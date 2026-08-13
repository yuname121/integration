#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datasets/co2/slope_ablation.py
Phase C-B1 — Controlled CO2_slope method / history ablation.

Selects an offline experimental slope reconstruction profile under one fixed
comparison universe, feature context, preprocessing policy, reference probe,
and metric contract. Does NOT perform final feature selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from datasets.co2.offline_experiment import (
    EXPECTED_LOCKED_TEST_SEALED,
    EXPECTED_TRAIN_COMMON,
    EXPECTED_VALIDATION_COMMON,
    EXPECTED_WARMUP_CANONICAL,
    MatrixBundle,
    TrainOnlyStandardScaler,
    _load_eligible_by_role,
    assert_no_forbidden_path_markers,
    build_sample_universe_manifest,
    compute_classification_metrics,
    load_comparison_matrix,
)
from datasets.co2.raw_reader import (
    UCIOccupancyRawReader,
    compute_sha256_bytes,
    compute_sha256_file,
    get_repo_root,
)
from datasets.co2.slope_feature import (
    COMPARISON_ABS_TOLERANCE,
    FEATURE_PROFILE_ID as A3_SLOPE_PROFILE_ID,
    FEATURE_UNIT,
    HISTORY_DURATION_SECONDS as A3_HISTORY_SECONDS,
    MAX_INTERNAL_GAP_SECONDS,
    SLOPE_METHOD as A3_SLOPE_METHOD,
    STATUS_AVAILABLE,
    reconstruct_all_slope_features,
    reconstruct_all_slope_features_with_params,
)

PHASE_ID = "C-B1"
PHASE_NAME = "CO2_SLOPE_METHOD_HISTORY_ABLATION"
ABLATION_CONTRACT_ID = "CO2_B1_SLOPE_ABLATION_CONTRACT_001"
CANDIDATE_REGISTRY_ID = "CO2_B1_SLOPE_CANDIDATE_REGISTRY_001"
PROBE_PROFILE_ID = "B1_FIXED_NEAREST_CENTROID_PROBE_001"
SELECTED_SLOPE_PROFILE_ID = "CO2_B1_SELECTED_SLOPE_PROFILE_001"
ARTIFACT_DIR_REL = "datasets/co2/manifests/c_b1_slope_method_history_ablation"
B0_DIR_REL = "datasets/co2/manifests/c_b0_offline_experiment_contract"
A6_LOCK_REL = "datasets/co2/manifests/c_a6_final_integrity_lock/artifact_lock_manifest.json"
A3_PROFILE_REL = "datasets/co2/manifests/c_a3_slope_feature/co2_slope_feature_profile.json"
CANONICAL_JSONL_REL = (
    "datasets/co2/manifests/c_a5_canonical_samples/canonical_source_samples.jsonl"
)
PRODUCTION_SCALER_REL = "models/co2/co2_scaling_metadata_v0.1.0.json"
A_SERIES_TAG = "co2-a-series-raw-to-canonical"
A_SERIES_TARGET = "bfd860cad2bb8dafe35ef7600cfa931d7d2d554d"
A_SERIES_LOCK_PROFILE = "CO2_A_SERIES_ARTIFACT_LOCK_PROFILE_001"
EXPECTED_LOCK_SHA256 = "b63f5e2da988f8e685cf1a01ec8e79c2c37f5bc77359be647f1147ecfb04e3da"
TARGET_PROFILE_ID = "CO2_OCCUPANCY_TARGET_PROFILE_001"
B0_CONTRACT_ID = "CO2_B0_OFFLINE_EXPERIMENT_CONTRACT_001"

AUTHORIZED_METHODS = ("ENDPOINT_DIFFERENCE", "CAUSAL_LINEAR_REGRESSION")
AUTHORIZED_HISTORIES = (60.0, 120.0, 150.0)
FIXED_BASE_FEATURES = ("CO2", "Temperature", "Humidity")
SLOPE_FEATURE_NAME = "candidate_CO2_slope"
NO_SLOPE_CONTROL_ID = "SCD40_NATIVE_NO_SLOPE_CONTROL"
BASELINE_CANDIDATE_ID = "ENDPOINT_H150"

B0_PREDECESSOR_FILES = (
    "experiment_contract.json",
    "sample_universe_manifest.json",
    "metric_contract.json",
    "feature_view_registry.json",
    "preprocessing_fit_evidence.json",
    "leakage_audit.json",
)


class CB1Error(RuntimeError):
    pass


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class LockedTestPolicyViolation(RuntimeError):
    pass


@dataclass(frozen=True)
class SlopeCandidateSpec:
    candidate_id: str
    method: str
    minimum_history_seconds: float
    is_a3_baseline: bool


def build_preregistered_candidates() -> List[SlopeCandidateSpec]:
    specs: List[SlopeCandidateSpec] = []
    for method, prefix in (
        ("ENDPOINT_DIFFERENCE", "ENDPOINT"),
        ("CAUSAL_LINEAR_REGRESSION", "LINEAR_REGRESSION"),
    ):
        for hist in AUTHORIZED_HISTORIES:
            cid = f"{prefix}_H{int(hist)}"
            specs.append(
                SlopeCandidateSpec(
                    candidate_id=cid,
                    method=method,
                    minimum_history_seconds=float(hist),
                    is_a3_baseline=(cid == BASELINE_CANDIDATE_ID),
                )
            )
    if len(specs) != 6:
        raise CB1Error(f"Expected 6 candidates, got {len(specs)}")
    return specs


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(obj: Any) -> str:
    return compute_sha256_bytes(_stable_json_dumps(obj).encode("utf-8"))


def _fingerprint_float64_le(values: Sequence[float], sample_ids: Sequence[str]) -> str:
    if len(values) != len(sample_ids):
        raise CB1Error("Fingerprint length mismatch")
    h = hashlib.sha256()
    h.update(b"CO2_B1_FEATURE_FP_V1|float64_le|sample_id_utf8_nul\n")
    for sid, val in zip(sample_ids, values):
        h.update(sid.encode("utf-8"))
        h.update(b"\0")
        h.update(np.float64(val).tobytes(order="C"))  # little-endian on this platform
    # Explicit endianness marker for portability documentation
    h.update(b"|endian=")
    h.update(sys.byteorder.encode("ascii"))
    return h.hexdigest()


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo))


def build_predecessor_fingerprint_registry(repo_root: Path) -> Dict[str, Any]:
    b0 = repo_root / B0_DIR_REL
    files = {}
    for name in B0_PREDECESSOR_FILES:
        path = b0 / name
        if not path.is_file():
            raise CB1Error(f"Missing B0 predecessor artifact: {name}")
        files[f"{B0_DIR_REL}/{name}"] = compute_sha256_file(path)
    lock_path = repo_root / A6_LOCK_REL
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_sha = compute_sha256_file(lock_path)
    if lock.get("lock_profile_id") != A_SERIES_LOCK_PROFILE:
        raise CB1Error("A-series lock profile id mismatch")
    if lock_sha != EXPECTED_LOCK_SHA256:
        raise CB1Error("C_B0_PREDECESSOR_FINGERPRINT_MISMATCH: A6 lock sha")
    a3_path = repo_root / A3_PROFILE_REL
    files[A6_LOCK_REL] = lock_sha
    files[A3_PROFILE_REL] = compute_sha256_file(a3_path)
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "registry_id": "CO2_B1_PREDECESSOR_FINGERPRINT_REGISTRY_001",
        "b0_contract_id": B0_CONTRACT_ID,
        "a_series_release_tag": A_SERIES_TAG,
        "a_series_release_target": A_SERIES_TARGET,
        "a_series_artifact_lock_profile": A_SERIES_LOCK_PROFILE,
        "a_series_artifact_lock_sha256": lock_sha,
        "a3_baseline_slope_profile": A3_SLOPE_PROFILE_ID,
        "a3_baseline_method": A3_SLOPE_METHOD,
        "a3_baseline_minimum_history_seconds": A3_HISTORY_SECONDS,
        "file_sha256": files,
        "closure_status": "LOCKED",
    }


def build_candidate_registry() -> Dict[str, Any]:
    specs = build_preregistered_candidates()
    candidates = []
    for spec in specs:
        candidates.append(
            {
                "candidate_id": spec.candidate_id,
                "method": spec.method,
                "minimum_history_seconds": spec.minimum_history_seconds,
                "unit": FEATURE_UNIT,
                "causality": "PAST_ONLY",
                "gap_restart_policy_seconds": MAX_INTERNAL_GAP_SECONDS,
                "actual_time_policy": "SOURCE_ACQUISITION_CLOCK_ELAPSED",
                "is_a3_baseline_correspondence": spec.is_a3_baseline,
                "participates_in_slope_ranking": True,
            }
        )
    payload = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "registry_id": CANDIDATE_REGISTRY_ID,
        "preregistered_before_validation_results": True,
        "authorized_methods": list(AUTHORIZED_METHODS),
        "authorized_history_thresholds_seconds": list(AUTHORIZED_HISTORIES),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "no_slope_control": {
            "control_id": NO_SLOPE_CONTROL_ID,
            "features": list(FIXED_BASE_FEATURES),
            "participates_in_slope_ranking": False,
            "purpose": "INCREMENTAL_SLOPE_EVIDENCE_ONLY",
        },
        "post_hoc_candidates_forbidden": True,
    }
    payload["registry_fingerprint"] = _sha256_json(
        {k: v for k, v in payload.items() if k != "registry_fingerprint"}
    )
    return payload


def build_ablation_contract(predecessor: Mapping[str, Any], registry: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "phase_name": PHASE_NAME,
        "contract_id": ABLATION_CONTRACT_ID,
        "objective": (
            "Determine which pre-registered causal CO2_slope reconstruction candidate "
            "provides the strongest VALIDATION evidence under one fixed comparison "
            "universe, feature context, preprocessing policy, reference probe, and "
            "metric contract."
        ),
        "experimental_factors": [
            "slope_reconstruction_method",
            "slope_minimum_history_threshold",
        ],
        "fixed_factors": [
            "b0_comparison_universe",
            "scd40_native_feature_context",
            "train_only_standardization",
            "nearest_centroid_reference_probe",
            "b0_metric_contract",
            "occupancy_target_semantics",
        ],
        "comparison_universe": {
            "train": EXPECTED_TRAIN_COMMON,
            "validation": EXPECTED_VALIDATION_COMMON,
            "locked_test": EXPECTED_LOCKED_TEST_SEALED,
            "locked_test_status": "SEALED",
            "warmup": EXPECTED_WARMUP_CANONICAL,
        },
        "feature_context": {
            "slope_candidate_features": list(FIXED_BASE_FEATURES) + [SLOPE_FEATURE_NAME],
            "no_slope_control_features": list(FIXED_BASE_FEATURES),
            "historical_compatibility_view_used_for_ranking": False,
            "light_excluded": True,
            "humidity_ratio_excluded": True,
        },
        "reference_probe": {
            "profile_id": PROBE_PROFILE_ID,
            "status": "NOT_A_PRODUCTION_MODEL",
            "deployable": False,
            "tflite_candidate": False,
            "architecture_candidate": False,
        },
        "primary_selection_metric": "validation_macro_f1",
        "tie_break_policy": [
            "higher_validation_macro_f1",
            "higher_validation_balanced_accuracy",
            "higher_validation_occupied_recall",
            "shorter_minimum_history_threshold",
            "simpler_method_ENDPOINT_DIFFERENCE_over_CAUSAL_LINEAR_REGRESSION",
            "lexicographically_smaller_candidate_id",
        ],
        "predecessor_fingerprint_registry_id": predecessor["registry_id"],
        "candidate_registry_id": registry["registry_id"],
        "candidate_registry_fingerprint": registry["registry_fingerprint"],
        "a_series_slope_profile_immutable": A3_SLOPE_PROFILE_ID,
        "selected_profile_id": SELECTED_SLOPE_PROFILE_ID,
        "final_feature_selection": "NOT_PERFORMED",
        "locked_test_policy": {
            "feature_value_access": False,
            "target_access_for_evaluation": False,
            "predictions": 0,
            "metrics": 0,
            "scaler_fit": False,
            "probe_fit": False,
            "candidate_ranking_use": False,
        },
        "device_claims": {
            "device_uci_cadence_domain_gap": "DEVICE_UCI_CADENCE_DOMAIN_GAP",
            "winner_classification": ["OFFLINE_UCI_SELECTED", "DEVICE_DOMAIN_UNVALIDATED"],
            "runtime_equivalence_claimed": False,
        },
        "lineage_limitations": [
            "CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED",
            "DEVICE_UCI_CADENCE_DOMAIN_GAP",
        ],
    }


@dataclass
class NearestCentroidProbe:
    """Deterministic dependency-light reference representation probe."""

    feature_names: Tuple[str, ...]
    vacant_centroid_: Optional[np.ndarray] = None
    occupied_centroid_: Optional[np.ndarray] = None
    n_train_: int = 0
    status: str = "C_B1_REFERENCE_PROBE_ONLY"
    profile_id: str = PROBE_PROFILE_ID

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NearestCentroidProbe":
        if x.ndim != 2 or x.shape[0] != y.shape[0]:
            raise CB1Error("Probe fit shape mismatch")
        if not np.isfinite(x).all():
            raise CB1Error("Non-finite features in probe fit")
        vac = x[y == 0]
        occ = x[y == 1]
        if vac.size == 0 or occ.size == 0:
            raise CB1Error("Both classes required for nearest-centroid probe")
        self.vacant_centroid_ = vac.mean(axis=0)
        self.occupied_centroid_ = occ.mean(axis=0)
        self.n_train_ = int(x.shape[0])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.vacant_centroid_ is None or self.occupied_centroid_ is None:
            raise CB1Error("Probe not fitted")
        d_vac = np.sum((x - self.vacant_centroid_) ** 2, axis=1)
        d_occ = np.sum((x - self.occupied_centroid_) ** 2, axis=1)
        # exact equal distance → VACANT / class 0
        pred = np.where(d_occ < d_vac, 1, 0).astype(np.int64)
        return pred

    def uncalibrated_scores(self, x: np.ndarray) -> np.ndarray:
        """UNCALIBRATED_REFERENCE_SCORE = d_vac - d_occ (larger ⇒ more OCCUPIED-like)."""
        if self.vacant_centroid_ is None or self.occupied_centroid_ is None:
            raise CB1Error("Probe not fitted")
        d_vac = np.sum((x - self.vacant_centroid_) ** 2, axis=1)
        d_occ = np.sum((x - self.occupied_centroid_) ** 2, axis=1)
        return d_vac - d_occ

    def to_metadata(self) -> Dict[str, Any]:
        if self.vacant_centroid_ is None or self.occupied_centroid_ is None:
            raise CB1Error("Probe not fitted")
        return {
            "profile_id": self.profile_id,
            "status": self.status,
            "feature_order": list(self.feature_names),
            "train_sample_count": self.n_train_,
            "vacant_centroid": [float(v) for v in self.vacant_centroid_],
            "occupied_centroid": [float(v) for v in self.occupied_centroid_],
            "tie_rule": "exact_equal_distance_predicts_VACANT",
            "class_priors_used": False,
            "class_weights_used": False,
            "threshold_tuned": False,
            "deployable": False,
            "production_model": False,
        }


def _index_slope_records(records) -> Dict[Tuple[str, str], Any]:
    out = {}
    for r in records:
        key = (r.target_source_member, r.target_source_row_identifier)
        out[key] = r
    return out


def _load_canonical_index(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    path = repo_root / CANONICAL_JSONL_REL
    by_id: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            by_id[row["canonical_sample_id"]] = row
    return by_id


def _member_row_key_from_canonical(row: Mapping[str, Any]) -> Tuple[str, str]:
    return (row["source_member_name"], row["source_row_identifier"])


def reconstruct_candidate_slopes(
    observations,
    spec: SlopeCandidateSpec,
) -> List[Any]:
    return reconstruct_all_slope_features_with_params(
        observations,
        method=spec.method,
        history_duration_seconds=spec.minimum_history_seconds,
        feature_contract_id=f"CO2_B1_CANDIDATE_{spec.candidate_id}",
    )


def extract_universe_slopes(
    *,
    records_by_key: Mapping[Tuple[str, str], Any],
    canonical_by_id: Mapping[str, Mapping[str, Any]],
    sample_ids: Sequence[str],
    candidate_id: str,
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    slopes: List[float] = []
    spans: List[float] = []
    unavailable = 0
    nonfinite = 0
    for sid in sample_ids:
        row = canonical_by_id[sid]
        key = _member_row_key_from_canonical(row)
        rec = records_by_key.get(key)
        if rec is None:
            raise CB1Error(f"Missing slope record for {sid} under {candidate_id}")
        if rec.feature_status != STATUS_AVAILABLE or rec.co2_slope is None:
            unavailable += 1
            slopes.append(float("nan"))
            continue
        val = float(rec.co2_slope)
        if not math.isfinite(val):
            nonfinite += 1
            slopes.append(val)
            continue
        slopes.append(val)
        if rec.history_elapsed_seconds is None:
            raise CB1Error(f"Missing history span for available slope {sid}")
        spans.append(float(rec.history_elapsed_seconds))

    diagnostics = {
        "eligible_count": len(sample_ids) - unavailable,
        "unavailable_count": unavailable,
        "non_finite_count": nonfinite + sum(1 for v in slopes if math.isfinite(v) is False and not math.isnan(v)),
        "nan_count": sum(1 for v in slopes if isinstance(v, float) and math.isnan(v)),
        "history_span_seconds": _span_stats(spans),
        "slope_ppm_per_min": _value_stats([v for v in slopes if math.isfinite(v)]),
    }
    # Clarify nonfinite: count non-finite among produced numeric slots excluding unavailable NaN placeholders
    finite_or_inf = [v for v in slopes if not (isinstance(v, float) and math.isnan(v))]
    diagnostics["non_finite_count"] = sum(1 for v in slopes if not math.isfinite(v))
    diagnostics["requested_count"] = len(sample_ids)
    return slopes, spans, diagnostics


def _span_stats(spans: Sequence[float]) -> Dict[str, Any]:
    if not spans:
        return {"min": None, "median": None, "max": None, "count": 0}
    ordered = sorted(spans)
    return {
        "min": float(ordered[0]),
        "median": float(statistics.median(ordered)),
        "max": float(ordered[-1]),
        "count": len(ordered),
    }


def _value_stats(vals: Sequence[float]) -> Dict[str, Any]:
    if not vals:
        return {
            "min": None,
            "median": None,
            "max": None,
            "mean": None,
            "std": None,
            "count": 0,
        }
    arr = np.asarray(list(vals), dtype=np.float64)
    return {
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "count": int(arr.size),
    }


def verify_endpoint_h150_parity(
    *,
    candidate_records,
    baseline_records,
    canonical_by_id: Mapping[str, Mapping[str, Any]],
    train_ids: Sequence[str],
    val_ids: Sequence[str],
) -> Dict[str, Any]:
    cand_idx = _index_slope_records(candidate_records)
    base_idx = _index_slope_records(baseline_records)
    status_mm = 0
    value_mm = 0
    max_abs = 0.0
    checked = 0
    for sid in list(train_ids) + list(val_ids):
        row = canonical_by_id[sid]
        key = _member_row_key_from_canonical(row)
        c = cand_idx[key]
        b = base_idx[key]
        checked += 1
        if c.feature_status != b.feature_status:
            status_mm += 1
        # Also compare against canonical persisted baseline value when available
        canon_slope = row.get("co2_slope")
        if c.feature_status != STATUS_AVAILABLE or c.co2_slope is None:
            if b.co2_slope is not None or canon_slope is not None:
                value_mm += 1
            continue
        cv = float(c.co2_slope)
        bv = float(b.co2_slope) if b.co2_slope is not None else None
        if bv is None or cv != bv:
            if bv is None or abs(cv - bv) > COMPARISON_ABS_TOLERANCE:
                value_mm += 1
            else:
                max_abs = max(max_abs, abs(cv - bv))
        else:
            max_abs = max(max_abs, 0.0)
        if canon_slope is not None:
            abs_d = abs(cv - float(canon_slope))
            max_abs = max(max_abs, abs_d)
            if abs_d > COMPARISON_ABS_TOLERANCE and cv != float(canon_slope):
                value_mm += 1
    status = "PASS" if status_mm == 0 and value_mm == 0 else "C_A3_BASELINE_PARITY_FAILURE"
    return {
        "candidate_id": BASELINE_CANDIDATE_ID,
        "checked_samples": checked,
        "status_mismatches": status_mm,
        "value_mismatches": value_mm,
        "max_abs_difference": max_abs,
        "comparison_abs_tolerance": COMPARISON_ABS_TOLERANCE,
        "equality_policy": "prefer_exact_float64_equality",
        "status": status,
    }


def build_feature_bundle(
    *,
    sample_ids: Sequence[str],
    labels: np.ndarray,
    base_features: np.ndarray,
    slope_values: Optional[Sequence[float]],
    split_role: str,
    include_slope: bool,
) -> MatrixBundle:
    if include_slope:
        if slope_values is None:
            raise CB1Error("slope_values required")
        slope_arr = np.asarray(list(slope_values), dtype=np.float64).reshape(-1, 1)
        if not np.isfinite(slope_arr).all():
            raise CB1Error("CANDIDATE_AVAILABILITY_CONTRACT_MISMATCH: non-finite slope")
        feats = np.concatenate([base_features, slope_arr], axis=1)
        names = FIXED_BASE_FEATURES + (SLOPE_FEATURE_NAME,)
    else:
        feats = base_features
        names = FIXED_BASE_FEATURES
    return MatrixBundle(
        sample_ids=list(sample_ids),
        features=np.asarray(feats, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        feature_names=tuple(names),
        split_role=split_role,
    )


def evaluate_probe_on_bundle(
    train_bundle: MatrixBundle,
    val_bundle: MatrixBundle,
) -> Dict[str, Any]:
    if train_bundle.split_role != "TRAIN" or val_bundle.split_role != "VALIDATION":
        raise CB1Error("Probe evaluation requires TRAIN fit / VALIDATION eval")
    scaler = TrainOnlyStandardScaler(feature_names=train_bundle.feature_names)
    scaler.status = "C_B1_REFERENCE_PROBE_ONLY"
    scaler.fit(train_bundle)
    x_train = scaler.transform(train_bundle)
    x_val = scaler.transform(val_bundle)
    probe = NearestCentroidProbe(feature_names=train_bundle.feature_names)
    probe.fit(x_train, train_bundle.labels)
    y_pred = probe.predict(x_val)
    metrics = compute_classification_metrics(val_bundle.labels, y_pred)
    pop_fp = _sha256_json({"sample_ids": train_bundle.sample_ids, "role": "TRAIN"})
    return {
        "metrics": metrics,
        "predictions": [int(x) for x in y_pred.tolist()],
        "scaler_metadata": scaler.to_metadata(
            fit_population_fingerprint=pop_fp,
            feature_view_id="C_B1_SCD40_NATIVE_PLUS_CANDIDATE_SLOPE"
            if SLOPE_FEATURE_NAME in train_bundle.feature_names
            else "C_B1_SCD40_NATIVE_NO_SLOPE",
        ),
        "probe_metadata": probe.to_metadata(),
        "uncalibrated_reference_score_note": (
            "Distance-difference scores are UNCALIBRATED_REFERENCE_SCORE and are "
            "not used for candidate selection."
        ),
    }


def rank_slope_candidates(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Pre-registered ranking among slope candidates only."""
    method_rank = {"ENDPOINT_DIFFERENCE": 0, "CAUSAL_LINEAR_REGRESSION": 1}

    def sort_key(item: Mapping[str, Any]):
        m = item["validation_metrics"]
        return (
            -float(m["macro_f1"]),
            -float(m["balanced_accuracy"]),
            -float(m["recall_occupied"]),
            float(item["minimum_history_seconds"]),
            method_rank[item["method"]],
            item["candidate_id"],
        )

    ordered = sorted(results, key=sort_key)
    ranked = []
    for i, item in enumerate(ordered, start=1):
        ranked.append(
            {
                "rank": i,
                "candidate_id": item["candidate_id"],
                "method": item["method"],
                "minimum_history_seconds": item["minimum_history_seconds"],
                "validation_macro_f1": item["validation_metrics"]["macro_f1"],
                "validation_balanced_accuracy": item["validation_metrics"]["balanced_accuracy"],
                "validation_occupied_recall": item["validation_metrics"]["recall_occupied"],
            }
        )
    return ranked


def build_selected_profile(
    winner: Mapping[str, Any],
    *,
    registry_fingerprint: str,
    incremental: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "profile_id": SELECTED_SLOPE_PROFILE_ID,
        "phase": PHASE_ID,
        "selected_candidate_id": winner["candidate_id"],
        "method": winner["method"],
        "minimum_history_seconds": winner["minimum_history_seconds"],
        "unit": FEATURE_UNIT,
        "causal_policy": "PAST_ONLY",
        "gap_restart_policy_seconds": MAX_INTERNAL_GAP_SECONDS,
        "actual_time_policy": "SOURCE_ACQUISITION_CLOCK_ELAPSED",
        "selection_universe": {
            "train": EXPECTED_TRAIN_COMMON,
            "validation": EXPECTED_VALIDATION_COMMON,
        },
        "reference_probe": PROBE_PROFILE_ID,
        "selection_metric": "validation_macro_f1",
        "tie_break_policy": [
            "higher_validation_macro_f1",
            "higher_validation_balanced_accuracy",
            "higher_validation_occupied_recall",
            "shorter_minimum_history_threshold",
            "simpler_method_ENDPOINT_DIFFERENCE_over_CAUSAL_LINEAR_REGRESSION",
            "lexicographically_smaller_candidate_id",
        ],
        "candidate_registry_fingerprint": registry_fingerprint,
        "b0_contract_id": B0_CONTRACT_ID,
        "a_series_baseline_profile": A3_SLOPE_PROFILE_ID,
        "a_series_baseline_retained": winner["candidate_id"] == BASELINE_CANDIDATE_ID,
        "validation_selection_status": "SELECTED",
        "locked_test_status": "SEALED",
        "deployment_status": "NOT_VALIDATED",
        "winner_classification": ["OFFLINE_UCI_SELECTED", "DEVICE_DOMAIN_UNVALIDATED"],
        "final_feature_selection": "NOT_PERFORMED",
        "incremental_slope_evidence": incremental,
        "relationship_to_a_series": {
            "a_series_canonical_baseline": A3_SLOPE_PROFILE_ID,
            "b_series_selected_experimental_reconstruction": SELECTED_SLOPE_PROFILE_ID,
            "a_series_source_of_truth_modified": False,
        },
    }


def run_slope_ablation(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root or get_repo_root()
    out_dir = root / ARTIFACT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    predecessor = build_predecessor_fingerprint_registry(root)
    registry = build_candidate_registry()
    contract = build_ablation_contract(predecessor, registry)
    universe = build_sample_universe_manifest(root)
    by_role = _load_eligible_by_role(root)
    train_ids = by_role["TRAIN"]
    val_ids = by_role["VALIDATION"]
    locked_ids = by_role["LOCKED_TEST"]
    if len(train_ids) != EXPECTED_TRAIN_COMMON:
        raise CB1Error("TRAIN universe mismatch")
    if len(val_ids) != EXPECTED_VALIDATION_COMMON:
        raise CB1Error("VALIDATION universe mismatch")
    if len(locked_ids) != EXPECTED_LOCKED_TEST_SEALED:
        raise CB1Error("LOCKED_TEST universe mismatch")

    # Base non-slope features from B0 harness (TRAIN/VAL only)
    train_base = load_comparison_matrix(
        repo_root=root, split_role="TRAIN", feature_names=FIXED_BASE_FEATURES
    )
    val_base = load_comparison_matrix(
        repo_root=root, split_role="VALIDATION", feature_names=FIXED_BASE_FEATURES
    )
    if train_base.sample_ids != train_ids or val_base.sample_ids != val_ids:
        raise CB1Error("B0 ordered ID mismatch vs load_comparison_matrix")

    reader = UCIOccupancyRawReader(repo_root=root)
    observations = reader.read_all_observations()
    baseline_records = reconstruct_all_slope_features(observations)
    canonical_by_id = _load_canonical_index(root)

    candidate_results: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {}
    fingerprints: Dict[str, Any] = {}
    preprocessing: Dict[str, Any] = {}
    probe_params: Dict[str, Any] = {}
    validation_metrics: Dict[str, Any] = {}
    validation_predictions: Dict[str, Any] = {}

    for spec in build_preregistered_candidates():
        records = reconstruct_candidate_slopes(observations, spec)
        by_key = _index_slope_records(records)
        train_slopes, _, train_diag = extract_universe_slopes(
            records_by_key=by_key,
            canonical_by_id=canonical_by_id,
            sample_ids=train_ids,
            candidate_id=spec.candidate_id,
        )
        val_slopes, _, val_diag = extract_universe_slopes(
            records_by_key=by_key,
            canonical_by_id=canonical_by_id,
            sample_ids=val_ids,
            candidate_id=spec.candidate_id,
        )
        if train_diag["unavailable_count"] or train_diag["non_finite_count"]:
            raise CB1Error(
                f"CANDIDATE_AVAILABILITY_CONTRACT_MISMATCH: {spec.candidate_id} TRAIN "
                f"unavailable={train_diag['unavailable_count']} "
                f"nonfinite={train_diag['non_finite_count']}"
            )
        if val_diag["unavailable_count"] or val_diag["non_finite_count"]:
            raise CB1Error(
                f"CANDIDATE_AVAILABILITY_CONTRACT_MISMATCH: {spec.candidate_id} VALIDATION "
                f"unavailable={val_diag['unavailable_count']} "
                f"nonfinite={val_diag['non_finite_count']}"
            )
        if train_diag["eligible_count"] != EXPECTED_TRAIN_COMMON:
            raise CB1Error("TRAIN availability != 8140")
        if val_diag["eligible_count"] != EXPECTED_VALIDATION_COMMON:
            raise CB1Error("VALIDATION availability != 2662")

        if spec.candidate_id == BASELINE_CANDIDATE_ID:
            parity = verify_endpoint_h150_parity(
                candidate_records=records,
                baseline_records=baseline_records,
                canonical_by_id=canonical_by_id,
                train_ids=train_ids,
                val_ids=val_ids,
            )
            if parity["status"] != "PASS":
                raise CB1Error("C_A3_BASELINE_PARITY_FAILURE")
        else:
            parity = None

        train_bundle = build_feature_bundle(
            sample_ids=train_ids,
            labels=train_base.labels,
            base_features=train_base.features,
            slope_values=train_slopes,
            split_role="TRAIN",
            include_slope=True,
        )
        val_bundle = build_feature_bundle(
            sample_ids=val_ids,
            labels=val_base.labels,
            base_features=val_base.features,
            slope_values=val_slopes,
            split_role="VALIDATION",
            include_slope=True,
        )
        eval_out = evaluate_probe_on_bundle(train_bundle, val_bundle)
        result = {
            "candidate_id": spec.candidate_id,
            "method": spec.method,
            "minimum_history_seconds": spec.minimum_history_seconds,
            "is_a3_baseline": spec.is_a3_baseline,
            "train_availability": train_diag,
            "validation_availability": val_diag,
            "validation_metrics": eval_out["metrics"],
            "a3_parity": parity,
        }
        candidate_results.append(result)
        diagnostics[spec.candidate_id] = {
            "TRAIN": train_diag,
            "VALIDATION": val_diag,
        }
        fingerprints[spec.candidate_id] = {
            "serialization": "float64_native_endian_tobytes_plus_utf8_sample_ids",
            "byte_order": sys.byteorder,
            "TRAIN": {
                "count": len(train_ids),
                "fingerprint": _fingerprint_float64_le(train_slopes, train_ids),
            },
            "VALIDATION": {
                "count": len(val_ids),
                "fingerprint": _fingerprint_float64_le(val_slopes, val_ids),
            },
            "LOCKED_TEST": {
                "status": "NOT_GENERATED",
                "reason": "LOCKED_TEST_SEALED_IN_C_B1",
            },
        }
        preprocessing[spec.candidate_id] = eval_out["scaler_metadata"]
        probe_params[spec.candidate_id] = eval_out["probe_metadata"]
        validation_metrics[spec.candidate_id] = eval_out["metrics"]
        validation_predictions[spec.candidate_id] = {
            "split": "VALIDATION",
            "count": len(eval_out["predictions"]),
            "predictions": eval_out["predictions"],
        }

    # No-slope control
    no_slope_train = build_feature_bundle(
        sample_ids=train_ids,
        labels=train_base.labels,
        base_features=train_base.features,
        slope_values=None,
        split_role="TRAIN",
        include_slope=False,
    )
    no_slope_val = build_feature_bundle(
        sample_ids=val_ids,
        labels=val_base.labels,
        base_features=val_base.features,
        slope_values=None,
        split_role="VALIDATION",
        include_slope=False,
    )
    no_slope_eval = evaluate_probe_on_bundle(no_slope_train, no_slope_val)
    no_slope_result = {
        "control_id": NO_SLOPE_CONTROL_ID,
        "features": list(FIXED_BASE_FEATURES),
        "participates_in_slope_ranking": False,
        "validation_metrics": no_slope_eval["metrics"],
        "scaler_metadata": no_slope_eval["scaler_metadata"],
        "probe_metadata": no_slope_eval["probe_metadata"],
        "predictions": {
            "split": "VALIDATION",
            "count": len(no_slope_eval["predictions"]),
            "predictions": no_slope_eval["predictions"],
        },
    }

    ranking = rank_slope_candidates(candidate_results)
    winner_id = ranking[0]["candidate_id"]
    winner = next(r for r in candidate_results if r["candidate_id"] == winner_id)
    delta_macro = float(winner["validation_metrics"]["macro_f1"]) - float(
        no_slope_result["validation_metrics"]["macro_f1"]
    )
    delta_bal = float(winner["validation_metrics"]["balanced_accuracy"]) - float(
        no_slope_result["validation_metrics"]["balanced_accuracy"]
    )
    delta_occ = float(winner["validation_metrics"]["recall_occupied"]) - float(
        no_slope_result["validation_metrics"]["recall_occupied"]
    )
    incremental_status = (
        "ESTABLISHED"
        if delta_macro > 0.0
        else "SLOPE_INCREMENTAL_VALUE_NOT_ESTABLISHED"
    )
    incremental = {
        "status": incremental_status,
        "delta_macro_f1": delta_macro,
        "delta_balanced_accuracy": delta_bal,
        "delta_occupied_recall": delta_occ,
        "no_slope_control_id": NO_SLOPE_CONTROL_ID,
        "note": (
            "No-slope control does not enter slope-candidate ranking; "
            "winner remains the best slope reconstruction candidate."
        ),
    }
    selected = build_selected_profile(
        winner, registry_fingerprint=registry["registry_fingerprint"], incremental=incremental
    )

    # Majority baseline context (recompute via B0-compatible TRAIN majority)
    y_train = train_base.labels
    counts = Counter(int(x) for x in y_train.tolist())
    majority_label = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    maj_pred = np.full(len(val_ids), majority_label, dtype=np.int64)
    majority_metrics = compute_classification_metrics(val_base.labels, maj_pred)

    leakage = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "train_validation_overlap": universe["overlaps"]["train_validation"],
        "train_locked_test_overlap": universe["overlaps"]["train_locked_test"],
        "validation_locked_test_overlap": universe["overlaps"]["validation_locked_test"],
        "target_included_as_feature": 0,
        "provenance_included_as_feature": 0,
        "validation_used_to_fit_scaler": False,
        "validation_used_to_fit_probe": False,
        "locked_test_used_to_fit_scaler": False,
        "locked_test_used_to_fit_probe": False,
        "locked_test_metrics": 0,
        "locked_test_predictions": 0,
        "locked_test_candidate_values_in_result_artifacts": 0,
        "candidate_grid_changed_after_evaluation": False,
        "uci_light_in_probe": False,
        "status": "PASS"
        if (
            universe["overlaps"]["train_validation"] == 0
            and universe["overlaps"]["train_locked_test"] == 0
            and universe["overlaps"]["validation_locked_test"] == 0
        )
        else "FAIL",
    }

    exceptions = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "limitations": [
            {
                "code": "DEVICE_UCI_CADENCE_DOMAIN_GAP",
                "severity": "UCI acquisition cadence is not the SCD40 runtime cadence.",
            },
            {
                "code": "CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED",
                "severity": (
                    "B1 does not retroactively establish historical TFLite/scaler training lineage."
                ),
            },
            {
                "code": "REFERENCE_PROBE_ONLY",
                "severity": (
                    "Nearest-centroid probe is not a production model or architecture candidate."
                ),
            },
            {
                "code": "FINAL_FEATURE_SELECTION_DEFERRED",
                "severity": "B1 selects slope reconstruction only; feature inclusion is later.",
            },
        ],
        "blockers": [],
    }

    probe_contract = {
        "manifest_version": "1.0",
        "profile_id": PROBE_PROFILE_ID,
        "probe_type": "NEAREST_CENTROID",
        "distance": "squared_euclidean",
        "tie_rule": "exact_equal_distance_predicts_VACANT",
        "class_priors": False,
        "class_weights": False,
        "threshold_tuning": False,
        "probability_calibration": False,
        "status": "NOT_A_PRODUCTION_MODEL",
        "deployable": False,
        "tflite_candidate": False,
        "dependency_policy": "numpy_only_no_sklearn_required",
    }

    selection_decision = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "candidate_registry_fingerprint": registry["registry_fingerprint"],
        "ranking_rule": contract["tie_break_policy"],
        "primary_metric": "validation_macro_f1",
        "ranking": ranking,
        "winning_slope_candidate": winner_id,
        "winning_method": winner["method"],
        "winning_minimum_history_seconds": winner["minimum_history_seconds"],
        "winning_validation_metrics": winner["validation_metrics"],
        "no_slope_control": {
            "control_id": NO_SLOPE_CONTROL_ID,
            "validation_metrics": no_slope_result["validation_metrics"],
            "included_in_slope_ranking": False,
        },
        "incremental_slope_evidence": incremental,
        "majority_class_baseline_context": {
            "train_majority_label": int(majority_label),
            "train_majority_class": "VACANT" if majority_label == 0 else "OCCUPIED",
            "validation_metrics": majority_metrics,
            "role": "REFERENCE_FLOOR_ONLY",
        },
        "endpoint_h150_a3_parity": next(
            r["a3_parity"] for r in candidate_results if r["candidate_id"] == BASELINE_CANDIDATE_ID
        ),
        "locked_test_status": "SEALED",
        "final_feature_selection_performed": False,
    }

    generation = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "generated_at_utc": None,
        "generation_clock_policy": "OMITTED_FOR_BIT_IDENTICAL_RERUNS",
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "artifact_dir": ARTIFACT_DIR_REL,
        "determinism_required": True,
    }

    # Persist artifacts
    artifacts = {
        "predecessor_fingerprint_registry.json": predecessor,
        "slope_ablation_contract.json": contract,
        "candidate_registry.json": registry,
        "candidate_feature_fingerprint_registry.json": {
            "manifest_version": "1.0",
            "phase": PHASE_ID,
            "candidates": fingerprints,
        },
        "candidate_diagnostics.json": {
            "manifest_version": "1.0",
            "phase": PHASE_ID,
            "diagnostics": diagnostics,
            "locked_test_value_diagnostics": "NOT_COMPUTED",
        },
        "fixed_probe_contract.json": probe_contract,
        "preprocessing_evidence.json": {
            "manifest_version": "1.0",
            "phase": PHASE_ID,
            "status": "C_B1_REFERENCE_PROBE_ONLY",
            "production_scaler_path": PRODUCTION_SCALER_REL,
            "production_scaler_modified": False,
            "per_candidate": preprocessing,
            "no_slope_control": no_slope_result["scaler_metadata"],
        },
        "probe_fit_evidence.json": {
            "manifest_version": "1.0",
            "phase": PHASE_ID,
            "fit_population": "TRAIN",
            "evaluation_population": "VALIDATION",
            "per_candidate": probe_params,
            "no_slope_control": no_slope_result["probe_metadata"],
        },
        "validation_metric_results.json": {
            "manifest_version": "1.0",
            "phase": PHASE_ID,
            "population": "VALIDATION",
            "locked_test_metrics": 0,
            "candidates": validation_metrics,
            "no_slope_control": no_slope_result["validation_metrics"],
            "majority_class_baseline": majority_metrics,
        },
        "validation_predictions.json": {
            "manifest_version": "1.0",
            "phase": PHASE_ID,
            "population": "VALIDATION",
            "locked_test_predictions": 0,
            "candidates": validation_predictions,
            "no_slope_control": no_slope_result["predictions"],
        },
        "no_slope_control_result.json": no_slope_result,
        "selection_decision.json": selection_decision,
        "selected_slope_profile.json": selected,
        "leakage_audit.json": leakage,
        "exceptions_and_limitations.json": exceptions,
        "generation_metadata.json": generation,
        "candidate_availability_summary.json": {
            "manifest_version": "1.0",
            "expected_train": EXPECTED_TRAIN_COMMON,
            "expected_validation": EXPECTED_VALIDATION_COMMON,
            "per_candidate": {
                cid: {
                    "train_available": diagnostics[cid]["TRAIN"]["eligible_count"],
                    "validation_available": diagnostics[cid]["VALIDATION"]["eligible_count"],
                    "train_unavailable": diagnostics[cid]["TRAIN"]["unavailable_count"],
                    "validation_unavailable": diagnostics[cid]["VALIDATION"]["unavailable_count"],
                    "train_non_finite": diagnostics[cid]["TRAIN"]["non_finite_count"],
                    "validation_non_finite": diagnostics[cid]["VALIDATION"]["non_finite_count"],
                }
                for cid in diagnostics
            },
        },
    }

    checksum_lines: List[str] = []
    for name, payload in sorted(artifacts.items()):
        path = out_dir / name
        write_json(path, payload)
        # Validate portable paths in serialized text
        text = path.read_text(encoding="utf-8")
        bad = assert_no_forbidden_path_markers(text)
        if bad:
            raise CB1Error(f"Forbidden path markers in {name}: {bad}")
        checksum_lines.append(f"{compute_sha256_file(path)}  {ARTIFACT_DIR_REL}/{name}")

    identity = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "artifact_dir": ARTIFACT_DIR_REL,
        "artifact_count": len(artifacts),
        "selected_profile_id": SELECTED_SLOPE_PROFILE_ID,
        "winning_candidate_id": winner_id,
        "a_series_slope_profile_modified": False,
        "production_scaler_modified": False,
        "production_model_modified": False,
    }
    identity_path = out_dir / "artifact_identity.json"
    write_json(identity_path, identity)
    checksum_lines.append(
        f"{compute_sha256_file(identity_path)}  {ARTIFACT_DIR_REL}/artifact_identity.json"
    )
    checksum_path = out_dir / "checksums.sha256"
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    return {
        "artifact_dir": str(out_dir.relative_to(root)).replace("\\", "/"),
        "winner": winner_id,
        "incremental_status": incremental_status,
        "parity": selection_decision["endpoint_h150_a3_parity"],
        "ranking": ranking,
    }


def load_b1_artifacts(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root or get_repo_root()
    d = root / ARTIFACT_DIR_REL
    out = {}
    for path in sorted(d.glob("*.json")):
        out[path.name] = json.loads(path.read_text(encoding="utf-8"))
    return out
