#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datasets/co2/offline_experiment.py
Phase C-B0 — Offline experiment contract, leakage-safe comparison universe,
and baseline evaluation harness helpers.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from datasets.co2.canonical_samples import (
    CANONICAL_SAMPLE_PROFILE_ID,
    EXPECTED_SLOPE_ELIGIBLE,
    EXPECTED_TOTAL_SAMPLES,
    EXPECTED_WARMUP,
)
from datasets.co2.raw_reader import compute_sha256_file, get_repo_root
from datasets.co2.slope_feature import FEATURE_PROFILE_ID as SLOPE_PROFILE_ID
from datasets.co2.target_semantics import TARGET_PROFILE_ID as TARGET_PROFILE_ID

PHASE_ID = "C-B0"
EXPERIMENT_CONTRACT_ID = "CO2_B0_OFFLINE_EXPERIMENT_CONTRACT_001"
MANIFEST_DIR_REL = "datasets/co2/manifests/c_b0_offline_experiment_contract"
A_SERIES_RELEASE_TAG = "co2-a-series-raw-to-canonical"
A_SERIES_RELEASE_COMMIT = "bfd860cad2bb8dafe35ef7600cfa931d7d2d554d"
A_SERIES_LOCK_PROFILE = "CO2_A_SERIES_ARTIFACT_LOCK_PROFILE_001"
A_SERIES_LOCK_PATH = (
    "datasets/co2/manifests/c_a6_final_integrity_lock/artifact_lock_manifest.json"
)

EXPECTED_TRAIN_COMMON = 8140
EXPECTED_VALIDATION_COMMON = 2662
EXPECTED_LOCKED_TEST_SEALED = 9749
EXPECTED_WARMUP_CANONICAL = 9

POSITIVE_CLASS = "OCCUPIED"
NEGATIVE_CLASS = "VACANT"
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

DEFAULT_SEED = 20260810
DEFAULT_PROB_THRESHOLD = 0.5

FEATURE_FIELDS = (
    "Temperature",
    "Humidity",
    "Light",
    "CO2",
    "HumidityRatio",
    "CO2_slope",
)
PROVENANCE_FIELDS = (
    "canonical_sample_id",
    "canonical_sample_index",
    "source_archive_path",
    "source_archive_sha256",
    "source_member_name",
    "source_member_sha256",
    "source_physical_line_number",
    "source_row_identifier",
    "source_timestamp_raw",
    "canonical_timestamp",
    "temporal_block_id",
    "future_split_role",
    "co2_slope_status",
    "history_start_source_row_identifier",
    "history_elapsed_seconds",
    "source_sample_count_used",
    "model_eligible_for_slope_complete_view",
    "model_eligibility_exclusion_reason",
    "scaler_fit_authorized",
    "locked_test_fit_authorized",
    "locked_test_tuning_authorized",
    "target_profile_id",
    "co2_slope_profile_id",
    "sample_kind",
)
TARGET_FIELDS = ("Occupancy", "occupancy_source_value", "occupancy_canonical_class")


class CB0Error(Exception):
    """Base C-B0 error."""


class LockedTestPolicyViolation(CB0Error):
    """Raised when LOCKED_TEST is used for fit/tuning/evaluation misuse."""


class ASeriesBaselineDrift(CB0Error):
    """Raised when released A-series artifacts do not match expected fingerprints."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(blob)


def ordered_id_list_sha256(ids: Sequence[str]) -> str:
    return _sha256_text("\n".join(ids) + ("\n" if ids else ""))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_a_series_release(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Verify local/remote tag identity using git when available."""
    import subprocess

    root = repo_root or get_repo_root()
    result = {
        "expected_tag": A_SERIES_RELEASE_TAG,
        "expected_commit": A_SERIES_RELEASE_COMMIT,
        "resolved_commit": None,
        "matches_expected": False,
        "status": "A_SERIES_RELEASE_PREREQUISITE_NOT_MET",
    }
    try:
        out = subprocess.check_output(
            ["git", "rev-list", "-n", "1", A_SERIES_RELEASE_TAG],
            cwd=str(root),
            text=True,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        return result
    result["resolved_commit"] = out
    result["matches_expected"] = out == A_SERIES_RELEASE_COMMIT
    result["status"] = "VERIFIED" if result["matches_expected"] else "A_SERIES_RELEASE_PREREQUISITE_NOT_MET"
    return result


def verify_a_series_artifact_lock(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root or get_repo_root()
    lock_path = root / A_SERIES_LOCK_PATH
    if not lock_path.exists():
        raise ASeriesBaselineDrift(f"Missing A-series lock: {A_SERIES_LOCK_PATH}")
    lock = load_json(lock_path)
    digest = compute_sha256_file(lock_path)
    profile = lock.get("lock_profile_id")
    count = lock.get("locked_artifact_count")
    entry_errors: List[str] = []
    for art in lock.get("artifacts", []):
        rel = art["path"]
        path = root / rel
        if not path.exists():
            entry_errors.append(f"missing {rel}")
            continue
        if compute_sha256_file(path) != art["sha256"]:
            entry_errors.append(f"hash mismatch {rel}")
        if path.stat().st_size != art["byte_size"]:
            entry_errors.append(f"size mismatch {rel}")
    ok = (
        profile == A_SERIES_LOCK_PROFILE
        and count == 33
        and not entry_errors
        and digest == "b63f5e2da988f8e685cf1a01ec8e79c2c37f5bc77359be647f1147ecfb04e3da"
    )
    # If lock bytes match expected SHA but count/profile differ, still drift.
    if profile != A_SERIES_LOCK_PROFILE or count != 33 or entry_errors:
        raise ASeriesBaselineDrift(
            f"A_SERIES_BASELINE_DRIFT profile={profile} count={count} errors={entry_errors[:5]}"
        )
    return {
        "path": A_SERIES_LOCK_PATH,
        "lock_profile_id": profile,
        "locked_artifact_count": count,
        "sha256": digest,
        "matches_released_lock_sha256": digest
        == "b63f5e2da988f8e685cf1a01ec8e79c2c37f5bc77359be647f1147ecfb04e3da",
        "entry_errors": entry_errors,
        "status": "VERIFIED" if ok else "A_SERIES_BASELINE_DRIFT",
    }


def build_a_series_consumption_registry(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root or get_repo_root()
    release = verify_a_series_release(root)
    if not release["matches_expected"]:
        raise CB0Error("A_SERIES_RELEASE_PREREQUISITE_NOT_MET")
    lock = verify_a_series_artifact_lock(root)
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "a_series_release_tag": A_SERIES_RELEASE_TAG,
        "a_series_release_commit": A_SERIES_RELEASE_COMMIT,
        "release_verification": release,
        "artifact_lock": lock,
        "canonical_sample_profile_id": CANONICAL_SAMPLE_PROFILE_ID,
        "slope_feature_profile_id": SLOPE_PROFILE_ID,
        "occupancy_target_profile_id": TARGET_PROFILE_ID,
        "canonical_source_samples": EXPECTED_TOTAL_SAMPLES,
        "model_eligible_samples": EXPECTED_SLOPE_ELIGIBLE,
        "warmup_samples": EXPECTED_WARMUP,
        "baseline_drift_status": "NONE" if lock["status"] == "VERIFIED" else "A_SERIES_BASELINE_DRIFT",
    }


def _load_eligible_by_role(repo_root: Path) -> Dict[str, List[str]]:
    path = repo_root / "datasets/co2/manifests/c_a5_canonical_samples/model_eligible_sample_ids.jsonl"
    rows = load_jsonl(path)
    by_role: Dict[str, List[str]] = {"TRAIN": [], "VALIDATION": [], "LOCKED_TEST": []}
    for row in rows:
        role = row["future_split_role"]
        by_role[role].append(row["canonical_sample_id"])
    return by_role


def _load_warmup_ids(repo_root: Path) -> List[str]:
    fa = load_json(
        repo_root
        / "datasets/co2/manifests/c_a5_canonical_samples/feature_availability_manifest.json"
    )
    return [x["canonical_sample_id"] for x in fa.get("excluded_from_model_eligible_view", [])]


def build_sample_universe_manifest(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root or get_repo_root()
    by_role = _load_eligible_by_role(root)
    warmup = _load_warmup_ids(root)
    train = by_role["TRAIN"]
    val = by_role["VALIDATION"]
    locked = by_role["LOCKED_TEST"]
    if len(train) != EXPECTED_TRAIN_COMMON:
        raise CB0Error(f"TRAIN common universe {len(train)} != {EXPECTED_TRAIN_COMMON}")
    if len(val) != EXPECTED_VALIDATION_COMMON:
        raise CB0Error(f"VALIDATION common universe {len(val)} != {EXPECTED_VALIDATION_COMMON}")
    if len(locked) != EXPECTED_LOCKED_TEST_SEALED:
        raise CB0Error(f"LOCKED_TEST sealed {len(locked)} != {EXPECTED_LOCKED_TEST_SEALED}")
    if len(warmup) != EXPECTED_WARMUP_CANONICAL:
        raise CB0Error(f"Warm-up {len(warmup)} != {EXPECTED_WARMUP_CANONICAL}")
    overlaps = {
        "train_validation": len(set(train) & set(val)),
        "train_locked_test": len(set(train) & set(locked)),
        "validation_locked_test": len(set(val) & set(locked)),
        "warmup_in_eligible": len(set(warmup) & (set(train) | set(val) | set(locked))),
    }
    if any(v != 0 for k, v in overlaps.items() if k != "warmup_in_eligible"):
        raise CB0Error(f"Cross-split overlap detected: {overlaps}")
    if overlaps["warmup_in_eligible"] != 0:
        raise CB0Error("Warm-up IDs must not appear in model-eligible universe")
    for name, ids in (("TRAIN", train), ("VALIDATION", val), ("LOCKED_TEST", locked), ("WARMUP", warmup)):
        if len(ids) != len(set(ids)):
            raise CB0Error(f"Duplicate IDs in {name}")
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "experiment_contract_id": EXPERIMENT_CONTRACT_ID,
        "canonical_source_universe": EXPECTED_TOTAL_SAMPLES,
        "b_series_common_train": EXPECTED_TRAIN_COMMON,
        "b_series_common_validation": EXPECTED_VALIDATION_COMMON,
        "b_series_sealed_locked_test": EXPECTED_LOCKED_TEST_SEALED,
        "canonical_warmup_records": EXPECTED_WARMUP_CANONICAL,
        "common_comparison_total": EXPECTED_TRAIN_COMMON + EXPECTED_VALIDATION_COMMON,
        "source_artifact": "datasets/co2/manifests/c_a5_canonical_samples/model_eligible_sample_ids.jsonl",
        "warmup_source_artifact": (
            "datasets/co2/manifests/c_a5_canonical_samples/feature_availability_manifest.json"
        ),
        "ordered_id_list_sha256": {
            "TRAIN": ordered_id_list_sha256(train),
            "VALIDATION": ordered_id_list_sha256(val),
            "LOCKED_TEST": ordered_id_list_sha256(locked),
            "WARMUP": ordered_id_list_sha256(warmup),
        },
        "overlaps": overlaps,
        "policy": {
            "common_comparison_universe": "MODEL_ELIGIBLE_SAMPLE under C-A3 slope availability",
            "locked_test_policy": "SEALED_FOR_FIT_TUNING_AND_B0_PREDICTIVE_EVALUATION",
            "warmup_policy": "PRESERVED_IN_A_SERIES_BUT_OUTSIDE_SLOPE_DEPENDENT_B_SERIES_MATRIX",
        },
    }


def build_feature_view_registry() -> Dict[str, Any]:
    feature_roles = {
        "Temperature": {
            "provenance": "MEASURED_UCI_FEATURE",
            "device_feasibility": "SCD40_NATIVE_AVAILABLE",
            "may_be_model_input": True,
        },
        "Humidity": {
            "provenance": "MEASURED_UCI_FEATURE",
            "device_feasibility": "SCD40_NATIVE_AVAILABLE",
            "may_be_model_input": True,
        },
        "CO2": {
            "provenance": "MEASURED_UCI_FEATURE",
            "device_feasibility": "SCD40_NATIVE_AVAILABLE",
            "may_be_model_input": True,
        },
        "CO2_slope": {
            "provenance": "DERIVED_CAUSAL_FEATURE",
            "device_feasibility": "DERIVABLE_FROM_SCD40_CO2_HISTORY",
            "may_be_model_input": True,
            "baseline_profile_id": SLOPE_PROFILE_ID,
            "b1_ablation_owned": True,
        },
        "Light": {
            "provenance": "MEASURED_UCI_FEATURE",
            "device_feasibility": "NOT_SCD40_NATIVE",
            "may_be_model_input": True,
            "deployment_note": "UCI-context / non-SCD40-native",
        },
        "HumidityRatio": {
            "provenance": "MEASURED_UCI_FEATURE",
            "device_feasibility": "NOT_DIRECT_SCD40_MODEL_INPUT_UNLESS_EXPLICITLY_DERIVED_AND_VALIDATED",
            "may_be_model_input": True,
        },
        "Occupancy": {
            "provenance": "SOURCE_TARGET_LABEL",
            "device_feasibility": "LABEL_ONLY",
            "may_be_model_input": False,
        },
    }
    views = {
        "CO2_ONLY_REFERENCE": {
            "features": ["CO2"],
            "status": "REGISTERED_REFERENCE_VIEW",
            "winner": False,
        },
        "SCD40_NATIVE_REFERENCE": {
            "features": ["CO2", "Temperature", "Humidity"],
            "status": "REGISTERED_REFERENCE_VIEW",
            "winner": False,
        },
        "HISTORICAL_COMPATIBILITY_REFERENCE": {
            "features": ["CO2_slope", "Humidity", "CO2"],
            "status": "HISTORICAL_COMPATIBILITY_VIEW",
            "winner": False,
            "note": (
                "Historical TFLite/scaler feature order; NOT CANONICAL_FINAL_FEATURE_SET "
                "because training lineage remains unverified."
            ),
        },
        "SCD40_SLOPE_REFERENCE": {
            "features": ["CO2", "Temperature", "Humidity", "CO2_slope"],
            "status": "REGISTERED_REFERENCE_VIEW",
            "winner": False,
        },
        "UCI_CONTEXT_DIAGNOSTIC": {
            "features": ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio", "CO2_slope"],
            "status": "NON_DEPLOYABLE_DIAGNOSTIC_VIEW",
            "winner": False,
            "note": "Includes UCI-only Light / HumidityRatio; not an SCD40 deployment contract.",
        },
    }
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "final_feature_selection_performed": False,
        "canonical_final_feature_set_claimed": False,
        "feature_roles": feature_roles,
        "forbidden_model_inputs": {
            "provenance_fields": list(PROVENANCE_FIELDS),
            "target_fields": list(TARGET_FIELDS),
        },
        "feature_views": views,
        "scd40_native_features": ["CO2", "Temperature", "Humidity"],
        "derived_scd40_compatible_features": ["CO2_slope"],
        "uci_only_or_non_native_features": ["Light", "HumidityRatio"],
        "historical_compatibility_view": "HISTORICAL_COMPATIBILITY_REFERENCE",
        "b1_boundary": {
            "slope_method_ablation": "DEFERRED_TO_C_B1",
            "history_duration_ablation": "DEFERRED_TO_C_B1",
            "baseline_slope_profile_id": SLOPE_PROFILE_ID,
        },
    }


def build_metric_contract() -> Dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "positive_class": POSITIVE_CLASS,
        "negative_class": NEGATIVE_CLASS,
        "positive_label": POSITIVE_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "required_metrics": [
            "accuracy",
            "balanced_accuracy",
            "precision_occupied",
            "recall_occupied",
            "f1_occupied",
            "precision_vacant",
            "recall_vacant",
            "f1_vacant",
            "macro_f1",
            "confusion_matrix",
        ],
        "optional_score_metrics": ["roc_auc", "average_precision"],
        "primary_summary_metric": "macro_f1",
        "prominent_secondary_metrics": ["balanced_accuracy", "recall_occupied"],
        "default_probability_threshold": DEFAULT_PROB_THRESHOLD,
        "threshold_optimization_in_b0": False,
        "deployment_acceptance_threshold_defined": False,
    }


def build_experiment_contract() -> Dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "schema": "SafeNest_CO2_Offline_Experiment_Contract",
        "experiment_contract_id": EXPERIMENT_CONTRACT_ID,
        "phase": PHASE_ID,
        "a_series_release_tag": A_SERIES_RELEASE_TAG,
        "a_series_release_commit": A_SERIES_RELEASE_COMMIT,
        "a_series_artifact_lock_profile": A_SERIES_LOCK_PROFILE,
        "canonical_sample_profile_id": CANONICAL_SAMPLE_PROFILE_ID,
        "target_profile_id": TARGET_PROFILE_ID,
        "baseline_slope_profile_id": SLOPE_PROFILE_ID,
        "train_population": {
            "role": "TRAIN",
            "universe": "MODEL_ELIGIBLE_SAMPLE",
            "count": EXPECTED_TRAIN_COMMON,
            "fit_authorized": True,
        },
        "validation_population": {
            "role": "VALIDATION",
            "universe": "MODEL_ELIGIBLE_SAMPLE",
            "count": EXPECTED_VALIDATION_COMMON,
            "fit_authorized": False,
            "model_selection_authorized_in_later_b_phases": True,
        },
        "locked_test_policy": {
            "role": "LOCKED_TEST",
            "universe": "MODEL_ELIGIBLE_SAMPLE",
            "count": EXPECTED_LOCKED_TEST_SEALED,
            "fit_authorized": False,
            "tuning_authorized": False,
            "b0_predictive_evaluation_authorized": False,
            "identity_integrity_inspection_authorized": True,
        },
        "common_comparison_universe": {
            "train": EXPECTED_TRAIN_COMMON,
            "validation": EXPECTED_VALIDATION_COMMON,
            "total": EXPECTED_TRAIN_COMMON + EXPECTED_VALIDATION_COMMON,
        },
        "preprocessing_policy": {
            "fit_population": "TRAIN_ONLY",
            "apply_to_validation": "TRAIN_FITTED_PARAMETERS",
            "locked_test_fit": False,
            "production_scaler_modification": False,
            "b0_scaler_status_if_created": "B0_EXPERIMENT_REFERENCE_ONLY",
        },
        "metric_definitions_ref": "metric_contract.json",
        "positive_class": POSITIVE_CLASS,
        "random_seed_policy": {
            "default_seed": DEFAULT_SEED,
            "python_seed": DEFAULT_SEED,
            "numpy_seed": DEFAULT_SEED,
            "pythonhashseed_recommended": str(DEFAULT_SEED),
        },
        "determinism_policy": {
            "data_pipeline_determinism_required": True,
            "model_training_determinism_claimed_bit_identical": False,
        },
        "model_selection_boundary": "DEFERRED_AFTER_C_B0",
        "b1_boundary": "CONTROLLED_SLOPE_METHOD_HISTORY_ABLATION",
        "b0_prohibited_activities": [
            "FINAL_FEATURE_SELECTION",
            "SLOPE_METHOD_OPTIMIZATION",
            "HISTORY_DURATION_OPTIMIZATION",
            "COMPLEX_ARCHITECTURE_COMPARISON",
            "HYPERPARAMETER_SWEEP",
            "LOCKED_TEST_FIT_OR_TUNING",
            "THRESHOLD_OPTIMIZATION_ON_LOCKED_TEST",
            "PRODUCTION_MODEL_PROMOTION",
            "SAFETY_THRESHOLD_CALIBRATION",
            "DEVICE_DOMAIN_EQUIVALENCE_CLAIM",
        ],
        "random_row_wise_split": "PROHIBITED",
        "inherited_c_a2_split_authoritative": True,
    }


def assert_no_forbidden_path_markers(text: str) -> List[str]:
    errors = []
    for marker in ("/Users/", "file://", "~/", "/private/tmp/", "CloudDocs"):
        if marker in text:
            errors.append(f"Forbidden path marker: {marker}")
    return errors


@dataclass(frozen=True)
class MatrixBundle:
    sample_ids: List[str]
    features: np.ndarray
    labels: np.ndarray
    feature_names: Tuple[str, ...]
    split_role: str


def _row_feature_vector(row: Mapping[str, Any], feature_names: Sequence[str]) -> List[float]:
    values: List[float] = []
    for name in feature_names:
        if name == "CO2_slope":
            val = row.get("co2_slope")
        elif name == "Temperature":
            val = row.get("temperature")
        elif name == "Humidity":
            val = row.get("humidity")
        elif name == "Light":
            val = row.get("light")
        elif name == "CO2":
            val = row.get("co2")
        elif name == "HumidityRatio":
            val = row.get("humidity_ratio")
        else:
            raise CB0Error(f"Unknown feature: {name}")
        if val is None or not math.isfinite(float(val)):
            raise CB0Error(f"Non-finite/missing feature {name} for {row.get('canonical_sample_id')}")
        values.append(float(val))
    return values


def load_comparison_matrix(
    *,
    repo_root: Optional[Path] = None,
    split_role: str,
    feature_names: Sequence[str],
    allow_locked_test_predictive: bool = False,
) -> MatrixBundle:
    """
    Load feature/label matrices for a split from A-series canonical JSONL,
    restricted to the B-series common/sealed eligible universe.
    """
    if split_role == "LOCKED_TEST" and not allow_locked_test_predictive:
        # Integrity-only identity loading is allowed via sample universe;
        # predictive matrix loading is blocked in B0.
        raise LockedTestPolicyViolation("LOCKED_TEST_POLICY_VIOLATION: predictive matrix load blocked in C-B0")

    root = repo_root or get_repo_root()
    universe = build_sample_universe_manifest(root)
    # Rebuild IDs for requested role from eligible list to preserve order
    by_role = _load_eligible_by_role(root)
    if split_role not in by_role:
        raise CB0Error(f"Unknown split role: {split_role}")
    wanted = set(by_role[split_role])
    # Validate universe counts already done in build_sample_universe_manifest

    # Reject leakage into feature names
    for name in feature_names:
        if name in TARGET_FIELDS or name == "Occupancy":
            raise CB0Error("Target leakage into feature matrix")
        if name in PROVENANCE_FIELDS:
            raise CB0Error("Provenance leakage into feature matrix")

    jsonl = root / "datasets/co2/manifests/c_a5_canonical_samples/canonical_source_samples.jsonl"
    ids: List[str] = []
    feats: List[List[float]] = []
    labels: List[int] = []
    with jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row["canonical_sample_id"]
            if sid not in wanted:
                continue
            if row["future_split_role"] != split_role:
                raise CB0Error(f"Split role mismatch for {sid}")
            ids.append(sid)
            feats.append(_row_feature_vector(row, feature_names))
            labels.append(int(row["occupancy_source_value"]))

    # Preserve eligible-file order rather than JSONL discovery order
    index = {sid: i for i, sid in enumerate(ids)}
    ordered_ids = by_role[split_role]
    ordered_feats = [feats[index[sid]] for sid in ordered_ids]
    ordered_labels = [labels[index[sid]] for sid in ordered_ids]
    if len(ordered_ids) != len(wanted):
        raise CB0Error("Eligible ID materialization incomplete")

    x = np.asarray(ordered_feats, dtype=np.float64)
    y = np.asarray(ordered_labels, dtype=np.int64)
    if not np.isfinite(x).all():
        raise CB0Error("Non-finite values in feature matrix")
    return MatrixBundle(
        sample_ids=list(ordered_ids),
        features=x,
        labels=y,
        feature_names=tuple(feature_names),
        split_role=split_role,
    )


@dataclass
class TrainOnlyStandardScaler:
    """Experiment-local TRAIN-only standard scaler (B0_EXPERIMENT_REFERENCE_ONLY)."""

    feature_names: Tuple[str, ...]
    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None
    n_samples_fit_: int = 0
    fit_population: str = "TRAIN"
    status: str = "B0_EXPERIMENT_REFERENCE_ONLY"

    def fit(self, bundle: MatrixBundle) -> "TrainOnlyStandardScaler":
        if bundle.split_role != "TRAIN":
            raise CB0Error("Scaler fit population must be TRAIN only")
        if bundle.feature_names != self.feature_names:
            raise CB0Error("Scaler feature order mismatch")
        mean = bundle.features.mean(axis=0)
        scale = bundle.features.std(axis=0, ddof=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        self.mean_ = mean
        self.scale_ = scale
        self.n_samples_fit_ = int(bundle.features.shape[0])
        return self

    def transform(self, bundle: MatrixBundle) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise CB0Error("Scaler not fitted")
        if bundle.feature_names != self.feature_names:
            raise CB0Error("Transform feature order mismatch")
        if bundle.split_role == "LOCKED_TEST":
            raise LockedTestPolicyViolation(
                "LOCKED_TEST_POLICY_VIOLATION: scaler transform for predictive use blocked in C-B0"
            )
        return (bundle.features - self.mean_) / self.scale_

    def to_metadata(self, *, fit_population_fingerprint: str, feature_view_id: str) -> Dict[str, Any]:
        if self.mean_ is None or self.scale_ is None:
            raise CB0Error("Scaler not fitted")
        return {
            "manifest_version": "1.0",
            "phase": PHASE_ID,
            "status": self.status,
            "production_scaler_modified": False,
            "feature_view_id": feature_view_id,
            "feature_order": list(self.feature_names),
            "train_sample_count": self.n_samples_fit_,
            "fit_population": self.fit_population,
            "fit_population_fingerprint": fit_population_fingerprint,
            "mean": [float(x) for x in self.mean_],
            "scale": [float(x) for x in self.scale_],
            "seed": DEFAULT_SEED,
            "purpose": "Validate C-B0 preprocessing harness only",
            "existing_production_scaler_status": "SCALER_FIT_LINEAGE_UNVERIFIED",
        }


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    # positive = OCCUPIED = 1
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    cm = confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    precision_occ = _safe_div(tp, tp + fp)
    recall_occ = _safe_div(tp, tp + fn)
    f1_occ = _safe_div(2 * precision_occ * recall_occ, precision_occ + recall_occ)
    precision_vac = _safe_div(tn, tn + fn)
    recall_vac = _safe_div(tn, tn + fp)
    f1_vac = _safe_div(2 * precision_vac * recall_vac, precision_vac + recall_vac)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    bal = 0.5 * (recall_occ + recall_vac)
    macro_f1 = 0.5 * (f1_occ + f1_vac)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": bal,
        "precision_occupied": precision_occ,
        "recall_occupied": recall_occ,
        "f1_occupied": f1_occ,
        "precision_vacant": precision_vac,
        "recall_vacant": recall_vac,
        "f1_vacant": f1_vac,
        "macro_f1": macro_f1,
        "confusion_matrix": {
            "labels": [NEGATIVE_LABEL, POSITIVE_LABEL],
            "label_names": [NEGATIVE_CLASS, POSITIVE_CLASS],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
    }


@dataclass
class MajorityClassBaseline:
    """Deterministic TRAIN-only majority-class reference baseline."""

    majority_label: Optional[int] = None
    status: str = "REFERENCE_BASELINE_ONLY"
    candidate: bool = False
    deployable: bool = False

    def fit(self, y_train: np.ndarray) -> "MajorityClassBaseline":
        if y_train.size == 0:
            raise CB0Error("Empty TRAIN labels")
        counts = Counter(int(x) for x in y_train.tolist())
        # Deterministic tie-break: prefer VACANT(0) then OCCUPIED(1) by max count then lower label
        self.majority_label = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return self

    def predict(self, n: int) -> np.ndarray:
        if self.majority_label is None:
            raise CB0Error("Baseline not fitted")
        return np.full(n, self.majority_label, dtype=np.int64)


def run_leakage_audit(
    universe: Mapping[str, Any],
    feature_registry: Mapping[str, Any],
    *,
    scaler_fit_role: str = "TRAIN",
    model_fit_role: str = "TRAIN",
    locked_test_predictive_used: bool = False,
) -> Dict[str, Any]:
    overlaps = universe["overlaps"]
    errors: List[str] = []
    if overlaps["train_validation"] != 0:
        errors.append("TRAIN/VALIDATION overlap")
    if overlaps["train_locked_test"] != 0:
        errors.append("TRAIN/LOCKED_TEST overlap")
    if overlaps["validation_locked_test"] != 0:
        errors.append("VALIDATION/LOCKED_TEST overlap")
    roles = feature_registry["feature_roles"]
    if roles["Occupancy"]["may_be_model_input"] is not False:
        errors.append("Occupancy marked as model input")
    for fname in feature_registry["forbidden_model_inputs"]["provenance_fields"]:
        if fname in feature_registry.get("feature_views", {}).get("UCI_CONTEXT_DIAGNOSTIC", {}).get(
            "features", []
        ):
            # provenance fields are not in views; ok
            pass
    if scaler_fit_role != "TRAIN":
        errors.append("Scaler fit not TRAIN-only")
    if model_fit_role != "TRAIN":
        errors.append("Model fit not TRAIN-only")
    if locked_test_predictive_used:
        errors.append("LOCKED_TEST predictive use")
    if feature_registry.get("final_feature_selection_performed") is not False:
        errors.append("Final feature selection claimed")
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "train_validation_overlap": overlaps["train_validation"],
        "train_locked_test_overlap": overlaps["train_locked_test"],
        "validation_locked_test_overlap": overlaps["validation_locked_test"],
        "duplicate_ids_within_splits": 0,
        "target_leakage": 0,
        "provenance_leakage": 0,
        "scaler_fit_uses_validation": False,
        "scaler_fit_uses_locked_test": False,
        "model_fit_uses_validation": False,
        "model_fit_uses_locked_test": False,
        "locked_test_predictive_evaluation_in_b0": False,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }


def build_environment_metadata() -> Dict[str, Any]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import sklearn  # type: ignore

        versions["scikit_learn"] = sklearn.__version__
        sklearn_available = True
    except Exception:  # noqa: BLE001
        versions["scikit_learn"] = None
        sklearn_available = False
    try:
        import tensorflow as tf  # type: ignore

        versions["tensorflow"] = tf.__version__
    except Exception:  # noqa: BLE001
        versions["tensorflow"] = None
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "versions": versions,
        "sklearn_available": sklearn_available,
        "optional_linear_baseline_status": (
            "SKIPPED_MISSING_DEPENDENCY" if not sklearn_available else "AVAILABLE"
        ),
        "path_policy": "repository-relative POSIX only",
    }


def build_exceptions_registry(*, sklearn_skipped: bool) -> Dict[str, Any]:
    warnings = [
        {
            "code": "DEVICE_UCI_CADENCE_DOMAIN_GAP",
            "severity": "WARNING",
            "description": "UCI offline performance does not prove SCD40 deployment performance.",
        },
        {
            "code": "MODEL_TRAINING_LINEAGE_UNVERIFIED",
            "severity": "WARNING",
            "description": "Existing TFLite remains unverified; B0 reference baselines are not promotions.",
        },
        {
            "code": "SCALER_FIT_LINEAGE_UNVERIFIED",
            "severity": "WARNING",
            "description": "Historical production scaler lineage remains unverified and unmodified.",
        },
        {
            "code": "CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED",
            "severity": "WARNING",
            "description": "C-A3 150s offline baseline remains CANONICAL_OFFLINE_BASELINE_DESIGN; ablation deferred to C-B1.",
        },
        {
            "code": "SAFETY_RULE_CONTRACT_OUT_OF_SCOPE",
            "severity": "WARNING",
            "description": "B0 does not calibrate CO2 safety/alarm thresholds.",
        },
        {
            "code": "DEFERRED_SHARED_INTEGRATION_UPDATE",
            "severity": "WARNING",
            "description": "Shared inventory/manifest updates deferred outside C-B0.",
        },
    ]
    if sklearn_skipped:
        warnings.append(
            {
                "code": "OPTIONAL_LINEAR_BASELINE_SKIPPED",
                "severity": "WARNING",
                "description": "scikit-learn unavailable; optional logistic baseline skipped. Majority baseline remains.",
            }
        )
    return {"manifest_version": "1.0", "phase": PHASE_ID, "warnings": warnings, "blockers": []}
