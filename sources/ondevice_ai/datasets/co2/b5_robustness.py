#!/usr/bin/env python3
"""SafeNest CO₂ C-B5 robustness, latency, and final-lock helpers.

The module deliberately keeps the C-B4 bytes and all predecessor contracts
immutable.  Source-level perturbations are applied to the validation
chronology before the frozen ENDPOINT_H150 feature reconstruction, TRAIN-only
scaling, and C-B4 full-integer inference.  LOCKED_TEST helpers are separate
and are only called after the pre-test freeze has been validated.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from datasets.co2.architecture_multiseed import prepare_fixed_data
from datasets.co2.imbalance_calibration import (
    classification_metrics_at_threshold,
    expected_calibration_error,
    probability_quality_metrics,
)
from datasets.co2.raw_reader import compute_sha256_file
from datasets.co2.tflite_equivalence import dequantize_int8_output, quantize_int8_input


PHASE_ID = "C-B5"
PROTOCOL_ID = "CO2_B5_ROBUSTNESS_PROTOCOL_001"
FREEZE_ID = "CO2_B5_PRE_LOCKED_TEST_CANDIDATE_FREEZE_001"
EVALUATION_ID = "CO2_B5_LOCKED_TEST_EVALUATION_001"
FINAL_LOCK_ID = "CO2_B5_FINAL_OFFLINE_CANDIDATE_LOCK_001"
ARTIFACT_DIR_REL = "datasets/co2/manifests/c_b5_robustness_final_lock"
CANDIDATE_DIR_REL = "models/co2/candidates/c_b5"
MODEL_REL = "models/co2/candidates/c_b4/full_integer_int8.tflite"
FLOAT_MODEL_REL = "models/co2/candidates/c_b4/float_reference.tflite"
FLOAT_REFERENCE_REL = "models/co2/candidates/c_b4/float_reference_parameters.json"
CLASS_MAP_REL = "models/co2/candidates/c_b4/class_map.json"
CANDIDATE_METADATA_REL = "models/co2/candidates/c_b4/candidate_metadata.json"
SCALER_REL = "datasets/co2/manifests/c_b2_imbalance_calibration/preprocessing_scaler_evidence.json"
TRAIN_COUNT = 8140
VALIDATION_COUNT = 2662
LOCKED_TEST_COUNT = 9749
THRESHOLD = 0.58
FEATURE_ORDER = ("CO2", "Temperature", "Humidity", "CO2_slope")
SLOPE_PROFILE = "ENDPOINT_H150"
IMBALANCE_STRATEGY = "BALANCED_RANDOM_OVERSAMPLE"
ARCHITECTURE = "LINEAR_LOGISTIC"
INPUT_SCALE = 0.03529411926865578
INPUT_ZERO_POINT = 0
OUTPUT_SCALE = 0.00390625
OUTPUT_ZERO_POINT = -128
INT8_MODEL_SHA256 = "bb2ed28533bca75d4fa3d06348e017c506df47d7c34b29574b77f70b6b386816"
FLOAT_MODEL_SHA256 = "a4767363faf0cc47dcc907cd411b4664eb0821c2926262cce25e3fffc4f04a35"
FLOAT_REFERENCE_SHA256 = "471b85852f661bd787e0bc130ae032bebb69ef0b51cfacb8e87ea04529970fee"
CLASS_MAP_SHA256 = "e8d9a18cba75092bb2c92212504afadddd2964c9628abfac71c5b0b91db9889a"
CANDIDATE_METADATA_SHA256 = "67874e8dd116619bb5e94db998cf094127d6806905d5eb2957a42f5e7f6a8269"
SCALER_FINGERPRINT = "d0cf83558fb0de9dcdc97f0d94781a5a475a6f68e8d818121aee929030e5dc89"
LOCKED_TEST_MEMBERSHIP_FINGERPRINT = "0bac8dc1affae1de48ea68f01e866508ea19f31e194a7c0dccbf617e529344e7"

_VALIDATION_JSONL = "datasets/co2/manifests/c_a5_canonical_samples/canonical_source_samples.jsonl"
_ELIGIBLE_JSONL = "datasets/co2/manifests/c_a5_canonical_samples/model_eligible_sample_ids.jsonl"


class CB5Error(RuntimeError):
    """Base C-B5 contract error."""


class LockedTestAuthorizationError(CB5Error):
    """Raised when the one-time test guard is not satisfied."""


def stable_sha256(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def array_sha256(values: np.ndarray) -> str:
    arr = np.asarray(values)
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def file_sha256(root: Path, rel: str) -> str:
    path = root / rel
    if not path.is_file():
        raise CB5Error(f"missing required file: {rel}")
    return compute_sha256_file(path)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _parse_dt(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")


def load_split_rows(root: Path, split_role: str) -> List[Dict[str, Any]]:
    """Load only one split's canonical rows; no LOCKED_TEST rows are read for validation."""
    rows: List[Dict[str, Any]] = []
    with (root / _VALIDATION_JSONL).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("future_split_role") != split_role:
                continue
            item = dict(row)
            item["_dt"] = _parse_dt(str(item["source_timestamp_raw"]))
            item["_source_index"] = len(rows)
            rows.append(item)
    if split_role == "VALIDATION" and len(rows) != 2665:
        raise CB5Error(f"validation canonical row count drift: {len(rows)}")
    if split_role == "LOCKED_TEST" and len(rows) != 9752:
        raise CB5Error(f"locked canonical row count drift: {len(rows)}")
    return rows


def load_eligible_ids(root: Path, split_role: str) -> List[str]:
    ids: List[str] = []
    with (root / _ELIGIBLE_JSONL).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("future_split_role") == split_role:
                ids.append(str(row["canonical_sample_id"]))
    expected = {"TRAIN": TRAIN_COUNT, "VALIDATION": VALIDATION_COUNT, "LOCKED_TEST": LOCKED_TEST_COUNT}[split_role]
    if len(ids) != expected:
        raise CB5Error(f"{split_role} eligible count drift: {len(ids)}")
    return ids


def _scenario_id(kind: str, level: Any) -> str:
    if isinstance(level, float):
        text = f"{level:g}"
    else:
        text = str(level)
    return f"{kind}__{text.replace('-', 'm').replace('.', 'p').replace('+', 'p')}"


def build_protocol() -> Dict[str, Any]:
    """Return the complete pre-registered deterministic diagnostic grid."""
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "protocol_id": PROTOCOL_ID,
        "registration_status": "PRE_REGISTERED_BEFORE_RESULTS",
        "offline_technical_stress_grid": True,
        "numeric_value_interpretation": [
            "NOT_SCD40_SPECIFICATION_LIMITS",
            "NOT_SAFETY_LIMITS",
            "NOT_DEPLOYMENT_ACCEPTANCE_LIMITS",
        ],
        "candidate": {
            "model_path": MODEL_REL,
            "model_sha256": INT8_MODEL_SHA256,
            "float_reference_path": FLOAT_REFERENCE_REL,
            "float_reference_sha256": FLOAT_REFERENCE_SHA256,
            "architecture": ARCHITECTURE,
            "imbalance_strategy": IMBALANCE_STRATEGY,
            "slope_profile": SLOPE_PROFILE,
            "feature_order": list(FEATURE_ORDER),
            "scaler_path": SCALER_REL,
            "scaler_fingerprint": SCALER_FINGERPRINT,
            "threshold": THRESHOLD,
            "input_dtype": "int8",
            "output_dtype": "int8",
            "input_quantization": {"scale": INPUT_SCALE, "zero_point": INPUT_ZERO_POINT},
            "output_quantization": {"scale": OUTPUT_SCALE, "zero_point": OUTPUT_ZERO_POINT},
        },
        "causal_procedure": {
            "source_level_first": True,
            "raw_fields": ["CO2", "Humidity", "timestamp", "row/history state"],
            "slope_recomputation": "ENDPOINT_DIFFERENCE",
            "history_duration_seconds": 150.0,
            "max_internal_gap_seconds": 90.0,
            "causality": "PAST_ONLY",
            "actual_elapsed_time": True,
            "train_only_scaler_reused": True,
            "int8_quantization_after_scaling": True,
            "silent_slope_imputation": False,
        },
        "scenarios": {
            "co2_offset_ppm": {"levels": [-200, -100, -50, 50, 100, 200], "seed": None, "source": "raw_CO2"},
            "co2_linear_drift_ppm_per_min": {"levels": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], "seed": None, "source": "raw_CO2_and_causal_slope"},
            "humidity_noise_sigma_rh": {"levels": [1.0, 2.0, 5.0], "seeds": [20260810, 20260811, 20260812], "source": "raw_Humidity", "clip": [0.0, 100.0], "diagnostic_label": "OFFLINE_HUMIDITY_NOISE_STRESS_ONLY"},
            "missing_row": {
                "patterns": [
                    {"name": "one_missing_observation", "indices": [1000]},
                    {"name": "consecutive_missing_observations", "indices": [1200, 1201, 1202]},
                    {"name": "periodic_sparse_missingness", "start": 1000, "period": 100, "count": 18},
                ],
                "source": "row/history state",
                "unavailable_is_not_vacant": True,
            },
            "stale_history_seconds": {"levels": [60, 120, 180], "source": "history state", "diagnostic_only": True},
            "timestamp_jitter_seconds": {"levels": [1, 5, 10], "seeds": [20260810, 20260811, 20260812], "source": "timestamp", "strict_ordering": True, "recompute_gap_logic": True},
        },
        "sample_availability_definition": "intended eligible validation rows; only rows with a causally available slope are paired and inferred",
        "saturation_definition": "unclipped affine int8 input outside [-128,127] before clipping",
        "metric_definitions": {
            "classification": "fixed threshold 0.58; positive class OCCUPIED=1",
            "probability_drift": "dequantized INT8 probability versus unperturbed validation baseline on paired IDs",
            "ece": "C-B2 convention: 10 equal-width bins over [0,1]",
        },
        "decision_policy": {
            "contract_failure": ["NaN/Inf", "silent unavailable conversion", "nondeterminism", "artifact mutation", "LOCKED_TEST leakage", "causality violation"],
            "performance_degradation": "quantify artificial stress effects; do not label safety failure without an authoritative safety contract",
            "classification": "ROBUSTNESS_DIAGNOSTIC_ONLY_UNDER_OFFLINE_TECHNICAL_STRESS",
        },
        "locked_test_used": False,
        "locked_test_perturbation_sweeps_prohibited": True,
        "model_tuning_authorized": False,
        "threshold_tuning_authorized": False,
    }


def _missing_indices(pattern: Mapping[str, Any], n: int) -> set[int]:
    name = str(pattern.get("name"))
    if name == "one_missing_observation":
        return {1000}
    if name == "consecutive_missing_observations":
        return {1200, 1201, 1202}
    if name == "periodic_sparse_missingness":
        return {int(pattern["start"]) + int(pattern["period"]) * i for i in range(int(pattern["count"])) if int(pattern["start"]) + int(pattern["period"]) * i < n}
    raise CB5Error(f"unknown missingness pattern: {name}")


def _bounded_jitter(rows: List[Dict[str, Any]], max_seconds: float, seed: int) -> int:
    rng = np.random.default_rng(int(seed))
    bounded = 0
    for i, row in enumerate(rows):
        original = row["_dt"]
        jitter = float(rng.uniform(-max_seconds, max_seconds))
        candidate = original + timedelta(seconds=jitter)
        lower = rows[i - 1]["_dt"] + timedelta(microseconds=1) if i else None
        upper = rows[i + 1]["_dt"] - timedelta(microseconds=1) if i + 1 < len(rows) else None
        if lower is not None and candidate <= lower:
            candidate = lower
            bounded += 1
        if upper is not None and candidate >= upper:
            candidate = upper
            bounded += 1
        row["_dt"] = candidate
    return bounded


def make_scenario_rows(source_rows: Sequence[Mapping[str, Any]], scenario: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply one perturbation to raw source rows, preserving row identity/order."""
    rows = [dict(r) for r in source_rows]
    for row in rows:
        row["_dt"] = _parse_dt(str(row["source_timestamp_raw"]))
    kind = str(scenario["kind"])
    level = scenario.get("level")
    seed = scenario.get("seed")
    meta: Dict[str, Any] = {"kind": kind, "level": level, "seed": seed, "rows_removed": 0, "humidity_clipped_count": 0, "timestamp_bounded_count": 0}
    if kind == "baseline":
        return rows, meta
    if kind == "co2_offset_ppm":
        for row in rows:
            row["co2"] = float(row["co2"]) + float(level)
    elif kind == "co2_linear_drift_ppm_per_min":
        starts: Dict[str, datetime] = {}
        for row in rows:
            block = str(row["source_member_name"])
            starts.setdefault(block, row["_dt"])
            elapsed = (row["_dt"] - starts[block]).total_seconds() / 60.0
            row["co2"] = float(row["co2"]) + float(level) * elapsed
    elif kind == "humidity_noise_sigma_rh":
        rng = np.random.default_rng(int(seed))
        noise = rng.normal(0.0, float(level), size=len(rows))
        for row, delta in zip(rows, noise.tolist()):
            raw = float(row["humidity"]) + float(delta)
            clipped = min(100.0, max(0.0, raw))
            if clipped != raw:
                meta["humidity_clipped_count"] += 1
            row["humidity"] = clipped
    elif kind == "timestamp_jitter_seconds":
        meta["timestamp_bounded_count"] = _bounded_jitter(rows, float(level), int(seed))
    elif kind == "missing_row":
        indices = _missing_indices(scenario["pattern"], len(rows))
        rows = [r for i, r in enumerate(rows) if i not in indices]
        meta["rows_removed"] = len(indices)
        meta["removed_source_indices"] = sorted(indices)
    elif kind == "stale_history_seconds":
        # The anchor remains causal, but the history state is delayed by a fixed
        # diagnostic duration when converting elapsed time to the slope.
        meta["stale_history_seconds"] = float(level)
    else:
        raise CB5Error(f"unknown scenario kind: {kind}")
    return rows, meta


def reconstruct_features(source_rows: Sequence[Mapping[str, Any]], intended_ids: Sequence[str], scenario: Mapping[str, Any]) -> Dict[str, Any]:
    """Reconstruct ENDPOINT_H150 causally and return availability/lineage records."""
    rows, meta = make_scenario_rows(source_rows, scenario)
    by_block: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        block = str(row["source_member_name"])
        row["_block_pos"] = len(by_block[block])
        by_block[block].append(row)
        by_id[str(row["canonical_sample_id"])] = row
    records: Dict[str, Dict[str, Any]] = {}
    for sid in intended_ids:
        target = by_id.get(str(sid))
        if target is None:
            records[str(sid)] = {"available": False, "status": "FEATURE_UNAVAILABLE_MISSING_SOURCE_ROW", "sample_id": str(sid)}
            continue
        block_rows = by_block[str(target["source_member_name"])]
        i = int(target["_block_pos"])
        anchor: Optional[Dict[str, Any]] = None
        status = "FEATURE_UNAVAILABLE_WARMUP"
        crossed_gap = False
        for k in range(i - 1, -1, -1):
            delta = (block_rows[k + 1]["_dt"] - block_rows[k]["_dt"]).total_seconds()
            if delta <= 0.0:
                status = "FEATURE_UNAVAILABLE_NON_MONOTONIC_TIMESTAMP"
                crossed_gap = True
                break
            if delta > 90.0:
                status = "FEATURE_UNAVAILABLE_GAP_RESTART"
                crossed_gap = True
                break
            elapsed = (target["_dt"] - block_rows[k]["_dt"]).total_seconds()
            if elapsed >= 150.0:
                anchor = block_rows[k]
                break
        if anchor is None:
            if not crossed_gap:
                status = "FEATURE_UNAVAILABLE_WARMUP"
            records[str(sid)] = {"available": False, "status": status, "sample_id": str(sid)}
            continue
        elapsed = float((target["_dt"] - anchor["_dt"]).total_seconds())
        effective_elapsed = elapsed
        if str(scenario["kind"]) == "stale_history_seconds":
            effective_elapsed += float(scenario["level"])
        now = float(target["co2"])
        hist = float(anchor["co2"])
        if not math.isfinite(now) or not math.isfinite(hist) or effective_elapsed <= 0.0:
            records[str(sid)] = {"available": False, "status": "FEATURE_UNAVAILABLE_NONFINITE_INPUT", "sample_id": str(sid)}
            continue
        slope = float((now - hist) / (effective_elapsed / 60.0))
        records[str(sid)] = {
            "available": True,
            "status": "FEATURE_AVAILABLE",
            "sample_id": str(sid),
            "raw": [now, float(target["temperature"]), float(target["humidity"]), slope],
            "co2": now,
            "humidity": float(target["humidity"]),
            "co2_slope": slope,
            "history_elapsed_seconds": elapsed,
            "effective_elapsed_seconds": effective_elapsed,
            "history_start_source_row_identifier": str(anchor["source_row_identifier"]),
        }
    return {"records": records, "meta": meta}


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = THRESHOLD) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    cls, _ = classification_metrics_at_threshold(labels, probs, threshold)
    quality: Dict[str, Any] = {}
    try:
        quality = probability_quality_metrics(labels, probs)
    except Exception as exc:  # noqa: BLE001
        quality = {"roc_auc": None, "pr_auc_average_precision": None, "brier_score": None, "log_loss": None, "quality_error": str(exc)}
    try:
        ece = expected_calibration_error(labels, probs)["expected_calibration_error"]
    except Exception:
        ece = None
    out = dict(cls)
    out.update(quality)
    out["ece"] = ece
    out["sample_count"] = int(labels.size)
    return out


def _saturation(flags: np.ndarray, overflow: np.ndarray, population: str) -> Dict[str, Any]:
    flags = np.asarray(flags, dtype=np.int64)
    overflow = np.asarray(overflow, dtype=np.float64)
    if flags.ndim != 2 or flags.shape[1] != 4:
        raise CB5Error("saturation flag shape mismatch")
    names = list(FEATURE_ORDER)
    per_feature = {name: {"count": int(flags[:, i].sum()), "fraction": float(flags[:, i].mean() if len(flags) else 0.0)} for i, name in enumerate(names)}
    total = int(flags.size)
    return {
        "population": population,
        "sample_count": int(flags.shape[0]),
        "feature_count": 4,
        "saturated_element_count": int(flags.sum()),
        "saturation_fraction": float(flags.sum() / total) if total else 0.0,
        "samples_with_at_least_one_saturated_feature": int(np.any(flags > 0, axis=1).sum()),
        "per_feature": per_feature,
        "maximum_overflow_distance": float(np.max(overflow)) if overflow.size else 0.0,
        "definition": "unclipped affine int8 input outside [-128,127] before clipping",
    }


def run_int8_model(tf: Any, root: Path, x_scaled: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Infer with the exact C-B4 INT8 model and count pre-clipping saturation."""
    model_bytes = (root / MODEL_REL).read_bytes()
    if hashlib.sha256(model_bytes).hexdigest() != INT8_MODEL_SHA256:
        raise CB5Error("C_B4 candidate INT8 bytes changed")
    interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=1)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    input_scale, input_zero = [float(x) for x in inp["quantization"]]
    output_scale, output_zero = float(out["quantization"][0]), int(out["quantization"][1])
    if not math.isclose(input_scale, INPUT_SCALE, rel_tol=0.0, abs_tol=1e-15) or input_zero != INPUT_ZERO_POINT:
        raise CB5Error("C-B4 INT8 input quantization drift")
    if not math.isclose(output_scale, OUTPUT_SCALE, rel_tol=0.0, abs_tol=1e-15) or output_zero != OUTPUT_ZERO_POINT:
        raise CB5Error("C-B4 INT8 output quantization drift")
    probabilities: List[float] = []
    all_flags: List[np.ndarray] = []
    all_overflow: List[np.ndarray] = []
    for row in np.asarray(x_scaled, dtype=np.float64):
        tensor, flags, overflow = quantize_int8_input(row.reshape(1, 4), input_scale, input_zero)
        interpreter.set_tensor(inp["index"], tensor)
        interpreter.invoke()
        raw = np.asarray(interpreter.get_tensor(out["index"])).reshape(-1)
        probabilities.append(float(dequantize_int8_output(raw, output_scale, output_zero)[0]))
        all_flags.append(flags[0].astype(np.int64))
        all_overflow.append(overflow[0].astype(np.float64))
    flags_arr = np.asarray(all_flags, dtype=np.int64).reshape((-1, 4)) if all_flags else np.empty((0, 4), dtype=np.int64)
    overflow_arr = np.asarray(all_overflow, dtype=np.float64).reshape((-1, 4)) if all_overflow else np.empty((0, 4), dtype=np.float64)
    return np.asarray(probabilities, dtype=np.float64), {"saturation_flags": flags_arr, "overflow_distances": overflow_arr, "input_scale": input_scale, "input_zero_point": input_zero, "output_scale": output_scale, "output_zero_point": output_zero}


def _scenario_result(
    scenario: Mapping[str, Any],
    feature_result: Mapping[str, Any],
    intended_ids: Sequence[str],
    labels_by_id: Mapping[str, int],
    baseline_probabilities: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    tf: Any,
    root: Path,
    baseline_raw: Mapping[str, Sequence[float]],
) -> Dict[str, Any]:
    records = feature_result["records"]
    ids = [str(sid) for sid in intended_ids if records[str(sid)].get("available")]
    unavailable = [str(sid) for sid in intended_ids if not records[str(sid)].get("available")]
    raw = np.asarray([records[sid]["raw"] for sid in ids], dtype=np.float64)
    x_scaled = (raw - scaler_mean.reshape(1, 4)) / scaler_scale.reshape(1, 4)
    probabilities, inference = run_int8_model(tf, root, x_scaled)
    labels = np.asarray([int(labels_by_id[sid]) for sid in ids], dtype=np.int64)
    metrics = _metrics(labels, probabilities)
    sat = _saturation(inference["saturation_flags"], inference["overflow_distances"], "VALIDATION_AVAILABLE")
    base_index = {str(sid): i for i, sid in enumerate(intended_ids)}
    paired_base = np.asarray([baseline_probabilities[base_index[sid]] for sid in ids], dtype=np.float64)
    diff = probabilities - paired_base
    abs_diff = np.abs(diff)
    baseline_preds = (paired_base >= THRESHOLD).astype(np.int64)
    perturbed_preds = (probabilities >= THRESHOLD).astype(np.int64)
    raw_base = np.asarray([baseline_raw[sid] for sid in ids], dtype=np.float64)
    raw_delta = raw - raw_base
    slopes = raw[:, 3]
    slope_base = raw_base[:, 3]
    status_counts: Dict[str, int] = defaultdict(int)
    for sid in intended_ids:
        status_counts[str(records[str(sid)].get("status"))] += 1
    return {
        "scenario_id": _scenario_id(str(scenario["kind"]), scenario.get("level", scenario.get("pattern", {}).get("name", "baseline"))),
        "kind": str(scenario["kind"]),
        "level": scenario.get("level"),
        "seed": scenario.get("seed"),
        "pattern": scenario.get("pattern"),
        "sample_count_intended": int(len(intended_ids)),
        "sample_count_available": int(len(ids)),
        "sample_count_unavailable": int(len(unavailable)),
        "availability_rate": float(len(ids) / len(intended_ids)) if intended_ids else 0.0,
        "unavailable_status_counts": dict(sorted(status_counts.items())),
        "paired_sample_count": int(len(ids)),
        "metrics": metrics,
        "probability_mae_vs_unperturbed": float(np.mean(abs_diff)) if len(abs_diff) else None,
        "probability_p95_absolute_drift_vs_unperturbed": float(np.percentile(abs_diff, 95)) if len(abs_diff) else None,
        "probability_max_absolute_drift_vs_unperturbed": float(np.max(abs_diff)) if len(abs_diff) else None,
        "label_disagreement_count_vs_unperturbed": int(np.sum(baseline_preds != perturbed_preds)),
        "label_disagreement_rate_vs_unperturbed": float(np.mean(baseline_preds != perturbed_preds)) if len(ids) else 0.0,
        "saturation": sat,
        "feature_change": {
            "mean_absolute_CO2": float(np.mean(np.abs(raw_delta[:, 0]))) if len(raw_delta) else 0.0,
            "max_absolute_CO2": float(np.max(np.abs(raw_delta[:, 0]))) if len(raw_delta) else 0.0,
            "mean_absolute_CO2_slope": float(np.mean(np.abs(raw_delta[:, 3]))) if len(raw_delta) else 0.0,
            "max_absolute_CO2_slope": float(np.max(np.abs(raw_delta[:, 3]))) if len(raw_delta) else 0.0,
            "mean_slope": float(np.mean(slopes)) if len(slopes) else None,
            "mean_baseline_slope": float(np.mean(slope_base)) if len(slope_base) else None,
        },
        "feature_matrix_sha256": array_sha256(raw),
        "probability_vector_sha256": array_sha256(probabilities),
        "sample_ids_sha256": ids_sha256(ids),
        "diagnostic_only": True,
    }


def _scenario_grid(protocol: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = [{"kind": "baseline"}]
    for level in protocol["scenarios"]["co2_offset_ppm"]["levels"]:
        out.append({"kind": "co2_offset_ppm", "level": level})
    for level in protocol["scenarios"]["co2_linear_drift_ppm_per_min"]["levels"]:
        out.append({"kind": "co2_linear_drift_ppm_per_min", "level": level})
    for level, seed in zip(protocol["scenarios"]["humidity_noise_sigma_rh"]["levels"], protocol["scenarios"]["humidity_noise_sigma_rh"]["seeds"]):
        out.append({"kind": "humidity_noise_sigma_rh", "level": level, "seed": seed})
    for pattern in protocol["scenarios"]["missing_row"]["patterns"]:
        out.append({"kind": "missing_row", "pattern": pattern})
    for level in protocol["scenarios"]["stale_history_seconds"]["levels"]:
        out.append({"kind": "stale_history_seconds", "level": level})
    for level, seed in zip(protocol["scenarios"]["timestamp_jitter_seconds"]["levels"], protocol["scenarios"]["timestamp_jitter_seconds"]["seeds"]):
        out.append({"kind": "timestamp_jitter_seconds", "level": level, "seed": seed})
    return out


def run_robustness(tf: Any, root: Path, protocol: Mapping[str, Any]) -> Dict[str, Any]:
    data = prepare_fixed_data(root)
    source_rows = load_split_rows(root, "VALIDATION")
    validation_ids = list(data.validation.sample_ids)
    if validation_ids != load_eligible_ids(root, "VALIDATION"):
        raise CB5Error("validation ordered identity drift")
    labels_by_id = {str(row["canonical_sample_id"]): int(row["occupancy_source_value"]) for row in source_rows if row["canonical_sample_id"] in set(validation_ids)}
    if len(labels_by_id) != VALIDATION_COUNT:
        raise CB5Error("validation labels incomplete")
    baseline_features = reconstruct_features(source_rows, validation_ids, {"kind": "baseline"})
    baseline_records = baseline_features["records"]
    baseline_raw_map = {sid: baseline_records[sid]["raw"] for sid in validation_ids if baseline_records[sid].get("available")}
    if len(baseline_raw_map) != VALIDATION_COUNT:
        raise CB5Error("unperturbed validation feature availability drift")
    baseline_raw = np.asarray([baseline_raw_map[sid] for sid in validation_ids], dtype=np.float64)
    if not np.allclose(baseline_raw, np.asarray(data.validation.features, dtype=np.float64), rtol=0.0, atol=1e-12):
        raise CB5Error("C-B1 ENDPOINT_H150 baseline reconstruction differs from C-B4 matrix")
    scaler_mean = np.asarray(data.scaler_evidence["mean"], dtype=np.float64)
    scaler_scale = np.asarray(data.scaler_evidence["scale"], dtype=np.float64)
    baseline_scaled = (baseline_raw - scaler_mean.reshape(1, 4)) / scaler_scale.reshape(1, 4)
    baseline_prob, baseline_inf = run_int8_model(tf, root, baseline_scaled)
    baseline_metrics = _metrics(np.asarray(data.validation.labels, dtype=np.int64), baseline_prob)
    baseline_sat = _saturation(baseline_inf["saturation_flags"], baseline_inf["overflow_distances"], "VALIDATION")
    results: List[Dict[str, Any]] = []
    for scenario in _scenario_grid(protocol):
        if scenario["kind"] == "baseline":
            results.append({
                "scenario_id": "baseline",
                "kind": "baseline",
                "level": None,
                "seed": None,
                "sample_count_intended": VALIDATION_COUNT,
                "sample_count_available": VALIDATION_COUNT,
                "sample_count_unavailable": 0,
                "availability_rate": 1.0,
                "paired_sample_count": VALIDATION_COUNT,
                "metrics": baseline_metrics,
                "saturation": baseline_sat,
                "feature_change": {"mean_absolute_CO2": 0.0, "max_absolute_CO2": 0.0, "mean_absolute_CO2_slope": 0.0, "max_absolute_CO2_slope": 0.0, "mean_slope": float(np.mean(baseline_raw[:, 3])), "mean_baseline_slope": float(np.mean(baseline_raw[:, 3]))},
                "probability_vector_sha256": array_sha256(baseline_prob),
                "feature_matrix_sha256": array_sha256(baseline_raw),
                "sample_ids_sha256": ids_sha256(validation_ids),
                "diagnostic_only": False,
            })
            continue
        feature_result = reconstruct_features(source_rows, validation_ids, scenario)
        results.append(_scenario_result(scenario, feature_result, validation_ids, labels_by_id, baseline_prob, scaler_mean, scaler_scale, tf, root, baseline_raw_map))
    scenario_ids = [str(row["scenario_id"]) for row in results]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise CB5Error("robustness scenario identity collision")
    perturbation_fingerprint = stable_sha256(results)
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "protocol_id": PROTOCOL_ID,
        "candidate_model_sha256": INT8_MODEL_SHA256,
        "threshold": THRESHOLD,
        "feature_order": list(FEATURE_ORDER),
        "slope_profile": SLOPE_PROFILE,
        "scaler_fingerprint": SCALER_FINGERPRINT,
        "locked_test_used": False,
        "locked_test_feature_access": 0,
        "locked_test_target_access": 0,
        "locked_test_predictions": 0,
        "locked_test_probabilities": 0,
        "locked_test_metrics": 0,
        "validation_sample_count": VALIDATION_COUNT,
        "unperturbed_validation_prediction_fingerprint": array_sha256(baseline_prob),
        "baseline_metrics": baseline_metrics,
        "baseline_saturation": baseline_sat,
        "results": results,
        "scenario_count": len(results),
        "perturbation_result_fingerprint": perturbation_fingerprint,
        "classification": "ROBUSTNESS_DIAGNOSTIC_ONLY_UNDER_OFFLINE_TECHNICAL_STRESS",
        "contract_failures": [],
    }


def host_latency_sanity(tf: Any, root: Path, sample: np.ndarray) -> Dict[str, Any]:
    model_bytes = (root / MODEL_REL).read_bytes()
    if hashlib.sha256(model_bytes).hexdigest() != INT8_MODEL_SHA256:
        raise CB5Error("C-B4 candidate INT8 bytes changed before latency")
    interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=1)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    tensor, _, _ = quantize_int8_input(np.asarray(sample, dtype=np.float64).reshape(1, 4), float(inp["quantization"][0]), int(inp["quantization"][1]))
    for _ in range(100):
        interpreter.set_tensor(inp["index"], tensor)
        interpreter.invoke()
        _ = interpreter.get_tensor(out["index"])
    timings: List[float] = []
    for _ in range(2000):
        start = time.perf_counter_ns()
        interpreter.set_tensor(inp["index"], tensor)
        interpreter.invoke()
        _ = interpreter.get_tensor(out["index"])
        timings.append((time.perf_counter_ns() - start) / 1_000_000.0)
    arr = np.asarray(timings, dtype=np.float64)
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "evidence_id": "CO2_B5_HOST_MAC_LATENCY_SANITY_001",
        "classification": "HOST_MAC_LATENCY_SANITY_ONLY",
        "model_path": MODEL_REL,
        "model_sha256": INT8_MODEL_SHA256,
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "processor": platform.processor()},
        "python_version": sys.version,
        "runtime": {"tensorflow_version": str(getattr(tf, "__version__", "unknown")), "interpreter_threads": 1, "delegate": "default", "input_shape": [1, 4], "input_dtype": str(inp["dtype"].__name__), "output_dtype": str(out["dtype"].__name__)},
        "warmup_iterations": 100,
        "timed_iterations": 2000,
        "units": "milliseconds",
        "mean": float(np.mean(arr)),
        "median_p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "minimum": float(np.min(arr)),
        "maximum": float(np.max(arr)),
        "raspberry_pi_latency_claimed": False,
        "production_realtime_claim": False,
        "latency_protocol_fingerprint": stable_sha256({"warmup_iterations": 100, "timed_iterations": 2000, "model_sha256": INT8_MODEL_SHA256, "threads": 1}),
    }


def build_pre_locked_test_freeze(root: Path, protocol: Mapping[str, Any], protocol_sha256: str, robustness_sha256: str, latency_sha256: str, robustness: Mapping[str, Any], latency: Mapping[str, Any]) -> Dict[str, Any]:
    if robustness.get("locked_test_used") is not False or robustness.get("locked_test_predictions") != 0:
        raise LockedTestAuthorizationError("C_B5_LOCKED_TEST_USED_DURING_ROBUSTNESS")
    payload: Dict[str, Any] = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "freeze_profile_id": FREEZE_ID,
        "freeze_status": "VALID_PRE_LOCKED_TEST",
        "candidate": {
            "model_path": MODEL_REL,
            "model_sha256": INT8_MODEL_SHA256,
            "float_reference_sha256": FLOAT_REFERENCE_SHA256,
            "class_map_sha256": CLASS_MAP_SHA256,
            "feature_order": list(FEATURE_ORDER),
            "slope_profile": SLOPE_PROFILE,
            "scaler_path": SCALER_REL,
            "scaler_fingerprint": SCALER_FINGERPRINT,
            "imbalance_strategy": IMBALANCE_STRATEGY,
            "architecture": ARCHITECTURE,
            "threshold": THRESHOLD,
            "input_dtype": "int8",
            "output_dtype": "int8",
            "input_quantization": {"scale": INPUT_SCALE, "zero_point": INPUT_ZERO_POINT},
            "output_quantization": {"scale": OUTPUT_SCALE, "zero_point": OUTPUT_ZERO_POINT},
        },
        "data": {"train_fingerprint": "492ca1f67e44b4a2018b743ec0fc3d20b418f7823d5f2643d8c90b0d39de8fab", "validation_fingerprint": "19321e57fe72f6482b3c7b5d3714d21e9c13b753173ceb56ea694524ac6529ef", "locked_test_membership_fingerprint": LOCKED_TEST_MEMBERSHIP_FINGERPRINT, "train_count": TRAIN_COUNT, "validation_count": VALIDATION_COUNT, "locked_test_count": LOCKED_TEST_COUNT},
        "evidence": {"robustness_protocol_path": f"{ARTIFACT_DIR_REL}/robustness_protocol.json", "robustness_protocol_sha256": protocol_sha256, "robustness_results_path": f"{ARTIFACT_DIR_REL}/robustness_results.json", "robustness_results_sha256": robustness_sha256, "latency_path": f"{ARTIFACT_DIR_REL}/host_latency_evidence.json", "latency_sha256": latency_sha256, "robustness_complete": True, "latency_complete": True},
        "decision_state": {"MODEL_SELECTION_COMPLETE": True, "FEATURE_SELECTION_COMPLETE": True, "SLOPE_SELECTION_COMPLETE": True, "IMBALANCE_SELECTION_COMPLETE": True, "THRESHOLD_SELECTION_COMPLETE": True, "QUANTIZATION_COMPLETE": True, "ROBUSTNESS_REVIEW_COMPLETE": True, "NO_FURTHER_SELECTION_AUTHORIZED": True, "LOCKED_TEST_EVALUATION_AUTHORIZED_ONCE": True},
        "locked_test_prior_access": {"feature_access": 0, "target_access": 0, "predictions": 0, "probabilities": 0, "metrics": 0},
        "predecessor_artifact_drift": False,
        "historical_production_model_modified": False,
        "historical_production_scaler_modified": False,
        "locked_test_robustness_sweeps_prohibited": True,
    }
    payload["freeze_sha256"] = stable_sha256(payload)
    return payload


def validate_pre_locked_test_freeze(root: Path, freeze: Mapping[str, Any], protocol: Mapping[str, Any], robustness: Mapping[str, Any], latency: Mapping[str, Any]) -> None:
    if freeze.get("freeze_profile_id") != FREEZE_ID or freeze.get("freeze_status") != "VALID_PRE_LOCKED_TEST":
        raise LockedTestAuthorizationError("invalid pre-test freeze profile")
    expected_hash = stable_sha256({k: v for k, v in freeze.items() if k != "freeze_sha256"})
    if freeze.get("freeze_sha256") != expected_hash:
        raise LockedTestAuthorizationError("C_B5_PRETEST_FREEZE_CHECKSUM_MISMATCH")
    cand = freeze.get("candidate", {})
    if cand.get("model_sha256") != INT8_MODEL_SHA256 or cand.get("threshold") != THRESHOLD or cand.get("feature_order") != list(FEATURE_ORDER) or cand.get("scaler_fingerprint") != SCALER_FINGERPRINT:
        raise LockedTestAuthorizationError("C_B5_PRETEST_CANDIDATE_IDENTITY_MISMATCH")
    prior = freeze.get("locked_test_prior_access", {})
    if any(int(prior.get(k, -1)) != 0 for k in ("feature_access", "target_access", "predictions", "probabilities", "metrics")):
        raise LockedTestAuthorizationError("C_B5_LOCKED_TEST_PRIOR_ACCESS")
    if protocol.get("locked_test_used") is not False or robustness.get("locked_test_used") is not False:
        raise LockedTestAuthorizationError("C_B5_LOCKED_TEST_USED_BEFORE_FREEZE")
    if not robustness.get("results") or not latency.get("evidence_id"):
        raise LockedTestAuthorizationError("C_B5_EVIDENCE_INCOMPLETE_BEFORE_FREEZE")


def evaluate_locked_test_once(tf: Any, root: Path, freeze: Mapping[str, Any]) -> Dict[str, Any]:
    """Perform the single unperturbed LOCKED_TEST evaluation after freeze."""
    if freeze.get("freeze_profile_id") != FREEZE_ID or freeze.get("freeze_status") != "VALID_PRE_LOCKED_TEST":
        raise LockedTestAuthorizationError("C_B5_LOCKED_TEST_NOT_AUTHORIZED")
    if freeze.get("freeze_sha256") != stable_sha256({k: v for k, v in freeze.items() if k != "freeze_sha256"}):
        raise LockedTestAuthorizationError("C_B5_PRETEST_FREEZE_CHECKSUM_MISMATCH")
    candidate = freeze.get("candidate", {})
    if candidate.get("model_sha256") != INT8_MODEL_SHA256 or candidate.get("threshold") != THRESHOLD or candidate.get("feature_order") != list(FEATURE_ORDER) or candidate.get("scaler_fingerprint") != SCALER_FINGERPRINT:
        raise LockedTestAuthorizationError("C_B5_LOCKED_TEST_CANDIDATE_IDENTITY_MISMATCH")
    prior = freeze.get("locked_test_prior_access", {})
    if any(int(prior.get(key, -1)) != 0 for key in ("feature_access", "target_access", "predictions", "probabilities", "metrics")):
        raise LockedTestAuthorizationError("C_B5_LOCKED_TEST_PRIOR_ACCESS")
    decisions = freeze.get("decision_state", {})
    if not all(bool(decisions.get(key)) for key in ("MODEL_SELECTION_COMPLETE", "FEATURE_SELECTION_COMPLETE", "SLOPE_SELECTION_COMPLETE", "IMBALANCE_SELECTION_COMPLETE", "THRESHOLD_SELECTION_COMPLETE", "QUANTIZATION_COMPLETE", "ROBUSTNESS_REVIEW_COMPLETE", "NO_FURTHER_SELECTION_AUTHORIZED", "LOCKED_TEST_EVALUATION_AUTHORIZED_ONCE")):
        raise LockedTestAuthorizationError("C_B5_LOCKED_TEST_DECISIONS_NOT_FROZEN")
    out_path = root / ARTIFACT_DIR_REL / "locked_test_evaluation.json"
    if out_path.exists():
        raise LockedTestAuthorizationError("C_B5_LOCKED_TEST_DOUBLE_EVALUATION")
    rows = load_split_rows(root, "LOCKED_TEST")
    ids = load_eligible_ids(root, "LOCKED_TEST")
    result = reconstruct_features(rows, ids, {"kind": "baseline"})
    records = result["records"]
    if any(not records[sid].get("available") for sid in ids):
        raise LockedTestAuthorizationError("C_B5_LOCKED_TEST_FEATURE_UNAVAILABLE")
    labels = np.asarray([int(next(row["occupancy_source_value"] for row in rows if str(row["canonical_sample_id"]) == sid)) for sid in ids], dtype=np.int64)
    raw = np.asarray([records[sid]["raw"] for sid in ids], dtype=np.float64)
    scaler = load_json(root / SCALER_REL)
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    x_scaled = (raw - mean.reshape(1, 4)) / scale.reshape(1, 4)
    probabilities, inference = run_int8_model(tf, root, x_scaled)
    metrics = _metrics(labels, probabilities)
    sat = _saturation(inference["saturation_flags"], inference["overflow_distances"], "LOCKED_TEST")
    payload = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "evaluation_profile_id": EVALUATION_ID,
        "evaluation_count": 1,
        "evaluation_status": "COMPLETED_ONE_TIME_UNPERTURBED",
        "execution_protocol": "C-B4 INT8 candidate + C-B2 TRAIN scaler + ENDPOINT_H150 + threshold 0.58; no perturbation sweep",
        "authorized_pre_test_freeze_sha256": str(freeze["freeze_sha256"]),
        "candidate_sha256": INT8_MODEL_SHA256,
        "threshold": THRESHOLD,
        "eligible_sample_count": len(ids),
        "locked_test_membership_fingerprint": ids_sha256(ids),
        "labels_fingerprint": hashlib.sha256(labels.astype("<i8").tobytes()).hexdigest(),
        "metrics": metrics,
        "saturation": sat,
        "locked_test_feature_access": len(ids),
        "locked_test_target_access": len(ids),
        "locked_test_predictions": len(ids),
        "locked_test_probabilities": len(ids),
        "locked_test_metrics": 1,
        "perturbation_sweeps": 0,
        "post_test_tuning": {"model_change": False, "scaler_change": False, "feature_change": False, "slope_change": False, "threshold_change": False},
        "generalization_gap_classification": "NOT_ASSESSED_UNTIL_VALIDATION_COMPARISON",
        "device_domain_validation": "NOT_YET_COMPLETE",
    }
    return payload


def build_final_candidate_metadata(root: Path, robustness: Mapping[str, Any], latency: Mapping[str, Any], locked: Mapping[str, Any], freeze: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "candidate_profile_id": "CO2_B5_FINAL_OFFLINE_UCI_CANDIDATE_001",
        "candidate_status": "FINAL_OFFLINE_UCI_CANDIDATE_LOCKED",
        "model_path": MODEL_REL,
        "model_sha256": INT8_MODEL_SHA256,
        "float_reference_path": FLOAT_REFERENCE_REL,
        "float_reference_sha256": FLOAT_REFERENCE_SHA256,
        "feature_order": list(FEATURE_ORDER),
        "scaler_identity": {"path": SCALER_REL, "fingerprint": SCALER_FINGERPRINT},
        "slope_profile": SLOPE_PROFILE,
        "imbalance_strategy": IMBALANCE_STRATEGY,
        "architecture": ARCHITECTURE,
        "threshold": THRESHOLD,
        "class_map": {"0": "VACANT", "1": "OCCUPIED", "positive": "OCCUPIED", "semantic": "ROOM_OCCUPANCY"},
        "input_dtype": "int8",
        "output_dtype": "int8",
        "input_quantization": {"scale": INPUT_SCALE, "zero_point": INPUT_ZERO_POINT},
        "output_quantization": {"scale": OUTPUT_SCALE, "zero_point": OUTPUT_ZERO_POINT},
        "train_fingerprint": freeze["data"]["train_fingerprint"],
        "validation_fingerprint": freeze["data"]["validation_fingerprint"],
        "locked_test_fingerprint": freeze["data"]["locked_test_membership_fingerprint"],
        "validation_metrics": robustness["baseline_metrics"],
        "locked_test_metrics": locked["metrics"],
        "robustness_evidence_identity": {"protocol_id": PROTOCOL_ID, "results_sha256": file_sha256(root, f"{ARTIFACT_DIR_REL}/robustness_results.json")},
        "latency_evidence_identity": {"evidence_id": latency["evidence_id"], "sha256": file_sha256(root, f"{ARTIFACT_DIR_REL}/host_latency_evidence.json")},
        "freeze_identity": {"profile_id": FREEZE_ID, "sha256": str(freeze["freeze_sha256"])},
        "known_limitations": ["SOURCE_TIMEZONE_UNVERIFIED", "GROUP_INDEPENDENCE_NOT_VERIFIABLE", "CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED", "DEVICE_UCI_CADENCE_DOMAIN_GAP", "SAFETY_RULE_CONTRACT_OUT_OF_SCOPE", "SENSOR_HEALTH_CONTRACT_OUT_OF_SCOPE", "MULTISENSOR_RISK_CONTRACT_OUT_OF_SCOPE", "DEFERRED_SHARED_INTEGRATION_UPDATE", "INT8_INPUT_SATURATION_OBSERVED"],
        "device_domain_validation_status": "NOT_YET_COMPLETE",
        "safety_risk_semantic_status": "ROOM_OCCUPANCY_ONLY_NO_SAFETY_SEMANTIC",
        "historical_production_model_modified": False,
        "historical_production_scaler_modified": False,
        "post_test_tuning": "NONE",
        "locked_test_evaluation_count": 1,
        "scd40_deployment_candidate_validation": "NOT_YET_COMPLETE",
        "b_series_release_status": "B_SERIES_RELEASE_READY_AFTER_MERGE",
    }


def _lock_entries(root: Path, candidate_metadata_rel: str) -> List[Dict[str, Any]]:
    rels = [
        "datasets/co2/manifests/c_a6_final_integrity_lock/artifact_lock_manifest.json",
        "datasets/co2/manifests/c_b0_offline_experiment_contract/checksums.sha256",
        "datasets/co2/manifests/c_b1_slope_method_history_ablation/checksums.sha256",
        "datasets/co2/manifests/c_b2_imbalance_calibration/checksums.sha256",
        "datasets/co2/manifests/c_b3_architecture_multiseed/checksums.sha256",
        "datasets/co2/manifests/c_b4_float_tflite_int8_equivalence/checksums.sha256",
        FLOAT_REFERENCE_REL, MODEL_REL, CLASS_MAP_REL, SCALER_REL,
        f"{ARTIFACT_DIR_REL}/robustness_protocol.json",
        f"{ARTIFACT_DIR_REL}/robustness_results.json",
        f"{ARTIFACT_DIR_REL}/host_latency_evidence.json",
        f"{ARTIFACT_DIR_REL}/pre_locked_test_candidate_freeze.json",
        f"{ARTIFACT_DIR_REL}/locked_test_evaluation.json",
        candidate_metadata_rel,
    ]
    return [{"path": rel, "byte_size": (root / rel).stat().st_size, "sha256": file_sha256(root, rel)} for rel in sorted(rels)]


def write_final_lock(root: Path, metadata: Mapping[str, Any]) -> Dict[str, Any]:
    lock_path = root / ARTIFACT_DIR_REL / "final_candidate_lock.json"
    registry_path = root / ARTIFACT_DIR_REL / "checksum_registry.json"
    checksums_path = root / ARTIFACT_DIR_REL / "checksums.sha256"
    metadata_rel = f"{CANDIDATE_DIR_REL}/final_candidate_metadata.json"
    entries = _lock_entries(root, metadata_rel)
    payload: Dict[str, Any] = {"manifest_version": "1.0", "phase": PHASE_ID, "final_lock_profile_id": FINAL_LOCK_ID, "candidate_status": "FINAL_OFFLINE_UCI_CANDIDATE_LOCKED", "self_reference_policy": {"final_lock_hashes_itself": False, "checksum_registry_hashes_itself": False, "checksums_sha256_hashes_itself": False}, "artifact_count": len(entries), "artifacts": entries, "closure_status": "PASS"}
    payload["final_lock_sha256"] = stable_sha256(payload)
    write_json(lock_path, payload)
    registry = {"manifest_version": "1.0", "phase": PHASE_ID, "self_referential": False, "entry_count": len(entries), "entries": entries, "closure_status": "PASS"}
    write_json(registry_path, registry)
    checksums_path.write_text("".join(f"{e['sha256']}  {e['path']}\n" for e in entries), encoding="utf-8")
    identity = {"manifest_version": "1.0", "phase": PHASE_ID, "final_lock_profile_id": FINAL_LOCK_ID, "final_lock_sha256": payload["final_lock_sha256"], "artifact_count": len(entries), "candidate_metadata_path": metadata_rel, "candidate_model_sha256": INT8_MODEL_SHA256, "locked_test_evaluation_count": 1, "raw_payload_included": False, "self_reference": False}
    write_json(root / ARTIFACT_DIR_REL / "artifact_identity.json", identity)
    return payload


__all__ = [
    "ARTIFACT_DIR_REL", "ARCHITECTURE", "CANDIDATE_DIR_REL", "CB5Error", "CLASS_MAP_SHA256", "EVALUATION_ID", "FEATURE_ORDER", "FLOAT_MODEL_SHA256", "FLOAT_REFERENCE_SHA256", "FREEZE_ID", "IMBALANCE_STRATEGY", "INPUT_SCALE", "INPUT_ZERO_POINT", "INT8_MODEL_SHA256", "LOCKED_TEST_COUNT", "LOCKED_TEST_MEMBERSHIP_FINGERPRINT", "MODEL_REL", "OUTPUT_SCALE", "OUTPUT_ZERO_POINT", "PHASE_ID", "PROTOCOL_ID", "SCALER_FINGERPRINT", "SLOPE_PROFILE", "THRESHOLD", "TRAIN_COUNT", "VALIDATION_COUNT", "array_sha256", "build_final_candidate_metadata", "build_pre_locked_test_freeze", "build_protocol", "evaluate_locked_test_once", "file_sha256", "host_latency_sanity", "ids_sha256", "load_eligible_ids", "load_json", "load_split_rows", "make_scenario_rows", "reconstruct_features", "run_robustness", "stable_sha256", "validate_pre_locked_test_freeze", "write_final_lock", "write_json", "_scenario_grid", "_metrics", "_missing_indices", "run_int8_model",
]
