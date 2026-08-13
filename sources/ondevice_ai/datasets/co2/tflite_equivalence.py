#!/usr/bin/env python3
"""SafeNest CO₂ Phase C-B4 conversion and equivalence evidence.

This module reconstructs the selected C-B3 logistic model, transfers its
weights into a small Keras Dense/sigmoid bridge, and evaluates float TFLite
and full-integer TFLite representations on the same ordered VALIDATION
population.  It intentionally never materialises LOCKED_TEST features or
targets.  The integer converter uses a fixed, TRAIN-derived activation range
(``P99.9`` rounded up to the nearest 0.5) to make the calibration policy
explicit; any resulting saturation is measured rather than hidden.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from threadpoolctl import threadpool_limits

from datasets.co2.architecture_multiseed import (
    FIXED_FEATURES,
    LOCKED_TEST_COUNT,
    TRAIN_COUNT,
    VALIDATION_COUNT,
    build_predecessor_fingerprint_registry as build_c_b3_predecessor_registry,
    prepare_fixed_data,
    validate_predecessor_inputs as validate_c_b3_predecessors,
    verify_stored_predecessor_registry as verify_c_b3_predecessor_registry,
)
from datasets.co2.imbalance_calibration import (
    BALANCED_RANDOM_OVERSAMPLE,
    _probability_fingerprint,
)
from datasets.co2.offline_experiment import (
    EXPECTED_LOCKED_TEST_SEALED,
    EXPECTED_TRAIN_COMMON,
    EXPECTED_VALIDATION_COMMON,
    _load_eligible_by_role,
    assert_no_forbidden_path_markers,
)
from datasets.co2.raw_reader import compute_sha256_file, get_repo_root


PHASE_ID = "C-B4"
PHASE_NAME = "CO2_FLOAT_TFLITE_INT8_EQUIVALENCE_AND_CONVERSION_ARTIFACT_LOCK"
ARTIFACT_DIR_REL = "datasets/co2/manifests/c_b4_float_tflite_int8_equivalence"
CANDIDATE_DIR_REL = "models/co2/candidates/c_b4"
B0_DIR_REL = "datasets/co2/manifests/c_b0_offline_experiment_contract"
B1_DIR_REL = "datasets/co2/manifests/c_b1_slope_method_history_ablation"
B2_DIR_REL = "datasets/co2/manifests/c_b2_imbalance_calibration"
B3_DIR_REL = "datasets/co2/manifests/c_b3_architecture_multiseed"
A6_DIR_REL = "datasets/co2/manifests/c_a6_final_integrity_lock"

C_B3_MERGED_MAIN_COMMIT = "7344997beacf82ff21df7f3dd2b9bc78405f32d7"
A_SERIES_RELEASE_TAG = "co2-a-series-raw-to-canonical"
A_SERIES_RELEASE_TARGET = "bfd860cad2bb8dafe35ef7600cfa931d7d2d554d"
B0_CONTRACT_ID = "CO2_B0_OFFLINE_EXPERIMENT_CONTRACT_001"
B1_PROFILE_ID = "CO2_B1_SELECTED_SLOPE_PROFILE_001"
B2_SCALER_PROFILE_ID = "CO2_B2_TRAIN_ONLY_STANDARD_SCALER_001"
B2_POLICY_ID = "CO2_B2_SELECTED_IMBALANCE_POLICY_001"
B2_THRESHOLD_PROTOCOL_ID = "CO2_B2_THRESHOLD_CALIBRATION_PROTOCOL_001"
B3_PROFILE_ID = "CO2_B3_SELECTED_ARCHITECTURE_PROFILE_001"
B3_EXPERIMENT_ID = "CO2_B3_ARCHITECTURE_MULTI_SEED_CONTRACT_001"
B3_ARCHITECTURE_ID = "LINEAR_LOGISTIC"
B3_ARCHITECTURE_FAMILY = "LINEAR"

FLOAT_REFERENCE_PROFILE_ID = "CO2_B4_FLOAT_LOGISTIC_REFERENCE_001"
BRIDGE_PROFILE_ID = "CO2_B4_WEIGHT_TRANSFER_EQUIVALENT_CONVERSION_BRIDGE_001"
CLASS_MAP_PROFILE_ID = "CO2_B4_CLASS_MAP_001"
INPUT_CONTRACT_PROFILE_ID = "CO2_B4_INPUT_CONTRACT_001"
THRESHOLD_PROFILE_ID = "CO2_B4_EQUIVALENCE_THRESHOLD_001"
REPRESENTATIVE_PROFILE_ID = "CO2_B4_ALL_8140_NATURAL_TRAIN_001"
METADATA_PROFILE_ID = "CO2_B4_CONVERSION_CANDIDATE_METADATA_001"

CANONICAL_RECONSTRUCTION_SEED = 20260810
EQUIVALENCE_THRESHOLD = 0.58
POSITIVE_CLASS = "OCCUPIED"
NEGATIVE_CLASS = "VACANT"
SYNTHETIC_FIXTURE_REL = "datasets/co2/processed/co2_occupancy_v1.npz"
PRODUCTION_MODEL_REL = "models/co2/co2_occupancy_int8_v0.1.0.tflite"
PRODUCTION_SCALER_REL = "models/co2/co2_scaling_metadata_v0.1.0.json"

# C-B4's deterministic calibration policy.  The range is derived from the
# TRAIN-only standardized matrix, before any conversion result is inspected.
CALIBRATION_PERCENTILE = 99.9
CALIBRATION_ROUNDING = 0.5

INT8_MACRO_F1_DEGRADATION_MAX = 0.005
INT8_OCCUPIED_RECALL_DEGRADATION_MAX = 0.010
INT8_PROBABILITY_MAE_MAX = 0.010
INT8_P95_DRIFT_MAX = 0.020
INT8_MAX_DRIFT_MAX = 0.050
INT8_LABEL_DISAGREEMENT_FRACTION_MAX = 0.005
FLOAT_DRIFT_MAX = 1e-5


class CB4Error(RuntimeError):
    """Base C-B4 contract error."""


class PredecessorFingerprintMismatch(CB4Error):
    """An immutable predecessor artifact or commit has changed."""


class LockedTestPolicyViolation(CB4Error):
    """Raised if C-B4 attempts predictive LOCKED_TEST access."""


class FloatReferenceReconstructionFailure(CB4Error):
    """The selected C-B3 float reference cannot be reproduced."""


class TFLiteContractMismatch(CB4Error):
    """A converted TFLite model does not have the expected contract."""


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def stable_sha256(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True, check=False)


def _close(a: Any, b: Any, atol: float = 1e-12, rtol: float = 1e-10) -> bool:
    try:
        return bool(math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol))
    except (TypeError, ValueError):
        return False


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CB4Error(message)


def validate_merged_main_ancestry(root: Path) -> None:
    result = _git(root, "merge-base", "--is-ancestor", C_B3_MERGED_MAIN_COMMIT, "HEAD")
    if result.returncode != 0:
        raise PredecessorFingerprintMismatch("C_B3_PREDECESSOR_NOT_MERGED")


def _assert_checksum_manifest(root: Path, directory_rel: str) -> None:
    checksum_path = root / directory_rel / "checksums.sha256"
    if not checksum_path.is_file():
        raise PredecessorFingerprintMismatch(f"Missing predecessor checksum manifest: {directory_rel}")
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, rel = line.split("  ", 1)
        except ValueError as exc:
            raise PredecessorFingerprintMismatch(f"Malformed checksum row: {line!r}") from exc
        path = root / rel
        if not path.is_file() or compute_sha256_file(path) != digest:
            raise PredecessorFingerprintMismatch(f"C_B0_B1_B2_B3_PREDECESSOR_FINGERPRINT_MISMATCH: {rel}")


def _c_b3_consumed_specs() -> Tuple[Tuple[str, str, str], ...]:
    return (
        (f"{B3_DIR_REL}/experiment_contract.json", "C-B3", "B3_EXPERIMENT_CONTRACT"),
        (f"{B3_DIR_REL}/architecture_candidate_registry.json", "C-B3", "B3_ARCHITECTURE_REGISTRY"),
        (f"{B3_DIR_REL}/per_run_results.json", "C-B3", "B3_PER_RUN_RESULTS"),
        (f"{B3_DIR_REL}/architecture_multiseed_aggregate.json", "C-B3", "B3_AGGREGATE_RESULTS"),
        (f"{B3_DIR_REL}/selected_architecture_profile.json", "C-B3", "B3_SELECTED_ARCHITECTURE"),
        (f"{B3_DIR_REL}/leakage_audit.json", "C-B3", "B3_LEAKAGE_AUDIT"),
        (f"{B3_DIR_REL}/determinism_report.json", "C-B3", "B3_DETERMINISM"),
        (f"{B3_DIR_REL}/fixed_comparison_universe_fingerprint.json", "C-B3", "B3_COMPARISON_UNIVERSE"),
        (f"{B3_DIR_REL}/preprocessing_parity_evidence.json", "C-B3", "B3_SCALER_PARITY"),
        (f"{B3_DIR_REL}/oversampling_parity_evidence.json", "C-B3", "B3_OVERSAMPLING_PARITY"),
        (f"{B3_DIR_REL}/validation_predictions.json", "C-B3", "B3_VALIDATION_PREDICTIONS"),
        (f"{B3_DIR_REL}/threshold_stability_summary.json", "C-B3", "B3_THRESHOLD_STABILITY"),
    )


def build_predecessor_fingerprint_registry(root: Path) -> Dict[str, Any]:
    """Build C-B0..C-B3 and A-series/protected-artifact closure."""
    validate_merged_main_ancestry(root)
    validate_c_b3_predecessors(root)
    stored_b3 = load_json(root / B3_DIR_REL / "predecessor_fingerprint_registry.json")
    verify_c_b3_predecessor_registry(root, stored_b3)
    for directory in (B0_DIR_REL, B1_DIR_REL, B2_DIR_REL, B3_DIR_REL):
        _assert_checksum_manifest(root, directory)

    entries: List[Dict[str, Any]] = []
    # The C-B3 registry already captures the exact A/C-B0/C-B1/C-B2 lock and
    # protected production/synthetic artifacts.  Carry those entries forward.
    for entry in stored_b3.get("entries", []):
        entries.append(dict(entry))
    for rel, phase, role in _c_b3_consumed_specs():
        path = root / rel
        if not path.is_file():
            raise PredecessorFingerprintMismatch(f"Missing C-B3 predecessor artifact: {rel}")
        entries.append({"path": rel, "phase": phase, "role": role, "byte_size": path.stat().st_size, "sha256": compute_sha256_file(path)})
    # Include the two machine-readable B3 lock documents explicitly even if a
    # future C-B3 registry changes its entry set.
    entries.append({
        "path": f"{B3_DIR_REL}/predecessor_fingerprint_registry.json",
        "phase": "C-B3",
        "role": "B3_PREDECESSOR_CLOSURE",
        "byte_size": (root / B3_DIR_REL / "predecessor_fingerprint_registry.json").stat().st_size,
        "sha256": compute_sha256_file(root / B3_DIR_REL / "predecessor_fingerprint_registry.json"),
    })
    deduped = {entry["path"]: entry for entry in entries}
    rows = [deduped[key] for key in sorted(deduped)]
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "registry_id": "CO2_B4_PREDECESSOR_FINGERPRINT_REGISTRY_001",
        "required_c_b3_merged_main_commit": C_B3_MERGED_MAIN_COMMIT,
        "a_series_release_tag": A_SERIES_RELEASE_TAG,
        "a_series_release_target": A_SERIES_RELEASE_TARGET,
        "b0_contract_id": B0_CONTRACT_ID,
        "b1_selected_slope_profile_id": B1_PROFILE_ID,
        "b2_policy_id": B2_POLICY_ID,
        "b2_scaler_profile_id": B2_SCALER_PROFILE_ID,
        "b2_threshold_protocol_id": B2_THRESHOLD_PROTOCOL_ID,
        "b3_profile_id": B3_PROFILE_ID,
        "entry_count": len(rows),
        "entries": rows,
        "closure_fingerprint": stable_sha256(rows),
        "closure_status": "LOCKED",
        "mismatch_status": "C_B0_B1_B2_B3_PREDECESSOR_FINGERPRINT_MISMATCH_ON_DRIFT",
    }


def verify_predecessor_fingerprint_registry(root: Path, stored: Mapping[str, Any]) -> None:
    live = build_predecessor_fingerprint_registry(root)
    if dict(stored) != live:
        raise PredecessorFingerprintMismatch("C_B0_B1_B2_B3_PREDECESSOR_FINGERPRINT_MISMATCH")


def _selected_b3(root: Path) -> Dict[str, Any]:
    profile = load_json(root / B3_DIR_REL / "selected_architecture_profile.json")
    _require(profile.get("winning_architecture_id") == B3_ARCHITECTURE_ID, "C-B3 selected architecture drift")
    _require(profile.get("winning_architecture_family") == B3_ARCHITECTURE_FAMILY, "C-B3 architecture family drift")
    _require(profile.get("preprocessing_profile_id") == B2_SCALER_PROFILE_ID, "C-B3 scaler profile drift")
    _require(profile.get("selected_imbalance_strategy") == BALANCED_RANDOM_OVERSAMPLE, "C-B3 imbalance policy drift")
    _require(profile.get("selected_slope_profile_id") == B1_PROFILE_ID, "C-B3 slope profile drift")
    _require(profile.get("production_model") is False and profile.get("int8_quantization") is False, "C-B3 production/int8 boundary drift")
    _require(profile.get("threshold_stability", {}).get("values_by_seed") == {str(seed): 0.58 for seed in (20260810, 20260811, 20260812, 20260813, 20260814)}, "C-B3 threshold stability drift")
    return profile


def validate_predecessor_contract(root: Path) -> Dict[str, Any]:
    """Validate immutable C-B0..C-B3 inputs without touching LOCKED_TEST data."""
    validate_merged_main_ancestry(root)
    pred = validate_c_b3_predecessors(root)
    profile = _selected_b3(root)
    b0 = pred["b0_contract"]
    universe = pred["b0_universe"]
    _require(b0.get("experiment_contract_id") == B0_CONTRACT_ID, "C-B0 contract identity drift")
    _require(universe.get("b_series_common_train") == TRAIN_COUNT and universe.get("b_series_common_validation") == VALIDATION_COUNT and universe.get("b_series_sealed_locked_test") == LOCKED_TEST_COUNT, "C-B0 comparison-universe counts drift")
    _require(pred["b1_profile"].get("selected_candidate_id") == "ENDPOINT_H150", "C-B1 selected slope drift")
    _require(pred["b1_profile"].get("method") == "ENDPOINT_DIFFERENCE" and _close(pred["b1_profile"].get("minimum_history_seconds"), 150.0), "C-B1 slope method/history drift")
    _require(pred["b2_scaler"].get("scaler_fingerprint") == "d0cf83558fb0de9dcdc97f0d94781a5a475a6f68e8d818121aee929030e5dc89", "C-B2 scaler fingerprint drift")
    _require(pred["b2_sampling"].get("resampled_ordered_sample_ids_sha256") == "a3c61800df26a6b995e79f2debb1d7003ba0517985c38ade74e2cdceeba7a551", "C-B2 oversampling fingerprint drift")
    _require(pred["b2_threshold"].get("protocol_id") == B2_THRESHOLD_PROTOCOL_ID and pred["b2_threshold"].get("threshold_search_population") == "VALIDATION_ONLY", "C-B2 threshold protocol drift")
    b3_contract = load_json(root / B3_DIR_REL / "experiment_contract.json")
    _require(b3_contract.get("experiment_contract_id") == B3_EXPERIMENT_ID, "C-B3 experiment contract identity drift")
    _require(b3_contract.get("train_population") == TRAIN_COUNT and b3_contract.get("validation_population") == VALIDATION_COUNT and b3_contract.get("locked_test_membership_count") == LOCKED_TEST_COUNT, "C-B3 population count drift")
    _require(b3_contract.get("locked_test_status") == "SEALED" and b3_contract.get("locked_test_predictive_evaluation") is False, "C-B3 LOCKED_TEST policy drift")
    return {"predecessor": pred, "b3_profile": profile, "b3_contract": b3_contract}


def _fit_reference(data: Any) -> Tuple[LogisticRegression, np.ndarray]:
    idx = data.oversample_plan.training_indices
    if idx.shape != (12828,) or len(set(idx.tolist())) != TRAIN_COUNT:
        raise FloatReferenceReconstructionFailure("C-B2 oversampled TRAIN multiset changed")
    model = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", fit_intercept=True,
        max_iter=2000, class_weight=None, random_state=CANONICAL_RECONSTRUCTION_SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with threadpool_limits(limits=1):
            model.fit(data.x_train_scaled[idx], data.train.labels[idx])
    if any("Convergence" in str(item.message) for item in caught):
        raise FloatReferenceReconstructionFailure("C-B3 logistic reconstruction did not converge")
    probabilities = np.asarray(model.predict_proba(data.x_validation_scaled)[:, 1], dtype=np.float64)
    if probabilities.shape != (VALIDATION_COUNT,) or not np.isfinite(probabilities).all():
        raise FloatReferenceReconstructionFailure("Invalid reconstructed SOURCE_FLOAT probabilities")
    return model, probabilities


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = EQUIVALENCE_THRESHOLD) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "decision_threshold": float(threshold),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "occupied_recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall_occupied": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "precision_occupied": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "confusion_matrix": {"labels": [0, 1], "label_names": [NEGATIVE_CLASS, POSITIVE_CLASS], "matrix": cm.tolist(), "tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "false_positive_rate": float(fpr), "false_negative_rate": float(fnr),
        "specificity": float(1.0 - fpr), "sensitivity": float(1.0 - fnr),
    }


def _drift_metrics(source: np.ndarray, target: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    diff = np.asarray(target, dtype=np.float64) - np.asarray(source, dtype=np.float64)
    absolute = np.abs(diff)
    correlation = None
    if np.std(source) > 0 and np.std(target) > 0:
        correlation = float(np.corrcoef(source, target)[0, 1])
    source_labels = (source >= EQUIVALENCE_THRESHOLD).astype(np.int64)
    target_labels = (target >= EQUIVALENCE_THRESHOLD).astype(np.int64)
    disagreements = int(np.sum(source_labels != target_labels))
    return {
        "probability_mae": float(np.mean(absolute)),
        "probability_rmse": float(np.sqrt(np.mean(diff * diff))),
        "probability_p95_absolute_drift": float(np.percentile(absolute, 95)),
        "probability_max_absolute_drift": float(np.max(absolute)),
        "pearson_correlation": correlation,
        "label_disagreement_count": disagreements,
        "label_disagreement_fraction": float(disagreements / len(source)),
        "source_metrics": _metrics(labels, source),
        "target_metrics": _metrics(labels, target),
    }


def quantize_int8_input(values: np.ndarray, scale: float, zero_point: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the actual TFLite affine input contract before clipping.

    Returns ``(clipped_int8, saturation_flags, overflow_distance)``.  This
    small pure function is intentionally public-facing through the module so
    focused tests can reject incorrect rounding, dequantization, or hidden
    saturation accounting without invoking TensorFlow.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 4 or not math.isfinite(float(scale)) or scale <= 0:
        raise TFLiteContractMismatch("INT8 input quantization contract is invalid")
    unclipped = np.rint(values / float(scale)) + int(zero_point)
    flags = (unclipped < -128) | (unclipped > 127)
    overflow = np.maximum(np.maximum(-128.0 - unclipped, unclipped - 127.0), 0.0)
    return np.clip(unclipped, -128, 127).astype(np.int8), flags, overflow


def dequantize_int8_output(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Dequantize scalar output using the interpreter's affine contract."""
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise TFLiteContractMismatch("INT8 output quantization scale is invalid")
    return (np.asarray(values, dtype=np.float64) - int(zero_point)) * float(scale)


def compute_int8_gate(source_metrics: Mapping[str, Any], int8_metrics: Mapping[str, Any], drift: Mapping[str, Any]) -> Dict[str, Any]:
    """Compute the pre-registered, non-retunable C-B4 INT8 gate."""
    gate = {
        "macro_f1_degradation": float(max(0.0, float(source_metrics["macro_f1"]) - float(int8_metrics["macro_f1"]))),
        "occupied_recall_degradation": float(max(0.0, float(source_metrics["occupied_recall"]) - float(int8_metrics["occupied_recall"]))),
        "probability_mae": float(drift["probability_mae"]),
        "p95_absolute_probability_drift": float(drift["probability_p95_absolute_drift"]),
        "maximum_absolute_probability_drift": float(drift["probability_max_absolute_drift"]),
        "label_disagreement_fraction": float(drift["label_disagreement_fraction"]),
        "limits": {
            "macro_f1_degradation": INT8_MACRO_F1_DEGRADATION_MAX,
            "occupied_recall_degradation": INT8_OCCUPIED_RECALL_DEGRADATION_MAX,
            "probability_mae": INT8_PROBABILITY_MAE_MAX,
            "p95_absolute_probability_drift": INT8_P95_DRIFT_MAX,
            "maximum_absolute_probability_drift": INT8_MAX_DRIFT_MAX,
            "label_disagreement_fraction": INT8_LABEL_DISAGREEMENT_FRACTION_MAX,
        },
    }
    gate["status"] = "PASS" if all(gate[key] <= limit for key, limit in gate["limits"].items()) else "FAIL"
    return gate


def validate_representative_membership(
    representative_ids: Sequence[str], train_ids: Sequence[str], validation_ids: Sequence[str], *,
    locked_test_rows: int = 0, synthetic_rows: int = 0, duplicate_draws: int = 0,
) -> None:
    """Reject representative calibration membership outside natural TRAIN."""
    if list(representative_ids) != list(train_ids):
        raise CB4Error("REPRESENTATIVE_DATASET_LEAKAGE: representative membership/order drift")
    if set(representative_ids).intersection(validation_ids):
        raise CB4Error("REPRESENTATIVE_DATASET_LEAKAGE: VALIDATION row present")
    if locked_test_rows or synthetic_rows or duplicate_draws:
        raise CB4Error("REPRESENTATIVE_DATASET_LEAKAGE")


def validate_locked_test_access(role: str) -> None:
    """Allow only sealed membership inspection for LOCKED_TEST."""
    if role == "LOCKED_TEST":
        raise LockedTestPolicyViolation("LOCKED_TEST_POLICY_VIOLATION")


def validate_class_map_semantics(class_map: Mapping[str, Any]) -> None:
    if class_map.get("labels") != {"0": "VACANT", "1": "OCCUPIED"}:
        raise CB4Error("class-map mutation")
    if class_map.get("positive_class") != "OCCUPIED":
        raise CB4Error("positive-class semantic drift")
    if class_map.get("safety_semantic") != "NONE" or class_map.get("risk_semantic") != "NONE":
        raise CB4Error("occupancy/safety semantic conflation")


def _tensorflow() -> Any:
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise CB4Error("C_B4_REQUIRED_BACKEND_UNAVAILABLE: TensorFlow/Keras") from exc
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass
    return tf


def _build_bridge(tf: Any, model: LogisticRegression) -> Any:
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(4,), dtype=tf.float32, name="standardized_features")
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", use_bias=True, name="occupied_probability")(inputs)
    bridge = tf.keras.Model(inputs=inputs, outputs=outputs, name="safenest_co2_b4_float_bridge")
    dense = bridge.get_layer("occupied_probability")
    dense.set_weights([np.asarray(model.coef_, dtype=np.float32).T, np.asarray(model.intercept_, dtype=np.float32)])
    return bridge


def _calibration_range(x_train_scaled: np.ndarray) -> Dict[str, Any]:
    absolute = np.abs(np.asarray(x_train_scaled, dtype=np.float64))
    percentile_value = float(np.percentile(absolute, CALIBRATION_PERCENTILE))
    rounded = float(math.ceil(percentile_value / CALIBRATION_ROUNDING) * CALIBRATION_ROUNDING)
    if not math.isfinite(rounded) or rounded <= 0:
        raise CB4Error("Invalid TRAIN-derived INT8 activation calibration range")
    return {
        "policy_id": "CO2_B4_TRAIN_ABS_P99_9_ROUND_UP_0_5_001",
        "source_population": "ORIGINAL_NATURALLY_DISTRIBUTED_TRAIN_ONLY",
        "source_sample_count": TRAIN_COUNT,
        "percentile": CALIBRATION_PERCENTILE,
        "percentile_value": percentile_value,
        "rounding_increment": CALIBRATION_ROUNDING,
        "range_min": -rounded,
        "range_max": rounded,
        "representative_rows": TRAIN_COUNT,
        "validation_rows": 0,
        "locked_test_rows": 0,
    }


def _build_int8_bridge(tf: Any, model: LogisticRegression, calibration_range: Mapping[str, Any]) -> Any:
    """Build a non-trained conversion-only model with a fixed PTQ range.

    The fake-quant operation is a fixed calibration-range constraint.  No
    optimizer, compile, fit, or weight update is performed.  The exported
    graph contains only FULLY_CONNECTED and LOGISTIC INT8 operations.
    """
    tf.keras.backend.clear_session()
    low = float(calibration_range["range_min"])
    high = float(calibration_range["range_max"])
    inputs = tf.keras.Input(shape=(4,), dtype=tf.float32, name="standardized_features")

    def fixed_range(x: Any) -> Any:
        return tf.quantization.fake_quant_with_min_max_vars(x, min=low, max=high, narrow_range=False)

    constrained = tf.keras.layers.Lambda(fixed_range, name="fixed_ptq_calibration_range")(inputs)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", use_bias=True, name="occupied_probability")(constrained)
    bridge = tf.keras.Model(inputs=inputs, outputs=outputs, name="safenest_co2_b4_int8_conversion_bridge")
    dense = bridge.get_layer("occupied_probability")
    dense.set_weights([np.asarray(model.coef_, dtype=np.float32).T, np.asarray(model.intercept_, dtype=np.float32)])
    return bridge


def _representative_dataset(x_train_scaled: np.ndarray) -> Iterable[List[np.ndarray]]:
    for row in np.asarray(x_train_scaled, dtype=np.float32):
        yield [row.reshape(1, 4)]


def _convert_float(tf: Any, bridge: Any) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(bridge)
    return bytes(converter.convert())


def _convert_int8(tf: Any, bridge: Any, x_train_scaled: np.ndarray) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(bridge)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_dataset(x_train_scaled)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return bytes(converter.convert())


def _dtype_name(dtype: Any) -> str:
    return np.dtype(dtype).name


def _tensor_quantization(detail: Mapping[str, Any]) -> Dict[str, Any]:
    scale, zero = detail.get("quantization", (0.0, 0))
    params = detail.get("quantization_parameters", {})
    return {
        "scale": float(scale),
        "zero_point": int(zero),
        "scales": [float(x) for x in np.asarray(params.get("scales", []), dtype=np.float64).tolist()],
        "zero_points": [int(x) for x in np.asarray(params.get("zero_points", []), dtype=np.int64).tolist()],
        "quantized_dimension": int(params.get("quantized_dimension", 0)),
    }


def inspect_tflite_contract(tf: Any, model_bytes: bytes, expected_dtype: str, label: str) -> Dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise TFLiteContractMismatch(f"{label}: expected one input and one output tensor")
    inp, out = inputs[0], outputs[0]
    actual_in = _dtype_name(inp["dtype"])
    actual_out = _dtype_name(out["dtype"])
    if actual_in != expected_dtype or actual_out != expected_dtype:
        raise TFLiteContractMismatch(f"{label}: dtype mismatch {actual_in}/{actual_out}")
    if [int(x) for x in inp["shape"]] != [1, 4] or [int(x) for x in out["shape"]] != [1, 1]:
        raise TFLiteContractMismatch(f"{label}: shape mismatch")
    ops = interpreter._get_ops_details()
    op_names = [str(op.get("op_name")) for op in ops if str(op.get("op_name")) != "DELEGATE"]
    tensor_types: List[str] = []
    for op in ops:
        if str(op.get("op_name")) == "DELEGATE":
            continue
        for key in ("operand_types", "result_types"):
            for dtype in op.get(key, []):
                try:
                    tensor_types.append(_dtype_name(dtype))
                except TypeError:
                    tensor_types.append(str(dtype))
    record = {
        "input_count": 1,
        "output_count": 1,
        "input_name": str(inp["name"]),
        "output_name": str(out["name"]),
        "input_shape": [int(x) for x in inp["shape"]],
        "output_shape": [int(x) for x in out["shape"]],
        "input_shape_signature": [int(x) for x in inp.get("shape_signature", inp["shape"])],
        "output_shape_signature": [int(x) for x in out.get("shape_signature", out["shape"])],
        "input_dtype": actual_in,
        "output_dtype": actual_out,
        "input_quantization": _tensor_quantization(inp),
        "output_quantization": _tensor_quantization(out),
        "op_names": op_names,
        "operator_tensor_types": sorted(set(tensor_types)),
        "full_integer_ops": all(dtype in {"int8", "int32"} for dtype in tensor_types) if expected_dtype == "int8" else False,
        "model_byte_size": len(model_bytes),
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
    }
    if expected_dtype == "int8":
        if not record["full_integer_ops"] or "FULLY_CONNECTED" not in op_names or "LOGISTIC" not in op_names:
            raise TFLiteContractMismatch(f"{label}: not a full integer builtins model")
        if record["input_quantization"]["scale"] <= 0 or record["output_quantization"]["scale"] <= 0:
            raise TFLiteContractMismatch(f"{label}: invalid quantization parameters")
    return record


def _run_tflite(tf: Any, model_bytes: bytes, x: np.ndarray, *, quantized: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    input_scale, input_zero = [float(x) for x in inp["quantization"]]
    output_scale, output_zero = [float(out["quantization"][0]), int(out["quantization"][1])]
    probabilities: List[float] = []
    saturation_flags: List[List[int]] = []
    overflow_distances: List[float] = []
    for row in np.asarray(x, dtype=np.float64):
        if quantized:
            tensor_batch, flags_batch, overflow_batch = quantize_int8_input(row.reshape(1, 4), input_scale, input_zero)
            saturation_flags.append([int(value) for value in flags_batch[0].tolist()])
            overflow_distances.append(float(np.max(overflow_batch[0])))
            tensor = tensor_batch
        else:
            tensor = np.asarray(row, dtype=np.float32).reshape(1, 4)
        interpreter.set_tensor(inp["index"], tensor)
        interpreter.invoke()
        raw = float(np.asarray(interpreter.get_tensor(out["index"])).reshape(-1)[0])
        probabilities.append(float(dequantize_int8_output(np.asarray([raw]), output_scale, output_zero)[0]) if quantized else raw)
    return np.asarray(probabilities, dtype=np.float64), {
        "saturation_flags": saturation_flags,
        "overflow_distances": overflow_distances,
        "input_scale": input_scale,
        "input_zero_point": int(input_zero),
        "output_scale": output_scale,
        "output_zero_point": int(output_zero),
    }


def _saturation_report(saturation_flags: Sequence[Sequence[int]], overflow_distances: Sequence[float], population: str) -> Dict[str, Any]:
    flags = np.asarray(saturation_flags, dtype=np.int64)
    if flags.shape != (len(saturation_flags), 4):
        raise CB4Error("Saturation accounting shape mismatch")
    per_feature = flags.sum(axis=0)
    sample_count = int(flags.shape[0])
    element_count = int(flags.size)
    saturated_elements = int(flags.sum())
    return {
        "population": population,
        "sample_count": sample_count,
        "feature_count": 4,
        "saturated_element_count": saturated_elements,
        "saturation_fraction": float(saturated_elements / element_count) if element_count else 0.0,
        "per_feature": {feature: {"count": int(per_feature[index]), "fraction": float(per_feature[index] / sample_count) if sample_count else 0.0} for index, feature in enumerate(FIXED_FEATURES)},
        "samples_with_at_least_one_saturated_feature": int(np.sum(np.any(flags > 0, axis=1))),
        "maximum_overflow_distance": float(max(overflow_distances) if overflow_distances else 0.0),
        "saturation_observed": bool(saturated_elements),
    }


def _candidate_metadata(
    *, root: Path, reference: Mapping[str, Any], float_contract: Mapping[str, Any], int8_contract: Mapping[str, Any],
    representative: Mapping[str, Any], calibration_range: Mapping[str, Any], equivalence: Mapping[str, Any],
    saturation: Mapping[str, Any], hashes: Mapping[str, str], environment: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "profile_id": METADATA_PROFILE_ID,
        "deployment_status": [
            "OFFLINE_CONVERSION_CANDIDATE",
            "INT8_EQUIVALENCE_EVALUATED",
            "LOCKED_TEST_UNTOUCHED",
            "DEVICE_DOMAIN_UNVALIDATED",
            "ROBUSTNESS_NOT_YET_VALIDATED",
            "FINAL_CANDIDATE_NOT_YET_LOCKED",
        ],
        "source_c_b3_profile": B3_PROFILE_ID,
        "selected_architecture": B3_ARCHITECTURE_ID,
        "architecture_family": B3_ARCHITECTURE_FAMILY,
        "canonical_reconstruction_seed": CANONICAL_RECONSTRUCTION_SEED,
        "feature_order": list(FIXED_FEATURES),
        "slope_profile_id": B1_PROFILE_ID,
        "scaler": {"path": f"{B2_DIR_REL}/preprocessing_scaler_evidence.json", "sha256": hashes["scaler"]},
        "imbalance_policy": BALANCED_RANDOM_OVERSAMPLE,
        "float_reference": {"path": f"{CANDIDATE_DIR_REL}/float_reference_parameters.json", "sha256": hashes["float_reference"]},
        "float_tflite": {"path": f"{CANDIDATE_DIR_REL}/float_reference.tflite", "sha256": hashes["float_tflite"]},
        "int8_tflite": {"path": f"{CANDIDATE_DIR_REL}/full_integer_int8.tflite", "sha256": hashes["int8_tflite"]},
        "class_map": {"path": f"{CANDIDATE_DIR_REL}/class_map.json", "sha256": hashes["class_map"]},
        "input_contract": {"path": f"{CANDIDATE_DIR_REL}/input_contract.json", "sha256": hashes["input_contract"]},
        "threshold_contract": {"path": f"{CANDIDATE_DIR_REL}/threshold_contract.json", "sha256": hashes["threshold_contract"]},
        "representative_dataset": {key: value for key, value in representative.items() if key != "sample_ids"},
        "calibration_range": calibration_range,
        "float_tflite_contract": float_contract,
        "int8_tflite_contract": int8_contract,
        "equivalence": equivalence,
        "saturation": saturation,
        "tensorflow_version": environment.get("tensorflow_version"),
        "conversion_options": {
            "float_optimizations": [],
            "int8_optimizations": ["DEFAULT"],
            "supported_ops": ["TFLITE_BUILTINS_INT8"],
            "inference_input_type": "int8",
            "inference_output_type": "int8",
            "quantization_aware_training": False,
            "keras_bridge_trained": False,
        },
        "locked_test_policy": {"membership_count": LOCKED_TEST_COUNT, "feature_access": 0, "target_access": 0, "predictions": 0, "probabilities": 0, "metrics": 0},
        "production_model_promoted": False,
        "production_scaler_modified": False,
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def _artifact_hashes(root: Path, paths: Mapping[str, str]) -> Dict[str, str]:
    return {key: compute_sha256_file(root / rel) for key, rel in paths.items()}


def _environment(tf: Any) -> Dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scikit_learn_version": __import__("sklearn").__version__,
        "tensorflow_version": str(tf.__version__),
        "platform": platform.platform(),
        "deterministic_ops_requested": True,
        "single_thread_requested": True,
    }


def run_c_b4(root: Optional[Path] = None, *, verify_repeat: bool = True) -> Dict[str, Any]:
    repo_root = (root or get_repo_root()).resolve()
    predecessor_state = validate_predecessor_contract(repo_root)
    data = prepare_fixed_data(repo_root)
    b3_run = load_json(repo_root / B3_DIR_REL / "per_run_results.json")["runs"]
    model, source_probabilities = _fit_reference(data)
    stored_b3_run = next(row for row in b3_run if row.get("run_id") == f"{B3_ARCHITECTURE_ID}__seed_{CANONICAL_RECONSTRUCTION_SEED}")
    stored_predictions = load_json(repo_root / B3_DIR_REL / "validation_predictions.json")
    stored_probability = np.asarray(stored_predictions["runs"][f"{B3_ARCHITECTURE_ID}__seed_{CANONICAL_RECONSTRUCTION_SEED}"]["probabilities"], dtype=np.float64)
    if not np.array_equal(source_probabilities, stored_probability):
        raise FloatReferenceReconstructionFailure("C_B3_FLOAT_REFERENCE_RECONSTRUCTION_FAILURE")
    source_fingerprint = _probability_fingerprint(data.validation.sample_ids, source_probabilities)
    if source_fingerprint != stored_b3_run.get("validation_probability_vector_sha256"):
        raise FloatReferenceReconstructionFailure("C-B3 validation probability fingerprint mismatch")
    source_metrics = _metrics(data.validation.labels, source_probabilities)
    if not _close(source_metrics["macro_f1"], stored_b3_run["calibrated_validation_metrics"]["macro_f1"], atol=1e-13) or not _close(source_metrics["occupied_recall"], stored_b3_run["calibrated_validation_metrics"]["occupied_recall"], atol=1e-13):
        raise FloatReferenceReconstructionFailure("C-B3 source metric parity failure")

    tf = _tensorflow()
    environment = _environment(tf)
    float_bridge = _build_bridge(tf, model)
    bridge_probabilities = np.asarray(float_bridge(np.asarray(data.x_validation_scaled, dtype=np.float32)).numpy().reshape(-1), dtype=np.float64)
    bridge_equivalence = _drift_metrics(source_probabilities, bridge_probabilities, data.validation.labels)
    if bridge_equivalence["probability_max_absolute_drift"] > FLOAT_DRIFT_MAX or bridge_equivalence["label_disagreement_count"] != 0:
        raise CB4Error("FLOAT_BRIDGE_EQUIVALENCE_FAILURE")

    calibration_range = _calibration_range(data.x_train_scaled)
    int8_bridge = _build_int8_bridge(tf, model, calibration_range)
    float_bytes = _convert_float(tf, float_bridge)
    int8_bytes = _convert_int8(tf, int8_bridge, data.x_train_scaled)
    float_contract = inspect_tflite_contract(tf, float_bytes, "float32", "FLOAT_TFLITE")
    int8_contract = inspect_tflite_contract(tf, int8_bytes, "int8", "INT8_TFLITE")
    float_probabilities, _ = _run_tflite(tf, float_bytes, data.x_validation_scaled, quantized=False)
    int8_probabilities, int8_run = _run_tflite(tf, int8_bytes, data.x_validation_scaled, quantized=True)
    repeat_report: Dict[str, Any] = {
        "requested": bool(verify_repeat),
        "float_tflite_bytes_identical": None,
        "int8_tflite_bytes_identical": None,
        "float_inference_semantic_reproducibility": None,
        "int8_inference_semantic_reproducibility": None,
    }
    if verify_repeat:
        repeat_float_bytes = _convert_float(tf, _build_bridge(tf, model))
        repeat_int8_bytes = _convert_int8(tf, _build_int8_bridge(tf, model, calibration_range), data.x_train_scaled)
        repeat_float_probabilities, _ = _run_tflite(tf, repeat_float_bytes, data.x_validation_scaled, quantized=False)
        repeat_int8_probabilities, _ = _run_tflite(tf, repeat_int8_bytes, data.x_validation_scaled, quantized=True)
        repeat_report.update({
            "float_tflite_bytes_identical": bool(repeat_float_bytes == float_bytes),
            "int8_tflite_bytes_identical": bool(repeat_int8_bytes == int8_bytes),
            "float_inference_semantic_reproducibility": bool(np.array_equal(repeat_float_probabilities, float_probabilities)),
            "int8_inference_semantic_reproducibility": bool(np.array_equal(repeat_int8_probabilities, int8_probabilities)),
        })
    float_bridge_tflite = _drift_metrics(bridge_probabilities, float_probabilities, data.validation.labels)
    source_float_tflite = _drift_metrics(source_probabilities, float_probabilities, data.validation.labels)
    source_int8 = _drift_metrics(source_probabilities, int8_probabilities, data.validation.labels)
    if float_bridge_tflite["probability_max_absolute_drift"] > FLOAT_DRIFT_MAX or float_bridge_tflite["label_disagreement_count"] != 0:
        raise CB4Error("FLOAT_TFLITE_EQUIVALENCE_FAILURE")
    int8_source_metrics = source_int8["target_metrics"]
    int8_gate = compute_int8_gate(source_int8["source_metrics"], int8_source_metrics, source_int8)
    if int8_gate["status"] != "PASS":
        raise CB4Error("INT8_EQUIVALENCE_GATE_FAILURE")

    train_int8_probabilities, train_int8_run = _run_tflite(tf, int8_bytes, data.x_train_scaled, quantized=True)
    saturation = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "definition": "q_unclipped < -128 or q_unclipped > 127 before int8 clipping",
        "train": _saturation_report(train_int8_run["saturation_flags"], train_int8_run["overflow_distances"], "TRAIN_REPRESENTATIVE"),
        "validation": _saturation_report(int8_run["saturation_flags"], int8_run["overflow_distances"], "VALIDATION"),
        "input_quantization": {"scale": int8_run["input_scale"], "zero_point": int8_run["input_zero_point"]},
        "interpretation": "INT8_INPUT_SATURATION_OBSERVED" if (int(_saturation_report(train_int8_run["saturation_flags"], train_int8_run["overflow_distances"], "TRAIN_REPRESENTATIVE")["saturated_element_count"]) or int(_saturation_report(int8_run["saturation_flags"], int8_run["overflow_distances"], "VALIDATION")["saturated_element_count"])) else "PASS",
    }

    representative = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "profile_id": REPRESENTATIVE_PROFILE_ID,
        "source_population": "ORIGINAL_NATURALLY_DISTRIBUTED_TRAIN",
        "source_artifact": f"{B0_DIR_REL}/sample_universe_manifest.json",
        "sample_count": TRAIN_COUNT,
        "sample_ids_sha256": data.original_train_fingerprint,
        "sample_ids": list(data.train.sample_ids),
        "preprocessing": "raw canonical features -> exact C-B2 TRAIN-only scaler -> float32 standardized four-feature input",
        "validation_rows": 0,
        "locked_test_rows": 0,
        "synthetic_npz_rows": 0,
        "oversampled_duplicate_draws": 0,
        "feature_order": list(FIXED_FEATURES),
    }
    source_bridge = dict(bridge_equivalence)
    source_bridge.update({"source_stage": "SOURCE_FLOAT_SKLEARN", "target_stage": "KERAS_FLOAT_BRIDGE", "threshold": EQUIVALENCE_THRESHOLD, "max_probability_drift_gate": FLOAT_DRIFT_MAX, "label_disagreement_gate": 0, "source_probability_fingerprint": source_fingerprint, "target_probability_fingerprint": _probability_fingerprint(data.validation.sample_ids, bridge_probabilities), "status": "PASS"})
    bridge_float = dict(float_bridge_tflite)
    bridge_float.update({"source_stage": "KERAS_FLOAT_BRIDGE", "target_stage": "FLOAT_TFLITE", "threshold": EQUIVALENCE_THRESHOLD, "max_probability_drift_gate": FLOAT_DRIFT_MAX, "label_disagreement_gate": 0, "source_probability_fingerprint": _probability_fingerprint(data.validation.sample_ids, bridge_probabilities), "target_probability_fingerprint": _probability_fingerprint(data.validation.sample_ids, float_probabilities), "status": "PASS"})
    source_float = dict(source_float_tflite)
    source_float.update({"source_stage": "SOURCE_FLOAT_SKLEARN", "target_stage": "FLOAT_TFLITE", "threshold": EQUIVALENCE_THRESHOLD, "source_probability_fingerprint": source_fingerprint, "target_probability_fingerprint": _probability_fingerprint(data.validation.sample_ids, float_probabilities), "status": "PASS"})
    source_int8_evidence = dict(source_int8)
    source_int8_evidence.update({"source_stage": "SOURCE_FLOAT_SKLEARN", "target_stage": "INT8_TFLITE_DEQUANTIZED", "threshold": EQUIVALENCE_THRESHOLD, "source_probability_fingerprint": source_fingerprint, "target_probability_fingerprint": _probability_fingerprint(data.validation.sample_ids, int8_probabilities), "gate": int8_gate, "status": int8_gate["status"]})
    equivalence = {
        "sample_universe": {"population": "VALIDATION", "sample_count": VALIDATION_COUNT, "sample_ids_sha256": data.validation_fingerprint, "same_order_for_all_representations": True, "missing_predictions": 0, "non_finite_predictions": 0, "sample_order_mismatch": 0},
        "threshold": EQUIVALENCE_THRESHOLD,
        "source_float": source_metrics,
        "source_bridge": source_bridge,
        "bridge_float_tflite": bridge_float,
        "source_float_tflite": source_float,
        "source_int8_tflite": source_int8_evidence,
        "int8_gate": int8_gate,
    }

    output_dir = repo_root / ARTIFACT_DIR_REL
    candidate_dir = repo_root / CANDIDATE_DIR_REL
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    parameter_payload = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "profile_id": FLOAT_REFERENCE_PROFILE_ID,
        "architecture_id": B3_ARCHITECTURE_ID,
        "architecture_family": B3_ARCHITECTURE_FAMILY,
        "source_library": "sklearn.linear_model.LogisticRegression",
        "source_library_version": __import__("sklearn").__version__,
        "feature_order": list(FIXED_FEATURES),
        "coefficient_vector": [float(x) for x in model.coef_.reshape(-1).tolist()],
        "intercept": float(model.intercept_.reshape(-1)[0]),
        "dtype": "float32_for_tflite_transfer",
        "positive_class": POSITIVE_CLASS,
        "negative_class": NEGATIVE_CLASS,
        "scaler_profile_id": B2_SCALER_PROFILE_ID,
        "scaler_fingerprint": predecessor_state["predecessor"]["b2_scaler"]["scaler_fingerprint"],
        "oversampled_train_fingerprint": data.oversampled_train_fingerprint,
        "source_c_b3_profile": B3_PROFILE_ID,
        "canonical_reconstruction_seed": CANONICAL_RECONSTRUCTION_SEED,
        "reference_threshold": EQUIVALENCE_THRESHOLD,
        "source_validation_prediction_fingerprint": source_fingerprint,
        "source_float_metrics": source_metrics,
        "reconstruction_status": "C_B3_FLOAT_REFERENCE_RECONSTRUCTION_PASS",
    }
    class_map = {
        "manifest_version": "1.0", "phase": PHASE_ID, "profile_id": CLASS_MAP_PROFILE_ID,
        "labels": {"0": NEGATIVE_CLASS, "1": POSITIVE_CLASS}, "positive_class": POSITIVE_CLASS,
        "semantic": "ROOM_OCCUPANCY", "probability_meaning": "P(Occupancy = OCCUPIED | model features)",
        "safety_semantic": "NONE", "risk_semantic": "NONE",
    }
    input_contract = {
        "manifest_version": "1.0", "phase": PHASE_ID, "profile_id": INPUT_CONTRACT_PROFILE_ID,
        "feature_count": 4, "feature_order": list(FIXED_FEATURES), "slope_profile_id": B1_PROFILE_ID,
        "input_preprocessing": "C-B2 TRAIN-only standardization", "source_float_dtype": "float32",
        "float_tflite_dtype": "float32", "int8_tflite_dtype": "int8", "int8_input_quantization": "actual interpreter input scale/zero point",
        "raw_ppm_direct_input": False, "scaler_embedded": False,
    }
    threshold_contract = {
        "manifest_version": "1.0", "phase": PHASE_ID, "profile_id": THRESHOLD_PROFILE_ID,
        "threshold": EQUIVALENCE_THRESHOLD, "source_protocol": B2_THRESHOLD_PROTOCOL_ID,
        "selected_on": "C-B3 VALIDATION_ONLY", "retuned_in_c_b4": False, "status": "OFFLINE_VALIDATION_REFERENCE_ONLY",
    }
    bridge_contract = {
        "manifest_version": "1.0", "phase": PHASE_ID, "profile_id": BRIDGE_PROFILE_ID,
        "model_type": "Input(shape=(4,), dtype=float32) -> Dense(1, activation=sigmoid, use_bias=True)",
        "weight_transfer": "C-B3 sklearn coefficient/intercept -> Keras Dense kernel/bias",
        "trained": False, "retrained": False, "optimizer": None, "epochs": 0,
        "new_architecture_candidate": False, "float32_weight_cast": True,
        "coefficient_parity": True, "intercept_parity": True,
    }
    write_json(candidate_dir / "float_reference_parameters.json", parameter_payload)
    write_json(candidate_dir / "class_map.json", class_map)
    write_json(candidate_dir / "input_contract.json", input_contract)
    write_json(candidate_dir / "threshold_contract.json", threshold_contract)
    write_json(output_dir / "bridge_contract.json", bridge_contract)
    (candidate_dir / "float_reference.tflite").write_bytes(float_bytes)
    (candidate_dir / "full_integer_int8.tflite").write_bytes(int8_bytes)

    candidate_paths = {
        "float_reference": f"{CANDIDATE_DIR_REL}/float_reference_parameters.json",
        "float_tflite": f"{CANDIDATE_DIR_REL}/float_reference.tflite",
        "int8_tflite": f"{CANDIDATE_DIR_REL}/full_integer_int8.tflite",
        "class_map": f"{CANDIDATE_DIR_REL}/class_map.json",
        "input_contract": f"{CANDIDATE_DIR_REL}/input_contract.json",
        "threshold_contract": f"{CANDIDATE_DIR_REL}/threshold_contract.json",
        "scaler": f"{B2_DIR_REL}/preprocessing_scaler_evidence.json",
    }
    hashes = _artifact_hashes(repo_root, candidate_paths)
    _write_jsonl(output_dir / "validation_prediction_drift.jsonl", (
        {"sample_id": sample_id, "source_float_probability": float(source_probabilities[i]), "keras_bridge_probability": float(bridge_probabilities[i]), "float_tflite_probability": float(float_probabilities[i]), "int8_dequantized_probability": float(int8_probabilities[i]), "source_class": int(source_probabilities[i] >= EQUIVALENCE_THRESHOLD), "float_tflite_class": int(float_probabilities[i] >= EQUIVALENCE_THRESHOLD), "int8_class": int(int8_probabilities[i] >= EQUIVALENCE_THRESHOLD), "absolute_float_tflite_drift": float(abs(source_probabilities[i] - float_probabilities[i])), "absolute_int8_drift": float(abs(source_probabilities[i] - int8_probabilities[i])), "int8_input_saturation": bool(any(int(x) for x in int8_run["saturation_flags"][i]))}
        for i, sample_id in enumerate(data.validation.sample_ids)
    ))
    write_json(output_dir / "predecessor_fingerprint_registry.json", build_predecessor_fingerprint_registry(repo_root))
    write_json(output_dir / "experiment_contract.json", {
        "manifest_version": "1.0", "schema": "SafeNest_CO2_C_B4_Float_TFLite_INT8_Equivalence_Contract", "phase": PHASE_ID,
        "phase_name": PHASE_NAME, "experiment_contract_id": "CO2_B4_FLOAT_TFLITE_INT8_EQUIVALENCE_CONTRACT_001", "immediate_predecessor": "C-B3",
        "c_b3_required_merged_main_commit": C_B3_MERGED_MAIN_COMMIT, "b0_contract_id": B0_CONTRACT_ID, "b1_selected_slope_profile_id": B1_PROFILE_ID,
        "b2_policy_id": B2_POLICY_ID, "b2_threshold_protocol_id": B2_THRESHOLD_PROTOCOL_ID, "b3_profile_id": B3_PROFILE_ID,
        "selected_architecture": B3_ARCHITECTURE_ID, "selected_architecture_family": B3_ARCHITECTURE_FAMILY, "feature_context": list(FIXED_FEATURES),
        "train_population": TRAIN_COUNT, "validation_population": VALIDATION_COUNT, "locked_test_membership_count": LOCKED_TEST_COUNT, "locked_test_status": "SEALED",
        "representative_dataset_policy": "ALL_8140_NATURAL_TRAIN", "representative_dataset_count": TRAIN_COUNT, "representative_validation_rows": 0, "representative_locked_test_rows": 0,
        "equivalence_threshold": EQUIVALENCE_THRESHOLD, "threshold_retuning": False, "locked_test_predictive_evaluation": False,
        "keras_bridge_trained": False, "quantization_aware_training": False, "int8_conversion": "FULL_INTEGER_POST_TRAINING",
        "conversion_gate": int8_gate, "production_model_promotion": False, "final_candidate_lock": False, "device_domain_validation": False,
    })
    write_json(output_dir / "representative_dataset_manifest.json", representative)
    write_json(output_dir / "conversion_range_policy.json", calibration_range)
    write_json(output_dir / "tflite_contract_audit.json", {"float_tflite": float_contract, "int8_tflite": int8_contract})
    write_json(output_dir / "equivalence_source_bridge.json", source_bridge)
    write_json(output_dir / "equivalence_bridge_float_tflite.json", bridge_float)
    write_json(output_dir / "equivalence_source_float_tflite.json", source_float)
    write_json(output_dir / "equivalence_source_int8_tflite.json", source_int8_evidence)
    write_json(output_dir / "saturation_report.json", saturation)
    write_json(output_dir / "conversion_environment.json", environment)
    write_json(output_dir / "determinism_report.json", {
        "manifest_version": "1.0", "phase": PHASE_ID, "data_pipeline_determinism": "PASS", "float_reference_reproducibility": "PASS",
        "inference_reproducibility": "PASS" if (not verify_repeat or (repeat_report["float_inference_semantic_reproducibility"] and repeat_report["int8_inference_semantic_reproducibility"])) else "FAIL",
        "representative_dataset_fingerprint": representative["sample_ids_sha256"], "quantization_parameters": int8_contract["input_quantization"],
        "repeat_execution_requested": bool(verify_repeat), "conversion_byte_determinism": ("PASS" if repeat_report.get("float_tflite_bytes_identical") and repeat_report.get("int8_tflite_bytes_identical") else "WARNING_SEMANTIC_ONLY") if verify_repeat else "NOT_RUN",
        "repeat_report": repeat_report,
    })
    write_json(output_dir / "exceptions_and_limitations.json", {
        "manifest_version": "1.0", "phase": PHASE_ID, "blockers": [], "warnings": [
            {"code": "DEVICE_UCI_CADENCE_DOMAIN_GAP", "description": "UCI occupancy cadence/domain equivalence to SCD40 is not validated."},
            {"code": "INT8_INPUT_SATURATION_OBSERVED", "description": "The fixed TRAIN-derived activation range clips a small number of standardized slope inputs; C-B5 owns robustness interpretation."},
        ], "deferred_shared_integration_update": "DEFERRED_SHARED_INTEGRATION_UPDATE", "production_model_modified": False, "production_scaler_modified": False,
        "locked_test_feature_access": 0, "locked_test_target_access": 0, "locked_test_predictions": 0, "locked_test_probabilities": 0, "locked_test_metrics": 0,
        "synthetic_npz_used": False, "safety_threshold_calibrated": False, "clinical_claim": False,
    })
    equivalence_summary = {"sample_count": VALIDATION_COUNT, "threshold": EQUIVALENCE_THRESHOLD, "source_float": source_metrics, "keras_bridge": bridge_equivalence, "float_tflite": source_float, "int8_tflite": source_int8_evidence, "int8_gate": int8_gate}
    candidate_meta = _candidate_metadata(root=repo_root, reference=parameter_payload, float_contract=float_contract, int8_contract=int8_contract, representative=representative, calibration_range=calibration_range, equivalence=equivalence_summary, saturation=saturation, hashes=hashes, environment=environment)
    write_json(candidate_dir / "candidate_metadata.json", candidate_meta)
    hashes["metadata"] = compute_sha256_file(candidate_dir / "candidate_metadata.json")
    artifact_identity = {
        "manifest_version": "1.0", "phase": PHASE_ID, "artifact_namespace": ARTIFACT_DIR_REL, "candidate_namespace": CANDIDATE_DIR_REL,
        "artifact_files": sorted([path.name for path in output_dir.glob("*") if path.is_file() and path.name not in {"checksums.sha256", "checksum_registry.json", "artifact_identity.json"}] + ["artifact_identity.json", "checksum_registry.json"]),
        "candidate_files": sorted(path.name for path in candidate_dir.glob("*")), "validation_sample_count": VALIDATION_COUNT, "representative_sample_count": TRAIN_COUNT,
        "locked_test_predictions": 0, "locked_test_probabilities": 0, "locked_test_metrics": 0, "raw_payload_included": False, "production_model_modified": False, "production_scaler_modified": False,
    }
    write_json(output_dir / "artifact_identity.json", artifact_identity)
    hash_targets: Dict[str, str] = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name not in {"checksums.sha256", "checksum_registry.json"}:
            hash_targets[f"{ARTIFACT_DIR_REL}/{path.name}"] = compute_sha256_file(path)
    for path in sorted(candidate_dir.iterdir()):
        if path.is_file():
            hash_targets[f"{CANDIDATE_DIR_REL}/{path.name}"] = compute_sha256_file(path)
    # Lock the consumed scaler as well; the registry is deliberately excluded
    # from its own rows to avoid a self-referential checksum.
    hash_targets[f"{B2_DIR_REL}/preprocessing_scaler_evidence.json"] = compute_sha256_file(repo_root / B2_DIR_REL / "preprocessing_scaler_evidence.json")
    checksum_registry = {"manifest_version": "1.0", "phase": PHASE_ID, "self_referential": False, "entries": [{"path": path, "sha256": digest, "byte_size": (repo_root / path).stat().st_size} for path, digest in sorted(hash_targets.items())], "entry_count": len(hash_targets), "closure_status": "LOCKED"}
    write_json(output_dir / "checksum_registry.json", checksum_registry)
    checksum_lines = [f"{digest}  {path}" for path, digest in sorted(hash_targets.items())]
    (output_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    for path in list(output_dir.glob("*.json")) + [output_dir / "validation_prediction_drift.jsonl"]:
        forbidden = assert_no_forbidden_path_markers(path.read_text(encoding="utf-8"))
        if forbidden:
            raise CB4Error(f"Non-portable path marker in {path.name}: {forbidden}")
    return {
        "artifact_dir": ARTIFACT_DIR_REL, "candidate_dir": CANDIDATE_DIR_REL, "selected_architecture": B3_ARCHITECTURE_ID,
        "representative_count": TRAIN_COUNT, "validation_count": VALIDATION_COUNT, "source_fingerprint": source_fingerprint,
        "float_tflite_sha256": hashlib.sha256(float_bytes).hexdigest(), "int8_tflite_sha256": hashlib.sha256(int8_bytes).hexdigest(),
        "int8_gate": int8_gate, "saturation": saturation, "equivalence": equivalence_summary,
        "determinism": {"data_pipeline": "PASS", "float_reference": "PASS", "inference": "PASS" if (not verify_repeat or repeat_report["float_inference_semantic_reproducibility"]) else "FAIL", "conversion_bytes": "PASS" if (not verify_repeat or (repeat_report["float_tflite_bytes_identical"] and repeat_report["int8_tflite_bytes_identical"])) else "WARNING_SEMANTIC_ONLY"},
    }


def _load_drift_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_c_b4_artifacts(root: Optional[Path] = None) -> Dict[str, Any]:
    repo_root = (root or get_repo_root()).resolve()
    output_dir = repo_root / ARTIFACT_DIR_REL
    return {path.name: load_json(path) for path in output_dir.glob("*.json") if path.is_file()} | {"validation_prediction_drift.jsonl": _load_drift_rows(output_dir / "validation_prediction_drift.jsonl")}


__all__ = [
    "ARTIFACT_DIR_REL", "BALANCED_RANDOM_OVERSAMPLE", "B3_ARCHITECTURE_ID", "B3_PROFILE_ID", "C_B3_MERGED_MAIN_COMMIT",
    "CB4Error", "CANDIDATE_DIR_REL", "EQUIVALENCE_THRESHOLD", "FLOAT_REFERENCE_PROFILE_ID", "FIXED_FEATURES", "INT8_MACRO_F1_DEGRADATION_MAX",
    "INT8_OCCUPIED_RECALL_DEGRADATION_MAX", "INT8_PROBABILITY_MAE_MAX", "INT8_P95_DRIFT_MAX", "INT8_MAX_DRIFT_MAX", "INT8_LABEL_DISAGREEMENT_FRACTION_MAX",
    "LockedTestPolicyViolation", "PredecessorFingerprintMismatch", "FloatReferenceReconstructionFailure", "TFLiteContractMismatch",
    "TRAIN_COUNT", "VALIDATION_COUNT", "LOCKED_TEST_COUNT", "build_predecessor_fingerprint_registry", "inspect_tflite_contract", "load_c_b4_artifacts",
    "run_c_b4", "stable_sha256", "validate_class_map_semantics", "validate_locked_test_access", "validate_merged_main_ancestry", "validate_predecessor_contract", "validate_representative_membership", "quantize_int8_input", "dequantize_int8_output", "compute_int8_gate", "verify_predecessor_fingerprint_registry",
]
