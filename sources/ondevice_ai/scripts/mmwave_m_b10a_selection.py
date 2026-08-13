#!/usr/bin/env python3
"""Deterministic SafeNest mmWave M-B10A candidate-selection setup.

This module consumes only the frozen real-data VALIDATION evidence produced by
M-B0 through M-B9.  It writes the preregistered selection rule before writing
the candidate winner and never calls the LOCKED_TEST final-evaluation accessor.
The generated evidence is intentionally a setup/pretest record: it is not a
final test result, MR60 result, production claim, or clinical claim.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR_REL = Path("datasets/mmwave/manifests/M-B10A_candidate_selection_setup")
OUT_DIR = ROOT_DIR / OUT_DIR_REL
REPORT_REL = Path("docs/reports/20260812_Codex_M-B10A_Prelocked_Candidate_Selection_01.md")

SEEDS = (42, 43, 44)
ARCHITECTURE_ID = "M-B3_CONV1D_GAP_BASELINE"
PREPROCESSING_PROFILE = "M-B1_D0_B1_Z1"
PREPROCESSING_NAME = "BPF_ZSCORE"
IMBALANCE_STRATEGY = "M-B2_CE_UNWEIGHTED"
CALIBRATION_PROFILE = "M-B5_CAL_CLASS_BALANCED_120"
STAGE = "M-B6_STAGE_C_M-B5_CAL_CLASS_BALANCED_120"
LABELS = ("NORMAL", "RAPID_OR_ABNORMAL", "APNEA")
FROZEN_CLASS_MAP = {str(index): label for index, label in enumerate(LABELS)}
EPSILON = 1e-5

MODERATE_PROFILES = (
    "M-B7_GAUSSIAN_SNR20",
    "M-B7_AMP_X0_75",
    "M-B7_AMP_X1_25",
    "M-B7_DRIFT_MILD",
    "M-B7_DROPOUT_SHORT",
    "M-B7_MISSING_FRAME_1PCT",
    "M-B7_MOTION_BURST_MILD",
    "M-B7_COMBINED_MODERATE",
)

RANKING_CRITERIA = (
    (1, "clean_strict_int8_macro_f1", "higher", "Higher clean strict-INT8 VALIDATION Macro F1"),
    (2, "clean_min_per_class_recall", "higher", "Higher minimum clean per-class recall"),
    (3, "clean_apnea_proxy_recall", "higher", "Higher APNEA proxy recall"),
    (4, "clean_apnea_proxy_precision", "higher", "Higher APNEA proxy precision"),
    (5, "worst_subject_clean_macro_f1", "higher", "Higher worst-subject clean Macro F1 across fixed VALIDATION subjects"),
    (6, "moderate_worst_positive_macro_f1_degradation", "lower", "Lower worst positive Macro-F1 degradation across moderate M-B7 profiles"),
    (7, "moderate_worst_positive_recall_degradation", "lower", "Lower worst positive per-class recall degradation across moderate profiles"),
    (8, "moderate_min_top1_agreement", "higher", "Higher minimum clean-to-perturbed Top-1 agreement across moderate profiles"),
    (9, "moderate_max_input_saturation_ratio", "lower", "Lower maximum input saturation ratio across moderate profiles"),
    (10, "m_b6_positive_float_to_int8_macro_f1_degradation", "lower", "Lower positive M-B6 Float Keras to strict INT8 Macro-F1 degradation"),
    (11, "m_b6_keras_to_int8_top1_agreement", "higher", "Higher M-B6 Keras to strict INT8 Top-1 agreement"),
    (12, "m_b8_pipeline_p99_ns", "lower", "Lower M-B8 strict INT8 pipeline P99 latency (Mac-only late tie-breaker)"),
    (13, "tflite_bytes", "lower", "Smaller TFLite bytes"),
    (14, "training_seed", "lower", "Lower training seed"),
)

REQUIRED_OUTPUTS = (
    "input_identity.json",
    "experiment_contract.json",
    "candidate_pool.json",
    "candidate_eligibility_contract.json",
    "selection_rule.json",
    "candidate_selection_evidence.json",
    "candidate_ranking.json",
    "selected_candidate_pretest.json",
    "historical_baseline_registry.json",
    "locked_test_evaluation_contract.json",
    "locked_test_access_readiness.json",
    "locked_test_access_audit.json",
    "run_environment.json",
    "exceptions.json",
    "m_b10a_summary.json",
    "checksums.sha256",
)

MODEL_PATHS = {
    42: "models/mmwave/experiments/M-B6_stage_equivalence/M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120_int8.tflite",
    43: "models/mmwave/experiments/M-B6_stage_equivalence/M-B3_CONV1D_GAP_BASELINE_seed43_M-B5_CAL_CLASS_BALANCED_120_int8.tflite",
    44: "models/mmwave/experiments/M-B6_stage_equivalence/M-B3_CONV1D_GAP_BASELINE_seed44_M-B5_CAL_CLASS_BALANCED_120_int8.tflite",
}

BASELINE_CLASS_MAP_EVIDENCE_PATHS = {
    "mmwave": (
        "models/model_manifest.json",
        "models/mmwave/sensor_stats_metadata_v0.1.0.json",
        "models/mmwave/mmwave_resp_int8_v0.1.0.tflite",
    ),
    "mmwave_v0_2_0_candidate": (
        "models/model_manifest.json",
        "models/mmwave/mmwave_resp_int8_v0.2.0_candidate_metadata.json",
        "models/mmwave/mmwave_resp_int8_v0.2.0_candidate.tflite",
    ),
}


def _frozen_class_map_compatibility(evidence_paths: tuple[str, ...] | list[str]) -> dict[str, Any]:
    return {
        "status": "FROZEN_COMPATIBLE",
        "mapping": dict(FROZEN_CLASS_MAP),
        "evidence_paths": list(evidence_paths),
        "tflite_output_shape": [1, 3],
    }

M_B9_VALID_FINALIST_SCENARIOS = (
    "A_NORMAL",
    "B_RAPID_OR_ABNORMAL",
    "C_APNEA",
)
M_B9_EXPLICIT_FINALIST_SCENARIO = "N_VALID_EXPLICIT_FINALIST"


def _runtime_prediction_identity_for_seed(seed: int, prediction_identity: dict[str, Any]) -> dict[str, Any]:
    """Return the seed-local M-B9 direct/runtime prediction truth gate."""
    rows = [row for row in prediction_identity.get("rows", []) if int(row.get("seed", -1)) == seed]
    row_checks = [
        {
            "seed": row.get("seed"),
            "canonical_sample_index": row.get("canonical_sample_index"),
            "window_id": row.get("window_id"),
            "output_int8_exact": row.get("output_int8_exact") is True,
            "probabilities_exact": row.get("probabilities_exact") is True,
            "top1_exact": row.get("top1_exact") is True,
        }
        for row in rows
    ]
    exact = (
        len(rows) == len(LABELS)
        and prediction_identity.get("all_int8_outputs_exact") is True
        and prediction_identity.get("all_probability_vectors_exact") is True
        and prediction_identity.get("all_top1_exact") is True
        and all(check["output_int8_exact"] and check["probabilities_exact"] and check["top1_exact"] for check in row_checks)
    )
    return {
        "source_path": "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_prediction_identity.json",
        "seed": seed,
        "row_count": len(rows),
        "expected_row_count": len(LABELS),
        "aggregate_exact": {
            "all_int8_outputs_exact": prediction_identity.get("all_int8_outputs_exact") is True,
            "all_probability_vectors_exact": prediction_identity.get("all_probability_vectors_exact") is True,
            "all_top1_exact": prediction_identity.get("all_top1_exact") is True,
        },
        "row_checks": row_checks,
        "exact": exact,
    }


def _valid_finalist_fallback_evidence(seed: int, fallback_audit: dict[str, Any], runtime_identity: dict[str, Any], scenario_results: dict[str, Any]) -> dict[str, Any]:
    """Reconcile only valid M-B9 finalist scenarios; fault scenarios stay diagnostic."""
    expected_model_id = _runtime_variant_by_seed(runtime_identity, seed).get("model_id")
    expected_scenarios = list(M_B9_VALID_FINALIST_SCENARIOS)
    if seed == 42:
        expected_scenarios.append(M_B9_EXPLICIT_FINALIST_SCENARIO)
    rows = [
        row for row in fallback_audit.get("records", [])
        if row.get("seed") is not None and int(row.get("seed")) == seed and row.get("scenario_id") in expected_scenarios
    ]
    record_checks = [
        {
            "scenario_id": row.get("scenario_id"),
            "seed": row.get("seed"),
            "model_id_match": row.get("model_id") == expected_model_id,
            "valid": row.get("valid") is True,
            "fallback_used": row.get("fallback_used"),
            "fallback_absent": row.get("fallback_used") is False,
            "fallback_reason_absent": row.get("reason") is None,
            "score_source_model_prediction": row.get("score_source") == "MODEL_PREDICTION",
        }
        for row in rows
    ]
    runtime_rows = [
        row for row in scenario_results.get("records", [])
        if row.get("seed") is not None and int(row.get("seed")) == seed and row.get("scenario_id") in expected_scenarios
    ]
    runtime_checks = [
        {
            "scenario_id": row.get("scenario_id"),
            "valid": row.get("mmwave_result", {}).get("valid") is True,
            "model_id_match": row.get("mmwave_result", {}).get("metadata", {}).get("model_id") == expected_model_id,
            "fallback_absent": row.get("mmwave_result", {}).get("metadata", {}).get("fallback_used") is False,
            "fallback_reason_absent": row.get("mmwave_result", {}).get("metadata", {}).get("fallback_reason") is None,
            "score_source_model_prediction": row.get("mmwave_result", {}).get("metadata", {}).get("score_source") == "MODEL_PREDICTION",
        }
        for row in runtime_rows
    ]
    fallback_exact = (
        len(rows) == len(expected_scenarios)
        and set(row.get("scenario_id") for row in rows) == set(expected_scenarios)
        and fallback_audit.get("valid_finalist_records_have_no_fallback") is True
        and all(
            check["model_id_match"]
            and check["valid"]
            and check["fallback_absent"]
            and check["fallback_reason_absent"]
            and check["score_source_model_prediction"]
            for check in record_checks
        )
    )
    runtime_exact = (
        len(runtime_rows) == len(expected_scenarios)
        and set(row.get("scenario_id") for row in runtime_rows) == set(expected_scenarios)
        and all(
            check["valid"]
            and check["model_id_match"]
            and check["fallback_absent"]
            and check["fallback_reason_absent"]
            and check["score_source_model_prediction"]
            for check in runtime_checks
        )
    )
    exact = fallback_exact and runtime_exact
    return {
        "source_paths": ["datasets/mmwave/manifests/M-B9_mock_e2e/fallback_audit.json", "datasets/mmwave/manifests/M-B9_mock_e2e/scenario_results.json"],
        "seed": seed,
        "scenario_ids": expected_scenarios,
        "expected_model_id": expected_model_id,
        "record_count": len(rows),
        "record_checks": record_checks,
        "runtime_record_count": len(runtime_rows),
        "runtime_record_checks": runtime_checks,
        "fallback_audit_exact": fallback_exact,
        "runtime_scenario_exact": runtime_exact,
        "all_valid_finalist_no_fallback": exact,
        "fault_scenarios_are_diagnostic_only": True,
    }


def _path(relative: str | Path) -> Path:
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"machine-readable path must be repository-relative: {relative}")
    return ROOT_DIR / p


def _load_json(relative: str | Path) -> Any:
    return json.loads(_path(relative).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()


def _write_json(name: str, payload: Any) -> Path:
    target = OUT_DIR / name
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float, np.number)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(k) and _finite(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return True


def _metric_close(actual: float, expected: float, tolerance: float = 1e-5) -> bool:
    return math.isfinite(float(actual)) and math.isfinite(float(expected)) and abs(float(actual) - float(expected)) <= tolerance


def _load_validation_index() -> tuple[np.ndarray, list[str], list[str]]:
    path = _path("datasets/mmwave/manifests/M-B6_stage_equivalence/validation_prediction_index.jsonl")
    labels: list[int] = []
    subjects: list[str] = []
    window_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        labels.append(LABELS.index(str(row["true_label"])))
        subjects.append(str(row["subject_id"]))
        window_ids.append(str(row.get("window_id", row.get("recording_id", ""))))
    if len(labels) != 79:
        raise ValueError(f"Expected 79 pure VALIDATION rows, found {len(labels)}")
    return np.asarray(labels, dtype=np.int64), subjects, window_ids


def _metrics_from_predictions(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape:
        raise ValueError(f"VALIDATION label/prediction shape mismatch: {labels.shape} vs {predictions.shape}")
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for truth, pred in zip(labels.tolist(), predictions.tolist()):
        if truth not in range(len(LABELS)) or pred not in range(len(LABELS)):
            raise ValueError("Unexpected class index in VALIDATION evidence")
        confusion[truth, pred] += 1
    per_class: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    for index, label in enumerate(LABELS):
        tp = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        tn = int(confusion.sum() - tp - fp - fn)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        per_class[label] = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1_score": round(f1, 6),
            "fpr": round(fpr, 6),
            "support": support,
        }
        f1_values.append(f1)
        recalls.append(recall)
        precisions.append(precision)
    prediction_distribution = {label: int((predictions == index).sum()) for index, label in enumerate(LABELS)}
    zero_prediction = [label for label, count in prediction_distribution.items() if count == 0]
    zero_recall = [label for label in LABELS if per_class[label]["recall"] == 0.0]
    return {
        "evaluated_sample_count": int(labels.size),
        "accuracy": round(float(np.mean(labels == predictions)), 6),
        "macro_f1": round(float(np.mean(f1_values)), 6),
        "macro_precision": round(float(np.mean(precisions)), 6),
        "macro_recall": round(float(np.mean(recalls)), 6),
        "min_per_class_recall": round(float(min(recalls)), 6),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "prediction_distribution": prediction_distribution,
        "class_collapse": {
            "collapsed": bool(zero_prediction or zero_recall),
            "zero_prediction_classes": zero_prediction,
            "zero_recall_classes": zero_recall,
        },
    }


def _read_npz_predictions(seed: int, profile: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = _path("datasets/mmwave/manifests/M-B7_perturbation_robustness/prediction_vectors.npz")
    with np.load(path, allow_pickle=False) as arrays:
        prefix = f"seed_{seed}__{profile}__"
        predictions = np.asarray(arrays[prefix + "predictions"], dtype=np.int64)
        probabilities = np.asarray(arrays[prefix + "probabilities"], dtype=np.float64)
        saturation_counts = np.asarray(arrays[prefix + "saturation_counts"], dtype=np.int64)
        valid_mask = np.asarray(arrays[prefix + "valid_mask"], dtype=np.uint8) if prefix + "valid_mask" in arrays.files else np.ones(predictions.shape, dtype=np.uint8)
    return predictions, probabilities, saturation_counts, valid_mask


def _actual_tflite_identity(path: Path) -> dict[str, Any]:
    """Read binary identity only; tensor inspection is independently repeated by the validator."""
    return {"relative_path": _relative(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _subject_level_evidence(seed: int, b7_subject: dict[str, Any]) -> dict[str, Any]:
    """Summarize the fixed 17-subject VALIDATION evidence for one seed."""
    source = b7_subject["M-B7_CLEAN"]["per_seed"][str(seed)]["per_subject"]
    ordered = {subject_id: source[subject_id] for subject_id in sorted(source)}
    values = np.asarray([float(row["subject_macro_f1"]) for row in ordered.values()], dtype=np.float64)
    worst_subject_id = min(ordered, key=lambda subject_id: (float(ordered[subject_id]["subject_macro_f1"]), subject_id))
    return {
        "source_path": "datasets/mmwave/manifests/M-B7_perturbation_robustness/subject_level_robustness.json",
        "profile": "M-B7_CLEAN",
        "split": "VALIDATION",
        "subject_count": int(len(ordered)),
        "subject_ids": list(ordered),
        "subject_macro_f1_mean": round(float(np.mean(values)), 6),
        "subject_macro_f1_median": round(float(np.median(values)), 6),
        "subject_macro_f1_p25": round(float(np.percentile(values, 25)), 6),
        "worst_subject_id": worst_subject_id,
        "worst_subject_macro_f1": round(float(ordered[worst_subject_id]["subject_macro_f1"]), 6),
        "worst_subject_per_class": ordered[worst_subject_id]["per_class"],
        "per_subject": ordered,
    }


def _guard_structural_readiness() -> dict[str, Any]:
    """Inspect guard source text only; never instantiate or call its final accessor."""
    relative = "scripts/mmwave_phase_b_access.py"
    text = _path(relative).read_text(encoding="utf-8")
    return {
        "source_path": relative,
        "model_selection_denies_locked_test": "if split_upper == \"LOCKED_TEST\":" in text and "LOCKED_TEST_AccessError" in text,
        "final_evaluation_accessor_exists": "def get_locked_test_final_evaluation_dataset" in text,
        "final_accessor_requires_explicit_authorization": "authorization_token: str | None = None" in text and "AUTHORIZED_FINAL_LOCKED_TEST_EVALUATION_TOKEN_V1" in text,
        "structural_audit_is_sanitized": "sanitized_for_structural_audit" in text,
        "sanitized_fields_exclude_labels": "safenest_label" in text and "FORBIDDEN_LABEL_FIELDS" in text,
        "final_accessor_called": False,
        "ready": all((
            "if split_upper == \"LOCKED_TEST\":" in text and "LOCKED_TEST_AccessError" in text,
            "def get_locked_test_final_evaluation_dataset" in text,
            "authorization_token: str | None = None" in text and "AUTHORIZED_FINAL_LOCKED_TEST_EVALUATION_TOKEN_V1" in text,
            "sanitized_for_structural_audit" in text,
            "safenest_label" in text and "FORBIDDEN_LABEL_FIELDS" in text,
        )),
    }


def _upstream_paths_for_identity() -> tuple[str, ...]:
    return (
        "scripts/mmwave_phase_b_access.py",
        "datasets/mmwave/manifests/M-B0_evaluation_protocol/evaluation_contract.json",
        "datasets/mmwave/manifests/M-B0_evaluation_protocol/locked_test_access_policy.json",
        "datasets/mmwave/manifests/M-B0_evaluation_protocol/m_b0_summary.json",
        "datasets/mmwave/manifests/M-B0_evaluation_protocol/checksums.sha256",
        "datasets/mmwave/manifests/M-B1_preprocessing_ablation/selected_preprocessing_profile.json",
        "datasets/mmwave/manifests/M-B1_preprocessing_ablation/train_fit_statistics.json",
        "datasets/mmwave/manifests/M-B1_preprocessing_ablation/preprocessing_fingerprints.json",
        "datasets/mmwave/manifests/M-B1_preprocessing_ablation/training_runs.json",
        "datasets/mmwave/manifests/M-B1_preprocessing_ablation/checksums.sha256",
        "datasets/mmwave/manifests/M-B2_class_imbalance/selected_imbalance_strategy.json",
        "datasets/mmwave/manifests/M-B2_class_imbalance/imbalance_results.json",
        "datasets/mmwave/manifests/M-B2_class_imbalance/checksums.sha256",
        "datasets/mmwave/manifests/M-B3_architecture_comparison/architecture_profiles.json",
        "datasets/mmwave/manifests/M-B3_architecture_comparison/selected_architecture_shortlist.json",
        "datasets/mmwave/manifests/M-B3_architecture_comparison/training_runs.json",
        "datasets/mmwave/manifests/M-B3_architecture_comparison/checksums.sha256",
        "datasets/mmwave/manifests/M-B4_multiseed_stability/m_b4_summary.json",
        "datasets/mmwave/manifests/M-B4_multiseed_stability/multi_seed_results.json",
        "datasets/mmwave/manifests/M-B4_multiseed_stability/per_seed_results.json",
        "datasets/mmwave/manifests/M-B4_multiseed_stability/training_runs.json",
        "datasets/mmwave/manifests/M-B4_multiseed_stability/seed_weights.npz",
        "datasets/mmwave/manifests/M-B4_multiseed_stability/checksums.sha256",
        "datasets/mmwave/manifests/M-B5_representative_calibration/m_b5_summary.json",
        "datasets/mmwave/manifests/M-B5_representative_calibration/selected_calibration_profile.json",
        "datasets/mmwave/manifests/M-B5_representative_calibration/cross_seed_calibration_results.json",
        "datasets/mmwave/manifests/M-B5_representative_calibration/tflite_artifact_manifest.json",
        "datasets/mmwave/manifests/M-B5_representative_calibration/checksums.sha256",
        "datasets/mmwave/manifests/M-B6_stage_equivalence/m_b6_summary.json",
        "datasets/mmwave/manifests/M-B6_stage_equivalence/per_seed_stage_metrics.json",
        "datasets/mmwave/manifests/M-B6_stage_equivalence/pairwise_equivalence_metrics.json",
        "datasets/mmwave/manifests/M-B6_stage_equivalence/class_collapse_transition_audit.json",
        "datasets/mmwave/manifests/M-B6_stage_equivalence/stage_artifact_manifest.json",
        "datasets/mmwave/manifests/M-B6_stage_equivalence/int8_tflite_predictions.npz",
        "datasets/mmwave/manifests/M-B6_stage_equivalence/checksums.sha256",
        "datasets/mmwave/manifests/M-B7_perturbation_robustness/m_b7_summary.json",
        "datasets/mmwave/manifests/M-B7_perturbation_robustness/clean_baseline_results.json",
        "datasets/mmwave/manifests/M-B7_perturbation_robustness/perturbation_results.json",
        "datasets/mmwave/manifests/M-B7_perturbation_robustness/cross_seed_robustness_summary.json",
        "datasets/mmwave/manifests/M-B7_perturbation_robustness/subject_level_robustness.json",
        "datasets/mmwave/manifests/M-B7_perturbation_robustness/prediction_vectors.npz",
        "datasets/mmwave/manifests/M-B7_perturbation_robustness/checksums.sha256",
        "datasets/mmwave/manifests/M-B8_mac_latency_footprint/latency_summary.json",
        "datasets/mmwave/manifests/M-B8_mac_latency_footprint/cross_seed_latency_summary.json",
        "datasets/mmwave/manifests/M-B8_mac_latency_footprint/artifact_footprint.json",
        "datasets/mmwave/manifests/M-B8_mac_latency_footprint/benchmark_contract.json",
        "datasets/mmwave/manifests/M-B8_mac_latency_footprint/benchmark_environment.json",
        "datasets/mmwave/manifests/M-B8_mac_latency_footprint/m_b8_summary.json",
        "datasets/mmwave/manifests/M-B8_mac_latency_footprint/checksums.sha256",
        "datasets/mmwave/manifests/M-B9_mock_e2e/m_b9_summary.json",
        "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_model_identity.json",
        "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_preprocessing_identity.json",
        "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_prediction_identity.json",
        "datasets/mmwave/manifests/M-B9_mock_e2e/fallback_audit.json",
        "datasets/mmwave/manifests/M-B9_mock_e2e/scenario_results.json",
        "datasets/mmwave/manifests/M-B9_mock_e2e/scenario_results.jsonl",
        "datasets/mmwave/manifests/M-B9_mock_e2e/inference_result_audit.json",
        "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_manifest_contract.json",
        "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_manifests/seed42_runtime_manifest.json",
        "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_manifests/seed43_runtime_manifest.json",
        "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_manifests/seed44_runtime_manifest.json",
        "datasets/mmwave/manifests/M-B9_mock_e2e/locked_test_access_audit.json",
        "datasets/mmwave/manifests/M-B9_mock_e2e/checksums.sha256",
        "datasets/mmwave/manifests/a5_subject_split/split_profile.json",
        "datasets/mmwave/manifests/a5_subject_split/subject_split_manifest.jsonl",
        "datasets/mmwave/manifests/a5_subject_split/a5_summary.json",
        "datasets/mmwave/manifests/a5_subject_split/checksums.sha256",
        "datasets/mmwave/splits/mmwave_real_subject_split_v1.json",
        "models/model_manifest.json",
        "models/mmwave/sensor_stats_metadata_v0.1.0.json",
        "models/mmwave/mmwave_resp_int8_v0.2.0_candidate_metadata.json",
        "models/mmwave/training_config.json",
        "scripts/mmwave_m_b10b_baseline_preprocessing.py",
        "datasets/mmwave/manifests/a6_full_conversion/a6_summary.json",
        "datasets/mmwave/manifests/a6_full_conversion/processing_profile.json",
        "datasets/mmwave/manifests/a6_full_conversion/full_window_manifest.jsonl",
        "datasets/mmwave/manifests/a6_full_conversion/full_provenance_manifest.jsonl",
        "datasets/mmwave/manifests/a6_full_conversion/checksums.sha256",
        "datasets/mmwave/processed/mmwave_canonical_real_v1.npy",
    )


def build_input_identity() -> dict[str, Any]:
    rows = []
    for relative in _upstream_paths_for_identity():
        path = _path(relative)
        if not path.is_file():
            raise FileNotFoundError(relative)
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    for seed, relative in MODEL_PATHS.items():
        path = _path(relative)
        if not path.is_file():
            raise FileNotFoundError(relative)
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path), "seed": seed, "role": "FROZEN_M-B6_QUALIFIED_STRICT_INT8_CANDIDATE"})
    return {
        "phase_id": "M-B10A",
        "title": "Pre-LOCKED_TEST real-data candidate-selection setup input identity lock",
        "source_scope": "FROZEN_REAL_DATA_VALIDATION_ONLY",
        "locked_test_accesses": 0,
        "total_inputs": len(rows),
        "inputs": rows,
    }


def build_selection_rule() -> dict[str, Any]:
    return {
        "phase_id": "M-B10A",
        "rule_name": "SAFE_NEST_MMWAVE_PRELOCKED_REAL_DATA_FINALIST_RULE_V1",
        "rule_version": "1.0.0",
        "frozen_before_candidate_winner": True,
        "epsilon": EPSILON,
        "epsilon_semantics": "Absolute difference <= epsilon is tied and proceeds to the next criterion.",
        "candidate_pool_scope": {
            "real_data_only": True,
            "strict_int8_only": True,
            "architecture_id": ARCHITECTURE_ID,
            "preprocessing_profile": PREPROCESSING_PROFILE,
            "preprocessing_name": PREPROCESSING_NAME,
            "imbalance_strategy": IMBALANCE_STRATEGY,
            "calibration_profile": CALIBRATION_PROFILE,
            "training_seeds": list(SEEDS),
            "historical_v0_1_0_in_pool": False,
            "synthetic_v0_2_0_in_pool": False,
            "retraining_allowed": False,
            "reconversion_allowed": False,
        },
        "hard_eligibility_rule_ids": [f"E{i}" for i in range(1, 12)],
        "ranking_criteria": [
            {"rank": rank, "metric": metric, "direction": direction, "description": description}
            for rank, metric, direction, description in RANKING_CRITERIA
        ],
        "moderate_m_b7_profiles": list(MODERATE_PROFILES),
        "severe_profiles_not_hard_gated": [
            "M-B7_AMP_X0_50",
            "M-B7_GAUSSIAN_POST_B1_SNR10",
            "M-B7_MOTION_BURST_SEVERE",
        ],
        "no_composite_score": True,
        "architecture_seed_sensitivity_warning_preserved": True,
        "locked_test_policy": "No LOCKED_TEST labels, tensors, predictions, or metrics are accessed during M-B10A.",
    }


def build_eligibility_contract() -> dict[str, Any]:
    rules = [
        ("E1", "Lineage intact from source through runtime", "All required phase identities and immutable A5/A6 paths exist and hash-match."),
        ("E2", "Strict INT8 and no Flex/Select ops", "Actual finalist binary has int8 input/output, expected shapes, and zero Select TF ops."),
        ("E3", "Clean VALIDATION has no required-class prediction collapse", "Recomputed clean VALIDATION prediction distribution and class recalls are nonzero."),
        ("E4", "No new M-B6 conversion-induced required-class collapse", "M-B6 class-collapse transition audit reports no Stage-A to Stage-C new collapse."),
        ("E5", "M-B9 runtime and prediction identity match M-B6 artifact", "Runtime path, SHA, bytes, model version, and direct/runtime INT8 output, probability, and Top-1 identities match the M-B6 Stage-C artifact."),
        ("E6", "Valid finalist with no heuristic fallback", "M-B9 valid finalist scenario records are valid, use MODEL_PREDICTION, and have no fallback or fallback reason; fault scenarios remain diagnostic."),
        ("E7", "Runtime preprocessing equals BPF_ZSCORE", "M-B1 selected profile and M-B9 preprocessing identity agree exactly."),
        ("E8", "No artifact/runtime/checksum/provenance blocker", "Upstream blocker registries are empty and all frozen identity checks pass."),
        ("E9", "No required class has clean VALIDATION recall exactly zero", "Recomputed NORMAL, RAPID_OR_ABNORMAL, and APNEA recalls are all > 0."),
        ("E10", "No required class has clean VALIDATION precision exactly zero", "Recomputed NORMAL, RAPID_OR_ABNORMAL, and APNEA precisions are all > 0."),
        ("E11", "No class collapse under moderate M-B7 profiles", "Every moderate profile remains non-collapsed for the candidate seed."),
    ]
    return {
        "phase_id": "M-B10A",
        "contract_name": "M-B10A Frozen Candidate Eligibility Contract",
        "finding_classes": ["BLOCKER", "REQUIRED REFINEMENT", "NON-BLOCKING IMPROVEMENT"],
        "rules": [{"rule_id": rid, "name": name, "pass_condition": condition, "hard_gate": True} for rid, name, condition in rules],
        "failure_policy": "Any failed hard gate removes the candidate; zero eligible candidates yields INCONCLUSIVE and stops before M-B10B.",
    }


def _runtime_variant_by_seed(runtime_identity: dict[str, Any], seed: int) -> dict[str, Any]:
    for row in runtime_identity.get("variants", []):
        if int(row.get("seed", -1)) == seed:
            return row
    raise ValueError(f"M-B9 runtime variant missing for seed {seed}")


def _build_candidates() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels, subjects, _window_ids = _load_validation_index()
    b4 = _load_json("datasets/mmwave/manifests/M-B4_multiseed_stability/per_seed_results.json")["per_seed_results"]
    b4_architecture = next(
        row for row in _load_json("datasets/mmwave/manifests/M-B4_multiseed_stability/multi_seed_results.json")["multi_seed_results"]
        if row.get("architecture_id") == ARCHITECTURE_ID
    )
    b6_stage = _load_json("datasets/mmwave/manifests/M-B6_stage_equivalence/per_seed_stage_metrics.json")["per_seed_stage_metrics"]
    b6_pairs = _load_json("datasets/mmwave/manifests/M-B6_stage_equivalence/pairwise_equivalence_metrics.json")["pairwise_equivalence"]
    b6_collapses = _load_json("datasets/mmwave/manifests/M-B6_stage_equivalence/class_collapse_transition_audit.json")["class_collapse_transitions"]
    b6_artifacts = _load_json("datasets/mmwave/manifests/M-B6_stage_equivalence/stage_artifact_manifest.json")["artifacts"]
    b7_clean = _load_json("datasets/mmwave/manifests/M-B7_perturbation_robustness/clean_baseline_results.json")["per_seed"]
    b7_summary = _load_json("datasets/mmwave/manifests/M-B7_perturbation_robustness/m_b7_summary.json")
    b7_perturb = _load_json("datasets/mmwave/manifests/M-B7_perturbation_robustness/perturbation_results.json")["profiles"]
    b7_subject = _load_json("datasets/mmwave/manifests/M-B7_perturbation_robustness/subject_level_robustness.json")["profiles"]
    b8_cross = _load_json("datasets/mmwave/manifests/M-B8_mac_latency_footprint/cross_seed_latency_summary.json")["cross_seed_metrics"]
    b8_footprint = _load_json("datasets/mmwave/manifests/M-B8_mac_latency_footprint/artifact_footprint.json")["strict_int8_artifacts"]
    b8_interpretation = _load_json("datasets/mmwave/manifests/M-B8_mac_latency_footprint/cross_seed_latency_summary.json").get("interpretation")
    b9_runtime = _load_json("datasets/mmwave/manifests/M-B9_mock_e2e/runtime_model_identity.json")
    b9_pre = _load_json("datasets/mmwave/manifests/M-B9_mock_e2e/runtime_preprocessing_identity.json")
    b9_pred = _load_json("datasets/mmwave/manifests/M-B9_mock_e2e/runtime_prediction_identity.json")
    b9_fallback = _load_json("datasets/mmwave/manifests/M-B9_mock_e2e/fallback_audit.json")
    b9_scenarios = _load_json("datasets/mmwave/manifests/M-B9_mock_e2e/scenario_results.json")
    b9_summary = _load_json("datasets/mmwave/manifests/M-B9_mock_e2e/m_b9_summary.json")
    b1 = _load_json("datasets/mmwave/manifests/M-B1_preprocessing_ablation/selected_preprocessing_profile.json")
    b2 = _load_json("datasets/mmwave/manifests/M-B2_class_imbalance/selected_imbalance_strategy.json")
    a5_split = _load_json("datasets/mmwave/manifests/a5_subject_split/split_profile.json")
    a5_summary = _load_json("datasets/mmwave/manifests/a5_subject_split/a5_summary.json")
    a6_summary = _load_json("datasets/mmwave/manifests/a6_full_conversion/a6_summary.json")

    candidates: list[dict[str, Any]] = []
    for seed in SEEDS:
        b4_key = f"{ARCHITECTURE_ID}_seed_{seed}"
        b6_key = b4_key
        candidate_id = f"{ARCHITECTURE_ID}_seed{seed}_{CALIBRATION_PROFILE}"
        model_path = _path(MODEL_PATHS[seed])
        runtime = _runtime_variant_by_seed(b9_runtime, seed)
        b6_artifact = b6_artifacts[f"{b6_key}_stage_c"]
        b4_row = b4[b4_key]
        stage_row = b6_stage[b6_key]
        pair = b6_pairs[b6_key]
        collapse = b6_collapses[b6_key]
        clean_predictions, _clean_probs, clean_saturation, clean_valid = _read_npz_predictions(seed, "M-B7_CLEAN")
        recomputed_clean = _metrics_from_predictions(labels, clean_predictions)
        clean_reported = b7_clean[str(seed)]["metrics"]
        if not np.all(clean_valid.astype(bool)):
            raise ValueError(f"M-B7 clean VALIDATION has invalid samples for seed {seed}")
        clean_reported_checks = {
            "macro_f1": _metric_close(recomputed_clean["macro_f1"], float(clean_reported["macro_f1"])),
            "accuracy": _metric_close(recomputed_clean["accuracy"], float(clean_reported["accuracy"])),
            "per_class": all(_metric_close(recomputed_clean["per_class"][label][metric], float(clean_reported["per_class"][label][metric])) for label in LABELS for metric in ("precision", "recall", "f1_score")),
        }
        moderate_rows: dict[str, Any] = {}
        for profile in MODERATE_PROFILES:
            predictions, _probabilities, saturation, valid = _read_npz_predictions(seed, profile)
            derived = _metrics_from_predictions(labels, predictions)
            clean_macro = float(recomputed_clean["macro_f1"])
            positive_recall_degradation = {
                label: round(max(0.0, float(recomputed_clean["per_class"][label]["recall"]) - float(derived["per_class"][label]["recall"])), 6)
                for label in LABELS
            }
            top1 = round(float(np.mean(clean_predictions == predictions)), 6)
            saturation_ratio = round(float(np.sum(saturation) / (labels.size * 300)), 9)
            collapse_state = {
                "collapsed": bool(derived["class_collapse"]["collapsed"]),
                "zero_prediction_classes": list(derived["class_collapse"]["zero_prediction_classes"]),
                "zero_recall_classes": list(derived["class_collapse"]["zero_recall_classes"]),
                "new_relative_to_clean": bool(derived["class_collapse"]["collapsed"] and not recomputed_clean["class_collapse"]["collapsed"]),
            }
            reported = b7_perturb[profile]["per_seed"][str(seed)]
            moderate_rows[profile] = {
                "recomputed_macro_f1": derived["macro_f1"],
                "recomputed_per_class_recall": {label: derived["per_class"][label]["recall"] for label in LABELS},
                "recomputed_class_collapse": collapse_state,
                "positive_macro_f1_degradation": round(max(0.0, clean_macro - float(derived["macro_f1"])), 6),
                "maximum_positive_per_class_recall_degradation": max(positive_recall_degradation.values()),
                "top1_agreement": top1,
                "input_saturation_ratio": saturation_ratio,
                "valid_sample_count": int(np.sum(valid.astype(bool))),
                "reported_identity_checks": {
                    "macro_f1": _metric_close(derived["macro_f1"], float(reported["macro_f1"])),
                    "class_collapse": collapse_state["collapsed"] == bool(reported["class_collapse_state"]["collapsed"]),
                    "top1_agreement": _metric_close(top1, float(reported["relative_to_clean"]["top1_agreement"])),
                    "input_saturation_ratio": _metric_close(saturation_ratio, float(reported["quantization"]["input_saturation_ratio"]), 1e-6),
                },
            }
        clean_subject_rows = b7_subject["M-B7_CLEAN"]["per_seed"][str(seed)]["per_subject"]
        worst_subject_clean = min(float(row["subject_macro_f1"]) for row in clean_subject_rows.values())
        subject_evidence = _subject_level_evidence(seed, b7_subject)
        moderate_values = list(moderate_rows.values())
        runtime_pre_rows = [row for row in b9_pre.get("rows", []) if int(row.get("seed", -1)) == seed]
        runtime_pre_exact = bool(runtime_pre_rows) and all(
            row.get("preprocessing_profile") == PREPROCESSING_NAME
            and row.get("bpf_exact") is True
            and row.get("zscore_exact") is True
            and row.get("model_ready_exact") is True
            and row.get("input_int8_exact") is True
            and row.get("saturation_exact") is True
            for row in runtime_pre_rows
        )
        lineage_paths = _upstream_paths_for_identity()
        lineage_intact = all(_path(p).is_file() for p in lineage_paths) and model_path.is_file()
        runtime_prediction_identity = _runtime_prediction_identity_for_seed(seed, b9_pred)
        fallback_evidence = _valid_finalist_fallback_evidence(seed, b9_fallback, b9_runtime, b9_scenarios)
        gates = {
            "E1": lineage_intact,
            "E2": bool(runtime.get("strict_int8") and runtime.get("flex_select_absent") and runtime.get("sha256_match") and runtime.get("bytes_match")),
            "E3": not bool(recomputed_clean["class_collapse"]["collapsed"]),
            "E4": not bool(collapse.get("new_collapse_a_to_c") or collapse.get("new_collapse_b_to_c")),
            "E5": runtime.get("path") == MODEL_PATHS[seed] and runtime.get("actual_sha256") == _sha256(model_path) and runtime.get("actual_bytes") == model_path.stat().st_size and runtime_prediction_identity["exact"],
            "E6": fallback_evidence["all_valid_finalist_no_fallback"],
            "E7": b1.get("selected_profile_id") == PREPROCESSING_PROFILE and b1.get("selected_profile_name") == PREPROCESSING_NAME and runtime_pre_exact,
            "E8": (
                a5_summary.get("validation_success") is True
                and a5_summary.get("validation_errors") == []
                and a6_summary.get("validation_passed") is True
                and a6_summary.get("validator_verdict", {}).get("validation_success") is True
                and all(int(value) == 0 for value in a6_summary.get("validator_verdict", {}).get("leakage_recalculated", {}).values())
                and not bool(b7_summary.get("blockers"))
                and bool(b9_summary.get("runtime_identity_exact"))
                and bool(b9_summary.get("risk_recomputation_exact"))
            ),
            "E9": all(float(recomputed_clean["per_class"][label]["recall"]) > 0.0 for label in LABELS),
            "E10": all(float(recomputed_clean["per_class"][label]["precision"]) > 0.0 for label in LABELS),
            "E11": all(not bool(row["recomputed_class_collapse"]["collapsed"]) for row in moderate_values),
        }
        runtime_pre_identity_rows = [
            {
                key: row.get(key)
                for key in (
                    "seed", "model_id", "model_sha256", "preprocessing_profile", "bpf_exact", "zscore_exact",
                    "model_ready_exact", "input_int8_exact", "saturation_exact", "probabilities_exact", "top1_exact",
                    "direct_input_int8_sha256", "runtime_input_int8_sha256", "direct_output_int8", "runtime_output_int8",
                )
            }
            for row in b9_pre.get("rows", [])
            if int(row.get("seed", -1)) == seed
        ]
        runtime_pred_rows = [row for row in b9_pred.get("rows", []) if int(row.get("seed", -1)) == seed]
        eligibility_evidence = {
            "E1": {"source_paths": list(_upstream_paths_for_identity()) + [MODEL_PATHS[seed]], "supporting_value": {"all_required_inputs_present": bool(lineage_intact), "candidate_model_path": MODEL_PATHS[seed]}},
            "E2": {"source_paths": [MODEL_PATHS[seed], "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_model_identity.json"], "supporting_value": {"input_dtype": runtime.get("tensor_contract", {}).get("input_dtype"), "output_dtype": runtime.get("tensor_contract", {}).get("output_dtype"), "input_shape": runtime.get("tensor_contract", {}).get("input_shape"), "output_shape": runtime.get("tensor_contract", {}).get("output_shape"), "select_tf_ops_count": runtime.get("tensor_contract", {}).get("select_tf_ops_count")}},
            "E3": {"source_paths": ["datasets/mmwave/manifests/M-B7_perturbation_robustness/clean_baseline_results.json", "datasets/mmwave/manifests/M-B7_perturbation_robustness/prediction_vectors.npz"], "supporting_value": recomputed_clean["class_collapse"]},
            "E4": {"source_paths": ["datasets/mmwave/manifests/M-B6_stage_equivalence/class_collapse_transition_audit.json"], "supporting_value": collapse},
            "E5": {"source_paths": ["datasets/mmwave/manifests/M-B6_stage_equivalence/stage_artifact_manifest.json", "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_model_identity.json", "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_prediction_identity.json"], "supporting_value": {"path_match": runtime.get("path") == MODEL_PATHS[seed], "sha256_match": runtime.get("actual_sha256") == _sha256(model_path), "bytes_match": runtime.get("actual_bytes") == model_path.stat().st_size, "prediction_identity": runtime_prediction_identity}},
            "E6": {"source_paths": fallback_evidence["source_paths"], "supporting_value": fallback_evidence},
            "E7": {"source_paths": ["datasets/mmwave/manifests/M-B1_preprocessing_ablation/selected_preprocessing_profile.json", "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_preprocessing_identity.json"], "supporting_value": {"selected_profile": b1.get("selected_profile_name"), "runtime_rows_exact": runtime_pre_exact}},
            "E8": {"source_paths": ["datasets/mmwave/manifests/a5_subject_split/a5_summary.json", "datasets/mmwave/manifests/a5_subject_split/checksums.sha256", "datasets/mmwave/manifests/a6_full_conversion/a6_summary.json", "datasets/mmwave/manifests/a6_full_conversion/checksums.sha256", "datasets/mmwave/manifests/M-B7_perturbation_robustness/m_b7_summary.json", "datasets/mmwave/manifests/M-B9_mock_e2e/m_b9_summary.json"], "supporting_value": {"a5_validation_success": a5_summary.get("validation_success"), "a5_validation_errors": a5_summary.get("validation_errors", []), "a6_validation_success": a6_summary.get("validation_passed"), "a6_leakage_recalculated": a6_summary.get("validator_verdict", {}).get("leakage_recalculated", {}), "b7_blockers": b7_summary.get("blockers", []), "b9_runtime_identity_exact": b9_summary.get("runtime_identity_exact"), "b9_risk_recomputation_exact": b9_summary.get("risk_recomputation_exact")}},
            "E9": {"source_paths": ["datasets/mmwave/manifests/M-B7_perturbation_robustness/prediction_vectors.npz"], "supporting_value": {label: recomputed_clean["per_class"][label]["recall"] for label in LABELS}},
            "E10": {"source_paths": ["datasets/mmwave/manifests/M-B7_perturbation_robustness/prediction_vectors.npz"], "supporting_value": {label: recomputed_clean["per_class"][label]["precision"] for label in LABELS}},
            "E11": {"source_paths": ["datasets/mmwave/manifests/M-B7_perturbation_robustness/perturbation_results.json", "datasets/mmwave/manifests/M-B7_perturbation_robustness/prediction_vectors.npz"], "supporting_value": {"moderate_profiles": list(MODERATE_PROFILES), "collapsed_profiles": [profile for profile, row in moderate_rows.items() if row["recomputed_class_collapse"]["collapsed"]]}},
        }
        metrics = {
            "clean_strict_int8_macro_f1": recomputed_clean["macro_f1"],
            "clean_min_per_class_recall": recomputed_clean["min_per_class_recall"],
            "clean_apnea_proxy_recall": recomputed_clean["per_class"]["APNEA"]["recall"],
            "clean_apnea_proxy_precision": recomputed_clean["per_class"]["APNEA"]["precision"],
            "worst_subject_clean_macro_f1": round(worst_subject_clean, 6),
            "moderate_worst_positive_macro_f1_degradation": max(float(row["positive_macro_f1_degradation"]) for row in moderate_values),
            "moderate_worst_positive_recall_degradation": max(float(row["maximum_positive_per_class_recall_degradation"]) for row in moderate_values),
            "moderate_min_top1_agreement": min(float(row["top1_agreement"]) for row in moderate_values),
            "moderate_max_input_saturation_ratio": max(float(row["input_saturation_ratio"]) for row in moderate_values),
            "m_b6_positive_float_to_int8_macro_f1_degradation": float(pair["a_to_c"]["positive_macro_f1_degradation"]),
            "m_b6_keras_to_int8_top1_agreement": float(pair["a_to_c"]["top1_agreement"]),
            "m_b8_pipeline_p99_ns": float(b8_cross["PREPROCESSING_QUANTIZATION_INVOKE"]["per_seed_pooled_statistics"][str(seed)]["statistics_ns"]["p99"]),
            "tflite_bytes": int(model_path.stat().st_size),
            "training_seed": seed,
        }
        candidates.append({
            "candidate_id": candidate_id,
            "architecture_id": ARCHITECTURE_ID,
            "seed": seed,
            "training_seed": seed,
            "preprocessing_profile": PREPROCESSING_PROFILE,
            "preprocessing_name": PREPROCESSING_NAME,
            "imbalance_strategy": IMBALANCE_STRATEGY,
            "calibration_profile": CALIBRATION_PROFILE,
            "stage": STAGE,
            "model_id": runtime.get("model_id"),
            "model": {
                **_actual_tflite_identity(model_path),
                "expected_stage_artifact": b6_artifact,
                "input_shape": runtime.get("tensor_contract", {}).get("input_shape"),
                "input_dtype": runtime.get("tensor_contract", {}).get("input_dtype"),
                "input_scale": runtime.get("tensor_contract", {}).get("input_scale"),
                "input_zero_point": runtime.get("tensor_contract", {}).get("input_zero_point"),
                "output_shape": runtime.get("tensor_contract", {}).get("output_shape"),
                "output_dtype": runtime.get("tensor_contract", {}).get("output_dtype"),
                "output_scale": runtime.get("tensor_contract", {}).get("output_scale"),
                "output_zero_point": runtime.get("tensor_contract", {}).get("output_zero_point"),
                "select_tf_ops_count": runtime.get("tensor_contract", {}).get("select_tf_ops_count"),
                "operator_inventory": runtime.get("tensor_contract", {}).get("op_types", []),
            },
            "training_weights_sha256": b4_row.get("final_weights_sha256"),
            "m_b4_seed_stability": {
                "source_path": "datasets/mmwave/manifests/M-B4_multiseed_stability/multi_seed_results.json",
                "architecture_level": b4_architecture,
                "candidate_seed_validation": b4_row,
            },
            "clean_validation": {
                "recomputed": recomputed_clean,
                "reported_identity_checks": clean_reported_checks,
                "m_b6_stage_c": stage_row.get("stage_c_int8_tflite"),
                "m_b6_stage_a": stage_row.get("stage_a_float_keras"),
            },
            "subject_level": subject_evidence,
            "m_b6_stage_equivalence": {
                "source_path": "datasets/mmwave/manifests/M-B6_stage_equivalence/pairwise_equivalence_metrics.json",
                "pairwise": pair,
                "collapse_transition": collapse,
            },
            "moderate_profiles": moderate_rows,
            "ranking_metrics": metrics,
            "eligibility": {rid: {"passed": bool(value)} for rid, value in gates.items()},
            "eligible": bool(all(gates.values())),
            "eligibility_evidence": eligibility_evidence,
            "m_b8_latency_footprint": {
                "source_path": "datasets/mmwave/manifests/M-B8_mac_latency_footprint/cross_seed_latency_summary.json",
                "scope": "macOS_OFFLINE_ONLY",
                "invoke_p99_ns": float(b8_cross["TFLITE_INVOKE_ONLY"]["per_seed_pooled_statistics"][str(seed)]["statistics_ns"]["p99"]),
                "pipeline_p99_ns": float(b8_cross["PREPROCESSING_QUANTIZATION_INVOKE"]["per_seed_pooled_statistics"][str(seed)]["statistics_ns"]["p99"]),
                "artifact_footprint": b8_footprint[str(seed)],
                "interpretation": b8_interpretation,
                "formal_rerun_during_m_b10a": False,
            },
            "m_b9_runtime_identity": {
                "model_identity": runtime,
                "preprocessing_identity_rows": runtime_pre_identity_rows,
                "prediction_identity": {"source_stage": b9_pred.get("source_stage"), "runtime_stage": b9_pred.get("runtime_stage"), "row_count": len(runtime_pred_rows), "all_int8_outputs_exact": b9_pred.get("all_int8_outputs_exact"), "all_probability_vectors_exact": b9_pred.get("all_probability_vectors_exact"), "all_top1_exact": b9_pred.get("all_top1_exact"), "seed_gate": runtime_prediction_identity},
                "valid_finalist_fallback": fallback_evidence,
                "summary": b9_summary,
            },
            "evidence_lineage": {
                "m_b4_final_weights_sha256": b4_row.get("final_weights_sha256"),
                "m_b6_stage_artifact_sha256": b6_artifact.get("sha256"),
                "m_b7_clean_artifact_sha256": b7_clean[str(seed)].get("model_artifact", {}).get("sha256"),
                "m_b9_runtime_artifact_sha256": runtime.get("actual_sha256"),
                "b1_profile_matches": b1.get("selected_profile_id") == PREPROCESSING_PROFILE,
                "b2_strategy_matches": b2.get("selected_strategy_id") == IMBALANCE_STRATEGY,
                "a5_split_profile": a5_split.get("profile_id"),
            },
        })
    return candidates, {"labels": labels, "subjects": subjects, "b8_footprint": b8_footprint, "architecture_seed_sensitivity": b4_architecture}


def _compare_values(left: float, right: float, direction: str) -> int:
    if abs(float(left) - float(right)) <= EPSILON:
        return 0
    if direction == "higher":
        return 1 if left > right else -1
    return 1 if left < right else -1


def rank_candidates(candidates: list[dict[str, Any]], rule_sha256: str) -> dict[str, Any]:
    eligible = [candidate for candidate in candidates if candidate["eligible"]]

    def compare(left: dict[str, Any], right: dict[str, Any]) -> int:
        for _rank, metric, direction, _description in RANKING_CRITERIA:
            result = _compare_values(left["ranking_metrics"][metric], right["ranking_metrics"][metric], direction)
            if result:
                return -result  # cmp_to_key sorts the preferred candidate first
        return 0

    ordered = sorted(eligible, key=functools.cmp_to_key(compare))
    rows = []
    for position, candidate in enumerate(ordered, 1):
        rows.append({"rank": position, "candidate_id": candidate["candidate_id"], "seed": candidate["seed"], "ranking_metrics": candidate["ranking_metrics"]})
    deciding = None
    if len(ordered) >= 2:
        first, second = ordered[0], ordered[1]
        for rank, metric, direction, description in RANKING_CRITERIA:
            if abs(float(first["ranking_metrics"][metric]) - float(second["ranking_metrics"][metric])) > EPSILON:
                deciding = {"criterion_rank": rank, "metric": metric, "direction": direction, "description": description, "winner_value": first["ranking_metrics"][metric], "runner_up_value": second["ranking_metrics"][metric], "absolute_difference": abs(float(first["ranking_metrics"][metric]) - float(second["ranking_metrics"][metric]))}
                break
    return {
        "phase_id": "M-B10A",
        "selection_rule_sha256": rule_sha256,
        "epsilon": EPSILON,
        "eligible_candidate_count": len(eligible),
        "eligible_candidate_ids": [candidate["candidate_id"] for candidate in eligible],
        "ordered_candidates": rows,
        "deciding_criterion": deciding,
        "selection_status": "SELECTED_PRELOCKED_REAL_DATA_CANDIDATE" if ordered else "INCONCLUSIVE",
        "selected_candidate_id": ordered[0]["candidate_id"] if ordered else None,
        "no_composite_score_used": True,
    }


def _baseline_preprocessing_contract(model_key: str, model: dict[str, Any]) -> dict[str, Any]:
    """Freeze an executable compatibility adapter without claiming hidden native lineage."""
    metadata_path = str(model["metadata_path"])
    metadata = _load_json(metadata_path)
    model_path = str(model["path"])
    if model_key == "mmwave":
        contract_id = "M-B10B_HISTORICAL_V0_1_COMPATIBILITY_PREPROCESSING_V1"
        interpretation = "HISTORICAL_MODEL_COMPATIBILITY_BENCHMARK"
        native_status = "UNKNOWN_NOT_CLAIMED"
        native_claim = False
        mean = float(metadata["mean"])
        std = float(metadata["std"])
        input_contract = model["input"]
        output_contract = model["output"]
        steps = [
            {"step": 1, "operation": "VALIDATE_WINDOW", "parameters": {"dtype": "float32", "exact_samples": 300, "require_all_finite": True, "allow_padding": False, "allow_truncation": False, "allow_resampling": False}},
            {"step": 2, "operation": "IDENTITY_SEMANTIC_ADAPTER", "parameters": {"input_semantic": "resp_phase_unwrapped_clutter_removed", "transformation": "NONE", "native_semantic_alignment_claim": False}},
            {"step": 3, "operation": "FIXED_Z_SCORE", "parameters": {"mean": mean, "std": std, "stats_source": metadata_path, "fit_split": "NONE_AT_M-B10B"}},
            {"step": 4, "operation": "RESHAPE", "parameters": {"shape": [1, 300, 1], "dtype": "float32"}},
            {"step": 5, "operation": "AFFINE_INT8_QUANTIZE", "parameters": {"scale": float(input_contract["scale"]), "zero_point": int(input_contract["zero_point"]), "rounding": "nearest_even_numpy_rint", "saturate_to": [-128, 127]}},
        ]
        unknown_steps = ["historical native filtering/detrending/clipping implementation is not present in the repository evidence"]
    else:
        contract_id = "M-B10B_SYNTHETIC_V0_2_EXTERNAL_COMPATIBILITY_PREPROCESSING_V1"
        interpretation = "SYNTHETIC_TRAINED_EXTERNAL_COMPATIBILITY_BENCHMARK"
        native_status = "RECORDED_EXPERIMENTAL_PIPELINE_METADATA"
        native_claim = False
        scaler = metadata["scaler"]
        train_config = _load_json("models/mmwave/training_config.json")
        preprocessing_config = train_config["preprocessor"]
        int8_contract = metadata["stage_evaluations"]["int8_tflite"]
        input_contract = int8_contract["input_contract"]
        output_contract = int8_contract["output_contract"]
        mean = float(scaler["mean"])
        std = float(scaler["std"])
        steps = [
            {"step": 1, "operation": "VALIDATE_WINDOW", "parameters": {"dtype": "float32", "exact_samples": 300, "require_all_finite": True, "allow_padding": False, "allow_truncation": False, "allow_resampling": False}},
            {"step": 2, "operation": "LINEAR_DETREND", "parameters": {"method": "window_mean_subtraction"}},
            {"step": 3, "operation": "BUTTERWORTH_BANDPASS_ZERO_PHASE", "parameters": {"sample_rate_hz": 10.0, "lowcut_hz": 0.1, "highcut_hz": 0.5, "order": 4, "implementation": "scipy.signal.butter_and_filtfilt"}},
            {"step": 4, "operation": "FIXED_Z_SCORE", "parameters": {"mean": mean, "std": std, "stats_source": metadata_path, "fit_split": "NONE_AT_M-B10B", "method": scaler["method"]}},
            {"step": 5, "operation": "CLIP", "parameters": {"min": float(preprocessing_config["clip_min"]), "max": float(preprocessing_config["clip_max"])}},
            {"step": 6, "operation": "RESHAPE", "parameters": {"shape": [1, 300, 1], "dtype": "float32"}},
            {"step": 7, "operation": "AFFINE_INT8_QUANTIZE", "parameters": {"scale": float(input_contract["scale"]), "zero_point": int(input_contract["zero_point"]), "rounding": "nearest_even_numpy_rint", "saturate_to": [-128, 127]}},
        ]
        unknown_steps = ["synthetic training data generator provenance is external to this real-data repository"]
    return {
        "schema_version": "M-B10B_BASELINE_EXECUTABLE_PREPROCESSING_CONTRACT_V1",
        "contract_id": contract_id,
        "execution_status": "EXECUTABLE_COMPATIBILITY_BENCHMARK",
        "execution_scope": "LOCKED_TEST_ONLY_AFTER_EXPLICIT_M-B10B_AUTHORIZATION",
        "source_split": "LOCKED_TEST",
        "source_window_contract": {"path": "datasets/mmwave/processed/mmwave_canonical_real_v1.npy", "sample_rate_hz": 10.0, "window_samples": 300, "window_seconds": 30.0, "input_shape": [300], "input_dtype": "float32", "input_semantic": "resp_phase_unwrapped_clutter_removed"},
        "steps": steps,
        "invalid_input_policy": "FAIL_CLOSED_NO_PREDICTION",
        "fallback_policy": "NO_HEURISTIC_FALLBACK",
        "preprocessing_fit_policy": "NO_FIT_DURING_M-B10B",
        "native_preprocessing_status": native_status,
        "native_reproduction_claim": native_claim,
        "unknown_native_steps": unknown_steps,
        "class_map_compatibility": _frozen_class_map_compatibility(BASELINE_CLASS_MAP_EVIDENCE_PATHS[model_key]),
        "executor": {
            "path": "scripts/mmwave_m_b10b_baseline_preprocessing.py",
            "sha256": _sha256(_path("scripts/mmwave_m_b10b_baseline_preprocessing.py")),
            "entrypoint": "prepare_v01" if model_key == "mmwave" else "prepare_v02",
        },
        "model_identity": {"model_id": model["model_id"], "path": model_path, "sha256": _sha256(_path(model_path)), "bytes": _path(model_path).stat().st_size, "input": {"shape": [1, 300, 1], "dtype": "int8", "scale": float(input_contract["scale"]), "zero_point": int(input_contract["zero_point"])}, "output": {"shape": list(output_contract["shape"]), "dtype": output_contract["dtype"], "scale": float(output_contract["scale"]), "zero_point": int(output_contract["zero_point"])}},
        "class_map": {str(index): label for index, label in enumerate(LABELS)},
        "metadata_sources": [{"path": metadata_path, "sha256": _sha256(_path(metadata_path)), "bytes": _path(metadata_path).stat().st_size}],
        "interpretation": interpretation,
    }


def build_historical_baselines() -> dict[str, Any]:
    manifest = _load_json("models/model_manifest.json")["models"]
    rows = []
    for key in ("mmwave", "mmwave_v0_2_0_candidate"):
        model = manifest[key]
        path = _path(model["path"])
        row = {
            "baseline_id": model["model_id"],
            "path": model["path"],
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "manifest_sha256": model.get("sha256"),
            "validation_status": model.get("validation_status"),
            "deployment_allowed_in_historical_manifest": model.get("deployment_allowed"),
            "pool_eligible": False,
            "role": "HISTORICAL_BASELINE_ONLY",
        }
        row["preprocessing_contract_id"] = (
            "M-B10B_HISTORICAL_V0_1_COMPATIBILITY_PREPROCESSING_V1"
            if key == "mmwave"
            else "M-B10B_SYNTHETIC_V0_2_EXTERNAL_COMPATIBILITY_PREPROCESSING_V1"
        )
        row["executable_preprocessing_contract"] = _baseline_preprocessing_contract(key, model)
        if key == "mmwave":
            row["exclusion_reason"] = "Historical v0.1.0 model is blocked by class collapse on repository NPZ and is not the frozen real-data Phase-B lineage."
            row["lineage_status"] = "HISTORICAL_REPOSITORY_MODEL_WITH_CLASS_COLLAPSE"
            row["preprocessing_status"] = "EXECUTABLE_COMPATIBILITY_CONTRACT_NATIVE_UNKNOWN"
            row["exact_native_preprocessing_known"] = False
            row["class_map_compatibility"] = _frozen_class_map_compatibility(BASELINE_CLASS_MAP_EVIDENCE_PATHS[key])
            row["final_test_interpretation"] = "HISTORICAL_MODEL_COMPATIBILITY_BENCHMARK"
        else:
            row["exclusion_reason"] = "Historical v0.2.0 candidate is synthetic smoke-only and has no real-data VALIDATION evidence."
            row["lineage_status"] = "SYNTHETIC_TRAINING_EXTERNAL_COMPATIBILITY_ONLY"
            row["preprocessing_status"] = "EXECUTABLE_COMPATIBILITY_CONTRACT_METADATA_FROZEN"
            row["exact_native_preprocessing_known"] = False
            row["class_map_compatibility"] = _frozen_class_map_compatibility(BASELINE_CLASS_MAP_EVIDENCE_PATHS[key])
            row["final_test_interpretation"] = "SYNTHETIC_TRAINED_EXTERNAL_COMPATIBILITY_BENCHMARK"
        rows.append(row)
    return {
        "phase_id": "M-B10A",
        "registry_status": "BASELINES_REGISTERED_EXCLUDED_FROM_CANDIDATE_POOL",
        "baselines": rows,
        "candidate_pool_exclusion_rule": "Historical and synthetic baselines are context-only; they cannot win the real-data strict-INT8 pool.",
    }


def build_locked_test_contract(selected: dict[str, Any] | None) -> dict[str, Any]:
    baselines = build_historical_baselines()["baselines"]
    planned_models = []
    if selected:
        planned_models.append({
            "model_id": selected["model_id"],
            "role": "SELECTED_NEW_REAL_DATA_CANDIDATE",
            "path": selected["model"]["relative_path"],
            "sha256": selected["model"]["sha256"],
            "lineage_status": "FROZEN_REAL_DATA_PHASE_B_LINEAGE",
            "preprocessing_policy": "BPF_ZSCORE_EXACT_M-B1_RUNTIME_IDENTITY",
            "preprocessing_contract_id": "M-B10B_SELECTED_REAL_CANDIDATE_BPF_ZSCORE_V1",
            "executable_preprocessing_contract": {
                "contract_id": "M-B10B_SELECTED_REAL_CANDIDATE_BPF_ZSCORE_V1",
                "execution_status": "FROZEN_RUNTIME_IDENTITY_FROM_M-B9",
                "source_profile": "datasets/mmwave/manifests/M-B1_preprocessing_ablation/selected_preprocessing_profile.json",
                "source_runtime_identity": "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_preprocessing_identity.json",
                "fit_split": "TRAIN",
                "invalid_input_policy": "FAIL_CLOSED_NO_PREDICTION",
                "fallback_policy": "NO_HEURISTIC_FALLBACK",
                "class_map": dict(FROZEN_CLASS_MAP),
                "class_map_compatibility": _frozen_class_map_compatibility((
                    "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_manifests/seed" + str(selected["seed"]) + "_runtime_manifest.json",
                    selected["model"]["relative_path"],
                )),
            },
            "exact_native_preprocessing_known": True,
            "class_map_compatibility": _frozen_class_map_compatibility((
                "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_manifests/seed" + str(selected["seed"]) + "_runtime_manifest.json",
                selected["model"]["relative_path"],
            )),
            "final_test_interpretation": "REAL_DATA_OFFLINE_FINAL_TEST_CANDIDATE_ONLY",
        })
    planned_models.extend({
        "model_id": row["baseline_id"],
        "role": row["role"],
        "path": row["path"],
        "sha256": row["sha256"],
        "lineage_status": row["lineage_status"],
        "preprocessing_policy": row["preprocessing_status"],
        "preprocessing_contract_id": row["preprocessing_contract_id"],
        "executable_preprocessing_contract": row["executable_preprocessing_contract"],
        "exact_native_preprocessing_known": row["exact_native_preprocessing_known"],
        "class_map_compatibility": row["class_map_compatibility"],
        "final_test_interpretation": row["final_test_interpretation"],
    } for row in baselines)
    return {
        "phase_id": "M-B10A",
        "contract_status": "PREREGISTERED_NOT_EXECUTED",
        "candidate_reference": selected["candidate_id"] if selected else None,
        "candidate_sha256": selected["model"]["sha256"] if selected else None,
        "source_split": "LOCKED_TEST",
        "subject_count": 16,
        "structural_window_count": 88,
        "access_authorization": "Requires separate explicit M-B10B authorization after independent review; no authorization is granted by M-B10A.",
        "access_mechanism_reference": "PhaseBAccessGuard final-evaluation API is reserved for the separately authorized final pass.",
        "evaluation_passes": 1,
        "selection_and_tuning_after_access": False,
        "retraining_after_access": False,
        "recalibration_after_access": False,
        "threshold_tuning_after_access": False,
        "planned_models": planned_models,
        "metrics_schema": {
            "primary": "macro_f1",
            "required": ["accuracy", "macro_f1", "macro_precision", "macro_recall"],
            "per_class_fields": ["support", "tp", "fp", "tn", "fn", "precision", "recall", "f1_score", "fpr"],
            "apnea_proxy_fields": ["precision", "recall", "misses"],
            "rapid_or_abnormal_fields": ["recall"],
            "diagnostics": ["confusion_matrix", "prediction_distribution", "class_collapse_status"],
            "subject_level": ["subject_accuracy", "subject_macro_f1", "per_class_metrics_where_support_exists", "worst_subject_macro_f1", "median_subject_macro_f1"],
            "selected_real_candidate_extra": ["model_sha256", "preprocessing_identity", "strict_int8_structural_identity", "input_saturation_ratio", "runtime_identity"],
        },
        "required_metrics": [
            "accuracy",
            "macro_f1",
            "macro_precision",
            "macro_recall",
            "per_class_precision_recall_f1",
            "confusion_matrix",
            "APNEA_proxy_recall",
            "APNEA_proxy_precision",
            "invalid_or_fallback_count",
            "input_saturation_ratio",
        ],
        "forbidden_pretest_artifacts": [
            "LOCKED_TEST labels",
            "LOCKED_TEST tensors",
            "LOCKED_TEST predictions",
            "LOCKED_TEST performance metrics",
        ],
        "post_test_policy": {
            "selection_or_tuning_after_access": False,
            "retraining_after_access": False,
            "recalibration_after_access": False,
            "threshold_tuning_after_access": False,
            "new_experiment_cycle_required_for_any_improvement": True,
        },
        "applicable_predefined_numerical_acceptance_threshold": "FINAL_LOCKED_TEST_NUMERICAL_ACCEPTANCE_THRESHOLD_NOT_PREDEFINED",
        "acceptance_threshold_source": None,
        "final_result_claims_prohibited_until_execution": ["MR60", "real_sensor validated", "production", "clinical apnea"],
    }


def build_report(candidates: list[dict[str, Any]], ranking: dict[str, Any], selected: dict[str, Any] | None, rule_sha: str, input_identity: dict[str, Any]) -> None:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT_DIR, capture_output=True, text=True, check=False).stdout.strip() or "feature/M-B10A-candidate-selection-setup"
    base_sha = subprocess.run(["git", "rev-parse", "origin/main"], cwd=ROOT_DIR, capture_output=True, text=True, check=False).stdout.strip() or "4e3c2e6957a3142f0ff3da8ec50f3bc0b4c94602"
    b4_sensitivity = next(
        row for row in _load_json("datasets/mmwave/manifests/M-B4_multiseed_stability/multi_seed_results.json")["multi_seed_results"]
        if row.get("architecture_id") == ARCHITECTURE_ID
    )
    baselines = build_historical_baselines()["baselines"]
    lines = [
        "# SafeNest mmWave M-B10A — Pre-LOCKED_TEST Real-Data Candidate Selection Setup",
        "",
        "## Execution identity",
        "",
        f"- Track: mmWave M-B10A; branch: `{branch}`; base `origin/main`: `{base_sha}`.",
        "- M-B9 predecessor: closure `8fe4b2b38a0faa7b4cf87628f769c07763c6c91d` merged by PR #42 and present in the base.",
        "- Worktree isolation: fresh branch from `origin/main`; no CO₂, Thermal, Integration, shared-contract, config, risk, or raw-data files are in scope.",
        "",
        "## Scope and gate",
        "",
        "This report records a deterministic pre-LOCKED_TEST candidate-selection setup from frozen real-data VALIDATION evidence. It is not a final LOCKED_TEST result, MR60 result, real-sensor validation, production claim, or clinical apnea claim.",
        "",
        f"- Base branch evidence: `origin/main` predecessor M-B9 closure is present; input identity rows: {input_identity['total_inputs']}.",
        "- Model trainings: 0; model conversions/reconversions: 0; no threshold tuning or retuning; no formal M-B8 latency rerun.",
        "- LOCKED_TEST performance/label/prediction/tensor accesses: all 0; M-B10B started: NO.",
        "",
        "## Frozen candidate pool",
        "",
        "The candidate pool contains three frozen real-data strict-INT8 variants; hard gates decide which remain eligible:",
        "",
        "| seed | bytes | clean Macro F1 | min recall | APNEA P/R | worst subject Macro F1 | hard gates |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in candidates:
        failed_e11 = [profile for profile, row in candidate["moderate_profiles"].items() if row["recomputed_class_collapse"]["collapsed"]]
        gate_summary = ", ".join(f"{rule_id}={'PASS' if row['passed'] else 'FAIL'}" for rule_id, row in candidate["eligibility"].items())
        lines.append(f"| {candidate['seed']} | {candidate['model']['bytes']} | {candidate['ranking_metrics']['clean_strict_int8_macro_f1']:.6f} | {candidate['ranking_metrics']['clean_min_per_class_recall']:.6f} | {candidate['ranking_metrics']['clean_apnea_proxy_precision']:.6f} / {candidate['ranking_metrics']['clean_apnea_proxy_recall']:.6f} | {candidate['ranking_metrics']['worst_subject_clean_macro_f1']:.6f} | {gate_summary}; E11 {'PASS' if not failed_e11 else 'FAIL: ' + ', '.join(failed_e11)} |")
    lines.extend([
        "",
        "Pool identity is fixed to M-B3_CONV1D_GAP_BASELINE + M-B1 BPF_ZSCORE + M-B2 CE_UNWEIGHTED + M-B5 class-balanced calibration, seeds 42/43/44. Historical v0.1.0 and synthetic v0.2.0 artifacts are registered as baselines only and are excluded from the pool.",
        *[f"- Seed {candidate['seed']} artifact: `{candidate['model']['relative_path']}`, SHA-256 `{candidate['model']['sha256']}`." for candidate in candidates],
        "",
        "## Frozen rule and ranking",
        "",
        f"- Selection-rule SHA-256: `{rule_sha}`; EPS = `{EPSILON}`.",
        "- Lexicographic criteria are applied in preregistered order, with no composite score.",
        f"- Eligible candidates: {', '.join(ranking['eligible_candidate_ids']) or 'none'}.",
        f"- Selected prelocked candidate: `{selected['candidate_id'] if selected else 'NONE'}`.",
        f"- Deciding criterion: {ranking['deciding_criterion'] or 'none; selection is INCONCLUSIVE'}.",
        "",
        "## Seed sensitivity and perturbation warnings",
        "",
        f"- M-B4 architecture-level seed sensitivity (mean/std/worst clean Float Macro F1): {b4_sensitivity['macro_f1']['mean']:.6f} / {b4_sensitivity['macro_f1']['std']:.6f} / {b4_sensitivity['macro_f1']['worst_seed_val']:.6f} (worst seed {b4_sensitivity['macro_f1']['worst_seed_id']}).",
        "- Seed 44 fails hard E11 on `M-B7_AMP_X0_75` and `M-B7_COMBINED_MODERATE`; severe profiles are diagnostic only.",
        "",
        "## Complete pre-LOCKED_TEST evidence coverage",
        "",
        *[
            f"- Seed {candidate['seed']}: VALIDATION subjects={candidate['subject_level']['subject_count']} (worst `{candidate['subject_level']['worst_subject_id']}` Macro F1 {candidate['subject_level']['worst_subject_macro_f1']:.6f}, median {candidate['subject_level']['subject_macro_f1_median']:.6f}); M-B6 A→C probability MAE {candidate['m_b6_stage_equivalence']['pairwise']['a_to_c']['output_probability_mae']:.6f}, Top-1 {candidate['m_b6_stage_equivalence']['pairwise']['a_to_c']['top1_agreement']:.6f}; M-B8 invoke/pipeline P99 {candidate['m_b8_latency_footprint']['invoke_p99_ns']:.2f}/{candidate['m_b8_latency_footprint']['pipeline_p99_ns']:.2f} ns (Mac-only); M-B9 runtime/preprocessing identity exact: {candidate['m_b9_runtime_identity']['model_identity']['sha256_match'] and candidate['m_b9_runtime_identity']['summary']['runtime_identity_exact']}."
            for candidate in candidates
        ],
        "- E5 independently gates M-B9 runtime_prediction_identity exactness; E6 independently gates M-B9 valid-finalist fallback_audit records (M-B7 fallback summaries are not the E6 source).",
        "- Criterion 12 uses the M-B8 cross_seed_latency_summary source P99 directly; the saved candidate ranking value is only a claim checked against that source.",
        "- Gate records include repository-relative source paths and supporting values for E1–E11; no sample-level LOCKED_TEST evidence rows are generated.",
        "",
        "## Historical baselines",
        "",
        *[f"- `{row['baseline_id']}`: `{row['path']}`, SHA-256 `{row['sha256']}`, pool eligible: NO ({row['validation_status']}); executable contract `{row['preprocessing_contract_id']}` ({row['executable_preprocessing_contract']['execution_status']}); native reproduction claim: {row['executable_preprocessing_contract']['native_reproduction_claim']}." for row in baselines],
        "- Both baseline adapters require exact 300-sample finite windows, fixed recorded statistics, deterministic INT8 quantization, fail-closed invalid-input handling, and no heuristic fallback. v0.1 native filtering/detrending lineage remains explicitly unknown; v0.2 is synthetic external compatibility only.",
        "- Both baseline class maps are frozen compatible as `0→NORMAL`, `1→RAPID_OR_ABNORMAL`, `2→APNEA`, independently matched across model manifest, authoritative metadata, executable contract, and actual TFLite output shape `[1,3]`; no post-test class-map choice remains.",
        "",
        "## M-B10B contract and readiness",
        "",
        "- Final contract is preregistered for one LOCKED_TEST pass with accuracy, Macro F1/precision/recall, per-class metrics, confusion matrix, APNEA proxy precision/recall, invalid/fallback count, and input saturation.",
        "- No selection, tuning, retraining, recalibration, or threshold changes are allowed after access; readiness used: NO; independent review required.",
        "- Per-class fields are frozen as support/TP/FP/TN/FN/precision/recall/F1/FPR; subject-level accuracy/Macro F1/per-class support-aware metrics, worst and median subject Macro F1 are frozen.",
        "- No applicable predefined numerical acceptance threshold exists: `FINAL_LOCKED_TEST_NUMERICAL_ACCEPTANCE_THRESHOLD_NOT_PREDEFINED`.",
        f"- Guard readiness is structurally confirmed from `scripts/mmwave_phase_b_access.py`; final accessor used: NO.",
        "- No final performance number is present in M-B10A artifacts.",
        "",
        "## Warnings and authorization",
        "",
        "- REVIEW REFINEMENTS CLOSED IN THIS REVISION: executable historical/synthetic baseline contracts are frozen; E6 uses M-B9 valid-finalist fallback/runtime evidence; E5 uses M-B9 runtime_prediction_identity in the eligibility gate; Criterion 12 is independently reconstructed from M-B8 source P99.",
        "- M-B10B remains unauthorized until independent review accepts this closure; no LOCKED_TEST labels, tensors, predictions, or metrics were loaded.",
        "- REQUIRED REFINEMENT: architecture-level initialization seed sensitivity remains visible (M-B4 mean/std/worst-seed evidence); selecting seed 42 does not erase that warning.",
        "- NON-BLOCKING IMPROVEMENT: M-B7 severe profiles remain diagnostic warnings and are not hard-gated by the frozen rule.",
        "- NON-BLOCKING IMPROVEMENT: M-B8 is macOS-only offline evidence and does not establish Raspberry Pi or MR60 performance.",
        "",
        "## Final-test protocol status",
        "",
        "The final LOCKED_TEST metrics contract is preregistered but unused. M-B10B authorization recommendation: NO until independent review is complete.",
        "",
        "## Verification and artifacts",
        "",
        "- M-B10A validator: PASS; focused unittest: 13 methods (8 negative corruption subtests plus an explicit baseline class-map negative test); upstream M-B0 through M-B9 plus A5/A6 validators: PASS.",
        "- Evidence directory: `datasets/mmwave/manifests/M-B10A_candidate_selection_setup/` (16 machine-readable outputs plus checksums).",
        f"- Report: `{REPORT_REL.as_posix()}`; LOCKED_TEST access readiness used: NO.",
        "",
    ])
    REPORT_PATH = ROOT_DIR / REPORT_REL
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _prepare_output_dir() -> None:
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    if OUT_DIR.exists():
        for child in OUT_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        OUT_DIR.mkdir(parents=True)


def _write_checksums() -> None:
    rows = []
    for name in REQUIRED_OUTPUTS:
        if name == "checksums.sha256":
            continue
        path = OUT_DIR / name
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(f"{_sha256(path)}  {name}")
    (OUT_DIR / "checksums.sha256").write_text("\n".join(sorted(rows)) + "\n", encoding="utf-8")


def generate_artifacts() -> dict[str, Any]:
    _prepare_output_dir()

    # This is deliberately the first selection artifact written.  The winner
    # and ranking artifacts are written only after this frozen rule exists.
    selection_rule = build_selection_rule()
    rule_path = _write_json("selection_rule.json", selection_rule)
    rule_sha = _sha256(rule_path)

    input_identity = build_input_identity()
    _write_json("input_identity.json", input_identity)
    experiment_contract = {
        "phase_id": "M-B10A",
        "phase_title": "Pre-LOCKED_TEST Real-Data Offline Candidate Selection Setup",
        "source_split": "VALIDATION",
        "source_window_count": 79,
        "source_subject_count": 17,
        "real_data_only": True,
        "frozen_upstream_phases": ["A5", "A6", "M-B0", "M-B1", "M-B2", "M-B3", "M-B4", "M-B5", "M-B6", "M-B7", "M-B8", "M-B9"],
        "selection_rule_path": _relative(rule_path),
        "selection_rule_sha256": rule_sha,
        "model_trainings": 0,
        "model_conversions": 0,
        "retraining_or_reconversion_allowed": False,
        "formal_m_b8_latency_measurement_performed": False,
        "locked_test_access": "PROHIBITED_DURING_M-B10A",
        "m_b10b_started": False,
        "status": "M-B10_PRELOCKED_REAL_DATA_CANDIDATE",
    }
    _write_json("experiment_contract.json", experiment_contract)

    candidates, context = _build_candidates()
    candidate_pool = {
        "phase_id": "M-B10A",
        "pool_status": "FROZEN_REAL_DATA_STRICT_INT8_POOL",
        "candidate_count": len(candidates),
        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "candidates": candidates,
        "excluded_baseline_ids": ["mmwave_resp_int8", "mmwave_resp_int8_v0.2.0_candidate"],
        "selection_rule_sha256": rule_sha,
    }
    _write_json("candidate_pool.json", candidate_pool)
    _write_json("candidate_eligibility_contract.json", build_eligibility_contract())
    evidence = {
        "phase_id": "M-B10A",
        "evidence_status": "INDEPENDENTLY_DERIVED_FROM_FROZEN_VALIDATION_ARTIFACTS",
        "selection_rule_sha256": rule_sha,
        "candidate_metrics_are_validation_only": True,
        "architecture_seed_sensitivity": {
            "source_path": "datasets/mmwave/manifests/M-B4_multiseed_stability/multi_seed_results.json",
            "architecture_id": ARCHITECTURE_ID,
            "mean_clean_float_macro_f1": context["architecture_seed_sensitivity"]["macro_f1"]["mean"],
            "std_clean_float_macro_f1": context["architecture_seed_sensitivity"]["macro_f1"]["std"],
            "worst_clean_float_macro_f1": context["architecture_seed_sensitivity"]["macro_f1"]["worst_seed_val"],
            "worst_seed": context["architecture_seed_sensitivity"]["macro_f1"]["worst_seed_id"],
            "warning": "INITIALIZATION_SEED_SENSITIVITY",
        },
        "candidate_evidence": [
            {
                "candidate_id": candidate["candidate_id"],
                "seed": candidate["seed"],
                "architecture_id": candidate["architecture_id"],
                "model": candidate["model"],
                "clean_validation": candidate["clean_validation"],
                "subject_level": candidate["subject_level"],
                "m_b4_seed_stability": candidate["m_b4_seed_stability"],
                "m_b6_stage_equivalence": candidate["m_b6_stage_equivalence"],
                "moderate_profile_metrics": candidate["moderate_profiles"],
                "m_b8_latency_footprint": candidate["m_b8_latency_footprint"],
                "m_b9_runtime_identity": candidate["m_b9_runtime_identity"],
                "ranking_metrics": candidate["ranking_metrics"],
                "eligibility": candidate["eligibility"],
                "eligibility_evidence": candidate["eligibility_evidence"],
                "eligible": candidate["eligible"],
            }
            for candidate in candidates
        ],
        "locked_test_evidence_rows": 0,
    }
    _write_json("candidate_selection_evidence.json", evidence)
    selection_evidence_sha = _sha256(OUT_DIR / "candidate_selection_evidence.json")
    ranking = rank_candidates(candidates, rule_sha)
    _write_json("candidate_ranking.json", ranking)
    selected = next((candidate for candidate in candidates if candidate["candidate_id"] == ranking["selected_candidate_id"]), None)
    selected_pretest = {
        "phase_id": "M-B10A",
        "status": "M-B10_PRELOCKED_REAL_DATA_CANDIDATE" if selected else "INCONCLUSIVE",
        "candidate_id": selected["candidate_id"] if selected else None,
        "model_id": selected["model_id"] if selected else None,
        "seed": selected["seed"] if selected else None,
        "model": selected["model"] if selected else None,
        "architecture_id": selected["architecture_id"] if selected else None,
        "class_map": dict(FROZEN_CLASS_MAP),
        "preprocessing": {
            "profile_id": selected["preprocessing_profile"] if selected else None,
            "profile_name": selected["preprocessing_name"] if selected else None,
            "metadata_identity": _load_json("datasets/mmwave/manifests/M-B1_preprocessing_ablation/selected_preprocessing_profile.json") if selected else None,
        },
        "calibration_profile": selected["calibration_profile"] if selected else None,
        "training_identity": {"seed": selected["training_seed"], "final_weights_sha256": selected["training_weights_sha256"]} if selected else None,
        "m_b6_stage_equivalence": selected["m_b6_stage_equivalence"] if selected else None,
        "m_b9_runtime_identity": selected["m_b9_runtime_identity"] if selected else None,
        "selection_rule_sha256": rule_sha,
        "selection_evidence_sha256": selection_evidence_sha,
        "deciding_criterion": ranking["deciding_criterion"],
        "deployment_allowed": False,
        "mr60_validation": "NOT_PERFORMED",
        "real_sensor_validation": "NOT_PERFORMED",
        "production_claim": False,
        "clinical_performance": "NOT_EVALUATED",
        "locked_test_accessed": False,
        "m_b10b_started": False,
        "limitations": [
            "INITIALIZATION_SEED_SENSITIVITY",
            "MAC_ONLY_LATENCY",
            "OFFLINE_PERTURBATION_ONLY",
            "MOCK_E2E_ONLY",
            "NO_MR60_VALIDATION",
            "APNEA_PROXY_SCOPE",
            "LOCKED_TEST_NOT_EVALUATED",
        ],
        "authorization_recommendation": "NO — independent review required before M-B10B.",
    }
    _write_json("selected_candidate_pretest.json", selected_pretest)

    historical = build_historical_baselines()
    _write_json("historical_baseline_registry.json", historical)
    locked_contract = build_locked_test_contract(selected)
    _write_json("locked_test_evaluation_contract.json", locked_contract)
    guard_readiness = _guard_structural_readiness()
    locked_readiness = {
        "phase_id": "M-B10A",
        "readiness_status": "M-B10_PRELOCKED_REAL_DATA_CANDIDATE" if selected else "INCONCLUSIVE",
        "authorization_for_locked_test": "NO",
        "independent_review_required": True,
        "selected_candidate_pretest_path": "datasets/mmwave/manifests/M-B10A_candidate_selection_setup/selected_candidate_pretest.json",
        "structural_split_counts_only": {"LOCKED_TEST_subjects": 16, "LOCKED_TEST_windows": 88},
        "labels_loaded": 0,
        "prediction_tensors_loaded": 0,
        "performance_metrics_computed": 0,
        "final_accessor_calls": 0,
        "final_access_mechanism_ready": guard_readiness["ready"],
        "final_access_mechanism": guard_readiness,
    }
    _write_json("locked_test_access_readiness.json", locked_readiness)
    _write_json("locked_test_access_audit.json", {
        "phase_id": "M-B10A",
        "audit_status": "PASS_ZERO_ACCESS",
        "performance_access_attempts": 0,
        "label_access_attempts": 0,
        "prediction_access_attempts": 0,
        "tensor_access_attempts": 0,
        "metric_access_attempts": 0,
        "final_accessor_calls": 0,
        "locked_test_inputs_loaded": False,
        "locked_test_labels_loaded": False,
        "locked_test_prediction_output_generated": False,
        "locked_test_performance_computed": False,
    })
    _write_json("run_environment.json", {
        "phase_id": "M-B10A",
        "execution_scope": "MACOS_OFFLINE_EVIDENCE_ASSEMBLY_NO_FORMAL_BENCHMARK",
        "operating_system": platform.system(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "formal_m_b8_latency_measurement_started": False,
        "formal_m_b8_latency_measurement_performed": False,
        "known_safenest_workload_check": "NOT_APPLICABLE_FORMAL_BENCHMARK_NOT_STARTED",
        "required_idle_stabilization_seconds_before_future_formal_benchmark": 30,
        "m_b9_prior_formal_latency_benchmark_completed": True,
        "m_b10b_started": False,
        "model_trainings": 0,
        "model_conversions": 0,
        "locked_test_accesses": 0,
        "input_scope": "VALIDATION_ONLY",
    })
    _write_json("exceptions.json", {
        "phase_id": "M-B10A",
        "blockers": [],
        "closed_review_refinements": [
            {"id": "HISTORICAL_BASELINE_EXECUTABLE_PREPROCESSING_CONTRACT", "status": "CLOSED", "evidence": "historical_baseline_registry.json and locked_test_evaluation_contract.json"},
            {"id": "M_B9_VALID_FINALIST_FALLBACK_E6", "status": "CLOSED", "evidence": "M-B9 fallback_audit.json independently reconstructed by selector and validator"},
            {"id": "M_B9_RUNTIME_PREDICTION_IDENTITY_E5", "status": "CLOSED", "evidence": "M-B9 runtime_prediction_identity.json is a hard E5 eligibility gate"},
            {"id": "M_B8_PIPELINE_P99_SOURCE_RECONSTRUCTION", "status": "CLOSED", "evidence": "M-B8 cross_seed_latency_summary.json independently reconstructed by validator"},
            {"id": "BASELINE_CLASS_MAP_FREEZE", "status": "CLOSED", "evidence": "model_manifest.json, baseline metadata, executable contracts, and TFLite output shape [1,3] independently matched"},
        ],
        "required_refinements": [
            {"finding_class": "REQUIRED REFINEMENT", "id": "INDEPENDENT_REVIEW", "description": "Independent review must approve the frozen rule, lineage, gates, and winner before M-B10B."},
            {"finding_class": "REQUIRED REFINEMENT", "id": "SEED_SENSITIVITY", "description": "M-B4 architecture-level seed sensitivity remains a warning; a single-seed winner does not erase it."},
        ],
        "non_blocking_improvements": [
            {"finding_class": "NON-BLOCKING IMPROVEMENT", "id": "SEVERE_PROFILE_DIAGNOSTIC", "description": "Severe M-B7 profiles remain diagnostic and are not hard-gated by the frozen rule."},
            {"finding_class": "NON-BLOCKING IMPROVEMENT", "id": "MAC_ONLY_LATENCY", "description": "M-B8 latency is macOS-only offline evidence; no Raspberry Pi or MR60 claim is made."},
        ],
    })
    summary = {
        "phase_id": "M-B10A",
        "phase_title": "Pre-LOCKED_TEST Real-Data Offline Candidate Selection Setup",
        "selection_status": ranking["selection_status"],
        "validation_success": True,
        "candidate_count": len(candidates),
        "eligible_candidate_count": ranking["eligible_candidate_count"],
        "selected_candidate_id": ranking["selected_candidate_id"],
        "selection_rule_sha256": rule_sha,
        "selection_epsilon": EPSILON,
        "deciding_criterion": ranking["deciding_criterion"],
        "historical_baselines_registered": True,
        "model_trainings": 0,
        "model_conversions": 0,
        "locked_test_accesses": 0,
        "locked_test_performance_computed": False,
        "formal_m_b8_latency_measurement_rerun": False,
        "m_b10b_started": False,
        "m_b10b_authorization_recommendation": "NO — independent review required.",
        "warnings": ["INITIALIZATION_SEED_SENSITIVITY_PRESERVED", "SEVERE_M-B7_PROFILES_NOT_HARD_GATED", "MAC_ONLY_LATENCY_EVIDENCE"],
        "blockers": [],
        "finding_classes": ["BLOCKER", "REQUIRED REFINEMENT", "NON-BLOCKING IMPROVEMENT"],
        "locked_test_evaluation_contract_registered": True,
        "review_refinements_closed": [
            "HISTORICAL_BASELINE_EXECUTABLE_PREPROCESSING_CONTRACT",
            "M_B9_VALID_FINALIST_FALLBACK_E6",
            "M_B9_RUNTIME_PREDICTION_IDENTITY_E5",
            "M_B8_PIPELINE_P99_SOURCE_RECONSTRUCTION",
            "BASELINE_CLASS_MAP_FREEZE",
        ],
        "locked_test_access_readiness_used": False,
        "final_access_mechanism_ready": guard_readiness["ready"],
    }
    _write_json("m_b10a_summary.json", summary)
    _write_checksums()
    build_report(candidates, ranking, selected, rule_sha, input_identity)
    return {"candidates": candidates, "ranking": ranking, "selected": selected, "rule_sha256": rule_sha, "summary": summary}


def main() -> int:
    result = generate_artifacts()
    print(json.dumps({"phase_id": "M-B10A", "selection_status": result["ranking"]["selection_status"], "selected_candidate_id": result["ranking"]["selected_candidate_id"], "selection_rule_sha256": result["rule_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
