#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/check_mmwave_candidate.py
SafeNest V6 Precise mmWave Candidate Defect Detector and Quality-Check CLI

Inspects a candidate mmWave TFLite model and candidate metadata JSON for technical defects:
- Artifact existence and SHA-256 integrity
- Strict metadata schema validation
- Class collapse derivation & flag consistency
- APNEA / RAPID zero-recall and per-class recall thresholds
- Float-to-INT8 macro F1 drop and overall performance limits
- Input saturation ratio and quantization error
- Scaler statistics and class map integrity
- Manifest-metadata consistency
- Sample count & prediction distribution consistency
"""

from __future__ import annotations
import os
import sys
import json
import math
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union

# Ensure canonical repository root is in python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import yaml
except ImportError:
    yaml = None

from scripts.evaluate_mmwave import calculate_sha256
from scripts.validate_metadata import validate_mmwave_candidate_metadata


class DefectItem(dict):
    """Structured defect failure item supporting dict access, attribute access, and string formatting."""

    def __init__(
        self,
        code: str,
        message: str,
        observed: Any = None,
        threshold: Any = None,
        remediation: str = "",
        source: str = "",
    ):
        super().__init__(
            code=code,
            message=message,
            observed=observed,
            threshold=threshold,
            remediation=remediation,
            source=source,
        )
        self.code = code
        self.message = message
        self.observed = observed
        self.threshold = threshold
        self.remediation = remediation
        self.source = source

    def __str__(self) -> str:
        obs_str = f" (Observed: {self.observed}, Threshold: {self.threshold})" if self.observed is not None else ""
        rem_str = f" [Action: {self.remediation}]" if self.remediation else ""
        return f"[{self.code}] {self.message}{obs_str}{rem_str}"


def load_acceptance_thresholds(
    contract_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Loads offline acceptance thresholds from contract YAML or returns built-in fallback.
    Returns: (thresholds_dict, threshold_source_name)
    """
    default_thresholds = {
        "accuracy_min": 0.40,
        "macro_f1_min": 0.60,
        "per_class_recall_min": {
            "NORMAL": 0.40,
            "RAPID_OR_ABNORMAL": 0.50,
            "APNEA": 0.50,
        },
        "max_int8_f1_drop": 0.05,
        "max_input_saturation_ratio": 0.05,
        "max_quantization_mae": 0.10,
        "prohibited_states": [
            "CLASS_COLLAPSE_ALL_SAME_PRED",
            "ZERO_APNEA_RECALL",
            "SATURATION_RATIO_EXCEEDED",
            "UNMATCHED_SHA256_HASH",
            "MISSING_SCALER_METADATA",
        ],
        "scope": "PASSED_ON_SYNTHETIC",
    }

    if contract_path and contract_path.exists():
        if yaml is None:
            return default_thresholds, "CONTRACT_YAML_PARSER_MISSING"
        try:
            with open(contract_path, "r", encoding="utf-8") as f:
                contract = yaml.safe_load(f)
            thresholds = contract.get("offline_acceptance_thresholds", {})
            if isinstance(thresholds, dict) and "accuracy_min" in thresholds:
                return thresholds, "INPUT_CONTRACT"
            else:
                return default_thresholds, "CONTRACT_MISSING_ACCEPTANCE_BLOCK"
        except Exception:
            return default_thresholds, "CONTRACT_PARSE_ERROR"

    return default_thresholds, "BUILT_IN_FALLBACK"


def check_candidate_quality(
    candidate_path: Path,
    metadata_path: Path,
    contract_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    model_root: Optional[Path] = None,
) -> Tuple[bool, List[DefectItem], Dict[str, Any]]:
    """
    Performs comprehensive technical defect detection on candidate model and metadata.
    Returns: (passed: bool, defects: List[DefectItem], report_dict: Dict[str, Any])
    """
    defects: List[DefectItem] = []
    if model_root is None:
        if metadata_path.parent.name == "mmwave" and metadata_path.parent.parent.name == "models":
            root = metadata_path.parent.parent.parent
        else:
            root = project_root
    else:
        root = model_root
    contract_file = contract_path or (root / "config/mmwave_input_contract.yaml")
    manifest_file = manifest_path or (root / "models/model_manifest.json")

    thresholds, threshold_source = load_acceptance_thresholds(contract_file)

    report: Dict[str, Any] = {
        "check_name": "mmwave_candidate_quality_check",
        "scope": "SYNTHETIC_OFFLINE_CANDIDATE_QA",
        "status": "FAILED",
        "candidate_path": (
            candidate_path.resolve().relative_to(root.resolve()).as_posix()
            if candidate_path.resolve().is_relative_to(root.resolve())
            else f"EXTERNAL_INPUT/{candidate_path.name}"
        ),
        "metadata_path": (
            metadata_path.resolve().relative_to(root.resolve()).as_posix()
            if metadata_path.resolve().is_relative_to(root.resolve())
            else f"EXTERNAL_INPUT/{metadata_path.name}"
        ),
        "threshold_source": threshold_source,
        "thresholds_used": thresholds,
        "checks": {},
        "failures": [],
        "limitations": {
            "real_sensor_performance": "NOT_VERIFIABLE",
            "hardware_validation": "BLOCKED_HARDWARE",
            "real_subject_generalization": "NOT_VERIFIABLE",
        },
    }

    # 6.1 Required Artifact Existence
    if not candidate_path.exists():
        defects.append(
            DefectItem(
                code="CANDIDATE_FILE_MISSING",
                message=f"Candidate TFLite file missing: {candidate_path}",
                remediation="Ensure train_mmwave.py completed model export.",
                source="candidate_path",
            )
        )
    if not metadata_path.exists():
        defects.append(
            DefectItem(
                code="METADATA_FILE_MISSING",
                message=f"Candidate metadata JSON missing: {metadata_path}",
                remediation="Ensure candidate metadata was generated.",
                source="metadata_path",
            )
        )

    if defects:
        report["failures"] = [dict(d) for d in defects]
        return False, defects, report

    # 6.2 JSON Parsing
    meta: Dict[str, Any] = {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            raise ValueError("Top-level JSON is not an object.")
        report["checks"]["json_parsing"] = "PASSED"
    except Exception as parse_err:
        defects.append(
            DefectItem(
                code="METADATA_JSON_PARSE_ERROR",
                message=f"Failed to parse candidate metadata JSON: {parse_err}",
                remediation="Check metadata file encoding and formatting.",
                source="metadata_path",
            )
        )
        report["failures"] = [dict(d) for d in defects]
        return False, defects, report

    # 6.3 Strict Metadata Schema Validation
    schema_passed = False
    try:
        validate_mmwave_candidate_metadata(meta, model_root=root)
        report["checks"]["metadata_schema"] = "PASSED"
        schema_passed = True
    except Exception as schema_err:
        defects.append(
            DefectItem(
                code="METADATA_SCHEMA_INVALID",
                message=f"Metadata schema validation failed: {schema_err}",
                remediation="Regenerate candidate metadata adhering to schema 1.0.",
                source="validate_mmwave_candidate_metadata",
            )
        )
        report["checks"]["metadata_schema"] = "FAILED"

    # 6.4 Candidate SHA-256 Integrity
    actual_sha = calculate_sha256(candidate_path).lower()
    recorded_sha = str(meta.get("sha256", "")).lower()
    report["actual_sha256"] = actual_sha
    report["recorded_sha256"] = recorded_sha

    if actual_sha != recorded_sha:
        defects.append(
            DefectItem(
                code="MODEL_METADATA_SHA_MISMATCH",
                message=f"Candidate file SHA-256 ({actual_sha}) != Metadata SHA-256 ({recorded_sha})",
                observed=actual_sha,
                threshold=recorded_sha,
                remediation="Re-save metadata with actual model file SHA-256 hash.",
                source="sha256",
            )
        )
        report["checks"]["sha256_integrity"] = "FAILED"
    else:
        report["checks"]["sha256_integrity"] = "PASSED"

    # Manifest SHA check if manifest exists
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            candidate_entry = manifest.get("models", {}).get("mmwave_v0_2_0_candidate", {})
            if candidate_entry:
                manifest_sha = str(candidate_entry.get("evaluation", {}).get("sha256", "")).lower()
                if manifest_sha and manifest_sha != actual_sha:
                    defects.append(
                        DefectItem(
                            code="MODEL_MANIFEST_SHA_MISMATCH",
                            message=f"Candidate file SHA ({actual_sha}) != Manifest SHA ({manifest_sha})",
                            observed=actual_sha,
                            threshold=manifest_sha,
                            remediation="Sync models/model_manifest.json with candidate SHA.",
                            source="manifest.evaluation.sha256",
                        )
                    )
                    report["checks"]["manifest_sha_integrity"] = "FAILED"
                else:
                    report["checks"]["manifest_sha_integrity"] = "PASSED"
        except Exception as man_err:
            pass

    # Extract Stage Evaluations
    stage_evals = meta.get("stage_evaluations", {})
    int8_eval = stage_evals.get("int8_tflite", {}) if isinstance(stage_evals, dict) else {}
    float_eval = (
        stage_evals.get("float_tflite", {})
        or stage_evals.get("float_keras", {})
        if isinstance(stage_evals, dict)
        else {}
    )

    # 6.14 Class Map Integrity
    class_map = meta.get("class_map", {})
    expected_class_map = {"0": "NORMAL", "1": "RAPID_OR_ABNORMAL", "2": "APNEA"}
    normalized_class_map = {str(k): str(v) for k, v in class_map.items()} if isinstance(class_map, dict) else {}

    if normalized_class_map != expected_class_map:
        defects.append(
            DefectItem(
                code="CLASS_MAP_MISMATCH",
                message=f"Metadata class map ({normalized_class_map}) != Expected ({expected_class_map})",
                observed=normalized_class_map,
                threshold=expected_class_map,
                remediation="Ensure class map is 0:NORMAL, 1:RAPID_OR_ABNORMAL, 2:APNEA.",
                source="class_map",
            )
        )
        report["checks"]["class_map_integrity"] = "FAILED"
    else:
        report["checks"]["class_map_integrity"] = "PASSED"

    # 6.13 Scaler Integrity
    scaler = meta.get("scaler", {}) if isinstance(meta.get("scaler"), dict) else {}
    scaler_mean = scaler.get("mean")
    scaler_std = scaler.get("std")
    scaler_method = scaler.get("method")
    scaler_source = scaler.get("stats_source")

    if scaler_mean is None or scaler_std is None:
        defects.append(
            DefectItem(
                code="MISSING_SCALER_METADATA",
                message="Scaler mean or std is missing from metadata.",
                remediation="Include preprocessor mean and std in metadata.",
                source="scaler",
            )
        )
        report["checks"]["scaler_integrity"] = "FAILED"
    elif not isinstance(scaler_mean, (int, float)) or math.isnan(scaler_mean) or math.isinf(scaler_mean):
        defects.append(
            DefectItem(
                code="INVALID_SCALER_MEAN",
                message=f"Scaler mean is invalid or non-finite: {scaler_mean}",
                observed=scaler_mean,
                remediation="Recalculate mean from train split.",
                source="scaler.mean",
            )
        )
        report["checks"]["scaler_integrity"] = "FAILED"
    elif not isinstance(scaler_std, (int, float)) or math.isnan(scaler_std) or math.isinf(scaler_std) or scaler_std <= 0:
        defects.append(
            DefectItem(
                code="INVALID_SCALER_STD",
                message=f"Scaler std must be finite positive number, got: {scaler_std}",
                observed=scaler_std,
                remediation="Recalculate std from train split.",
                source="scaler.std",
            )
        )
        report["checks"]["scaler_integrity"] = "FAILED"
    elif scaler_source != "train_split_only":
        defects.append(
            DefectItem(
                code="SCALER_SOURCE_MISMATCH",
                message=f"Scaler stats_source ({scaler_source}) != train_split_only",
                observed=scaler_source,
                threshold="train_split_only",
                remediation="Compute scaler statistics exclusively from X_train.",
                source="scaler.stats_source",
            )
        )
        report["checks"]["scaler_integrity"] = "FAILED"
    else:
        report["checks"]["scaler_integrity"] = "PASSED"

    # Evaluation Evidence Checks
    pred_dist = int8_eval.get("prediction_distribution", {}) if isinstance(int8_eval, dict) else {}
    eval_count = (
        int8_eval.get("total_samples")
        or int8_eval.get("evaluated_sample_count")
        or (sum(v for v in pred_dist.values() if isinstance(v, (int, float))) if isinstance(pred_dist, dict) and pred_dist else 0)
    )

    # 6.15 Prediction Distribution Count Consistency
    if isinstance(pred_dist, dict) and pred_dist:
        total_pred_samples = sum(v for v in pred_dist.values() if isinstance(v, (int, float)))
        if eval_count > 0 and total_pred_samples != eval_count:
            defects.append(
                DefectItem(
                    code="PREDICTION_DISTRIBUTION_COUNT_MISMATCH",
                    message=f"Sum of prediction distribution ({total_pred_samples}) != evaluated_sample_count ({eval_count})",
                    observed=total_pred_samples,
                    threshold=eval_count,
                    remediation="Re-evaluate predictions over identical test set.",
                    source="int8_tflite.prediction_distribution",
                )
            )
            report["checks"]["prediction_count_consistency"] = "FAILED"
        else:
            report["checks"]["prediction_count_consistency"] = "PASSED"
    else:
        defects.append(
            DefectItem(
                code="EVALUATION_EVIDENCE_MISSING",
                message="INT8 evaluation prediction_distribution is missing or empty.",
                remediation="Run INT8 TFLite evaluation on test set.",
                source="int8_tflite.prediction_distribution",
            )
        )
        report["checks"]["prediction_count_consistency"] = "FAILED"

    # 6.5 Class Collapse Derivation & Cross-check
    non_zero_predicted = [k for k, v in pred_dist.items() if isinstance(v, (int, float)) and v > 0]
    derived_collapse = len(non_zero_predicted) == 1 if eval_count > 0 else False
    recorded_collapse = int8_eval.get("class_collapse") if isinstance(int8_eval, dict) else None

    if derived_collapse:
        defects.append(
            DefectItem(
                code="CLASS_COLLAPSE_ALL_SAME_PRED",
                message=f"Class collapse detected! Model predicts only 1 class: {non_zero_predicted}",
                observed=non_zero_predicted,
                remediation="Inspect class balancing, learning rate, and loss convergence.",
                source="int8_tflite.prediction_distribution",
            )
        )
        report["checks"]["class_collapse"] = "FAILED"

    if recorded_collapse is not None and derived_collapse != recorded_collapse:
        defects.append(
            DefectItem(
                code="CLASS_COLLAPSE_FLAG_INCONSISTENT",
                message=f"Derived class collapse ({derived_collapse}) != recorded flag ({recorded_collapse})",
                observed=recorded_collapse,
                threshold=derived_collapse,
                remediation="Recompute class_collapse flag from prediction distribution.",
                source="int8_tflite.class_collapse",
            )
        )
        report["checks"]["class_collapse"] = "FAILED"
    else:
        report["checks"]["class_collapse"] = "PASSED"

    # 6.6 & 6.7 Zero Recall Checks for Minority Classes
    apnea_miss_rate = int8_eval.get("apnea_window_miss_rate") if isinstance(int8_eval, dict) else None
    apnea_recall: Optional[float] = None
    if isinstance(apnea_miss_rate, (int, float)):
        apnea_recall = 1.0 - float(apnea_miss_rate)

    apnea_pred_cnt = pred_dist.get("APNEA", 0) if isinstance(pred_dist, dict) else 0
    rapid_pred_cnt = pred_dist.get("RAPID_OR_ABNORMAL", 0) if isinstance(pred_dist, dict) else 0
    normal_pred_cnt = pred_dist.get("NORMAL", 0) if isinstance(pred_dist, dict) else 0

    # Recall Checks
    if apnea_recall is not None and apnea_recall == 0.0:
        defects.append(
            DefectItem(
                code="ZERO_APNEA_RECALL",
                message="APNEA class has 0.0 recall (100% miss rate).",
                observed=0.0,
                threshold=thresholds.get("per_class_recall_min", {}).get("APNEA", 0.50),
                remediation="Adjust loss weights or augment APNEA respiration windows.",
                source="int8_tflite.apnea_window_miss_rate",
            )
        )
        report["checks"]["zero_apnea_recall"] = "FAILED"
    elif apnea_pred_cnt == 0:
        defects.append(
            DefectItem(
                code="ZERO_APNEA_RECALL",
                message="APNEA class has 0 predicted windows (zero recall).",
                observed=0,
                remediation="Re-train model to detect APNEA event signatures.",
                source="int8_tflite.prediction_distribution.APNEA",
            )
        )
        report["checks"]["zero_apnea_recall"] = "FAILED"
    else:
        report["checks"]["zero_apnea_recall"] = "PASSED"

    if rapid_pred_cnt == 0:
        defects.append(
            DefectItem(
                code="ZERO_RAPID_RECALL",
                message="RAPID_OR_ABNORMAL class has 0 predicted windows.",
                observed=0,
                remediation="Ensure RAPID_OR_ABNORMAL samples are present in training batch.",
                source="int8_tflite.prediction_distribution.RAPID_OR_ABNORMAL",
            )
        )
        report["checks"]["zero_rapid_recall"] = "FAILED"
    else:
        report["checks"]["zero_rapid_recall"] = "PASSED"

    # 6.8 Per-Class Recall Minimum Thresholds
    per_class_min = thresholds.get("per_class_recall_min", {})
    per_class_rec = int8_eval.get("per_class_recall", {}) if isinstance(int8_eval, dict) else {}

    # APNEA recall
    min_apnea = per_class_min.get("APNEA", 0.50)
    cur_apnea_rec = per_class_rec.get("APNEA", apnea_recall)
    if cur_apnea_rec is not None:
        if cur_apnea_rec < min_apnea:
            defects.append(
                DefectItem(
                    code="APNEA_RECALL_BELOW_THRESHOLD",
                    message=f"APNEA recall ({cur_apnea_rec:.4f}) < minimum threshold ({min_apnea})",
                    observed=cur_apnea_rec,
                    threshold=min_apnea,
                    remediation="Improve APNEA feature extraction and recall sensitivity.",
                    source="int8_tflite.apnea_window_miss_rate",
                )
            )
            report["checks"]["apnea_recall_threshold"] = "FAILED"
        else:
            report["checks"]["apnea_recall_threshold"] = "PASSED"
    else:
        report["checks"]["apnea_recall_threshold"] = "NOT_AVAILABLE"

    # RAPID_OR_ABNORMAL recall
    min_rapid = per_class_min.get("RAPID_OR_ABNORMAL", 0.50)
    cur_rapid_rec = per_class_rec.get("RAPID_OR_ABNORMAL", 1.0 if rapid_pred_cnt > 0 else 0.0)
    if cur_rapid_rec < min_rapid:
        defects.append(
            DefectItem(
                code="RAPID_RECALL_BELOW_THRESHOLD",
                message=f"RAPID_OR_ABNORMAL recall ({cur_rapid_rec:.4f}) < minimum threshold ({min_rapid})",
                observed=cur_rapid_rec,
                threshold=min_rapid,
                remediation="Improve RAPID_OR_ABNORMAL training balance.",
                source="int8_tflite.prediction_distribution.RAPID_OR_ABNORMAL",
            )
        )
        report["checks"]["rapid_recall_threshold"] = "FAILED"
    else:
        report["checks"]["rapid_recall_threshold"] = "PASSED"

    # NORMAL recall
    min_normal = per_class_min.get("NORMAL", 0.40)
    cur_normal_rec = per_class_rec.get("NORMAL", 1.0 if normal_pred_cnt > 0 else 0.0)
    if cur_normal_rec < min_normal:
        defects.append(
            DefectItem(
                code="NORMAL_RECALL_BELOW_THRESHOLD",
                message=f"NORMAL recall ({cur_normal_rec:.4f}) < minimum threshold ({min_normal})",
                observed=cur_normal_rec,
                threshold=min_normal,
                remediation="Ensure NORMAL respiration signals are recognized.",
                source="int8_tflite.prediction_distribution.NORMAL",
            )
        )
        report["checks"]["normal_recall_threshold"] = "FAILED"
    else:
        report["checks"]["normal_recall_threshold"] = "PASSED"

    report["per_class_recall"] = {
        "NORMAL": cur_normal_rec,
        "RAPID_OR_ABNORMAL": cur_rapid_rec,
        "APNEA": cur_apnea_rec,
    }

    # 6.9 Float-to-INT8 Macro F1 Drop
    float_f1 = float_eval.get("macro_f1") if isinstance(float_eval, dict) else None
    int8_f1 = int8_eval.get("macro_f1") if isinstance(int8_eval, dict) else None
    max_f1_drop = thresholds.get("max_int8_f1_drop", 0.05)

    if float_f1 is None or int8_f1 is None:
        defects.append(
            DefectItem(
                code="MACRO_F1_METRIC_MISSING",
                message="Float or INT8 macro F1 metric is missing in evaluation evidence.",
                remediation="Record macro_f1 for float and int8 evaluation stages.",
                source="stage_evaluations",
            )
        )
        report["checks"]["macro_f1_drop"] = "FAILED"
    else:
        f1_drop = float(float_f1) - float(int8_f1)
        report["float_macro_f1"] = float_f1
        report["int8_macro_f1"] = int8_f1
        report["f1_drop"] = f1_drop

        if f1_drop > max_f1_drop:
            defects.append(
                DefectItem(
                    code="INT8_MACRO_F1_DROP_EXCEEDED",
                    message=f"INT8 macro F1 drop ({f1_drop:.4f}) > maximum limit ({max_f1_drop})",
                    observed=f1_drop,
                    threshold=max_f1_drop,
                    remediation="Inspect calibration dataset representative quality.",
                    source="int8_tflite.macro_f1",
                )
            )
            report["checks"]["macro_f1_drop"] = "FAILED"
        else:
            report["checks"]["macro_f1_drop"] = "PASSED"

    # 6.10 Minimum Accuracy & Macro F1
    min_acc = thresholds.get("accuracy_min", 0.40)
    min_f1 = thresholds.get("macro_f1_min", 0.60)
    int8_acc = int8_eval.get("accuracy") if isinstance(int8_eval, dict) else None

    if isinstance(int8_acc, (int, float)):
        if int8_acc < min_acc:
            defects.append(
                DefectItem(
                    code="INT8_ACCURACY_BELOW_THRESHOLD",
                    message=f"INT8 accuracy ({int8_acc:.4f}) < minimum limit ({min_acc})",
                    observed=int8_acc,
                    threshold=min_acc,
                    remediation="Re-train model with higher capacity or longer epochs.",
                    source="int8_tflite.accuracy",
                )
            )
            report["checks"]["overall_accuracy"] = "FAILED"
        else:
            report["checks"]["overall_accuracy"] = "PASSED"

    if isinstance(int8_f1, (int, float)):
        if int8_f1 < min_f1:
            defects.append(
                DefectItem(
                    code="INT8_MACRO_F1_BELOW_THRESHOLD",
                    message=f"INT8 macro F1 ({int8_f1:.4f}) < minimum limit ({min_f1})",
                    observed=int8_f1,
                    threshold=min_f1,
                    remediation="Re-balance classes or tune cross-entropy loss.",
                    source="int8_tflite.macro_f1",
                )
            )
            report["checks"]["overall_macro_f1"] = "FAILED"
        else:
            report["checks"]["overall_macro_f1"] = "PASSED"

    # 6.11 Input Saturation Ratio
    sat_ratio = int8_eval.get("input_saturation_ratio") if isinstance(int8_eval, dict) else None
    max_sat_ratio = thresholds.get("max_input_saturation_ratio", 0.05)

    if sat_ratio is None or not isinstance(sat_ratio, (int, float)) or math.isnan(sat_ratio) or math.isinf(sat_ratio):
        defects.append(
            DefectItem(
                code="INPUT_SATURATION_RATIO_INVALID",
                message=f"Input saturation ratio is missing or invalid: {sat_ratio}",
                observed=sat_ratio,
                remediation="Record valid input_saturation_ratio in [0.0, 1.0].",
                source="int8_tflite.input_saturation_ratio",
            )
        )
        report["checks"]["input_saturation"] = "FAILED"
    elif sat_ratio > max_sat_ratio:
        defects.append(
            DefectItem(
                code="SATURATION_RATIO_EXCEEDED",
                message=f"Input saturation ratio ({sat_ratio:.4f}) > maximum limit ({max_sat_ratio})",
                observed=sat_ratio,
                threshold=max_sat_ratio,
                remediation="Adjust clipping range or Z-score normalization scaling.",
                source="int8_tflite.input_saturation_ratio",
            )
        )
        report["checks"]["input_saturation"] = "FAILED"
    else:
        report["checks"]["input_saturation"] = "PASSED"

    # 6.12 Quantization MAE
    quant_mae = int8_eval.get("quantization_mae") if isinstance(int8_eval, dict) else None
    max_quant_mae = thresholds.get("max_quantization_mae", 0.10)
    if isinstance(quant_mae, (int, float)) and not math.isnan(quant_mae):
        if quant_mae > max_quant_mae:
            defects.append(
                DefectItem(
                    code="QUANTIZATION_MAE_EXCEEDED",
                    message=f"Quantization MAE ({quant_mae:.4f}) > maximum limit ({max_quant_mae})",
                    observed=quant_mae,
                    threshold=max_quant_mae,
                    remediation="Inspect Float-to-INT8 tensor quantization error.",
                    source="int8_tflite.quantization_mae",
                )
            )
            report["checks"]["quantization_mae"] = "FAILED"
        else:
            report["checks"]["quantization_mae"] = "PASSED"
        report["quantization_mae"] = float(quant_mae)
    else:
        report["checks"]["quantization_mae"] = "NOT_AVAILABLE"
        report["quantization_mae"] = "NOT_AVAILABLE"

    passed = len(defects) == 0
    report["status"] = "PASSED" if passed else "FAILED"
    report["failures"] = [dict(d) for d in defects]

    return passed, defects, report


def check_pipeline_smoke_gate(
    candidate_path: Path,
    metadata_path: Path,
) -> Tuple[bool, List[str]]:
    """Run the technical candidate gate without making a deployment claim."""
    passed, defects, _ = check_candidate_quality(
        candidate_path=Path(candidate_path),
        metadata_path=Path(metadata_path),
    )
    failures = [f"{item.get('code')}: {item.get('message')}" for item in defects]
    return passed, failures


def check_release_deployment_gate(metadata_path: Path) -> Tuple[bool, List[str]]:
    """Separate a synthetic pipeline-smoke pass from real deployment readiness."""
    try:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"Release Gate Blocked: metadata unreadable ({type(exc).__name__})."]

    failures: List[str] = []
    if metadata.get("validation_status") == "SYNTHETIC_SMOKE_ONLY":
        failures.append("Release Gate Blocked: Model validated solely on synthetic NPZ data.")
    if metadata.get("real_sensor_performance") != "VERIFIED":
        failures.append("Release Gate Blocked: Real-sensor performance is not verified.")
    if metadata.get("hardware_validation") != "VERIFIED":
        failures.append("Release Gate Blocked: Target hardware validation is incomplete.")
    return not failures, failures


def main():
    parser = argparse.ArgumentParser(description="SafeNest V6 Precise mmWave Candidate Defect Detector")
    parser.add_argument(
        "--candidate",
        type=str,
        default="models/mmwave/mmwave_resp_int8_v0.2.0_candidate.tflite",
        help="Path to candidate TFLite model",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="models/mmwave/mmwave_resp_int8_v0.2.0_candidate_metadata.json",
        help="Path to candidate metadata JSON",
    )
    parser.add_argument(
        "--contract",
        type=str,
        default="config/mmwave_input_contract.yaml",
        help="Path to mmWave input contract YAML",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="models/model_manifest.json",
        help="Path to model manifest JSON",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="benchmarks/mmwave_candidate_quality_check.json",
        help="Output check report path",
    )

    args = parser.parse_args()

    candidate_path = (project_root / args.candidate).resolve() if not Path(args.candidate).is_absolute() else Path(args.candidate)
    metadata_path = (project_root / args.metadata).resolve() if not Path(args.metadata).is_absolute() else Path(args.metadata)
    contract_path = (project_root / args.contract).resolve() if not Path(args.contract).is_absolute() else Path(args.contract)
    manifest_path = (project_root / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)
    report_path = (project_root / args.report).resolve() if not Path(args.report).is_absolute() else Path(args.report)

    print(f"🛡️ Running Candidate Quality & Defect Detection: {candidate_path.name}")
    passed, defects, report_dict = check_candidate_quality(
        candidate_path=candidate_path,
        metadata_path=metadata_path,
        contract_path=contract_path,
        manifest_path=manifest_path,
        model_root=project_root,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    if passed:
        print(f"\n✅ Candidate Quality Check PASSED")
        print(f"  - Scope: {report_dict['scope']}")
        print(f"  - Threshold Source: {report_dict['threshold_source']}")
        print(f"  - Report saved to: {report_path}")
        sys.exit(0)
    else:
        print(f"\n❌ Candidate Quality Check FAILED ({len(defects)} technical defects detected):")
        for defect in defects:
            print(f"  - {defect}")
        print(f"  - Defect report saved to: {report_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
