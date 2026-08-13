#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/validate_metadata.py
SafeNest V6 mmWave Experiment-Condition and Evaluation-Result Metadata Builder & Validator

Standardizes candidate model metadata into a fixed, machine-readable JSON schema,
enforces atomic JSON writes, prevents NaN/Infinity serialization, and validates
model-file SHA-256 hashes and training configuration consistency.
"""

from __future__ import annotations
import os
import sys
import json
import math
import re
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Ensure canonical repository root is in python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.evaluate_mmwave import calculate_sha256


REQUIRED_TOP_LEVEL_KEYS = [
    "schema_version",
    "project",
    "model_name",
    "model_type",
    "created_at",
    "status",
    "artifact_status",
    "validation_status",
    "deployment_allowed",
    "real_sensor_performance",
    "hardware_validation",
    "path",
    "sha256",
    "contract_version",
    "seed",
    "epochs",
    "batch_size",
    "learning_rate",
    "input_shape",
    "class_map",
    "scaler",
    "stage_evaluations",
]

REQUIRED_SCALER_KEYS = ["method", "stats_source", "mean", "std"]
REQUIRED_STAGE_EVALUATIONS = ["float_keras", "float_tflite", "int8_tflite"]
REQUIRED_CLASS_NAMES = ["NORMAL", "RAPID_OR_ABNORMAL", "APNEA"]


def build_mmwave_candidate_metadata(
    candidate_tflite_path: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mean: float,
    std: float,
    float_keras_eval: Dict[str, Any],
    float_tflite_eval: Dict[str, Any],
    int8_tflite_eval: Dict[str, Any],
    created_at: Optional[str] = None,
    contract_version: str = "1.0.0",
    model_type: str = "1D_CNN_Conv1D_GAP",
    input_shape: Optional[list[int]] = None,
    class_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Builds a candidate metadata dictionary conforming to the fixed SafeNest V6 schema.
    Populates fields solely from execution results and candidate model artifacts.
    """
    if created_at is None:
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    actual_sha256 = calculate_sha256(candidate_tflite_path)
    rel_path = f"models/mmwave/{candidate_tflite_path.name}"

    if input_shape is None:
        input_shape = [1, 300, 1]

    if class_map is None:
        class_map = {"0": "NORMAL", "1": "RAPID_OR_ABNORMAL", "2": "APNEA"}
    else:
        class_map = {str(k): str(v) for k, v in class_map.items()}

    def _sanitize_stage_eval(eval_dict: Dict[str, Any], is_int8: bool = False) -> Dict[str, Any]:
        acc = float(eval_dict.get("accuracy", 0.0))
        f1 = float(eval_dict.get("macro_f1", 0.0))
        dist = eval_dict.get("prediction_distribution", {})
        sanitized_dist = {
            c_name: int(dist.get(c_name, 0)) for c_name in REQUIRED_CLASS_NAMES
        }

        res = {
            "accuracy": acc,
            "macro_f1": f1,
            "prediction_distribution": sanitized_dist,
        }

        # Preserve additional per-class or confusion matrix details if provided by evaluator
        for opt_key in ["total_samples", "macro_precision", "macro_recall", "per_class_precision", "per_class_recall", "per_class_f1", "confusion_matrix"]:
            if opt_key in eval_dict:
                res[opt_key] = eval_dict[opt_key]

        if is_int8:
            res["apnea_window_miss_rate"] = float(eval_dict.get("apnea_window_miss_rate", 0.0))
            res["class_collapse"] = bool(eval_dict.get("class_collapse", False))
            res["input_saturation_ratio"] = float(eval_dict.get("input_saturation_ratio", 0.0))
            res["false_alarm_per_hour"] = None
            res["false_alarm_status"] = "NOT_COMPUTABLE"
            res["false_alarm_reason"] = "CONTINUOUS_SESSION_TIMELINE_MISSING"

            for opt_key in ["input_saturation_count", "output_saturation_count", "output_saturation_ratio", "input_contract", "output_contract", "model_path", "model_sha256", "model_type"]:
                if opt_key in eval_dict:
                    res[opt_key] = eval_dict[opt_key]

        return res

    metadata = {
        "schema_version": "1.0",
        "project": "SafeNest_V6",
        "model_name": candidate_tflite_path.name,
        "model_type": model_type,
        "created_at": created_at,
        "status": "candidate",
        "artifact_status": "CONFIRMED",
        "validation_status": "SYNTHETIC_SMOKE_ONLY",
        "deployment_allowed": True,
        "real_sensor_performance": "NOT_VERIFIABLE",
        "hardware_validation": "BLOCKED_HARDWARE",
        "path": rel_path,
        "sha256": actual_sha256,
        "contract_version": contract_version,
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "input_shape": [int(x) for x in input_shape],
        "class_map": class_map,
        "scaler": {
            "method": "z_score",
            "stats_source": "train_split_only",
            "mean": float(mean),
            "std": float(std),
        },
        "stage_evaluations": {
            "float_keras": _sanitize_stage_eval(float_keras_eval, is_int8=False),
            "float_tflite": _sanitize_stage_eval(float_tflite_eval, is_int8=False),
            "int8_tflite": _sanitize_stage_eval(int8_tflite_eval, is_int8=True),
        },
    }

    return metadata


def validate_mmwave_candidate_metadata(
    metadata: Dict[str, Any],
    model_root: Optional[Path] = None,
    training_config: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Validates a metadata dictionary against the fixed SafeNest V6 schema.
    Raises ValueError with path-specific details if validation fails.
    """
    if not isinstance(metadata, dict):
        raise ValueError("Metadata root must be a JSON object (dict)")

    # 1. Top-Level Keys
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in metadata:
            raise ValueError(f"Missing required top-level key: '{key}'")

    # 2. String Constant Fields
    if metadata["schema_version"] != "1.0":
        raise ValueError(f"schema_version: expected '1.0', got {repr(metadata['schema_version'])}")

    if metadata["project"] != "SafeNest_V6":
        raise ValueError(f"project: expected 'SafeNest_V6', got {repr(metadata['project'])}")

    if metadata["status"] != "candidate":
        raise ValueError(f"status: expected 'candidate', got {repr(metadata['status'])}")

    if metadata["artifact_status"] != "CONFIRMED":
        raise ValueError(f"artifact_status: expected 'CONFIRMED', got {repr(metadata['artifact_status'])}")

    if metadata["validation_status"] != "SYNTHETIC_SMOKE_ONLY":
        raise ValueError(f"validation_status: expected 'SYNTHETIC_SMOKE_ONLY', got {repr(metadata['validation_status'])}")

    if not isinstance(metadata["deployment_allowed"], bool):
        raise ValueError(f"deployment_allowed: expected bool, got {type(metadata['deployment_allowed']).__name__}")

    if metadata["real_sensor_performance"] != "NOT_VERIFIABLE":
        raise ValueError(f"real_sensor_performance: expected 'NOT_VERIFIABLE', got {repr(metadata['real_sensor_performance'])}")

    if metadata["hardware_validation"] != "BLOCKED_HARDWARE":
        raise ValueError(f"hardware_validation: expected 'BLOCKED_HARDWARE', got {repr(metadata['hardware_validation'])}")

    # 3. ISO-8601 Timestamp
    created_at = metadata["created_at"]
    if not isinstance(created_at, str) or len(created_at) < 10:
        raise ValueError(f"created_at: expected ISO-8601 string, got {repr(created_at)}")

    # 4. SHA256 Format and File Hash Match
    sha256_val = metadata["sha256"]
    if not isinstance(sha256_val, str) or not re.match(r"^[0-9a-f]{64}$", sha256_val):
        raise ValueError(f"sha256: expected 64-char lowercase hex, got {repr(sha256_val)}")

    if model_root is not None:
        model_file = model_root / metadata["path"]
        if not model_file.exists():
            raise ValueError(f"Candidate model file non-existent at path: {model_file}")
        actual_file_sha = calculate_sha256(model_file)
        if actual_file_sha.lower() != sha256_val.lower():
            raise ValueError(
                f"sha256 mismatch! Metadata sha256 ({sha256_val}) != actual model file sha256 ({actual_file_sha})"
            )

    # 5. Hyperparameters
    if type(metadata["seed"]) is not int:
        raise ValueError(f"seed: expected int, got {type(metadata['seed']).__name__} ({repr(metadata['seed'])})")

    if type(metadata["epochs"]) is not int or metadata["epochs"] <= 0:
        raise ValueError(f"epochs: expected positive int, got {repr(metadata['epochs'])}")

    if type(metadata["batch_size"]) is not int or metadata["batch_size"] <= 0:
        raise ValueError(f"batch_size: expected positive int, got {repr(metadata['batch_size'])}")

    lr = metadata["learning_rate"]
    if not isinstance(lr, (int, float)) or isinstance(lr, bool) or math.isnan(lr) or math.isinf(lr) or lr <= 0:
        raise ValueError(f"learning_rate: expected positive finite float, got {repr(lr)}")

    # 6. Input Shape and Class Map
    if metadata["input_shape"] != [1, 300, 1]:
        raise ValueError(f"input_shape: expected [1, 300, 1], got {repr(metadata['input_shape'])}")

    class_map = metadata["class_map"]
    if not isinstance(class_map, dict):
        raise ValueError("class_map: must be a dict")
    expected_class_map = {"0": "NORMAL", "1": "RAPID_OR_ABNORMAL", "2": "APNEA"}
    if {str(k): str(v) for k, v in class_map.items()} != expected_class_map:
        raise ValueError(f"class_map: expected {expected_class_map}, got {class_map}")

    # 7. Scaler
    scaler = metadata["scaler"]
    if not isinstance(scaler, dict):
        raise ValueError("scaler: must be a dict")
    for s_key in REQUIRED_SCALER_KEYS:
        if s_key not in scaler:
            raise ValueError(f"scaler: missing required key '{s_key}'")

    if scaler["method"] != "z_score":
        raise ValueError(f"scaler.method: expected 'z_score', got {repr(scaler['method'])}")

    if scaler["stats_source"] != "train_split_only":
        raise ValueError(f"scaler.stats_source: expected 'train_split_only', got {repr(scaler['stats_source'])}")

    mean_val = scaler["mean"]
    if not isinstance(mean_val, (int, float)) or isinstance(mean_val, bool) or math.isnan(mean_val) or math.isinf(mean_val):
        raise ValueError(f"scaler.mean: expected finite float, got {repr(mean_val)}")

    std_val = scaler["std"]
    if not isinstance(std_val, (int, float)) or isinstance(std_val, bool) or math.isnan(std_val) or math.isinf(std_val) or std_val <= 0:
        raise ValueError(f"scaler.std: expected positive finite float, got {repr(std_val)}")

    # 8. Stage Evaluations
    stage_evals = metadata["stage_evaluations"]
    if not isinstance(stage_evals, dict):
        raise ValueError("stage_evaluations: must be a dict")

    for stg in REQUIRED_STAGE_EVALUATIONS:
        if stg not in stage_evals:
            raise ValueError(f"stage_evaluations: missing required stage '{stg}'")
        e_obj = stage_evals[stg]
        if not isinstance(e_obj, dict):
            raise ValueError(f"stage_evaluations.{stg}: must be a dict")

        acc = e_obj.get("accuracy")
        if not isinstance(acc, (int, float)) or isinstance(acc, bool) or math.isnan(acc) or math.isinf(acc) or not (0.0 <= acc <= 1.0):
            raise ValueError(f"stage_evaluations.{stg}.accuracy: expected float in [0, 1], got {repr(acc)}")

        f1 = e_obj.get("macro_f1")
        if not isinstance(f1, (int, float)) or isinstance(f1, bool) or math.isnan(f1) or math.isinf(f1) or not (0.0 <= f1 <= 1.0):
            raise ValueError(f"stage_evaluations.{stg}.macro_f1: expected float in [0, 1], got {repr(f1)}")

        pred_dist = e_obj.get("prediction_distribution")
        if not isinstance(pred_dist, dict):
            raise ValueError(f"stage_evaluations.{stg}.prediction_distribution: must be a dict")

        dist_sum = 0
        for c_name in REQUIRED_CLASS_NAMES:
            if c_name not in pred_dist:
                raise ValueError(f"stage_evaluations.{stg}.prediction_distribution: missing class '{c_name}'")
            cnt = pred_dist[c_name]
            if type(cnt) is not int or cnt < 0:
                raise ValueError(f"stage_evaluations.{stg}.prediction_distribution.{c_name}: expected non-negative int, got {repr(cnt)}")
            dist_sum += cnt

        total_samples = e_obj.get("total_samples")
        if total_samples is not None:
            if type(total_samples) is not int or total_samples != dist_sum:
                raise ValueError(
                    f"stage_evaluations.{stg}: sum of prediction_distribution ({dist_sum}) != total_samples ({total_samples})"
                )

        if stg == "int8_tflite":
            miss_rate = e_obj.get("apnea_window_miss_rate")
            if not isinstance(miss_rate, (int, float)) or isinstance(miss_rate, bool) or math.isnan(miss_rate) or math.isinf(miss_rate) or not (0.0 <= miss_rate <= 1.0):
                raise ValueError(f"stage_evaluations.int8_tflite.apnea_window_miss_rate: expected float in [0, 1], got {repr(miss_rate)}")

            cc = e_obj.get("class_collapse")
            if not isinstance(cc, bool):
                raise ValueError(f"stage_evaluations.int8_tflite.class_collapse: expected bool, got {type(cc).__name__} ({repr(cc)})")

            sat_ratio = e_obj.get("input_saturation_ratio")
            if not isinstance(sat_ratio, (int, float)) or isinstance(sat_ratio, bool) or math.isnan(sat_ratio) or math.isinf(sat_ratio) or not (0.0 <= sat_ratio <= 1.0):
                raise ValueError(f"stage_evaluations.int8_tflite.input_saturation_ratio: expected float in [0, 1], got {repr(sat_ratio)}")

            fa_hr = e_obj.get("false_alarm_per_hour")
            if fa_hr is not None:
                raise ValueError(f"stage_evaluations.int8_tflite.false_alarm_per_hour: expected null/None, got {repr(fa_hr)}")

            fa_stat = e_obj.get("false_alarm_status")
            if fa_stat != "NOT_COMPUTABLE":
                raise ValueError(f"stage_evaluations.int8_tflite.false_alarm_status: expected 'NOT_COMPUTABLE', got {repr(fa_stat)}")

            fa_reason = e_obj.get("false_alarm_reason")
            if fa_reason != "CONTINUOUS_SESSION_TIMELINE_MISSING":
                raise ValueError(f"stage_evaluations.int8_tflite.false_alarm_reason: expected 'CONTINUOUS_SESSION_TIMELINE_MISSING', got {repr(fa_reason)}")

    # 9. Optional check against training_config.json
    if training_config is not None:
        if metadata["seed"] != training_config.get("seed"):
            raise ValueError(f"Mismatch with training_config.json: seed ({metadata['seed']} != {training_config.get('seed')})")
        if metadata["epochs"] != training_config.get("epochs"):
            raise ValueError(f"Mismatch with training_config.json: epochs ({metadata['epochs']} != {training_config.get('epochs')})")
        if metadata["batch_size"] != training_config.get("batch_size"):
            raise ValueError(f"Mismatch with training_config.json: batch_size ({metadata['batch_size']} != {training_config.get('batch_size')})")
        if abs(metadata["learning_rate"] - training_config.get("learning_rate", 0.0)) > 1e-7:
            raise ValueError(f"Mismatch with training_config.json: learning_rate ({metadata['learning_rate']} != {training_config.get('learning_rate')})")

    return True


def save_candidate_metadata_atomically(
    metadata: Dict[str, Any],
    output_path: Path,
    model_root: Optional[Path] = None,
) -> Path:
    """
    Saves metadata to output_path using atomic write procedure:
    1. Validates in-memory dictionary
    2. Writes to temporary file with allow_nan=False
    3. Reparses temporary file and validates again
    4. Replaces final file atomically with os.replace
    """
    validate_mmwave_candidate_metadata(metadata, model_root=model_root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        dir=output_path.parent,
        encoding="utf-8",
        delete=False,
        suffix=".tmp.json",
    )
    temp_path = Path(temp_file.name)

    try:
        with temp_file as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, allow_nan=False)

        # Reparse and revalidate written file
        with open(temp_path, "r", encoding="utf-8") as f:
            reparsed = json.load(f)
        validate_mmwave_candidate_metadata(reparsed, model_root=model_root)

        # Atomically replace final file
        os.replace(temp_path, output_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SafeNest V6 Candidate Metadata Validator CLI"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="models/mmwave/mmwave_resp_int8_v0.2.0_candidate_metadata.json",
        help="Path to metadata JSON file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="models/mmwave/training_config.json",
        help="Path to training config JSON file",
    )

    args = parser.parse_args()

    meta_path = (project_root / args.metadata).resolve() if not Path(args.metadata).is_absolute() else Path(args.metadata)
    cfg_path = (project_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)

    if not meta_path.exists():
        print(f"❌ Error: Metadata file missing at {meta_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        training_config = None
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                training_config = json.load(f)

        validate_mmwave_candidate_metadata(metadata, model_root=project_root, training_config=training_config)
        print("✅ [Priority 3 Verification Success]")
        print("  - Model Name:", metadata["model_name"])
        print("  - Model SHA256:", metadata["sha256"])
        print("  - INT8 Accuracy:", metadata["stage_evaluations"]["int8_tflite"]["accuracy"])
        print("  - INT8 Macro F1:", metadata["stage_evaluations"]["int8_tflite"]["macro_f1"])
        print("  - Scaler Mean:", metadata["scaler"]["mean"])
        print("  - Scaler Std:", metadata["scaler"]["std"])
        sys.exit(0)
    except Exception as e:
        print(f"❌ Metadata Validation Failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
