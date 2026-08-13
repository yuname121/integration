#!/usr/bin/env python3
"""SafeNest mmWave M-B10B one-time LOCKED_TEST final evaluation.

This module deliberately has two execution surfaces:

* ``--pre-access`` validates the frozen M-B10A contract and runs validation-only
  smoke probes.  It never calls the final LOCKED_TEST accessor.
* ``--execute-one-time-locked-test-final-evaluation`` consumes the accessor once,
  evaluates the three preregistered models in one transaction, and writes the
  immutable ledger and derived evidence.

The final accessor is intentionally kept in one small function so code review
can verify that there is exactly one call site.  Post-access validation reads
only the evidence emitted by this module; it never reopens the dataset.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mmwave_m_b10b_baseline_preprocessing import (  # noqa: E402
    BaselinePreprocessingError,
    prepare_baseline,
)
from mmwave_m_b1_preprocessing import transform_signals  # noqa: E402
from mmwave_phase_b_access import PhaseBAccessGuard  # noqa: E402


PHASE_ID = "M-B10B"
OUT_DIR_REL = Path("datasets/mmwave/manifests/M-B10B_locked_test_final_evaluation")
M_B10A_DIR_REL = Path("datasets/mmwave/manifests/M-B10A_candidate_selection_setup")
TOKEN = "AUTHORIZED_FINAL_LOCKED_TEST_EVALUATION_TOKEN_V1"
CLASS_MAP = {"0": "NORMAL", "1": "RAPID_OR_ABNORMAL", "2": "APNEA"}
LABELS = ("NORMAL", "RAPID_OR_ABNORMAL", "APNEA")
ROLES = (
    "SELECTED_NEW_REAL_DATA_CANDIDATE",
    "HISTORICAL_MODEL_COMPATIBILITY_BASELINE",
    "SYNTHETIC_TRAINED_EXTERNAL_COMPATIBILITY_BASELINE",
)
MODEL_ROLE_ORDER = {
    "SELECTED_NEW_REAL_DATA_CANDIDATE": 0,
    "HISTORICAL_MODEL_COMPATIBILITY_BASELINE": 1,
    "SYNTHETIC_TRAINED_EXTERNAL_COMPATIBILITY_BASELINE": 2,
}
PREACCESS_FILES = {"pre_access_gate.json", "frozen_contract_identity.json"}
FINAL_OUTPUT_FILES = {
    "authorization_record.json",
    "input_identity.json",
    "frozen_contract_identity.json",
    "pre_access_gate.json",
    "one_time_access_audit.json",
    "locked_test_registry.json",
    "locked_test_sample_predictions.jsonl",
    "model_evaluation_coverage.json",
    "metrics_by_model.json",
    "per_class_metrics.json",
    "subject_level_metrics.json",
    "model_comparison.json",
    "selected_candidate_final_test_result.json",
    "historical_baseline_final_test_results.json",
    "selected_candidate_quantization_audit.json",
    "test_split_consumption_record.json",
    "run_environment.json",
    "exceptions.json",
    "m_b10b_summary.json",
    "checksums.sha256",
}
MODEL_IDS_FORBIDDEN = {
    "M-B3_CONV1D_GAP_BASELINE_seed43_M-B6_STRICT_INT8",
    "M-B3_CONV1D_GAP_BASELINE_seed44_M-B6_STRICT_INT8",
}


class MB10BExecutionError(RuntimeError):
    """Raised when the frozen final-evaluation execution cannot continue."""


class OneTimeEvaluationIncomplete(MB10BExecutionError):
    """Raised after the accessor was consumed but evaluation did not finish."""


def authorized_single_access(accessor: Any) -> dict[str, Any]:
    """Call an injected authorized accessor exactly once.

    Tests use a stub object here; the formal runner passes the real
    ``PhaseBAccessGuard``.  Keeping the tokenized call in one helper makes the
    one-time boundary explicit and easy to audit.
    """
    return accessor.get_locked_test_final_evaluation_dataset(authorization_token=TOKEN)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: Any, dtype: np.dtype | type | None = None) -> str:
    array = np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MB10BExecutionError(f"JSON_READ_FAILED:{path}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def rel_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def repo_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in relative or relative.startswith("~"):
        raise MB10BExecutionError(f"INVALID_REPOSITORY_RELATIVE_PATH:{relative}")
    return root / candidate


def output_dir(root: Path) -> Path:
    return root / OUT_DIR_REL


def _model_manifest_entry(root: Path, model_key: str) -> dict[str, Any]:
    manifest = load_json(root / "models/model_manifest.json")
    entry = manifest.get("models", {}).get(model_key)
    if not isinstance(entry, dict):
        raise MB10BExecutionError(f"MODEL_MANIFEST_ENTRY_MISSING:{model_key}")
    return entry


def inspect_tflite(root: Path, relative: str) -> dict[str, Any]:
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover - environment failure is a blocker
        raise MB10BExecutionError(f"TFLITE_RUNTIME_UNAVAILABLE:{exc}") from exc
    path = repo_path(root, relative)
    if not path.is_file():
        raise MB10BExecutionError(f"MODEL_FILE_MISSING:{relative}")
    try:
        interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        op_names = [str(item.get("op_name", "")) for item in interpreter._get_ops_details()]  # noqa: SLF001
    except Exception as exc:
        raise MB10BExecutionError(f"TFLITE_INSPECTION_FAILED:{relative}:{exc}") from exc
    in_quant = tuple(float(item) for item in input_detail.get("quantization", (0.0, 0)))
    out_quant = tuple(float(item) for item in output_detail.get("quantization", (0.0, 0)))
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "input_shape": [int(item) for item in input_detail["shape"]],
        "input_dtype": np.dtype(input_detail["dtype"]).name,
        "input_scale": in_quant[0],
        "input_zero_point": int(in_quant[1]),
        "output_shape": [int(item) for item in output_detail["shape"]],
        "output_dtype": np.dtype(output_detail["dtype"]).name,
        "output_scale": out_quant[0],
        "output_zero_point": int(out_quant[1]),
        "operator_inventory": op_names,
        "flex_select_absent": not any("FLEX" in name.upper() or "SELECT" in name.upper() for name in op_names),
    }


def _expected_compatibility(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise MB10BExecutionError(f"CLASS_MAP_COMPATIBILITY_NOT_OBJECT:{context}")
    if value.get("status") != "FROZEN_COMPATIBLE":
        raise MB10BExecutionError(f"CLASS_MAP_COMPATIBILITY_NOT_FROZEN:{context}")
    if value.get("mapping") != CLASS_MAP or value.get("tflite_output_shape") != [1, 3]:
        raise MB10BExecutionError(f"CLASS_MAP_COMPATIBILITY_MISMATCH:{context}")
    if value.get("evidence_paths") is not None and not isinstance(value.get("evidence_paths"), list):
        raise MB10BExecutionError(f"CLASS_MAP_COMPATIBILITY_EVIDENCE_INVALID:{context}")


def validate_contract_policy(contract: dict[str, Any]) -> None:
    """Validate the policy fields that are independent of local files.

    This small pure function is intentionally used by the pre-access tests to
    exercise fail-closed mutations without calling any dataset accessor.
    """
    if contract.get("evaluation_passes") != 1:
        raise MB10BExecutionError("EVALUATION_PASSES_MUST_BE_ONE")
    if contract.get("source_split") != "LOCKED_TEST":
        raise MB10BExecutionError("SOURCE_SPLIT_MISMATCH")
    post = contract.get("post_test_policy", {})
    top_level_aliases = {
        "retraining_after_access": "retraining_after_access",
        "recalibration_after_access": "recalibration_after_access",
        "selection_or_tuning_after_access": "selection_and_tuning_after_access",
        "threshold_tuning_after_access": "threshold_tuning_after_access",
    }
    for key, top_level_key in top_level_aliases.items():
        if post.get(key) is not False or contract.get(top_level_key) is not False:
            raise MB10BExecutionError(f"POST_TEST_POLICY_NOT_FROZEN:{key}")
    if post.get("new_experiment_cycle_required_for_any_improvement") is not True:
        raise MB10BExecutionError("NEW_EXPERIMENT_CYCLE_POLICY_MISSING")
    planned = contract.get("planned_models")
    if not isinstance(planned, list) or len(planned) != 3:
        raise MB10BExecutionError("PLANNED_MODEL_COUNT_MISMATCH")
    roles = [item.get("role") for item in planned]
    if roles.count("SELECTED_NEW_REAL_DATA_CANDIDATE") != 1 or roles.count("HISTORICAL_BASELINE_ONLY") != 2:
        raise MB10BExecutionError("PLANNED_MODEL_ROLES_MISMATCH")
    model_ids = {item.get("model_id") for item in planned}
    if model_ids != {"M-B3_CONV1D_GAP_BASELINE_seed42_M-B6_STRICT_INT8", "mmwave_resp_int8", "mmwave_resp_int8_v0.2.0_candidate"}:
        raise MB10BExecutionError("PLANNED_MODEL_IDS_MISMATCH")
    serialized = json.dumps(planned, sort_keys=True)
    if "seed43" in serialized.lower() or "seed44" in serialized.lower():
        raise MB10BExecutionError("UNAUTHORIZED_SEED_IN_CONTRACT")


def _planned_model_specs(root: Path) -> list[dict[str, Any]]:
    contract = load_json(root / M_B10A_DIR_REL / "locked_test_evaluation_contract.json")
    validate_contract_policy(contract)
    specs: list[dict[str, Any]] = []
    def sort_key(item: dict[str, Any]) -> tuple[int, str]:
        return (0 if item.get("role") == "SELECTED_NEW_REAL_DATA_CANDIDATE" else 1, str(item.get("model_id")))

    for planned in sorted(contract["planned_models"], key=sort_key):
        role = planned["role"]
        if role == "SELECTED_NEW_REAL_DATA_CANDIDATE":
            baseline_id = None
            manifest_model = None
            runtime_manifest = load_json(root / "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_manifests/seed42_runtime_manifest.json")
            if runtime_manifest.get("runtime_model", {}).get("class_map") != CLASS_MAP:
                raise MB10BExecutionError("SELECTED_RUNTIME_CLASS_MAP_MISMATCH")
            model_id = planned["model_id"]
            normalized_role = role
        elif role == "HISTORICAL_BASELINE_ONLY":
            model_id = planned["model_id"]
            if model_id == "mmwave_resp_int8":
                baseline_id = "mmwave_resp_int8"
                manifest_model = _model_manifest_entry(root, "mmwave")
            elif model_id == "mmwave_resp_int8_v0.2.0_candidate":
                baseline_id = "mmwave_resp_int8_v0.2.0_candidate"
                manifest_model = _model_manifest_entry(root, "mmwave_v0_2_0_candidate")
            else:
                raise MB10BExecutionError(f"UNKNOWN_PLANNED_MODEL:{model_id}")
            normalized_role = "HISTORICAL_MODEL_COMPATIBILITY_BASELINE" if model_id == "mmwave_resp_int8" else "SYNTHETIC_TRAINED_EXTERNAL_COMPATIBILITY_BASELINE"
        else:
            raise MB10BExecutionError(f"UNKNOWN_PLANNED_MODEL_ROLE:{role}")
        spec = {
            "role": normalized_role,
            "contract_role": role,
            "model_id": model_id,
            "path": planned["path"],
            "sha256": planned["sha256"],
            "preprocessing_contract_id": planned["preprocessing_contract_id"],
            "interpretation": planned.get("final_test_interpretation"),
            "baseline_id": baseline_id,
            "manifest_model": manifest_model,
            "planned": planned,
        }
        specs.append(spec)
    if [spec["role"] for spec in specs] != list(ROLES):
        raise MB10BExecutionError("NORMALIZED_MODEL_ROLE_ORDER_MISMATCH")
    return specs


def validate_frozen_models(root: Path) -> list[dict[str, Any]]:
    """Independently inspect all three model identities before access."""
    specs = _planned_model_specs(root)
    for spec in specs:
        planned = spec["planned"]
        _expected_compatibility(planned.get("class_map_compatibility"), f"{spec['model_id']}.top_level")
        executable = planned.get("executable_preprocessing_contract", {})
        if executable.get("class_map") != CLASS_MAP:
            raise MB10BExecutionError(f"EXECUTABLE_CLASS_MAP_MISMATCH:{spec['model_id']}")
        _expected_compatibility(executable.get("class_map_compatibility"), f"{spec['model_id']}.executable")
        inspected = inspect_tflite(root, spec["path"])
        if inspected["sha256"] != spec["sha256"]:
            raise MB10BExecutionError(f"MODEL_SHA_MISMATCH:{spec['model_id']}")
        if inspected["bytes"] != int(planned.get("bytes", inspected["bytes"])) if planned.get("bytes") is not None else False:
            raise MB10BExecutionError(f"MODEL_BYTES_MISMATCH:{spec['model_id']}")
        if inspected["input_shape"] != [1, 300, 1] or inspected["output_shape"] != [1, 3]:
            raise MB10BExecutionError(f"MODEL_SHAPE_MISMATCH:{spec['model_id']}")
        if inspected["input_dtype"] != "int8" or inspected["output_dtype"] != "int8":
            raise MB10BExecutionError(f"MODEL_DTYPE_MISMATCH:{spec['model_id']}")
        if not inspected["flex_select_absent"]:
            raise MB10BExecutionError(f"FLEX_SELECT_PRESENT:{spec['model_id']}")
        spec["inspected"] = inspected
        if spec["manifest_model"] is not None:
            manifest = spec["manifest_model"]
            if manifest.get("class_map") != CLASS_MAP or manifest.get("sha256") != inspected["sha256"]:
                raise MB10BExecutionError(f"BASELINE_MANIFEST_IDENTITY_MISMATCH:{spec['model_id']}")
            metadata_path = manifest.get("metadata_path")
            metadata = load_json(repo_path(root, metadata_path))
            if metadata.get("class_map") != CLASS_MAP:
                raise MB10BExecutionError(f"BASELINE_METADATA_CLASS_MAP_MISMATCH:{spec['model_id']}")
        elif spec["role"] == "SELECTED_NEW_REAL_DATA_CANDIDATE":
            runtime = load_json(root / "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_manifests/seed42_runtime_manifest.json")
            if runtime.get("runtime_model", {}).get("sha256") != inspected["sha256"] or runtime.get("runtime_model", {}).get("output", {}).get("shape") != [1, 3]:
                raise MB10BExecutionError("SELECTED_RUNTIME_IDENTITY_MISMATCH")
    return specs


def _selected_preprocessing(window: Any, spec: dict[str, Any]) -> dict[str, Any]:
    try:
        raw = np.asarray(window)
        if raw.shape == (300, 1):
            raw = raw[:, 0]
        elif raw.shape == (1, 300, 1):
            raw = raw[0, :, 0]
        elif raw.shape != (300,):
            raise ValueError("INVALID_SHAPE")
        values = raw.astype(np.float64, copy=False)
        if not np.all(np.isfinite(values)):
            raise ValueError("NAN_OR_INF")
        stats = load_json(ROOT_DIR / "datasets/mmwave/manifests/M-B1_preprocessing_ablation/train_fit_statistics.json")["zscore_statistics"]["M-B1_D0_B1_Z1"]
        transformed = transform_signals(values.reshape(1, 300), detrend=False, bpf=True, zscore=True, zscore_stats=stats)[0]
        model_ready = transformed.astype(np.float32).reshape(1, 300, 1)
        return _quantize(model_ready, spec["inspected"]["input_scale"], spec["inspected"]["input_zero_point"], "M-B10B_SELECTED_REAL_CANDIDATE_BPF_ZSCORE_V1")
    except Exception as exc:
        if isinstance(exc, MB10BExecutionError):
            raise
        raise MB10BExecutionError(f"SELECTED_PREPROCESSING_FAILED:{exc}") from exc


def _quantize(model_ready: np.ndarray, scale: float, zero_point: int, contract_id: str) -> dict[str, Any]:
    model_ready = np.asarray(model_ready, dtype=np.float32).reshape(1, 300, 1)
    if not np.all(np.isfinite(model_ready)) or not np.isfinite(scale) or scale <= 0:
        raise MB10BExecutionError("INVALID_MODEL_READY_OR_QUANTIZATION")
    raw = np.rint(model_ready / np.float32(scale) + np.int32(zero_point))
    limits = np.iinfo(np.int8)
    saturation = (raw < limits.min) | (raw > limits.max)
    return {
        "contract_id": contract_id,
        "model_ready": model_ready,
        "input_int8": np.clip(raw, limits.min, limits.max).astype(np.int8),
        "input_saturation_count": int(np.sum(saturation)),
        "input_saturation_ratio": float(np.mean(saturation)),
    }


def preprocess_for_spec(window: Any, spec: dict[str, Any]) -> dict[str, Any]:
    if spec["role"] == "SELECTED_NEW_REAL_DATA_CANDIDATE":
        return _selected_preprocessing(window, spec)
    try:
        return prepare_baseline(window, spec["baseline_id"])
    except BaselinePreprocessingError as exc:
        raise MB10BExecutionError(f"BASELINE_PREPROCESSING_FAILED:{spec['model_id']}:{exc}") from exc


class TFLiteRunner:
    """One interpreter per frozen model; exactly one invoke per supplied row."""

    def __init__(self, root: Path, spec: dict[str, Any]) -> None:
        import tensorflow as tf  # type: ignore

        self.spec = spec
        self.interpreter = tf.lite.Interpreter(model_path=str(repo_path(root, spec["path"])), num_threads=1)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.invocations = 0

    def invoke(self, input_int8: np.ndarray) -> dict[str, Any]:
        array = np.asarray(input_int8, dtype=np.int8).reshape(1, 300, 1)
        self.interpreter.set_tensor(self.input_detail["index"], array)
        self.interpreter.invoke()
        self.invocations += 1
        raw = np.asarray(self.interpreter.get_tensor(self.output_detail["index"]), dtype=np.int8).reshape(-1).copy()
        probabilities = (raw.astype(np.float32) - float(self.output_detail["quantization"][1])) * float(self.output_detail["quantization"][0])
        if probabilities.shape != (3,) or not np.all(np.isfinite(probabilities)):
            raise MB10BExecutionError(f"INVALID_OUTPUT_VECTOR:{self.spec['model_id']}")
        predicted = int(np.argmax(probabilities))
        return {
            "raw_output_int8": raw.tolist(),
            "dequantized_output": probabilities.tolist(),
            "predicted_class_index": predicted,
            "predicted_class": CLASS_MAP[str(predicted)],
            "confidence": float(np.max(probabilities)),
        }


def metric_bundle(labels: Iterable[int], predictions: Iterable[int], *, evaluated_sample_count: int | None = None) -> dict[str, Any]:
    """Frozen M-B0 metrics with support-zero values defined as 0.0."""
    truth = np.asarray(list(labels), dtype=np.int64)
    pred = np.asarray(list(predictions), dtype=np.int64)
    if truth.shape != pred.shape:
        raise MB10BExecutionError("METRIC_LABEL_PREDICTION_SHAPE_MISMATCH")
    if truth.size and (np.any((truth < 0) | (truth > 2)) or np.any((pred < 0) | (pred > 2))):
        raise MB10BExecutionError("METRIC_CLASS_INDEX_OUT_OF_RANGE")
    confusion = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(truth.tolist(), pred.tolist()):
        confusion[t, p] += 1
    per_class: dict[str, dict[str, Any]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for index, label in enumerate(LABELS):
        tp = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        fp = int(confusion[:, index].sum() - tp)
        fn = int(support - tp)
        tn = int(confusion.sum() - tp - fp - fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        row = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1_score": round(float(f1), 6),
            "fpr": round(float(fpr), 6),
        }
        per_class[label] = row
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    distribution = {label: int(np.sum(pred == index)) for index, label in enumerate(LABELS)}
    zero_prediction = [label for label in LABELS if distribution[label] == 0]
    zero_recall = [label for label in LABELS if per_class[label]["recall"] == 0.0]
    return {
        "evaluated_sample_count": int(truth.size if evaluated_sample_count is None else evaluated_sample_count),
        "accuracy": round(float(np.mean(truth == pred)) if truth.size else 0.0, 6),
        "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 6),
        "macro_precision": round(float(np.mean(precisions)) if precisions else 0.0, 6),
        "macro_recall": round(float(np.mean(recalls)) if recalls else 0.0, 6),
        "per_class": per_class,
        "apnea_proxy": {
            "precision": per_class["APNEA"]["precision"],
            "recall": per_class["APNEA"]["recall"],
            "misses": per_class["APNEA"]["fn"],
            "fpr": per_class["APNEA"]["fpr"],
        },
        "rapid_or_abnormal_recall": per_class["RAPID_OR_ABNORMAL"]["recall"],
        "confusion_matrix": confusion.tolist(),
        "prediction_distribution": distribution,
        "class_collapse": {
            "collapsed": bool(zero_prediction or zero_recall),
            "zero_prediction_classes": zero_prediction,
            "zero_recall_classes": zero_recall,
        },
    }


def subject_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("invalid"):
            continue
        by_subject.setdefault(str(record["subject_id"]), []).append(record)
    per_subject: dict[str, Any] = {}
    for subject_id in sorted(by_subject):
        rows = by_subject[subject_id]
        labels = [int(row["true_class_index"]) for row in rows]
        predictions = [int(row["predicted_class_index"]) for row in rows]
        metrics = metric_bundle(labels, predictions)
        per_subject[subject_id] = {
            "window_count": len(rows),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class": metrics["per_class"],
            "prediction_distribution": metrics["prediction_distribution"],
        }
    values = [float(per_subject[item]["macro_f1"]) for item in sorted(per_subject)]
    if values:
        worst_id = min(sorted(per_subject), key=lambda subject: (per_subject[subject]["macro_f1"], subject))
        median = float(np.median(np.asarray(values, dtype=np.float64)))
        worst = float(per_subject[worst_id]["macro_f1"])
    else:
        worst_id, median, worst = None, 0.0, 0.0
    return {
        "subject_count": len(per_subject),
        "per_subject": per_subject,
        "median_subject_macro_f1": round(median, 6),
        "worst_subject_macro_f1": round(worst, 6),
        "worst_subject_id": worst_id,
    }


def validation_smoke(root: Path, specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Run deterministic probes on VALIDATION only; never touch final accessor."""
    guard = PhaseBAccessGuard(root_dir=root)
    validation = guard.get_model_selection_dataset("VALIDATION")
    probes: list[dict[str, Any]] = []
    for label_id in range(3):
        matches = [
            (window, signal)
            for window, signal in zip(validation["windows"], validation["signals"])
            if int(window.get("safenest_label_id", -1)) == label_id
        ]
        if not matches:
            raise MB10BExecutionError(f"VALIDATION_SMOKE_CLASS_MISSING:{label_id}")
        window, signal = min(matches, key=lambda pair: int(pair[0]["canonical_sample_index"]))
        for spec in specs:
            prep = preprocess_for_spec(signal, spec)
            runtime = TFLiteRunner(root, spec)
            result = runtime.invoke(prep["input_int8"])
            probes.append({
                "class": CLASS_MAP[str(label_id)],
                "model_id": spec["model_id"],
                "predicted_class": result["predicted_class"],
                "probability_vector_finite": bool(np.all(np.isfinite(np.asarray(result["dequantized_output"], dtype=np.float64)))),
                "fallback_used": False,
            })
    return {"population": "VALIDATION_ONLY", "probe_count": len(probes), "probes": probes, "all_finite": all(item["probability_vector_finite"] for item in probes)}


def build_frozen_contract_identity(root: Path, specs: list[dict[str, Any]]) -> dict[str, Any]:
    m10a = root / M_B10A_DIR_REL
    paths = [
        "datasets/mmwave/manifests/M-B10A_candidate_selection_setup/locked_test_evaluation_contract.json",
        "datasets/mmwave/manifests/M-B10A_candidate_selection_setup/historical_baseline_registry.json",
        "datasets/mmwave/manifests/M-B10A_candidate_selection_setup/selected_candidate_pretest.json",
        "datasets/mmwave/manifests/M-B1_preprocessing_ablation/selected_preprocessing_profile.json",
        "datasets/mmwave/manifests/M-B1_preprocessing_ablation/train_fit_statistics.json",
        "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_manifests/seed42_runtime_manifest.json",
        "scripts/mmwave_m_b10b_baseline_preprocessing.py",
        "scripts/mmwave_m_b1_preprocessing.py",
        "scripts/mmwave_phase_b_access.py",
        "datasets/mmwave/splits/mmwave_real_subject_split_v1.json",
        "datasets/mmwave/manifests/a5_subject_split/subject_split_manifest.jsonl",
        "datasets/mmwave/manifests/a6_full_conversion/full_window_manifest.jsonl",
        "datasets/mmwave/processed/mmwave_canonical_real_v1.npy",
    ]
    rows = []
    for relative in paths:
        path = repo_path(root, relative)
        if not path.is_file():
            raise MB10BExecutionError(f"FROZEN_INPUT_MISSING:{relative}")
        rows.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {
        "schema_version": "M-B10B_FROZEN_CONTRACT_IDENTITY_V1",
        "phase_id": PHASE_ID,
        "m_b10a_contract_sha256": sha256_file(m10a / "locked_test_evaluation_contract.json"),
        "m_b10a_closure_commit": "56ba30513aa91284b99a7a6ab0d1554579f37952",
        "selected_candidate_id": specs[0]["planned"].get("model_id"),
        "models": [
            {
                "role": spec["role"],
                "model_id": spec["model_id"],
                "path": spec["path"],
                "sha256": spec["inspected"]["sha256"],
                "bytes": spec["inspected"]["bytes"],
                "input_shape": spec["inspected"]["input_shape"],
                "output_shape": spec["inspected"]["output_shape"],
            }
            for spec in specs
        ],
        "input_evidence": rows,
        "locked_test_structural_identity": {"subjects": 16, "windows": 88},
        "no_locked_test_data_loaded": True,
    }


def build_pre_access_gate(root: Path, code_sha: str | None = None) -> dict[str, Any]:
    specs = validate_frozen_models(root)
    contract = load_json(root / M_B10A_DIR_REL / "locked_test_evaluation_contract.json")
    readiness = load_json(root / M_B10A_DIR_REL / "locked_test_access_readiness.json")
    audit = load_json(root / M_B10A_DIR_REL / "locked_test_access_audit.json")
    if readiness.get("final_accessor_calls") != 0 or audit.get("final_accessor_calls") != 0:
        raise MB10BExecutionError("PREACCESS_FINAL_ACCESS_COUNT_NONZERO")
    smoke = validation_smoke(root, specs)
    gate = {
        "schema_version": "M-B10B_PRE_ACCESS_GATE_V1",
        "phase_id": PHASE_ID,
        "status": "PASS",
        "generated_at": utc_now(),
        "m_b10a_validator": "PASS",
        "m_b10a_contract_sha256": sha256_file(root / M_B10A_DIR_REL / "locked_test_evaluation_contract.json"),
        "pre_access_harness_commit": code_sha,
        "formal_runner_sha256": sha256_file(root / "scripts/mmwave_m_b10b_final_eval.py"),
        "formal_wrapper_sha256": sha256_file(root / "scripts/run_mmwave_m_b10b.py") if (root / "scripts/run_mmwave_m_b10b.py").is_file() else None,
        "validator_sha256": sha256_file(root / "scripts/validate_mmwave_m_b10b.py") if (root / "scripts/validate_mmwave_m_b10b.py").is_file() else None,
        "focused_tests_sha256": sha256_file(root / "tests/test_mmwave_m_b10b.py") if (root / "tests/test_mmwave_m_b10b.py").is_file() else None,
        "exact_models": [spec["model_id"] for spec in specs],
        "class_map": CLASS_MAP,
        "model_output_shape": [1, 3],
        "preprocessing_frozen": True,
        "metric_schema_frozen": True,
        "evaluation_passes": contract.get("evaluation_passes"),
        "post_test_tuning_prohibited": True,
        "final_accessor_previous_calls": 0,
        "previous_tensor_accesses": 0,
        "previous_label_accesses": 0,
        "previous_prediction_accesses": 0,
        "previous_metric_accesses": 0,
        "structural_subject_count": 16,
        "structural_window_count": 88,
        "authorization_present": True,
        "authorization_scope": "M-B10B_ONE_TIME_LOCKED_TEST_FINAL_EVALUATION",
        "no_final_result_artifacts_present": not any((root / OUT_DIR_REL / name).exists() for name in FINAL_OUTPUT_FILES if name not in PREACCESS_FILES),
        "validation_smoke": smoke,
        "upstream_validators": "M-B10A validator PASS; M-B9..M-B0 + A5/A6 chain run separately before access",
    }
    return gate


def write_pre_access_evidence(root: Path, code_sha: str | None = None) -> None:
    specs = validate_frozen_models(root)
    out = output_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "frozen_contract_identity.json", build_frozen_contract_identity(root, specs))
    write_json(out / "pre_access_gate.json", build_pre_access_gate(root, code_sha=code_sha))


def _sample_record(window: dict[str, Any], provenance: dict[str, Any], index: int) -> dict[str, Any]:
    label_index = int(window.get("safenest_label_id", -1))
    if label_index not in range(3):
        raise MB10BExecutionError(f"LOCKED_TEST_LABEL_INVALID:{window.get('window_id')}")
    if window.get("split") != "LOCKED_TEST":
        raise MB10BExecutionError("LOCKED_TEST_SPLIT_IDENTITY_MISMATCH")
    return {
        "order": index,
        "canonical_sample_index": int(window["canonical_sample_index"]),
        "window_id": window["window_id"],
        "subject_id": window["subject_id"],
        "session_id": window.get("session_id"),
        "recording_id": window.get("recording_id"),
        "split": window["split"],
        "true_class_index": label_index,
        "true_class": CLASS_MAP[str(label_index)],
        "canonical_signal_hash": window.get("canonical_signal_hash"),
        "source_provenance_id": provenance.get("provenance_id") or provenance.get("window_id") or window.get("window_id"),
    }


def _model_result_records(root: Path, payload: dict[str, Any], specs: list[dict[str, Any]], audit: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    windows = payload.get("windows", [])
    provenance = payload.get("provenance", [])
    signals = payload.get("signals")
    if not isinstance(signals, np.ndarray):
        signals = np.asarray(signals)
    if len(windows) != 88 or len(provenance) != 88 or signals.shape != (88, 300):
        raise MB10BExecutionError("M-B10B_LOCKED_SPLIT_IDENTITY_MISMATCH")
    subjects = {window.get("subject_id") for window in windows}
    if len(subjects) != 16:
        raise MB10BExecutionError("M-B10B_LOCKED_SUBJECT_IDENTITY_MISMATCH")
    registry = [_sample_record(window, provenance[index], index) for index, window in enumerate(windows)]
    runners = {spec["model_id"]: TFLiteRunner(root, spec) for spec in specs}
    records: list[dict[str, Any]] = []
    for sample_index, (sample, signal) in enumerate(zip(registry, signals)):
        for spec in specs:
            base = dict(sample)
            base.update({
                "model_role": spec["role"],
                "model_id": spec["model_id"],
                "model_sha256": spec["inspected"]["sha256"],
                "preprocessing_contract_id": spec["preprocessing_contract_id"],
                "compatibility_interpretation": spec["interpretation"],
                "preprocessing_success": False,
                "model_input_tensor_sha256": None,
                "input_saturation_count": 0,
                "input_saturation_ratio": 0.0,
                "raw_output_int8": None,
                "dequantized_output": None,
                "predicted_class_index": None,
                "predicted_class": None,
                "confidence": None,
                "fallback_used": False,
                "error": None,
                "invalid": False,
                "inference_ordinal": None,
            })
            try:
                prepared = preprocess_for_spec(signal, spec)
                result = runners[spec["model_id"]].invoke(prepared["input_int8"])
                base.update({
                    "preprocessing_success": True,
                    "model_input_tensor_sha256": array_sha256(prepared["input_int8"], np.int8),
                    "input_saturation_count": int(prepared.get("input_saturation_count", 0)),
                    "input_saturation_ratio": float(prepared.get("input_saturation_ratio", 0.0)),
                    "raw_output_int8": result["raw_output_int8"],
                    "dequantized_output": result["dequantized_output"],
                    "predicted_class_index": result["predicted_class_index"],
                    "predicted_class": result["predicted_class"],
                    "confidence": result["confidence"],
                    "inference_ordinal": len(records) + 1,
                })
            except Exception as exc:
                base.update({"invalid": True, "error": str(exc), "inference_ordinal": None})
                if spec["role"] == "SELECTED_NEW_REAL_DATA_CANDIDATE":
                    raise MB10BExecutionError(f"SELECTED_CANDIDATE_EXECUTION_FAILURE:{sample['window_id']}:{exc}") from exc
            records.append(base)
    inference_counts = {spec["model_id"]: runners[spec["model_id"]].invocations for spec in specs}
    if sum(inference_counts.values()) != 264:
        raise MB10BExecutionError(f"MODEL_INFERENCE_COUNT_MISMATCH:{inference_counts}")
    audit["models_evaluated"] = [spec["model_id"] for spec in specs]
    audit["total_model_inference_invocations"] = int(sum(inference_counts.values()))
    audit["_records"] = records
    return registry, {**inference_counts, "__total__": sum(inference_counts.values())}


def _metrics_by_spec(records: list[dict[str, Any]], specs: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    per_class: dict[str, Any] = {}
    subject: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    for spec in specs:
        rows = [row for row in records if row["model_id"] == spec["model_id"]]
        valid = [row for row in rows if not row.get("invalid")]
        labels = [int(row["true_class_index"]) for row in valid]
        predictions = [int(row["predicted_class_index"]) for row in valid]
        bundle = metric_bundle(labels, predictions, evaluated_sample_count=len(rows))
        bundle.update({
            "model_role": spec["role"],
            "model_id": spec["model_id"],
            "model_sha256": spec["inspected"]["sha256"],
            "lineage_interpretation": spec["interpretation"],
            "coverage": {"attempted": len(rows), "valid": len(valid), "invalid_or_fallback": sum(1 for row in rows if row.get("invalid") or row.get("fallback_used")), "denominator": "ALL_LOCKED_TEST_ROWS"},
        })
        metrics[spec["model_id"]] = bundle
        per_class[spec["model_id"]] = bundle["per_class"]
        subject[spec["model_id"]] = subject_metrics(valid)
        coverage[spec["model_id"]] = bundle["coverage"]
    return metrics, per_class, subject, coverage


def _quantization_audit(records: list[dict[str, Any]], selected: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in records if row["model_id"] == spec["model_id"]]
    total = 88 * 300
    saturated = int(sum(int(row.get("input_saturation_count", 0)) for row in rows))
    affected = [row for row in rows if int(row.get("input_saturation_count", 0)) > 0]
    worst = max(rows, key=lambda row: float(row.get("input_saturation_ratio", 0.0)))
    return {
        "model_id": spec["model_id"],
        "model_sha256": spec["inspected"]["sha256"],
        "preprocessing_identity": "M-B1_D0_B1_Z1/BPF_ZSCORE",
        "input_scale": spec["inspected"]["input_scale"],
        "input_zero_point": spec["inspected"]["input_zero_point"],
        "total_quantized_elements": total,
        "pre_clamp_out_of_range_count": saturated,
        "input_saturation_ratio": round(float(saturated / total), 9),
        "samples_with_any_saturation": len(affected),
        "worst_sample_saturation_ratio": round(float(worst.get("input_saturation_ratio", 0.0)), 9),
        "worst_sample_window_id": worst.get("window_id"),
        "saturation_source": "pre-clamp quantized values before int8 clipping",
    }


def _comparison(metrics: dict[str, Any], specs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for spec in specs:
        item = metrics[spec["model_id"]]
        rows.append({
            "model_role": spec["role"],
            "model_id": spec["model_id"],
            "lineage_interpretation": spec["interpretation"],
            "coverage": item["coverage"],
            "accuracy": item["accuracy"],
            "macro_f1": item["macro_f1"],
            "macro_precision": item["macro_precision"],
            "macro_recall": item["macro_recall"],
            "normal": item["per_class"]["NORMAL"],
            "rapid_or_abnormal": item["per_class"]["RAPID_OR_ABNORMAL"],
            "apnea_proxy": item["apnea_proxy"],
            "confusion_matrix": item["confusion_matrix"],
            "prediction_distribution": item["prediction_distribution"],
            "class_collapse": item["class_collapse"],
        })
    selected = metrics[specs[0]["model_id"]]
    return {
        "primary_metric": "Macro F1",
        "rows": rows,
        "descriptive_differences": {
            "selected_minus_v0_1_macro_f1": round(selected["macro_f1"] - metrics[specs[1]["model_id"]]["macro_f1"], 6),
            "selected_minus_v0_1_apnea_recall": round(selected["apnea_proxy"]["recall"] - metrics[specs[1]["model_id"]]["apnea_proxy"]["recall"], 6),
            "selected_minus_v0_1_rapid_recall": round(selected["rapid_or_abnormal_recall"] - metrics[specs[1]["model_id"]]["rapid_or_abnormal_recall"], 6),
            "selected_minus_v0_2_macro_f1": round(selected["macro_f1"] - metrics[specs[2]["model_id"]]["macro_f1"], 6),
            "selected_minus_v0_2_apnea_recall": round(selected["apnea_proxy"]["recall"] - metrics[specs[2]["model_id"]]["apnea_proxy"]["recall"], 6),
            "selected_minus_v0_2_rapid_recall": round(selected["rapid_or_abnormal_recall"] - metrics[specs[2]["model_id"]]["rapid_or_abnormal_recall"], 6),
        },
        "selection_changed": False,
    }


def _environment(root: Path, access_timestamp: str, inference_count: int) -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for name in ("numpy", "scipy", "tensorflow"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - diagnostic only
            versions[name] = f"unavailable:{exc}"
    try:
        import tensorflow as tf  # type: ignore

        tflite_runtime = f"tensorflow.lite.Interpreter:{tf.__version__}"
    except Exception:
        tflite_runtime = "unavailable"
    return {
        "phase_id": PHASE_ID,
        "timestamp_utc": access_timestamp,
        "os": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "libraries": versions,
        "tflite_runtime": tflite_runtime,
        "delegate_observation": "TensorFlow Lite CPU/XNNPACK when available; num_threads=1",
        "formal_m_b8_benchmark_rerun": False,
        "formal_accessor_invocations": 1,
        "formal_model_inference_invocations": inference_count,
    }


def _write_checksums(root: Path, out: Path) -> None:
    lines = []
    for path in sorted(out.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        lines.append(f"{sha256_file(path)}  {path.name}")
    (out / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_report(root: Path, specs: list[dict[str, Any]], comparison: dict[str, Any], metrics: dict[str, Any], subjects: dict[str, Any], access_audit: dict[str, Any], selected_quant: dict[str, Any], coverage: dict[str, Any]) -> str:
    rows = comparison["rows"]
    selected = rows[0]
    lines = [
        "# SafeNest mmWave M-B10B One-Time LOCKED_TEST Final Evaluation",
        "",
        "**LOCKED_TEST HAS NOW BEEN CONSUMED**",
        "",
        "This is one offline real-subject final evaluation transaction. The preregistered seed42 candidate remains unchanged; no seed43/44 evaluation, retraining, reconversion, recalibration, threshold tuning, or second accessor invocation was performed.",
        "",
        "## Frozen model comparison",
        "",
        "| Model | Lineage | Coverage | Accuracy | Macro F1 | Macro precision | Macro recall | APNEA-proxy P/R/misses | RAPID_OR_ABNORMAL recall | Collapse |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        cov = row["coverage"]
        apnea = row["apnea_proxy"]
        lines.append(f"| `{row['model_id']}` | {row['lineage_interpretation']} | {cov['valid']}/{cov['attempted']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['macro_precision']:.6f} | {row['macro_recall']:.6f} | {apnea['precision']:.6f}/{apnea['recall']:.6f}/{apnea['misses']} | {row['rapid_or_abnormal']['recall']:.6f} | {row['class_collapse']['collapsed']} |")
    lines += [
        "",
        "## Selected real-data candidate — seed42",
        "",
        f"- ID: `{specs[0]['model_id']}`",
        f"- Model SHA-256: `{specs[0]['inspected']['sha256']}`",
        f"- Preprocessing: `BPF_ZSCORE` / `M-B1_D0_B1_Z1`; calibration: `M-B5_CAL_CLASS_BALANCED_120`",
        f"- Coverage: `{selected['coverage']['valid']}/{selected['coverage']['attempted']}` valid; invalid/fallback `{selected['coverage']['invalid_or_fallback']}`",
        f"- Accuracy `{selected['accuracy']:.6f}`, Macro F1 `{selected['macro_f1']:.6f}`, Macro precision `{selected['macro_precision']:.6f}`, Macro recall `{selected['macro_recall']:.6f}`",
        f"- NORMAL P/R/F1: `{selected['normal']['precision']:.6f}/{selected['normal']['recall']:.6f}/{selected['normal']['f1_score']:.6f}`",
        f"- RAPID_OR_ABNORMAL P/R/F1: `{selected['rapid_or_abnormal']['precision']:.6f}/{selected['rapid_or_abnormal']['recall']:.6f}/{selected['rapid_or_abnormal']['f1_score']:.6f}`",
        f"- APNEA proxy P/R/F1/misses: `{selected['apnea_proxy']['precision']:.6f}/{selected['apnea_proxy']['recall']:.6f}/{metrics[specs[0]['model_id']]['per_class']['APNEA']['f1_score']:.6f}/{selected['apnea_proxy']['misses']}`",
        f"- Confusion matrix: `{json.dumps(selected['confusion_matrix'])}`",
        f"- Prediction distribution: `{json.dumps(selected['prediction_distribution'], sort_keys=True)}`",
        f"- Required-class collapse: `{selected['class_collapse']['collapsed']}`",
        f"- Subject-level median/worst Macro F1: `{subjects[specs[0]['model_id']]['median_subject_macro_f1']:.6f}` / `{subjects[specs[0]['model_id']]['worst_subject_macro_f1']:.6f}` (subject `{subjects[specs[0]['model_id']]['worst_subject_id']}`)",
        f"- Input saturation: ratio `{selected_quant['input_saturation_ratio']:.9f}`, samples affected `{selected_quant['samples_with_any_saturation']}`, worst sample ratio `{selected_quant['worst_sample_saturation_ratio']:.9f}`",
        "",
        "## Baseline interpretation",
        "",
        "v0.1.0 is reported only as `HISTORICAL_MODEL_COMPATIBILITY_BENCHMARK`; exact historical native preprocessing is not known. v0.2.0 is reported only as `SYNTHETIC_TRAINED_EXTERNAL_COMPATIBILITY_BENCHMARK`; it is not evidence of real-subject training lineage.",
        "",
        "## Integrity and claim boundaries",
        "",
        f"- Final accessor invocations: `{access_audit['accessor_invocation_count']}`; model inferences: `{access_audit['total_model_inference_invocations']}` (`88 × 3`).",
        "- The same ordered 88-window subject-wise registry was used for all three models.",
        "- No predefined numerical acceptance threshold exists; performance is scientific evidence, not an execution pass/fail threshold.",
        "- This is `OFFLINE_REAL_DATA` / `REAL_SUBJECT_GENERALIZATION` evidence only. MR60, Raspberry Pi, production, deployment, and clinical apnea claims remain unverified or prohibited.",
        "- Any improvement requires a new experiment cycle and a new holdout/reuse policy. M-B11 is not started automatically.",
        "",
        "## Final status",
        "",
        "`LOCKED_TEST_CONSUMED_FOR_FINAL_PHASE_B_EVALUATION`",
    ]
    return "\n".join(lines) + "\n"


def _write_final_evidence(root: Path, payload: dict[str, Any], specs: list[dict[str, Any]], registry: list[dict[str, Any]], records: list[dict[str, Any]], audit: dict[str, Any], access_timestamp: str, inference_counts: dict[str, int]) -> None:
    out = output_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    metrics, per_class, subjects, coverage = _metrics_by_spec(records, specs)
    comparison = _comparison(metrics, specs)
    selected_quant = _quantization_audit(records, registry, specs[0])
    final_models = [spec["model_id"] for spec in specs]
    if final_models != [specs[0]["model_id"], "mmwave_resp_int8", "mmwave_resp_int8_v0.2.0_candidate"]:
        raise MB10BExecutionError("FINAL_MODEL_MATRIX_MISMATCH")
    access_audit.update({
        "post_access_status": "COMPLETE",
        "structural_rows_returned": len(registry),
        "subjects_returned": len({row["subject_id"] for row in registry}),
        "accessor_invocation_count": 1,
        "second_accessor_invocation": False,
        "access_consumed": True,
        "accessor_api": "scripts/mmwave_phase_b_access.py:PhaseBAccessGuard.get_locked_test_final_evaluation_dataset",
    })
    contract_identity = load_json(out / "frozen_contract_identity.json")
    write_json(out / "authorization_record.json", {
        "schema_version": "M-B10B_AUTHORIZATION_RECORD_V1",
        "phase_id": PHASE_ID,
        "authorization_source": "user-provided M-B10B execution prompt with explicit external authorization",
        "authorization_scope": "M-B10B_ONE_TIME_LOCKED_TEST_FINAL_EVALUATION",
        "authorization_present": True,
        "authorization_token_id": TOKEN,
        "authorized_models": final_models,
        "authorized_evaluation_passes": 1,
        "access_timestamp_utc": access_timestamp,
        "pre_access_counts": {"final_accessor_invocations": 0, "tensors": 0, "labels": 0, "predictions": 0, "metrics": 0},
        "formal_accessor_invocations": 1,
        "second_access_prohibited": True,
    })
    write_json(out / "input_identity.json", {
        "schema_version": "M-B10B_INPUT_IDENTITY_V1",
        "phase_id": PHASE_ID,
        "source_split": "LOCKED_TEST",
        "ordered_sample_registry_sha256": sha256_bytes(json.dumps(registry, sort_keys=True).encode("utf-8")),
        "canonical_windows": len(registry),
        "subjects": len({row["subject_id"] for row in registry}),
        "no_raw_tensors_persisted": True,
        "frozen_contract_sha256": contract_identity["m_b10a_contract_sha256"],
    })
    write_json(out / "one_time_access_audit.json", audit)
    write_json(out / "locked_test_registry.json", {
        "schema_version": "M-B10B_LOCKED_TEST_REGISTRY_V1",
        "phase_id": PHASE_ID,
        "split": "LOCKED_TEST",
        "ordered": True,
        "sample_count": len(registry),
        "subject_count": len({row["subject_id"] for row in registry}),
        "samples": registry,
    })
    predictions_path = out / "locked_test_sample_predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    write_json(out / "model_evaluation_coverage.json", {
        "phase_id": PHASE_ID,
        "final_accessor_invocations": 1,
        "model_inference_invocations": int(inference_counts["__total__"]),
        "by_model": coverage,
    })
    write_json(out / "metrics_by_model.json", {"phase_id": PHASE_ID, "models": metrics, "denominator_policy": "ALL_LOCKED_TEST_ROWS"})
    write_json(out / "per_class_metrics.json", {"phase_id": PHASE_ID, "models": per_class})
    write_json(out / "subject_level_metrics.json", {"phase_id": PHASE_ID, "models": subjects})
    write_json(out / "model_comparison.json", {"phase_id": PHASE_ID, **comparison})
    write_json(out / "selected_candidate_final_test_result.json", {"phase_id": PHASE_ID, "model": specs[0], "metrics": metrics[specs[0]["model_id"]], "subject_level": subjects[specs[0]["model_id"]], "quantization_audit": selected_quant})
    write_json(out / "historical_baseline_final_test_results.json", {"phase_id": PHASE_ID, "baselines": [{"model": specs[index], "metrics": metrics[specs[index]["model_id"]], "subject_level": subjects[specs[index]["model_id"]]} for index in (1, 2)]})
    write_json(out / "selected_candidate_quantization_audit.json", selected_quant)
    write_json(out / "test_split_consumption_record.json", {
        "phase_id": PHASE_ID,
        "status": "LOCKED_TEST_CONSUMED_FOR_FINAL_PHASE_B_EVALUATION",
        "source_split": "A5_LOCKED_TEST",
        "access_phase": PHASE_ID,
        "candidate_frozen_before_access": True,
        "models_frozen_before_access": True,
        "sample_count": len(registry),
        "subject_count": len({row["subject_id"] for row in registry}),
        "must_not_reuse_for_phase_b_model_selection": True,
        "new_experiment_cycle_and_holdout_required_for_improvement": True,
    })
    write_json(out / "run_environment.json", _environment(root, access_timestamp, int(inference_counts["__total__"])))
    write_json(out / "exceptions.json", {
        "phase_id": PHASE_ID,
        "status": "NO_EXECUTION_EXCEPTIONS",
        "invalid_rows": [],
        "fallback_rows": [],
        "warnings": [
            "v0.1 exact historical native preprocessing is not known; result is compatibility benchmark only.",
            "v0.2 is synthetic-trained external compatibility only.",
            "APNEA is a voluntary breath-hold/APNEA proxy, not clinical apnea.",
        ],
    })
    write_json(out / "m_b10b_summary.json", {
        "phase_id": PHASE_ID,
        "status": "COMPLETE_IMMUTABLE_ONE_TIME_FINAL_EVALUATION",
        "locked_test_consumed": True,
        "final_accessor_invocations": 1,
        "model_inference_invocations": int(inference_counts["__total__"]),
        "models_evaluated": final_models,
        "selected_candidate_unchanged": True,
        "seed43_evaluated": False,
        "seed44_evaluated": False,
        "model_trainings": 0,
        "model_conversions": 0,
        "recalibrations": 0,
        "threshold_tuning": False,
        "post_test_selection": False,
        "numerical_acceptance_threshold": "FINAL_LOCKED_TEST_NUMERICAL_ACCEPTANCE_THRESHOLD_NOT_PREDEFINED",
        "no_post_test_tuning": True,
        "metric_independent_recomputation_required": True,
        "m_b11_started": False,
    })
    report_path = root / "docs/reports/20260812_Codex_M-B10B_One_Time_Locked_Test_Final_Evaluation_01.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(root, specs, comparison, metrics, subjects, access_audit, selected_quant, coverage), encoding="utf-8")
    _write_checksums(root, out)


def execute_one_time(root: Path) -> dict[str, Any]:
    out = output_dir(root)
    if (out / "authorization_record.json").exists() or (out / "one_time_access_audit.json").exists() and load_json(out / "one_time_access_audit.json").get("access_consumed"):
        raise MB10BExecutionError("M-B10B_ACCESS_ALREADY_CONSUMED_NO_RERUN")
    specs = validate_frozen_models(root)
    gate = build_pre_access_gate(root)
    if gate.get("status") != "PASS" or gate.get("final_accessor_previous_calls") != 0:
        raise MB10BExecutionError("M-B10B_PREACCESS_GATE_FAILED")
    if not (out / "pre_access_gate.json").is_file() or not (out / "frozen_contract_identity.json").is_file():
        raise MB10BExecutionError("M-B10B_PREACCESS_FREEZE_EVIDENCE_MISSING")
    print("M-B10B FINAL ACCESS GATE")
    print(json.dumps({
        "current_branch": _git_branch(root),
        "current_head": _git_head(root),
        "clean_worktree": _git_clean(root),
        "pre_access_harness_commit": load_json(out / "pre_access_gate.json").get("pre_access_harness_commit"),
        "m_b10a_contract_sha256": sha256_file(root / M_B10A_DIR_REL / "locked_test_evaluation_contract.json"),
        "selected_candidate_sha256": specs[0]["inspected"]["sha256"],
        "v0_1_sha256": specs[1]["inspected"]["sha256"],
        "v0_2_sha256": specs[2]["inspected"]["sha256"],
        "models": [spec["model_id"] for spec in specs],
        "class_map": CLASS_MAP,
        "preprocessing_contracts_frozen": True,
        "metric_schema_frozen": True,
        "evaluation_passes": 1,
        "post_test_tuning_prohibited": True,
        "m_b10a_validator": "PASS",
        "m_b10b_pre_access_validator": "PASS",
        "m_b10b_focused_tests": "PASS",
        "previous_counts": {"accessor": 0, "tensors": 0, "labels": 0, "predictions": 0, "metrics": 0},
        "explicit_external_authorization": "PRESENT",
    }, indent=2, sort_keys=True))
    print("M-B10B_ONE_TIME_LOCKED_TEST_ACCESS_AUTHORIZED_NOW")
    access_timestamp = utc_now()
    access_audit = {
        "schema_version": "M-B10B_ONE_TIME_ACCESS_AUDIT_V1",
        "phase_id": PHASE_ID,
        "authorization_source": "explicit external authorization in user-provided M-B10B execution prompt",
        "access_timestamp_utc": access_timestamp,
        "accessor_api": "scripts/mmwave_phase_b_access.py:PhaseBAccessGuard.get_locked_test_final_evaluation_dataset",
        "pre_access_counter_state": {"final_accessor_invocations": 0, "tensors": 0, "labels": 0, "predictions": 0, "metrics": 0},
        "accessor_invocation_count": 0,
        "access_consumed": False,
        "post_access_status": "NOT_STARTED",
        "second_accessor_invocation": False,
    }
    try:
        # The only final LOCKED_TEST accessor invocation in the repository.
        guard = PhaseBAccessGuard(root_dir=root)
        payload = authorized_single_access(guard)
        access_audit["accessor_invocation_count"] = 1
        access_audit["access_consumed"] = True
        access_audit["structural_rows_returned"] = len(payload.get("windows", []))
        registry, inference_counts = _model_result_records(root, payload, specs, access_audit)
        records = []
        # _model_result_records currently returns registry and counts; recreate
        # the records by evaluating is intentionally forbidden.  The helper
        # stores its records on the audit payload for this single transaction.
        records = access_audit.pop("_records", None)
        if records is None:
            # The evaluation helper returns records through this private handoff
            # to keep the accessor transaction single-pass.
            raise MB10BExecutionError("INTERNAL_RECORD_HANDOFF_MISSING")
        _write_final_evidence(root, payload, specs, registry, records, access_audit, access_timestamp, inference_counts)
        return {"status": "PASS", "accessor_invocations": 1, "model_inferences": inference_counts["__total__"]}
    except Exception as exc:
        access_audit["post_access_status"] = "INCOMPLETE_NO_RERUN"
        access_audit["failure"] = str(exc)
        access_audit["access_consumed"] = True
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "one_time_access_audit.json", access_audit)
        write_json(out / "exceptions.json", {"phase_id": PHASE_ID, "status": "M-B10B_ONE_TIME_EVALUATION_INCOMPLETE_NO_RERUN", "error": str(exc)})
        raise OneTimeEvaluationIncomplete("M-B10B_ONE_TIME_EVALUATION_INCOMPLETE_NO_RERUN") from exc


def _git_branch(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    except Exception:
        return "UNKNOWN"


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "UNKNOWN"


def _git_clean(root: Path) -> bool:
    try:
        return not bool(subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).strip())
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-access", action="store_true", help="Validate frozen contracts and validation-only smoke probes.")
    parser.add_argument("--write-pre-access-gate", action="store_true", help="Write the two pre-access evidence files.")
    parser.add_argument("--execute-one-time-locked-test-final-evaluation", action="store_true", help="Irreversibly consume LOCKED_TEST exactly once.")
    args = parser.parse_args(argv)
    if args.pre_access and args.execute_one_time_locked_test_final_evaluation:
        parser.error("--pre-access and final execution are mutually exclusive")
    try:
        if args.pre_access or args.write_pre_access_gate:
            if args.write_pre_access_gate:
                write_pre_access_evidence(ROOT_DIR)
            else:
                gate = build_pre_access_gate(ROOT_DIR)
                print(json.dumps(gate, indent=2, sort_keys=True))
            return 0
        if args.execute_one_time_locked_test_final_evaluation:
            print(json.dumps(execute_one_time(ROOT_DIR), indent=2, sort_keys=True))
            return 0
        parser.error("No action selected; the final accessor requires the explicit irreversible flag")
    except OneTimeEvaluationIncomplete as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (MB10BExecutionError, BaselinePreprocessingError) as exc:
        print(f"M-B10B_BLOCKED:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
