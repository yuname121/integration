#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/evaluate_mmwave.py
SafeNest V6 mmWave Evaluator Script

Supports 3-stage evaluation (Float Keras, Float TFLite, INT8 TFLite).
Computes accuracy, macro precision/recall/F1, per-class metrics, confusion matrix,
apnea_window_miss_rate, prediction distribution, class collapse, and input/output saturation ratios.
"""

from __future__ import annotations
import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np

# Ensure canonical repository root is in python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preprocessing.mmwave import MMWavePreprocessor


def artifact_path(path: Path) -> str:
    """Serialize an input path without persisting a machine-specific absolute path."""
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return f"EXTERNAL_INPUT/{path.name}"


def calculate_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def evaluate_tflite_model(
    model_path: Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    preprocessor: MMWavePreprocessor,
    class_map: Dict[int, str]
) -> Dict[str, Any]:
    import tensorflow as tf

    sha256_val = calculate_sha256(model_path)
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_dtype = input_details["dtype"]
    out_dtype = output_details["dtype"]

    in_scale, in_zero_point = input_details.get("quantization", (0.0, 0))
    out_scale, out_zero_point = output_details.get("quantization", (0.0, 0))

    y_preds = []
    input_sat_count = 0
    output_sat_count = 0

    for i in range(len(X_test)):
        sample_x, _ = preprocessor.preprocess_window(X_test[i])  # shape (1, 300, 1)

        if in_dtype == np.int8:
            if in_scale > 0:
                q_x = np.clip(np.round(sample_x / in_scale + in_zero_point), -128, 127).astype(np.int8)
            else:
                q_x = sample_x.astype(np.int8)
            
            # Check input saturation (clipping at extreme bounds)
            if np.max(q_x) == 127 or np.min(q_x) == -128:
                if np.sum(q_x == 127) + np.sum(q_x == -128) > 30:
                    input_sat_count += 1
            interpreter.set_tensor(input_details["index"], q_x)
        else:
            interpreter.set_tensor(input_details["index"], sample_x.astype(np.float32))

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details["index"])[0]  # shape (3,)

        if out_dtype == np.int8:
            dequantized = (output_data.astype(np.float32) - out_zero_point) * out_scale
            pred_class = int(np.argmax(dequantized))
        else:
            pred_class = int(np.argmax(output_data))

        y_preds.append(pred_class)

    y_preds = np.array(y_preds, dtype=np.int64)
    metrics = compute_metrics(y_test, y_preds, class_map)

    total_samples = len(X_test)
    metrics["model_path"] = artifact_path(model_path)
    metrics["model_sha256"] = sha256_val
    metrics["model_type"] = "INT8 TFLite" if in_dtype == np.int8 else "Float TFLite"
    metrics["input_saturation_count"] = input_sat_count
    metrics["input_saturation_ratio"] = float(input_sat_count / total_samples) if total_samples > 0 else 0.0
    metrics["output_saturation_count"] = output_sat_count
    metrics["output_saturation_ratio"] = float(output_sat_count / total_samples) if total_samples > 0 else 0.0
    metrics["input_contract"] = {
        "shape": input_details["shape"].tolist(),
        "dtype": str(in_dtype.__name__),
        "scale": float(in_scale),
        "zero_point": int(in_zero_point)
    }
    metrics["output_contract"] = {
        "shape": output_details["shape"].tolist(),
        "dtype": str(out_dtype.__name__),
        "scale": float(out_scale),
        "zero_point": int(out_zero_point)
    }

    return metrics


def evaluate_keras_model(
    model_path: Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    preprocessor: MMWavePreprocessor,
    class_map: Dict[int, str]
) -> Dict[str, Any]:
    import tensorflow as tf

    sha256_val = calculate_sha256(model_path)
    model = tf.keras.models.load_model(str(model_path))

    X_prep = preprocessor.preprocess_batch(X_test)
    preds = model.predict(X_prep, verbose=0)
    y_preds = np.argmax(preds, axis=1).astype(np.int64)

    metrics = compute_metrics(y_test, y_preds, class_map)
    metrics["model_path"] = artifact_path(model_path)
    metrics["model_sha256"] = sha256_val
    metrics["model_type"] = "Float Keras"
    metrics["input_saturation_count"] = 0
    metrics["input_saturation_ratio"] = 0.0
    metrics["output_saturation_count"] = 0
    metrics["output_saturation_ratio"] = 0.0

    return metrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_map: Dict[int, str]
) -> Dict[str, Any]:
    num_classes = len(class_map)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            conf_matrix[t, p] += 1

    total_samples = int(len(y_true))
    correct = int(np.sum(y_true == y_pred))
    accuracy = float(correct / total_samples) if total_samples > 0 else 0.0

    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}

    for c in range(num_classes):
        c_name = class_map.get(c, str(c))
        tp = conf_matrix[c, c]
        fp = np.sum(conf_matrix[:, c]) - tp
        fn = np.sum(conf_matrix[c, :]) - tp

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class_precision[c_name] = prec
        per_class_recall[c_name] = rec
        per_class_f1[c_name] = f1

    macro_precision = float(np.mean(list(per_class_precision.values())))
    macro_recall = float(np.mean(list(per_class_recall.values())))
    macro_f1 = float(np.mean(list(per_class_f1.values())))

    apnea_recall = per_class_recall.get("APNEA", 0.0)
    apnea_window_miss_rate = float(1.0 - apnea_recall)

    pred_dist = {}
    for c in range(num_classes):
        c_name = class_map.get(c, str(c))
        pred_dist[c_name] = int(np.sum(y_pred == c))

    unique_preds = len(set(y_pred))
    class_collapse = bool(unique_preds <= 1)

    return {
        "total_samples": total_samples,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "confusion_matrix": conf_matrix.tolist(),
        "apnea_window_miss_rate": apnea_window_miss_rate,
        "false_alarm_per_hour": None,
        "false_alarm_status": "NOT_COMPUTABLE",
        "false_alarm_reason": "CONTINUOUS_SESSION_TIMELINE_MISSING",
        "prediction_distribution": pred_dist,
        "class_collapse": class_collapse
    }


def main():
    parser = argparse.ArgumentParser(description="SafeNest V6 mmWave Model Evaluator")
    parser.add_argument("--model", type=str, default="models/mmwave/mmwave_resp_int8_v0.1.0.tflite", help="Path to model file")
    parser.add_argument("--dataset", type=str, default="datasets/mmwave/processed/mmwave_respiration_v1.npz", help="Path to processed NPZ dataset")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test", help="Dataset split to evaluate")
    parser.add_argument("--is-legacy", action="store_true", help="Set for legacy v0.1.0 baseline evaluation")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSON result file")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    model_path = (project_root / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    dataset_path = (project_root / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset)

    if not dataset_path.exists() or not model_path.exists():
        print("❌ Error: Dataset or model file missing")
        sys.exit(1)

    data = np.load(dataset_path, allow_pickle=True)
    X_split = data[f"X_{args.split}"]
    y_split = data[f"y_{args.split}"]
    mean_val = float(data["mean"]) if "mean" in data else 0.006091983988881111
    std_val = float(data["std"]) if "std" in data else 2.5013835430145264
    class_map = {0: "NORMAL", 1: "RAPID_OR_ABNORMAL", 2: "APNEA"}

    # Use legacy preprocessor (no bandpass filter) for v0.1.0 baseline, experimental for V6 candidate
    apply_filter = not args.is_legacy
    preprocessor = MMWavePreprocessor(mean=mean_val, std=std_val, apply_filter=apply_filter)

    print(f"🔍 Evaluating model {model_path.name} on split '{args.split}' (filter={apply_filter})...")

    if model_path.suffix.lower() == ".tflite":
        results = evaluate_tflite_model(model_path, X_split, y_split, preprocessor, class_map)
    elif model_path.suffix.lower() in [".h5", ".keras"]:
        results = evaluate_keras_model(model_path, X_split, y_split, preprocessor, class_map)
    else:
        print(f"❌ Error: Unsupported model format {model_path.suffix}")
        sys.exit(1)

    results["dataset_path"] = artifact_path(dataset_path)
    results["dataset_split"] = args.split
    results["synthetic_data"] = True

    print("\n--- Evaluation Summary ---")
    print(f"Model Path:             {results['model_path']}")
    print(f"Model SHA256:           {results['model_sha256']}")
    print(f"Accuracy:               {results['accuracy']:.4f}")
    print(f"Macro F1:               {results['macro_f1']:.4f}")
    print(f"Apnea Window Miss Rate: {results['apnea_window_miss_rate']:.4f}")
    print(f"False Alarm Status:     {results['false_alarm_status']} ({results['false_alarm_reason']})")
    print(f"Class Collapse:         {results['class_collapse']}")
    print(f"Input Saturation Ratio: {results['input_saturation_ratio']:.4f}")
    print(f"Predictions:            {results['prediction_distribution']}")

    if args.output:
        out_path = (project_root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Saved results to {out_path}")


if __name__ == "__main__":
    main()
