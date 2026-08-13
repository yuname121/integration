# SafeNest mmWave Track — Phase M-B6 Stage-Equivalence Helpers

import hashlib
import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from mmwave_m_b1_preprocessing import fit_train_zscore_statistics, transform_signals
from mmwave_m_b2_imbalance import LABEL_NAMES, compute_one_vs_rest_false_positives, compute_subject_level_diagnostics
from mmwave_m_b3_architecture import build_model_by_id, compute_numerical_weights_sha256
from mmwave_phase_b_access import PhaseBAccessGuard

SHORTLIST_SEEDS = [42, 43, 44]


def convert_model_to_unoptimized_float32_tflite(model: tf.keras.Model) -> Tuple[bytes, Dict[str, Any]]:
    """Convert float Keras model to true unoptimized Float32 TFLite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Do NOT set optimizations (keep pure Float32)
    tflite_bytes = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()

    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]
    op_details = interpreter._get_ops_details()
    op_types = [op["op_name"] for op in op_details]

    metadata = {
        "bytes": len(tflite_bytes),
        "sha256": hashlib.sha256(tflite_bytes).hexdigest(),
        "input_dtype": str(in_details["dtype"].__name__),
        "output_dtype": str(out_details["dtype"].__name__),
        "input_shape": [int(x) for x in in_details["shape"]],
        "output_shape": [int(x) for x in out_details["shape"]],
        "op_types": op_types,
        "select_tf_ops_count": sum(1 for t in op_types if "Flex" in t or "Select" in t),
    }

    return tflite_bytes, metadata


def evaluate_tflite_float32_model(
    tflite_model_bytes: bytes,
    val_x_3d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute unoptimized Float32 TFLite model on VALIDATION set. Returns (preds_array, probs_array)."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interpreter.allocate_tensors()

    in_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    N = len(val_x_3d)
    float_preds = []
    float_probs = []

    for i in range(N):
        x_sample = val_x_3d[i : i + 1].astype(np.float32)
        interpreter.set_tensor(in_idx, x_sample)
        interpreter.invoke()
        y_prob = interpreter.get_tensor(out_idx)[0]
        pred_c = int(np.argmax(y_prob))

        float_preds.append(pred_c)
        float_probs.append(y_prob.tolist())

    return np.array(float_preds, dtype=int), np.array(float_probs, dtype=np.float32)


from mmwave_m_b5_calibration import evaluate_tflite_int8_model


def evaluate_tflite_int8_model_full(
    tflite_model_bytes: bytes,
    val_x_3d: np.ndarray,
    val_y: np.ndarray,
    float_probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Execute strict INT8 TFLite model on VALIDATION set using authoritative M-B5 evaluation logic."""
    if float_probs is None:
        float_probs = np.zeros((len(val_y), 3), dtype=np.float32)

    res = evaluate_tflite_int8_model(tflite_model_bytes, val_x_3d, val_y, float_probs)

    int8_preds_arr = np.array(res["int8_predictions"], dtype=int)
    dequant_probs_arr = np.array(res["dequantized_probabilities"], dtype=np.float32)

    return {
        "predictions": int8_preds_arr,
        "probabilities": dequant_probs_arr,
        "macro_f1": res["val_macro_f1"],
        "accuracy": res["val_accuracy"],
        "class_metrics": res["class_metrics"],
        "input_saturation_ratio": res["input_saturation_ratio"],
        "saturated_sample_count": res["saturated_sample_count"],
        "output_endpoint_ratio": res["output_endpoint_ratio"],
        "input_scale": res["input_scale"],
        "input_zero_point": res["input_zero_point"],
        "output_scale": res["output_scale"],
        "output_zero_point": res["output_zero_point"],
    }


def compute_pairwise_equivalence(
    source_preds: np.ndarray,
    source_probs: np.ndarray,
    target_preds: np.ndarray,
    target_probs: np.ndarray,
    val_y: np.ndarray,
    val_windows: List[Dict[str, Any]],
    source_stage: str,
    target_stage: str,
) -> Dict[str, Any]:
    """Compute exact top-1 agreement, probability MAE/RMSE/max, and class metric deltas between two stages."""
    N = len(val_y)
    top1_matches = int(np.sum(source_preds == target_preds))
    top1_agreement = float(top1_matches / N)

    prob_diff = target_probs - source_probs
    abs_diff = np.abs(prob_diff)

    output_probability_mae = float(np.mean(abs_diff))
    output_probability_rmse = float(np.sqrt(np.mean(prob_diff ** 2)))
    output_probability_max_abs_error = float(np.max(abs_diff))

    source_cm = compute_one_vs_rest_false_positives(val_y, source_preds)
    target_cm = compute_one_vs_rest_false_positives(val_y, target_preds)

    source_macro_f1 = float(np.mean([source_cm[c]["f1_score"] for c in LABEL_NAMES]))
    target_macro_f1 = float(np.mean([target_cm[c]["f1_score"] for c in LABEL_NAMES]))

    signed_macro_f1_delta = round(target_macro_f1 - source_macro_f1, 6)
    pos_macro_f1_deg = round(max(0.0, source_macro_f1 - target_macro_f1), 6)

    source_acc = float(np.mean(source_preds == val_y))
    target_acc = float(np.mean(target_preds == val_y))

    signed_acc_delta = round(target_acc - source_acc, 6)
    pos_acc_deg = round(max(0.0, source_acc - target_acc), 6)

    per_class_rec_deg = {}
    per_class_f1_deg = {}

    for cname in LABEL_NAMES:
        src_rec = source_cm[cname]["recall"]
        tgt_rec = target_cm[cname]["recall"]
        per_class_rec_deg[cname] = round(max(0.0, src_rec - tgt_rec), 6)

        src_f1 = source_cm[cname]["f1_score"]
        tgt_f1 = target_cm[cname]["f1_score"]
        per_class_f1_deg[cname] = round(max(0.0, src_f1 - tgt_f1), 6)

    max_pos_rec_deg = round(max(per_class_rec_deg.values()), 6)

    # Collect mismatch samples
    mismatch_samples = []
    for i in range(N):
        if source_preds[i] != target_preds[i] or np.max(abs_diff[i]) > 0.05:
            mismatch_samples.append({
                "validation_sample_index": i,
                "window_id": val_windows[i]["window_id"],
                "subject_id": val_windows[i]["subject_id"],
                "recording_id": val_windows[i]["recording_id"],
                "true_class": int(val_y[i]),
                "true_label": val_windows[i]["safenest_label"],
                "source_stage": source_stage,
                "target_stage": target_stage,
                "source_prediction": int(source_preds[i]),
                "target_prediction": int(target_preds[i]),
                "source_probs": source_probs[i].tolist(),
                "target_probs": target_probs[i].tolist(),
                "abs_error_mean": float(np.mean(abs_diff[i])),
                "abs_error_max": float(np.max(abs_diff[i])),
            })

    return {
        "source_stage": source_stage,
        "target_stage": target_stage,
        "top1_matches": top1_matches,
        "total_samples": N,
        "top1_agreement": round(top1_agreement, 6),
        "mismatch_count": len([m for m in mismatch_samples if m["source_prediction"] != m["target_prediction"]]),
        "output_probability_mae": round(output_probability_mae, 6),
        "output_probability_rmse": round(output_probability_rmse, 6),
        "output_probability_max_abs_error": round(output_probability_max_abs_error, 6),
        "source_macro_f1": round(source_macro_f1, 6),
        "target_macro_f1": round(target_macro_f1, 6),
        "signed_macro_f1_delta": signed_macro_f1_delta,
        "positive_macro_f1_degradation": pos_macro_f1_deg,
        "source_accuracy": round(source_acc, 6),
        "target_accuracy": round(target_acc, 6),
        "signed_accuracy_delta": signed_acc_delta,
        "positive_accuracy_degradation": pos_acc_deg,
        "per_class_positive_recall_degradation": per_class_rec_deg,
        "max_positive_recall_degradation": max_pos_rec_deg,
        "per_class_positive_f1_degradation": per_class_f1_deg,
        "mismatch_samples": mismatch_samples,
    }
