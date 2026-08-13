# SafeNest mmWave Track — Phase M-B5 Calibration Profiles & Strict INT8 Evaluator

import hashlib
import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from mmwave_m_b1_preprocessing import fit_train_zscore_statistics, transform_signals
from mmwave_m_b2_imbalance import LABEL_NAMES, compute_one_vs_rest_false_positives, compute_subject_level_diagnostics
from mmwave_m_b3_architecture import build_model_by_id, compute_numerical_weights_sha256
from mmwave_phase_b_access import PhaseBAccessGuard

CALIBRATION_SAMPLE_COUNT = 120
CALIBRATION_RNG_SEED = 20260810
SHORTLIST_SEEDS = [42, 43, 44]

PROFILE_IDS = [
    "M-B5_CAL_TRAIN_ORDER_120",
    "M-B5_CAL_RANDOM_PROPORTIONAL_120",
    "M-B5_CAL_CLASS_BALANCED_120",
    "M-B5_CAL_DISTRIBUTION_AWARE_120",
]


def build_profile_a_train_order(train_windows: List[Dict[str, Any]], sample_count: int = CALIBRATION_SAMPLE_COUNT) -> List[int]:
    """Profile A: First sample_count pure-class TRAIN rows in canonical TRAIN order."""
    if len(train_windows) < sample_count:
        raise ValueError(f"TRAIN dataset has fewer than {sample_count} samples: {len(train_windows)}")
    return list(range(sample_count))


def build_profile_b_random_proportional(
    train_windows: List[Dict[str, Any]],
    sample_count: int = CALIBRATION_SAMPLE_COUNT,
    seed: int = CALIBRATION_RNG_SEED,
) -> List[int]:
    """Profile B: Proportional class random sample without replacement using pinned RNG."""
    total = len(train_windows)
    label_ids = [w["safenest_label_id"] for w in train_windows]
    class_counts = {c: label_ids.count(idx) for idx, c in enumerate(LABEL_NAMES)}

    # Largest-remainder allocation
    raw_alloc = {c: sample_count * class_counts[c] / total for c in LABEL_NAMES}
    int_alloc = {c: int(np.floor(raw_alloc[c])) for c in LABEL_NAMES}
    remainders = {c: raw_alloc[c] - int_alloc[c] for c in LABEL_NAMES}
    rem_needed = sample_count - sum(int_alloc.values())
    sorted_classes = sorted(LABEL_NAMES, key=lambda c: remainders[c], reverse=True)
    for i in range(rem_needed):
        int_alloc[sorted_classes[i]] += 1

    rng = np.random.RandomState(seed)
    selected_indices = []

    for idx, cname in enumerate(LABEL_NAMES):
        c_count_needed = int_alloc[cname]
        c_eligible = [i for i, w in enumerate(train_windows) if w["safenest_label_id"] == idx]
        if len(c_eligible) < c_count_needed:
            raise ValueError(f"Class '{cname}' has fewer eligible samples ({len(c_eligible)}) than needed ({c_count_needed})")
        sampled = rng.choice(c_eligible, size=c_count_needed, replace=False)
        selected_indices.extend(sampled.tolist())

    return sorted(selected_indices)


def build_profile_c_class_balanced(
    train_windows: List[Dict[str, Any]],
    sample_count: int = CALIBRATION_SAMPLE_COUNT,
    seed: int = CALIBRATION_RNG_SEED,
) -> List[int]:
    """Profile C: Equal class balanced sample (40 per class for 120 samples) without replacement."""
    target_per_class = sample_count // len(LABEL_NAMES)
    rng = np.random.RandomState(seed)
    selected_indices = []

    for idx, cname in enumerate(LABEL_NAMES):
        c_eligible = [i for i, w in enumerate(train_windows) if w["safenest_label_id"] == idx]
        if len(c_eligible) < target_per_class:
            raise ValueError(f"Balanced profile infeasible: class '{cname}' has only {len(c_eligible)} samples, need {target_per_class}")
        sampled = rng.choice(c_eligible, size=target_per_class, replace=False)
        selected_indices.extend(sampled.tolist())

    return sorted(selected_indices)


UNKNOWN_OR_MISSING = "UNKNOWN_OR_MISSING"
PROFILE_D_CONTINUOUS_FEATURES = ["peak_abs", "RMS", "p01", "p99", "dynamic_range"]
PROFILE_D_TIE_POLICY = "LOWER_AUTHORITATIVE_CANONICAL_TRAIN_INDEX"
REQUIRED_SUPERVISED_CLASSES_FOR_COLLAPSE = ("RAPID_OR_ABNORMAL", "APNEA")


def _categorical_value(raw: Any) -> str:
    if raw is None:
        return UNKNOWN_OR_MISSING
    text = str(raw).strip()
    if not text:
        return UNKNOWN_OR_MISSING
    return text


def derive_train_categorical_vocabulary(
    train_windows: List[Dict[str, Any]],
    field_name: str,
) -> List[str]:
    """Derive sorted categorical vocabulary from actual TRAIN rows; preserve UNKNOWN_OR_MISSING when needed."""
    values = [_categorical_value(w.get(field_name)) for w in train_windows]
    present = sorted({v for v in values if v != UNKNOWN_OR_MISSING})
    if any(v == UNKNOWN_OR_MISSING for v in values):
        return present + [UNKNOWN_OR_MISSING]
    return present


def one_hot_encode(value: str, vocabulary: List[str]) -> List[float]:
    if value not in vocabulary:
        raise ValueError(f"Categorical value '{value}' not in vocabulary {vocabulary}")
    return [1.0 if value == token else 0.0 for token in vocabulary]


def build_profile_d_feature_matrix(
    train_windows: List[Dict[str, Any]],
    preprocessed_train_x: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build Profile-D feature matrix from TRAIN-only continuous stats + authoritative categorical metadata."""
    if len(train_windows) != len(preprocessed_train_x):
        raise ValueError("TRAIN windows and preprocessed tensors must align 1:1 for Profile D")

    posture_vocab = derive_train_categorical_vocabulary(train_windows, "posture")
    condition_vocab = derive_train_categorical_vocabulary(train_windows, "source_test_condition")

    continuous_rows = []
    categorical_rows = []
    for i, window in enumerate(train_windows):
        sig_flat = np.asarray(preprocessed_train_x[i], dtype=np.float64).reshape(-1)
        peak_abs = float(np.max(np.abs(sig_flat)))
        rms = float(np.sqrt(np.mean(sig_flat ** 2)))
        p01 = float(np.percentile(sig_flat, 1))
        p99 = float(np.percentile(sig_flat, 99))
        dyn_range = p99 - p01
        continuous_rows.append([peak_abs, rms, p01, p99, dyn_range])

        lbl_id = int(window["safenest_label_id"])
        class_onehot = [1.0 if lbl_id == idx else 0.0 for idx in range(len(LABEL_NAMES))]
        posture_onehot = one_hot_encode(_categorical_value(window.get("posture")), posture_vocab)
        condition_onehot = one_hot_encode(
            _categorical_value(window.get("source_test_condition")),
            condition_vocab,
        )
        categorical_rows.append(class_onehot + posture_onehot + condition_onehot)

    cont_feats = np.asarray(continuous_rows, dtype=np.float64)
    cat_feats = np.asarray(categorical_rows, dtype=np.float64)

    medians = np.median(cont_feats, axis=0)
    iqr = np.percentile(cont_feats, 75, axis=0) - np.percentile(cont_feats, 25, axis=0)
    iqr = np.where(iqr == 0.0, 1.0, iqr)
    norm_cont = (cont_feats - medians) / iqr
    norm_matrix = np.hstack([norm_cont, cat_feats])

    metadata = {
        "posture_vocabulary": posture_vocab,
        "source_test_condition_vocabulary": condition_vocab,
        "unknown_or_missing_token": UNKNOWN_OR_MISSING,
        "continuous_features": list(PROFILE_D_CONTINUOUS_FEATURES),
        "class_onehot_order": list(LABEL_NAMES),
        "snr_available": False,
        "snr_source": "NOT_AVAILABLE",
        "robust_scaling": {
            "statistic": "median_iqr",
            "fit_population": "TRAIN_ONLY",
            "continuous_medians": [float(x) for x in medians],
            "continuous_iqr": [float(x) for x in iqr],
        },
        "tie_policy": PROFILE_D_TIE_POLICY,
        "feature_definition": (
            "robust_scaled(peak_abs,RMS,p01,p99,dynamic_range) + "
            "class_onehot + posture_onehot(source TRAIN posture) + "
            "source_test_condition_onehot"
        ),
    }
    return norm_matrix, metadata


def build_profile_d_distribution_aware(
    train_windows: List[Dict[str, Any]],
    preprocessed_train_x: np.ndarray,
    sample_count: int = CALIBRATION_SAMPLE_COUNT,
) -> Tuple[List[int], Dict[str, Any]]:
    """Profile D: Deterministic farthest-point coverage with authoritative metadata and subject-cap policy."""
    N = len(train_windows)
    if N < sample_count:
        raise ValueError(f"TRAIN dataset has fewer than {sample_count} samples: {N}")

    norm_matrix, metadata = build_profile_d_feature_matrix(train_windows, preprocessed_train_x)
    center = np.median(norm_matrix, axis=0)
    distances_from_center = np.linalg.norm(norm_matrix - center, axis=1)

    # First sample: maximum distance from global TRAIN feature center; exact ties → lower index.
    max_dist = float(np.max(distances_from_center))
    first_candidates = [i for i in range(N) if abs(float(distances_from_center[i]) - max_dist) <= 1e-12]
    first_idx = int(min(first_candidates))

    selected = [first_idx]
    selected_set = {first_idx}
    subject_counts = {w["subject_id"]: 0 for w in train_windows}
    subject_counts[train_windows[first_idx]["subject_id"]] += 1

    subject_cap: float = 2.0
    subject_cap_state = "MAX_2"
    subject_cap_relaxations: List[str] = []

    while len(selected) < sample_count:
        cand_indices = [i for i in range(N) if i not in selected_set]
        if not cand_indices:
            raise ValueError(f"Unable to select {sample_count} Profile-D samples; exhausted TRAIN pool")

        if subject_cap == float("inf"):
            eligible_cands = cand_indices
        else:
            eligible_cands = [
                i for i in cand_indices if subject_counts[train_windows[i]["subject_id"]] < subject_cap
            ]

        if not eligible_cands:
            if subject_cap == 2.0:
                subject_cap = 3.0
                subject_cap_state = "RELAXED_MAX_3"
                subject_cap_relaxations.append("MAX_2_TO_MAX_3")
                continue
            if subject_cap == 3.0:
                subject_cap = float("inf")
                subject_cap_state = "CAP_REMOVED_FOR_REMAINING_SLOTS"
                subject_cap_relaxations.append("MAX_3_TO_UNCAPPED")
                continue
            raise ValueError("Profile-D selection stalled with no eligible candidates under uncapped policy")

        selected_feats = norm_matrix[selected]
        best_cand = None
        max_min_dist = -1.0

        for cand_i in eligible_cands:
            min_d = float(np.min(np.linalg.norm(selected_feats - norm_matrix[cand_i], axis=1)))
            if best_cand is None or min_d > max_min_dist + 1e-12:
                max_min_dist = min_d
                best_cand = cand_i
            elif abs(min_d - max_min_dist) <= 1e-12 and cand_i < best_cand:
                best_cand = cand_i

        assert best_cand is not None
        selected.append(best_cand)
        selected_set.add(best_cand)
        subject_counts[train_windows[best_cand]["subject_id"]] += 1

    selected_sorted = sorted(selected)
    selected_subjects = [train_windows[i]["subject_id"] for i in selected_sorted]
    selected_labels = [train_windows[i]["safenest_label"] for i in selected_sorted]
    metadata = {
        **metadata,
        "algorithm": "deterministic_farthest_point",
        "sample_count": sample_count,
        "subject_cap_initial": 2,
        "subject_cap_final_state": subject_cap_state,
        "subject_cap_relaxations": subject_cap_relaxations,
        "selected_subject_count": len(set(selected_subjects)),
        "selected_class_distribution": {
            cname: int(selected_labels.count(cname)) for cname in LABEL_NAMES
        },
        "max_selected_per_subject": int(max(selected_subjects.count(s) for s in set(selected_subjects))),
        "rng_used": False,
    }
    return selected_sorted, metadata


def build_all_calibration_profiles(
    train_windows: List[Dict[str, Any]],
    preprocessed_train_x: np.ndarray,
    sample_count: int = CALIBRATION_SAMPLE_COUNT,
) -> Dict[str, List[int]]:
    """Build all 4 preregistered M-B5 calibration profiles."""
    prof_a = build_profile_a_train_order(train_windows, sample_count=sample_count)
    prof_b = build_profile_b_random_proportional(train_windows, sample_count=sample_count)
    prof_c = build_profile_c_class_balanced(train_windows, sample_count=sample_count)
    prof_d, _profile_d_meta = build_profile_d_distribution_aware(
        train_windows, preprocessed_train_x, sample_count=sample_count
    )

    return {
        "M-B5_CAL_TRAIN_ORDER_120": prof_a,
        "M-B5_CAL_RANDOM_PROPORTIONAL_120": prof_b,
        "M-B5_CAL_CLASS_BALANCED_120": prof_c,
        "M-B5_CAL_DISTRIBUTION_AWARE_120": prof_d,
    }


def build_all_calibration_profiles_with_metadata(
    train_windows: List[Dict[str, Any]],
    preprocessed_train_x: np.ndarray,
    sample_count: int = CALIBRATION_SAMPLE_COUNT,
) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
    """Build all profiles and return Profile-D construction metadata for the contract artifact."""
    prof_a = build_profile_a_train_order(train_windows, sample_count=sample_count)
    prof_b = build_profile_b_random_proportional(train_windows, sample_count=sample_count)
    prof_c = build_profile_c_class_balanced(train_windows, sample_count=sample_count)
    prof_d, profile_d_meta = build_profile_d_distribution_aware(
        train_windows, preprocessed_train_x, sample_count=sample_count
    )
    return {
        "M-B5_CAL_TRAIN_ORDER_120": prof_a,
        "M-B5_CAL_RANDOM_PROPORTIONAL_120": prof_b,
        "M-B5_CAL_CLASS_BALANCED_120": prof_c,
        "M-B5_CAL_DISTRIBUTION_AWARE_120": prof_d,
    }, profile_d_meta


def compute_positive_recall_degradation(
    float_class_metrics: Dict[str, Dict[str, Any]],
    int8_class_metrics: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, float], float]:
    """Independently compute per-class positive recall degradation and its maximum."""
    per_class = {}
    for cname in LABEL_NAMES:
        fl_rec = float(float_class_metrics[cname]["recall"])
        int8_rec = float(int8_class_metrics[cname]["recall"])
        per_class[cname] = round(max(0.0, fl_rec - int8_rec), 6)
    return per_class, round(max(per_class.values()), 6)


def required_class_collapse_state(
    preds: np.ndarray,
    class_metrics: Dict[str, Dict[str, Any]],
) -> bool:
    """True when a required supervised class is collapsed in the prediction state."""
    unique_count = int(len(np.unique(preds)))
    for cname in REQUIRED_SUPERVISED_CLASSES_FOR_COLLAPSE:
        if float(class_metrics[cname]["recall"]) == 0.0:
            return True
    return unique_count < len(LABEL_NAMES)


def detect_new_quantization_collapse(
    float_preds: np.ndarray,
    float_class_metrics: Dict[str, Dict[str, Any]],
    int8_preds: np.ndarray,
    int8_class_metrics: Dict[str, Dict[str, Any]],
) -> bool:
    """New collapse only if Float is not collapsed on required classes but INT8 is."""
    float_collapsed = required_class_collapse_state(float_preds, float_class_metrics)
    int8_collapsed = required_class_collapse_state(int8_preds, int8_class_metrics)
    return (not float_collapsed) and int8_collapsed


def inspect_tflite_model_bytes(tflite_model_bytes: bytes) -> Dict[str, Any]:
    """Independently inspect an on-disk/in-memory TFLite artifact for dtype/ops/quantization."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]
    op_details = interpreter._get_ops_details()
    op_types = [op["op_name"] for op in op_details]
    return {
        "bytes": len(tflite_model_bytes),
        "sha256": hashlib.sha256(tflite_model_bytes).hexdigest(),
        "input_dtype": str(in_details["dtype"].__name__),
        "output_dtype": str(out_details["dtype"].__name__),
        "input_shape": [int(x) for x in in_details["shape"]],
        "output_shape": [int(x) for x in out_details["shape"]],
        "input_scale": float(in_details["quantization"][0]),
        "input_zero_point": int(in_details["quantization"][1]),
        "output_scale": float(out_details["quantization"][0]),
        "output_zero_point": int(out_details["quantization"][1]),
        "op_types": op_types,
        "select_tf_ops_count": sum(1 for t in op_types if "Flex" in t or "Select" in t),
    }


def compute_tensor_statistics(data_x: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive tensor statistics over a preprocessed float dataset."""
    flat = data_x.flatten()
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p01": float(np.percentile(flat, 1)),
        "p05": float(np.percentile(flat, 5)),
        "p25": float(np.percentile(flat, 25)),
        "p50": float(np.percentile(flat, 50)),
        "p75": float(np.percentile(flat, 75)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "peak_abs": float(np.max(np.abs(flat))),
        "rms": float(np.sqrt(np.mean(flat ** 2))),
    }


def convert_model_to_strict_int8_tflite(
    model: tf.keras.Model,
    calib_x_float32: np.ndarray,
) -> Tuple[bytes, Dict[str, Any]]:
    """Convert float Keras model to strict INT8 TFLite model using specified representative float32 samples."""
    def representative_dataset_gen():
        for i in range(len(calib_x_float32)):
            sample = calib_x_float32[i : i + 1]  # shape (1, 250, 1)
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model_bytes = converter.convert()
    metadata = inspect_tflite_model_bytes(tflite_model_bytes)
    return tflite_model_bytes, metadata


def evaluate_tflite_int8_model(
    tflite_model_bytes: bytes,
    val_x_float32: np.ndarray,
    val_y: np.ndarray,
    float_probs: np.ndarray,
) -> Dict[str, Any]:
    """Execute strict INT8 TFLite model on VALIDATION set and compute quantization metrics."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interpreter.allocate_tensors()

    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]

    in_idx = in_details["index"]
    out_idx = out_details["index"]

    in_scale = float(in_details["quantization"][0])
    in_zp = int(in_details["quantization"][1])
    out_scale = float(out_details["quantization"][0])
    out_zp = int(out_details["quantization"][1])

    N = len(val_x_float32)
    int8_preds = []
    dequant_probs = []

    total_input_elements = 0
    saturated_input_elements = 0
    saturated_sample_count = 0

    endpoint_occupancy_count = 0
    total_output_elements = 0

    mismatch_samples = []

    for i in range(N):
        x_sample = val_x_float32[i : i + 1]  # shape (1, 250, 1)

        # Pre-clamp quantization math for input saturation calculation
        q_raw = np.round(x_sample / in_scale + in_zp)
        sat_mask = (q_raw < -128) | (q_raw > 127)
        sat_cnt = int(np.sum(sat_mask))

        total_input_elements += q_raw.size
        saturated_input_elements += sat_cnt
        if sat_cnt > 0:
            saturated_sample_count += 1

        x_int8 = np.clip(q_raw, -128, 127).astype(np.int8)

        interpreter.set_tensor(in_idx, x_int8)
        interpreter.invoke()
        y_int8 = interpreter.get_tensor(out_idx)  # shape (1, 3)

        # Output endpoint occupancy (-128 or 127)
        endpoint_mask = (y_int8 == -128) | (y_int8 == 127)
        endpoint_occupancy_count += int(np.sum(endpoint_mask))
        total_output_elements += y_int8.size

        # Dequantize output probabilities
        y_dequant = (y_int8.astype(np.float32) - out_zp) * out_scale
        pred_class = int(np.argmax(y_dequant, axis=1)[0])

        int8_preds.append(pred_class)
        dequant_probs.append(y_dequant[0].tolist())

        float_pred_class = int(np.argmax(float_probs[i]))
        abs_err = np.mean(np.abs(y_dequant[0] - float_probs[i]))

        if pred_class != float_pred_class or sat_cnt > 0 or abs_err > 0.05:
            mismatch_samples.append({
                "validation_sample_index": i,
                "float_pred_class": float_pred_class,
                "int8_pred_class": pred_class,
                "true_class": int(val_y[i]),
                "float_probs": float_probs[i].tolist(),
                "dequant_probs": y_dequant[0].tolist(),
                "abs_output_error": float(abs_err),
                "input_saturation_count": sat_cnt,
            })

    int8_preds_arr = np.array(int8_preds, dtype=int)
    dequant_probs_arr = np.array(dequant_probs, dtype=np.float32)

    # One-vs-rest confusion & metrics
    cm = compute_one_vs_rest_false_positives(val_y, int8_preds_arr)
    macro_f1 = float(np.mean([cm[c]["f1_score"] for c in LABEL_NAMES]))
    accuracy = float(np.mean(int8_preds_arr == val_y))
    min_rec = float(min(cm[c]["recall"] for c in LABEL_NAMES))
    apnea_rec = cm["APNEA"]["recall"]
    rapid_rec = cm["RAPID_OR_ABNORMAL"]["recall"]

    collapsed = (apnea_rec == 0.0) or (rapid_rec == 0.0) or (len(np.unique(int8_preds_arr)) < 3)

    pred_dist = {c: int(np.sum(int8_preds_arr == idx)) for idx, c in enumerate(LABEL_NAMES)}

    # Quantization error metrics vs Float baseline
    float_preds_arr = np.argmax(float_probs, axis=1)
    top1_agreement = float(np.mean(int8_preds_arr == float_preds_arr))
    dequantized_output_mae = float(np.mean(np.abs(dequant_probs_arr - float_probs)))
    dequantized_output_max_err = float(np.max(np.abs(dequant_probs_arr - float_probs)))

    input_saturation_ratio = float(saturated_input_elements / total_input_elements) if total_input_elements > 0 else 0.0
    output_endpoint_ratio = float(endpoint_occupancy_count / total_output_elements) if total_output_elements > 0 else 0.0

    dequant_min = float(np.min(dequant_probs_arr)) if dequant_probs_arr.size else 0.0
    dequant_max = float(np.max(dequant_probs_arr)) if dequant_probs_arr.size else 0.0

    return {
        "val_macro_f1": round(macro_f1, 6),
        "val_accuracy": round(accuracy, 6),
        "min_per_class_recall": round(min_rec, 6),
        "apnea_recall": round(apnea_rec, 6),
        "rapid_recall": round(rapid_rec, 6),
        "collapsed": collapsed,
        "prediction_distribution": pred_dist,
        "class_metrics": cm,
        "int8_predictions": int8_preds_arr,
        "dequantized_probabilities": dequant_probs_arr,
        "top1_agreement": round(top1_agreement, 6),
        "dequantized_output_mae": round(dequantized_output_mae, 6),
        "dequantized_output_max_err": round(dequantized_output_max_err, 6),
        "dequantized_output_min": round(dequant_min, 6),
        "dequantized_output_max": round(dequant_max, 6),
        "input_saturation_ratio": round(input_saturation_ratio, 6),
        "saturated_input_elements": saturated_input_elements,
        "saturated_sample_count": saturated_sample_count,
        "output_endpoint_ratio": round(output_endpoint_ratio, 6),
        "mismatch_samples": mismatch_samples,
        "input_scale": in_scale,
        "input_zero_point": in_zp,
        "output_scale": out_scale,
        "output_zero_point": out_zp,
    }


def rank_cross_seed_calibration_profiles(
    cross_seed_results: List[Dict[str, Any]],
    eps: float = 1e-5,
) -> List[Dict[str, Any]]:
    """Rank calibration profiles with epsilon-aware 8-criterion preregistered ordering."""
    import functools

    eligible_profiles = [p for p in cross_seed_results if p["eligible"]]
    policy_order = {
        "M-B5_CAL_TRAIN_ORDER_120": 1,
        "M-B5_CAL_RANDOM_PROPORTIONAL_120": 2,
        "M-B5_CAL_CLASS_BALANCED_120": 3,
        "M-B5_CAL_DISTRIBUTION_AWARE_120": 4,
    }

    def compare_pair(a: Dict[str, Any], b: Dict[str, Any]) -> int:
        if a["profile_id"] == b["profile_id"]:
            return 0

        # Criterion 1: LOWER worst positive Macro-F1 degradation
        d1 = a["worst_positive_macro_f1_degradation"] - b["worst_positive_macro_f1_degradation"]
        if abs(d1) > eps:
            return -1 if d1 < 0 else 1

        # Criterion 2: LOWER worst positive per-class recall degradation
        d2 = a["worst_positive_recall_degradation"] - b["worst_positive_recall_degradation"]
        if abs(d2) > eps:
            return -1 if d2 < 0 else 1

        # Criterion 3: HIGHER minimum Top-1 agreement
        d3 = a["min_top1_agreement"] - b["min_top1_agreement"]
        if abs(d3) > eps:
            return -1 if d3 > 0 else 1

        # Criterion 4: LOWER maximum output probability MAE
        d4 = a["max_dequantized_output_mae"] - b["max_dequantized_output_mae"]
        if abs(d4) > eps:
            return -1 if d4 < 0 else 1

        # Criterion 5: LOWER maximum input saturation ratio
        d5 = a["max_input_saturation_ratio"] - b["max_input_saturation_ratio"]
        if abs(d5) > eps:
            return -1 if d5 < 0 else 1

        # Criterion 6: LOWER maximum output endpoint ratio
        d6 = a["max_output_endpoint_ratio"] - b["max_output_endpoint_ratio"]
        if abs(d6) > eps:
            return -1 if d6 < 0 else 1

        # Criterion 7: simpler calibration policy
        p7 = policy_order.get(a["profile_id"], 99) - policy_order.get(b["profile_id"], 99)
        if p7 != 0:
            return -1 if p7 < 0 else 1

        # Criterion 8: lexicographic profile ID
        if a["profile_id"] < b["profile_id"]:
            return -1
        if a["profile_id"] > b["profile_id"]:
            return 1
        return 0

    return sorted(eligible_profiles, key=functools.cmp_to_key(compare_pair))


def explain_ranking_decision(
    ranked_profiles: List[Dict[str, Any]],
    eps: float = 1e-5,
) -> Dict[str, Any]:
    """Explain which ranking criterion decided the winner under epsilon-aware ties."""
    if not ranked_profiles:
        return {
            "winner_profile_id": None,
            "deciding_criterion": "NO_ELIGIBLE_PROFILE",
            "tie_tolerance_eps": eps,
        }

    winner = ranked_profiles[0]
    if len(ranked_profiles) == 1:
        return {
            "winner_profile_id": winner["profile_id"],
            "deciding_criterion": "SOLE_ELIGIBLE_PROFILE",
            "tie_tolerance_eps": eps,
        }

    runner = ranked_profiles[1]
    checks = [
        ("worst_positive_macro_f1_degradation", "lower", "CRITERION_1_WORST_POSITIVE_MACRO_F1_DEGRADATION"),
        ("worst_positive_recall_degradation", "lower", "CRITERION_2_WORST_POSITIVE_RECALL_DEGRADATION"),
        ("min_top1_agreement", "higher", "CRITERION_3_MIN_TOP1_AGREEMENT"),
        ("max_dequantized_output_mae", "lower", "CRITERION_4_MAX_OUTPUT_PROBABILITY_MAE"),
        ("max_input_saturation_ratio", "lower", "CRITERION_5_MAX_INPUT_SATURATION_RATIO"),
        ("max_output_endpoint_ratio", "lower", "CRITERION_6_MAX_OUTPUT_ENDPOINT_RATIO"),
    ]
    for field, direction, label in checks:
        delta = float(winner[field]) - float(runner[field])
        if abs(delta) > eps:
            return {
                "winner_profile_id": winner["profile_id"],
                "runner_up_profile_id": runner["profile_id"],
                "deciding_criterion": label,
                "direction": direction,
                "winner_value": winner[field],
                "runner_up_value": runner[field],
                "abs_delta": abs(delta),
                "tie_tolerance_eps": eps,
            }

    policy_order = {
        "M-B5_CAL_TRAIN_ORDER_120": 1,
        "M-B5_CAL_RANDOM_PROPORTIONAL_120": 2,
        "M-B5_CAL_CLASS_BALANCED_120": 3,
        "M-B5_CAL_DISTRIBUTION_AWARE_120": 4,
    }
    if policy_order.get(winner["profile_id"], 99) != policy_order.get(runner["profile_id"], 99):
        return {
            "winner_profile_id": winner["profile_id"],
            "runner_up_profile_id": runner["profile_id"],
            "deciding_criterion": "CRITERION_7_SIMPLER_CALIBRATION_POLICY",
            "tie_tolerance_eps": eps,
        }
    return {
        "winner_profile_id": winner["profile_id"],
        "runner_up_profile_id": runner["profile_id"],
        "deciding_criterion": "CRITERION_8_LEXICOGRAPHIC_PROFILE_ID",
        "tie_tolerance_eps": eps,
    }
