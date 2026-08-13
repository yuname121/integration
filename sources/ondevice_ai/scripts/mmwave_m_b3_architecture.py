# SafeNest mmWave Track — Phase M-B3 TinyML Architecture Comparison & Screening

import gc
import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent

LABEL_NAMES = ["NORMAL", "RAPID_OR_ABNORMAL", "APNEA"]

ARCHITECTURES = [
    {
        "architecture_id": "M-B3_CONV1D_GAP_BASELINE",
        "name": "Conv1D + GAP Baseline",
        "family": "Conv1D_GAP",
        "topology": "Input(300,1)->Conv1D(16,k7,s2)->BN->MaxPool(2)->Conv1D(32,k5,s2)->BN->MaxPool(2)->Conv1D(64,k3,s1)->GAP->Dropout(0.3)->Dense(3)",
    },
    {
        "architecture_id": "M-B3_SEPARABLECONV1D_GAP",
        "name": "SeparableConv1D + GAP",
        "family": "SeparableConv1D_GAP",
        "topology": "Input(300,1)->SepConv1D(16,k7,s2,d1)->BN->MaxPool(2)->SepConv1D(32,k5,s2,d1)->BN->MaxPool(2)->SepConv1D(64,k3,s1,d1)->GAP->Dropout(0.3)->Dense(3)",
    },
    {
        "architecture_id": "M-B3_CONV1D_BILSTM",
        "name": "Conv1D + BiLSTM",
        "family": "Conv1D_BiLSTM",
        "topology": "Input(300,1)->Conv1D(16,k7,s2)->BN->MaxPool(2)->Conv1D(32,k5,s2)->BN->MaxPool(2)->BiLSTM(32)->Dropout(0.3)->Dense(3)",
    },
]


def reset_seeds(seed: int = 42) -> None:
    """Reset all random seeds and clear Keras session for deterministic execution."""
    tf.keras.backend.clear_session()
    gc.collect()
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_architecture_a(input_shape: Tuple[int, int] = (300, 1)) -> tf.keras.Model:
    """Architecture A: Conv1D + GAP Baseline (Exact M-B1/M-B2 Probe Architecture)."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(16, kernel_size=7, strides=2, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(32, kernel_size=5, strides=2, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(3, activation="softmax"),
        ],
        name="M-B3_CONV1D_GAP_BASELINE",
    )
    return model


def build_architecture_b(input_shape: Tuple[int, int] = (300, 1)) -> tf.keras.Model:
    """Architecture B: SeparableConv1D + GAP (Depthwise Separable 1D Conv)."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.SeparableConv1D(16, kernel_size=7, strides=2, padding="same", depth_multiplier=1, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.SeparableConv1D(32, kernel_size=5, strides=2, padding="same", depth_multiplier=1, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.SeparableConv1D(64, kernel_size=3, strides=1, padding="same", depth_multiplier=1, activation="relu"),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(3, activation="softmax"),
        ],
        name="M-B3_SEPARABLECONV1D_GAP",
    )
    return model


def build_architecture_c(input_shape: Tuple[int, int] = (300, 1)) -> tf.keras.Model:
    """Architecture C: Conv1D + BiLSTM (Recurrent Hybrid Architecture)."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(16, kernel_size=7, strides=2, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(32, kernel_size=5, strides=2, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(3, activation="softmax"),
        ],
        name="M-B3_CONV1D_BILSTM",
    )
    return model


def build_model_by_id(arch_id: str, input_shape: Tuple[int, int] = (300, 1)) -> tf.keras.Model:
    """Factory function to build architecture model by ID."""
    if arch_id == "M-B3_CONV1D_GAP_BASELINE":
        return build_architecture_a(input_shape)
    elif arch_id == "M-B3_SEPARABLECONV1D_GAP":
        return build_architecture_b(input_shape)
    elif arch_id == "M-B3_CONV1D_BILSTM":
        return build_architecture_c(input_shape)
    else:
        raise ValueError(f"Unknown architecture ID: {arch_id}")


def compute_numerical_weights_sha256(model: tf.keras.Model) -> str:
    """Compute deterministic SHA-256 digest across all numerical float32 weight arrays."""
    import hashlib

    hasher = hashlib.sha256()
    for w in model.get_weights():
        hasher.update(np.ascontiguousarray(w, dtype=np.float32).tobytes())
    return hasher.hexdigest()


def train_architecture(
    arch_id: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    seed: int = 42,
    batch_size: int = 32,
    epochs: int = 25,
    learning_rate: float = 0.001,
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """Train a single architecture model under frozen training contract and return (model, metrics_dict)."""
    reset_seeds(seed)

    model = build_model_by_id(arch_id, input_shape=(train_x.shape[1], train_x.shape[2]))
    initial_weights_sha = compute_numerical_weights_sha256(model)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=0,
        )
    ]

    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    final_weights_sha = compute_numerical_weights_sha256(model)
    stopped_epoch = int(callbacks[0].stopped_epoch) if callbacks[0].stopped_epoch > 0 else len(history.history["loss"])

    param_counts = {
        "total_params": int(model.count_params()),
        "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        "non_trainable_params": int(sum(np.prod(w.shape) for w in model.non_trainable_weights)),
    }

    info = {
        "architecture_id": arch_id,
        "initial_weights_sha256": initial_weights_sha,
        "final_weights_sha256": final_weights_sha,
        "stopped_epoch": stopped_epoch,
        "param_counts": param_counts,
        "history": {k: [round(float(v), 6) for v in vals] for k, vals in history.history.items()},
    }

    return model, info


def convert_to_tflite_float(model: tf.keras.Model) -> Tuple[bytes, bool]:
    """Convert Keras model to unoptimized Float32 TFLite binary. Returns (tflite_bytes, select_tf_ops_required)."""
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        return converter.convert(), False
    except Exception:
        # Fallback allowing Select TF Ops for architectures like BiLSTM
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        return converter.convert(), True


def convert_to_tflite_strict_int8(
    model: tf.keras.Model, rep_dataset_gen: Callable[[], Any]
) -> Tuple[bool, Optional[bytes], str, Optional[str]]:
    """Attempt strict full-INT8 TFLite conversion (TFLITE_BUILTINS_INT8 only).

    Returns (success, tflite_bytes_or_none, status_code, error_message_or_none).
    """
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_bytes = converter.convert()

        # Sanity check TFLite interpreter initialization & tensor dtypes
        interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
        interpreter.allocate_tensors()

        in_details = interpreter.get_input_details()
        out_details = interpreter.get_output_details()

        in_dtype = in_details[0]["dtype"]
        out_dtype = out_details[0]["dtype"]

        if in_dtype != np.int8 or out_dtype != np.int8:
            return False, None, "STRICT_INT8_DTYPE_MISMATCH", f"Input dtype={in_dtype}, output dtype={out_dtype}"

        return True, tflite_bytes, "FULL_INT8_SUPPORTED", None
    except Exception as e:
        err_msg = str(e)
        return False, None, "STRICT_INT8_UNSUPPORTED", err_msg


def convert_to_tflite_select_tf_ops(
    model: tf.keras.Model
) -> Tuple[bool, Optional[bytes], str, Optional[str]]:
    """Fallback diagnostic conversion allowing Select TF Ops (Flex Ops)."""
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]

        tflite_bytes = converter.convert()
        return True, tflite_bytes, "SELECT_TF_OPS_REQUIRED", None
    except Exception as e:
        return False, None, "CONVERSION_FAILED", str(e)


def evaluate_tflite_model(
    tflite_bytes: bytes, val_x: np.ndarray, is_int8: bool = False
) -> Optional[np.ndarray]:
    """Invoke TFLite model over input samples and return predicted class IDs (or None if Flex delegate is required)."""
    try:
        interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
        interpreter.allocate_tensors()
    except Exception:
        # Standard TFLite interpreter cannot allocate tensors for Select TF Ops models without Flex delegate
        return None

    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    in_index = in_details[0]["index"]
    out_index = out_details[0]["index"]

    in_scale, in_zero_point = in_details[0].get("quantization", (0.0, 0))
    out_scale, out_zero_point = out_details[0].get("quantization", (0.0, 0))

    preds = []
    for i in range(len(val_x)):
        sample = val_x[i : i + 1]  # shape (1, 300, 1)

        if is_int8:
            if in_scale != 0:
                q_sample = np.round(sample / in_scale + in_zero_point).astype(np.int8)
            else:
                q_sample = sample.astype(np.int8)
            interpreter.set_tensor(in_index, q_sample)
        else:
            interpreter.set_tensor(in_index, sample.astype(np.float32))

        interpreter.invoke()
        output_tensor = interpreter.get_tensor(out_index)

        if is_int8 and out_scale != 0:
            output_tensor = (output_tensor.astype(np.float32) - out_zero_point) * out_scale

        pred_class = int(np.argmax(output_tensor[0]))
        preds.append(pred_class)

    return np.array(preds, dtype=int)


def rank_architectures(
    results: List[Dict[str, Any]], eps: float = 1e-5
) -> List[Dict[str, Any]]:
    """Rank architectures under pre-registered deployment eligibility and ranking rules."""
    # Filter for deployment eligible architectures
    eligible = [r for r in results if r.get("deployment_eligibility") == "DEPLOYMENT_ELIGIBLE_SINGLE_SEED"]

    def compare_pair(a: Dict[str, Any], b: Dict[str, Any]) -> int:
        if a["architecture_id"] == b["architecture_id"]:
            return 0

        # Step 2: Float Macro F1
        f1_diff = a["float_macro_f1"] - b["float_macro_f1"]
        if abs(f1_diff) > eps:
            return 1 if f1_diff > 0 else -1

        # Step 3: Larger min per-class recall
        rec_diff = a["float_min_per_class_recall"] - b["float_min_per_class_recall"]
        if abs(rec_diff) > eps:
            return 1 if rec_diff > 0 else -1

        # Step 4: Higher APNEA proxy recall
        apnea_diff = a["float_apnea_recall"] - b["float_apnea_recall"]
        if abs(apnea_diff) > eps:
            return 1 if apnea_diff > 0 else -1

        # Step 5: Lower total parameter count
        param_diff = b["total_params"] - a["total_params"]
        if param_diff != 0:
            return 1 if param_diff > 0 else -1

        # Step 6: Smaller strict INT8 byte size
        size_a = a.get("strict_int8_bytes") or 999999999
        size_b = b.get("strict_int8_bytes") or 999999999
        size_diff = size_b - size_a
        if size_diff != 0:
            return 1 if size_diff > 0 else -1

        # Step 7: Lexicographic architecture ID
        return 1 if a["architecture_id"] < b["architecture_id"] else -1

    import functools

    eligible.sort(key=functools.cmp_to_key(compare_pair), reverse=True)
    return eligible
