#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/train_mmwave.py
SafeNest V6 mmWave Model Trainer & Quantizer

Train a lightweight 1D-CNN respiration classifier, convert to Float TFLite
and INT8 TFLite using train-split representative dataset, run 3-stage evaluation,
and export candidate artifacts.
"""

from __future__ import annotations
import os
import sys
import json
import hashlib
import random
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Generator

import numpy as np
import tensorflow as tf

# Ensure the canonical repository root is in python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preprocessing.mmwave import MMWavePreprocessor
from scripts.evaluate_mmwave import compute_metrics, evaluate_tflite_model, calculate_sha256


def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        tf.random.set_seed(seed)


def build_lightweight_model(input_shape=(300, 1), num_classes=3) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="resp_phase_input")
    
    x = tf.keras.layers.Conv1D(16, kernel_size=7, strides=2, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    x = tf.keras.layers.Conv1D(32, kernel_size=5, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    x = tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_output")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SafeNest_mmWave_Respiration_Net")
    return model


def train_float_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    preprocessor: MMWavePreprocessor,
    epochs: int = 25,
    batch_size: int = 32
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    X_tr_prep = preprocessor.preprocess_batch(X_train)
    X_va_prep = preprocessor.preprocess_batch(X_val)

    class_counts = np.bincount(y_train, minlength=3)
    total_samples = len(y_train)
    class_weights = {}
    for c in range(3):
        cnt = class_counts[c]
        class_weights[c] = float(total_samples / (3.0 * max(cnt, 1)))

    print(f"  Class counts (train): {class_counts.tolist()}")
    print(f"  Computed class weights: {class_weights}")

    model = build_lightweight_model(input_shape=(300, 1), num_classes=3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
    ]

    history = model.fit(
        X_tr_prep,
        y_train,
        validation_data=(X_va_prep, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    history_dict = {
        "loss": [float(v) for v in history.history.get("loss", [])],
        "accuracy": [float(v) for v in history.history.get("accuracy", [])],
        "val_loss": [float(v) for v in history.history.get("val_loss", [])],
        "val_accuracy": [float(v) for v in history.history.get("val_accuracy", [])]
    }

    return model, history_dict


def convert_to_float_tflite(keras_model: tf.keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    return converter.convert()


def convert_to_int8_tflite(
    keras_model: tf.keras.Model,
    X_train_prep: np.ndarray,
    num_calibration_samples: int = 200
) -> Tuple[bytes, list[int]]:
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    indices = list(range(min(len(X_train_prep), num_calibration_samples)))

    def representative_dataset_gen() -> Generator[list, None, None]:
        for idx in indices:
            sample = X_train_prep[idx:idx+1].astype(np.float32)
            yield [sample]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert(), indices


def main():
    parser = argparse.ArgumentParser(description="SafeNest V6 mmWave Model Trainer")
    parser.add_argument("--dataset", type=str, default="datasets/mmwave/processed/mmwave_respiration_v1.npz", help="Dataset NPZ path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory for model artifacts")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Enable strict op determinism")
    args = parser.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)

    project_root = Path(__file__).resolve().parent.parent
    dataset_path = (project_root / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset)

    if not dataset_path.exists():
        print(f"❌ Error: Dataset file non-existent at {dataset_path}")
        sys.exit(1)

    print(f"🚀 Training mmWave candidate model (Seed {args.seed}, Deterministic={args.deterministic})...")
    data = np.load(dataset_path, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    class_map = {0: "NORMAL", 1: "RAPID_OR_ABNORMAL", 2: "APNEA"}

    # Compute normalization statistics strictly from train split
    preprocessor = MMWavePreprocessor.from_train_split(X_train)
    print(f"  Train-only normalization stats -> Mean: {preprocessor.mean:.6f}, Std: {preprocessor.std:.6f}")

    # Train Float model
    keras_model, history_dict = train_float_model(X_train, y_train, X_val, y_val, preprocessor, epochs=args.epochs)

    # Save Keras Model Checkpoint
    if args.output_dir:
        models_dir = Path(args.output_dir).resolve() / "models/mmwave"
    else:
        models_dir = project_root / "models/mmwave"
    models_dir.mkdir(parents=True, exist_ok=True)
    keras_path = models_dir / "mmwave_resp_float_v0.2.0_candidate.keras"
    keras_model.save(str(keras_path))
    print(f"  Saved Keras Float Model: {keras_path}")

    # Save Training Config and History JSONs
    train_config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": 32,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "loss": "sparse_categorical_crossentropy",
        "class_map": class_map,
        "preprocessor": {
            "mean": preprocessor.mean,
            "std": preprocessor.std,
            "clip_min": preprocessor.clip_min,
            "clip_max": preprocessor.clip_max,
            "stage": "EXPERIMENTAL_PREPROCESSING_V1"
        }
    }
    with open(models_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)

    with open(models_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)

    # 1. Evaluate Float Keras model
    X_test_prep = preprocessor.preprocess_batch(X_test)
    float_preds = keras_model.predict(X_test_prep, verbose=0)
    float_y_preds = np.argmax(float_preds, axis=1)
    float_metrics = compute_metrics(y_test, float_y_preds, class_map)

    print("\n📊 Stage 1: Float Keras Model Evaluation")
    print(f"  Accuracy: {float_metrics['accuracy']:.4f}, Macro F1: {float_metrics['macro_f1']:.4f}")

    # 2. Convert to Float TFLite and evaluate
    float_tflite_bytes = convert_to_float_tflite(keras_model)
    float_tflite_path = models_dir / "mmwave_resp_float_v0.2.0_candidate.tflite"
    with open(float_tflite_path, "wb") as f:
        f.write(float_tflite_bytes)
    float_tflite_metrics = evaluate_tflite_model(float_tflite_path, X_test, y_test, preprocessor, class_map)

    print("\n📊 Stage 2: Float TFLite Model Evaluation")
    print(f"  Accuracy: {float_tflite_metrics['accuracy']:.4f}, Macro F1: {float_tflite_metrics['macro_f1']:.4f}")

    # 3. Convert to INT8 TFLite and evaluate
    X_train_prep = preprocessor.preprocess_batch(X_train)
    int8_tflite_bytes, calib_indices = convert_to_int8_tflite(keras_model, X_train_prep)
    candidate_tflite_path = models_dir / "mmwave_resp_int8_v0.2.0_candidate.tflite"
    with open(candidate_tflite_path, "wb") as f:
        f.write(int8_tflite_bytes)

    with open(models_dir / "representative_dataset_indices.json", "w", encoding="utf-8") as f:
        json.dump({"calibration_sample_indices": calib_indices, "count": len(calib_indices)}, f, indent=2)

    int8_tflite_metrics = evaluate_tflite_model(candidate_tflite_path, X_test, y_test, preprocessor, class_map)

    print("\n📊 Stage 3: INT8 TFLite Candidate Model Evaluation")
    print(f"  Accuracy: {int8_tflite_metrics['accuracy']:.4f}, Macro F1: {int8_tflite_metrics['macro_f1']:.4f}")
    print(f"  Apnea Window Miss Rate: {int8_tflite_metrics['apnea_window_miss_rate']:.4f}")
    print(f"  Class Collapse: {int8_tflite_metrics['class_collapse']}")
    print(f"  Input Saturation Ratio: {int8_tflite_metrics['input_saturation_ratio']:.4f}")
    print(f"  Predictions: {int8_tflite_metrics['prediction_distribution']}")

    from scripts.validate_metadata import (
        build_mmwave_candidate_metadata,
        save_candidate_metadata_atomically,
    )

    candidate_sha256 = calculate_sha256(candidate_tflite_path)

    # Build Candidate Metadata JSON using fixed schema builder
    metadata = build_mmwave_candidate_metadata(
        candidate_tflite_path=candidate_tflite_path,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=32,
        learning_rate=0.001,
        mean=preprocessor.mean,
        std=preprocessor.std,
        float_keras_eval=float_metrics,
        float_tflite_eval=float_tflite_metrics,
        int8_tflite_eval=int8_tflite_metrics,
        class_map=class_map,
    )

    metadata_path = models_dir / "mmwave_resp_int8_v0.2.0_candidate_metadata.json"
    save_candidate_metadata_atomically(metadata, metadata_path, model_root=models_dir.parent.parent)
    print(f"\n✅ Saved and validated metadata atomically: {metadata_path}")

    # Update models/model_manifest.json in V6 only if default run
    manifest_path = project_root / "models/model_manifest.json"
    if not args.output_dir and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        eval_copy = dict(int8_tflite_metrics)
        if "model_path" in eval_copy:
            eval_copy["model_path"] = f"models/mmwave/{Path(eval_copy['model_path']).name}"

        manifest["project"] = "SafeNest_V6"
        manifest["models"]["mmwave_v0_2_0_candidate"] = {
            "model_id": "mmwave_resp_int8_v0.2.0_candidate",
            "version": "0.2.0_candidate",
            "status": "candidate",
            "artifact_status": "CONFIRMED",
            "validation_status": "SYNTHETIC_SMOKE_ONLY",
            "deployment_allowed": True,
            "real_sensor_performance": "NOT_VERIFIABLE",
            "hardware_validation": "BLOCKED_HARDWARE",
            "role": "respiration_anomaly_classification",
            "framework": "TensorFlow Lite",
            "quantization": "full_int8",
            "path": "models/mmwave/mmwave_resp_int8_v0.2.0_candidate.tflite",
            "metadata_path": "models/mmwave/mmwave_resp_int8_v0.2.0_candidate_metadata.json",
            "sha256": candidate_sha256,
            "input": {
                "shape": [1, 300, 1],
                "dtype": "int8",
                "sample_rate_hz": 10,
                "window_seconds": 30,
                "semantic": "resp_phase"
            },
            "class_map": class_map,
            "evaluation": eval_copy
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"✅ Updated manifest entry in {manifest_path}")


if __name__ == "__main__":
    main()
