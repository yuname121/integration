#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
thermal_train.py
SafeNest Thermal-44 (80x62 IR Array) Keras 2D-CNN 대규모 학습 및 INT8 Full Quantization

1. thermal/processed_thermal_80x62.npz 로드 (54,218 프레임 전수)
2. Keras 2D-CNN 경량 파이프라인 훈련 (Epochs=20, EarlyStopping, ReduceLROnPlateau)
3. Representative Dataset INT8 Calibration (Full Integer Quantization)
4. models/thermal/thermal_fall_int8_v0.1.0.tflite & tflite/thermal_fall_quant.tflite 갱신
5. models/model_manifest.json SHA256 & 바이너리 메타데이터 동기화
"""

import os
import sys
import time
import json
import hashlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def update_manifest(project_root, tflite_path, input_details, output_details):
    manifest_path = os.path.join(project_root, "models", "model_manifest.json")
    if not os.path.exists(manifest_path):
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(tflite_path, "rb") as f:
        content = f.read()
        sha256_hash = hashlib.sha256(content).hexdigest()
        file_size = len(content)

    in_quant = input_details[0].get("quantization", (0.0, 0))
    out_quant = output_details[0].get("quantization", (0.0, 0))

    data["models"]["thermal"]["path"] = os.path.relpath(tflite_path, project_root)
    data["models"]["thermal"]["size_bytes"] = file_size
    data["models"]["thermal"]["sha256"] = sha256_hash
    data["models"]["thermal"]["input"]["scale"] = float(in_quant[0])
    data["models"]["thermal"]["input"]["zero_point"] = int(in_quant[1])
    data["models"]["thermal"]["output"]["scale"] = float(out_quant[0])
    data["models"]["thermal"]["output"]["zero_point"] = int(out_quant[1])

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"📝 models/model_manifest.json SHA256({sha256_hash[:8]}...) 및 바이너리 메타데이터 동기화 완료!")

def train_and_quantize_thermal_model(base_dir):
    print("🚀 [Step 2] Thermal-44 (80x62 IR Array) Keras 2D-CNN 대규모 학습 & INT8 양자화 변환 시작...")
    
    npz_path = os.path.join(base_dir, "thermal", "processed_thermal_80x62.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"❌ 전처리 데이터셋을 찾을 수 없습니다: {npz_path}\n👉 python3 thermal_prep.py를 먼저 실행해 주세요!")
        
    data = np.load(npz_path)
    X_raw, y = data['X'], data['y']
    X = np.expand_dims(X_raw, axis=-1)  # (N, 62, 80, 1)
    
    # 80:20 Train / Test Split
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * 0.2)
    X_train, X_test = X[indices[test_size:]], X[indices[:test_size]]
    y_train, y_test = y[indices[test_size:]], y[indices[:test_size]]
    
    print(f"📊 80x62 전수 데이터셋 로드 완료:")
    print(f"   - Train 세트: {X_train.shape}, 클래스 분포: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"   - Test  세트: {X_test.shape}, 클래스 분포: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # Keras 2D-CNN 아키텍처
    model = models.Sequential([
        layers.Input(shape=(62, 80, 1)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    cb_early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )
    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, verbose=1
    )
    
    print("\n⚡ Keras 모델 학습 실행 중 (Epochs: 20, Batch Size: 64)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=64,
        callbacks=[cb_early_stop, cb_reduce_lr],
        verbose=1
    )
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n🎯 [Float32 Keras 검증 성과] Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")
    
    # Keras H5 저장
    keras_model_path = os.path.join(base_dir, "models", "thermal", "thermal_fall_model.h5")
    model.save(keras_model_path)
    print(f"💾 Keras H5 모델 저장 완료: {keras_model_path}")
    
    # --- INT8 Full Quantization ---
    print("\n⚙️ Representative Dataset Calibration (INT8 Full Quantization) 수행 중...")
    
    def representative_dataset_gen():
        for i in range(min(500, len(X_train))):
            yield [X_train[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_quant_model = converter.convert()
    
    # 1) models/thermal/thermal_fall_int8_v0.1.0.tflite 저장
    manifest_tflite_path = os.path.join(base_dir, "models", "thermal", "thermal_fall_int8_v0.1.0.tflite")
    os.makedirs(os.path.dirname(manifest_tflite_path), exist_ok=True)
    with open(manifest_tflite_path, "wb") as f:
        f.write(tflite_quant_model)
        
    # 2) tflite/thermal_fall_quant.tflite 저장
    legacy_tflite_path = os.path.join(base_dir, "tflite", "thermal_fall_quant.tflite")
    os.makedirs(os.path.dirname(legacy_tflite_path), exist_ok=True)
    with open(legacy_tflite_path, "wb") as f:
        f.write(tflite_quant_model)

    print(f"🎉 [INT8 TFLite 모델 저장 성공]")
    print(f"   - 공식 모델 경로: {manifest_tflite_path} ({len(tflite_quant_model)/1024:.2f} KB)")
    print(f"   - 호환 모델 경로: {legacy_tflite_path}")

    # TFLite 인터프리터 검증 및 Manifest 갱신
    interpreter = tf.lite.Interpreter(model_path=manifest_tflite_path)
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    update_manifest(base_dir, manifest_tflite_path, in_details, out_details)
    print("\n✅ Thermal-44 80x62 파이프라인 학습 및 INT8 양자화 모델 갱신 완료!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_and_quantize_thermal_model(base_dir)
