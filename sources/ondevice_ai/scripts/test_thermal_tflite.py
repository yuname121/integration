#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_thermal_tflite.py
SafeNest Thermal-44 INT8 TFLite 모델 추론 단위 검증 스크립트

models/model_manifest.json 및 inference/thermal_interpreter.py 공용 Wrapper를 사용하여
단일화된 공식 모델(thermal_fall_int8_v0.1.0.tflite)을 검증합니다.
"""

import sys
from pathlib import Path
import numpy as np

# 프로젝트 루트를 Python 모듈 경로에 추가
base_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir))

from inference.thermal_interpreter import ThermalInterpreter

def main():
    print("=" * 70)
    print("🛡️ SafeNest Thermal-44 INT8 TFLite 공용 Wrapper 단밀 검증")
    print("=" * 70)

    try:
        runner = ThermalInterpreter(project_root=base_dir)
        print(f"✅ [성공] Manifest 모델 로드: {runner.model_path.name}")
        print(f"  - Model ID: {runner.model_meta['model_id']} (v{runner.model_meta['version']})")
        print(f"  - Input Tensor:  Shape {runner.input_info['shape'].tolist()}, Dtype {runner.input_info['dtype'].__name__}")
        print(f"  - Output Tensor: Shape {runner.output_info['shape'].tolist()}, Dtype {runner.output_info['dtype'].__name__}")
    except Exception as e:
        print(f"❌ [오류] 모델 인스턴스화 실패: {e}")
        sys.exit(1)

    # 샘플 테스트 1: Zero Frame
    frame_zero = np.zeros((62, 80), dtype=np.float32)
    pred_zero = runner.predict(frame_zero)
    print("\n[테스트 1: Zero Grid Frame]")
    print(f"  - 예측 클래스: {pred_zero.class_name} (Index {pred_zero.class_index})")
    print(f"  - 신뢰도 (Confidence): {pred_zero.confidence * 100:.2f}%")
    print(f"  - 추론 지연시간: {pred_zero.latency_ms:.2f} ms")
    print(f"  - Class Probabilities: Not Human={pred_zero.probabilities[0]:.4f}, Normal={pred_zero.probabilities[1]:.4f}, Fall={pred_zero.probabilities[2]:.4f}")

    # 샘플 테스트 2: Simulated Fall Heatmap (바닥 부분 체온 부하)
    frame_fall = np.zeros((62, 80), dtype=np.float32)
    frame_fall[45:58, 15:65] = 0.85  # 바닥 가로 체온 부하
    pred_fall = runner.predict(frame_fall)
    print("\n[테스트 2: Simulated Fall Grid Frame]")
    print(f"  - 예측 클래스: {pred_fall.class_name} (Index {pred_fall.class_index})")
    print(f"  - 신뢰도 (Confidence): {pred_fall.confidence * 100:.2f}%")
    print(f"  - 추론 지연시간: {pred_fall.latency_ms:.2f} ms")
    print(f"  - Class Probabilities: Not Human={pred_fall.probabilities[0]:.4f}, Normal={pred_fall.probabilities[1]:.4f}, Fall={pred_fall.probabilities[2]:.4f}")

    print("\n" + "=" * 70)
    print("✅ PASS: Thermal TFLite 모델 추론 테스트 완료!")
    print("=" * 70)

if __name__ == "__main__":
    main()
