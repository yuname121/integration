#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inference/thermal_interpreter.py
SafeNest 공용 Thermal Interpreter Wrapper

[역할]
1. models/model_manifest.json에서 공식 모델 경로 및 텐서 스펙 로드
2. Mac TensorFlow와 Raspberry Pi tflite-runtime 이중 호환
3. 입력 shape 검증 및 NaN/Inf 안전 검사
4. INT8 입력 양자화 (np.rint 반올림 & np.clip(-128, 127) 적용)
5. INT8 출력 역양자화 (Softmax 이중 적용 방지)
6. 추론 지연시간(latency_ms) 및 모델 버전 반환
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import numpy as np

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite
        except ImportError:
            import tensorflow as tf
            tflite = tf.lite


@dataclass(frozen=True)
class ThermalPrediction:
    class_index: int
    class_name: str
    confidence: float
    probabilities: list[float]
    latency_ms: float
    model_id: str
    model_version: str


class ThermalInterpreter:
    def __init__(
        self,
        project_root: str | Path | None = None,
        manifest_path: str = "models/model_manifest.json",
    ) -> None:
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = Path(project_root).resolve()

        manifest_file = self.project_root / manifest_path
        if not manifest_file.is_file():
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        self.model_meta = manifest["models"]["thermal"]
        self.class_map = {
            int(key): value
            for key, value in self.model_meta["class_map"].items()
        }

        self.model_path = self.project_root / self.model_meta["path"]
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.sha256_hash = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        expected_sha256 = self.model_meta.get("sha256")
        self.sha256_matches = bool(expected_sha256 and self.sha256_hash == expected_sha256)
        if not self.sha256_matches:
            raise ValueError(
                "thermal model SHA-256 mismatch: "
                f"expected={expected_sha256}, actual={self.sha256_hash}"
            )

        self.interpreter = tflite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()

        self.input_info = self.interpreter.get_input_details()[0]
        self.output_info = self.interpreter.get_output_details()[0]

        self._validate_contract()

    def _validate_contract(self) -> None:
        expected_input = self.model_meta["input"]
        expected_output = self.model_meta["output"]

        actual_input_shape = self.input_info["shape"].tolist()
        actual_output_shape = self.output_info["shape"].tolist()
        actual_input_dtype = self.input_info["dtype"].__name__
        actual_output_dtype = self.output_info["dtype"].__name__

        if actual_input_shape != expected_input["shape"]:
            raise ValueError(
                f"input shape mismatch: {actual_input_shape} != {expected_input['shape']}"
            )
        if actual_output_shape != expected_output["shape"]:
            raise ValueError(
                f"output shape mismatch: {actual_output_shape} != {expected_output['shape']}"
            )
        if actual_input_dtype != expected_input["dtype"]:
            raise ValueError(
                f"input dtype mismatch: {actual_input_dtype} != {expected_input['dtype']}"
            )
        if actual_output_dtype != expected_output["dtype"]:
            raise ValueError(
                f"output dtype mismatch: {actual_output_dtype} != {expected_output['dtype']}"
            )

        for label, actual, expected in (
            ("input", self.input_info, expected_input),
            ("output", self.output_info, expected_output),
        ):
            expected_scale = expected.get("scale")
            expected_zero_point = expected.get("zero_point")
            if expected_scale is None:
                continue
            actual_scale, actual_zero_point = actual["quantization"]
            if not np.isclose(float(actual_scale), float(expected_scale), rtol=0, atol=1e-12):
                raise ValueError(
                    f"{label} scale mismatch: {actual_scale} != {expected_scale}"
                )
            if int(actual_zero_point) != int(expected_zero_point):
                raise ValueError(
                    f"{label} zero_point mismatch: {actual_zero_point} != {expected_zero_point}"
                )

    @staticmethod
    def _prepare_float_frame(frame: np.ndarray) -> np.ndarray:
        array = np.asarray(frame, dtype=np.float32)

        if array.shape == (62, 80):
            array = array[None, ..., None]
        elif array.shape == (62, 80, 1):
            array = array[None, ...]
        elif array.shape != (1, 62, 80, 1):
            raise ValueError(
                f"thermal frame must have shape (62,80), (62,80,1), or (1,62,80,1), got {array.shape}"
            )

        if not np.all(np.isfinite(array)):
            raise ValueError("thermal frame contains NaN or infinity")

        min_value = float(array.min())
        max_value = float(array.max())
        if min_value < 0.0 or max_value > 1.0:
            # Min-Max normalize array to [0.0, 1.0] safely if unnormalized
            range_val = max_value - min_value
            if range_val > 0:
                array = (array - min_value) / range_val
            else:
                array = np.clip(array, 0.0, 1.0)

        return array

    def _encode_input(self, frame: np.ndarray) -> np.ndarray:
        float_input = self._prepare_float_frame(frame)
        dtype = self.input_info["dtype"]

        if np.issubdtype(dtype, np.integer):
            scale, zero_point = self.input_info["quantization"]
            if scale <= 0:
                raise ValueError("invalid input quantization scale")

            quantized = np.rint(float_input / scale + zero_point)
            limits = np.iinfo(dtype)
            quantized = np.clip(quantized, limits.min, limits.max)
            return quantized.astype(dtype)

        return float_input.astype(dtype)

    def _decode_output(self, raw_output: np.ndarray) -> np.ndarray:
        dtype = self.output_info["dtype"]

        if np.issubdtype(dtype, np.integer):
            scale, zero_point = self.output_info["quantization"]
            if scale <= 0:
                raise ValueError("invalid output quantization scale")
            probabilities = (
                raw_output.astype(np.float32) - zero_point
            ) * scale
        else:
            probabilities = raw_output.astype(np.float32)

        probabilities = probabilities[0]
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("model output contains NaN or infinity")

        probabilities = np.clip(probabilities, 0.0, None)
        total = float(probabilities.sum())
        if total <= 0.0:
            return np.array([0.333, 0.333, 0.334], dtype=np.float32)

        return probabilities / total

    def predict(self, frame: np.ndarray) -> ThermalPrediction:
        input_tensor = self._encode_input(frame)

        started = time.perf_counter()
        self.interpreter.set_tensor(
            self.input_info["index"],
            input_tensor,
        )
        self.interpreter.invoke()
        raw_output = self.interpreter.get_tensor(
            self.output_info["index"]
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        probabilities = self._decode_output(raw_output)
        class_index = int(np.argmax(probabilities))

        return ThermalPrediction(
            class_index=class_index,
            class_name=self.class_map.get(class_index, f"CLASS_{class_index}"),
            confidence=float(probabilities[class_index]),
            probabilities=[float(value) for value in probabilities],
            latency_ms=float(latency_ms),
            model_id=self.model_meta["model_id"],
            model_version=self.model_meta["version"],
        )
