#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inference/co2_interpreter.py
SafeNest CO2 & SCD40 Occupancy/Risk TFLite Interpreter Wrapper
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
class CO2Prediction:
    class_index: int
    class_name: str
    confidence: float
    probabilities: list[float]
    latency_ms: float
    model_id: str
    model_version: str


class CO2Interpreter:
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
        self.model_meta = manifest["models"]["co2"]
        self.class_map = {
            int(key): value
            for key, value in self.model_meta["class_map"].items()
        }

        # Metadata 로드 (scaling mean/std)
        meta_file = self.project_root / self.model_meta.get(
            "metadata_path", "models/co2/co2_scaling_metadata_v0.1.0.json"
        )
        if not meta_file.is_file():
            raise FileNotFoundError(f"CO2 metadata file not found: {meta_file}")
        self.stats_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        self.means = np.asarray(self.stats_meta["mean"], dtype=np.float32).reshape(-1)
        self.scales = np.asarray(self.stats_meta["scale"], dtype=np.float32).reshape(-1)
        if self.means.size != 3 or not np.all(np.isfinite(self.means)):
            raise ValueError("CO2 metadata mean must contain three finite values")
        if self.scales.size != 3 or not np.all(np.isfinite(self.scales)) or np.any(self.scales <= 0):
            raise ValueError("CO2 metadata scale must contain three positive finite values")

        self.model_path = self.project_root / self.model_meta["path"]
        if not self.model_path.is_file():
            raise FileNotFoundError(f"CO2 Model file not found: {self.model_path}")

        self.sha256_hash = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        expected_sha256 = self.model_meta.get("sha256")
        self.sha256_matches = bool(expected_sha256 and self.sha256_hash == expected_sha256)
        if not self.sha256_matches:
            raise ValueError(
                f"CO2 model SHA-256 mismatch: expected={expected_sha256}, actual={self.sha256_hash}"
            )

        self.interpreter = tflite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()

        self.input_info = self.interpreter.get_input_details()[0]
        self.output_info = self.interpreter.get_output_details()[0]
        self._validate_contract()

    def _validate_contract(self) -> None:
        for label, actual, expected in (
            ("input", self.input_info, self.model_meta["input"]),
            ("output", self.output_info, self.model_meta["output"]),
        ):
            actual_shape = [int(value) for value in actual["shape"]]
            if actual_shape != expected["shape"]:
                raise ValueError(
                    f"CO2 {label} shape mismatch: {actual_shape} != {expected['shape']}"
                )
            actual_dtype = np.dtype(actual["dtype"]).name
            if actual_dtype != expected["dtype"]:
                raise ValueError(
                    f"CO2 {label} dtype mismatch: {actual_dtype} != {expected['dtype']}"
                )
            expected_scale = expected.get("scale")
            if expected_scale is not None:
                actual_scale, actual_zero_point = actual["quantization"]
                if not np.isclose(float(actual_scale), float(expected_scale), rtol=0, atol=1e-12):
                    raise ValueError(
                        f"CO2 {label} scale mismatch: {actual_scale} != {expected_scale}"
                    )
                if int(actual_zero_point) != int(expected["zero_point"]):
                    raise ValueError(
                        f"CO2 {label} zero_point mismatch: "
                        f"{actual_zero_point} != {expected['zero_point']}"
                    )

    def prepare_input(self, co2_slope: float, humidity: float, co2_ppm: float) -> np.ndarray:
        raw_features = np.array([co2_slope, humidity, co2_ppm], dtype=np.float32)
        if not np.all(np.isfinite(raw_features)):
            raise ValueError("CO2 features contain NaN or infinity")
        scaled_features = (raw_features - self.means) / self.scales

        dtype = self.input_info["dtype"]
        if np.issubdtype(dtype, np.integer):
            scale, zero_point = self.input_info["quantization"]
            if scale <= 0:
                raise ValueError("invalid CO2 input quantization scale")
            quantized = np.rint(scaled_features / scale + zero_point)
            limits = np.iinfo(dtype)
            quantized = np.clip(quantized, limits.min, limits.max)
            return quantized.reshape(1, 3).astype(dtype)

        return scaled_features.reshape(1, 3).astype(dtype)

    def decode_output(self, raw_output: np.ndarray) -> np.ndarray:
        dtype = self.output_info["dtype"]
        if np.issubdtype(dtype, np.integer):
            scale, zero_point = self.output_info["quantization"]
            if scale <= 0:
                raise ValueError("invalid CO2 output quantization scale")
            probs = (raw_output.astype(np.float32) - zero_point) * scale
        else:
            probs = raw_output.astype(np.float32)

        probs = probs[0]
        if not np.all(np.isfinite(probs)):
            raise ValueError("CO2 model output contains NaN or infinity")
        probs = np.clip(probs, 0.0, None)
        total = float(np.sum(probs))
        if total > 0:
            return probs / total
        return np.array([0.5, 0.5], dtype=np.float32)

    def predict(self, co2_slope: float, humidity: float, co2_ppm: float) -> CO2Prediction:
        input_tensor = self.prepare_input(co2_slope, humidity, co2_ppm)

        started = time.perf_counter()
        self.interpreter.set_tensor(self.input_info["index"], input_tensor)
        self.interpreter.invoke()
        raw_output = self.interpreter.get_tensor(self.output_info["index"])
        latency_ms = (time.perf_counter() - started) * 1000.0

        probabilities = self.decode_output(raw_output)
        class_index = int(np.argmax(probabilities))

        return CO2Prediction(
            class_index=class_index,
            class_name=self.class_map.get(class_index, f"CLASS_{class_index}"),
            confidence=float(probabilities[class_index]),
            probabilities=[float(p) for p in probabilities],
            latency_ms=float(latency_ms),
            model_id=self.model_meta["model_id"],
            model_version=self.model_meta["version"],
        )
