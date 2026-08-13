#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inference/mmwave_interpreter.py
SafeNest 공용 mmWave 30초 시계열 호흡 파형 TFLite 추론 Wrapper

[검수 2차 지적사항 반영 완료]
1. INT8 / FLOAT32 입력/출력 텐서 스펙 검사 및 양자화/역양자화 연산 수립
2. 모델 SHA-256 및 입출력 텐서 계약을 Manifest와 대조
3. TFLite 모델 파일 미존재/로드 실패 원인을 구분하여 fallback provenance에 명시
4. 휴리스틱 결과 산출 시 model_id를 "mmwave_heuristic_fallback"으로 투명하게 반환하여 가짜 AI 표기 방지
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import scipy.signal

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
class MMWavePrediction:
    class_index: int
    class_name: str
    confidence: float
    probabilities: list[float]
    latency_ms: float
    model_id: str
    model_version: str
    fallback_used: bool
    fallback_reason: str | None
    preprocessing_profile: str | None = None
    model_sha256: str | None = None


class MMWaveInterpreter:
    def __init__(
        self,
        project_root: str | Path | None = None,
        manifest_path: str = "models/model_manifest.json",
        runtime_manifest_path: str | Path | None = None,
    ) -> None:
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = Path(project_root).resolve()

        # A phase-local runtime manifest is deliberately opt-in.  The shared
        # project manifest remains the legacy/backward-compatible default and
        # is never silently replaced by a finalist artifact.
        self.runtime_manifest_path = runtime_manifest_path
        selected_manifest_path = runtime_manifest_path if runtime_manifest_path is not None else manifest_path
        manifest_file = Path(selected_manifest_path)
        if not manifest_file.is_absolute():
            manifest_file = self.project_root / manifest_file
        manifest_file = manifest_file.resolve()
        if not manifest_file.is_file():
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        self.is_phase_runtime_manifest = runtime_manifest_path is not None
        if self.is_phase_runtime_manifest:
            self.model_meta = manifest.get("runtime_model")
            if not isinstance(self.model_meta, dict):
                raise ValueError("M-B9 runtime manifest missing runtime_model")
            self.preprocessing_meta = manifest.get("preprocessing")
            self._validate_runtime_preprocessing_metadata(self.preprocessing_meta)
        else:
            self.model_meta = manifest["models"]["mmwave"]
            self.preprocessing_meta = {
                "profile_id": "LEGACY_MANIFEST_ZSCORE_ONLY",
                "profile_name": "LEGACY_ZSCORE_ONLY",
                "bpf": False,
                "detrend": False,
                "zscore": True,
            }
        self.class_map = {
            int(key): value
            for key, value in self.model_meta["class_map"].items()
        }

        # Metadata 로드 (Z-Score mean/std). 학습 산출물이 [value] 형태로
        # 저장되는 경우도 허용하되, 다변량 통계는 조용히 축약하지 않는다.
        if self.is_phase_runtime_manifest:
            self.stats_meta = {
                "mean": self.preprocessing_meta["zscore_mean"],
                "std": self.preprocessing_meta["zscore_std"],
            }
        else:
            meta_file = self.project_root / self.model_meta.get(
                "metadata_path", "models/mmwave/sensor_stats_metadata_v0.1.0.json"
            )
            if not meta_file.is_file():
                raise FileNotFoundError(f"mmWave metadata file not found: {meta_file}")
            self.stats_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if self.is_phase_runtime_manifest:
            self.mean = float(self.stats_meta["mean"])
            self.std = float(self.stats_meta["std"])
            if not np.isfinite(self.mean) or not np.isfinite(self.std):
                raise ValueError("M-B9_RUNTIME_PREPROCESSING_DEPENDENCY_BLOCKER")
        else:
            self.mean = self._read_scalar_stat(self.stats_meta.get("mean"), "mean")
            self.std = self._read_scalar_stat(self.stats_meta.get("std"), "std")
        if self.std <= 0:
            raise ValueError("mmWave metadata std must be greater than zero")

        model_path_value = self.model_meta["path"]
        model_path_obj = Path(model_path_value)
        if self.is_phase_runtime_manifest and model_path_obj.is_absolute():
            raise ValueError("M-B9 runtime model path must be repository-relative")
        self.model_path = (self.project_root / model_path_obj).resolve()
        self.interpreter = None
        self.model_file_exists = self.model_path.is_file()
        self.sha256_hash = None
        self.sha256_matches = False
        self.load_error_reason = None

        # TFLite 바이너리가 존재하는 경우 SHA-256과 텐서 계약을 검증한다.
        if self.model_file_exists:
            try:
                model_bytes = self.model_path.read_bytes()
                self.sha256_hash = hashlib.sha256(model_bytes).hexdigest()
                expected_sha256 = self.model_meta.get("expected_sha256", self.model_meta.get("sha256"))
                if expected_sha256 and self.sha256_hash != expected_sha256:
                    self.load_error_reason = (
                        "M-B9_FINALIST_ARTIFACT_IDENTITY_MISMATCH"
                        if self.is_phase_runtime_manifest
                        else "TFLITE_MODEL_SHA256_MISMATCH"
                    )
                    raise ValueError(
                        f"mmWave model SHA-256 mismatch: expected={expected_sha256}, "
                        f"actual={self.sha256_hash}"
                    )
                expected_bytes = self.model_meta.get("expected_bytes", self.model_meta.get("bytes"))
                if expected_bytes is not None and len(model_bytes) != int(expected_bytes):
                    self.load_error_reason = (
                        "M-B9_FINALIST_ARTIFACT_IDENTITY_MISMATCH"
                        if self.is_phase_runtime_manifest
                        else "TFLITE_MODEL_BYTES_MISMATCH"
                    )
                    raise ValueError(
                        f"mmWave model byte-size mismatch: expected={expected_bytes}, "
                        f"actual={len(model_bytes)}"
                    )
                self.sha256_matches = True
                self.interpreter = tflite.Interpreter(model_path=str(self.model_path))
                self.interpreter.allocate_tensors()
                self.input_info = self.interpreter.get_input_details()[0]
                self.output_info = self.interpreter.get_output_details()[0]
                self._validate_tensor_contract()
            except Exception as e:
                print(f"⚠️ [MMWaveInterpreter] TFLite 로드 경고: {e}")
                self.interpreter = None
                if self.load_error_reason is None:
                    self.load_error_reason = (
                        "M-B9_FINALIST_ARTIFACT_IDENTITY_MISMATCH"
                        if self.is_phase_runtime_manifest
                        else "TFLITE_MODEL_LOAD_ERROR"
                    )
        else:
            self.load_error_reason = (
                "M-B9_FINALIST_ARTIFACT_IDENTITY_MISMATCH"
                if self.is_phase_runtime_manifest
                else "TFLITE_MODEL_FILE_MISSING"
            )

        if self.is_phase_runtime_manifest and self.interpreter is None:
            # Strict phase manifests never fall through to the historical
            # heuristic.  The provider records this as an unavailable model.
            raise ValueError(self.load_error_reason or "M-B9_FINALIST_ARTIFACT_IDENTITY_MISMATCH")

    @staticmethod
    def _read_scalar_stat(value: object, field_name: str) -> float:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size != 1 or not np.all(np.isfinite(array)):
            raise ValueError(f"mmWave metadata {field_name} must contain one finite value")
        return float(array[0])

    @staticmethod
    def _validate_runtime_preprocessing_metadata(metadata: object) -> None:
        if not isinstance(metadata, dict):
            raise ValueError("M-B9_RUNTIME_PREPROCESSING_DEPENDENCY_BLOCKER")
        required = {
            "profile_id",
            "profile_name",
            "detrend",
            "bpf",
            "zscore",
            "sample_rate_hz",
            "bpf_lowcut_hz",
            "bpf_highcut_hz",
            "bpf_order",
            "zscore_fit_split",
            "zscore_mean",
            "zscore_std",
        }
        if not required.issubset(metadata):
            raise ValueError("M-B9_RUNTIME_PREPROCESSING_DEPENDENCY_BLOCKER")
        if (
            metadata.get("profile_name") != "BPF_ZSCORE"
            or metadata.get("profile_id") != "M-B1_D0_B1_Z1"
            or metadata.get("detrend") is not False
            or metadata.get("bpf") is not True
            or metadata.get("zscore") is not True
            or not np.isclose(float(metadata.get("sample_rate_hz")), 10.0, rtol=0, atol=1e-12)
            or not np.isclose(float(metadata.get("bpf_lowcut_hz")), 0.1, rtol=0, atol=1e-12)
            or not np.isclose(float(metadata.get("bpf_highcut_hz")), 0.5, rtol=0, atol=1e-12)
            or int(metadata.get("bpf_order")) != 4
            or metadata.get("zscore_fit_split") != "TRAIN"
        ):
            raise ValueError("M-B9_RUNTIME_PREPROCESSING_MISMATCH")
        for field_name in ("zscore_mean", "zscore_std"):
            value = float(metadata[field_name])
            if not np.isfinite(value):
                raise ValueError("M-B9_RUNTIME_PREPROCESSING_DEPENDENCY_BLOCKER")
        if float(metadata["zscore_std"]) <= 0:
            raise ValueError("M-B9_RUNTIME_PREPROCESSING_MISMATCH")

    @staticmethod
    def _dtype_name(dtype: object) -> str:
        return np.dtype(dtype).name

    def _validate_tensor_contract(self) -> None:
        for label, actual, expected in (
            ("input", self.input_info, self.model_meta["input"]),
            ("output", self.output_info, self.model_meta["output"]),
        ):
            actual_shape = [int(value) for value in actual["shape"]]
            if actual_shape != expected["shape"]:
                raise ValueError(
                    f"mmWave {label} shape mismatch: expected={expected['shape']}, "
                    f"actual={actual_shape}"
                )
            actual_dtype = self._dtype_name(actual["dtype"])
            if actual_dtype != expected["dtype"]:
                raise ValueError(
                    f"mmWave {label} dtype mismatch: expected={expected['dtype']}, "
                    f"actual={actual_dtype}"
                )

            expected_scale = expected.get("scale")
            expected_zero_point = expected.get("zero_point")
            if expected_scale is not None:
                actual_scale, actual_zero_point = actual["quantization"]
                if not np.isclose(float(actual_scale), float(expected_scale), rtol=0, atol=1e-12):
                    raise ValueError(
                        f"mmWave {label} scale mismatch: expected={expected_scale}, "
                        f"actual={actual_scale}"
                    )
                if int(actual_zero_point) != int(expected_zero_point):
                    raise ValueError(
                        f"mmWave {label} zero_point mismatch: expected={expected_zero_point}, "
                        f"actual={actual_zero_point}"
                    )

    def prepare_window(self, window: np.ndarray) -> np.ndarray:
        """Validate, preprocess, and quantize one 300-sample resp_phase window."""
        trace = self.preprocess_trace(window)
        if self.interpreter is not None:
            return trace["quantized_input"]
        return trace["model_ready"]

    def preprocess_trace(self, window: np.ndarray) -> dict[str, np.ndarray | str | float | int]:
        """Return each frozen preprocessing stage for independent M-B9 auditing."""
        dtype = np.float64 if self.is_phase_runtime_manifest else np.float32
        array = np.asarray(window, dtype=dtype)

        if array.shape == (300,):
            array = array[None, ..., None]
        elif array.shape == (300, 1):
            array = array[None, ...]
        elif array.shape != (1, 300, 1):
            raise ValueError(f"mmWave window shape must be (300,), (300,1), or (1,300,1), got {array.shape}")

        if not np.all(np.isfinite(array)):
            raise ValueError("mmWave window contains NaN or infinity")

        canonical = array.astype(np.float64, copy=False)
        squeezed = canonical.reshape(1, 300)
        if self.is_phase_runtime_manifest:
            b, a = scipy.signal.butter(
                int(self.preprocessing_meta["bpf_order"]),
                [
                    float(self.preprocessing_meta["bpf_lowcut_hz"]),
                    float(self.preprocessing_meta["bpf_highcut_hz"]),
                ],
                btype="bandpass",
                fs=float(self.preprocessing_meta["sample_rate_hz"]),
            )
            bpf_output = scipy.signal.filtfilt(b, a, squeezed, axis=-1)
        else:
            bpf_output = squeezed.astype(np.float32, copy=False)

        normalized = (bpf_output - self.mean) / self.std
        model_ready = normalized.astype(np.float32)
        model_ready_tensor = model_ready.reshape(1, 300, 1)

        # INT8 양자화 처리 (Interpreter 존재 시)
        if self.interpreter is not None:
            dtype = self.input_info["dtype"]
            if np.issubdtype(dtype, np.integer):
                scale, zero_point = self.input_info["quantization"]
                if scale > 0:
                    quantized = np.rint(model_ready_tensor / scale + zero_point)
                    limits = np.iinfo(dtype)
                    quantized = np.clip(quantized, limits.min, limits.max)
                    raw_quantized = np.rint(model_ready_tensor / scale + zero_point)
                    return {
                        "canonical": canonical,
                        "bpf_output": bpf_output,
                        "zscore_output": normalized,
                        "model_ready": model_ready_tensor,
                        "quantized_input": quantized.astype(dtype),
                        "input_saturation_count": int(np.sum((raw_quantized < limits.min) | (raw_quantized > limits.max))),
                        "preprocessing_profile": self.preprocessing_meta.get("profile_name", "LEGACY_ZSCORE_ONLY"),
                    }

        return {
            "canonical": canonical,
            "bpf_output": bpf_output,
            "zscore_output": normalized,
            "model_ready": model_ready_tensor,
            "quantized_input": model_ready_tensor,
            "input_saturation_count": 0,
            "preprocessing_profile": self.preprocessing_meta.get("profile_name", "LEGACY_ZSCORE_ONLY"),
        }

    def decode_output(self, raw_output: np.ndarray) -> np.ndarray:
        """INT8 / FLOAT32 역양자화 처리"""
        if self.interpreter is not None:
            dtype = self.output_info["dtype"]
            if np.issubdtype(dtype, np.integer):
                scale, zero_point = self.output_info["quantization"]
                if scale > 0:
                    probs = (raw_output.astype(np.float32) - zero_point) * scale
                    return probs[0]

        return raw_output[0].astype(np.float32)

    def predict(self, window: np.ndarray) -> MMWavePrediction:
        trace = self.preprocess_trace(window)
        input_tensor = trace["quantized_input"]
        self.last_preprocess_trace = trace
        started = time.perf_counter()

        if self.interpreter is not None:
            self.interpreter.set_tensor(self.input_info["index"], input_tensor)
            self.interpreter.invoke()
            raw_output = self.interpreter.get_tensor(self.output_info["index"])
            probabilities = self.decode_output(raw_output)
            probabilities = np.asarray(probabilities, dtype=np.float32)
            if not np.all(np.isfinite(probabilities)):
                raise ValueError("M-B9_MODEL_OUTPUT_NON_FINITE")
            if self.is_phase_runtime_manifest:
                if float(np.sum(probabilities)) <= 0.0:
                    raise ValueError("M-B9_MODEL_OUTPUT_INVALID")
            else:
                probabilities = np.clip(probabilities, 0.0, None)
                total = float(np.sum(probabilities))
                if total > 0:
                    probabilities = probabilities / total
                else:
                    probabilities = np.array([0.333, 0.333, 0.334], dtype=np.float32)
            fallback_used = False
            fallback_reason = None
            model_id_str = self.model_meta["model_id"]
            self.last_raw_output = np.asarray(raw_output).copy()
        else:
            # Fallback heuristic calculation for 300-sample window
            std_val = float(np.std(window))
            if std_val < 0.05:  # Flat line -> Apnea
                probabilities = np.array([0.02, 0.03, 0.95], dtype=np.float32)
            elif std_val > 0.5:  # Rapid/Abnormal
                probabilities = np.array([0.10, 0.85, 0.05], dtype=np.float32)
            else:  # Normal
                probabilities = np.array([0.92, 0.05, 0.03], dtype=np.float32)
            fallback_used = True
            fallback_reason = self.load_error_reason or "TFLITE_MODEL_LOAD_ERROR"
            model_id_str = "mmwave_heuristic_fallback"

        latency_ms = (time.perf_counter() - started) * 1000.0
        class_index = int(np.argmax(probabilities))

        return MMWavePrediction(
            class_index=class_index,
            class_name=self.class_map.get(class_index, f"CLASS_{class_index}"),
            confidence=float(probabilities[class_index]),
            probabilities=[float(p) for p in probabilities],
            latency_ms=float(latency_ms),
            model_id=model_id_str,
            model_version=self.model_meta["version"],
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            preprocessing_profile=self.preprocessing_meta.get("profile_name"),
            model_sha256=self.sha256_hash,
        )
