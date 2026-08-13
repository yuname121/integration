#!/usr/bin/env python3
"""Strict M-B9 finalist-backed mock mmWave provider.

The historical :class:`MockMMWaveSensor` remains unchanged for existing smoke
tests.  This provider is intentionally separate: scenario names describe
metadata-only truth, while the returned state, score, and confidence are
derived from the explicitly supplied finalist model prediction.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from inference.inference_result import InferenceResult
from inference.mmwave_interpreter import MMWaveInterpreter
from sensors.base_sensor import BaseSensor, SensorState


CLASS_SCORE = {0: 0.0, 1: 0.5, 2: 1.0}


class FinalistMockProvider(BaseSensor):
    """A deterministic provider for one explicit M-B6 strict-INT8 finalist."""

    def __init__(
        self,
        project_root: str | Path,
        runtime_manifest_path: str | Path,
        *,
        raw_window: np.ndarray | None = None,
        scenario_truth_class: str | None = None,
        selection_metadata: dict[str, Any] | None = None,
        failure_mode: str | None = None,
        timeout_sec: float = 2.0,
        stale_sec: float = 3.0,
        sample_rate_hz: float = 10.0,
        window_samples: int = 300,
        window_seconds: float = 30.0,
    ) -> None:
        super().__init__(sensor_id="mmwave", timeout_sec=timeout_sec, stale_sec=stale_sec)
        self.sample_rate_hz = sample_rate_hz
        self.window_samples = window_samples
        self.window_seconds = window_seconds
        self.runtime_manifest_path = str(runtime_manifest_path)
        self.raw_window = None if raw_window is None else np.asarray(raw_window).copy()
        self.scenario_truth_class = scenario_truth_class
        self.selection_metadata = dict(selection_metadata or {})
        self.failure_mode = failure_mode
        self.last_result: InferenceResult | None = None
        self.initialization_error: str | None = None
        self.interpreter: MMWaveInterpreter | None = None
        try:
            self.interpreter = MMWaveInterpreter(
                project_root=project_root,
                runtime_manifest_path=runtime_manifest_path,
            )
        except Exception as exc:  # strict mode records, never hides, load failure
            message = str(exc) or "M-B9_FINALIST_ARTIFACT_IDENTITY_MISMATCH"
            if "Manifest file not found" in message:
                message = "M-B9_FINALIST_MODEL_MANIFEST_MISSING"
            self.initialization_error = message

    def connect(self) -> bool:
        if self.failure_mode == "NOT_CONNECTED":
            self.connected = False
            self.current_state = SensorState.NOT_CONNECTED
            return False
        self.connected = True
        self.current_state = SensorState.NORMAL
        return True

    def _invalid(
        self,
        now: float,
        error: str,
        *,
        state: str = "MMWAVE_INVALID",
        metadata: dict[str, Any] | None = None,
        fallback_used: bool = False,
        model_id: str = "mmwave_finalist_unavailable",
        fallback_reason: str | None = None,
    ) -> InferenceResult:
        self.error_count += 1
        self.current_state = SensorState.INFER_FAILED
        details = {
            "model_id": model_id,
            "model_version": None,
            "fallback_used": bool(fallback_used),
            "fallback_reason": fallback_reason or error,
            "score_source": "NO_VALID_PREDICTION",
            "scenario_truth_class": self.scenario_truth_class,
            "selection_metadata": self.selection_metadata,
        }
        details.update(metadata or {})
        result = InferenceResult(
            sensor_id="mmwave",
            timestamp=now,
            score=0.0,
            state=state,
            confidence=0.0,
            valid=False,
            latency_ms=0.0,
            error=error,
            metadata=details,
        )
        self.last_result = result
        return result

    def read(self) -> InferenceResult:
        started = time.perf_counter()
        now = time.time()
        self.read_count += 1
        self.last_read_ts = now

        if not self.connected:
            return self._invalid(now, "SENSOR_NOT_CONNECTED", state="NOT_CONNECTED")

        if self.failure_mode == "READ_EXCEPTION":
            raise RuntimeError("M-B9_SIMULATED_PROVIDER_READ_EXCEPTION")
        if self.failure_mode == "TIMEOUT":
            # The integrated node owns the timeout decision.  The sleep is
            # finite and deliberately just beyond the configured 2-second
            # provider contract.
            time.sleep(self.timeout_sec + 0.05)
            return self._invalid(now, "PROVIDER_READ_TIMEOUT", state="READ_TIMEOUT")
        if self.failure_mode == "INSUFFICIENT_HISTORY":
            return self._invalid(
                now,
                "INSUFFICIENT_HISTORY",
                state="WARMING_UP",
                metadata={
                    "buffer_len": 0,
                    "required_samples": self.window_samples,
                    "remaining_samples": self.window_samples,
                },
            )
        if self.failure_mode == "PROVIDER_FAULT":
            return self._invalid(now, "SIMULATED_MMWAVE_SENSOR_FAULT", state="FAULT")
        if self.failure_mode in {"MISSING_MODEL", "SHA_MISMATCH"} or self.initialization_error:
            reason = (
                "M-B9_FINALIST_ARTIFACT_IDENTITY_MISMATCH"
                if self.failure_mode == "SHA_MISMATCH"
                else self.initialization_error or "M-B9_FINALIST_MODEL_MISSING"
            )
            # This is an explicit record of the legacy fallback identity.  It
            # is invalid and is never reported as a finalist prediction.
            return self._invalid(
                now,
                reason,
                state="MODEL_UNAVAILABLE",
                fallback_used=True,
                model_id="mmwave_heuristic_fallback",
                fallback_reason=reason,
                metadata={"legacy_fallback_behavior": "HEURISTIC_NOT_ACCEPTED_AS_FINALIST"},
            )

        if self.interpreter is None or self.raw_window is None:
            return self._invalid(now, "M-B9_FINALIST_MODEL_UNAVAILABLE", state="MODEL_UNAVAILABLE")

        window = self.raw_window.copy()
        if self.failure_mode == "INVALID_SHAPE":
            window = np.zeros((299,), dtype=np.float64)
        elif self.failure_mode == "NAN":
            window = window.astype(np.float64, copy=True).reshape(-1)
            window[0] = np.nan
        elif self.failure_mode == "INF":
            window = window.astype(np.float64, copy=True).reshape(-1)
            window[0] = np.inf

        try:
            prediction = self.interpreter.predict(window)
        except ValueError as exc:
            message = str(exc)
            if self.failure_mode == "INVALID_SHAPE":
                code = "INVALID_SHAPE"
            elif self.failure_mode == "NAN":
                code = "NAN_OR_INF"
            elif self.failure_mode == "INF":
                code = "NAN_OR_INF"
            else:
                code = message or "M-B9_RUNTIME_PREDICTION_ERROR"
            return self._invalid(now, code, state=code)
        except Exception as exc:
            return self._invalid(now, "M-B9_RUNTIME_PREDICTION_ERROR", state="INFER_ERROR", metadata={"detail": str(exc)})

        if self.failure_mode == "STALE":
            now = now - (self.stale_sec + 1.0)

        trace = getattr(self.interpreter, "last_preprocess_trace", {})
        raw_output = np.asarray(getattr(self.interpreter, "last_raw_output", []))
        quantized_input = np.asarray(trace.get("quantized_input", []))
        class_index = int(prediction.class_index)
        state = prediction.class_name
        score = CLASS_SCORE.get(class_index)
        if score is None:
            return self._invalid(now, "M-B9_UNKNOWN_MODEL_CLASS", state="UNKNOWN_CLASS")
        metadata = {
            "model_id": prediction.model_id,
            "model_version": prediction.model_version,
            "model_sha256": prediction.model_sha256,
            "class_index": class_index,
            "model_predicted_class": state,
            "probabilities": list(prediction.probabilities),
            "fallback_used": bool(prediction.fallback_used),
            "fallback_reason": prediction.fallback_reason,
            "preprocessing_profile": prediction.preprocessing_profile,
            "score_source": "MODEL_PREDICTION",
            "scenario_truth_class": self.scenario_truth_class,
            "scenario_truth_source": "METADATA_ONLY",
            "selection_metadata": self.selection_metadata,
            "input_quantized_dtype": quantized_input.dtype.name,
            "input_quantized_shape": list(quantized_input.shape),
            "input_quantized_sha256": hashlib.sha256(np.ascontiguousarray(quantized_input).tobytes()).hexdigest(),
            "input_saturation_count": int(trace.get("input_saturation_count", 0)),
            "output_int8_dtype": raw_output.dtype.name if raw_output.size else None,
            "output_int8_shape": list(raw_output.shape),
            "output_int8_sha256": hashlib.sha256(np.ascontiguousarray(raw_output).tobytes()).hexdigest() if raw_output.size else None,
        }
        result = InferenceResult(
            sensor_id="mmwave",
            timestamp=now,
            score=float(score),
            state=state,
            confidence=float(prediction.confidence),
            valid=True,
            latency_ms=float((time.perf_counter() - started) * 1000.0),
            error=None,
            metadata=metadata,
        )
        self.current_state = SensorState.NORMAL
        self.last_result = result
        return result

    def close(self) -> None:
        self.connected = False
        self.current_state = SensorState.SHUTDOWN
