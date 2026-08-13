#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sensors/mmwave/mmwave_adapter.py
Hardware mmWave Radar (Seeed Studio MR60BHA2 60GHz) UART Serial Adapter
"""

from __future__ import annotations
import time
from pathlib import Path
from collections import deque
import numpy as np

from sensors.base_sensor import BaseSensor, SensorState, HardwareBackendUnavailable
from inference.inference_result import InferenceResult
from inference.mmwave_interpreter import MMWaveInterpreter, MMWavePrediction


class MMWaveSensorAdapter(BaseSensor):
    def __init__(
        self,
        port: str = "EXTERNAL_SENSOR_PROVIDER_REQUIRED",
        baudrate: int = 115200,
        project_root: str | Path | None = None,
        manifest_path: str = "models/model_manifest.json",
        timeout_sec: float = 2.0,
        stale_sec: float = 3.0,
        sample_rate_hz: float = 10.0,
        window_samples: int = 300,
        window_seconds: float = 30.0,
    ):
        super().__init__(sensor_id="mmwave", timeout_sec=timeout_sec, stale_sec=stale_sec)
        self.port = port
        self.baudrate = baudrate
        self.sample_rate_hz = sample_rate_hz
        self.window_samples = window_samples
        self.window_seconds = window_seconds
        self.interpreter = MMWaveInterpreter(project_root=project_root, manifest_path=manifest_path)
        self.ring_buffer = deque(maxlen=window_samples)
        self.last_ts: float | None = None

    def connect(self) -> bool:
        self.ring_buffer.clear()
        self.last_ts = None
        raise HardwareBackendUnavailable(
            "Real MR60BHA2 UART backend is not installed"
        )

    def push_sample(self, phase_val: float, timestamp_s: float) -> bool:
        if np.isnan(phase_val) or np.isinf(phase_val):
            self.current_state = SensorState.NAN_OR_INF
            self.last_error = "NAN_OR_INF_SAMPLE"
            return False
        if self.last_ts is not None and timestamp_s <= self.last_ts:
            self.current_state = SensorState.INVALID_FORMAT
            self.last_error = "NON_MONOTONIC_TIMESTAMP"
            return False  # Non-monotonic or duplicate timestamp
        self.ring_buffer.append(float(phase_val))
        self.last_ts = timestamp_s
        return True

    def read(self) -> InferenceResult:
        t0 = time.perf_counter()
        now = time.time()
        self.read_count += 1
        self.last_read_ts = now

        if not self.connected:
            self.current_state = SensorState.NOT_CONNECTED
            return InferenceResult(
                sensor_id=self.sensor_id,
                timestamp=now,
                score=0.0,
                state="NOT_CONNECTED",
                confidence=0.0,
                valid=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error="SENSOR_NOT_CONNECTED"
            )

        if len(self.ring_buffer) < self.window_samples:
            self.current_state = SensorState.WARMING_UP
            return InferenceResult(
                sensor_id=self.sensor_id,
                timestamp=now,
                score=0.0,
                state="WARMING_UP",
                confidence=0.0,
                valid=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error="INSUFFICIENT_HISTORY",
                metadata={
                    "buffer_len": len(self.ring_buffer),
                    "required_samples": self.window_samples,
                    "remaining_samples": self.window_samples - len(self.ring_buffer)
                }
            )

        window = np.array(self.ring_buffer, dtype=np.float32)
        if np.isnan(window).any() or np.isinf(window).any():
            self.current_state = SensorState.NAN_OR_INF
            return InferenceResult(
                sensor_id=self.sensor_id,
                timestamp=now,
                score=0.0,
                state="NAN_OR_INF",
                confidence=0.0,
                valid=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error="NAN_OR_INF_IN_BUFFER"
            )

        try:
            pred: MMWavePrediction = self.interpreter.predict(window)
            if pred.class_index == 2:
                score = 1.0
                state_str = "APNEA"
            elif pred.class_index == 1:
                score = 0.5
                state_str = "RAPID_OR_ABNORMAL"
            else:
                score = 0.0
                state_str = "NORMAL"

            self.current_state = SensorState.NORMAL

            return InferenceResult(
                sensor_id=self.sensor_id,
                timestamp=now,
                score=score,
                state=state_str,
                confidence=pred.confidence,
                valid=True,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                metadata={
                    "model_id": pred.model_id,
                    "class_index": pred.class_index,
                    "probabilities": pred.probabilities,
                    "buffer_len": len(self.ring_buffer)
                }
            )
        except Exception as exc:
            self.error_count += 1
            self.current_state = SensorState.INFER_FAILED
            return InferenceResult(
                sensor_id=self.sensor_id,
                timestamp=now,
                score=0.0,
                state="INFER_ERROR",
                confidence=0.0,
                valid=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error=str(exc)
            )

    def close(self) -> None:
        self.connected = False
        self.ring_buffer.clear()
        self.last_ts = None
        self.current_state = SensorState.SHUTDOWN
