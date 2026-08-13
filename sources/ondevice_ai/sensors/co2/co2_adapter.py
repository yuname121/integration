#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sensors/co2/co2_adapter.py
Hardware SCD40 CO2 / Temperature / Humidity Sensor I2C Adapter
"""

from __future__ import annotations
import time
from pathlib import Path
from collections import deque
import numpy as np

from sensors.base_sensor import BaseSensor, SensorState, HardwareBackendUnavailable
from inference.inference_result import InferenceResult
from inference.co2_interpreter import CO2Interpreter, CO2Prediction


class CO2SensorAdapter(BaseSensor):
    def __init__(
        self,
        i2c_bus: int = 1,
        address: int = 0x62,
        project_root: str | Path | None = None,
        manifest_path: str = "models/model_manifest.json",
        timeout_sec: float = 5.0,
        stale_sec: float = 10.0,
        sample_rate_hz: float = 0.2,
        window_samples: int = 30,
        window_seconds: float = 150.0,
    ):
        super().__init__(sensor_id="co2", timeout_sec=timeout_sec, stale_sec=stale_sec)
        self.i2c_bus = i2c_bus
        self.address = address
        self.sample_rate_hz = sample_rate_hz
        self.window_samples = window_samples
        self.window_seconds = window_seconds
        self.interpreter = CO2Interpreter(project_root=project_root, manifest_path=manifest_path)
        self.co2_history = deque(maxlen=window_samples)  # Timestamp & ppm history for slope calculation

    def connect(self) -> bool:
        self.co2_history.clear()
        raise HardwareBackendUnavailable(
            "Real SCD40 I2C driver is not installed"
        )

    def calculate_co2_slope(self, current_ts: float, current_ppm: float, required_history_sec: float = 5.0) -> tuple[float | None, str | None]:
        if len(self.co2_history) > 0:
            last_ts, _ = self.co2_history[-1]
            if current_ts <= last_ts:
                return None, "NON_MONOTONIC_TIMESTAMP"

        self.co2_history.append((current_ts, current_ppm))
        if len(self.co2_history) < 2:
            return None, "INSUFFICIENT_HISTORY"

        ts_first, ppm_first = self.co2_history[0]
        elapsed_sec = current_ts - ts_first
        if elapsed_sec < required_history_sec:
            return None, "INSUFFICIENT_HISTORY"

        elapsed_min = elapsed_sec / 60.0
        slope = float((current_ppm - ppm_first) / elapsed_min)
        return slope, None

    def read_raw_values(self) -> tuple[float, float, float]:
        raise HardwareBackendUnavailable(
            "Real SCD40 I2C driver is not installed"
        )

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

        try:
            co2_ppm, humidity, _ = self.read_raw_values()
            required_history_sec = 5.0
            co2_slope, err_code = self.calculate_co2_slope(now, co2_ppm, required_history_sec=required_history_sec)

            if err_code == "NON_MONOTONIC_TIMESTAMP":
                self.current_state = SensorState.INVALID_FORMAT
                return InferenceResult(
                    sensor_id=self.sensor_id,
                    timestamp=now,
                    score=0.0,
                    state="INVALID_FORMAT",
                    confidence=0.0,
                    valid=False,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    error="NON_MONOTONIC_TIMESTAMP"
                )

            if err_code == "INSUFFICIENT_HISTORY" or co2_slope is None:
                self.current_state = SensorState.WARMING_UP
                ts_first = self.co2_history[0][0] if len(self.co2_history) > 0 else now
                current_span = now - ts_first
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
                        "history_samples": len(self.co2_history),
                        "history_span_sec": current_span,
                        "required_history_sec": required_history_sec
                    }
                )

            pred: CO2Prediction = self.interpreter.predict(co2_slope, humidity, co2_ppm)
            score = 1.0 if (pred.class_index == 1 or co2_ppm > 1500.0) else 0.0
            self.current_state = SensorState.NORMAL

            return InferenceResult(
                sensor_id=self.sensor_id,
                timestamp=now,
                score=score,
                state="OCCUPIED_ELEVATED" if score == 1.0 else pred.class_name,
                confidence=pred.confidence,
                valid=True,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                metadata={
                    "model_id": pred.model_id,
                    "class_index": pred.class_index,
                    "probabilities": pred.probabilities,
                    "co2_ppm": co2_ppm,
                    "co2_slope_ppm_min": co2_slope
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
        self.co2_history.clear()
        self.current_state = SensorState.SHUTDOWN
