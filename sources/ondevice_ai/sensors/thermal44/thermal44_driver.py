#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sensors/thermal44/thermal44_driver.py
Thermal-44 (80x62 IR Array) Hardware Driver & SPI/I2C Adapter
"""

from __future__ import annotations
import time
from pathlib import Path
import numpy as np

from sensors.base_sensor import BaseSensor, SensorState, HardwareBackendUnavailable
from sensors.thermal44.frame_parser import ThermalFrameParser
from inference.inference_result import InferenceResult
from inference.thermal_interpreter import ThermalInterpreter, ThermalPrediction


class Thermal44Sensor(BaseSensor):
    def __init__(
        self,
        project_root: str | Path | None = None,
        manifest_path: str = "models/model_manifest.json",
        timeout_sec: float = 2.0,
        stale_sec: float = 3.0,
        sample_rate_hz: float = 10.0,
    ):
        super().__init__(sensor_id="thermal44", timeout_sec=timeout_sec, stale_sec=stale_sec)
        self.sample_rate_hz = sample_rate_hz
        self.interpreter = ThermalInterpreter(project_root=project_root, manifest_path=manifest_path)

    def connect(self) -> bool:
        raise HardwareBackendUnavailable(
            "Real Thermal-44 SPI/I2C driver is not installed"
        )

    def read_frame(self) -> np.ndarray:
        raise HardwareBackendUnavailable(
            "Real Thermal-44 SPI/I2C driver is not installed"
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
            frame_62x80 = self.read_frame()
            if frame_62x80 is None:
                self.current_state = SensorState.WARMING_UP
                return InferenceResult(
                    sensor_id=self.sensor_id,
                    timestamp=now,
                    score=0.0,
                    state="WARMING_UP",
                    confidence=0.0,
                    valid=False,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    error="WAITING_FOR_FIRST_FRAME"
                )

            if frame_62x80.shape != (62, 80):
                self.current_state = SensorState.INVALID_FORMAT
                return InferenceResult(
                    sensor_id=self.sensor_id,
                    timestamp=now,
                    score=0.0,
                    state="INVALID_FORMAT",
                    confidence=0.0,
                    valid=False,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    error="INVALID_FRAME_SHAPE"
                )

            if np.isnan(frame_62x80).any() or np.isinf(frame_62x80).any():
                self.current_state = SensorState.NAN_OR_INF
                return InferenceResult(
                    sensor_id=self.sensor_id,
                    timestamp=now,
                    score=0.0,
                    state="NAN_OR_INF",
                    confidence=0.0,
                    valid=False,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    error="NAN_OR_INF_FRAME"
                )

            pred: ThermalPrediction = self.interpreter.predict(frame_62x80)
            score = 1.0 if pred.class_index == 2 else 0.0
            self.current_state = SensorState.NORMAL

            return InferenceResult(
                sensor_id=self.sensor_id,
                timestamp=now,
                score=score,
                state="HUMAN_FALL" if score == 1.0 else pred.class_name,
                confidence=pred.confidence,
                valid=True,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                metadata={
                    "model_id": pred.model_id,
                    "class_index": pred.class_index,
                    "probabilities": pred.probabilities,
                    "infer_latency_ms": pred.latency_ms
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
        self.current_state = SensorState.SHUTDOWN
