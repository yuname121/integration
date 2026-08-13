#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sensors/thermal44/mock_sensor.py
Mock Thermal-44 Sensor Adapter for Mac/Simulation Environments
"""

from __future__ import annotations
import time
from pathlib import Path
import numpy as np

from sensors.base_sensor import BaseSensor, SensorState
from inference.inference_result import InferenceResult
from inference.thermal_interpreter import ThermalInterpreter, ThermalPrediction


class MockThermalSensor(BaseSensor):
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
        self.simulated_scenario = "NORMAL"  # "NORMAL", "FALL", "FAULT"

    def connect(self) -> bool:
        self.connected = True
        self.current_state = SensorState.NORMAL
        return True

    def set_scenario(self, scenario: str) -> None:
        self.simulated_scenario = scenario

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

        if self.simulated_scenario == "FAULT":
            self.current_state = SensorState.NAN_OR_INF
            self.error_count += 1
            return InferenceResult(
                sensor_id=self.sensor_id,
                timestamp=now,
                score=0.0,
                state="FAULT",
                confidence=0.0,
                valid=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error="SIMULATED_THERMAL_SENSOR_FAULT"
            )

        if self.simulated_scenario == "WARMING_UP":
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

        # Generate synthetic 80x62 frame based on scenario
        frame = np.full((62, 80), 22.0, dtype=np.float32)
        if self.simulated_scenario == "FALL":
            # High intensity fall blob in lower grid region
            frame[45:60, 20:60] = 34.5
        else: # NORMAL
            # Standing human blob in center
            frame[15:50, 30:50] = 33.0

        try:
            pred: ThermalPrediction = self.interpreter.predict(frame)
            score = 1.0 if (self.simulated_scenario == "FALL" or pred.class_index == 2) else 0.0
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
                    "simulated_scenario": self.simulated_scenario
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
