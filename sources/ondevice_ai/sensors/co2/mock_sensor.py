#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sensors/co2/mock_sensor.py
Mock CO2 Sensor Adapter for Simulation / Mac Testing
"""

from __future__ import annotations
import time
from pathlib import Path
import numpy as np

from sensors.base_sensor import BaseSensor, SensorState
from inference.inference_result import InferenceResult
from inference.co2_interpreter import CO2Interpreter, CO2Prediction


class MockCO2Sensor(BaseSensor):
    def __init__(
        self,
        project_root: str | Path | None = None,
        manifest_path: str = "models/model_manifest.json",
        timeout_sec: float = 5.0,
        stale_sec: float = 10.0,
        sample_rate_hz: float = 0.2,
        window_samples: int = 30,
        window_seconds: float = 150.0,
    ):
        super().__init__(sensor_id="co2", timeout_sec=timeout_sec, stale_sec=stale_sec)
        self.sample_rate_hz = sample_rate_hz
        self.window_samples = window_samples
        self.window_seconds = window_seconds
        self.interpreter = CO2Interpreter(project_root=project_root, manifest_path=manifest_path)
        self.simulated_scenario = "NORMAL"  # "NORMAL", "ELEVATED", "FAULT"

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
                error="SIMULATED_CO2_SENSOR_FAULT"
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
                error="INSUFFICIENT_HISTORY",
                metadata={
                    "history_samples": 0,
                    "history_span_sec": 0.0,
                    "required_history_sec": 5.0
                }
            )

        # Feature vector: [CO2_slope (ppm/min), Humidity (%), CO2 (ppm)]
        if self.simulated_scenario == "ELEVATED":
            features = np.array([45.0, 65.0, 1850.0], dtype=np.float32)
        else: # NORMAL
            features = np.array([2.0, 42.0, 520.0], dtype=np.float32)

        try:
            if hasattr(self.interpreter, "predict"):
                pred: CO2Prediction = self.interpreter.predict(features[0], features[1], features[2])
                score = 1.0 if (self.simulated_scenario == "ELEVATED" or pred.class_index == 1) else 0.0
                state_str = "OCCUPIED_ELEVATED" if score == 1.0 else pred.class_name
                conf = pred.confidence
                probs = pred.probabilities
                model_id = pred.model_id
            else:
                score = 1.0 if self.simulated_scenario == "ELEVATED" else 0.0
                state_str = "OCCUPIED_ELEVATED" if score == 1.0 else "UNOCCUPIED_NORMAL"
                conf = 1.0
                probs = [1.0 - score, score]
                model_id = "mock"

            self.current_state = SensorState.NORMAL

            return InferenceResult(
                sensor_id=self.sensor_id,
                timestamp=now,
                score=score,
                state=state_str,
                confidence=conf,
                valid=True,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                metadata={
                    "model_id": model_id,
                    "probabilities": probs,
                    "features": features.tolist()
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
