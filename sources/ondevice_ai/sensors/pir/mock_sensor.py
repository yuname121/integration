#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sensors/pir/mock_sensor.py
Mock PIR Motion Sensor Adapter for Simulation / Mac Testing
"""

from __future__ import annotations
import time
from sensors.base_sensor import BaseSensor, SensorState
from inference.inference_result import InferenceResult


class MockPIRSensor(BaseSensor):
    def __init__(
        self,
        no_motion_threshold_sec: float = 15.0,
        timeout_sec: float = 5.0,
        stale_sec: float = 10.0,
    ):
        super().__init__(sensor_id="pir", timeout_sec=timeout_sec, stale_sec=stale_sec)
        self.no_motion_threshold_sec = no_motion_threshold_sec
        self.last_motion_ts = time.time()
        self.simulated_scenario = "MOTION"  # "MOTION", "NO_MOTION", "STUCK_FAULT"

    def connect(self) -> bool:
        self.connected = True
        self.current_state = SensorState.NORMAL
        return True

    def set_scenario(self, scenario: str) -> None:
        self.simulated_scenario = scenario
        if scenario == "MOTION":
            self.last_motion_ts = time.time()
        elif scenario == "NO_MOTION":
            self.last_motion_ts = time.time() - (self.no_motion_threshold_sec + 5.0)

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

        if self.simulated_scenario == "STUCK_FAULT":
            self.current_state = SensorState.INVALID_FORMAT
            self.error_count += 1
            return InferenceResult(
                sensor_id=self.sensor_id,
                timestamp=now,
                score=0.0,
                state="STUCK_FAULT",
                confidence=0.0,
                valid=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error="PIR_GPIO_STUCK_HIGH_OR_LOW_FAULT"
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
                error="PIR_WARMING_UP",
                metadata={
                    "no_motion_threshold_sec": self.no_motion_threshold_sec
                }
            )

        elapsed_since_motion = now - self.last_motion_ts
        is_no_motion = elapsed_since_motion >= self.no_motion_threshold_sec

        # PIR score S3: 1.0 if long no motion detected, 0.0 if motion active
        score = 1.0 if is_no_motion else 0.0
        state_str = "LONG_NO_MOTION" if is_no_motion else "MOTION"
        self.current_state = SensorState.NORMAL

        return InferenceResult(
            sensor_id=self.sensor_id,
            timestamp=now,
            score=score,
            state=state_str,
            confidence=1.0,
            valid=True,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            metadata={
                "elapsed_since_motion_sec": elapsed_since_motion,
                "no_motion_threshold_sec": self.no_motion_threshold_sec
            }
        )

    def close(self) -> None:
        self.connected = False
        self.current_state = SensorState.SHUTDOWN
