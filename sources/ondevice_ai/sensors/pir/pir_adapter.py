#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sensors/pir/pir_adapter.py
Hardware PIR Motion Sensor Raspberry Pi GPIO Adapter
"""

from __future__ import annotations
import time
from sensors.base_sensor import BaseSensor, SensorState, HardwareBackendUnavailable
from inference.inference_result import InferenceResult



class PIRSensorAdapter(BaseSensor):
    def __init__(
        self,
        gpio_pin: int = 17,
        no_motion_threshold_sec: float = 15.0,
        startup_grace_period_sec: float = 5.0,
        timeout_sec: float = 5.0,
        stale_sec: float = 10.0,
    ):
        super().__init__(sensor_id="pir", timeout_sec=timeout_sec, stale_sec=stale_sec)
        self.gpio_pin = gpio_pin
        self.no_motion_threshold_sec = no_motion_threshold_sec
        self.startup_grace_period_sec = startup_grace_period_sec
        self.connect_monotonic_ts: float | None = None
        self.last_motion_monotonic_ts: float | None = None
        self.has_observation: bool = False
        self.has_motion_event: bool = False

    def connect(self) -> bool:
        self.connect_monotonic_ts = time.monotonic()
        self.last_motion_monotonic_ts = None
        self.has_observation = False
        self.has_motion_event = False
        raise HardwareBackendUnavailable(
            "Real PIR GPIO backend is not installed"
        )

    def read_gpio(self) -> bool:
        raise HardwareBackendUnavailable(
            "Real PIR GPIO backend is not installed"
        )

    def read(self) -> InferenceResult:
        t0 = time.perf_counter()
        now = time.time()
        now_mono = time.monotonic()
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
            motion_detected = self.read_gpio()
            self.has_observation = True
            if motion_detected:
                self.has_motion_event = True
                self.last_motion_monotonic_ts = now_mono

            if not self.has_motion_event:
                start_ts = self.connect_monotonic_ts if self.connect_monotonic_ts is not None else now_mono
                elapsed_since_start = now_mono - start_ts
                if elapsed_since_start < self.startup_grace_period_sec:
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
                            "elapsed_since_start_sec": elapsed_since_start,
                            "startup_grace_period_sec": self.startup_grace_period_sec
                        }
                    )
                elapsed = elapsed_since_start
            else:
                elapsed = now_mono - (self.last_motion_monotonic_ts if self.last_motion_monotonic_ts is not None else now_mono)

            is_no_motion = elapsed >= self.no_motion_threshold_sec
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
                    "gpio_pin": self.gpio_pin,
                    "motion_detected": motion_detected,
                    "elapsed_since_motion_sec": elapsed
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
        self.connect_monotonic_ts = None
        self.last_motion_monotonic_ts = None
        self.has_observation = False
        self.has_motion_event = False
        self.current_state = SensorState.SHUTDOWN
