#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sensors/base_sensor.py
SafeNest V4 On-Device AI BaseSensor Contract & State Machine
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass
import time
from typing import Optional

from inference.inference_result import InferenceResult


class HardwareBackendUnavailable(RuntimeError):
    """Selected real sensor backend has no production hardware implementation."""


class SensorState(Enum):
    NORMAL = auto()
    NOT_CONNECTED = auto()
    READ_TIMEOUT = auto()
    INVALID_FORMAT = auto()
    NAN_OR_INF = auto()
    OUT_OF_BOUNDS = auto()
    INFER_FAILED = auto()
    STALE = auto()
    SHUTDOWN = auto()
    HARDWARE_BACKEND_NOT_IMPLEMENTED = auto()
    WARMING_UP = auto()


@dataclass
class SensorHealth:
    sensor_id: str
    connected: bool
    state: SensorState
    last_read_ts: float
    age_sec: float
    read_count: int
    error_count: int
    last_error: Optional[str] = None


class BaseSensor(ABC):
    def __init__(self, sensor_id: str, timeout_sec: float = 2.0, stale_sec: float = 5.0):
        self.sensor_id = sensor_id
        self.timeout_sec = timeout_sec
        self.stale_sec = stale_sec
        self.connected = False
        self.last_read_ts = 0.0
        self.read_count = 0
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.current_state = SensorState.NOT_CONNECTED

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to physical sensor hardware or mock interface."""
        pass

    @abstractmethod
    def read(self) -> InferenceResult:
        """Read sensor telemetry, execute preprocessing/inference, and return InferenceResult."""
        pass

    def health(self) -> SensorHealth:
        now = time.time()
        age = now - self.last_read_ts if self.last_read_ts > 0 else 9999.0
        if age > self.stale_sec and self.connected:
            self.current_state = SensorState.STALE

        return SensorHealth(
            sensor_id=self.sensor_id,
            connected=self.connected,
            state=self.current_state,
            last_read_ts=self.last_read_ts,
            age_sec=age,
            read_count=self.read_count,
            error_count=self.error_count,
            last_error=self.last_error
        )

    def backend_unavailable_result(self, error: str) -> InferenceResult:
        now = time.time()

        self.connected = False
        self.current_state = SensorState.HARDWARE_BACKEND_NOT_IMPLEMENTED
        self.last_error = error
        self.error_count += 1

        return InferenceResult(
            sensor_id=self.sensor_id,
            timestamp=now,
            score=0.0,
            state="HARDWARE_BACKEND_NOT_IMPLEMENTED",
            confidence=0.0,
            valid=False,
            latency_ms=0.0,
            error=error,
            metadata={
                "mode": "real",
                "backend_available": False,
            },
        )

    def warming_up_result(self, error: str = "INSUFFICIENT_HISTORY", metadata: Optional[dict] = None) -> InferenceResult:
        now = time.time()
        self.current_state = SensorState.WARMING_UP
        self.last_error = error

        meta = {"warming_up": True}
        if metadata:
            meta.update(metadata)

        return InferenceResult(
            sensor_id=self.sensor_id,
            timestamp=now,
            score=0.0,
            state="WARMING_UP",
            confidence=0.0,
            valid=False,
            latency_ms=0.0,
            error=error,
            metadata=meta
        )

    @abstractmethod
    def close(self) -> None:
        """Safely release SPI, I2C, Serial UART, GPIO, or memory resources."""
        pass
