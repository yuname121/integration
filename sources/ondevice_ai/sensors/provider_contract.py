#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""External sensor-provider contract and fail-closed result validation."""

from __future__ import annotations

from numbers import Real
from typing import Any, Protocol, runtime_checkable
import math
import time

from inference.inference_result import InferenceResult


EXPECTED_PROVIDER_IDS = {
    "thermal44": "thermal44",
    "mmwave": "mmwave",
    "co2": "co2",
    "pir": "pir",
}


@runtime_checkable
class SensorProvider(Protocol):
    def connect(self) -> bool:
        ...

    def read(self) -> InferenceResult:
        ...

    def close(self) -> None:
        ...


def invalid_provider_result(
    sensor_id: str,
    error: str,
    *,
    state: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> InferenceResult:
    return InferenceResult(
        sensor_id=sensor_id,
        timestamp=time.time(),
        score=0.0,
        state=state or error,
        confidence=0.0,
        valid=False,
        latency_ms=0.0,
        error=error,
        metadata=dict(metadata or {}),
    )


class MissingExternalSensorProvider:
    """Explicit sentinel used when real mode has no team-owned provider."""

    def __init__(self, sensor_id: str, runtime_settings: dict[str, Any]):
        self.sensor_id = sensor_id
        self.runtime_settings = dict(runtime_settings)

    def connect(self) -> bool:
        return False

    def read(self) -> InferenceResult:
        return invalid_provider_result(
            self.sensor_id,
            "EXTERNAL_SENSOR_PROVIDER_REQUIRED",
            metadata={"mode": "real", "provider_injected": False},
        )

    def close(self) -> None:
        return None


def validate_provider_interface(provider: object, sensor_key: str) -> None:
    if sensor_key not in EXPECTED_PROVIDER_IDS:
        raise ValueError(f"Unknown sensor provider key: {sensor_key!r}")
    missing = [
        method
        for method in ("connect", "read", "close")
        if not callable(getattr(provider, method, None))
    ]
    if missing:
        raise TypeError(
            f"Provider {sensor_key!r} is missing callable methods: {missing}"
        )


def _has_non_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return False
    if isinstance(value, Real):
        return not math.isfinite(float(value))
    if isinstance(value, dict):
        return any(_has_non_finite(k) or _has_non_finite(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_non_finite(item) for item in value)
    return False


def validate_provider_result(
    result: object,
    sensor_key: str,
) -> tuple[bool, str | None]:
    expected_id = EXPECTED_PROVIDER_IDS[sensor_key]
    if not isinstance(result, InferenceResult):
        return False, "PROVIDER_RESULT_TYPE_INVALID"
    if result.sensor_id != expected_id:
        return False, "PROVIDER_SENSOR_ID_MISMATCH"
    if result.timestamp < 0.0 or not math.isfinite(float(result.timestamp)):
        return False, "PROVIDER_TIMESTAMP_INVALID"
    if result.latency_ms < 0.0 or not math.isfinite(float(result.latency_ms)):
        return False, "PROVIDER_LATENCY_INVALID"
    if _has_non_finite(result.metadata):
        return False, "PROVIDER_METADATA_NON_FINITE"
    if not result.valid and not result.error:
        return False, "PROVIDER_INVALID_RESULT_MISSING_ERROR"
    return True, None
