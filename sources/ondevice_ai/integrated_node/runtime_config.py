#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Validated V5 runtime settings loaded from ``config/sensors.yaml``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import math

import yaml


EXPECTED_SENSOR_IDS = {
    "thermal44": "thermal44",
    "mmwave": "mmwave",
    "co2": "co2",
    "pir": "pir",
}


class RuntimeConfigError(ValueError):
    """Raised when configured and effective runtime settings cannot agree."""


@dataclass(frozen=True)
class SensorRuntimeSettings:
    key: str
    sensor_id: str
    enabled: bool
    stale_sec: float
    timeout_sec: float
    sample_rate_hz: float | None = None
    window_samples: int | None = None
    window_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NodeRuntimeSettings:
    loop_interval_sec: float
    sensors: dict[str, SensorRuntimeSettings]
    source_path: str = "config/sensors.yaml"

    @property
    def stale_by_sensor(self) -> dict[str, float]:
        return {
            ("thermal" if key == "thermal44" else key): settings.stale_sec
            for key, settings in self.sensors.items()
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "source": self.source_path,
            "loop_interval_sec": self.loop_interval_sec,
            "sensors": {
                key: settings.to_dict()
                for key, settings in self.sensors.items()
            },
        }


def _positive_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise RuntimeConfigError(f"{field} must be a positive finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(f"{field} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeConfigError(f"{field} must be a positive finite number")
    return number


def _optional_positive_float(value: Any, field: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, field)


def _optional_positive_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise RuntimeConfigError(f"{field} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(f"{field} must be a positive integer") from exc
    if number <= 0 or float(value) != float(number):
        raise RuntimeConfigError(f"{field} must be a positive integer")
    return number


def load_runtime_settings(
    project_root: str | Path,
    config_path: str | Path = "config/sensors.yaml",
) -> NodeRuntimeSettings:
    root = Path(project_root).resolve()
    path_obj = Path(config_path)
    if path_obj.is_absolute():
        config_abs = path_obj.resolve()
        try:
            source_path = config_abs.relative_to(root).as_posix()
        except ValueError as exc:
            raise RuntimeConfigError("Sensor config must be inside the project root") from exc
    else:
        config_abs = (root / path_obj).resolve()
        source_path = path_obj.as_posix()

    if not config_abs.is_file():
        raise RuntimeConfigError(f"Sensor config not found: {source_path}")

    raw = yaml.safe_load(config_abs.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise RuntimeConfigError("Sensor config root must be a mapping")

    system = raw.get("system")
    sensor_map = raw.get("sensors")
    if not isinstance(system, Mapping) or not isinstance(sensor_map, Mapping):
        raise RuntimeConfigError("Sensor config requires system and sensors mappings")

    loop_interval_sec = _positive_float(
        system.get("loop_interval_sec"),
        "system.loop_interval_sec",
    )

    unknown = sorted(set(sensor_map) - set(EXPECTED_SENSOR_IDS))
    missing = sorted(set(EXPECTED_SENSOR_IDS) - set(sensor_map))
    if unknown or missing:
        raise RuntimeConfigError(
            f"Sensor config key mismatch: missing={missing}, unknown={unknown}"
        )

    parsed: dict[str, SensorRuntimeSettings] = {}
    for key, expected_id in EXPECTED_SENSOR_IDS.items():
        item = sensor_map[key]
        if not isinstance(item, Mapping):
            raise RuntimeConfigError(f"sensors.{key} must be a mapping")
        if item.get("sensor_id") != expected_id:
            raise RuntimeConfigError(
                f"sensors.{key}.sensor_id must be {expected_id!r}"
            )
        if not isinstance(item.get("enabled"), bool):
            raise RuntimeConfigError(f"sensors.{key}.enabled must be boolean")

        sample_rate_hz = _optional_positive_float(
            item.get("sample_rate_hz"),
            f"sensors.{key}.sample_rate_hz",
        )
        window_samples = _optional_positive_int(
            item.get("window_samples"),
            f"sensors.{key}.window_samples",
        )
        window_seconds = _optional_positive_float(
            item.get("window_seconds"),
            f"sensors.{key}.window_seconds",
        )

        settings = SensorRuntimeSettings(
            key=key,
            sensor_id=expected_id,
            enabled=item["enabled"],
            stale_sec=_positive_float(item.get("stale_sec"), f"sensors.{key}.stale_sec"),
            timeout_sec=_positive_float(item.get("timeout_sec"), f"sensors.{key}.timeout_sec"),
            sample_rate_hz=sample_rate_hz,
            window_samples=window_samples,
            window_seconds=window_seconds,
        )
        if sample_rate_hz is not None and settings.stale_sec < (1.0 / sample_rate_hz):
            raise RuntimeConfigError(
                f"sensors.{key}.stale_sec is shorter than one configured sample period"
            )
        if key == "mmwave":
            if None in (sample_rate_hz, window_samples, window_seconds):
                raise RuntimeConfigError(
                    "mmWave requires sample_rate_hz, window_samples, and window_seconds"
                )
            calculated_window = window_samples / sample_rate_hz
            if not math.isclose(calculated_window, window_seconds, abs_tol=1e-6):
                raise RuntimeConfigError(
                    "mmWave window_samples/sample_rate_hz must equal window_seconds"
                )
        parsed[key] = settings

    return NodeRuntimeSettings(
        loop_interval_sec=loop_interval_sec,
        sensors=parsed,
        source_path=source_path,
    )


def validate_provider_settings(
    provider: object,
    settings: SensorRuntimeSettings,
) -> None:
    """Fail startup when a provider explicitly declares conflicting settings."""

    declared = getattr(provider, "runtime_settings", None)
    for field in (
        "stale_sec",
        "timeout_sec",
        "sample_rate_hz",
        "window_samples",
        "window_seconds",
    ):
        expected = getattr(settings, field)
        if expected is None:
            continue
        if isinstance(declared, Mapping) and field in declared:
            actual = declared[field]
        elif hasattr(provider, field):
            actual = getattr(provider, field)
        else:
            continue
        if isinstance(expected, float):
            try:
                matches = math.isclose(float(actual), expected, abs_tol=1e-9)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            raise RuntimeConfigError(
                f"Provider {settings.key!r} {field} mismatch: "
                f"config={expected!r}, provider={actual!r}"
            )
