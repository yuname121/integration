#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SafeNest V5 standardized inference and risk output schemas."""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
import json
import math
import time


@dataclass(frozen=True)
class InferenceResult:
    sensor_id: str
    timestamp: float
    score: float           # Normalized risk score in range [0.0, 1.0]
    state: str             # State string (e.g. "NORMAL", "HUMAN_FALL", "APNEA", "ELEVATED", "MOTION")
    confidence: float      # Model confidence or rule reliability [0.0, 1.0]
    valid: bool            # True if sensor telemetry and inference are healthy
    latency_ms: float      # Total latency (preprocessing + inference + postprocessing)
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.sensor_id, str) or not self.sensor_id:
            raise ValueError("InferenceResult sensor_id must be a non-empty string")
        if not isinstance(self.state, str) or not self.state:
            raise ValueError("InferenceResult state must be a non-empty string")
        if not math.isfinite(float(self.timestamp)) or self.timestamp < 0.0:
            raise ValueError("InferenceResult timestamp must be finite Unix seconds")
        if not math.isfinite(float(self.score)) or not (0.0 <= self.score <= 1.0):
            raise ValueError(f"InferenceResult score must be between 0.0 and 1.0, got {self.score}")
        if not math.isfinite(float(self.confidence)) or not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"InferenceResult confidence must be between 0.0 and 1.0, got {self.confidence}")
        if not math.isfinite(float(self.latency_ms)) or self.latency_ms < 0.0:
            raise ValueError("InferenceResult latency_ms must be finite and non-negative")
        if not isinstance(self.valid, bool):
            raise ValueError("InferenceResult valid must be boolean")
        if not self.valid and not self.error:
            raise ValueError("InferenceResult valid=False requires a non-empty error code")
        if not isinstance(self.metadata, dict):
            raise ValueError("InferenceResult metadata must be a dictionary")

    def to_dict(self) -> dict:
        data = asdict(self)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class SafeNestRiskOutput:
    timestamp: float
    risk_score: float | None           # Human risk score [0.0, 100.0] or None if FAILED
    risk_level: str | None             # "NORMAL", "CAUTION", "DANGER" or None if FAILED
    system_health: str                 # "HEALTHY", "DEGRADED", "FAILED"
    degraded_mode: bool                # True if fallback used or system degraded/failed
    invalid_sensors: list[str] = field(default_factory=list)
    stale_sensors: list[str] = field(default_factory=list)
    component_scores: dict[str, float | None] = field(default_factory=dict)
    is_emergency: bool = False
    reasons: list[str] = field(default_factory=list)
    sensors: dict[str, dict] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    level: str | None = None           # Backward compatibility alias for risk_level
    system_status: str = ""            # Backward compatibility alias for system_health
    fallback_used: bool = False        # Backward compatibility alias for degraded_mode

    def __post_init__(self):
        if not math.isfinite(float(self.timestamp)) or self.timestamp < 0.0:
            raise ValueError("SafeNestRiskOutput timestamp must be finite Unix seconds")
        if self.risk_score is not None and (
            not math.isfinite(float(self.risk_score))
            or not (0.0 <= self.risk_score <= 100.0)
        ):
            raise ValueError("SafeNestRiskOutput risk_score must be null or in [0, 100]")
        metadata = dict(self.metadata)
        metadata.setdefault("schema_version", "5.0")
        object.__setattr__(self, "metadata", metadata)

        # Synchronize backward compatibility aliases if not set
        if self.level is None and self.risk_level is not None:
            object.__setattr__(self, "level", self.risk_level)
        elif self.level is None and self.system_health == "FAILED":
            object.__setattr__(self, "level", "FAULT")

        if not self.system_status:
            status_map = {"HEALTHY": "OK", "DEGRADED": "DEGRADED", "FAILED": "FAULT"}
            object.__setattr__(self, "system_status", status_map.get(self.system_health, self.system_health))

        if not self.fallback_used and self.degraded_mode:
            object.__setattr__(self, "fallback_used", True)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)
