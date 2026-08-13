"""Thread-safe latest publication and bounded PHASE 7 transition events."""

from __future__ import annotations

from collections import deque
import copy
import json
import math
import threading
import time
import uuid
from typing import Any, Mapping


class RuntimeStore:
    def __init__(self, event_capacity: int = 500) -> None:
        if isinstance(event_capacity, bool) or not isinstance(event_capacity, int) or event_capacity <= 0:
            raise ValueError("event_capacity must be a positive integer")
        self._lock = threading.RLock()
        self._latest: dict[str, Any] | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=event_capacity)
        self._publication_revision = 0
        self._event_sequence = 0
        self._last_error: dict[str, Any] | None = None

    def publish(
        self,
        state: Mapping[str, Any],
        ai: Mapping[str, Any],
        risk: Mapping[str, Any],
    ) -> dict[str, Any]:
        timestamp = _finite_timestamp(state.get("timestamp"))
        candidate = {
            "timestamp": timestamp,
            "state": copy.deepcopy(dict(state)),
            "ai": copy.deepcopy(dict(ai)),
            "risk": copy.deepcopy(dict(risk)),
        }
        _strict_json(candidate)
        with self._lock:
            previous = self._latest
            self._publication_revision += 1
            candidate["publication_revision"] = self._publication_revision
            self._latest = candidate
            self._record_transitions(previous, candidate)
            return copy.deepcopy(candidate)

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            return copy.deepcopy(self._latest)

    def events(self, limit: int = 100) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ValueError("event limit must be between 1 and 200")
        with self._lock:
            return copy.deepcopy(list(reversed(self._events))[:limit])

    def history(self, limit: int = 100) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ValueError("history limit must be between 1 and 200")
        latest = self.latest()
        return [] if latest is None else [latest]

    def record_runtime_error(self, source: str, error: Exception | str) -> dict[str, Any]:
        detail = str(error)
        with self._lock:
            timestamp = time.time()
            event = self._append_event(
                timestamp,
                "RUNTIME_ERROR",
                {"source": str(source), "detail": detail[:1000]},
            )
            self._last_error = copy.deepcopy(event)
            return copy.deepcopy(event)

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ready": self._latest is not None,
                "publication_revision": self._publication_revision,
                "event_count": len(self._events),
                "last_error": copy.deepcopy(self._last_error),
            }

    def _record_transitions(
        self,
        previous: Mapping[str, Any] | None,
        current: Mapping[str, Any],
    ) -> None:
        timestamp = float(current["timestamp"])
        current_risk = _mapping(current.get("risk"))
        if previous is None:
            self._append_event(
                timestamp,
                "SNAPSHOT_INITIALIZED",
                {
                    "risk_level": current_risk.get("risk_level"),
                    "system_health": current_risk.get("system_health"),
                },
            )
            return
        previous_risk = _mapping(previous.get("risk"))
        for field, event_type in (
            ("risk_level", "RISK_LEVEL_CHANGED"),
            ("system_health", "SYSTEM_HEALTH_CHANGED"),
        ):
            before = previous_risk.get(field)
            after = current_risk.get(field)
            if before != after:
                self._append_event(timestamp, event_type, {"from": before, "to": after})
        before_emergency = bool(previous_risk.get("is_emergency"))
        after_emergency = bool(current_risk.get("is_emergency"))
        if before_emergency != after_emergency:
            self._append_event(
                timestamp,
                "EMERGENCY_STARTED" if after_emergency else "EMERGENCY_CLEARED",
                {"is_emergency": after_emergency},
            )
        previous_sensors = _mapping(_mapping(previous.get("state")).get("sensors"))
        current_sensors = _mapping(_mapping(current.get("state")).get("sensors"))
        for sensor_id in ("mmwave", "thermal", "co2", "pir"):
            before = _mapping(previous_sensors.get(sensor_id)).get("status")
            after = _mapping(current_sensors.get(sensor_id)).get("status")
            if before != after:
                self._append_event(
                    timestamp,
                    "SENSOR_STATUS_CHANGED",
                    {"sensor_id": sensor_id, "from": before, "to": after},
                )

    def _append_event(
        self,
        timestamp: float,
        event_type: str,
        details: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._event_sequence += 1
        event = {
            "event_id": str(uuid.uuid4()),
            "sequence": self._event_sequence,
            "timestamp": float(timestamp),
            "event_type": event_type,
            "details": copy.deepcopy(dict(details)),
        }
        _strict_json(event)
        self._events.append(event)
        return event


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_timestamp(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError("publication timestamp must be finite and non-negative")
    return float(value)


def _strict_json(value: object) -> None:
    json.dumps(value, ensure_ascii=False, allow_nan=False)
