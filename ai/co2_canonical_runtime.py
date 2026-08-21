"""Live CO2_slope reconstruction for the C-B6 reduced-feature runtime.

Implements ``CO2_SLOPE_FEATURE_PROFILE_001``
(``sources/ondevice_ai/models/rp_x0_b_complete/co2/co2_slope_feature_profile.json``)
exactly as declared:

* ``feature_unit``           ppm/min
* ``slope_method``           ENDPOINT_DIFFERENCE
* ``formula``                (co2_now - co2_history_start) / (elapsed_s / 60.0)
* endpoint selection         earliest past sample whose age >= history 150 s
* ``causality``              PAST_ONLY - no future or centred windows
* ``timestamp_basis``        SOURCE_ACQUISITION_CLOCK, i.e. the ESP's
                             ``co2_measurement_monotonic_ms`` physical
                             measurement clock, never the Pi wall clock
* ``max_internal_gap_seconds`` 90 s, ``gap_policy`` RESTART_HISTORY_AFTER_FORBIDDEN_GAP
* ``interpolation_allowed``  false
* ``calculation_precision``  float64
* ``nonfinite_policy``       FAIL_CLOSED_STATUS_NO_CANONICAL_SLOPE

Status codes are the profile's own vocabulary, so an unavailable slope is never
silently reported as 0.0 ppm/min:

* ``CO2_SLOPE_READY``
* ``FEATURE_UNAVAILABLE_WARMUP``          (warm_up_status)
* ``FEATURE_UNAVAILABLE_GAP_RESTART``     (gap_restart_status)
* ``NO_CANONICAL_SLOPE``                  (nonfinite_policy)
* ``CO2_MEASUREMENT_CLOCK_UNAVAILABLE``   (no source clock to anchor on)

Only physical measurement events advance the history. The runtime republishes the
last CO2 reading on every telemetry packet and additionally throttles the
presentation value to once per minute, so keying on anything other than
``measurement_event_id`` would fabricate a flat slope out of repeated values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import threading
from typing import Any, Mapping

PROFILE_PATH = (
    Path(__file__).resolve().parent.parent
    / "sources/ondevice_ai/models/rp_x0_b_complete/co2/co2_slope_feature_profile.json"
)
PROFILE_ID = "CO2_SLOPE_FEATURE_PROFILE_001"


@dataclass(frozen=True)
class CO2SlopeResult:
    status: str
    reason: str | None
    ppm: float | None
    slope_ppm_per_min: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "CO2_SLOPE_READY"


def _load_profile() -> dict[str, Any]:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if profile.get("profile_id") != PROFILE_ID:
        raise ValueError("CO2_SLOPE_PROFILE_IDENTITY_MISMATCH")
    if profile.get("feature_unit") != "ppm/min":
        raise ValueError("CO2_SLOPE_PROFILE_UNIT_MISMATCH")
    if profile.get("slope_method") != "ENDPOINT_DIFFERENCE":
        raise ValueError("CO2_SLOPE_PROFILE_METHOD_MISMATCH")
    if profile.get("causality") != "PAST_ONLY":
        raise ValueError("CO2_SLOPE_PROFILE_CAUSALITY_MISMATCH")
    if profile.get("interpolation_allowed") is not False:
        raise ValueError("CO2_SLOPE_PROFILE_INTERPOLATION_MISMATCH")
    if profile.get("future_samples_allowed") is not False:
        raise ValueError("CO2_SLOPE_PROFILE_FUTURE_SAMPLES_MISMATCH")
    return profile


class CO2SlopeWindowBuilder:
    """Accumulate CO2 measurement events and derive the canonical ppm/min slope.

    ``observe`` runs on the receiver thread while ``latest`` is read by the
    publication thread, so the history is lock-protected.
    """

    def __init__(self, profile: Mapping[str, Any] | None = None) -> None:
        self._profile = dict(profile) if profile is not None else _load_profile()
        self.history_seconds = float(self._profile["history_duration_seconds"])
        self.minimum_elapsed_seconds = float(self._profile["minimum_elapsed_seconds"])
        self.minimum_samples = int(self._profile["minimum_source_samples"])
        self.max_internal_gap_seconds = float(self._profile["max_internal_gap_seconds"])
        self._lock = threading.RLock()
        # (source_clock_seconds, ppm) in float64, oldest first, PAST_ONLY.
        self._samples: list[tuple[float, float]] = []
        self._last_event_key: tuple[Any, Any, Any] | None = None
        self._boot_id: str | None = None
        self._gap_restarts = 0
        self._gap_restart_pending = False
        self._accepted_events = 0

    def reset(self, reason: str) -> None:
        with self._lock:
            self._samples.clear()
            self._last_event_key = None
            if reason == "GAP":
                self._gap_restarts += 1
                self._gap_restart_pending = True

    def observe(self, sensor: Mapping[str, Any]) -> None:
        """Ingest one CO2 sensor record; only new measurement events advance."""

        with self._lock:
            self._observe_locked(sensor)

    def _observe_locked(self, sensor: Mapping[str, Any]) -> None:
        values = sensor.get("values")
        if not isinstance(values, Mapping):
            return
        if values.get("measurement_event_valid") is not True:
            return

        boot_id = sensor.get("boot_id")
        boot_id = boot_id if isinstance(boot_id, str) and boot_id else None
        if self._boot_id is not None and boot_id is not None and boot_id != self._boot_id:
            # boundary_policy: DERIVED_TEMPORAL_FEATURES_MUST_NOT_CROSS_BLOCK_BOUNDARIES
            self.reset("BOOT_BOUNDARY")
        self._boot_id = boot_id

        event_id = values.get("measurement_event_id")
        event_key = (sensor.get("device_id"), boot_id, event_id)
        if event_id is None or event_key == self._last_event_key:
            return

        clock_ms = values.get("measurement_monotonic_ms")
        ppm = values.get("latest_measurement_ppm")
        if not _finite(clock_ms) or not _finite(ppm):
            return
        clock_s = float(clock_ms) / 1000.0
        ppm = float(ppm)

        if self._samples:
            gap = clock_s - self._samples[-1][0]
            if gap <= 0.0:
                # Non-monotonic source clock cannot anchor a PAST_ONLY feature.
                self.reset("GAP")
            elif gap > self.max_internal_gap_seconds:
                self.reset("GAP")

        self._last_event_key = event_key
        self._samples.append((clock_s, ppm))
        self._accepted_events += 1
        # Keep just enough past history to select the >= 150 s endpoint.
        horizon = clock_s - (self.history_seconds * 2.0 + self.max_internal_gap_seconds)
        while len(self._samples) > 2 and self._samples[1][0] < horizon:
            self._samples.pop(0)

    def latest(self) -> CO2SlopeResult:
        with self._lock:
            return self._latest_locked()

    def _latest_locked(self) -> CO2SlopeResult:
        base: dict[str, Any] = {
            "slope_profile_id": PROFILE_ID,
            "slope_method": "ENDPOINT_DIFFERENCE",
            "slope_unit": "ppm/min",
            "timestamp_basis": "SOURCE_ACQUISITION_CLOCK",
            "required_history_seconds": self.history_seconds,
            "max_internal_gap_seconds": self.max_internal_gap_seconds,
            "accepted_measurement_events": self._accepted_events,
            "gap_restarts": self._gap_restarts,
            "retained_samples": len(self._samples),
            "gap_restart_pending": self._gap_restart_pending,
            "source_boot_id": self._boot_id,
        }
        if not self._samples:
            return CO2SlopeResult(
                "CO2_MEASUREMENT_CLOCK_UNAVAILABLE",
                "NO_VALID_MEASUREMENT_EVENT",
                None,
                None,
                base,
            )

        now_s, now_ppm = self._samples[-1]
        base["ppm"] = now_ppm
        if len(self._samples) < self.minimum_samples:
            return self._unrecovered(
                "INSUFFICIENT_SOURCE_SAMPLES", now_ppm, base
            )

        # Earliest past observation whose source-clock age is at least the
        # configured history duration. No interpolation, no future samples.
        endpoint: tuple[float, float] | None = None
        for sample in self._samples[:-1]:
            if now_s - sample[0] >= self.history_seconds:
                endpoint = sample
                break
        base["available_history_seconds"] = round(now_s - self._samples[0][0], 3)
        if endpoint is None:
            return self._unrecovered("INSUFFICIENT_ELAPSED_HISTORY", now_ppm, base)

        elapsed_s = now_s - endpoint[0]
        if elapsed_s < self.minimum_elapsed_seconds or elapsed_s <= 0.0:
            return self._unrecovered("INSUFFICIENT_ELAPSED_HISTORY", now_ppm, base)
        slope = (now_ppm - endpoint[1]) / (elapsed_s / 60.0)
        if not math.isfinite(slope):
            return CO2SlopeResult(
                "NO_CANONICAL_SLOPE", "NONFINITE_SLOPE", now_ppm, None, base
            )
        base.update(
            {
                "endpoint_span_seconds": round(elapsed_s, 3),
                "endpoint_ppm": endpoint[1],
            }
        )
        self._gap_restart_pending = False
        base["gap_restart_pending"] = False
        return CO2SlopeResult("CO2_SLOPE_READY", None, now_ppm, slope, base)

    def _unrecovered(
        self, reason: str, ppm: float | None, base: dict[str, Any]
    ) -> CO2SlopeResult:
        """Distinguish a cold start from an unrecovered forbidden-gap restart."""

        if self._gap_restart_pending:
            return CO2SlopeResult(
                "FEATURE_UNAVAILABLE_GAP_RESTART",
                f"GAP_RESTART_{reason}",
                ppm,
                None,
                base,
            )
        return CO2SlopeResult("FEATURE_UNAVAILABLE_WARMUP", reason, ppm, None, base)


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


__all__ = ["CO2SlopeWindowBuilder", "CO2SlopeResult", "PROFILE_ID", "PROFILE_PATH"]
