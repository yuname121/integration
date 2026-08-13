"""Thread-safe latest-value and freshness state for all SafeNest sensors."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
import threading
import time
from typing import Final, Mapping

from gateway.protocol import (
    DecodedPacket,
    TelemetryPayload,
    ThermalFrame,
)


SENSOR_IDS: Final = ("mmwave", "thermal", "co2", "pir")
DEFAULT_STALE_SECONDS: Final = {
    "mmwave": 3.0,
    "thermal": 3.0,
    "co2": 10.0,
    "pir": 10.0,
}


@dataclass
class _SensorRecord:
    sensor_id: str
    ttl_seconds: float
    connected: bool = False
    valid: bool = False
    peer: str | None = None
    sequence: int | None = None
    source_uptime_ms: int | None = None
    last_received_at: float | None = None
    last_received_monotonic: float | None = None
    last_valid_at: float | None = None
    disconnected_at: float | None = None
    error: str | None = None
    values: dict[str, object] = field(default_factory=dict)


class SensorStateManager:
    """Own the latest state without pretending old or invalid values are live.

    Wall-clock Unix seconds are exposed to downstream consumers. Freshness is
    calculated from monotonic receive time so system clock adjustments cannot
    make a stale packet look new.
    """

    def __init__(self, stale_seconds: Mapping[str, float] | None = None) -> None:
        configured = dict(DEFAULT_STALE_SECONDS)
        if stale_seconds is not None:
            unknown = set(stale_seconds) - set(SENSOR_IDS)
            if unknown:
                raise ValueError(f"unknown sensor TTL keys: {sorted(unknown)}")
            configured.update(stale_seconds)
        for sensor_id, ttl in configured.items():
            if not math.isfinite(float(ttl)) or float(ttl) <= 0:
                raise ValueError(f"{sensor_id} stale TTL must be positive and finite")

        self._lock = threading.RLock()
        self._records = {
            sensor_id: _SensorRecord(sensor_id, float(configured[sensor_id]))
            for sensor_id in SENSOR_IDS
        }
        self._latest_thermal_frame: ThermalFrame | None = None
        self._revision = 0

    def ingest(
        self,
        packet: DecodedPacket,
        peer: tuple[str, int],
        *,
        received_at: float | None = None,
        monotonic_at: float | None = None,
    ) -> int:
        wall = time.time() if received_at is None else _finite_time(received_at, "received_at")
        monotonic = (
            time.monotonic()
            if monotonic_at is None
            else _finite_time(monotonic_at, "monotonic_at")
        )
        peer_label = f"{peer[0]}:{peer[1]}"
        with self._lock:
            if isinstance(packet, TelemetryPayload):
                self._ingest_telemetry(packet, peer_label, wall, monotonic)
            elif isinstance(packet, ThermalFrame):
                self._ingest_thermal(packet, peer_label, wall, monotonic)
            else:  # Defensive boundary for callbacks outside the strict decoder.
                raise TypeError(f"unsupported packet object: {type(packet).__name__}")
            self._revision += 1
            return self._revision

    def mark_peer_disconnected(
        self,
        peer: tuple[str, int],
        *,
        disconnected_at: float | None = None,
    ) -> int:
        wall = (
            time.time()
            if disconnected_at is None
            else _finite_time(disconnected_at, "disconnected_at")
        )
        peer_label = f"{peer[0]}:{peer[1]}"
        changed = False
        with self._lock:
            for record in self._records.values():
                if record.peer == peer_label and record.connected:
                    record.connected = False
                    record.disconnected_at = wall
                    changed = True
            if changed:
                self._revision += 1
            return self._revision

    def snapshot(
        self,
        *,
        now: float | None = None,
        monotonic_now: float | None = None,
    ) -> dict[str, object]:
        wall = time.time() if now is None else _finite_time(now, "now")
        monotonic = (
            time.monotonic()
            if monotonic_now is None
            else _finite_time(monotonic_now, "monotonic_now")
        )
        with self._lock:
            sensors = {
                sensor_id: self._snapshot_record(record, monotonic)
                for sensor_id, record in self._records.items()
            }
            statuses = [entry["status"] for entry in sensors.values()]
            live_count = statuses.count("LIVE")
            connected_count = sum(bool(entry["connected"]) for entry in sensors.values())
            if live_count == len(SENSOR_IDS):
                system = "ONLINE"
            elif connected_count > 0:
                system = "DEGRADED"
            else:
                system = "OFFLINE"
            return {
                "timestamp": wall,
                "revision": self._revision,
                "system": system,
                "sensors": sensors,
            }

    def latest_thermal_frame(self) -> ThermalFrame | None:
        """Return the immutable latest frame; raw bytes stay out of state JSON."""

        with self._lock:
            return self._latest_thermal_frame

    def _ingest_telemetry(
        self,
        packet: TelemetryPayload,
        peer: str,
        wall: float,
        monotonic: float,
    ) -> None:
        mmwave_valid = packet.valid["respiration"] or packet.valid["heart"]
        self._update(
            "mmwave",
            peer=peer,
            sequence=packet.header.sequence,
            uptime_ms=packet.uptime_ms,
            wall=wall,
            monotonic=monotonic,
            valid=mmwave_valid,
            error=None if mmwave_valid else "MMWAVE_VALUES_INVALID",
            values={
                "presence": None,
                "presence_available": False,
                "respiration_rate_bpm": packet.respiration_rate_bpm,
                "heart_rate_bpm": packet.heart_rate_bpm,
                "respiration_valid": packet.valid["respiration"],
                "heart_valid": packet.valid["heart"],
            },
        )
        self._update(
            "co2",
            peer=peer,
            sequence=packet.header.sequence,
            uptime_ms=packet.uptime_ms,
            wall=wall,
            monotonic=monotonic,
            valid=packet.valid["co2"],
            error=None if packet.valid["co2"] else "CO2_VALUE_INVALID",
            values={"ppm": packet.co2_ppm},
        )
        self._update(
            "pir",
            peer=peer,
            sequence=packet.header.sequence,
            uptime_ms=packet.uptime_ms,
            wall=wall,
            monotonic=monotonic,
            valid=True,
            error=None,
            values={"motion": packet.pir_motion},
        )

    def _ingest_thermal(
        self,
        packet: ThermalFrame,
        peer: str,
        wall: float,
        monotonic: float,
    ) -> None:
        self._latest_thermal_frame = packet
        self._update(
            "thermal",
            peer=peer,
            sequence=packet.header.sequence,
            uptime_ms=packet.uptime_ms,
            wall=wall,
            monotonic=monotonic,
            valid=True,
            error=None,
            values={
                "width": packet.width,
                "height": packet.height,
                "frame_sequence": packet.frame_sequence,
                "minimum_raw": packet.minimum_raw,
                "maximum_raw": packet.maximum_raw,
                "frame_available": True,
            },
        )

    def _update(
        self,
        sensor_id: str,
        *,
        peer: str,
        sequence: int,
        uptime_ms: int,
        wall: float,
        monotonic: float,
        valid: bool,
        error: str | None,
        values: dict[str, object],
    ) -> None:
        record = self._records[sensor_id]
        record.connected = True
        record.valid = valid
        record.peer = peer
        record.sequence = sequence
        record.source_uptime_ms = uptime_ms
        record.last_received_at = wall
        record.last_received_monotonic = monotonic
        record.disconnected_at = None
        record.error = error
        record.values = values
        if valid:
            record.last_valid_at = wall

    @staticmethod
    def _snapshot_record(record: _SensorRecord, monotonic_now: float) -> dict[str, object]:
        if record.last_received_monotonic is None:
            age = None
            stale = False
            status = "NO_DATA"
        else:
            age = max(0.0, monotonic_now - record.last_received_monotonic)
            stale = age > record.ttl_seconds
            if not record.connected:
                status = "DISCONNECTED"
            elif stale:
                status = "STALE"
            elif not record.valid:
                status = "INVALID"
            else:
                status = "LIVE"
        return {
            "sensor_id": record.sensor_id,
            "status": status,
            "connected": record.connected,
            "stale": stale,
            "valid": record.valid,
            "current": status == "LIVE",
            "ttl_seconds": record.ttl_seconds,
            "age_seconds": age,
            "last_update": record.last_received_at,
            "last_received_at": record.last_received_at,
            "last_valid_at": record.last_valid_at,
            "disconnected_at": record.disconnected_at,
            "peer": record.peer,
            "sequence": record.sequence,
            "source_uptime_ms": record.source_uptime_ms,
            "error": record.error,
            "values": copy.deepcopy(record.values),
        }


def _finite_time(value: float, field: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return converted
