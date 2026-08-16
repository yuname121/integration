"""Capture session and event identity rules.

session_id
    Format: sncap-<UTC compact>-<12 hex entropy>
    Example: sncap-20260816T093012Z-a1b2c3d4e5f6

    A plain timestamp is not sufficient. The entropy suffix remains unique
    across process restarts, Pi reboots, and concurrent experiments that
    could share the same UTC second.

capture_event_id
    Canonical lowercase UUID4. This is the Pi-assigned persistent evidence
    identity. It is never reused as a substitute for sensor-native IDs such
    as CO2 measurement_event_id, Thermal frame_sequence, mmWave phase
    sample sequence, or PIR transition identity. Those remain in
    source_measurement_event_id / payload.frame_sequence.

frame_id
    Pi-assigned Thermal frame identity, distinct from capture_event_id and
    from the device frame_sequence.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
import secrets
from typing import Callable
import uuid

SESSION_ID_PATTERN = re.compile(r"^sncap-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}$")
CAPTURE_EVENT_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
FRAME_ID_PATTERN = re.compile(r"^snfrm-[0-9a-f]{12}$")
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ERROR_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
THERMAL_PAYLOAD_REFERENCE_PATTERN = re.compile(
    r"^thermal/frames_[0-9]{4,}\.npz(?:#[0-9]+)?$"
)


def new_session_id(
    *,
    wall_time: float,
    entropy: str | None = None,
) -> str:
    utc = datetime.fromtimestamp(wall_time, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = entropy if entropy is not None else secrets.token_hex(6)
    session_id = f"sncap-{utc}-{suffix}"
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError(f"generated session_id is malformed: {session_id}")
    return session_id


def new_capture_event_id(uuid_factory: Callable[[], uuid.UUID] = uuid.uuid4) -> str:
    event_id = str(uuid_factory())
    if not CAPTURE_EVENT_ID_PATTERN.fullmatch(event_id):
        raise ValueError(f"generated capture_event_id is not UUID4: {event_id}")
    return event_id


def new_frame_id(*, entropy: str | None = None) -> str:
    suffix = entropy if entropy is not None else secrets.token_hex(6)
    frame_id = f"snfrm-{suffix}"
    if not FRAME_ID_PATTERN.fullmatch(frame_id):
        raise ValueError(f"generated frame_id is malformed: {frame_id}")
    return frame_id
