"""Deterministic Capture v1 structural validator.

The validator checks schema correctness only. It does not run AI, talk to
hardware, read real Capture payloads, modify records, or repair malformed
input.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .constants import (
    CAPTURE_SCHEMA_VERSION,
    CO2_FORBIDDEN_FIELDS,
    CO2_PAYLOAD_ALLOWED,
    DEVICE_BOOT_ID_UNAVAILABLE,
    DEVICE_ID_UNAVAILABLE_THERMAL_UDP,
    ERROR_CODES,
    EVENT_ALLOWED_FIELDS,
    EVENT_KINDS,
    EVENT_REQUIRED_FIELDS,
    INTEGRATION_GIT_SHA_UNAVAILABLE,
    MMWAVE_FORBIDDEN_FIELDS,
    MMWAVE_PAYLOAD_ALLOWED,
    MMWAVE_PHASE_STATUS,
    PIR_PAYLOAD_ALLOWED,
    RUNTIME_CAPTURE_ROOT,
    RUNTIME_PAYLOAD_ALLOWED,
    SENSOR_TYPES,
    SESSION_ALLOWED_FIELDS,
    SESSION_REQUIRED_FIELDS,
    SESSION_STATUSES,
    SOURCE_TIMING_UNAVAILABLE,
    THERMAL_DTYPE,
    THERMAL_ENDIANNESS,
    THERMAL_FORBIDDEN_FIELDS,
    THERMAL_HEIGHT,
    THERMAL_PAYLOAD_ALLOWED,
    THERMAL_PAYLOAD_CONTAINER,
    THERMAL_PIXEL_COUNT,
    THERMAL_WIDTH,
    UNAVAILABILITY_REASONS,
)
from .errors import CaptureValidationIssue, CaptureValidationResult, merge
from .identities import (
    CAPTURE_EVENT_ID_PATTERN,
    FRAME_ID_PATTERN,
    GIT_SHA_PATTERN,
    SESSION_ID_PATTERN,
    SHA256_PATTERN,
    THERMAL_PAYLOAD_REFERENCE_PATTERN,
)


def _issue(path: str, code: str, message: str) -> CaptureValidationIssue:
    return CaptureValidationIssue(path=path, code=code, message=message)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def load_json_document(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_json_constant)


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _unknown_fields(document: Mapping[str, Any], allowed: set[str], path: str) -> list[CaptureValidationIssue]:
    extra = sorted(set(document) - allowed)
    return [
        _issue(f"{path}.{name}" if path else name, "UNKNOWN_FIELD", "unknown fields are rejected in Capture v1")
        for name in extra
    ]


def _require_object(value: Any, path: str) -> list[CaptureValidationIssue]:
    if not isinstance(value, dict):
        return [_issue(path, "TYPE_INVALID", "expected a JSON object")]
    return []


def _check_schema_version(value: Any, path: str) -> list[CaptureValidationIssue]:
    if value != CAPTURE_SCHEMA_VERSION:
        return [
            _issue(
                path,
                "SCHEMA_VERSION_INVALID",
                f"expected {CAPTURE_SCHEMA_VERSION!r}, got {value!r}",
            )
        ]
    return []


def _check_session_id(value: Any, path: str) -> list[CaptureValidationIssue]:
    if not isinstance(value, str) or not SESSION_ID_PATTERN.fullmatch(value):
        return [
            _issue(
                path,
                "SESSION_ID_INVALID",
                "session_id must match sncap-<YYYYMMDDTHHMMSSZ>-<12 hex>",
            )
        ]
    return []


def _check_event_id(value: Any, path: str) -> list[CaptureValidationIssue]:
    if not isinstance(value, str) or not CAPTURE_EVENT_ID_PATTERN.fullmatch(value):
        return [
            _issue(
                path,
                "CAPTURE_EVENT_ID_INVALID",
                "capture_event_id must be a canonical lowercase UUID4",
            )
        ]
    return []


def _check_unix_time(value: Any, path: str) -> list[CaptureValidationIssue]:
    if not _is_finite_number(value) or float(value) < 0:
        return [_issue(path, "TIMESTAMP_INVALID", "expected a finite non-negative Unix time")]
    return []


def _check_monotonic(value: Any, path: str) -> list[CaptureValidationIssue]:
    if not _is_finite_number(value) or float(value) < 0:
        return [_issue(path, "TIMESTAMP_INVALID", "expected a finite non-negative monotonic time")]
    return []


def _check_nullable_int(value: Any, path: str, *, minimum: int | None = 0) -> list[CaptureValidationIssue]:
    if value is None:
        return []
    if not _is_int(value):
        return [_issue(path, "TYPE_INVALID", "expected an integer or null")]
    if minimum is not None and value < minimum:
        return [_issue(path, "VALUE_INVALID", f"expected integer >= {minimum} or null")]
    return []


def _check_nullable_string(value: Any, path: str) -> list[CaptureValidationIssue]:
    if value is None:
        return []
    if isinstance(value, str) and value:
        return []
    if isinstance(value, str):
        return [_issue(path, "VALUE_INVALID", "empty string is not a substitute for null")]
    return [_issue(path, "TYPE_INVALID", "expected a string or null")]


def _reason_allowed(value: Any, path: str) -> list[CaptureValidationIssue]:
    if value is None:
        return []
    if not isinstance(value, str) or value not in UNAVAILABILITY_REASONS:
        return [
            _issue(
                path,
                "UNAVAILABLE_REASON_INVALID",
                f"reason must be one of {sorted(UNAVAILABILITY_REASONS)} or null",
            )
        ]
    return []


def validate_session_manifest(document: Any) -> CaptureValidationResult:
    errors = _require_object(document, "$")
    if errors:
        return CaptureValidationResult(tuple(errors))
    assert isinstance(document, dict)
    errors.extend(_unknown_fields(document, set(SESSION_ALLOWED_FIELDS), ""))
    for field in SESSION_REQUIRED_FIELDS:
        if field not in document:
            errors.append(_issue(field, "MISSING_FIELD", "required field is missing"))

    if "capture_schema_version" in document:
        errors.extend(_check_schema_version(document.get("capture_schema_version"), "capture_schema_version"))
    if "session_id" in document:
        errors.extend(_check_session_id(document.get("session_id"), "session_id"))
    if "session_start_wall_time" in document:
        errors.extend(_check_unix_time(document.get("session_start_wall_time"), "session_start_wall_time"))
    if "session_start_monotonic_time" in document:
        errors.extend(_check_monotonic(document.get("session_start_monotonic_time"), "session_start_monotonic_time"))

    provenance = document.get("software_provenance")
    if "software_provenance" in document and (not isinstance(provenance, str) or not provenance.strip()):
        errors.append(_issue("software_provenance", "TYPE_INVALID", "expected a non-empty string"))

    git_sha = document.get("integration_git_sha")
    git_reason = document.get("integration_git_sha_unavailable_reason")
    if git_sha is None:
        if git_reason != INTEGRATION_GIT_SHA_UNAVAILABLE:
            errors.append(
                _issue(
                    "integration_git_sha_unavailable_reason",
                    "UNAVAILABLE_REASON_REQUIRED",
                    f"null integration_git_sha requires {INTEGRATION_GIT_SHA_UNAVAILABLE}",
                )
            )
    elif not isinstance(git_sha, str) or not GIT_SHA_PATTERN.fullmatch(git_sha):
        errors.append(_issue("integration_git_sha", "GIT_SHA_INVALID", "expected 40 lowercase hex chars or null"))
    elif git_reason is not None:
        errors.append(
            _issue(
                "integration_git_sha_unavailable_reason",
                "UNAVAILABLE_REASON_INVALID",
                "reason must be null when integration_git_sha is present",
            )
        )

    runtime_version = document.get("pi_runtime_version")
    if runtime_version is not None and (not isinstance(runtime_version, str) or not runtime_version.strip()):
        errors.append(_issue("pi_runtime_version", "TYPE_INVALID", "expected a non-empty string or null/omitted"))

    device_ids = document.get("device_ids")
    if "device_ids" in document:
        if not isinstance(device_ids, list) or any(not isinstance(item, str) or not item for item in device_ids):
            errors.append(_issue("device_ids", "TYPE_INVALID", "expected a list of non-empty strings"))
        elif len(device_ids) != len(set(device_ids)):
            errors.append(_issue("device_ids", "VALUE_INVALID", "device_ids must be unique"))

    status = document.get("session_status")
    if "session_status" in document and status not in SESSION_STATUSES:
        errors.append(
            _issue("session_status", "SESSION_STATUS_INVALID", f"expected one of {sorted(SESSION_STATUSES)}")
        )

    root = document.get("runtime_capture_root")
    if "runtime_capture_root" in document and root != RUNTIME_CAPTURE_ROOT:
        errors.append(
            _issue(
                "runtime_capture_root",
                "CAPTURE_ROOT_INVALID",
                f"expected relative root {RUNTIME_CAPTURE_ROOT!r}, not an absolute or alternate path",
            )
        )

    return CaptureValidationResult(tuple(errors))


def validate_capture_event(document: Any) -> CaptureValidationResult:
    errors = _require_object(document, "$")
    if errors:
        return CaptureValidationResult(tuple(errors))
    assert isinstance(document, dict)
    errors.extend(_unknown_fields(document, set(EVENT_ALLOWED_FIELDS), ""))
    for field in EVENT_REQUIRED_FIELDS:
        if field not in document:
            errors.append(_issue(field, "MISSING_FIELD", "required field is missing"))

    if "capture_schema_version" in document:
        errors.extend(_check_schema_version(document.get("capture_schema_version"), "capture_schema_version"))
    if "session_id" in document:
        errors.extend(_check_session_id(document.get("session_id"), "session_id"))
    if "capture_event_id" in document:
        errors.extend(_check_event_id(document.get("capture_event_id"), "capture_event_id"))

    event_kind = document.get("event_kind")
    if "event_kind" in document and event_kind not in EVENT_KINDS:
        errors.append(_issue("event_kind", "EVENT_KIND_INVALID", f"expected one of {sorted(EVENT_KINDS)}"))

    sensor_type = document.get("sensor_type")
    if "sensor_type" in document and sensor_type not in SENSOR_TYPES:
        errors.append(_issue("sensor_type", "SENSOR_TYPE_INVALID", f"expected one of {sorted(SENSOR_TYPES)}"))

    errors.extend(_check_nullable_string(document.get("device_id"), "device_id"))
    errors.extend(_check_nullable_string(document.get("boot_id"), "boot_id"))
    errors.extend(_reason_allowed(document.get("device_id_unavailable_reason"), "device_id_unavailable_reason"))
    errors.extend(_reason_allowed(document.get("boot_id_unavailable_reason"), "boot_id_unavailable_reason"))

    device_id = document.get("device_id")
    device_reason = document.get("device_id_unavailable_reason")
    if device_id is None:
        expected = DEVICE_ID_UNAVAILABLE_THERMAL_UDP if sensor_type == "thermal" else None
        if device_reason is None:
            errors.append(
                _issue(
                    "device_id_unavailable_reason",
                    "UNAVAILABLE_REASON_REQUIRED",
                    "null device_id requires an explicit unavailability reason",
                )
            )
        elif sensor_type == "thermal" and device_reason != expected:
            errors.append(
                _issue(
                    "device_id_unavailable_reason",
                    "UNAVAILABLE_REASON_INVALID",
                    f"Thermal UDP currently requires {DEVICE_ID_UNAVAILABLE_THERMAL_UDP}",
                )
            )
    elif device_reason is not None:
        errors.append(
            _issue(
                "device_id_unavailable_reason",
                "UNAVAILABLE_REASON_INVALID",
                "reason must be null/omitted when device_id is present",
            )
        )

    boot_id = document.get("boot_id")
    boot_reason = document.get("boot_id_unavailable_reason")
    if boot_id is None:
        if boot_reason != DEVICE_BOOT_ID_UNAVAILABLE:
            errors.append(
                _issue(
                    "boot_id_unavailable_reason",
                    "UNAVAILABLE_REASON_REQUIRED",
                    f"null boot_id requires {DEVICE_BOOT_ID_UNAVAILABLE}",
                )
            )
    elif boot_reason is not None:
        errors.append(
            _issue(
                "boot_id_unavailable_reason",
                "UNAVAILABLE_REASON_INVALID",
                "reason must be null/omitted when boot_id is present",
            )
        )

    errors.extend(_check_nullable_int(document.get("packet_sequence"), "packet_sequence", minimum=0))
    errors.extend(_check_nullable_int(document.get("device_uptime_ms"), "device_uptime_ms", minimum=0))

    source_event_id = document.get("source_measurement_event_id")
    source_monotonic = document.get("source_measurement_monotonic_ms")
    source_reason = document.get("source_timing_unavailable_reason")
    errors.extend(_check_nullable_string(source_event_id, "source_measurement_event_id"))
    errors.extend(
        _check_nullable_int(source_monotonic, "source_measurement_monotonic_ms", minimum=0)
    )
    errors.extend(_reason_allowed(source_reason, "source_timing_unavailable_reason"))
    has_source_id = source_event_id is not None
    has_source_time = source_monotonic is not None
    if has_source_id != has_source_time:
        errors.append(
            _issue(
                "source_measurement_event_id",
                "SOURCE_PROVENANCE_INCOMPLETE",
                "source identity and source monotonic time must both be present or both be null",
            )
        )
    elif not has_source_id:
        if source_reason != SOURCE_TIMING_UNAVAILABLE:
            errors.append(
                _issue(
                    "source_timing_unavailable_reason",
                    "UNAVAILABLE_REASON_REQUIRED",
                    f"missing source timing/identity requires {SOURCE_TIMING_UNAVAILABLE}, not a fabricated zero",
                )
            )
    elif source_reason is not None:
        errors.append(
            _issue(
                "source_timing_unavailable_reason",
                "UNAVAILABLE_REASON_INVALID",
                "reason must be null when source timing and identity are both present",
            )
        )
    if source_reason == SOURCE_TIMING_UNAVAILABLE and (has_source_id or has_source_time):
        errors.append(
            _issue(
                "source_measurement_monotonic_ms",
                "FAKE_ZERO_FORBIDDEN",
                "unavailable source timing must use null; 0 is a real value, not a missing marker",
            )
        )

    if "pi_receive_wall_time" in document:
        errors.extend(_check_unix_time(document.get("pi_receive_wall_time"), "pi_receive_wall_time"))
    if "pi_receive_monotonic_time" in document:
        errors.extend(_check_monotonic(document.get("pi_receive_monotonic_time"), "pi_receive_monotonic_time"))

    for flag in ("parse_valid", "sensor_valid", "stale"):
        if flag in document and not _is_bool(document.get(flag)):
            errors.append(_issue(flag, "TYPE_INVALID", "expected a boolean"))

    error_code = document.get("error_code")
    error_reason = document.get("error_reason")
    if error_code is not None and (not isinstance(error_code, str) or error_code not in ERROR_CODES):
        errors.append(
            _issue("error_code", "ERROR_CODE_INVALID", f"expected one of {sorted(ERROR_CODES)} or null")
        )
    if error_reason is not None and not isinstance(error_reason, str):
        errors.append(_issue("error_reason", "TYPE_INVALID", "expected a string or null"))
    if error_reason is not None and ("\n" in error_reason or "Traceback" in error_reason):
        errors.append(
            _issue("error_reason", "ERROR_REASON_INVALID", "stack traces are not part of the core event contract")
        )

    parse_valid = document.get("parse_valid")
    if parse_valid is False and error_code is None:
        errors.append(_issue("error_code", "ERROR_CODE_REQUIRED", "parse_valid=false requires error_code"))

    payload = document.get("payload")
    if "payload" in document:
        errors.extend(_validate_payload(sensor_type, payload))
        if (
            sensor_type == "co2"
            and isinstance(payload, dict)
            and _is_bool(payload.get("measurement_identity_unavailable"))
        ):
            flag = payload["measurement_identity_unavailable"]
            if flag and source_event_id is not None:
                errors.append(
                    _issue(
                        "payload.measurement_identity_unavailable",
                        "SOURCE_PROVENANCE_INCOMPLETE",
                        "measurement_identity_unavailable=true requires null source identity",
                    )
                )
            if not flag and source_event_id is None:
                errors.append(
                    _issue(
                        "payload.measurement_identity_unavailable",
                        "SOURCE_PROVENANCE_INCOMPLETE",
                        "measurement_identity_unavailable=false requires source identity",
                    )
                )

    return CaptureValidationResult(tuple(errors))


def _validate_payload(sensor_type: Any, payload: Any) -> list[CaptureValidationIssue]:
    errors = _require_object(payload, "payload")
    if errors:
        return errors
    assert isinstance(payload, dict)
    if sensor_type == "co2":
        return _validate_co2_payload(payload)
    if sensor_type == "thermal":
        return _validate_thermal_payload(payload)
    if sensor_type == "pir":
        return _validate_pir_payload(payload)
    if sensor_type == "mmwave":
        return _validate_mmwave_payload(payload)
    if sensor_type == "runtime":
        return _validate_runtime_payload(payload)
    return []


def _validate_co2_payload(payload: dict[str, Any]) -> list[CaptureValidationIssue]:
    errors = _unknown_fields(payload, set(CO2_PAYLOAD_ALLOWED), "payload")
    for name in sorted(set(payload) & CO2_FORBIDDEN_FIELDS):
        errors.append(
            _issue(
                f"payload.{name}",
                "FORBIDDEN_FIELD",
                "humidity, temperature, slope, and pixel arrays are not part of the CO2 Capture contract",
            )
        )
    if "co2_ppm" not in payload:
        errors.append(_issue("payload.co2_ppm", "MISSING_FIELD", "required field is missing"))
    else:
        value = payload["co2_ppm"]
        if value is not None and not _is_finite_number(value):
            errors.append(_issue("payload.co2_ppm", "TYPE_INVALID", "expected a finite number or null"))
    identity_flag = payload.get("measurement_identity_unavailable")
    if "measurement_identity_unavailable" not in payload:
        errors.append(
            _issue(
                "payload.measurement_identity_unavailable",
                "MISSING_FIELD",
                "CO2 must state whether measurement identity is unavailable",
            )
        )
    elif not _is_bool(identity_flag):
        errors.append(_issue("payload.measurement_identity_unavailable", "TYPE_INVALID", "expected a boolean"))
    return errors


def _validate_thermal_payload(payload: dict[str, Any]) -> list[CaptureValidationIssue]:
    errors = _unknown_fields(payload, set(THERMAL_PAYLOAD_ALLOWED), "payload")
    for name in sorted(set(payload) & THERMAL_FORBIDDEN_FIELDS):
        errors.append(
            _issue(
                f"payload.{name}",
                "FORBIDDEN_FIELD",
                "full pixel arrays must not appear in Capture event JSON",
            )
        )
    required = (
        "frame_id",
        "frame_sequence",
        "width",
        "height",
        "dtype",
        "endianness",
        "minimum_raw",
        "maximum_raw",
        "pixel_count",
        "payload_container",
        "payload_reference",
        "payload_sha256",
    )
    for field in required:
        if field not in payload:
            errors.append(_issue(f"payload.{field}", "MISSING_FIELD", "required field is missing"))

    frame_id = payload.get("frame_id")
    if "frame_id" in payload and (not isinstance(frame_id, str) or not FRAME_ID_PATTERN.fullmatch(frame_id)):
        errors.append(_issue("payload.frame_id", "FRAME_ID_INVALID", "frame_id must match snfrm-<12 hex>"))

    errors.extend(_check_nullable_int(payload.get("frame_sequence"), "payload.frame_sequence", minimum=0))
    if payload.get("frame_sequence") is None and "frame_sequence" in payload:
        errors.append(_issue("payload.frame_sequence", "TYPE_INVALID", "frame_sequence is required and may not be null"))

    width = payload.get("width")
    height = payload.get("height")
    if "width" in payload and width != THERMAL_WIDTH:
        errors.append(_issue("payload.width", "DIMENSION_INVALID", f"expected {THERMAL_WIDTH}"))
    if "height" in payload and height != THERMAL_HEIGHT:
        errors.append(_issue("payload.height", "DIMENSION_INVALID", f"expected {THERMAL_HEIGHT}"))
    if "dtype" in payload and payload.get("dtype") != THERMAL_DTYPE:
        errors.append(_issue("payload.dtype", "DTYPE_INVALID", f"expected {THERMAL_DTYPE}"))
    if "endianness" in payload and payload.get("endianness") != THERMAL_ENDIANNESS:
        errors.append(_issue("payload.endianness", "ENDIANNESS_INVALID", f"expected {THERMAL_ENDIANNESS}"))
    if "pixel_count" in payload and payload.get("pixel_count") != THERMAL_PIXEL_COUNT:
        errors.append(
            _issue("payload.pixel_count", "DIMENSION_INVALID", f"expected {THERMAL_PIXEL_COUNT}")
        )
    if "payload_container" in payload and payload.get("payload_container") != THERMAL_PAYLOAD_CONTAINER:
        errors.append(
            _issue(
                "payload.payload_container",
                "PAYLOAD_CONTAINER_INVALID",
                f"RP-A1 reserves {THERMAL_PAYLOAD_CONTAINER}; the production writer is not implemented",
            )
        )

    for field in ("minimum_raw", "maximum_raw"):
        value = payload.get(field)
        if field in payload and not _is_int(value):
            errors.append(_issue(f"payload.{field}", "TYPE_INVALID", "expected an integer"))
        elif _is_int(value) and value < 0:
            errors.append(_issue(f"payload.{field}", "VALUE_INVALID", "raw values must be >= 0"))
    minimum_raw = payload.get("minimum_raw")
    maximum_raw = payload.get("maximum_raw")
    if _is_int(minimum_raw) and _is_int(maximum_raw) and minimum_raw > maximum_raw:
        errors.append(_issue("payload.maximum_raw", "VALUE_INVALID", "maximum_raw must be >= minimum_raw"))

    reference = payload.get("payload_reference")
    if "payload_reference" in payload:
        if not isinstance(reference, str) or not THERMAL_PAYLOAD_REFERENCE_PATTERN.fullmatch(reference):
            errors.append(
                _issue(
                    "payload.payload_reference",
                    "PAYLOAD_REFERENCE_INVALID",
                    "expected session-relative thermal/frames_NNNN.npz or thermal/frames_NNNN.npz#index",
                )
            )
        elif reference.startswith("/") or ".." in reference or ":\\" in reference:
            errors.append(
                _issue(
                    "payload.payload_reference",
                    "PAYLOAD_REFERENCE_INVALID",
                    "absolute paths are forbidden",
                )
            )

    digest = payload.get("payload_sha256")
    if "payload_sha256" in payload and (not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest)):
        errors.append(
            _issue("payload.payload_sha256", "CHECKSUM_INVALID", "expected 64 lowercase hex SHA-256 chars")
        )
    return errors


def _validate_pir_payload(payload: dict[str, Any]) -> list[CaptureValidationIssue]:
    errors = _unknown_fields(payload, set(PIR_PAYLOAD_ALLOWED), "payload")
    if "pir_motion" not in payload:
        errors.append(_issue("payload.pir_motion", "MISSING_FIELD", "required field is missing"))
    elif not _is_bool(payload.get("pir_motion")):
        errors.append(_issue("payload.pir_motion", "TYPE_INVALID", "expected a boolean"))
    if "is_transition" in payload and not _is_bool(payload.get("is_transition")):
        errors.append(_issue("payload.is_transition", "TYPE_INVALID", "expected a boolean"))
    return errors


def _validate_mmwave_payload(payload: dict[str, Any]) -> list[CaptureValidationIssue]:
    errors = _unknown_fields(payload, set(MMWAVE_PAYLOAD_ALLOWED), "payload")
    for name in sorted(set(payload) & MMWAVE_FORBIDDEN_FIELDS):
        errors.append(
            _issue(
                f"payload.{name}",
                "FORBIDDEN_FIELD",
                "mmWave phase payload remains PENDING_MMWAVE_DEVICE_CONTRACT_VALIDATION",
            )
        )
    status = payload.get("phase_payload_status")
    if status != MMWAVE_PHASE_STATUS:
        errors.append(
            _issue(
                "payload.phase_payload_status",
                "MMWAVE_PHASE_DEFERRED",
                f"expected {MMWAVE_PHASE_STATUS}",
            )
        )
    for field in ("respiration_rate_bpm", "heart_rate_bpm"):
        if field in payload and payload[field] is not None and not _is_finite_number(payload[field]):
            errors.append(_issue(f"payload.{field}", "TYPE_INVALID", "expected a finite number or null"))
    for field in ("respiration_valid", "heart_valid"):
        if field in payload and not _is_bool(payload[field]):
            errors.append(_issue(f"payload.{field}", "TYPE_INVALID", "expected a boolean"))
    return errors


def _validate_runtime_payload(payload: dict[str, Any]) -> list[CaptureValidationIssue]:
    errors = _unknown_fields(payload, set(RUNTIME_PAYLOAD_ALLOWED), "payload")
    detail = payload.get("detail")
    if "detail" in payload and not isinstance(detail, str):
        errors.append(_issue("payload.detail", "TYPE_INVALID", "expected a string"))
    return errors


def validate_event_collection(
    events: list[Any],
    *,
    session_id: str | None = None,
) -> CaptureValidationResult:
    collected: list[CaptureValidationIssue] = []
    seen: dict[str, int] = {}
    if not isinstance(events, list):
        return CaptureValidationResult((_issue("$", "TYPE_INVALID", "expected a JSON array of events"),))
    for index, event in enumerate(events):
        result = validate_capture_event(event)
        for issue in result.errors:
            collected.append(
                CaptureValidationIssue(path=f"[{index}].{issue.path}", code=issue.code, message=issue.message)
            )
        if isinstance(event, dict):
            event_id = event.get("capture_event_id")
            if isinstance(event_id, str):
                if event_id in seen:
                    collected.append(
                        _issue(
                            f"[{index}].capture_event_id",
                            "CAPTURE_EVENT_ID_DUPLICATE",
                            f"duplicates [{seen[event_id]}].capture_event_id",
                        )
                    )
                else:
                    seen[event_id] = index
            if session_id is not None and event.get("session_id") != session_id:
                collected.append(
                    _issue(
                        f"[{index}].session_id",
                        "SESSION_ID_MISMATCH",
                        "event session_id must match the session manifest",
                    )
                )
    return CaptureValidationResult(tuple(collected))


def validate_document(document: Any, *, kind: str) -> CaptureValidationResult:
    if kind == "session":
        return validate_session_manifest(document)
    if kind == "event":
        return validate_capture_event(document)
    if kind == "events":
        return validate_event_collection(document)
    return CaptureValidationResult((_issue("$", "KIND_INVALID", f"unsupported document kind {kind!r}"),))


def infer_kind(path: Path, document: Any) -> str:
    name = path.name
    if "session" in name or name == "manifest.json":
        return "session"
    if isinstance(document, list):
        return "events"
    return "event"


def validate_path(path: Path) -> CaptureValidationResult:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        return CaptureValidationResult((_issue(str(path), "READ_FAILED", str(error)),))
    if path.suffix == ".jsonl":
        events: list[Any] = []
        collected: list[CaptureValidationIssue] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                events.append(load_json_document(line))
            except ValueError as error:
                collected.append(_issue(f"{path}:{line_number}", "JSON_INVALID", str(error)))
        if collected:
            return CaptureValidationResult(tuple(collected))
        return validate_event_collection(events)
    try:
        document = load_json_document(text)
    except ValueError as error:
        return CaptureValidationResult((_issue(str(path), "JSON_INVALID", str(error)),))
    return validate_document(document, kind=infer_kind(path, document))


def validate_paths(paths: list[Path]) -> CaptureValidationResult:
    return merge(*(validate_path(path) for path in paths))
