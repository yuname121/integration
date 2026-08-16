"""Machine-readable Capture v1 constants.

Capture schema version is independent of:
- network telemetry schema (`safenest.telemetry.v1`)
- SQLite operational schema
- AI input contracts
"""

from __future__ import annotations

from typing import Final

CAPTURE_SCHEMA_VERSION: Final = "safenest.capture.v1"
CAPTURE_SCHEMA_FAMILY: Final = "SAFENEST_CAPTURE_V1"
RUNTIME_CAPTURE_ROOT: Final = "captures"

SENSOR_TYPES: Final = frozenset({"co2", "thermal", "pir", "mmwave", "runtime"})
EVENT_KINDS: Final = frozenset(
    {
        "observation",
        "transport_duplicate",
        "transport_gap",
        "sensor_invalid",
        "continuity_break",
        "capture_drop",
        "capture_error",
    }
)
SESSION_STATUSES: Final = frozenset({"open", "closed", "unclean"})

THERMAL_WIDTH: Final = 80
THERMAL_HEIGHT: Final = 62
THERMAL_PIXEL_COUNT: Final = THERMAL_WIDTH * THERMAL_HEIGHT
THERMAL_DTYPE: Final = "uint16"
THERMAL_ENDIANNESS: Final = "big"
THERMAL_PAYLOAD_CONTAINER: Final = "npz_uint16_lossless"

MMWAVE_PHASE_STATUS: Final = "PENDING_MMWAVE_DEVICE_CONTRACT_VALIDATION"

SOURCE_TIMING_UNAVAILABLE: Final = "SOURCE_TIMING_UNAVAILABLE"
DEVICE_BOOT_ID_UNAVAILABLE: Final = "DEVICE_BOOT_ID_NOT_IN_TELEMETRY_V1"
DEVICE_ID_UNAVAILABLE_THERMAL_UDP: Final = "DEVICE_ID_UNAVAILABLE_THERMAL_UDP"
INTEGRATION_GIT_SHA_UNAVAILABLE: Final = "INTEGRATION_GIT_SHA_UNAVAILABLE"

UNAVAILABILITY_REASONS: Final = frozenset(
    {
        SOURCE_TIMING_UNAVAILABLE,
        DEVICE_BOOT_ID_UNAVAILABLE,
        DEVICE_ID_UNAVAILABLE_THERMAL_UDP,
        INTEGRATION_GIT_SHA_UNAVAILABLE,
        "MEASUREMENT_IDENTITY_UNAVAILABLE",
        MMWAVE_PHASE_STATUS,
    }
)

ERROR_CODES: Final = frozenset(
    {
        "PARSE_FAILED",
        "SENSOR_INVALID",
        "TRANSPORT_GAP",
        "TRANSPORT_DUPLICATE",
        "CAPTURE_DROP",
        "CAPTURE_ERROR",
        "CHECKSUM_INVALID",
        "PAYLOAD_REFERENCE_INVALID",
        "DIMENSION_INVALID",
    }
)

CO2_FORBIDDEN_FIELDS: Final = frozenset(
    {
        "humidity",
        "relative_humidity",
        "temperature",
        "temperature_c",
        "co2_slope",
        "CO2_slope",
        "pixels",
    }
)
THERMAL_FORBIDDEN_FIELDS: Final = frozenset(
    {
        "pixels",
        "values",
        "frame",
        "raw_pixels",
    }
)
MMWAVE_FORBIDDEN_FIELDS: Final = frozenset(
    {
        "breath_phase",
        "respiration_phase_window",
        "phase_samples",
        "phase_window",
        "bpf_input",
    }
)

SESSION_REQUIRED_FIELDS: Final = (
    "capture_schema_version",
    "session_id",
    "session_start_wall_time",
    "session_start_monotonic_time",
    "software_provenance",
    "integration_git_sha",
    "device_ids",
    "session_status",
    "runtime_capture_root",
)
SESSION_OPTIONAL_FIELDS: Final = (
    "integration_git_sha_unavailable_reason",
    "pi_runtime_version",
)
SESSION_ALLOWED_FIELDS: Final = frozenset(SESSION_REQUIRED_FIELDS + SESSION_OPTIONAL_FIELDS)

EVENT_REQUIRED_FIELDS: Final = (
    "capture_schema_version",
    "session_id",
    "capture_event_id",
    "event_kind",
    "sensor_type",
    "device_id",
    "boot_id",
    "packet_sequence",
    "device_uptime_ms",
    "source_measurement_event_id",
    "source_measurement_monotonic_ms",
    "source_timing_unavailable_reason",
    "pi_receive_wall_time",
    "pi_receive_monotonic_time",
    "parse_valid",
    "sensor_valid",
    "stale",
    "error_code",
    "error_reason",
    "payload",
)
EVENT_OPTIONAL_FIELDS: Final = (
    "device_id_unavailable_reason",
    "boot_id_unavailable_reason",
)
EVENT_ALLOWED_FIELDS: Final = frozenset(EVENT_REQUIRED_FIELDS + EVENT_OPTIONAL_FIELDS)

CO2_PAYLOAD_ALLOWED: Final = frozenset(
    {
        "co2_ppm",
        "measurement_identity_unavailable",
    }
)
THERMAL_PAYLOAD_ALLOWED: Final = frozenset(
    {
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
    }
)
PIR_PAYLOAD_ALLOWED: Final = frozenset(
    {
        "pir_motion",
        "is_transition",
    }
)
MMWAVE_PAYLOAD_ALLOWED: Final = frozenset(
    {
        "phase_payload_status",
        "respiration_rate_bpm",
        "heart_rate_bpm",
        "respiration_valid",
        "heart_valid",
    }
)
RUNTIME_PAYLOAD_ALLOWED: Final = frozenset(
    {
        "detail",
    }
)
