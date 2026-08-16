"""Capture v1 schema, identities, and deterministic validator.

This package defines the storage language for future Raspberry Pi Capture.
It does not write runtime sessions.
"""

from .constants import (
    CAPTURE_SCHEMA_FAMILY,
    CAPTURE_SCHEMA_VERSION,
    MMWAVE_PHASE_STATUS,
    RUNTIME_CAPTURE_ROOT,
    SOURCE_TIMING_UNAVAILABLE,
)
from .errors import CaptureValidationIssue, CaptureValidationResult
from .identities import new_capture_event_id, new_frame_id, new_session_id
from .validator import (
    validate_capture_event,
    validate_document,
    validate_event_collection,
    validate_path,
    validate_session_manifest,
)

__all__ = [
    "CAPTURE_SCHEMA_FAMILY",
    "CAPTURE_SCHEMA_VERSION",
    "MMWAVE_PHASE_STATUS",
    "RUNTIME_CAPTURE_ROOT",
    "SOURCE_TIMING_UNAVAILABLE",
    "CaptureValidationIssue",
    "CaptureValidationResult",
    "new_capture_event_id",
    "new_frame_id",
    "new_session_id",
    "validate_capture_event",
    "validate_document",
    "validate_event_collection",
    "validate_path",
    "validate_session_manifest",
]
