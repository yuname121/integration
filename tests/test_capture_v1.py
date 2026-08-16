from __future__ import annotations

import copy
import json
from io import StringIO
from pathlib import Path
import subprocess
from unittest import mock
import unittest
import uuid

from storage.capture_v1 import (
    CAPTURE_SCHEMA_FAMILY,
    CAPTURE_SCHEMA_VERSION,
    MMWAVE_PHASE_STATUS,
    RUNTIME_CAPTURE_ROOT,
    new_capture_event_id,
    new_session_id,
    validate_capture_event,
    validate_event_collection,
    validate_path,
    validate_session_manifest,
)
from storage.capture_v1.__main__ import main as capture_cli_main
from storage.capture_v1.identities import SESSION_ID_PATTERN


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "capture_v1"


def load_fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def codes(result) -> set[str]:
    return {issue.code for issue in result.errors}


class CaptureV1ContractTests(unittest.TestCase):
    def test_schema_version_is_independent_capture_domain(self) -> None:
        catalog = json.loads(
            (REPO_ROOT / "storage" / "capture_v1" / "schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(CAPTURE_SCHEMA_VERSION, "safenest.capture.v1")
        self.assertEqual(CAPTURE_SCHEMA_FAMILY, "SAFENEST_CAPTURE_V1")
        self.assertEqual(catalog["capture_schema_version"], CAPTURE_SCHEMA_VERSION)
        self.assertNotEqual(CAPTURE_SCHEMA_VERSION, "safenest.telemetry.v1")
        self.assertEqual(catalog["unknown_field_policy"], "reject")
        self.assertEqual(catalog["mmwave_phase_payload_status"], MMWAVE_PHASE_STATUS)

    def test_session_id_is_not_a_plain_timestamp(self) -> None:
        session_id = new_session_id(wall_time=1_786_872_612.0, entropy="aaaaaaaaaaaa")
        self.assertEqual(session_id, "sncap-20260816T093012Z-aaaaaaaaaaaa")
        self.assertRegex(session_id, SESSION_ID_PATTERN)
        other = new_session_id(wall_time=1_786_872_612.0, entropy="bbbbbbbbbbbb")
        self.assertNotEqual(session_id, other)

    def test_capture_event_id_is_uuid4_not_sensor_native(self) -> None:
        event_id = new_capture_event_id(lambda: uuid.UUID("11111111-1111-4111-8111-111111111111"))
        self.assertEqual(event_id, "11111111-1111-4111-8111-111111111111")
        self.assertNotEqual(event_id, "co2-meas-42")

    def test_valid_session_manifest_accepted(self) -> None:
        result = validate_session_manifest(load_fixture("session_valid.json"))
        self.assertTrue(result.ok, result.format_errors())

    def test_invalid_session_manifest_rejected(self) -> None:
        result = validate_session_manifest(load_fixture("session_invalid_missing_id.json"))
        self.assertFalse(result.ok)
        self.assertIn("MISSING_FIELD", codes(result))

    def test_valid_co2_event_accepted(self) -> None:
        result = validate_capture_event(load_fixture("event_co2_valid.json"))
        self.assertTrue(result.ok, result.format_errors())
        payload = load_fixture("event_co2_valid.json")["payload"]
        self.assertNotIn("humidity", payload)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("co2_slope", payload)

    def test_co2_without_source_timing_accepted(self) -> None:
        result = validate_capture_event(load_fixture("event_co2_without_source_timing.json"))
        self.assertTrue(result.ok, result.format_errors())

    def test_invalid_co2_with_humidity_rejected(self) -> None:
        result = validate_capture_event(load_fixture("event_co2_invalid.json"))
        self.assertFalse(result.ok)
        self.assertIn("FORBIDDEN_FIELD", codes(result))

    def test_stale_valid_co2_is_still_evidence(self) -> None:
        document = load_fixture("event_co2_stale_valid.json")
        result = validate_capture_event(document)
        self.assertTrue(result.ok, result.format_errors())
        self.assertTrue(document["stale"])
        self.assertTrue(document["parse_valid"])
        self.assertTrue(document["sensor_valid"])

    def test_valid_thermal_metadata_accepted(self) -> None:
        result = validate_capture_event(load_fixture("event_thermal_valid.json"))
        self.assertTrue(result.ok, result.format_errors())
        payload = load_fixture("event_thermal_valid.json")["payload"]
        self.assertNotIn("pixels", payload)

    def test_invalid_thermal_dimensions_rejected(self) -> None:
        result = validate_capture_event(load_fixture("event_thermal_invalid_dimensions.json"))
        self.assertFalse(result.ok)
        self.assertIn("DIMENSION_INVALID", codes(result))
        self.assertIn("VALUE_INVALID", codes(result))

    def test_invalid_thermal_reference_and_checksum_rejected(self) -> None:
        result = validate_capture_event(load_fixture("event_thermal_invalid_reference.json"))
        self.assertFalse(result.ok)
        self.assertIn("PAYLOAD_REFERENCE_INVALID", codes(result))
        self.assertIn("CHECKSUM_INVALID", codes(result))

    def test_valid_pir_event_accepted(self) -> None:
        result = validate_capture_event(load_fixture("event_pir_valid.json"))
        self.assertTrue(result.ok, result.format_errors())

    def test_mmwave_placeholder_accepted_and_phase_remains_deferred(self) -> None:
        document = load_fixture("event_mmwave_placeholder.json")
        result = validate_capture_event(document)
        self.assertTrue(result.ok, result.format_errors())
        self.assertEqual(document["payload"]["phase_payload_status"], MMWAVE_PHASE_STATUS)
        blocked = copy.deepcopy(document)
        blocked["payload"]["breath_phase"] = [0.1, 0.2]
        blocked_result = validate_capture_event(blocked)
        self.assertFalse(blocked_result.ok)
        self.assertIn("FORBIDDEN_FIELD", codes(blocked_result))

    def test_invalid_sensor_type_rejected(self) -> None:
        result = validate_capture_event(load_fixture("event_invalid_sensor_type.json"))
        self.assertFalse(result.ok)
        self.assertIn("SENSOR_TYPE_INVALID", codes(result))

    def test_missing_required_fields_rejected(self) -> None:
        result = validate_capture_event(load_fixture("event_missing_required.json"))
        self.assertFalse(result.ok)
        self.assertIn("MISSING_FIELD", codes(result))

    def test_malformed_capture_event_id_rejected(self) -> None:
        result = validate_capture_event(load_fixture("event_malformed_event_id.json"))
        self.assertFalse(result.ok)
        self.assertIn("CAPTURE_EVENT_ID_INVALID", codes(result))

    def test_invalid_timestamps_rejected(self) -> None:
        result = validate_capture_event(load_fixture("event_invalid_timestamp.json"))
        self.assertFalse(result.ok)
        self.assertIn("TIMESTAMP_INVALID", codes(result))

    def test_fake_zero_source_timing_rejected(self) -> None:
        result = validate_capture_event(load_fixture("event_fake_zero_source_timing.json"))
        self.assertFalse(result.ok)
        self.assertTrue({"FAKE_ZERO_FORBIDDEN", "SOURCE_PROVENANCE_INCOMPLETE"} & codes(result))

    def test_zero_is_allowed_when_the_value_is_actually_known(self) -> None:
        document = copy.deepcopy(load_fixture("event_co2_valid.json"))
        document["packet_sequence"] = 0
        document["device_uptime_ms"] = 0
        result = validate_capture_event(document)
        self.assertTrue(result.ok, result.format_errors())

    def test_unknown_fields_rejected(self) -> None:
        result = validate_capture_event(load_fixture("event_unknown_field.json"))
        self.assertFalse(result.ok)
        self.assertIn("UNKNOWN_FIELD", codes(result))

    def test_boolean_is_not_accepted_as_integer_sequence(self) -> None:
        document = copy.deepcopy(load_fixture("event_co2_without_source_timing.json"))
        document["packet_sequence"] = True
        result = validate_capture_event(document)
        self.assertFalse(result.ok)
        self.assertIn("TYPE_INVALID", codes(result))

    def test_duplicate_capture_event_ids_rejected(self) -> None:
        first = load_fixture("event_pir_valid.json")
        second = copy.deepcopy(first)
        result = validate_event_collection([first, second])
        self.assertFalse(result.ok)
        self.assertIn("CAPTURE_EVENT_ID_DUPLICATE", codes(result))

    def test_jsonl_fixture_accepted(self) -> None:
        result = validate_path(FIXTURES / "events_pir.jsonl")
        self.assertTrue(result.ok, result.format_errors())

    def test_cli_accepts_valid_and_rejects_invalid(self) -> None:
        with mock.patch("sys.stdout", new=StringIO()), mock.patch("sys.stderr", new=StringIO()):
            self.assertEqual(capture_cli_main(["validate", str(FIXTURES / "session_valid.json")]), 0)
            self.assertEqual(capture_cli_main(["validate", str(FIXTURES / "event_co2_invalid.json")]), 1)

    def test_runtime_captures_path_is_gitignored(self) -> None:
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "captures/demo-session/events_0001.jsonl"],
            cwd=REPO_ROOT,
        )
        self.assertEqual(ignored.returncode, 0)

    def test_synthetic_fixtures_are_not_gitignored(self) -> None:
        tracked = subprocess.run(
            ["git", "check-ignore", "-q", "tests/fixtures/capture_v1/session_valid.json"],
            cwd=REPO_ROOT,
        )
        self.assertEqual(tracked.returncode, 1)
        self.assertEqual(RUNTIME_CAPTURE_ROOT, "captures")

    def test_validator_does_not_mutate_input(self) -> None:
        document = load_fixture("event_co2_invalid.json")
        before = json.dumps(document, sort_keys=True)
        validate_capture_event(document)
        self.assertEqual(json.dumps(document, sort_keys=True), before)


if __name__ == "__main__":
    unittest.main()
