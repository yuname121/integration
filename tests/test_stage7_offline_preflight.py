"""Stage 7 Mac-offline deployment preflight and configuration contract tests."""

from __future__ import annotations

from pathlib import Path
import json
import unittest

from backend.runtime_status import runtime_status_document
from backend.views import status_document
from deployment.verify_bundle import verify
from hil.preflight import EXPECTED_ENV_KEYS, offline_preflight_document
from tests.test_runtime_status import ai_result, sensor, state


ROOT = Path(__file__).resolve().parent.parent


class Stage7OfflinePreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = offline_preflight_document(ROOT)

    def test_offline_preflight_passes_without_claiming_pi_or_sensors(self) -> None:
        self.assertTrue(self.document["ok"], json.dumps(self.document, indent=2)[:4000])
        self.assertEqual(self.document["mode"], "MAC_OFFLINE_PREFLIGHT")
        self.assertEqual(self.document["pi_checks"], "NOT_RUN")
        self.assertEqual(self.document["sensor_checks"], "NOT_RUN")
        self.assertTrue(self.document["mac_pass_does_not_imply_pi_pass"])

    def test_runtime_entrypoint_and_lcd_web_assets_exist(self) -> None:
        names = {item["name"]: item for item in self.document["checks"]}
        self.assertTrue(names["file_backend_run_backend.py"]["passed"])
        self.assertTrue(names["file_deployment_run_pi.sh"]["passed"])
        self.assertTrue(names["file_web_dashboard_index.html"]["passed"])
        self.assertTrue(names["file_sources_display-test2_raspberry_pi_lcd_static_display.html"]["passed"])
        self.assertTrue(names["dashboard_runtime_badge"]["passed"])
        self.assertTrue(names["lcd_consumes_backend_runtime_status"]["passed"])

    def test_listener_contract_remains_tcp_9000_udp_5005(self) -> None:
        names = {item["name"]: item for item in self.document["checks"]}
        self.assertTrue(names["tcp_default_port_9000"]["passed"])
        self.assertTrue(names["udp_default_port_5005"]["passed"])
        self.assertTrue(names["http_default_port_8000"]["passed"])

    def test_artifact_selection_does_not_silently_activate_tb5_or_old_b(self) -> None:
        names = {item["name"]: item for item in self.document["checks"]}
        self.assertTrue(names["thermal_production_path_is_historical_v0_1_0"]["passed"])
        self.assertTrue(names["mmwave_primary_deployment_blocked"]["passed"])
        self.assertTrue(names["model_thermal_sha256"]["passed"])
        self.assertTrue(names["model_mmwave_sha256"]["passed"])

    def test_no_developer_absolute_runtime_paths(self) -> None:
        names = {item["name"]: item for item in self.document["checks"]}
        self.assertTrue(names["no_developer_absolute_runtime_paths"]["passed"])
        self.assertTrue(names["db_path_repository_relative"]["passed"])
        self.assertTrue(names["venv_path_repository_relative"]["passed"])

    def test_env_example_covers_runtime_keys(self) -> None:
        names = {item["name"]: item for item in self.document["checks"]}
        self.assertTrue(names["env_example_documents_runtime_keys"]["passed"], names["env_example_documents_runtime_keys"])
        self.assertIn("SAFENEST_VENV_PATH", EXPECTED_ENV_KEYS)
        self.assertIn("SAFENEST_THERMAL_UDP_PORT", EXPECTED_ENV_KEYS)

    def test_status_contract_and_partial_availability_are_preserved(self) -> None:
        names = {item["name"]: item for item in self.document["checks"]}
        self.assertTrue(names["ready_with_limitations_preserved"]["passed"])
        self.assertTrue(names["thermal_ai_blocked_reason_preserved"]["passed"])
        self.assertTrue(names["pir_not_applicable_preserved"]["passed"])
        publication = {
            "timestamp": 1.0,
            "state": state(),
            "ai": {"ai": {"co2": ai_result(available=True)}},
            "risk": {"system_health": "HEALTHY", "components": {}},
            "emergency": {},
        }
        document = status_document(publication)
        self.assertEqual(document["runtime_status"]["status"], "READY_WITH_LIMITATIONS")
        self.assertEqual(document["thermal"]["runtime_status"]["ai_status"], "BLOCKED")
        self.assertEqual(document["thermal"]["runtime_status"]["sensor_status"], "AVAILABLE")
        self.assertEqual(document["pir"]["runtime_status"]["ai_status"], "NOT_APPLICABLE")
        self.assertEqual(document["pir"]["runtime_status"]["sensor_value_status"], "NO_MOTION")
        self.assertEqual(document["co2"]["runtime_status"]["ai_status"], "ACTIVE")
        blocked = runtime_status_document(state(), {})
        self.assertNotEqual(blocked["sensors"]["co2"]["ai_status"], "ACTIVE")

    def test_bundle_verifier_includes_runtime_status_module(self) -> None:
        result = verify(ROOT)
        self.assertNotIn("backend/runtime_status.py", result["missing"])
        self.assertNotIn("backend/views.py", result["missing"])


if __name__ == "__main__":
    unittest.main()
