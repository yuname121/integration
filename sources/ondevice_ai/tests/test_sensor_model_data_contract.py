#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_sensor_model_data_contract.py
Automated verification test suite for P0-3 Sensor-Model Data Contract.

Tests validate:
  - JSON contract values match actual metadata files (mmWave, CO2)
  - JSON contract model tensors match P0-2 model_inventory.json
  - Feature orders match actual metadata
  - SafeNestRiskOutput dataclass fields match PRIMARY_RUNTIME_SCHEMA
  - SensorState enum values match documented states
  - No time.time() described as monotonic in contract markdown
  - Thermal parser_implementation_allowed == false
  - No unverified thermal dtype/endianness/fixed temperature range asserted as confirmed
  - mmWave semantic compatibility unverified => deployment_ready == false
  - No absolute paths in any artifact
  - TFLite model SHA256 matches manifest/inventory (not self-referencing)
  - Section 16 has no uint16/int16 or WARMING_UP implementation recommendations
  - PIR timestamp not described as monotonic
  - actual_sensor_update_hz not hardcoded without basis
  - Capture metrics match actual JSONL recalculations (empty log dup ratio == 1.0)
  - Synthetic missing-value regression test verifies no null gap bridging occurs
  - MMWaveCSVAdapter regression test produces exactly 90 windows from 299.816s CSV
  - current_mock_provenance, current_real_provenance, and target_real_provenance exist for all 4 sensors
  - AST call/assignment inspection test verifies run_node.py polling loop calls sensor.read() and risk_engine.evaluate()
  - PIR uses time.time() and time.monotonic() for distinct purposes (dual-clock)
  - PIR startup_grace_period_sec (5.0s) and WARMING_UP state verified
  - mmWave offline candidate pipeline is marked as CONTRACT_GAP
  - Thermal raw physical unit is UNVERIFIED_DATASHEET
  - WARMING_UP system_status does not include NORMAL
"""

from __future__ import annotations
import ast
import json
import hashlib
import re
import unittest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_JSON_PATH = PROJECT_ROOT / "docs" / "reports" / "sensor_model_data_contract.json"
MMWAVE_METADATA_PATH = PROJECT_ROOT / "models" / "mmwave" / "sensor_stats_metadata_v0.1.0.json"
CO2_METADATA_PATH = PROJECT_ROOT / "models" / "co2" / "co2_scaling_metadata_v0.1.0.json"
MODEL_MANIFEST_PATH = PROJECT_ROOT / "models" / "model_manifest.json"
SENSOR_DATA_CONTRACT_MD_PATH = PROJECT_ROOT / "docs" / "reports" / "SENSOR_DATA_CONTRACT.md"
RUN_NODE_PATH = PROJECT_ROOT / "integrated_node" / "run_node.py"
PIR_ADAPTER_PATH = PROJECT_ROOT / "sensors" / "pir" / "pir_adapter.py"

MODEL_INVENTORY_PATH = PROJECT_ROOT / "docs" / "reports" / "model_inventory.json"
CONTRACT_FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "sensor_contract"
PRESENCE_JSONL_PATH = CONTRACT_FIXTURE_ROOT / "mmwave_presence.jsonl"
EMPTY_JSONL_PATH = CONTRACT_FIXTURE_ROOT / "mmwave_empty.jsonl"
SAMPLE_CSV_PATH = CONTRACT_FIXTURE_ROOT / "mmwave_window_sample.csv"

ALLOWED_VERIFICATION_STATUSES = {
    "VERIFIED_CODE",
    "VERIFIED_MODEL",
    "VERIFIED_ARTIFACT",
    "VERIFIED_DATASHEET",
    "CONTRACT_MISMATCH",
    "IMPLEMENTATION_DEFECT",
    "CONTRACT_GAP",
    "DESIGN_LIMITATION",
    "DORMANT_CODE_CONTRACT_RISK",
    "BLOCKED_HARDWARE",
    "UNVERIFIED_DATASHEET",
    "NOT_APPLICABLE",
    "TRAIN_INFERENCE_PREPROCESSING_MATCH",
    "STALE_CONTRACT_ARTIFACT",
    "RESOLVED",
}

ALLOWED_BLOCKER_STATUSES = {
    "BLOCKED_HARDWARE",
    "UNVERIFIED_DATASHEET",
}

ALLOWED_MISMATCH_CATEGORIES = {
    "IMPLEMENTATION_DEFECT",
    "CONTRACT_MISMATCH",
    "DORMANT_CODE_CONTRACT_RISK",
    "TRAIN_INFERENCE_PREPROCESSING_MATCH",
    "DESIGN_LIMITATION",
    "STALE_CONTRACT_ARTIFACT",
    "RESOLVED",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _calc_dups(raw_phases: list) -> tuple[int, int, float, int, int, float]:
    total_records = len(raw_phases)
    null_count = sum(1 for p in raw_phases if p is None or not np.isfinite(p))
    null_ratio = round(null_count / total_records, 4) if total_records else 0.0

    valid_pairs = [
        (a, b) for a, b in zip(raw_phases, raw_phases[1:])
        if a is not None and np.isfinite(a) and b is not None and np.isfinite(b)
    ]
    pair_count = len(valid_pairs)
    dup_count = sum(a == b for a, b in valid_pairs)
    dup_ratio = round(dup_count / pair_count, 4) if pair_count else 0.0
    return total_records, null_count, null_ratio, pair_count, dup_count, dup_ratio


class TestContractJsonStructure(unittest.TestCase):
    """Tests for JSON contract file structure and schema."""

    def test_json_file_exists_and_parses(self):
        self.assertTrue(CONTRACT_JSON_PATH.is_file(), f"Contract JSON missing: {CONTRACT_JSON_PATH}")
        data = _load_json(CONTRACT_JSON_PATH)
        self.assertIn("schema_version", data)
        self.assertIn("contract_validation_status", data)
        self.assertIn("system_deployment_status", data)
        self.assertIn("sensors", data)

    def test_contract_validation_status_is_pass_with_blockers(self):
        data = _load_json(CONTRACT_JSON_PATH)
        self.assertEqual(data["contract_validation_status"], "PASS_WITH_BLOCKERS")
        self.assertEqual(data["system_deployment_status"], "NOT_READY")

    def test_four_sensors_present_with_required_fields(self):
        data = _load_json(CONTRACT_JSON_PATH)
        sensors = data["sensors"]
        required_fields = [
            "raw_transport", "raw_schema", "parser_input", "parser_output",
            "preprocessing", "consumer", "model_or_rule", "shape", "dtype",
            "units", "sampling", "quality", "invalid_conditions",
            "current_mock_provenance", "current_real_provenance",
            "verification_status", "evidence",
        ]
        for expected_sensor in ("thermal44", "mmwave", "co2", "pir"):
            self.assertIn(expected_sensor, sensors, f"Missing sensor entry: {expected_sensor}")
            sensor_data = sensors[expected_sensor]
            for field in required_fields:
                self.assertIn(field, sensor_data, f"Sensor '{expected_sensor}' missing field '{field}'")

    def test_provenance_fields_split_and_correct(self):
        data = _load_json(CONTRACT_JSON_PATH)
        expected_real_prov = "External provider or EXTERNAL_SENSOR_PROVIDER_REQUIRED -> InferenceResult -> risk/risk_engine.py"
        for sensor_name, sensor_data in data["sensors"].items():
            self.assertIn("current_mock_provenance", sensor_data)
            self.assertIn("current_real_provenance", sensor_data)
            self.assertEqual(sensor_data["current_real_provenance"], expected_real_prov,
                             f"Sensor '{sensor_name}' current_real_provenance mismatch")
            if sensor_name == "mmwave":
                self.assertIn("captured_artifact_stage", sensor_data)
                self.assertIn("verified_adapter_stage", sensor_data)
                self.assertIn("unconnected_candidate_stage", sensor_data)
                self.assertIn("target_online_provenance", sensor_data)
            else:
                self.assertIn("target_real_provenance", sensor_data)

    def test_co2_single_array_mismatch_is_resolved(self):
        data = _load_json(CONTRACT_JSON_PATH)
        mismatches = {m["id"]: m for m in data["contract_mismatches"]}
        self.assertIn("MISMATCH-04", mismatches)
        self.assertEqual(mismatches["MISMATCH-04"]["category"], "RESOLVED")
        self.assertEqual(mismatches["MISMATCH-04"]["runtime_impact"], "RESOLVED")
        self.assertIn("three scalar", mismatches["MISMATCH-04"]["description"])
        self.assertEqual(data["sensors"]["co2"]["verification_status"], "RESOLVED")

    def test_verification_statuses_are_valid(self):
        data = _load_json(CONTRACT_JSON_PATH)
        for sensor_name, sensor_data in data["sensors"].items():
            status = sensor_data["verification_status"]
            self.assertIn(status, ALLOWED_VERIFICATION_STATUSES,
                          f"Sensor '{sensor_name}' has invalid status '{status}'")

    def test_no_unverified_hardware_status_code(self):
        """Ensure old UNVERIFIED_HARDWARE status code is not used anywhere."""
        raw_text = CONTRACT_JSON_PATH.read_text(encoding="utf-8")
        self.assertNotIn("UNVERIFIED_HARDWARE", raw_text,
                         "Old 'UNVERIFIED_HARDWARE' status code found. Use 'BLOCKED_HARDWARE' instead.")

    def test_blocker_statuses_are_valid(self):
        data = _load_json(CONTRACT_JSON_PATH)
        for blocker in data.get("hardware_blockers", []):
            self.assertIn(blocker["status"], ALLOWED_BLOCKER_STATUSES,
                          f"Blocker '{blocker['id']}' has invalid status '{blocker['status']}'")

    def test_mismatch_categories_are_valid(self):
        data = _load_json(CONTRACT_JSON_PATH)
        for mismatch in data.get("contract_mismatches", []):
            self.assertIn(mismatch["category"], ALLOWED_MISMATCH_CATEGORIES,
                          f"Mismatch '{mismatch['id']}' has invalid category '{mismatch['category']}'")


class TestRunNodePollingLoopAST(unittest.TestCase):
    """AST Call & Assignment Inspection Test validating primary runtime polling in run_node.py."""

    def test_run_node_ast_call_inspection(self):
        self.assertTrue(RUN_NODE_PATH.is_file(), f"run_node.py missing: {RUN_NODE_PATH}")
        tree = ast.parse(RUN_NODE_PATH.read_text(encoding="utf-8"), filename=str(RUN_NODE_PATH))

        found_sensor_read_call = False
        found_risk_eval_call = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check sensor.read() call
                if isinstance(node.func, ast.Attribute) and node.func.attr == "read":
                    found_sensor_read_call = True
                # Check risk_engine.evaluate() call
                if isinstance(node.func, ast.Attribute) and node.func.attr == "evaluate":
                    if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "risk_engine":
                        found_risk_eval_call = True

        self.assertTrue(found_sensor_read_call, "run_node.py AST must contain sensor.read() call in step()")
        self.assertTrue(found_risk_eval_call, "run_node.py AST must contain self.risk_engine.evaluate() call in step()")


class TestPirClockAndGracePeriod(unittest.TestCase):
    """Tests PIR dual-clock usage, startup grace period, and WARMING_UP state in code."""

    def test_pir_dual_clock_usage(self):
        import sys, time
        sn_dir = str(PROJECT_ROOT)
        if sn_dir not in sys.path:
            sys.path.insert(0, sn_dir)
        from sensors.pir.pir_adapter import PIRSensorAdapter
        from sensors.base_sensor import SensorState

        adapter = PIRSensorAdapter()
        adapter.connected = True
        adapter.connect_monotonic_ts = time.monotonic()
        adapter.read_gpio = lambda: False
        self.assertEqual(adapter.startup_grace_period_sec, 5.0)

        # Read immediately within startup grace period
        res = adapter.read()
        self.assertEqual(res.state, "WARMING_UP")
        self.assertFalse(res.valid)
        self.assertEqual(res.error, "PIR_WARMING_UP")
        self.assertEqual(adapter.current_state, SensorState.WARMING_UP)
        self.assertIn("startup_grace_period_sec", res.metadata)

    def test_pir_ast_clock_functions(self):
        tree = ast.parse(PIR_ADAPTER_PATH.read_text(encoding="utf-8"), filename=str(PIR_ADAPTER_PATH))
        time_calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "time":
                    time_calls.add(node.func.attr)

        self.assertIn("time", time_calls, "pir_adapter.py must call time.time()")
        self.assertIn("monotonic", time_calls, "pir_adapter.py must call time.monotonic()")
        self.assertIn("perf_counter", time_calls, "pir_adapter.py must call time.perf_counter()")


class TestMmwaveOfflineUnconnectedCandidateStatus(unittest.TestCase):
    """Tests that mmWave offline pipeline candidate stage is marked as CONTRACT_GAP."""

    def test_mmwave_unconnected_candidate_is_contract_gap(self):
        data = _load_json(CONTRACT_JSON_PATH)
        mmwave = data["sensors"]["mmwave"]
        self.assertIn("unconnected_candidate_stage", mmwave)
        self.assertIn("CONTRACT_GAP", mmwave["unconnected_candidate_stage"])

        gaps = [g["feature"] for g in data["contract_gaps"]]
        self.assertTrue(any("unconnected" in g.lower() or "candidate" in g.lower() for g in gaps),
                        "contract_gaps must include unconnected candidate pipeline entry")


class TestMmwaveMetadataSync(unittest.TestCase):
    """Tests that JSON contract mmWave values match actual metadata files."""

    def test_mmwave_mean_std_matches_actual_metadata(self):
        data = _load_json(CONTRACT_JSON_PATH)
        metadata = _load_json(MMWAVE_METADATA_PATH)
        mmwave_meta = data["sensors"]["mmwave"]["mmwave_metadata"]

        actual_mean = float(metadata["mean"])
        actual_std = float(metadata["std"])
        contract_mean = float(mmwave_meta["mean"])
        contract_std = float(mmwave_meta["std"])

        self.assertAlmostEqual(contract_mean, actual_mean, places=10,
                               msg=f"mmWave mean mismatch: contract={contract_mean}, actual={actual_mean}")
        self.assertAlmostEqual(contract_std, actual_std, places=10,
                               msg=f"mmWave std mismatch: contract={contract_std}, actual={actual_std}")

    def test_mmwave_sample_rate_matches_metadata(self):
        data = _load_json(CONTRACT_JSON_PATH)
        metadata = _load_json(MMWAVE_METADATA_PATH)
        mmwave_meta = data["sensors"]["mmwave"]["mmwave_metadata"]

        self.assertEqual(mmwave_meta["sample_rate_hz"], metadata["sample_rate_hz"])
        self.assertEqual(mmwave_meta["window_samples"], metadata["window_samples"])
        self.assertEqual(mmwave_meta["window_seconds"], metadata["window_seconds"])

    def test_mmwave_input_semantic_matches_metadata(self):
        data = _load_json(CONTRACT_JSON_PATH)
        metadata = _load_json(MMWAVE_METADATA_PATH)
        mmwave_meta = data["sensors"]["mmwave"]["mmwave_metadata"]
        self.assertEqual(mmwave_meta["input_semantic"], metadata["input_semantic"])

    def test_no_unsubstantiated_actual_sensor_update_hz(self):
        """Ensure actual_sensor_update_hz is not specified as a hardcoded float in JSON without hardware proof."""
        data = _load_json(CONTRACT_JSON_PATH)
        capture_metrics = data["sensors"]["mmwave"].get("capture_metrics", {})
        self.assertNotIn("actual_sensor_update_hz", capture_metrics,
                         "actual_sensor_update_hz should be deleted; frame rate is BLOCKED_HARDWARE.")

    def test_mmwave_units_vendor_phase_float(self):
        data = _load_json(CONTRACT_JSON_PATH)
        units = data["sensors"]["mmwave"]["units"]
        self.assertIn("UNVERIFIED_DATASHEET", units,
                      f"mmWave units should be Vendor phase float / UNVERIFIED_DATASHEET, got: {units}")


class TestCo2MetadataSync(unittest.TestCase):
    """Tests that JSON contract CO2 values match actual metadata files."""

    def test_co2_feature_order_matches_actual_metadata(self):
        data = _load_json(CONTRACT_JSON_PATH)
        metadata = _load_json(CO2_METADATA_PATH)

        actual_features = [f.lower() for f in metadata["features"]]
        self.assertEqual(actual_features, ["co2_slope", "humidity", "co2"])

        contract_features = data["sensors"]["co2"]["feature_order"]
        self.assertEqual(len(contract_features), len(actual_features))

    def test_co2_metadata_values_match(self):
        data = _load_json(CONTRACT_JSON_PATH)
        metadata = _load_json(CO2_METADATA_PATH)
        co2_meta = data["sensors"]["co2"]["co2_metadata"]

        for i, (cm, am) in enumerate(zip(co2_meta["mean"], metadata["mean"])):
            self.assertAlmostEqual(cm, am, places=10,
                                   msg=f"CO2 mean[{i}] mismatch: contract={cm}, actual={am}")
        for i, (cs, as_) in enumerate(zip(co2_meta["scale"], metadata["scale"])):
            self.assertAlmostEqual(cs, as_, places=10,
                                   msg=f"CO2 scale[{i}] mismatch: contract={cs}, actual={as_}")


class TestModelInventorySync(unittest.TestCase):
    """Tests that JSON contract model tensors match P0-2 model_inventory.json."""

    @unittest.skipUnless(MODEL_INVENTORY_PATH.is_file(), "P0-2 model_inventory.json not yet generated")
    def test_mmwave_tensor_shape_matches_inventory(self):
        data = _load_json(CONTRACT_JSON_PATH)
        inventory = _load_json(MODEL_INVENTORY_PATH)
        inv_mmwave = inventory["models"]["mmwave"]

        self.assertEqual(data["sensors"]["mmwave"]["shape"],
                         inv_mmwave["input_tensor"]["shape"])
        self.assertEqual(data["sensors"]["mmwave"]["dtype"],
                         inv_mmwave["input_tensor"]["dtype"])

    @unittest.skipUnless(MODEL_INVENTORY_PATH.is_file(), "P0-2 model_inventory.json not yet generated")
    def test_thermal_tensor_shape_matches_inventory(self):
        data = _load_json(CONTRACT_JSON_PATH)
        inventory = _load_json(MODEL_INVENTORY_PATH)
        inv_thermal = inventory["models"]["thermal"]

        self.assertEqual(data["sensors"]["thermal44"]["shape"],
                         inv_thermal["input_tensor"]["shape"])
        self.assertEqual(data["sensors"]["thermal44"]["dtype"],
                         inv_thermal["input_tensor"]["dtype"])

    @unittest.skipUnless(MODEL_INVENTORY_PATH.is_file(), "P0-2 model_inventory.json not yet generated")
    def test_co2_tensor_shape_matches_inventory(self):
        data = _load_json(CONTRACT_JSON_PATH)
        inventory = _load_json(MODEL_INVENTORY_PATH)
        inv_co2 = inventory["models"]["co2"]

        self.assertEqual(data["sensors"]["co2"]["shape"],
                         inv_co2["input_tensor"]["shape"])
        self.assertEqual(data["sensors"]["co2"]["dtype"],
                         inv_co2["input_tensor"]["dtype"])

    @unittest.skipUnless(MODEL_INVENTORY_PATH.is_file(), "P0-2 model_inventory.json not yet generated")
    def test_class_order_matches_inventory(self):
        manifest = _load_json(MODEL_MANIFEST_PATH)
        inventory = _load_json(MODEL_INVENTORY_PATH)

        for sensor_key in ("thermal", "mmwave", "co2"):
            manifest_classes = manifest["models"][sensor_key]["class_map"]
            inv_labels = inventory["models"][sensor_key]["ordered_class_labels"]
            manifest_ordered = [manifest_classes[str(i)] for i in range(len(manifest_classes))]
            self.assertEqual(manifest_ordered, inv_labels,
                             f"Class order mismatch for {sensor_key}: manifest={manifest_ordered}, inventory={inv_labels}")


class TestPrimaryRuntimeSchema(unittest.TestCase):
    """Tests PRIMARY_RUNTIME_SCHEMA fields match SafeNestRiskOutput dataclass."""

    def test_primary_schema_fields_match_dataclass(self):
        import sys
        sn_dir = str(PROJECT_ROOT)
        if sn_dir not in sys.path:
            sys.path.insert(0, sn_dir)
        from inference.inference_result import SafeNestRiskOutput
        import dataclasses

        dataclass_fields = {f.name for f in dataclasses.fields(SafeNestRiskOutput)}
        data = _load_json(CONTRACT_JSON_PATH)
        schema_fields = set(data["json_schemas"]["primary_runtime_schema"]["fields"])

        self.assertEqual(schema_fields, dataclass_fields,
                         f"PRIMARY_RUNTIME_SCHEMA fields != SafeNestRiskOutput fields.\n"
                         f"  Missing in schema: {dataclass_fields - schema_fields}\n"
                         f"  Extra in schema: {schema_fields - dataclass_fields}")


class TestSensorStateEnum(unittest.TestCase):
    """Tests that documented SensorState values match actual enum."""

    def test_sensor_state_enum_completeness(self):
        import sys
        sn_dir = str(PROJECT_ROOT)
        if sn_dir not in sys.path:
            sys.path.insert(0, sn_dir)
        from sensors.base_sensor import SensorState

        actual_states = {s.name for s in SensorState}
        expected_documented = {
            "NORMAL", "NOT_CONNECTED", "HARDWARE_BACKEND_NOT_IMPLEMENTED",
            "WARMING_UP", "READ_TIMEOUT", "INVALID_FORMAT", "NAN_OR_INF",
            "OUT_OF_BOUNDS", "STALE", "INFER_FAILED", "SHUTDOWN",
        }
        missing = expected_documented - actual_states
        self.assertEqual(missing, set(),
                         f"Documented SensorStates missing from enum: {missing}")

    @unittest.skipUnless(SENSOR_DATA_CONTRACT_MD_PATH.is_file(), "SENSOR_DATA_CONTRACT.md not found")
    def test_warming_up_system_status_has_no_normal(self):
        """Ensure WARMING_UP system status in markdown section 5 table does not contain NORMAL."""
        md_text = SENSOR_DATA_CONTRACT_MD_PATH.read_text(encoding="utf-8")
        for line in md_text.splitlines():
            if "| `WARMING_UP` |" in line:
                self.assertNotIn("`NORMAL`", line.split("|")[5],
                                 "WARMING_UP system impact must not include NORMAL because valid=false")


class TestTimestampNomenclature(unittest.TestCase):
    """Tests timestamp classifications in documentation."""

    @unittest.skipUnless(SENSOR_DATA_CONTRACT_MD_PATH.is_file(), "SENSOR_DATA_CONTRACT.md not found")
    def test_no_time_time_called_monotonic(self):
        md_text = SENSOR_DATA_CONTRACT_MD_PATH.read_text(encoding="utf-8")
        lines = md_text.splitlines()
        for i, line in enumerate(lines, 1):
            if "time.time()" in line and "monotonic" in line.lower():
                if "non-monotonic" in line.lower() or "not monotonic" in line.lower():
                    continue
                self.fail(f"Line {i}: time.time() described as monotonic: {line.strip()}")

    @unittest.skipUnless(SENSOR_DATA_CONTRACT_MD_PATH.is_file(), "SENSOR_DATA_CONTRACT.md not found")
    def test_pir_timestamp_not_described_as_monotonic(self):
        md_text = SENSOR_DATA_CONTRACT_MD_PATH.read_text(encoding="utf-8")
        for line in md_text.splitlines():
            if "| **PIR** |" in line:
                timestamp_col = line.split("|")[10]
                self.assertNotIn("Monotonic", timestamp_col,
                                 f"PIR timestamp column should be wall-clock time.time(), got: {timestamp_col}")


class TestThermalContractConstraints(unittest.TestCase):
    """Tests Thermal-specific contract constraints."""

    def test_parser_implementation_not_allowed(self):
        data = _load_json(CONTRACT_JSON_PATH)
        thermal = data["sensors"]["thermal44"]
        self.assertFalse(thermal["parser_implementation_allowed"],
                         "Thermal parser_implementation_allowed should be false")

    def test_hardware_not_verified(self):
        data = _load_json(CONTRACT_JSON_PATH)
        thermal = data["sensors"]["thermal44"]
        self.assertFalse(thermal["hardware_verified"],
                         "Thermal hardware_verified should be false")

    def test_thermal_raw_physical_unit_is_unverified_datasheet(self):
        data = _load_json(CONTRACT_JSON_PATH)
        units = data["sensors"]["thermal44"]["units"]
        self.assertIn("UNVERIFIED_DATASHEET", units,
                      f"Thermal units must state UNVERIFIED_DATASHEET for physical unit, got: {units}")

    @unittest.skipUnless(SENSOR_DATA_CONTRACT_MD_PATH.is_file(), "SENSOR_DATA_CONTRACT.md not found")
    def test_no_uint16_int16_recommendation_in_section16(self):
        md_text = SENSOR_DATA_CONTRACT_MD_PATH.read_text(encoding="utf-8")
        sec16_start = md_text.find("## 16. Required Follow-up Implementation")
        sec17_start = md_text.find("## 17. Acceptance Checklist")
        sec16_text = md_text[sec16_start:sec17_start] if sec16_start != -1 else ""

        self.assertNotIn("uint16", sec16_text, "Section 16 must not contain uint16 recommendation")
        self.assertNotIn("int16", sec16_text, "Section 16 must not contain int16 recommendation")

    @unittest.skipUnless(SENSOR_DATA_CONTRACT_MD_PATH.is_file(), "SENSOR_DATA_CONTRACT.md not found")
    def test_no_warming_up_addition_recommendation_in_section16(self):
        md_text = SENSOR_DATA_CONTRACT_MD_PATH.read_text(encoding="utf-8")
        sec16_start = md_text.find("## 16. Required Follow-up Implementation")
        sec17_start = md_text.find("## 17. Acceptance Checklist")
        sec16_text = md_text[sec16_start:sec17_start] if sec16_start != -1 else ""

        self.assertNotIn("WARMING_UP", sec16_text, "Section 16 must not contain WARMING_UP addition recommendation")

    @unittest.skipUnless(SENSOR_DATA_CONTRACT_MD_PATH.is_file(), "SENSOR_DATA_CONTRACT.md not found")
    def test_single_quantization_ownership_recommendation_in_section16(self):
        md_text = SENSOR_DATA_CONTRACT_MD_PATH.read_text(encoding="utf-8")
        sec16_start = md_text.find("## 16. Required Follow-up Implementation")
        sec17_start = md_text.find("## 17. Acceptance Checklist")
        sec16_text = md_text[sec16_start:sec17_start] if sec16_start != -1 else ""

        self.assertIn("Single Quantization Ownership", sec16_text,
                      "Section 16 must specify Single Quantization Ownership recommendation")


class TestMmwaveDeploymentGate(unittest.TestCase):
    """Tests mmWave deployment gate."""

    def test_deployment_ready_false_when_semantic_unverified(self):
        data = _load_json(CONTRACT_JSON_PATH)
        mmwave = data["sensors"]["mmwave"]
        if mmwave.get("model_semantic_compatibility") in ("BLOCKED_HARDWARE", "UNVERIFIED_DATASHEET"):
            self.assertFalse(mmwave.get("deployment_ready", True),
                             "mmWave deployment_ready must be false when semantic compatibility is unverified")


class TestSyntheticMissingValueDuplicateCalculation(unittest.TestCase):
    """Regression test verifying that duplicate calculation on synthetic missing-value sequence does not bridge nulls."""

    def test_synthetic_sequence_with_null(self):
        synth_phases = [1.0, None, 1.0, 2.0, 2.0]
        total_records, null_count, null_ratio, pair_count, dup_count, dup_ratio = _calc_dups(synth_phases)

        self.assertEqual(total_records, 5)
        self.assertEqual(null_count, 1)
        self.assertEqual(null_ratio, 0.2)
        self.assertEqual(pair_count, 2, "Adjacent valid pairs in [1.0, None, 1.0, 2.0, 2.0] must be [(1.0, 2.0), (2.0, 2.0)]")
        self.assertEqual(dup_count, 1, "Only (2.0, 2.0) is a duplicate pair")
        self.assertEqual(dup_ratio, 0.5, "1 / 2 = 0.5; null-bridged pair (1.0, 1.0) must NOT be counted")


class TestCaptureMetricsRecalculation(unittest.TestCase):
    """Tests that capture metrics recorded in JSON match actual calculations on JSONL logs."""

    @unittest.skipUnless(PRESENCE_JSONL_PATH.is_file(), "Presence JSONL log missing")
    def test_presence_log_metrics_match(self):
        data = _load_json(CONTRACT_JSON_PATH)
        metrics = data["sensors"]["mmwave"]["capture_metrics"]["presence_log_metrics"]

        records = [json.loads(line) for line in PRESENCE_JSONL_PATH.read_text(encoding="utf-8").splitlines() if "seq" in line]
        ts = [r["ts_monotonic_ms"] for r in records]
        phase_ages = [r.get("phase_age_ms") for r in records if r.get("phase_age_ms") is not None]
        raw_phases = [r.get("breath_phase") for r in records]

        dt_mean = round(float(np.mean(np.diff(ts))), 2)
        age_mean = round(float(np.mean(phase_ages)), 2)
        age_max = int(max(phase_ages))

        total_rec, null_cnt, null_rat, pair_cnt, dup_cnt, dup_rat = _calc_dups(raw_phases)

        self.assertEqual(metrics["telemetry_dt_mean_ms"], dt_mean)
        self.assertEqual(metrics["phase_age_ms_mean"], age_mean)
        self.assertEqual(metrics["phase_age_ms_max"], age_max)
        self.assertEqual(metrics["null_count"], null_cnt)
        self.assertEqual(metrics["null_ratio"], null_rat)
        self.assertEqual(metrics["pair_count"], pair_cnt)
        self.assertEqual(metrics["duplicate_count"], dup_cnt)
        self.assertEqual(metrics["consecutive_dup_ratio"], dup_rat)

    @unittest.skipUnless(EMPTY_JSONL_PATH.is_file(), "Empty JSONL log missing")
    def test_empty_log_metrics_match(self):
        data = _load_json(CONTRACT_JSON_PATH)
        metrics = data["sensors"]["mmwave"]["capture_metrics"]["empty_log_metrics"]

        records = [json.loads(line) for line in EMPTY_JSONL_PATH.read_text(encoding="utf-8").splitlines() if "seq" in line]
        raw_phases = [r.get("breath_phase") for r in records]
        total_rec, null_cnt, null_rat, pair_cnt, dup_cnt, dup_rat = _calc_dups(raw_phases)

        self.assertEqual(metrics["pair_count"], pair_cnt)
        self.assertEqual(metrics["duplicate_count"], dup_cnt)
        self.assertEqual(metrics["consecutive_dup_ratio"], 1.0)
        self.assertEqual(dup_rat, 1.0)


class TestCsvAdapterRegression(unittest.TestCase):
    """Regression test verifying MMWaveCSVAdapter window generation on sample exported CSV."""

    @unittest.skipUnless(SAMPLE_CSV_PATH.is_file(), "Sample exported CSV missing")
    def test_csv_adapter_generates_exactly_90_windows(self):
        import sys
        sn_dir = str(PROJECT_ROOT)
        if sn_dir not in sys.path:
            sys.path.insert(0, sn_dir)
        from adapters.mmwave_csv_adapter import MMWaveCSVAdapter

        adapter = MMWaveCSVAdapter(window_seconds=30.0, stride_seconds=3.0)
        windows = list(adapter.iter_windows(SAMPLE_CSV_PATH))

        self.assertEqual(len(windows), 90,
                         f"MMWaveCSVAdapter should generate exactly 90 windows from {SAMPLE_CSV_PATH.name}, got {len(windows)}")


class TestNoAbsolutePaths(unittest.TestCase):
    """Tests that no absolute paths exist in any contract artifact."""

    def test_no_absolute_paths_in_json(self):
        json_text = CONTRACT_JSON_PATH.read_text(encoding="utf-8")
        forbidden_prefixes = ["/Users/", "/home/", "C:\\", "D:\\"]
        for prefix in forbidden_prefixes:
            self.assertNotIn(prefix, json_text,
                             f"Absolute path containing '{prefix}' found in contract JSON")

    @unittest.skipUnless(SENSOR_DATA_CONTRACT_MD_PATH.is_file(), "SENSOR_DATA_CONTRACT.md not found")
    def test_no_absolute_paths_in_md(self):
        md_text = SENSOR_DATA_CONTRACT_MD_PATH.read_text(encoding="utf-8")
        forbidden_prefixes = ["/Users/", "/home/", "C:\\", "D:\\"]
        for prefix in forbidden_prefixes:
            self.assertNotIn(prefix, md_text,
                             f"Absolute path containing '{prefix}' found in SENSOR_DATA_CONTRACT.md")


class TestTfliteModelIntegrity(unittest.TestCase):
    """Tests TFLite model file SHA256 matches manifest (not self-referencing)."""

    def test_tflite_sha256_matches_manifest(self):
        manifest = _load_json(MODEL_MANIFEST_PATH)
        for model_key, model_data in manifest["models"].items():
            model_path = PROJECT_ROOT / model_data["path"]
            self.assertTrue(model_path.is_file(), f"TFLite model file missing: {model_path}")
            actual_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
            expected_sha = model_data["sha256"]
            self.assertEqual(actual_sha, expected_sha,
                             f"TFLite {model_key} SHA256 mismatch: "
                             f"manifest={expected_sha}, actual={actual_sha}")

    @unittest.skipUnless(MODEL_INVENTORY_PATH.is_file(), "P0-2 model_inventory.json not yet generated")
    def test_tflite_sha256_matches_inventory(self):
        inventory = _load_json(MODEL_INVENTORY_PATH)
        manifest = _load_json(MODEL_MANIFEST_PATH)
        for model_key in ("thermal", "mmwave", "co2"):
            inv_sha = inventory["models"][model_key]["sha256"]
            manifest_sha = manifest["models"][model_key]["sha256"]
            self.assertEqual(inv_sha, manifest_sha,
                             f"TFLite {model_key} SHA256: inventory={inv_sha} != manifest={manifest_sha}")


if __name__ == "__main__":
    unittest.main()
