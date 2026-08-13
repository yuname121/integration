#!/usr/bin/env python3
"""Focused M-B9 runtime-contract and corruption tests."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validate_mmwave_m_b9 import NEGATIVE_CASES, REQUIRED_FILES, _negative_case_detected


OUT = ROOT / "datasets/mmwave/manifests/M-B9_mock_e2e"


class TestMMWaveMB9(unittest.TestCase):
    def test_required_machine_outputs_and_variants(self):
        for name in REQUIRED_FILES:
            self.assertTrue((OUT / name).is_file(), name)
        for seed in (42, 43, 44):
            self.assertTrue((OUT / "runtime_manifests" / f"seed{seed}_runtime_manifest.json").is_file())

    def test_explicit_identity_and_no_default_finalist(self):
        identity = json.loads((OUT / "runtime_model_identity.json").read_text())
        self.assertTrue(identity["all_sha256_match"])
        self.assertTrue(identity["all_bytes_match"])
        self.assertTrue(identity["all_strict_int8"])
        self.assertTrue(identity["all_flex_select_absent"])
        contract = json.loads((OUT / "runtime_manifest_contract.json").read_text())
        self.assertFalse(contract["shared_default_manifest_used_for_finalist_inference"])
        self.assertNotEqual(contract["shared_default_model_id"], contract["explicit_model_ids"]["42"])

    def test_preprocessing_and_runtime_prediction_identity(self):
        pre = json.loads((OUT / "runtime_preprocessing_identity.json").read_text())
        pred = json.loads((OUT / "runtime_prediction_identity.json").read_text())
        self.assertTrue(pre["all_bpf_exact"])
        self.assertTrue(pre["all_zscore_exact"])
        self.assertTrue(pre["all_model_ready_exact"])
        self.assertTrue(pre["all_int8_exact"])
        self.assertTrue(pred["all_top1_exact"])
        self.assertTrue(pred["all_probability_vectors_exact"])
        self.assertTrue(pred["all_int8_outputs_exact"])

    def test_scenario_truth_is_metadata_only(self):
        records = json.loads((OUT / "scenario_results.json").read_text())["records"]
        model_records = [row for row in records if row["scenario_id"] in {"A_NORMAL", "B_RAPID_OR_ABNORMAL", "C_APNEA", "N_VALID_EXPLICIT_FINALIST"}]
        self.assertTrue(model_records)
        for row in model_records:
            mm = row["mmwave_result"]
            self.assertEqual(mm["metadata"]["score_source"], "MODEL_PREDICTION")
            self.assertFalse(mm["metadata"]["fallback_used"])
            self.assertEqual(mm["state"], mm["metadata"]["model_predicted_class"])

    def test_fault_timeout_stale_matrix(self):
        audit = json.loads((OUT / "fault_timeout_stale_audit.json").read_text())
        actual = {row["scenario_id"] for row in audit["records"]}
        self.assertTrue(set(audit["required_fault_ids"]).issubset(actual))
        stale = next(row for row in audit["records"] if row["scenario_id"] == "H_STALE")
        self.assertIn("mmwave", stale["stale_sensors"])
        timeout = next(row for row in audit["records"] if row["scenario_id"] == "K_TIMEOUT")
        self.assertFalse(timeout["valid"])

    def test_risk_and_json_audits(self):
        risk = json.loads((OUT / "risk_input_audit.json").read_text())
        output = json.loads((OUT / "json_output_audit.json").read_text())
        self.assertTrue(risk["all_equal"])
        self.assertTrue(output["all_valid"])
        self.assertEqual(risk["phase_id"], "M-B9")

    def test_locked_test_is_zero(self):
        locked = json.loads((OUT / "locked_test_access_audit.json").read_text())
        self.assertEqual(locked["performance_access_attempts"], 0)
        self.assertEqual(locked["label_access_attempts"], 0)
        self.assertFalse(locked["locked_test_inputs_loaded"])

    def test_m_b8_wording_is_unambiguous(self):
        experiment = json.loads((OUT / "experiment_contract.json").read_text())
        environment = json.loads((OUT / "run_environment.json").read_text())
        summary = json.loads((OUT / "m_b9_summary.json").read_text())
        for payload in (experiment, environment, summary):
            self.assertNotIn("formal_m_b8_latency_measurement_started", payload)
            self.assertNotIn("m_b8_latency_measurement_started", payload)
            self.assertFalse(payload["formal_m_b8_latency_measurement_rerun_during_m_b9"])
        self.assertTrue(experiment["m_b8_prior_formal_latency_benchmark_completed"])
        self.assertTrue(environment["m_b8_prior_formal_latency_benchmark_completed"])
        self.assertTrue(summary["m_b8_prior_formal_latency_benchmark_completed"])

    def test_negative_corruption_cases(self):
        """All real isolated corruption workspaces must fail closed."""
        for case_id in NEGATIVE_CASES:
            with self.subTest(case_id=case_id):
                self.assertTrue(_negative_case_detected(case_id, ROOT))

    def test_machine_outputs_have_no_absolute_paths(self):
        for path in OUT.rglob("*.json"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/Users/", text, str(path))
            self.assertNotIn("/private/", text, str(path))
            self.assertNotIn("file://", text, str(path))


if __name__ == "__main__":
    unittest.main()
