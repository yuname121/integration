#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_three_model_integration.py
P0-7 3대 모델(Thermal, CO2, mmWave) 통합 엔드투엔드 테스트 수트
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.model_registry import ModelRegistry
from integrated_node.safenest_risk_engine import SafeNestRiskEngine


class TestThreeModelIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = ModelRegistry(project_root=PROJECT_ROOT)
        cls.engine = SafeNestRiskEngine(manifest_path="models/model_manifest.json")

    def test_model_registry_health(self):
        health = self.registry.health()
        self.assertTrue(health["thermal"]["loaded"])
        self.assertTrue(health["co2"]["loaded"])
        self.assertTrue(health["mmwave"]["loaded"])
        self.assertTrue(health["thermal"]["sha256_matches"])
        self.assertTrue(health["co2"]["sha256_matches"])
        self.assertTrue(health["mmwave"]["sha256_matches"])

    def test_co2_interpreter_predict(self):
        res = self.registry.co2.predict(0.01, 45.0, 650.0)
        self.assertIn(res.class_index, (0, 1))
        self.assertEqual(len(res.probabilities), 2)
        self.assertAlmostEqual(sum(res.probabilities), 1.0, places=4)

    def test_scenario_1_all_normal(self):
        packet = {
            "thermal_80x62": np.zeros((62, 80), dtype=np.float32),
            "co2_scd40": {"co2_ppm": 500, "humidity": 45.0},
            "mmwave_mr60": {"breath_rpm": 16.0, "apnea": 0, "heart_bpm": 72.0, "resp_phase": 0.01},
            "pir": {"motion": 1}
        }
        res = self.engine.evaluate_risk(packet)
        self.assertEqual(res["status_str"], "NORMAL")
        self.assertFalse(res["is_emergency"])

    def test_scenario_2_thermal_fall(self):
        grid = np.zeros((62, 80), dtype=np.float32)
        grid[45:, :] = 100.0
        packet = {
            "thermal_80x62": grid,
            "co2_scd40": {"co2_ppm": 500, "humidity": 45.0},
            "mmwave_mr60": {"breath_rpm": 16.0, "apnea": 0, "heart_bpm": 72.0},
            "pir": {"motion": 0}
        }
        res = self.engine.evaluate_risk(packet)
        self.assertEqual(res["status_str"], "DANGER")
        self.assertTrue(res["is_emergency"])

    def test_scenario_3_mmwave_hardware_apnea(self):
        packet = {
            "thermal_80x62": np.zeros((62, 80), dtype=np.float32),
            "co2_scd40": {"co2_ppm": 500, "humidity": 45.0},
            "mmwave_mr60": {"breath_rpm": 0.0, "apnea": 1, "heart_bpm": 35.0},
            "pir": {"motion": 0}
        }
        res = self.engine.evaluate_risk(packet)
        self.assertEqual(res["status_str"], "DANGER")
        self.assertTrue(res["is_emergency"])

    def test_scenario_4_co2_rise(self):
        packet = {
            "thermal_80x62": np.zeros((62, 80), dtype=np.float32),
            "co2_scd40": {"co2_ppm": 2800, "humidity": 65.0},
            "mmwave_mr60": {"breath_rpm": 16.0, "apnea": 0, "heart_bpm": 75.0},
            "pir": {"motion": 1}
        }
        res = self.engine.evaluate_risk(packet)
        self.assertIn("HIGH_CO2_DANGER", res["reasons"])

    def test_scenario_5_real_thermal_invoke_exception_fallback(self):
        original_runner = self.engine.thermal_runner
        try:
            mock_runner = MagicMock()
            mock_runner.predict.side_effect = RuntimeError("Simulated thermal invoke failure")
            self.engine.thermal_runner = mock_runner

            packet = {
                "thermal_80x62": np.zeros((62, 80), dtype=np.float32),
                "co2_scd40": {"co2_ppm": 500, "humidity": 45.0},
                "mmwave_mr60": {"breath_rpm": 16.0, "apnea": 0, "heart_bpm": 72.0},
                "pir": {"motion": 1}
            }
            res = self.engine.evaluate_risk(packet)
            self.assertEqual(res["sensor_quality"]["thermal"], 0.0)
            self.assertIn("THERMAL_MODEL_INVOKE_ERROR", res["reasons"])
            self.assertEqual(res["system_status"], "DEGRADED")
        finally:
            self.engine.thermal_runner = original_runner

    def test_scenario_5_b_co2_invoke_exception_fallback(self):
        original_runner = self.engine.co2_runner
        try:
            mock_runner = MagicMock()
            mock_runner.predict.side_effect = RuntimeError("Simulated CO2 invoke failure")
            self.engine.co2_runner = mock_runner

            packet = {
                "thermal_80x62": np.zeros((62, 80), dtype=np.float32),
                "co2_scd40": {"co2_ppm": 500, "humidity": 45.0},
                "mmwave_mr60": {"breath_rpm": 16.0, "apnea": 0, "heart_bpm": 72.0},
                "pir": {"motion": 1}
            }
            res = self.engine.evaluate_risk(packet)
            self.assertIn("CO2_MODEL_INVOKE_ERROR", res["reasons"])
            self.assertEqual(res["system_status"], "DEGRADED")
            self.assertTrue(res["model_meta"]["co2"]["fallback_used"])
        finally:
            self.engine.co2_runner = original_runner

    def test_real_mmwave_tflite_runs_after_window_ready(self):
        """300샘플 window 준비 후 반입된 mmWave TFLite가 fallback 없이 실행되는지 검증"""
        start_ts = 100.0
        res = None
        for i in range(300):
            packet = {
                "timestamp_s": start_ts + i * 0.1,
                "thermal_80x62": np.zeros((62, 80), dtype=np.float32),
                "co2_scd40": {"co2_ppm": 500, "humidity": 45.0},
                "mmwave_mr60": {"breath_rpm": 16.0, "apnea": 0, "heart_bpm": 72.0, "resp_phase": 0.01, "presence": 1},
                "pir": {"motion": 1}
            }
            res = self.engine.evaluate_risk(packet)

        self.assertIsNotNone(res)
        self.assertNotIn("TFLITE_MODEL_FILE_MISSING", res["reasons"])
        self.assertEqual(res["model_meta"]["mmwave"]["source"], "mmwave_resp_int8")
        self.assertFalse(res["model_meta"]["mmwave"]["fallback_used"])
        self.assertEqual(res["model_meta"]["mmwave"]["ai_status"], "OK")

    def test_scenario_6_real_world_empty_packet_is_fault(self):
        packet = {
            "thermal_80x62": None,
            "co2_scd40": {},
            "mmwave_mr60": {},
            "pir": {}
        }
        res = self.engine.evaluate_risk(packet)
        self.assertEqual(res["status_str"], "FAULT")
        self.assertEqual(res["system_status"], "FAULT")
        self.assertIn("ALL_SENSORS_MISSING", res["reasons"])

    def test_scenario_7_missing_hr_is_degraded(self):
        packet = {
            "thermal_80x62": np.zeros((62, 80), dtype=np.float32),
            "co2_scd40": {"co2_ppm": 500, "humidity": 45.0},
            "mmwave_mr60": {"breath_rpm": 16.0, "apnea": 0},
            "pir": {"motion": 1}
        }
        res = self.engine.evaluate_risk(packet)
        self.assertIn("HR_SENSOR_MISSING", res["reasons"])
        self.assertEqual(res["system_status"], "DEGRADED")

    def test_scenario_8_presence_zero(self):
        packet = {
            "thermal_80x62": np.zeros((62, 80), dtype=np.float32),
            "co2_scd40": {"co2_ppm": 500, "humidity": 45.0},
            "mmwave_mr60": {"breath_rpm": 16.0, "apnea": 0, "heart_bpm": 72.0, "resp_phase": 0.01, "presence": 0},
            "pir": {"motion": 1}
        }
        res = self.engine.evaluate_risk(packet)
        self.assertIn("MMWAVE_PRESENCE_NOT_DETECTED", res["reasons"])

    def test_scenario_9_all_sensors_fault(self):
        packet = {}
        res = self.engine.evaluate_risk(packet)
        self.assertEqual(res["status_str"], "FAULT")
        self.assertEqual(res["system_status"], "FAULT")
        self.assertIn("ALL_SENSORS_MISSING", res["reasons"])


if __name__ == "__main__":
    unittest.main()
