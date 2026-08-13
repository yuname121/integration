#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_fault_injection.py
P0-8 Fault Injection 방어 테스트 (NaN, missing sensors, empty packet {}, confirmed apnea bypass)
"""

import os
import sys
from pathlib import Path
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk.risk_rules import RiskRulesEvaluator
from integrated_node.safenest_risk_engine import SafeNestRiskEngine


class TestFaultInjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = RiskRulesEvaluator()
        cls.engine = SafeNestRiskEngine()

    def test_empty_packet_returns_fault_not_normal(self):
        """빈 패킷 {} 입력 시 R=0.0 NORMAL 은폐 방지 ➔ status=FAULT, system_status=FAULT, reasons=["ALL_SENSORS_MISSING"] 검증"""
        res = self.engine.evaluate_risk({})
        self.assertEqual(res["status_str"], "FAULT")
        self.assertEqual(res["system_status"], "FAULT")
        self.assertIn("ALL_SENSORS_MISSING", res["reasons"])
        self.assertNotEqual(res["status_str"], "NORMAL")

    def test_nan_respiration_is_not_normal(self):
        res_resp = self.evaluator.evaluate_respiration(breath_rpm=np.nan, apnea=0)
        res_env = self.evaluator.evaluate_environment(co2_ppm=500.0)
        res_hr = self.evaluator.evaluate_vital_hr(72.0)
        res_post = self.evaluator.evaluate_posture(thermal_fall_class=1)
        res_mot = self.evaluator.evaluate_motion(1)

        sys_eval = self.evaluator.evaluate_system(res_resp, res_env, res_hr, res_post, res_mot)
        self.assertIn(sys_eval.system_status, {"DEGRADED", "FAULT"})
        self.assertIn("RESP_SENSOR_FAULT", sys_eval.reasons)

    def test_missing_co2_is_degraded_not_normal(self):
        res_resp = self.evaluator.evaluate_respiration(breath_rpm=16.0, apnea=0)
        res_env = self.evaluator.evaluate_environment(co2_ppm=None, valid=False)
        res_hr = self.evaluator.evaluate_vital_hr(72.0)
        res_post = self.evaluator.evaluate_posture(thermal_fall_class=1)
        res_mot = self.evaluator.evaluate_motion(1)

        sys_eval = self.evaluator.evaluate_system(res_resp, res_env, res_hr, res_post, res_mot)
        self.assertEqual(sys_eval.sensor_status["co2"], "DEGRADED")
        self.assertIn("CO2_SENSOR_FAULT", sys_eval.reasons)

    def test_confirmed_apnea_bypasses_smoothing(self):
        res_resp = self.evaluator.evaluate_respiration(breath_rpm=0.0, apnea=1, dt_s=2.0)
        res_env = self.evaluator.evaluate_environment(co2_ppm=500.0)
        res_hr = self.evaluator.evaluate_vital_hr(72.0)
        res_post = self.evaluator.evaluate_posture(thermal_fall_class=1)
        res_mot = self.evaluator.evaluate_motion(1)

        sys_eval = self.evaluator.evaluate_system(res_resp, res_env, res_hr, res_post, res_mot)
        self.assertEqual(sys_eval.level, "DANGER")
        self.assertTrue(sys_eval.is_emergency)
        self.assertEqual(sys_eval.risk_score, 100.0)


if __name__ == "__main__":
    unittest.main()
