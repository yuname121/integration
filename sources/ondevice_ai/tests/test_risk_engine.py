#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_risk_engine.py
Unit tests for SafeNest V4 Risk Fusion Engine & Boundary Conditions
"""

import unittest
import time

from inference.inference_result import InferenceResult
from risk.risk_engine import SafeNestRiskEngine


class TestRiskEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SafeNestRiskEngine()
        self.now = time.time()

    def make_res(self, sensor_id: str, score: float, state: str = "NORMAL", valid: bool = True):
        return InferenceResult(
            sensor_id=sensor_id,
            timestamp=self.now,
            score=score,
            state=state,
            confidence=1.0,
            valid=valid,
            latency_ms=0.1
        )

    def test_all_zero_normal(self):
        sensors = {
            "mmwave": self.make_res("mmwave", 0.0),
            "co2": self.make_res("co2", 0.0),
            "pir": self.make_res("pir", 0.0),
            "thermal44": self.make_res("thermal44", 0.0)
        }
        out = self.engine.evaluate(sensors, now=self.now)
        self.assertEqual(out.risk_score, 0.0)
        self.assertEqual(out.level, "NORMAL")
        self.assertEqual(out.system_status, "OK")

    def test_exact_boundary_conditions(self):
        # Boundary 29.999 -> NORMAL
        # S1=0.857114 -> R = 100 * (0.35 * 0.857114) = 29.99899
        s_299 = {
            "mmwave": self.make_res("mmwave", 0.857114, state="RAPID_OR_ABNORMAL"),
            "co2": self.make_res("co2", 0.0),
            "pir": self.make_res("pir", 0.0),
            "thermal44": self.make_res("thermal44", 0.0)
        }
        out_299 = self.engine.evaluate(s_299, now=self.now)
        self.assertLess(out_299.risk_score, 30.0)
        self.assertEqual(out_299.level, "NORMAL")

        # Boundary 30.0 -> CAUTION
        s_300 = {
            "mmwave": self.make_res("mmwave", 30.0 / 35.0, state="RAPID_OR_ABNORMAL"),
            "co2": self.make_res("co2", 0.0),
            "pir": self.make_res("pir", 0.0),
            "thermal44": self.make_res("thermal44", 0.0)
        }
        out_300 = self.engine.evaluate(s_300, now=self.now)
        self.assertAlmostEqual(out_300.risk_score, 30.0, places=4)
        self.assertEqual(out_300.level, "CAUTION")

        # Boundary 59.999 -> CAUTION
        # S1=0.9, S2=(59.999 - 31.5)/35 = 0.81425714
        s_599 = {
            "mmwave": self.make_res("mmwave", 0.9, state="RAPID_OR_ABNORMAL"),
            "co2": self.make_res("co2", (59.999 - 31.5) / 35.0, state="ELEVATED"),
            "pir": self.make_res("pir", 0.0),
            "thermal44": self.make_res("thermal44", 0.0)
        }
        out_599 = self.engine.evaluate(s_599, now=self.now)
        self.assertLess(out_599.risk_score, 60.0)
        self.assertEqual(out_599.level, "CAUTION")

        # Boundary 60.0 -> DANGER
        # S1=0.9, S2=(60.0 - 31.5)/35 = 0.814285714
        s_600 = {
            "mmwave": self.make_res("mmwave", 0.9, state="RAPID_OR_ABNORMAL"),
            "co2": self.make_res("co2", (60.0 - 31.5) / 35.0, state="ELEVATED"),
            "pir": self.make_res("pir", 0.0),
            "thermal44": self.make_res("thermal44", 0.0)
        }
        out_600 = self.engine.evaluate(s_600, now=self.now)
        self.assertAlmostEqual(out_600.risk_score, 60.0, places=4)
        self.assertEqual(out_600.level, "DANGER")

    def test_emergency_overrides(self):
        # Thermal fall emergency
        s_fall = {
            "mmwave": self.make_res("mmwave", 0.0),
            "co2": self.make_res("co2", 0.0),
            "pir": self.make_res("pir", 0.0),
            "thermal44": self.make_res("thermal44", 1.0, state="HUMAN_FALL")
        }
        out_fall = self.engine.evaluate(s_fall, now=self.now)
        self.assertEqual(out_fall.risk_score, 100.0)
        self.assertEqual(out_fall.level, "DANGER")
        self.assertTrue(out_fall.is_emergency)
        self.assertIn("EMERGENCY_HUMAN_FALL", out_fall.reasons)

        # mmWave apnea emergency
        s_apnea = {
            "mmwave": self.make_res("mmwave", 1.0, state="APNEA"),
            "co2": self.make_res("co2", 0.0),
            "pir": self.make_res("pir", 0.0),
            "thermal44": self.make_res("thermal44", 0.0)
        }
        out_apnea = self.engine.evaluate(s_apnea, now=self.now)
        self.assertEqual(out_apnea.risk_score, 100.0)
        self.assertEqual(out_apnea.level, "DANGER")
        self.assertTrue(out_apnea.is_emergency)
        self.assertIn("EMERGENCY_HARDWARE_APNEA", out_apnea.reasons)


if __name__ == "__main__":
    unittest.main()
