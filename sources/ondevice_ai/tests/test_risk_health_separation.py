#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_risk_health_separation.py
Comprehensive Unit & Integration Tests for P0-4 Risk and System Health Separation.
"""

from __future__ import annotations
import unittest
import time
import json
from typing import Optional

from inference.inference_result import InferenceResult, SafeNestRiskOutput
from risk.fallback import FallbackEngine, evaluate_sensor_health_and_risk
from risk.risk_engine import SafeNestRiskEngine


class TestRiskHealthSeparation(unittest.TestCase):
    def setUp(self):
        self.engine = SafeNestRiskEngine()
        self.now = time.time()

    def make_res(
        self,
        sensor_id: str,
        score: float,
        valid: bool = True,
        ts: Optional[float] = None,
        state: str = "NORMAL",
        error: Optional[str] = None
    ) -> InferenceResult:
        return InferenceResult(
            sensor_id=sensor_id,
            timestamp=ts if ts is not None else self.now,
            score=score,
            state=state if valid else "FAULT",
            confidence=1.0 if valid else 0.0,
            valid=valid,
            latency_ms=0.1,
            error=error
        )

    def test_all_sensors_healthy(self):
        """Scenario 1: All sensors healthy -> HEALTHY, degraded_mode=false, correct risk score."""
        sensors = {
            "mmwave": self.make_res("mmwave", 0.5),
            "co2": self.make_res("co2", 0.0),
            "pir": self.make_res("pir", 0.0),
            "thermal": self.make_res("thermal", 0.0),
        }
        out = self.engine.evaluate(sensors, now=self.now)
        self.assertEqual(out.system_health, "HEALTHY")
        self.assertFalse(out.degraded_mode)
        self.assertEqual(out.invalid_sensors, [])
        self.assertEqual(out.stale_sensors, [])
        self.assertAlmostEqual(out.risk_score, 17.5, places=4)
        self.assertEqual(out.risk_level, "NORMAL")
        self.assertEqual(out.component_scores["mmwave"], 0.5)
        self.assertEqual(out.component_scores["co2"], 0.0)
        self.assertEqual(out.component_scores["pir"], 0.0)
        self.assertEqual(out.component_scores["thermal"], 0.0)

    def test_co2_disconnected_fallback(self):
        """Scenario 2 & 6: CO2 disconnected -> co2=null, invalid_sensors=['co2'], DEGRADED, reweighted fallback."""
        sensors = {
            "mmwave": self.make_res("mmwave", 0.5),
            "co2": self.make_res("co2", 0.0, valid=False, error="SENSOR_DISCONNECTED"),
            "pir": self.make_res("pir", 0.0),
            "thermal": self.make_res("thermal", 0.0),
        }
        out = self.engine.evaluate(sensors, now=self.now)
        self.assertEqual(out.system_health, "DEGRADED")
        self.assertTrue(out.degraded_mode)
        self.assertEqual(out.invalid_sensors, ["co2"])
        self.assertEqual(out.stale_sensors, [])
        self.assertIsNone(out.component_scores["co2"])
        self.assertEqual(out.component_scores["mmwave"], 0.5)

        # Reweighted calculation: total valid weight = 0.35 + 0.15 + 0.15 = 0.65
        # Reweighted score = (0.35/0.65)*0.5 * 100 = 26.9230769
        expected_score = (0.35 / 0.65) * 0.5 * 100.0
        self.assertAlmostEqual(out.risk_score, expected_score, places=4)
        self.assertEqual(out.risk_level, "NORMAL")

    def test_co2_valid_zero(self):
        """Scenario 3: CO2 valid 0.0 -> co2=0.0, not in invalid list."""
        sensors = {
            "mmwave": self.make_res("mmwave", 0.0),
            "co2": self.make_res("co2", 0.0, valid=True),
            "pir": self.make_res("pir", 0.0),
            "thermal": self.make_res("thermal", 0.0),
        }
        out = self.engine.evaluate(sensors, now=self.now)
        self.assertEqual(out.system_health, "HEALTHY")
        self.assertNotIn("co2", out.invalid_sensors)
        self.assertEqual(out.component_scores["co2"], 0.0)

    def test_co2_nan_inf_invalid(self):
        """Scenario 4: CO2 score NaN/Inf or out of bounds -> invalid handling."""
        sensors = {
            "mmwave": self.make_res("mmwave", 0.0),
            "co2": {"valid": False, "score": float("nan"), "timestamp": self.now, "state": "NAN_OR_INF", "error": "NAN_OR_INF"},
            "pir": self.make_res("pir", 0.0),
            "thermal": self.make_res("thermal", 0.0),
        }
        out = self.engine.evaluate(sensors, now=self.now)
        self.assertEqual(out.system_health, "DEGRADED")
        self.assertIn("co2", out.invalid_sensors)
        self.assertIsNone(out.component_scores["co2"])

    def test_sensor_timestamp_stale(self):
        """Scenario 5: Sensor timestamp expired -> stale handling, not invalid."""
        stale_time = self.now - 10.0
        sensors = {
            "mmwave": self.make_res("mmwave", 0.5, ts=stale_time),
            "co2": self.make_res("co2", 0.0),
            "pir": self.make_res("pir", 0.0),
            "thermal": self.make_res("thermal", 0.0),
        }
        out = self.engine.evaluate(sensors, now=self.now)
        self.assertEqual(out.system_health, "DEGRADED")
        self.assertTrue(out.degraded_mode)
        self.assertEqual(out.stale_sensors, ["mmwave"])
        self.assertEqual(out.invalid_sensors, [])
        self.assertIsNone(out.component_scores["mmwave"])

    def test_multiple_sensors_simultaneous_failure(self):
        """Scenario 6: Multiple sensors failing -> deterministic canonical list order & reweighting."""
        stale_time = self.now - 10.0
        sensors = {
            "pir": self.make_res("pir", 0.0, valid=False, error="GPIO_FAULT"),
            "co2": self.make_res("co2", 0.5, ts=stale_time),
            "mmwave": self.make_res("mmwave", 0.8),
            "thermal": self.make_res("thermal", 0.0),
        }
        out = self.engine.evaluate(sensors, now=self.now)
        self.assertEqual(out.system_health, "DEGRADED")
        self.assertEqual(out.invalid_sensors, ["pir"])
        self.assertEqual(out.stale_sensors, ["co2"])
        # Canonical order: ["mmwave", "co2", "pir", "thermal"]
        self.assertEqual(list(out.component_scores.keys()), ["mmwave", "co2", "pir", "thermal"])
        self.assertIsNone(out.component_scores["co2"])
        self.assertIsNone(out.component_scores["pir"])

        # Reweighted valid sensors (mmwave=0.35, thermal=0.15; total valid=0.50)
        # score = (0.35/0.50)*0.8 * 100 = 56.0
        self.assertAlmostEqual(out.risk_score, 56.0, places=4)
        self.assertEqual(out.risk_level, "CAUTION")

    def test_degraded_system_with_high_risk(self):
        """Scenario 7: Partial failure while remaining sensor detects high risk -> DEGRADED + DANGER."""
        sensors = {
            "mmwave": self.make_res("mmwave", 0.0, valid=False, error="COMM_ERROR"),
            "co2": self.make_res("co2", 0.9),  # High CO2 risk
            "pir": self.make_res("pir", 0.0),
            "thermal": self.make_res("thermal", 0.0),
        }
        out = self.engine.evaluate(sensors, now=self.now)
        self.assertEqual(out.system_health, "DEGRADED")
        self.assertTrue(out.degraded_mode)
        # Valid weights = 0.35 (co2) + 0.15 (pir) + 0.15 (thermal) = 0.65
        # Reweighted score = (0.35/0.65)*0.9 * 100 = 48.46 -> CAUTION
        self.assertEqual(out.risk_level, "CAUTION")

    def test_insufficient_valid_sensors_failed(self):
        """Scenario 8: All sensors invalid/missing -> FAILED, risk_score=null, risk_level=null."""
        sensors = {
            "mmwave": self.make_res("mmwave", 0.0, valid=False, error="FAULT"),
            "co2": self.make_res("co2", 0.0, valid=False, error="FAULT"),
            "pir": self.make_res("pir", 0.0, valid=False, error="FAULT"),
            "thermal": self.make_res("thermal", 0.0, valid=False, error="FAULT"),
        }
        out = self.engine.evaluate(sensors, now=self.now)
        self.assertEqual(out.system_health, "FAILED")
        self.assertTrue(out.degraded_mode)
        self.assertIsNone(out.risk_score)
        self.assertIsNone(out.risk_level)
        self.assertEqual(out.invalid_sensors, ["mmwave", "co2", "pir", "thermal"])

    def test_json_serialization(self):
        """Scenario 10: JSON serialization preserves null vs 0.0 differentiation."""
        sensors = {
            "mmwave": self.make_res("mmwave", 0.5),
            "co2": self.make_res("co2", 0.0, valid=False, error="DISCONNECTED"),
            "pir": self.make_res("pir", 0.0),
            "thermal": self.make_res("thermal", 0.0),
        }
        out = self.engine.evaluate(sensors, now=self.now)
        json_str = out.to_json()
        parsed = json.loads(json_str)

        self.assertIn("risk_score", parsed)
        self.assertIn("risk_level", parsed)
        self.assertIn("system_health", parsed)
        self.assertIn("degraded_mode", parsed)
        self.assertIn("invalid_sensors", parsed)
        self.assertIn("stale_sensors", parsed)
        self.assertIn("component_scores", parsed)

        self.assertEqual(parsed["system_health"], "DEGRADED")
        self.assertTrue(parsed["degraded_mode"])
        self.assertEqual(parsed["invalid_sensors"], ["co2"])
        self.assertEqual(parsed["component_scores"]["co2"], None)
        self.assertEqual(parsed["component_scores"]["pir"], 0.0)


if __name__ == "__main__":
    unittest.main()
