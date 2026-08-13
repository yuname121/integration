#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_fallback.py
Unit tests for SafeNest V4 Safe Fallback Handler & Sensor Fault Isolation
"""

from __future__ import annotations
import unittest
import time
from typing import Optional

from inference.inference_result import InferenceResult
from risk.fallback import FallbackEngine
from risk.risk_engine import SafeNestRiskEngine


class TestFallbackEngine(unittest.TestCase):
    def setUp(self):
        self.fallback = FallbackEngine(stale_sec=2.0)
        self.risk_engine = SafeNestRiskEngine()
        self.now = time.time()

    def make_res(self, sensor_id: str, score: float, valid: bool = True, ts: Optional[float] = None, error: Optional[str] = None):
        return InferenceResult(
            sensor_id=sensor_id,
            timestamp=ts if ts is not None else self.now,
            score=score,
            state="NORMAL" if valid else "FAULT",
            confidence=1.0 if valid else 0.0,
            valid=valid,
            latency_ms=0.1,
            error=error
        )

    def test_missing_one_sensor_degrades_system(self):
        sensors = {
            "mmwave": self.make_res("mmwave", 0.5),
            "co2": self.make_res("co2", 0.5),
            "pir": self.make_res("pir", 0.0),
            # thermal44 missing
        }
        out = self.risk_engine.evaluate(sensors, now=self.now)
        self.assertEqual(out.system_status, "DEGRADED")
        self.assertTrue(out.fallback_used)
        self.assertIn("THERMAL44_MISSING", out.reasons)
        self.assertGreater(out.risk_score, 0.0)

    def test_stale_timestamp_triggers_fallback(self):
        stale_time = self.now - 10.0  # 10 seconds old
        sensors = {
            "mmwave": self.make_res("mmwave", 0.0, ts=stale_time),
            "co2": self.make_res("co2", 0.0),
            "pir": self.make_res("pir", 0.0),
            "thermal44": self.make_res("thermal44", 0.0)
        }
        out = self.risk_engine.evaluate(sensors, now=self.now)
        self.assertEqual(out.system_status, "DEGRADED")
        self.assertIn("MMWAVE_STALE_TIMESTAMP", out.reasons)

    def test_all_sensors_fault(self):
        sensors = {
            "mmwave": self.make_res("mmwave", 0.0, valid=False, error="SENSOR_DISCONNECTED"),
            "co2": self.make_res("co2", 0.0, valid=False, error="READ_TIMEOUT"),
            "pir": self.make_res("pir", 0.0, valid=False, error="GPIO_FAULT"),
            "thermal44": self.make_res("thermal44", 0.0, valid=False, error="I2C_FAULT")
        }
        out = self.risk_engine.evaluate(sensors, now=self.now)
        self.assertEqual(out.level, "FAULT")
        self.assertEqual(out.system_status, "FAULT")
        self.assertTrue(out.fallback_used)


if __name__ == "__main__":
    unittest.main()
