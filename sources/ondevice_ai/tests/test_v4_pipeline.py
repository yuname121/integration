#!/usr/bin/env python3
"""SafeNest v4 fusion and Thermal-44 real-time runner tests."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.infer_pi_thermal import NpyThermal44Source, ThermalRealtimeRunner, VirtualThermal44Source
from inference.thermal_interpreter import ThermalPrediction
from risk.risk_engine import RiskEngineV4
from risk.risk_rules import calculate_v4_risk, classify_v4_risk
from integrated_node.safenest_risk_engine import SafeNestRiskEngine


class TestV4RiskEngine(unittest.TestCase):
    def test_exact_formula(self):
        score = calculate_v4_risk({"S1": 1, "S2": 0.5, "S3": 0.25, "S4": 1})
        self.assertAlmostEqual(score, 71.25)

    def test_exact_boundaries(self):
        self.assertEqual(classify_v4_risk(29.999), "NORMAL")
        self.assertEqual(classify_v4_risk(30.0), "CAUTION")
        self.assertEqual(classify_v4_risk(59.999), "CAUTION")
        self.assertEqual(classify_v4_risk(60.0), "DANGER")

    def test_nan_uses_last_good_within_budget(self):
        engine = RiskEngineV4()
        engine.evaluate({"S1": 0.4, "S2": 0, "S3": 0, "S4": 0})
        result = engine.evaluate({"S1": np.nan, "S2": 0, "S3": 0, "S4": 0})
        self.assertEqual(result.sensor_scores["S1"], 0.4)
        self.assertEqual(result.sensor_status["S1"], "FALLBACK_LAST_GOOD")
        self.assertTrue(result.fallback_used)
        self.assertTrue(result.fallback_budget_met)

    def test_missing_channel_uses_raw_rule(self):
        result = RiskEngineV4().evaluate(
            {"S1": None, "S2": None, "S3": None, "S4": None},
            {"apnea": 1, "co2_ppm": 2500, "pir_motion": 0,
             "presence": 1, "thermal_class": 2, "thermal_confidence": 0.9},
        )
        self.assertEqual(result.sensor_scores, {"S1": 1.0, "S2": 1.0, "S3": 1.0, "S4": 1.0})
        self.assertEqual(result.level, "DANGER")

    def test_thermal_alone_obeys_weighted_formula(self):
        result = RiskEngineV4().evaluate({"S1": 0, "S2": 0, "S3": 0, "S4": 1})
        self.assertEqual(result.risk_score, 15.0)
        self.assertEqual(result.level, "NORMAL")

    def test_packet_missing_channel_is_degraded_not_ok(self):
        result = RiskEngineV4().evaluate_packet(
            {
                "timestamp_s": 1.0,
                "mmwave_mr60": {"breath_rpm": 16.0, "apnea": 0, "presence": 1},
                "pir": {"motion": 1},
            },
            s4=0.0,
        )
        self.assertEqual(result.sensor_status["S2"], "FALLBACK_RULE")
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.system_status, "DEGRADED")

    def test_packet_valid_false_ignores_stale_numeric_value(self):
        result = RiskEngineV4().evaluate_packet(
            {
                "timestamp_s": 1.0,
                "mmwave_mr60": {"breath_rpm": 16.0, "apnea": 0, "presence": 1},
                "co2_scd40": {"co2_ppm": 9000.0, "valid": False},
                "pir": {"motion": 1},
            },
            s4=0.0,
        )
        self.assertEqual(result.sensor_scores["S2"], 0.0)
        self.assertEqual(result.sensor_status["S2"], "FALLBACK_RULE")

    def test_packet_non_numeric_values_and_nan_s4_fall_back(self):
        result = RiskEngineV4().evaluate_packet(
            {
                "timestamp_s": "invalid",
                "mmwave_mr60": {"breath_rpm": "16", "apnea": 0},
                "co2_scd40": {"co2_ppm": "500"},
                "pir": {"motion": "1"},
            },
            s4=np.nan,
        )
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.system_status, "DEGRADED")
        self.assertTrue(all(value == "FALLBACK_RULE" for value in result.sensor_status.values()))

    def test_last_good_expires_after_ttl(self):
        engine = RiskEngineV4(last_good_ttl_s={
            "S1": 1.0, "S2": 1.0, "S3": 1.0, "S4": 1.0,
        })
        engine.evaluate({"S1": 1, "S2": 1, "S3": 1, "S4": 1}, timestamp_s=1.0)
        result = engine.evaluate(
            {"S1": None, "S2": None, "S3": None, "S4": None},
            timestamp_s=3.0,
        )
        self.assertEqual(result.sensor_scores, {"S1": 0.0, "S2": 0.0, "S3": 0.0, "S4": 0.0})
        self.assertTrue(all(value == "FALLBACK_RULE" for value in result.sensor_status.values()))

    def test_confirmed_fall_applies_documented_override(self):
        result = RiskEngineV4().evaluate_packet(
            {
                "timestamp_s": 1.0,
                "mmwave_mr60": {"breath_rpm": 16.0, "apnea": 0, "presence": 1},
                "co2_scd40": {"co2_ppm": 500.0},
                "pir": {"motion": 1},
            },
            s4=1.0,
        )
        self.assertEqual(result.weighted_risk_score, 15.0)
        self.assertEqual(result.risk_score, 100.0)
        self.assertEqual(result.level, "DANGER")
        self.assertTrue(result.emergency_override)


class TestV4IntegratedContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SafeNestRiskEngine()

    def test_top_level_is_official_v4_and_legacy_is_namespaced(self):
        packet = {
            "timestamp_s": 1.0,
            "thermal_80x62": np.zeros((62, 80), dtype=np.float32),
            "co2_scd40": {"co2_ppm": 500.0, "humidity": 45.0},
            "mmwave_mr60": {
                "breath_rpm": 16.0, "apnea": 0, "heart_bpm": 72.0,
                "presence": 1, "resp_phase": 0.01,
            },
            "pir": {"motion": 1},
        }
        result = self.engine.evaluate_risk(packet)
        self.assertEqual(result["risk_score"], result["v4_fusion"]["risk_score"])
        self.assertEqual(result["status_str"], result["v4_fusion"]["level"])
        self.assertIn("legacy_fusion", result)

    def test_malformed_and_invalid_channels_do_not_crash(self):
        packet = {
            "timestamp_s": 1.0,
            "thermal_80x62": np.zeros((62, 80), dtype=np.float32),
            "co2_scd40": {"co2_ppm": "650", "valid": False},
            "mmwave_mr60": {"breath_rpm": "16", "apnea": 0, "valid": False},
            "pir": {"motion": 1},
        }
        result = self.engine.evaluate_risk(packet)
        self.assertEqual(result["system_status"], "DEGRADED")
        self.assertTrue(result["v4_fusion"]["fallback_used"])

    def test_empty_packet_keeps_stable_v4_schema(self):
        result = self.engine.evaluate_risk({})
        self.assertEqual(result["status_str"], "FAULT")
        self.assertIn("v4_fusion", result)
        self.assertIn("legacy_fusion", result)


class TestV4ThermalRunner(unittest.TestCase):
    def test_single_channel_npy_is_one_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.npy"
            np.save(path, np.zeros((62, 80, 1), dtype=np.float32))
            frames = list(NpyThermal44Source(path).frames())
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].shape, (62, 80, 1))

    def test_npy_batch_preserves_frame_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.npy"
            np.save(path, np.zeros((3, 62, 80, 1), dtype=np.float32))
            frames = list(NpyThermal44Source(path).frames())
        self.assertEqual(len(frames), 3)

    def test_fall_dispatches_s4_alarm(self):
        prediction = ThermalPrediction(
            class_index=2, class_name="HUMAN_FALL", confidence=0.95,
            probabilities=[0.01, 0.04, 0.95], latency_ms=0.05,
            model_id="thermal_fall_int8", model_version="0.1.0",
        )
        interpreter = Mock()
        interpreter.predict.return_value = prediction
        alarms = []
        runner = ThermalRealtimeRunner(
            VirtualThermal44Source("fall", realtime=False),
            interpreter=interpreter,
            alarm_sink=alarms.append,
        )
        event = runner.process_frame(np.zeros((62, 80), dtype=np.float32), 1)
        self.assertEqual(event.s4, 1.0)
        self.assertTrue(event.latency_target_met)
        self.assertEqual(len(alarms), 1)
        self.assertEqual(alarms[0].fusion["sensor_scores"]["S4"], 1.0)
        self.assertEqual(event.fusion["weighted_risk_score"], 15.0)
        self.assertEqual(event.fusion["risk_score"], 100.0)
        self.assertEqual(event.fusion["level"], "DANGER")
        self.assertTrue(event.fusion["emergency_override"])


if __name__ == "__main__":
    unittest.main()
