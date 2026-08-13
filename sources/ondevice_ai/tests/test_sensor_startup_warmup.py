#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_sensor_startup_warmup.py
Comprehensive unit & integration tests for SafeNest V4 P0-5 Sensor Startup and Warming-Up Safety
"""

import unittest
import time
import math
import numpy as np
from unittest.mock import Mock

from sensors.base_sensor import SensorState, HardwareBackendUnavailable
from sensors.mmwave.mmwave_adapter import MMWaveSensorAdapter
from sensors.mmwave.mock_sensor import MockMMWaveSensor
from sensors.co2.co2_adapter import CO2SensorAdapter
from sensors.co2.mock_sensor import MockCO2Sensor
from sensors.pir.pir_adapter import PIRSensorAdapter
from sensors.pir.mock_sensor import MockPIRSensor
from sensors.thermal44.thermal44_driver import Thermal44Sensor
from sensors.thermal44.mock_sensor import MockThermalSensor

from integrated_node.run_node import SafeNestIntegratedNode


class TestMMWaveWarmingUp(unittest.TestCase):
    def test_mmwave_buffer_thresholds_and_interpreter_suppression(self):
        sensor = MMWaveSensorAdapter()
        sensor.connected = True
        sensor.interpreter.predict = Mock(side_effect=AssertionError("predict must not be called during warmup"))

        # 0 samples -> WARMING_UP
        res = sensor.read()
        self.assertFalse(res.valid)
        self.assertEqual(res.state, "WARMING_UP")
        self.assertEqual(res.error, "INSUFFICIENT_HISTORY")
        self.assertEqual(res.metadata["buffer_len"], 0)

        # 1 sample -> WARMING_UP
        sensor.push_sample(1.2, 100.0)
        res = sensor.read()
        self.assertFalse(res.valid)
        self.assertEqual(res.state, "WARMING_UP")
        self.assertEqual(res.metadata["buffer_len"], 1)

        # 299 samples -> WARMING_UP
        for i in range(1, 299):
            sensor.push_sample(1.2 + i * 0.001, 100.0 + i * 0.1)
        res = sensor.read()
        self.assertFalse(res.valid)
        self.assertEqual(res.state, "WARMING_UP")
        self.assertEqual(res.metadata["buffer_len"], 299)
        sensor.interpreter.predict.assert_not_called()

        # 300th sample -> predict called
        sensor.interpreter.predict = Mock(return_value=Mock(class_index=0, confidence=0.9, model_id="mmwave_v4"))
        sensor.push_sample(1.5, 100.0 + 299 * 0.1)
        res = sensor.read()
        self.assertTrue(res.valid)
        self.assertEqual(res.state, "NORMAL")
        sensor.interpreter.predict.assert_called_once()

    def test_mmwave_timestamp_reversal_and_nan_rejection(self):
        sensor = MMWaveSensorAdapter()
        sensor.connected = True

        # Valid push
        self.assertTrue(sensor.push_sample(1.0, 10.0))

        # Duplicate timestamp
        self.assertFalse(sensor.push_sample(1.1, 10.0))
        self.assertEqual(sensor.current_state, SensorState.INVALID_FORMAT)

        # Reversed timestamp
        self.assertFalse(sensor.push_sample(1.2, 9.0))
        self.assertEqual(sensor.current_state, SensorState.INVALID_FORMAT)

        # NaN sample
        self.assertFalse(sensor.push_sample(float("nan"), 11.0))
        self.assertEqual(sensor.current_state, SensorState.NAN_OR_INF)

        # Inf sample
        self.assertFalse(sensor.push_sample(float("inf"), 12.0))
        self.assertEqual(sensor.current_state, SensorState.NAN_OR_INF)

    def test_mmwave_reconnect_clears_buffer(self):
        sensor = MMWaveSensorAdapter()
        sensor.connected = True
        sensor.interpreter.predict = Mock(return_value=Mock(class_index=0, confidence=0.9, model_id="mmwave_v4"))

        for i in range(300):
            sensor.push_sample(1.0, i * 0.1)
        res = sensor.read()
        self.assertTrue(res.valid)

        # Reconnect resets buffer
        try:
            sensor.connect()
        except HardwareBackendUnavailable:
            pass

        sensor.connected = True
        res_after = sensor.read()
        self.assertFalse(res_after.valid)
        self.assertEqual(res_after.state, "WARMING_UP")
        self.assertEqual(res_after.metadata["buffer_len"], 0)


class TestCO2WarmingUp(unittest.TestCase):
    def test_co2_history_duration_and_unit(self):
        sensor = CO2SensorAdapter()
        sensor.connected = True
        sensor.interpreter.predict = Mock(side_effect=AssertionError("predict must not be called during warmup"))

        sensor.read_raw_values = Mock(return_value=(600.0, 50.0, 24.0))

        # 0 samples -> WARMING_UP (first read adds 1 sample, still < 2)
        res = sensor.read()
        self.assertFalse(res.valid)
        self.assertEqual(res.state, "WARMING_UP")
        self.assertEqual(res.error, "INSUFFICIENT_HISTORY")

        # 2nd sample with insufficient elapsed time (< 5 sec) -> WARMING_UP
        res = sensor.read()
        self.assertFalse(res.valid)
        self.assertEqual(res.state, "WARMING_UP")

        # Manually set history with sufficient span (>= 5.0 sec) to test inference path
        now = time.time()
        sensor.co2_history.clear()
        sensor.co2_history.append((now - 60.0, 500.0))  # 60 sec ago

        # Next read() will add (now, 600.0) -> elapsed = 60 sec, slope = (600-500)/1 = 100 ppm/min
        sensor.interpreter.predict = Mock(return_value=Mock(class_index=0, confidence=0.95, class_name="UNOCCUPIED_NORMAL", model_id="co2_v4"))
        res = sensor.read()
        self.assertTrue(res.valid)
        sensor.interpreter.predict.assert_called_once()
        # Verify ppm/min slope unit and exact value in metadata
        self.assertIn("co2_slope_ppm_min", res.metadata)
        self.assertAlmostEqual(res.metadata["co2_slope_ppm_min"], 100.0, places=3)

    def test_co2_predict_signature_autospec(self):
        """Enforce strict signature match using create_autospec so single-array argument calls fail with TypeError."""
        from unittest.mock import create_autospec
        from inference.co2_interpreter import CO2Interpreter, CO2Prediction

        sensor = CO2SensorAdapter()
        sensor.connected = True
        sensor.read_raw_values = Mock(return_value=(600.0, 50.0, 24.0))

        sensor.interpreter = create_autospec(CO2Interpreter, instance=True)
        sensor.interpreter.predict.return_value = CO2Prediction(
            class_index=0,
            class_name="UNOCCUPIED_NORMAL",
            confidence=0.95,
            probabilities=[0.95, 0.05],
            latency_ms=0.1,
            model_id="co2_v4",
            model_version="4.0",
        )

        now = time.time()
        sensor.co2_history.clear()
        sensor.co2_history.append((now - 60.0, 500.0))

        res = sensor.read()
        self.assertTrue(res.valid)
        self.assertEqual(res.state, "UNOCCUPIED_NORMAL")

        sensor.interpreter.predict.assert_called_once()
        args = sensor.interpreter.predict.call_args.args
        self.assertEqual(len(args), 3)
        self.assertAlmostEqual(args[0], 100.0, places=3)  # co2_slope
        self.assertAlmostEqual(args[1], 50.0, places=3)   # humidity
        self.assertAlmostEqual(args[2], 600.0, places=3)  # co2_ppm

    def test_co2_real_interpreter_integration(self):
        """Integration test executing CO2SensorAdapter.read() with real TFLite CO2Interpreter."""
        sensor = CO2SensorAdapter()
        sensor.connected = True
        sensor.read_raw_values = Mock(return_value=(600.0, 50.0, 24.0))

        now = time.time()
        sensor.co2_history.clear()
        sensor.co2_history.append((now - 60.0, 500.0))

        res = sensor.read()
        self.assertTrue(res.valid)
        self.assertNotEqual(res.state, "INFER_ERROR")
        self.assertIsNone(res.error)
        self.assertIn("co2_slope_ppm_min", res.metadata)
        self.assertAlmostEqual(res.metadata["co2_slope_ppm_min"], 100.0, places=3)
        self.assertIn("probabilities", res.metadata)
        self.assertTrue(all(np.isfinite(p) for p in res.metadata["probabilities"]))
        self.assertTrue(0.0 <= res.score <= 1.0)

    def test_co2_timestamp_reversal(self):
        sensor = CO2SensorAdapter()
        sensor.connected = True
        t_base = time.time()
        sensor.co2_history.append((t_base + 10.0, 600.0))

        # Call slope calculation with past timestamp (t_base + 5.0 <= t_base + 10.0)
        slope, err = sensor.calculate_co2_slope(t_base + 5.0, 650.0)
        self.assertEqual(err, "NON_MONOTONIC_TIMESTAMP")
        self.assertIsNone(slope)

    def test_co2_reconnect_resets_history(self):
        sensor = CO2SensorAdapter()
        sensor.co2_history.append((time.time(), 500.0))

        try:
            sensor.connect()
        except HardwareBackendUnavailable:
            pass

        self.assertEqual(len(sensor.co2_history), 0)


class TestPIRStartup(unittest.TestCase):
    def test_pir_startup_grace_period_and_monotonic(self):
        sensor = PIRSensorAdapter(no_motion_threshold_sec=15.0, startup_grace_period_sec=5.0)
        sensor.connected = True
        sensor.read_gpio = Mock(return_value=False)

        # Before grace period ends (0 elapsed sec) -> WARMING_UP
        sensor.connect_monotonic_ts = time.monotonic()
        res = sensor.read()
        self.assertFalse(res.valid)
        self.assertEqual(res.state, "WARMING_UP")
        self.assertEqual(res.error, "PIR_WARMING_UP")

        # After grace period ends (6 sec elapsed), but no motion event -> motion calculated from startup
        sensor.connect_monotonic_ts = time.monotonic() - 6.0
        res = sensor.read()
        self.assertTrue(res.valid)
        self.assertEqual(res.state, "MOTION")

        # Motion event triggers -> MOTION state
        sensor.read_gpio = Mock(return_value=True)
        res = sensor.read()
        self.assertTrue(res.valid)
        self.assertEqual(res.state, "MOTION")

        # After motion event, test threshold boundaries (14.9s vs 15.0s vs 15.1s)
        sensor.read_gpio = Mock(return_value=False)
        now_m = time.monotonic()

        # 14.9s elapsed -> MOTION
        sensor.last_motion_monotonic_ts = now_m - 14.9
        res = sensor.read()
        self.assertEqual(res.state, "MOTION")

        # 15.0s elapsed -> LONG_NO_MOTION
        sensor.last_motion_monotonic_ts = now_m - 15.0
        res = sensor.read()
        self.assertEqual(res.state, "LONG_NO_MOTION")

        # 15.1s elapsed -> LONG_NO_MOTION
        sensor.last_motion_monotonic_ts = now_m - 15.1
        res = sensor.read()
        self.assertEqual(res.state, "LONG_NO_MOTION")

    def test_pir_reconnect_resets_state(self):
        sensor = PIRSensorAdapter()
        sensor.has_motion_event = True
        sensor.has_observation = True

        try:
            sensor.connect()
        except HardwareBackendUnavailable:
            pass

        self.assertFalse(sensor.has_motion_event)
        self.assertFalse(sensor.has_observation)


class TestThermalStartup(unittest.TestCase):
    def test_thermal_frame_validation(self):
        sensor = Thermal44Sensor()
        sensor.connected = True

        # Missing frame (None) -> WARMING_UP
        sensor.read_frame = Mock(return_value=None)
        res = sensor.read()
        self.assertFalse(res.valid)
        self.assertEqual(res.state, "WARMING_UP")
        self.assertEqual(res.error, "WAITING_FOR_FIRST_FRAME")

        # Wrong shape (10, 10) -> INVALID_FORMAT
        sensor.read_frame = Mock(return_value=np.full((10, 10), 25.0, dtype=np.float32))
        res = sensor.read()
        self.assertFalse(res.valid)
        self.assertEqual(res.state, "INVALID_FORMAT")

        # NaN frame -> NAN_OR_INF
        bad_frame = np.full((62, 80), 25.0, dtype=np.float32)
        bad_frame[0, 0] = np.nan
        sensor.read_frame = Mock(return_value=bad_frame)
        res = sensor.read()
        self.assertFalse(res.valid)
        self.assertEqual(res.state, "NAN_OR_INF")

        # Inf frame -> NAN_OR_INF
        bad_frame_inf = np.full((62, 80), 25.0, dtype=np.float32)
        bad_frame_inf[0, 0] = np.inf
        sensor.read_frame = Mock(return_value=bad_frame_inf)
        res = sensor.read()
        self.assertFalse(res.valid)
        self.assertEqual(res.state, "NAN_OR_INF")

        # Valid frame -> inference runs
        sensor.interpreter.predict = Mock(return_value=Mock(class_index=0, class_name="NORMAL", confidence=0.99, model_id="thermal_v4", latency_ms=1.2))
        sensor.read_frame = Mock(return_value=np.full((62, 80), 25.0, dtype=np.float32))
        res = sensor.read()
        self.assertTrue(res.valid)
        self.assertEqual(res.state, "NORMAL")

    def test_thermal_real_mode_fail_closed_preserved(self):
        sensor = Thermal44Sensor()
        with self.assertRaises(HardwareBackendUnavailable):
            sensor.connect()
        with self.assertRaises(HardwareBackendUnavailable):
            sensor.read_frame()


class TestIntegratedNodeWarmingUp(unittest.TestCase):
    def test_all_mock_sensors_warming_up(self):
        node = SafeNestIntegratedNode(mode="mock")
        node.start()

        # Set all mock sensors to WARMING_UP
        for sensor in node.sensors.values():
            if hasattr(sensor, "set_scenario"):
                sensor.set_scenario("WARMING_UP")

        output = node.step().to_dict()

        # System should be FAULT when all sensors are WARMING_UP / invalid
        self.assertEqual(output["system_status"], "FAULT")
        self.assertEqual(output["level"], "FAULT")
        self.assertTrue(output["fallback_used"])

        for sensor_id in ("thermal44", "mmwave", "co2", "pir"):
            self.assertFalse(output["sensors"][sensor_id]["valid"])
            self.assertEqual(output["sensors"][sensor_id]["state"], "WARMING_UP")

    def test_partial_sensors_warming_up_degrades_system(self):
        node = SafeNestIntegratedNode(mode="mock")
        node.start()

        # Set only mmwave and co2 to WARMING_UP, thermal and pir normal
        node.sensors["mmwave"].set_scenario("WARMING_UP")
        node.sensors["co2"].set_scenario("WARMING_UP")
        node.sensors["thermal44"].set_scenario("NORMAL")
        node.sensors["pir"].set_scenario("MOTION")

        output = node.step().to_dict()

        self.assertEqual(output["system_status"], "DEGRADED")
        self.assertEqual(output["level"], "NORMAL")  # thermal and pir valid low risk
        self.assertTrue(output["fallback_used"])
        self.assertFalse(output["sensors"]["mmwave"]["valid"])
        self.assertFalse(output["sensors"]["co2"]["valid"])
        self.assertTrue(output["sensors"]["thermal44"]["valid"])
        self.assertTrue(output["sensors"]["pir"]["valid"])


if __name__ == "__main__":
    unittest.main()
