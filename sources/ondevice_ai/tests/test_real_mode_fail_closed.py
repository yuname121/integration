#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_real_mode_fail_closed.py
Regression tests for SafeNest V4 Real Mode Fail-Closed architecture
"""

import unittest
import json
from unittest.mock import Mock

from sensors.base_sensor import HardwareBackendUnavailable
from sensors.thermal44.thermal44_driver import Thermal44Sensor
from sensors.mmwave.mmwave_adapter import MMWaveSensorAdapter
from sensors.co2.co2_adapter import CO2SensorAdapter
from sensors.pir.pir_adapter import PIRSensorAdapter

from integrated_node.run_node import SafeNestIntegratedNode


class TestRealModeFailClosed(unittest.TestCase):
    def test_thermal_real_backend_does_not_generate_frame(self):
        sensor = Thermal44Sensor()
        with self.assertRaises(HardwareBackendUnavailable):
            sensor.connect()
        with self.assertRaises(HardwareBackendUnavailable):
            sensor.read_frame()

    def test_co2_real_backend_does_not_return_synthetic_values(self):
        sensor = CO2SensorAdapter()
        with self.assertRaises(HardwareBackendUnavailable):
            sensor.connect()
        with self.assertRaises(HardwareBackendUnavailable):
            sensor.read_raw_values()

    def test_pir_real_backend_does_not_return_synthetic_motion(self):
        sensor = PIRSensorAdapter()
        with self.assertRaises(HardwareBackendUnavailable):
            sensor.connect()
        with self.assertRaises(HardwareBackendUnavailable):
            sensor.read_gpio()

    def test_mmwave_real_backend_does_not_connect(self):
        sensor = MMWaveSensorAdapter()
        with self.assertRaises(HardwareBackendUnavailable):
            sensor.connect()

    def test_real_mode_reports_all_unimplemented_backends(self):
        node = SafeNestIntegratedNode(mode="real")
        node.start()
        result = node.step().to_dict()

        self.assertEqual(result["level"], "FAULT")
        self.assertEqual(result["system_status"], "FAULT")
        self.assertTrue(result["fallback_used"])

        for sensor_id in ("thermal44", "mmwave", "co2", "pir"):
            sensor = result["sensors"][sensor_id]
            self.assertFalse(sensor["valid"])
            self.assertEqual(sensor["state"], "EXTERNAL_SENSOR_PROVIDER_REQUIRED")
            self.assertEqual(sensor["error"], "EXTERNAL_SENSOR_PROVIDER_REQUIRED")

    def test_prediction_call_suppression_when_backends_unavailable(self):
        node = SafeNestIntegratedNode(mode="real")
        for sensor in node.sensors.values():
            if hasattr(sensor, "interpreter"):
                sensor.interpreter.predict = Mock(
                    side_effect=AssertionError("predict must not be called")
                )

        node.start()
        result = node.step().to_dict()

        self.assertEqual(result["level"], "FAULT")
        for sensor_id in ("thermal44", "mmwave", "co2", "pir"):
            self.assertFalse(result["sensors"][sensor_id]["valid"])

    def test_mock_mode_remains_operational(self):
        node = SafeNestIntegratedNode(mode="mock")
        node.start()
        result = node.step().to_dict()

        self.assertNotEqual(result["system_status"], "FAULT")
        self.assertNotIn(
            "HARDWARE_BACKEND_NOT_IMPLEMENTED",
            json.dumps(result),
        )


if __name__ == "__main__":
    unittest.main()
