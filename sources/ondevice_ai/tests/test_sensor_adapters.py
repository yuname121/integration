#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_sensor_adapters.py
Unit tests for SafeNest V4 BaseSensor contract and Mock Sensor Adapters
"""

import unittest
import time

from sensors.base_sensor import SensorState
from sensors.thermal44.mock_sensor import MockThermalSensor
from sensors.mmwave.mock_sensor import MockMMWaveSensor
from sensors.co2.mock_sensor import MockCO2Sensor
from sensors.pir.mock_sensor import MockPIRSensor


class TestSensorAdapters(unittest.TestCase):
    def test_mock_thermal_sensor(self):
        sensor = MockThermalSensor()
        self.assertTrue(sensor.connect())

        # Test normal scenario
        res_norm = sensor.read()
        self.assertTrue(res_norm.valid)
        self.assertEqual(res_norm.score, 0.0)

        # Test fall scenario
        sensor.set_scenario("FALL")
        res_fall = sensor.read()
        self.assertTrue(res_fall.valid)
        self.assertEqual(res_fall.score, 1.0)
        self.assertEqual(res_fall.state, "HUMAN_FALL")

        sensor.close()

    def test_mock_mmwave_sensor(self):
        sensor = MockMMWaveSensor()
        self.assertTrue(sensor.connect())

        # Test normal
        res_norm = sensor.read()
        self.assertTrue(res_norm.valid)

        # Test apnea
        sensor.set_scenario("APNEA")
        res_apnea = sensor.read()
        self.assertTrue(res_apnea.valid)
        self.assertEqual(res_apnea.score, 1.0)
        self.assertEqual(res_apnea.state, "APNEA")

        sensor.close()

    def test_mock_co2_sensor(self):
        sensor = MockCO2Sensor()
        self.assertTrue(sensor.connect())

        # Test normal
        res_norm = sensor.read()
        self.assertTrue(res_norm.valid)

        # Test elevated
        sensor.set_scenario("ELEVATED")
        res_elevated = sensor.read()
        self.assertTrue(res_elevated.valid)
        self.assertEqual(res_elevated.score, 1.0)

        sensor.close()

    def test_mock_pir_sensor(self):
        sensor = MockPIRSensor(no_motion_threshold_sec=1.0)
        self.assertTrue(sensor.connect())

        sensor.set_scenario("MOTION")
        res_motion = sensor.read()
        self.assertTrue(res_motion.valid)
        self.assertEqual(res_motion.score, 0.0)
        self.assertEqual(res_motion.state, "MOTION")

        sensor.set_scenario("NO_MOTION")
        time.sleep(0.1)
        res_no_motion = sensor.read()
        self.assertTrue(res_no_motion.valid)
        self.assertEqual(res_no_motion.score, 1.0)
        self.assertEqual(res_no_motion.state, "LONG_NO_MOTION")

        sensor.close()


if __name__ == "__main__":
    unittest.main()
