#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_risk_rules.py
RiskRulesEvaluator 5대 피처 순수 위험도 룰 단위 테스트 (타임스탬프 경계조건, presence 조건 포함)
"""

import os
import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk.risk_rules import RiskRulesEvaluator, validate_timestamp


class TestRiskRules(unittest.TestCase):
    def setUp(self):
        self.evaluator = RiskRulesEvaluator()

    def test_weights_sum_to_one(self):
        w = self.evaluator.weights
        self.assertAlmostEqual(sum(w.values()), 1.0, places=4)

    def test_normal_respiration(self):
        res = self.evaluator.evaluate_respiration(16.0, 0)
        self.assertEqual(res.status, "NORMAL")
        self.assertEqual(res.score, 0.0)

    def test_abnormal_respiration_rpm(self):
        res = self.evaluator.evaluate_respiration(8.0, 0)
        self.assertEqual(res.status, "CAUTION")
        self.assertIn("ABNORMAL_RESPIRATION_RPM", res.reasons)

    def test_apnea_timer_exact_boundary(self):
        """무호흡 1.9초(비응급) 및 정확히 2.0초(응급) 경계조건 및 리셋 검증"""
        # 1. 0.0s시작
        res1 = self.evaluator.evaluate_respiration(0.0, 0, sample_timestamp=10.0)
        self.assertFalse(res1.emergency_override)

        # 2. 1.9s 경과 -> 비응급
        res2 = self.evaluator.evaluate_respiration(0.0, 0, sample_timestamp=11.9)
        self.assertFalse(res2.emergency_override)

        # 3. 2.0s 경과 -> 응급 발동
        res3 = self.evaluator.evaluate_respiration(0.0, 0, sample_timestamp=12.0)
        self.assertTrue(res3.emergency_override)
        self.assertEqual(res3.status, "CRITICAL")

        # 4. 정상 호흡 수신 시 타이머 초기화 검증
        self.evaluator.evaluate_respiration(16.0, 0, sample_timestamp=13.0)
        self.assertIsNone(self.evaluator.apnea_started_at)

    def test_pir_no_motion_presence_boundary(self):
        """presence 미확인 시 LONG_NO_MOTION 누적 차단 및 14.9s/15.0s 경계 검증"""
        # 1. presence=False 상태에서 20초간 motion=0 -> LONG_NO_MOTION 발생 금지
        res_nopres = self.evaluator.evaluate_motion(0, presence_confirmed=False, sample_timestamp=10.0)
        self.assertNotIn("LONG_NO_MOTION", res_nopres.reasons)

        res_nopres_2 = self.evaluator.evaluate_motion(0, presence_confirmed=False, sample_timestamp=30.0)
        self.assertNotIn("LONG_NO_MOTION", res_nopres_2.reasons)

        # 2. presence=True로 14.9s 무움직임 -> 미확정
        self.evaluator.evaluate_motion(0, presence_confirmed=True, sample_timestamp=100.0)
        res_149 = self.evaluator.evaluate_motion(0, presence_confirmed=True, sample_timestamp=114.9)
        self.assertNotIn("LONG_NO_MOTION", res_149.reasons)

        # 3. presence=True로 15.0s 무움직임 -> LONG_NO_MOTION 발동
        res_150 = self.evaluator.evaluate_motion(0, presence_confirmed=True, sample_timestamp=115.0)
        self.assertIn("LONG_NO_MOTION", res_150.reasons)

        # 4. presence 상실 시 타이머 리셋 검증
        self.evaluator.evaluate_motion(0, presence_confirmed=False, sample_timestamp=116.0)
        self.assertIsNone(self.evaluator.no_motion_started_at)

    def test_timestamp_non_monotonic_validation(self):
        """타임스탬프 역행(10.0 -> 9.0) 시 오류 반환 및 타이머 초기화 검증"""
        valid1, err1 = validate_timestamp(10.0, previous=None)
        self.assertTrue(valid1)

        valid2, err2 = validate_timestamp(9.0, previous=10.0)
        self.assertFalse(valid2)
        self.assertEqual(err2, "SENSOR_TIMESTAMP_NON_MONOTONIC")

        # 무호흡 연산 중 타임스탬프 역행 발생 시 FAULT 반환
        self.evaluator.evaluate_respiration(0.0, 0, sample_timestamp=10.0)
        res_rev = self.evaluator.evaluate_respiration(0.0, 0, sample_timestamp=9.0)
        self.assertEqual(res_rev.status, "FAULT")
        self.assertIn("SENSOR_TIMESTAMP_NON_MONOTONIC", res_rev.reasons)

    def test_thermal_fall_emergency(self):
        res = self.evaluator.evaluate_posture(2, confidence=0.9)
        self.assertTrue(res.emergency_override)
        self.assertEqual(res.status, "CRITICAL")


if __name__ == "__main__":
    unittest.main()
