#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_mmwave_stream_adapter.py
P0-6 MMWaveStreamAdapter 단위 검증 테스트 (NaN 거부, 중복/역순 거부, presence=0 초기화, stale 차단, 0.0 timestamp 경계)
"""

import os
import sys
from pathlib import Path
import time
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.mmwave_stream_adapter import MMWaveStreamAdapter


class TestMMWaveStreamAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = MMWaveStreamAdapter(window_samples=300, sample_rate_hz=10.0, max_gap_seconds=0.5)

    def test_nan_inf_rejection(self):
        res1 = self.adapter.push_sample(np.nan, timestamp_s=1.0)
        self.assertFalse(res1.accepted)
        self.assertEqual(res1.reason, "MMWAVE_VALUE_NAN_OR_INF")
        self.assertEqual(len(self.adapter.buffer), 0)

        res2 = self.adapter.push_sample(np.inf, timestamp_s=1.1)
        self.assertFalse(res2.accepted)
        self.assertEqual(res2.reason, "MMWAVE_VALUE_NAN_OR_INF")
        self.assertEqual(len(self.adapter.buffer), 0)

    def test_non_monotonic_timestamp_rejection(self):
        self.adapter.push_sample(0.1, timestamp_s=10.0)
        res_dup = self.adapter.push_sample(0.2, timestamp_s=10.0)
        self.assertFalse(res_dup.accepted)
        self.assertEqual(res_dup.reason, "MMWAVE_TIMESTAMP_NON_MONOTONIC")

        res_rev = self.adapter.push_sample(0.3, timestamp_s=9.5)
        self.assertFalse(res_rev.accepted)
        self.assertEqual(res_rev.reason, "MMWAVE_TIMESTAMP_NON_MONOTONIC")

    def test_duplicate_timestamp_is_rejected_when_stream_starts_at_zero(self):
        """timestamp 0.0에서 시작하는 세션의 0.0 -> 0.0 중복 거부 검증"""
        self.adapter.push_sample(0.1, timestamp_s=0.0)
        res = self.adapter.push_sample(0.2, timestamp_s=0.0)
        self.assertFalse(res.accepted)
        self.assertEqual(res.reason, "MMWAVE_TIMESTAMP_NON_MONOTONIC")

    def test_large_gap_is_rejected_when_stream_starts_at_zero(self):
        """timestamp 0.0에서 시작하는 세션의 0.0 -> 10.0 큰 gap 거부 검증"""
        self.adapter.push_sample(0.1, timestamp_s=0.0)
        res = self.adapter.push_sample(0.2, timestamp_s=10.0)
        self.assertFalse(res.accepted)
        self.assertEqual(res.reason, "MMWAVE_STREAM_GAP_TOO_LARGE")
        self.assertEqual(res.buffer_size, 0)

    def test_large_gap_clears_buffer(self):
        for i in range(10):
            self.adapter.push_sample(0.1, timestamp_s=1.0 + i * 0.1)
        self.assertEqual(len(self.adapter.buffer), 10)

        res_gap = self.adapter.push_sample(0.2, timestamp_s=10.0)
        self.assertFalse(res_gap.accepted)
        self.assertEqual(res_gap.reason, "MMWAVE_STREAM_GAP_TOO_LARGE")
        self.assertEqual(len(self.adapter.buffer), 0)

    def test_presence_zero_clears_buffer(self):
        for i in range(10):
            self.adapter.push_sample(0.1, timestamp_s=1.0 + i * 0.1, presence=1)
        self.assertEqual(len(self.adapter.buffer), 10)

        res_nopresence = self.adapter.push_sample(0.1, timestamp_s=2.0, presence=0)
        self.assertFalse(res_nopresence.accepted)
        self.assertEqual(res_nopresence.reason, "MMWAVE_PRESENCE_NOT_DETECTED")
        self.assertEqual(len(self.adapter.buffer), 0)
        self.assertFalse(self.adapter.is_ready())

    def test_stale_window_returns_none(self):
        start_ts = 100.0
        for i in range(300):
            self.adapter.push_sample(0.1, timestamp_s=start_ts + i * 0.1)

        self.assertTrue(self.adapter.is_ready())
        current_now = start_ts + 29.9 + 5.0
        self.assertTrue(self.adapter.is_stale(current_time_s=current_now))
        self.assertIsNone(self.adapter.get_window(current_time_s=current_now))


if __name__ == "__main__":
    unittest.main()
