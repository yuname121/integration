#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_mmwave_input_adapter.py
P0-6 CSV Adapter 세션 분리 및 Gap/보간 정밀 검증 테스트
"""

import os
import sys
from pathlib import Path
import tempfile
import unittest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.mmwave_csv_adapter import MMWaveCSVAdapter


class TestMMWaveInputAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = MMWaveCSVAdapter(
            sample_rate_hz=10.0,
            window_seconds=30.0,
            stride_seconds=3.0,
            max_gap_seconds=0.5,
            max_interpolated_fraction=0.05
        )

    def test_session_boundary_isolation(self):
        """두 개 이상의 세션이 섞이지 않고 독립적으로 window가 생성되는지 검증"""
        # Session A: 350 samples (35 seconds)
        ts_a = np.arange(350, dtype=np.float64) * 0.1
        df_a = pd.DataFrame({
            "timestamp_s": ts_a,
            "resp_phase": np.sin(ts_a),
            "subject_id": "SUBJ_01",
            "session_id": "SESS_A",
            "label": "NORMAL"
        })

        # Session B: 350 samples (35 seconds)
        ts_b = np.arange(350, dtype=np.float64) * 0.1 + 100.0
        df_b = pd.DataFrame({
            "timestamp_s": ts_b,
            "resp_phase": np.cos(ts_b),
            "subject_id": "SUBJ_01",
            "session_id": "SESS_B",
            "label": "APNEA"
        })

        df_combined = pd.concat([df_a, df_b], ignore_index=True)

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            df_combined.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            windows = list(self.adapter.iter_windows(csv_path))
            # SESS_A -> (350 - 300) / 30 + 1 = 2 windows
            # SESS_B -> 2 windows
            # Total 4 windows, none crossing boundaries
            self.assertEqual(len(windows), 4)

            sess_a_count = sum(1 for w in windows if w.session_id == "SESS_A")
            sess_b_count = sum(1 for w in windows if w.session_id == "SESS_B")
            self.assertEqual(sess_a_count, 2)
            self.assertEqual(sess_b_count, 2)

            for w in windows:
                self.assertEqual(w.values.shape, (300,))
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_large_gap_rejection(self):
        """0.5초 초과 타임스탬프 공백 발생 시 window 폐기 검증"""
        ts = np.arange(350, dtype=np.float64) * 0.1
        ts[150:] += 5.0  # 5 second gap at index 150

        df = pd.DataFrame({
            "timestamp_s": ts,
            "resp_phase": np.sin(ts),
            "subject_id": "SUBJ_01",
            "session_id": "SESS_GAP",
            "label": "NORMAL"
        })

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            windows = list(self.adapter.iter_windows(csv_path))
            # Window crossing index 150 must be discarded due to gap > 0.5s
            for w in windows:
                # Assert no window spans the gap between index 149 and 150
                self.assertFalse(w.started_at_s < 15.0 and w.ended_at_s > 20.0)
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)


if __name__ == "__main__":
    unittest.main()
