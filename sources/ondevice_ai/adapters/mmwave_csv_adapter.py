#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
adapters/mmwave_csv_adapter.py
P0-6 mmWave CSV 입력 Adapter 정밀 연산 구현

[검수 3차 정밀 연산]
1. (subject_id, session_id) 단위로 DataFrame 그룹화하여 세션 경계 간 window 누수 완전 차단
2. 역순/비단조 타임스탬프 발생 시 해당 세션 거부
3. max_gap_seconds (0.5초) 초과 공백 검출 시 해당 window 폐기
4. 정확한 0.1초 간격 10Hz target grid (29.9초) 생성 및 np.interp 선형 재샘플링 수행 (끝 구간 외삽 금지)
5. max_interpolated_fraction (0.05) 초과 보간 시 해당 window 폐기
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class MMWaveWindow:
    values: np.ndarray             # (300,) float32 resp_phase (정확히 10Hz 재샘플링)
    sample_rate_hz: float
    started_at_s: float
    ended_at_s: float
    subject_id: str
    session_id: str
    label: str | None
    quality: float
    interpolated_fraction: float
    source: str


class MMWaveCSVAdapter:
    def __init__(
        self,
        sample_rate_hz: float = 10.0,
        window_seconds: float = 30.0,
        stride_seconds: float = 3.0,
        max_gap_seconds: float = 0.5,
        max_interpolated_fraction: float = 0.05,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.window_seconds = window_seconds
        self.window_samples = int(sample_rate_hz * window_seconds)
        self.stride_samples = int(sample_rate_hz * stride_seconds)
        self.max_gap_seconds = max_gap_seconds
        self.max_interpolated_fraction = max_interpolated_fraction

    def iter_windows(self, csv_path: str | Path):
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path)
        required_cols = {"timestamp_s", "resp_phase"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV missing required columns {required_cols - set(df.columns)}")

        if "subject_id" not in df.columns:
            df["subject_id"] = "S_UNKNOWN"
        if "session_id" not in df.columns:
            df["session_id"] = "SES_UNKNOWN"
        if "label" not in df.columns:
            df["label"] = None

        for (subj, sess), group in df.groupby(["subject_id", "session_id"]):
            phases = group["resp_phase"].values.astype(np.float32)
            timestamps = group["timestamp_s"].values.astype(np.float64)
            labels = group["label"].values

            ts_diffs = np.diff(timestamps)
            if np.any(ts_diffs <= 0):
                continue

            if len(phases) < self.window_samples:
                continue

            idx = 0
            while idx + self.window_samples <= len(phases):
                raw_window_data = phases[idx : idx + self.window_samples]
                raw_window_ts = timestamps[idx : idx + self.window_samples]
                window_label = str(labels[idx]) if labels[idx] is not None else None

                if np.any(~np.isfinite(raw_window_data)) or np.any(~np.isfinite(raw_window_ts)):
                    idx += self.stride_samples
                    continue

                sub_diffs = np.diff(raw_window_ts)
                if np.any(sub_diffs > self.max_gap_seconds):
                    idx += self.stride_samples
                    continue

                expected_dt = 1.0 / self.sample_rate_hz
                target_10hz_ts = raw_window_ts[0] + np.arange(self.window_samples) * expected_dt

                if target_10hz_ts[-1] > raw_window_ts[-1] + 1e-5:
                    idx += self.stride_samples
                    continue

                resampled_values = np.interp(target_10hz_ts, raw_window_ts, raw_window_data).astype(np.float32)

                gap_fractions = np.maximum(0.0, sub_diffs - expected_dt)
                interpolated_fraction = float(np.sum(gap_fractions) / self.window_seconds)

                if interpolated_fraction > self.max_interpolated_fraction:
                    idx += self.stride_samples
                    continue

                yield MMWaveWindow(
                    values=resampled_values,
                    sample_rate_hz=self.sample_rate_hz,
                    started_at_s=float(target_10hz_ts[0]),
                    ended_at_s=float(target_10hz_ts[-1]),
                    subject_id=str(subj),
                    session_id=str(sess),
                    label=window_label,
                    quality=float(1.0 - interpolated_fraction),
                    interpolated_fraction=interpolated_fraction,
                    source="CSV_FILE"
                )

                idx += self.stride_samples
