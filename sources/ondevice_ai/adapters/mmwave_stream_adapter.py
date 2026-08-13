#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
adapters/mmwave_stream_adapter.py
P0-6 mmWave 실시간 파이프라인 Ring Buffer Stream Adapter 정밀 구현

[검수 3차 핵심 수정]
1. PushResult dataclass 기반 명시적 입출력 결과 반환 (accepted, reason, buffer_size)
2. timestamp 0.0 경계 버그 수정 (self.last_timestamp: float | None = None 사용)
3. 0.0 -> 0.0 중복 및 0.0 -> 10.0 큰 gap 검사 완벽 지원
4. NaN/Inf 거부 (MMWAVE_VALUE_NAN_OR_INF)
5. presence 미감지(0) 시 버퍼 즉시 초기화 및 거부 (MMWAVE_PRESENCE_NOT_DETECTED)
6. stale (age_ms > 2000ms) 시 get_window()가 None 반환
"""

from __future__ import annotations
from dataclasses import dataclass
import time
import collections
import numpy as np


@dataclass(frozen=True)
class PushResult:
    accepted: bool
    reason: str | None
    buffer_size: int


class MMWaveStreamAdapter:
    def __init__(self, window_samples: int = 300, sample_rate_hz: float = 10.0, max_gap_seconds: float = 0.5):
        self.window_samples = window_samples
        self.sample_rate_hz = sample_rate_hz
        self.max_gap_seconds = max_gap_seconds
        self.buffer = collections.deque(maxlen=window_samples)
        self.timestamps = collections.deque(maxlen=window_samples)
        self.last_push_time: float = 0.0
        self.last_timestamp: float | None = None
        self.presence = 1

    def clear(self):
        self.buffer.clear()
        self.timestamps.clear()
        self.last_timestamp = None
        self.last_push_time = 0.0

    def push_sample(self, resp_phase_val: float | None, timestamp_s: float | None = None, presence: int = 1) -> PushResult:
        if presence != 1:
            self.presence = 0
            self.clear()
            return PushResult(accepted=False, reason="MMWAVE_PRESENCE_NOT_DETECTED", buffer_size=len(self.buffer))

        self.presence = 1

        if resp_phase_val is None or not np.isfinite(resp_phase_val):
            return PushResult(accepted=False, reason="MMWAVE_VALUE_NAN_OR_INF", buffer_size=len(self.buffer))

        ts = timestamp_s if timestamp_s is not None else time.time()
        if not np.isfinite(ts):
            return PushResult(accepted=False, reason="MMWAVE_TIMESTAMP_NON_FINITE", buffer_size=len(self.buffer))

        # timestamp 0.0 경계 정밀 검사
        if self.last_timestamp is not None:
            dt = ts - self.last_timestamp
            if dt <= 0:
                return PushResult(accepted=False, reason="MMWAVE_TIMESTAMP_NON_MONOTONIC", buffer_size=len(self.buffer))
            elif dt > self.max_gap_seconds:
                self.clear()
                return PushResult(accepted=False, reason="MMWAVE_STREAM_GAP_TOO_LARGE", buffer_size=len(self.buffer))

        self.buffer.append(float(resp_phase_val))
        self.timestamps.append(float(ts))
        self.last_push_time = time.time() if timestamp_s is None else float(ts)
        self.last_timestamp = float(ts)

        return PushResult(accepted=True, reason=None, buffer_size=len(self.buffer))

    def is_ready(self) -> bool:
        if self.presence == 0:
            return False
        return len(self.buffer) == self.window_samples

    def is_stale(self, current_time_s: float | None = None, max_age_ms: float = 2000.0) -> bool:
        if self.last_push_time == 0.0 or len(self.buffer) == 0:
            return True
        now = current_time_s if current_time_s is not None else time.time()
        age_ms = (now - self.last_push_time) * 1000.0
        return age_ms > max_age_ms or self.presence == 0

    def get_window(self, current_time_s: float | None = None) -> np.ndarray | None:
        if not self.is_ready() or self.is_stale(current_time_s=current_time_s):
            return None
        return np.array(self.buffer, dtype=np.float32)

    def get_status(self) -> dict:
        ready = self.is_ready()
        age_ms = (time.time() - self.last_push_time) * 1000.0 if self.last_push_time > 0 else 9999.0
        stale = age_ms > 2000.0 or self.presence == 0
        quality = 1.0 if (ready and not stale) else 0.0

        return {
            "ready": ready,
            "window_samples": len(self.buffer),
            "sample_rate_hz": self.sample_rate_hz,
            "age_ms": age_ms,
            "stale": stale,
            "presence": self.presence,
            "quality": quality,
            "source": "MR60_RESP_PHASE"
        }
