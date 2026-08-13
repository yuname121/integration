#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocessing/mmwave.py
SafeNest V6 mmWave Common Preprocessing Module

Status: EXPERIMENTAL_PREPROCESSING_V1
Strict 7-Stage Preprocessing Pipeline:
 1. Finite check & NaN/Inf replacement
 2. Missing value handling
 3. Linear detrending (window mean subtraction)
 4. Bandpass filtering (Butterworth 0.1-0.5 Hz, order 4)
 5. Train-only Z-score normalization (mean, std)
 6. Clipping [-5.0, 5.0]
 7. Final shape [batch_size, 300, 1] float32
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
import numpy as np
try:
    from scipy.signal import butter, filtfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class MMWavePreprocessor:
    """
    Common preprocessing pipeline for mmWave 60GHz respiration signal (resp_phase).
    Status: EXPERIMENTAL_PREPROCESSING_V1
    """

    def __init__(
        self,
        mean: float = 0.006091983988881111,
        std: float = 2.5013835430145264,
        clip_min: float = -5.0,
        clip_max: float = 5.0,
        expected_samples: int = 300,
        fs_hz: float = 10.0,
        lowcut_hz: float = 0.1,
        highcut_hz: float = 0.5,
        filter_order: int = 4,
        apply_filter: bool = True
    ):
        self.mean = float(mean)
        self.std = float(std) if std > 0 else 1.0
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.expected_samples = expected_samples
        self.fs_hz = fs_hz
        self.lowcut_hz = lowcut_hz
        self.highcut_hz = highcut_hz
        self.filter_order = filter_order
        self.apply_filter = apply_filter

        # Design Butterworth bandpass filter if scipy available
        if HAS_SCIPY:
            nyq = 0.5 * fs_hz
            low = lowcut_hz / nyq
            high = highcut_hz / nyq
            self.b, self.a = butter(filter_order, [low, high], btype='bandpass')
        else:
            self.b, self.a = None, None

    def preprocess_window(self, raw_signal: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocesses a single 1D 300-sample window or 2D (300, 1) signal following 7-step sequence.
        """
        arr = np.asarray(raw_signal, dtype=np.float32).flatten()
        quality_info = {
            "valid": True,
            "reason": "OK",
            "original_length": len(arr),
            "nan_count": 0,
            "inf_count": 0,
            "preprocessing_stage": "EXPERIMENTAL_PREPROCESSING_V1"
        }

        # Step 1: Check length
        if len(arr) != self.expected_samples:
            quality_info["valid"] = False
            quality_info["reason"] = f"Expected {self.expected_samples} samples, got {len(arr)}"
            if len(arr) < self.expected_samples:
                arr = np.pad(arr, (0, self.expected_samples - len(arr)), mode='edge')
            else:
                arr = arr[:self.expected_samples]

        # Step 2: Finite check & NaN/Inf replacement
        nan_mask = np.isnan(arr)
        inf_mask = np.isinf(arr)
        nan_cnt = int(np.sum(nan_mask))
        inf_cnt = int(np.sum(inf_mask))
        quality_info["nan_count"] = nan_cnt
        quality_info["inf_count"] = inf_cnt

        if nan_cnt > 0 or inf_cnt > 0:
            quality_info["valid"] = False
            quality_info["reason"] = f"Signal contains {nan_cnt} NaNs and {inf_cnt} Infs"
            arr = np.nan_to_num(arr, nan=self.mean, posinf=self.clip_max, neginf=self.clip_min)

        # Step 3: Linear detrending (window mean subtraction)
        detrended = arr - np.mean(arr)

        # Step 4: Bandpass filtering (Butterworth 0.1-0.5 Hz)
        if self.apply_filter and HAS_SCIPY and self.b is not None and len(detrended) > 15:
            try:
                filtered = filtfilt(self.b, self.a, detrended)
            except Exception as e:
                quality_info["filter_warning"] = str(e)
                filtered = detrended
        else:
            filtered = detrended

        # Step 5: Train-only Z-score normalization
        normalized = (filtered - self.mean) / self.std

        # Step 6: Clipping [-5.0, 5.0]
        clipped = np.clip(normalized, self.clip_min, self.clip_max)

        # Step 7: Final shape conversion [1, 300, 1] float32
        final_signal = clipped.reshape(1, self.expected_samples, 1).astype(np.float32)
        return final_signal, quality_info

    def preprocess_batch(self, raw_batch: np.ndarray) -> np.ndarray:
        """
        Preprocesses a batch of shape (N, 300, 1) or (N, 300).
        """
        arr = np.asarray(raw_batch, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.squeeze(-1)
        
        N = len(arr)
        processed = np.zeros((N, self.expected_samples, 1), dtype=np.float32)
        for i in range(N):
            proc_win, _ = self.preprocess_window(arr[i])
            processed[i] = proc_win[0]
        return processed

    @classmethod
    def from_train_split(cls, X_train: np.ndarray, clip_min: float = -5.0, clip_max: float = 5.0, apply_filter: bool = True) -> MMWavePreprocessor:
        flat_tr = X_train.flatten()
        mean_val = float(np.mean(flat_tr))
        std_val = float(np.std(flat_tr))
        if std_val <= 0 or np.isnan(std_val):
            std_val = 1.0
        return cls(mean=mean_val, std=std_val, clip_min=clip_min, clip_max=clip_max, apply_filter=apply_filter)
