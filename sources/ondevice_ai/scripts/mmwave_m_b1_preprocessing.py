#!/usr/bin/env python3
"""SafeNest mmWave Phase M-B1 — Preprocessing Factorial Module.

Defines the 8 pre-registered preprocessing profiles (2^3 factorial over Detrend, BPF, Z-score)
and applies deterministic signal transformations to mmWave canonical phase windows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal

ROOT_DIR = Path(__file__).resolve().parents[1]

# 8 Pre-registered Preprocessing Profiles
PROFILES = [
    {
        "profile_id": "M-B1_D0_B0_Z0",
        "name": "RAW",
        "detrend": False,
        "bpf": False,
        "zscore": False,
        "description": "Raw canonical phase window without preprocessing",
    },
    {
        "profile_id": "M-B1_D1_B0_Z0",
        "name": "DETREND_ONLY",
        "detrend": True,
        "bpf": False,
        "zscore": False,
        "description": "Linear detrending only",
    },
    {
        "profile_id": "M-B1_D0_B1_Z0",
        "name": "BPF_ONLY",
        "detrend": False,
        "bpf": True,
        "zscore": False,
        "description": "Fixed 0.1-0.5 Hz 4th-order Butterworth BPF only",
    },
    {
        "profile_id": "M-B1_D1_B1_Z0",
        "name": "DETREND_BPF",
        "detrend": True,
        "bpf": True,
        "zscore": False,
        "description": "Linear detrending followed by fixed 0.1-0.5 Hz BPF",
    },
    {
        "profile_id": "M-B1_D0_B0_Z1",
        "name": "ZSCORE_ONLY",
        "detrend": False,
        "bpf": False,
        "zscore": True,
        "description": "TRAIN-fitted global Z-score standardization only",
    },
    {
        "profile_id": "M-B1_D1_B0_Z1",
        "name": "DETREND_ZSCORE",
        "detrend": True,
        "bpf": False,
        "zscore": True,
        "description": "Linear detrending followed by TRAIN-fitted global Z-score",
    },
    {
        "profile_id": "M-B1_D0_B1_Z1",
        "name": "BPF_ZSCORE",
        "detrend": False,
        "bpf": True,
        "zscore": True,
        "description": "Fixed 0.1-0.5 Hz BPF followed by TRAIN-fitted global Z-score",
    },
    {
        "profile_id": "M-B1_D1_B1_Z1",
        "name": "DETREND_BPF_ZSCORE",
        "detrend": True,
        "bpf": True,
        "zscore": True,
        "description": "Linear detrending, fixed 0.1-0.5 Hz BPF, and TRAIN-fitted global Z-score",
    },
]


def apply_linear_detrend(signal: np.ndarray) -> np.ndarray:
    """Apply genuine per-window linear detrending along time axis (axis=-1)."""
    return scipy.signal.detrend(signal, axis=-1, type="linear")


def apply_bpf(signal: np.ndarray, fs: float = 10.0, lowcut: float = 0.1, highcut: float = 0.5, order: int = 4) -> np.ndarray:
    """Apply zero-phase 4th-order Butterworth bandpass filter (0.1-0.5 Hz) along time axis."""
    b, a = scipy.signal.butter(order, [lowcut, highcut], btype="bandpass", fs=fs)
    return scipy.signal.filtfilt(b, a, signal, axis=-1)


def fit_train_zscore_statistics(train_signals: np.ndarray, detrend: bool, bpf: bool) -> dict[str, float]:
    """Fit global scalar mean and std from TRAIN split AFTER applying D/B transformations."""
    transformed = train_signals.copy()
    if detrend:
        transformed = apply_linear_detrend(transformed)
    if bpf:
        transformed = apply_bpf(transformed)

    flattened = transformed.ravel()
    mean_val = float(np.mean(flattened))
    std_val = float(np.std(flattened))
    if std_val == 0:
        std_val = 1.0

    return {"mean": mean_val, "std": std_val}


def transform_signals(
    signals: np.ndarray,
    detrend: bool,
    bpf: bool,
    zscore: bool,
    zscore_stats: dict[str, float] | None = None,
) -> np.ndarray:
    """Apply transformation pipeline to input signals matrix (N x 300)."""
    out = signals.astype(np.float64, copy=True)

    if detrend:
        out = apply_linear_detrend(out)

    if bpf:
        out = apply_bpf(out)

    if zscore:
        if zscore_stats is None:
            raise ValueError("zscore_stats required when zscore is True!")
        m = zscore_stats["mean"]
        s = zscore_stats["std"]
        out = (out - m) / s

    return out


def compute_tensor_fingerprint(tensor: np.ndarray) -> str:
    """Compute canonical SHA-256 hash for float32/float64 array bytes."""
    contiguous = np.ascontiguousarray(tensor)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def compute_signal_diagnostics(signals: np.ndarray) -> dict[str, Any]:
    """Compute comprehensive signal-domain diagnostic statistics for a signal array."""
    flat = signals.ravel()
    clip_exceed_count = int(np.sum(np.abs(flat) > 5.0))
    clip_exceed_ratio = float(clip_exceed_count / flat.size)

    window_stds = np.std(signals, axis=-1)
    constant_win_count = int(np.sum(window_stds < 1e-12))

    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "rms": float(np.sqrt(np.mean(flat**2))),
        "median": float(np.median(flat)),
        "p01": float(np.percentile(flat, 1)),
        "p05": float(np.percentile(flat, 5)),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "nan_count": int(np.isnan(flat).sum()),
        "inf_count": int(np.isinf(flat).sum()),
        "constant_window_count": constant_win_count,
        "legacy_clip_exceedance_ratio_at_abs_5": clip_exceed_ratio,
    }
