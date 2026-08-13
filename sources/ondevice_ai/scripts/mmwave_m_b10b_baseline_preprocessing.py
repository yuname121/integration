#!/usr/bin/env python3
"""Frozen, fail-closed compatibility preprocessors for historical mmWave baselines.

These adapters are intentionally separate from the selected real-data runtime
pipeline.  They provide deterministic, executable compatibility semantics for
the separately authorized M-B10B pass without claiming that v0.1 reproduces
unknown native preprocessing or that v0.2 has real-data training lineage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt


class BaselinePreprocessingError(ValueError):
    """Raised when a baseline input cannot be processed fail-closed."""


V01_CONTRACT_ID = "M-B10B_HISTORICAL_V0_1_COMPATIBILITY_PREPROCESSING_V1"
V02_CONTRACT_ID = "M-B10B_SYNTHETIC_V0_2_EXTERNAL_COMPATIBILITY_PREPROCESSING_V1"


def _window_300(window: Any) -> np.ndarray:
    """Normalize accepted tensor spellings to one finite float32 300-vector."""
    try:
        raw = np.asarray(window)
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise BaselinePreprocessingError("INVALID_WINDOW") from exc
    if raw.shape == (300, 1):
        raw = raw[:, 0]
    elif raw.shape == (1, 300, 1):
        raw = raw[0, :, 0]
    elif raw.shape != (300,):
        raise BaselinePreprocessingError("INVALID_SHAPE")
    try:
        values = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BaselinePreprocessingError("INVALID_DTYPE") from exc
    if not np.all(np.isfinite(values)):
        raise BaselinePreprocessingError("NAN_OR_INF")
    return values.astype(np.float32)


def _quantize(model_ready: np.ndarray, scale: float, zero_point: int) -> dict[str, Any]:
    if not np.isfinite(scale) or scale <= 0:
        raise BaselinePreprocessingError("INVALID_QUANTIZATION")
    raw = np.rint(model_ready / np.float32(scale) + np.int32(zero_point))
    limits = np.iinfo(np.int8)
    saturation = (raw < limits.min) | (raw > limits.max)
    return {
        "model_ready": model_ready.astype(np.float32, copy=False),
        "input_int8": np.clip(raw, limits.min, limits.max).astype(np.int8),
        "input_saturation_count": int(np.sum(saturation)),
        "input_saturation_ratio": float(np.mean(saturation)),
    }


def prepare_v01(window: Any) -> dict[str, Any]:
    """Execute the v0.1 historical compatibility adapter."""
    values = _window_300(window)
    normalized = (values - np.float32(0.006091983988881111)) / np.float32(2.5013835430145264)
    model_ready = normalized.reshape(1, 300, 1)
    result = _quantize(model_ready, 0.03259856998920441, -13)
    result.update({"contract_id": V01_CONTRACT_ID, "preprocessing_steps": ["VALIDATE_WINDOW", "IDENTITY_SEMANTIC_ADAPTER", "FIXED_Z_SCORE", "RESHAPE", "AFFINE_INT8_QUANTIZE"]})
    return result


def prepare_v02(window: Any) -> dict[str, Any]:
    """Execute the v0.2 recorded experimental compatibility pipeline."""
    values = _window_300(window)
    detrended = values - np.float32(np.mean(values, dtype=np.float64))
    b, a = butter(4, [0.1, 0.5], btype="bandpass", fs=10.0)
    filtered = filtfilt(b, a, detrended.astype(np.float64)).astype(np.float32)
    normalized = (filtered - np.float32(0.17212218046188354)) / np.float32(1.7171541452407837)
    clipped = np.clip(normalized, -5.0, 5.0).reshape(1, 300, 1).astype(np.float32)
    result = _quantize(clipped, 0.012282303534448147, 12)
    result.update({"contract_id": V02_CONTRACT_ID, "preprocessing_steps": ["VALIDATE_WINDOW", "LINEAR_DETREND", "BUTTERWORTH_BANDPASS_ZERO_PHASE", "FIXED_Z_SCORE", "CLIP", "RESHAPE", "AFFINE_INT8_QUANTIZE"]})
    return result


def prepare_baseline(window: Any, baseline_id: str) -> dict[str, Any]:
    """Dispatch only known historical/synthetic baseline IDs; never fallback."""
    if baseline_id == "mmwave_resp_int8":
        return prepare_v01(window)
    if baseline_id == "mmwave_resp_int8_v0.2.0_candidate":
        return prepare_v02(window)
    raise BaselinePreprocessingError("UNKNOWN_BASELINE_NO_FALLBACK")
