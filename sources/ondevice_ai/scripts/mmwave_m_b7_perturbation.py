#!/usr/bin/env python3
"""Deterministic perturbation and strict-INT8 helpers for SafeNest M-B7.

The module contains no training or conversion path.  It applies the frozen M-B1
BPF/Z-score contract, constructs the preregistered M-B7 perturbations, and runs
already-qualified strict-INT8 TFLite artifacts with one reusable interpreter per
seed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from mmwave_m_b1_preprocessing import transform_signals
from mmwave_m_b2_imbalance import LABEL_NAMES, compute_one_vs_rest_false_positives
from mmwave_timeline import TimelineProfile, analyze_timeline, resample_timeline


GLOBAL_PERTURBATION_SEED = 20260811
SAMPLE_RATE_HZ = 10.0
WINDOW_SAMPLES = 300
DRIFT_FREQUENCY_HZ = 0.05
NUMERICAL_EPS = 1e-18
FROZEN_SEEDS = (42, 43, 44)

CLEAN_PROFILE_ID = "M-B7_CLEAN"
PERTURBATION_PROFILE_ORDER = (
    "M-B7_GAUSSIAN_SNR20",
    "M-B7_GAUSSIAN_SNR10",
    "M-B7_GAUSSIAN_POST_B1_SNR20",
    "M-B7_GAUSSIAN_POST_B1_SNR10",
    "M-B7_AMP_X0_50",
    "M-B7_AMP_X0_75",
    "M-B7_AMP_X1_25",
    "M-B7_AMP_X1_50",
    "M-B7_DRIFT_MILD",
    "M-B7_DRIFT_SEVERE",
    "M-B7_DROPOUT_SHORT",
    "M-B7_DROPOUT_LONG",
    "M-B7_MISSING_FRAME_1PCT",
    "M-B7_MISSING_FRAME_5PCT",
    "M-B7_MOTION_BURST_MILD",
    "M-B7_MOTION_BURST_SEVERE",
    "M-B7_COMBINED_MODERATE",
)
ALL_PROFILE_ORDER = (CLEAN_PROFILE_ID,) + PERTURBATION_PROFILE_ORDER


PROFILE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    CLEAN_PROFILE_ID: {
        "family": "CLEAN",
        "injection_domain": "CLEAN_FROZEN_B1_OUTPUT",
        "stochastic": False,
    },
    "M-B7_GAUSSIAN_SNR20": {
        "family": "GAUSSIAN_NOISE",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "target_snr_db": 20.0,
    },
    "M-B7_GAUSSIAN_SNR10": {
        "family": "GAUSSIAN_NOISE",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "target_snr_db": 10.0,
    },
    "M-B7_GAUSSIAN_POST_B1_SNR20": {
        "family": "GAUSSIAN_NOISE",
        "injection_domain": "POST_B1_MODEL_INPUT",
        "stochastic": True,
        "target_snr_db": 20.0,
    },
    "M-B7_GAUSSIAN_POST_B1_SNR10": {
        "family": "GAUSSIAN_NOISE",
        "injection_domain": "POST_B1_MODEL_INPUT",
        "stochastic": True,
        "target_snr_db": 10.0,
    },
    "M-B7_AMP_X0_50": {
        "family": "AMPLITUDE_SCALING",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": False,
        "scale": 0.50,
    },
    "M-B7_AMP_X0_75": {
        "family": "AMPLITUDE_SCALING",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": False,
        "scale": 0.75,
    },
    "M-B7_AMP_X1_25": {
        "family": "AMPLITUDE_SCALING",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": False,
        "scale": 1.25,
    },
    "M-B7_AMP_X1_50": {
        "family": "AMPLITUDE_SCALING",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": False,
        "scale": 1.50,
    },
    "M-B7_DRIFT_MILD": {
        "family": "BASELINE_DRIFT",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "frequency_hz": DRIFT_FREQUENCY_HZ,
        "amplitude_rms_multiplier": 0.25,
    },
    "M-B7_DRIFT_SEVERE": {
        "family": "BASELINE_DRIFT",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "frequency_hz": DRIFT_FREQUENCY_HZ,
        "amplitude_rms_multiplier": 0.50,
    },
    "M-B7_DROPOUT_SHORT": {
        "family": "CONTIGUOUS_DROPOUT",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "duration_samples": 5,
        "duration_seconds": 0.5,
        "replacement_policy": "LINEAR_INTERPOLATION_INTERIOR_NEAREST_VALID_HOLD_BOUNDARY",
    },
    "M-B7_DROPOUT_LONG": {
        "family": "CONTIGUOUS_DROPOUT",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "duration_samples": 30,
        "duration_seconds": 3.0,
        "replacement_policy": "LINEAR_INTERPOLATION_INTERIOR_NEAREST_VALID_HOLD_BOUNDARY",
    },
    "M-B7_MISSING_FRAME_1PCT": {
        "family": "MISSING_FRAME",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "missing_fraction": 0.01,
        "missing_count": 3,
        "repair_policy": "A3_MMWAVE_TIMELINE_PROFILE_001_LINEAR_INTERPOLATION",
    },
    "M-B7_MISSING_FRAME_5PCT": {
        "family": "MISSING_FRAME",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "missing_fraction": 0.05,
        "missing_count": 15,
        "repair_policy": "A3_MMWAVE_TIMELINE_PROFILE_001_LINEAR_INTERPOLATION",
    },
    "M-B7_MOTION_BURST_MILD": {
        "family": "MOTION_BURST",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "duration_samples": 5,
        "duration_seconds": 0.5,
        "std_multiplier": 3.0,
        "waveform": "SIGNED_RECTANGULAR_ADDITIVE_BURST",
    },
    "M-B7_MOTION_BURST_SEVERE": {
        "family": "MOTION_BURST",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "duration_samples": 5,
        "duration_seconds": 0.5,
        "std_multiplier": 6.0,
        "waveform": "SIGNED_RECTANGULAR_ADDITIVE_BURST",
    },
    "M-B7_COMBINED_MODERATE": {
        "family": "COMBINED",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "gaussian_target_snr_db": 20.0,
        "amplitude_scale": 0.75,
        "dropout_duration_samples": 5,
        "dropout_duration_seconds": 0.5,
        "application_order": [
            "GAUSSIAN_SNR20",
            "AMPLITUDE_X0_75",
            "DROPOUT_SHORT_LINEAR_INTERPOLATION",
        ],
    },
}


def perturbation_profile_contract() -> Dict[str, Any]:
    """Return the immutable preregistered profile and RNG contract."""
    return {
        "phase_id": "M-B7",
        "global_perturbation_seed": GLOBAL_PERTURBATION_SEED,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window_samples": WINDOW_SAMPLES,
        "profile_order": list(ALL_PROFILE_ORDER),
        "perturbation_profile_count": len(PERTURBATION_PROFILE_ORDER),
        "total_inference_profile_count": len(ALL_PROFILE_ORDER),
        "rng_derivation": {
            "identity_string": "{global_seed}|{canonical_sample_index}|{window_id}|{profile_id}",
            "hash": "SHA-256",
            "integer_rule": "unsigned big-endian integer from first 8 digest bytes",
            "numpy_generator": "numpy.random.default_rng(derived_uint64_seed)",
            "processing_order_independent": True,
        },
        "frozen_preprocessing": {
            "profile_id": "M-B1_D0_B1_Z1",
            "profile_name": "BPF_ZSCORE",
            "bpf": {
                "type": "Butterworth band-pass",
                "order": 4,
                "lowcut_hz": 0.1,
                "highcut_hz": 0.5,
                "zero_phase": True,
            },
            "zscore": "global TRAIN-fitted scalar mean/std after BPF",
        },
        "profiles": {pid: dict(PROFILE_DEFINITIONS[pid]) for pid in ALL_PROFILE_ORDER},
        "timestamp_jitter": {
            "status": "OPTIONAL_PROFILE_NOT_ADDED",
            "reason": "The fixed prompt matrix defines missing-frame profiles but no preregistered timestamp-jitter magnitude; M-B7 does not add a post-hoc profile.",
            "a3_resampling_machinery_available": True,
        },
    }


def _round(value: float, digits: int = 9) -> float:
    return round(float(value), digits)


def array_sha256(array: np.ndarray, dtype: Optional[np.dtype] = None) -> str:
    arr = np.asarray(array, dtype=dtype) if dtype is not None else np.asarray(array)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def root_mean_square(array: np.ndarray) -> float:
    arr = np.asarray(array, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(arr))))


def derive_sample_seed(window: Dict[str, Any], profile_id: str) -> int:
    identity = (
        f"{GLOBAL_PERTURBATION_SEED}|{int(window['canonical_sample_index'])}|"
        f"{window['window_id']}|{profile_id}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _gaussian_noise(
    signal: np.ndarray,
    target_snr_db: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    signal64 = np.asarray(signal, dtype=np.float64)
    signal_power = float(np.mean(np.square(signal64)))
    if not np.isfinite(signal_power) or signal_power <= NUMERICAL_EPS:
        raise ValueError("DEGENERATE_SIGNAL_POWER")
    noise_power = signal_power / (10.0 ** (float(target_snr_db) / 10.0))
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=signal64.shape).astype(np.float64)
    achieved_noise_power = float(np.mean(np.square(noise)))
    if not np.isfinite(achieved_noise_power) or achieved_noise_power <= NUMERICAL_EPS:
        raise ValueError("DEGENERATE_GENERATED_NOISE_POWER")
    achieved_snr_db = 10.0 * np.log10(signal_power / achieved_noise_power)
    return noise, {
        "target_snr_db": float(target_snr_db),
        "signal_power": _round(signal_power, 12),
        "theoretical_noise_power": _round(noise_power, 12),
        "achieved_noise_power": _round(achieved_noise_power, 12),
        "achieved_snr_db": _round(achieved_snr_db, 9),
        "noise_sha256_float64": array_sha256(noise, np.float64),
        "formula": "noise_power = mean(x^2) / 10^(SNR_dB/10); noise ~ N(0, noise_power)",
    }


def _dropout_repair(
    signal: np.ndarray,
    start: int,
    length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    source = np.asarray(signal, dtype=np.float64)
    if start < 0 or length <= 0 or start + length > source.size:
        raise ValueError("INVALID_DROPOUT_RANGE")
    repaired = source.copy()
    mask = np.zeros(source.size, dtype=np.uint8)
    mask[start : start + length] = 1
    end = start + length
    if start > 0 and end < source.size:
        repaired[start:end] = np.linspace(source[start - 1], source[end], length + 2)[1:-1]
    elif start == 0 and end < source.size:
        repaired[start:end] = source[end]
    elif start > 0 and end == source.size:
        repaired[start:end] = source[start - 1]
    else:
        raise ValueError("DROPOUT_REMOVES_COMPLETE_WINDOW")
    return repaired, mask


def _missing_frame_gap_records(removed_indices: Sequence[int]) -> List[Dict[str, Any]]:
    indices = sorted(int(v) for v in removed_indices)
    if not indices:
        return []
    groups: List[List[int]] = [[indices[0]]]
    for index in indices[1:]:
        if index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [
        {
            "first_removed_index": group[0],
            "last_removed_index": group[-1],
            "missing_count": len(group),
            "effective_gap_seconds": _round((len(group) + 1) / SAMPLE_RATE_HZ, 6),
        }
        for group in groups
    ]


def _apply_missing_frame_a3(
    signal: np.ndarray,
    removed_indices: Sequence[int],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    source = np.asarray(signal, dtype=np.float64)
    removed = np.asarray(sorted(int(v) for v in removed_indices), dtype=int)
    keep_mask = np.ones(WINDOW_SAMPLES, dtype=bool)
    keep_mask[removed] = False
    # Match A3's canonical grid construction exactly.  Using ``index / 10``
    # makes the final 29.9-second value round one ULP below the value produced
    # by A3's ``index * 0.1`` policy, which would make its floor-based grid one
    # sample short even though both timelines represent the same 10 Hz window.
    native_timestamps = (
        np.arange(WINDOW_SAMPLES, dtype=np.float64) * (1.0 / SAMPLE_RATE_HZ)
    )
    damaged_signal = source[keep_mask]
    damaged_timestamps = native_timestamps[keep_mask]
    profile = TimelineProfile()
    analysis = analyze_timeline(damaged_timestamps, profile)
    repaired, repaired_timestamps, _, metadata = resample_timeline(
        damaged_signal,
        damaged_timestamps,
        dt.datetime(2000, 1, 1),
        profile,
        analysis,
    )
    permissible = bool(metadata.get("resampling_permissible", False))
    valid = (
        permissible
        and bool(metadata.get("resampling_performed", False))
        and repaired.shape == source.shape
        and repaired_timestamps.shape == native_timestamps.shape
        and np.all(np.isfinite(repaired))
    )
    evidence = {
        "removed_indices": removed.tolist(),
        "removed_count": int(removed.size),
        "removed_mask": (~keep_mask).astype(np.uint8).tolist(),
        "effective_gaps": _missing_frame_gap_records(removed.tolist()),
        "a3_profile_id": profile.profile_id,
        "a3_decision_code": metadata.get("decision_code"),
        "a3_resampling_required": bool(metadata.get("resampling_required", False)),
        "a3_resampling_permissible": permissible,
        "a3_resampling_performed": bool(metadata.get("resampling_performed", False)),
        "interpolated_count": int(metadata.get("interpolated_sample_count", 0)),
        "large_gap_count": int(analysis.get("large_gap_count", 0)),
        "small_gap_count": int(analysis.get("small_gap_count", 0)),
        "rejected_count": 0 if valid else 1,
    }
    return (repaired if valid else None), evidence


def _preprocess_b1(signal: np.ndarray, zscore_stats: Dict[str, float]) -> np.ndarray:
    result = transform_signals(
        np.asarray(signal, dtype=np.float64).reshape(1, WINDOW_SAMPLES),
        detrend=False,
        bpf=True,
        zscore=True,
        zscore_stats=zscore_stats,
    )[0]
    return np.asarray(result, dtype=np.float64)


def generate_profile_sample(
    profile_id: str,
    canonical_signal: np.ndarray,
    clean_b1_signal: np.ndarray,
    window: Dict[str, Any],
    zscore_stats: Dict[str, float],
) -> Dict[str, Any]:
    """Generate one deterministic profile sample and its compact evidence."""
    if profile_id not in PROFILE_DEFINITIONS:
        raise KeyError(f"Unknown M-B7 profile: {profile_id}")
    clean = np.asarray(canonical_signal, dtype=np.float64).reshape(-1)
    clean_b1 = np.asarray(clean_b1_signal, dtype=np.float64).reshape(-1)
    if clean.shape != (WINDOW_SAMPLES,) or clean_b1.shape != (WINDOW_SAMPLES,):
        raise ValueError("M-B7 inputs must be 300-sample vectors")

    definition = PROFILE_DEFINITIONS[profile_id]
    family = definition["family"]
    derived_seed = derive_sample_seed(window, profile_id)
    rng = np.random.default_rng(derived_seed)
    params: Dict[str, Any] = {}
    perturbed_canonical: Optional[np.ndarray] = clean.copy()
    model_input: Optional[np.ndarray] = None
    invalid_reasons: List[str] = []

    try:
        if family == "CLEAN":
            model_input = clean_b1.copy()
            params = {"operation": "IDENTITY"}

        elif family == "GAUSSIAN_NOISE":
            signal_reference = clean if definition["injection_domain"] == "CANONICAL_PHASE_PRE_B1" else clean_b1
            noise, noise_meta = _gaussian_noise(signal_reference, definition["target_snr_db"], rng)
            params = noise_meta
            if definition["injection_domain"] == "CANONICAL_PHASE_PRE_B1":
                perturbed_canonical = clean + noise
            else:
                perturbed_canonical = clean.copy()
                model_input = clean_b1 + noise

        elif family == "AMPLITUDE_SCALING":
            scale = float(definition["scale"])
            perturbed_canonical = scale * clean
            params = {
                "scale": scale,
                "max_abs_formula_error": _round(np.max(np.abs(perturbed_canonical - scale * clean)), 12),
            }

        elif family == "BASELINE_DRIFT":
            scale = root_mean_square(clean)
            if not np.isfinite(scale) or scale <= NUMERICAL_EPS:
                raise ValueError("DEGENERATE_WINDOW_RMS")
            amplitude = float(definition["amplitude_rms_multiplier"]) * scale
            phase_offset = float(rng.uniform(0.0, 2.0 * np.pi))
            time_s = np.arange(WINDOW_SAMPLES, dtype=np.float64) / SAMPLE_RATE_HZ
            drift = amplitude * np.sin(2.0 * np.pi * float(definition["frequency_hz"]) * time_s + phase_offset)
            perturbed_canonical = clean + drift
            params = {
                "frequency_hz": float(definition["frequency_hz"]),
                "window_scale_rms": _round(scale, 12),
                "amplitude_multiplier": float(definition["amplitude_rms_multiplier"]),
                "amplitude": _round(amplitude, 12),
                "phase_offset_rad": _round(phase_offset, 12),
                "drift_rms": _round(root_mean_square(drift), 12),
                "drift_sha256_float64": array_sha256(drift, np.float64),
            }

        elif family == "CONTIGUOUS_DROPOUT":
            length = int(definition["duration_samples"])
            start = int(rng.integers(0, WINDOW_SAMPLES - length + 1))
            perturbed_canonical, mask = _dropout_repair(clean, start, length)
            params = {
                "start_index": start,
                "end_index_exclusive": start + length,
                "duration_samples": length,
                "duration_seconds": float(definition["duration_seconds"]),
                "replacement_policy": definition["replacement_policy"],
                "dropout_mask": mask.tolist(),
                "dropout_mask_sha256_uint8": array_sha256(mask, np.uint8),
            }

        elif family == "MISSING_FRAME":
            missing_count = int(definition["missing_count"])
            removed = sorted(
                int(v)
                for v in rng.choice(
                    np.arange(1, WINDOW_SAMPLES - 1, dtype=int),
                    size=missing_count,
                    replace=False,
                ).tolist()
            )
            repaired, missing_meta = _apply_missing_frame_a3(clean, removed)
            params = {
                "missing_fraction": float(definition["missing_fraction"]),
                "repair_policy": definition["repair_policy"],
                **missing_meta,
            }
            if repaired is None:
                invalid_reasons.append("UNRECOVERABLE_TIMELINE_CORRUPTION")
                perturbed_canonical = None
            else:
                perturbed_canonical = repaired

        elif family == "MOTION_BURST":
            std = float(np.std(clean))
            if not np.isfinite(std) or std <= NUMERICAL_EPS:
                raise ValueError("DEGENERATE_WINDOW_STANDARD_DEVIATION")
            length = int(definition["duration_samples"])
            start = int(rng.integers(0, WINDOW_SAMPLES - length + 1))
            sign = int(rng.choice(np.asarray([-1, 1], dtype=int)))
            amplitude = sign * float(definition["std_multiplier"]) * std
            burst = np.zeros(WINDOW_SAMPLES, dtype=np.float64)
            burst[start : start + length] = amplitude
            perturbed_canonical = clean + burst
            params = {
                "start_index": start,
                "end_index_exclusive": start + length,
                "duration_samples": length,
                "duration_seconds": float(definition["duration_seconds"]),
                "window_std": _round(std, 12),
                "std_multiplier": float(definition["std_multiplier"]),
                "sign": sign,
                "signed_amplitude": _round(amplitude, 12),
                "waveform": definition["waveform"],
                "burst_sha256_float64": array_sha256(burst, np.float64),
            }

        elif family == "COMBINED":
            noise, noise_meta = _gaussian_noise(clean, definition["gaussian_target_snr_db"], rng)
            stage_gaussian = clean + noise
            stage_amplitude = float(definition["amplitude_scale"]) * stage_gaussian
            length = int(definition["dropout_duration_samples"])
            start = int(rng.integers(0, WINDOW_SAMPLES - length + 1))
            perturbed_canonical, mask = _dropout_repair(stage_amplitude, start, length)
            params = {
                "application_order": list(definition["application_order"]),
                "gaussian": noise_meta,
                "amplitude_scale": float(definition["amplitude_scale"]),
                "dropout_start_index": start,
                "dropout_end_index_exclusive": start + length,
                "dropout_duration_samples": length,
                "dropout_duration_seconds": float(definition["dropout_duration_seconds"]),
                "dropout_mask": mask.tolist(),
                "dropout_mask_sha256_uint8": array_sha256(mask, np.uint8),
                "stage_gaussian_sha256_float64": array_sha256(stage_gaussian, np.float64),
                "stage_amplitude_sha256_float64": array_sha256(stage_amplitude, np.float64),
            }

        else:
            raise ValueError(f"Unsupported perturbation family: {family}")

        if perturbed_canonical is not None and model_input is None:
            model_input = _preprocess_b1(perturbed_canonical, zscore_stats)

        if model_input is not None and not np.all(np.isfinite(model_input)):
            invalid_reasons.append("NON_FINITE_MODEL_INPUT")
            model_input = None

    except (FloatingPointError, ValueError) as exc:
        invalid_reasons.append(str(exc))
        perturbed_canonical = None
        model_input = None

    valid = model_input is not None
    if valid:
        assert model_input is not None
        pre_delta_rms: Optional[float]
        if definition["injection_domain"] == "CANONICAL_PHASE_PRE_B1":
            assert perturbed_canonical is not None
            pre_delta_rms = root_mean_square(perturbed_canonical - clean)
        else:
            pre_delta_rms = None
        post_delta_rms = root_mean_square(model_input - clean_b1)
        attenuation_ratio = (
            float(post_delta_rms / pre_delta_rms)
            if pre_delta_rms is not None and pre_delta_rms > NUMERICAL_EPS
            else None
        )
        perturbed_canonical_sha = (
            array_sha256(perturbed_canonical, np.float64)
            if perturbed_canonical is not None
            else None
        )
        model_input_sha = array_sha256(model_input.astype(np.float32), np.float32)
    else:
        pre_delta_rms = None
        post_delta_rms = None
        attenuation_ratio = None
        perturbed_canonical_sha = None
        model_input_sha = None

    evidence = {
        "phase_id": "M-B7",
        "profile_id": profile_id,
        "family": family,
        "injection_domain": definition["injection_domain"],
        "canonical_sample_index": int(window["canonical_sample_index"]),
        "window_id": window["window_id"],
        "subject_id": window["subject_id"],
        "recording_id": window["recording_id"],
        "split": window["split"],
        "true_class": int(window["safenest_label_id"]),
        "true_label": window["safenest_label"],
        "derived_rng_seed": derived_seed if definition["stochastic"] else None,
        "rng_used": bool(definition["stochastic"]),
        "parameters": params,
        "validity_status": (
            "INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION"
            if valid
            else "INVALID_OR_FALLBACK_RECOMMENDED"
        ),
        "invalid_reason_codes": sorted(set(invalid_reasons)),
        "pre_b1_delta_rms": _round(pre_delta_rms, 12) if pre_delta_rms is not None else None,
        "post_b1_delta_rms": _round(post_delta_rms, 12) if post_delta_rms is not None else None,
        "preprocessing_attenuation_ratio": (
            _round(attenuation_ratio, 12) if attenuation_ratio is not None else None
        ),
        "perturbed_canonical_sha256_float64": perturbed_canonical_sha,
        "model_input_sha256_float32": model_input_sha,
    }
    return {
        "valid": valid,
        "model_input": model_input.astype(np.float32) if model_input is not None else None,
        "perturbed_canonical": perturbed_canonical,
        "evidence": evidence,
    }


class StrictInt8Runner:
    """Reusable strict-INT8 interpreter plus exact pre-clamp diagnostics."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self.model_bytes = self.model_path.read_bytes()
        self.sha256 = hashlib.sha256(self.model_bytes).hexdigest()
        self.interpreter = tf.lite.Interpreter(model_content=self.model_bytes)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.input_index = int(self.input_detail["index"])
        self.output_index = int(self.output_detail["index"])
        self.input_scale = float(self.input_detail["quantization"][0])
        self.input_zero_point = int(self.input_detail["quantization"][1])
        self.output_scale = float(self.output_detail["quantization"][0])
        self.output_zero_point = int(self.output_detail["quantization"][1])
        self.op_types = [op["op_name"] for op in self.interpreter._get_ops_details()]

    def structure(self) -> Dict[str, Any]:
        return {
            "relative_path": None,
            "bytes": len(self.model_bytes),
            "sha256": self.sha256,
            "input_dtype": self.input_detail["dtype"].__name__,
            "output_dtype": self.output_detail["dtype"].__name__,
            "input_shape": [int(v) for v in self.input_detail["shape"]],
            "output_shape": [int(v) for v in self.output_detail["shape"]],
            "input_scale": self.input_scale,
            "input_zero_point": self.input_zero_point,
            "output_scale": self.output_scale,
            "output_zero_point": self.output_zero_point,
            "op_types": list(self.op_types),
            "select_tf_ops_count": sum(
                1 for name in self.op_types if "Flex" in name or "Select" in name
            ),
        }

    def infer(self, model_inputs: np.ndarray) -> Dict[str, np.ndarray]:
        inputs = np.asarray(model_inputs, dtype=np.float32)
        if inputs.ndim == 2:
            inputs = np.expand_dims(inputs, axis=-1)
        if inputs.ndim != 3 or inputs.shape[1:] != (WINDOW_SAMPLES, 1):
            raise ValueError(f"Expected (N,300,1) model inputs, got {inputs.shape}")
        count = inputs.shape[0]
        predictions = np.empty(count, dtype=np.int16)
        probabilities = np.empty((count, len(LABEL_NAMES)), dtype=np.float32)
        saturation_counts = np.zeros(count, dtype=np.int32)
        output_endpoint_counts = np.zeros(count, dtype=np.int32)

        for index in range(count):
            sample = inputs[index : index + 1]
            q_raw = np.round(sample / self.input_scale + self.input_zero_point)
            saturation_counts[index] = int(np.count_nonzero((q_raw < -128) | (q_raw > 127)))
            quantized = np.clip(q_raw, -128, 127).astype(np.int8)
            self.interpreter.set_tensor(self.input_index, quantized)
            self.interpreter.invoke()
            output_int8 = self.interpreter.get_tensor(self.output_index)
            output_endpoint_counts[index] = int(
                np.count_nonzero((output_int8 == -128) | (output_int8 == 127))
            )
            dequantized = (
                output_int8.astype(np.float32) - self.output_zero_point
            ) * self.output_scale
            probabilities[index] = dequantized[0]
            predictions[index] = int(np.argmax(dequantized[0]))

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "saturation_counts": saturation_counts,
            "output_endpoint_counts": output_endpoint_counts,
        }


def confidence_statistics(values: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
        }
    return {
        "count": int(arr.size),
        "mean": _round(np.mean(arr), 6),
        "median": _round(np.median(arr), 6),
        "p05": _round(np.percentile(arr, 5), 6),
        "p95": _round(np.percentile(arr, 95), 6),
    }


def confusion_matrix_rows(y_true: np.ndarray, predictions: np.ndarray) -> List[List[int]]:
    return [
        [
            int(np.count_nonzero((y_true == true_id) & (predictions == predicted_id)))
            for predicted_id in range(len(LABEL_NAMES))
        ]
        for true_id in range(len(LABEL_NAMES))
    ]


def class_collapse_state(
    predictions: np.ndarray,
    class_metrics: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    prediction_counts = {
        name: int(np.count_nonzero(predictions == class_id))
        for class_id, name in enumerate(LABEL_NAMES)
    }
    zero_recall = [name for name in LABEL_NAMES if class_metrics[name]["recall"] == 0.0]
    zero_predictions = [name for name in LABEL_NAMES if prediction_counts[name] == 0]
    collapsed = (
        "APNEA" in zero_recall
        or "RAPID_OR_ABNORMAL" in zero_recall
        or len(np.unique(predictions)) < len(LABEL_NAMES)
    )
    return {
        "collapsed": bool(collapsed),
        "zero_recall_classes": zero_recall,
        "zero_prediction_classes": zero_predictions,
    }


def compute_run_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    saturation_counts: np.ndarray,
    output_endpoint_counts: np.ndarray,
    clean_predictions: np.ndarray,
    clean_probabilities: np.ndarray,
    window_ids: Sequence[str],
) -> Dict[str, Any]:
    """Compute one seed/profile record relative to that seed's clean baseline."""
    y = np.asarray(y_true, dtype=int)
    preds = np.asarray(predictions, dtype=int)
    probs = np.asarray(probabilities, dtype=np.float32)
    clean_preds = np.asarray(clean_predictions, dtype=int)
    clean_probs = np.asarray(clean_probabilities, dtype=np.float32)
    sat = np.asarray(saturation_counts, dtype=int)
    endpoints = np.asarray(output_endpoint_counts, dtype=int)
    if not (len(y) == len(preds) == len(probs) == len(clean_preds) == len(clean_probs)):
        raise ValueError("Metric inputs are not aligned")

    class_metrics = compute_one_vs_rest_false_positives(y, preds)
    clean_class_metrics = compute_one_vs_rest_false_positives(y, clean_preds)
    macro_f1 = float(np.mean([class_metrics[name]["f1_score"] for name in LABEL_NAMES]))
    clean_macro_f1 = float(
        np.mean([clean_class_metrics[name]["f1_score"] for name in LABEL_NAMES])
    )
    accuracy = float(np.mean(preds == y))
    clean_accuracy = float(np.mean(clean_preds == y))
    confidence = np.max(probs, axis=1)
    clean_confidence = np.max(clean_probs, axis=1)
    correct_mask = preds == y
    collapse = class_collapse_state(preds, class_metrics)
    clean_collapse = class_collapse_state(clean_preds, clean_class_metrics)
    per_class_signed_recall_delta = {
        name: _round(class_metrics[name]["recall"] - clean_class_metrics[name]["recall"], 6)
        for name in LABEL_NAMES
    }
    per_class_positive_recall_degradation = {
        name: _round(
            max(0.0, clean_class_metrics[name]["recall"] - class_metrics[name]["recall"]),
            6,
        )
        for name in LABEL_NAMES
    }
    worst_sat_index = int(np.argmax(sat)) if sat.size else 0
    total_elements = int(len(y) * WINDOW_SAMPLES)
    total_outputs = int(len(y) * len(LABEL_NAMES))
    return {
        "evaluated_sample_count": int(len(y)),
        "macro_f1": _round(macro_f1, 6),
        "accuracy": _round(accuracy, 6),
        "per_class": class_metrics,
        "confusion_matrix": confusion_matrix_rows(y, preds),
        "prediction_distribution": {
            name: int(np.count_nonzero(preds == class_id))
            for class_id, name in enumerate(LABEL_NAMES)
        },
        "class_collapse_state": {
            **collapse,
            "new_relative_to_clean": bool((not clean_collapse["collapsed"]) and collapse["collapsed"]),
        },
        "relative_to_clean": {
            "clean_macro_f1": _round(clean_macro_f1, 6),
            "signed_macro_f1_delta": _round(macro_f1 - clean_macro_f1, 6),
            "positive_macro_f1_degradation": _round(max(0.0, clean_macro_f1 - macro_f1), 6),
            "clean_accuracy": _round(clean_accuracy, 6),
            "signed_accuracy_delta": _round(accuracy - clean_accuracy, 6),
            "per_class_signed_recall_delta": per_class_signed_recall_delta,
            "per_class_positive_recall_degradation": per_class_positive_recall_degradation,
            "maximum_positive_per_class_recall_degradation": _round(
                max(per_class_positive_recall_degradation.values()), 6
            ),
            "top1_agreement": _round(np.mean(preds == clean_preds), 6),
            "prediction_change_count": int(np.count_nonzero(preds != clean_preds)),
            "mean_confidence_change": _round(np.mean(confidence - clean_confidence), 6),
        },
        "confidence": {
            "definition": "max(dequantized output probability vector)",
            "all_predictions": confidence_statistics(confidence),
            "correct_predictions": confidence_statistics(confidence[correct_mask]),
            "incorrect_predictions": confidence_statistics(confidence[~correct_mask]),
            "calibrated_probability_claimed": False,
        },
        "quantization": {
            "saturated_element_count": int(np.sum(sat)),
            "input_saturation_ratio": _round(np.sum(sat) / total_elements, 9),
            "saturated_sample_count": int(np.count_nonzero(sat > 0)),
            "worst_per_sample_saturation_count": int(sat[worst_sat_index]) if sat.size else 0,
            "worst_per_sample_saturation_ratio": _round(
                sat[worst_sat_index] / WINDOW_SAMPLES if sat.size else 0.0, 9
            ),
            "worst_sample_window_id": window_ids[worst_sat_index] if sat.size else None,
            "output_endpoint_count": int(np.sum(endpoints)),
            "output_endpoint_ratio": _round(np.sum(endpoints) / total_outputs, 9),
        },
    }


def subject_level_metrics(
    windows: Sequence[Dict[str, Any]],
    predictions: np.ndarray,
    clean_predictions: np.ndarray,
) -> Dict[str, Any]:
    """Compute required per-subject metrics and clean-to-perturbed deltas."""
    result: Dict[str, Any] = {"subject_count": 0, "per_subject": {}}
    subjects = sorted({str(window["subject_id"]) for window in windows})
    result["subject_count"] = len(subjects)
    for subject_id in subjects:
        indices = np.asarray(
            [index for index, window in enumerate(windows) if window["subject_id"] == subject_id],
            dtype=int,
        )
        y = np.asarray([windows[index]["safenest_label_id"] for index in indices], dtype=int)
        pred = np.asarray(predictions, dtype=int)[indices]
        clean_pred = np.asarray(clean_predictions, dtype=int)[indices]
        class_metrics = compute_one_vs_rest_false_positives(y, pred)
        clean_class_metrics = compute_one_vs_rest_false_positives(y, clean_pred)
        supported_f1 = [
            class_metrics[name]["f1_score"]
            for name in LABEL_NAMES
            if class_metrics[name]["support"] > 0
        ]
        clean_supported_f1 = [
            clean_class_metrics[name]["f1_score"]
            for name in LABEL_NAMES
            if clean_class_metrics[name]["support"] > 0
        ]
        accuracy = float(np.mean(pred == y))
        clean_accuracy = float(np.mean(clean_pred == y))
        macro_f1 = float(np.mean(supported_f1)) if supported_f1 else 0.0
        clean_macro_f1 = float(np.mean(clean_supported_f1)) if clean_supported_f1 else 0.0
        distribution = {
            name: int(np.count_nonzero(pred == class_id))
            for class_id, name in enumerate(LABEL_NAMES)
        }
        clean_distribution = {
            name: int(np.count_nonzero(clean_pred == class_id))
            for class_id, name in enumerate(LABEL_NAMES)
        }
        result["per_subject"][subject_id] = {
            "window_count": int(len(indices)),
            "accuracy": _round(accuracy, 6),
            "subject_macro_f1": _round(macro_f1, 6),
            "prediction_distribution": distribution,
            "per_class": class_metrics,
            "clean_to_perturbed_change": {
                "accuracy_delta": _round(accuracy - clean_accuracy, 6),
                "subject_macro_f1_delta": _round(macro_f1 - clean_macro_f1, 6),
                "prediction_distribution_delta": {
                    name: distribution[name] - clean_distribution[name] for name in LABEL_NAMES
                },
                "per_class_recall_delta": {
                    name: _round(
                        class_metrics[name]["recall"] - clean_class_metrics[name]["recall"], 6
                    )
                    for name in LABEL_NAMES
                },
            },
        }
    return result


def aggregate_cross_seed(
    perturbation_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate each perturbation without hiding per-seed values."""
    summaries: Dict[str, Any] = {}
    for profile_id in PERTURBATION_PROFILE_ORDER:
        per_seed = perturbation_results[profile_id]["per_seed"]
        f1_degradation = np.asarray(
            [per_seed[str(seed)]["relative_to_clean"]["positive_macro_f1_degradation"] for seed in FROZEN_SEEDS],
            dtype=np.float64,
        )
        top1 = np.asarray(
            [per_seed[str(seed)]["relative_to_clean"]["top1_agreement"] for seed in FROZEN_SEEDS],
            dtype=np.float64,
        )
        recall_degradation = np.asarray(
            [
                per_seed[str(seed)]["relative_to_clean"][
                    "maximum_positive_per_class_recall_degradation"
                ]
                for seed in FROZEN_SEEDS
            ],
            dtype=np.float64,
        )
        saturation = np.asarray(
            [per_seed[str(seed)]["quantization"]["input_saturation_ratio"] for seed in FROZEN_SEEDS],
            dtype=np.float64,
        )
        confidence_change = np.asarray(
            [per_seed[str(seed)]["relative_to_clean"]["mean_confidence_change"] for seed in FROZEN_SEEDS],
            dtype=np.float64,
        )
        collapses = [
            seed for seed in FROZEN_SEEDS if per_seed[str(seed)]["class_collapse_state"]["collapsed"]
        ]
        f1_worst_index = int(np.argmax(f1_degradation))
        top1_worst_index = int(np.argmin(top1))
        recall_worst_index = int(np.argmax(recall_degradation))
        saturation_worst_index = int(np.argmax(saturation))
        confidence_worst_index = int(np.argmin(confidence_change))
        summaries[profile_id] = {
            "macro_f1_degradation": {
                "mean": _round(np.mean(f1_degradation), 6),
                "median": _round(np.median(f1_degradation), 6),
                "std": _round(np.std(f1_degradation), 6),
                "min": _round(np.min(f1_degradation), 6),
                "max": _round(np.max(f1_degradation), 6),
                "worst_seed": int(FROZEN_SEEDS[f1_worst_index]),
                "per_seed": {
                    str(seed): _round(f1_degradation[index], 6)
                    for index, seed in enumerate(FROZEN_SEEDS)
                },
            },
            "top1_agreement": {
                "mean": _round(np.mean(top1), 6),
                "minimum": _round(np.min(top1), 6),
                "worst_seed": int(FROZEN_SEEDS[top1_worst_index]),
                "per_seed": {
                    str(seed): _round(top1[index], 6)
                    for index, seed in enumerate(FROZEN_SEEDS)
                },
            },
            "maximum_positive_per_class_recall_degradation": {
                "mean": _round(np.mean(recall_degradation), 6),
                "max": _round(np.max(recall_degradation), 6),
                "worst_seed": int(FROZEN_SEEDS[recall_worst_index]),
                "per_seed": {
                    str(seed): _round(recall_degradation[index], 6)
                    for index, seed in enumerate(FROZEN_SEEDS)
                },
            },
            "input_saturation": {
                "mean": _round(np.mean(saturation), 9),
                "max": _round(np.max(saturation), 9),
                "worst_seed": int(FROZEN_SEEDS[saturation_worst_index]),
                "per_seed": {
                    str(seed): _round(saturation[index], 9)
                    for index, seed in enumerate(FROZEN_SEEDS)
                },
            },
            "confidence_change": {
                "mean": _round(np.mean(confidence_change), 6),
                "worst": _round(np.min(confidence_change), 6),
                "worst_seed": int(FROZEN_SEEDS[confidence_worst_index]),
                "per_seed": {
                    str(seed): _round(confidence_change[index], 6)
                    for index, seed in enumerate(FROZEN_SEEDS)
                },
            },
            "collapse": {
                "total": len(collapses),
                "affected_seeds": collapses,
            },
        }
    return {"phase_id": "M-B7", "profiles": summaries}


__all__ = [
    "ALL_PROFILE_ORDER",
    "CLEAN_PROFILE_ID",
    "FROZEN_SEEDS",
    "GLOBAL_PERTURBATION_SEED",
    "PERTURBATION_PROFILE_ORDER",
    "PROFILE_DEFINITIONS",
    "StrictInt8Runner",
    "aggregate_cross_seed",
    "array_sha256",
    "compute_run_metrics",
    "derive_sample_seed",
    "generate_profile_sample",
    "perturbation_profile_contract",
    "subject_level_metrics",
]
