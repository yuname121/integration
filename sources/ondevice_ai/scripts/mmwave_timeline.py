#!/usr/bin/env python3
"""Canonical timeline, resampling policy, gap handling, and 30-second window generator.

Phase A3 of the SafeNest mmWave real-data reconstruction pipeline converts
continuous canonical phase and radar timestamps into a deterministic time-domain
contract suitable for downstream label mapping and dataset construction.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
import math
import re
import statistics
from typing import Any, Iterable, Sequence

import numpy as np


PROFILE_ID = "MMWAVE_TIMELINE_PROFILE_001"


class TimelineError(ValueError):
    """Raised when timeline reconstruction or validation fails."""


@dataclass(frozen=True)
class TimelineProfile:
    """Deterministic configuration for timeline reconstruction and windowing."""

    profile_id: str = PROFILE_ID
    target_sampling_rate_hz: float = 10.0
    expected_dt_seconds: float = 0.1
    native_timeline_preferred: bool = True
    jitter_tolerance_seconds: float = 0.005
    normal_max_dt_seconds: float = 0.105
    small_gap_max_seconds: float = 0.5
    large_gap_min_seconds: float = 0.5
    resampling_enabled_when_required: bool = True
    resampling_method: str = "LINEAR_INTERPOLATION"
    extrapolation_allowed: bool = False
    large_gap_interpolation_allowed: bool = False
    window_duration_seconds: float = 30.0
    window_samples: int = 300
    window_stride_samples: int = 300
    window_overlap_samples: int = 0
    boundary_convention: str = "[start,end)"
    incomplete_tail_policy: str = "DROP_INCOMPLETE_TAIL"
    label_independent: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "target_sampling_rate_hz": self.target_sampling_rate_hz,
            "expected_dt_seconds": self.expected_dt_seconds,
            "native_timeline_preferred": self.native_timeline_preferred,
            "jitter_policy": {
                "tolerance_seconds": self.jitter_tolerance_seconds,
                "normal_max_dt_seconds": self.normal_max_dt_seconds,
            },
            "gap_policy": {
                "small_gap_max_seconds": self.small_gap_max_seconds,
                "large_gap_min_seconds": self.large_gap_min_seconds,
            },
            "resampling": {
                "enabled_when_required": self.resampling_enabled_when_required,
                "method": self.resampling_method,
                "extrapolation_allowed": self.extrapolation_allowed,
                "large_gap_interpolation_allowed": self.large_gap_interpolation_allowed,
            },
            "window": {
                "duration_seconds": self.window_duration_seconds,
                "samples": self.window_samples,
                "stride_samples": self.window_stride_samples,
                "overlap_samples": self.window_overlap_samples,
                "boundary_convention": self.boundary_convention,
                "incomplete_tail_policy": self.incomplete_tail_policy,
            },
            "label_independent": self.label_independent,
        }


def format_canonical_iso(first_dt: dt.datetime, relative_seconds: float) -> str:
    """Format a relative offset from first_dt as an exact ISO-8601 string."""
    target_dt = first_dt + dt.timedelta(seconds=relative_seconds)
    iso_str = target_dt.isoformat()
    if not iso_str.endswith("Z") and "+" not in iso_str[-6:] and "-" not in iso_str[-6:]:
        iso_str += "Z"
    return iso_str


def parse_timestamps_to_seconds(
    timestamps_raw: bytes | str | Sequence[str],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Parse ISO-8601 timestamps and convert to relative seconds t_i from t_0 = 0.0."""
    if isinstance(timestamps_raw, bytes):
        try:
            text = timestamps_raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TimelineError(f"timestamps text is not UTF-8: {exc}") from exc
        lines = [line.strip() for line in text.splitlines() if line.strip()]
    elif isinstance(timestamps_raw, str):
        lines = [line.strip() for line in timestamps_raw.splitlines() if line.strip()]
    else:
        lines = [str(line).strip() for line in timestamps_raw if str(line).strip()]

    if not lines:
        raise TimelineError("timestamp sequence is empty")
    if len(lines) < 2:
        raise TimelineError("at least two timestamps are required to reconstruct a timeline")

    parsed_tuples: list[tuple[dt.datetime, int]] = []
    failures: list[dict[str, Any]] = []

    for line_number, value in enumerate(lines, 1):
        match = re.fullmatch(
            r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})?",
            value,
        )
        try:
            if match is None:
                raise ValueError("not an ISO-8601 timestamp with at most nanosecond precision")
            base, fraction, timezone = match.groups()
            fraction = (fraction or "").ljust(9, "0")
            microseconds = fraction[:6]
            submicro_nanoseconds = int(fraction[6:] or "0")
            timezone = "+00:00" if timezone == "Z" else (timezone or "")
            parsed_dt = dt.datetime.fromisoformat(
                base + (f".{microseconds}" if microseconds else "") + timezone
            )
            parsed_tuples.append((parsed_dt, submicro_nanoseconds))
        except (ValueError, OverflowError) as exc:
            failures.append({"line": line_number, "value": value, "error": str(exc)})

    if failures:
        raise TimelineError(f"{len(failures)} timestamp rows failed ISO-8601 parsing")

    # Calculate exact relative seconds from t_0 = 0.0
    first_dt, first_ns = parsed_tuples[0]
    seconds = np.zeros(len(parsed_tuples), dtype=np.float64)

    for idx in range(1, len(parsed_tuples)):
        curr_dt, curr_ns = parsed_tuples[idx]
        delta_sec = (curr_dt - first_dt).total_seconds() + (curr_ns - first_ns) / 1_000_000_000.0
        seconds[idx] = delta_sec

    metadata = {
        "timestamp_count": len(lines),
        "first_timestamp": lines[0],
        "last_timestamp": lines[-1],
        "first_datetime": first_dt,
        "timestamp_format": "ISO8601_HEADERLESS_UTF8",
        "duration_seconds": float(seconds[-1] - seconds[0]),
    }

    return seconds, lines, metadata


def analyze_timeline(
    timestamps_sec: np.ndarray, profile: TimelineProfile
) -> dict[str, Any]:
    """Perform deterministic timing, jitter, and gap analysis on relative timestamps."""
    sec = np.asarray(timestamps_sec, dtype=np.float64)
    if sec.ndim != 1 or sec.size < 2 or not np.all(np.isfinite(sec)):
        raise TimelineError("timestamps_sec must be a finite 1D float array with len >= 2")

    deltas = np.diff(sec)
    duplicate_count = int(np.count_nonzero(deltas == 0.0))
    backward_count = int(np.count_nonzero(deltas < 0.0))
    non_monotonic_count = duplicate_count + backward_count

    median_dt = float(np.median(deltas))
    mean_dt = float(np.mean(deltas))
    min_dt = float(np.min(deltas))
    max_dt = float(np.max(deltas))

    empirical_rate = float(1.0 / median_dt) if median_dt > 0 else 0.0

    # Jitter calculation relative to expected_dt
    timing_errors = np.abs(deltas - profile.expected_dt_seconds)
    median_abs_jitter = float(np.median(timing_errors))
    max_abs_jitter = float(np.max(timing_errors))

    if timing_errors.size > 0:
        pcts = np.percentile(timing_errors, [50, 90, 95, 99]).tolist()
        percentiles = {
            "p50": float(pcts[0]),
            "p90": float(pcts[1]),
            "p95": float(pcts[2]),
            "p99": float(pcts[3]),
        }
    else:
        percentiles = {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

    jitter_exceeded_count = int(np.count_nonzero(timing_errors > profile.jitter_tolerance_seconds + 1e-12))
    fraction_outside_tolerance = (
        float(jitter_exceeded_count / timing_errors.size) if timing_errors.size > 0 else 0.0
    )

    # Gap classification
    # dt <= normal_max_dt -> NORMAL
    # normal_max_dt < dt <= small_gap_max -> SMALL_GAP
    # dt > large_gap_min -> LARGE_GAP
    small_gaps = np.flatnonzero(
        (deltas > profile.normal_max_dt_seconds + 1e-12)
        & (deltas <= profile.small_gap_max_seconds + 1e-12)
    )
    large_gaps = np.flatnonzero(deltas > profile.large_gap_min_seconds + 1e-12)

    small_gap_count = int(small_gaps.size)
    large_gap_count = int(large_gaps.size)

    return {
        "timestamp_count": int(sec.size),
        "duration_seconds": float(sec[-1] - sec[0]),
        "median_dt_seconds": round(median_dt, 9),
        "mean_dt_seconds": round(mean_dt, 9),
        "min_dt_seconds": round(min_dt, 9),
        "max_dt_seconds": round(max_dt, 9),
        "empirical_sampling_rate_hz": round(empirical_rate, 6),
        "duplicate_timestamp_count": duplicate_count,
        "backward_timestamp_count": backward_count,
        "non_monotonic_count": non_monotonic_count,
        "median_abs_jitter_seconds": round(median_abs_jitter, 9),
        "max_abs_jitter_seconds": round(max_abs_jitter, 9),
        "jitter_percentiles_seconds": {
            k: round(v, 9) for k, v in percentiles.items()
        },
        "jitter_exceeded_count": jitter_exceeded_count,
        "fraction_outside_jitter_tolerance": round(fraction_outside_tolerance, 6),
        "small_gap_count": small_gap_count,
        "large_gap_count": large_gap_count,
        "small_gap_indices": small_gaps.tolist(),
        "large_gap_indices": large_gaps.tolist(),
    }


def evaluate_resampling_decision(
    analysis: dict[str, Any], profile: TimelineProfile
) -> dict[str, Any]:
    """Determine whether explicit timeline resampling is required or permissible."""
    dup = analysis["duplicate_timestamp_count"]
    back = analysis["backward_timestamp_count"]
    max_jitter = analysis["max_abs_jitter_seconds"]
    small_gaps = analysis["small_gap_count"]
    large_gaps = analysis["large_gap_count"]

    if dup > 0 or back > 0:
        return {
            "resampling_required": True,
            "resampling_permissible": False,
            "decision_code": "RECORDING_NOT_SAFELY_RESAMPLEABLE",
            "reason": f"Non-monotonic timestamps detected (duplicate={dup}, backward={back})",
        }

    if large_gaps > 0:
        return {
            "resampling_required": False,
            "resampling_permissible": False,
            "decision_code": "LARGE_GAP_PRESENT_NO_RESAMPLING",
            "reason": f"Large gap(s) present ({large_gaps}); timeline cannot be continuously resampled",
        }

    if max_jitter <= profile.jitter_tolerance_seconds and small_gaps == 0:
        return {
            "resampling_required": False,
            "resampling_permissible": True,
            "decision_code": "NATIVE_10HZ_NO_RESAMPLING",
            "reason": "Native timeline is exact 10 Hz within jitter tolerance",
        }

    if profile.resampling_enabled_when_required:
        return {
            "resampling_required": True,
            "resampling_permissible": True,
            "decision_code": "RESAMPLING_PERFORMED",
            "reason": "Resampling required and performed due to minor jitter or small gaps",
        }

    return {
        "resampling_required": True,
        "resampling_permissible": False,
        "decision_code": "RESAMPLING_REQUIRED_BUT_DISABLED",
        "reason": "Resampling is required by timing variation but disabled in profile",
    }


def resample_timeline(
    phase: np.ndarray,
    timestamps_sec: np.ndarray,
    first_dt: dt.datetime,
    profile: TimelineProfile,
    analysis: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Resample canonical unwrapped phase onto a regular 10 Hz grid if required."""
    decision = evaluate_resampling_decision(analysis, profile)

    if not decision["resampling_required"] or not decision["resampling_permissible"]:
        native_iso = [format_canonical_iso(first_dt, s) for s in timestamps_sec]
        return (
            phase,
            timestamps_sec,
            native_iso,
            {
                **decision,
                "resampling_performed": False,
                "native_sample_count": len(phase),
                "canonical_sample_count": len(phase),
                "interpolated_sample_count": 0,
            },
        )

    # Build exact regular 10 Hz target grid: integer steps * dt_target
    dt_target = 1.0 / profile.target_sampling_rate_hz  # 0.1
    duration = float(timestamps_sec[-1] - timestamps_sec[0])
    num_steps = int(math.floor(duration / dt_target))
    grid_sec = np.arange(0, num_steps + 1, dtype=np.float64) * dt_target

    resampled_phase = np.interp(grid_sec, timestamps_sec, phase)
    canonical_iso = [format_canonical_iso(first_dt, s) for s in grid_sec]

    # Identify interpolated vs native samples
    interpolated_mask = np.zeros(grid_sec.shape, dtype=bool)
    for i, t in enumerate(grid_sec):
        min_dist = float(np.min(np.abs(timestamps_sec - t)))
        if min_dist > profile.jitter_tolerance_seconds:
            interpolated_mask[i] = True

    interpolated_count = int(np.count_nonzero(interpolated_mask))

    return (
        resampled_phase,
        grid_sec,
        canonical_iso,
        {
            **decision,
            "resampling_performed": True,
            "native_sample_count": len(phase),
            "canonical_sample_count": len(grid_sec),
            "interpolated_sample_count": interpolated_count,
            "interpolated_mask": interpolated_mask.tolist(),
            "native_timestamps_sec": timestamps_sec.tolist(),
        },
    )


def generate_30s_windows(
    phase: np.ndarray,
    timestamps_sec: np.ndarray,
    canonical_timestamps_iso: list[str],
    first_dt: dt.datetime,
    recording_id: str,
    subject_id: str,
    profile: TimelineProfile,
    extraction_profile_id: str,
    analysis: dict[str, Any],
    resampling_meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Generate 30-second deterministic windows (300 samples @ 10 Hz)."""
    n_samples = len(phase)
    win_len = profile.window_samples  # 300
    stride = profile.window_stride_samples  # 300

    if n_samples < win_len:
        dropped_tail = n_samples
        return [], dropped_tail

    num_windows = n_samples // stride
    dropped_tail = n_samples - (num_windows * win_len)

    windows: list[dict[str, Any]] = []

    resampled_performed = resampling_meta.get("resampling_performed", False)
    interpolated_mask = resampling_meta.get("interpolated_mask", None)
    native_timestamps_sec = resampling_meta.get("native_timestamps_sec", None)

    for w_idx in range(num_windows):
        start_idx = w_idx * stride
        end_idx_exclusive = start_idx + win_len
        last_sample_idx = end_idx_exclusive - 1

        window_id = f"{recording_id}__W{w_idx:04d}"

        w_start_ts = canonical_timestamps_iso[start_idx]
        w_last_sample_ts = canonical_timestamps_iso[last_sample_idx]

        # Calculate exact exclusive end timestamp: t_start + 30.0 s
        start_sec = float(timestamps_sec[start_idx])
        end_exclusive_sec = start_sec + profile.window_duration_seconds
        w_end_exclusive_ts = format_canonical_iso(first_dt, end_exclusive_sec)

        # Source native index mapping
        if resampled_performed and native_timestamps_sec is not None:
            native_sec = np.array(native_timestamps_sec, dtype=np.float64)
            src_start = int(np.searchsorted(native_sec, start_sec - 1e-9, side="left"))
            src_end_excl = int(np.searchsorted(native_sec, end_exclusive_sec - 1e-9, side="left"))
        else:
            src_start = start_idx
            src_end_excl = end_idx_exclusive

        # Count interpolated samples in window
        if resampled_performed and interpolated_mask is not None:
            w_interp_count = int(sum(interpolated_mask[start_idx:end_idx_exclusive]))
        else:
            w_interp_count = 0

        # Check for large gaps crossing window
        large_gap_indices = analysis.get("large_gap_indices", [])
        w_large_gaps = [
            idx for idx in large_gap_indices if start_idx <= idx < end_idx_exclusive
        ]
        large_gap_in_window = len(w_large_gaps) > 0

        timeline_valid = True
        quality_flags: list[str] = []

        if not resampled_performed and analysis["max_abs_jitter_seconds"] <= profile.jitter_tolerance_seconds:
            quality_flags.append("TIMELINE_EXACT_NATIVE_10HZ")
        elif resampled_performed:
            quality_flags.append("TIMELINE_RESAMPLED")

        if w_interp_count > 0:
            quality_flags.append("WINDOW_CONTAINS_INTERPOLATION")

        if large_gap_in_window:
            quality_flags.append("LARGE_GAP_PRESENT")
            timeline_valid = False

        if analysis["non_monotonic_count"] > 0:
            quality_flags.append("NON_MONOTONIC_TIMESTAMP")
            timeline_valid = False

        window_entry = {
            "window_id": window_id,
            "recording_id": recording_id,
            "subject_id": subject_id,
            "timeline_profile": profile.profile_id,
            "phase_profile": extraction_profile_id,
            "window_index": w_idx,
            "source_start_index": src_start,
            "source_end_index_exclusive": src_end_excl,
            "canonical_start_index": start_idx,
            "canonical_end_index_exclusive": end_idx_exclusive,
            "start_timestamp": w_start_ts,
            "last_sample_timestamp": w_last_sample_ts,
            "end_timestamp_exclusive": w_end_exclusive_ts,
            "sample_count": win_len,
            "duration_seconds": float(profile.window_duration_seconds),
            "interpolated_sample_count": w_interp_count,
            "large_gap_count": len(w_large_gaps),
            "timeline_valid": timeline_valid,
            "quality_flags": sorted(list(set(quality_flags))),
        }
        windows.append(window_entry)

    return windows, dropped_tail


def process_recording_timeline(
    phase: np.ndarray,
    timestamps_raw: bytes | str | Sequence[str],
    recording_id: str,
    subject_id: str,
    extraction_profile_id: str,
    profile: TimelineProfile = TimelineProfile(),
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Process a single recording phase series into validated timeline results and windows."""
    timestamps_sec, timestamps_iso, ts_meta = parse_timestamps_to_seconds(timestamps_raw)
    first_dt = ts_meta["first_datetime"]

    if len(phase) != len(timestamps_sec):
        raise TimelineError(
            f"Phase length ({len(phase)}) does not match timestamp count ({len(timestamps_sec)})"
        )

    analysis = analyze_timeline(timestamps_sec, profile)
    resample_phase, resample_sec, canonical_iso, resample_meta = resample_timeline(
        phase, timestamps_sec, first_dt, profile, analysis
    )

    windows, dropped_tail_samples = generate_30s_windows(
        phase=resample_phase,
        timestamps_sec=resample_sec,
        canonical_timestamps_iso=canonical_iso,
        first_dt=first_dt,
        recording_id=recording_id,
        subject_id=subject_id,
        profile=profile,
        extraction_profile_id=extraction_profile_id,
        analysis=analysis,
        resampling_meta=resample_meta,
    )

    quality_flags: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    exceptions: list[dict[str, Any]] = []

    if analysis["max_abs_jitter_seconds"] <= profile.jitter_tolerance_seconds and analysis["non_monotonic_count"] == 0:
        quality_flags.append("TIMELINE_EXACT_NATIVE_10HZ")
    elif analysis["max_abs_jitter_seconds"] <= profile.jitter_tolerance_seconds * 2.0:
        quality_flags.append("TIMELINE_JITTER_WITHIN_TOLERANCE")

    if resample_meta.get("resampling_performed", False):
        quality_flags.append("TIMELINE_RESAMPLED")

    if analysis["small_gap_count"] > 0:
        quality_flags.append("SMALL_GAP_INTERPOLATED")
        warnings.append("SMALL_GAP_PRESENT")

    if analysis["large_gap_count"] > 0:
        quality_flags.append("LARGE_GAP_PRESENT")
        warnings.append("LARGE_GAP_PRESENT")
        exceptions.append(
            {
                "recording_id": recording_id,
                "category": "LARGE_GAP",
                "severity": "WARNING",
                "message": f"Recording contains {analysis['large_gap_count']} large gap(s)",
            }
        )

    if analysis["duplicate_timestamp_count"] > 0:
        quality_flags.append("DUPLICATE_TIMESTAMP")
        warnings.append("DUPLICATE_TIMESTAMPS_PRESENT")

    if analysis["backward_timestamp_count"] > 0:
        quality_flags.append("NON_MONOTONIC_TIMESTAMP")
        errors.append("BACKWARD_TIMESTAMPS_PRESENT")
        exceptions.append(
            {
                "recording_id": recording_id,
                "category": "TIMESTAMP_PARSE",
                "severity": "ERROR",
                "message": f"Recording contains {analysis['backward_timestamp_count']} backward timestamp(s)",
            }
        )

    if dropped_tail_samples > 0:
        quality_flags.append("INCOMPLETE_TAIL_DROPPED")
        warnings.append("INCOMPLETE_TAIL_DROPPED")
        exceptions.append(
            {
                "recording_id": recording_id,
                "category": "INCOMPLETE_TAIL",
                "severity": "WARNING",
                "message": f"Dropped incomplete tail of {dropped_tail_samples} samples after {len(windows)} full 30s windows",
            }
        )

    quality_status = "SUCCESS"
    if errors:
        quality_status = "FAILURE"
    elif warnings:
        quality_status = "SUCCESS_WITH_WARNINGS"

    rec_result = {
        "recording_id": recording_id,
        "subject_id": subject_id,
        "source_phase_profile": extraction_profile_id,
        "timeline_profile": profile.profile_id,
        "source_sample_count": len(phase),
        "source_timestamp_count": len(timestamps_sec),
        "first_timestamp": ts_meta["first_timestamp"],
        "last_timestamp": ts_meta["last_timestamp"],
        "duration_seconds": ts_meta["duration_seconds"],
        "median_dt_seconds": analysis["median_dt_seconds"],
        "mean_dt_seconds": analysis["mean_dt_seconds"],
        "min_dt_seconds": analysis["min_dt_seconds"],
        "max_dt_seconds": analysis["max_dt_seconds"],
        "empirical_sampling_rate_hz": analysis["empirical_sampling_rate_hz"],
        "duplicate_timestamp_count": analysis["duplicate_timestamp_count"],
        "backward_timestamp_count": analysis["backward_timestamp_count"],
        "small_gap_count": analysis["small_gap_count"],
        "large_gap_count": analysis["large_gap_count"],
        "resampling_required": resample_meta["resampling_required"],
        "resampling_performed": resample_meta.get("resampling_performed", False),
        "native_sample_count": resample_meta.get("native_sample_count", len(phase)),
        "canonical_sample_count": resample_meta.get("canonical_sample_count", len(phase)),
        "interpolated_sample_count": resample_meta.get("interpolated_sample_count", 0),
        "window_count": len(windows),
        "dropped_tail_samples": dropped_tail_samples,
        "quality_status": quality_status,
        "quality_flags": sorted(list(set(quality_flags))),
        "warnings": warnings,
        "errors": errors,
    }

    return rec_result, windows, exceptions


__all__ = [
    "PROFILE_ID",
    "TimelineError",
    "TimelineProfile",
    "analyze_timeline",
    "evaluate_resampling_decision",
    "format_canonical_iso",
    "generate_30s_windows",
    "parse_timestamps_to_seconds",
    "process_recording_timeline",
    "resample_timeline",
]
