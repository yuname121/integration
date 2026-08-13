#!/usr/bin/env python3
"""Deterministic, label-independent mmWave target and phase extraction.

The canonical output is the unfiltered, unnormalised unwrapped phase from one
stored range bin and one anonymous virtual channel.  Detrending and spectra in
this module are diagnostics only and never replace the canonical signal.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable

import numpy as np


PROFILE_ID = "MMWAVE_PHASE_EXTRACTION_PROFILE_001"
RESPIRATION_BAND_HZ = (0.1, 0.5)
TOTAL_DIAGNOSTIC_BAND_HZ = (0.05, 2.0)
TIE_TOLERANCE = 1e-12


class PhaseExtractionError(ValueError):
    """Raised when the structural A1 contract cannot be safely extracted."""


@dataclass(frozen=True)
class SearchRegion:
    profile_id: str
    minimum_range_m: float
    maximum_range_m: float

    def eligible_indices(self, rbins: np.ndarray) -> np.ndarray:
        bins = np.asarray(rbins)
        if bins.ndim != 1 or bins.size == 0 or not np.all(np.isfinite(bins)):
            raise PhaseExtractionError("stored rBins must be a finite non-empty vector")
        indices = np.flatnonzero(
            (bins >= self.minimum_range_m) & (bins <= self.maximum_range_m)
        )
        if indices.size == 0:
            raise PhaseExtractionError("search region contains no stored rBins")
        return indices


def _validate_inputs(rffts: np.ndarray, rbins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(rffts)
    bins = np.asarray(rbins)
    if values.ndim != 3 or not np.issubdtype(values.dtype, np.complexfloating):
        raise PhaseExtractionError("rffts must be complex[frame, virtual_channel, range]")
    if bins.ndim != 1 or bins.size != values.shape[2]:
        raise PhaseExtractionError("stored rBins length must equal the range dimension")
    if values.shape[0] < 2 or values.shape[1] < 1:
        raise PhaseExtractionError("rffts require at least two frames and one channel")
    return values, bins


def deterministic_argmax(
    candidates: Iterable[tuple[float, int, int]], *, tolerance: float = TIE_TOLERANCE
) -> tuple[float, int, int]:
    """Maximise score, resolving ties by lowest bin then lowest channel."""
    rows = [(float(score), int(bin_index), int(channel)) for score, bin_index, channel in candidates]
    if not rows:
        raise PhaseExtractionError("no selection candidates")
    finite = [row for row in rows if math.isfinite(row[0])]
    if not finite:
        raise PhaseExtractionError("all candidate scores are non-finite")
    best_score = max(row[0] for row in finite)
    tied = [row for row in finite if abs(row[0] - best_score) <= tolerance]
    return min(tied, key=lambda row: (row[1], row[2]))


def _linear_detrend(values: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or y.size < 2 or not np.all(np.isfinite(y)):
        return np.full(y.shape, np.nan, dtype=np.float64)
    x = np.arange(y.size, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    return y - (slope * x + intercept)


def diagnostic_periodogram(
    phase: np.ndarray,
    sampling_rate_hz: float,
    respiration_band_hz: tuple[float, float] = RESPIRATION_BAND_HZ,
    total_band_hz: tuple[float, float] = TOTAL_DIAGNOSTIC_BAND_HZ,
) -> dict[str, float | None]:
    """Hann periodogram of a temporary linearly detrended phase copy."""
    if not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise PhaseExtractionError("sampling_rate_hz must be positive")
    y = _linear_detrend(np.asarray(phase, dtype=np.float64))
    if y.size < 4 or not np.all(np.isfinite(y)):
        return {
            "dominant_frequency_hz": None,
            "respiration_band_energy": None,
            "total_spectral_energy": None,
            "respiration_band_fraction": None,
        }
    windowed = y * np.hanning(y.size)
    frequencies = np.fft.rfftfreq(y.size, d=1.0 / sampling_rate_hz)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    total_mask = (frequencies >= total_band_hz[0]) & (frequencies <= total_band_hz[1])
    respiration_mask = (frequencies >= respiration_band_hz[0]) & (
        frequencies <= respiration_band_hz[1]
    )
    total = float(np.sum(power[total_mask]))
    respiration = float(np.sum(power[respiration_mask]))
    dominant = None
    if np.any(total_mask) and total > 0:
        eligible = np.flatnonzero(total_mask)
        dominant = float(frequencies[eligible[int(np.argmax(power[eligible]))]])
    return {
        "dominant_frequency_hz": dominant,
        "respiration_band_energy": respiration,
        "total_spectral_energy": total,
        "respiration_band_fraction": float(respiration / total) if total > 0 else 0.0,
    }


def _safe_abs_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = _linear_detrend(left)
    b = _linear_detrend(right)
    if a.size != b.size or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    if float(np.std(a)) <= 0 or float(np.std(b)) <= 0:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def phase_statistics(complex_signal: np.ndarray, sampling_rate_hz: float) -> dict[str, Any]:
    z = np.asarray(complex_signal)
    magnitude = np.abs(z)
    finite_complex = np.isfinite(z.real) & np.isfinite(z.imag)
    positive = magnitude[(magnitude > 0) & np.isfinite(magnitude)]
    reference = float(np.median(positive)) if positive.size else 0.0
    near_zero_threshold = max(np.finfo(np.float64).tiny, reference * 1e-6)
    near_zero = finite_complex & (magnitude <= near_zero_threshold)
    wrapped = np.angle(z)
    unwrapped = np.unwrap(wrapped)
    wrapped_steps = np.diff(wrapped)
    unwrapped_steps = np.diff(unwrapped)
    unwrap_corrections = int(np.count_nonzero(np.abs(wrapped_steps) > np.pi))
    finite_unwrapped_steps = np.abs(unwrapped_steps[np.isfinite(unwrapped_steps)])
    percentiles = (
        np.percentile(finite_unwrapped_steps, [50, 90, 95, 99]).tolist()
        if finite_unwrapped_steps.size
        else [None, None, None, None]
    )
    median_mag = float(np.median(magnitude[np.isfinite(magnitude)])) if np.any(np.isfinite(magnitude)) else None
    mad = (
        float(np.median(np.abs(magnitude[np.isfinite(magnitude)] - median_mag)))
        if median_mag is not None
        else 0.0
    )
    magnitude_outliers = int(
        np.count_nonzero(np.abs(magnitude - median_mag) > 6.0 * mad)
    ) if median_mag is not None and mad > 0 else 0
    return {
        "magnitude": magnitude,
        "wrapped_phase": wrapped,
        "unwrapped_phase": unwrapped,
        "near_zero_threshold": near_zero_threshold,
        "near_zero_magnitude_count": int(np.count_nonzero(near_zero)),
        "near_zero_magnitude_ratio": float(np.mean(near_zero)),
        "nonfinite_complex_count": int(np.count_nonzero(~finite_complex)),
        "nonfinite_phase_count": int(np.count_nonzero(~np.isfinite(unwrapped))),
        "unwrap_correction_count": unwrap_corrections,
        "wrapped_phase_jump_count": unwrap_corrections,
        "large_unwrapped_step_count": int(np.count_nonzero(finite_unwrapped_steps > 1.0)),
        "largest_wrapped_jump_rad": float(np.nanmax(np.abs(wrapped_steps))) if wrapped_steps.size else 0.0,
        "largest_unwrapped_step_rad": float(np.nanmax(finite_unwrapped_steps)) if finite_unwrapped_steps.size else 0.0,
        "phase_step_percentiles_rad": {
            "p50": percentiles[0], "p90": percentiles[1], "p95": percentiles[2], "p99": percentiles[3]
        },
        "magnitude_outlier_count": magnitude_outliers,
        "spectrum": diagnostic_periodogram(unwrapped, sampling_rate_hz),
    }


def _candidate_metrics(
    rffts: np.ndarray,
    bin_index: int,
    channel: int,
    sampling_rate_hz: float,
    eligible_set: set[int],
) -> dict[str, Any]:
    signal = rffts[:, channel, bin_index]
    stats = phase_statistics(signal, sampling_rate_hz)
    magnitude = stats["magnitude"]
    finite = np.isfinite(signal.real) & np.isfinite(signal.imag)
    if not np.all(finite):
        mean_magnitude = median_magnitude = dynamic_energy = float("-inf")
    else:
        mean_magnitude = float(np.mean(magnitude))
        median_magnitude = float(np.median(magnitude))
        dynamic = signal - np.mean(signal)
        dynamic_energy = float(np.mean(np.abs(dynamic) ** 2))
    neighbors = []
    for neighbor in (bin_index - 1, bin_index + 1):
        if neighbor in eligible_set:
            other = phase_statistics(rffts[:, channel, neighbor], sampling_rate_hz)["unwrapped_phase"]
            neighbors.append(_safe_abs_correlation(stats["unwrapped_phase"], other))
    other_channels = []
    for other_channel in range(rffts.shape[1]):
        if other_channel != channel:
            other = phase_statistics(rffts[:, other_channel, bin_index], sampling_rate_hz)["unwrapped_phase"]
            other_channels.append(_safe_abs_correlation(stats["unwrapped_phase"], other))
    return {
        "bin_index": int(bin_index),
        "channel": int(channel),
        "mean_magnitude": mean_magnitude,
        "median_magnitude": median_magnitude,
        "dynamic_energy": dynamic_energy,
        **stats["spectrum"],
        "phase_continuity": float(1.0 - stats["large_unwrapped_step_count"] / max(1, signal.size - 1)),
        "near_zero_magnitude_ratio": stats["near_zero_magnitude_ratio"],
        "neighbor_bin_agreement": float(np.mean(neighbors)) if neighbors else 0.0,
        "channel_agreement": float(np.median(other_channels)) if other_channels else 0.0,
        "unwrap_correction_count": stats["unwrap_correction_count"],
        "large_unwrapped_step_count": stats["large_unwrapped_step_count"],
        "nonfinite_complex_count": stats["nonfinite_complex_count"],
    }


def _rank_scores(rows: list[dict[str, Any]], keys: list[str]) -> list[float]:
    """Unitless average percentile ranks; higher metric values are preferred."""
    n = len(rows)
    output = np.zeros(n, dtype=np.float64)
    for key in keys:
        order = sorted(range(n), key=lambda i: (-float(rows[i][key]), rows[i]["bin_index"], rows[i]["channel"]))
        ranks = np.empty(n, dtype=np.float64)
        for rank, index in enumerate(order):
            ranks[index] = 1.0 if n == 1 else 1.0 - rank / (n - 1)
        output += ranks
    return (output / len(keys)).tolist()


def _aligned_phase_aggregate(
    rffts: np.ndarray,
    bin_index: int,
    channels: list[int],
    weights: np.ndarray | None,
) -> np.ndarray:
    phases = np.vstack([np.unwrap(np.angle(rffts[:, channel, bin_index])) for channel in channels])
    phases = phases - phases[:, :1]
    if weights is None:
        return np.median(phases, axis=0)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(channels),) or not np.all(np.isfinite(weights)) or np.sum(weights) <= 0:
        raise PhaseExtractionError("invalid aggregation weights")
    return np.average(phases, axis=0, weights=weights)


class MmwavePhaseExtractor:
    """Compare A--E/V1--V3 candidates and apply the selected A2 profile."""

    def __init__(self, search_region: SearchRegion, sampling_rate_hz: float = 10.0) -> None:
        self.search_region = search_region
        self.sampling_rate_hz = float(sampling_rate_hz)

    def analyze_candidates(self, rffts: np.ndarray, rbins: np.ndarray) -> dict[str, Any]:
        values, bins = _validate_inputs(rffts, rbins)
        eligible = self.search_region.eligible_indices(bins)
        eligible_set = set(map(int, eligible))
        metrics = [
            _candidate_metrics(values, int(bin_index), channel, self.sampling_rate_hz, eligible_set)
            for bin_index in eligible
            for channel in range(values.shape[1])
        ]
        for row, score in zip(metrics, _rank_scores(metrics, [
            "dynamic_energy", "phase_continuity", "neighbor_bin_agreement", "channel_agreement"
        ])):
            row["phase_quality_rank_score"] = score

        def best(metric: str) -> dict[str, Any]:
            _, bin_index, channel = deterministic_argmax(
                (
                    row[metric] if row[metric] is not None else float("-inf"),
                    row["bin_index"],
                    row["channel"],
                )
                for row in metrics
            )
            return next(row for row in metrics if row["bin_index"] == bin_index and row["channel"] == channel)

        def aggregated_winner(metric: str) -> tuple[dict[str, Any], float]:
            aggregate = []
            for bin_index in eligible:
                rows = [row for row in metrics if row["bin_index"] == int(bin_index)]
                values_for_bin = [row[metric] for row in rows if row[metric] is not None]
                score = float(np.median(values_for_bin)) if values_for_bin else float("-inf")
                aggregate.append((score, int(bin_index), 0))
            score, winning_bin, _ = deterministic_argmax(aggregate)
            rows = [row for row in metrics if row["bin_index"] == winning_bin]
            _, _, winning_channel = deterministic_argmax(
                (
                    row[metric] if row[metric] is not None else float("-inf"),
                    winning_bin,
                    row["channel"],
                )
                for row in rows
            )
            return next(row for row in rows if row["channel"] == winning_channel), score

        # Range B uses a robust channel aggregate; channel V1 is selected only
        # after the winning bin is fixed, so arbitrary physical channel ordering
        # is never assumed.
        aggregate_dynamic_winner, range_score = aggregated_winner("dynamic_energy")
        selected_bin = aggregate_dynamic_winner["bin_index"]
        selected_rows = [row for row in metrics if row["bin_index"] == selected_bin]
        channel_score, _, selected_channel = deterministic_argmax(
            (row["phase_quality_rank_score"], selected_bin, row["channel"]) for row in selected_rows
        )
        selected_metric = next(row for row in selected_rows if row["channel"] == selected_channel)

        strategies = []
        for strategy_id, metric in (
            ("A_MEAN_MAGNITUDE_PER_CHANNEL", "mean_magnitude"),
            ("B_DYNAMIC_ENERGY_PER_CHANNEL", "dynamic_energy"),
            ("C_RESPIRATION_BAND_ENERGY", "respiration_band_energy"),
            ("D_PHASE_QUALITY", "phase_quality_rank_score"),
            ("E_NEIGHBOR_BIN_SUPPORT", "neighbor_bin_agreement"),
        ):
            winner = best(metric)
            strategies.append({"strategy_id": strategy_id, "virtual_channel_strategy": "V1_SINGLE_BEST", **winner})
        for strategy_id, metric in (
            ("A_MEAN_MAGNITUDE_CHANNEL_AGGREGATED", "mean_magnitude"),
            ("B_DYNAMIC_ENERGY_CHANNEL_AGGREGATED", "dynamic_energy"),
        ):
            winner, aggregate_score = aggregated_winner(metric)
            strategies.append({
                "strategy_id": strategy_id,
                "virtual_channel_strategy": "V1_WITH_MEDIAN_CHANNEL_RANGE_SCORE",
                "channel_aggregated_selection_score": aggregate_score,
                **winner,
            })
        neighbor_winner = best("neighbor_bin_agreement")
        neighbor_bins = [
            index for index in (
                neighbor_winner["bin_index"] - 1,
                neighbor_winner["bin_index"],
                neighbor_winner["bin_index"] + 1,
            )
            if index in eligible_set
        ]
        neighbor_phases = np.vstack([
            np.unwrap(np.angle(values[:, neighbor_winner["channel"], index]))
            for index in neighbor_bins
        ])
        neighbor_phases -= neighbor_phases[:, :1]
        neighbor_phase = np.median(neighbor_phases, axis=0)
        strategies.append({
            "strategy_id": "E_NEIGHBOR_ALIGNED_MEDIAN_PHASE_DIAGNOSTIC",
            "virtual_channel_strategy": "V1_SINGLE_BEST",
            "neighbor_bin_policy": "PLUS_MINUS_ONE_INDIVIDUAL_PHASE_CENTER_THEN_MEDIAN",
            "selected_bin_index": neighbor_winner["bin_index"],
            "selected_range_m": float(bins[neighbor_winner["bin_index"]]),
            "selected_channels": [neighbor_winner["channel"]],
            "neighbor_bin_indices": neighbor_bins,
            "selection_score": neighbor_winner["neighbor_bin_agreement"],
            **diagnostic_periodogram(neighbor_phase, self.sampling_rate_hz),
        })

        quality_values = np.array([max(0.0, row["phase_quality_rank_score"]) for row in selected_rows])
        if float(np.sum(quality_values)) <= 0:
            quality_values = np.ones(values.shape[1], dtype=np.float64)
        v2 = _aligned_phase_aggregate(values, selected_bin, list(range(values.shape[1])), quality_values)
        v3 = _aligned_phase_aggregate(values, selected_bin, list(range(values.shape[1])), None)
        for strategy_id, phase in (("V2_QUALITY_WEIGHTED_PHASE", v2), ("V3_MEDIAN_CONSENSUS_PHASE", v3)):
            spectrum = diagnostic_periodogram(phase, self.sampling_rate_hz)
            strategies.append({
                "strategy_id": "B_DYNAMIC_ENERGY_CHANNEL_AGGREGATED",
                "virtual_channel_strategy": strategy_id,
                "selected_bin_index": selected_bin,
                "selected_range_m": float(bins[selected_bin]),
                "selected_channels": list(range(values.shape[1])),
                "selection_score": range_score,
                **spectrum,
            })
        return {
            "eligible_bin_indices": eligible.tolist(),
            "candidate_metrics": metrics,
            "strategy_results": strategies,
            "selected_bin_index": selected_bin,
            "selected_range_m": float(bins[selected_bin]),
            "selected_channel": selected_channel,
            "range_selection_score": range_score,
            "channel_selection_score": channel_score,
            "selected_metric": selected_metric,
        }

    def extract(
        self,
        *,
        rffts: np.ndarray,
        rbins: np.ndarray,
        timestamps: Any = None,
        config: Any = None,
        profile: str = PROFILE_ID,
        annotation: Any = None,
    ) -> dict[str, Any]:
        if profile != PROFILE_ID:
            raise PhaseExtractionError(f"unsupported extraction profile: {profile}")
        analysis = self.analyze_candidates(rffts, rbins)
        values, bins = _validate_inputs(rffts, rbins)
        bin_index = analysis["selected_bin_index"]
        channel = analysis["selected_channel"]
        signal = np.array(values[:, channel, bin_index], copy=True)
        stats = phase_statistics(signal, self.sampling_rate_hz)
        warnings = []
        errors = []
        if any(row["nonfinite_complex_count"] for row in analysis["candidate_metrics"]):
            warnings.append("NONFINITE_CANDIDATES_EXCLUDED_FROM_ENERGY_SELECTION")
        if stats["near_zero_magnitude_count"]:
            warnings.append("NEAR_ZERO_MAGNITUDE_SAMPLES_PRESERVED")
        if stats["nonfinite_complex_count"] or stats["nonfinite_phase_count"]:
            errors.append("NONFINITE_CANONICAL_SIGNAL")
        # annotation is deliberately not read by candidate analysis.  Keeping it
        # in the public API makes the non-leakage boundary testable.
        del timestamps, config, annotation
        return {
            "profile_id": PROFILE_ID,
            "complex_signal": signal,
            "magnitude": stats["magnitude"],
            "wrapped_phase": stats["wrapped_phase"],
            "unwrapped_phase": stats["unwrapped_phase"],
            "selection": {
                "selected_range_bin_index": bin_index,
                "selected_range_m": float(bins[bin_index]),
                "selected_virtual_channels": [channel],
                "virtual_channel_aggregation": "V1_SINGLE_BEST_PHASE_QUALITY_CHANNEL",
                "range_bin_selection_method": "B_MEDIAN_CHANNEL_DYNAMIC_ENERGY",
                "selection_score": analysis["range_selection_score"],
                "selection_score_components": analysis["selected_metric"],
                "label_independent": True,
            },
            "statistics": {key: value for key, value in stats.items() if key not in {"magnitude", "wrapped_phase", "unwrapped_phase"}},
            "candidate_analysis": analysis,
            "warnings": warnings,
            "errors": errors,
        }


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


__all__ = [
    "MmwavePhaseExtractor", "PROFILE_ID", "PhaseExtractionError", "SearchRegion",
    "array_sha256", "deterministic_argmax", "diagnostic_periodogram", "phase_statistics"
]
