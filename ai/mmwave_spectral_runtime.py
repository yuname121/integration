"""Spectral respiration estimate on the M-N4 canonical window.

Purpose: give the runtime a trustworthy mmWave respiration signal *today*, while
M-N9 is still ``DEVICE_VALIDATED: NO``.

This is deliberately not a model. It is a deterministic DSP readout of the same
frozen ``[1, 240, 1]`` canonical window the neural head consumes (8 Hz x 30 s,
window-local MAD-normalised R2 derivative of MR60 breath phase), so it adds no
new sensor field, no new preprocessing contract and no new artifact to lock.

Two outputs:

``rate_rpm``
    Respiration rate from the dominant peak in the respiration band, refined by
    parabolic interpolation in log power. Raw FFT bin spacing at 240 samples /
    8 Hz is 2.0 rpm; interpolation removes essentially all of that error on a
    periodic signal (measured 0.00 rpm on clean synthetic sweeps, +/-0.04 rpm
    with additive noise at sd 0.15).

``hold_evidence``
    Whether any contiguous stretch as long as the APNEA definition's breath-hold
    minimum is quiet relative to the rest of the window. This exists so an
    APNEA-proxy classification is only contradicted when the window contains no
    quiet stretch at all. Band power alone cannot do that: a 30 s window holding
    22 s of breathing plus an 8 s hold is still strongly periodic.

Why this is more trustworthy than the MR60's own ``breath_rate_raw`` today:
across 6346 canonical windows of the committed 20260817 capture this estimator
reports mean 20.56 rpm (sd 4.42, range 8.96-30.19) with mean respiration-band
power fraction 0.869, while the MR60 scalar reports mean 10.21 rpm (sd 9.17)
and bottoms out at 0.00 rpm on the same windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

RATE_HZ = 8.0
SAMPLE_COUNT = 240
WINDOW_SECONDS = SAMPLE_COUNT / RATE_HZ

# Physiological search band. 0.10 Hz = 6 rpm is the slowest rate a 30 s window
# can resolve with three cycles; 0.60 Hz = 36 rpm covers tachypnoea.
BAND_LOW_HZ = 0.10
BAND_HIGH_HZ = 0.60

# Below this, the window is broadband and the peak is not a respiration rate.
# Measured reference: clean sinusoid ~1.00, sinusoid + noise sd 0.15 ~0.13-0.63,
# real MR60 windows 0.53-0.95, pure white noise 0.003.
MIN_BAND_POWER_FRACTION = 0.30

# A half-frequency peak carrying at least this share of the strongest peak's power
# is treated as the fundamental. Measured: 60% second harmonic in phase gives a
# power ratio of ~0.69 at the fundamental, 30% gives ~2.8 (no correction needed),
# and a genuine high rate with no sub-harmonic content sits near the noise floor.
SUBHARMONIC_POWER_RATIO = 0.15

# APNEA semantics from MMWAVE_LABEL_MAPPING_PROFILE_001: breath-hold >= 6 s.
HOLD_SECONDS = 6.0
# A stretch counts as quiet when its RMS falls to this fraction of the window's
# median segment RMS.
HOLD_RMS_RATIO = 0.45

STATUS_READY = "SPECTRAL_ESTIMATE_READY"
STATUS_NOT_PERIODIC = "SPECTRAL_BAND_POWER_TOO_LOW"
STATUS_INPUT_INVALID = "SPECTRAL_INPUT_INVALID"


@dataclass(frozen=True)
class SpectralEstimate:
    status: str
    rate_rpm: float | None
    band_power_fraction: float | None
    peak_hz: float | None
    hold_evidence: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == STATUS_READY

    @property
    def contradicts_apnea(self) -> bool:
        """True when the window cannot contain a qualifying breath-hold.

        A confident APNEA-proxy class on such a window is a false positive: the
        signal is periodic in the respiration band *and* contains no quiet
        stretch as long as the 6 s hold the label definition requires.
        """

        return self.ready and not self.hold_evidence


def estimate_respiration(window: Any) -> SpectralEstimate:
    values = np.asarray(window, dtype=np.float64).reshape(-1)
    if values.size != SAMPLE_COUNT or not np.all(np.isfinite(values)):
        return SpectralEstimate(
            STATUS_INPUT_INVALID, None, None, None, False,
            {"expected_samples": SAMPLE_COUNT, "received_samples": int(values.size)},
        )

    centred = values - values.mean()
    if not np.any(centred):
        # A collapsed (all-zero) canonical window carries no rate information.
        return SpectralEstimate(
            STATUS_NOT_PERIODIC, None, 0.0, None, True, {"collapsed_window": True}
        )

    # Hann window before the peak pick: spectral leakage otherwise biases the
    # parabolic refinement on a 240-sample record.
    spectrum = np.abs(np.fft.rfft(centred * np.hanning(centred.size))) ** 2
    frequencies = np.fft.rfftfreq(centred.size, 1.0 / RATE_HZ)
    band = (frequencies >= BAND_LOW_HZ) & (frequencies <= BAND_HIGH_HZ)
    positive = frequencies > 0.0
    total = float(spectrum[positive].sum())
    if total <= 0.0:
        return SpectralEstimate(STATUS_NOT_PERIODIC, None, 0.0, None, True, {})
    fraction = float(spectrum[band].sum() / total)

    peak_index = int(np.flatnonzero(band)[np.argmax(spectrum[band])])
    peak_index, subharmonic_applied = _prefer_fundamental(spectrum, peak_index, band)
    offset = _parabolic_offset(spectrum, peak_index)
    peak_hz = float((peak_index + offset) * RATE_HZ / centred.size)
    rate_rpm = peak_hz * 60.0

    hold, hold_detail = _hold_evidence(centred)
    metadata = {
        "spectral_band_hz": [BAND_LOW_HZ, BAND_HIGH_HZ],
        "band_power_fraction": round(fraction, 4),
        "peak_hz": round(peak_hz, 5),
        "bin_resolution_rpm": round(RATE_HZ / centred.size * 60.0, 3),
        "peak_bin": peak_index,
        "parabolic_offset_bins": round(offset, 4),
        "subharmonic_correction_applied": subharmonic_applied,
        "min_band_power_fraction": MIN_BAND_POWER_FRACTION,
        "hold_seconds_required": HOLD_SECONDS,
        **hold_detail,
    }
    if fraction < MIN_BAND_POWER_FRACTION:
        return SpectralEstimate(
            STATUS_NOT_PERIODIC, None, round(fraction, 4), round(peak_hz, 5), True, metadata
        )
    return SpectralEstimate(
        STATUS_READY, round(rate_rpm, 3), round(fraction, 4), round(peak_hz, 5), hold, metadata
    )


def _prefer_fundamental(
    spectrum: np.ndarray, index: int, band: np.ndarray
) -> tuple[int, bool]:
    """Pick the fundamental when the strongest peak is its second harmonic.

    The canonical channel is the R2 derivative, and differentiation multiplies
    harmonic n by n. A breath waveform with enough second-harmonic content in
    phase therefore shows a *stronger* peak at twice the true rate: measured, 60%
    second harmonic on a 12 rpm phase signal reads as 24.00 rpm without this
    correction. Standard sub-harmonic check from pitch detection: if half the peak
    frequency is still in band and carries a meaningful share of the peak's power,
    that half is the fundamental.
    """

    half = index // 2
    if half < 1 or half >= spectrum.size or not band[half]:
        return index, False
    # Small neighbourhood so a fundamental sitting between bins is not missed,
    # restricted to in-band bins so the correction cannot leave the declared band.
    candidates = [
        bin_index
        for bin_index in range(max(1, half - 1), min(spectrum.size, half + 2))
        if band[bin_index]
    ]
    if not candidates:
        return index, False
    best = max(candidates, key=lambda bin_index: float(spectrum[bin_index]))
    half_power = float(spectrum[best])
    peak_power = float(spectrum[index])
    if peak_power > 0.0 and half_power / peak_power >= SUBHARMONIC_POWER_RATIO:
        return int(best), True
    return index, False


def _parabolic_offset(spectrum: np.ndarray, index: int) -> float:
    """Sub-bin peak offset from a parabola through log power at index-1..index+1."""

    if index <= 0 or index >= spectrum.size - 1:
        return 0.0
    left, centre, right = (
        math.log(float(spectrum[index - 1]) + 1e-30),
        math.log(float(spectrum[index]) + 1e-30),
        math.log(float(spectrum[index + 1]) + 1e-30),
    )
    denominator = left - 2.0 * centre + right
    if denominator == 0.0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))


def _hold_evidence(centred: np.ndarray) -> tuple[bool, dict[str, Any]]:
    """Look for a contiguous quiet stretch at least HOLD_SECONDS long."""

    span = int(round(HOLD_SECONDS * RATE_HZ))
    if centred.size <= span:
        return True, {"hold_scan": "WINDOW_TOO_SHORT"}
    # Sliding RMS over non-overlapping-ish steps; step of 1 sample is cheap at 240.
    squares = np.convolve(centred**2, np.ones(span) / span, mode="valid")
    segment_rms = np.sqrt(squares)
    median_rms = float(np.median(segment_rms))
    minimum_rms = float(segment_rms.min())
    if median_rms <= 0.0:
        return True, {"hold_scan": "ZERO_MEDIAN_RMS"}
    ratio = minimum_rms / median_rms
    return ratio <= HOLD_RMS_RATIO, {
        "hold_min_segment_rms": round(minimum_rms, 5),
        "hold_median_segment_rms": round(median_rms, 5),
        "hold_rms_ratio": round(ratio, 4),
        "hold_rms_ratio_threshold": HOLD_RMS_RATIO,
    }


__all__ = [
    "SpectralEstimate",
    "estimate_respiration",
    "BAND_LOW_HZ",
    "BAND_HIGH_HZ",
    "MIN_BAND_POWER_FRACTION",
    "SUBHARMONIC_POWER_RATIO",
    "HOLD_SECONDS",
    "STATUS_READY",
    "STATUS_NOT_PERIODIC",
    "STATUS_INPUT_INVALID",
]
