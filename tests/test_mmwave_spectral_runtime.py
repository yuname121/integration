"""Spectral respiration readout on the M-N4 canonical window.

Synthetic signals with known ground truth plus the committed field capture.
No hardware. This is DSP, not a model, so it is fully deterministic.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np

from ai.mmwave_canonical_runtime import MR60CanonicalWindowBuilder
from ai.mmwave_spectral_runtime import (
    BAND_HIGH_HZ,
    BAND_LOW_HZ,
    HOLD_SECONDS,
    MIN_BAND_POWER_FRACTION,
    RATE_HZ,
    SAMPLE_COUNT,
    STATUS_INPUT_INVALID,
    STATUS_NOT_PERIODIC,
    STATUS_READY,
    estimate_respiration,
)

ROOT = Path(__file__).resolve().parent.parent
CAPTURE = ROOT / "data" / "mmwave" / "20260817_09_mmwave.jsonl"


def _canonical_module():
    path = ROOT / "sources/ondevice_ai/scripts/mmwave_m_n4_canonical.py"
    spec = importlib.util.spec_from_file_location("_test_m_n4", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M4 = _canonical_module()
TIME = np.arange(0.0, 31.0, 1.0 / RATE_HZ)
START = float(TIME[-1] - M4.WINDOW_SECONDS)


def window(signal):
    return M4.form_canonical_window(TIME, np.asarray(signal, dtype=np.float64), START).values


def phase(rpm: float):
    return 2.0 * np.pi * (rpm / 60.0) * TIME


class RateAccuracyTests(unittest.TestCase):
    def test_exact_on_clean_sinusoids(self):
        # Raw bin spacing is 2.0 rpm; parabolic refinement removes it.
        for rpm in (8, 10, 12, 14, 15, 16, 18, 20, 22, 24, 28, 30):
            with self.subTest(rpm=rpm):
                estimate = estimate_respiration(window(np.sin(phase(rpm))))
                self.assertEqual(estimate.status, STATUS_READY)
                self.assertAlmostEqual(estimate.rate_rpm, rpm, delta=0.2)

    def test_robust_to_additive_noise(self):
        rng = np.random.default_rng(20260821)
        for rpm in (15, 18, 20, 24):
            with self.subTest(rpm=rpm):
                signal = np.sin(phase(rpm)) + rng.normal(0, 0.15, TIME.size)
                estimate = estimate_respiration(window(signal))
                if estimate.ready:
                    self.assertAlmostEqual(estimate.rate_rpm, rpm, delta=1.0)

    def test_second_harmonic_does_not_double_the_reported_rate(self):
        """R2 is a derivative, so harmonic n is amplified by n."""

        for rpm in (10, 12, 15, 18):
            for weight in (0.6, 1.0):
                with self.subTest(rpm=rpm, harmonic_weight=weight):
                    signal = np.sin(phase(rpm)) + weight * np.sin(2 * phase(rpm))
                    estimate = estimate_respiration(window(signal))
                    self.assertEqual(estimate.status, STATUS_READY)
                    self.assertAlmostEqual(estimate.rate_rpm, rpm, delta=0.5)
                    self.assertTrue(estimate.metadata["subharmonic_correction_applied"])

    def test_asymmetric_waveform_keeps_the_fundamental(self):
        for rpm in (12, 15, 20):
            with self.subTest(rpm=rpm):
                signal = np.sign(np.sin(phase(rpm))) * np.abs(np.sin(phase(rpm))) ** 0.5
                estimate = estimate_respiration(window(signal))
                self.assertAlmostEqual(estimate.rate_rpm, rpm, delta=0.5)

    def test_reported_rate_always_stays_inside_the_declared_band(self):
        rng = np.random.default_rng(7)
        for rpm in (8, 10, 14, 20, 26, 32):
            for weight in (0.0, 0.6, 1.0):
                signal = np.sin(phase(rpm)) + weight * np.sin(2 * phase(rpm))
                signal = signal + rng.normal(0, 0.05, TIME.size)
                estimate = estimate_respiration(window(signal))
                if estimate.ready:
                    self.assertGreaterEqual(estimate.peak_hz, BAND_LOW_HZ - 1e-9)
                    self.assertLessEqual(estimate.peak_hz, BAND_HIGH_HZ + 1e-9)


class RejectionTests(unittest.TestCase):
    def test_white_noise_is_not_a_respiration_rate(self):
        signal = np.random.default_rng(1).normal(0, 1.0, TIME.size)
        estimate = estimate_respiration(window(signal))
        self.assertEqual(estimate.status, STATUS_NOT_PERIODIC)
        self.assertIsNone(estimate.rate_rpm)
        self.assertLess(estimate.band_power_fraction, MIN_BAND_POWER_FRACTION)
        self.assertFalse(estimate.contradicts_apnea)

    def test_flat_window_is_rejected_and_does_not_contradict_apnea(self):
        estimate = estimate_respiration(np.zeros(SAMPLE_COUNT))
        self.assertNotEqual(estimate.status, STATUS_READY)
        self.assertFalse(estimate.contradicts_apnea)

    def test_wrong_length_input_is_refused(self):
        self.assertEqual(estimate_respiration(np.zeros(120)).status, STATUS_INPUT_INVALID)
        bad = np.zeros(SAMPLE_COUNT)
        bad[3] = np.nan
        self.assertEqual(estimate_respiration(bad).status, STATUS_INPUT_INVALID)


class ApneaContradictionTests(unittest.TestCase):
    """The gate may only fire when a qualifying breath-hold is impossible."""

    def test_continuous_breathing_contradicts_apnea(self):
        estimate = estimate_respiration(window(np.sin(phase(20))))
        self.assertTrue(estimate.ready)
        self.assertFalse(estimate.hold_evidence)
        self.assertTrue(estimate.contradicts_apnea)

    def test_a_real_breath_hold_is_never_contradicted(self):
        base = np.sin(phase(20))
        for hold in (HOLD_SECONDS, 8.0, 10.0, 15.0):
            with self.subTest(hold_seconds=hold):
                signal = base.copy()
                mask = (TIME >= START + 5.0) & (TIME < START + 5.0 + hold)
                signal[mask] = signal[int(np.argmin(np.abs(TIME - (START + 5.0))))]
                estimate = estimate_respiration(window(signal))
                self.assertTrue(estimate.hold_evidence, estimate.metadata)
                self.assertFalse(estimate.contradicts_apnea)

    def test_unready_estimates_never_contradict(self):
        signal = np.random.default_rng(2).normal(0, 1.0, TIME.size)
        self.assertFalse(estimate_respiration(window(signal)).contradicts_apnea)


class FieldCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CAPTURE.is_file():
            raise unittest.SkipTest(f"field capture not present: {CAPTURE}")
        builder = MR60CanonicalWindowBuilder()
        cls.estimates = []
        with CAPTURE.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= 3000:
                    break
                record = json.loads(line)
                mmwave = record.get("mmwave") or {}
                builder.ingest(
                    {
                        "sequence": record.get("sequence"),
                        "boot_id": record.get("boot_id"),
                        "values": {
                            "breath_phase": mmwave.get("breath_phase"),
                            "ts_monotonic_ms": mmwave.get("ts_monotonic_ms"),
                            "phase_age_ms": mmwave.get("phase_age_ms"),
                        },
                    }
                )
                result = builder.latest()
                if result.status == "CANONICAL_WINDOW_READY":
                    cls.estimates.append(estimate_respiration(result.tensor))

    def test_the_capture_yields_usable_estimates(self):
        self.assertGreater(len(self.estimates), 100)
        ready = [item for item in self.estimates if item.ready]
        self.assertGreater(len(ready) / len(self.estimates), 0.9)

    def test_rates_are_physiologically_plausible(self):
        rates = [item.rate_rpm for item in self.estimates if item.ready]
        self.assertGreaterEqual(min(rates), 5.0)
        self.assertLessEqual(max(rates), 36.0)
        self.assertGreater(float(np.mean(rates)), 8.0)
        self.assertLess(float(np.mean(rates)), 28.0)

    def test_the_capture_is_strongly_periodic(self):
        """This is why a confident APNEA-proxy on it is a false positive."""

        fractions = [item.band_power_fraction for item in self.estimates if item.ready]
        self.assertGreater(float(np.mean(fractions)), 0.5)

    def test_more_stable_than_the_mr60_scalar(self):
        raw = []
        with CAPTURE.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= 3000:
                    break
                value = json.loads(line).get("respiration_rate_bpm")
                if isinstance(value, (int, float)):
                    raw.append(float(value))
        spectral = [item.rate_rpm for item in self.estimates if item.ready]
        # The MR60 scalar reaches 0.0 rpm on windows the spectrum reads as normal.
        self.assertEqual(min(raw), 0.0)
        self.assertGreater(min(spectral), 0.0)
        self.assertLess(float(np.std(spectral)), float(np.std(raw)))


if __name__ == "__main__":
    unittest.main()
