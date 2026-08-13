#!/usr/bin/env python3
"""Unit test suite for SafeNest mmWave M-B1 Preprocessing Ablation & Validator."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from scripts.mmwave_m_b1_preprocessing import (
    PROFILES,
    apply_bpf,
    apply_linear_detrend,
    compute_tensor_fingerprint,
    fit_train_zscore_statistics,
    transform_signals,
)
from scripts.mmwave_phase_b_access import LOCKED_TEST_AccessError, PhaseBAccessGuard
from scripts.validate_mmwave_m_b1 import MB1ValidationError, validate_m_b1_artifacts

ROOT_DIR = Path(__file__).resolve().parents[1]


class TestMMWaveMB1(unittest.TestCase):
    """Test suite for Phase M-B1 preprocessing factorial, access guard, and validator."""

    def setUp(self) -> None:
        self.guard = PhaseBAccessGuard(root_dir=ROOT_DIR)
        self.manifest_dir = ROOT_DIR / "datasets/mmwave/manifests/M-B1_preprocessing_ablation"

    def test_factorial_profiles_count_and_structure(self) -> None:
        self.assertEqual(len(PROFILES), 8)
        profile_ids = [p["profile_id"] for p in PROFILES]
        expected_ids = [
            "M-B1_D0_B0_Z0",
            "M-B1_D1_B0_Z0",
            "M-B1_D0_B1_Z0",
            "M-B1_D1_B1_Z0",
            "M-B1_D0_B0_Z1",
            "M-B1_D1_B0_Z1",
            "M-B1_D0_B1_Z1",
            "M-B1_D1_B1_Z1",
        ]
        self.assertEqual(profile_ids, expected_ids)

    def test_linear_detrend_semantics(self) -> None:
        t = np.linspace(0, 10, 300)
        linear_ramp = 2.5 * t + 1.0 + np.sin(2 * np.pi * 0.2 * t)
        detrended = apply_linear_detrend(linear_ramp[np.newaxis, :])[0]
        # Linear slope should be removed, leaving oscillating component
        self.assertLess(abs(np.polyfit(t, detrended, 1)[0]), 1e-4)

    def test_bpf_semantics(self) -> None:
        t = np.linspace(0, 30, 300)  # 10 Hz, 30 seconds
        dc_trend = np.ones_like(t) * 10.0  # 0 Hz
        resp = np.sin(2 * np.pi * 0.25 * t)  # 0.25 Hz (within 0.1-0.5 Hz passband)
        hf_noise = np.cos(2 * np.pi * 2.0 * t)  # 2.0 Hz (outside 0.5 Hz cutoff)

        sig = (dc_trend + resp + hf_noise)[np.newaxis, :]
        filtered = apply_bpf(sig, fs=10.0, lowcut=0.1, highcut=0.5, order=4)[0]

        # DC trend and HF noise should be strongly attenuated
        self.assertLess(abs(np.mean(filtered)), 0.1)
        # Respiration component should be preserved
        corr = np.corrcoef(resp[20:-20], filtered[20:-20])[0, 1]
        self.assertGreater(corr, 0.95)

    def test_train_fitted_zscore(self) -> None:
        train_signals = np.random.RandomState(42).randn(10, 300) * 5.0 + 12.0
        zstats = fit_train_zscore_statistics(train_signals, detrend=False, bpf=False)
        self.assertAlmostEqual(zstats["mean"], float(np.mean(train_signals)), places=4)
        self.assertAlmostEqual(zstats["std"], float(np.std(train_signals)), places=4)

        transformed = transform_signals(train_signals, detrend=False, bpf=False, zscore=True, zscore_stats=zstats)
        self.assertAlmostEqual(float(np.mean(transformed)), 0.0, places=4)
        self.assertAlmostEqual(float(np.std(transformed)), 1.0, places=4)

    def test_locked_test_model_selection_prohibited(self) -> None:
        with self.assertRaises(LOCKED_TEST_AccessError):
            self.guard.get_model_selection_dataset("LOCKED_TEST")

    def test_standalone_m_b1_validator_clean(self) -> None:
        if self.manifest_dir.is_dir():
            res = validate_m_b1_artifacts(root_dir=ROOT_DIR, manifest_dir=self.manifest_dir)
            self.assertTrue(res["validation_success"])
            self.assertEqual(res["m_b1_gate_status"], "PASS_WITH_WARNINGS")
            self.assertEqual(res["m_b2_entry_status"], "READY_WITH_CONDITIONS")
            self.assertTrue(res["independently_measured"]["zscore_statistics_verified"])

    def test_validator_detects_corrupted_winner_selection(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            sel_file = tmp_manifest / "selected_preprocessing_profile.json"
            data = json.loads(sel_file.read_text(encoding="utf-8"))
            data["selected_profile_id"] = "M-B1_D0_B0_Z0"  # Corrupt winner
            sel_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(MB1ValidationError):
                validate_m_b1_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_malformed_checksum_line(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            chk_file = tmp_manifest / "checksums.sha256"
            content = chk_file.read_text(encoding="utf-8")
            chk_file.write_text("malformed_line_without_space\n" + content, encoding="utf-8")

            with self.assertRaises(MB1ValidationError):
                validate_m_b1_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_unpinned_environment(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            env_file = tmp_manifest / "run_environment.json"
            data = json.loads(env_file.read_text(encoding="utf-8"))
            data["numpy_version"] = "2.0.2"  # Unpinned version
            env_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(MB1ValidationError):
                validate_m_b1_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_contradictory_reproducibility_verdict(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            repro_file = tmp_manifest / "reproducibility_comparison.json"
            data = json.loads(repro_file.read_text(encoding="utf-8"))
            data["winner_changed"] = True
            data["reproducibility_verdict"] = "VERIFIED_IDENTICAL (Contradiction!)"
            repro_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(MB1ValidationError):
                validate_m_b1_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)


if __name__ == "__main__":
    unittest.main()
