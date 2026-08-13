#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_verify_reproducibility.py
Unit tests for Priority 4 Reproducibility Verification script & functions.
"""

import os
import sys
import json
import unittest
import numpy as np
from pathlib import Path

# Add canonical repository root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.verify_reproducibility import get_environment_fingerprint


class TestVerifyReproducibility(unittest.TestCase):

    def test_01_seed_permutation_determinism(self):
        """Verify same seed produces same permutation, different seed produces different permutation."""
        rng1 = np.random.default_rng(42)
        perm1_a = rng1.permutation(100)

        rng2 = np.random.default_rng(42)
        perm1_b = rng2.permutation(100)

        rng3 = np.random.default_rng(999)
        perm2 = rng3.permutation(100)

        np.testing.assert_array_equal(perm1_a, perm1_b, "Same seed must produce identical permutation")
        self.assertFalse(np.array_equal(perm1_a, perm2), "Different seed must produce different permutation")

    def test_02_environment_fingerprint_keys(self):
        """Verify get_environment_fingerprint returns required keys."""
        env = get_environment_fingerprint()
        required_keys = [
            "python_version",
            "executable",
            "platform",
            "machine",
            "tensorflow_version",
            "numpy_version",
            "pyyaml_version",
            "visible_cpus",
            "deterministic_env_vars",
        ]
        for key in required_keys:
            self.assertIn(key, env)

    def test_03_report_file_structure(self):
        """Verify benchmarks/mmwave_reproducibility_report.json exists and adheres to schema."""
        report_path = project_root / "benchmarks/mmwave_reproducibility_report.json"
        self.assertTrue(report_path.exists(), f"Report file non-existent at {report_path}")

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertEqual(report["audit_name"], "mmwave_mac_reproducibility_verification")
        self.assertEqual(report["scope"], "MACOS_CPU_ONLY")
        self.assertEqual(report["overall_status"], "PASSED")
        self.assertTrue(report["exact_binary_target"])
        self.assertTrue(report["comparisons"]["resolved_config_match"])
        self.assertTrue(report["comparisons"]["canonical_weight_sha_match"])
        self.assertTrue(report["comparisons"]["float_tflite_sha_match"])
        self.assertTrue(report["comparisons"]["int8_tflite_sha_match"])
        self.assertEqual(report["limitations"]["cross_platform_reproducibility"], "NOT_VERIFIABLE")
        self.assertEqual(report["limitations"]["raspberry_pi_validation"], "BLOCKED_HARDWARE")
        self.assertEqual(report["limitations"]["real_sensor_performance"], "NOT_VERIFIABLE")


if __name__ == "__main__":
    unittest.main()
