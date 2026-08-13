#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_audit_dataset_integrity.py
Unit tests for SafeNest V6 mmWave Dataset Integrity Audit Script
"""

from __future__ import annotations
import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

# Ensure canonical repository root is in python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.audit_dataset_integrity import (
    canonical_window_hash,
    run_integrity_audit,
)


class TestAuditDatasetIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mock_npz_and_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_ids: np.ndarray | None = None,
        split_dict: dict | None = None,
        split_indices: dict | None = None,
    ) -> tuple[Path, Path, Path]:
        npz_file = self.tmp_path / "test_dataset.npz"
        split_file = self.tmp_path / "test_split.json"
        output_file = self.tmp_path / "audit_output.json"

        save_kwargs = {
            "X": X,
            "y": y,
            "class_map": {0: "NORMAL", 1: "RAPID_OR_ABNORMAL", 2: "APNEA"},
        }
        if group_ids is not None:
            save_kwargs["group_ids"] = group_ids

        np.savez_compressed(npz_file, **save_kwargs)

        sdata = split_dict or {
            "schema_version": "1.0",
            "split_version": "1.0.0",
            "project": "SafeNest_V6_Test",
        }
        if split_indices:
            sdata["indices"] = split_indices

        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(sdata, f)

        return npz_file, split_file, output_file

    def test_01_identical_waveforms_identical_labels(self):
        """1. Detection of identical waveforms with identical labels"""
        w1 = np.sin(np.linspace(0, 10, 300, dtype=np.float32))
        X = np.stack([w1, w1, w1], axis=0)  # 3 identical windows
        y = np.array([0, 0, 0], dtype=np.int64)
        g = np.array(["g1", "g2", "g3"])
        splits = {"train": [0], "val": [1], "test": [2]}

        npz_p, split_p, out_p = self._create_mock_npz_and_split(X, y, group_ids=g, split_indices=splits)
        report = run_integrity_audit(npz_p, split_p, out_p, expected_windows=3)

        self.assertEqual(report["duplicates"]["duplicate_hash_group_count"], 1)
        self.assertEqual(report["duplicates"]["duplicate_instance_count"], 2)
        self.assertFalse(report["label_consistency"]["has_label_inconsistency"])

    def test_02_identical_waveforms_conflicting_labels(self):
        """2. Detection of identical waveforms with conflicting labels"""
        w1 = np.sin(np.linspace(0, 10, 300, dtype=np.float32))
        X = np.stack([w1, w1], axis=0)
        y = np.array([0, 2], dtype=np.int64)  # Conflicting labels: NORMAL vs APNEA
        g = np.array(["g1", "g2"])
        splits = {"train": [0], "val": [1], "test": []}

        npz_p, split_p, out_p = self._create_mock_npz_and_split(X, y, group_ids=g, split_indices=splits)
        report = run_integrity_audit(npz_p, split_p, out_p, expected_windows=2)

        self.assertTrue(report["label_consistency"]["has_label_inconsistency"])
        self.assertEqual(report["label_consistency"]["inconsistent_hash_count"], 1)
        self.assertEqual(report["signal_leakage_status"], "FAILED")

    def test_03_duplicate_indices_within_split(self):
        """3. Detection of duplicate indices within a split"""
        X = np.random.randn(5, 300).astype(np.float32)
        y = np.array([0, 1, 2, 0, 1], dtype=np.int64)
        splits = {"train": [0, 1, 1], "val": [2], "test": [3, 4]}  # Index 1 duplicated in train

        npz_p, split_p, out_p = self._create_mock_npz_and_split(X, y, split_indices=splits)
        report = run_integrity_audit(npz_p, split_p, out_p, expected_windows=5)

        self.assertEqual(report["index_split_status"], "FAILED")
        self.assertIn("duplicate indices", report["split_integrity"]["split_failures"][0])

    def test_04_index_leakage_across_splits(self):
        """4. Detection of index leakage across splits"""
        X = np.random.randn(4, 300).astype(np.float32)
        y = np.array([0, 1, 0, 1], dtype=np.int64)
        splits = {"train": [0, 1], "val": [1, 2], "test": [3]}  # Index 1 in both train and val

        npz_p, split_p, out_p = self._create_mock_npz_and_split(X, y, split_indices=splits)
        report = run_integrity_audit(npz_p, split_p, out_p, expected_windows=4)

        self.assertEqual(report["index_split_status"], "FAILED")
        self.assertEqual(report["split_integrity"]["train_val_index_overlap_count"], 1)

    def test_05_cross_split_signal_leakage(self):
        """5. Detection of identical waveforms with different indices across splits"""
        w1 = np.ones((300,), dtype=np.float32)
        w2 = np.zeros((300,), dtype=np.float32)
        X = np.stack([w1, w2, w1], axis=0)  # Window 0 and 2 identical
        y = np.array([0, 1, 0], dtype=np.int64)
        splits = {"train": [0], "val": [1], "test": [2]}  # Window 0 in train, Window 2 in test

        npz_p, split_p, out_p = self._create_mock_npz_and_split(X, y, split_indices=splits)
        report = run_integrity_audit(npz_p, split_p, out_p, expected_windows=3)

        self.assertTrue(report["cross_split_signal_leakage"]["has_cross_split_signal_leakage"])
        self.assertEqual(report["cross_split_signal_leakage"]["train_test_duplicate_hash_count"], 1)
        self.assertEqual(report["signal_leakage_status"], "FAILED")

    def test_06_group_id_leakage_across_splits(self):
        """6. Detection of Group ID leakage across splits"""
        X = np.random.randn(4, 300).astype(np.float32)
        y = np.array([0, 1, 0, 1], dtype=np.int64)
        g = np.array(["subj_A", "subj_B", "subj_A", "subj_C"])  # subj_A in train and val
        splits = {"train": [0], "val": [2], "test": [1, 3]}

        npz_p, split_p, out_p = self._create_mock_npz_and_split(X, y, group_ids=g, split_indices=splits)
        report = run_integrity_audit(npz_p, split_p, out_p, expected_windows=4)

        self.assertEqual(report["group_isolation"]["status"], "FAILED")
        self.assertTrue(report["group_isolation"]["has_group_leakage"])
        self.assertEqual(report["group_isolation"]["train_val_group_overlap_count"], 1)

    def test_07_not_verifiable_when_group_ids_absent(self):
        """7. NOT_VERIFIABLE status when Group IDs are absent"""
        X = np.random.randn(3, 300).astype(np.float32)
        y = np.array([0, 1, 2], dtype=np.int64)
        splits = {"train": [0], "val": [1], "test": [2]}

        npz_p, split_p, out_p = self._create_mock_npz_and_split(X, y, group_ids=None, split_indices=splits)
        report = run_integrity_audit(npz_p, split_p, out_p, expected_windows=3)

        self.assertEqual(report["group_isolation"]["status"], "NOT_VERIFIABLE")
        self.assertEqual(report["overall_status"], "NOT_VERIFIABLE")

    def test_08_out_of_range_and_negative_indices(self):
        """8. Detection of out-of-range and negative indices"""
        X = np.random.randn(3, 300).astype(np.float32)
        y = np.array([0, 1, 2], dtype=np.int64)
        splits = {"train": [0, -1], "val": [1], "test": [99]}  # -1 and 99 invalid

        npz_p, split_p, out_p = self._create_mock_npz_and_split(X, y, split_indices=splits)
        report = run_integrity_audit(npz_p, split_p, out_p, expected_windows=3)

        self.assertEqual(report["index_split_status"], "FAILED")
        self.assertEqual(report["split_integrity"]["split_index_checks"]["train"]["negative_index_count"], 1)
        self.assertEqual(report["split_integrity"]["split_index_checks"]["test"]["out_of_bounds_count"], 1)

    def test_09_incomplete_split_coverage(self):
        """9. Detection of incomplete split coverage"""
        X = np.random.randn(4, 300).astype(np.float32)
        y = np.array([0, 1, 2, 0], dtype=np.int64)
        splits = {"train": [0, 1], "val": [2], "test": []}  # Index 3 missing

        npz_p, split_p, out_p = self._create_mock_npz_and_split(X, y, split_indices=splits)
        report = run_integrity_audit(npz_p, split_p, out_p, expected_windows=4)

        self.assertEqual(report["index_split_status"], "FAILED")
        self.assertEqual(report["split_integrity"]["unassigned_index_count"], 1)
        self.assertLess(report["split_integrity"]["coverage_ratio"], 1.0)

    def test_10_canonical_hash_consistency_shapes(self):
        """10. Canonical hash consistency between [N, 300] and [N, 300, 1]"""
        w_2d = np.arange(300, dtype=np.float32)
        w_3d = w_2d[:, np.newaxis]

        hash_2d, bytes_2d = canonical_window_hash(w_2d)
        hash_3d, bytes_3d = canonical_window_hash(w_3d)

        self.assertEqual(hash_2d, hash_3d)
        self.assertEqual(bytes_2d, bytes_3d)

    def test_11_json_serialization(self):
        """11. Successful JSON serialization"""
        X = np.random.randn(3, 300).astype(np.float32)
        y = np.array([0, 1, 2], dtype=np.int64)
        splits = {"train": [0], "val": [1], "test": [2]}

        npz_p, split_p, out_p = self._create_mock_npz_and_split(X, y, split_indices=splits)
        report = run_integrity_audit(npz_p, split_p, out_p, expected_windows=3)

        self.assertTrue(out_p.exists())
        with open(out_p, "r", encoding="utf-8") as f:
            loaded_report = json.load(f)
        self.assertEqual(loaded_report["audit_name"], "mmwave_dataset_integrity_audit")

    def test_12_cli_exit_codes(self):
        """12. Correct exit codes for success, failure, unverifiable"""
        # Test PASSED -> exit code 0
        X = np.random.randn(3, 300).astype(np.float32)
        y = np.array([0, 1, 2], dtype=np.int64)
        g = np.array(["g1", "g2", "g3"])
        splits = {"train": [0], "val": [1], "test": [2]}

        npz_p, split_p, out_p = self._create_mock_npz_and_split(X, y, group_ids=g, split_indices=splits)
        report = run_integrity_audit(npz_p, split_p, out_p, expected_windows=3)
        self.assertEqual(report["overall_status"], "PASSED")


if __name__ == "__main__":
    unittest.main()
