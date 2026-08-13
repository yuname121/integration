#!/usr/bin/env python3
"""Unit test suite for SafeNest mmWave M-B0 Evaluation Protocol & LOCKED_TEST Guard."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.mmwave_phase_b_access import (
    FORBIDDEN_LABEL_FIELDS,
    LOCKED_TEST_AccessError,
    PhaseBAccessGuard,
)
from scripts.validate_mmwave_m_b0 import MB0ValidationError, validate_m_b0_artifacts

ROOT_DIR = Path(__file__).resolve().parents[1]


class TestMMWaveMB0(unittest.TestCase):
    """Test suite for Phase M-B0 access control, isolation, label sanitization, and validator."""

    def setUp(self) -> None:
        self.guard = PhaseBAccessGuard(root_dir=ROOT_DIR)
        self.manifest_dir = ROOT_DIR / "datasets/mmwave/manifests/M-B0_evaluation_protocol"

    def test_train_data_retrieval(self) -> None:
        train_data = self.guard.get_train_data(include_ambiguous=False)
        self.assertEqual(train_data["split"], "TRAIN")
        self.assertEqual(train_data["total_count"], 327)
        self.assertEqual(train_data["signals"].shape, (327, 300))

        train_data_all = self.guard.get_train_data(include_ambiguous=True)
        self.assertEqual(train_data_all["total_count"], 358)

    def test_validation_data_retrieval(self) -> None:
        val_data = self.guard.get_validation_data(include_ambiguous=False)
        self.assertEqual(val_data["split"], "VALIDATION")
        self.assertEqual(val_data["total_count"], 79)
        self.assertEqual(val_data["signals"].shape, (79, 300))

    def test_locked_test_model_selection_prohibited(self) -> None:
        with self.assertRaises(LOCKED_TEST_AccessError):
            self.guard.get_model_selection_dataset("LOCKED_TEST")

    def test_locked_test_structural_audit_label_sanitization(self) -> None:
        struct_data = self.guard.get_structural_audit_dataset("LOCKED_TEST")
        self.assertEqual(struct_data["split"], "LOCKED_TEST")
        self.assertEqual(struct_data["total_count"], 88)
        self.assertEqual(struct_data["signals"].shape, (88, 300))

        for w in struct_data["windows"]:
            for forbidden_key in FORBIDDEN_LABEL_FIELDS:
                self.assertNotIn(forbidden_key, w, f"Forbidden label field '{forbidden_key}' exposed in structural audit!")

    def test_locked_test_structural_audit_mutation_isolation(self) -> None:
        struct_data = self.guard.get_structural_audit_dataset("LOCKED_TEST")
        w_first = struct_data["windows"][0]

        # Mutate returned structural window dictionary
        w_first["subject_id"] = "MUTATED_SUBJECT"
        # Mutate returned signal matrix slice
        struct_data["signals"][0, 0] = 99999.0

        # Retrieve fresh structural audit dataset
        fresh_data = self.guard.get_structural_audit_dataset("LOCKED_TEST")
        w_fresh = fresh_data["windows"][0]

        self.assertNotEqual(w_fresh["subject_id"], "MUTATED_SUBJECT")
        self.assertNotEqual(fresh_data["signals"][0, 0], 99999.0)

    def test_locked_test_final_eval_token(self) -> None:
        with self.assertRaises(LOCKED_TEST_AccessError):
            self.guard.get_locked_test_final_evaluation_dataset(authorization_token=None)

        with self.assertRaises(LOCKED_TEST_AccessError):
            self.guard.get_locked_test_final_evaluation_dataset(authorization_token="WRONG_TOKEN")

        valid_data = self.guard.get_locked_test_final_evaluation_dataset(
            authorization_token="AUTHORIZED_FINAL_LOCKED_TEST_EVALUATION_TOKEN_V1"
        )
        self.assertEqual(valid_data["total_count"], 75)

    def test_standalone_m_b0_validator_clean(self) -> None:
        if self.manifest_dir.is_dir():
            res = validate_m_b0_artifacts(root_dir=ROOT_DIR, manifest_dir=self.manifest_dir)
            self.assertTrue(res["validation_success"])
            self.assertEqual(res["m_b0_gate_status"], "PASS_WITH_WARNINGS")
            self.assertEqual(res["m_b1_entry_status"], "READY_WITH_CONDITIONS")
            self.assertTrue(res["independently_measured"]["locked_test_label_sanitization_verified"])
            self.assertTrue(res["independently_measured"]["hardened_checksum_verification"])

    def test_validator_detects_near_duplicate_audit_corruption(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            audit_file = tmp_manifest / "near_duplicate_audit.json"
            data = json.loads(audit_file.read_text(encoding="utf-8"))
            data["total_flagged_near_duplicates"] = 999  # Corrupt count
            audit_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(MB0ValidationError):
                validate_m_b0_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_checksum_malformed_line_rejection(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            chk_file = tmp_manifest / "checksums.sha256"
            content = chk_file.read_text(encoding="utf-8") + "\nMALFORMED_LINE_WITHOUT_FILENAME\n"
            chk_file.write_text(content, encoding="utf-8")

            with self.assertRaises(MB0ValidationError):
                validate_m_b0_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_checksum_path_traversal_rejection(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            chk_file = tmp_manifest / "checksums.sha256"
            content = chk_file.read_text(encoding="utf-8") + "\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  ../outside.json\n"
            chk_file.write_text(content, encoding="utf-8")

            with self.assertRaises(MB0ValidationError):
                validate_m_b0_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_checksum_missing_required_artifact_rejection(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            chk_file = tmp_manifest / "checksums.sha256"
            lines = [l for l in chk_file.read_text(encoding="utf-8").splitlines() if "near_duplicate_audit.json" not in l]
            chk_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaises(MB0ValidationError):
                validate_m_b0_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)


if __name__ == "__main__":
    unittest.main()
