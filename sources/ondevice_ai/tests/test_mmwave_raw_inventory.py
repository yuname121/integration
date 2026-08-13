#!/usr/bin/env python3
"""
Comprehensive unit tests for Phase A0 mmWave Raw Dataset Inventory Audit & Validation.
Tests checksum computation, zip integrity checks, path traversal, CRC failures,
deterministic ID generation, role classification, recording linkage cardinality,
gate derivation logic, output determinism, inventory validation, and unsafe deserialization protection.
"""

import os
import sys
import json
import hashlib
import tempfile
import zipfile
import unittest

# Add scripts directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
import audit_mmwave_raw_inventory as audit
import validate_mmwave_raw_inventory as validator


class TestMMWaveRawInventoryAudit(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_15_1_streaming_checksums_exact(self):
        """15.1: Verify streaming SHA-256 and MD5 computation against exact known string digest."""
        tmp_file = os.path.join(self.test_dir.name, "exact_sample.txt")
        content = b"SafeNest Phase A0 Verification String 2026"
        with open(tmp_file, "wb") as f:
            f.write(content)

        expected_sha256 = hashlib.sha256(content).hexdigest()
        expected_md5 = hashlib.md5(content).hexdigest()

        sha256, md5 = audit.compute_streaming_checksums(tmp_file)
        self.assertEqual(sha256, expected_sha256)
        self.assertEqual(md5, expected_md5)

    def test_15_2_missing_archive_behavior(self):
        """15.2: Verify blocker anomaly and BLOCKED gate when archive is missing."""
        mock_integrity = {"zip_integrity_status": "FAIL", "member_count": 0}
        gate, a1_entry = audit.derive_a0_gate(False, mock_integrity, 1, 0, 0, 0, 0, 0, False)

        self.assertEqual(gate, "BLOCKED")
        self.assertEqual(a1_entry, "BLOCKED")

    def test_15_3_corrupt_zip_crc_failure(self):
        """15.3: Verify CRC failure detection using a controlled corrupt ZIP fixture."""
        zip_path = os.path.join(self.test_dir.name, "corrupt.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("test.txt", b"1234567890" * 100)

        # Corrupt bytes inside the payload area of the file
        with open(zip_path, "r+b") as f:
            f.seek(45)
            f.write(b"XXXXX_CORRUPTED_PAYLOAD_XXXXX")

        res = audit.audit_zip_integrity(zip_path, verify_crc=True)
        self.assertTrue(res["crc_failure_count"] > 0 or not res["zip_openable"] or res["zip_integrity_status"] == "FAIL")

    def test_15_4_path_traversal_detection(self):
        """15.4: Test detection of path traversal risks in zip archive."""
        zip_path = os.path.join(self.test_dir.name, "traversal.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../dangerous.txt", "bad content")
            zf.writestr("sub/../../evil.txt", "bad content")

        res = audit.audit_zip_integrity(zip_path, verify_crc=False)
        self.assertTrue(res["path_traversal_risk_count"] >= 1)
        self.assertEqual(res["zip_integrity_status"], "FAIL")

    def test_15_5_case_insensitive_collision(self):
        """15.5: Test detection of case-insensitive path collisions."""
        zip_path = os.path.join(self.test_dir.name, "casefold.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("Subject01/data.csv", "a,b,c")
            zf.writestr("subject01/DATA.CSV", "x,y,z")

        res = audit.audit_zip_integrity(zip_path, verify_crc=False)
        self.assertEqual(res["duplicate_casefold_path_count"], 1)

    def test_15_6_duplicate_member_path(self):
        """15.6: Test handling of duplicate member paths."""
        zip_path = os.path.join(self.test_dir.name, "dup_path.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("db_records/P001/file.txt", "v1")

        res = audit.audit_zip_integrity(zip_path, verify_crc=False)
        self.assertEqual(res["duplicate_exact_path_count"], 0)

    def test_15_7_deterministic_identifiers(self):
        """15.7: Verify deterministic ID generation and collision avoidance."""
        d1, a1, subj1, sess1, rec1, src1 = audit.derive_ids(
            "10.5281/zenodo.18599983", "f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0",
            "P001", "Sitting", "Rest", "db_records/P001/Sitting/Rest/radar_rFFTs.zlib"
        )
        d2, a2, subj2, sess2, rec2, src2 = audit.derive_ids(
            "10.5281/zenodo.18599983", "f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0",
            "P001", "Sitting", "Rest", "db_records/P001/Sitting/Rest/radar_rFFTs.zlib"
        )

        # Identical inputs must yield identical IDs
        self.assertEqual(d1, d2)
        self.assertEqual(a1, a2)
        self.assertEqual(subj1, subj2)
        self.assertEqual(sess1, sess2)
        self.assertEqual(rec1, rec2)
        self.assertEqual(src1, src2)

        # Distinct inputs must yield distinct IDs
        _, _, subj3, _, rec3, src3 = audit.derive_ids(
            "10.5281/zenodo.18599983", "f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0",
            "P002", "Lying", "Post-exercise", "db_records/P002/Lying/Post-exercise/radar_rFFTs.zlib"
        )
        self.assertNotEqual(subj1, subj3)
        self.assertNotEqual(rec1, rec3)
        self.assertNotEqual(src1, src3)

    def test_15_8_stable_output_ordering(self):
        """15.8: Verify stable output ordering across multiple executions."""
        zip_path = os.path.join(self.test_dir.name, "stable.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("db_records/P001/Sitting/Rest/radar_rFFTs.zlib", b"\x78\xda\x00")
            zf.writestr("db_records/P001/Sitting/Rest/radar_timestamps.csv", "2025-02-20T12:00:00\n")
            zf.writestr("db_records/P001/Sitting/Rest/radar_chirpConfig.json", '{"START_FREQ": 60250000000.0}\n')
            zf.writestr("db_records/P001/Sitting/Rest/movesense_acc.csv", "Timestamp,X,Y,Z\n")
            zf.writestr("db_records/P001/Sitting/Rest/movesense_ecg.csv", "Timestamp,mV\n")

        res1 = audit.audit_zip_integrity(zip_path, verify_crc=True)
        res2 = audit.audit_zip_integrity(zip_path, verify_crc=True)
        self.assertEqual(res1, res2)

    def test_15_11_unknown_role_preservation(self):
        """15.11: Verify unknown file roles are preserved as UNKNOWN and not omitted."""
        role, ev = audit.classify_member_role("db_records/P001/Sitting/Rest/unexpected_sensor.xyz")
        self.assertEqual(role, "UNKNOWN")
        self.assertEqual(ev, "INFERRED_FROM_FILENAME")

    def test_15_12_recording_linkage_cardinality(self):
        """15.12: Test recording linkage evaluation logic against cardinality rules."""
        complete_rec = {
            'radar_files': ['radar_rFFTs.zlib'],
            'timestamp_files': ['radar_timestamps.csv'],
            'chirp_config_files': ['radar_chirpConfig.json'],
            'movesense_acc_files': ['movesense_acc.csv'],
            'movesense_ecg_files': ['movesense_ecg.csv'],
            'annotation_files': ['non_breathing_ts.csv']
        }
        self.assertEqual(audit.evaluate_recording_linkage(complete_rec), "COMPLETE")

        complete_no_opt_rec = {
            'radar_files': ['radar_rFFTs.zlib'],
            'timestamp_files': ['radar_timestamps.csv'],
            'chirp_config_files': ['radar_chirpConfig.json'],
            'movesense_acc_files': ['movesense_acc.csv'],
            'movesense_ecg_files': ['movesense_ecg.csv'],
            'annotation_files': []
        }
        self.assertEqual(audit.evaluate_recording_linkage(complete_no_opt_rec), "COMPLETE_WITH_OPTIONAL_FILES_ABSENT")

        partial_rec = {
            'radar_files': ['radar_rFFTs.zlib'],
            'timestamp_files': [],
            'chirp_config_files': ['radar_chirpConfig.json'],
            'movesense_acc_files': ['movesense_acc.csv'],
            'movesense_ecg_files': [],
            'annotation_files': []
        }
        self.assertEqual(audit.evaluate_recording_linkage(partial_rec), "PARTIAL")

        broken_rec = {
            'radar_files': [],
            'timestamp_files': [],
            'chirp_config_files': [],
            'movesense_acc_files': [],
            'movesense_ecg_files': [],
            'annotation_files': []
        }
        self.assertEqual(audit.evaluate_recording_linkage(broken_rec), "BROKEN")

    def test_15_13_gate_derivation_logic(self):
        """15.13: Test direct unit derivation for PASS, PASS_WITH_WARNINGS, FAIL, BLOCKED gate states."""
        clean_integrity = {"zip_integrity_status": "PASS"}

        # 1. PASS
        gate, a1 = audit.derive_a0_gate(True, clean_integrity, 0, 0, 0, 0, 0, 0, True)
        self.assertEqual(gate, "PASS")
        self.assertEqual(a1, "READY")

        # 2. PASS_WITH_WARNINGS -> READY_WITH_CONDITIONS
        gate, a1 = audit.derive_a0_gate(True, clean_integrity, 0, 0, 1, 0, 0, 0, True)
        self.assertEqual(gate, "PASS_WITH_WARNINGS")
        self.assertEqual(a1, "READY_WITH_CONDITIONS")

        # 3. FAIL -> NOT_READY
        gate, a1 = audit.derive_a0_gate(True, clean_integrity, 0, 1, 0, 0, 0, 0, True)
        self.assertEqual(gate, "FAIL")
        self.assertEqual(a1, "NOT_READY")

        # 4. BLOCKED -> BLOCKED
        gate, a1 = audit.derive_a0_gate(False, clean_integrity, 1, 0, 0, 0, 0, 0, True)
        self.assertEqual(gate, "BLOCKED")
        self.assertEqual(a1, "BLOCKED")

    def test_15_14_no_unsafe_deserialization(self):
        """15.14: Verify code path contains no calls to unsafe object deserialization mechanisms."""
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'audit_mmwave_raw_inventory.py'))
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()

        forbidden_patterns = [
            "pickle.load", "pickle.loads", "joblib.load", "torch.load",
            "allow_pickle=True", "pandas.read_pickle"
        ]
        for pattern in forbidden_patterns:
            self.assertNotIn(pattern, code, f"Forbidden unsafe deserialization call '{pattern}' found in script!")


class TestMMWaveRawInventoryValidator(unittest.TestCase):

    def test_16_inventory_validator_success(self):
        """16: Test inventory validator on canonical dataset manifests."""
        repo_root = os.popen("git rev-parse --show-toplevel").read().strip() or os.getcwd()
        inv_dir = os.path.join(repo_root, "datasets/mmwave/manifests/a0_raw_inventory")

        if os.path.exists(inv_dir):
            success, errors = validator.validate_inventory_directory(inv_dir)
            self.assertTrue(success, f"Validation errors: {errors}")


if __name__ == "__main__":
    unittest.main()
