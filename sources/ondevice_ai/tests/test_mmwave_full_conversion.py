#!/usr/bin/env python3
"""Unit test suite for SafeNest Phase A6 Full Conversion and Integrity Audit.

Exercises the A6 contract and critical negative paths using synthetic fixtures.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mmwave_full_converter import (
    PROFILE_ID,
    FullConversionError,
    FullConversionProfile,
    compute_canonical_signal_hash,
    process_single_recording,
)
from validate_mmwave_full_conversion import (
    A6ValidationError,
    WINDOW_PROVENANCE_FIELDS,
    _validate_alignment,
    _validate_checksums,
    _validate_recording_accounting,
    validate_full_conversion_artifacts,
)


class TestMmwaveFullConversion(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = FullConversionProfile()
        self.sample_signal = np.sin(np.linspace(0, 2 * np.pi, 300))

    # 1. Deterministic signal hashing
    def test_01_deterministic_signal_hashing(self) -> None:
        h1 = compute_canonical_signal_hash(self.sample_signal)
        h2 = compute_canonical_signal_hash(self.sample_signal)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

    # 2. Canonical signal remains unfiltered and unnormalized
    def test_02_canonical_signal_unfiltered_unnormalized(self) -> None:
        self.assertEqual(self.profile.canonical_signal, "UNFILTERED_UNNORMALIZED_PHASE")

    # 3. Naive timestamp contract & utc_conversion_claimed == False
    def test_03_timestamp_contract_defaults(self) -> None:
        self.assertEqual(self.profile.timestamp_reference, "COMMON_ACQUISITION_COMPUTER_CLOCK")
        self.assertEqual(self.profile.source_timezone, "UNVERIFIED")
        self.assertFalse(self.profile.utc_conversion_claimed)

    # 4. Profile serialization to dict
    def test_04_profile_serialization(self) -> None:
        d = self.profile.to_dict()
        self.assertEqual(d["profile_id"], PROFILE_ID)
        self.assertEqual(d["a1_decoder_profile"], "RFFT_DECODER_PROFILE_001")
        self.assertEqual(d["a2_extraction_profile"], "MMWAVE_PHASE_EXTRACTION_PROFILE_001")
        self.assertEqual(d["a3_timeline_profile"], "MMWAVE_TIMELINE_PROFILE_001")
        self.assertEqual(d["a4_label_profile"], "MMWAVE_LABEL_MAPPING_PROFILE_001")
        self.assertEqual(d["a5_split_profile"], "MMWAVE_SUBJECT_SPLIT_PROFILE_001")

    # 5. LOCKED_TEST training eligibility false check in validator
    def test_05_locked_test_training_eligibility_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            manifest_dir = tmppath / "datasets/mmwave/manifests/a6_full_conversion"
            manifest_dir.mkdir(parents=True, exist_ok=True)

            invalid_window = {
                "window_id": "WIN_0001",
                "subject_id": "P001",
                "split": "LOCKED_TEST",
                "assignment_status": "ASSIGNED",
                "training_eligible": True,  # INVALID!
                "validation_eligible": False,
                "locked_test_evaluation_eligible": True,
            }
            (manifest_dir / "full_window_manifest.jsonl").write_text(json.dumps(invalid_window) + "\n")
            (manifest_dir / "full_recording_results.jsonl").write_text("{}\n")
            (manifest_dir / "full_provenance_manifest.jsonl").write_text("{}\n")
            (manifest_dir / "full_quality_audit.json").write_text(json.dumps({"nan_sample_count": 0, "inf_sample_count": 0, "mean_window_phase_std_dev": 0.5}))
            (manifest_dir / "checksums.sha256").write_text("")

            with self.assertRaises(A6ValidationError):
                validate_full_conversion_artifacts(root_dir=tmppath, manifest_dir=manifest_dir)

    # 6. AMBIGUOUS pure-class eligibility false check in validator
    def test_06_ambiguous_pure_class_eligibility_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            manifest_dir = tmppath / "datasets/mmwave/manifests/a6_full_conversion"
            manifest_dir.mkdir(parents=True, exist_ok=True)

            invalid_window = {
                "window_id": "WIN_0002",
                "subject_id": "P001",
                "split": "TRAIN",
                "assignment_status": "AMBIGUOUS",
                "training_eligible": True,  # INVALID!
                "validation_eligible": False,
                "locked_test_evaluation_eligible": False,
            }
            (manifest_dir / "full_window_manifest.jsonl").write_text(json.dumps(invalid_window) + "\n")
            (manifest_dir / "full_recording_results.jsonl").write_text("{}\n")
            (manifest_dir / "full_provenance_manifest.jsonl").write_text("{}\n")
            (manifest_dir / "full_quality_audit.json").write_text(json.dumps({"nan_sample_count": 0, "inf_sample_count": 0, "mean_window_phase_std_dev": 0.5}))
            (manifest_dir / "checksums.sha256").write_text("")

            with self.assertRaises(A6ValidationError):
                validate_full_conversion_artifacts(root_dir=tmppath, manifest_dir=manifest_dir)

    # 7. Rejection of absolute local paths in canonical provenance fields
    def test_07_absolute_local_path_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            manifest_dir = tmppath / "datasets/mmwave/manifests/a6_full_conversion"
            manifest_dir.mkdir(parents=True, exist_ok=True)

            invalid_prov = {
                "window_id": "WIN_0001",
                "archive_identifier": "/Users/junwoo/db_records.zip",  # INVALID absolute path!
                "source_radar_member": "db_records/P001/Lying/Rest/radar_rFFTs.zlib",
                "source_timestamp_member": "db_records/P001/Lying/Rest/radar_timestamps.csv",
                "a1_decoder_profile": "RFFT_DECODER_PROFILE_001",
                "timestamp_reference": "COMMON_ACQUISITION_COMPUTER_CLOCK",
                "source_timezone": "UNVERIFIED",
                "utc_conversion_claimed": False,
            }
            (manifest_dir / "full_window_manifest.jsonl").write_text("{}\n")
            (manifest_dir / "full_recording_results.jsonl").write_text("{}\n")
            (manifest_dir / "full_provenance_manifest.jsonl").write_text(json.dumps(invalid_prov) + "\n")
            (manifest_dir / "full_quality_audit.json").write_text(json.dumps({"nan_sample_count": 0, "inf_sample_count": 0, "mean_window_phase_std_dev": 0.5}))
            (manifest_dir / "checksums.sha256").write_text("")

            with self.assertRaises(A6ValidationError):
                validate_full_conversion_artifacts(root_dir=tmppath, manifest_dir=manifest_dir)

    # 8. Rejection of trailing Z timestamp in newly generated window manifest
    def test_08_trailing_z_timestamp_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            manifest_dir = tmppath / "datasets/mmwave/manifests/a6_full_conversion"
            manifest_dir.mkdir(parents=True, exist_ok=True)

            invalid_window = {
                "window_id": "WIN_0001",
                "subject_id": "P001",
                "split": "TRAIN",
                "assignment_status": "ASSIGNED",
                "start_timestamp": "2025-02-20T12:34:30.238545Z",  # INVALID trailing Z!
                "last_sample_timestamp": "2025-02-20T12:35:00.138545",
                "end_timestamp_exclusive": "2025-02-20T12:35:00.238545",
                "training_eligible": True,
                "validation_eligible": False,
                "locked_test_evaluation_eligible": False,
            }
            (manifest_dir / "full_window_manifest.jsonl").write_text(json.dumps(invalid_window) + "\n")
            (manifest_dir / "full_recording_results.jsonl").write_text("{}\n")
            (manifest_dir / "full_provenance_manifest.jsonl").write_text("{}\n")
            (manifest_dir / "full_quality_audit.json").write_text(json.dumps({"nan_sample_count": 0, "inf_sample_count": 0, "mean_window_phase_std_dev": 0.5}))
            (manifest_dir / "checksums.sha256").write_text("")

            with self.assertRaises(A6ValidationError):
                validate_full_conversion_artifacts(root_dir=tmppath, manifest_dir=manifest_dir)

    def _aligned_sample(self) -> tuple[dict, dict, np.ndarray]:
        signal = np.ascontiguousarray(self.sample_signal, dtype=np.float64)
        signal_hash = compute_canonical_signal_hash(signal)
        common = {
            "canonical_sample_index": 0,
            "window_id": "WIN_0001",
            "recording_id": "REC_0001",
            "subject_id": "P001",
            "split": "TRAIN",
            "safenest_label": "NORMAL",
            "safenest_label_id": 0,
            "mapping_type": "DERIVED_LABEL",
            "mapping_rule_id": "RULE_001",
            "assignment_status": "ASSIGNED",
            "canonical_signal_hash": signal_hash,
            "training_eligible": True,
            "validation_eligible": False,
            "locked_test_evaluation_eligible": False,
        }
        return dict(common), dict(common), signal.reshape(1, -1)

    def test_09_rejects_window_provenance_semantic_mismatch(self) -> None:
        window, provenance, matrix = self._aligned_sample()
        provenance["split"] = "VALIDATION"
        with self.assertRaisesRegex(A6ValidationError, "Window/provenance mismatch"):
            _validate_alignment([window], [provenance], matrix)

    def test_10_rejects_provenance_signal_hash_mismatch(self) -> None:
        window, provenance, matrix = self._aligned_sample()
        provenance["canonical_signal_hash"] = "0" * 64
        window["canonical_signal_hash"] = "0" * 64
        with self.assertRaisesRegex(A6ValidationError, "signal hash mismatch"):
            _validate_alignment([window], [provenance], matrix)

    def test_11_rejects_failed_recording_status(self) -> None:
        a0 = [{"recording_id": "REC_0001", "subject_id": "P001"}]
        results = [
            {
                "recording_id": "REC_0001",
                "subject_id": "P001",
                "status": "FAILED_ANNOTATION_PARSE",
                "window_count": 0,
            }
        ]
        with self.assertRaisesRegex(A6ValidationError, "non-success A6 status"):
            _validate_recording_accounting(a0, results, [], [])

    def test_12_rejects_recording_sample_count_mismatch(self) -> None:
        window, provenance, _ = self._aligned_sample()
        a0 = [{"recording_id": "REC_0001", "subject_id": "P001"}]
        results = [
            {
                "recording_id": "REC_0001",
                "subject_id": "P001",
                "status": "SUCCESS",
                "window_count": 2,
            }
        ]
        with self.assertRaisesRegex(A6ValidationError, "sample accounting mismatch"):
            _validate_recording_accounting(a0, results, [window], [provenance])

    def test_13_rejects_malformed_checksum_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_dir = root / "datasets/mmwave/manifests/a6_full_conversion"
            manifest_dir.mkdir(parents=True)
            (manifest_dir / "checksums.sha256").write_text("not-a-valid-entry\n", encoding="utf-8")
            with self.assertRaisesRegex(A6ValidationError, "Malformed checksum entry"):
                _validate_checksums(root, manifest_dir)

    def test_14_rejects_missing_required_checksum_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_dir = root / "datasets/mmwave/manifests/a6_full_conversion"
            manifest_dir.mkdir(parents=True)
            one_file = manifest_dir / "processing_profile.json"
            one_file.write_text("{}", encoding="utf-8")
            digest = hashlib.sha256(one_file.read_bytes()).hexdigest()
            (manifest_dir / "checksums.sha256").write_text(
                f"{digest}  processing_profile.json\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(A6ValidationError, "Required checksum targets missing"):
                _validate_checksums(root, manifest_dir)

    @mock.patch("mmwave_full_converter.parse_annotation_file", side_effect=ValueError("bad annotation"))
    @mock.patch("mmwave_full_converter.process_recording_timeline")
    @mock.patch("mmwave_full_converter.MmwavePhaseExtractor.extract")
    @mock.patch("mmwave_full_converter.SafeRFFTReader.read_recording")
    def test_15_annotation_failure_is_recorded_and_blocks_output(
        self,
        mock_read_recording: mock.Mock,
        mock_extract: mock.Mock,
        mock_timeline: mock.Mock,
        _mock_parse_annotation: mock.Mock,
    ) -> None:
        mock_read_recording.return_value = {
            "tensor": np.zeros((300, 8, 64), dtype=np.complex128),
            "range_bins": np.linspace(0.0, 2.0, 64),
        }
        mock_extract.return_value = {
            "unwrapped_phase": np.zeros(300, dtype=np.float64),
            "selection": {
                "selected_range_bin_index": 2,
                "selected_range_m": 0.6,
                "selected_virtual_channels": [0],
            },
        }
        mock_timeline.return_value = (
            {"first_timestamp": "2025-01-01T00:00:00.000000", "dropped_tail_samples": 0},
            [],
            [],
        )

        class FakeZip:
            filename = "/tmp/fake.zip"

            def namelist(self) -> list[str]:
                return [
                    "source/radar_rFFTs.zlib",
                    "source/radar_timestamps.csv",
                    "source/radar_chirpConfig.json",
                    "source/non_breathing_ts.csv",
                ]

            def read(self, member: str) -> bytes:
                return b"placeholder"

        result = process_single_recording(
            {
                "recording_id": "REC_0001",
                "subject_id": "P001",
                "source_recording_path": "source",
                "activity_or_test": {"value": "Rest"},
                "posture": {"value": "Lying"},
            },
            FakeZip(),  # type: ignore[arg-type]
            "TRAIN",
        )
        self.assertEqual(result["status"], "FAILED_ANNOTATION_PARSE")
        self.assertEqual(result["window_count"], 0)
        self.assertEqual(result["annotation_event_count"], 0)
        self.assertEqual(result["exceptions"][-1]["category"], "ANNOTATION_PARSE_FAILED")
        self.assertEqual(result["exceptions"][-1]["severity"], "ERROR")


if __name__ == "__main__":
    unittest.main()
