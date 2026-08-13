#!/usr/bin/env python3
"""Archive-independent tests for the Phase A1 safe rFFT reader."""

from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
import unittest
import zipfile
import zlib

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from mmwave_rfft_reader import (  # noqa: E402
    DecompressionError,
    PayloadFormatError,
    SafeRFFTReader,
    UnsafeSerializationError,
    bounded_zlib_decompress_bytes,
    classify_alignment,
    decode_restricted_numpy_pickle,
    identify_payload_format,
    parse_chirp_config,
    parse_radar_timestamps,
)
from run_mmwave_rfft_pilot import (  # noqa: E402
    assign_decoder_profiles,
    deterministic_pilot_selection,
    scan_timestamp_counts,
)
from validate_mmwave_rfft_pilot import derive_validated_gate  # noqa: E402


def controlled_pickle(frames: int = 3, antennas: int = 8, bins: int = 4) -> bytes:
    values = np.arange(frames * antennas * bins, dtype=np.float64)
    tensor = values.reshape(frames, antennas, bins).astype(np.complex128)
    tensor += 1j * (tensor + 0.5)
    range_bins = np.linspace(0.0, 3.0, bins, dtype=np.float64)
    return pickle.dumps([tensor, range_bins], protocol=5)


def fake_record(subject: int, posture: str, activity: str) -> dict:
    subject_text = f"p{subject:03d}"
    activity_id = activity.lower().replace("-", "_")
    profile = "SCHEMA_PROFILE_001" if activity == "Rest" else "SCHEMA_PROFILE_002"
    annotation = ["annotation.csv"] if activity == "Rest" else []
    base = f"db_records/P{subject:03d}/{posture}/{activity}"
    return {
        "recording_id": f"dataset-{subject_text}-{posture.lower()}-{activity_id}",
        "subject_id": f"dataset-{subject_text}",
        "source_recording_path": base,
        "posture": {"value": posture},
        "activity_or_test": {"value": activity},
        "schema_profile": profile,
        "annotation_files": annotation,
        "timestamp_files": [base + "/radar_timestamps.csv"],
    }


class TestSafeRFFTReader(unittest.TestCase):
    def test_01_valid_zlib_decoding(self):
        original = b"SafeNest" * 200
        decoded, metadata = bounded_zlib_decompress_bytes(zlib.compress(original, 9))
        self.assertEqual(decoded, original)
        self.assertTrue(metadata["zlib_eof"])
        self.assertEqual(metadata["unused_trailing_bytes"], 0)

    def test_02_invalid_zlib_header(self):
        with self.assertRaises(DecompressionError):
            bounded_zlib_decompress_bytes(b"\x00\x00not-zlib")

    def test_03_truncated_zlib_stream(self):
        compressed = zlib.compress(b"payload" * 100)
        with self.assertRaisesRegex(DecompressionError, "truncated|incomplete"):
            bounded_zlib_decompress_bytes(compressed[:-2])

    def test_04_corrupt_zlib_stream(self):
        compressed = bytearray(zlib.compress(b"payload" * 100))
        compressed[len(compressed) // 2] ^= 0xFF
        with self.assertRaises(DecompressionError):
            bounded_zlib_decompress_bytes(bytes(compressed))

    def test_05_maximum_decompressed_size_enforced(self):
        with self.assertRaisesRegex(DecompressionError, "exceeds"):
            bounded_zlib_decompress_bytes(
                zlib.compress(b"x" * 4096), max_decompressed_bytes=128
            )

    def test_06_trailing_data_rejected_and_measured(self):
        data = zlib.compress(b"abc") + b"tail"
        with self.assertRaisesRegex(DecompressionError, "trailing data"):
            bounded_zlib_decompress_bytes(data)
        decoded, metadata = bounded_zlib_decompress_bytes(data, allow_trailing_data=True)
        self.assertEqual(decoded, b"abc")
        self.assertEqual(metadata["unused_trailing_bytes"], 4)
        self.assertFalse(metadata["concatenated_stream_detected"])

    def test_07_concatenated_zlib_stream_rejected(self):
        data = zlib.compress(b"first") + zlib.compress(b"second")
        with self.assertRaisesRegex(DecompressionError, "concatenated"):
            bounded_zlib_decompress_bytes(data)

    def test_08_safe_payload_format_detection(self):
        payload = controlled_pickle()
        self.assertEqual(identify_payload_format(payload), "PYTHON_PICKLE_PROTOCOL_5")
        self.assertEqual(identify_payload_format(b"\x93NUMPYrest"), "NUMPY_NPY")
        self.assertEqual(identify_payload_format(b"unknown"), "UNKNOWN_BINARY")

    def test_09_unsupported_payload_format_rejected(self):
        with self.assertRaises(PayloadFormatError):
            decode_restricted_numpy_pickle(b"not a pickle")

    def test_10_unsafe_pickle_global_rejected(self):
        unsafe_candidate = pickle.dumps(len, protocol=5)
        with self.assertRaises(UnsafeSerializationError):
            decode_restricted_numpy_pickle(unsafe_candidate)

    def test_11_deterministic_controlled_fixture_decode(self):
        payload = controlled_pickle(frames=3, antennas=8, bins=4)
        tensor_a, ranges_a, metadata_a = decode_restricted_numpy_pickle(payload)
        tensor_b, ranges_b, metadata_b = decode_restricted_numpy_pickle(payload)
        np.testing.assert_array_equal(tensor_a, tensor_b)
        np.testing.assert_array_equal(ranges_a, ranges_b)
        self.assertEqual(metadata_a, metadata_b)
        self.assertEqual(tensor_a.shape, (3, 8, 4))
        self.assertEqual(tensor_a.dtype.name, "complex128")
        self.assertEqual(ranges_a.dtype.name, "float64")
        self.assertFalse(metadata_a["arbitrary_object_execution"])

    def test_12_frame_count_extraction_through_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = os.path.join(tmp, "fixture.zip")
            base = "db_records/P001/Sitting/Rest/"
            config = {
                "ADC_SAMPLES": 4,
                "PERIODICITY": 100.0,
                "TX_ANTENNAS": 2,
                "RX_ANTENNAS": 4,
                "R_BIN": 1.0,
            }
            timestamps = "\n".join(
                [
                    "2026-01-01T00:00:00.000000000",
                    "2026-01-01T00:00:00.100000000",
                    "2026-01-01T00:00:00.200000000",
                ]
            )
            with zipfile.ZipFile(archive, "w") as fixture:
                fixture.writestr(
                    base + "radar_rFFTs.zlib", zlib.compress(controlled_pickle(3, 8, 4), 9)
                )
                fixture.writestr(base + "radar_timestamps.csv", timestamps)
                fixture.writestr(base + "radar_chirpConfig.json", json.dumps(config))
            result = SafeRFFTReader().read_recording(
                archive_path=archive,
                radar_member=base + "radar_rFFTs.zlib",
                timestamp_member=base + "radar_timestamps.csv",
                chirp_config_member=base + "radar_chirpConfig.json",
            )
            self.assertEqual(result["structural_metadata"]["frame_count"], 3)
            self.assertEqual(result["structural_metadata"]["alignment_status"], "EXACT_ALIGNMENT")
            self.assertEqual(result["tensor"].shape, (3, 8, 4))

    def test_13_timestamp_parsing_statistics(self):
        raw = (
            "2026-01-01T00:00:00.000000001\n"
            "2026-01-01T00:00:00.100000001\n"
            "2026-01-01T00:00:00.200000001\n"
        ).encode()
        stats = parse_radar_timestamps(raw)
        self.assertEqual(stats["timestamp_count"], 3)
        self.assertEqual(stats["timestamp_median_dt_seconds"], 0.1)
        self.assertEqual(stats["empirical_frame_rate_hz"], 10.0)
        self.assertEqual(stats["duplicate_timestamp_count"], 0)
        self.assertEqual(stats["backward_timestamp_count"], 0)
        self.assertEqual(stats["large_gap_count"], 0)

    def test_14_exact_and_mismatch_alignment(self):
        self.assertEqual(classify_alignment(500, 500), ("EXACT_ALIGNMENT", 0))
        self.assertEqual(classify_alignment(500, 499), ("OFF_BY_ONE", 1))
        self.assertEqual(classify_alignment(500, 490), ("FRAME_COUNT_MISMATCH", 10))
        self.assertEqual(classify_alignment(None, 500), ("DECODE_FAILURE", None))

    def test_15_chirp_config_parsing(self):
        config = {
            "START_FREQ": 60_250_000_000.0,
            "B": 480_000_000.0,
            "ADC_SAMPLES": 64,
            "PERIODICITY": 100.0,
            "TX_ANTENNAS": 2,
            "RX_ANTENNAS": 4,
        }
        parsed = parse_chirp_config(json.dumps(config).encode())
        self.assertEqual(parsed["interpreted"]["virtual_antenna_count"], 8)
        self.assertEqual(parsed["interpreted"]["frame_period_seconds"], 0.1)
        self.assertEqual(parsed["interpreted"]["configured_frame_rate_hz"], 10.0)
        self.assertEqual(parsed["interpreted"]["sampled_end_frequency_hz"], 60_730_000_000.0)
        self.assertEqual(len(parsed["chirp_config_hash"]), 64)

    def test_16_deterministic_pilot_selection(self):
        records = [
            fake_record(subject, posture, activity)
            for subject in (1, 50, 100)
            for posture in ("Lying", "Sitting")
            for activity in ("Rest", "Post-exercise")
        ]
        anomalies = {
            "anomalies": [
                {
                    "category": "SCHEMA",
                    "affected_files": [
                        "db_records/P050/Sitting/Post-exercise/radar_timestamps.csv"
                    ],
                }
            ]
        }
        recommended = {"P050/Lying/Post-exercise"}
        timestamp_counts = {item["recording_id"]: 500 for item in records}
        anomaly_record = next(
            item
            for item in records
            if item["source_recording_path"]
            == "db_records/P050/Sitting/Post-exercise"
        )
        six_hundred_record = next(
            item
            for item in records
            if item["source_recording_path"] == "db_records/P050/Sitting/Rest"
        )
        timestamp_counts[anomaly_record["recording_id"]] = 400
        timestamp_counts[six_hundred_record["recording_id"]] = 600
        first = deterministic_pilot_selection(
            records,
            anomalies,
            recommended_paths=recommended,
            timestamp_counts=timestamp_counts,
            target_count=11,
        )
        second = deterministic_pilot_selection(
            list(reversed(records)),
            anomalies,
            recommended_paths=recommended,
            timestamp_counts=timestamp_counts,
            target_count=11,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 11)
        self.assertEqual(
            {item["a0_schema_profile"] for item in first},
            {"SCHEMA_PROFILE_001", "SCHEMA_PROFILE_002"},
        )
        self.assertTrue(
            any(
                item["selection_reason"] == "A0_RECORDED_TIMESTAMP_LENGTH_EXCEPTION"
                for item in first
            )
        )
        self.assertEqual(
            {item["timestamp_count_prescan"] for item in first},
            {400, 500, 600},
        )
        self.assertTrue(
            any(
                item["selection_reason"]
                == "ZIP_TIMESTAMP_COUNT_STRATUM_REPRESENTATIVE"
                for item in first
            )
        )
        self.assertTrue(
            any(
                item["selection_reason"] == "A0_REPORT_RECOMMENDED_PILOT"
                for item in first
            )
        )

    def test_17_decoder_profile_classification(self):
        def result(recording: str, profile: str, frames: int) -> dict:
            return {
                "recording_id": recording,
                "a0_schema_profile": profile,
                "payload_decode_status": "SUCCESS_WITH_WARNING",
                "payload_format": "PYTHON_PICKLE_PROTOCOL_5_NUMPY_ARRAY_PAIR",
                "radar_header_signature": "78da",
                "dtype": "complex128",
                "endianness": "little",
                "is_complex": True,
                "complex_representation": "NATIVE_NUMPY_COMPLEX_INTERLEAVED_REAL_IMAG_FLOAT64",
                "shape": [frames, 8, 64],
                "frame_count": frames,
                "frame_axis": 0,
                "antenna_axis": 1,
                "range_bin_axis": 2,
                "chirp_config_hash": "fullhash",
                "a0_compatible_chirp_config_hash": "a0hash",
                "safe_decode_method": "PICKLETOOLS_ALLOWLISTED_SYMBOLIC_VM",
                "timestamp_median_dt_seconds": 0.1,
                "chirp_config_summary": {
                    "interpreted": {
                        "adc_samples": 64,
                        "tx_antenna_count": 2,
                        "rx_antenna_count": 4,
                    }
                },
            }

        results = [
            result("a", "SCHEMA_PROFILE_001", 400),
            result("b", "SCHEMA_PROFILE_002", 600),
        ]
        profiles = assign_decoder_profiles(results)
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0]["shape_pattern"], [None, 8, 64])
        self.assertEqual(profiles[0]["observed_frame_counts"], [400, 600])
        self.assertEqual(
            profiles[0]["supported_a0_schema_profiles"],
            ["SCHEMA_PROFILE_001", "SCHEMA_PROFILE_002"],
        )
        self.assertEqual(results[0]["decoder_profile_id"], results[1]["decoder_profile_id"])

    def test_18_bounded_timestamp_count_prescan(self):
        records = [
            fake_record(1, "Lying", "Rest"),
            fake_record(2, "Sitting", "Rest"),
            fake_record(3, "Lying", "Post-exercise"),
        ]
        expected = [400, 500, 600]
        with tempfile.TemporaryDirectory() as tmp:
            archive = os.path.join(tmp, "timestamps.zip")
            with zipfile.ZipFile(archive, "w") as fixture:
                for record, count in zip(records, expected):
                    rows = "\n".join(
                        f"2026-01-01T00:00:{index // 10:02d}.{index % 10}00000000"
                        for index in range(count)
                    )
                    fixture.writestr(record["timestamp_files"][0], rows)
                fixture.writestr(
                    "db_records/P001/Lying/Rest/radar_rFFTs.zlib",
                    b"must-not-be-opened",
                )
            measured = scan_timestamp_counts(os.path.abspath(archive), records)
        self.assertEqual(
            [measured[record["recording_id"]] for record in records], expected
        )

    def test_19_validation_failure_forces_gate_failure(self):
        successful_result = {
            "payload_decode_status": "SUCCESS_WITH_WARNING",
            "errors": [],
            "frame_axis": 0,
            "antenna_axis": 1,
            "range_bin_axis": 2,
        }
        warning = {"severity": "WARNING"}
        self.assertEqual(
            derive_validated_gate(
                results=[successful_result],
                exceptions=[warning],
                validation_success=True,
            ),
            ("PASS_WITH_WARNINGS", "READY_WITH_CONDITIONS"),
        )
        self.assertEqual(
            derive_validated_gate(
                results=[successful_result],
                exceptions=[warning],
                validation_success=False,
            ),
            ("FAIL", "NOT_READY"),
        )


if __name__ == "__main__":
    unittest.main()
