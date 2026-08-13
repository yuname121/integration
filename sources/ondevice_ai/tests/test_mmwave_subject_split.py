#!/usr/bin/env python3
"""Phase A5 subject isolation, inheritance, provenance, and gate tests."""

from __future__ import annotations

from copy import deepcopy
import json
import unittest

from scripts.mmwave_subject_split import (
    SPLIT_PROFILE_ID, SPLITS, SPLIT_SEED, SubjectSplitError,
    assign_subject_splits, attach_pilot_window_provenance,
    build_recording_split_manifest, build_subject_catalog,
    calculate_split_counts, cross_split_duplicate_hashes, measure_inventory,
)
from scripts.validate_mmwave_subject_split import contains_local_path, derive_gate


def recording(subject: int, posture: str = "Lying", condition: str = "Rest", suffix: str = "0") -> dict:
    sid = f"dataset-p{subject:03d}"
    rid = f"{sid}-{posture.lower()}-{condition.lower().replace('-', '_')}-{suffix}"
    return {
        "dataset_id": "dataset-test", "archive_id": "archive-test", "subject_id": sid,
        "source_subject_id": f"P{subject:03d}", "session_id": f"{sid}-session-01",
        "recording_id": rid, "source_recording_path": f"db_records/P{subject:03d}/{posture}/{condition}",
        "posture": {"value": posture}, "activity_or_test": {"value": condition},
        "radar_files": [f"db_records/P{subject:03d}/{posture}/{condition}/radar_rFFTs.zlib"],
        "timestamp_files": [f"db_records/P{subject:03d}/{posture}/{condition}/radar_timestamps.csv"],
        "schema_profile": "TEST", "quality_status": "NOT_YET_SIGNAL_ASSESSED",
        "annotation_files": ["annotation.csv"] if condition == "Rest" else [],
    }


def window(subject: int = 1, status: str = "ASSIGNED") -> dict:
    sid = f"dataset-p{subject:03d}"
    return {
        "window_id": f"{sid}-lying-rest-0__W0000", "recording_id": f"{sid}-lying-rest-0",
        "subject_id": sid, "mapping_type": "DERIVED" if status == "ASSIGNED" else "AMBIGUOUS",
        "mapping_rule_id": "A4_RULE_TEST", "assignment_status": status,
        "safenest_label": "NORMAL" if status == "ASSIGNED" else None,
        "safenest_label_id": 0 if status == "ASSIGNED" else None,
        "source_start_index": 0, "source_end_index_exclusive": 300,
        "phase_profile": "MMWAVE_PHASE_EXTRACTION_PROFILE_001",
        "timeline_profile": "MMWAVE_TIMELINE_PROFILE_001", "quality_flags": [],
    }


class TestMmwaveSubjectSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.inventory = [
            recording(subject, posture, condition, str(index))
            for subject in range(1, 111)
            for index, (posture, condition) in enumerate((
                ("Lying", "Rest"), ("Lying", "Post-exercise"),
                ("Sitting", "Rest"), ("Sitting", "Post-exercise"),
            ))
        ]
        self.catalog = build_subject_catalog(self.inventory)
        self.subjects = assign_subject_splits(self.catalog)

    def test_each_subject_exactly_once_all_covered_and_no_overlap(self) -> None:
        ids = [row["subject_id"] for row in self.subjects]
        self.assertEqual(len(ids), len(set(ids)), 110)
        sets = {split: {row["subject_id"] for row in self.subjects if row["split"] == split} for split in SPLITS}
        self.assertFalse(sets["TRAIN"] & sets["VALIDATION"])
        self.assertFalse(sets["TRAIN"] & sets["LOCKED_TEST"])
        self.assertFalse(sets["VALIDATION"] & sets["LOCKED_TEST"])

    def test_deterministic_and_input_order_invariant(self) -> None:
        self.assertEqual(self.subjects, assign_subject_splits(list(reversed(self.catalog))))
        self.assertEqual(self.subjects, assign_subject_splits(self.catalog))

    def test_target_ratio_integer_rounding(self) -> None:
        self.assertEqual(calculate_split_counts(110), {"TRAIN": 77, "VALIDATION": 17, "LOCKED_TEST": 16})
        self.assertEqual(dict(__import__("collections").Counter(row["split"] for row in self.subjects)), calculate_split_counts(110))

    def test_recordings_and_multiple_recordings_inherit_subject_split(self) -> None:
        manifest = build_recording_split_manifest(self.inventory, self.subjects)
        split_map = {row["subject_id"]: row["split"] for row in self.subjects}
        self.assertEqual(len(manifest), 440)
        self.assertTrue(all(row["split"] == split_map[row["subject_id"]] for row in manifest))
        self.assertTrue(all(row["synthetic"] is False for row in manifest))

    def test_windows_and_multiple_windows_inherit_subject_split(self) -> None:
        source = window()
        second = deepcopy(source); second["window_id"] = second["window_id"].replace("W0000", "W0001")
        rows = attach_pilot_window_provenance([second, source], self.subjects, signal_hashes={source["window_id"]: "a", second["window_id"]: "b"})
        self.assertEqual({row["split"] for row in rows}, {next(row["split"] for row in self.subjects if row["subject_id"] == source["subject_id"])})

    def test_duplicate_subject_id_rejected(self) -> None:
        with self.assertRaises(SubjectSplitError):
            assign_subject_splits(self.catalog + [deepcopy(self.catalog[0])])

    def test_duplicate_recording_id_rejected(self) -> None:
        with self.assertRaises(SubjectSplitError):
            build_subject_catalog(self.inventory + [deepcopy(self.inventory[0])])

    def test_unknown_recording_subject_fails(self) -> None:
        bad = deepcopy(self.inventory[0]); bad["recording_id"] += "-unknown"; bad["subject_id"] = "unknown"
        with self.assertRaises(SubjectSplitError):
            build_recording_split_manifest([bad], self.subjects)

    def test_unknown_window_subject_and_recording_fail(self) -> None:
        bad_subject = window(); bad_subject["subject_id"] = "unknown"
        with self.assertRaises(SubjectSplitError):
            attach_pilot_window_provenance([bad_subject], self.subjects)
        recordings = build_recording_split_manifest(self.inventory, self.subjects)
        bad_recording = window(); bad_recording["recording_id"] = "unknown"
        with self.assertRaises(SubjectSplitError):
            attach_pilot_window_provenance([bad_recording], self.subjects, recordings)

    def test_invalid_split_enum_rejected(self) -> None:
        bad = deepcopy(self.subjects); bad[0]["split"] = "TEST"
        with self.assertRaises(SubjectSplitError):
            build_recording_split_manifest(self.inventory, bad)

    def test_a4_fields_unchanged_and_ambiguous_preserved_ineligible(self) -> None:
        source = window(status="AMBIGUOUS")
        row = attach_pilot_window_provenance([source], self.subjects, signal_hashes={source["window_id"]: "hash"})[0]
        for field in ("mapping_type", "mapping_rule_id", "assignment_status", "safenest_label", "safenest_label_id"):
            self.assertEqual(row[field], source[field])
        self.assertFalse(row["training_eligible"])
        self.assertFalse(row["validation_eligible"])
        self.assertFalse(row["locked_test_evaluation_eligible"])
        self.assertEqual(row["timestamp_reference"], "COMMON_ACQUISITION_COMPUTER_CLOCK")
        self.assertEqual(row["source_timezone"], "UNVERIFIED")
        self.assertFalse(row["utc_conversion_claimed"])

    def test_inventory_counts_and_range_are_evidence_derived(self) -> None:
        irregular = [recording(1, suffix="a"), recording(1, suffix="b"), recording(2, suffix="a")]
        self.assertEqual(measure_inventory(irregular), {
            "subject_count": 2,
            "recording_count": 3,
            "unique_recording_id_count": 3,
            "recording_count_per_subject": {"minimum": 1, "maximum": 2, "distribution": {"1": 1, "2": 1}},
        })

    def test_locked_test_clean_window_never_training_eligible(self) -> None:
        locked = next(row for row in self.subjects if row["split"] == "LOCKED_TEST")
        number = int(locked["subject_id"].split("p")[-1])
        source = window(number)
        row = attach_pilot_window_provenance([source], self.subjects, signal_hashes={source["window_id"]: "hash"})[0]
        self.assertFalse(row["training_eligible"])
        self.assertTrue(row["locked_test_evaluation_eligible"])

    def test_window_id_overlap_rejected(self) -> None:
        source = window()
        with self.assertRaises(SubjectSplitError):
            attach_pilot_window_provenance([source, deepcopy(source)], self.subjects)

    def test_exact_duplicate_signal_cross_split_audit(self) -> None:
        rows = [
            {"window_id": "a", "split": "TRAIN", "pilot_signal_sha256": "same"},
            {"window_id": "b", "split": "LOCKED_TEST", "pilot_signal_sha256": "same"},
        ]
        self.assertEqual(len(cross_split_duplicate_hashes(rows)), 1)
        rows[1]["pilot_signal_sha256"] = "different"
        self.assertEqual(cross_split_duplicate_hashes(rows), [])

    def test_no_absolute_path_required_for_provenance(self) -> None:
        self.assertFalse(contains_local_path({"source_radar_member": "db_records/P001/radar.zlib"}))
        self.assertTrue(contains_local_path({"bad": "/Users/name/data.zip"}))

    def test_profile_constants_and_stable_json_regeneration(self) -> None:
        self.assertEqual(SPLIT_PROFILE_ID, "MMWAVE_SUBJECT_SPLIT_PROFILE_001")
        self.assertEqual(SPLIT_SEED, 20260808)
        first = json.dumps(self.subjects, sort_keys=True, separators=(",", ":"))
        second = json.dumps(assign_subject_splits(list(reversed(self.catalog))), sort_keys=True, separators=(",", ":"))
        self.assertEqual(first, second)

    def test_validator_failure_forces_a5_fail_and_a6_not_ready(self) -> None:
        self.assertEqual(derive_gate(False), ("FAIL", "NOT_READY"))


if __name__ == "__main__":
    unittest.main()
