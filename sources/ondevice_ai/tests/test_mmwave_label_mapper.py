#!/usr/bin/env python3
"""Unit tests for Phase A4 annotation alignment, label mapping, and policy validator."""

from __future__ import annotations

import json
import unittest

from scripts.mmwave_label_mapper import (
    LabelMappingError,
    LabelMappingProfile,
    compute_window_annotation_overlap,
    map_window_label,
    parse_annotation_file,
)
from scripts.validate_mmwave_label_pilot import derive_a4_gate, validate_label_manifests


class TestMmwaveLabelMapper(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = LabelMappingProfile()
        self.base_window = {
            "window_id": "test_rec__W0000",
            "recording_id": "test_rec",
            "subject_id": "test_subj",
            "window_index": 0,
            "canonical_start_index": 0,
            "canonical_end_index_exclusive": 300,
            "start_timestamp": "2025-02-20T12:00:00.000000Z",
            "last_sample_timestamp": "2025-02-20T12:00:29.900000Z",
            "end_timestamp_exclusive": "2025-02-20T12:00:30.000000Z",
            "sample_count": 300,
            "duration_seconds": 30.0,
        }

    # 1. Exact [start,end) annotation overlap
    def test_exact_interval_annotation_overlap(self) -> None:
        events = [
            {
                "event_id": "EVT_1",
                "start_seconds_relative": 10.0,
                "end_seconds_relative": 20.0,
                "duration_seconds": 10.0,
            }
        ]
        info = compute_window_annotation_overlap(0.0, 30.0, events)
        self.assertEqual(info["annotation_overlap_seconds"], 10.0)
        self.assertAlmostEqual(info["annotation_overlap_fraction"], 10.0 / 30.0, places=5)

    # 2. Event exactly on window start
    def test_event_on_window_start(self) -> None:
        events = [
            {
                "event_id": "EVT_1",
                "start_seconds_relative": 0.0,
                "end_seconds_relative": 10.0,
                "duration_seconds": 10.0,
            }
        ]
        info = compute_window_annotation_overlap(0.0, 30.0, events)
        self.assertEqual(info["annotation_overlap_seconds"], 10.0)

    # 3. Event exactly on exclusive window end
    def test_event_on_exclusive_window_end(self) -> None:
        events = [
            {
                "event_id": "EVT_1",
                "start_seconds_relative": 20.0,
                "end_seconds_relative": 30.0,
                "duration_seconds": 10.0,
            }
        ]
        info = compute_window_annotation_overlap(0.0, 30.0, events)
        self.assertEqual(info["annotation_overlap_seconds"], 10.0)

    # 4. Event spanning two windows
    def test_event_spanning_two_windows(self) -> None:
        events = [
            {
                "event_id": "EVT_1",
                "start_seconds_relative": 25.0,
                "end_seconds_relative": 35.0,
                "duration_seconds": 10.0,
            }
        ]
        info_w0 = compute_window_annotation_overlap(0.0, 30.0, events)
        info_w1 = compute_window_annotation_overlap(30.0, 60.0, events)
        self.assertEqual(info_w0["annotation_overlap_seconds"], 5.0)
        self.assertEqual(info_w1["annotation_overlap_seconds"], 5.0)

    # 5. 10-second overlap candidate calculation
    def test_10s_overlap_candidate(self) -> None:
        events = [
            {
                "event_id": "EVT_1",
                "start_seconds_relative": 10.0,
                "end_seconds_relative": 20.5,
                "duration_seconds": 10.5,
            }
        ]
        mapped = map_window_label(self.base_window, events, "Rest", "Lying", self.profile)
        self.assertEqual(mapped["safenest_label"], "APNEA")
        self.assertEqual(mapped["safenest_label_id"], 2)
        self.assertEqual(mapped["mapping_type"], "DERIVED")

    # 6. 15-second / 50% legacy rule is not hardcoded as canonical
    def test_legacy_15s_rule_not_canonical(self) -> None:
        events = [
            {
                "event_id": "EVT_1",
                "start_seconds_relative": 10.0,
                "end_seconds_relative": 21.0,
                "duration_seconds": 11.0,
            }
        ]
        # Overlap is 11.0s (< 15.0s). Under selected profile (>= 6.0s), it maps to APNEA.
        mapped = map_window_label(self.base_window, events, "Rest", "Lying", self.profile)
        self.assertEqual(mapped["safenest_label"], "APNEA")

    # 7. Partial event -> ambiguity behavior
    def test_partial_event_ambiguity_behavior(self) -> None:
        events = [
            {
                "event_id": "EVT_1",
                "start_seconds_relative": 26.0,
                "end_seconds_relative": 36.0,
                "duration_seconds": 10.0,
            }
        ]
        # Overlap with window 0 [0, 30) is 4.0s (0 < overlap < 6s) -> AMBIGUOUS
        mapped = map_window_label(self.base_window, events, "Rest", "Lying", self.profile)
        self.assertIsNone(mapped["safenest_label"])
        self.assertEqual(mapped["mapping_type"], "AMBIGUOUS")
        self.assertEqual(mapped["assignment_status"], "AMBIGUOUS")

    # 8. Transition-window handling
    def test_transition_window_handling(self) -> None:
        events = [
            {
                "event_id": "EVT_1",
                "start_seconds_relative": 27.0,
                "end_seconds_relative": 35.0,
                "duration_seconds": 8.0,
            }
        ]
        mapped = map_window_label(self.base_window, events, "Rest", "Lying", self.profile)
        self.assertEqual(mapped["mapping_rule_id"], "A4_RULE_TRANSITION_WINDOW")
        self.assertIsNone(mapped["safenest_label"])

    # 9. Voluntary breath hold -> APNEA mapping type is DERIVED
    def test_voluntary_breath_hold_mapping_type_derived(self) -> None:
        events = [
            {
                "event_id": "EVT_1",
                "start_seconds_relative": 5.0,
                "end_seconds_relative": 18.0,
                "duration_seconds": 13.0,
            }
        ]
        mapped = map_window_label(self.base_window, events, "Rest", "Lying", self.profile)
        self.assertEqual(mapped["safenest_label"], "APNEA")
        self.assertEqual(mapped["mapping_type"], "DERIVED")

    # 10. No clinical-apnea claim
    def test_no_clinical_apnea_claimed(self) -> None:
        self.assertFalse(self.profile.clinical_apnea_claimed)
        self.assertFalse(self.profile.to_dict()["apnea_policy"]["clinical_apnea_claimed"])

    # 11. Rest does not automatically become DIRECT NORMAL
    def test_rest_mapping_type_is_derived(self) -> None:
        events = []
        mapped = map_window_label(self.base_window, events, "Rest", "Lying", self.profile)
        self.assertEqual(mapped["safenest_label"], "NORMAL")
        self.assertEqual(mapped["mapping_type"], "DERIVED")
        self.assertEqual(mapped["mapping_rule_id"], "A4_RULE_NORMAL_REST_PROXY")

    # 12. Post-exercise alone cannot automatically become RAPID_OR_ABNORMAL
    def test_post_exercise_not_auto_rapid(self) -> None:
        events = []
        mapped = map_window_label(self.base_window, events, "Post-exercise", "Sitting", self.profile)
        self.assertIsNone(mapped["safenest_label"])
        self.assertEqual(mapped["mapping_type"], "AMBIGUOUS")
        self.assertEqual(mapped["assignment_status"], "AMBIGUOUS")
        self.assertEqual(mapped["mapping_rule_id"], "A4_RULE_POST_EXERCISE_UNVERIFIED")

    # 13. Missing physiological evidence -> ambiguous/unmapped behavior
    def test_missing_evidence_ambiguous_unmapped(self) -> None:
        mapped = map_window_label(self.base_window, [], "Post-exercise", "Lying", self.profile)
        self.assertEqual(mapped["assignment_status"], "AMBIGUOUS")
        self.assertIn("respiration rate", mapped["ambiguity_reasons"][0])

    # 14. Dropped-tail annotation accounting
    def test_dropped_tail_annotation_accounting(self) -> None:
        events = [
            {
                "event_id": "EVT_1",
                "start_seconds_relative": 25.0,
                "end_seconds_relative": 35.0,
                "duration_seconds": 10.0,
            }
        ]
        # In A3 30s window (cutoff at 30.0s), 5s is inside window, 5s is lost in tail
        info = compute_window_annotation_overlap(0.0, 30.0, events)
        self.assertEqual(info["annotation_overlap_seconds"], 5.0)

    # 15. Original condition preserved separately
    def test_original_condition_preserved_separately(self) -> None:
        mapped = map_window_label(self.base_window, [], "Rest", "Lying", self.profile)
        self.assertEqual(mapped["source_test_condition"], "Rest")
        self.assertEqual(mapped["posture"], "Lying")
        self.assertEqual(mapped["safenest_label"], "NORMAL")

    # 16. Deterministic mapping
    def test_deterministic_mapping(self) -> None:
        events = [
            {
                "event_id": "EVT_1",
                "start_seconds_relative": 10.0,
                "end_seconds_relative": 22.0,
                "duration_seconds": 12.0,
            }
        ]
        mapped1 = map_window_label(self.base_window, events, "Rest", "Lying", self.profile)
        mapped2 = map_window_label(self.base_window, events, "Rest", "Lying", self.profile)
        self.assertEqual(json.dumps(mapped1, sort_keys=True), json.dumps(mapped2, sort_keys=True))

    # 17. A3 window timestamps remain unchanged
    def test_a3_window_timestamps_unchanged(self) -> None:
        mapped = map_window_label(self.base_window, [], "Rest", "Lying", self.profile)
        self.assertEqual(mapped["start_timestamp"], self.base_window["start_timestamp"])
        self.assertEqual(mapped["last_sample_timestamp"], self.base_window["last_sample_timestamp"])
        self.assertEqual(mapped["end_timestamp_exclusive"], self.base_window["end_timestamp_exclusive"])

    # 18. No split fields in window mapping
    def test_no_split_fields(self) -> None:
        mapped = map_window_label(self.base_window, [], "Rest", "Lying", self.profile)
        forbidden = {"split", "train_val_test", "is_train", "model_prediction"}
        self.assertEqual(forbidden.intersection(mapped.keys()), set())

    # 19. Validator failure -> A4 FAIL / A5 NOT_READY
    def test_validator_failure_triggers_fail(self) -> None:
        gate, ready = derive_a4_gate(False, [], [])
        self.assertEqual(gate, "FAIL")
        self.assertEqual(ready, "NOT_READY")

    # 20. Deterministic artifact regeneration
    def test_deterministic_artifact_regeneration(self) -> None:
        events = [
            {
                "event_id": "EVT_1",
                "start_seconds_relative": 5.0,
                "end_seconds_relative": 17.0,
                "duration_seconds": 12.0,
            }
        ]
        res1 = map_window_label(self.base_window, events, "Rest", "Sitting", self.profile)
        res2 = map_window_label(self.base_window, events, "Rest", "Sitting", self.profile)
        self.assertEqual(json.dumps(res1, sort_keys=True), json.dumps(res2, sort_keys=True))

    # 21. Movesense chest ACC reference respiration rate >= 25 bpm -> RAPID_OR_ABNORMAL
    def test_movesense_acc_rapid_respiration(self) -> None:
        acc_info = {"rr_bpm": 30.0, "peak_freq_hz": 0.50, "reference_sensor": "MOVESENSE_CHEST_ACC"}
        mapped = map_window_label(
            self.base_window, [], "Post-exercise", "Lying", profile=self.profile, movesense_rr_info=acc_info
        )
        self.assertEqual(mapped["safenest_label"], "RAPID_OR_ABNORMAL")
        self.assertEqual(mapped["safenest_label_id"], 1)
        self.assertEqual(mapped["mapping_type"], "DERIVED")
        self.assertEqual(mapped["mapping_rule_id"], "A4_RULE_RAPID_MOVESENSE_ACC_REF")

    # 22. Movesense chest ACC reference respiration rate 10 <= RR < 25 bpm -> NORMAL
    def test_movesense_acc_normal_respiration(self) -> None:
        acc_info = {"rr_bpm": 15.0, "peak_freq_hz": 0.25, "reference_sensor": "MOVESENSE_CHEST_ACC"}
        mapped = map_window_label(
            self.base_window, [], "Post-exercise", "Sitting", profile=self.profile, movesense_rr_info=acc_info
        )
        self.assertEqual(mapped["safenest_label"], "NORMAL")
        self.assertEqual(mapped["safenest_label_id"], 0)
        self.assertEqual(mapped["mapping_type"], "DERIVED")
        self.assertEqual(mapped["mapping_rule_id"], "A4_RULE_NORMAL_MOVESENSE_ACC_REF")

    # 23. Movesense chest ACC reference respiration rate < 10 bpm (bradypnea) -> RAPID_OR_ABNORMAL
    def test_movesense_acc_bradypnea_respiration(self) -> None:
        acc_info = {"rr_bpm": 8.0, "peak_freq_hz": 0.133, "reference_sensor": "MOVESENSE_CHEST_ACC"}
        mapped = map_window_label(
            self.base_window, [], "Post-exercise", "Sitting", profile=self.profile, movesense_rr_info=acc_info
        )
        self.assertEqual(mapped["safenest_label"], "RAPID_OR_ABNORMAL")
        self.assertEqual(mapped["safenest_label_id"], 1)
        self.assertEqual(mapped["mapping_type"], "DERIVED")
        self.assertEqual(mapped["mapping_rule_id"], "A4_RULE_ABNORMAL_BRADYPNEA_MOVESENSE_ACC_REF")


if __name__ == "__main__":
    unittest.main()
