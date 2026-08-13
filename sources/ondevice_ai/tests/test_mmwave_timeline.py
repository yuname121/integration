#!/usr/bin/env python3
"""Unit tests for Phase A3 timeline reconstruction, gap handling, and windowing."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import unittest

from scripts.mmwave_timeline import (
    TimelineError,
    TimelineProfile,
    analyze_timeline,
    evaluate_resampling_decision,
    format_canonical_iso,
    generate_30s_windows,
    parse_timestamps_to_seconds,
    process_recording_timeline,
    resample_timeline,
)
from scripts.validate_mmwave_timeline_pilot import derive_gate, validate_manifests


def make_iso_timestamps(start_iso: str, count: int, dt_sec: float = 0.1) -> list[str]:
    start_dt = dt.datetime.fromisoformat(start_iso)
    result = []
    for i in range(count):
        curr = start_dt + dt.timedelta(seconds=i * dt_sec)
        result.append(curr.isoformat() + "Z")
    return result


class TestMmwaveTimeline(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = TimelineProfile()

    # 1. Exact 10 Hz native timeline
    def test_exact_10hz_native_timeline(self) -> None:
        ts_iso = make_iso_timestamps("2025-02-20T12:00:00.000", 300, 0.1)
        sec, lines, meta = parse_timestamps_to_seconds(ts_iso)
        analysis = analyze_timeline(sec, self.profile)

        self.assertEqual(analysis["empirical_sampling_rate_hz"], 10.0)
        self.assertEqual(analysis["duplicate_timestamp_count"], 0)
        self.assertEqual(analysis["backward_timestamp_count"], 0)
        self.assertEqual(analysis["small_gap_count"], 0)
        self.assertEqual(analysis["large_gap_count"], 0)

        decision = evaluate_resampling_decision(analysis, self.profile)
        self.assertFalse(decision["resampling_required"])
        self.assertEqual(decision["decision_code"], "NATIVE_10HZ_NO_RESAMPLING")

    # 2. Empirical dt calculation
    def test_empirical_dt_calculation(self) -> None:
        ts_iso = make_iso_timestamps("2025-02-20T12:00:00.000", 5, 0.1)
        sec, _, _ = parse_timestamps_to_seconds(ts_iso)
        analysis = analyze_timeline(sec, self.profile)

        self.assertAlmostEqual(analysis["median_dt_seconds"], 0.1)
        self.assertAlmostEqual(analysis["mean_dt_seconds"], 0.1)
        self.assertAlmostEqual(analysis["min_dt_seconds"], 0.1)
        self.assertAlmostEqual(analysis["max_dt_seconds"], 0.1)

    # 3. Small jitter within policy
    def test_small_jitter_within_policy(self) -> None:
        # dt = 0.102 (within 0.005 s tolerance)
        ts_dt = [dt.datetime(2025, 2, 20, 12, 0, 0) + dt.timedelta(seconds=i * 0.102) for i in range(10)]
        ts_iso = [t.isoformat() for t in ts_dt]
        sec, _, _ = parse_timestamps_to_seconds(ts_iso)
        analysis = analyze_timeline(sec, self.profile)

        self.assertLessEqual(analysis["max_abs_jitter_seconds"], 0.005)
        decision = evaluate_resampling_decision(analysis, self.profile)
        self.assertFalse(decision["resampling_required"])

    # 4. Duplicate timestamp detection
    def test_duplicate_timestamp_detection(self) -> None:
        ts_iso = [
            "2025-02-20T12:00:00.000000000Z",
            "2025-02-20T12:00:00.100000000Z",
            "2025-02-20T12:00:00.100000000Z",  # Duplicate
            "2025-02-20T12:00:00.300000000Z",
        ]
        sec, _, _ = parse_timestamps_to_seconds(ts_iso)
        analysis = analyze_timeline(sec, self.profile)

        self.assertEqual(analysis["duplicate_timestamp_count"], 1)
        decision = evaluate_resampling_decision(analysis, self.profile)
        self.assertEqual(decision["decision_code"], "RECORDING_NOT_SAFELY_RESAMPLEABLE")

    # 5. Backward timestamp detection
    def test_backward_timestamp_detection(self) -> None:
        ts_iso = [
            "2025-02-20T12:00:00.000000000Z",
            "2025-02-20T12:00:00.200000000Z",
            "2025-02-20T12:00:00.100000000Z",  # Backward
            "2025-02-20T12:00:00.300000000Z",
        ]
        sec, _, _ = parse_timestamps_to_seconds(ts_iso)
        analysis = analyze_timeline(sec, self.profile)

        self.assertEqual(analysis["backward_timestamp_count"], 1)
        decision = evaluate_resampling_decision(analysis, self.profile)
        self.assertEqual(decision["decision_code"], "RECORDING_NOT_SAFELY_RESAMPLEABLE")

    # 6. Small gap detection
    def test_small_gap_detection(self) -> None:
        ts_iso = [
            "2025-02-20T12:00:00.000Z",
            "2025-02-20T12:00:00.100Z",
            "2025-02-20T12:00:00.400Z",  # dt = 0.3s -> SMALL_GAP (<= 0.5s)
            "2025-02-20T12:00:00.500Z",
        ]
        sec, _, _ = parse_timestamps_to_seconds(ts_iso)
        analysis = analyze_timeline(sec, self.profile)

        self.assertEqual(analysis["small_gap_count"], 1)
        self.assertEqual(analysis["large_gap_count"], 0)
        decision = evaluate_resampling_decision(analysis, self.profile)
        self.assertTrue(decision["resampling_required"])
        self.assertEqual(decision["decision_code"], "RESAMPLING_PERFORMED")

    # 7. Large gap detection
    def test_large_gap_detection(self) -> None:
        ts_iso = [
            "2025-02-20T12:00:00.000Z",
            "2025-02-20T12:00:00.100Z",
            "2025-02-20T12:00:01.000Z",  # dt = 0.9s -> LARGE_GAP (> 0.5s)
            "2025-02-20T12:00:01.100Z",
        ]
        sec, _, _ = parse_timestamps_to_seconds(ts_iso)
        analysis = analyze_timeline(sec, self.profile)

        self.assertEqual(analysis["large_gap_count"], 1)
        decision = evaluate_resampling_decision(analysis, self.profile)
        self.assertFalse(decision["resampling_permissible"])
        self.assertEqual(decision["decision_code"], "LARGE_GAP_PRESENT_NO_RESAMPLING")

    # 8 & 9. Bounded interpolation behavior without crossing large gap
    def test_bounded_interpolation(self) -> None:
        sec_list = [i * 0.1 for i in range(50)] + [(50 + 2) * 0.1 + i * 0.1 for i in range(50)]
        sec = np.array(sec_list, dtype=np.float64)
        phase = np.sin(2 * np.pi * 0.2 * sec)
        first_dt = dt.datetime(2025, 2, 20, 12, 0, 0)

        analysis = analyze_timeline(sec, self.profile)
        res_phase, res_sec, canonical_iso, meta = resample_timeline(phase, sec, first_dt, self.profile, analysis)

        self.assertTrue(meta["resampling_performed"])
        self.assertGreater(meta["interpolated_sample_count"], 0)
        self.assertEqual(len(res_phase), len(canonical_iso))

    # 10. Deterministic regular 10 Hz grid
    def test_deterministic_regular_grid(self) -> None:
        sec = np.array([0.0, 0.102, 0.205, 0.301, 0.400], dtype=np.float64)
        phase = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        first_dt = dt.datetime(2025, 2, 20, 12, 0, 0)
        analysis = analyze_timeline(sec, self.profile)
        res_phase1, res_sec1, iso1, _ = resample_timeline(phase, sec, first_dt, self.profile, analysis)
        res_phase2, res_sec2, iso2, _ = resample_timeline(phase, sec, first_dt, self.profile, analysis)

        np.testing.assert_array_equal(res_phase1, res_phase2)
        np.testing.assert_array_equal(res_sec1, res_sec2)
        self.assertEqual(iso1, iso2)

    # 11. Exact 300-sample window and timestamp semantics
    def test_exact_300_sample_window_and_timestamp_semantics(self) -> None:
        ts_iso = make_iso_timestamps("2025-02-20T12:00:00.000", 300, 0.1)
        phase = np.linspace(-1.0, 1.0, 300)
        rec_res, windows, exceptions = process_recording_timeline(
            phase=phase,
            timestamps_raw=ts_iso,
            recording_id="rec-300",
            subject_id="subj-1",
            extraction_profile_id="EXT_001",
            profile=self.profile,
        )

        self.assertEqual(len(windows), 1)
        w = windows[0]
        self.assertEqual(w["sample_count"], 300)
        self.assertEqual(w["start_timestamp"], "2025-02-20T12:00:00+00:00")
        self.assertEqual(w["last_sample_timestamp"], "2025-02-20T12:00:29.900000+00:00")
        self.assertEqual(w["end_timestamp_exclusive"], "2025-02-20T12:00:30+00:00")
        self.assertEqual(rec_res["dropped_tail_samples"], 0)

    # 12. 400-sample recording -> 1 window + 100 tail samples
    def test_400_sample_recording(self) -> None:
        ts_iso = make_iso_timestamps("2025-02-20T12:00:00.000", 400, 0.1)
        phase = np.linspace(-1.0, 1.0, 400)
        rec_res, windows, exceptions = process_recording_timeline(
            phase=phase,
            timestamps_raw=ts_iso,
            recording_id="rec-400",
            subject_id="subj-1",
            extraction_profile_id="EXT_001",
            profile=self.profile,
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["sample_count"], 300)
        self.assertEqual(rec_res["dropped_tail_samples"], 100)
        self.assertIn("INCOMPLETE_TAIL_DROPPED", rec_res["quality_flags"])
        self.assertEqual(len(exceptions), 1)
        self.assertEqual(exceptions[0]["severity"], "WARNING")

    # 13. 500-sample recording -> 1 window + 200 tail samples
    def test_500_sample_recording(self) -> None:
        ts_iso = make_iso_timestamps("2025-02-20T12:00:00.000", 500, 0.1)
        phase = np.linspace(-1.0, 1.0, 500)
        rec_res, windows, exceptions = process_recording_timeline(
            phase=phase,
            timestamps_raw=ts_iso,
            recording_id="rec-500",
            subject_id="subj-1",
            extraction_profile_id="EXT_001",
            profile=self.profile,
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["sample_count"], 300)
        self.assertEqual(rec_res["dropped_tail_samples"], 200)

    # 14. 600-sample recording -> 2 windows + 0 tail samples
    def test_600_sample_recording(self) -> None:
        ts_iso = make_iso_timestamps("2025-02-20T12:00:00.000", 600, 0.1)
        phase = np.linspace(-1.0, 1.0, 600)
        rec_res, windows, exceptions = process_recording_timeline(
            phase=phase,
            timestamps_raw=ts_iso,
            recording_id="rec-600",
            subject_id="subj-1",
            extraction_profile_id="EXT_001",
            profile=self.profile,
        )

        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["sample_count"], 300)
        self.assertEqual(windows[1]["sample_count"], 300)
        self.assertEqual(windows[0]["canonical_start_index"], 0)
        self.assertEqual(windows[0]["canonical_end_index_exclusive"], 300)
        self.assertEqual(windows[1]["canonical_start_index"], 300)
        self.assertEqual(windows[1]["canonical_end_index_exclusive"], 600)
        self.assertEqual(windows[0]["start_timestamp"], "2025-02-20T12:00:00+00:00")
        self.assertEqual(windows[0]["last_sample_timestamp"], "2025-02-20T12:00:29.900000+00:00")
        self.assertEqual(windows[0]["end_timestamp_exclusive"], "2025-02-20T12:00:30+00:00")
        self.assertEqual(windows[1]["start_timestamp"], "2025-02-20T12:00:30+00:00")
        self.assertEqual(windows[1]["last_sample_timestamp"], "2025-02-20T12:00:59.900000+00:00")
        self.assertEqual(windows[1]["end_timestamp_exclusive"], "2025-02-20T12:01:00+00:00")
        self.assertEqual(rec_res["dropped_tail_samples"], 0)

    # 15. Incomplete tail accounting
    def test_incomplete_tail_accounting(self) -> None:
        ts_iso = make_iso_timestamps("2025-02-20T12:00:00.000", 450, 0.1)
        phase = np.linspace(-1.0, 1.0, 450)
        rec_res, windows, exceptions = process_recording_timeline(
            phase=phase,
            timestamps_raw=ts_iso,
            recording_id="rec-450",
            subject_id="subj-1",
            extraction_profile_id="EXT_001",
            profile=self.profile,
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(rec_res["dropped_tail_samples"], 150)

    # 16. Deterministic window IDs
    def test_deterministic_window_ids(self) -> None:
        ts_iso = make_iso_timestamps("2025-02-20T12:00:00.000", 600, 0.1)
        phase = np.linspace(-1.0, 1.0, 600)
        _, windows1, _ = process_recording_timeline(
            phase, ts_iso, "rec-det", "subj-1", "EXT_001", self.profile
        )
        _, windows2, _ = process_recording_timeline(
            phase, ts_iso, "rec-det", "subj-1", "EXT_001", self.profile
        )

        self.assertEqual([w["window_id"] for w in windows1], ["rec-det__W0000", "rec-det__W0001"])
        self.assertEqual(
            [w["window_id"] for w in windows1], [w["window_id"] for w in windows2]
        )

    # 17. Label-free window manifest
    def test_label_free_window_manifest(self) -> None:
        ts_iso = make_iso_timestamps("2025-02-20T12:00:00.000", 300, 0.1)
        phase = np.linspace(-1.0, 1.0, 300)
        _, windows, _ = process_recording_timeline(
            phase, ts_iso, "rec-labelfree", "subj-1", "EXT_001", self.profile
        )
        w = windows[0]
        forbidden = {"NORMAL", "RAPID_OR_ABNORMAL", "APNEA", "label", "labels", "target", "targets", "class"}
        self.assertEqual(forbidden.intersection(w.keys()), set())

    # 18. Validator failure -> A3 FAIL / A4 NOT_READY
    def test_validator_failure_triggers_fail_gate(self) -> None:
        gate, ready = derive_gate(False, [], [])
        self.assertEqual(gate, "FAIL")
        self.assertEqual(ready, "NOT_READY")

    # 19. Deterministic regeneration
    def test_deterministic_regeneration(self) -> None:
        ts_iso = make_iso_timestamps("2025-02-20T12:00:00.000", 500, 0.1)
        phase = np.sin(np.linspace(0, 10, 500))

        res1, wins1, exc1 = process_recording_timeline(
            phase, ts_iso, "rec-regen", "subj-1", "EXT_001", self.profile
        )
        res2, wins2, exc2 = process_recording_timeline(
            phase, ts_iso, "rec-regen", "subj-1", "EXT_001", self.profile
        )

        self.assertEqual(json.dumps(res1, sort_keys=True), json.dumps(res2, sort_keys=True))
        self.assertEqual(json.dumps(wins1, sort_keys=True), json.dumps(wins2, sort_keys=True))
        self.assertEqual(json.dumps(exc1, sort_keys=True), json.dumps(exc2, sort_keys=True))

    # 20. Resampling provenance source vs canonical index mapping
    def test_resampling_source_vs_canonical_index_mapping(self) -> None:
        # Native 8 Hz recording for 40 seconds -> dt = 0.125s, 321 samples
        ts_iso = make_iso_timestamps("2025-02-20T12:00:00.000", 321, 0.125)
        phase = np.sin(np.linspace(0, 10, 321))

        rec_res, windows, exceptions = process_recording_timeline(
            phase=phase,
            timestamps_raw=ts_iso,
            recording_id="rec-resampled-8hz",
            subject_id="subj-resample",
            extraction_profile_id="EXT_001",
            profile=self.profile,
        )

        self.assertTrue(rec_res["resampling_performed"])
        self.assertEqual(rec_res["source_sample_count"], 321)
        self.assertEqual(rec_res["canonical_sample_count"], 401)  # 40s @ 10Hz = 401 samples (0..400)
        self.assertEqual(len(windows), 1)  # 1 full 300-sample window (30s)

        w = windows[0]
        # Canonical indices: [0, 300)
        self.assertEqual(w["canonical_start_index"], 0)
        self.assertEqual(w["canonical_end_index_exclusive"], 300)

        # Source native indices: 30s @ 8Hz = 240 samples -> [0, 240)
        self.assertEqual(w["source_start_index"], 0)
        self.assertEqual(w["source_end_index_exclusive"], 240)
        self.assertNotEqual(w["canonical_end_index_exclusive"], w["source_end_index_exclusive"])

        # Validate with manifest validator
        val_success, val_errors = validate_manifests(
            a2_pilot={"recordings": [{"recording_id": "rec-resampled-8hz"}]},
            profile=self.profile.to_dict(),
            rec_results=[rec_res],
            windows=windows,
            exceptions=exceptions,
            summary={
                "pilot_recording_count": 1,
                "total_window_count": 1,
                "total_dropped_tail_samples": rec_res["dropped_tail_samples"],
                "a3_gate_status": "PASS_WITH_WARNINGS",
                "a4_entry_status": "READY_WITH_CONDITIONS",
            },
        )
        self.assertTrue(val_success, f"Validation failed: {val_errors}")


if __name__ == "__main__":
    unittest.main()
