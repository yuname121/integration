# Phase A3 — Canonical Timeline, Resampling Policy, Gap Handling, and 30-Second Window Contract Report

- **Author**: Antigravity Implementation Engineer
- **Date**: 2026-08-07
- **Target Repository**: `https://github.com/sheepmeat/test.git`
- **Branch**: `feature/phase-a3-timeline-window-policy`
- **Phase A3 Gate**: `PASS_WITH_WARNINGS`
- **Phase A4 Entry Status**: `READY_WITH_CONDITIONS`

---

## 1. Executive Summary

Phase A3 of the SafeNest mmWave real-data reconstruction pipeline converts continuous canonical unwrapped radar phase (established in Phase A2) and its associated raw radar timestamps into a deterministic, reproducible time-domain contract.

Following user feedback on initial commit `859cd52`, Phase A3 incorporates three targeted refinements:
1. **Unambiguous `[start, end)` Timestamp Semantics**: Separates `start_timestamp` ($t_{start}$), `last_sample_timestamp` ($t_{last\_sample} = t_{start} + 29.9\text{ s}$), and `end_timestamp_exclusive` ($t_{end\_exclusive} = t_{start} + 30.0\text{ s}$).
2. **Canonical Resampling Grid & Provenance**: Enforces step-wise regular sampling grid $t_k = k \cdot 0.1\text{ s}$ and generates canonical ISO timestamp strings mapped 1-to-1 with resampled phase indices. Activates active validator sampling-rate verification.
3. **Streamlined Warning Exceptions**: Consolidates duplicate `INCOMPLETE_TAIL` exceptions into exactly 11 `WARNING` severity entries matching the 11 warning-bearing recordings.

A3 establishes:
1. An empirical sampling assessment confirming native 10.0 Hz frame sampling across all 13 pilot recordings.
2. A strict timing jitter policy with a 5 ms tolerance (`0.005 s`).
3. Numerical gap thresholds (`NORMAL`: $\le 0.105$ s, `SMALL_GAP`: $\le 0.5$ s, `LARGE_GAP`: $> 0.5$ s).
4. A deterministic 30-second canonical windowing policy producing exactly 300 samples per window at 10 Hz with zero overlap.
5. Incomplete-tail sample accounting (`DROP_INCOMPLETE_TAIL`), preserving dropped sample counts without silent discarding or data manipulation.
6. A label-independent window manifest linking every window deterministically to its source recording and sample indices.

---

## 2. Git Baseline

- **Repository Root**: canonical repository root containing `AGENTS.md`
- **Branch**: `feature/phase-a3-timeline-window-policy`
- **Base Commit**: `b935b0d` (merged Phase A2)
- **Raw Archive SHA-256 (Pre/Post)**: `f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0` (Unchanged)

---

## 3. Approved A2 Input Contract

Phase A3 consumes the approved Phase A2 extraction baseline (`MMWAVE_PHASE_EXTRACTION_PROFILE_001`):
- **Phase Data**: Unfiltered, unnormalized, continuous unwrapped phase in radians (`np.angle` $\rightarrow$ `np.unwrap`).
- **Target Selection**: Range-bin 2 ($0.6345$ m) from stored `rBins`.
- **Channel Selection**: Anonymous virtual antenna channel 6 (or single best consensus channel).
- **Label Independence**: Range-bin selection and phase extraction performed without inspecting subject annotations or class labels.

---

## 4. Pilot Composition

The A3 pilot preserves the approved 13-recording pilot dataset across 6 subjects:
- **Subjects**: `p001`, `p002`, `p004`, `p007`, `p075`, `p110` (6 subjects).
- **Postures**: `Lying`, `Sitting`.
- **Activities**: `Rest`, `Post-exercise`.
- **Frame Lengths**:
  - `400` frames (2 recordings: `p007-sitting-post_exercise`, `p075-sitting-rest`)
  - `500` frames (9 recordings: `p001-...` [4], `p002-...` [1], `p110-...` [4])
  - `600` frames (2 recordings: `p004-lying-post_exercise`, `p004-lying-rest`)

---

## 5. Timestamp Statistics

Raw ISO-8601 timestamps were extracted from `radar_timestamps.csv` for all 13 pilot recordings:
- **Timestamp Format**: ISO-8601 headerless UTF-8 text (nanosecond precision).
- **Monotonicity**: 100% monotonic across all 13 pilot recordings.
- **Duplicate Timestamps** ($\Delta t = 0$): 0 observed.
- **Backward Timestamps** ($\Delta t < 0$): 0 observed.
- **Median $\Delta t$**: $0.100000000$ s ($10.000000$ Hz empirical sampling rate).
- **Min / Max $\Delta t$**: $0.100000000$ s / $0.100000000$ s across all 6,400 raw frames.

---

## 6. Native Sampling Assessment

All 13 pilot recordings exhibit exact 10.0 Hz frame timing ($\Delta t = 0.1$ s) from hardware acquisition without missing frames or clock drift.

- **Decision Code**: `NATIVE_10HZ_NO_RESAMPLING`
- **Native 10 Hz Preferred**: `True`
- **Resampling Required**: `False`
- **Resampling Performed**: `False`
- **Interpolated Samples**: 0

---

## 7. Jitter Analysis

Jitter deviation from target $0.1$ s:
- **Median Absolute Timing Error**: $0.000000000$ s
- **Maximum Absolute Timing Error**: $0.000000000$ s
- **Percentiles (p50, p90, p95, p99)**: $0.000000000$ s
- **Jitter Tolerance**: $0.005$ s ($5$ ms)
- **Fraction Outside Tolerance**: $0.000000$

---

## 8. Gap Policy

Numerical gap thresholds defined under `MMWAVE_TIMELINE_PROFILE_001`:
- **Normal $\Delta t$ Threshold**: $\le 0.105$ s (`NORMAL`)
- **Small-Gap Threshold**: $0.105\text{ s} < \Delta t \le 0.500\text{ s}$ (`SMALL_GAP`, bounded linear interpolation permitted if resampling required)
- **Large-Gap Threshold**: $> 0.500$ s (`LARGE_GAP`, interpolation prohibited; invalidates overlapping windows)
- **Observed Small Gaps**: 0
- **Observed Large Gaps**: 0

---

## 9. Resampling Decision & Canonical Grid Construction

- **Grid Formula**: $t_k = k \cdot 0.1\text{ s}$ for $k \in [0, \lfloor \text{duration} / 0.1 \rfloor]$.
- **Native Grid Preserved**: Raw phase and timestamp alignment preserved without synthetic interpolation when native timing is exact.
- **Resampling Infrastructure**: `scripts/mmwave_timeline.py` includes step-wise grid construction and canonical ISO timestamp formatting (`format_canonical_iso`) ensuring 1-to-1 alignment between resampled phase indices and timestamp strings.

---

## 10. Canonical 10 Hz Timeline Policy

Profile ID: `MMWAVE_TIMELINE_PROFILE_001`
```json
{
  "profile_id": "MMWAVE_TIMELINE_PROFILE_001",
  "target_sampling_rate_hz": 10.0,
  "expected_dt_seconds": 0.1,
  "native_timeline_preferred": true,
  "jitter_policy": {
    "tolerance_seconds": 0.005,
    "normal_max_dt_seconds": 0.105
  },
  "gap_policy": {
    "small_gap_max_seconds": 0.5,
    "large_gap_min_seconds": 0.5
  },
  "resampling": {
    "enabled_when_required": true,
    "method": "LINEAR_INTERPOLATION",
    "extrapolation_allowed": false,
    "large_gap_interpolation_allowed": false
  },
  "window": {
    "duration_seconds": 30.0,
    "samples": 300,
    "stride_samples": 300,
    "overlap_samples": 0,
    "boundary_convention": "[start,end)",
    "incomplete_tail_policy": "DROP_INCOMPLETE_TAIL"
  },
  "label_independent": true
}
```

---

## 11. 30-Second Window Contract

- **Window Duration**: $30.0$ seconds
- **Target Rate**: $10.0$ Hz
- **Samples per Window**: $300$ samples
- **Window Stride**: $300$ samples
- **Overlap**: $0$ samples (non-overlapping canonical baseline)
- **Boundary Convention**: `[start_index, end_index_exclusive)`
- **Timestamp Semantics**:
  - `start_timestamp`: ISO string for $t_{start}$ (sample 0)
  - `last_sample_timestamp`: ISO string for $t_{start} + 29.9\text{ s}$ (sample 299)
  - `end_timestamp_exclusive`: ISO string for $t_{start} + 30.0\text{ s}$ (exclusive boundary)

---

## 12. Incomplete-Tail Handling

Recordings with frame counts not divisible by 300 have their trailing samples dropped according to `DROP_INCOMPLETE_TAIL`:
- **400-sample recordings** (2): 1 full window (300 samples) + 100 dropped tail samples.
- **500-sample recordings** (9): 1 full window (300 samples) + 200 dropped tail samples.
- **600-sample recordings** (2): 2 full windows (600 samples) + 0 dropped tail samples.
- **Total Valid Windows Cut**: 15 windows ($2 \times 1 + 9 \times 1 + 2 \times 2 = 15$).
- **Total Dropped Tail Samples**: 2,000 samples ($2 \times 100 + 9 \times 200 = 2,000$).

---

## 13. Window Provenance

Every window in `window_manifest.jsonl` contains deterministic lineage metadata:
- `window_id`: `<recording_id>__W<window_index:04d>`
- `recording_id`: Parent recording identifier.
- `subject_id`: Subject identifier.
- `timeline_profile`: `MMWAVE_TIMELINE_PROFILE_001`
- `phase_profile`: `MMWAVE_PHASE_EXTRACTION_PROFILE_001`
- `canonical_start_index` & `canonical_end_index_exclusive`: `[0, 300)` or `[300, 600)`
- `start_timestamp`, `last_sample_timestamp`, `end_timestamp_exclusive`: Exact ISO-8601 timestamp strings.

---

## 14. Quality Flags

Machine-readable quality flags assigned:
- `TIMELINE_EXACT_NATIVE_10HZ`: Assigned to all 13 pilot recordings.
- `INCOMPLETE_TAIL_DROPPED`: Assigned to the 11 recordings with dropped tail samples.

---

## 15. Exceptions

A total of 11 non-blocking `INCOMPLETE_TAIL` warning exceptions were recorded for the 11 recordings with dropped tail samples. 0 errors, 0 blockers.

---

## 16. Validation

The standalone validator `scripts/validate_mmwave_timeline_pilot.py` verified all 15 structural and timing rules:
1. Every A3 recording references an A2 pilot recording.
2. Every window references a valid recording.
3. Phase/timestamp lengths agree before timeline transformation.
4. Timestamps are monotonic.
5. Duplicate/backward counts match measured values.
6. Target sampling rate matches profile and grid counts are verified.
7. All valid canonical windows have exactly 300 samples and contain required timestamp fields (`start_timestamp`, `last_sample_timestamp`, `end_timestamp_exclusive`).
8. Window IDs are unique.
9. Window boundaries are deterministic and non-overlapping.
10. No window crosses a prohibited large gap.
11. No label fields were introduced in window manifest.
12. Interpolation counts match detailed data.
13. Dropped-tail counts match recording lengths.
14. Summary counts match detailed manifests.
15. Final A3 gate matches live validator result.

---

## 17. A3 Gate

- **A3 Gate Status**: `PASS_WITH_WARNINGS`
- **Reason**: All timeline reconstruction, gap handling, and 30s windowing rules are deterministic and fully validated. The gate includes non-blocking warnings for dropped incomplete tail samples (2,000 samples total) and limited pilot scope (13 recordings).

---

## 18. A4 Entry Decision

- **A4 Entry Status**: `READY_WITH_CONDITIONS`
- **Conditions**:
  1. A4 must consume `[start_timestamp, end_timestamp_exclusive)` from `window_manifest.jsonl` when checking annotation overlap.
  2. A4 must preserve label independence when performing label mapping.
  3. Tail samples dropped in A3 must not be padded or artificially synthesized in A4.

---

## 19. Remaining Limitations

1. Pilot scope is limited to 13 recordings; full dataset conversion across all 440 recordings is deferred.
2. Physical TX/RX antenna mapping remains unverified (preserved from A2).

---

## 20. Explicit Non-Scope Confirmation

As required, the following tasks were **NOT** performed in Phase A3:

```text
SafeNest label mapping: NOT PERFORMED
Breath-hold/APNEA mapping: NOT PERFORMED
Subject splitting: NOT PERFORMED
Full 440-recording conversion: NOT PERFORMED
Training NPZ generation: NOT PERFORMED
Permanent BPF: NOT PERFORMED
Z-score normalization: NOT PERFORMED
Model training: NOT PERFORMED
A4: NOT PERFORMED
```

---

## 21. Files Changed

- `scripts/mmwave_timeline.py`: Canonical timeline reconstruction, resampling policy, gap handling, and 30s window generator module.
- `scripts/run_mmwave_timeline_pilot.py`: Phase A3 pilot runner script.
- `scripts/validate_mmwave_timeline_pilot.py`: Phase A3 in-memory and standalone validator script.
- `tests/test_mmwave_timeline.py`: Unit tests for timeline reconstruction, jitter, gaps, resampling, and windowing.
- `datasets/mmwave/manifests/a3_timeline_pilot/`: Manifest output directory (`pilot_selection.json`, `timeline_profile.json`, `recording_timeline_results.jsonl`, `window_manifest.jsonl`, `exceptions.json`, `a3_summary.json`, `checksums.sha256`).
- `docs/reports/20260807_Antigravity_A3_Timeline_Window_Policy_Pilot_01.md`: This report.

---

## 22. Verification Commands and Test Execution

```bash
# 1. Run Phase A3 Pilot and Validator
python3 scripts/run_mmwave_timeline_pilot.py
python3 scripts/validate_mmwave_timeline_pilot.py

# 2. Run Unit Tests and A2 Regression Tests
python3 -m unittest tests/test_mmwave_timeline.py -v
python3 -m unittest tests/test_mmwave_phase_extractor.py -v

# 3. Confirm Deterministic Regeneration
python3 scripts/run_mmwave_timeline_pilot.py
```
