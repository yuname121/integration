# Phase A4 — Annotation Alignment, SafeNest Label Policy Analysis, and Deterministic Window Label Mapping Report

- **Author**: Antigravity Implementation Engineer
- **Date**: 2026-08-08
- **Target Repository**: `https://github.com/sheepmeat/test.git`
- **Branch**: `feature/phase-a4-annotation-label-mapping`
- **Phase A4 Gate**: `PASS_WITH_WARNINGS`
- **Phase A5 Entry Status**: `READY_WITH_CONDITIONS`

---

## 1. Executive Summary

Phase A4 of the SafeNest mmWave real-data reconstruction pipeline establishes the semantic mapping between original dataset test conditions / non-breathing annotations / Movesense chest accelerometer reference respiration rates and the SafeNest target classes (`NORMAL`, `RAPID_OR_ABNORMAL`, `APNEA`).

Key outcomes of Phase A4:
1. **Empirical Annotation Audit & Terminology Alignment**: All 6 annotation-bearing recordings in the 13-recording pilot contain headerless 2-line `non_breathing_ts.csv` files with microsecond storage precision. Annotation temporal accuracy is documented as `NOT_QUANTIFIED` (push-button timestamps adjusted post-hoc via reference accelerometer).
2. **Re-Evaluation & Rejection of the Legacy 15-Second Rule**: Empirical duration analysis revealed that all voluntary breath-hold events in the dataset last between $9.771$ s and $12.205$ s (mean $11.321$ s). Requiring $\ge 15.0$ seconds of non-breathing overlap would discard 100% of all non-breathing events in the dataset.
3. **Selected APNEA Proxy Policy**: Established `MMWAVE_LABEL_MAPPING_PROFILE_001` requiring $\ge 6.0$ seconds of non-breathing annotation overlap within a 30-second window to assign the `APNEA` target label. Mapping type is strictly recorded as `DERIVED` (voluntary breath-hold proxy, never clinical apnea).
4. **Movesense Chest ACC Respiration Reference Integration**: Official Nature Scientific Data documentation (`10.1038/s41597-026-07172-9`) confirms the Movesense chest accelerometer as the independent reference sensor for breathing validation. Spectral analysis over 30s windows using profile-configurable search band `movesense_rr_search_band_hz: [0.1, 0.7]` yielded reference respiration rates:
   - Respiration rate $\ge 25.0$ bpm ($0.417$ Hz) or $< 10.0$ bpm bradypnea $\rightarrow$ `RAPID_OR_ABNORMAL` (`DERIVED`, rules `A4_RULE_RAPID_MOVESENSE_ACC_REF` / `A4_RULE_ABNORMAL_BRADYPNEA_MOVESENSE_ACC_REF`, 3 windows).
   - Respiration rate $10.0 \le \text{RR} < 25.0$ bpm $\rightarrow$ `NORMAL` (`DERIVED`, rule `A4_RULE_NORMAL_MOVESENSE_ACC_REF`, 5 windows).
5. **Corrected Annotation Coverage Audit**: Measured against recording canonical window spans:
   - **Fully represented events**: 2 events (`p004-lying-rest` preserved across W0+W1 union, `p075-sitting-rest` contained inside W0).
   - **Partially represented events**: 4 events (events ending past 30.0s in 500-sample recordings where the tail [30, 50s) was dropped).
   - **Not represented events**: 0 events.
   - Total annotated seconds represented in A3 windows: $54.319$ s ($80.0\%$), lost to dropped tails: $13.606$ s ($20.0\%$).
6. **Deterministic Window Label Manifest**: Produced [`window_label_manifest.jsonl`](../../datasets/mmwave/manifests/a4_label_pilot/window_label_manifest.jsonl) mapping all 15 A3 windows (5 `NORMAL`, 3 `RAPID_OR_ABNORMAL`, 6 `APNEA`, 1 `AMBIGUOUS`).

---

## 2. Git Baseline

- **Repository Root**: canonical repository root containing `AGENTS.md`
- **Branch**: `feature/phase-a4-annotation-label-mapping`
- **Base Commit**: `05c3211` (merged Phase A3)
- **Raw Archive SHA-256 (Pre/Post)**: `f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0` (Unchanged)

---

## 3. Approved A3 Input Contract

Phase A4 consumes the approved Phase A3 timeline and window contract (`MMWAVE_TIMELINE_PROFILE_001`):
- **Windows**: 15 canonical 30-second windows across 13 pilot recordings.
- **Rate & Duration**: $10.0$ Hz, $300$ samples per window, $30.0$ seconds duration, 0 overlap.
- **Timestamp Contract**: `start_timestamp` ($t_{start}$), `last_sample_timestamp` ($t_{start} + 29.9\text{ s}$), `end_timestamp_exclusive` ($t_{start} + 30.0\text{ s}$).

---

## 4. Original Dataset Conditions

The pilot covers 13 recordings across 6 subjects (`p001`, `p002`, `p004`, `p007`, `p075`, `p110`):
- **Rest** (6 recordings): All 6 Rest recordings contain voluntary breath-hold annotations (`non_breathing_ts.csv`).
- **Post-exercise** (7 recordings): None of the Post-exercise recordings contain non-breathing annotations. All 7 contain 3D chest accelerometer reference data (`movesense_acc.csv`).

---

## 5. Annotation File Structure

- **Member Name**: `<source_recording_path>/non_breathing_ts.csv`
- **File Format**: Headerless 2-line text file:
  - Line 1: `begin,<ISO-8601 timestamp string>` (e.g. `begin,2025-02-20 12:24:27.352571`)
  - Line 2: `end,<ISO-8601 timestamp string>` (e.g. `end,2025-02-20 12:24:37.904324`)
- **Timestamp Storage Precision**: Microsecond ($10^{-6}\text{ s}$).
- **Annotation Temporal Accuracy**: Documented as `NOT_QUANTIFIED` (push-button timestamps adjusted post-hoc via reference accelerometer).

---

## 6. Annotation Event Statistics

Across the 6 annotated pilot recordings:
- **Total Non-Breathing Events**: 6 events (1 per annotated recording).
- **Mean Event Duration**: $11.321$ seconds.
- **Min / Max Event Duration**: $9.771$ seconds (`p075-sitting-rest`) to $12.205$ seconds (`p110-sitting-rest`).
- **Total Annotated Duration**: $67.925$ seconds.

---

## 7. Window / Event Alignment

1D interval intersection $[t_{start}, t_{end\_exclusive}) \cap [t_{begin}, t_{end})$ measured:
- `p001-lying-rest`: $8.725$ s overlap with Window 0 ($[0, 30)$ s), $1.826$ s in dropped tail ($[30, 50)$ s).
- `p001-sitting-rest`: $9.001$ s overlap with Window 0 ($[0, 30)$ s), $2.474$ s in dropped tail ($[30, 50)$ s).
- `p004-lying-rest` (600 frames, 2 windows): $7.254$ s overlap with Window 0 ($[0, 30)$ s), $4.739$ s overlap with Window 1 ($[30, 60)$ s). Entire event is preserved across W0+W1!
- `p075-sitting-rest`: $9.771$ s overlap with Window 0 ($[0, 30)$ s), $0.000$ s in tail. Entire event is contained inside W0!
- `p110-lying-rest`: $6.787$ s overlap with Window 0 ($[0, 30)$ s), $5.143$ s in dropped tail ($[30, 50)$ s).
- `p110-sitting-rest`: $8.042$ s overlap with Window 0 ($[0, 30)$ s), $4.163$ s in dropped tail ($[30, 50)$ s).

---

## 8. Legacy 15-Second Rule Re-Evaluation

The previously proposed rule ($\ge 15.0$s overlap or $\ge 50\%$ window fraction) was evaluated against empirical dataset evidence:
- **Max Event Duration in Dataset**: $12.205$ seconds.
- **Assigned APNEA Windows under Legacy Rule**: 0 windows.
- **Captured Events**: 0 events.
- **Conclusion**: The legacy 15-second requirement is **unusable** because no non-breathing event in the dataset reaches 15 seconds.

---

## 9. APNEA Proxy Policy Comparison

Candidate policies evaluated in [`policy_comparison.json`](../../datasets/mmwave/manifests/a4_label_pilot/policy_comparison.json):
1. **Legacy 15-second Rule** ($\ge 15.0$s): 0 APNEA windows assigned (100% events lost). Discarded.
2. **Policy A — 10-second Candidate** ($\ge 10.0$s): 0 APNEA windows assigned in fixed 30s grid due to window boundary truncation (events start at $t \approx 21-23$s and span across $t=30.0$s, yielding overlaps of $6.79$s to $9.00$s). A 10.0s threshold would discard all 6 of 6 dataset events under the current fixed A3 window grid. Discarded.
3. **Policy B — Overlap + Event Duration** ($\ge 6.0$s overlap AND total event duration $\ge 8.0$s): 6 APNEA windows assigned. Valid equivalent.
4. **Policy C — Event-Centered Diagnostic** (30s window centered on event midpoint): 6 APNEA windows assigned. Diagnostic alternative only.
5. **Selected Policy — $\ge 6.0$s Overlap Canonical**: 6 APNEA windows assigned, captures all 6 voluntary breath-hold events. **Selected as canonical profile.**

---

## 10. NORMAL Mapping Analysis

- **Rule**: Controlled `Rest` condition or Movesense chest ACC reference respiration rate in normal range ($10.0 \le \text{RR} < 25.0$ bpm) $\rightarrow$ `NORMAL` (`safenest_label_id: 0`).
- **Mapping Type**: `DERIVED` (resting-breathing or Movesense ACC reference proxy, rule `A4_RULE_NORMAL_MOVESENSE_ACC_REF`).
- **Pilot Count**: 5 windows (all 5 Post-exercise windows with normal Movesense ACC reference respiration rates).

---

## 11. RAPID_OR_ABNORMAL Mapping Analysis — Movesense Reference Integration

- **Reference Sensor**: Movesense chest accelerometer (`movesense_acc.csv`) as documented in Nature Scientific Data paper.
- **Respiration Rate Extraction**: Spectral peak analysis over 30s window using `movesense_rr_search_band_hz: [0.1, 0.7]`.
- **Rule**: Movesense chest ACC reference respiration rate $\ge 25.0$ bpm ($0.417$ Hz) or $< 10.0$ bpm bradypnea $\rightarrow$ `RAPID_OR_ABNORMAL` (`safenest_label_id: 1`, `mapping_type: DERIVED`, rules `A4_RULE_RAPID_MOVESENSE_ACC_REF` / `A4_RULE_ABNORMAL_BRADYPNEA_MOVESENSE_ACC_REF`).
- **Pilot Count**: 3 windows (`p001-lying-post_exercise` W0 at 32.0 bpm, `p002-lying-post_exercise` W0 at 26.0 bpm, `p004-lying-post_exercise` W1 at 36.1 bpm).

---

## 12. Transition / Mixed Window Policy

- **Rule**: Windows with non-zero non-breathing overlap $< 6.0$s (e.g. `p004-lying-rest` Window 1 with $4.739$s overlap) represent a breathing $\leftrightarrow$ non-breathing transition state.
- **Classification**: Classified as `AMBIGUOUS` (`safenest_label: null`, `assignment_status: AMBIGUOUS`, `mapping_rule_id: A4_RULE_TRANSITION_WINDOW`).

---

## 13. Selected Mapping Profile

Profile ID: `MMWAVE_LABEL_MAPPING_PROFILE_001`
```json
{
  "profile_id": "MMWAVE_LABEL_MAPPING_PROFILE_001",
  "target_classes": {
    "NORMAL": 0,
    "RAPID_OR_ABNORMAL": 1,
    "APNEA": 2
  },
  "apnea_policy": {
    "min_overlap_seconds": 6.0,
    "min_event_duration_seconds": 8.0,
    "voluntary_breath_hold_as_apnea_proxy": true,
    "clinical_apnea_claimed": false
  },
  "normal_policy": {
    "rest_condition_as_normal_proxy": true,
    "requires_zero_non_breathing_overlap": true,
    "movesense_acc_normal_rr_range_bpm": [10.0, 25.0]
  },
  "rapid_or_abnormal_policy": {
    "rapid_min_rr_bpm": 25.0,
    "bradypnea_max_rr_bpm": 10.0,
    "post_exercise_auto_rapid": false,
    "requires_independent_respiration_rate_reference": true,
    "reference_sensor": "MOVESENSE_CHEST_ACC",
    "movesense_rr_search_band_hz": [0.1, 0.7]
  },
  "a3_window_contract_modified": false
}
```

---

## 14. Window Label Distribution

Across all 15 evaluated A3 windows:
- **NORMAL**: 5 (33.3%)
- **RAPID_OR_ABNORMAL**: 3 (20.0%)
- **APNEA**: 6 (40.0%)
- **AMBIGUOUS**: 1 (6.7%)
- **UNMAPPED**: 0
- **EXCLUDED**: 0

---

## 15. Ambiguous / Unmapped / Excluded Windows

- **AMBIGUOUS Windows (1)**: `p004-lying-rest` Window 1 ($4.739$s non-breathing overlap, transition state).
- **UNMAPPED / EXCLUDED**: 0

---

## 16. Annotation Coverage Lost to A3 Tails

- **Total Annotated Events**: 6
- **Events Fully Represented**: 2 (`p004-lying-rest` across W0+W1 union, `p075-sitting-rest` inside W0)
- **Events Partially Represented**: 4 (events ending past 30.0s in 500-sample recordings)
- **Events Not Represented**: 0
- **Total Annotated Seconds in Pilot**: $67.925$ seconds.
- **Annotated Seconds Represented in A3 Windows**: $54.319$ seconds (80.0%).
- **Annotated Seconds Lost to A3 Dropped Tails**: $13.606$ seconds (20.0%).

---

## 17. Condition / Posture Artifact Audit

Contingency Summary:
- **APNEA**: Rest (6 windows) | Lying (3), Sitting (3)
- **NORMAL**: Post-exercise (5 windows) | Lying (2), Sitting (3)
- **RAPID_OR_ABNORMAL**: Post-exercise (3 windows) | Lying (3)
- **AMBIGUOUS**: Rest (1 window) | Lying (1)
- **Movesense Chest ACC Reference Used**: `True`
- **Post-Exercise Auto-Rapid Flag**: `False`
- **Clinical Apnea Claimed Flag**: `False`

Posture does not direct label assignment; recording condition is preserved separately from SafeNest target classes.

---

## 18. Exceptions

A total of 5 exceptions recorded in [`exceptions.json`](../../datasets/mmwave/manifests/a4_label_pilot/exceptions.json):
- 4 `ANNOTATION_IN_DROPPED_TAIL` (WARNING)
- 1 `TRANSITION_WINDOW` (INFO)
- 0 errors, 0 blockers.

---

## 19. Validation

The standalone validator `scripts/validate_mmwave_label_pilot.py` verified all 20 structural and semantic rules:
1. Every A3 window appears exactly once in A4.
2. Every A4 window refers to a valid A3 window.
3. A3 timestamp boundaries are preserved exactly.
4. A3 window boundaries are not silently changed.
5. Label class values match class contract.
6. Every assigned label has a mapping type.
7. Every assigned label has a mapping rule ID.
8. Every APNEA assignment from voluntary non-breathing is marked DERIVED.
9. No result claims clinical apnea.
10. Post-exercise alone is insufficient evidence for RAPID_OR_ABNORMAL (verified Movesense ACC reference used).
11. Window annotation overlap values match re-calculated event-window intersections.
12. Ambiguous windows remain explicit.
13. Unmapped windows remain in the manifest.
14. Source labels/conditions remain preserved.
15. Summary class counts match detailed manifest.
16. Exception counts match.
17. Annotation coverage counts match re-calculated values (2 full, 4 partial, 0 absent).
18. No train/val/test split fields introduced.
19. No model predictions used.
20. Validator wired into final gate.

---

## 20. A4 Gate

- **A4 Gate Status**: `PASS_WITH_WARNINGS`
- **Reason**: Annotation semantics, overlap mathematics, Movesense ACC reference respiration rate extraction, and SafeNest label mappings are deterministic and fully validated. Non-blocking warnings are logged for voluntary breath-hold proxy usage and tail-dropped annotation seconds.

---

## 21. A5 Entry Decision

- **A5 Entry Status**: `READY_WITH_CONDITIONS`
- **Conditions**:
  1. A5 (dataset packaging & subject split) must preserve `mapping_type` and `assignment_status` fields.
  2. `AMBIGUOUS` windows (transition states) must be handled explicitly during subject-wise train/validation/test split construction (e.g. held out from pure class training sets).
  3. No clinical apnea claims or unverified Post-exercise RAPID claims may be introduced in A5.

---

## 22. Remaining Limitations

1. Pilot dataset scope is 13 recordings (15 windows); full 440-recording conversion is deferred to A6.
2. Voluntary breath-hold is used as an APNEA class proxy rather than clinical sleep apnea.

---

## 23. Explicit Non-Scope Confirmation

As required, the following tasks were **NOT** performed in Phase A4:

```text
Subject-wise split: NOT PERFORMED
Train/validation/test assignment: NOT PERFORMED
Full 440-recording conversion: NOT PERFORMED
Final training NPZ generation: NOT PERFORMED
Class balancing: NOT PERFORMED
Preprocessing ablation: NOT PERFORMED
Model training: NOT PERFORMED
TFLite conversion: NOT PERFORMED
INT8 quantization: NOT PERFORMED
A5: NOT PERFORMED
```

---

## 24. Files Changed

- `scripts/mmwave_label_mapper.py`: Phase A4 SafeNest label mapper library module (with Movesense ACC respiration rate extraction and bradypnea rule).
- `scripts/run_mmwave_label_pilot.py`: Phase A4 pilot runner script (with updated coverage & policy comparison text).
- `scripts/validate_mmwave_label_pilot.py`: Phase A4 in-memory and standalone validator script (with active overlap & coverage re-calculation).
- `tests/test_mmwave_label_mapper.py`: Comprehensive unit test suite (23 unit tests covering label mapping, Movesense ACC extraction, bradypnea rule, and validation rules).
- `datasets/mmwave/manifests/a4_label_pilot/`: Manifest output directory (`pilot_selection.json`, `annotation_inventory.jsonl`, `policy_comparison.json`, `label_mapping_profile.json`, `window_label_manifest.jsonl`, `exceptions.json`, `a4_summary.json`, `checksums.sha256`).
- `docs/reports/20260808_Antigravity_A4_Annotation_Label_Mapping_Pilot_01.md`: This report.

---

## 25. Verification Commands and Test Execution

```bash
# 1. Run Phase A4 Pilot and Validator
python3 scripts/run_mmwave_label_pilot.py
python3 scripts/validate_mmwave_label_pilot.py

# 2. Run Unit Tests and A3/A2 Regression Tests
python3 -m unittest tests/test_mmwave_label_mapper.py -v
python3 -m unittest tests/test_mmwave_timeline.py -v
python3 -m unittest tests/test_mmwave_phase_extractor.py -v

# 3. Confirm Deterministic Regeneration
python3 scripts/run_mmwave_label_pilot.py
```
