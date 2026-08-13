# Phase A6 — Full mmWave Conversion, End-to-End Provenance, Integrity / Leakage Audit, and Phase-A Exit Gate Report

- **Author**: Antigravity Implementation Engineer
- **Date**: 2026-08-08
- **Target Repository**: `https://github.com/sheepmeat/test.git`
- **Branch**: `feature/phase-a6-full-conversion-integrity-audit`
- **Baseline Commit**: `be24476` (merged Phase A5 ancestor `575010592073913c97f2f0201ebe1beef5264b55`)
- **Phase A6 Gate Status**: `PASS_WITH_WARNINGS`
- **Phase-B Entry Status**: `READY_WITH_CONDITIONS`

---

## 1. Executive Summary

Phase A6 represents the final execution, end-to-end provenance, and integrity auditing phase of the SafeNest mmWave real-data reconstruction pipeline.

Key achievements of Phase A6:
1. **Full A0 Inventory Processing**: Converted 100% of all 440 recordings across 110 subjects from `db_records.zip` into canonical, label-independent unwrapped phase signals without altering the raw dataset archive.
2. **Immutable A5 Split Inheritance**: Inherited the approved SUBJECT-wise split (`77` TRAIN, `17` VALIDATION, `16` LOCKED_TEST subjects) from `MMWAVE_SUBJECT_SPLIT_PROFILE_001` without recomputing or altering split assignments.
3. **Zero Cross-Split Leakage**: Achieved complete isolation across splits:
   - Cross-split subject overlap = `0`
   - Cross-split recording overlap = `0`
   - Cross-split window-ID overlap = `0`
   - Cross-split exact signal hash overlap = `0`
4. **Deterministic Canonical Window & Label Generation**: Generated **530 canonical 30-second 10 Hz phase windows** ($300$ samples each, zero overlap):
   - `RAPID_OR_ABNORMAL`: 119 windows
   - `APNEA`: 213 windows
   - `NORMAL`: 149 windows
   - `AMBIGUOUS`: 49 windows
5. **Canonical Numeric Dataset (.npy)**: Generated `datasets/mmwave/processed/mmwave_canonical_real_v1.npy` containing the full 530 $\times$ 300 float64 matrix with verified 1:1 index alignment to window and provenance manifests.
6. **Real Quality Audit**: Performed explicit array calculations on all 530 window phase slices: NaN count = 0, Inf count = 0, exact constant count = 0, near constant count = 0, mean window std dev = 6.535.
7. **LOCKED_TEST Isolation**: Verified that `training_eligible == False` for all 88 LOCKED_TEST windows and that `AMBIGUOUS` transition windows are excluded from pure-class training/validation eligibility.
8. **Strict Machine-Readable Path Provenance**: Guaranteed that all persistent JSON/JSONL manifest fields store repository-relative POSIX paths only, rejecting absolute `/Users/...` or `file://...` URIs.
9. **Raw Archive Immutability**: Confirmed that `pre_a6_archive_sha256` and `post_a6_archive_sha256` are byte-identical (`f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0`).

---

## 2. Git Baseline

- **Repository Root**: `SafeNest POSIX active workspace root`
- **Branch**: `feature/phase-a6-full-conversion-integrity-audit`
- **Ancestor Commit**: `575010592073913c97f2f0201ebe1beef5264b55` (`git merge-base --is-ancestor` verified = `0`)
- **Raw Archive SHA-256 (Pre/Post)**: `f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0` (Unchanged)

---

## 3. Approved A0–A5 Contracts

Phase A6 inherited and orchestrated the upstream Phase-A contracts:
- **A1 Decoder**: `RFFT_DECODER_PROFILE_001` (safe non-executing VM, protocol-5 pickle allowlist).
- **A2 Phase Extractor**: `MMWAVE_PHASE_EXTRACTION_PROFILE_001` (label-independent range-bin selection over `[0.3, 1.91]` m).
- **A3 Timeline**: `MMWAVE_TIMELINE_PROFILE_001` ($10.0$ Hz, 30-second windows, $300$ samples, zero overlap).
- **A4 Label Mapping**: `MMWAVE_LABEL_MAPPING_PROFILE_001` ($\ge 6.0$s non-breathing overlap $\to$ `APNEA`, Movesense ACC reference $\ge 25$ bpm or $< 10$ bpm bradypnea $\to$ `RAPID_OR_ABNORMAL`).
- **A5 Subject Split**: `MMWAVE_SUBJECT_SPLIT_PROFILE_001` (SUBJECT-wise split: `77` TRAIN, `17` VALIDATION, `16` LOCKED_TEST).

---

## 4. A0-Measured Inventory

Measured dynamically from `datasets/mmwave/manifests/a0_raw_inventory/recording_index.jsonl`:
- **Unique Subjects**: `110`
- **Unique Recordings**: `440`
- **Recordings per Subject**: `4` for all 110 subjects (distribution: `{ "4": 110 }`).
- **Evidence Source**: `MEASURED_FROM_A0`

---

## 5. A5 Fixed Split Verification

Subject assignments loaded from `datasets/mmwave/splits/mmwave_real_subject_split_v1.json`:
- **TRAIN Subjects**: `77` (70.0%)
- **VALIDATION Subjects**: `17` (15.5%)
- **LOCKED_TEST Subjects**: `16` (14.5%)
- **Missing Subjects**: `0`
- **Duplicate Subjects**: `0`
- **Split Conflicts**: `0`
- **Split Recomputed**: `NO`

---

## 6. Raw Archive Integrity

- **Pre-A6 SHA-256**: `f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0`
- **Post-A6 SHA-256**: `f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0`
- **Archive Unchanged**: `True`

---

## 7. Source Preflight

Preflight inspection across 440 recordings:
- `READY`: 440 recordings (100.0%)
- `MISSING_REQUIRED_MEMBER`: 0
- `INVENTORY_CONTRADICTION`: 0

---

## 8. Full Safe Decode Results

- **Decoder Profile**: `RFFT_DECODER_PROFILE_001`
- **Successful Decodes**: 440 (100.0%)
- **Decode Failures**: 0
- **Frame-Count Distribution**:
  - `500` frames: `348` recordings
  - `600` frames: `90` recordings
  - `400` frames: `2` recordings
- **Tensor Shape Distribution**:
  - `(500, 8, 64)`: `348` recordings
  - `(600, 8, 64)`: `90` recordings
  - `(400, 8, 64)`: `2` recordings
- **Nonfinite Raw Complex Values**: 0

---

## 9. Full Range-Bin / Channel Selection

- **Extraction Profile**: `MMWAVE_PHASE_EXTRACTION_PROFILE_001`
- **Selected Range Bin Distribution**: Range Bin index `2` (394 recordings), Range Bin index `1` (46 recordings).
- **Selected Virtual Channel Distribution**:
  - Virtual Channel `6`: 70 recordings
  - Virtual Channel `7`: 68 recordings
  - Virtual Channel `5`: 67 recordings
  - Virtual Channel `4`: 67 recordings
  - Virtual Channel `2`: 49 recordings
  - Virtual Channel `0`: 45 recordings
  - Virtual Channel `3`: 39 recordings
  - Virtual Channel `1`: 35 recordings
- **Search Boundary Cases**: 0

---

## 10. Full Canonical Phase Results

- **Canonical Signal**: Unfiltered, unnormalized unwrapped phase (`np.float64`).
- **Nonfinite Phase Count**: 0.
- **Near-Zero Magnitude Samples**: Preserved safely.

---

## 11. Timestamp Contract

- **Timestamp Reference**: `COMMON_ACQUISITION_COMPUTER_CLOCK`
- **Source Timezone**: `UNVERIFIED`
- **UTC Conversion Claimed**: `False`
- **New Automatic Z Suffixes**: `0`
- **Historical Pilot Trailing-Z**: Treated as historical string artifact, not verified UTC proof.

---

## 12. Full Timeline / Window Results

- **Timeline Profile**: `MMWAVE_TIMELINE_PROFILE_001`
- **Native Timeline Rate**: 10.0 Hz (exact 100 ms spacing).
- **Resampling Performed**: False (0 recordings required resampling).
- **Duplicate / Backward Timestamps**: 0.
- **Total Canonical 30s Windows**: `530` windows (30.0s duration, 300 samples, 0 overlap).
- **Dropped Tail Sample Distribution**:
  - `200` samples dropped: `348` recordings (500 frames $\to$ 1 window)
  - `0` samples dropped: `90` recordings (600 frames $\to$ 2 windows)
  - `100` samples dropped: `2` recordings (400 frames $\to$ 1 window)
- **Interpolated Samples**: `0`

---

## 13. Annotation Coverage

Across 440 recordings:
- **Annotation-Bearing Recordings**: `220` (Rest condition recordings with `non_breathing_ts.csv`).
- **Annotation-Absent Recordings**: `220` (Post-exercise condition recordings).
- **Total Non-Breathing Events**: `220` voluntary breath-hold events.
- Event count is derived from successfully parsed annotation records, not inferred from annotation file count; all 220 annotation files produced one valid event and no parse failure.
- **Overlap Mathematics**: Computed exact 1D interval intersections $[t_{start}, t_{end\_exclusive}) \cap [t_{begin}, t_{end})$.

---

## 14. Movesense Reference Processing

- **Reference Sensor**: Movesense chest accelerometer (`movesense_acc.csv`).
- **Respiration Extraction**: FFT spectral peak analysis over `[0.1, 0.7]` Hz ($6.0$ to $42.0$ bpm).
- **Normal Range**: $10.0 \le \text{RR} < 25.0$ bpm.
- **Rapid / Abnormal Range**: $\text{RR} \ge 25.0$ bpm or $\text{RR} < 10.0$ bpm (bradypnea).

---

## 15. Full Label / Assignment Accounting

Across all 530 canonical windows:
- **RAPID_OR_ABNORMAL**: `119` windows (22.5%)
- **APNEA**: `213` windows (40.2%)
- **NORMAL**: `149` windows (28.1%)
- **AMBIGUOUS**: `49` windows (9.2%)
- **UNMAPPED**: 0
- **EXCLUDED**: 0

---

## 16. Fixed Split Inheritance

Every window inherits its subject's split:
- **TRAIN Windows**: `358` (67.5%)
- **VALIDATION Windows**: `84` (15.8%)
- **LOCKED_TEST Windows**: `88` (16.6%)
- **Split Mismatches**: 0.

---

## 17. Eligibility Accounting

- **training_eligible**: `327` windows (TRAIN windows with clean ASSIGNED label).
- **validation_eligible**: `79` windows (VALIDATION windows with clean ASSIGNED label).
- **locked_test_evaluation_eligible**: `75` windows (LOCKED_TEST windows with clean ASSIGNED label).
- **AMBIGUOUS Pure-Class Eligible**: `0` windows (hard constraint verified).
- **LOCKED_TEST Training Eligible**: `0` windows (hard constraint verified).

---

## 18. Signal Quality Audit

- **NaN Samples**: `0` (real array check)
- **Inf Samples**: `0` (real array check)
- **Exact Constant Windows**: `0` (real array check)
- **Near-Constant Windows**: `0` (real array check)
- **Mean Window Phase Standard Deviation**: `6.535004` (measured across all 530 windows).
- **Quality Flags**: `TIMELINE_EXACT_NATIVE_10HZ` (530 windows).

---

## 19. Exact Duplicate Audit

- **Total Unique Signal Hashes**: `530`
- **Same-Subject Exact Duplicates**: 0
- **Cross-Subject Exact Duplicates**: 0
- **Cross-Split Exact Signal Hash Overlap**: `0`

---

## 20. Near-Duplicate Diagnostic

- **Near-Duplicate Diagnostic Status**: `NOT_PERFORMED` (explicitly unperformed diagnostic).

---

## 21. Subject Leakage Audit

- **TRAIN $\cap$ VALIDATION Subjects**: $\emptyset$
- **TRAIN $\cap$ LOCKED_TEST Subjects**: $\emptyset$
- **VALIDATION $\cap$ LOCKED_TEST Subjects**: $\emptyset$
- **Cross-Split Subject Overlap**: `0` (independently re-calculated by validator).

---

## 22. Recording Leakage Audit

- **Cross-Split Recording Overlap**: `0` (independently re-calculated by validator).

---

## 23. Window Leakage Audit

- **Cross-Split Window ID Overlap**: `0` (independently re-calculated by validator).

---

## 24. LOCKED_TEST Isolation Audit

- **Mechanical Conversion Only**: True
- **Used for Training**: `False`
- **Used for Preprocessing Selection**: `False`
- **Used for Threshold Tuning**: `False`
- **Used for Model Selection**: `False`
- **Structural Accounting Only**: `True`

---

## 25. Provenance Completeness

- **Total Provenance Manifest Rows**: `530`
- **One-to-One Correspondence**: `True`
- **Missing Required Fields**: `0`
- **Absolute Local Paths**: `0`
- The standalone validator compares every row's window/recording/subject identity, split, SafeNest label, mapping, assignment status, eligibility flags, and canonical signal SHA-256 across the window manifest, provenance manifest, and `.npy` matrix.

---

## 26. Canonical Sample Index Audit

- **Sample Indices**: Bounded 0-indexed integers `0` to `529`.
- **Future NPZ Sample Index**: `null` (`None`) until Phase B training NPZ creation.

---

## 27. Deterministic Spot Checks

Selected sample indices `0`, `132`, `265`, `397`, `529` traced lineage backwards from `canonical_sample_index` to `.npy` row, window manifest, provenance row, and `A0 raw archive member`:
- **Spot Check Failures**: `0`

---

## 28. Deterministic Regeneration

Ran full conversion twice in isolated paths:
- **Output Manifest Byte Comparison**: Identical SHA-256 digests across all output files.

---

## 29. Checksums

All output manifests and the canonical `.npy` file are checksummed in `datasets/mmwave/manifests/a6_full_conversion/checksums.sha256`. The validator rejects malformed or duplicate entries, missing required targets, paths escaping the canonical root, missing files, and digest mismatches.

---

## 30. Exceptions / Warnings

- **Blockers**: 0
- **Errors**: 0
- **Warnings**: `350` recordings logged `INCOMPLETE_TAIL_DROPPED`: 348 recordings dropped 200 samples (20 s), and 2 recordings dropped 100 samples (10 s).

---

## 31. A6 Gate

- **A6 Gate Status**: `PASS_WITH_WARNINGS`
- **Reason**: Full conversion of all 440 A0 recordings is complete, deterministic, and independently validated by standalone validator. Zero cross-split leakage observed.

---

## 32. Phase-A Exit Decision

- **Phase A Status**: `COMPLETE`
- **Reason**: The full reconstruction chain $A0 \to A1 \to A2 \to A3 \to A4 \to A5 \to A6$ is reproducible and fully audited for the entire 110-subject / 440-recording inventory.

---

## 33. Phase-B Entry Decision

- **Phase-B Entry Status**: `READY_WITH_CONDITIONS`
- **Conditions**:
  1. Phase B preprocessing ablation must consume canonical phase matrix `mmwave_canonical_real_v1.npy` without modifying A0-A6 manifests.
  2. `LOCKED_TEST` partition must remain strictly isolated from hyperparameter tuning and model selection.
  3. `AMBIGUOUS` transition windows must remain excluded from pure-class training.

---

## 34. Scientific Limitations

1. Voluntary breath-hold is used as an APNEA class proxy rather than clinical sleep apnea.
2. Offline Zenodo dataset validation is not a substitute for deployed MR60 real-sensor hardware field validation.
3. Source timestamps are recorded on acquisition-computer clock; source timezone is `UNVERIFIED`.

---

## 35. Explicit Non-Scope Section

The following tasks were **NOT** performed in Phase A6:
```text
Split recalculation: NOT PERFORMED
Subject reassignment: NOT PERFORMED
Preprocessing ablation: NOT PERFORMED
Train-only scaler fitting: NOT PERFORMED
Class balancing: NOT PERFORMED
Oversampling / augmentation: NOT PERFORMED
Feature selection: NOT PERFORMED
Threshold tuning: NOT PERFORMED
Model training: NOT PERFORMED
Validation-based model selection: NOT PERFORMED
LOCKED_TEST model evaluation: NOT PERFORMED
TFLite conversion: NOT PERFORMED
INT8 quantization: NOT PERFORMED
Phase B: NOT PERFORMED
```

---

## 36. Files Changed

- `scripts/mmwave_full_converter.py`: Phase A6 full conversion and provenance library module.
- `scripts/run_mmwave_full_conversion.py`: Phase A6 full conversion runner script.
- `scripts/validate_mmwave_full_conversion.py`: Phase A6 standalone and in-memory validator script.
- `tests/test_mmwave_full_conversion.py`: Unit test suite testing A6 scenarios.
- `datasets/mmwave/processed/mmwave_canonical_real_v1.npy`: Canonical 530 $\times$ 300 float64 phase matrix dataset.
- `datasets/mmwave/manifests/a6_full_conversion/`: Output manifest directory (`processing_profile.json`, `full_recording_results.jsonl`, `full_window_manifest.jsonl`, `full_provenance_manifest.jsonl`, `full_label_distribution.json`, `full_split_distribution.json`, `full_quality_audit.json`, `full_duplicate_audit.json`, `spot_check_results.json`, `exceptions.json`, `a6_summary.json`, `checksums.sha256`).
- `docs/reports/20260808_Antigravity_A6_Full_Conversion_Integrity_Audit_01.md`: This human-readable report.

---

## 37. Commands / Tests

```bash
# 1. Run Full A6 Conversion & Auditing
python3 scripts/run_mmwave_full_conversion.py

# 2. Run Standalone A6 Validator
python3 scripts/validate_mmwave_full_conversion.py

# 3. Run Full Test Suite (A0-A6 Unit & Regression Tests)
python3 -m unittest tests/test_mmwave_full_conversion.py -v
python3 -m unittest tests/test_mmwave_subject_split.py -v
python3 -m unittest tests/test_mmwave_label_mapper.py -v
python3 -m unittest tests/test_mmwave_timeline.py -v
python3 -m unittest tests/test_mmwave_phase_extractor.py -v

# 4. Check Git Diff & Raw Archive Immutability
git diff --check
python3 -c "import hashlib; print(hashlib.sha256(open('datasets/raw_archives/external_datasets/db_records.zip', 'rb').read()).hexdigest())"
```
