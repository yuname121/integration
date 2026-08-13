# SafeNest mmWave Phase A0 Raw Dataset Identity, Schema, Inventory, and Integrity Lock Audit Report

**Audit Date**: 2026-08-07
**Auditor**: Autonomous AI Data Lineage & Radar Integrity Engineer (Antigravity Agent)
**Target Repository Root**: `<REPO_ROOT>`
**Git Branch**: `feature/phase-a0-raw-inventory`
**Git Commit**: `9f65d3165088c5a89c213956d32f9aac78f132e2`
**Target Raw Archive**: `datasets/raw_archives/external_datasets/db_records.zip`
**Phase A0 Gate Status**: **`PASS_WITH_WARNINGS`**
**Phase A1 Entry Status**: **`READY_WITH_CONDITIONS`**

---

## 1. Executive Summary

This report establishes the Phase A0 audit baseline for the Zenodo 60 GHz FMCW mmWave Vital Signs Radar Dataset (`10.5281/zenodo.18599983`). All conclusions in this report are programmatically derived from empirical audit measurements.

### Measured Key Highlights
- **Primary Archive Presence**: `datasets/raw_archives/external_datasets/db_records.zip` (EXISTS)
- **Archive Byte Size**: `246,597,320` bytes
- **Archive Checksums**:
  - SHA-256: `f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0`
  - MD5: `370de95033f1a98b78e57dbbea92a8bc`
- **ZIP Container Integrity**: `PASS` (6,382 total members; 0 CRC failures, 0 path risks)
- **Official Zenodo Remote Status**: `REMOTE_VERIFIED`
- **Official vs Local Relationship**: `LIKELY_REPACKAGED_NOT_FULLY_VERIFIED`
- **Dataset Inventory Scale**:
  - Unique Participants: **110**
  - Explicit Source Sessions: **0**
  - Normalized Derived Sessions: **110** (One deterministic normalized session per subject because the source archive exposes no explicit session identifier.)
  - Total Logical Recordings: **440**
- **Companion Linkage Summary**:
  - `COMPLETE`: **220**
  - `COMPLETE_WITH_OPTIONAL_FILES_ABSENT`: **220**
  - `PARTIAL`: **0**
  - `AMBIGUOUS`: **0**
  - `BROKEN`: **0**
- **Discovered Multi-Factor Schema Profiles**: **2**
- **Identifier Collision Count**: **0**
- **Registered Anomalies**: 0 Blockers, 0 Errors, 1 Warnings, 5 Info
- **A0 Gate Decision**: **`PASS_WITH_WARNINGS`** (A1 Entry Status: **`READY_WITH_CONDITIONS`**)

---

## 2. Scope

This Phase A0 audit performed the following evidence-derived operations:
1. Dynamic Git repository baseline and worktree status recording.
2. Direct streaming checksum and byte size measurement of `db_records.zip` before and after audit.
3. Live query against the official Zenodo REST API for DOI `10.5281/zenodo.18599983`.
4. Stream CRC check and structural path integrity audit across all ZIP members.
5. Complete enumeration of ZIP members into `archive_members.jsonl` with explicit evidence types.
6. Reconstructing recording companion-file linkage into `recording_index.jsonl` with schema cardinality contract.
7. Deep bounded inspection of measured schema signatures (measured header bytes, measured role cardinalities, ISO-8601 timestamp deltas, chirp config hashes).
8. Dynamic derivation of anomalies, inventory summary counts, A0 gate status, and A1 entry status.

---

## 3. Non-Scope

The following operations were **EXPLICITLY NOT PERFORMED** during Phase A0:
- **No rFFT Decoding**: Radar range FFT tensor arrays inside `radar_rFFTs.zlib` were not decompressed or decoded into numpy arrays.
- **No Range-Bin Selection**: Target range-bin indices were not selected.
- **No Antenna Beamforming/Selection**: Antenna channel combination was not performed.
- **No Phase Extraction**: Complex phase computation and phase unwrap were not executed.
- **No Signal Preprocessing**: Linear detrending, Butterworth BPF (0.1–0.5 Hz), and Z-score normalization were not applied.
- **No Resampling/Windowing**: 10 Hz resampling and 30-second windowing were not performed.
- **No Label Mapping**: Class label assignment was not performed.
- **No Subject Splitting**: Train/validation/test split was not generated.
- **No NPZ Generation**: Processed NPZ files were not generated or modified.
- **No Model Training / Quantization**: Model training, conversion, quantization, or evaluation was not performed.
- **No Git Commit/Push**: No git commits or pushes were performed.

---

## 4. Repository State

- **Repository Root**: `<REPO_ROOT>`
- **Git Branch**: `feature/phase-a0-raw-inventory`
- **Git Commit**: `9f65d3165088c5a89c213956d32f9aac78f132e2`
- **Git Remote Origin**: `https://github.com/sheepmeat/test.git`

---

## 5. Input Assets

| Asset Path | Status | Byte Size | SHA-256 Checksum | MD5 Checksum |
|---|---|---|---|---|
| `datasets/raw_archives/external_datasets/db_records.zip` | EXISTS | 246,597,320 | `f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0` | `370de95033f1a98b78e57dbbea92a8bc` |

---

## 6. Official Dataset Identity

- **Zenodo DOI**: `10.5281/zenodo.18599983`
- **Zenodo Record ID**: `18599983`
- **Official Title**: `Extensive Age-Balanced and Subject-Varied mmWave Radar Dataset of Referenced Records for Vital Signs`
- **Publication Date**: `2026-02-10`
- **Creators**: Parralejo, Felipe, Paredes, José A., Álvarez, Fernando J., Vicario, África
- **Official License**: `CC-BY-4.0`
- **Remote Verification Status**: `REMOTE_VERIFIED`

---

## 7. Official-to-Local Relationship

- **Relationship Status**: **`LIKELY_REPACKAGED_NOT_FULLY_VERIFIED`**
- **Official Container MD5**: `408c5b347c751c553abe6d0f640a6f98`
- **Local Container MD5**: `370de95033f1a98b78e57dbbea92a8bc`
- **MD5 Match**: `False`
- **Local Container SHA-256**: `f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0`
- **Internal Content Match Confirmed**: `False`

### Limitations & Evidence
1. The observed hash (370de95033f1a98b78e57dbbea92a8bc), size (246597320 bytes), and archive structure differences are consistent with local repackaging, but member-level identity with the official Zenodo archive has not been verified.
2. `content_match_confirmed` is explicitly set to `False` because official Zenodo member-level files were not fetched or byte-compared locally in A0.

---

## 8. ZIP Integrity Results

| Metric | Measured Value | Status |
|---|---|---|
| Openable Central Directory | `True` | PASS |
| Member Count | `6,382` | PASS |
| CRC Read Failures | `0` | PASS |
| Path Traversal Risks | `0` | PASS |
| Duplicate Exact Paths | `0` | PASS |
| Duplicate Casefold Paths | `0` | PASS |
| Encrypted Members | `0` | PASS |
| Overall ZIP Integrity | **`PASS`** | **PASS** |

---

## 9. Observation-Derived Schema Profiles

### Profile: `SCHEMA_PROFILE_001`
- **Recordings using Profile**: 220
- **Measured Signature Hash**: `2e892fce2d6776cd`
- **Observed Radar Header Signature**: `78da`
- **Measured Timestamp & Interval Properties**:
  - Parsed Timestamp Format: `ISO8601_UTC_CSV`
  - Measured Median Δt: `0.1s`
  - Measured Frame Rate: `10.0 Hz`
  - Duplicate Timestamps: `0`
  - Backward Timestamps: `0`
  - Large Timestamp Gaps (>0.2s): `0`
- **FMCW Chirp Parameters**:
  - Start Frequency: 60250000000.0 Hz
  - Ramp Slope: 30000000000000.0 Hz/s
  - ADC Samples: 64
  - Frame Periodicity: 100.0 ms (10 Hz)
- **Phase A1 Reader Requirements**:
  - Decompress zlib stream for radar_rFFTs.zlib
  - Parse float/complex array safely without object pickle
  - Parse ISO-8601 timestamps from radar_timestamps.csv
  - Read FMCW chirp parameters from radar_chirpConfig.json

### Profile: `SCHEMA_PROFILE_002`
- **Recordings using Profile**: 220
- **Measured Signature Hash**: `5aa1154747af06f6`
- **Observed Radar Header Signature**: `78da`
- **Measured Timestamp & Interval Properties**:
  - Parsed Timestamp Format: `ISO8601_UTC_CSV`
  - Measured Median Δt: `0.1s`
  - Measured Frame Rate: `10.0 Hz`
  - Duplicate Timestamps: `0`
  - Backward Timestamps: `0`
  - Large Timestamp Gaps (>0.2s): `0`
- **FMCW Chirp Parameters**:
  - Start Frequency: 60250000000.0 Hz
  - Ramp Slope: 30000000000000.0 Hz/s
  - ADC Samples: 64
  - Frame Periodicity: 100.0 ms (10 Hz)
- **Phase A1 Reader Requirements**:
  - Decompress zlib stream for radar_rFFTs.zlib
  - Parse float/complex array safely without object pickle
  - Parse ISO-8601 timestamps from radar_timestamps.csv
  - Read FMCW chirp parameters from radar_chirpConfig.json

---

## 10. Documented Claims Versus Observed Evidence

| Claimed Field | Documented Claim | Locally Measured Value | Status |
|---|---|---|---|
| `doi` | `10.5281/zenodo.18599983` | `10.5281/zenodo.18599983` | `MATCH` |
| `participant_count` | `110` | `110` | `MATCH` |
| `recording_count` | `440` | `440` | `MATCH` |
| `archive_size_bytes` | `246597320` | `246597320` | `PARTIAL_MATCH` |
| `archive_sha256` | `f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0` | `f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0` | `MATCH` |
| `postures` | `['Sitting', 'Lying']` | `['Lying', 'Sitting']` | `MATCH` |
| `activities` | `['Rest', 'Post-exercise']` | `['Post-exercise', 'Rest']` | `MATCH` |

---

## 11. Anomalies Registry

| Anomaly ID | Severity | Category | Observed Evidence | Impact |
|---|---|---|---|---|
| `A0-ANOM-0001` | `INFO` | `REPOSITORY_STATE` | Pre-existing modified and untracked files exist in the repository worktree prior to Phase A0 execution. | Requires careful tracking to ensure Phase A0 changes are isolated. |
| `A0-ANOM-0002` | `INFO` | `VERSION_CONTEXT` | Repository contains legacy SafeNest V4 and SafeNest V5 directories alongside top-level datasets/. | Historical manifest files exist in V4/V5 subdirectories. |
| `A0-ANOM-0003` | `WARNING` | `REMOTE_VERIFICATION` | Zenodo record 18599983 includes 3 companion files (ParticipantsInfo.xlsx, ExampleCode.ipynb, helper_fns.py) that are not present in the local workspace clone. | Demographic participant metadata (age, sex, height, weight) is currently missing locally. |
| `A0-ANOM-0004` | `INFO` | `CHECKSUM` | The observed hash (370de95033f1a98b78e57dbbea92a8bc), size (246597320), and archive structure differences are consistent with local repackaging, but member-level identity with the official Zenodo archive has not been verified. | Container hash mismatch; content_match_confirmed set to false. |
| `A0-ANOM-0005` | `INFO` | `ZIP_PATH` | Archive contains 3191 __MACOSX/ resource fork metadata files created during macOS re-archiving. | Filter out __MACOSX entries during dataset reading. |
| `A0-ANOM-0006` | `INFO` | `SCHEMA` | Recordings ['db_records/P075/Sitting/Rest/radar_timestamps.csv', 'db_records/P007/Sitting/Post-exercise/radar_timestamps.csv'] have 400 timestamp lines (40s duration) rather than 500 (50s) or 600 (60s). | Phase A1 window generator must handle 40s recordings. |

---

## 12. Dynamic A0 Gate Decision

- **A0 Gate Status**: **`PASS_WITH_WARNINGS`**
- **A1 Entry Status**: **`READY_WITH_CONDITIONS`**
- **Archive Unchanged After Audit**: `True`

---

## 13. A1 Pilot Recommendations

The following candidate recordings are recommended for Phase A1 decoder testing:
1. `P001/Sitting/Rest`: Baseline 500-frame sitting rest recording with voluntary breath-hold annotation.
2. `P001/Lying/Rest`: Baseline 500-frame lying rest recording with annotation.
3. `P001/Sitting/Post-exercise`: Post-exercise elevated respiration rate recording without annotation.
4. `P002/Lying/Post-exercise`: 600-frame (60s) duration recording.
5. `P075/Sitting/Rest`: 400-frame (40s) duration edge-case recording.
