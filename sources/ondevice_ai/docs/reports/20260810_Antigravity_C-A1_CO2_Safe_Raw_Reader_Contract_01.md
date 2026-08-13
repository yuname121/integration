# SafeNest CO₂ Phase C-A1 — Safe Raw Reader and Source-Row Contract Audit Report

- Document Version: `01`
- Author: `Antigravity` (CO₂ Track Implementation Agent)
- Execution Date: `2026-08-10`
- Phase: `C-A1 — CO₂ Safe Raw Reader and Source-Row Contract`
- Target Dataset: UCI Occupancy Detection Dataset
- Status: `PASS_WITH_WARNINGS`
- C-A2 Authorization: `YES` (All foundational safe reader and provenance contract criteria satisfied)

---

## 1. Executive Summary

Phase **C-A1** establishes a safe, deterministic, read-only reader module ([`datasets/co2/raw_reader.py`](../../datasets/co2/raw_reader.py)) and source-row provenance contract for the real UCI Occupancy Detection dataset used by the SafeNest CO₂ track.

The reader consumes the raw read-only zip payload (`datasets/raw_archives/external_datasets/occupancy+detection.zip`) directly in memory without unzipping files to disk, resolves the 7-header vs. 8-field physical column mismatch, preserves original raw timestamp strings, enforces naive local clock semantics (`SOURCE_ACQUISITION_CLOCK`, `UNVERIFIED`), and guarantees 100% full raw provenance traceability for all 20,560 source observations without row loss.

The standalone C-A1 validator passed with 0 errors and 5 non-blocking warnings, authorizing entry into Phase **C-A2**.

---

## 2. Predecessor C-A0 Identity Verification

- **Predecessor Phase**: `C-A0 — CO₂ Source Identity, License, and Raw Inventory`
- **Predecessor Manifest Directory**: `datasets/co2/manifests/c_a0_raw_inventory/`
- **Predecessor Status**: Verified in canonical `main` branch (`0c14b52 Merge pull request #14`).
- **Raw Archive Relative Path**: `datasets/raw_archives/external_datasets/occupancy+detection.zip`
- **Raw Archive Byte Size**: `335,713` bytes
- **Raw Archive SHA-256 Hash**: `4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a`
- **Verification Result**: Current C-A1 reader verification matches C-A0 predecessor evidence exactly.

---

## 3. Reader Architecture & Module Contract

The C-A1 reader is implemented in [`datasets/co2/raw_reader.py`](../../datasets/co2/raw_reader.py) as `UCIOccupancyRawReader`.

```text
raw archive zip (read-only)
   ↓
verify zip size & SHA-256
   ↓
verify member size & SHA-256
   ↓
verify 7-field header string
   ↓
parse 8 physical CSV fields (Field 0 = Row ID, Field 1 = Date, Fields 2..6 = Features, Field 7 = Occupancy)
   ↓
CO2SourceRowObservation (Dataclass with 100% Raw Provenance + Raw Measurements)
```

---

## 4. Source Schema Profile & Schema Mismatch Resolution

- **Visible Named Header**: `"date","Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"` (7 named fields).
- **Physical Row Structure**: 8 physical CSV fields per line across all 20,560 data rows.
- **Physical Field Mapping**:
  - `Field 0`: `source_row_identifier` (Exported dataframe integer index string, e.g. `"1"`, `"2"`)
  - `Field 1`: `source_timestamp_raw` (Raw timestamp string, e.g. `"2015-02-04 17:51:00"`)
  - `Field 2`: `temperature` (Raw float in °C)
  - `Field 3`: `humidity` (Raw float in %)
  - `Field 4`: `light` (Raw float in Lux)
  - `Field 5`: `co2` (Raw float in ppm)
  - `Field 6`: `humidity_ratio` (Raw float in kg/kg)
  - `Field 7`: `occupancy` (Raw binary integer label, `0` or `1`)

---

## 5. Source-Row Provenance & Traceability Contract

Every `CO2SourceRowObservation` emitted by `UCIOccupancyRawReader` carries the following 16 fields:

| Field Name | Type | Value / Semantics |
|---|---|---|
| `source_archive_path` | `str` | `datasets/raw_archives/external_datasets/occupancy+detection.zip` |
| `source_archive_sha256` | `str` | `4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a` |
| `source_member_name` | `str` | `datatest.txt` / `datatest2.txt` / `datatraining.txt` |
| `source_member_sha256` | `str` | Exact member SHA-256 hash |
| `source_physical_line_number` | `int` | 1-indexed physical line number in member file (≥ 2) |
| `source_row_identifier` | `str` | Original exported dataframe index string |
| `source_timestamp_raw` | `str` | Original raw acquisition timestamp string |
| `timestamp_reference` | `str` | `SOURCE_ACQUISITION_CLOCK` |
| `source_timezone` | `str` | `UNVERIFIED` |
| `utc_conversion_claimed` | `bool` | `False` |
| `temperature` | `float` | Raw measured temperature (°C) |
| `humidity` | `float` | Raw measured humidity (%) |
| `light` | `float` | Raw measured light (Lux) |
| `co2` | `float` | Raw measured CO₂ concentration (ppm) |
| `humidity_ratio` | `float` | Raw measured humidity ratio (kg/kg) |
| `occupancy` | `int` | Original room occupancy label (`0` or `1`) |

---

## 6. Timestamp & Timezone Semantics

- **Timestamp Format**: Standard `YYYY-MM-DD HH:MM:SS` (100% parseable across all 20,560 rows).
- **Timezone Status**: `UNVERIFIED` (Naive local clock timestamp; UTC conversion is explicitly **not** claimed).
- **Enforcement**: Timestamps are preserved as raw strings. No timezone offsets or `Z` suffixes are appended in C-A1.

---

## 7. Exact Row Preservation & Label Distribution

The reader guarantees zero row loss across all members:

| Member Name | Uncompressed Size (Bytes) | SHA-256 Hash | Raw Data Rows Read | Occupancy 0 | Occupancy 1 |
|---|---|---|---|---|---|
| `datatest.txt` | 200,766 | `1b92c7c1b2838963464fa891a610cf3c5db4becb7189189b29b330107a584c7f` | 2,665 | 1,693 | 972 |
| `datatest2.txt` | 699,664 | `d026d1bd5aeccd4aff4f3b3710d48e40613bd5fc370db7e61bbdcaa50d985095` | 9,752 | 7,703 | 2,049 |
| `datatraining.txt` | 596,674 | `b2c4d0ce2b9e4e453c476f7125ef31aeec2d1f5c7f5572d0e80de3df6521ab56` | 8,143 | 6,414 | 1,729 |
| **Total** | **1,497,104** | — | **20,560** | **15,810 (76.89%)** | **4,750 (23.11%)** |

- **Silent Row Loss**: `0` dropped rows.
- **Corrupt / Unparseable Rows**: `0` corrupt rows.

---

## 8. Reader Failure Modes & Error Handling

`UCIOccupancyRawReader` defines strict defensive exceptions in `datasets/co2/raw_reader.py`:

- `ArchiveNotFoundError`: Raised if the raw zip archive path does not exist.
- `ArchiveIntegrityError`: Raised if zip file byte size or SHA-256 hash fails verification.
- `MemberIntegrityError`: Raised if a zip member is missing, corrupted, or fails SHA-256 check.
- `SchemaValidationError`: Raised if header strings, named field counts, or physical column counts mismatch.
- `SourceRowParseError`: Raised if numeric values or Occupancy labels fail typed parsing.

---

## 9. Lineage Isolation & Synthetic-vs-Real Boundaries

- **Real Raw Source**: `datasets/raw_archives/external_datasets/occupancy+detection.zip` (Used exclusively by `UCIOccupancyRawReader`).
- **Synthetic Smoke Fixture**: `datasets/co2/processed/co2_occupancy_v1.npz` (Classified as `SYNTHETIC_SMOKE_FIXTURE`, not used as reader source, preserved unchanged).
- **Existing TFLite Model**: `models/co2/co2_occupancy_int8_v0.1.0.tflite` (`MODEL_TRAINING_LINEAGE_UNVERIFIED`).
- **Existing Scaling Metadata**: `models/co2/co2_scaling_metadata_v0.1.0.json` (`SCALER_FIT_LINEAGE_UNVERIFIED`).

---

## 10. Non-blocking Limitations & Warnings

The C-A1 validator records 5 non-blocking warnings carried forward from C-A0:

1. `HEADER_DATA_WIDTH_MISMATCH`: 7 header fields vs 8 physical data fields.
2. `SOURCE_TIMEZONE_UNVERIFIED`: Naive local clock timestamps without UTC offset.
3. `MODEL_TRAINING_LINEAGE_UNVERIFIED`: Existing TFLite model lineage unverified against raw ZIP.
4. `SCALER_FIT_LINEAGE_UNVERIFIED`: Existing scaling metadata fit data lineage unverified against raw ZIP.
5. `GROUP_INDEPENDENCE_NOT_VERIFIABLE`: Dataset gathered from a single office room across 3 continuous time windows.

---

## 11. Explicitly Deferred Work

- **C-A2**: Timestamp canonicalization, strongest defensible temporal grouping & subject/session split policy definition.
- **C-A3**: Derived feature reconstruction (`CO2_slope` in ppm/min, history window duration, regression vs difference).
- **C-A4–C-A6**: Quality audit, full dataset conversion, and Phase A exit gate.
- **C-B**: Offline model comparison.
- **C-C**: SCD40 device-domain validation.

---

## 12. C-A2 Authorization Result

```text
C-A2 Authorized: YES
Reason:
- Predecessor C-A0 evidence verified and validator passed.
- Safe read-only reader UCIOccupancyRawReader implemented and verified.
- Schema mismatch (7-header vs 8-field) safely resolved.
- Provenance contract complete (1:1 traceability for all 20,560 source rows).
- Zero row loss verified across all 3 raw dataset files.
- Raw measurements remain unnormalized (no scaling, no CO2_slope created).
- Synthetic NPZ explicitly isolated from real raw source.
- Standalone validator passed (PASS_WITH_WARNINGS).
- Focused test suite passed (11/11 passed).
- 0 raw payload files staged in Git index.
- 0 mmWave, Thermal, or shared integration files modified.
```
