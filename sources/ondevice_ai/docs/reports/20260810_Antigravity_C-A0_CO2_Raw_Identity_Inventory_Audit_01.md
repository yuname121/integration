# SafeNest CO₂ Phase C-A0 — Source Identity, License, and Raw Inventory Audit

- Document Version: `01`
- Author: `Antigravity` (CO₂ Track Implementation Agent)
- Execution Date: `2026-08-10`
- Phase: `C-A0 — CO₂ Source Identity, License, and Raw Inventory`
- Target Dataset: UCI Occupancy Detection Dataset
- Status: `PASS_WITH_WARNINGS`
- C-A1 Authorization: `YES` (All foundational identity, license, and raw inventory criteria satisfied)

---

## 1. Executive Summary

Phase **C-A0** establishes the verified, deterministic, checksum-backed provenance and raw inventory foundation for the real UCI Occupancy Detection dataset used by the SafeNest CO₂ track.

This phase resolves the historical lineage ambiguity between the tracked synthetic smoke fixture (`co2_occupancy_v1.npz`) and the real UCI raw source archive (`occupancy+detection.zip`). The standalone validator passed with 0 blocking errors and 5 non-blocking warnings, authorizing entry into Phase **C-A1**.

---

## 2. Official Dataset Identity and Source

- **Official Dataset Name**: UCI Occupancy Detection Dataset
- **Official Source URL**: `https://archive.ics.uci.edu/dataset/357/occupancy+detection`
- **Stable Identifier**: UCI Machine Learning Repository Dataset ID 357
- **DOI (UCI Dataset)**: `10.24432/C5X01N`
- **DOI (Publication)**: `10.1016/j.enbuild.2015.11.071`
- **Primary Publication**: Candanedo, L. M., & Feldheim, V. (2016). *Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models*. Energy and Buildings, 112, 28-39.
- **Authors / Institution**: Luis M. Candanedo and V. Feldheim (University of Mons, Department of Thermal Engineering and Combustion, Mons, Belgium).

---

## 3. License and Permitted Use

- **License Name**: Creative Commons Attribution 4.0 International
- **SPDX License ID**: `CC-BY-4.0`
- **Research Use**: `VERIFIED_PERMITTED`
- **Model Training Use**: `VERIFIED_PERMITTED`
- **Redistribution**: `VERIFIED_PERMITTED_WITH_ATTRIBUTION`
- **Modified Redistribution**: `VERIFIED_PERMITTED_WITH_ATTRIBUTION`
- **Commercial Use**: `VERIFIED_PERMITTED_WITH_ATTRIBUTION`
- **Citation Requirement**: `VERIFIED_REQUIRED`
- **License Classification Status**: `VERIFIED`

---

## 4. Local Raw Archive Identity and Git Visibility

- **Repository-Relative Archive Path**: `datasets/raw_archives/external_datasets/occupancy+detection.zip`
- **Archive Materialization Status**: `MATERIALIZED_LOCAL` (Local payload accessible to execution environment)
- **Git Visibility**: `GIT_IGNORED` via `/datasets/raw_archives/` rule in `.gitignore`.
- **Measured Byte Size**: `335,713` bytes
- **Measured SHA-256 Hash**: `4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a`
- **Owner-Confirmed Match**: Measured size and SHA-256 match the repository owner's confirmed reference values.

---

## 5. Raw Archive Member Inventory and Measurements

The zip archive contains exactly 3 uncompressed text files:

| Member File | Uncompressed Size (Bytes) | SHA-256 Hash | Raw Data Rows | Occupancy 0 | Occupancy 1 |
|---|---|---|---|---|---|
| `datatraining.txt` | 596,674 | `b2c4d0ce2b9e4e453c476f7125ef31aeec2d1f5c7f5572d0e80de3df6521ab56` | 8,143 | 6,414 | 1,729 |
| `datatest.txt` | 200,766 | `1b92c7c1b2838963464fa891a610cf3c5db4becb7189189b29b330107a584c7f` | 2,665 | 1,693 | 972 |
| `datatest2.txt` | 699,664 | `d026d1bd5aeccd4aff4f3b3710d48e40613bd5fc370db7e61bbdcaa50d985095` | 9,752 | 7,703 | 2,049 |
| **Total** | **1,497,104** | — | **20,560** | **15,810 (76.89%)** | **4,750 (23.11%)** |

---

## 6. Schema Mismatch Audit

- **Raw Visible Header String**: `"date","Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"` (7 named fields)
- **Observed CSV Data Row Fields**: 8 comma-separated physical fields per line across all 20,560 rows.
- **Root Cause & Resolution**:
  - Physical Field 0: Unnamed 1-based integer index exported from an R/Pandas dataframe (`"1"`, `"2"`, ...).
  - Physical Field 1: Timestamp string (`"2015-02-04 17:51:00"`).
  - Physical Field 2: Temperature (°C, `float`).
  - Physical Field 3: Humidity (%, `float`).
  - Physical Field 4: Light (Lux, `float`).
  - Physical Field 5: CO₂ (ppm, `float`).
  - Physical Field 6: HumidityRatio (kg/kg, `float`).
  - Physical Field 7: Occupancy (0 or 1, target label).
- **Enforcement**: Future raw readers must map columns using physical index offsets (Field 0 = Row ID, Field 1 = Timestamp, Field 5 = CO₂, Field 7 = Occupancy) rather than positional index matching header field count.

---

## 7. Timestamp, Timeline, and Data Integrity Findings

- **Timestamp Format**: Standard ISO-like `YYYY-MM-DD HH:MM:SS` (100% parseable across all 20,560 rows).
- **Timezone Status**: `UNVERIFIED` (Source acquisition clock; UTC conversion is explicitly NOT claimed).
- **Chronological Order**: 0 timestamp reversals observed. Timestamps are monotonically increasing within each file.
- **Duplicate Records**: 0 duplicate full rows, 0 duplicate row IDs, 0 duplicate timestamps.
- **Missing / Non-finite Values**: 0 missing values, 0 NaN values, 0 Inf values across all columns.
- **Measured Numeric Ranges**:
  - `Temperature`: `19.00 °C` to `24.40 °C`
  - `Humidity`: `16.74 %` to `39.13 %`
  - `Light`: `0.00 Lux` to `1697.25 Lux`
  - `CO2`: `412.75 ppm` to `2076.50 ppm`
  - `HumidityRatio`: `0.002674 kg/kg` to `0.006476 kg/kg`

---

## 8. Target Label Semantics vs Safety Boundaries

- **Target Label Semantics**: The dataset label `Occupancy` represents physical room occupancy (0 = vacant, 1 = occupied).
- **Safety Boundary**: The dataset label does **not** represent:
  - Clinical respiratory apnea
  - Acute CO₂ toxic emergency / asphyxiation
  - SCD40 hardware health
  - SafeNest multisensor risk score
- **Enforcement**: C-A0 explicitly restricts label interpretations to room occupancy semantics.

---

## 9. Lineage Separation Registry

The project explicitly separates four distinct asset lineages:

1. **Lineage A (Real UCI Raw Source)**: `datasets/raw_archives/external_datasets/occupancy+detection.zip` (SHA-256: `4ae3f46...`, `REAL_EXTERNAL_SOURCE`, `GIT_IGNORED`).
2. **Lineage B (Synthetic Smoke Fixture)**: `datasets/co2/processed/co2_occupancy_v1.npz` (SHA-256: `bff5cd7...`, `SYNTHETIC_SMOKE_FIXTURE`, generated via `datasets/build_processed_npz.py`).
3. **Lineage C (Existing TFLite Model)**: `models/co2/co2_occupancy_int8_v0.1.0.tflite` (SHA-256: `3a8c86c...`, `CONFIRMED_SYNTHETIC_ONLY`, training lineage unverified).
4. **Lineage D (Existing Scaling Metadata)**: `models/co2/co2_scaling_metadata_v0.1.0.json` (SHA-256: `9195be0...`, `FIT_DATA_LINEAGE_UNVERIFIED`).

No attempt is made to claim that existing models or NPZ files were generated from the local UCI raw archive without direct proof.

---

## 10. Warnings and Exception Registry

The C-A0 audit records 5 non-blocking warnings in `anomalies_and_limitations.json`:

1. `HEADER_DATA_WIDTH_MISMATCH`: 7 header fields vs 8 physical data fields.
2. `SOURCE_TIMEZONE_UNVERIFIED`: Naive local clock timestamps without UTC offset.
3. `MODEL_TRAINING_LINEAGE_UNVERIFIED`: Existing TFLite model lineage unverified against raw ZIP.
4. `SCALER_FIT_LINEAGE_UNVERIFIED`: Existing scaling metadata fit data lineage unverified against raw ZIP.
5. `GROUP_INDEPENDENCE_NOT_VERIFIABLE`: Dataset gathered from a single office room across 3 continuous time windows.

---

## 11. Explicitly Deferred Work

- **C-A1**: Safe raw reader contract implementation.
- **C-A2**: Timestamp canonicalization and subject/session/temporal split policy definition.
- **C-A3**: Derived feature reconstruction (`CO2_slope` in ppm/min, history window duration, regression vs difference).
- **C-A4–C-A6**: Quality audit, full dataset conversion, and Phase A exit gate.
- **C-B**: Offline model comparison.
- **C-C**: SCD40 device-domain validation.

---

## 12. C-A1 Authorization Result

```text
C-A1 Authorized: YES
Reason:
- Official UCI dataset identity and CC-BY-4.0 license verified.
- Local raw archive readability and checksum (4ae3f46...) verified.
- Deterministic 3-member inventory (20,560 rows) generated.
- Header mismatch (7 header vs 8 data fields) resolved.
- Standalone validator passed (PASS_WITH_WARNINGS).
- Focused test suite passed (10/10 passed).
- Upstream regression test suite passed (117/117 passed).
- 0 raw payload files staged in Git.
- 0 mmWave, Thermal, or shared integration files modified.
```
