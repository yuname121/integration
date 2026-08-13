# SafeNest CO₂ Phase C-A2 — Timestamp Canonicalization, Temporal Blocks, and Grouping/Split Contract Audit Report

- Document Version: `01`
- Author: `Antigravity` (CO₂ Track Implementation Agent)
- Execution Date: `2026-08-10`
- Phase: `C-A2 — CO₂ Timestamp Canonicalization, Temporal Blocks, and Grouping/Split Contract`
- Target Dataset: UCI Occupancy Detection Dataset
- Status: `PASS_WITH_WARNINGS`
- C-A3 Authorization: `YES` (All foundational temporal block reconstruction and split contract criteria satisfied)

---

## 1. Executive Summary

Phase **C-A2** establishes a deterministic, evidence-backed temporal interpretation and group-aware split policy contract for the real UCI Occupancy Detection source rows read via the C-A1 safe reader ([`datasets/co2/raw_reader.py`](../../datasets/co2/raw_reader.py)).

The analysis confirmed that the 20,560 source rows form a contiguous 3-block chronological timeline from Feb 2, 2015 to Feb 18, 2015 with nominal 1-minute sampling (`60.0s` average cadence, 59-61s quantization jitter). Exactly **3** temporal acquisition blocks were reconstructed without any row loss.

Random row-wise splitting is explicitly **prohibited** to prevent autocorrelation data leakage. A group-aware future split contract was established (`BLOCK_02_DATATRAINING` -> `TRAIN`, `BLOCK_01_DATATEST` -> `VALIDATION`, `BLOCK_03_DATATEST2` -> `LOCKED_TEST`), along with strict rules requiring scaler statistics to be fit on `TRAIN` only and forbidding derived temporal features (e.g. `CO2_slope` in C-A3) from crossing temporal block boundaries.

The standalone C-A2 validator passed with 0 errors and 5 non-blocking warnings, authorizing entry into Phase **C-A3**.

---

## 2. Predecessor C-A1 Status & Source Archive Identity

- **Predecessor Phase**: `C-A1 — CO₂ Safe Raw Reader and Source-Row Contract`
- **Predecessor Manifest Directory**: `datasets/co2/manifests/c_a1_safe_reader/`
- **Predecessor Status**: Verified in canonical `main` branch (`9ef42f1 Merge pull request #15`).
- **Raw Archive Relative Path**: `datasets/raw_archives/external_datasets/occupancy+detection.zip`
- **Raw Archive SHA-256 Hash**: `4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a`
- **Source Observations Read**: `20,560` rows (100% read via `UCIOccupancyRawReader`).

---

## 3. Timestamp Semantics & Cadence Analysis

- **Raw Timestamp Format**: `YYYY-MM-DD HH:MM:SS` (e.g., `'2015-02-04 17:51:00'`).
- **Canonical Representation**: `YYYY-MM-DDTHH:MM:SS` (Timezone-naive ISO representation).
- **Timestamp Reference**: `SOURCE_ACQUISITION_CLOCK`
- **Source Timezone**: `UNVERIFIED` (Naive local clock readings; UTC conversion explicitly **not** claimed).
- **Sampling Cadence**:
  - Official documented sampling interval: `60 seconds` (1 minute).
  - Observed dominant adjacent delta: `60.0 seconds` (60.0% of all adjacent pairs).
  - Observed adjacent delta range: `59.0` to `61.0` seconds (Integer second timestamp quantization jitter; 59.0s and 61.0s counts are 100% symmetric, yielding an average cadence of exactly `60.0000` seconds).
  - Timestamp reversals: `0`
  - Duplicate timestamps: `0`

---

## 4. Reconstructed Temporal Acquisition Blocks

The 20,560 source rows partition deterministically into exactly **3** contiguous temporal acquisition blocks matching the 3 raw source files:

| Block ID | Member File | Row Count | Start Timestamp (Canonical) | End Timestamp (Canonical) | Duration | Preceding Gap | Occupancy Distribution |
|---|---|---|---|---|---|---|---|
| `BLOCK_01_DATATEST` | `datatest.txt` | 2,665 | `2015-02-02T14:19:00` | `2015-02-04T10:43:00` | 44.40 hours | None (Start) | Vacant: 1,693 / Occupied: 972 |
| `BLOCK_02_DATATRAINING` | `datatraining.txt` | 8,143 | `2015-02-04T17:51:00` | `2015-02-10T09:33:00` | 135.70 hours | 7.13 hours (25,680s) | Vacant: 6,414 / Occupied: 1,729 |
| `BLOCK_03_DATATEST2` | `datatest2.txt` | 9,752 | `2015-02-11T14:48:00` | `2015-02-18T09:19:00` | 162.52 hours | 29.25 hours (105,300s) | Vacant: 7,703 / Occupied: 2,049 |
| **Total** | — | **20,560** | **2015-02-02T14:19:00** | **2015-02-18T09:19:00** | **15.79 days** | — | **Vacant: 15,810 / Occupied: 4,750** |

- **Rows Omitted**: `0`
- **Duplicate Block Membership**: `0`

---

## 5. Cross-Member Chronology & Inter-Block Discontinuities

The 3 raw dataset files form a non-overlapping, strictly chronological sequence spanning 15.79 days:

```text
datatest.txt (Feb 2-4) ──[7.13h Gap]──> datatraining.txt (Feb 4-10) ──[29.25h Gap]──> datatest2.txt (Feb 11-18)
```

1. **Gap 1 (7.13 hours / 25,680s)**: Inter-block discontinuity between Block 1 (`datatest.txt`) end and Block 2 (`datatraining.txt`) start.
2. **Gap 2 (29.25 hours / 105,300s)**: Inter-block discontinuity between Block 2 (`datatraining.txt`) end and Block 3 (`datatest2.txt`) start.

---

## 6. Grouping Unit & Leakage Prevention Contract

- **Strongest Defensible Grouping Unit**: `TEMPORAL_ACQUISITION_BLOCK`.
- **Group Independence Status**: `GROUP_INDEPENDENCE_NOT_VERIFIABLE` (All 3 blocks originate from a single office room over continuous time windows. Cross-room/cross-building generalization cannot be claimed).
- **Random Row-Wise Split Policy**: `PROHIBITED` (`allowed = False`). Adjacent time-series samples are highly autocorrelated; random shuffling causes severe evaluation data leakage.

---

## 7. SafeNest Future Split Policy Contract

The C-A2 phase establishes the machine-readable split assignment contract for future phases:

| Future ML Split Role | Assigned Temporal Block | Original UCI Source File | Row Count | Percentage | Occupancy 0 / 1 |
|---|---|---|---|---|---|
| `TRAIN` | `BLOCK_02_DATATRAINING` | `datatraining.txt` | 8,143 | 39.61% | 6,414 / 1,729 |
| `VALIDATION` | `BLOCK_01_DATATEST` | `datatest.txt` | 2,665 | 12.96% | 1,693 / 972 |
| `LOCKED_TEST` | `BLOCK_03_DATATEST2` | `datatest2.txt` | 9,752 | 47.43% | 7,703 / 2,049 |

### Invariant Rules for Future Phases

1. **Scaler Fit Rule**: Any scaler or normalization statistics in future phases **must be fit on `TRAIN` (`BLOCK_02_DATATRAINING`) only**. `VALIDATION` and `LOCKED_TEST` must never contribute to scaler fitting.
2. **Feature History Cross-Block Isolation Rule**: Derived temporal features (such as `CO2_slope` in Phase C-A3) **must never cross temporal block boundaries**. History windows must be reset at the start of each temporal acquisition block.
3. **No Model Feature / NPZ Generation in C-A2**: No `CO2_slope` was created, no normalization was performed, and no model-ready NPZ tensors were generated in C-A2.

---

## 8. Non-blocking Limitations & Warnings

The C-A2 validator records 5 non-blocking warnings:

1. `HEADER_DATA_WIDTH_MISMATCH`: 7 named header fields vs 8 physical data fields.
2. `SOURCE_TIMEZONE_UNVERIFIED`: Source timestamps are timezone-naive local clock readings.
3. `MODEL_TRAINING_LINEAGE_UNVERIFIED`: Existing TFLite model lineage unverified against raw source.
4. `SCALER_FIT_LINEAGE_UNVERIFIED`: Existing scaling metadata fit data lineage unverified against raw source.
5. `GROUP_INDEPENDENCE_NOT_VERIFIABLE`: All 3 temporal acquisition blocks originate from a single office room over continuous time windows.

---

## 9. Explicitly Deferred Work

- **C-A3**: Derived feature reconstruction (`CO2_slope` in ppm/min, history window duration, regression vs difference, warm-up handling).
- **C-A4–C-A6**: Quality audit, full dataset conversion, and Phase A exit gate.
- **C-B**: Offline model comparison.
- **C-C**: SCD40 device-domain validation.

---

## 10. C-A3 Authorization Result

```text
C-A3 Authorized: YES
Reason:
- Predecessor C-A1 evidence verified and validator passed.
- All 20,560 source timestamps parsed and canonicalized deterministically.
- 0 timestamp reversals, 0 duplicate timestamps.
- Sampling cadence audited (dominant 60s sampling, 59-61s quantization jitter).
- Exactly 3 contiguous temporal acquisition blocks reconstructed with 100% row assignment (0 row loss).
- Group-aware split policy contract defined (Random row-wise split PROHIBITED).
- Invariant rules established (TRAIN-only scaler fit, feature history cross-block isolation for C-A3).
- No CO2_slope derived, no measurements normalized.
- Standalone validator passed (PASS_WITH_WARNINGS).
- Focused test suite passed (10/10 passed).
- Combined regression test suite passed (31/31 passed).
- 0 raw payload files staged in Git index.
- 0 mmWave, Thermal, or shared integration files modified.
```
