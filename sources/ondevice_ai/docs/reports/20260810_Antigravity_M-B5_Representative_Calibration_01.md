# SafeNest mmWave M-B5 — Representative Calibration Dataset Comparison Report

- **Author**: Cursor Implementation / Validation Engineer
- **Date**: 2026-08-10
- **Target Repository**: `https://github.com/sheepmeat/test.git`
- **Branch**: `feature/M-B5-representative-calibration`
- **Phase M-B5 Gate Status**: `PASS_WITH_WARNINGS`
- **M-B6 Entry Status**: `READY_WITH_CONDITIONS`
- **Pinned Environment**: Python 3.9.6 / TensorFlow 2.20.0 / NumPy 1.26.4 / SciPy 1.13.1
- **Frozen Primary Architecture**: `M-B3_CONV1D_GAP_BASELINE`
- **Frozen Weight Seeds**: `[42, 43, 44]`
- **TRAIN Population**: 327 pure-class windows (77 subjects)
- **VALIDATION Population**: 79 pure-class windows (17 subjects)
- **Selected Calibration Profile**: `M-B5_CAL_CLASS_BALANCED_120`
- **Ranking Deciding Criterion**: `CRITERION_4_MAX_OUTPUT_PROBABILITY_MAE`
- **Tie Tolerance**: `eps=1e-5`
- **Profile-D conversions rerun this closure**: `3`
- **Profiles A/B/C conversions rerun this closure**: `0`
- **Neural-network models retrained**: `0`

---

## 1. Executive Summary

Phase M-B5 compares four pre-registered TRAIN-only representative calibration dataset profiles across the three frozen M-B4 primary-architecture weight sets (`42`, `43`, `44`). M-B5 selects a **calibration profile**, not a model seed.

M-B4 already demonstrated substantial **initialization-seed sensitivity** for `M-B3_CONV1D_GAP_BASELINE` (seed42 Macro F1≈0.663708, seed43≈0.451010, seed44≈0.329107). That scientific result is preserved; M-B5 measures quantization behavior under those frozen weights.

Profile D (`M-B5_CAL_DISTRIBUTION_AWARE_120`) was repaired to use authoritative TRAIN metadata:
- posture vocabulary: `['Lying', 'Sitting']`
- source_test_condition vocabulary: `['Post-exercise', 'Rest']`
- unknown/missing policy: `UNKNOWN_OR_MISSING`
- continuous features: `['peak_abs', 'RMS', 'p01', 'p99', 'dynamic_range']`
- subject-cap final state: `MAX_2`

Final selected calibration profile under the preregistered epsilon-aware ranking rule: **`M-B5_CAL_CLASS_BALANCED_120`** (decided by `CRITERION_4_MAX_OUTPUT_PROBABILITY_MAE`).

LOCKED_TEST performance access attempts: **0**. Formal M-B6 Float Keras → Float TFLite → INT8 stage equivalence remains pending. MR60 hardware and Raspberry Pi performance are **not** validated. APNEA remains a voluntary breath-hold proxy, not clinical apnea. This phase does **not** claim production/deployment readiness.

---

## 2. Cross-Seed Calibration Profile Performance Matrix (VALIDATION Set)

| Profile ID | Eligibility | Worst F1 Deg. | Worst Rec Deg. | Min Top-1 | Max Output MAE | Max Input Sat. | Max End. Ratio |
|---|---|---|---|---|---|---|---|
| `M-B5_CAL_TRAIN_ORDER_120` | `ELIGIBLE` | `0.009770` | `0.081082` | `0.936709` | `0.008551` | `0.000000` | `0.000000` |
| `M-B5_CAL_RANDOM_PROPORTIONAL_120` | `ELIGIBLE` | `0.016925` | `0.054054` | `0.962025` | `0.008554` | `0.000000` | `0.000000` |
| `M-B5_CAL_CLASS_BALANCED_120` | `ELIGIBLE` | `0.009770` | `0.081082` | `0.936709` | `0.008439` | `0.000000` | `0.000000` |
| `M-B5_CAL_DISTRIBUTION_AWARE_120` | `ELIGIBLE` | `0.009770` | `0.081082` | `0.936709` | `0.009179` | `0.000000` | `0.000000` |

---

## 3. Selected Profile Details

Selected Calibration Profile: **`M-B5_CAL_CLASS_BALANCED_120`**
- Worst Positive Macro F1 Degradation: `0.00977`
- Worst Positive Recall Degradation: `0.081082`
- Minimum Top-1 Agreement: `0.936709`
- Maximum Output Probability MAE: `0.008439`
- Maximum Input Saturation Ratio: `0.0`
- Maximum Output Endpoint Ratio: `0.0`

Selected-profile per-seed diagnostics:
- Seed 42: Float Macro F1=`0.663708`, INT8 Macro F1=`0.666231`, Top-1=`0.987342`, MAE=`0.006319`, Input sat=`0.000000`
- Seed 43: Float Macro F1=`0.451010`, INT8 Macro F1=`0.441240`, Top-1=`0.936709`, MAE=`0.001030`, Input sat=`0.000000`
- Seed 44: Float Macro F1=`0.329107`, INT8 Macro F1=`0.329107`, Top-1=`1.000000`, MAE=`0.008439`, Input sat=`0.000000`

Selected-profile three-seed conversion replay functional equality: `True`

---

## 4. Limitations & Scope

- Inherited immutable A5 subject split (TRAIN=77 subjects, VALIDATION=17 subjects).
- LOCKED_TEST remained unused for representative sampling, calibration, ranking, and mismatch inspection.
- No clinical apnea claims.
- M-B6 formal stage equivalence is still pending.
- MR60 / Raspberry Pi hardware validation is not claimed.

---

## 5. Validation & Exit Gate Summary

- Standalone M-B5 validator (`scripts/validate_mmwave_m_b5.py`) must pass independently against these artifacts.
- Checksum coverage: 19 machine-readable manifests in `checksums.sha256`
- M-B5 Gate Status: `PASS_WITH_WARNINGS`
- M-B6 Entry Status: `READY_WITH_CONDITIONS`
