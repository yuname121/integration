# SafeNest mmWave M-B6 — Stage-Equivalence Validation Report

- **Author**: Antigravity Implementation Engineer
- **Date**: 2026-08-10
- **Target Repository**: `https://github.com/sheepmeat/test.git`
- **Branch**: `feature/M-B6-stage-equivalence`
- **Phase M-B6 Gate Status**: `PASS_WITH_WARNINGS`
- **M-B7 Entry Status**: `READY_WITH_CONDITIONS`
- **Pinned Environment**: Python 3.9.6 / TensorFlow 2.20.0 / NumPy 1.26.4 / SciPy 1.13.1 (`requirements-mac.txt` compliant)
- **Frozen Primary Architecture**: `M-B3_CONV1D_GAP_BASELINE`
- **Frozen Selected Calibration Profile**: `M-B5_CAL_CLASS_BALANCED_120`
- **Frozen Seeds**: `[42, 43, 44]`

---

## 1. Executive Summary

Phase M-B6 measures formal three-stage conversion equivalence across **Stage A (Float Keras)**, **Stage B (unoptimized Float32 TFLite)**, and **Stage C (selected-profile strict INT8 TFLite)** for all three frozen M-B4 initialization seeds (`42`, `43`, `44`).

Key findings:
1. **Stage A → B (Float Keras → Float TFLite)**: Perfect functional equivalence (`1.000000` Top-1 agreement, `0.000000` probability MAE) across all 3 seeds.
2. **Stage B → C / Stage A → C (Float → Strict INT8)**: Quantization drift matches M-B5 evidence. Cross-seed A->C worst positive Macro F1 degradation is `0.009770`, with minimum Top-1 agreement of `0.936709`.
3. **Class Collapse Transitions**: Zero new conversion-induced class collapses detected across all stages.
4. **LOCKED_TEST Guard**: Confirmed `0` performance access attempts to LOCKED_TEST (`scripts/mmwave_phase_b_access.py` guard verified).

---

## 2. Stage-Equivalence Matrix Across Frozen Seeds

| Seed | Stage A (Float Keras) F1 | Stage B (Float TFLite) F1 | Stage C (Strict INT8) F1 | A->B Top-1 Agree | B->C Top-1 Agree | A->C Top-1 Agree | A->C Prob MAE | A->C F1 Deg. |
|---|---|---|---|---|---|---|---|---|
| `42` | `0.663708` | `0.663708` | `0.666231` | `1.000000` | `0.987342` | `0.987342` | `0.006319` | `0.000000` |
| `43` | `0.451010` | `0.451010` | `0.441240` | `1.000000` | `0.936709` | `0.936709` | `0.001030` | `0.009770` |
| `44` | `0.329107` | `0.329107` | `0.329107` | `1.000000` | `1.000000` | `1.000000` | `0.008439` | `0.000000` |

---

## 3. Limitations & Scope

- **Fixed Subject Split**: Inherited immutable A5 subject split (TRAIN=77 subjects, VALIDATION=17 subjects).
- **LOCKED_TEST Preserved**: LOCKED_TEST (20 subjects) remained strictly un-accessed (0 access attempts).
- **No Clinical Claims**: Voluntary breath-hold labels remain APNEA proxies, not clinical apnea.
- **Hardware Validation Unverified**: Hardware performance on MR60 real sensor and Raspberry Pi remains unverified until hardware testing.

---

## 4. Validation & Exit Gate Summary

