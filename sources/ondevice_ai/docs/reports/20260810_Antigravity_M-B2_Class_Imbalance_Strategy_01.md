# SafeNest mmWave M-B2 — Real-Data Class-Imbalance Strategy Comparison Report (Pinned Environment)

- **Author**: Antigravity Implementation Engineer
- **Date**: 2026-08-10
- **Target Repository**: `https://github.com/sheepmeat/test.git`
- **Branch**: `feature/M-B2-class-imbalance`
- **Phase M-B2 Gate Status**: `PASS_WITH_WARNINGS`
- **M-B3 Entry Status**: `READY_WITH_CONDITIONS`
- **Pinned Environment**: Python 3.9.6 / TensorFlow 2.20.0 / NumPy 1.26.4 / SciPy 1.13.1 (`requirements-mac.txt` compliant)
- **Frozen Preprocessing Profile**: `M-B1_D0_B1_Z1` (`BPF_ZSCORE`)
- **Selected Class-Imbalance Strategy**: `M-B2_CE_UNWEIGHTED` (`CE_UNWEIGHTED`)

---

## 1. Executive Summary

Phase M-B2 compares four pre-registered class-imbalance handling strategies (**Standard Unweighted CE**, **Real-TRAIN Class Weighting**, **TRAIN-Only Random Oversampling**, and **Multiclass Focal Loss with $\gamma=2.0$**) on the canonical real mmWave dataset with frozen M-B1 `BPF_ZSCORE` preprocessing in the pinned macOS environment.

Key achievements of Phase M-B2:
1. **Frozen Preprocessing & Lineage Hardening**: Preserved frozen M-B1 BPF and TRAIN-fitted Z-score statistics. Verified 100% tensor fingerprint match with M-B1.
2. **Real-TRAIN Imbalance Evidence**: Audited pure-class TRAIN distribution (102 NORMAL, 80 RAPID, 145 APNEA; total 327 windows across 77 subjects). Derived inverse-frequency weights ($w_0=1.0686$, $w_1=1.3625$, $w_2=0.7517$) and minority-only oversampling plan (435 windows total: 102+43 NORMAL, 80+65 RAPID, 145+0 APNEA) exclusively from TRAIN data.
3. **Controlled Imbalance Comparison**: Trained all 4 strategies under identical fixed probe architecture (9,315 params), fixed initial weights SHA-256 (`03253f5697701f5fe7dce436d1368320936d9ba837432e2d8f2710e6fa93a6e3`), and seed 42.
4. **Strategy Winner Selection**: Under the pre-registered 7-step ranking rule, strategy **`M-B2_CE_UNWEIGHTED` (`CE_UNWEIGHTED`)** achieved highest VALIDATION Macro F1 = **`0.663708`**, Accuracy = `0.721519`, APNEA Recall = `1.000000`.
5. **Strict Prediction Index Provenance**: Generated `validation_prediction_index.jsonl` establishing 1:1 window mapping strictly for the 79 VALIDATION samples across 17 subjects with `0` TRAIN or LOCKED_TEST exposure.
6. **Strict LOCKED_TEST Isolation**: Confirmed `0` performance access attempts to LOCKED_TEST (`scripts/mmwave_phase_b_access.py` guard verified).
7. **Deterministic Rerun Verification**: Verified 100% prediction match when rerunning `M-B2_CE_UNWEIGHTED` under fixed initialization seed `42`.

---

## 2. Class-Imbalance Strategy Performance Results (Pinned Environment)

| Strategy ID | Name | Macro F1 | Macro Precision | Macro FPR | Accuracy | APNEA Proxy Recall | RAPID Recall | Class Collapsed |
|---|---|---|---|---|---|---|---|---|
| `M-B2_CE_UNWEIGHTED` | `CE_UNWEIGHTED` | `0.663708` | `0.737179` | `0.158992` | `0.721519` | `1.0000` | `0.5000` | `NO` |
| `M-B2_CE_CLASS_WEIGHT` | `CE_CLASS_WEIGHT` | `0.663341` | `0.693603` | `0.148748` | `0.721519` | `0.9459` | `0.4000` | `NO` |
| `M-B2_CE_RANDOM_OVERSAMPLE` | `CE_RANDOM_OVERSAMPLE` | `0.642179` | `0.708770` | `0.165236` | `0.708861` | `1.0000` | `0.4000` | `NO` |
| `M-B2_FOCAL_CLASS_ALPHA` | `FOCAL_CLASS_ALPHA` | `0.603019` | `0.605348` | `0.172819` | `0.658228` | `0.8649` | `0.4000` | `NO` |

---

## 3. Strategy Selection & Ranking Rationale

Under the pre-registered 7-step ranking rule:
1. **Class-Collapse Filtering**: Evaluated all 4 strategies for zero recall on APNEA proxy or RAPID classes. Zero strategies collapsed.
2. **Macro F1 Ranking**: Strategy **`M-B2_CE_UNWEIGHTED`** achieved the highest VALIDATION Macro F1 (**`0.663708`**).
3. **Selected Strategy Contract**: `M-B2_CE_UNWEIGHTED` (`CE_UNWEIGHTED`) is frozen in `selected_imbalance_strategy.json` for subsequent Phase M-B3 experiments.

---

## 4. Subject-Level Diagnostic Summary Across 17 Validation Subjects

- Subject Count: `17`
- Mean Subject Accuracy: `0.711485` (median = `0.750000`, std = `0.168781`)
- Mean Subject Macro F1: `0.610598` (median = `0.600000`, std = `0.256498`)
- Min / Max Subject Macro F1: `0.222222` / `1.000000`

---

## 5. Validation & Exit Gate Summary

- Fixed Probe Model Parameter Count: `9315`
- Standalone M-B2 validator (`scripts/validate_mmwave_m_b2.py`): `PASS` (`validation_success: True`)
- Standalone M-B1 validator (`scripts/validate_mmwave_m_b1.py`): `PASS`
- Standalone M-B0 validator (`scripts/validate_mmwave_m_b0.py`): `PASS`
- Upstream M-A5 validator (`scripts/validate_mmwave_subject_split.py`): `PASS`
- Upstream M-A6 validator (`scripts/validate_mmwave_full_conversion.py`): `PASS`
- Checksum Coverage: All 19 machine-readable manifests checksummed in `checksums.sha256`
- M-B2 Gate Status: `PASS_WITH_WARNINGS`
- M-B3 Entry Status: `READY_WITH_CONDITIONS`
