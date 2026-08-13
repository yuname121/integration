# SafeNest mmWave M-B3 — TinyML Architecture Comparison Report (Pinned Environment)

- **Author**: Antigravity Implementation Engineer
- **Date**: 2026-08-10
- **Target Repository**: `https://github.com/sheepmeat/test.git`
- **Branch**: `feature/M-B3-architecture-comparison`
- **Phase M-B3 Gate Status**: `PASS_WITH_WARNINGS`
- **M-B4 Entry Status**: `READY_WITH_CONDITIONS`
- **Pinned Environment**: Python 3.9.6 / TensorFlow 2.20.0 / NumPy 1.26.4 / SciPy 1.13.1 (`requirements-mac.txt` compliant)
- **Frozen Preprocessing Profile**: `M-B1_D0_B1_Z1` (`BPF_ZSCORE`)
- **Frozen Class-Imbalance Strategy**: `M-B2_CE_UNWEIGHTED` (`CE_UNWEIGHTED`)
- **Selected Shortlisted Deployment Architectures**: `M-B3_CONV1D_GAP_BASELINE, M-B3_SEPARABLECONV1D_GAP`

---

## 1. Executive Summary

Phase M-B3 compares three pre-registered TinyML model architectures (**Conv1D+GAP Baseline**, **SeparableConv1D+GAP**, and **Conv1D+BiLSTM**) under frozen M-B1 `BPF_ZSCORE` preprocessing and frozen M-B2 `CE_UNWEIGHTED` imbalance strategy in the pinned macOS environment.

Key achievements of Phase M-B3:
1. **Frozen Lineage & Baseline Equivalence**: Preserved frozen M-B1 BPF and TRAIN-fitted Z-score statistics. Architecture A (`M-B3_CONV1D_GAP_BASELINE`) reproduced the frozen M-B2 CE_UNWEIGHTED baseline with 100% parameter, weight SHA, prediction vector, and metric match.
2. **TinyML Screening & INT8 Qualification**: Evaluated Float Keras, Float TFLite, and Strict INT8 TFLite models using the frozen all-TRAIN compatibility representative dataset (`M-B3_COMPATIBILITY_REPSET_ALL_TRAIN_001`, 327 samples).
3. **Deployment Shortlist Selection**:
   - `M-B3_CONV1D_GAP_BASELINE` (9315 params): Float Macro F1 = **`0.663708`**, Strict INT8 = `FULL_INT8_SUPPORTED` (22080 bytes), Eligible.
   - `M-B3_SEPARABLECONV1D_GAP` (3258 params): Float Macro F1 = **`0.470833`**, Strict INT8 = `FULL_INT8_SUPPORTED` (19072 bytes), Eligible.
   - `M-B3_CONV1D_BILSTM` (19747 params): Float Macro F1 = `0.467728`, Strict INT8 = `STRICT_INT8_UNSUPPORTED` (SELECT_TF_OPS_REQUIRED), Excluded from deployment shortlist.
4. **Strict LOCKED_TEST Isolation**: Confirmed `0` performance access attempts to LOCKED_TEST (`scripts/mmwave_phase_b_access.py` guard verified).
5. **Deterministic Rerun Verification**: Verified 100% initial/final weight SHA and prediction match when rerunning shortlisted architectures under fixed initialization seed `42`.

---

## 2. Architecture Comparison Results (Pinned Environment)

| Architecture ID | Name | Parameters | Float Macro F1 | Float Accuracy | APNEA Proxy Recall | RAPID Recall | Strict INT8 Success | Deployment Eligibility |
|---|---|---|---|---|---|---|---|---|
| `M-B3_CONV1D_GAP_BASELINE` | `Conv1D + GAP Baseline` | `9315` | `0.663708` | `0.721519` | `1.0000` | `0.5000` | `YES` | `DEPLOYMENT_ELIGIBLE_SINGLE_SEED` |
| `M-B3_SEPARABLECONV1D_GAP` | `SeparableConv1D + GAP` | `3258` | `0.470833` | `0.594937` | `1.0000` | `0.3000` | `YES` | `DEPLOYMENT_ELIGIBLE_SINGLE_SEED` |
| `M-B3_CONV1D_BILSTM` | `Conv1D + BiLSTM` | `19747` | `0.467728` | `0.607595` | `0.9459` | `0.0500` | `NO` | `SELECT_TF_OPS_REQUIRED` |

---

## 3. Deployment Shortlist Rationale

Under the pre-registered ranking rules:
1. **`M-B3_CONV1D_GAP_BASELINE`** (Rank 1): Float Macro F1 = `0.663708`, strict full-INT8 TFLite compatible (TFLITE_BUILTINS_INT8 only, 22080 bytes).
2. **`M-B3_SEPARABLECONV1D_GAP`** (Rank 2): Float Macro F1 = `0.470833`, strict full-INT8 TFLite compatible (TFLITE_BUILTINS_INT8 only, 19072 bytes).
3. **`M-B3_CONV1D_BILSTM`** (Excluded): Strict INT8 conversion failed (`SELECT_TF_OPS_REQUIRED`). Excluded from TinyML deployment shortlist.

---

## 4. Subject-Level Diagnostic Summary for Winner (M-B3_CONV1D_GAP_BASELINE)

- Subject Count: `17`
- Mean Subject Accuracy: `0.711485` (median = `0.750000`, std = `0.168781`)
- Mean Subject Macro F1: `0.610598` (median = `0.600000`, std = `0.256498`)
- Min / Max Subject Macro F1: `0.222222` / `1.000000`

---

## 5. Validation & Exit Gate Summary

- Standalone M-B3 validator (`scripts/validate_mmwave_m_b3.py`): `PASS`
- Standalone M-B2 validator (`scripts/validate_mmwave_m_b2.py`): `PASS`
- Standalone M-B1 validator (`scripts/validate_mmwave_m_b1.py`): `PASS`
- Standalone M-B0 validator (`scripts/validate_mmwave_m_b0.py`): `PASS`
- Upstream M-A5 validator (`scripts/validate_mmwave_subject_split.py`): `PASS`
- Upstream M-A6 validator (`scripts/validate_mmwave_full_conversion.py`): `PASS`
- Checksum Coverage: All 18 machine-readable manifests checksummed in `checksums.sha256`
- M-B3 Gate Status: `PASS_WITH_WARNINGS`
- M-B4 Entry Status: `READY_WITH_CONDITIONS`
