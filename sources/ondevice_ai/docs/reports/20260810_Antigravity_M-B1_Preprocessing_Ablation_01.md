# SafeNest mmWave M-B1 — Real-Data Preprocessing Full-Factorial Ablation Report (Pinned Environment)

- **Author**: Antigravity Implementation Engineer
- **Date**: 2026-08-10
- **Target Repository**: `https://github.com/sheepmeat/test.git`
- **Branch**: `feature/M-B1-clean-final`
- **Phase M-B1 Gate Status**: `PASS_WITH_WARNINGS`
- **M-B2 Entry Status**: `READY_WITH_CONDITIONS`
- **Pinned Environment**: Python 3.9.6 / TensorFlow 2.20.0 / NumPy 1.26.4 / SciPy 1.13.1 (`requirements-mac.txt` compliant)
- **Selected Preprocessing Profile**: `M-B1_D0_B1_Z1` (`BPF_ZSCORE`)

---

## 1. Executive Summary

Phase M-B1 conducts a $2^3$ full-factorial offline preprocessing ablation experiment over **Linear Detrending ($D$)**, **Fixed 0.1–0.5 Hz 4th-order Butterworth BPF ($B$)**, and **TRAIN-fitted Global Z-score Standardization ($Z$)** on the approved real mmWave canonical dataset (`mmwave_canonical_real_v1.npy`, 530 windows) in the pinned macOS execution environment.

Key achievements of Phase M-B1 Refinement:
1. **Pinned Environment Execution**: Reproduced the complete $2^3$ full-factorial ablation experiment under pinned `numpy==1.26.4`, `tensorflow==2.20.0`, `scipy==1.13.1`.
2. **Winner Selection**: Under pinned environment `numpy==1.26.4`, profile **`M-B1_D0_B1_Z1` (`BPF_ZSCORE`)** achieved highest VALIDATION Macro F1 = **`0.663708`**, Accuracy = `0.721519`, APNEA Recall = `1.000000` under the pre-registered 6-step ranking rule.
3. **Reproducibility Analysis**: Compared pinned NumPy 1.26.4 results directly against historical NumPy 2.0.2 results. Historical winner `DETREND_ONLY` (Macro F1 = 0.652975) was superseded by `BPF_ZSCORE` (Macro F1 = 0.663708).
4. **Hardened Upstream Identity Chain**: Standalone validator independently verified the immutable M-B0 checksum chain (`checksums.sha256`), M-B0 evaluation contract, A5 subject split, A6 canonical NPY, and A6 window manifest.
5. **Strict Prediction Index Provenance**: Generated `validation_prediction_index.jsonl` establishing 1:1 window mapping strictly for the 79 VALIDATION samples with `0` TRAIN or LOCKED_TEST exposure.
6. **Strict LOCKED_TEST Isolation**: Confirmed `0` performance access attempts to LOCKED_TEST (`scripts/mmwave_phase_b_access.py` guard verified).
7. **Deterministic Rerun Verification**: Verified 100% prediction match when rerunning `M-B1_D0_B1_Z1` under fixed initialization seed `42`.

---

## 2. Full-Factorial Ablation Performance Results (Pinned Environment)

| Profile ID | Name | Detrend ($D$) | BPF ($B$) | Z-Score ($Z$) | Macro F1 | Accuracy | APNEA Proxy Recall | RAPID Recall | Class Collapsed |
|---|---|---|---|---|---|---|---|---|---|
| `M-B1_D0_B0_Z0` | `RAW` | `OFF` | `OFF` | `OFF` | `0.5784` | `0.6709` | `0.9730` | `0.2500` | `NO` |
| `M-B1_D1_B0_Z0` | `DETREND_ONLY` | `ON` | `OFF` | `OFF` | `0.6530` | `0.7215` | `0.9730` | `0.3500` | `NO` |
| `M-B1_D0_B1_Z0` | `BPF_ONLY` | `OFF` | `ON` | `OFF` | `0.6179` | `0.6456` | `0.7027` | `0.4500` | `NO` |
| `M-B1_D1_B1_Z0` | `DETREND_BPF` | `ON` | `ON` | `OFF` | `0.6141` | `0.6456` | `0.7027` | `0.4000` | `NO` |
| `M-B1_D0_B0_Z1` | `ZSCORE_ONLY` | `OFF` | `OFF` | `ON` | `0.2763` | `0.4937` | `1.0000` | `0.1000` | `NO` |
| `M-B1_D1_B0_Z1` | `DETREND_ZSCORE` | `ON` | `OFF` | `ON` | `0.2126` | `0.4684` | `1.0000` | `0.0000` | `YES` |
| `M-B1_D0_B1_Z1` | `BPF_ZSCORE` | `OFF` | `ON` | `ON` | `0.6637` | `0.7215` | `1.0000` | `0.5000` | `NO` |
| `M-B1_D1_B1_Z1` | `DETREND_BPF_ZSCORE` | `ON` | `ON` | `ON` | `0.6113` | `0.6835` | `1.0000` | `0.3500` | `NO` |

---

## 3. Winner Selection & Ranking Rationale

Under the pre-registered 6-step ranking rule:
1. **Class-Collapse Filtering**: Evaluated all 8 profiles for zero recall or prediction collapse on APNEA proxy or RAPID classes. Profile `M-B1_D1_B0_Z1` collapsed on RAPID class (recall = 0.0) and was rejected.
2. **Macro F1 Ranking**: Profile **`M-B1_D0_B1_Z1`** achieved the highest VALIDATION Macro F1 (**`0.663708`**).
3. **Selected Profile Contract**: `M-B1_D0_B1_Z1` (`BPF_ZSCORE`) is frozen in `selected_preprocessing_profile.json` for subsequent Phase-B experiments.

---

## 4. Environment Reproducibility Comparison (NumPy 2.0.2 vs Pinned NumPy 1.26.4)

- **Historical Winner**: `M-B1_D1_B0_Z0` (`DETREND_ONLY`, Macro F1 = 0.652975)
- **Pinned Winner**: `M-B1_D0_B1_Z1` (`BPF_ZSCORE`, Macro F1 = 0.663708)
- **Winner Changed**: YES (`M-B1_D1_B0_Z0` superseded by `M-B1_D0_B1_Z1`)
- **Verdict**: `WINNER_CHANGED (Historical NumPy 2.0.2 selected DETREND_ONLY, whereas pinned NumPy 1.26.4 selected BPF_ZSCORE)`

### Measured Profile Comparisons

| Profile ID | Name | Old NumPy 2.0.2 Macro F1 | New Pinned NumPy 1.26.4 Macro F1 | Delta Macro F1 | Status |
|---|---|---|---|---|---|
| `M-B1_D0_B0_Z0` | `RAW` | `0.578420` | `0.578435` | `+0.000015` | `PREDICTIONS_DIFFERENT` |
| `M-B1_D1_B0_Z0` | `DETREND_ONLY` | `0.652975` | `0.652975` | `+0.000000` | `PREDICTIONS_DIFFERENT` |
| `M-B1_D0_B1_Z0` | `BPF_ONLY` | `0.617935` | `0.617859` | `-0.000076` | `PREDICTIONS_DIFFERENT` |
| `M-B1_D1_B1_Z0` | `DETREND_BPF` | `0.626101` | `0.614146` | `-0.011955` | `PREDICTIONS_DIFFERENT` |
| `M-B1_D0_B0_Z1` | `ZSCORE_ONLY` | `0.276332` | `0.276260` | `-0.000072` | `PREDICTIONS_DIFFERENT` |
| `M-B1_D1_B0_Z1` | `DETREND_ZSCORE` | `0.212598` | `0.212644` | `+0.000046` | `IDENTICAL` |
| `M-B1_D0_B1_Z1` | `BPF_ZSCORE` | `0.622384` | `0.663708` | `+0.041324` | `PREDICTIONS_DIFFERENT` |
| `M-B1_D1_B1_Z1` | `DETREND_BPF_ZSCORE` | `0.608933` | `0.611265` | `+0.002332` | `PREDICTIONS_DIFFERENT` |

---

## 5. Signal Domain & Diagnostic Results

### 5.1 BPF Frequency Response Diagnostic (0.1–0.5 Hz, 4th Order)
- **30 bpm (0.50 Hz)**: -3.0 dB attenuation (gain 0.707)
- **40 bpm (0.67 Hz)**: -14.6 dB attenuation (gain 0.186)
- **48 bpm (0.80 Hz)**: -20.5 dB attenuation (gain 0.094)
- **Finding**: The 0.1–0.5 Hz BPF naturally suppresses respiration frequencies above 30 bpm. This filter parameter is frozen for M-B1 and will be evaluated for potential tuning in later phases if required.

### 5.2 APNEA-Proxy Preprocessing Diagnostic
- Voluntary breath-hold APNEA proxy windows retain near-zero respiration amplitude characteristics after linear detrending and bandpass filtering, while low-frequency baseline drift is successfully removed.

---

## 6. Validation & Exit Gate Summary

- Fixed Probe Model Parameter Count: `9315`
- Standalone M-B1 validator (`scripts/validate_mmwave_m_b1.py`): `PASS` (`validation_success: True`)
- Standalone M-B0 validator (`scripts/validate_mmwave_m_b0.py`): `PASS`
- Upstream M-A5 validator (`scripts/validate_mmwave_subject_split.py`): `PASS`
- Upstream M-A6 validator (`scripts/validate_mmwave_full_conversion.py`): `PASS`
- Unit tests (`tests/test_mmwave_m_b1.py`): `PASS`
- Deterministic Rerun: `PASS` (`validation_predictions_match: True`)
- Checksum Coverage: All 19 machine-readable manifests checksummed in `checksums.sha256`
- M-B1 Gate Status: `PASS_WITH_WARNINGS`
- M-B2 Entry Status: `READY_WITH_CONDITIONS`
