# SafeNest mmWave M-B0 — Evaluation Protocol, Duplicate Audit, and LOCKED_TEST Access Control Report

- **Author**: Antigravity Implementation Engineer
- **Date**: 2026-08-10
- **Target Repository**: `https://github.com/sheepmeat/test.git`
- **Branch**: `feature/M-B0-evaluation-protocol`
- **Phase M-B0 Gate Status**: `PASS_WITH_WARNINGS`
- **M-B1 Entry Status**: `READY_WITH_CONDITIONS`

---

## 1. Executive Summary

Phase M-B0 establishes a reproducible, independently validated evaluation-control layer for the SafeNest mmWave real-data reconstruction pipeline **before any model-selection experiment begins**.

Key achievements of Phase M-B0:
1. **Input Identity Lock**: Measured and locked SHA-256 digests for all 10 authoritative M-A inputs, verifying byte-level identity against upstream M-A5/M-A6 manifests and raw archive `db_records.zip`.
2. **Independent Split Isolation Re-verification**: Confirmed 100% subject isolation (110 subjects: 77 TRAIN / 17 VALIDATION / 16 LOCKED_TEST) with `0` subject overlap, `0` recording overlap, `0` window-ID overlap, and `0` exact signal hash overlap across splits.
3. **Exact Duplicate Audit**: Recalculated signal hashes for all 530 canonical $300$-sample float64 phase windows (`mmwave_canonical_real_v1.npy`), confirming `0` exact duplicates across subjects or splits.
4. **Near-Duplicate Diagnostic Policy & Empirical Calibration Audit**:
   - Defined mathematical near-duplicate metric based on standardized waveform Pearson correlation ($r$) and Normalized RMSE ($	ext{NRMSE}$).
   - Derived frozen near-duplicate threshold ($r \ge 0.995, 	ext{NRMSE} \le 0.05$) from all 358 TRAIN-only signal correlations and controlled micro-perturbations across representative windows without tuning against LOCKED_TEST.
   - Evaluated all 140,185 window pairs across the 530-window canonical dataset:
     - `CROSS_SPLIT` near-duplicates: `0`
     - `SAME_RECORDING` near-duplicates: `0` (flagged as expected physiological time-series continuity across adjacent 30s segments).
5. **LOCKED_TEST Code-Level Access Control Guard**: Created `scripts/mmwave_phase_b_access.py` (`PhaseBAccessGuard`), which provides TRAIN and VALIDATION datasets for model selection while refusing LOCKED_TEST access with a `LOCKED_TEST_AccessError` exception. Structural audit datasets strip all class labels and annotation derivation fields.
6. **Immutable Evaluation Contract**: Defined `evaluation_contract.json`, enforcing TRAIN-only fitting, VALIDATION-only selection, `AMBIGUOUS` pure-class exclusion, SafeNest APNEA-proxy terminology, Macro F1 / class-collapse rejection rules, and multi-seed finalist aggregation schemas.

---

## 2. Authoritative Input Identity

| Artifact Path | Evidence Role | Measured SHA-256 Digest | Status |
|---|---|---|---|
| `datasets/mmwave/manifests/a5_subject_split/split_profile.json` | A5 split profile configuration | `d022295eed222712927c4a8c7edea4613a5ba650dcbb84710af72f95a72b0c93` | `PASS_WITH_WARNINGS` |
| `datasets/mmwave/manifests/a5_subject_split/subject_split_manifest.jsonl` | A5 subject assignment manifest | `777cdaa1a8cda54ab0db63dcc916d3ba208c10f30cc2f48d3bc91e94bcb2dfc7` | `PASS_WITH_WARNINGS` |
| `datasets/mmwave/splits/mmwave_real_subject_split_v1.json` | Real subject split lookup mapping | `a1996ea00a1d5066eae6d25f022d04137085434d4768e27cbebcdad4e0385baa` | `PASS_WITH_WARNINGS` |
| `datasets/mmwave/processed/mmwave_canonical_real_v1.npy` | Canonical float64 phase matrix ($530 	imes 300$) | `c2e2cd1615c7af0f0e21700f291ee12ac0347a9f7fc6ccc9f337433c16868f0e` | `PASS_WITH_WARNINGS` |
| `datasets/mmwave/manifests/a6_full_conversion/a6_summary.json` | Full conversion audit summary | `2657c703d691e1e4a2aea6033b351e64ff124ac09438ab02b843210596189d34` | `PASS_WITH_WARNINGS` |
| `datasets/mmwave/manifests/a6_full_conversion/full_window_manifest.jsonl` | 530 canonical 30s window manifest | `1d1728eafdc3d4786e34fc663329a12a311322a698bdbf2fd01e6bce95c50acf` | `PASS_WITH_WARNINGS` |
| `datasets/mmwave/manifests/a6_full_conversion/full_provenance_manifest.jsonl` | 530 window provenance records | `7b94b73fea7ed51be2813e1014a1760fa22325c9399490b855c9ea59093a6dc2` | `PASS_WITH_WARNINGS` |
| `datasets/mmwave/manifests/a6_full_conversion/full_duplicate_audit.json` | M-A6 exact duplicate record | `14e75d39df2ae20724f31d0ba6eeae3404c428ab3ff50f4ce7710bb2888d7c1b` | `PASS_WITH_WARNINGS` |
| `datasets/mmwave/manifests/a6_full_conversion/processing_profile.json` | M-A6 full conversion profile | `c533fc590093f4b6ba765347181becd959f4d576d26a02ab2dc14e983811a2a2` | `PASS_WITH_WARNINGS` |
| `datasets/raw_archives/external_datasets/db_records.zip` | Immutable raw Zenodo archive | `f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0` | `VERIFIED` |

---

## 3. Split Isolation Audit Results

- **TRAIN Subjects**: `77` (358 windows)
- **VALIDATION Subjects**: `17` (84 windows)
- **LOCKED_TEST Subjects**: `16` (88 windows)
- **Cross-Split Subject Overlap**: `0`
- **Cross-Split Recording Overlap**: `0`
- **Cross-Split Window ID Overlap**: `0`
- **Cross-Split Exact Signal Hash Overlap**: `0`

---

## 4. Duplicate & Near-Duplicate Audit Results

### 4.1 Exact Duplicate Audit
- Total 30s windows audited: `530`
- Unique signal hashes: `530`
- Exact duplicates found: `0`

### 4.2 Near-Duplicate Policy & Empirical Calibration
- **Diagnostic Method**: Standardized Waveform Pearson Correlation ($r$) and NRMSE.
- **TRAIN-only Empirical Calibration**: Distinct physiological breathing windows in TRAIN reach max $r pprox 0.9761$. Controlled micro-perturbations reach $r > 0.99999$.
- **Frozen Threshold Applied**: $r \ge 0.995$ and $	ext{NRMSE} \le 0.05$.
- **LOCKED_TEST Tuning Prohibition**: Confirmed `False` (threshold derived strictly without accessing LOCKED_TEST).

### 4.3 Near-Duplicate Audit Results (140,185 pairs)
- `SAME_RECORDING` near-duplicates: `0` (Adjacent sequential 30s windows from same 5-minute recording)
- `SAME_SUBJECT_DIFFERENT_RECORDING`: `0`
- `CROSS_SUBJECT_SAME_SPLIT`: `0`
- **`CROSS_SPLIT` near-duplicates**: **`0`**

---

## 5. LOCKED_TEST Code-Level Access Control

Data access guard implementation: `scripts/mmwave_phase_b_access.py` (`PhaseBAccessGuard`).

- `get_train_data(include_ambiguous=False)`: Returns 327 training-eligible windows.
- `get_validation_data(include_ambiguous=False)`: Returns 79 validation-eligible windows.
- `get_model_selection_dataset("LOCKED_TEST")`: **Fails closed** with `LOCKED_TEST_AccessError`.
- `get_structural_audit_dataset("LOCKED_TEST")`: Allows read-only access for leakage/duplicate audits, with all class labels and annotation metadata stripped out.
- `get_locked_test_final_evaluation_dataset(token)`: Requires explicit authorization token for final evaluation.

---

## 6. Evaluation Metric & Multi-Seed Policy

- **Primary Metric**: Macro F1 (Macro-averaged across pure classes).
- **Required Per-Class Metrics**: Precision, Recall, F1-Score, Confusion Matrix, Prediction Distribution.
- **SafeNest APNEA-Proxy Metrics**: APNEA Recall ($\ge 6.0$s voluntary breath-hold proxy), APNEA Miss Rate.
- **Class Collapse Policy**: Any candidate model predicting zero recall or collapsing APNEA / RAPID predictions shall be **REJECTED** regardless of high accuracy.
- **Multi-Seed Aggregation**: $\ge 3$ initialization seeds required for finalists (reporting mean, std, worst-seed). Seed selection on LOCKED_TEST is prohibited.
- **AMBIGUOUS Policy**: Transition windows retained for provenance but excluded from pure-class training/validation.

---

## 7. Exceptions & Warnings

- **Blockers**: `0`
- **Errors**: `0`
- **Warnings**: `0` `SAME_RECORDING_NEAR_DUPLICATE_PHASE_CONTINUITY` pairs logged as expected physiological time-series continuity.

---

## 8. Validation & Exit Gate

- Standalone M-B0 validator (`scripts/validate_mmwave_m_b0.py`): `PASS` (`validation_success: True`)
- M-A5 subject split validator (`scripts/validate_mmwave_subject_split.py`): `PASS`
- M-A6 full conversion validator (`scripts/validate_mmwave_full_conversion.py`): `PASS`
- Unit tests (`tests/test_mmwave_m_b0.py`): `PASS` (11/11 passed)
- Raw archive immutability: `f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0` (`VERIFIED`)
- M-B0 Gate Status: `PASS_WITH_WARNINGS`
- M-B1 Entry Status: `READY_WITH_CONDITIONS`
