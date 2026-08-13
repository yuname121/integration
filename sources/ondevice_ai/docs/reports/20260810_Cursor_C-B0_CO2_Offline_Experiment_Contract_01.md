# SafeNest CO₂ Phase C-B0 — Offline Experiment Contract, Leakage-Safe Comparison Universe, and Baseline Evaluation Harness

- Document Version: `01`
- Author: `Cursor` (CO₂ Track Implementation Agent)
- Execution Date: `2026-08-10`
- Phase: `C-B0`
- Status: `PASS_WITH_WARNINGS`
- Worktree: `/private/tmp/safenest-co2-cb0`
- Branch: `feature/C-B0-co2-offline-experiment-contract`

---

## 1. Executive Summary

C-B0 locks the immutable offline experiment foundation for the CO₂ B-series using the released A-series baseline. It defines the common TRAIN/VALIDATION comparison universe, seals LOCKED_TEST, registers feature/device roles without selecting a winner, defines metrics/preprocessing rules, validates a TRAIN-only majority-class reference baseline on VALIDATION, and records machine-readable evidence for C-B1+.

No final feature selection, slope ablation, complex architecture comparison, production scaler/model promotion, or LOCKED_TEST predictive use was performed.

---

## 2. A-Series Prerequisite

| Field | Value |
|---|---|
| Release tag | `co2-a-series-raw-to-canonical` |
| Release commit | `bfd860cad2bb8dafe35ef7600cfa931d7d2d554d` |
| Tag verification | VERIFIED |
| Artifact lock profile | `CO2_A_SERIES_ARTIFACT_LOCK_PROFILE_001` |
| Artifact lock SHA-256 | `b63f5e2da988f8e685cf1a01ec8e79c2c37f5bc77359be647f1147ecfb04e3da` |
| Baseline drift | NONE |

Branch base may be later `origin/main`, but B0 data contracts are fingerprinted against the A-series release/lock, not against arbitrary later HEAD semantics.

---

## 3. Comparison Universe

| Universe | Count |
|---|---:|
| Canonical source | 20560 |
| Canonical warm-up (preserved, outside slope-dependent matrix) | 9 |
| B-series TRAIN common | 8140 |
| B-series VALIDATION common | 2662 |
| B-series sealed LOCKED_TEST | 9749 |
| Tunable comparison total (TRAIN+VAL) | 10802 |

Cross-split ID overlaps: **0**

---

## 4. Feature / Device Registry

Registered views (no winner):

- `CO2_ONLY_REFERENCE`
- `SCD40_NATIVE_REFERENCE`
- `HISTORICAL_COMPATIBILITY_REFERENCE` (`CO2_slope`, `Humidity`, `CO2`) — historical compatibility only, **not** final feature set
- `SCD40_SLOPE_REFERENCE`
- `UCI_CONTEXT_DIAGNOSTIC` (non-deployable diagnostic)

SCD40-native: CO2, Temperature, Humidity  
Derived SCD40-compatible: CO2_slope  
UCI-only/non-native: Light, HumidityRatio  

Target `Occupancy` and provenance fields are forbidden model inputs.

C-B1 boundary: slope method/history ablation deferred; baseline remains `CO2_SLOPE_FEATURE_PROFILE_001`.

---

## 5. Preprocessing / Metrics / Baseline

- Preprocessing: TRAIN-only standard scaler, status `B0_EXPERIMENT_REFERENCE_ONLY`
- Production scaler/model: **not modified**
- Metrics: accuracy, balanced accuracy, class-wise precision/recall/F1, macro F1, confusion matrix
- Primary summary metric: macro F1
- Reference baseline: `MAJORITY_CLASS_BASELINE` fitted on TRAIN, evaluated on VALIDATION only
- Optional logistic baseline: skipped (`scikit-learn` unavailable)

---

## 6. Leakage / Policy

| Check | Result |
|---|---|
| TRAIN/VAL/LOCKED overlaps | 0 |
| Target/provenance as features | rejected by harness |
| LOCKED_TEST fit/tuning | 0 |
| LOCKED_TEST predictive evaluation in B0 | blocked |
| Random row-wise resplit | prohibited |

---

## 7. Artifacts

Under `datasets/co2/manifests/c_b0_offline_experiment_contract/`:

- `experiment_contract.json`
- `a_series_consumption_registry.json`
- `sample_universe_manifest.json`
- `feature_view_registry.json`
- `metric_contract.json`
- `leakage_audit.json`
- `preprocessing_fit_evidence.json`
- `reference_baseline_result.json`
- `generation_metadata.json`
- `exceptions_and_limitations.json`
- `run_environment.json`
- `checksums.sha256`

Code/tests:

- `datasets/co2/offline_experiment.py`
- `scripts/audit_co2_offline_experiment.py`
- `scripts/validate_co2_offline_experiment.py`
- `tests/test_co2_offline_experiment.py`

---

## 8. Validation Evidence

| Check | Result |
|---|---|
| C-A0..C-A6 validators | all `PASS_WITH_WARNINGS` |
| C-B0 validator | `PASS_WITH_WARNINGS` |
| CO₂ tests | **92 passed** |
| Full repository tests | **647 passed, 5 skipped, 0 failed** |
| compile/import | PASS |
| `git diff --check` | PASS |
| Deterministic double generation | PASS |

---

## 9. C-B1 Authorization

After C-B0 merge, C-B1 may perform controlled CO₂ slope method/history ablation while consuming this locked experiment contract. C-B1 must not redefine target, split, metric contract, LOCKED_TEST policy, or sample-identity semantics.
