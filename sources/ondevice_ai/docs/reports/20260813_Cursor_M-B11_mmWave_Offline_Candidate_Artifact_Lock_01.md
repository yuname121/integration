# M-B11 mmWave Offline Real-Data Candidate Artifact Lock

Generated from stored machine-readable evidence. This report does not create a new model.

## Prominent lock statements

M-B11 DOES NOT CREATE A NEW MODEL.

THE LOCKED CANDIDATE IS THE PREVIOUSLY SELECTED SEED42 STRICT-INT8 MODEL.

THE FINAL OFFLINE EVALUATION USED A NON-PRISTINE HOLDOUT REUSE EXCEPTION.

- Artifact status: `REAL_DATA_OFFLINE_CANDIDATE`
- Result limitation: `REUSED_LOCKED_TEST_AFTER_PREINFERENCE_STRUCTURAL_ABORT`
- `result_not_pristine`: true
- Candidate ID: `M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120`
- Runtime model ID: `M-B3_CONV1D_GAP_BASELINE_seed42_M-B6_STRICT_INT8`
- Artifact: `models/mmwave/experiments/M-B6_stage_equivalence/M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120_int8.tflite`
- SHA-256: `6dff6aaa72c79d76715d40cf7e32bb1e6cd9b2c2e3ac78eaf2fda737561430c5`
- Bytes: 22080

## What this lock is not

This lock is not deployment ready, MR60 validated, Raspberry Pi validated, production ready, or clinical apnea validated. Phase B release remains incomplete. M-B12 is still required.

Offline real-data model evidence is not physical MR60 sensor evidence, not Raspberry Pi runtime evidence, and not future multisensor integration evidence. Team MR60 measurement evidence and the approximately-20-rpm issue belong to future M-C/M-D device-domain work. M-B11 does not resolve that issue. Old team `ondevice_ai` behavior is not validation evidence for this locked candidate.

## Source and canonical lineage

- Raw archive: `datasets/raw_archives/external_datasets/db_records.zip`
- Raw SHA-256: `f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0`
- DOI: 10.5281/zenodo.18599983 version v1.1
- Population: 110 participants / 440 recordings
- Canonical dataset: `datasets/mmwave/processed/mmwave_canonical_real_v1.npy`
- Canonical SHA-256: `c2e2cd1615c7af0f0e21700f291ee12ac0347a9f7fc6ccc9f337433c16868f0e`
- Shape/dtype: [530, 300] / float64
- A5 split: `datasets/mmwave/splits/mmwave_real_subject_split_v1.json`
- A5 SHA-256: `a1996ea00a1d5066eae6d25f022d04137085434d4768e27cbebcdad4e0385baa`
- A5 split seed: 20260808
- Subjects TRAIN/VALIDATION/LOCKED_TEST: 77/17/16
- A6 manifest SHA-256: `1d1728eafdc3d4786e34fc663329a12a311322a698bdbf2fd01e6bce95c50acf`
- Windows structural TRAIN/VALIDATION/LOCKED_TEST: 358/84/88
- Eligible TRAIN/VALIDATION/LOCKED_TEST: 327/79/75
- Class totals NORMAL/RAPID_OR_ABNORMAL/APNEA/AMBIGUOUS: 149/119/213/49

## Locked candidate contracts

- Preprocessing profile: `M-B1_D0_B1_Z1` / `BPF_ZSCORE`
- Execution contract: `M-B10B_SELECTED_REAL_CANDIDATE_BPF_ZSCORE_V1`
- BPF: Butterworth 0.1-0.5 Hz, order 4, zero-phase filtfilt, fs=10.0 Hz
- Z-score: TRAIN-only mean=0.0031162832173884064, std=2.955399434649939
- Training strategy: `M-B2_CE_UNWEIGHTED`
- Loss: sparse_categorical_crossentropy unweighted; optimizer Adam lr=0.001; batch 32; max epochs 25; patience 7; restore-best True
- Seed: 42
- Calibration: `M-B5_CAL_CLASS_BALANCED_120`
- Input: shape [1, 300, 1] dtype int8 scale=0.041720833629369736 zp=-3
- Output: shape [1, 3] dtype int8 scale=0.00390625 zp=-128
- Strict INT8: true; Flex/Select TF Ops: false
- Class map: 0→NORMAL, 1→RAPID_OR_ABNORMAL, 2→APNEA (APNEA remains a proxy)

## B-series lineage

Selected path: M-B1 `M-B1_D0_B1_Z1` → M-B2 `M-B2_CE_UNWEIGHTED` → M-B3 `M-B3_CONV1D_GAP_BASELINE` → M-B4 seeds 42/43/44 → M-B5 `M-B5_CAL_CLASS_BALANCED_120` → M-B6 frozen strict INT8 → M-B7 perturbation robustness → M-B8 Mac/M2 latency only → M-B9 mock runtime/E2E.

M-B4 initialization sensitivity is locked, not hidden. VALIDATION Macro F1: seed42=0.663708, seed43=0.45101, seed44=0.329107. Seed42 was materially better than seed44. M-B7 recorded moderate-profile collapse evidence for seed44 while seed42 was retained.

M-B8 is Mac/M2 latency and footprint evidence only. It is not Raspberry Pi latency. M-B9 is mock/runtime path equivalence and fail-closed behavior. It is not physical sensor integration.

## Abnormal final-test history (must not be erased)

1. M-B10A preregistered the candidate before LOCKED_TEST.
2. M-B10B original final accessor invocation = 1; payload returned; model inference = 0; original LOCKED_TEST consumed = true; root cause = PRETEST_CONTRACT_COUNT_SEMANTICS_CONFLATION.
3. M-B10R0 policy = LIMITED_LOCKED_TEST_REUSE_EXCEPTION_RECOMMENDED.
4. M-B10R1-A froze the recovery harness before reuse; new access = 0.
5. M-B10R1-B recovery accessor = 1; recovery payload release = 1; TFLite invokes = 225; second recovery = NO; rerun = NO.
6. Historical total payload releases = 2.
7. Result designation remains `REUSED_LOCKED_TEST_AFTER_PREINFERENCE_STRUCTURAL_ABORT`. This is not FINAL_LOCKED_TEST and not a pristine holdout.

## Final selected candidate summary

Recomputed from the persisted 75 seed42 prediction rows. No model inference was performed in M-B11.

- Eligible evaluated / valid / invalid: 75 / 75 / 0
- Accuracy: 0.56
- Macro F1: 0.494836
- Macro precision: 0.557692
- Macro recall: 0.518846
- NORMAL: support=25 precision=0.5 recall=0.2 F1=0.285714 FPR=0.1
- RAPID_OR_ABNORMAL: support=19 precision=0.615385 recall=0.421053 F1=0.5 FPR=0.089286
- APNEA proxy: support=31 precision=0.557692 recall=0.935484 F1=0.698795 misses=2 FPR=0.522727
- Confusion: [[5, 5, 15], [3, 8, 8], [2, 0, 29]]
- Prediction distribution: {'NORMAL': 10, 'RAPID_OR_ABNORMAL': 13, 'APNEA': 52}
- Class collapse: False
- Subjects: 16
- Median subject Macro F1: 0.388888
- Worst subject Macro F1: 0.095238
- Worst subject: `dataset-10_5281_zenodo_18599983-p019`
- Saturation ratio: 0.0 (pre-clamp out-of-range 0 / 22500)

These limitations are locked scientific facts for future M-C/M-D. They are not M-B11 blockers and they are not defects requiring immediate B-series retuning.

## Baseline comparison

This is not a new model-selection event. v0.1 and v0.2 remain compatibility benchmarks only.

- seed42 Macro F1: 0.494836 (no required-class collapse)
- v0.1 `HISTORICAL_MODEL_COMPATIBILITY_BENCHMARK` Macro F1: 0.166667 (class collapse; all 75 predicted NORMAL)
- v0.2 `SYNTHETIC_TRAINED_EXTERNAL_COMPATIBILITY_BENCHMARK` Macro F1: 0.391074 (RAPID_OR_ABNORMAL zero-prediction collapse)

## Device-domain handoff for future M-C

M-B11 does not begin M-C. Future M-C must independently investigate:

- physical MR60BHA2 signal-domain compatibility with this offline candidate
- device preprocessing correspondence to `M-B10B_SELECTED_REAL_CANDIDATE_BPF_ZSCORE_V1`
- observed team approximately-20-rpm behavior
- domain shift between the offline Zenodo dataset and the physical sensor
- runtime input identity on device
- Raspberry Pi / device execution behavior

## Release readiness

- M-B11 artifact lock complete: True
- M-B12 offline final report required: True
- Phase B release ready: False
- LOCKED_TEST reopen allowed: False
- Recovery reopen allowed: False

Do not create a GitHub Release or tag in M-B11. Do not begin M-B12 until this lock is independently reviewed and merged.

## Validator-truth closure

- Forbidden-claim recursive enforcement: PASS
- Non-claim-boundary corruption tests: PASS
- Locked cross-model recording mismatches: 0
- Recording corruption tests: PASS
- Generator high-level ledger analyzer reused by validator: NO
- Validator-owned source ledger:
  - unique IDs = 75
  - models = 3
  - pairs = 225
  - duplicates = 0
  - missing = 0
  - unexpected = 0
  - label mismatches = 0
  - subject mismatches = 0
  - recording mismatches = 0
- New LOCKED_TEST access = 0
- New recovery access = 0
- New inference = 0

YES — M-B11 artifact lock validator is independently fail-closed; await independent review before M-B12.
