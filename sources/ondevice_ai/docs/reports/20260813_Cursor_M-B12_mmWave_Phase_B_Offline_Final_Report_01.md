# M-B12 mmWave Phase-B Offline Final Report

Generated from stored M-B11 lock evidence and immutable predecessor artifacts.
This report does not create a new model, does not re-evaluate LOCKED_TEST, and
does not begin M-C.

## Prominent closure statements

M-B12 DOES NOT CREATE A NEW MODEL.

THE LOCKED CANDIDATE REMAINS THE M-B11 SEED42 STRICT-INT8 OFFLINE CANDIDATE.

THE FINAL OFFLINE EVALUATION REMAINS A NON-PRISTINE HOLDOUT REUSE EXCEPTION.

- Artifact status: `REAL_DATA_OFFLINE_CANDIDATE`
- Result limitation: `REUSED_LOCKED_TEST_AFTER_PREINFERENCE_STRUCTURAL_ABORT`
- `result_not_pristine`: true
- Intermediate-release status: `PHASE_B_OFFLINE_INTERMEDIATE_RELEASE_READY_AFTER_MERGE`
- Unqualified Phase-B product/deployment release: false
- Git tag created: false
- GitHub Release created: false
- M-C started: false

## What this closure is not

This closure is not deployment ready, MR60 validated, Raspberry Pi validated,
production ready, or clinical apnea validated. It is not a pristine LOCKED_TEST
result. It is not physical sensor integration. It does not authorize reopening
LOCKED_TEST or recovery. It does not create a git tag or GitHub Release.

APNEA labels remain SafeNest proxies derived from voluntary breath-hold
windows. They are not clinical apnea.

## A0–A6 frozen source, representation, and split

- Raw archive: `datasets/raw_archives/external_datasets/db_records.zip`
- Raw SHA-256: `f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0`
- DOI: 10.5281/zenodo.18599983 version v1.1
- Population: 110 participants / 440 recordings
- Canonical dataset: `datasets/mmwave/processed/mmwave_canonical_real_v1.npy`
- Canonical SHA-256: `c2e2cd1615c7af0f0e21700f291ee12ac0347a9f7fc6ccc9f337433c16868f0e`
- Shape/dtype: [530, 300] / float64
- A4 labels: SafeNest APNEA proxies; not clinical apnea
- A5 split: `datasets/mmwave/splits/mmwave_real_subject_split_v1.json`
- A5 SHA-256: `a1996ea00a1d5066eae6d25f022d04137085434d4768e27cbebcdad4e0385baa`
- A5 split seed: 20260808
- Subjects TRAIN/VALIDATION/LOCKED_TEST: 77/17/16
- A6 manifest SHA-256: `1d1728eafdc3d4786e34fc663329a12a311322a698bdbf2fd01e6bce95c50acf`
- Windows structural TRAIN/VALIDATION/LOCKED_TEST: 358/84/88
- Eligible TRAIN/VALIDATION/LOCKED_TEST: 327/79/75
- Class totals NORMAL/RAPID_OR_ABNORMAL/APNEA/AMBIGUOUS: 149/119/213/49
- LOCKED_TEST excluded ambiguous/non-eligible: 13

## B-series selected path

- M-B0: evaluation / leakage / LOCKED_TEST protocol. Cross-split subject/recording/window overlap = 0.
- M-B1 selected preprocessing: `M-B1_D0_B1_Z1` / `BPF_ZSCORE`
- M-B2 selected training strategy: `M-B2_CE_UNWEIGHTED` (unweighted CE)
- M-B3 selected architecture: `M-B3_CONV1D_GAP_BASELINE`
- M-B4 cross-seed VALIDATION Macro F1: seed42=0.663708, seed43=0.45101, seed44=0.329107. Initialization sensitivity is locked, not hidden.
- M-B5 selected calibration: `M-B5_CAL_CLASS_BALANCED_120`
- M-B6 frozen strict INT8 SHA-256: `6dff6aaa72c79d76715d40cf7e32bb1e6cd9b2c2e3ac78eaf2fda737561430c5`
- M-B7 perturbation robustness: seed42 retained; seed44 moderate-profile collapse recorded
- M-B8 Mac/M2 latency and footprint only. Not Raspberry Pi latency. seed42 median=0.006583 ms, p99=0.00933301 ms
- M-B9 mock runtime/E2E fail-closed path. Not physical sensor integration.
- M-B10A preregistered `M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120` before LOCKED_TEST
- M-B10B original accessor=1, payload release=1, inference=0, consumed=true, root cause=`PRETEST_CONTRACT_COUNT_SEMANTICS_CONFLATION`
- M-B10R0 policy=`LIMITED_LOCKED_TEST_REUSE_EXCEPTION_RECOMMENDED`
- M-B10R1-A froze the recovery harness; new access=0
- M-B10R1-B recovery accessor=1, recovery payload release=1, TFLite invokes=225, second recovery=NO, rerun=NO
- Historical total payload releases = 2
- M-B11 locked the candidate as `REAL_DATA_OFFLINE_CANDIDATE`

## Locked candidate

- Candidate ID: `M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120`
- Runtime model ID: `M-B3_CONV1D_GAP_BASELINE_seed42_M-B6_STRICT_INT8`
- Artifact: `models/mmwave/experiments/M-B6_stage_equivalence/M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120_int8.tflite`
- SHA-256: `6dff6aaa72c79d76715d40cf7e32bb1e6cd9b2c2e3ac78eaf2fda737561430c5`
- Bytes: 22080
- Seed: 42
- Preprocessing contract: `M-B10B_SELECTED_REAL_CANDIDATE_BPF_ZSCORE_V1`
- BPF: Butterworth 0.1-0.5 Hz, order 4, zero-phase filtfilt, fs=10.0 Hz
- Z-score: TRAIN-only mean=0.0031162832173884064, std=2.955399434649939
- Training: `M-B2_CE_UNWEIGHTED`; sparse_categorical_crossentropy unweighted; Adam lr=0.001; batch 32; max epochs 25; patience 7
- Input: shape [1, 300, 1] dtype int8 scale=0.041720833629369736 zp=-3
- Output: shape [1, 3] dtype int8 scale=0.00390625 zp=-128
- Strict INT8: true; Flex/Select TF Ops: false
- Class map: 0→NORMAL, 1→RAPID_OR_ABNORMAL, 2→APNEA (APNEA remains a proxy)

## Final offline evaluation (non-pristine)

Reused stored M-B10R1-B / M-B11 evidence. No M-B12 inference.

- Result designation: `REUSED_LOCKED_TEST_AFTER_PREINFERENCE_STRUCTURAL_ABORT`
- Unique eligible IDs / models / pairs: 75 / 3 / 225
- Duplicates / missing / unexpected: 0 / 0 / 0
- Label / subject / recording mismatches: 0 / 0 / 0
- Eligible evaluated / valid / invalid: 75 / 75 / 0
- Accuracy: 0.56
- Macro F1: 0.494836
- Macro precision: 0.557692
- Macro recall: 0.518846
- NORMAL: support=25 precision=0.5 recall=0.2 F1=0.285714 FPR=0.1
- RAPID_OR_ABNORMAL: support=19 precision=0.615385 recall=0.421053 F1=0.5 FPR=0.089286
- APNEA proxy: support=31 precision=0.557692 recall=0.935484 F1=0.698795 misses=2 FPR=0.522727
- Confusion: [[5, 5, 15], [3, 8, 8], [2, 0, 29]]
- Prediction distribution: {'APNEA': 52, 'NORMAL': 10, 'RAPID_OR_ABNORMAL': 13}
- Class collapse: False
- Subjects: 16
- Median subject Macro F1: 0.388888
- Worst subject Macro F1: 0.095238
- Worst subject: `dataset-10_5281_zenodo_18599983-p019`
- Saturation ratio: 0.0 (pre-clamp out-of-range 0 / 22500)

These limitations are locked scientific facts for future M-C/M-D. They are not
M-B12 blockers and they are not defects requiring immediate B-series retuning.

Locked limitation facts: NORMAL recall=0.2; RAPID recall=0.421053; APNEA proxy recall=0.935484; APNEA FPR=0.522727; worst-subject Macro F1=0.095238; M-B4 seed42 VAL=0.663708 vs seed44 VAL=0.329107.

## Baseline comparison

This is not a new model-selection event.

- seed42 Macro F1: 0.494836 (no required-class collapse)
- v0.1 `HISTORICAL_MODEL_COMPATIBILITY_BENCHMARK` Macro F1: 0.166667 (class collapse; all 75 predicted NORMAL)
- v0.2 `SYNTHETIC_TRAINED_EXTERNAL_COMPATIBILITY_BENCHMARK` Macro F1: 0.391074 (RAPID_OR_ABNORMAL zero-prediction collapse)

## Intermediate-release readiness

- Status label: `PHASE_B_OFFLINE_INTERMEDIATE_RELEASE_READY_AFTER_MERGE`
- Ready after merge onto canonical `main`: True
- Unqualified `Phase_B_release_ready`: False
- Proposed future tag (not created): `mmwave-phase-b-offline-candidate`
- Git tag created: False
- GitHub Release created: False
- Explicit exclusions remain false: clinical_apnea_validated, deployment_ready, first_locked_test_evaluation, locked_test_reopen_allowed, mr60_device_validated, multisensor_integration_complete, pristine_locked_test, production_ready, raspberry_pi_validated, recovery_reopen_allowed

Do not create a GitHub Release or tag in M-B12. Any future tag must target the
exact M-B12 merge commit on canonical `main` after independent review.

## Device-domain handoff for future M-C

M-B12 does not begin M-C. Future M-C must independently investigate:

- physical MR60BHA2 signal-domain compatibility with this offline candidate
- device preprocessing correspondence to M-B10B_SELECTED_REAL_CANDIDATE_BPF_ZSCORE_V1
- observed team approximately-20-rpm behavior
- domain shift between the offline Zenodo dataset and the physical sensor
- runtime input identity on device
- Raspberry Pi / device execution behavior

## Claim boundary

- PRISTINE_LOCKED_TEST: False
- MR60 validated: False
- Raspberry Pi validated: False
- Deployment ready: False
- Production ready: False
- Clinical apnea validated: False
- LOCKED_TEST reopen allowed: False
- Recovery reopen allowed: False
- M-C started: False
- Phase-B offline final report complete: True
- Phase-B offline intermediate release ready after merge: True

## Machine-Verified Final Facts

| Fact | Value |
| --- | --- |
| candidate_status | `REAL_DATA_OFFLINE_CANDIDATE` |
| selected_model_sha | `6dff6aaa72c79d76715d40cf7e32bb1e6cd9b2c2e3ac78eaf2fda737561430c5` |
| result_designation | `REUSED_LOCKED_TEST_AFTER_PREINFERENCE_STRUCTURAL_ABORT` |
| result_not_pristine | true |
| final_accuracy | 0.56 |
| final_macro_f1 | 0.494836 |
| normal_recall | 0.2 |
| rapid_recall | 0.421053 |
| apnea_recall | 0.935484 |
| apnea_fpr | 0.522727 |
| v0_1_macro_f1 | 0.166667 |
| v0_2_macro_f1 | 0.391074 |
| original_release | 1 |
| recovery_release | 1 |
| historical_total_release | 2 |
| mr60_validated | false |
| raspberry_pi_validated | false |
| deployment_ready | false |
| clinical_apnea_validated | false |
| intermediate_release_ready | true |
| tag_created | false |
| github_release_created | false |
| m_c_started | false |

New LOCKED_TEST access = 0
New recovery access = 0
New inference = 0

<!-- MACHINE_VERIFIED_FINAL_FACTS -->
candidate_status=REAL_DATA_OFFLINE_CANDIDATE
selected_model_sha=6dff6aaa72c79d76715d40cf7e32bb1e6cd9b2c2e3ac78eaf2fda737561430c5
result_designation=REUSED_LOCKED_TEST_AFTER_PREINFERENCE_STRUCTURAL_ABORT
result_not_pristine=true
final_accuracy=0.56
final_macro_f1=0.494836
normal_recall=0.2
rapid_recall=0.421053
apnea_recall=0.935484
apnea_fpr=0.522727
v0_1_macro_f1=0.166667
v0_2_macro_f1=0.391074
original_release=1
recovery_release=1
historical_total_release=2
mr60_validated=false
raspberry_pi_validated=false
deployment_ready=false
clinical_apnea_validated=false
intermediate_release_ready=true
tag_created=false
github_release_created=false
m_c_started=false
<!-- END_MACHINE_VERIFIED_FINAL_FACTS -->
