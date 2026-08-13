# M-B10R1-B — Authorized One-Time Limited-Reuse Recovery Evaluation

**THIS RESULT USES A NON-PRISTINE HOLDOUT REUSE EXCEPTION**

**ORIGINAL M-B10B ACCESS WAS CONSUMED BEFORE INFERENCE**

**M-B10R1-B IS THE SINGLE AUTHORIZED RECOVERY RELEASE**

**RECOVERY HAS BEEN EXECUTED ONCE. DO NOT RETRY. DO NOT REOPEN LOCKED_TEST.**

## Status

| Field | Value |
|---|---|
| Phase | M-B10R1-B |
| Status | `RECOVERY_EXECUTED` |
| Result designation | `REUSED_LOCKED_TEST_AFTER_PREINFERENCE_STRUCTURAL_ABORT` |
| `result_not_pristine` | true |
| Pristine locked-test claim | **NO** |
| Recovery payload releases | **1** |
| Original M-B10B payload releases | **1** |
| Historical total payload releases | **2** |
| Recovery accessor invocations | **1** |
| Second recovery | **NO** |
| Rerun | **NO** |
| M-B11 started | **NO** |

Allowed scientific wording: `OFFLINE_REAL_DATA_RECOVERY_EVALUATION_WITH_HOLDOUT_REUSE_LIMITATION`.

This is **not** MR60-validated, Raspberry Pi-validated, deployment-ready, production-ready, or clinical apnea validation. APNEA remains a SafeNest proxy. Offline dataset performance is distinct from team MR60 device-domain evidence (PR #56 overlay); the ~20 rpm investigation remains a future M-C/M-D input and is not solved by this recovery.

## Binding

| Field | Value |
|---|---|
| Reviewed M-B10R1-A head | `ca4b55b09555de34172a98f1a7ac05c24c1f2dac` |
| Execution freeze identity SHA-256 | `174c539074f04c4d57979439bbd3e58bf8cf687cdd27ce5dc28abed0a1b18a6c` |
| A directory modified | **NO** |
| Execution code modified | **NO** |
| Authorization overlay | `datasets/mmwave/manifests/M-B10R1B_recovery_execution/authorization_record.json` |
| Authorization status | `AUTHORIZED_FOR_ONE_RECOVERY_RELEASE` |
| Independent reviewer authorization | true |
| Recovery token used | `M_B10R1_LIMITED_REUSE_RECOVERY_AUTHORIZATION_V1` |
| Original final token used | **NO** |

A `recovery_execution_authorized` and `recovery_payload_release_authorized` remain historically **false**.

## Historical original access (unchanged)

- Original M-B10B final accessor invocations: **1**
- Original payload releases: **1**
- Original rows returned: **75**
- Original model inference: **0**
- Original predictions: **0**
- Original metrics: **0**
- Original LOCKED_TEST consumed: **true**
- Original pristine status: **false**

## Population

| Field | Value |
|---|---|
| Structural LOCKED_TEST windows | 88 |
| Supervised eligible windows | 75 |
| Excluded AMBIGUOUS / non-eligible | 13 |
| Eligible subjects | 16 |
| Eligibility provenance | `PREEXISTING_A6_METADATA_VERIFIED` |
| Positional first-N truncation | false |
| Actual returned eligible rows | 75 |
| Actual returned eligible subjects | 16 |

## Models (exactly 3)

1. Selected seed42 real-data strict-INT8  
   Candidate: `M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120`  
   Runtime: `M-B3_CONV1D_GAP_BASELINE_seed42_M-B6_STRICT_INT8`  
   SHA `6dff6aaa72c79d76715d40cf7e32bb1e6cd9b2c2e3ac78eaf2fda737561430c5`  
   Contract: `M-B10B_SELECTED_REAL_CANDIDATE_BPF_ZSCORE_V1`
2. Historical v0.1 compatibility baseline `mmwave_resp_int8`  
   SHA `43cdd6f321c2b645232162233e098c2b65549388cbc9e680a17a0eccdc8f0158`  
   Contract: `M-B10B_HISTORICAL_V0_1_COMPATIBILITY_PREPROCESSING_V1`  
   Interpretation: `HISTORICAL_MODEL_COMPATIBILITY_BENCHMARK`
3. Synthetic v0.2 compatibility baseline `mmwave_resp_int8_v0.2.0_candidate`  
   SHA `85c023d3eefca13ecbb72a841974e53a56d5f4173920645d46df49a9088452ff`  
   Contract: `M-B10B_SYNTHETIC_V0_2_EXTERNAL_COMPATIBILITY_PREPROCESSING_V1`  
   Interpretation: `SYNTHETIC_TRAINED_EXTERNAL_COMPATIBILITY_BENCHMARK`

Seed43 evaluated: **NO**. Seed44 evaluated: **NO**. Fourth model: **NO**. Candidate unchanged: **YES**.

## Inference execution

| Field | Value |
|---|---|
| Expected ledger rows | 225 |
| Actual ledger rows persisted | 225 |
| Expected TFLite invocations | 225 |
| Actual TFLite invocations | 225 |
| seed42 invokes / valid / invalid | 75 / 75 / 0 |
| v0.1 invokes / valid / invalid | 75 / 75 / 0 |
| v0.2 invokes / valid / invalid | 75 / 75 / 0 |

Durable ledger: `datasets/mmwave/manifests/M-B10R1B_recovery_execution/recovery_sample_predictions.jsonl`

## Selected candidate (seed42)

Coverage: 75 attempted, 75 valid, 0 invalid, 75 TFLite invokes.

| Metric | Value |
|---|---|
| Accuracy | 0.56 |
| Macro F1 | 0.494836 |
| Macro precision | 0.557692 |
| Macro recall | 0.518846 |

### NORMAL

support 25; precision 0.5; recall 0.2; F1 0.285714; FPR 0.1; TP 5; FP 5; TN 45; FN 20

### RAPID_OR_ABNORMAL

support 19; precision 0.615385; recall 0.421053; F1 0.5; FPR 0.089286; TP 8; FP 5; TN 51; FN 11

### APNEA proxy

support 31; precision 0.557692; recall 0.935484; F1 0.698795; FPR 0.522727; misses 2; TP 29; FP 23; TN 21; FN 2

Confusion matrix (rows = true NORMAL / RAPID_OR_ABNORMAL / APNEA):

```text
[[5, 5, 15],
 [3, 8,  8],
 [2, 0, 29]]
```

Prediction distribution: NORMAL 10, RAPID_OR_ABNORMAL 13, APNEA 52.

Class collapse: **false** (no zero-prediction or zero-recall class).

Input saturation (selected, pre-clamp): total elements 22500; out-of-range 0; ratio 0.0; samples affected 0; worst-sample ratio 0.0.

Subject metrics: subject count 16; median Macro F1 0.388888; worst Macro F1 0.095238; worst subject `dataset-10_5281_zenodo_18599983-p019`.

## v0.1 historical compatibility baseline

Coverage: 75 attempted, 75 valid, 0 invalid, 75 TFLite invokes.

This is a compatibility benchmark. It does **not** claim exact historical preprocessing reconstruction.

| Metric | Value |
|---|---|
| Accuracy | 0.333333 |
| Macro F1 | 0.166667 |
| Macro precision | 0.111111 |
| Macro recall | 0.333333 |

NORMAL: support 25; P 0.333333; R 1.0; F1 0.5; FPR 1.0  
RAPID_OR_ABNORMAL: support 19; P 0.0; R 0.0; F1 0.0; FPR 0.0  
APNEA proxy: support 31; P 0.0; R 0.0; F1 0.0; FPR 0.0; misses 31

Confusion:

```text
[[25, 0, 0],
 [19, 0, 0],
 [31, 0, 0]]
```

Prediction distribution: NORMAL 75, RAPID_OR_ABNORMAL 0, APNEA 0.

Class collapse: **true** (zero prediction/recall for RAPID_OR_ABNORMAL and APNEA).

Subject metrics: subject count 16; median Macro F1 0.166667; worst Macro F1 0.0; worst subject `dataset-10_5281_zenodo_18599983-p039`.

## v0.2 synthetic external compatibility baseline

Coverage: 75 attempted, 75 valid, 0 invalid, 75 TFLite invokes.

| Metric | Value |
|---|---|
| Accuracy | 0.506667 |
| Macro F1 | 0.391074 |
| Macro precision | 0.412281 |
| Macro recall | 0.467957 |

NORMAL: support 25; P 0.403509; R 0.92; F1 0.560976; FPR 0.68  
RAPID_OR_ABNORMAL: support 19; P 0.0; R 0.0; F1 0.0; FPR 0.0  
APNEA proxy: support 31; P 0.833333; R 0.483871; F1 0.612245; FPR 0.068182; misses 16

Confusion:

```text
[[23, 0, 2],
 [18, 0, 1],
 [16, 0, 15]]
```

Prediction distribution: NORMAL 57, RAPID_OR_ABNORMAL 0, APNEA 18.

Class collapse: **true** (zero prediction/recall for RAPID_OR_ABNORMAL).

Subject metrics: subject count 16; median Macro F1 0.388889; worst Macro F1 0.095238; worst subject `dataset-10_5281_zenodo_18599983-p044`.

## Descriptive comparisons only

These differences do **not** trigger reselection, retraining, recalibration, or a second recovery.

| Comparison | Value |
|---|---|
| seed42 Macro F1 − v0.1 Macro F1 | 0.328169 |
| seed42 Macro F1 − v0.2 Macro F1 | 0.103762 |
| seed42 APNEA recall − v0.1 | 0.935484 |
| seed42 APNEA recall − v0.2 | 0.451613 |
| seed42 RAPID recall − v0.1 | 0.421053 |
| seed42 RAPID recall − v0.2 | 0.421053 |

Selected candidate remains seed42. Low Macro F1, baseline collapse, and APNEA FPR are scientific evidence, not execution-integrity failures. No predefined numerical acceptance threshold exists (`FINAL_LOCKED_TEST_NUMERICAL_ACCEPTANCE_THRESHOLD_NOT_PREDEFINED`).

## Post-result prohibitions (observed)

- Training: none
- Conversion: none
- Recalibration: none
- Threshold tuning: none
- Seed reselection: none
- Architecture reselection: none
- Preprocessing change: none

## Durable evidence

Directory: `datasets/mmwave/manifests/M-B10R1B_recovery_execution/`

Frozen post-access validator `scripts/validate_mmwave_m_b10r1b.py` independently recomputed metrics from the persisted 225-row ledger: **PASS**. Checksums: **PASS**. No-reaccess validation: validator does not call recovery or final accessors.

## M-B11

M-B11 artifact lock may begin only after independent review of B authorization history, recovery counters, the 225-row ledger, validator, metrics, and Git isolation. **Do not begin M-B11 in this phase.**
