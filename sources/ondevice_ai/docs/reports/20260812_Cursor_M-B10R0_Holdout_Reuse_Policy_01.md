# M-B10R0 — Holdout Reuse Exception vs New-Holdout Policy Review

Track / corrective phase: mmWave / M-B10R0

Canonical base: `origin/main` (post PR #47 merge)

Branch: `feature/M-B10R0-holdout-policy`

## Executive summary

M-B10B consumed LOCKED_TEST once via the final accessor but aborted before model
inference due to `PRETEST_CONTRACT_COUNT_SEMANTICS_CONFLATION`. M-B10R0
determines whether a limited holdout-reuse exception is scientifically
defensible before any additional access.

**Policy recommendation:** `LIMITED_LOCKED_TEST_REUSE_EXCEPTION_RECOMMENDED`

LOCKED_TEST REOPENED: NO

RECOVERY EVALUATION RUN: NO

MODEL INFERENCE: 0

M-B11 STARTED: NO

This PR preregisters policy only. Recovery execution is **not** authorized.

---

## Original incident

- Runtime detection: `M-B10B_LOCKED_SPLIT_IDENTITY_MISMATCH`
- Forensic root cause: `PRETEST_CONTRACT_COUNT_SEMANTICS_CONFLATION`
- Final accessor invocations: `1`
- Rows returned: `75` (pure-class eligible; accessor excludes AMBIGUOUS)
- Model inference invocations: `0`
- LOCKED_TEST consumed: `true` (pristine status lost)
- Rerun performed: `false`

`88` was the entire structural LOCKED_TEST split. `75` was the pre-existing
pure-class evaluation-eligible population (`locked_test_evaluation_eligible`).
The returned count matched pre-access A6 evidence; it is
`PREEXISTING_STRUCTURAL_INFORMATION_CONFIRMED_BY_ABORT`, not new performance
information.

---

## Why pristine status is lost

The withheld evaluation payload was returned to the M-B10B process once
(E0). Under the existing strict consumption policy, LOCKED_TEST is not
pristine even though no model predictions or metrics were produced.

---

## Exposure assessment

| Category | Result |
|----------|--------|
| Payload release occurred | YES |
| Prediction exposure | NO |
| Performance exposure | NO |
| Persisted sample registry | NO |
| Raw tensors persisted | NO |
| Sample/subject IDs persisted from returned payload | NO |
| Labels persisted from returned payload | NO |
| Human/agent performance-driven decisions | NO |
| Pre-existing eligible count confirmed (75) | YES |

Do not claim `NO_INFORMATION_EXPOSURE`. Payload release occurred; distinguish
prediction/performance/registry exposure from structural count confirmation.

---

## Count semantics (immutable)

| Population | Count |
|------------|-------|
| LOCKED_TEST structural subjects | 16 |
| LOCKED_TEST structural windows | 88 |
| Supervised evaluation eligible windows | 75 |
| AMBIGUOUS / excluded | 13 |

All 16 LOCKED_TEST subjects have at least one eligible window per pre-access
A6 window metadata. Subject count for future recovery is known from A6
metadata; sample-level identity was not persisted during M-B10B.

---

## Candidate and baseline freeze

Selected candidate: `M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120`

- Seed: `42`
- Model ID: `M-B3_CONV1D_GAP_BASELINE_seed42_M-B6_STRICT_INT8`
- SHA: `6dff6aaa72c79d76715d40cf7e32bb1e6cd9b2c2e3ac78eaf2fda737561430c5`
- Preprocessing: `BPF_ZSCORE` (`M-B1_D0_B1_Z1`)
- Calibration: `M-B5_CAL_CLASS_BALANCED_120`
- Class map: `0 NORMAL`, `1 RAPID_OR_ABNORMAL`, `2 APNEA`

Candidate changed after original access: NO

v0.1.0 and v0.2.0 compatibility baselines remain frozen per M-B10A contract.

---

## Existing unused holdout inventory

| Split | Subjects |
|-------|----------|
| Total approved corpus subjects | 110 |
| TRAIN | 77 |
| VALIDATION | 17 |
| LOCKED_TEST | 16 |
| Unassigned untouched | 0 |

**Independent existing replacement holdout:** NO

All 110 subjects are assigned. No genuinely unused subject holdout exists
within the approved Zenodo corpus. TRAIN and VALIDATION subjects cannot
become a new holdout because they already influenced model development. A5
reshuffle is prohibited.

Future external data (MR60, new Zenodo acquisition) is classified
`NEW_DATA_REQUIRED`, not `EXISTING_UNUSED_HOLDOUT_AVAILABLE`.

---

## Reuse exception gates R1–R10

| Gate | Result |
|------|--------|
| R1 incident truth closed | PASS |
| R2 exactly one previous access | PASS |
| R3 zero model evaluation | PASS |
| R4 no persisted sample-level payload | PASS |
| R5 candidate immutable | PASS |
| R6 baselines immutable | PASS |
| R7 count-semantics correction only | PASS |
| R8 no post-access tuning | PASS |
| R9 future contract unchanged models/metrics | PASS |
| R10 contamination disclosure accepted | PASS |

Failed gates: none

---

## Policy decision

**Decision:** `LIMITED_LOCKED_TEST_REUSE_EXCEPTION_RECOMMENDED`

**Basis:** No untouched existing replacement holdout is available and all
reuse gates R1–R10 pass. A limited reuse exception may be scientifically
defensible subject to **independent review**.

**Not authorized in this PR:**

- recovery execution
- LOCKED_TEST reopen
- M-B11

---

## If limited reuse is later authorized (proposed, not authorized)

Future recovery contract status: `PROPOSED_NOT_AUTHORIZED`

- Supervised population: `75` pure-class eligible windows (13 AMBIGUOUS excluded)
- Structural context: `16` subjects / `88` total windows
- Models: exactly 3 (seed42 selected, v0.1.0, v0.2.0)
- Expected inference count: `75 × 3 = 225`
- Metrics: exact frozen M-B10A schema
- Required result designation: `REUSED_LOCKED_TEST_AFTER_PREINFERENCE_STRUCTURAL_ABORT`
- Allowed wording: `OFFLINE_REAL_DATA_RECOVERY_EVALUATION_WITH_HOLDOUT_REUSE_LIMITATION`
- Forbidden wording includes: `PRISTINE_REAL_SUBJECT_FINAL_TEST`,
  `PRISTINE_ONE_TIME_LOCKED_TEST`, `PRISTINE_LOCKED_TEST`,
  `FIRST_LOCKED_TEST_EVALUATION`, `LOCKED_TEST_NOT_CONSUMED`,
  `NO_INFORMATION_EXPOSURE`, `ORIGINAL_ACCESS_UNUSED`

Original `original_final_accessor_invocations = 1` must remain immutable.
A future authorized recovery would be a second payload release event, not a
rewrite of history.

---

## Alternative if reuse is not accepted

If independent review rejects reuse:

- `NEW_INDEPENDENT_HOLDOUT_REQUIRED` via `NEW_UNSEEN_SUBJECT_DATA`
- No A5 reshuffle
- No TRAIN/VALIDATION promotion to test

---

## M-B10R0 access audit

| Counter | Value |
|---------|-------|
| Previous historical original access events | 1 |
| New M-B10R0 accessor invocations | 0 |
| Recovery runner executions | 0 |

No LOCKED_TEST payload was accessed during M-B10R0.

---

## Claim boundaries

Allowed: `LIMITED_LOCKED_TEST_REUSE_EXCEPTION_RECOMMENDED`,
`PREINFERENCE_STRUCTURAL_ABORT`, `NO_PERFORMANCE_INFORMATION_OBSERVED`,
`NO_EXISTING_UNUSED_SUBJECT_HOLDOUT`, `RECOVERY_REQUIRES_INDEPENDENT_AUTHORIZATION`

Forbidden: `LOCKED_TEST_PRISTINE`, `RECOVERY_ALREADY_AUTHORIZED`,
`FINAL_PERFORMANCE_VALIDATED`, `M-B11_READY`

---

## Validation

- M-B10R0 validator: PASS (independently recomputes R1–R10 and policy decision;
  does not import generator `_reuse_gates` / `_policy_decision`)
- M-B10B incident regression: PASS
- M-B10A regression: PASS (`--skip-upstream`)

---

## Recommendation

Await independent review of this policy recommendation before any recovery
implementation, accessor modification, or additional LOCKED_TEST access.

This is a **limited exception** recommendation, not restoration of pristine
test status.
