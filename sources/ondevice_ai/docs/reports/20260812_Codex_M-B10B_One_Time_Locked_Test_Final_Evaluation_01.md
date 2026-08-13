# M-B10B RESULT

Track / phase: mmWave / M-B10B

Canonical base: `de7effb1a5cc3a7a95483d9dc5d135500a8cefa9`

Branch: `feature/M-B10B-locked-test-final-evaluation`

Pre-access harness commit: `7073374`

Final evidence commit: `a2992e0`

Incident-truth closure: `docs(mmwave): close M-B10B count-semantics incident`

Head commit: branch `feature/M-B10B-locked-test-final-evaluation` (PR #47)

PR: [#47](https://github.com/sheepmeat/test/pull/47)

## Forensic count-semantics truth

`88` was the entire structural LOCKED_TEST split.

`75` was the pre-existing pure-class evaluation-eligible population.

The final accessor was already defined to exclude `AMBIGUOUS` rows.

Therefore the accessor's 75-row return was consistent with existing Phase-B
data-access semantics.

The M-B10B pretest harness incorrectly expected `88` returned supervised rows
and aborted before inference.

A6 machine evidence:

- LOCKED_TEST structural subjects: `16`
- LOCKED_TEST total structural windows: `88`
- LOCKED_TEST supervised evaluation eligible windows: `75`
- Difference / AMBIGUOUS excluded from pure-class supervised evaluation: `13`

Returned-count claim:

- `RETURNED_COUNT_MATCHES_PREEXISTING_A6_ELIGIBILITY_COUNT`
- Not claimed: `FULL_75_ROW_IDENTITY_VERIFIED` (subject IDs / sample IDs were not persisted before abort)

## Root-cause classification

Runtime detection code (historical, preserved):

- `M-B10B_LOCKED_SPLIT_IDENTITY_MISMATCH`

Forensic root cause:

- `PRETEST_CONTRACT_COUNT_SEMANTICS_CONFLATION`

Accessor behavior classification:

- `EXPECTED_EXISTING_ACCESSOR_BEHAVIOR`

Do not label the accessor as broken. Dataset corruption, split mutation, and
model failure were not evidenced.

## M-B10A frozen contract

Selected candidate:
- ID: `M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120`
- seed: `42`
- model path: `models/mmwave/experiments/M-B6_stage_equivalence/M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120_int8.tflite`
- SHA: `6dff6aaa72c79d76715d40cf7e32bb1e6cd9b2c2e3ac78eaf2fda737561430c5`
- preprocessing: `BPF_ZSCORE`
- calibration: `M-B5_CAL_CLASS_BALANCED_120`
- class map: `0 NORMAL`, `1 RAPID_OR_ABNORMAL`, `2 APNEA`

M-B10A contract SHA: `ba6429ecfe685de1807ec85b55e697ee12e24138e6b96e94715b0a1a6b19e0f7`

Candidate changed after test:
- NO

M-B10A artifacts modified during incident closure:
- `0` — the incorrect structural `88`-window pretest contract remains frozen historical evidence

## LOCKED_TEST access

Authorization:
- explicit M-B10B authorization present: YES

Accessor implementation: `scripts/mmwave_phase_b_access.py:PhaseBAccessGuard.get_locked_test_final_evaluation_dataset`

Pre-access accessor count: `0`

Formal accessor invocations: `1`

Second accessor invocation:
- NO

LOCKED_TEST consumed:
- YES — consumed by the single authorized accessor; no result rerun is permitted

Structural subjects: expected `16`; actual subject count was not recorded before abort

Structural windows: expected `88` by the incorrect pretest contract; accessor returned `75` evaluation-eligible rows

Actual registry subjects: NOT GENERATED

Actual registry windows: NOT GENERATED — pretest expected-count gate failed before registry preservation

## Model execution

Expected models: seed42 selected candidate, v0.1.0 historical compatibility, v0.2.0 synthetic external compatibility

Actually evaluated models: none

Unexpected models: none

seed43 evaluated:
- NO

seed44 evaluated:
- NO

Model trainings: `0`

Model conversions: `0`

Recalibration: `0`

Threshold tuning: `0`

Post-test selection: `0`

Total formal model inference invocations: `0`

## Final result status

Internal status: `M-B10B_ONE_TIME_EVALUATION_INCOMPLETE_NO_RERUN`

Forensic status: `INCIDENT_ROOT_CAUSE_CLOSED`

Performance result: `FINAL_PERFORMANCE_NOT_AVAILABLE_PREINFERENCE_ABORT`

Accuracy: NOT_AVAILABLE

Macro F1: NOT_AVAILABLE

APNEA proxy metrics: NOT_AVAILABLE

RAPID metrics: NOT_AVAILABLE

No model performance was observed. No candidate was reselected. The test was
not reopened. The test is still treated as consumed. Future reuse requires a
separately preregistered holdout-reuse or new-holdout policy. This PR does not
authorize that policy and does not begin M-B11.

The one-time accessor returned `75` pure-class rows because the existing final
accessor excludes `AMBIGUOUS` windows, while M-B10A/M-B10B preregistered the
full structural identity of `88` windows as the expected supervised row count.
The harness therefore aborted before model inference. No labels, tensors,
predictions, or metrics were persisted from the returned payload.

The execution is invalid/incomplete as final performance evidence. The consumed
split must not be reopened or reused in this experimental cycle.

Predefined numerical final-test threshold: `FINAL_LOCKED_TEST_NUMERICAL_ACCEPTANCE_THRESHOLD_NOT_PREDEFINED`

## One-time evidence gates

- Sample registry gate: NOT REACHED
- Same-test/same-order gate: NOT REACHED
- Model artifact gate: PASS before access
- Preprocessing contract gate: PASS before access
- Class-map gate: PASS before access
- Prediction-ledger gate: NOT REACHED
- Metric independent recomputation: NOT REACHED
- Subject-level recomputation: NOT REACHED
- Quantization audit: NOT REACHED
- No-retuning gate: PASS (`0` training/conversion/recalibration/tuning)
- Test-consumption gate: CONSUMED / incomplete; not restored to pristine
- Incident-root-cause gate: PASS (`PRETEST_CONTRACT_COUNT_SEMANTICS_CONFLATION`)
- Checksums: PASS for incomplete evidence directory including incident closure

## Tests / regressions

- M-B10A validator: PASS (immutable contract regression)
- M-B10B terminal validator: `INCOMPLETE_NO_RERUN`
- M-B10B incident validator: PASS
- M-B10B incident focused tests: PASS
- Successful-result corruption matrix not run because no successful result ledger exists

## Git isolation

- Unique M-B10B commits before access: `2`
- Pre-access commit precedes incomplete evidence: YES
- Unrelated-track commits: `0`
- AGENTS.md: `0`
- models/model_manifest.json: `0`
- M-B10A: `0`
- A5/A6: `0`
- mmwave_phase_b_access.py: `0`
- CO₂: `0`
- Thermal: `0`
- Integration/shared: `0`
- raw payload: `0`
- Working tree: clean at incident-truth handoff

## Claim boundaries

- OFFLINE_REAL_DATA: NOT CLAIMED — final inference did not complete
- REAL_SUBJECT_GENERALIZATION: NOT CLAIMED
- MR60_VALIDATED: NO
- RASPBERRY_PI_VALIDATED: NO
- PRODUCTION_READY: NO
- CLINICAL_APNEA_VALIDATED: NO
- ACCESSOR_BROKEN: NO
- DATASET_CORRUPTED: NO
- SPLIT_MUTATED: NO
- RECOVERY_AUTHORIZED: NO
- M-B11_READY: NO

## Warnings

- The authorized LOCKED_TEST split is consumed and cannot be reopened under this cycle's one-time policy.
- Incident truth is closed, but final scientific performance remains unavailable.
- A separately reviewed holdout-reuse exception or new-holdout policy is required before any additional final evaluation.

## Blockers

- None for incident-truth closure itself.
- Terminal experimental status remains: final performance unavailable (`FINAL_PERFORMANCE_NOT_AVAILABLE_PREINFERENCE_ABORT`).
- Historical runtime detection preserved: `M-B10B_LOCKED_SPLIT_IDENTITY_MISMATCH`.

## M-B11 authorization recommendation

NO — LOCKED_TEST remains consumed and must NOT be reopened without a separately reviewed reuse/new-holdout policy
