# M-B10R1-A — Limited Holdout-Reuse Recovery Harness Pre-Freeze

**RECOVERY HAS NOT BEEN EXECUTED**

**LOCKED_TEST HAS NOT BEEN REOPENED DURING M-B10R1-A**

## Status

| Field | Value |
|---|---|
| Phase | M-B10R1-A |
| Generated at (UTC) | 2026-08-12T19:44:45Z |
| Recovery execution authorized | false |
| Recovery payload release authorized | false |
| New recovery accessor invocations | 0 |
| New recovery payload releases | 0 |
| Recovery model inference | 0 |
| M-B10R1-B started | false |
| M-B11 started | false |

## Purpose

Freeze the limited-reuse recovery harness, access controller, metric engine,
future runner, validators, and pre-access contracts **without** releasing any
LOCKED_TEST recovery payload.

## Historical original access (preserved)

- Original M-B10B final accessor invocations: **1**
- Original rows returned: **75**
- Original model inference: **0**
- Original LOCKED_TEST consumed: **true**
- Original pristine status: **false**

## Frozen recovery population

- Structural windows: **88**
- Supervised eligible: **75**
- Excluded AMBIGUOUS: **13**
- Subjects: **16**
- Provenance: `PREEXISTING_A6_METADATA_VERIFIED`
- Positional truncation: **false**
- Eligibility: `assignment_status != AMBIGUOUS` (A6 semantics)

## Exact future model set (3)

1. `M-B3_CONV1D_GAP_BASELINE_seed42_M-B6_STRICT_INT8` SHA `6dff6aaa72c79d76715d40cf7e32bb1e6cd9b2c2e3ac78eaf2fda737561430c5`
2. `mmwave_resp_int8` SHA `43cdd6f321c2b645232162233e098c2b65549388cbc9e680a17a0eccdc8f0158`
3. `mmwave_resp_int8_v0.2.0_candidate` SHA `85c023d3eefca13ecbb72a841974e53a56d5f4173920645d46df49a9088452ff`

Forbidden: seed43, seed44, fourth model.

Expected future inferences (M-B10R1-B only): **225** (= 75 × 3).

## Access design

- Module: `scripts/mmwave_m_b10r1_recovery_access.py`
- Distinct token id: `M_B10R1_LIMITED_REUSE_RECOVERY_AUTHORIZATION_V1`
- Original final token rejected: `AUTHORIZED_FINAL_LOCKED_TEST_EVALUATION_TOKEN_V1`
- `mmwave_phase_b_access.py` unmodified (0 diff required)
- At-most-one recovery payload release
- Historical original counters never reset
- Result designation: `REUSED_LOCKED_TEST_AFTER_PREINFERENCE_STRUCTURAL_ABORT`

## Result limitation

Future recovery results (if independently authorized) are **not pristine** and
must carry designation `REUSED_LOCKED_TEST_AFTER_PREINFERENCE_STRUCTURAL_ABORT`.

## Explicit non-claims

- No recovery performance numbers
- No LOCKED_TEST reopen during this phase
- No M-B10R1-B authorization
- No M-B11 start

## Recovery-path truth closures (M-B10R1A final)

- Window-vs-signal: evaluation uses numeric ``signal`` ndarray for
  ``preprocess_for_spec`` (never window metadata dict).
- Exact preprocessing contract IDs:
  - selected: `M-B10B_SELECTED_REAL_CANDIDATE_BPF_ZSCORE_V1`
  - v0.1: `M-B10B_HISTORICAL_V0_1_COMPATIBILITY_PREPROCESSING_V1`
  - v0.2: `M-B10B_SYNTHETIC_V0_2_EXTERNAL_COMPATIBILITY_PREPROCESSING_V1`
- Ledger attempts distinguished from TFLite invocations
  (``tflite_invoke_count`` / ``actual_total_tflite_invocations``).
- Selected candidate requires 75 valid predictions for complete status;
  ``metric_bundle`` refuses empty-labels-with-positive evaluated count.
- Payload release recorded at loader return boundary, before verify.
- Authoritative ``execution_freeze_identity.json`` binds harness module SHAs
  and policy/model artifacts; execute path compares live against frozen.
- M-B10R1-A directory is immutable after merge. Future M-B10R1-B must not
  mutate A readiness, A runtime state, A audit, or A checksums.
- B authorization overlay lives at
  ``datasets/mmwave/manifests/M-B10R1B_recovery_execution/authorization_record.json``
  with status ``NOT_AUTHORIZED_NOT_EXECUTED`` and ``approval=false``.
- Future execution reads the B overlay; A
  ``recovery_execution_authorized`` / ``recovery_payload_release_authorized``
  remain historically false forever.
- Durable result writer and post-access validator
  (``scripts/validate_mmwave_m_b10r1b.py``) are frozen now; they validate
  stored B evidence only and never reopen LOCKED_TEST.
- Frozen bound-contract fallback removed: missing
  ``bound_contract_identity`` stops before payload release.
