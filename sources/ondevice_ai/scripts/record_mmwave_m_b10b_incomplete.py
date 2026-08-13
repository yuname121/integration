#!/usr/bin/env python3
"""Record the immutable M-B10B incomplete state without reopening LOCKED_TEST.

This command is intentionally usable only after the formal runner has already
written an ``INCOMPLETE_NO_RERUN`` access audit.  It reads that audit, enriches
it with the expected/observed structural identity, and writes explicit
``NOT_GENERATED`` placeholders for result files.  It never imports or calls
the PhaseBAccessGuard final accessor.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

from mmwave_m_b10b_final_eval import (
    FINAL_OUTPUT_FILES,
    M_B10A_DIR_REL,
    OUT_DIR_REL,
    PHASE_ID,
    _write_checksums,
    load_json,
    sha256_file,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / OUT_DIR_REL


def _placeholder(status: str = "NOT_GENERATED_DUE_TO_LOCKED_SPLIT_IDENTITY_MISMATCH") -> dict[str, object]:
    return {"phase_id": PHASE_ID, "status": status, "results_available": False}


def record() -> None:
    audit_path = OUT / "one_time_access_audit.json"
    audit = load_json(audit_path)
    if audit.get("access_consumed") is not True or audit.get("accessor_invocation_count") != 1:
        raise RuntimeError("INCOMPLETE_RECORD_REQUIRES_CONSUMED_SINGLE_ACCESS_AUDIT")
    if audit.get("post_access_status") != "INCOMPLETE_NO_RERUN" or audit.get("second_accessor_invocation") is not False:
        raise RuntimeError("INCOMPLETE_RECORD_AUDIT_STATE_INVALID")
    if audit.get("failure") != "M-B10B_LOCKED_SPLIT_IDENTITY_MISMATCH":
        raise RuntimeError("INCOMPLETE_RECORD_EXPECTED_STRUCTURAL_MISMATCH")

    audit.update({
        "expected_structural_subjects": 16,
        "expected_structural_windows": 88,
        "actual_structural_windows": int(audit.get("structural_rows_returned", 0)),
        "actual_structural_subjects": "NOT_RECORDED_BEFORE_ABORT",
        "completed_model_inference_invocations": 0,
        "partial_evaluation_state": "ABORTED_BEFORE_MODEL_INFERENCE",
        "no_rerun_performed": True,
        "result_validity": "INVALID_INCOMPLETE_STRUCTURAL_IDENTITY",
    })
    write_json(audit_path, audit)

    write_json(OUT / "authorization_record.json", {
        "schema_version": "M-B10B_AUTHORIZATION_RECORD_V1",
        "phase_id": PHASE_ID,
        "authorization_source": "explicit external authorization in user-provided M-B10B execution prompt",
        "authorization_scope": "M-B10B_ONE_TIME_LOCKED_TEST_FINAL_EVALUATION",
        "authorization_present": True,
        "formal_accessor_invocations": 1,
        "pre_access_counts": {"final_accessor_invocations": 0, "tensors": 0, "labels": 0, "predictions": 0, "metrics": 0},
        "second_access_prohibited": True,
        "access_consumed": True,
        "result_status": "INCOMPLETE_NO_RERUN",
    })
    write_json(OUT / "input_identity.json", {
        "schema_version": "M-B10B_INPUT_IDENTITY_V1",
        "phase_id": PHASE_ID,
        "source_split": "LOCKED_TEST",
        "expected_structural_windows": 88,
        "actual_structural_windows": int(audit.get("structural_rows_returned", 0)),
        "structural_identity_match": False,
        "labels_or_tensors_persisted": False,
        "inference_started": False,
    })
    write_json(OUT / "locked_test_registry.json", {
        "schema_version": "M-B10B_LOCKED_TEST_REGISTRY_V1",
        "phase_id": PHASE_ID,
        "split": "LOCKED_TEST",
        "status": "NOT_GENERATED_DUE_TO_LOCKED_SPLIT_IDENTITY_MISMATCH",
        "expected_window_count": 88,
        "actual_window_count": int(audit.get("structural_rows_returned", 0)),
        "samples": [],
        "raw_tensors_persisted": False,
    })
    (OUT / "locked_test_sample_predictions.jsonl").write_text("", encoding="utf-8")
    for name in (
        "model_evaluation_coverage.json",
        "metrics_by_model.json",
        "per_class_metrics.json",
        "subject_level_metrics.json",
        "model_comparison.json",
        "selected_candidate_final_test_result.json",
        "historical_baseline_final_test_results.json",
        "selected_candidate_quantization_audit.json",
    ):
        write_json(OUT / name, _placeholder())
    write_json(OUT / "model_evaluation_coverage.json", {
        **_placeholder(),
        "final_accessor_invocations": 1,
        "model_inference_invocations": 0,
        "evaluation_started": False,
    })
    write_json(OUT / "test_split_consumption_record.json", {
        "phase_id": PHASE_ID,
        "status": "LOCKED_TEST_CONSUMED_FOR_FINAL_PHASE_B_EVALUATION_INCOMPLETE",
        "access_phase": PHASE_ID,
        "candidate_frozen_before_access": True,
        "models_frozen_before_access": True,
        "must_not_reuse_for_phase_b_model_selection": True,
        "new_experiment_cycle_and_holdout_required_for_improvement": True,
        "no_rerun_performed": True,
    })
    write_json(OUT / "run_environment.json", {
        "phase_id": PHASE_ID,
        "timestamp_utc": audit.get("access_timestamp_utc"),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "formal_m_b8_benchmark_rerun": False,
        "formal_accessor_invocations": 1,
        "formal_model_inference_invocations": 0,
    })
    write_json(OUT / "exceptions.json", {
        "phase_id": PHASE_ID,
        "status": "BLOCKER",
        "classification": "BLOCKER",
        "code": "M-B10B_LOCKED_SPLIT_IDENTITY_MISMATCH",
        "detail": "Authorized accessor returned 75 pure-class rows while M-B10A preregistered structural identity requires 88 LOCKED_TEST windows.",
        "access_consumed": True,
        "completed_model_inference_invocations": 0,
        "no_rerun_performed": True,
    })
    write_json(OUT / "m_b10b_summary.json", {
        "phase_id": PHASE_ID,
        "status": "INCOMPLETE_NO_RERUN",
        "locked_test_consumed": True,
        "final_accessor_invocations": 1,
        "model_inference_invocations": 0,
        "models_evaluated": [],
        "selected_candidate_unchanged": True,
        "seed43_evaluated": False,
        "seed44_evaluated": False,
        "model_trainings": 0,
        "model_conversions": 0,
        "recalibrations": 0,
        "threshold_tuning": False,
        "post_test_selection": False,
        "no_post_test_tuning": True,
        "m_b11_started": False,
        "m_b11_authorization_recommendation": "NO",
    })
    report_path = ROOT / "docs/reports/20260812_Codex_M-B10B_One_Time_Locked_Test_Final_Evaluation_01.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"""# M-B10B RESULT

Track / phase: mmWave / M-B10B

Canonical base: `de7effb1a5cc3a7a95483d9dc5d135500a8cefa9`

Branch: `feature/M-B10B-locked-test-final-evaluation`

Pre-access harness commit: `7073374`

Final evidence commit: `a2992e0`

Head commit: `a2992e0`

PR: [#47](https://github.com/sheepmeat/test/pull/47)

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

Structural windows: expected `88`; actual returned `{{audit_actual}}`

Actual registry subjects: NOT GENERATED

Actual registry windows: NOT GENERATED — structural gate failed before registry preservation

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

The one-time accessor returned `{{audit_actual}}` pure-class rows because the existing final accessor excludes `AMBIGUOUS` windows, while M-B10A preregistered structural identity is `88` windows. The structural identity gate therefore failed before model inference. No labels, tensors, predictions, or metrics were persisted from the returned payload.

The execution is invalid/incomplete as final performance evidence. The consumed split must not be reopened or reused in this experimental cycle.

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
- Test-consumption gate: BLOCKED by structural identity mismatch
- Checksums: PASS for incomplete evidence directory

## Tests / regressions

- M-B10A validator: PASS before access
- M-B10B pre-access validator: PASS
- M-B10B pre-access focused tests: PASS (post-access corruption matrix not run because no successful result ledger exists)
- M-B9–M-B0 plus A5/A6 validators: PASS before access

## Git isolation

- Unique M-B10B commits before access: `2`
- Pre-access commit precedes incomplete evidence: YES
- Unrelated-track commits: `0`
- AGENTS.md: `0`
- models/model_manifest.json: `0`
- CO₂: `0`
- Thermal: `0`
- Integration/shared: `0`
- raw payload: `0`
- Working tree: clean at evidence handoff

## Claim boundaries

- OFFLINE_REAL_DATA: NOT CLAIMED — final inference did not complete
- REAL_SUBJECT_GENERALIZATION: NOT CLAIMED
- MR60_VALIDATED: NO
- RASPBERRY_PI_VALIDATED: NO
- PRODUCTION_READY: NO
- CLINICAL_APNEA_VALIDATED: NO

## Warnings

- The authorized LOCKED_TEST split is consumed and cannot be reopened under this cycle's one-time policy.
- The accessor/contract structural identity mismatch requires independent review and a new holdout/reuse policy for any future attempt.

## Blockers

- `BLOCKER: M-B10B_LOCKED_SPLIT_IDENTITY_MISMATCH` — accessor returned `{{audit_actual}}` pure-class rows versus preregistered `88` structural windows.

## M-B11 authorization recommendation

NO — M-B10B execution integrity requires review; LOCKED_TEST must NOT be reopened
""".replace("{audit_actual}", str(audit.get("structural_rows_returned", 0))).rstrip() + "\n",
        encoding="utf-8",
    )
    _write_checksums(ROOT, OUT)


if __name__ == "__main__":
    record()
