#!/usr/bin/env python3
"""Generate M-B10R1-A recovery harness pre-freeze evidence.

Never calls recovery accessor or LOCKED_TEST final accessor.
"""

from __future__ import annotations

import copy
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.mmwave_m_b10r1_recovery_access import (  # noqa: E402
    EXPECTED_AMBIGUOUS,
    EXPECTED_ELIGIBLE,
    EXPECTED_INFERENCES,
    EXPECTED_STRUCTURAL,
    EXPECTED_SUBJECTS,
    ORIGINAL_FINAL_TOKEN,
    RECOVERY_AUTHORIZATION_TOKEN,
    RESULT_LIMITATION,
    LimitedReuseRecoveryAccessController,
)
from scripts.mmwave_m_b10r1_result_writer import (  # noqa: E402
    B_OUT_DIR_REL,
    a_directory_immutability_contract,
    future_b_result_directory_schema,
    not_authorized_overlay_template,
)
from scripts.mmwave_m_b10r1_recovery_eval import (  # noqa: E402
    EXECUTOR_PATH,
    EXECUTOR_SHA,
    META_V01_PATH,
    META_V01_SHA,
    META_V02_PATH,
    META_V02_SHA,
    M_B10A_CONTRACT_SHA,
    M_B10A_DIR_REL,
    M_B10B_DIR_REL,
    M_B10R0_DIR_REL,
    OUT_DIR_REL,
    SELECTED_CANDIDATE_ID,
    SELECTED_MODEL_ID,
    SELECTED_PATH,
    SELECTED_PREPROCESSING_CONTRACT_ID,
    SELECTED_SHA,
    V01_PATH,
    V01_PREPROCESSING_CONTRACT_ID,
    V01_SHA,
    V02_PATH,
    V02_PREPROCESSING_CONTRACT_ID,
    V02_SHA,
    build_bound_contract_identity,
    build_execution_freeze_identity,
    build_preaccess_readiness,
    future_ledger_schema,
    future_result_schema,
    sha256_file,
    validate_frozen_recovery_models,
)

REPORT_REL = Path("docs/reports/20260813_Cursor_M-B10R1A_Recovery_PreFreeze_01.md")

REQUIRED_OUTPUTS = {
    "input_identity.json",
    "incident_identity.json",
    "reuse_policy_identity.json",
    "frozen_recovery_contract.json",
    "model_identity_registry.json",
    "baseline_identity_registry.json",
    "recovery_population_contract.json",
    "metric_contract.json",
    "recovery_access_contract.json",
    "recovery_access_readiness.json",
    "recovery_access_audit.json",
    "future_result_schema.json",
    "future_ledger_schema.json",
    "execution_freeze_identity.json",
    "a_directory_immutability_contract.json",
    "future_b_authorization_overlay.json",
    "future_b_result_directory_schema.json",
    "run_environment.json",
    "exceptions.json",
    "m_b10r1a_summary.json",
    "checksums.sha256",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(root: Path, summary: dict[str, Any]) -> None:
    path = root / REPORT_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    body = f"""# M-B10R1-A — Limited Holdout-Reuse Recovery Harness Pre-Freeze

**RECOVERY HAS NOT BEEN EXECUTED**

**LOCKED_TEST HAS NOT BEEN REOPENED DURING M-B10R1-A**

## Status

| Field | Value |
|---|---|
| Phase | M-B10R1-A |
| Generated at (UTC) | {summary.get("generated_at")} |
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

1. `{SELECTED_MODEL_ID}` SHA `{SELECTED_SHA}`
2. `mmwave_resp_int8` SHA `{V01_SHA}`
3. `mmwave_resp_int8_v0.2.0_candidate` SHA `{V02_SHA}`

Forbidden: seed43, seed44, fourth model.

Expected future inferences (M-B10R1-B only): **225** (= 75 × 3).

## Access design

- Module: `scripts/mmwave_m_b10r1_recovery_access.py`
- Distinct token id: `{RECOVERY_AUTHORIZATION_TOKEN}`
- Original final token rejected: `{ORIGINAL_FINAL_TOKEN}`
- `mmwave_phase_b_access.py` unmodified (0 diff required)
- At-most-one recovery payload release
- Historical original counters never reset
- Result designation: `{RESULT_LIMITATION}`

## Result limitation

Future recovery results (if independently authorized) are **not pristine** and
must carry designation `{RESULT_LIMITATION}`.

## Explicit non-claims

- No recovery performance numbers
- No LOCKED_TEST reopen during this phase
- No M-B10R1-B authorization
- No M-B11 start

## Recovery-path truth closures (M-B10R1A final)

- Window-vs-signal: evaluation uses numeric ``signal`` ndarray for
  ``preprocess_for_spec`` (never window metadata dict).
- Exact preprocessing contract IDs:
  - selected: `{SELECTED_PREPROCESSING_CONTRACT_ID}`
  - v0.1: `{V01_PREPROCESSING_CONTRACT_ID}`
  - v0.2: `{V02_PREPROCESSING_CONTRACT_ID}`
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
"""
    path.write_text(body, encoding="utf-8")


def generate_m_b10r1a_prefreeze(
    root: Path | None = None,
    *,
    validator_pass_placeholder: bool = False,
) -> dict[str, Any]:
    """Write all M-B10R1-A pre-freeze artifacts. Never accesses recovery payload."""
    root = Path(root) if root is not None else ROOT_DIR
    out = root / OUT_DIR_REL
    out.mkdir(parents=True, exist_ok=True)

    # Ensure runtime audit state exists with historical facts, zero recovery counters.
    controller = LimitedReuseRecoveryAccessController(root, audit_state_path=out / "recovery_access_runtime_state.json")
    controller.assert_preaccess_only()

    specs = validate_frozen_recovery_models(root)
    bound = build_bound_contract_identity(root)
    r0 = root / M_B10R0_DIR_REL
    policy = _load_json(r0 / "policy_decision.json")
    proposed = _load_json(r0 / "proposed_recovery_evaluation_contract.json")
    m_b10a_contract = _load_json(root / M_B10A_DIR_REL / "locked_test_evaluation_contract.json")
    incident_r0 = _load_json(r0 / "incident_identity.json")
    baseline_registry = _load_json(root / M_B10A_DIR_REL / "historical_baseline_registry.json")

    if policy.get("decision") != "LIMITED_LOCKED_TEST_REUSE_EXCEPTION_RECOMMENDED":
        raise RuntimeError("M_B10R0_POLICY_NOT_LIMITED_REUSE")

    input_identity = {
        "phase_id": "M-B10R1-A",
        "schema_version": "M-B10R1A_INPUT_IDENTITY_V1",
        "upstream": {
            "M-B10R0": str(M_B10R0_DIR_REL),
            "M-B10B": str(M_B10B_DIR_REL),
            "M-B10A": str(M_B10A_DIR_REL),
        },
        "bound_contract_identity": bound,
        "mmwave_phase_b_access_modified": False,
        "recovery_executed": False,
    }

    incident_identity = {
        "phase_id": "M-B10R1-A",
        "schema_version": "M-B10R1A_INCIDENT_IDENTITY_V1",
        "preserved_from": "M-B10B/M-B10R0",
        "original_accessor_invocations": 1,
        "rows_returned": 75,
        "model_inference_invocations": 0,
        "predictions_generated": False,
        "metrics_generated": False,
        "locked_test_consumed": True,
        "original_pristine_status": False,
        "forensic_root_cause": incident_r0.get("forensic_root_cause"),
        "runtime_detection_code": incident_r0.get("runtime_detection_code"),
        "structural_windows": EXPECTED_STRUCTURAL,
        "supervised_eligible_windows": EXPECTED_ELIGIBLE,
        "excluded_ambiguous_windows": EXPECTED_AMBIGUOUS,
    }

    reuse_policy_identity = {
        "phase_id": "M-B10R1-A",
        "schema_version": "M-B10R1A_REUSE_POLICY_IDENTITY_V1",
        "decision": policy["decision"],
        "reuse_exception_eligible": policy.get("reuse_exception_eligible"),
        "recovery_execution_authorized_in_r0": False,
        "required_result_limitation": RESULT_LIMITATION,
        "bindings": {
            "policy_decision_sha256": bound["policy_decision_sha256"],
            "reuse_exception_gate_results_sha256": bound["reuse_exception_gate_results_sha256"],
            "proposed_recovery_evaluation_contract_sha256": bound[
                "proposed_recovery_evaluation_contract_sha256"
            ],
            "future_recovery_access_requirements_sha256": bound[
                "future_recovery_access_requirements_sha256"
            ],
            "m_b10r0_summary_sha256": bound["m_b10r0_summary_sha256"],
        },
    }

    frozen_recovery_contract = copy.deepcopy(proposed)
    frozen_recovery_contract["phase_id"] = "M-B10R1-A"
    frozen_recovery_contract["schema_version"] = "M-B10R1A_FROZEN_RECOVERY_CONTRACT_V1"
    frozen_recovery_contract["status"] = "FROZEN_PREACCESS_NOT_AUTHORIZED"
    frozen_recovery_contract["recovery_execution_authorized"] = False
    frozen_recovery_contract["locked_test_reopen_authorized"] = False
    frozen_recovery_contract["bound_contract_identity"] = bound
    frozen_recovery_contract["bindings"] = {
        "m_b10a_metric_contract_sha256": M_B10A_CONTRACT_SHA,
        "selected_model_sha256": SELECTED_SHA,
        "baseline_v01_sha256": V01_SHA,
        "baseline_v02_sha256": V02_SHA,
        "executor_sha256": EXECUTOR_SHA,
        "preprocessing_contract_ids": {
            SELECTED_MODEL_ID: SELECTED_PREPROCESSING_CONTRACT_ID,
            "mmwave_resp_int8": V01_PREPROCESSING_CONTRACT_ID,
            "mmwave_resp_int8_v0.2.0_candidate": V02_PREPROCESSING_CONTRACT_ID,
        },
    }

    model_identity_registry = {
        "phase_id": "M-B10R1-A",
        "schema_version": "M-B10R1A_MODEL_IDENTITY_REGISTRY_V1",
        "model_count": 3,
        "seed42_exact": True,
        "forbidden_seeds": [43, 44],
        "models": specs,
        "selected_candidate_id": SELECTED_CANDIDATE_ID,
        "selected_model_id": SELECTED_MODEL_ID,
        "selected_path": SELECTED_PATH,
        "selected_sha256": SELECTED_SHA,
        "class_map": {"0": "NORMAL", "1": "RAPID_OR_ABNORMAL", "2": "APNEA"},
    }

    # Bind baselines with executor/metadata SHAs from live files + M-B10A registry.
    baselines_out = []
    for entry in baseline_registry.get("baselines", []):
        item = copy.deepcopy(entry)
        baselines_out.append(item)
    baseline_identity_registry = {
        "phase_id": "M-B10R1-A",
        "schema_version": "M-B10R1A_BASELINE_IDENTITY_REGISTRY_V1",
        "executor_path": EXECUTOR_PATH,
        "executor_sha256": EXECUTOR_SHA,
        "v0_1": {
            "model_id": "mmwave_resp_int8",
            "path": V01_PATH,
            "sha256": V01_SHA,
            "metadata_path": META_V01_PATH,
            "metadata_sha256": META_V01_SHA,
            "executor_path": EXECUTOR_PATH,
            "executor_sha256": EXECUTOR_SHA,
        },
        "v0_2": {
            "model_id": "mmwave_resp_int8_v0.2.0_candidate",
            "path": V02_PATH,
            "sha256": V02_SHA,
            "metadata_path": META_V02_PATH,
            "metadata_sha256": META_V02_SHA,
            "executor_path": EXECUTOR_PATH,
            "executor_sha256": EXECUTOR_SHA,
        },
        "baselines": baselines_out,
        "live_verified": True,
    }

    recovery_population_contract = {
        "phase_id": "M-B10R1-A",
        "schema_version": "M-B10R1A_RECOVERY_POPULATION_CONTRACT_V1",
        "structural_windows": EXPECTED_STRUCTURAL,
        "supervised_eligible_windows": EXPECTED_ELIGIBLE,
        "excluded_ambiguous_windows": EXPECTED_AMBIGUOUS,
        "subjects": EXPECTED_SUBJECTS,
        "subject_count_policy": "PREEXISTING_A6_METADATA_VERIFIED",
        "eligibility_provenance": "PREEXISTING_A6_METADATA_VERIFIED",
        "eligibility_rule": (
            "split == LOCKED_TEST AND assignment_status != AMBIGUOUS; "
            "equivalent to locked_test_evaluation_eligible == true via "
            "PhaseBAccessGuard._get_split_dataset(include_ambiguous=False)"
        ),
        "positional_truncation": False,
        "include_ambiguous": False,
        "expected_model_inference_count": EXPECTED_INFERENCES,
        "not_first_n_rows": True,
    }

    metrics_schema = copy.deepcopy(m_b10a_contract["metrics_schema"])
    metric_contract = {
        "phase_id": "M-B10R1-A",
        "schema_version": "M-B10R1A_METRIC_CONTRACT_V1",
        "metrics_schema": metrics_schema,
        "metrics_schema_source": str(M_B10A_DIR_REL / "locked_test_evaluation_contract.json"),
        "m_b10a_contract_sha256": M_B10A_CONTRACT_SHA,
        "deep_equal_to_m_b10a_metrics_schema": True,
        "applicable_predefined_numerical_acceptance_threshold": (
            "FINAL_LOCKED_TEST_NUMERICAL_ACCEPTANCE_THRESHOLD_NOT_PREDEFINED"
        ),
        "acceptance_threshold": "NOT_PREDEFINED",
        "status": "PREREGISTERED_NOT_EXECUTED",
        "metrics_populated": False,
    }

    recovery_access_contract = {
        "phase_id": "M-B10R1-A",
        "schema_version": "M-B10R1A_RECOVERY_ACCESS_CONTRACT_V1",
        "module_path": "scripts/mmwave_m_b10r1_recovery_access.py",
        "accessor_api": (
            "scripts/mmwave_m_b10r1_recovery_access.py:"
            "LimitedReuseRecoveryAccessController.get_locked_test_recovery_evaluation_dataset"
        ),
        "authorization_token_id": "M_B10R1_LIMITED_REUSE_RECOVERY_AUTHORIZATION_V1",
        "authorization_token_constant": RECOVERY_AUTHORIZATION_TOKEN,
        "original_final_token_rejected": ORIGINAL_FINAL_TOKEN,
        "at_most_one_recovery_payload_release": True,
        "original_counter_reset_forbidden": True,
        "modifies_mmwave_phase_b_access": False,
        "loads_via": "PhaseBAccessGuard._get_split_dataset('LOCKED_TEST', include_ambiguous=False)",
        "forbids_final_accessor": "get_locked_test_final_evaluation_dataset",
        "include_ambiguous": False,
        "expected_eligible_windows": EXPECTED_ELIGIBLE,
        "expected_subjects": EXPECTED_SUBJECTS,
        "result_limitation": RESULT_LIMITATION,
        "authorization_during_m_b10r1a": False,
        "b_authorization_overlay_path": str(B_OUT_DIR_REL / "authorization_record.json"),
        "a_readiness_mutation_required_for_b": False,
        "future_b_runtime_state_path": str(B_OUT_DIR_REL / "recovery_access_runtime_state.json"),
        "a_runtime_state_mutation_required": False,
    }

    readiness = build_preaccess_readiness(root, validator_pass=validator_pass_placeholder)
    # Generator sets pending; validator confirms pass and may rewrite.
    if not validator_pass_placeholder:
        readiness["pre_access_validator_pass"] = False
        readiness["pre_access_validator_status"] = "PENDING_VALIDATOR"

    execution_freeze_identity = build_execution_freeze_identity(root)

    audit = {
        "phase_id": "M-B10R1-A",
        "schema_version": "M-B10R1A_RECOVERY_ACCESS_AUDIT_V1",
        "historical_original_final_accessor_invocations": 1,
        "historical_original_payload_release_events": 1,
        "M-B10R1A_recovery_accessor_invocations": 0,
        "M-B10R1A_recovery_payload_release_events": 0,
        "historical_total_payload_release_events": 1,
        "original_locked_test_consumed": True,
        "original_pristine_status": False,
        "runtime_state": controller.snapshot(),
    }

    run_environment = {
        "phase_id": "M-B10R1-A",
        "generated_at": _utc_now(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "recovery_accessor_invoked": False,
        "locked_test_final_accessor_invoked": False,
        "prefreeze_only": True,
    }

    exceptions = {
        "phase_id": "M-B10R1-A",
        "status": "NO_EXECUTION_EXCEPTIONS",
        "classification": "PREFREEZE_ONLY",
        "recovery_executed": False,
    }

    summary = {
        "phase_id": "M-B10R1-A",
        "status": "RECOVERY_HARNESS_PREFREEZE_COMPLETE",
        "generated_at": run_environment["generated_at"],
        "policy_decision": policy["decision"],
        "result_limitation": RESULT_LIMITATION,
        "recovery_execution_authorized": False,
        "recovery_payload_release_authorized": False,
        "M-B10R1B_started": False,
        "m_b11_started": False,
        "new_recovery_accessor_invocations": 0,
        "new_payload_release_events": 0,
        "structural_windows": EXPECTED_STRUCTURAL,
        "supervised_eligible_windows": EXPECTED_ELIGIBLE,
        "ambiguous_windows": EXPECTED_AMBIGUOUS,
        "subjects": EXPECTED_SUBJECTS,
        "expected_future_inferences": EXPECTED_INFERENCES,
        "model_count": 3,
        "mmwave_phase_b_access_diff": 0,
        "preprocessing_contract_ids": {
            SELECTED_MODEL_ID: SELECTED_PREPROCESSING_CONTRACT_ID,
            "mmwave_resp_int8": V01_PREPROCESSING_CONTRACT_ID,
            "mmwave_resp_int8_v0.2.0_candidate": V02_PREPROCESSING_CONTRACT_ID,
        },
        "execution_freeze_identity": "execution_freeze_identity.json",
        "b_authorization_overlay": str(B_OUT_DIR_REL / "authorization_record.json"),
        "b_authorization_status": "NOT_AUTHORIZED_NOT_EXECUTED",
        "a_directory_immutable_after_merge": True,
        "a_readiness_mutation_required_for_b": False,
        "report": str(REPORT_REL),
    }

    artifacts = {
        "input_identity.json": input_identity,
        "incident_identity.json": incident_identity,
        "reuse_policy_identity.json": reuse_policy_identity,
        "frozen_recovery_contract.json": frozen_recovery_contract,
        "model_identity_registry.json": model_identity_registry,
        "baseline_identity_registry.json": baseline_identity_registry,
        "recovery_population_contract.json": recovery_population_contract,
        "metric_contract.json": metric_contract,
        "recovery_access_contract.json": recovery_access_contract,
        "recovery_access_readiness.json": readiness,
        "recovery_access_audit.json": audit,
        "future_result_schema.json": future_result_schema(),
        "future_ledger_schema.json": future_ledger_schema(),
        "execution_freeze_identity.json": execution_freeze_identity,
        "a_directory_immutability_contract.json": a_directory_immutability_contract(),
        "future_b_result_directory_schema.json": future_b_result_directory_schema(),
        "run_environment.json": run_environment,
        "exceptions.json": exceptions,
        "m_b10r1a_summary.json": summary,
    }

    for name, payload in artifacts.items():
        (out / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    freeze_sha = sha256_file(out / "execution_freeze_identity.json")
    overlay = not_authorized_overlay_template(
        freeze_sha=freeze_sha,
        a_head=None,
    )
    overlay["reviewed_m_b10r1a_head_sha_status"] = "PENDING_INDEPENDENT_REVIEW"
    overlay["prefreeze_head_informational"] = execution_freeze_identity.get("pre_freeze_head")
    (out / "future_b_authorization_overlay.json").write_text(
        json.dumps(overlay, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    b_out = root / B_OUT_DIR_REL
    b_out.mkdir(parents=True, exist_ok=True)
    # Template only — never populate measured B results during M-B10R1-A.
    (b_out / "authorization_record.json").write_text(
        json.dumps(overlay, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Runtime state is supporting audit; include in checksums.
    checksum_lines = []
    for path in sorted(out.iterdir(), key=lambda p: p.name):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_lines.append(f"{sha256_file(path)}  {path.name}")
    (out / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    _write_report(root, summary)
    return {
        "phase_id": "M-B10R1-A",
        "status": summary["status"],
        "output_dir": str(OUT_DIR_REL),
        "report": str(REPORT_REL),
        "recovery_executed": False,
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    result = generate_m_b10r1a_prefreeze()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
