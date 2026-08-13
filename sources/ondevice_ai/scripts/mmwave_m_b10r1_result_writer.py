#!/usr/bin/env python3
"""Durable M-B10R1-B result persistence (atomic JSON/JSONL writes).

M-B10R1-A never populates this directory with measured recovery results.
Future authorized recovery must persist the full 225-row ledger before
successful CLI exit. After payload release, evaluation failure persists a
terminal incomplete artifact and never retries.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.mmwave_m_b10r1_metrics import CLASS_MAP  # noqa: E402
from scripts.mmwave_m_b10r1_recovery_access import (  # noqa: E402
    EXPECTED_ELIGIBLE,
    EXPECTED_INFERENCES,
    RESULT_LIMITATION,
)

B_OUT_DIR_REL = Path("datasets/mmwave/manifests/M-B10R1B_recovery_execution")
A_OUT_DIR_REL = Path("datasets/mmwave/manifests/M-B10R1A_recovery_prefreeze")

SELECTED_MODEL_ID = "M-B3_CONV1D_GAP_BASELINE_seed42_M-B6_STRICT_INT8"
V01_MODEL_ID = "mmwave_resp_int8"
V02_MODEL_ID = "mmwave_resp_int8_v0.2.0_candidate"

REQUIRED_B_RESULT_FILES = (
    "authorization_record.json",
    "execution_identity.json",
    "recovery_access_runtime_state.json",
    "one_time_recovery_access_audit.json",
    "recovery_registry.json",
    "recovery_sample_predictions.jsonl",
    "model_evaluation_coverage.json",
    "metrics_by_model.json",
    "per_class_metrics.json",
    "subject_level_metrics.json",
    "model_comparison.json",
    "selected_candidate_recovery_result.json",
    "historical_baseline_recovery_results.json",
    "selected_candidate_quantization_audit.json",
    "recovery_consumption_record.json",
    "run_environment.json",
    "exceptions.json",
    "m_b10r1b_summary.json",
    "checksums.sha256",
)

B_AUTHORIZATION_STATUS_TEMPLATE = "NOT_AUTHORIZED_NOT_EXECUTED"
B_AUTHORIZATION_STATUS_GRANTED = "AUTHORIZED_FOR_ONE_RECOVERY_RELEASE"


class MB10R1ResultWriterError(Exception):
    """Fail-closed result persistence error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write text via temp file + flush + atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def write_checksums(out: Path) -> None:
    lines = []
    for path in sorted(out.iterdir(), key=lambda p: p.name):
        if path.is_file() and path.name != "checksums.sha256":
            lines.append(f"{sha256_file(path)}  {path.name}")
    atomic_write_text(out / "checksums.sha256", "\n".join(lines) + "\n")


def not_authorized_overlay_template(*, freeze_sha: str | None = None, a_head: str | None = None) -> dict[str, Any]:
    """M-B10R1-A template. approval remains false forever in this phase."""
    return {
        "schema_version": "M-B10R1B_AUTHORIZATION_RECORD_V1",
        "phase_id": "M-B10R1-B",
        "status": B_AUTHORIZATION_STATUS_TEMPLATE,
        "approval": False,
        "recovery_execution_authorized": False,
        "recovery_payload_release_authorized": False,
        "independent_reviewer_authorization": False,
        "independent_review_required": True,
        "reviewed_m_b10r1a_head_sha": a_head,
        "execution_freeze_identity_sha256": freeze_sha,
        "one_recovery_release_only": True,
        "retry_prohibited": True,
        "expected_eligible_windows": EXPECTED_ELIGIBLE,
        "expected_subjects": 16,
        "expected_model_inference_count": EXPECTED_INFERENCES,
        "model_count": 3,
        "exact_three_model_contract": True,
        "selected_model_id": SELECTED_MODEL_ID,
        "baseline_model_ids": [V01_MODEL_ID, V02_MODEL_ID],
        "expected_selected_valid_predictions": EXPECTED_ELIGIBLE,
        "result_limitation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "a_readiness_must_remain_historically_false": True,
        "a_directory_immutable": True,
        "a_readiness_mutation_required_for_b": False,
        "reviewed_m_b10r1a_head_sha_status": (
            "PENDING_INDEPENDENT_REVIEW" if not a_head else "BOUND"
        ),
        "notes": (
            "M-B10R1-A overlay template only. Independent reviewer must populate "
            "this B-side record in M-B10R1-B. A recovery_execution_authorized and "
            "recovery_payload_release_authorized remain historically false forever."
        ),
    }


def load_b_authorization_record(root: Path) -> dict[str, Any]:
    path = Path(root) / B_OUT_DIR_REL / "authorization_record.json"
    if not path.is_file():
        raise MB10R1ResultWriterError("B_AUTHORIZATION_RECORD_MISSING")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MB10R1ResultWriterError("B_AUTHORIZATION_RECORD_INVALID")
    return payload


def initialize_b_runtime_from_a(root: Path, b_state_path: Path) -> dict[str, Any]:
    """Create B runtime state from immutable A historical evidence. Never mutates A."""
    b_state_path = Path(b_state_path)
    if b_state_path.is_file():
        return json.loads(b_state_path.read_text(encoding="utf-8"))
    a_path = Path(root) / A_OUT_DIR_REL / "recovery_access_runtime_state.json"
    if not a_path.is_file():
        raise MB10R1ResultWriterError("A_RUNTIME_STATE_MISSING_FOR_B_INIT")
    a_state = json.loads(a_path.read_text(encoding="utf-8"))
    if int(a_state.get("recovery_accessor_invocations", -1)) != 0:
        raise MB10R1ResultWriterError("A_RECOVERY_ACCESS_NOT_ZERO_AT_B_INIT")
    if int(a_state.get("recovery_payload_release_events", -1)) != 0:
        raise MB10R1ResultWriterError("A_RECOVERY_RELEASE_NOT_ZERO_AT_B_INIT")
    if int(a_state.get("original_final_accessor_invocations", 0)) != 1:
        raise MB10R1ResultWriterError("A_ORIGINAL_ACCESS_NOT_1_AT_B_INIT")
    if a_state.get("original_locked_test_consumed") is not True:
        raise MB10R1ResultWriterError("A_ORIGINAL_CONSUMED_NOT_TRUE_AT_B_INIT")
    state = {
        "schema_version": "M-B10R1B_RECOVERY_ACCESS_RUNTIME_STATE_V1",
        "original_final_accessor_invocations": 1,
        "original_locked_test_consumed": True,
        "original_final_payload_release_events": int(
            a_state.get("original_final_payload_release_events", 1)
        ),
        "historical_total_payload_release_events": int(
            a_state.get("historical_total_payload_release_events", 1)
        ),
        "recovery_accessor_invocations": 0,
        "recovery_payload_release_events": 0,
        "payload_consumed": False,
        "rerun_performed": False,
        "automatic_retry": False,
        "initialized_from_immutable_a_historical_evidence": True,
        "a_runtime_state_not_mutated": True,
        "a_runtime_state_path": str(A_OUT_DIR_REL / "recovery_access_runtime_state.json"),
    }
    atomic_write_json(b_state_path, state)
    return state


def future_b_result_directory_schema() -> dict[str, Any]:
    return {
        "schema_version": "M-B10R1B_RESULT_DIRECTORY_SCHEMA_V1",
        "status": "NOT_POPULATED",
        "output_dir": str(B_OUT_DIR_REL),
        "required_files": list(REQUIRED_B_RESULT_FILES),
        "ledger_file": "recovery_sample_predictions.jsonl",
        "expected_ledger_rows": EXPECTED_INFERENCES,
        "stores_raw_phase_tensors": False,
        "populated_during_m_b10r1a": False,
    }


def a_directory_immutability_contract() -> dict[str, Any]:
    return {
        "schema_version": "M-B10R1A_DIRECTORY_IMMUTABILITY_CONTRACT_V1",
        "phase_id": "M-B10R1-A",
        "m_b10r1a_directory_immutable_after_merge": True,
        "directory": str(A_OUT_DIR_REL),
        "future_b_must_not_modify": [
            "recovery_access_readiness.json",
            "recovery_access_runtime_state.json",
            "recovery_access_audit.json",
            "checksums.sha256",
            "execution_freeze_identity.json",
            "any other M-B10R1-A evidence",
        ],
        "historical_proof": {
            "recovery_access": 0,
            "payload_release": 0,
            "authorization": False,
        },
        "future_b_runtime_state_path": str(B_OUT_DIR_REL / "recovery_access_runtime_state.json"),
        "future_b_must_not_mutate_a_readiness": True,
        "a_readiness_recovery_execution_authorized_forever": False,
        "a_readiness_recovery_payload_release_authorized_forever": False,
    }


def _per_class_by_model(metrics_by_model: dict[str, Any]) -> dict[str, Any]:
    return {mid: body.get("per_class", {}) for mid, body in metrics_by_model.items()}


def _model_comparison(metrics_by_model: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary": "macro_f1",
        "numerical_acceptance_threshold": "FINAL_LOCKED_TEST_NUMERICAL_ACCEPTANCE_THRESHOLD_NOT_PREDEFINED",
        "by_model": {
            mid: {
                "accuracy": body.get("accuracy"),
                "macro_f1": body.get("macro_f1"),
                "macro_precision": body.get("macro_precision"),
                "macro_recall": body.get("macro_recall"),
                "valid_count": body.get("valid_count"),
                "invalid_count": body.get("invalid_count"),
                "planned_count": body.get("planned_count"),
            }
            for mid, body in metrics_by_model.items()
        },
        "no_post_result_branching": True,
        "no_performance_threshold_gating": True,
    }


def persist_recovery_results(
    out: Path,
    evaluation: dict[str, Any],
    *,
    runtime_state: dict[str, Any],
    authorization: dict[str, Any],
    frozen: dict[str, Any],
    specs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Persist a complete recovery evaluation. Ledger is written before return."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ledger = list(evaluation.get("ledger") or [])
    if len(ledger) != EXPECTED_INFERENCES:
        raise MB10R1ResultWriterError(f"LEDGER_ROW_COUNT:{len(ledger)}")

    metrics_by_model = evaluation["metrics_by_model"]
    coverage = evaluation["coverage_by_model"]
    subjects = evaluation["subject_metrics_by_model"]
    saturation = evaluation["saturation_audit_seed42"]

    registry = {
        "schema_version": "M-B10R1B_RECOVERY_REGISTRY_V1",
        "status": "RECOVERY_EXECUTED",
        "eligible_window_count": EXPECTED_ELIGIBLE,
        "model_count": 3,
        "ledger_row_count": len(ledger),
        "stores_raw_phase_tensors": False,
        "result_limitation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "ordered_window_ids": [
            row["window_id"] for row in ledger if row["model_id"] == SELECTED_MODEL_ID
        ],
    }

    audit = {
        "schema_version": "M-B10R1B_ONE_TIME_RECOVERY_ACCESS_AUDIT_V1",
        "historical_original_final_accessor_invocations": int(
            runtime_state.get("original_final_accessor_invocations", 1)
        ),
        "historical_original_payload_release_events": int(
            runtime_state.get("original_final_payload_release_events", 1)
        ),
        "recovery_accessor_invocations": int(runtime_state.get("recovery_accessor_invocations", 0)),
        "recovery_payload_release_events": int(runtime_state.get("recovery_payload_release_events", 0)),
        "historical_total_payload_release_events": int(
            runtime_state.get("historical_total_payload_release_events", 0)
        ),
        "payload_consumed": bool(runtime_state.get("payload_consumed")),
        "rerun_performed": False,
        "automatic_retry": False,
        "result_limitation": RESULT_LIMITATION,
        "result_not_pristine": True,
    }

    consumption = {
        "schema_version": "M-B10R1B_RECOVERY_CONSUMPTION_RECORD_V1",
        "recovery_payload_consumed": True,
        "rerun_performed": False,
        "second_recovery_prohibited": True,
        "original_final_payload_release_events": 1,
        "recovery_payload_release_events": int(runtime_state.get("recovery_payload_release_events", 0)),
        "historical_total_payload_release_events": int(
            runtime_state.get("historical_total_payload_release_events", 0)
        ),
    }

    execution_identity = {
        "schema_version": "M-B10R1B_EXECUTION_IDENTITY_V1",
        "frozen_from": "datasets/mmwave/manifests/M-B10R1A_recovery_prefreeze/execution_freeze_identity.json",
        "harness_module_sha256": frozen.get("harness_module_sha256"),
        "preprocessing_contract_ids": frozen.get("preprocessing_contract_ids"),
        "result_limitation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "model_count": 3,
        "specs": [
            {
                "model_id": spec["model_id"],
                "role": spec["role"],
                "sha256": spec["sha256"],
                "preprocessing_contract_id": spec["preprocessing_contract_id"],
            }
            for spec in specs
        ],
    }

    selected = metrics_by_model[SELECTED_MODEL_ID]
    selected_result = {
        "model_id": SELECTED_MODEL_ID,
        "role": "SELECTED_NEW_REAL_DATA_CANDIDATE",
        "metrics": selected,
        "coverage": coverage[SELECTED_MODEL_ID],
        "result_limitation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "class_map": dict(CLASS_MAP),
    }
    baseline_results = {
        V01_MODEL_ID: {
            "model_id": V01_MODEL_ID,
            "metrics": metrics_by_model[V01_MODEL_ID],
            "coverage": coverage[V01_MODEL_ID],
        },
        V02_MODEL_ID: {
            "model_id": V02_MODEL_ID,
            "metrics": metrics_by_model[V02_MODEL_ID],
            "coverage": coverage[V02_MODEL_ID],
        },
    }

    summary = {
        "phase_id": "M-B10R1-B",
        "status": "RECOVERY_EXECUTED",
        "result_limitation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "ledger_row_count": len(ledger),
        "actual_total_tflite_invocations": evaluation.get("actual_total_tflite_invocations"),
        "recovery_payload_release_events": audit["recovery_payload_release_events"],
        "historical_total_payload_release_events": audit["historical_total_payload_release_events"],
        "rerun_performed": False,
        "selected_valid_count": coverage[SELECTED_MODEL_ID]["valid_count"],
        "acceptance_threshold": "FINAL_LOCKED_TEST_NUMERICAL_ACCEPTANCE_THRESHOLD_NOT_PREDEFINED",
        "candidate_unchanged": True,
        "no_tuning": True,
        "no_retraining": True,
        "no_recalibration": True,
        "stores_raw_phase_tensors": False,
    }

    run_env = {
        "phase_id": "M-B10R1-B",
        "generated_at": _utc_now(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "stores_raw_phase_tensors": False,
    }
    exceptions = {
        "phase_id": "M-B10R1-B",
        "status": "NO_EXECUTION_EXCEPTIONS",
        "recovery_executed": True,
    }

    atomic_write_json(out / "authorization_record.json", authorization)
    atomic_write_json(out / "execution_identity.json", execution_identity)
    atomic_write_json(out / "recovery_access_runtime_state.json", runtime_state)
    atomic_write_json(out / "one_time_recovery_access_audit.json", audit)
    atomic_write_json(out / "recovery_registry.json", registry)
    atomic_write_jsonl(out / "recovery_sample_predictions.jsonl", ledger)
    atomic_write_json(out / "model_evaluation_coverage.json", coverage)
    atomic_write_json(out / "metrics_by_model.json", metrics_by_model)
    atomic_write_json(out / "per_class_metrics.json", _per_class_by_model(metrics_by_model))
    atomic_write_json(out / "subject_level_metrics.json", subjects)
    atomic_write_json(out / "model_comparison.json", _model_comparison(metrics_by_model))
    atomic_write_json(out / "selected_candidate_recovery_result.json", selected_result)
    atomic_write_json(out / "historical_baseline_recovery_results.json", baseline_results)
    atomic_write_json(out / "selected_candidate_quantization_audit.json", saturation)
    atomic_write_json(out / "recovery_consumption_record.json", consumption)
    atomic_write_json(out / "run_environment.json", run_env)
    atomic_write_json(out / "exceptions.json", exceptions)
    atomic_write_json(out / "m_b10r1b_summary.json", summary)
    write_checksums(out)
    return summary


def persist_terminal_failure(
    out: Path,
    *,
    runtime_state: dict[str, Any],
    authorization: dict[str, Any] | None,
    exception: BaseException,
    failure_stage: str,
    ledger: list[dict[str, Any]] | None = None,
    coverage: dict[str, Any] | None = None,
    completed_inference_count: int | None = None,
) -> None:
    """After payload release, persist incomplete evidence. No retry. No fabricated metrics."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(ledger or [])
    invoke_count = completed_inference_count
    if invoke_count is None and coverage:
        invoke_count = sum(int(c.get("tflite_invoke_count", 0)) for c in coverage.values())
    if invoke_count is None:
        invoke_count = sum(1 for r in rows if not r.get("invalid") and r.get("predicted_class_index", -1) >= 0)

    audit = {
        "schema_version": "M-B10R1B_ONE_TIME_RECOVERY_ACCESS_AUDIT_V1",
        "status": "PARTIAL_INCOMPLETE",
        "historical_original_payload_release_events": int(
            runtime_state.get("original_final_payload_release_events", 1)
        ),
        "recovery_payload_release_events": int(runtime_state.get("recovery_payload_release_events", 0)),
        "historical_total_payload_release_events": int(
            runtime_state.get("historical_total_payload_release_events", 0)
        ),
        "recovery_accessor_invocations": int(runtime_state.get("recovery_accessor_invocations", 0)),
        "payload_consumed": True,
        "rerun_performed": False,
        "automatic_retry": False,
        "completed_inference_count": invoke_count,
        "failure_stage": failure_stage,
        "exception": str(exception),
        "result_limitation": RESULT_LIMITATION,
        "result_not_pristine": True,
    }
    consumption = {
        "schema_version": "M-B10R1B_RECOVERY_CONSUMPTION_RECORD_V1",
        "status": "PARTIAL_INCOMPLETE",
        "recovery_payload_consumed": True,
        "rerun_performed": False,
        "recovery_payload_release_events": audit["recovery_payload_release_events"],
        "historical_total_payload_release_events": audit["historical_total_payload_release_events"],
    }
    exceptions = {
        "phase_id": "M-B10R1-B",
        "status": "TERMINAL_FAILURE_AFTER_PAYLOAD_RELEASE_NO_RETRY",
        "failure_stage": failure_stage,
        "exception": str(exception),
        "metrics_fabricated": False,
    }
    summary = {
        "phase_id": "M-B10R1-B",
        "status": "PARTIAL_INCOMPLETE",
        "result_limitation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "ledger_row_count": len(rows),
        "completed_inference_count": invoke_count,
        "recovery_payload_release_events": audit["recovery_payload_release_events"],
        "historical_total_payload_release_events": audit["historical_total_payload_release_events"],
        "rerun_performed": False,
        "metrics_populated": False,
    }
    if authorization is not None:
        atomic_write_json(out / "authorization_record.json", authorization)
    atomic_write_json(out / "recovery_access_runtime_state.json", runtime_state)
    atomic_write_json(out / "one_time_recovery_access_audit.json", audit)
    atomic_write_json(out / "recovery_consumption_record.json", consumption)
    atomic_write_json(out / "exceptions.json", exceptions)
    atomic_write_json(out / "m_b10r1b_summary.json", summary)
    if rows:
        atomic_write_jsonl(out / "recovery_sample_predictions.jsonl", rows)
        atomic_write_json(
            out / "recovery_registry.json",
            {
                "status": "PARTIAL_INCOMPLETE",
                "ledger_row_count": len(rows),
                "not_completed_performance": True,
            },
        )
    if coverage is not None:
        atomic_write_json(out / "model_evaluation_coverage.json", coverage)
    write_checksums(out)
