#!/usr/bin/env python3
"""M-B10R1 recovery evaluation runner (pre-freeze + future authorized path).

Default / pre-access modes never release recovery payload.
``execute_authorized_recovery`` is irreversible and MUST NOT be invoked during
M-B10R1-A.

Evaluation after payload release is ``evaluate_recovery_payload`` so mock tests
can exercise the signal→preprocess→invoke path without LOCKED_TEST access.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.mmwave_m_b10r1_metrics import (  # noqa: E402
    CLASS_MAP,
    metric_bundle,
    saturation_audit_from_rows,
    subject_metrics,
)
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
    RecoveryReadiness,
)
from scripts.mmwave_m_b10r1_result_writer import (  # noqa: E402
    B_AUTHORIZATION_STATUS_GRANTED,
    B_OUT_DIR_REL,
    initialize_b_runtime_from_a,
    load_b_authorization_record,
    persist_recovery_results,
    persist_terminal_failure,
)

OUT_DIR_REL = Path("datasets/mmwave/manifests/M-B10R1A_recovery_prefreeze")
M_B10R0_DIR_REL = Path("datasets/mmwave/manifests/M-B10R0_holdout_policy_review")
M_B10A_DIR_REL = Path("datasets/mmwave/manifests/M-B10A_candidate_selection_setup")
M_B10B_DIR_REL = Path("datasets/mmwave/manifests/M-B10B_locked_test_final_evaluation")
EXECUTION_FREEZE_IDENTITY_NAME = "execution_freeze_identity.json"

SELECTED_CANDIDATE_ID = "M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120"
SELECTED_MODEL_ID = "M-B3_CONV1D_GAP_BASELINE_seed42_M-B6_STRICT_INT8"
SELECTED_PATH = (
    "models/mmwave/experiments/M-B6_stage_equivalence/"
    "M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120_int8.tflite"
)
SELECTED_SHA = "6dff6aaa72c79d76715d40cf7e32bb1e6cd9b2c2e3ac78eaf2fda737561430c5"
V01_PATH = "models/mmwave/mmwave_resp_int8_v0.1.0.tflite"
V01_SHA = "43cdd6f321c2b645232162233e098c2b65549388cbc9e680a17a0eccdc8f0158"
V02_PATH = "models/mmwave/mmwave_resp_int8_v0.2.0_candidate.tflite"
V02_SHA = "85c023d3eefca13ecbb72a841974e53a56d5f4173920645d46df49a9088452ff"
EXECUTOR_PATH = "scripts/mmwave_m_b10b_baseline_preprocessing.py"
EXECUTOR_SHA = "8ca87f457d0a151cffa2da23ae9ab9d87764b144fa826b91444776f3dc58ec4f"
META_V01_PATH = "models/mmwave/sensor_stats_metadata_v0.1.0.json"
META_V01_SHA = "a875a8369ff7adf5477cec009b99c0c6d0fbb8b0e60e5b0b07a551f3780d2e37"
META_V02_PATH = "models/mmwave/mmwave_resp_int8_v0.2.0_candidate_metadata.json"
META_V02_SHA = "36039a6cffbc57162dbb4c720034da6dcfa49ef2f2d33238bee65a62aa133127"
M_B10A_CONTRACT_SHA = "ba6429ecfe685de1807ec85b55e697ee12e24138e6b96e94715b0a1a6b19e0f7"

# Exact M-B10A locked_test_evaluation_contract preprocessing contract IDs.
SELECTED_PREPROCESSING_CONTRACT_ID = "M-B10B_SELECTED_REAL_CANDIDATE_BPF_ZSCORE_V1"
V01_PREPROCESSING_CONTRACT_ID = "M-B10B_HISTORICAL_V0_1_COMPATIBILITY_PREPROCESSING_V1"
V02_PREPROCESSING_CONTRACT_ID = "M-B10B_SYNTHETIC_V0_2_EXTERNAL_COMPATIBILITY_PREPROCESSING_V1"

FORBIDDEN_MODEL_IDS = {
    "M-B3_CONV1D_GAP_BASELINE_seed43_M-B6_STRICT_INT8",
    "M-B3_CONV1D_GAP_BASELINE_seed44_M-B6_STRICT_INT8",
}

HARNESS_MODULE_RELS = (
    "scripts/mmwave_m_b10r1_recovery_access.py",
    "scripts/mmwave_m_b10r1_recovery_eval.py",
    "scripts/mmwave_m_b10r1_metrics.py",
    "scripts/run_mmwave_m_b10r1.py",
    "scripts/mmwave_m_b10b_baseline_preprocessing.py",
    "scripts/mmwave_m_b10r1_result_writer.py",
    "scripts/validate_mmwave_m_b10r1b.py",
)

# Frozen SHA keys → repository-relative paths (authoritative freeze identity).
FROZEN_ARTIFACT_BINDINGS = {
    "policy_decision_sha256": "datasets/mmwave/manifests/M-B10R0_holdout_policy_review/policy_decision.json",
    "reuse_exception_gate_results_sha256": (
        "datasets/mmwave/manifests/M-B10R0_holdout_policy_review/reuse_exception_gate_results.json"
    ),
    "proposed_recovery_evaluation_contract_sha256": (
        "datasets/mmwave/manifests/M-B10R0_holdout_policy_review/"
        "proposed_recovery_evaluation_contract.json"
    ),
    "future_recovery_access_requirements_sha256": (
        "datasets/mmwave/manifests/M-B10R0_holdout_policy_review/"
        "future_recovery_access_requirements.json"
    ),
    "m_b10r0_summary_sha256": "datasets/mmwave/manifests/M-B10R0_holdout_policy_review/m_b10r0_summary.json",
    "m_b10a_metric_contract_sha256": (
        "datasets/mmwave/manifests/M-B10A_candidate_selection_setup/locked_test_evaluation_contract.json"
    ),
    "selected_model_sha256": SELECTED_PATH,
    "baseline_v01_sha256": V01_PATH,
    "baseline_v02_sha256": V02_PATH,
    "executor_sha256": EXECUTOR_PATH,
    "metadata_v01_sha256": META_V01_PATH,
    "metadata_v02_sha256": META_V02_PATH,
}

LEDGER_ROW_FIELDS = (
    "window_id",
    "subject_id",
    "recording_id",
    "true_class",
    "true_class_index",
    "model_role",
    "model_id",
    "model_sha256",
    "preprocessing_contract_id",
    "model_input_tensor_sha256",
    "raw_output_int8",
    "dequantized_output",
    "predicted_class_index",
    "predicted_class",
    "confidence",
    "input_saturation_count",
    "input_saturation_ratio",
    "fallback_used",
    "invalid",
    "error",
    "result_limitation",
    "result_not_pristine",
)

COVERAGE_TRACKING_FIELDS = (
    "evaluation_rows_attempted",
    "preprocessing_success_count",
    "tflite_invoke_count",
    "tflite_invoke_attempts",
    "invalid_preprocessing_count",
    "invalid_inference_count",
    "planned_count",
    "valid_count",
    "invalid_count",
)

LEDGER_SCHEMA_STATUS = "NOT_EXECUTED"
RESULT_SCHEMA_STATUS = "NOT_POPULATED"


class MB10R1EvalError(Exception):
    """Recovery evaluation harness error."""

    def __init__(
        self,
        message: str,
        *,
        ledger: list[dict[str, Any]] | None = None,
        coverage: dict[str, Any] | None = None,
        completed_inference_count: int | None = None,
    ) -> None:
        super().__init__(message)
        self.ledger = ledger
        self.coverage = coverage
        self.completed_inference_count = completed_inference_count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_head(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
    except OSError:
        pass
    return "pre_freeze_head_pending"


def frozen_model_specs() -> list[dict[str, Any]]:
    return [
        {
            "role": "SELECTED_NEW_REAL_DATA_CANDIDATE",
            "candidate_id": SELECTED_CANDIDATE_ID,
            "model_id": SELECTED_MODEL_ID,
            "path": SELECTED_PATH,
            "sha256": SELECTED_SHA,
            "seed": 42,
            "bytes": 22080,
            "preprocessing": "BPF_ZSCORE",
            "preprocessing_profile": "M-B1_D0_B1_Z1",
            "preprocessing_contract_id": SELECTED_PREPROCESSING_CONTRACT_ID,
            "calibration": "M-B5_CAL_CLASS_BALANCED_120",
            "baseline_id": None,
            "class_map": dict(CLASS_MAP),
        },
        {
            "role": "HISTORICAL_MODEL_COMPATIBILITY_BASELINE",
            "model_id": "mmwave_resp_int8",
            "path": V01_PATH,
            "sha256": V01_SHA,
            "bytes": 466616,
            "baseline_id": "mmwave_resp_int8",
            "preprocessing_contract_id": V01_PREPROCESSING_CONTRACT_ID,
            "executor_path": EXECUTOR_PATH,
            "executor_sha256": EXECUTOR_SHA,
            "metadata_path": META_V01_PATH,
            "metadata_sha256": META_V01_SHA,
            "interpretation": "HISTORICAL_MODEL_COMPATIBILITY_BENCHMARK",
            "class_map": dict(CLASS_MAP),
        },
        {
            "role": "SYNTHETIC_TRAINED_EXTERNAL_COMPATIBILITY_BASELINE",
            "model_id": "mmwave_resp_int8_v0.2.0_candidate",
            "path": V02_PATH,
            "sha256": V02_SHA,
            "bytes": 22472,
            "baseline_id": "mmwave_resp_int8_v0.2.0_candidate",
            "preprocessing_contract_id": V02_PREPROCESSING_CONTRACT_ID,
            "executor_path": EXECUTOR_PATH,
            "executor_sha256": EXECUTOR_SHA,
            "metadata_path": META_V02_PATH,
            "metadata_sha256": META_V02_SHA,
            "interpretation": "SYNTHETIC_TRAINED_EXTERNAL_COMPATIBILITY_BENCHMARK",
            "class_map": dict(CLASS_MAP),
        },
    ]


def validate_frozen_recovery_models(root: Path) -> list[dict[str, Any]]:
    """Exact 3 models; reject seed43/44/fourth; exact preprocessing contract IDs."""
    specs = frozen_model_specs()
    if len(specs) != 3:
        raise MB10R1EvalError("MODEL_COUNT_MUST_BE_THREE")
    ids = [s["model_id"] for s in specs]
    if len(set(ids)) != 3:
        raise MB10R1EvalError("DUPLICATE_MODEL_IDS")
    for forbidden in FORBIDDEN_MODEL_IDS:
        if forbidden in ids:
            raise MB10R1EvalError(f"FORBIDDEN_MODEL:{forbidden}")
    serialized = json.dumps(specs, sort_keys=True).lower()
    if "seed43" in serialized or "seed44" in serialized:
        raise MB10R1EvalError("FORBIDDEN_SEED_IN_MODEL_SET")
    expected_contracts = {
        SELECTED_MODEL_ID: SELECTED_PREPROCESSING_CONTRACT_ID,
        "mmwave_resp_int8": V01_PREPROCESSING_CONTRACT_ID,
        "mmwave_resp_int8_v0.2.0_candidate": V02_PREPROCESSING_CONTRACT_ID,
    }
    for spec in specs:
        expected = expected_contracts[spec["model_id"]]
        if spec.get("preprocessing_contract_id") != expected:
            raise MB10R1EvalError(f"PREPROCESSING_CONTRACT_ID_MISMATCH:{spec['model_id']}")
        path = root / spec["path"]
        if not path.is_file():
            raise MB10R1EvalError(f"MODEL_MISSING:{spec['path']}")
        live = sha256_file(path)
        if live != spec["sha256"]:
            raise MB10R1EvalError(f"MODEL_SHA_MISMATCH:{spec['model_id']}")
        if path.stat().st_size != int(spec["bytes"]):
            raise MB10R1EvalError(f"MODEL_BYTES_MISMATCH:{spec['model_id']}")
        if spec.get("executor_path"):
            executor = root / spec["executor_path"]
            if sha256_file(executor) != spec["executor_sha256"]:
                raise MB10R1EvalError(f"EXECUTOR_SHA_MISMATCH:{spec['model_id']}")
        if spec.get("metadata_path"):
            meta = root / spec["metadata_path"]
            if sha256_file(meta) != spec["metadata_sha256"]:
                raise MB10R1EvalError(f"METADATA_SHA_MISMATCH:{spec['model_id']}")
    return specs


def build_bound_contract_identity(root: Path) -> dict[str, Any]:
    """SHA-bind M-B10R0 + M-B10A + model/baseline identities (prefreeze snapshot helper)."""
    validate_frozen_recovery_models(root)
    r0 = root / M_B10R0_DIR_REL
    return {
        "schema_version": "M-B10R1_BOUND_CONTRACT_IDENTITY_V1",
        "include_ambiguous": False,
        "positional_truncation": False,
        "eligibility_rule": (
            "split==LOCKED_TEST AND assignment_status!=AMBIGUOUS "
            "(A6 locked_test_evaluation_eligible semantics via PhaseBAccessGuard._get_split_dataset)"
        ),
        "expected_eligible_windows": EXPECTED_ELIGIBLE,
        "expected_subjects": EXPECTED_SUBJECTS,
        "expected_structural_windows": EXPECTED_STRUCTURAL,
        "expected_ambiguous_windows": EXPECTED_AMBIGUOUS,
        "expected_model_inference_count": EXPECTED_INFERENCES,
        "result_limitation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "policy_decision_sha256": sha256_file(r0 / "policy_decision.json"),
        "reuse_exception_gate_results_sha256": sha256_file(r0 / "reuse_exception_gate_results.json"),
        "proposed_recovery_evaluation_contract_sha256": sha256_file(
            r0 / "proposed_recovery_evaluation_contract.json"
        ),
        "future_recovery_access_requirements_sha256": sha256_file(
            r0 / "future_recovery_access_requirements.json"
        ),
        "m_b10r0_summary_sha256": sha256_file(r0 / "m_b10r0_summary.json"),
        "m_b10a_metric_contract_sha256": sha256_file(
            root / M_B10A_DIR_REL / "locked_test_evaluation_contract.json"
        ),
        "selected_model_sha256": SELECTED_SHA,
        "baseline_v01_sha256": V01_SHA,
        "baseline_v02_sha256": V02_SHA,
        "executor_sha256": EXECUTOR_SHA,
        "metadata_v01_sha256": META_V01_SHA,
        "metadata_v02_sha256": META_V02_SHA,
        "selected_candidate_id": SELECTED_CANDIDATE_ID,
        "selected_model_id": SELECTED_MODEL_ID,
        "model_count": 3,
        "preprocessing_contract_ids": {
            SELECTED_MODEL_ID: SELECTED_PREPROCESSING_CONTRACT_ID,
            "mmwave_resp_int8": V01_PREPROCESSING_CONTRACT_ID,
            "mmwave_resp_int8_v0.2.0_candidate": V02_PREPROCESSING_CONTRACT_ID,
        },
        "recovery_authorization_token_id": "M_B10R1_LIMITED_REUSE_RECOVERY_AUTHORIZATION_V1",
        "original_final_token_rejected": ORIGINAL_FINAL_TOKEN,
    }


def build_execution_freeze_identity(root: Path) -> dict[str, Any]:
    """Authoritative freeze snapshot for prefreeze generation (not execute-time truth rebuild)."""
    bound = build_bound_contract_identity(root)
    harness_sha = {rel: sha256_file(root / rel) for rel in HARNESS_MODULE_RELS}
    artifact_sha = {key: sha256_file(root / rel) for key, rel in FROZEN_ARTIFACT_BINDINGS.items()}
    return {
        "schema_version": "M-B10R1A_EXECUTION_FREEZE_IDENTITY_V1",
        "phase_id": "M-B10R1-A",
        "pre_freeze_head": _git_head(root),
        "pre_freeze_head_note": (
            "Informational HEAD at prefreeze generation. M-B10R1-B must bind the "
            "reviewed PR head that contains these harness module SHAs."
        ),
        "bound_contract_identity": bound,
        "artifact_sha256": artifact_sha,
        "harness_module_sha256": harness_sha,
        "preprocessing_contract_ids": {
            SELECTED_MODEL_ID: SELECTED_PREPROCESSING_CONTRACT_ID,
            "mmwave_resp_int8": V01_PREPROCESSING_CONTRACT_ID,
            "mmwave_resp_int8_v0.2.0_candidate": V02_PREPROCESSING_CONTRACT_ID,
        },
        "m_b10r0_proposed_contract_sha256": artifact_sha[
            "proposed_recovery_evaluation_contract_sha256"
        ],
        "m_b10a_metric_contract_sha256": artifact_sha["m_b10a_metric_contract_sha256"],
        "selected_model_sha256": SELECTED_SHA,
        "baseline_v01_sha256": V01_SHA,
        "baseline_v02_sha256": V02_SHA,
        "executor_sha256": EXECUTOR_SHA,
        "metadata_v01_sha256": META_V01_SHA,
        "metadata_v02_sha256": META_V02_SHA,
        "result_limitation": RESULT_LIMITATION,
        "expected_eligible_windows": EXPECTED_ELIGIBLE,
        "expected_model_inference_count": EXPECTED_INFERENCES,
    }


def load_frozen_execution_identity(root: Path) -> dict[str, Any]:
    """Load authoritative freeze identity; never rebuild as sole execute-time truth."""
    path = root / OUT_DIR_REL / EXECUTION_FREEZE_IDENTITY_NAME
    if not path.is_file():
        raise MB10R1EvalError("EXECUTION_FREEZE_IDENTITY_MISSING")
    frozen = load_json(path)
    if not isinstance(frozen, dict):
        raise MB10R1EvalError("EXECUTION_FREEZE_IDENTITY_INVALID")
    if frozen.get("schema_version") != "M-B10R1A_EXECUTION_FREEZE_IDENTITY_V1":
        raise MB10R1EvalError("EXECUTION_FREEZE_IDENTITY_SCHEMA_MISMATCH")
    return frozen


def verify_freeze_identity_in_checksums(root: Path) -> str:
    """Require execution_freeze_identity.json checksum entry matches live file bytes."""
    out = root / OUT_DIR_REL
    checksum_path = out / "checksums.sha256"
    if not checksum_path.is_file():
        raise MB10R1EvalError("CHECKSUMS_MISSING_FOR_FREEZE_IDENTITY")
    mapped: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise MB10R1EvalError(f"CHECKSUM_LINE_INVALID:{line}")
        mapped[parts[1]] = parts[0]
    if EXECUTION_FREEZE_IDENTITY_NAME not in mapped:
        raise MB10R1EvalError("EXECUTION_FREEZE_IDENTITY_NOT_IN_CHECKSUMS")
    live = sha256_file(out / EXECUTION_FREEZE_IDENTITY_NAME)
    if live != mapped[EXECUTION_FREEZE_IDENTITY_NAME]:
        raise MB10R1EvalError("EXECUTION_FREEZE_IDENTITY_CHECKSUM_MISMATCH")
    return live


def verify_live_against_frozen(root: Path, frozen: dict[str, Any]) -> None:
    """Compare live file SHAs against frozen recorded values. Fail before payload release."""
    artifact_sha = frozen.get("artifact_sha256") or {}
    for key, rel in FROZEN_ARTIFACT_BINDINGS.items():
        expected = artifact_sha.get(key) or frozen.get(key)
        if expected is None:
            raise MB10R1EvalError(f"FROZEN_ARTIFACT_SHA_MISSING:{key}")
        path = root / rel
        if not path.is_file():
            raise MB10R1EvalError(f"FROZEN_ARTIFACT_FILE_MISSING:{rel}")
        live = sha256_file(path)
        if live != expected:
            raise MB10R1EvalError(f"FROZEN_LIVE_MISMATCH:{key}")

    harness = frozen.get("harness_module_sha256") or {}
    for rel in HARNESS_MODULE_RELS:
        expected = harness.get(rel)
        if expected is None:
            raise MB10R1EvalError(f"FROZEN_HARNESS_SHA_MISSING:{rel}")
        live = sha256_file(root / rel)
        if live != expected:
            raise MB10R1EvalError(f"FROZEN_HARNESS_LIVE_MISMATCH:{rel}")

    contracts = frozen.get("preprocessing_contract_ids") or {}
    expected_contracts = {
        SELECTED_MODEL_ID: SELECTED_PREPROCESSING_CONTRACT_ID,
        "mmwave_resp_int8": V01_PREPROCESSING_CONTRACT_ID,
        "mmwave_resp_int8_v0.2.0_candidate": V02_PREPROCESSING_CONTRACT_ID,
    }
    for model_id, contract_id in expected_contracts.items():
        if contracts.get(model_id) != contract_id:
            raise MB10R1EvalError(f"FROZEN_PREPROCESSING_CONTRACT_MISMATCH:{model_id}")

    if frozen.get("m_b10a_metric_contract_sha256") != sha256_file(
        root / FROZEN_ARTIFACT_BINDINGS["m_b10a_metric_contract_sha256"]
    ):
        raise MB10R1EvalError("FROZEN_M_B10A_METRIC_LIVE_MISMATCH")
    if frozen.get("m_b10r0_proposed_contract_sha256") != sha256_file(
        root / FROZEN_ARTIFACT_BINDINGS["proposed_recovery_evaluation_contract_sha256"]
    ):
        raise MB10R1EvalError("FROZEN_M_B10R0_PROPOSED_LIVE_MISMATCH")


def require_frozen_bound_contract(frozen: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed: never rebuild bound contract identity at execute time."""
    bound = frozen.get("bound_contract_identity")
    if not isinstance(bound, dict) or not bound:
        raise MB10R1EvalError("FROZEN_BOUND_CONTRACT_IDENTITY_MISSING_STOP_BEFORE_PAYLOAD")
    return bound


def authorize_pre_access_freeze_binding(root: Path) -> dict[str, Any]:
    """Load frozen identity, verify checksum coverage, compare live→frozen. No payload."""
    frozen = load_frozen_execution_identity(root)
    verify_freeze_identity_in_checksums(root)
    verify_live_against_frozen(root, frozen)
    require_frozen_bound_contract(frozen)
    return frozen


def build_preaccess_readiness(root: Path, *, validator_pass: bool = False) -> dict[str, Any]:
    """All authorization flags FALSE during M-B10R1-A."""
    del root  # identity is global; root reserved for future path checks
    return {
        "schema_version": "M-B10R1A_RECOVERY_ACCESS_READINESS_V1",
        "phase_id": "M-B10R1-A",
        "mechanism_implemented": True,
        "runner_implemented": True,
        "pre_access_validator_pass": bool(validator_pass),
        "independent_review_required": True,
        "recovery_execution_authorized": False,
        "recovery_payload_release_authorized": False,
        "M-B10R1B_started": False,
        "new_recovery_accessor_invocations": 0,
        "new_payload_release_events": 0,
        "authorization_token_supplied_during_m_b10r1a": False,
        "notes": (
            "Mechanism and runner are implemented for future M-B10R1-B. "
            "No recovery execution authorization is granted by M-B10R1-A."
        ),
    }


def future_ledger_schema() -> dict[str, Any]:
    return {
        "schema_version": "M-B10R1_FUTURE_LEDGER_SCHEMA_V1",
        "status": LEDGER_SCHEMA_STATUS,
        "row_unit": "eligible_sample_x_model",
        "expected_rows": EXPECTED_INFERENCES,
        "expected_eligible_windows": EXPECTED_ELIGIBLE,
        "expected_models": 3,
        "required_fields": list(LEDGER_ROW_FIELDS),
        "coverage_tracking_fields": list(COVERAGE_TRACKING_FIELDS),
        "stores_raw_phase_tensors": False,
        "result_limitation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "populated": False,
        "population_note": "NOT_EXECUTED — ledger rows are not populated during M-B10R1-A",
        "inference_count_note": (
            "Never treat len(ledger)==225 as proof of 225 TFLite inferences; "
            "use actual_total_tflite_invocations / per-model tflite_invoke_count."
        ),
    }


def future_result_schema() -> dict[str, Any]:
    return {
        "schema_version": "M-B10R1_FUTURE_RESULT_SCHEMA_V1",
        "status": RESULT_SCHEMA_STATUS,
        "required_result_designation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "original_pristine_final_access_consumed": True,
        "original_model_inferences": 0,
        "reuse_exception_reviewed": True,
        "forbidden_scientific_wording": [
            "PRISTINE_REAL_SUBJECT_FINAL_TEST",
            "PRISTINE_ONE_TIME_LOCKED_TEST",
            "PRISTINE_LOCKED_TEST",
            "FIRST_LOCKED_TEST_EVALUATION",
            "LOCKED_TEST_NOT_CONSUMED",
            "NO_INFORMATION_EXPOSURE",
            "ORIGINAL_ACCESS_UNUSED",
        ],
        "allowed_scientific_wording": "OFFLINE_REAL_DATA_RECOVERY_EVALUATION_WITH_HOLDOUT_REUSE_LIMITATION",
        "acceptance_threshold": "FINAL_LOCKED_TEST_NUMERICAL_ACCEPTANCE_THRESHOLD_NOT_PREDEFINED",
        "metrics_populated": False,
        "predictions_populated": False,
        "coverage_requirements": {
            "selected_must_have_valid_predictions": EXPECTED_ELIGIBLE,
            "fail_closed_on_selected_incomplete": True,
            "metrics_from_valid_rows_only": True,
            "never_claim_planned_as_evaluated_when_empty": True,
        },
        "note": "NOT_POPULATED — no recovery metrics or predictions during M-B10R1-A",
    }


def _mb10b_spec_for_preprocess(spec: dict[str, Any], inspected: dict[str, Any] | None) -> dict[str, Any]:
    """Build M-B10B preprocess_for_spec / TFLiteRunner spec with exact contract IDs."""
    role = spec["role"]
    mb10b_spec: dict[str, Any] = {
        "role": role,
        "model_id": spec["model_id"],
        "path": spec["path"],
        "baseline_id": spec.get("baseline_id"),
        "preprocessing_contract_id": spec["preprocessing_contract_id"],
        "sha256": spec["sha256"],
    }
    if inspected is not None:
        mb10b_spec["inspected"] = inspected
    return mb10b_spec


def evaluate_recovery_payload(
    root: Path,
    payload: dict[str, Any],
    specs: list[dict[str, Any]],
    *,
    runners: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a recovery payload (real or mock) without calling the recovery accessor.

    ``window`` is metadata only; ``signal`` (numeric ndarray) is the preprocess input.
    """
    from scripts.mmwave_m_b10b_final_eval import TFLiteRunner, inspect_tflite, preprocess_for_spec

    windows = payload.get("windows") or []
    signals = payload.get("signals")
    if signals is None:
        raise MB10R1EvalError("PAYLOAD_SIGNALS_MISSING")
    signals_arr = np.asarray(signals)
    if len(windows) != EXPECTED_ELIGIBLE:
        raise MB10R1EvalError(f"PAYLOAD_WINDOW_COUNT:{len(windows)}")
    if signals_arr.shape != (EXPECTED_ELIGIBLE, 300):
        raise MB10R1EvalError(f"PAYLOAD_SIGNAL_SHAPE:{signals_arr.shape}")

    # Materialize runners / inspected (selected path needs quantization params).
    local_runners: dict[str, Any] = {}
    mb10b_specs: dict[str, dict[str, Any]] = {}
    for spec in specs:
        mid = spec["model_id"]
        inspected = None
        if runners is None or mid not in (runners or {}):
            # Real path: inspect for selected quantization; TFLiteRunner created below.
            if spec["role"] == "SELECTED_NEW_REAL_DATA_CANDIDATE":
                inspected = inspect_tflite(root, spec["path"])
            mb10b_specs[mid] = _mb10b_spec_for_preprocess(spec, inspected)
            local_runners[mid] = TFLiteRunner(root, mb10b_specs[mid])
        else:
            # Injected runner (tests): still inspect selected model for real preprocess.
            if spec["role"] == "SELECTED_NEW_REAL_DATA_CANDIDATE":
                inspected = inspect_tflite(root, spec["path"])
            mb10b_specs[mid] = _mb10b_spec_for_preprocess(spec, inspected)
            local_runners[mid] = runners[mid]

    coverage: dict[str, dict[str, int]] = {
        spec["model_id"]: {
            "evaluation_rows_attempted": 0,
            "preprocessing_success_count": 0,
            "tflite_invoke_count": 0,
            "tflite_invoke_attempts": 0,
            "invalid_preprocessing_count": 0,
            "invalid_inference_count": 0,
            "planned_count": EXPECTED_ELIGIBLE,
            "valid_count": 0,
            "invalid_count": 0,
        }
        for spec in specs
    }

    ledger: list[dict[str, Any]] = []
    for spec in specs:
        mid = spec["model_id"]
        mb10b_spec = mb10b_specs[mid]
        runner = local_runners[mid]
        is_selected = spec["role"] == "SELECTED_NEW_REAL_DATA_CANDIDATE"
        for window, signal in zip(windows, signals_arr):
            cov = coverage[mid]
            cov["evaluation_rows_attempted"] += 1
            row: dict[str, Any] = {
                "window_id": window.get("window_id"),
                "subject_id": window.get("subject_id"),
                "recording_id": window.get("recording_id"),
                "true_class": window.get("safenest_label"),
                "true_class_index": int(window.get("safenest_label_id", -1)),
                "model_role": spec["role"],
                "model_id": mid,
                "model_sha256": spec["sha256"],
                "preprocessing_contract_id": spec["preprocessing_contract_id"],
                "fallback_used": False,
                "invalid": False,
                "error": None,
                "result_limitation": RESULT_LIMITATION,
                "result_not_pristine": True,
            }
            try:
                # CRITICAL: pass numeric signal ndarray — never window metadata dict.
                prepared = preprocess_for_spec(signal, mb10b_spec)
                cov["preprocessing_success_count"] += 1
            except Exception as exc:
                cov["invalid_preprocessing_count"] += 1
                cov["invalid_count"] += 1
                row["invalid"] = True
                row["error"] = f"PREPROCESS:{exc}"
                row["predicted_class_index"] = -1
                row["predicted_class"] = None
                row["confidence"] = None
                row["input_saturation_count"] = 0
                row["input_saturation_ratio"] = 0.0
                ledger.append(row)
                if is_selected:
                    # Selected failures are terminal after payload release (no retry).
                    # Continue loop to finish ledger accounting, then raise below.
                    continue
                continue

            try:
                cov["tflite_invoke_attempts"] += 1
                out = runner.invoke(prepared["input_int8"])
                cov["tflite_invoke_count"] += 1
                dequant = out["dequantized_output"]
                if len(dequant) != 3 or not all(np.isfinite(dequant)):
                    raise MB10R1EvalError(f"NONFINITE_OR_BAD_OUTPUT:{mid}")
                row.update(
                    {
                        "model_input_tensor_sha256": hashlib.sha256(
                            prepared["input_int8"].tobytes()
                        ).hexdigest(),
                        "raw_output_int8": out["raw_output_int8"],
                        "dequantized_output": out["dequantized_output"],
                        "predicted_class_index": out["predicted_class_index"],
                        "predicted_class": out["predicted_class"],
                        "confidence": out["confidence"],
                        "input_saturation_count": prepared.get("input_saturation_count", 0),
                        "input_saturation_ratio": prepared.get("input_saturation_ratio", 0.0),
                    }
                )
                cov["valid_count"] += 1
            except Exception as exc:
                cov["invalid_inference_count"] += 1
                cov["invalid_count"] += 1
                row["invalid"] = True
                row["error"] = f"INFER:{exc}"
                row["predicted_class_index"] = -1
                row["predicted_class"] = None
                row["confidence"] = None
                row["input_saturation_count"] = prepared.get("input_saturation_count", 0)
                row["input_saturation_ratio"] = prepared.get("input_saturation_ratio", 0.0)
            ledger.append(row)

    if len(ledger) != EXPECTED_INFERENCES:
        raise MB10R1EvalError(
            f"INFERENCE_COUNT_MISMATCH:{len(ledger)}",
            ledger=ledger,
            coverage=coverage,
            completed_inference_count=sum(c["tflite_invoke_count"] for c in coverage.values()),
        )

    # Prefer runner.invocations attribute when present (actual invoke calls).
    runner_invocation_total = 0
    runner_invocation_by_model: dict[str, int] = {}
    for mid, runner in local_runners.items():
        inv = int(getattr(runner, "invocations", coverage[mid]["tflite_invoke_count"]))
        runner_invocation_by_model[mid] = inv
        runner_invocation_total += inv

    explicit_total = sum(c["tflite_invoke_count"] for c in coverage.values())
    actual_total_tflite_invocations = runner_invocation_total
    if runner_invocation_total != explicit_total:
        # Prefer runner attribute when it tracks calls; still record both.
        actual_total_tflite_invocations = runner_invocation_total

    # Selected must have 75 valid predictions for complete RECOVERY_EXECUTED.
    selected_cov = coverage[SELECTED_MODEL_ID]
    selected_invalid = selected_cov["invalid_count"]
    if selected_invalid != 0 or selected_cov["valid_count"] != EXPECTED_ELIGIBLE:
        raise MB10R1EvalError(
            "SELECTED_INCOMPLETE_COVERAGE:"
            f"valid={selected_cov['valid_count']};invalid={selected_invalid}",
            ledger=ledger,
            coverage=coverage,
            completed_inference_count=actual_total_tflite_invocations,
        )
    if selected_cov["tflite_invoke_count"] != EXPECTED_ELIGIBLE:
        raise MB10R1EvalError(
            f"SELECTED_TFLITE_INVOKE_MISMATCH:{selected_cov['tflite_invoke_count']}",
            ledger=ledger,
            coverage=coverage,
            completed_inference_count=actual_total_tflite_invocations,
        )

    metrics_by_model: dict[str, Any] = {}
    subject_by_model: dict[str, Any] = {}
    try:
        for spec in specs:
            mid = spec["model_id"]
            model_rows = [r for r in ledger if r["model_id"] == mid]
            valid_rows = [r for r in model_rows if not r.get("invalid")]
            labels = [int(r["true_class_index"]) for r in valid_rows]
            preds = [int(r["predicted_class_index"]) for r in valid_rows]
            # Metrics only from actual valid prediction rows; denominators explicit.
            # Never metric_bundle([], [], evaluated_sample_count=75).
            bundle = metric_bundle(labels, preds, evaluated_sample_count=len(labels))
            bundle["planned_count"] = EXPECTED_ELIGIBLE
            bundle["valid_count"] = len(valid_rows)
            bundle["invalid_count"] = len(model_rows) - len(valid_rows)
            bundle["coverage"] = dict(coverage[mid])
            metrics_by_model[mid] = bundle
            subject_by_model[mid] = subject_metrics(valid_rows)

        seed42_rows = [r for r in ledger if r["model_id"] == SELECTED_MODEL_ID and not r.get("invalid")]
        return {
            "status": "RECOVERY_EXECUTED",
            "result_limitation": RESULT_LIMITATION,
            "result_not_pristine": True,
            "ledger_row_count": len(ledger),
            "expected_inferences": EXPECTED_INFERENCES,
            "actual_total_tflite_invocations": actual_total_tflite_invocations,
            "runner_invocations_by_model": runner_invocation_by_model,
            "coverage_by_model": coverage,
            "metrics_by_model": metrics_by_model,
            "subject_metrics_by_model": subject_by_model,
            "saturation_audit_seed42": saturation_audit_from_rows(seed42_rows),
            "ledger": ledger,
            "acceptance_threshold": "FINAL_LOCKED_TEST_NUMERICAL_ACCEPTANCE_THRESHOLD_NOT_PREDEFINED",
            "note": "No performance threshold gating; no retry; no post-result model branching.",
        }
    except MB10R1EvalError:
        raise
    except Exception as exc:
        raise MB10R1EvalError(
            f"EVALUATION_FAILED:{exc}",
            ledger=ledger,
            coverage=coverage,
            completed_inference_count=actual_total_tflite_invocations,
        ) from exc


def run_validation_smoke(root: Path, *, attempt_tflite: bool = False) -> dict[str, Any]:
    """VALIDATION-only smoke; never LOCKED_TEST. Labeled mock/VALIDATION inference."""
    from scripts.mmwave_phase_b_access import PhaseBAccessGuard

    specs = validate_frozen_recovery_models(root)
    guard = PhaseBAccessGuard(root_dir=root)
    validation = guard.get_validation_data(include_ambiguous=False)
    selection = guard.get_model_selection_dataset("VALIDATION", include_ambiguous=False)
    if selection["total_count"] != validation["total_count"]:
        raise MB10R1EvalError("VALIDATION_ACCESSOR_COUNT_MISMATCH")

    probes: list[dict[str, Any]] = [
        {
            "split": "VALIDATION",
            "label": "VALIDATION_ACCESSOR_PROBE",
            "status": "OK",
            "sample_count": int(validation["total_count"]),
            "model_count_frozen": len(specs),
            "locked_test_accessed": False,
        }
    ]
    inference_count = 0
    if attempt_tflite:
        try:
            from scripts.mmwave_m_b10b_final_eval import TFLiteRunner

            # Synthetic int8 probe — VALIDATION wiring only; not recovery.
            dummy = np.zeros((1, 300, 1), dtype=np.int8)
            for spec in specs:
                mb10b_spec = {
                    "role": "SELECTED_NEW_REAL_DATA_CANDIDATE",
                    "model_id": spec["model_id"],
                    "path": spec["path"],
                    "baseline_id": spec.get("baseline_id"),
                    "preprocessing_contract_id": spec["preprocessing_contract_id"],
                }
                runner = TFLiteRunner(root, mb10b_spec)
                out = runner.invoke(dummy)
                inference_count += 1
                probes.append(
                    {
                        "split": "VALIDATION_WIRING",
                        "label": "MOCK_SYNTHETIC_INT8_PROBE",
                        "model_id": spec["model_id"],
                        "predicted_class": out["predicted_class"],
                        "locked_test_accessed": False,
                        "status": "OK",
                    }
                )
        except Exception as exc:
            probes.append(
                {
                    "split": "VALIDATION",
                    "label": "TFLITE_SMOKE_SKIPPED",
                    "status": f"SKIPPED:{exc}",
                    "locked_test_accessed": False,
                }
            )
    return {
        "status": "VALIDATION_SMOKE_COMPLETE",
        "split": "VALIDATION",
        "locked_test_accessed": False,
        "recovery_accessor_invoked": False,
        "validation_sample_count": int(validation["total_count"]),
        "validation_inferences_attempted_or_completed": inference_count,
        "probes": probes,
        "note": "MOCK/VALIDATION only — not recovery LOCKED_TEST inference",
    }


def _assert_a_readiness_historically_false(root: Path) -> dict[str, Any]:
    """A readiness flags describe M-B10R1-A history and must remain false forever."""
    readiness_path = root / OUT_DIR_REL / "recovery_access_readiness.json"
    if not readiness_path.is_file():
        raise MB10R1EvalError("READINESS_MANIFEST_MISSING")
    readiness_doc = load_json(readiness_path)
    if readiness_doc.get("recovery_execution_authorized") is not False:
        raise MB10R1EvalError("A_READINESS_MUST_REMAIN_HISTORICALLY_FALSE")
    if readiness_doc.get("recovery_payload_release_authorized") is not False:
        raise MB10R1EvalError("A_PAYLOAD_AUTH_MUST_REMAIN_HISTORICALLY_FALSE")
    return readiness_doc


def _assert_b_authorization_granted(overlay: dict[str, Any], freeze_sha: str) -> None:
    """B-side overlay is the only grant. A flags are never flipped to true."""
    if overlay.get("approval") is not True:
        raise MB10R1EvalError("B_AUTHORIZATION_APPROVAL_FALSE")
    if overlay.get("status") != B_AUTHORIZATION_STATUS_GRANTED:
        raise MB10R1EvalError("B_AUTHORIZATION_STATUS_NOT_GRANTED")
    if overlay.get("independent_reviewer_authorization") is not True:
        raise MB10R1EvalError("B_INDEPENDENT_REVIEWER_NOT_AUTHORIZED")
    if overlay.get("recovery_execution_authorized") is not True:
        raise MB10R1EvalError("B_OVERLAY_EXECUTION_NOT_AUTHORIZED")
    if overlay.get("recovery_payload_release_authorized") is not True:
        raise MB10R1EvalError("B_OVERLAY_PAYLOAD_NOT_AUTHORIZED")
    if overlay.get("one_recovery_release_only") is not True:
        raise MB10R1EvalError("B_OVERLAY_ONE_RELEASE_REQUIRED")
    if overlay.get("retry_prohibited") is not True:
        raise MB10R1EvalError("B_OVERLAY_RETRY_MUST_BE_PROHIBITED")
    if int(overlay.get("expected_eligible_windows", -1)) != EXPECTED_ELIGIBLE:
        raise MB10R1EvalError("B_OVERLAY_ELIGIBLE_MISMATCH")
    if int(overlay.get("expected_subjects", -1)) != EXPECTED_SUBJECTS:
        raise MB10R1EvalError("B_OVERLAY_SUBJECTS_MISMATCH")
    if int(overlay.get("expected_model_inference_count", -1)) != EXPECTED_INFERENCES:
        raise MB10R1EvalError("B_OVERLAY_INFERENCE_COUNT_MISMATCH")
    if int(overlay.get("model_count", -1)) != 3:
        raise MB10R1EvalError("B_OVERLAY_MODEL_COUNT_NOT_3")
    if overlay.get("result_not_pristine") is not True:
        raise MB10R1EvalError("B_OVERLAY_RESULT_NOT_PRISTINE_REQUIRED")
    if overlay.get("result_limitation") != RESULT_LIMITATION:
        raise MB10R1EvalError("B_OVERLAY_RESULT_LIMITATION_MISMATCH")
    if overlay.get("execution_freeze_identity_sha256") != freeze_sha:
        raise MB10R1EvalError("B_OVERLAY_FREEZE_SHA_MISMATCH")
    head = overlay.get("reviewed_m_b10r1a_head_sha")
    if not isinstance(head, str) or len(head) != 40:
        raise MB10R1EvalError("B_OVERLAY_REVIEWED_HEAD_MISSING")


def execute_authorized_recovery(root: Path, authorization_token: str) -> dict[str, Any]:
    """Irreversible recovery path. MUST NOT be called during M-B10R1-A.

    Authorization is the B overlay, not A readiness. A flags stay historically false.
    Runtime state is persisted under the B output directory only.
    """
    _assert_a_readiness_historically_false(root)
    try:
        overlay = load_b_authorization_record(root)
    except Exception as exc:
        raise MB10R1EvalError(f"B_AUTHORIZATION_RECORD_UNREADABLE:{exc}") from exc

    # Freeze binding BEFORE payload release — live compared against frozen snapshot.
    frozen = authorize_pre_access_freeze_binding(root)
    bound = require_frozen_bound_contract(frozen)
    freeze_sha = sha256_file(root / OUT_DIR_REL / EXECUTION_FREEZE_IDENTITY_NAME)
    _assert_b_authorization_granted(overlay, freeze_sha)

    specs = validate_frozen_recovery_models(root)
    b_out = root / B_OUT_DIR_REL
    b_state_path = b_out / "recovery_access_runtime_state.json"
    try:
        initialize_b_runtime_from_a(root, b_state_path)
    except Exception as exc:
        raise MB10R1EvalError(f"B_RUNTIME_INIT_FAILED:{exc}") from exc

    # Grant comes from B overlay. A readiness is not mutated to true.
    readiness = RecoveryReadiness(
        recovery_execution_authorized=True,
        recovery_payload_release_authorized=True,
        independent_review_required=True,
        mechanism_implemented=True,
        runner_implemented=True,
        pre_access_validator_pass=True,
        M_B10R1B_started=True,
    )
    controller = LimitedReuseRecoveryAccessController(root, audit_state_path=b_state_path)
    try:
        payload = controller.get_locked_test_recovery_evaluation_dataset(
            authorization_token=authorization_token,
            bound_contract_identity=bound,
            readiness=readiness,
        )
    except Exception as exc:
        snap = controller.snapshot()
        if int(snap.get("recovery_payload_release_events", 0)) >= 1:
            persist_terminal_failure(
                b_out,
                runtime_state=snap,
                authorization=overlay,
                exception=exc,
                failure_stage="PAYLOAD_VERIFY",
            )
        raise

    if int(payload["total_count"]) != EXPECTED_ELIGIBLE:
        persist_terminal_failure(
            b_out,
            runtime_state=controller.snapshot(),
            authorization=overlay,
            exception=MB10R1EvalError("POST_RELEASE_COUNT_MISMATCH"),
            failure_stage="POST_RELEASE_COUNT",
        )
        raise MB10R1EvalError("POST_RELEASE_COUNT_MISMATCH")

    try:
        evaluation = evaluate_recovery_payload(root, payload, specs, runners=None)
        persist_recovery_results(
            b_out,
            evaluation,
            runtime_state=controller.snapshot(),
            authorization=overlay,
            frozen=frozen,
            specs=specs,
        )
        return evaluation
    except Exception as exc:
        persist_terminal_failure(
            b_out,
            runtime_state=controller.snapshot(),
            authorization=overlay,
            exception=exc,
            failure_stage="POST_PAYLOAD_EVALUATION",
            ledger=getattr(exc, "ledger", None),
            coverage=getattr(exc, "coverage", None),
            completed_inference_count=getattr(exc, "completed_inference_count", None),
        )
        raise


def readiness_summary(root: Path) -> dict[str, Any]:
    """Default CLI payload — never accesses recovery."""
    out = root / OUT_DIR_REL
    readiness_path = out / "recovery_access_readiness.json"
    audit_path = out / "recovery_access_audit.json"
    summary = {
        "phase_id": "M-B10R1-A",
        "mode": "PRE_ACCESS_READINESS_SUMMARY",
        "recovery_accessor_invoked": False,
        "recovery_payload_released": False,
        "locked_test_reopened": False,
        "recovery_execution_authorized": False,
        "recovery_authorization_token_constant_present": RECOVERY_AUTHORIZATION_TOKEN,
        "original_final_token_rejected_for_recovery": ORIGINAL_FINAL_TOKEN,
        "result_limitation": RESULT_LIMITATION,
        "expected_eligible": EXPECTED_ELIGIBLE,
        "expected_inferences": EXPECTED_INFERENCES,
        "generated_at": _utc_now(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
    }
    if readiness_path.is_file():
        summary["readiness"] = load_json(readiness_path)
    else:
        summary["readiness"] = build_preaccess_readiness(root, validator_pass=False)
    if audit_path.is_file():
        summary["audit"] = load_json(audit_path)
    return summary
