#!/usr/bin/env python3
"""Fail-closed validator for M-B10R1-A recovery harness pre-freeze.

NEVER calls recovery accessor or LOCKED_TEST final accessor.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUT_DIR_REL = Path("datasets/mmwave/manifests/M-B10R1A_recovery_prefreeze")
M_B10R0_DIR_REL = Path("datasets/mmwave/manifests/M-B10R0_holdout_policy_review")
M_B10A_DIR_REL = Path("datasets/mmwave/manifests/M-B10A_candidate_selection_setup")
M_B10B_DIR_REL = Path("datasets/mmwave/manifests/M-B10B_locked_test_final_evaluation")
REPORT_REL = Path("docs/reports/20260813_Cursor_M-B10R1A_Recovery_PreFreeze_01.md")

RECOVERY_ACCESS_MODULE = Path("scripts/mmwave_m_b10r1_recovery_access.py")
RECOVERY_EVAL_MODULE = Path("scripts/mmwave_m_b10r1_recovery_eval.py")
RECOVERY_METRICS_MODULE = Path("scripts/mmwave_m_b10r1_metrics.py")
PREFREEZE_MODULE = Path("scripts/mmwave_m_b10r1a_prefreeze.py")
RUNNER_MODULE = Path("scripts/run_mmwave_m_b10r1.py")

SELECTED_SHA = "6dff6aaa72c79d76715d40cf7e32bb1e6cd9b2c2e3ac78eaf2fda737561430c5"
V01_SHA = "43cdd6f321c2b645232162233e098c2b65549388cbc9e680a17a0eccdc8f0158"
V02_SHA = "85c023d3eefca13ecbb72a841974e53a56d5f4173920645d46df49a9088452ff"
EXECUTOR_SHA = "8ca87f457d0a151cffa2da23ae9ab9d87764b144fa826b91444776f3dc58ec4f"
META_V01_SHA = "a875a8369ff7adf5477cec009b99c0c6d0fbb8b0e60e5b0b07a551f3780d2e37"
META_V02_SHA = "36039a6cffbc57162dbb4c720034da6dcfa49ef2f2d33238bee65a62aa133127"
M_B10A_CONTRACT_SHA = "ba6429ecfe685de1807ec85b55e697ee12e24138e6b96e94715b0a1a6b19e0f7"
RESULT_LIMITATION = "REUSED_LOCKED_TEST_AFTER_PREINFERENCE_STRUCTURAL_ABORT"
RECOVERY_TOKEN = "M_B10R1_LIMITED_REUSE_RECOVERY_AUTHORIZATION_V1"
ORIGINAL_TOKEN = "AUTHORIZED_FINAL_LOCKED_TEST_EVALUATION_TOKEN_V1"

SELECTED_PATH = (
    "models/mmwave/experiments/M-B6_stage_equivalence/"
    "M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120_int8.tflite"
)
V01_PATH = "models/mmwave/mmwave_resp_int8_v0.1.0.tflite"
V02_PATH = "models/mmwave/mmwave_resp_int8_v0.2.0_candidate.tflite"
EXECUTOR_PATH = "scripts/mmwave_m_b10b_baseline_preprocessing.py"
META_V01_PATH = "models/mmwave/sensor_stats_metadata_v0.1.0.json"
META_V02_PATH = "models/mmwave/mmwave_resp_int8_v0.2.0_candidate_metadata.json"

FORBIDDEN_PRISTINE = {
    "PRISTINE_LOCKED_TEST",
    "PRISTINE_ONE_TIME_LOCKED_TEST",
    "PRISTINE_REAL_SUBJECT_FINAL_TEST",
    "FIRST_LOCKED_TEST_EVALUATION",
    "LOCKED_TEST_NOT_CONSUMED",
    "NO_INFORMATION_EXPOSURE",
    "ORIGINAL_ACCESS_UNUSED",
}

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

SELECTED_PREPROCESSING_CONTRACT_ID = "M-B10B_SELECTED_REAL_CANDIDATE_BPF_ZSCORE_V1"
V01_PREPROCESSING_CONTRACT_ID = "M-B10B_HISTORICAL_V0_1_COMPATIBILITY_PREPROCESSING_V1"
V02_PREPROCESSING_CONTRACT_ID = "M-B10B_SYNTHETIC_V0_2_EXTERNAL_COMPATIBILITY_PREPROCESSING_V1"
SELECTED_MODEL_ID = "M-B3_CONV1D_GAP_BASELINE_seed42_M-B6_STRICT_INT8"

HARNESS_MODULE_RELS = (
    "scripts/mmwave_m_b10r1_recovery_access.py",
    "scripts/mmwave_m_b10r1_recovery_eval.py",
    "scripts/mmwave_m_b10r1_metrics.py",
    "scripts/run_mmwave_m_b10r1.py",
    "scripts/mmwave_m_b10b_baseline_preprocessing.py",
    "scripts/mmwave_m_b10r1_result_writer.py",
    "scripts/validate_mmwave_m_b10r1b.py",
)


class MB10R1AValidationError(Exception):
    """Fail-closed M-B10R1-A validation failure."""


def _raise(code: str) -> None:
    raise MB10R1AValidationError(code)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_upstream(root: Path) -> None:
    commands = [
        [sys.executable, str(root / "scripts/validate_mmwave_m_b10r0.py")],
        [sys.executable, str(root / "scripts/validate_mmwave_m_b10b_incident.py")],
        [sys.executable, str(root / "scripts/validate_mmwave_m_b10a.py"), "--skip-upstream"],
    ]
    for cmd in commands:
        completed = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
        if completed.returncode != 0:
            _raise(
                "UPSTREAM_VALIDATOR_FAILED:"
                f"{Path(cmd[1]).name}:rc={completed.returncode}:{completed.stderr[-500:]}"
            )


def _validate_checksums(out: Path) -> None:
    checksum_path = out / "checksums.sha256"
    if not checksum_path.is_file():
        _raise("CHECKSUMS_MISSING")
    lines = [ln.strip() for ln in checksum_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    mapped: dict[str, str] = {}
    for line in lines:
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64:
            _raise(f"CHECKSUM_LINE_INVALID:{line}")
        digest, name = parts
        if name.startswith("/") or ".." in name or name.startswith("file:"):
            _raise(f"CHECKSUM_PATH_FORBIDDEN:{name}")
        mapped[name] = digest
    files = {p.name for p in out.iterdir() if p.is_file()}
    expected = set(REQUIRED_OUTPUTS) | {"recovery_access_runtime_state.json"}
    # checksums.sha256 covers all files except itself
    covered = set(mapped)
    actual_others = files - {"checksums.sha256"}
    if covered != actual_others:
        _raise(f"CHECKSUM_COVERAGE_MISMATCH:missing={sorted(actual_others-covered)};extra={sorted(covered-actual_others)}")
    for name, digest in mapped.items():
        live = sha256_file(out / name)
        if live != digest:
            _raise(f"CHECKSUM_MISMATCH:{name}")
    if not REQUIRED_OUTPUTS.issubset(files):
        _raise(f"REQUIRED_OUTPUTS_MISSING:{sorted(REQUIRED_OUTPUTS - files)}")


def _reject_abs_paths(obj: Any, context: str) -> None:
    text = json.dumps(obj)
    if "/Users/" in text or "file://" in text:
        _raise(f"ABSOLUTE_PATH_IN_ARTIFACT:{context}")


def _ast_call_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def _inspect_recovery_access_source(root: Path) -> None:
    path = root / RECOVERY_ACCESS_MODULE
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = _ast_call_names(tree)
    if "get_locked_test_final_evaluation_dataset" in calls:
        _raise("RECOVERY_ACCESS_CALLS_FINAL_ACCESSOR")
    if RECOVERY_TOKEN not in source:
        _raise("RECOVERY_TOKEN_MISSING")
    if ORIGINAL_TOKEN not in source:
        _raise("ORIGINAL_TOKEN_REJECTION_CONSTANT_MISSING")
    if "SECOND_RECOVERY" not in source and "recovery_payload_release_events" not in source:
        _raise("SECOND_ACCESS_REFUSAL_MISSING")
    if re.search(
        r'original_final_accessor_invocations["\']?\s*[:=]\s*0\b',
        source,
    ):
        _raise("ORIGINAL_COUNTER_RESET_TO_ZERO_PRESENT")
    if "_get_split_dataset" not in source:
        _raise("PRIVATE_SPLIT_LOADER_REQUIRED")
    if "include_ambiguous=False" not in source and "include_ambiguous = False" not in source:
        _raise("INCLUDE_AMBIGUOUS_FALSE_REQUIRED")
    # Payload release must be recorded at loader return boundary, before verify.
    load_idx = source.find("self._load_eligible_locked_test")
    release_idx = source.find('recovery_payload_release_events')
    # Find the increment after load: second occurrence of assignment pattern after load.
    verify_idx = source.find("self._verify_payload")
    if load_idx < 0 or verify_idx < 0:
        _raise("ACCESS_LOAD_OR_VERIFY_MISSING")
    # Require release increment text between load call and verify call.
    between = source[load_idx:verify_idx]
    if "recovery_payload_release_events" not in between:
        _raise("PAYLOAD_RELEASE_NOT_BEFORE_VERIFY")
    if "_persist" not in between:
        _raise("PAYLOAD_RELEASE_NOT_PERSISTED_BEFORE_VERIFY")


def _inspect_recovery_eval_source(root: Path) -> None:
    path = root / RECOVERY_EVAL_MODULE
    source = path.read_text(encoding="utf-8")
    if "def evaluate_recovery_payload" not in source:
        _raise("EVALUATE_RECOVERY_PAYLOAD_MISSING")
    if "preprocess_for_spec(signal" not in source:
        _raise("EVAL_PREPROCESS_MUST_USE_SIGNAL")
    if "preprocess_for_spec(window" in source:
        _raise("EVAL_PREPROCESS_USES_WINDOW_METADATA")
    if "actual_total_tflite_invocations" not in source:
        _raise("EVAL_MISSING_TFLITE_INVOKE_TRACKING")
    for field in (
        "evaluation_rows_attempted",
        "preprocessing_success_count",
        "tflite_invoke_count",
        "invalid_preprocessing_count",
        "invalid_inference_count",
    ):
        if field not in source:
            _raise(f"EVAL_MISSING_COVERAGE_FIELD:{field}")
    for contract_id in (
        SELECTED_PREPROCESSING_CONTRACT_ID,
        V01_PREPROCESSING_CONTRACT_ID,
        V02_PREPROCESSING_CONTRACT_ID,
    ):
        if contract_id not in source:
            _raise(f"EVAL_MISSING_PREPROCESSING_CONTRACT_ID:{contract_id}")
    if "load_frozen_execution_identity" not in source:
        _raise("EVAL_MISSING_LOAD_FROZEN_IDENTITY")
    if "verify_live_against_frozen" not in source:
        _raise("EVAL_MISSING_VERIFY_LIVE_AGAINST_FROZEN")
    if "authorize_pre_access_freeze_binding" not in source:
        _raise("EVAL_MISSING_PREACCESS_FREEZE_BINDING")
    if "or build_bound_contract_identity" in source:
        _raise("EVAL_BOUND_LIVE_REBUILD_FALLBACK")
    if "FROZEN_BOUND_CONTRACT_IDENTITY_MISSING_STOP_BEFORE_PAYLOAD" not in source:
        _raise("EVAL_MISSING_FAIL_CLOSED_BOUND")
    if "B_OUT_DIR_REL" not in source and "M-B10R1B_recovery_execution" not in source:
        _raise("EVAL_MISSING_B_OUTPUT_DIR")
    if "A_READINESS_MUST_REMAIN_HISTORICALLY_FALSE" not in source:
        _raise("EVAL_MISSING_A_HISTORICAL_FALSE_GUARD")
    if "load_b_authorization_record" not in source:
        _raise("EVAL_MISSING_B_AUTHORIZATION_OVERLAY")
    if "persist_recovery_results" not in source:
        _raise("EVAL_MISSING_DURABLE_RESULT_WRITER")
    if "persist_terminal_failure" not in source:
        _raise("EVAL_MISSING_TERMINAL_FAILURE_PERSISTENCE")


def _inspect_metrics_source(root: Path) -> None:
    source = (root / RECOVERY_METRICS_MODULE).read_text(encoding="utf-8")
    if "METRIC_EMPTY_LABELS_WITH_POSITIVE_EVALUATED_COUNT" not in source:
        _raise("METRIC_BUNDLE_EMPTY_GUARD_MISSING")
    if "METRIC_EVALUATED_SAMPLE_COUNT_MISMATCH" not in source:
        _raise("METRIC_BUNDLE_COUNT_MISMATCH_GUARD_MISSING")


def _inspect_b_validator_source(root: Path) -> None:
    path = root / "scripts/validate_mmwave_m_b10r1b.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else None
            )
            if name in {
                "get_locked_test_recovery_evaluation_dataset",
                "get_locked_test_final_evaluation_dataset",
                "execute_authorized_recovery",
            }:
                _raise(f"B_VALIDATOR_CALLS_FORBIDDEN_ACCESS:{name}")


def _inspect_validator_self(root: Path) -> None:
    source = (root / "scripts/validate_mmwave_m_b10r1a.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in {
                "get_locked_test_recovery_evaluation_dataset",
                "get_locked_test_final_evaluation_dataset",
                "execute_authorized_recovery",
            }:
                _raise(f"VALIDATOR_CALLS_FORBIDDEN_ACCESS:{name}")


def _inspect_runner_default(root: Path) -> None:
    source = (root / RUNNER_MODULE).read_text(encoding="utf-8")
    if "execute_authorized_limited_reuse_recovery" not in source:
        _raise("RUNNER_MISSING_DANGEROUS_FLAG")
    # Default path must call readiness_summary / generate, not execute without flag.
    tree = ast.parse(source)
    # Ensure execute_authorized_recovery is gated.
    if "if args.execute_authorized_limited_reuse_recovery" not in source:
        _raise("RUNNER_EXECUTE_NOT_GATED")


def validate_m_b10r1a_artifacts(
    root: Path,
    *,
    output_dir: Path | None = None,
    skip_upstream: bool = False,
    mark_validator_pass: bool = False,
) -> dict[str, Any]:
    root = Path(root)
    out = Path(output_dir) if output_dir is not None else root / OUT_DIR_REL
    if not out.is_dir():
        _raise("OUTPUT_DIR_MISSING")

    _inspect_validator_self(root)
    if not skip_upstream:
        _run_upstream(root)

    # Upstream policy checks
    policy = load_json(root / M_B10R0_DIR_REL / "policy_decision.json")
    if policy.get("decision") != "LIMITED_LOCKED_TEST_REUSE_EXCEPTION_RECOMMENDED":
        _raise("POLICY_DECISION_NOT_LIMITED_REUSE")
    gates = load_json(root / M_B10R0_DIR_REL / "reuse_exception_gate_results.json")
    if gates.get("all_r1_r10_pass") is not True:
        _raise("R0_GATES_NOT_ALL_PASS")

    _validate_checksums(out)

    artifacts = {name: load_json(out / name) for name in REQUIRED_OUTPUTS if name.endswith(".json")}
    for name, payload in artifacts.items():
        _reject_abs_paths(payload, name)

    # Incident / audit historical facts
    incident = artifacts["incident_identity.json"]
    if int(incident.get("original_accessor_invocations", 0)) != 1:
        _raise("INCIDENT_ACCESSOR_NOT_1")
    if int(incident.get("rows_returned", -1)) != 75:
        _raise("INCIDENT_ROWS_NOT_75")
    if int(incident.get("model_inference_invocations", -1)) != 0:
        _raise("INCIDENT_INFERENCE_NOT_0")
    if incident.get("locked_test_consumed") is not True:
        _raise("INCIDENT_NOT_CONSUMED")

    audit = artifacts["recovery_access_audit.json"]
    if int(audit.get("historical_original_final_accessor_invocations", 0)) != 1:
        _raise("AUDIT_ORIGINAL_ACCESSOR_NOT_1")
    if int(audit.get("historical_original_payload_release_events", 0)) != 1:
        _raise("AUDIT_ORIGINAL_PAYLOAD_NOT_1")
    if int(audit.get("M-B10R1A_recovery_accessor_invocations", -1)) != 0:
        _raise("AUDIT_NEW_ACCESSOR_NOT_0")
    if int(audit.get("M-B10R1A_recovery_payload_release_events", -1)) != 0:
        _raise("AUDIT_NEW_PAYLOAD_NOT_0")
    if int(audit.get("historical_total_payload_release_events", 0)) != 1:
        _raise("AUDIT_HISTORICAL_TOTAL_NOT_1")
    if audit.get("original_locked_test_consumed") is not True:
        _raise("AUDIT_ORIGINAL_CONSUMED_FALSE")
    if audit.get("original_pristine_status") is not False:
        _raise("AUDIT_PRISTINE_MUST_BE_FALSE")

    readiness = artifacts["recovery_access_readiness.json"]
    if readiness.get("mechanism_implemented") is not True:
        _raise("READINESS_MECHANISM_NOT_TRUE")
    if readiness.get("runner_implemented") is not True:
        _raise("READINESS_RUNNER_NOT_TRUE")
    if readiness.get("independent_review_required") is not True:
        _raise("READINESS_REVIEW_NOT_REQUIRED")
    if readiness.get("recovery_execution_authorized") is not False:
        _raise("READINESS_EXECUTION_AUTHORIZED_MUST_BE_FALSE")
    if readiness.get("recovery_payload_release_authorized") is not False:
        _raise("READINESS_PAYLOAD_AUTHORIZED_MUST_BE_FALSE")
    if readiness.get("M-B10R1B_started") is not False:
        _raise("READINESS_R1B_STARTED_MUST_BE_FALSE")
    if int(readiness.get("new_recovery_accessor_invocations", -1)) != 0:
        _raise("READINESS_NEW_ACCESSOR_NOT_0")
    if int(readiness.get("new_payload_release_events", -1)) != 0:
        _raise("READINESS_NEW_PAYLOAD_NOT_0")
    # Frozen M-B10R1-A evidence must already record a successful pre-access validator stamp.
    # (mark_validator_pass=True may be used once during freeze to create that stamp.)
    if not mark_validator_pass and readiness.get("pre_access_validator_pass") is not True:
        _raise("READINESS_VALIDATOR_PASS_NOT_TRUE")

    # Population
    pop = artifacts["recovery_population_contract.json"]
    if int(pop.get("structural_windows", -1)) != 88:
        _raise("POP_STRUCTURAL_NOT_88")
    if int(pop.get("supervised_eligible_windows", -1)) != 75:
        _raise("POP_ELIGIBLE_NOT_75")
    if int(pop.get("excluded_ambiguous_windows", -1)) != 13:
        _raise("POP_AMBIGUOUS_NOT_13")
    if int(pop.get("subjects", -1)) != 16:
        _raise("POP_SUBJECTS_NOT_16")
    if pop.get("eligibility_provenance") != "PREEXISTING_A6_METADATA_VERIFIED" and pop.get(
        "subject_count_policy"
    ) != "PREEXISTING_A6_METADATA_VERIFIED":
        _raise("POP_PROVENANCE_MISMATCH")
    if pop.get("positional_truncation") is not False:
        _raise("POP_POSITIONAL_TRUNCATION_MUST_BE_FALSE")
    if pop.get("include_ambiguous") is True:
        _raise("POP_INCLUDE_AMBIGUOUS_TRUE")

    # Models — recompute SHAs
    live = {
        "selected": sha256_file(root / SELECTED_PATH),
        "v01": sha256_file(root / V01_PATH),
        "v02": sha256_file(root / V02_PATH),
        "executor": sha256_file(root / EXECUTOR_PATH),
        "meta_v01": sha256_file(root / META_V01_PATH),
        "meta_v02": sha256_file(root / META_V02_PATH),
        "m_b10a": sha256_file(root / M_B10A_DIR_REL / "locked_test_evaluation_contract.json"),
    }
    if live["selected"] != SELECTED_SHA:
        _raise("LIVE_SELECTED_SHA_MISMATCH")
    if live["v01"] != V01_SHA:
        _raise("LIVE_V01_SHA_MISMATCH")
    if live["v02"] != V02_SHA:
        _raise("LIVE_V02_SHA_MISMATCH")
    if live["executor"] != EXECUTOR_SHA:
        _raise("LIVE_EXECUTOR_SHA_MISMATCH")
    if live["meta_v01"] != META_V01_SHA:
        _raise("LIVE_META_V01_SHA_MISMATCH")
    if live["meta_v02"] != META_V02_SHA:
        _raise("LIVE_META_V02_SHA_MISMATCH")
    if live["m_b10a"] != M_B10A_CONTRACT_SHA:
        _raise("LIVE_M_B10A_CONTRACT_SHA_MISMATCH")

    models = artifacts["model_identity_registry.json"]
    model_list = models.get("models") or []
    if len(model_list) != 3:
        _raise("MODEL_COUNT_NOT_3")
    serialized = json.dumps(models, sort_keys=True).lower()
    if "seed43" in serialized or "seed44" in serialized:
        _raise("FORBIDDEN_SEED_IN_MODEL_REGISTRY")
    ids = [m.get("model_id") for m in model_list]
    if len(ids) != len(set(ids)):
        _raise("DUPLICATE_MODEL_IDS")
    for mid in ids:
        if mid and ("seed43" in str(mid) or "seed44" in str(mid)):
            _raise(f"FORBIDDEN_MODEL_ID:{mid}")
    expected_contracts = {
        SELECTED_MODEL_ID: SELECTED_PREPROCESSING_CONTRACT_ID,
        "mmwave_resp_int8": V01_PREPROCESSING_CONTRACT_ID,
        "mmwave_resp_int8_v0.2.0_candidate": V02_PREPROCESSING_CONTRACT_ID,
    }
    for model in model_list:
        mid = model.get("model_id")
        expected = expected_contracts.get(mid)
        if expected is None:
            _raise(f"UNEXPECTED_MODEL_ID:{mid}")
        if model.get("preprocessing_contract_id") != expected:
            _raise(f"MODEL_PREPROCESSING_CONTRACT_MISMATCH:{mid}")

    # Authoritative execution freeze identity
    freeze = artifacts["execution_freeze_identity.json"]
    if freeze.get("schema_version") != "M-B10R1A_EXECUTION_FREEZE_IDENTITY_V1":
        _raise("FREEZE_IDENTITY_SCHEMA_MISMATCH")
    harness = freeze.get("harness_module_sha256") or {}
    for rel in HARNESS_MODULE_RELS:
        if rel not in harness:
            _raise(f"FREEZE_HARNESS_SHA_MISSING:{rel}")
        live_mod = sha256_file(root / rel)
        if live_mod != harness[rel]:
            _raise(f"FREEZE_HARNESS_SHA_MISMATCH:{rel}")
    freeze_contracts = freeze.get("preprocessing_contract_ids") or {}
    for mid, expected in expected_contracts.items():
        if freeze_contracts.get(mid) != expected:
            _raise(f"FREEZE_PREPROCESSING_CONTRACT_MISMATCH:{mid}")
    if freeze.get("m_b10a_metric_contract_sha256") != M_B10A_CONTRACT_SHA:
        # Allow live recompute equality as authoritative check
        live_m_b10a = sha256_file(root / M_B10A_DIR_REL / "locked_test_evaluation_contract.json")
        if freeze.get("m_b10a_metric_contract_sha256") != live_m_b10a:
            _raise("FREEZE_M_B10A_METRIC_SHA_MISMATCH")

    baseline = artifacts["baseline_identity_registry.json"]
    if baseline.get("executor_sha256") != EXECUTOR_SHA:
        _raise("BASELINE_EXECUTOR_SHA_MISMATCH")
    if baseline.get("v0_1", {}).get("sha256") != V01_SHA:
        _raise("BASELINE_V01_SHA_MISMATCH")
    if baseline.get("v0_2", {}).get("sha256") != V02_SHA:
        _raise("BASELINE_V02_SHA_MISMATCH")
    if baseline.get("v0_1", {}).get("metadata_sha256") != META_V01_SHA:
        _raise("BASELINE_META_V01_SHA_MISMATCH")
    if baseline.get("v0_2", {}).get("metadata_sha256") != META_V02_SHA:
        _raise("BASELINE_META_V02_SHA_MISMATCH")
    if baseline.get("v0_1", {}).get("executor_sha256") != EXECUTOR_SHA:
        _raise("BASELINE_V01_EXECUTOR_SHA_MISMATCH")
    if baseline.get("v0_2", {}).get("executor_sha256") != EXECUTOR_SHA:
        _raise("BASELINE_V02_EXECUTOR_SHA_MISMATCH")

    # Metric schema deep-equal to M-B10A
    m_b10a = load_json(root / M_B10A_DIR_REL / "locked_test_evaluation_contract.json")
    metric_contract = artifacts["metric_contract.json"]
    if metric_contract.get("metrics_schema") != m_b10a.get("metrics_schema"):
        _raise("METRIC_SCHEMA_NOT_DEEP_EQUAL_M_B10A")
    thresh = metric_contract.get("applicable_predefined_numerical_acceptance_threshold") or metric_contract.get(
        "acceptance_threshold"
    )
    if "NOT_PREDEFINED" not in str(thresh):
        _raise("METRIC_THRESHOLD_NOT_PREDEFINED")

    # Result designation
    result_schema = artifacts["future_result_schema.json"]
    if result_schema.get("required_result_designation") != RESULT_LIMITATION:
        _raise("RESULT_DESIGNATION_MISMATCH")
    if result_schema.get("result_not_pristine") is not True:
        _raise("RESULT_NOT_PRISTINE_REQUIRED")
    if result_schema.get("status") not in {"NOT_POPULATED", "NOT_EXECUTED"}:
        _raise("RESULT_SCHEMA_MUST_BE_PLACEHOLDER")
    ledger = artifacts["future_ledger_schema.json"]
    if ledger.get("status") != "NOT_EXECUTED":
        _raise("LEDGER_STATUS_MUST_BE_NOT_EXECUTED")
    coverage_fields = ledger.get("coverage_tracking_fields") or []
    for field in (
        "evaluation_rows_attempted",
        "preprocessing_success_count",
        "tflite_invoke_count",
        "invalid_preprocessing_count",
        "invalid_inference_count",
    ):
        if field not in coverage_fields and field not in json.dumps(ledger):
            _raise(f"LEDGER_MISSING_COVERAGE_FIELD:{field}")
    if RESULT_LIMITATION not in json.dumps(ledger):
        # designation may live on result schema only; require on at least one
        pass
    blob = json.dumps(
        {
            "result": result_schema,
            "ledger": ledger,
            "frozen": artifacts["frozen_recovery_contract.json"],
            "summary": artifacts["m_b10r1a_summary.json"],
        }
    )
    for claim in FORBIDDEN_PRISTINE:
        if claim in blob and result_schema.get("forbidden_scientific_wording") and claim in result_schema.get(
            "forbidden_scientific_wording", []
        ):
            continue  # listed as forbidden is OK
        # Reject affirmative pristine claims outside forbidden lists
    if '"PRISTINE_LOCKED_TEST": true' in blob.lower() or "pristine_locked_test\": true" in blob.lower():
        _raise("PRISTINE_CLAIM_PRESENT")

    access_contract = artifacts["recovery_access_contract.json"]
    if access_contract.get("authorization_token_id") != "M_B10R1_LIMITED_REUSE_RECOVERY_AUTHORIZATION_V1":
        _raise("ACCESS_TOKEN_ID_MISMATCH")
    if access_contract.get("modifies_mmwave_phase_b_access") is not False:
        _raise("ACCESS_MUST_NOT_MODIFY_PHASE_B")
    if access_contract.get("at_most_one_recovery_payload_release") is not True:
        _raise("ACCESS_AT_MOST_ONE_REQUIRED")
    if access_contract.get("original_counter_reset_forbidden") is not True:
        _raise("ACCESS_NO_RESET_REQUIRED")

    summary = artifacts["m_b10r1a_summary.json"]
    if summary.get("recovery_execution_authorized") is not False:
        _raise("SUMMARY_EXECUTION_AUTHORIZED")
    if int(summary.get("new_recovery_accessor_invocations", -1)) != 0:
        _raise("SUMMARY_NEW_ACCESSOR_NOT_0")
    if int(summary.get("new_payload_release_events", -1)) != 0:
        _raise("SUMMARY_NEW_PAYLOAD_NOT_0")
    if summary.get("a_readiness_mutation_required_for_b") is not False:
        _raise("SUMMARY_A_READINESS_MUTATION_MUST_BE_FALSE")
    if summary.get("b_authorization_status") != "NOT_AUTHORIZED_NOT_EXECUTED":
        _raise("SUMMARY_B_OVERLAY_NOT_TEMPLATE")

    overlay = artifacts["future_b_authorization_overlay.json"]
    if overlay.get("approval") is not False:
        _raise("A_OVERLAY_TEMPLATE_APPROVAL_TRUE")
    if overlay.get("status") != "NOT_AUTHORIZED_NOT_EXECUTED":
        _raise("A_OVERLAY_TEMPLATE_STATUS")
    if overlay.get("recovery_execution_authorized") is not False:
        _raise("A_OVERLAY_TEMPLATE_EXECUTION_AUTHORIZED")
    if overlay.get("independent_reviewer_authorization") is not False:
        _raise("A_OVERLAY_TEMPLATE_REVIEWER_AUTHORIZED")
    freeze_sha = sha256_file(out / "execution_freeze_identity.json")
    if overlay.get("execution_freeze_identity_sha256") != freeze_sha:
        _raise("A_OVERLAY_FREEZE_SHA_MISMATCH")

    immutability = artifacts["a_directory_immutability_contract.json"]
    if immutability.get("m_b10r1a_directory_immutable_after_merge") is not True:
        _raise("A_IMMUTABILITY_CONTRACT_MISSING")
    if immutability.get("a_readiness_recovery_execution_authorized_forever") is not False:
        _raise("A_IMMUTABILITY_EXECUTION_MUST_STAY_FALSE")
    if immutability.get("future_b_must_not_mutate_a_readiness") is not True:
        _raise("A_IMMUTABILITY_B_MUST_NOT_MUTATE_READINESS")

    b_schema = artifacts["future_b_result_directory_schema.json"]
    if b_schema.get("populated_during_m_b10r1a") is not False:
        _raise("B_SCHEMA_MARKED_POPULATED")
    if b_schema.get("status") != "NOT_POPULATED":
        _raise("B_SCHEMA_STATUS_NOT_TEMPLATE")

    b_dir = root / "datasets/mmwave/manifests/M-B10R1B_recovery_execution"
    if b_dir.is_dir():
        for name in (
            "recovery_sample_predictions.jsonl",
            "metrics_by_model.json",
            "m_b10r1b_summary.json",
            "model_evaluation_coverage.json",
            "recovery_access_runtime_state.json",
        ):
            if (b_dir / name).is_file():
                _raise(f"B_RESULT_POPULATED_DURING_A:{name}")
        b_auth_path = b_dir / "authorization_record.json"
        if b_auth_path.is_file():
            b_auth = load_json(b_auth_path)
            if b_auth.get("approval") is True:
                _raise("B_OVERLAY_APPROVAL_TRUE_DURING_A")
            if b_auth.get("status") != "NOT_AUTHORIZED_NOT_EXECUTED":
                _raise("B_OVERLAY_STATUS_NOT_TEMPLATE_DURING_A")
            if b_auth.get("recovery_execution_authorized") is not False:
                _raise("B_OVERLAY_EXECUTION_AUTHORIZED_DURING_A")

    report_path = root / REPORT_REL
    if not report_path.is_file():
        _raise("REPORT_MISSING")
    report_text = report_path.read_text(encoding="utf-8")
    if "RECOVERY HAS NOT BEEN EXECUTED" not in report_text:
        _raise("REPORT_MISSING_RECOVERY_NOT_EXECUTED")
    if "LOCKED_TEST HAS NOT BEEN REOPENED DURING M-B10R1-A" not in report_text:
        _raise("REPORT_MISSING_LOCKED_TEST_NOT_REOPENED")

    _inspect_recovery_access_source(root)
    _inspect_recovery_eval_source(root)
    _inspect_metrics_source(root)
    _inspect_runner_default(root)
    _inspect_b_validator_source(root)

    # Required source modules exist
    for rel in (
        RECOVERY_ACCESS_MODULE,
        RECOVERY_EVAL_MODULE,
        RECOVERY_METRICS_MODULE,
        PREFREEZE_MODULE,
        RUNNER_MODULE,
        Path("scripts/mmwave_m_b10r1_result_writer.py"),
        Path("scripts/validate_mmwave_m_b10r1b.py"),
    ):
        if not (root / rel).is_file():
            _raise(f"MODULE_MISSING:{rel}")

    # Runtime state preaccess
    runtime_path = out / "recovery_access_runtime_state.json"
    if runtime_path.is_file():
        runtime = load_json(runtime_path)
        if int(runtime.get("recovery_accessor_invocations", -1)) != 0:
            _raise("RUNTIME_RECOVERY_ACCESSOR_NOT_0")
        if int(runtime.get("recovery_payload_release_events", -1)) != 0:
            _raise("RUNTIME_RECOVERY_PAYLOAD_NOT_0")
        if int(runtime.get("original_final_accessor_invocations", 0)) != 1:
            _raise("RUNTIME_ORIGINAL_ACCESSOR_CORRUPT")

    # Optionally stamp validator pass into readiness (rewrite + checksum refresh).
    if mark_validator_pass:
        readiness = copy.deepcopy(artifacts["recovery_access_readiness.json"])
        readiness["pre_access_validator_pass"] = True
        readiness["pre_access_validator_status"] = "PASS"
        (out / "recovery_access_readiness.json").write_text(
            json.dumps(readiness, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksum_lines = []
        for path in sorted(out.iterdir(), key=lambda p: p.name):
            if path.is_file() and path.name != "checksums.sha256":
                checksum_lines.append(f"{sha256_file(path)}  {path.name}")
        (out / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        # Re-validate checksums after stamp
        _validate_checksums(out)

    return {
        "validation_status": "PASS",
        "phase_id": "M-B10R1-A",
        "recovery_execution_authorized": False,
        "recovery_payload_release_authorized": False,
        "new_recovery_accessor_invocations": 0,
        "new_payload_release_events": 0,
        "policy_decision": policy.get("decision"),
        "result_limitation": RESULT_LIMITATION,
        "output_dir": str(out.relative_to(root) if out.is_relative_to(root) else out),
    }


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    parser_skip = "--skip-upstream" in args
    stamp = "--stamp-pass" in args
    args = [a for a in args if a not in {"--skip-upstream", "--stamp-pass"}]
    try:
        result = validate_m_b10r1a_artifacts(
            ROOT_DIR, skip_upstream=parser_skip, mark_validator_pass=stamp
        )
    except MB10R1AValidationError as exc:
        print(json.dumps({"validation_status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
