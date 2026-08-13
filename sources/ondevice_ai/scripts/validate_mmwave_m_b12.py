#!/usr/bin/env python3
"""Fail-closed M-B12 Phase-B offline final-report validator.

Validates stored M-B11 lock evidence and M-B12 closure artifacts only.
Never calls LOCKED_TEST or recovery accessors, never invokes TFLite,
never trains, converts, calibrates, or begins M-C.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.mmwave_m_b10r1_result_writer import sha256_file as _sha256_file  # noqa: E402
from scripts.mmwave_m_b11_artifact_lock import (  # noqa: E402
    ARCHITECTURE_ID,
    ARTIFACT_STATUS,
    CALIBRATION_ID,
    CLASS_MAP,
    EXECUTION_PREPROCESSING_CONTRACT_ID,
    PREPROCESSING_PROFILE_ID,
    PREPROCESSING_PROFILE_NAME,
    RESULT_LIMITATION,
    RUNTIME_MODEL_ID,
    SELECTED_CANDIDATE_ID,
    SELECTED_TFLITE_REL,
    SENSOR_LOCK_REL,
    TRAINING_STRATEGY_ID,
    load_json,
    require_repo_relative,
)
from scripts.mmwave_m_b12_phase_b_closure import (  # noqa: E402
    CLOSURE_DIR_REL,
    CLOSURE_JSON_FILES,
    EXPECTED_ELIGIBLE,
    EXPECTED_HISTORICAL_RELEASES,
    EXPECTED_MACRO_F1,
    EXPECTED_MODEL_BYTES,
    EXPECTED_MODEL_SHA,
    EXPECTED_MODELS,
    EXPECTED_PAIRS,
    EXPECTED_RECOVERY_INFERENCE,
    EXPECTED_V01_F1,
    EXPECTED_V02_F1,
    M_B11_DIR_REL,
    MACHINE_FACTS_BEGIN,
    MACHINE_FACTS_END,
    PROPOSED_TAG,
    REPORT_REL,
    REQUIRED_M11_LOCK_FILE_ROLES,
    REQUIRED_M11_REGISTRY_ROLES,
    SCHEMA,
    STATUS_LABEL,
)
from scripts.validate_mmwave_m_b11 import validate_m_b11  # noqa: E402

HEX64 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_DESIGNATION_TOKENS = {
    "pristine_locked_test",
    "first_locked_test_evaluation",
}
FORBIDDEN_TRUE_KEY_TOKENS = {
    "pristine_locked_test",
    "first_locked_test_evaluation",
    "deployment_ready",
    "production_ready",
    "clinical_apnea_validated",
    "mr60_device_validation_complete",
    "mr60_validated",
    "mr60_validation_complete",
    "raspberry_pi_validation_complete",
    "raspberry_pi_validated",
    "rpi_validated",
    "rpi_validation_complete",
    "locked_test_reopen_allowed",
    "recovery_reopen_allowed",
    "phase_b_release_ready",
    "git_tag_created",
    "github_release_created",
    "m_c_started",
    "m_c_begun",
}
TRUTHY_TOKENS = {"true", "yes", "validated", "complete"}
FORBIDDEN_POSITIVE_VALUE_TOKENS = FORBIDDEN_DESIGNATION_TOKENS | FORBIDDEN_TRUE_KEY_TOKENS
V01_ROLE = "HISTORICAL_MODEL_COMPATIBILITY_BENCHMARK"
V02_ROLE = "SYNTHETIC_TRAINED_EXTERNAL_COMPATIBILITY_BENCHMARK"
ALLOWED_GENERATOR_IMPORTS = {
    "CLOSURE_DIR_REL",
    "CLOSURE_JSON_FILES",
    "EXPECTED_ELIGIBLE",
    "EXPECTED_HISTORICAL_RELEASES",
    "EXPECTED_MACRO_F1",
    "EXPECTED_MODEL_BYTES",
    "EXPECTED_MODEL_SHA",
    "EXPECTED_MODELS",
    "EXPECTED_PAIRS",
    "EXPECTED_RECOVERY_INFERENCE",
    "EXPECTED_V01_F1",
    "EXPECTED_V02_F1",
    "M_B11_DIR_REL",
    "MACHINE_FACTS_BEGIN",
    "MACHINE_FACTS_END",
    "PROPOSED_TAG",
    "REPORT_REL",
    "REQUIRED_M11_LOCK_FILE_ROLES",
    "REQUIRED_M11_REGISTRY_ROLES",
    "SCHEMA",
    "STATUS_LABEL",
}


class MB12ValidationError(Exception):
    """Fail-closed M-B12 validation failure."""


def _raise(code: str) -> None:
    raise MB12ValidationError(code)


def _inspect_no_accessor_or_invoke() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "get_locked_test_recovery_evaluation_dataset",
        "get_locked_test_final_evaluation_dataset",
        "invoke",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "scripts.mmwave_m_b12_phase_b_closure":
            for alias in node.names:
                if alias.name == "*" or alias.name not in ALLOWED_GENERATOR_IMPORTS:
                    _raise(f"VALIDATOR_IMPORTS_GENERATOR:{alias.name}")
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else None
            )
            if name in forbidden:
                _raise(f"M_B12_VALIDATOR_FORBIDDEN_CALL:{name}")


def _validate_checksums(out: Path) -> None:
    checksum_path = out / "checksums.sha256"
    if not checksum_path.is_file():
        _raise("CHECKSUMS_MISSING")
    mapped: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2 or not HEX64.fullmatch(parts[0]):
            _raise(f"CHECKSUM_LINE_INVALID:{line}")
        rel = parts[1]
        if ".." in rel or rel.startswith("/") or "\\" in rel:
            _raise(f"CHECKSUM_UNSAFE_PATH:{rel}")
        if rel in mapped and mapped[rel] != parts[0]:
            _raise(f"CHECKSUM_DUPLICATE_INCONSISTENT:{rel}")
        mapped[rel] = parts[0]
        target = out / rel
        if not target.is_file():
            _raise(f"CHECKSUM_TARGET_MISSING:{rel}")
        if _sha256_file(target) != parts[0]:
            _raise(f"CHECKSUM_MISMATCH:{rel}")
    expected = set(CLOSURE_JSON_FILES)
    if set(mapped) != expected:
        missing = sorted(expected - set(mapped))
        extra = sorted(set(mapped) - expected)
        _raise(f"CHECKSUM_ENTRY_SET_MISMATCH:missing={missing}:extra={extra}")
    if "checksums.sha256" in mapped:
        _raise("CHECKSUM_SELF_HASH")
    if "phase_b_required_role_registry.json" not in mapped:
        _raise("ROLE_REGISTRY_NOT_CHECKSUMMED")
    if "final_report_identity.json" not in mapped:
        _raise("REPORT_IDENTITY_NOT_CHECKSUMMED")


def _walk_strings(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_walk_strings(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_walk_strings(item))
    return found


def _reject_unsafe_paths(payload: Any, *, context: str) -> None:
    for text in _walk_strings(payload):
        if text.startswith("/") or text.startswith("file:") or "\\" in text:
            _raise(f"UNSAFE_PATH:{context}:{text}")
        if ".." in Path(text).parts:
            _raise(f"UNSAFE_PATH:{context}:{text}")


def _normalize_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _is_truthy_claim(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str) and _normalize_token(value) in TRUTHY_TOKENS:
        return True
    return False


def _reject_forbidden_claims(payload: Any, *, context: str) -> None:
    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child = f"{path}.{key}"
                token = _normalize_token(key)
                if token in FORBIDDEN_TRUE_KEY_TOKENS and _is_truthy_claim(value):
                    _raise(f"FORBIDDEN_POSITIVE_CLAIM:{context}:{child}:{value}")
                if isinstance(value, str) and _normalize_token(value) in FORBIDDEN_POSITIVE_VALUE_TOKENS:
                    _raise(f"FORBIDDEN_POSITIVE_CLAIM:{context}:{child}:{value}")
                walk(value, child)
            return
        if isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")
            return
        if isinstance(node, str) and _normalize_token(node) in FORBIDDEN_POSITIVE_VALUE_TOKENS:
            _raise(f"FORBIDDEN_POSITIVE_CLAIM:{context}:{path}:{node}")

    walk(payload, "$")


def _require_equal(actual: Any, expected: Any, code: str) -> None:
    if actual != expected:
        _raise(f"{code}:{actual}!={expected}")


def _require_non_pristine_fields(payload: dict[str, Any], *, context: str) -> None:
    if "result_limitation" in payload:
        _require_equal(payload.get("result_limitation"), RESULT_LIMITATION, f"LIMITATION:{context}")
    if "result_designation" in payload:
        _require_equal(payload.get("result_designation"), RESULT_LIMITATION, f"DESIGNATION:{context}")
    if "result_not_pristine" in payload and payload.get("result_not_pristine") is not True:
        _raise(f"RESULT_NOT_PRISTINE_FALSE:{context}")


def _fact(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


def inspect_required_roles(
    root: Path,
    role_spec: dict[str, Any],
    m11_registry: dict[str, Any],
) -> dict[str, Any]:
    """Validator-owned M-B11 registry completeness. Does not trust generator PASS flags."""
    expected_registry_roles = {item[0] for item in REQUIRED_M11_REGISTRY_ROLES}
    expected_lock_roles = {item[0] for item in REQUIRED_M11_LOCK_FILE_ROLES}
    expected_paths = {item[0]: item[3] for item in REQUIRED_M11_LOCK_FILE_ROLES}
    for role, _phase, _category in REQUIRED_M11_REGISTRY_ROLES:
        expected_paths.setdefault(role, "")
    roles = list(role_spec.get("roles") or [])
    spec_registry = {str(item.get("artifact_role")) for item in roles if item.get("binding") == "m_b11_registry"}
    spec_lock = {str(item.get("artifact_role")) for item in roles if item.get("binding") == "m_b11_lock_file"}
    if spec_registry != expected_registry_roles:
        _raise(f"REQUIRED_ROLE_SET_DRIFT:registry:{sorted(expected_registry_roles ^ spec_registry)}")
    if spec_lock != expected_lock_roles:
        _raise(f"REQUIRED_ROLE_SET_DRIFT:lock:{sorted(expected_lock_roles ^ spec_lock)}")

    artifacts = list(m11_registry.get("artifacts") or [])
    by_role: dict[str, list[dict[str, Any]]] = {}
    paths_by_role: dict[str, set[str]] = {}
    for item in artifacts:
        role = str(item.get("artifact_role"))
        by_role.setdefault(role, []).append(item)
        paths_by_role.setdefault(role, set()).add(str(item.get("repo_relative_path")))
    duplicates = sorted(role for role, rows in by_role.items() if role in expected_registry_roles and len(rows) != 1)
    ambiguities = sorted(role for role, paths in paths_by_role.items() if role in expected_registry_roles and len(paths) > 1)
    missing = sorted(expected_registry_roles - set(by_role))
    if duplicates:
        _raise(f"DUPLICATE_REQUIRED_ROLE:{duplicates}")
    if ambiguities:
        _raise(f"ROLE_PATH_AMBIGUITY:{ambiguities}")
    if missing:
        _raise(f"MISSING_REQUIRED_ROLE:{missing}")

    spec_by_role = {str(item.get("artifact_role")): item for item in roles}
    sha_mismatches = 0
    for role, _phase, _category in REQUIRED_M11_REGISTRY_ROLES:
        item = by_role[role][0]
        spec = spec_by_role[role]
        rel = require_repo_relative(str(item.get("repo_relative_path")), context=role)
        if rel != spec.get("repo_relative_path"):
            _raise(f"ROLE_PATH_MISMATCH:{role}:{rel}!={spec.get('repo_relative_path')}")
        target = root / rel
        if not target.is_file():
            _raise(f"MISSING_REFERENCED_FILE:{role}:{rel}")
        live = _sha256_file(target)
        if live != item.get("sha256"):
            sha_mismatches += 1
            _raise(f"LIVE_SHA_MISMATCH:{role}")
        if live != spec.get("expected_sha256"):
            _raise(f"FROZEN_SHA_MISMATCH:{role}")
        if item.get("immutable") is not True:
            _raise(f"REQUIRED_ROLE_NOT_IMMUTABLE:{role}")
    for role, _phase, _category, rel in REQUIRED_M11_LOCK_FILE_ROLES:
        spec = spec_by_role[role]
        frozen = require_repo_relative(str(spec.get("repo_relative_path")), context=role)
        if frozen != rel:
            _raise(f"ROLE_PATH_MISMATCH:{role}:{frozen}!={rel}")
        target = root / frozen
        if not target.is_file():
            _raise(f"MISSING_REFERENCED_FILE:{role}:{frozen}")
        live = _sha256_file(target)
        if live != spec.get("expected_sha256"):
            _raise(f"LIVE_SHA_MISMATCH:{role}")
    return {
        "required_role_count": len(expected_registry_roles) + len(expected_lock_roles),
        "present_required_role_count": len(expected_registry_roles) + len(expected_lock_roles),
        "missing": 0,
        "duplicate_roles": 0,
        "role_path_ambiguities": 0,
        "live_sha_mismatches": sha_mismatches,
    }


def _parse_report_facts(text: str) -> dict[str, str]:
    if MACHINE_FACTS_BEGIN not in text or MACHINE_FACTS_END not in text:
        _raise("REPORT_FACTS_TABLE_MISSING")
    block = text.split(MACHINE_FACTS_BEGIN, 1)[1].split(MACHINE_FACTS_END, 1)[0]
    facts: dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        facts[key.strip()] = value.strip()
    return facts


def validate_m_b12(
    root: Path | None = None,
    *,
    closure_dir: Path | None = None,
    m11_registry_path: Path | None = None,
    report_path: Path | None = None,
    skip_m_b11: bool = False,
) -> dict[str, Any]:
    _inspect_no_accessor_or_invoke()
    root = Path(root) if root is not None else ROOT_DIR
    closure_dir = Path(closure_dir) if closure_dir is not None else root / CLOSURE_DIR_REL
    if not closure_dir.is_dir():
        _raise("CLOSURE_DIR_MISSING")
    gates = {
        "m_b11_validator_pass": False,
        "required_registry_roles_complete": False,
        "candidate_contract_exact": False,
        "final_result_contract_exact": False,
        "incident_history_exact": False,
        "claim_boundaries_valid": False,
        "report_sha_valid": False,
        "report_machine_consistent": False,
        "m_b12_checksums_valid": False,
        "no_new_locked_test_access": False,
        "no_new_recovery_access": False,
        "no_new_inference": False,
    }
    if not skip_m_b11:
        m11 = validate_m_b11(root)
        if m11.get("status") != "PASS":
            _raise(f"M_B11_VALIDATOR_NOT_PASS:{m11.get('status')}")
        _require_equal(m11.get("model_sha256"), EXPECTED_MODEL_SHA, "M11_LIVE_MODEL_SHA")
        _require_equal(m11.get("macro_f1"), EXPECTED_MACRO_F1, "M11_LIVE_MACRO_F1")
        _require_equal(m11.get("generator_ledger_analyzer_reused"), False, "M11_ANALYZER_REUSED")
        source_ledger = m11.get("source_ledger") or {}
        _require_equal(source_ledger.get("unique_ids"), EXPECTED_ELIGIBLE, "M11_UNIQUE")
        _require_equal(source_ledger.get("models"), EXPECTED_MODELS, "M11_MODELS")
        _require_equal(source_ledger.get("pairs"), EXPECTED_PAIRS, "M11_PAIRS")
        _require_equal(source_ledger.get("duplicates"), 0, "M11_DUP")
        _require_equal(source_ledger.get("missing"), 0, "M11_MISSING")
        _require_equal(source_ledger.get("unexpected"), 0, "M11_UNEXPECTED")
        _require_equal(source_ledger.get("label_mismatches"), 0, "M11_LABEL")
        _require_equal(source_ledger.get("subject_mismatches"), 0, "M11_SUBJECT")
        _require_equal(source_ledger.get("recording_mismatches"), 0, "M11_RECORDING")
        gates["m_b11_validator_pass"] = True
    else:
        gates["m_b11_validator_pass"] = True
    _validate_checksums(closure_dir)
    gates["m_b12_checksums_valid"] = True

    artifacts = {name: load_json(closure_dir / name) for name in CLOSURE_JSON_FILES}
    for name, payload in artifacts.items():
        _reject_unsafe_paths(payload, context=name)
        if isinstance(payload, dict):
            _reject_forbidden_claims(payload, context=name)
            _require_non_pristine_fields(payload, context=name)
    gates["claim_boundaries_valid"] = True

    identity = artifacts["phase_b_closure_identity.json"]
    predecessor = artifacts["predecessor_gate.json"]
    population = artifacts["source_and_population_summary.json"]
    lineage = artifacts["selected_path_lineage.json"]
    candidate = artifacts["locked_candidate_summary.json"]
    evaluation = artifacts["final_evaluation_summary.json"]
    claims = artifacts["claim_boundary.json"]
    readiness = artifacts["release_readiness_manifest.json"]
    handoff = artifacts["device_domain_handoff.json"]
    evidence = artifacts["immutable_evidence_registry.json"]
    summary = artifacts["phase_b_closure_summary.json"]
    role_spec = artifacts["phase_b_required_role_registry.json"]
    report_identity = artifacts["final_report_identity.json"]

    m11_dir = root / M_B11_DIR_REL
    m11_identity = load_json(m11_dir / "artifact_lock_identity.json")
    m11_metrics = load_json(m11_dir / "final_metric_lock.json")
    m11_sample = load_json(m11_dir / "final_sample_registry_lock.json")
    m11_history = load_json(m11_dir / "recovery_access_history_lock.json")
    m11_model = load_json(m11_dir / "model_artifact_lock.json")
    m11_source = load_json(m11_dir / "source_lineage_lock.json")
    m11_baselines = load_json(m11_dir / "baseline_comparison_lock.json")
    m11_prep = load_json(m11_dir / "preprocessing_lock.json")
    m11_train = load_json(m11_dir / "training_lock.json")
    m11_subjects = load_json(m11_dir / "final_subject_metric_lock.json")
    m11_quant = load_json(m11_dir / "quantization_lock.json")
    m11_final = load_json(m11_dir / "final_evaluation_lock.json")
    registry_path = Path(m11_registry_path) if m11_registry_path is not None else m11_dir / "immutable_artifact_registry.json"
    m11_imm = load_json(registry_path)
    role_stats = inspect_required_roles(root, role_spec, m11_imm)
    gates["required_registry_roles_complete"] = True

    _require_equal(identity.get("schema_version"), SCHEMA, "SCHEMA")
    _require_equal(identity.get("artifact_status"), ARTIFACT_STATUS, "ARTIFACT_STATUS")
    _require_equal(identity.get("result_limitation"), RESULT_LIMITATION, "RESULT_LIMITATION")
    _require_equal(identity.get("candidate_id"), SELECTED_CANDIDATE_ID, "CANDIDATE_ID")
    _require_equal(identity.get("runtime_model_id"), RUNTIME_MODEL_ID, "RUNTIME_MODEL_ID")
    _require_equal(identity.get("class_map"), CLASS_MAP, "CLASS_MAP")
    if identity.get("m_b12_creates_new_model") is not False:
        _raise("CREATES_NEW_MODEL")
    if identity.get("m_c_started") is not False:
        _raise("M_C_STARTED")
    if identity.get("selected_candidate_changed") is not False:
        _raise("CANDIDATE_CHANGED")

    live_model = root / require_repo_relative(SELECTED_TFLITE_REL, context="model")
    _require_equal(_sha256_file(live_model), EXPECTED_MODEL_SHA, "LIVE_MODEL_SHA")
    _require_equal(int(live_model.stat().st_size), EXPECTED_MODEL_BYTES, "LIVE_MODEL_BYTES")
    _require_equal(candidate.get("artifact_status"), m11_identity.get("artifact_status"), "CANDIDATE_STATUS")
    _require_equal(candidate.get("candidate_id"), m11_identity.get("candidate_id"), "CANDIDATE_ID")
    _require_equal(candidate.get("runtime_model_id"), m11_model.get("runtime_model_id"), "RUNTIME_MODEL_ID")
    _require_equal(candidate.get("repo_relative_path"), m11_model.get("repo_relative_path"), "MODEL_PATH")
    _require_equal(candidate.get("sha256"), m11_model.get("sha256"), "CANDIDATE_SHA")
    _require_equal(candidate.get("bytes"), m11_model.get("bytes"), "CANDIDATE_BYTES")
    _require_equal(candidate.get("seed"), m11_model.get("seed"), "CANDIDATE_SEED")
    _require_equal(candidate.get("architecture_id"), m11_model.get("architecture_id"), "ARCHITECTURE")
    _require_equal(candidate.get("training_strategy_id"), m11_train.get("selected_strategy_id"), "TRAINING_STRATEGY")
    _require_equal(candidate.get("preprocessing_profile_id"), m11_prep.get("selected_profile_id"), "PREPROCESSING_ID")
    _require_equal(candidate.get("preprocessing_profile_name"), m11_prep.get("selected_profile_name"), "PREPROCESSING_NAME")
    _require_equal(
        candidate.get("execution_preprocessing_contract_id"),
        m11_prep.get("execution_preprocessing_contract_id"),
        "PREPROCESSING_CONTRACT",
    )
    _require_equal(candidate.get("calibration_profile"), m11_model.get("calibration_profile"), "CALIBRATION_ID")
    _require_equal(candidate.get("input_tensor"), m11_model.get("input_tensor"), "INPUT_TENSOR")
    _require_equal(candidate.get("output_tensor"), m11_model.get("output_tensor"), "OUTPUT_TENSOR")
    _require_equal(candidate.get("strict_int8"), m11_model.get("strict_int8"), "STRICT_INT8")
    _require_equal(candidate.get("flex_ops_present"), m11_model.get("flex_ops_present"), "FLEX")
    _require_equal(candidate.get("select_tf_ops_present"), m11_model.get("select_tf_ops_present"), "SELECT_TF")
    _require_equal(candidate.get("builtin_op_status"), m11_model.get("builtin_op_status"), "BUILTIN")
    _require_equal(candidate.get("class_map"), m11_model.get("class_map"), "CLASS_MAP")
    _require_equal(candidate.get("apnea_is_proxy"), True, "APNEA_PROXY")
    _require_equal(candidate.get("training_strategy_id"), TRAINING_STRATEGY_ID, "TRAINING_CONST")
    _require_equal(candidate.get("preprocessing_profile_id"), PREPROCESSING_PROFILE_ID, "PREP_CONST")
    _require_equal(candidate.get("preprocessing_profile_name"), PREPROCESSING_PROFILE_NAME, "PREP_NAME_CONST")
    _require_equal(candidate.get("execution_preprocessing_contract_id"), EXECUTION_PREPROCESSING_CONTRACT_ID, "PREP_CONTRACT_CONST")
    _require_equal(candidate.get("calibration_profile"), CALIBRATION_ID, "CAL_CONST")
    _require_equal(candidate.get("architecture_id"), ARCHITECTURE_ID, "ARCH_CONST")
    if candidate.get("strict_int8") is not True:
        _raise("STRICT_INT8_FALSE")
    gates["candidate_contract_exact"] = True

    _require_equal(evaluation.get("macro_f1"), EXPECTED_MACRO_F1, "MACRO_F1")
    _require_equal(evaluation.get("macro_f1"), m11_metrics.get("macro_f1"), "M11_MACRO_F1")
    _require_equal(evaluation.get("accuracy"), m11_metrics.get("accuracy"), "ACCURACY")
    _require_equal(evaluation.get("macro_precision"), m11_metrics.get("macro_precision"), "MACRO_PRECISION")
    _require_equal(evaluation.get("macro_recall"), m11_metrics.get("macro_recall"), "MACRO_RECALL")
    _require_equal(evaluation.get("eligible_evaluated"), m11_final.get("eligible_evaluated"), "ELIGIBLE")
    _require_equal(evaluation.get("valid"), m11_final.get("valid"), "VALID")
    _require_equal(evaluation.get("invalid"), m11_final.get("invalid"), "INVALID")
    _require_equal(evaluation.get("tflite_invocations_selected"), m11_final.get("tflite_invocations_selected"), "TFLITE_SELECTED")
    _require_equal(evaluation.get("tflite_invocations_all_models"), m11_final.get("tflite_invocations_all_models"), "TFLITE_ALL")
    for cls_name in ("NORMAL", "RAPID_OR_ABNORMAL", "APNEA"):
        _require_equal(evaluation["per_class"][cls_name], m11_metrics["per_class"][cls_name], f"PER_CLASS_{cls_name}")
    _require_equal(evaluation.get("apnea_proxy"), m11_metrics.get("apnea_proxy"), "APNEA_PROXY_METRICS")
    _require_equal(evaluation.get("confusion_matrix"), m11_metrics.get("confusion_matrix"), "CONFUSION")
    _require_equal(evaluation.get("prediction_distribution"), m11_metrics.get("prediction_distribution"), "PRED_DIST")
    _require_equal(evaluation.get("class_collapse"), m11_metrics.get("class_collapse"), "COLLAPSE")
    _require_equal(evaluation.get("subject_count"), m11_subjects.get("subject_count"), "SUBJECT_COUNT")
    _require_equal(evaluation.get("median_subject_macro_f1"), m11_subjects.get("median_subject_macro_f1"), "SUBJECT_MEDIAN")
    _require_equal(evaluation.get("worst_subject_macro_f1"), m11_subjects.get("worst_subject_macro_f1"), "SUBJECT_WORST")
    _require_equal(evaluation.get("worst_subject_id"), m11_subjects.get("worst_subject_id"), "WORST_ID")
    _require_equal(evaluation.get("input_saturation_ratio"), m11_quant.get("input_saturation_ratio"), "SAT_RATIO")
    _require_equal(evaluation.get("pre_clamp_out_of_range_count"), m11_quant.get("pre_clamp_out_of_range_count"), "SAT_OOR")
    _require_equal(evaluation.get("total_quantized_elements"), m11_quant.get("total_quantized_elements"), "SAT_ELEMENTS")
    _require_equal(evaluation.get("samples_with_any_saturation"), m11_quant.get("samples_with_any_saturation"), "SAT_SAMPLES")
    _require_equal(evaluation.get("worst_sample_saturation_ratio"), m11_quant.get("worst_sample_saturation_ratio"), "SAT_WORST")
    _require_equal(evaluation.get("unique_eligible_window_ids"), m11_sample.get("unique_eligible_window_ids"), "M11_UNIQUE")
    _require_equal(evaluation.get("actual_pairs"), m11_sample.get("actual_pairs"), "M11_PAIRS")
    _require_equal(evaluation.get("cross_model_recording_mismatches"), m11_sample.get("cross_model_recording_mismatches"), "M11_RECORDING")
    _require_equal(evaluation.get("v0_1", {}).get("role"), V01_ROLE, "V01_ROLE")
    _require_equal(evaluation.get("v0_2", {}).get("role"), V02_ROLE, "V02_ROLE")
    _require_equal(evaluation.get("v0_1", {}).get("macro_f1"), m11_baselines["v0_1"]["macro_f1"], "V01_F1")
    _require_equal(evaluation.get("v0_2", {}).get("macro_f1"), m11_baselines["v0_2"]["macro_f1"], "V02_F1")
    _require_equal(evaluation.get("v0_1", {}).get("sha256"), m11_baselines["v0_1"]["sha256"], "V01_SHA")
    _require_equal(evaluation.get("v0_2", {}).get("sha256"), m11_baselines["v0_2"]["sha256"], "V02_SHA")
    _require_equal(evaluation.get("v0_1", {}).get("class_collapse"), m11_baselines["v0_1"]["class_collapse"], "V01_COLLAPSE")
    _require_equal(evaluation.get("v0_2", {}).get("class_collapse"), m11_baselines["v0_2"]["class_collapse"], "V02_COLLAPSE")
    _require_equal(evaluation.get("v0_1_macro_f1"), EXPECTED_V01_F1, "V01_F1_CONST")
    _require_equal(evaluation.get("v0_2_macro_f1"), EXPECTED_V02_F1, "V02_F1_CONST")
    gates["final_result_contract_exact"] = True

    _require_equal(evaluation.get("original_m_b10b_accessor_invocations"), m11_history.get("original_m_b10b_accessor_invocations"), "ORIG_ACCESS")
    _require_equal(evaluation.get("original_m_b10b_payload_releases"), m11_history.get("original_m_b10b_payload_releases"), "ORIG_RELEASE")
    _require_equal(evaluation.get("original_m_b10b_model_inference"), m11_history.get("original_m_b10b_model_inference"), "ORIG_INFER")
    _require_equal(evaluation.get("m_b10r1b_recovery_accessor_invocations"), m11_history.get("m_b10r1b_recovery_accessor_invocations"), "REC_ACCESS")
    _require_equal(evaluation.get("m_b10r1b_recovery_payload_releases"), m11_history.get("m_b10r1b_recovery_payload_releases"), "REC_RELEASE")
    _require_equal(evaluation.get("recovery_model_inference"), m11_history.get("recovery_model_inference"), "REC_INFER")
    _require_equal(evaluation.get("historical_total_payload_releases"), m11_history.get("historical_total_payload_releases"), "HIST_TOTAL")
    _require_equal(evaluation.get("historical_total_payload_releases"), EXPECTED_HISTORICAL_RELEASES, "HIST_CONST")
    _require_equal(evaluation.get("recovery_model_inference"), EXPECTED_RECOVERY_INFERENCE, "REC_INFER_CONST")
    if evaluation.get("rerun") is not False:
        _raise("RERUN_TRUE")
    if evaluation.get("second_recovery") is not False:
        _raise("SECOND_RECOVERY_TRUE")
    if evaluation.get("inference_rerun_in_m_b12") is not False:
        _raise("M12_INFERENCE_RERUN")
    if evaluation.get("new_model_selection_event") is not False:
        _raise("NEW_SELECTION")
    if evaluation.get("result_not_pristine") is not True:
        _raise("RESULT_NOT_PRISTINE_FALSE:final_evaluation_summary.json")
    _require_equal(evaluation.get("result_designation"), RESULT_LIMITATION, "RESULT_DESIGNATION")
    _require_equal(population.get("raw_archive_sha256"), m11_source.get("raw_archive_sha256"), "RAW_SHA")
    gates["incident_history_exact"] = True

    required_phases = {
        "M-B0", "M-B1", "M-B2", "M-B3", "M-B4", "M-B5", "M-B6", "M-B7", "M-B8", "M-B9",
        "M-B10A", "M-B10B", "M-B10R0", "M-B10R1-A", "M-B10R1-B",
    }
    selected_path = lineage.get("selected_path") or {}
    missing_phases = sorted(required_phases - set(selected_path))
    if missing_phases:
        _raise(f"LINEAGE_MISSING:{missing_phases}")
    for key in ("A0", "A1", "A2", "A3", "A4", "A5", "A6"):
        if key not in (lineage.get("a_series") or {}):
            _raise(f"A_SERIES_MISSING:{key}")
    if lineage.get("m_b12", {}).get("begins_m_c") is not False:
        _raise("LINEAGE_BEGINS_MC")
    if lineage.get("m_b12", {}).get("creates_git_tag") is not False:
        _raise("LINEAGE_CREATES_TAG")

    if claims.get("git_tag_created") is not False:
        _raise("GIT_TAG_CREATED")
    if claims.get("github_release_created") is not False:
        _raise("GITHUB_RELEASE_CREATED")
    if claims.get("m_c_started") is not False:
        _raise("CLAIM_MC_STARTED")
    if claims.get("Phase_B_release_ready") is not False:
        _raise("UNQUALIFIED_PHASE_B_RELEASE_TRUE")
    _require_equal(readiness.get("status_label"), STATUS_LABEL, "STATUS_LABEL")
    _require_equal(readiness.get("proposed_release_tag"), PROPOSED_TAG, "PROPOSED_TAG")
    if readiness.get("git_tag_created") is not False or readiness.get("github_release_created") is not False:
        _raise("READINESS_CREATED_RELEASE")
    if readiness.get("do_not_create_tag_or_github_release_in_this_pr") is not True:
        _raise("READINESS_ALLOWS_TAG")
    if handoff.get("m_c_started") is not False:
        _raise("HANDOFF_MC_STARTED")
    if summary.get("new_locked_test_access") != 0:
        _raise("SUMMARY_NEW_ACCESS")
    if summary.get("new_recovery_access") != 0:
        _raise("SUMMARY_NEW_RECOVERY")
    if summary.get("new_model_inference") != 0:
        _raise("SUMMARY_NEW_INFERENCE")
    _require_equal(predecessor.get("new_locked_test_access"), 0, "PRED_LOCKED")
    _require_equal(predecessor.get("new_recovery_access"), 0, "PRED_RECOVERY")
    _require_equal(predecessor.get("new_model_inference"), 0, "PRED_INFER")
    gates["no_new_locked_test_access"] = True
    gates["no_new_recovery_access"] = True
    gates["no_new_inference"] = True
    _require_equal(
        predecessor.get("m_b11_checksums_sha256"),
        _sha256_file(m11_dir / "checksums.sha256"),
        "M11_CHECKSUMS_SHA",
    )

    artifacts_list = evidence.get("artifacts") or []
    if not artifacts_list:
        _raise("EVIDENCE_EMPTY")
    for item in artifacts_list:
        rel = require_repo_relative(str(item.get("repo_relative_path")), context=str(item.get("artifact_role")))
        target = root / rel
        if not target.is_file():
            _raise(f"EVIDENCE_MISSING:{rel}")
        if _sha256_file(target) != item.get("sha256"):
            _raise(f"EVIDENCE_SHA_MISMATCH:{rel}")
        if item.get("immutable") is not True:
            _raise(f"EVIDENCE_NOT_IMMUTABLE:{rel}")

    sensor = load_json(root / SENSOR_LOCK_REL)
    _reject_forbidden_claims(sensor, context="sensor_lock")
    _require_equal(sensor.get("sha256"), EXPECTED_MODEL_SHA, "SENSOR_SHA")
    if sensor.get("deployment_ready") is True:
        _raise("SENSOR_DEPLOYMENT_READY")

    report_rel = require_repo_relative(str(report_identity.get("repo_relative_path")), context="report")
    _require_equal(report_rel, str(REPORT_REL), "REPORT_PATH")
    resolved_report = Path(report_path) if report_path is not None else root / report_rel
    if not resolved_report.is_file():
        _raise("REPORT_MISSING")
    live_report_sha = _sha256_file(resolved_report)
    _require_equal(live_report_sha, report_identity.get("sha256"), "REPORT_SHA")
    _require_equal(int(resolved_report.stat().st_size), report_identity.get("bytes"), "REPORT_BYTES")
    if report_identity.get("generated_from_machine_evidence") is not True:
        _raise("REPORT_NOT_FROM_MACHINE_EVIDENCE")
    gates["report_sha_valid"] = True
    report_text = resolved_report.read_text(encoding="utf-8")
    facts = _parse_report_facts(report_text)
    expected_facts = {
        "candidate_status": ARTIFACT_STATUS,
        "selected_model_sha": candidate["sha256"],
        "result_designation": RESULT_LIMITATION,
        "result_not_pristine": "true",
        "final_accuracy": _fact(evaluation["accuracy"]),
        "final_macro_f1": _fact(evaluation["macro_f1"]),
        "normal_recall": _fact(evaluation["per_class"]["NORMAL"]["recall"]),
        "rapid_recall": _fact(evaluation["per_class"]["RAPID_OR_ABNORMAL"]["recall"]),
        "apnea_recall": _fact(evaluation["per_class"]["APNEA"]["recall"]),
        "apnea_fpr": _fact(evaluation["per_class"]["APNEA"]["fpr"]),
        "v0_1_macro_f1": _fact(evaluation["v0_1_macro_f1"]),
        "v0_2_macro_f1": _fact(evaluation["v0_2_macro_f1"]),
        "original_release": _fact(evaluation["original_m_b10b_payload_releases"]),
        "recovery_release": _fact(evaluation["m_b10r1b_recovery_payload_releases"]),
        "historical_total_release": _fact(evaluation["historical_total_payload_releases"]),
        "mr60_validated": "false",
        "raspberry_pi_validated": "false",
        "deployment_ready": "false",
        "clinical_apnea_validated": "false",
        "intermediate_release_ready": "true",
        "tag_created": "false",
        "github_release_created": "false",
        "m_c_started": "false",
    }
    for key, expected in expected_facts.items():
        if facts.get(key) != expected:
            _raise(f"REPORT_FACT_MISMATCH:{key}:{facts.get(key)}!={expected}")
    gates["report_machine_consistent"] = True

    intermediate = all(gates.values())
    if not intermediate:
        _raise(f"INTERMEDIATE_READY_FALSE:{sorted(name for name, ok in gates.items() if not ok)}")
    if claims.get("phase_b_offline_intermediate_release_ready_after_merge") is not True:
        _raise("CLAIM_INTERMEDIATE_READY_FALSE")
    if claims.get("phase_b_offline_final_report_complete") is not True:
        _raise("REPORT_INCOMPLETE")
    if readiness.get("phase_b_offline_intermediate_release_ready_after_merge") is not True:
        _raise("READINESS_INTERMEDIATE_FALSE")
    if readiness.get("git_tag_created") is not False or claims.get("git_tag_created") is not False:
        _raise("GIT_TAG_CREATED")
    if readiness.get("github_release_created") is not False:
        _raise("GITHUB_RELEASE_CREATED")

    return {
        "status": "PASS",
        "candidate_id": SELECTED_CANDIDATE_ID,
        "model_sha256": EXPECTED_MODEL_SHA,
        "macro_f1": EXPECTED_MACRO_F1,
        "status_label": STATUS_LABEL,
        "result_limitation": RESULT_LIMITATION,
        "phase_b_offline_intermediate_release_ready_after_merge": True,
        "Phase_B_release_ready": False,
        "git_tag_created": False,
        "github_release_created": False,
        "m_c_started": False,
        "new_locked_test_access": 0,
        "new_recovery_access": 0,
        "new_model_inference": 0,
        "role_completeness": role_stats,
        "gates": gates,
        "report_sha256": live_report_sha,
        "source_ledger": {
            "unique_ids": EXPECTED_ELIGIBLE,
            "models": EXPECTED_MODELS,
            "pairs": EXPECTED_PAIRS,
            "duplicates": 0,
            "missing": 0,
            "unexpected": 0,
            "label_mismatches": 0,
            "subject_mismatches": 0,
            "recording_mismatches": 0,
        },
    }


def main() -> int:
    try:
        result = validate_m_b12()
    except MB12ValidationError as exc:
        print(f"M-B12 VALIDATION FAIL: {exc}", file=sys.stderr)
        return 1
    print("M-B12 VALIDATION PASS")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
