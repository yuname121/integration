#!/usr/bin/env python3
"""Fail-closed forensic validator for the M-B10B count-semantics incident.

This validator is post-access-evidence-only. It never instantiates
``PhaseBAccessGuard`` and never calls
``get_locked_test_final_evaluation_dataset``. Accessor semantics are verified
only by static AST/source inspection.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR_REL = Path("datasets/mmwave/manifests/M-B10B_locked_test_final_evaluation")
M_B10A_DIR_REL = Path("datasets/mmwave/manifests/M-B10A_candidate_selection_setup")
A6_DIR_REL = Path("datasets/mmwave/manifests/a6_full_conversion")
ACCESSOR_REL = Path("scripts/mmwave_phase_b_access.py")
RUNNER_REL = Path("scripts/mmwave_m_b10b_final_eval.py")
REPORT_REL = Path("docs/reports/20260812_Codex_M-B10B_One_Time_Locked_Test_Final_Evaluation_01.md")

ROOT_CAUSE_ID = "PRETEST_CONTRACT_COUNT_SEMANTICS_CONFLATION"
RUNTIME_DETECTION_CODE = "M-B10B_LOCKED_SPLIT_IDENTITY_MISMATCH"
FROZEN_M_B10A_CONTRACT_SHA256 = "ba6429ecfe685de1807ec85b55e697ee12e24138e6b96e94715b0a1a6b19e0f7"
SELECTED_CANDIDATE_ID = "M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120"
SELECTED_MODEL_ID = "M-B3_CONV1D_GAP_BASELINE_seed42_M-B6_STRICT_INT8"

NOT_GENERATED_MARKERS = (
    "NOT_GENERATED_DUE_TO_LOCKED_SPLIT_IDENTITY_MISMATCH",
    "NOT_GENERATED",
)


class MB10BIncidentValidationError(RuntimeError):
    """Raised when M-B10B incident-truth evidence fails closed."""


def _raise(message: str) -> None:
    raise MB10BIncidentValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - fail closed on any parse issue
        _raise(f"JSON_PARSE_ERROR:{path.as_posix()}:{exc}")


def _hex_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value.lower()))


def _safe_relative(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or relative.startswith("~") or "file://" in relative:
        _raise(f"ABSOLUTE_OR_TRAVERSAL_PATH:{relative}")
    return path


def _validate_machine_paths(out: Path) -> None:
    for path in out.iterdir():
        if path.suffix not in {".json", ".jsonl", ".sha256"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "/Users/" in text or "/private/" in text or "file://" in text or "\\\\" in text:
            _raise(f"LOCAL_ABSOLUTE_PATH:{path.name}")


def _validate_checksums(out: Path) -> None:
    path = out / "checksums.sha256"
    if not path.is_file():
        _raise("CHECKSUM_MANIFEST_MISSING")
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not _hex_digest(parts[0]):
            _raise(f"CHECKSUM_SYNTAX:{line_number}")
        digest, relative = parts[0].lower(), parts[1].strip()
        _safe_relative(relative)
        if relative in seen:
            _raise(f"CHECKSUM_DUPLICATE:{relative}")
        seen.add(relative)
        target = out / relative
        if target.parent.resolve() != out.resolve() or not target.is_file():
            _raise(f"CHECKSUM_TARGET_INVALID:{relative}")
        if sha256_file(target) != digest:
            _raise(f"CHECKSUM_MISMATCH:{relative}")
    if "incident_root_cause.json" not in seen:
        _raise("CHECKSUM_MISSING_INCIDENT_ROOT_CAUSE")
    actual = {item.name for item in out.iterdir() if item.is_file() and item.name != "checksums.sha256"}
    if seen != actual:
        _raise(f"CHECKSUM_COVERAGE_MISMATCH:missing={sorted(actual - seen)}:extra={sorted(seen - actual)}")


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _static_verify_accessor_semantics(root: Path) -> dict[str, Any]:
    source_path = root / ACCESSOR_REL
    if not source_path.is_file():
        _raise("ACCESSOR_SOURCE_MISSING")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    final_fn: ast.FunctionDef | None = None
    split_fn: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "PhaseBAccessGuard":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "get_locked_test_final_evaluation_dataset":
                    final_fn = item
                if isinstance(item, ast.FunctionDef) and item.name == "_get_split_dataset":
                    split_fn = item
    if final_fn is None or split_fn is None:
        _raise("ACCESSOR_METHODS_MISSING")

    calls_split = False
    include_ambiguous_false = False
    for node in ast.walk(final_fn):
        if isinstance(node, ast.Call) and _call_name(node.func) == "_get_split_dataset":
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "LOCKED_TEST":
                calls_split = True
            for keyword in node.keywords:
                if keyword.arg == "include_ambiguous" and isinstance(keyword.value, ast.Constant) and keyword.value.value is False:
                    include_ambiguous_false = True
    if not calls_split or not include_ambiguous_false:
        _raise("FINAL_ACCESSOR_DOES_NOT_EXCLUDE_AMBIGUOUS")

    excludes_ambiguous = False
    for node in ast.walk(split_fn):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        # Match: if not include_ambiguous and w["assignment_status"] == "AMBIGUOUS"
        if not isinstance(test, ast.BoolOp) or not isinstance(test.op, ast.And) or len(test.values) != 2:
            continue
        left, right = test.values
        left_ok = isinstance(left, ast.UnaryOp) and isinstance(left.op, ast.Not) and isinstance(left.operand, ast.Name) and left.operand.id == "include_ambiguous"
        right_ok = False
        if isinstance(right, ast.Compare) and len(right.ops) == 1 and isinstance(right.ops[0], ast.Eq):
            if (
                isinstance(right.left, ast.Subscript)
                and isinstance(right.left.slice, ast.Constant)
                and right.left.slice.value == "assignment_status"
                and isinstance(right.comparators[0], ast.Constant)
                and right.comparators[0].value == "AMBIGUOUS"
            ):
                right_ok = True
        if left_ok and right_ok:
            for body_item in node.body:
                if isinstance(body_item, ast.Continue):
                    excludes_ambiguous = True
                    break
    if not excludes_ambiguous:
        _raise("SPLIT_DATASET_AMBIGUOUS_EXCLUSION_MISSING")

    # Ensure this validator source itself never invokes the final accessor.
    validator_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=__file__)
    for node in ast.walk(validator_tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in {"get_locked_test_final_evaluation_dataset", "PhaseBAccessGuard"}:
                _raise("INCIDENT_VALIDATOR_CALLS_FINAL_ACCESSOR")

    return {
        "accessor_include_ambiguous": False,
        "ambiguous_exclusion_verified_from_source": True,
        "accessor_behavior_classification": "EXPECTED_EXISTING_ACCESSOR_BEHAVIOR",
    }


def _validate_a6_counts(root: Path) -> dict[str, Any]:
    summary = _load(root / A6_DIR_REL / "a6_summary.json")
    distribution = _load(root / A6_DIR_REL / "full_split_distribution.json")
    labels = _load(root / A6_DIR_REL / "full_label_distribution.json")
    subjects = summary.get("split_subject_distribution", {}).get("LOCKED_TEST")
    windows = summary.get("split_window_distribution", {}).get("LOCKED_TEST")
    eligible = distribution.get("eligibility_counts", {}).get("locked_test_evaluation_eligible")
    ambiguous = labels.get("split_label_breakdown", {}).get("LOCKED_TEST", {}).get("AMBIGUOUS")
    if subjects != 16:
        _raise(f"A6_LOCKED_TEST_SUBJECTS:{subjects}")
    if windows != 88:
        _raise(f"A6_LOCKED_TEST_WINDOWS:{windows}")
    if eligible != 75:
        _raise(f"A6_LOCKED_TEST_ELIGIBLE:{eligible}")
    if ambiguous != 13 or windows - eligible != 13:
        _raise(f"A6_LOCKED_TEST_EXCLUDED:{ambiguous}:{windows}:{eligible}")
    if distribution.get("window_counts", {}).get("LOCKED_TEST") != 88:
        _raise("A6_DISTRIBUTION_WINDOW_MISMATCH")
    return {
        "subjects": subjects,
        "windows": windows,
        "eligible": eligible,
        "difference": windows - eligible,
        "ambiguous": ambiguous,
    }


def _validate_m_b10a_frozen(root: Path, out: Path) -> None:
    contract_path = root / M_B10A_DIR_REL / "locked_test_evaluation_contract.json"
    selected_path = root / M_B10A_DIR_REL / "selected_candidate_pretest.json"
    if not contract_path.is_file() or not selected_path.is_file():
        _raise("M_B10A_ARTIFACTS_MISSING")
    digest = sha256_file(contract_path)
    if digest != FROZEN_M_B10A_CONTRACT_SHA256:
        _raise(f"M_B10A_CONTRACT_HASH_CHANGED:{digest}")
    frozen = _load(out / "frozen_contract_identity.json")
    if frozen.get("m_b10a_contract_sha256") != FROZEN_M_B10A_CONTRACT_SHA256:
        _raise("FROZEN_CONTRACT_IDENTITY_HASH_MISMATCH")
    contract = _load(contract_path)
    if contract.get("structural_window_count") != 88 or contract.get("subject_count") != 16:
        _raise("M_B10A_STRUCTURAL_IDENTITY_CHANGED")
    if contract.get("evaluation_passes") != 1:
        _raise("M_B10A_ONE_TIME_POLICY_CHANGED")
    selected = _load(selected_path)
    if selected.get("seed") != 42 or selected.get("candidate_id") != SELECTED_CANDIDATE_ID:
        _raise("M_B10A_SELECTED_CANDIDATE_CHANGED")
    if selected.get("model_id") != SELECTED_MODEL_ID:
        _raise("M_B10A_SELECTED_MODEL_CHANGED")


def _validate_runner_expectation_preserved(root: Path) -> None:
    text = (root / RUNNER_REL).read_text(encoding="utf-8")
    if "len(windows) != 88" not in text or "len(provenance) != 88" not in text or "(88, 300)" not in text:
        _raise("FROZEN_RUNNER_EXPECTED_88_ROWS_MISSING")
    if "!= 264" not in text and "!= 88 * 3" not in text:
        # runner uses sum(inference_counts.values()) != 264
        if "!= 264" not in text:
            _raise("FROZEN_RUNNER_EXPECTED_264_INFERENCES_MISSING")
    # Closure must not have rewritten the formal runner to expect 75.
    if "len(windows) != 75" in text:
        _raise("FROZEN_RUNNER_RETROACTIVELY_CHANGED_TO_75")


def _validate_no_completed_performance(out: Path) -> None:
    placeholders = [
        "locked_test_registry.json",
        "metrics_by_model.json",
        "per_class_metrics.json",
        "subject_level_metrics.json",
        "model_comparison.json",
        "selected_candidate_final_test_result.json",
    ]
    for name in placeholders:
        payload = _load(out / name)
        if payload.get("results_available") is True:
            _raise(f"PERFORMANCE_ARTIFACT_CLAIMS_AVAILABLE:{name}")
        status = str(payload.get("status", ""))
        if name == "locked_test_registry.json":
            if payload.get("samples") not in ([], None) and payload.get("samples"):
                _raise("REGISTRY_SAMPLES_PRESENT")
            if not any(marker in status for marker in NOT_GENERATED_MARKERS):
                _raise("REGISTRY_NOT_MARKED_NOT_GENERATED")
        elif not any(marker in status for marker in NOT_GENERATED_MARKERS) and payload.get("results_available") is not False:
            _raise(f"PERFORMANCE_ARTIFACT_NOT_MARKED_ABSENT:{name}")
    ledger = out / "locked_test_sample_predictions.jsonl"
    if ledger.is_file() and any(line.strip() for line in ledger.read_text(encoding="utf-8").splitlines()):
        _raise("PREDICTION_LEDGER_HAS_ROWS")


def _validate_incident(root: Path, out: Path, a6: dict[str, Any], accessor: dict[str, Any]) -> dict[str, Any]:
    incident = _load(out / "incident_root_cause.json")
    audit = _load(out / "one_time_access_audit.json")
    summary = _load(out / "m_b10b_summary.json")
    consumption = _load(out / "test_split_consumption_record.json")
    exceptions = _load(out / "exceptions.json")

    if incident.get("schema_version") != "M-B10B_INCIDENT_ROOT_CAUSE_V1":
        _raise("INCIDENT_SCHEMA_INVALID")
    if incident.get("phase_id") != "M-B10B":
        _raise("INCIDENT_PHASE_INVALID")
    if incident.get("incident_status") != "INCIDENT_ROOT_CAUSE_CLOSED":
        _raise("INCIDENT_STATUS_INVALID")
    if incident.get("runtime_detection_code") != RUNTIME_DETECTION_CODE:
        _raise("INCIDENT_RUNTIME_DETECTION_REWRITTEN")
    if incident.get("root_cause_id") != ROOT_CAUSE_ID:
        _raise("INCIDENT_ROOT_CAUSE_ID_INVALID")
    if incident.get("a6_total_locked_test_windows") != 88:
        _raise("INCIDENT_TOTAL_WINDOWS_INVALID")
    if incident.get("a6_locked_test_evaluation_eligible_windows") != 75:
        _raise("INCIDENT_ELIGIBLE_WINDOWS_INVALID")
    if incident.get("count_difference") != 13:
        _raise("INCIDENT_DIFFERENCE_INVALID")
    if incident.get("a6_total_locked_test_windows") == 75:
        _raise("INCIDENT_CONFLATES_TOTAL_WITH_ELIGIBLE")
    if incident.get("a6_locked_test_evaluation_eligible_windows") == 88:
        _raise("INCIDENT_CONFLATES_ELIGIBLE_WITH_TOTAL")
    if incident.get("accessor_include_ambiguous") is not False:
        _raise("INCIDENT_ACCESSOR_INCLUDE_AMBIGUOUS_INVALID")
    if incident.get("accessor_behavior_classification") != "EXPECTED_EXISTING_ACCESSOR_BEHAVIOR":
        _raise("INCIDENT_CLAIMS_ACCESSOR_MALFUNCTION")
    if incident.get("dataset_corruption_evidence") is not False:
        _raise("INCIDENT_CLAIMS_DATASET_CORRUPTION")
    if incident.get("split_mutation_evidence") is not False:
        _raise("INCIDENT_CLAIMS_SPLIT_MUTATION")
    if incident.get("accessor_malfunction_evidence") is not False:
        _raise("INCIDENT_CLAIMS_ACCESSOR_MALFUNCTION_EVIDENCE")
    if incident.get("model_failure_evidence") is not False:
        _raise("INCIDENT_CLAIMS_MODEL_FAILURE")
    if incident.get("formal_accessor_invocations") != 1:
        _raise("INCIDENT_ACCESSOR_INVOCATIONS_INVALID")
    if incident.get("second_accessor_invocation") is not False:
        _raise("INCIDENT_SECOND_ACCESSOR_INVALID")
    if incident.get("rows_returned") != 75:
        _raise("INCIDENT_ROWS_RETURNED_INVALID")
    if incident.get("model_inference_invocations") != 0:
        _raise("INCIDENT_MODEL_INFERENCE_INVALID")
    if incident.get("metrics_generated") is not False or incident.get("predictions_generated") is not False or incident.get("registry_generated") is not False:
        _raise("INCIDENT_CLAIMS_GENERATED_RESULTS")
    if incident.get("scientific_final_performance_available") is not False:
        _raise("INCIDENT_CLAIMS_PERFORMANCE_AVAILABLE")
    if incident.get("returned_subject_count") != "NOT_RECORDED_BEFORE_ABORT":
        _raise("INCIDENT_FABRICATES_RETURNED_SUBJECT_COUNT")
    if incident.get("locked_test_consumed") is not True:
        _raise("INCIDENT_CLAIMS_NOT_CONSUMED")
    if incident.get("rerun_performed") is not False:
        _raise("INCIDENT_AUTHORIZES_OR_CLAIMS_RERUN")
    if incident.get("recovery_evaluation_authorized") is not False:
        _raise("INCIDENT_AUTHORIZES_RECOVERY")
    if incident.get("locked_test_reopen_authorized") is not False:
        _raise("INCIDENT_AUTHORIZES_REOPEN")
    if incident.get("m_b11_authorized") is not False:
        _raise("INCIDENT_AUTHORIZES_M_B11")
    if incident.get("m_b10a_contract_frozen") is not True or incident.get("m_b10a_contract_modified_after_access") is not False:
        _raise("INCIDENT_M_B10A_MUTATION_CLAIM_INVALID")
    if incident.get("required_followup") != "SEPARATE_HOLDOUT_REUSE_OR_NEW_HOLDOUT_POLICY_REVIEW":
        _raise("INCIDENT_FOLLOWUP_INVALID")
    if incident.get("recovery_policy_status") != "NOT_AUTHORIZED_REQUIRES_SEPARATE_REVIEW":
        _raise("INCIDENT_RECOVERY_POLICY_INVALID")

    crosswalk = incident.get("count_semantics_crosswalk")
    if not isinstance(crosswalk, dict):
        _raise("INCIDENT_CROSSWALK_MISSING")
    structural = crosswalk.get("LOCKED_TEST_STRUCTURAL_POPULATION", {})
    supervised = crosswalk.get("LOCKED_TEST_SUPERVISED_EVALUATION_POPULATION", {})
    excluded = crosswalk.get("EXCLUDED_FROM_SUPERVISED_EVALUATION", {})
    if structural.get("subjects") != 16 or structural.get("windows") != 88:
        _raise("INCIDENT_CROSSWALK_STRUCTURAL_INVALID")
    if supervised.get("windows") != 75:
        _raise("INCIDENT_CROSSWALK_SUPERVISED_INVALID")
    if excluded.get("windows") != 13:
        _raise("INCIDENT_CROSSWALK_EXCLUDED_INVALID")
    if supervised.get("returned_count_identity_claim") != "RETURNED_COUNT_MATCHES_PREEXISTING_A6_ELIGIBILITY_COUNT":
        _raise("INCIDENT_OVERCLAIMS_ROW_IDENTITY")
    allowed = incident.get("allowed_claims")
    if isinstance(allowed, list) and "FULL_75_ROW_IDENTITY_VERIFIED" in allowed:
        _raise("INCIDENT_FORBIDDEN_FULL_IDENTITY_CLAIM")
    if incident.get("returned_count_identity_claim") == "FULL_75_ROW_IDENTITY_VERIFIED":
        _raise("INCIDENT_FORBIDDEN_FULL_IDENTITY_CLAIM")
    if incident.get("full_75_row_identity_verified") is True:
        _raise("INCIDENT_FORBIDDEN_FULL_IDENTITY_CLAIM")

    # Historical access audit must remain consistent.
    if audit.get("accessor_invocation_count") != 1 or audit.get("second_accessor_invocation") is not False:
        _raise("ACCESS_AUDIT_INVOCATION_INVALID")
    if audit.get("structural_rows_returned") != 75 or audit.get("completed_model_inference_invocations") != 0:
        _raise("ACCESS_AUDIT_RESULT_INVALID")
    if audit.get("failure") != RUNTIME_DETECTION_CODE or audit.get("no_rerun_performed") is not True:
        _raise("ACCESS_AUDIT_DETECTION_REWRITTEN")
    if audit.get("post_access_status") != "INCOMPLETE_NO_RERUN" or audit.get("access_consumed") is not True:
        _raise("ACCESS_AUDIT_STATUS_INVALID")
    if audit.get("actual_structural_subjects") != "NOT_RECORDED_BEFORE_ABORT":
        _raise("ACCESS_AUDIT_SUBJECT_COUNT_FABRICATED")

    if summary.get("status") != "INCOMPLETE_NO_RERUN":
        _raise("SUMMARY_STATUS_REWRITTEN_TO_PASS")
    if summary.get("runtime_detection_code") != RUNTIME_DETECTION_CODE:
        _raise("SUMMARY_RUNTIME_DETECTION_REWRITTEN")
    if summary.get("forensic_root_cause") != ROOT_CAUSE_ID:
        _raise("SUMMARY_ROOT_CAUSE_INVALID")
    if summary.get("forensic_status") != "INCIDENT_ROOT_CAUSE_CLOSED":
        _raise("SUMMARY_FORENSIC_STATUS_INVALID")
    if summary.get("performance_result") != "NOT_AVAILABLE":
        _raise("SUMMARY_PERFORMANCE_AVAILABLE")
    if summary.get("final_accessor_invocations") != 1 or summary.get("model_inference_invocations") != 0:
        _raise("SUMMARY_INVOCATION_COUNTS_INVALID")
    if summary.get("locked_test_consumed") is not True or summary.get("m_b11_started") is not False:
        _raise("SUMMARY_CONSUMPTION_OR_M_B11_INVALID")
    if summary.get("selected_candidate_unchanged") is not True or summary.get("seed43_evaluated") is not False or summary.get("seed44_evaluated") is not False:
        _raise("SUMMARY_CANDIDATE_MUTATION")

    if consumption.get("status") != "LOCKED_TEST_CONSUMED_FOR_FINAL_PHASE_B_EVALUATION_INCOMPLETE":
        _raise("CONSUMPTION_STATUS_INVALID")
    if consumption.get("must_not_reuse_for_phase_b_model_selection") is not True:
        _raise("CONSUMPTION_REUSE_FOR_SELECTION_ALLOWED")
    if consumption.get("no_rerun_performed") is not True:
        _raise("CONSUMPTION_RERUN_CLAIM_INVALID")
    if consumption.get("status") in {"LOCKED_TEST_NOT_USED", "LOCKED_TEST_PRISTINE"}:
        _raise("CONSUMPTION_CLAIMS_PRISTINE")

    if exceptions.get("code") != RUNTIME_DETECTION_CODE:
        _raise("EXCEPTIONS_RUNTIME_DETECTION_REWRITTEN")
    if exceptions.get("forensic_root_cause") != ROOT_CAUSE_ID:
        _raise("EXCEPTIONS_ROOT_CAUSE_MISSING")

    if a6["windows"] != incident["a6_total_locked_test_windows"] or a6["eligible"] != incident["a6_locked_test_evaluation_eligible_windows"]:
        _raise("INCIDENT_A6_MISMATCH")
    if accessor["accessor_include_ambiguous"] is not False:
        _raise("ACCESSOR_STATIC_INCLUDE_AMBIGUOUS")

    report = root / REPORT_REL
    text = report.read_text(encoding="utf-8") if report.is_file() else ""
    required_phrases = [
        "M-B10B_ONE_TIME_EVALUATION_INCOMPLETE_NO_RERUN",
        ROOT_CAUSE_ID,
        RUNTIME_DETECTION_CODE,
        "EXPECTED_EXISTING_ACCESSOR_BEHAVIOR",
        "`88` was the entire structural LOCKED_TEST split",
        "`75` was the pre-existing pure-class evaluation-eligible population",
        "FINAL_PERFORMANCE_NOT_AVAILABLE_PREINFERENCE_ABORT",
    ]
    for phrase in required_phrases:
        if phrase not in text:
            _raise(f"REPORT_MISSING_PHRASE:{phrase}")
    forbidden_positive = [
        "ACCESSOR_BROKEN",
        "DATASET_CORRUPTED",
        "SPLIT_MUTATED",
        "RECOVERY_AUTHORIZED",
        "M-B11_READY",
        "LOCKED_TEST_STILL_PRISTINE",
        "M-B10B_FINAL_PERFORMANCE_COMPLETE",
        "FULL_75_ROW_IDENTITY_VERIFIED",
    ]
    for phrase in forbidden_positive:
        # Allow explicit negations in claim-boundary tables / forensic notes.
        for match in re.finditer(re.escape(phrase), text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            line = text[line_start:line_end if line_end != -1 else len(text)]
            lowered = line.lower()
            if (
                ": NO" in line
                or "NOT CLAIMED" in line
                or "Not claimed" in line
                or "not claimed" in lowered
                or "NO —" in line
                or "must NOT" in line
            ):
                continue
            _raise(f"REPORT_FORBIDDEN_POSITIVE_CLAIM:{phrase}")
    if "recovery evaluation authorized: YES" in text.lower() or "LOCKED_TEST reopen authorized: YES" in text:
        _raise("REPORT_AUTHORIZES_RECOVERY")

    return incident


def validate_m_b10b_incident_artifacts(root_dir: Path = ROOT_DIR, output_dir: Path | None = None) -> dict[str, Any]:
    root = root_dir.resolve()
    out = (output_dir or root / OUT_DIR_REL).resolve()
    if not out.is_dir():
        _raise("INCIDENT_OUTPUT_DIRECTORY_MISSING")
    _validate_checksums(out)
    _validate_machine_paths(out)
    a6 = _validate_a6_counts(root)
    accessor = _static_verify_accessor_semantics(root)
    _validate_m_b10a_frozen(root, out)
    _validate_runner_expectation_preserved(root)
    _validate_no_completed_performance(out)
    incident = _validate_incident(root, out, a6, accessor)
    return {
        "validation_status": "PASS",
        "phase_id": "M-B10B",
        "mode": "INCIDENT_TRUTH_CLOSURE",
        "incident_status": incident.get("incident_status"),
        "runtime_detection_code": RUNTIME_DETECTION_CODE,
        "forensic_root_cause": ROOT_CAUSE_ID,
        "a6": a6,
        "accessor": accessor,
        "final_accessor_invocations": 1,
        "model_inference_invocations": 0,
        "scientific_final_performance_available": False,
        "recovery_evaluation_authorized": False,
        "locked_test_reopen_authorized": False,
        "m_b11_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    del argv  # no CLI options that can reopen LOCKED_TEST
    try:
        result = validate_m_b10b_incident_artifacts()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except MB10BIncidentValidationError as exc:
        print(f"M-B10B incident validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
