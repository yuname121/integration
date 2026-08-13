#!/usr/bin/env python3
"""Fail-closed validator for the SafeNest M-B10B one-time final evidence.

``--pre-access`` validates only frozen contracts and VALIDATION smoke evidence.
The post-access mode validates the stored registry/ledger and independently
recomputes every metric.  Neither mode calls the final LOCKED_TEST accessor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.mmwave_m_b10b_final_eval import (  # noqa: E402
    CLASS_MAP,
    FINAL_OUTPUT_FILES,
    LABELS,
    M_B10A_DIR_REL,
    MODEL_IDS_FORBIDDEN,
    OUT_DIR_REL,
    PREACCESS_FILES,
    ROLES,
    MB10BExecutionError,
    _comparison,
    _quantization_audit,
    _write_checksums,
    build_pre_access_gate,
    load_json,
    metric_bundle,
    repo_path,
    sha256_file,
    subject_metrics,
    validate_contract_policy,
    validate_frozen_models,
)


class MB10BValidationError(RuntimeError):
    """Raised when M-B10B evidence fails closed."""


def _raise(message: str) -> None:
    raise MB10BValidationError(message)


def _load(path: Path) -> Any:
    try:
        return load_json(path)
    except Exception as exc:
        _raise(f"JSON_PARSE_ERROR:{path.name}:{exc}")


def _hex_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value.lower()))


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float, np.number)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(key) and _finite(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _safe_relative(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or relative.startswith("~") or "file://" in relative:
        _raise(f"ABSOLUTE_OR_TRAVERSAL_PATH:{relative}")
    return path


INCOMPLETE_FORENSIC_EXTRA_FILES = {"incident_root_cause.json"}


def _validate_checksums(out: Path, *, extra_allowed: set[str] | None = None) -> None:
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
    expected = (FINAL_OUTPUT_FILES | (extra_allowed or set())) - {"checksums.sha256"}
    if seen != expected:
        _raise(f"CHECKSUM_COVERAGE:missing={sorted(expected - seen)}:unexpected={sorted(seen - expected)}")
    actual = {item.name for item in out.iterdir() if item.is_file() and item.name != "checksums.sha256"}
    if actual != expected:
        _raise(f"UNREGISTERED_OUTPUT_FILES:{sorted(actual ^ expected)}")


def _validate_machine_paths(out: Path) -> None:
    for path in out.iterdir():
        if path.suffix not in {".json", ".jsonl", ".sha256"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "/Users/" in text or "/private/" in text or "file://" in text or "\\\\" in text:
            _raise(f"LOCAL_ABSOLUTE_PATH:{path.name}")


def _validate_pre_access(root: Path, out: Path) -> dict[str, Any]:
    try:
        from scripts.validate_mmwave_m_b10a import validate_m_b10a_artifacts

        validate_m_b10a_artifacts(root_dir=root, run_upstream=False)
    except Exception as exc:
        _raise(f"M-B10A_VALIDATOR_FAILED:{exc}")
    try:
        specs = validate_frozen_models(root)
    except MB10BExecutionError as exc:
        _raise(f"FROZEN_MODEL_GATE:{exc}")
    if not out.is_dir():
        _raise("PREACCESS_EVIDENCE_DIRECTORY_MISSING")
    files = {item.name for item in out.iterdir() if item.is_file()}
    if not files <= PREACCESS_FILES:
        _raise(f"PREACCESS_RESULT_ARTIFACT_PRESENT:{sorted(files - PREACCESS_FILES)}")
    if not (out / "pre_access_gate.json").is_file() or not (out / "frozen_contract_identity.json").is_file():
        _raise("PREACCESS_GATE_FILES_MISSING")
    gate = _load(out / "pre_access_gate.json")
    identity = _load(out / "frozen_contract_identity.json")
    if gate.get("status") != "PASS" or gate.get("phase_id") != "M-B10B":
        _raise("PREACCESS_GATE_NOT_PASS")
    for key in ("final_accessor_previous_calls", "previous_tensor_accesses", "previous_label_accesses", "previous_prediction_accesses", "previous_metric_accesses"):
        if gate.get(key) != 0:
            _raise(f"PREACCESS_NONZERO:{key}")
    if gate.get("exact_models") != [spec["model_id"] for spec in specs] or gate.get("class_map") != CLASS_MAP or gate.get("model_output_shape") != [1, 3]:
        _raise("PREACCESS_FROZEN_MATRIX_MISMATCH")
    if gate.get("evaluation_passes") != 1 or gate.get("post_test_tuning_prohibited") is not True or gate.get("authorization_present") is not True:
        _raise("PREACCESS_POLICY_MISMATCH")
    if gate.get("no_final_result_artifacts_present") is not True:
        _raise("PREACCESS_RESULT_ARTIFACT_CLAIM_MISMATCH")
    if identity.get("no_locked_test_data_loaded") is not True or identity.get("locked_test_structural_identity") != {"subjects": 16, "windows": 88}:
        _raise("PREACCESS_LOCKED_IDENTITY_MISMATCH")
    if gate.get("validation_smoke", {}).get("population") != "VALIDATION_ONLY" or gate.get("validation_smoke", {}).get("all_finite") is not True:
        _raise("PREACCESS_VALIDATION_SMOKE_FAILED")
    return {"validation_status": "PASS", "phase_id": "M-B10B", "mode": "PRE_ACCESS", "final_accessor_calls": 0, "models": [spec["model_id"] for spec in specs]}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            _raise(f"LEDGER_JSONL_PARSE:{line_number}:{exc}")
        if not isinstance(row, dict) or not _finite(row):
            _raise(f"LEDGER_ROW_INVALID:{line_number}")
        rows.append(row)
    return rows


def _validate_registry(registry: dict[str, Any]) -> list[dict[str, Any]]:
    if registry.get("split") != "LOCKED_TEST" or registry.get("ordered") is not True or registry.get("sample_count") != 88 or registry.get("subject_count") != 16:
        _raise("LOCKED_TEST_REGISTRY_IDENTITY")
    samples = registry.get("samples")
    if not isinstance(samples, list) or len(samples) != 88:
        _raise("LOCKED_TEST_REGISTRY_COUNT")
    subjects = set()
    for index, sample in enumerate(samples):
        if sample.get("order") != index or sample.get("split") != "LOCKED_TEST" or sample.get("canonical_sample_index") is None:
            _raise(f"LOCKED_TEST_REGISTRY_ORDER:{index}")
        if sample.get("true_class_index") not in range(3) or sample.get("true_class") != CLASS_MAP[str(sample["true_class_index"])]:
            _raise(f"LOCKED_TEST_REGISTRY_LABEL:{index}")
        if not sample.get("window_id") or not sample.get("subject_id"):
            _raise(f"LOCKED_TEST_REGISTRY_ID_FIELDS:{index}")
        subjects.add(sample["subject_id"])
    if len(subjects) != 16:
        _raise("LOCKED_TEST_REGISTRY_SUBJECT_COUNT")
    return samples


def _validate_ledger(root: Path, records: list[dict[str, Any]], registry: list[dict[str, Any]], specs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    expected_ids = [spec["model_id"] for spec in specs]
    if len(records) != 264:
        _raise(f"LEDGER_ROW_COUNT:{len(records)}")
    by_model: dict[str, list[dict[str, Any]]] = {model_id: [] for model_id in expected_ids}
    seen: set[tuple[int, str]] = set()
    details = {spec["model_id"]: spec["inspected"] for spec in specs}
    for row in records:
        model_id = row.get("model_id")
        if model_id not in by_model or model_id in MODEL_IDS_FORBIDDEN:
            _raise(f"UNAUTHORIZED_MODEL:{model_id}")
        order = row.get("order")
        if not isinstance(order, int) or order not in range(88):
            _raise("LEDGER_SAMPLE_ORDER_INVALID")
        key = (order, model_id)
        if key in seen:
            _raise(f"LEDGER_DUPLICATE:{key}")
        seen.add(key)
        sample = registry[order]
        for field in ("canonical_sample_index", "window_id", "subject_id", "recording_id", "true_class_index", "true_class", "split"):
            if row.get(field) != sample.get(field):
                _raise(f"LEDGER_SAMPLE_MISMATCH:{order}:{field}")
        if row.get("model_sha256") != details[model_id]["sha256"] or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("model_input_tensor_sha256", ""))):
            _raise(f"LEDGER_MODEL_OR_INPUT_HASH:{order}:{model_id}")
        if row.get("invalid") is not False or row.get("fallback_used") is not False or row.get("preprocessing_success") is not True:
            _raise(f"LEDGER_INVALID_OR_FALLBACK:{order}:{model_id}")
        raw = row.get("raw_output_int8")
        probs = row.get("dequantized_output")
        if not isinstance(raw, list) or len(raw) != 3 or not all(isinstance(item, int) and -128 <= item <= 127 for item in raw):
            _raise(f"LEDGER_RAW_OUTPUT_INVALID:{order}:{model_id}")
        if not isinstance(probs, list) or len(probs) != 3 or not all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in probs):
            _raise(f"LEDGER_PROBABILITY_INVALID:{order}:{model_id}")
        actual = details[model_id]
        expected_probs = [(int(raw[index]) - actual["output_zero_point"]) * actual["output_scale"] for index in range(3)]
        if not np.allclose(np.asarray(probs, dtype=np.float64), np.asarray(expected_probs, dtype=np.float64), rtol=0.0, atol=1e-7):
            _raise(f"LEDGER_DEQUANTIZATION_MISMATCH:{order}:{model_id}")
        predicted = int(np.argmax(np.asarray(probs, dtype=np.float64)))
        if row.get("predicted_class_index") != predicted or row.get("predicted_class") != CLASS_MAP[str(predicted)]:
            _raise(f"LEDGER_ARGMAX_MISMATCH:{order}:{model_id}")
        if not math.isfinite(float(row.get("confidence"))) or abs(float(row["confidence"]) - max(float(item) for item in probs)) > 1e-7:
            _raise(f"LEDGER_CONFIDENCE_MISMATCH:{order}:{model_id}")
        by_model[model_id].append(row)
    if seen != {(index, model_id) for index in range(88) for model_id in expected_ids}:
        _raise("LEDGER_MISSING_SAMPLE_MODEL_ROWS")
    for model_id, rows in by_model.items():
        if [row["order"] for row in rows] != list(range(88)):
            _raise(f"LEDGER_SAMPLE_ORDER_MISMATCH:{model_id}")
    return by_model


def _validate_metrics(root: Path, out: Path, by_model: dict[str, list[dict[str, Any]]], specs: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_metrics: dict[str, Any] = {}
    expected_per_class: dict[str, Any] = {}
    expected_subject: dict[str, Any] = {}
    expected_coverage: dict[str, Any] = {}
    for spec in specs:
        model_id = spec["model_id"]
        rows = by_model[model_id]
        bundle = metric_bundle([int(row["true_class_index"]) for row in rows], [int(row["predicted_class_index"]) for row in rows], evaluated_sample_count=88)
        bundle.update({
            "model_role": spec["role"],
            "model_id": model_id,
            "model_sha256": spec["inspected"]["sha256"],
            "lineage_interpretation": spec["interpretation"],
            "coverage": {"attempted": 88, "valid": 88, "invalid_or_fallback": 0, "denominator": "ALL_LOCKED_TEST_ROWS"},
        })
        expected_metrics[model_id] = bundle
        expected_per_class[model_id] = bundle["per_class"]
        expected_subject[model_id] = subject_metrics(rows)
        expected_coverage[model_id] = bundle["coverage"]
    stored_metrics = _load(out / "metrics_by_model.json")
    stored_per_class = _load(out / "per_class_metrics.json")
    stored_subject = _load(out / "subject_level_metrics.json")
    stored_coverage = _load(out / "model_evaluation_coverage.json")
    if stored_metrics.get("models") != expected_metrics or stored_per_class.get("models") != expected_per_class or stored_subject.get("models") != expected_subject or stored_coverage.get("by_model") != expected_coverage:
        _raise("INDEPENDENT_METRIC_RECOMPUTATION_MISMATCH")
    if stored_coverage.get("final_accessor_invocations") != 1 or stored_coverage.get("model_inference_invocations") != 264:
        _raise("COVERAGE_INVOCATION_COUNT_MISMATCH")
    return expected_metrics, expected_per_class, expected_subject, expected_coverage


def _validate_post_access(root: Path, out: Path) -> dict[str, Any]:
    try:
        specs = validate_frozen_models(root)
        validate_contract_policy(_load(root / M_B10A_DIR_REL / "locked_test_evaluation_contract.json"))
    except Exception as exc:
        _raise(f"FROZEN_CONTRACT_GATE:{exc}")
    if not out.is_dir():
        _raise("RESULT_DIRECTORY_MISSING")
    # A structural mismatch after the one authorized accessor is a terminal
    # state.  Validate the preserved audit and explicit NOT_GENERATED evidence
    # without attempting to interpret or reacquire any LOCKED_TEST payload.
    if (out / "one_time_access_audit.json").is_file():
        terminal_audit = _load(out / "one_time_access_audit.json")
        if terminal_audit.get("post_access_status") == "INCOMPLETE_NO_RERUN":
            if terminal_audit.get("access_consumed") is not True or terminal_audit.get("accessor_invocation_count") != 1 or terminal_audit.get("second_accessor_invocation") is not False:
                _raise("INCOMPLETE_ACCESS_AUDIT_INVALID")
            if terminal_audit.get("failure") != "M-B10B_LOCKED_SPLIT_IDENTITY_MISMATCH" or terminal_audit.get("completed_model_inference_invocations") != 0 or terminal_audit.get("no_rerun_performed") is not True:
                _raise("INCOMPLETE_FAILURE_AUDIT_INVALID")
            if terminal_audit.get("expected_structural_windows") != 88 or terminal_audit.get("actual_structural_windows") != 75:
                _raise("INCOMPLETE_STRUCTURAL_IDENTITY_NOT_PRESERVED")
            _validate_checksums(out, extra_allowed=INCOMPLETE_FORENSIC_EXTRA_FILES)
            _validate_machine_paths(out)
            exceptions = _load(out / "exceptions.json")
            summary = _load(out / "m_b10b_summary.json")
            consumption = _load(out / "test_split_consumption_record.json")
            incident = _load(out / "incident_root_cause.json")
            if exceptions.get("classification") != "BLOCKER" or exceptions.get("code") != "M-B10B_LOCKED_SPLIT_IDENTITY_MISMATCH" or exceptions.get("no_rerun_performed") is not True:
                _raise("INCOMPLETE_EXCEPTION_REGISTRY_INVALID")
            if summary.get("status") != "INCOMPLETE_NO_RERUN" or summary.get("final_accessor_invocations") != 1 or summary.get("model_inference_invocations") != 0 or summary.get("m_b11_started") is not False:
                _raise("INCOMPLETE_SUMMARY_INVALID")
            if (
                summary.get("runtime_detection_code") != "M-B10B_LOCKED_SPLIT_IDENTITY_MISMATCH"
                or summary.get("forensic_root_cause") != "PRETEST_CONTRACT_COUNT_SEMANTICS_CONFLATION"
                or summary.get("forensic_status") != "INCIDENT_ROOT_CAUSE_CLOSED"
                or summary.get("performance_result") != "NOT_AVAILABLE"
            ):
                _raise("INCOMPLETE_FORENSIC_SUMMARY_INVALID")
            if (
                incident.get("root_cause_id") != "PRETEST_CONTRACT_COUNT_SEMANTICS_CONFLATION"
                or incident.get("runtime_detection_code") != "M-B10B_LOCKED_SPLIT_IDENTITY_MISMATCH"
                or incident.get("recovery_evaluation_authorized") is not False
                or incident.get("locked_test_reopen_authorized") is not False
                or incident.get("m_b11_authorized") is not False
            ):
                _raise("INCOMPLETE_INCIDENT_ROOT_CAUSE_INVALID")
            if consumption.get("status") != "LOCKED_TEST_CONSUMED_FOR_FINAL_PHASE_B_EVALUATION_INCOMPLETE" or consumption.get("no_rerun_performed") is not True:
                _raise("INCOMPLETE_CONSUMPTION_RECORD_INVALID")
            report = root / "docs/reports/20260812_Codex_M-B10B_One_Time_Locked_Test_Final_Evaluation_01.md"
            report_text = report.read_text(encoding="utf-8") if report.is_file() else ""
            if (
                not report.is_file()
                or "M-B10B_ONE_TIME_EVALUATION_INCOMPLETE_NO_RERUN" not in report_text
                or "PRETEST_CONTRACT_COUNT_SEMANTICS_CONFLATION" not in report_text
            ):
                _raise("INCOMPLETE_REPORT_MISSING")
            return {
                "validation_status": "INCOMPLETE_NO_RERUN",
                "phase_id": "M-B10B",
                "mode": "POST_ACCESS_TERMINAL_FAILURE",
                "final_accessor_invocations": 1,
                "model_inference_invocations": 0,
                "blocker": "M-B10B_LOCKED_SPLIT_IDENTITY_MISMATCH",
                "forensic_root_cause": "PRETEST_CONTRACT_COUNT_SEMANTICS_CONFLATION",
                "forensic_status": "INCIDENT_ROOT_CAUSE_CLOSED",
            }
    _validate_checksums(out)
    _validate_machine_paths(out)
    registry = _validate_registry(_load(out / "locked_test_registry.json"))
    records = _read_jsonl(out / "locked_test_sample_predictions.jsonl")
    by_model = _validate_ledger(root, records, registry, specs)
    metrics, per_class, subjects, coverage = _validate_metrics(root, out, by_model, specs)
    comparison = _comparison(metrics, specs)
    stored_comparison = _load(out / "model_comparison.json")
    if stored_comparison != {"phase_id": "M-B10B", **comparison}:
        _raise("MODEL_COMPARISON_RECOMPUTATION_MISMATCH")
    selected_quant = _quantization_audit(records, registry, specs[0])
    if _load(out / "selected_candidate_quantization_audit.json") != selected_quant:
        _raise("QUANTIZATION_AUDIT_RECOMPUTATION_MISMATCH")
    selected_result = _load(out / "selected_candidate_final_test_result.json")
    if selected_result.get("metrics") != metrics[specs[0]["model_id"]] or selected_result.get("subject_level") != subjects[specs[0]["model_id"]] or selected_result.get("quantization_audit") != selected_quant:
        _raise("SELECTED_RESULT_RECOMPUTATION_MISMATCH")
    historical = _load(out / "historical_baseline_final_test_results.json").get("baselines", [])
    if len(historical) != 2 or [item.get("model", {}).get("model_id") for item in historical] != [specs[1]["model_id"], specs[2]["model_id"]]:
        _raise("HISTORICAL_RESULT_MATRIX_MISMATCH")
    for index, item in enumerate(historical, start=1):
        if item.get("metrics") != metrics[specs[index]["model_id"]] or item.get("subject_level") != subjects[specs[index]["model_id"]]:
            _raise("HISTORICAL_RESULT_RECOMPUTATION_MISMATCH")
    access = _load(out / "one_time_access_audit.json")
    if access.get("access_consumed") is not True or access.get("accessor_invocation_count") != 1 or access.get("second_accessor_invocation") is not False or access.get("structural_rows_returned") != 88 or access.get("subjects_returned") != 16 or access.get("total_model_inference_invocations") != 264:
        _raise("ONE_TIME_ACCESS_AUDIT_INVALID")
    authorization = _load(out / "authorization_record.json")
    if authorization.get("authorization_present") is not True or authorization.get("formal_accessor_invocations") != 1 or authorization.get("second_access_prohibited") is not True or authorization.get("pre_access_counts", {}).get("final_accessor_invocations") != 0:
        _raise("AUTHORIZATION_RECORD_INVALID")
    consumption = _load(out / "test_split_consumption_record.json")
    if consumption.get("status") != "LOCKED_TEST_CONSUMED_FOR_FINAL_PHASE_B_EVALUATION" or consumption.get("candidate_frozen_before_access") is not True or consumption.get("models_frozen_before_access") is not True or consumption.get("must_not_reuse_for_phase_b_model_selection") is not True:
        _raise("TEST_SPLIT_CONSUMPTION_INVALID")
    summary = _load(out / "m_b10b_summary.json")
    if summary.get("status") != "COMPLETE_IMMUTABLE_ONE_TIME_FINAL_EVALUATION" or summary.get("locked_test_consumed") is not True or summary.get("final_accessor_invocations") != 1 or summary.get("model_inference_invocations") != 264 or summary.get("selected_candidate_unchanged") is not True or summary.get("seed43_evaluated") is not False or summary.get("seed44_evaluated") is not False or summary.get("no_post_test_tuning") is not True or summary.get("m_b11_started") is not False:
        _raise("SUMMARY_INTEGRITY_INVALID")
    exceptions = _load(out / "exceptions.json")
    if exceptions.get("status") != "NO_EXECUTION_EXCEPTIONS" or exceptions.get("invalid_rows") or exceptions.get("fallback_rows"):
        _raise("EXCEPTION_REGISTRY_INVALID")
    report = root / "docs/reports/20260812_Codex_M-B10B_One_Time_Locked_Test_Final_Evaluation_01.md"
    if not report.is_file() or "LOCKED_TEST HAS NOW BEEN CONSUMED" not in report.read_text(encoding="utf-8"):
        _raise("REPORT_MISSING_OR_CONSUMPTION_CLAIM_MISSING")
    return {"validation_status": "PASS", "phase_id": "M-B10B", "mode": "POST_ACCESS", "final_accessor_invocations": 1, "model_inference_invocations": 264, "models": [spec["model_id"] for spec in specs]}


def validate_m_b10b_artifacts(root_dir: Path = ROOT_DIR, output_dir: Path | None = None, pre_access: bool = False) -> dict[str, Any]:
    root = root_dir.resolve()
    out = (output_dir or root / OUT_DIR_REL).resolve()
    try:
        return _validate_pre_access(root, out) if pre_access else _validate_post_access(root, out)
    except MB10BValidationError:
        raise
    except Exception as exc:
        raise MB10BValidationError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-access", action="store_true", help="Validate only pre-access evidence; never call LOCKED_TEST accessor.")
    args = parser.parse_args(argv)
    try:
        result = validate_m_b10b_artifacts(pre_access=args.pre_access)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except MB10BValidationError as exc:
        print(f"M-B10B validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
