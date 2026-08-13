#!/usr/bin/env python3
"""Fail-closed post-access validator for M-B10R1-B stored recovery evidence.

NEVER calls recovery accessor or LOCKED_TEST final accessor.
Validates persisted B evidence only. Independently recomputes metrics from
the 225-row ledger.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.mmwave_m_b10r1_metrics import (  # noqa: E402
    CLASS_MAP,
    LABELS,
    metric_bundle,
    saturation_audit_from_rows,
    subject_metrics,
)
from scripts.mmwave_m_b10r1_recovery_access import (  # noqa: E402
    EXPECTED_ELIGIBLE,
    EXPECTED_INFERENCES,
    RESULT_LIMITATION,
)
from scripts.mmwave_m_b10r1_result_writer import (  # noqa: E402
    B_OUT_DIR_REL,
    REQUIRED_B_RESULT_FILES,
    SELECTED_MODEL_ID,
    V01_MODEL_ID,
    V02_MODEL_ID,
    sha256_file,
)

SELECTED_PREPROCESSING_CONTRACT_ID = "M-B10B_SELECTED_REAL_CANDIDATE_BPF_ZSCORE_V1"
V01_PREPROCESSING_CONTRACT_ID = "M-B10B_HISTORICAL_V0_1_COMPATIBILITY_PREPROCESSING_V1"
V02_PREPROCESSING_CONTRACT_ID = "M-B10B_SYNTHETIC_V0_2_EXTERNAL_COMPATIBILITY_PREPROCESSING_V1"
SELECTED_SHA = "6dff6aaa72c79d76715d40cf7e32bb1e6cd9b2c2e3ac78eaf2fda737561430c5"
V01_SHA = "43cdd6f321c2b645232162233e098c2b65549388cbc9e680a17a0eccdc8f0158"
V02_SHA = "85c023d3eefca13ecbb72a841974e53a56d5f4173920645d46df49a9088452ff"
EXPECTED_CONTRACTS = {
    SELECTED_MODEL_ID: SELECTED_PREPROCESSING_CONTRACT_ID,
    V01_MODEL_ID: V01_PREPROCESSING_CONTRACT_ID,
    V02_MODEL_ID: V02_PREPROCESSING_CONTRACT_ID,
}
EXPECTED_MODEL_SHA = {
    SELECTED_MODEL_ID: SELECTED_SHA,
    V01_MODEL_ID: V01_SHA,
    V02_MODEL_ID: V02_SHA,
}
FORBIDDEN_SEEDS = ("seed43", "seed44")


class MB10R1BValidationError(Exception):
    """Fail-closed M-B10R1-B stored-evidence validation failure."""


def _raise(code: str) -> None:
    raise MB10R1BValidationError(code)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _hex_digest(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", value))


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
        if len(parts) != 2 or not _hex_digest(parts[0]):
            _raise(f"CHECKSUM_LINE_INVALID:{line}")
        if ".." in parts[1] or parts[1].startswith("/") or "\\" in parts[1]:
            _raise(f"CHECKSUM_UNSAFE_PATH:{parts[1]}")
        if parts[1] in mapped and mapped[parts[1]] != parts[0]:
            _raise(f"CHECKSUM_DUPLICATE_INCONSISTENT:{parts[1]}")
        mapped[parts[1]] = parts[0]
        target = out / parts[1]
        if not target.is_file():
            _raise(f"CHECKSUM_TARGET_MISSING:{parts[1]}")
        live = sha256_file(target)
        if live != parts[0]:
            _raise(f"CHECKSUM_MISMATCH:{parts[1]}")


def _inspect_no_accessor_calls() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
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
            }:
                _raise(f"B_VALIDATOR_CALLS_ACCESSOR:{name}")


def _recompute_from_ledger(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_INFERENCES:
        _raise(f"LEDGER_ROW_COUNT:{len(rows)}")
    model_ids = sorted({str(r.get("model_id")) for r in rows})
    if model_ids != sorted([SELECTED_MODEL_ID, V01_MODEL_ID, V02_MODEL_ID]):
        _raise(f"MODEL_SET_MISMATCH:{model_ids}")
    blob = json.dumps(rows).lower()
    for seed in FORBIDDEN_SEEDS:
        if seed in blob:
            _raise(f"FORBIDDEN_SEED:{seed}")
    if len(model_ids) != 3:
        _raise("MODEL_COUNT_NOT_3")

    metrics_by_model: dict[str, Any] = {}
    subjects_by_model: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    for mid in [SELECTED_MODEL_ID, V01_MODEL_ID, V02_MODEL_ID]:
        model_rows = [r for r in rows if r.get("model_id") == mid]
        if len(model_rows) != EXPECTED_ELIGIBLE:
            _raise(f"MODEL_ROW_COUNT:{mid}:{len(model_rows)}")
        expected_contract = EXPECTED_CONTRACTS[mid]
        for row in model_rows:
            if row.get("preprocessing_contract_id") != expected_contract:
                _raise(f"PREPROCESSING_CONTRACT_MISMATCH:{mid}")
            if row.get("result_not_pristine") is not True:
                _raise("RESULT_NOT_PRISTINE_FALSE")
            if row.get("result_limitation") != RESULT_LIMITATION:
                _raise("RESULT_LIMITATION_MISMATCH")
            dequant = row.get("dequantized_output")
            if not row.get("invalid"):
                if not isinstance(dequant, list) or len(dequant) != 3:
                    _raise(f"DEQUANT_SHAPE:{mid}")
                if not all(np.isfinite(dequant)):
                    _raise(f"DEQUANT_NONFINITE:{mid}")
                recomputed = int(np.argmax(np.asarray(dequant, dtype=np.float64)))
                if recomputed != int(row["predicted_class_index"]):
                    _raise(f"ARGMAX_MISMATCH:{mid}:{row.get('window_id')}")
                if CLASS_MAP[str(recomputed)] != row.get("predicted_class"):
                    _raise(f"CLASS_MAP_MISMATCH:{mid}")
        valid = [r for r in model_rows if not r.get("invalid")]
        if mid == SELECTED_MODEL_ID and len(valid) != EXPECTED_ELIGIBLE:
            _raise(f"SELECTED_VALID_COUNT:{len(valid)}")
        labels = [int(r["true_class_index"]) for r in valid]
        preds = [int(r["predicted_class_index"]) for r in valid]
        bundle = metric_bundle(labels, preds, evaluated_sample_count=len(labels))
        bundle["planned_count"] = EXPECTED_ELIGIBLE
        bundle["valid_count"] = len(valid)
        bundle["invalid_count"] = len(model_rows) - len(valid)
        metrics_by_model[mid] = bundle
        subjects_by_model[mid] = subject_metrics(valid)
        coverage[mid] = {
            "evaluation_rows_attempted": len(model_rows),
            "valid_count": len(valid),
            "invalid_count": len(model_rows) - len(valid),
            "planned_count": EXPECTED_ELIGIBLE,
        }
    selected_rows = [r for r in rows if r.get("model_id") == SELECTED_MODEL_ID and not r.get("invalid")]
    saturation = saturation_audit_from_rows(selected_rows)
    return {
        "metrics_by_model": metrics_by_model,
        "subject_metrics_by_model": subjects_by_model,
        "coverage": coverage,
        "saturation_audit_seed42": saturation,
    }


def _approx_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) < 1e-9
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_approx_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(_approx_equal(a[k], b[k]) for k in a)
    return a == b


def validate_m_b10r1b_artifacts(
    root: Path | None = None,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate stored B evidence. Never accesses LOCKED_TEST."""
    del root  # B validator uses stored evidence only; root reserved for future identity checks.
    _inspect_no_accessor_calls()
    out = Path(output_dir) if output_dir is not None else ROOT_DIR / B_OUT_DIR_REL
    if not out.is_dir():
        _raise("B_OUTPUT_DIR_MISSING")

    for name in REQUIRED_B_RESULT_FILES:
        if not (out / name).is_file():
            _raise(f"B_REQUIRED_FILE_MISSING:{name}")
    _validate_checksums(out)

    summary = load_json(out / "m_b10r1b_summary.json")
    if summary.get("status") == "PARTIAL_INCOMPLETE":
        _raise("B_EVIDENCE_PARTIAL_INCOMPLETE")
    if summary.get("status") != "RECOVERY_EXECUTED":
        _raise("B_STATUS_NOT_EXECUTED")
    if summary.get("result_not_pristine") is not True:
        _raise("SUMMARY_RESULT_NOT_PRISTINE_FALSE")
    if summary.get("result_limitation") != RESULT_LIMITATION:
        _raise("SUMMARY_RESULT_LIMITATION")
    if summary.get("rerun_performed") is True:
        _raise("SUMMARY_RERUN_TRUE")

    audit = load_json(out / "one_time_recovery_access_audit.json")
    if int(audit.get("recovery_payload_release_events", -1)) != 1:
        _raise("AUDIT_RECOVERY_RELEASE_NOT_1")
    if int(audit.get("historical_original_payload_release_events", -1)) != 1:
        _raise("AUDIT_ORIGINAL_RELEASE_NOT_1")
    if int(audit.get("historical_total_payload_release_events", -1)) != 2:
        _raise("AUDIT_HISTORICAL_TOTAL_NOT_2")
    if audit.get("rerun_performed") is True:
        _raise("AUDIT_RERUN_TRUE")
    if audit.get("result_not_pristine") is not True:
        _raise("AUDIT_RESULT_NOT_PRISTINE_FALSE")

    consumption = load_json(out / "recovery_consumption_record.json")
    if consumption.get("rerun_performed") is True:
        _raise("CONSUMPTION_RERUN_TRUE")
    if int(consumption.get("historical_total_payload_release_events", -1)) != 2:
        _raise("CONSUMPTION_TOTAL_NOT_2")

    rows = load_jsonl(out / "recovery_sample_predictions.jsonl")
    recomputed = _recompute_from_ledger(rows)

    stored_metrics = load_json(out / "metrics_by_model.json")
    stored_subjects = load_json(out / "subject_level_metrics.json")
    stored_coverage = load_json(out / "model_evaluation_coverage.json")
    stored_saturation = load_json(out / "selected_candidate_quantization_audit.json")
    stored_per_class = load_json(out / "per_class_metrics.json")

    for mid in [SELECTED_MODEL_ID, V01_MODEL_ID, V02_MODEL_ID]:
        exp = recomputed["metrics_by_model"][mid]
        got = stored_metrics.get(mid) or {}
        for key in (
            "accuracy",
            "macro_f1",
            "macro_precision",
            "macro_recall",
            "confusion_matrix",
            "prediction_distribution",
            "class_collapse",
            "apnea_proxy",
            "rapid_or_abnormal_recall",
            "per_class",
        ):
            if not _approx_equal(exp.get(key), got.get(key)):
                _raise(f"METRIC_RECOMPUTE_MISMATCH:{mid}:{key}")
        if not _approx_equal(exp["per_class"], stored_per_class.get(mid)):
            _raise(f"PER_CLASS_RECOMPUTE_MISMATCH:{mid}")
        sub_exp = recomputed["subject_metrics_by_model"][mid]
        sub_got = stored_subjects.get(mid) or {}
        for key in ("worst_subject_macro_f1", "median_subject_macro_f1", "subject_count", "per_subject"):
            if not _approx_equal(sub_exp.get(key), sub_got.get(key)):
                _raise(f"SUBJECT_RECOMPUTE_MISMATCH:{mid}:{key}")
        cov = stored_coverage.get(mid) or {}
        if int(cov.get("evaluation_rows_attempted", -1)) != EXPECTED_ELIGIBLE:
            _raise(f"COVERAGE_ATTEMPTED_NOT_75:{mid}")
        if mid == SELECTED_MODEL_ID:
            if int(cov.get("valid_count", -1)) != EXPECTED_ELIGIBLE:
                _raise("SELECTED_COVERAGE_VALID_NOT_75")
            if int(cov.get("tflite_invoke_count", -1)) != EXPECTED_ELIGIBLE:
                _raise("SELECTED_TFLITE_INVOKE_NOT_75")

    invoke_sum = sum(
        int((stored_coverage.get(mid) or {}).get("tflite_invoke_count", -1))
        for mid in [SELECTED_MODEL_ID, V01_MODEL_ID, V02_MODEL_ID]
    )
    if int(summary.get("actual_total_tflite_invocations", -1)) != invoke_sum:
        _raise("SUMMARY_TFLITE_INVOKE_NOT_COVERAGE_SUM")
    if int(summary.get("actual_total_tflite_invocations", -1)) != EXPECTED_INFERENCES:
        _raise("SUMMARY_TFLITE_INVOKE_NOT_225")
    if int(summary.get("ledger_row_count", -1)) != EXPECTED_INFERENCES:
        _raise("SUMMARY_LEDGER_NOT_225")
    if summary.get("candidate_unchanged") is not True:
        _raise("SUMMARY_CANDIDATE_CHANGED")
    if summary.get("no_tuning") is not True:
        _raise("SUMMARY_TUNING_PRESENT")
    if summary.get("no_retraining") is not True:
        _raise("SUMMARY_RETRAINING_PRESENT")
    if summary.get("no_recalibration") is not True:
        _raise("SUMMARY_RECALIBRATION_PRESENT")

    sat_exp = recomputed["saturation_audit_seed42"]
    for key in (
        "input_saturation_ratio",
        "samples_with_any_saturation",
        "worst_sample_saturation_ratio",
        "pre_clamp_out_of_range_count",
    ):
        if key in sat_exp and key in stored_saturation:
            if not _approx_equal(sat_exp[key], stored_saturation[key]):
                _raise(f"SATURATION_RECOMPUTE_MISMATCH:{key}")

    identity = load_json(out / "execution_identity.json")
    specs = identity.get("specs") or []
    if len(specs) != 3:
        _raise("EXECUTION_IDENTITY_MODEL_COUNT")
    ids = [s.get("model_id") for s in specs]
    if "seed43" in json.dumps(ids).lower() or "seed44" in json.dumps(ids).lower():
        _raise("EXECUTION_IDENTITY_FORBIDDEN_SEED")
    spec_sha = {s.get("model_id"): s.get("sha256") for s in specs}
    for mid, expected in EXPECTED_MODEL_SHA.items():
        if spec_sha.get(mid) != expected:
            _raise(f"EXECUTION_IDENTITY_MODEL_SHA_MISMATCH:{mid}")
    for row in rows:
        mid = row.get("model_id")
        if row.get("model_sha256") != spec_sha.get(mid):
            _raise(f"LEDGER_MODEL_SHA_MISMATCH:{mid}")
        if row.get("model_sha256") != EXPECTED_MODEL_SHA.get(mid):
            _raise(f"LEDGER_MODEL_SHA_NOT_FROZEN:{mid}")

    return {
        "validation_status": "PASS",
        "phase_id": "M-B10R1-B",
        "ledger_row_count": len(rows),
        "actual_total_tflite_invocations": summary.get("actual_total_tflite_invocations"),
        "recovery_payload_release_events": 1,
        "historical_total_payload_release_events": 2,
        "rerun_performed": False,
        "result_not_pristine": True,
        "locked_test_accessed": False,
        "output_dir": str(out),
    }


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    output_dir = None
    if "--output-dir" in args:
        idx = args.index("--output-dir")
        output_dir = Path(args[idx + 1])
    try:
        result = validate_m_b10r1b_artifacts(output_dir=output_dir)
    except MB10R1BValidationError as exc:
        print(json.dumps({"validation_status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
