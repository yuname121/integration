#!/usr/bin/env python3
"""Independent SafeNest mmWave M-B9 validator.

The validator intentionally recomputes identity and executes fresh bounded
scenarios.  It does not treat ``m_b9_summary.json``, saved PASS flags, or saved
fallback flags as authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inference.mmwave_interpreter import MMWaveInterpreter, tflite
from integrated_node.run_node import SafeNestIntegratedNode
from risk.risk_engine import SafeNestRiskEngine
from sensors.mmwave.finalist_mock_provider import FinalistMockProvider
from mmwave_m_b1_preprocessing import fit_train_zscore_statistics, transform_signals
from mmwave_m_b9_mock_e2e import (
    LABELS,
    OUT_DIR_REL,
    RUNTIME_DIR_REL,
    SEEDS,
    NeutralSupportProvider,
    array_sha256,
    build_runtime_model_identity,
    direct_prediction,
    inspect_tflite,
    load_json,
    load_stage_artifacts,
    rel,
    run_node_once,
    select_validation_inputs,
    sha256_file,
)
from mmwave_phase_b_access import PhaseBAccessGuard
from validate_mmwave_m_b8 import validate_m_b8_artifacts


REQUIRED_FILES = (
    "input_identity.json",
    "experiment_contract.json",
    "runtime_manifest_contract.json",
    "runtime_model_identity.json",
    "runtime_preprocessing_identity.json",
    "scenario_contract.json",
    "scenario_input_selection.json",
    "scenario_results.json",
    "scenario_results.jsonl",
    "inference_result_audit.json",
    "risk_input_audit.json",
    "json_output_audit.json",
    "fallback_audit.json",
    "fault_timeout_stale_audit.json",
    "runtime_prediction_identity.json",
    "locked_test_access_audit.json",
    "run_environment.json",
    "exceptions.json",
    "m_b9_summary.json",
    "checksums.sha256",
)
SCENARIOS = (
    "A_NORMAL",
    "B_RAPID_OR_ABNORMAL",
    "C_APNEA",
    "D_INSUFFICIENT_HISTORY",
    "E_INVALID_SHAPE",
    "F_NAN",
    "G_INF",
    "H_STALE",
    "I_PROVIDER_SENSOR_FAULT",
    "J_READ_EXCEPTION",
    "K_TIMEOUT",
    "L_MISSING_MODEL",
    "M_SHA_MISMATCH",
    "N_VALID_EXPLICIT_FINALIST",
    "O_NOT_CONNECTED_PROVIDER",
)
NEGATIVE_CASES = (
    "default_historical_model_substituted",
    "wrong_phase_sha256",
    "wrong_phase_bytes",
    "wrong_seed",
    "wrong_input_dtype",
    "wrong_input_quantization",
    "wrong_output_quantization",
    "wrong_preprocessing_profile",
    "wrong_bpf_contract",
    "wrong_zscore_stats",
    "runtime_prediction_mismatch",
    "scenario_truth_forces_state",
    "scenario_truth_forces_score",
    "hidden_fallback",
    "missing_fallback_reason",
    "invalid_shape",
    "nan",
    "inf",
    "insufficient_history",
    "stale_timestamp",
    "provider_fault",
    "read_exception",
    "timeout",
    "missing_model",
    "sha_mismatch",
    "risk_input_mismatch",
    "json_schema",
    "json_nonfinite",
    "locked_test_access",
    "checksum_corruption",
    "manifest_absolute_path",
    "manifest_path_traversal",
    "model_binary_duplicated",
)

NONDETERMINISTIC_KEYS = frozenset({"timestamp", "latency_ms", "calc_latency_ms", "elapsed_sec"})
RISK_CORE_FIELDS = (
    "risk_score",
    "risk_level",
    "system_health",
    "degraded_mode",
    "invalid_sensors",
    "stale_sensors",
    "component_scores",
    "is_emergency",
    "reasons",
)
INFERENCE_FIELDS = (
    "sensor_id",
    "state",
    "score",
    "confidence",
    "valid",
    "error",
    "metadata",
)


class MB9ValidationError(RuntimeError):
    pass


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float, np.number)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(k) and _finite(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return True


def _load_all_json(out_dir: Path) -> dict[str, Any]:
    loaded = {}
    for name in REQUIRED_FILES:
        path = out_dir / name
        if not path.is_file():
            raise MB9ValidationError(f"M-B9_REQUIRED_OUTPUT_MISSING:{name}")
        if name.endswith(".json"):
            try:
                loaded[name] = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise MB9ValidationError(f"M-B9_JSON_PARSE_ERROR:{name}:{exc}") from exc
            if not _finite(loaded[name]):
                raise MB9ValidationError(f"M-B9_JSON_NONFINITE:{name}")
    return loaded


def _canonical(value: Any, *, key: str | None = None) -> Any:
    """Normalize only runtime-generated timing fields for evidence comparison."""
    if key in NONDETERMINISTIC_KEYS:
        return "<NONDETERMINISTIC>"
    if isinstance(value, dict):
        return {
            str(k): _canonical(v, key=str(k))
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            if str(k) not in NONDETERMINISTIC_KEYS
        }
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, str) and key == "json_output":
        try:
            return _canonical(json.loads(value))
        except Exception:
            return value
    return value


def _assert_equal(actual: Any, expected: Any, code: str) -> None:
    if _canonical(actual) != _canonical(expected):
        raise MB9ValidationError(code)


def _record_key(row: dict[str, Any]) -> tuple[str, int | None]:
    return str(row.get("scenario_id")), row.get("seed")


def _records_by_key(records: list[dict[str, Any]], code: str) -> dict[tuple[str, int | None], dict[str, Any]]:
    result: dict[tuple[str, int | None], dict[str, Any]] = {}
    for row in records:
        key = _record_key(row)
        if key in result:
            raise MB9ValidationError(f"{code}:DUPLICATE:{key}")
        result[key] = row
    return result


def _expected_fallback_row(row: dict[str, Any]) -> dict[str, Any]:
    mm = row["mmwave_result"]
    metadata = mm.get("metadata", {})
    return {
        "scenario_id": row["scenario_id"],
        "seed": row.get("seed"),
        "fallback_used": metadata.get("fallback_used"),
        "model_id": metadata.get("model_id"),
        "reason": metadata.get("fallback_reason"),
        "valid": row.get("valid"),
        "score_source": metadata.get("score_source", row.get("score_source")),
    }


def _expected_fault_row(row: dict[str, Any]) -> dict[str, Any]:
    output = row["node_output"]
    return {
        "scenario_id": row["scenario_id"],
        "seed": row.get("seed"),
        "valid": row.get("valid"),
        "error": row.get("error"),
        "system_health": output.get("system_health"),
        "invalid_sensors": output.get("invalid_sensors"),
        "stale_sensors": output.get("stale_sensors"),
    }


def _expected_inference_row(row: dict[str, Any]) -> dict[str, Any]:
    mm = row["mmwave_result"]
    metadata = mm.get("metadata", {})
    return {
        "scenario_id": row["scenario_id"],
        "seed": row.get("seed"),
        "mmwave_result": mm,
        "valid": row.get("valid"),
        "error": row.get("error"),
        "metadata_contract": {
            "score_source": metadata.get("score_source", row.get("score_source")),
            "fallback_used": metadata.get("fallback_used"),
        },
    }


def _expected_risk_row(row: dict[str, Any], *, recomputed: dict[str, Any]) -> dict[str, Any]:
    output = row["node_output"]
    sensors = output.get("sensors", {})
    node_core = {key: output.get(key) for key in RISK_CORE_FIELDS}
    recomputed_core = {key: recomputed.get(key) for key in RISK_CORE_FIELDS}
    return {
        "scenario_id": row["scenario_id"],
        "seed": row.get("seed"),
        "exact_inference_results_entering_risk": sensors,
        "node_risk_output_core": node_core,
        "fresh_risk_engine_recomputation_core": recomputed_core,
        "equal": _canonical(node_core) == _canonical(recomputed_core),
    }


def _risk_engine_recompute(row: dict[str, Any]) -> dict[str, Any]:
    output = row.get("node_output", {})
    sensors = output.get("sensors")
    if not isinstance(sensors, dict):
        raise MB9ValidationError(f"M-B9_RISK_INPUT_MISSING:{row.get('scenario_id')}")
    stale_sec = output.get("metadata", {}).get("stale_sec", 3.0)
    engine = SafeNestRiskEngine(stale_sec=stale_sec)
    try:
        result = engine.evaluate(sensors, now=float(output["timestamp"]))
    except Exception as exc:
        raise MB9ValidationError(f"M-B9_RISK_RECOMPUTATION_ERROR:{row.get('scenario_id')}:{exc}") from exc
    return result.to_dict()


def _validate_inference_shape(row: dict[str, Any]) -> None:
    mm = row.get("mmwave_result")
    if not isinstance(mm, dict) or any(field not in mm for field in INFERENCE_FIELDS):
        raise MB9ValidationError(f"M-B9_INFERENCE_RESULT_FIELDS:{row.get('scenario_id')}")
    metadata = mm.get("metadata")
    if not isinstance(metadata, dict):
        raise MB9ValidationError(f"M-B9_INFERENCE_METADATA:{row.get('scenario_id')}")
    for key in ("fallback_used", "fallback_reason", "score_source"):
        # Provider-contract failures created by SafeNestIntegratedNode may
        # legitimately omit fallback metadata; the audit normalizes those
        # absent fields to None/NO_VALID_PREDICTION without inventing a value.
        if key not in metadata and bool(mm.get("valid")):
            raise MB9ValidationError(f"M-B9_INFERENCE_METADATA_FIELD:{row.get('scenario_id')}:{key}")
    if bool(mm.get("valid")):
        for key in ("model_id", "model_sha256", "class_index", "model_predicted_class", "probabilities", "preprocessing_profile"):
            if key not in metadata:
                raise MB9ValidationError(f"M-B9_VALID_INFERENCE_METADATA_FIELD:{row.get('scenario_id')}:{key}")
        if metadata.get("score_source") != "MODEL_PREDICTION" or metadata.get("fallback_used") is not False:
            raise MB9ValidationError(f"M-B9_VALID_INFERENCE_FALLBACK:{row.get('scenario_id')}")
        if metadata.get("fallback_reason") is not None:
            raise MB9ValidationError(f"M-B9_VALID_INFERENCE_FALLBACK_REASON:{row.get('scenario_id')}")
    else:
        if not mm.get("error"):
            raise MB9ValidationError(f"M-B9_INVALID_INFERENCE_ERROR:{row.get('scenario_id')}")


def _validate_stored_audits(loaded: dict[str, Any]) -> dict[str, Any]:
    """Validate stored evidence without trusting any stored PASS boolean."""
    scenario_payload = loaded["scenario_results.json"]
    stored_records = scenario_payload.get("records")
    if not isinstance(stored_records, list) or not stored_records:
        raise MB9ValidationError("M-B9_SCENARIO_RESULTS_EMPTY")
    stored_by_key = _records_by_key(stored_records, "M-B9_SCENARIO_RESULTS")
    for row in stored_records:
        _validate_inference_shape(row)
        if row.get("node_mode") != "real_with_injected_mock_providers":
            raise MB9ValidationError(f"M-B9_NODE_MODE:{row.get('scenario_id')}")
        if row.get("mmwave_result", {}).get("sensor_id") != "mmwave":
            raise MB9ValidationError(f"M-B9_SENSOR_ID:{row.get('scenario_id')}")
        if row.get("scenario_truth_source") not in {None, "METADATA_ONLY"}:
            raise MB9ValidationError(f"M-B9_TRUTH_SOURCE:{row.get('scenario_id')}")
        if row.get("scenario_id") in {"A_NORMAL", "B_RAPID_OR_ABNORMAL", "C_APNEA", "N_VALID_EXPLICIT_FINALIST"}:
            mm = row["mmwave_result"]
            metadata = mm["metadata"]
            if mm.get("state") != metadata.get("model_predicted_class"):
                raise MB9ValidationError(f"M-B9_SCENARIO_STATE_FORCED:{row.get('scenario_id')}")
            expected_score = {"NORMAL": 0.0, "RAPID_OR_ABNORMAL": 0.5, "APNEA": 1.0}.get(metadata.get("model_predicted_class"))
            if expected_score is None or float(mm.get("score")) != expected_score:
                raise MB9ValidationError(f"M-B9_SCENARIO_SCORE_FORCED:{row.get('scenario_id')}")

    inference_records = loaded["inference_result_audit.json"].get("records")
    if not isinstance(inference_records, list):
        raise MB9ValidationError("M-B9_INFERENCE_AUDIT_RECORDS")
    inference_by_key = _records_by_key(inference_records, "M-B9_INFERENCE_AUDIT")
    for key, row in stored_by_key.items():
        expected = _expected_inference_row(row)
        actual = inference_by_key.get(key)
        if actual is None:
            raise MB9ValidationError(f"M-B9_INFERENCE_AUDIT_MISSING:{key}")
        _assert_equal(actual, expected, f"M-B9_INFERENCE_AUDIT_MISMATCH:{key}")

    fallback_records = loaded["fallback_audit.json"].get("records")
    if not isinstance(fallback_records, list):
        raise MB9ValidationError("M-B9_FALLBACK_AUDIT_RECORDS")
    fallback_by_key = _records_by_key(fallback_records, "M-B9_FALLBACK_AUDIT")
    for key, row in stored_by_key.items():
        actual = fallback_by_key.get(key)
        if actual is None:
            raise MB9ValidationError(f"M-B9_FALLBACK_AUDIT_MISSING:{key}")
        _assert_equal(actual, _expected_fallback_row(row), f"M-B9_FALLBACK_AUDIT_MISMATCH:{key}")
    if loaded["fallback_audit.json"].get("valid_finalist_records_have_no_fallback") is not True:
        raise MB9ValidationError("M-B9_FALLBACK_AUDIT_FLAG")

    fault_records = loaded["fault_timeout_stale_audit.json"].get("records")
    if not isinstance(fault_records, list):
        raise MB9ValidationError("M-B9_FAULT_AUDIT_RECORDS")
    fault_by_key = _records_by_key(fault_records, "M-B9_FAULT_AUDIT")
    required_faults = set(loaded["fault_timeout_stale_audit.json"].get("required_fault_ids", []))
    required_fault_contract = set(SCENARIOS[3:]) - {"N_VALID_EXPLICIT_FINALIST"}
    if not required_fault_contract.issubset(required_faults):
        raise MB9ValidationError("M-B9_FAULT_AUDIT_REQUIRED_IDS")
    for key, row in stored_by_key.items():
        if row["scenario_id"] not in required_faults:
            continue
        actual = fault_by_key.get(key)
        if actual is None:
            raise MB9ValidationError(f"M-B9_FAULT_AUDIT_MISSING:{key}")
        _assert_equal(actual, _expected_fault_row(row), f"M-B9_FAULT_AUDIT_MISMATCH:{key}")
    expected_fault_semantics = {
        "D_INSUFFICIENT_HISTORY": (False, "INSUFFICIENT_HISTORY", False),
        "E_INVALID_SHAPE": (False, "INVALID_SHAPE", False),
        "F_NAN": (False, "NAN_OR_INF", False),
        "G_INF": (False, "NAN_OR_INF", False),
        "H_STALE": (True, None, True),
        "I_PROVIDER_SENSOR_FAULT": (False, "SIMULATED_MMWAVE_SENSOR_FAULT", False),
        "J_READ_EXCEPTION": (False, "PROVIDER_READ_EXCEPTION", False),
        "K_TIMEOUT": (False, "PROVIDER_READ_TIMEOUT", False),
        "L_MISSING_MODEL": (False, "M-B9_FINALIST_MODEL_MANIFEST_MISSING", False),
        "M_SHA_MISMATCH": (False, "M-B9_FINALIST_ARTIFACT_IDENTITY_MISMATCH", False),
        "O_NOT_CONNECTED_PROVIDER": (False, "PROVIDER_CONNECT_FAILED", False),
    }
    for scenario_id, (valid, error, stale) in expected_fault_semantics.items():
        row = fault_by_key.get((scenario_id, 42 if scenario_id != "L_MISSING_MODEL" else None))
        if row is None:
            raise MB9ValidationError(f"M-B9_FAULT_AUDIT_SEMANTICS_MISSING:{scenario_id}")
        if bool(row.get("valid")) != valid or (error is not None and row.get("error") != error) or bool(row.get("stale_sensors")) != stale:
            raise MB9ValidationError(f"M-B9_FAULT_AUDIT_SEMANTICS:{scenario_id}")

    risk_records = loaded["risk_input_audit.json"].get("records")
    if not isinstance(risk_records, list):
        raise MB9ValidationError("M-B9_RISK_AUDIT_RECORDS")
    risk_by_key = _records_by_key(risk_records, "M-B9_RISK_AUDIT")
    for key, row in stored_by_key.items():
        actual = risk_by_key.get(key)
        if actual is None:
            raise MB9ValidationError(f"M-B9_RISK_AUDIT_MISSING:{key}")
        recomputed = _risk_engine_recompute(row)
        expected = _expected_risk_row(row, recomputed=recomputed)
        _assert_equal(actual, expected, f"M-B9_RISK_AUDIT_MISMATCH:{key}")
        if actual.get("equal") is not True:
            raise MB9ValidationError(f"M-B9_RISK_AUDIT_NOT_EQUAL:{key}")
    if loaded["risk_input_audit.json"].get("all_equal") is not True:
        raise MB9ValidationError("M-B9_RISK_AUDIT_FLAG")

    json_records = loaded["json_output_audit.json"].get("records")
    if not isinstance(json_records, list):
        raise MB9ValidationError("M-B9_JSON_AUDIT_RECORDS")
    json_by_key = _records_by_key(json_records, "M-B9_JSON_AUDIT")
    required_output_fields = {"timestamp", "risk_score", "risk_level", "system_health", "degraded_mode", "invalid_sensors", "stale_sensors", "component_scores", "is_emergency", "reasons", "sensors", "metadata"}
    for key, row in stored_by_key.items():
        serialized = row.get("json_output")
        if not isinstance(serialized, str):
            raise MB9ValidationError(f"M-B9_JSON_SERIALIZED_MISSING:{key}")
        try:
            parsed = json.loads(serialized)
        except Exception as exc:
            raise MB9ValidationError(f"M-B9_JSON_PARSE:{key}:{exc}") from exc
        if not _finite(parsed) or not required_output_fields.issubset(parsed):
            raise MB9ValidationError(f"M-B9_JSON_SCHEMA_OR_NONFINITE:{key}")
        _assert_equal(parsed, row["node_output"], f"M-B9_JSON_NODE_OUTPUT_MISMATCH:{key}")
        actual = json_by_key.get(key)
        expected = {
            "scenario_id": row["scenario_id"],
            "seed": row.get("seed"),
            "serialized_with": "SafeNestRiskOutput.to_json",
            "parsed_schema_version": parsed.get("metadata", {}).get("schema_version"),
            "finite": True,
            "parse_success": True,
            "schema_fields_present": True,
        }
        if actual is None:
            raise MB9ValidationError(f"M-B9_JSON_AUDIT_MISSING:{key}")
        _assert_equal(actual, expected, f"M-B9_JSON_AUDIT_MISMATCH:{key}")
    if loaded["json_output_audit.json"].get("all_valid") is not True:
        raise MB9ValidationError("M-B9_JSON_AUDIT_FLAG")

    locked = loaded["locked_test_access_audit.json"]
    if locked.get("locked_test_inputs_loaded") is not False or locked.get("model_selection_access_attempts") != 0 or locked.get("performance_access_attempts") != 0 or locked.get("label_access_attempts") != 0 or locked.get("lock_preserved") is not True or locked.get("source_split") != "VALIDATION":
        raise MB9ValidationError("M-B9_LOCKED_TEST_AUDIT_CORRUPTED")
    return {"stored_scenario_count": len(stored_records), "stored_audits_reconstructed": True}


def _load_scenario_jsonl(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "scenario_results.jsonl"
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise MB9ValidationError(f"M-B9_SCENARIO_JSONL_PARSE:{line_number}:{exc}") from exc
        if not isinstance(row, dict) or not _finite(row):
            raise MB9ValidationError(f"M-B9_SCENARIO_JSONL_INVALID:{line_number}")
        rows.append(row)
    return rows


def _validate_scenario_jsonl_consistency(out_dir: Path, stored_records: list[dict[str, Any]]) -> dict[str, Any]:
    jsonl_records = _load_scenario_jsonl(out_dir)
    _assert_equal(jsonl_records, stored_records, "M-B9_SCENARIO_JSONL_MISMATCH")
    return {"stored_jsonl_match": True, "stored_jsonl_count": len(jsonl_records)}


def _validate_stored_runtime_prediction_identity(loaded: dict[str, Any]) -> dict[str, Any]:
    payload = loaded["runtime_prediction_identity.json"]
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise MB9ValidationError("M-B9_RUNTIME_PREDICTION_AUDIT_RECORDS")
    for row in rows:
        required = {
            "direct_class_index",
            "runtime_class_index",
            "top1_exact",
            "direct_probabilities",
            "runtime_probabilities",
            "probabilities_exact",
            "direct_output_int8",
            "runtime_output_int8",
            "output_int8_exact",
        }
        if not required.issubset(row):
            raise MB9ValidationError("M-B9_RUNTIME_PREDICTION_AUDIT_FIELDS")
        if row["direct_class_index"] != row["runtime_class_index"] or row["top1_exact"] is not True:
            raise MB9ValidationError("M-B9_RUNTIME_PREDICTION_AUDIT_TOP1_MISMATCH")
        if _canonical(row["direct_probabilities"]) != _canonical(row["runtime_probabilities"]) or row["probabilities_exact"] is not True:
            raise MB9ValidationError("M-B9_RUNTIME_PREDICTION_AUDIT_PROBABILITY_MISMATCH")
        if _canonical(row["direct_output_int8"]) != _canonical(row["runtime_output_int8"]) or row["output_int8_exact"] is not True:
            raise MB9ValidationError("M-B9_RUNTIME_PREDICTION_AUDIT_OUTPUT_MISMATCH")
    for flag in ("all_top1_exact", "all_probability_vectors_exact", "all_int8_outputs_exact"):
        if payload.get(flag) is not True:
            raise MB9ValidationError(f"M-B9_RUNTIME_PREDICTION_AUDIT_FLAG:{flag}")
    return {"runtime_prediction_identity_rows": len(rows), "runtime_prediction_identity_exact": True}


def _validate_fresh_risk_records(fresh_records: list[dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    for row in fresh_records:
        recomputed = _risk_engine_recompute(row)
        output = row["node_output"]
        node_core = {key: output.get(key) for key in RISK_CORE_FIELDS}
        recomputed_core = {key: recomputed.get(key) for key in RISK_CORE_FIELDS}
        if _canonical(node_core) != _canonical(recomputed_core):
            raise MB9ValidationError(f"M-B9_FRESH_RISK_ENGINE_MISMATCH:{row.get('scenario_id')}:{row.get('seed')}")
        stored_recomputed = row.get("risk_recomputed_independently")
        if stored_recomputed is not None and _canonical(stored_recomputed) != _canonical(recomputed):
            raise MB9ValidationError(f"M-B9_FRESH_RISK_SECOND_RECOMPUTATION_MISMATCH:{row.get('scenario_id')}:{row.get('seed')}")
        sensors = output.get("sensors", {})
        if set(sensors) != {"thermal44", "mmwave", "co2", "pir"}:
            raise MB9ValidationError(f"M-B9_FRESH_SENSOR_SET:{row.get('scenario_id')}:{row.get('seed')}")
        for sensor_id, sensor in sensors.items():
            if any(field not in sensor for field in INFERENCE_FIELDS):
                raise MB9ValidationError(f"M-B9_FRESH_SENSOR_FIELDS:{sensor_id}:{row.get('scenario_id')}")
            if sensor.get("sensor_id") != sensor_id:
                raise MB9ValidationError(f"M-B9_FRESH_SENSOR_ID:{sensor_id}:{row.get('scenario_id')}")
        checked += 1
    return {"fresh_risk_records_checked": checked, "fresh_risk_engine_recomputation_exact": True}


def _validate_fresh_json_records(fresh_records: list[dict[str, Any]]) -> dict[str, Any]:
    required_output_fields = {"timestamp", "risk_score", "risk_level", "system_health", "degraded_mode", "invalid_sensors", "stale_sensors", "component_scores", "is_emergency", "reasons", "sensors", "metadata"}
    checked = 0
    for row in fresh_records:
        serialized = row.get("json_output")
        if not isinstance(serialized, str):
            raise MB9ValidationError(f"M-B9_FRESH_JSON_MISSING:{row.get('scenario_id')}")
        try:
            parsed = json.loads(serialized)
        except Exception as exc:
            raise MB9ValidationError(f"M-B9_FRESH_JSON_PARSE:{row.get('scenario_id')}:{exc}") from exc
        if not _finite(parsed) or not required_output_fields.issubset(parsed):
            raise MB9ValidationError(f"M-B9_FRESH_JSON_SCHEMA_OR_NONFINITE:{row.get('scenario_id')}")
        _assert_equal(parsed, row["node_output"], f"M-B9_FRESH_JSON_NODE_MISMATCH:{row.get('scenario_id')}:{row.get('seed')}")
        recomputed = _risk_engine_recompute(row)
        for field in RISK_CORE_FIELDS:
            if _canonical(parsed.get(field)) != _canonical(recomputed.get(field)):
                raise MB9ValidationError(f"M-B9_FRESH_JSON_RISK_MISMATCH:{field}:{row.get('scenario_id')}")
        _assert_equal(parsed.get("sensors"), row["node_output"].get("sensors"), f"M-B9_FRESH_JSON_SENSOR_MISMATCH:{row.get('scenario_id')}")
        checked += 1
    return {"fresh_json_records_checked": checked, "fresh_json_exact": True}


def _compare_fresh_audits(root: Path, loaded: dict[str, Any], fresh_records: list[dict[str, Any]]) -> dict[str, Any]:
    stored_records = loaded["scenario_results.json"]["records"]
    stored_by_key = _records_by_key(stored_records, "M-B9_STORED_SCENARIOS")
    fresh_by_key = _records_by_key(fresh_records, "M-B9_FRESH_SCENARIOS")
    stored_keys = set(stored_by_key)
    if not stored_keys.issubset(fresh_by_key):
        raise MB9ValidationError(f"M-B9_STORED_SCENARIO_MISSING_FROM_FRESH:{sorted(stored_keys - set(fresh_by_key))}")
    for key in sorted(stored_keys, key=str):
        _assert_equal(stored_by_key[key], fresh_by_key[key], f"M-B9_STORED_VS_FRESH_SCENARIO_MISMATCH:{key}")

    def compare_audit(name: str, expected_rows: list[dict[str, Any]]) -> None:
        actual_rows = loaded[name].get("records")
        if not isinstance(actual_rows, list):
            raise MB9ValidationError(f"M-B9_FRESH_AUDIT_RECORDS:{name}")
        actual_by_key = _records_by_key(actual_rows, f"M-B9_FRESH_AUDIT:{name}")
        expected_by_key = _records_by_key(expected_rows, f"M-B9_EXPECTED_AUDIT:{name}")
        if set(actual_by_key) != set(expected_by_key):
            raise MB9ValidationError(f"M-B9_FRESH_AUDIT_KEYS:{name}")
        for key in expected_by_key:
            _assert_equal(actual_by_key[key], expected_by_key[key], f"M-B9_STORED_VS_FRESH_AUDIT_MISMATCH:{name}:{key}")

    expected_inference = [_expected_inference_row(fresh_by_key[key]) for key in sorted(stored_keys, key=str)]
    expected_fallback = [_expected_fallback_row(fresh_by_key[key]) for key in sorted(stored_keys, key=str)]
    required_fault_contract = set(SCENARIOS[3:]) - {"N_VALID_EXPLICIT_FINALIST"}
    expected_fault = [_expected_fault_row(fresh_by_key[key]) for key in sorted(stored_keys, key=str) if fresh_by_key[key]["scenario_id"] in required_fault_contract]
    expected_risk = []
    for key in sorted(stored_keys, key=str):
        fresh = fresh_by_key[key]
        expected_risk.append(_expected_risk_row(fresh, recomputed=_risk_engine_recompute(fresh)))
    expected_json = []
    for key in sorted(stored_keys, key=str):
        fresh = fresh_by_key[key]
        parsed = json.loads(fresh["json_output"])
        expected_json.append({
            "scenario_id": fresh["scenario_id"],
            "seed": fresh.get("seed"),
            "serialized_with": "SafeNestRiskOutput.to_json",
            "parsed_schema_version": parsed.get("metadata", {}).get("schema_version"),
            "finite": True,
            "parse_success": True,
            "schema_fields_present": True,
        })
    compare_audit("inference_result_audit.json", expected_inference)
    compare_audit("fallback_audit.json", expected_fallback)
    compare_audit("fault_timeout_stale_audit.json", expected_fault)
    compare_audit("risk_input_audit.json", expected_risk)
    compare_audit("json_output_audit.json", expected_json)
    return {
        "stored_vs_fresh_scenario_count": len(stored_keys),
        "stored_vs_fresh_scenario_gate": True,
        "fresh_inference_result_gate": True,
        "fresh_fallback_gate": True,
        "fresh_fault_stale_timeout_gate": True,
        "fresh_risk_input_gate": True,
        "fresh_risk_engine_recomputation_gate": True,
        "fresh_json_gate": True,
    }


def _assert_phase_manifest(root: Path, seed: int, expected_stage: dict[str, Any]) -> dict[str, Any]:
    path = root / RUNTIME_DIR_REL / f"seed{seed}_runtime_manifest.json"
    if not path.is_file():
        raise MB9ValidationError(f"M-B9_RUNTIME_MANIFEST_MISSING:{seed}")
    manifest = load_json(path)
    if manifest.get("schema_version") != "M-B9_RUNTIME_MANIFEST_V1" or manifest.get("phase_id") != "M-B9":
        raise MB9ValidationError(f"M-B9_RUNTIME_MANIFEST_SCHEMA:{seed}")
    model = manifest.get("runtime_model")
    preprocessing = manifest.get("preprocessing")
    if not isinstance(model, dict) or not isinstance(preprocessing, dict):
        raise MB9ValidationError(f"M-B9_RUNTIME_MANIFEST_SECTIONS:{seed}")
    if int(model.get("seed", -1)) != seed:
        raise MB9ValidationError(f"M-B9_RUNTIME_MANIFEST_SEED:{seed}")
    if model.get("model_id") == "mmwave_resp_int8" or model.get("path") == "models/mmwave/mmwave_resp_int8_v0.1.0.tflite":
        raise MB9ValidationError(f"M-B9_SHARED_DEFAULT_MODEL_USED:{seed}")
    model_path = Path(model.get("path", ""))
    if model_path.is_absolute() or ".." in model_path.parts:
        raise MB9ValidationError(f"M-B9_ABSOLUTE_OR_ESCAPE_PATH:{seed}")
    artifact_path = root / model_path
    if not artifact_path.is_file():
        raise MB9ValidationError(f"M-B9_FINALIST_ARTIFACT_MISSING:{seed}")
    actual_sha = sha256_file(artifact_path)
    actual_bytes = artifact_path.stat().st_size
    if actual_sha != model.get("expected_sha256") or actual_sha != model.get("sha256") or actual_bytes != int(model.get("expected_bytes", -1)) or actual_bytes != int(model.get("bytes", -1)):
        raise MB9ValidationError(f"M-B9_FINALIST_ARTIFACT_IDENTITY_MISMATCH:{seed}")
    if actual_sha != expected_stage.get("sha256") or actual_bytes != int(expected_stage.get("bytes", -1)):
        raise MB9ValidationError(f"M-B9_M6_STAGE_IDENTITY_MISMATCH:{seed}")
    tensor = inspect_tflite(artifact_path)
    expected_input = model.get("input", {})
    expected_output = model.get("output", {})
    for actual, expected, side in ((tensor, expected_input, "input"), (tensor, expected_output, "output")):
        prefix = "input_" if side == "input" else "output_"
        if actual[f"{prefix}dtype"] != expected.get("dtype") or actual[f"{prefix}shape"] != expected.get("shape"):
            raise MB9ValidationError(f"M-B9_TENSOR_CONTRACT:{seed}:{side}")
        if not np.isclose(actual[f"{prefix}scale"], float(expected.get("scale")), rtol=0, atol=1e-12) or actual[f"{prefix}zero_point"] != int(expected.get("zero_point")):
            raise MB9ValidationError(f"M-B9_QUANTIZATION_CONTRACT:{seed}:{side}")
    if tensor["input_dtype"] != "int8" or tensor["output_dtype"] != "int8" or not tensor["flex_select_absent"]:
        raise MB9ValidationError(f"M-B9_STRICT_INT8_OR_FLEX_SELECT:{seed}")
    required_pre = {
        "profile_id": "M-B1_D0_B1_Z1",
        "profile_name": "BPF_ZSCORE",
        "detrend": False,
        "bpf": True,
        "zscore": True,
        "sample_rate_hz": 10.0,
        "bpf_lowcut_hz": 0.1,
        "bpf_highcut_hz": 0.5,
        "bpf_order": 4,
        "zscore_fit_split": "TRAIN",
    }
    for key, value in required_pre.items():
        actual = preprocessing.get(key)
        if isinstance(value, float):
            if not np.isclose(float(actual), value, rtol=0, atol=1e-12):
                raise MB9ValidationError(f"M-B9_PREPROCESSING_CONTRACT:{seed}:{key}")
        elif actual != value:
            raise MB9ValidationError(f"M-B9_PREPROCESSING_CONTRACT:{seed}:{key}")
    if not np.isfinite(float(preprocessing.get("zscore_mean"))) or not np.isfinite(float(preprocessing.get("zscore_std"))) or float(preprocessing["zscore_std"]) <= 0:
        raise MB9ValidationError(f"M-B9_PREPROCESSING_STATS:{seed}")
    return manifest


def _fresh_runtime_identity(root: Path, manifests: dict[int, dict[str, Any]], selected: dict[str, Any], stats: dict[str, float]) -> dict[str, Any]:
    rows = []
    for seed in SEEDS:
        runtime_path = root / RUNTIME_DIR_REL / f"seed{seed}_runtime_manifest.json"
        runtime = MMWaveInterpreter(root, runtime_manifest_path=runtime_path)
        pre = manifests[seed]["preprocessing"]
        if not np.isclose(float(pre["zscore_mean"]), stats["mean"], rtol=0, atol=0) or not np.isclose(float(pre["zscore_std"]), stats["std"], rtol=0, atol=0):
            raise MB9ValidationError(f"M-B9_STATS_NOT_M6:{seed}")
        for label in LABELS:
            window = np.asarray(selected[label]["signal"], dtype=np.float64)
            bpf = transform_signals(window.reshape(1, 300), False, True, False, None)[0]
            zscore = transform_signals(window.reshape(1, 300), False, True, True, stats)[0]
            model_ready = zscore.astype(np.float32).reshape(1, 300, 1)
            trace = runtime.preprocess_trace(window)
            direct = direct_prediction(runtime.model_path, model_ready, float(runtime.input_info["quantization"][0]), int(runtime.input_info["quantization"][1]))
            pred = runtime.predict(window)
            if not np.array_equal(np.asarray(trace["bpf_output"]), bpf.reshape(1, 300)):
                raise MB9ValidationError(f"M-B9_RUNTIME_PREPROCESSING_MISMATCH:BPF:{seed}:{label}")
            if not np.array_equal(np.asarray(trace["zscore_output"]), zscore.reshape(1, 300)):
                raise MB9ValidationError(f"M-B9_RUNTIME_PREPROCESSING_MISMATCH:ZSCORE:{seed}:{label}")
            if not np.array_equal(np.asarray(trace["model_ready"]), model_ready):
                raise MB9ValidationError(f"M-B9_RUNTIME_PREPROCESSING_MISMATCH:MODEL_READY:{seed}:{label}")
            if not np.array_equal(np.asarray(trace["quantized_input"]), direct["input_int8"]):
                raise MB9ValidationError(f"M-B9_RUNTIME_PREPROCESSING_MISMATCH:INT8:{seed}:{label}")
            if not np.array_equal(np.asarray(runtime.last_raw_output), direct["output_int8"]):
                raise MB9ValidationError(f"M-B9_RUNTIME_PREDICTION_MISMATCH:INT8_OUTPUT:{seed}:{label}")
            if not np.array_equal(np.asarray(pred.probabilities, dtype=np.float32), np.asarray(direct["probabilities"], dtype=np.float32)) or pred.class_index != direct["class_index"]:
                raise MB9ValidationError(f"M-B9_RUNTIME_PREDICTION_MISMATCH:PROBABILITY_OR_TOP1:{seed}:{label}")
            rows.append({"seed": seed, "label": label, "bpf_exact": True, "zscore_exact": True, "model_ready_exact": True, "input_int8_exact": True, "output_int8_exact": True, "probabilities_exact": True, "top1_exact": True})
    return {"rows": rows, "all_exact": True}


def _fresh_scenarios(root: Path, runtime_paths: dict[int, Path], selected: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def run(provider: FinalistMockProvider, scenario_id: str, seed: int | None) -> None:
        records.append(run_node_once(root, provider, scenario_id=scenario_id, seed=seed))

    # Every finalist and every model-driven class.
    for seed in SEEDS:
        for scenario_id, label in (("A_NORMAL", "NORMAL"), ("B_RAPID_OR_ABNORMAL", "RAPID_OR_ABNORMAL"), ("C_APNEA", "APNEA")):
            item = selected[label]
            run(FinalistMockProvider(root, runtime_paths[seed], raw_window=item["signal"], scenario_truth_class=label, selection_metadata={k: v for k, v in item.items() if k != "signal"}), scenario_id, seed)
    item = selected["NORMAL"]
    for scenario_id, seed, mode in (
        ("D_INSUFFICIENT_HISTORY", 42, "INSUFFICIENT_HISTORY"),
        ("E_INVALID_SHAPE", 42, "INVALID_SHAPE"),
        ("F_NAN", 42, "NAN"),
        ("G_INF", 42, "INF"),
        ("H_STALE", 42, "STALE"),
        ("I_PROVIDER_SENSOR_FAULT", 42, "PROVIDER_FAULT"),
        ("J_READ_EXCEPTION", 42, "READ_EXCEPTION"),
        ("K_TIMEOUT", 42, "TIMEOUT"),
        ("M_SHA_MISMATCH", 42, "SHA_MISMATCH"),
        ("O_NOT_CONNECTED_PROVIDER", 42, "NOT_CONNECTED"),
    ):
        run(FinalistMockProvider(root, runtime_paths[seed], raw_window=item["signal"], selection_metadata={k: v for k, v in item.items() if k != "signal"}, failure_mode=mode), scenario_id, seed)
    run(FinalistMockProvider(root, root / RUNTIME_DIR_REL / "missing_runtime_manifest.json", raw_window=item["signal"], selection_metadata={k: v for k, v in item.items() if k != "signal"}, failure_mode="MISSING_MODEL"), "L_MISSING_MODEL", None)
    run(FinalistMockProvider(root, runtime_paths[42], raw_window=item["signal"], scenario_truth_class="NORMAL", selection_metadata={k: v for k, v in item.items() if k != "signal"}), "N_VALID_EXPLICIT_FINALIST", 42)
    # Negative scenario-truth injection: the model must still own the state and score.
    disagreement = FinalistMockProvider(root, runtime_paths[42], raw_window=item["signal"], scenario_truth_class="APNEA", selection_metadata={k: v for k, v in item.items() if k != "signal"})
    run(disagreement, "NEGATIVE_SCENARIO_TRUTH_DISAGREEMENT", 42)
    return records


def _validate_scenarios(records: list[dict[str, Any]]) -> dict[str, Any]:
    base = {item["scenario_id"] for item in records}
    missing = [item for item in SCENARIOS if item not in base]
    if missing:
        raise MB9ValidationError(f"M-B9_SCENARIO_MISSING:{missing}")
    for item in records:
        mm = item["node_output"]["sensors"]["mmwave"]
        metadata = mm.get("metadata", {})
        if item["scenario_id"] in {"A_NORMAL", "B_RAPID_OR_ABNORMAL", "C_APNEA", "N_VALID_EXPLICIT_FINALIST", "NEGATIVE_SCENARIO_TRUTH_DISAGREEMENT"} and mm.get("valid"):
            predicted = metadata.get("model_predicted_class")
            if metadata.get("score_source") != "MODEL_PREDICTION" or metadata.get("fallback_used") is not False:
                raise MB9ValidationError(f"M-B9_VALID_MODEL_FALLBACK_OR_SCORE_SOURCE:{item['scenario_id']}:{item.get('seed')}")
            expected_score = {"NORMAL": 0.0, "RAPID_OR_ABNORMAL": 0.5, "APNEA": 1.0}.get(predicted)
            if expected_score is None or float(mm.get("score")) != expected_score or mm.get("state") != predicted:
                raise MB9ValidationError(f"M-B9_SCENARIO_TRUTH_FORCED_STATE_OR_SCORE:{item['scenario_id']}:{item.get('seed')}")
        if item["scenario_id"] == "H_STALE" and "mmwave" not in item["node_output"].get("stale_sensors", []):
            raise MB9ValidationError("M-B9_STALE_NOT_DETECTED")
        if item["scenario_id"] == "K_TIMEOUT" and item["node_output"]["sensors"]["mmwave"].get("error") != "PROVIDER_READ_TIMEOUT":
            raise MB9ValidationError("M-B9_TIMEOUT_NOT_DETECTED")
        if item["scenario_id"] == "NEGATIVE_SCENARIO_TRUTH_DISAGREEMENT":
            truth = item.get("scenario_truth_class")
            if truth != "APNEA" or truth == item.get("model_predicted_class"):
                raise MB9ValidationError("M-B9_NEGATIVE_TRUTH_DISAGREEMENT_NOT_INJECTED")
    return {"count": len(records), "all_contracts_valid": True, "required_scenarios_present": {item: item in base for item in SCENARIOS}, "negative_truth_disagreement_passed": True}


def _validate_saved_checksums(root: Path, out_dir: Path) -> dict[str, Any]:
    checksum_path = out_dir / "checksums.sha256"
    entries = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, path = line.split("  ", 1)
        entries[path] = digest
        target = root / path
        if not target.is_file() or sha256_file(target) != digest:
            raise MB9ValidationError(f"M-B9_CHECKSUM_MISMATCH:{path}")
    expected = {
        rel(path, root)
        for path in out_dir.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    }
    report_path = root / "docs/reports/20260812_Codex_M-B9_Mock_E2E_Runtime_01.md"
    if report_path.is_file():
        expected.add(rel(report_path, root))
    if set(entries) != expected:
        raise MB9ValidationError(
            f"M-B9_CHECKSUM_COVERAGE_MISMATCH:missing={sorted(expected - set(entries))}:extra={sorted(set(entries) - expected)}"
        )
    if any(path.suffix.lower() in {".tflite", ".onnx", ".h5", ".keras"} for path in out_dir.rglob("*")):
        raise MB9ValidationError("M-B9_MODEL_BINARY_DUPLICATED_IN_EVIDENCE")
    return {"entry_count": len(entries), "coverage_complete": True}


def _negative_case_detected(case_id: str, root: Path) -> bool:
    """Run the real validator against a temporary, copy-on-write workspace.

    Every mutation below changes an actual M-B9 evidence artifact.  The
    mutated artifact's checksum is refreshed only when needed to reach the
    intended semantic gate; checksum-corruption intentionally leaves it stale.
    """
    if case_id not in NEGATIVE_CASES:
        return False

    def clone_workspace() -> Path:
        temp_root = Path(tempfile.mkdtemp(prefix="safenest_m_b9_corrupt_", dir=str(root.parent)))
        shutil.copytree(
            root,
            temp_root,
            symlinks=True,
            copy_function=os.link,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
            dirs_exist_ok=True,
        )
        return temp_root

    def write_json(path: Path, value: Any, *, allow_nan: bool = False) -> None:
        path.unlink()
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=allow_nan) + "\n", encoding="utf-8")

    def update_checksum(temp_root: Path, path: Path) -> None:
        checksum_path = temp_root / OUT_DIR_REL / "checksums.sha256"
        rel_path = rel(path, temp_root)
        digest = sha256_file(path)
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
        replaced = False
        for index, line in enumerate(lines):
            if "  " in line and line.split("  ", 1)[1] == rel_path:
                lines[index] = f"{digest}  {rel_path}"
                replaced = True
                break
        if not replaced:
            raise RuntimeError(f"checksum entry missing for {rel_path}")
        checksum_path.unlink()
        checksum_path.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")

    def mutate(temp_root: Path) -> None:
        out = temp_root / OUT_DIR_REL
        manifest_path = out / "runtime_manifests" / "seed42_runtime_manifest.json"
        manifest = load_json(manifest_path)
        model = manifest["runtime_model"]
        preprocessing = manifest["preprocessing"]
        if case_id == "default_historical_model_substituted":
            model["model_id"] = "mmwave_resp_int8"
            write_json(manifest_path, manifest); update_checksum(temp_root, manifest_path); return
        if case_id == "wrong_phase_sha256":
            model["expected_sha256"] = "0" * 64
            write_json(manifest_path, manifest); update_checksum(temp_root, manifest_path); return
        if case_id == "wrong_phase_bytes":
            model["expected_bytes"] = int(model["bytes"]) + 1
            write_json(manifest_path, manifest); update_checksum(temp_root, manifest_path); return
        if case_id == "wrong_seed":
            model["seed"] = 99
            write_json(manifest_path, manifest); update_checksum(temp_root, manifest_path); return
        if case_id == "wrong_input_dtype":
            model["input"]["dtype"] = "float32"
            write_json(manifest_path, manifest); update_checksum(temp_root, manifest_path); return
        if case_id == "wrong_input_quantization":
            model["input"]["scale"] = 0.1
            write_json(manifest_path, manifest); update_checksum(temp_root, manifest_path); return
        if case_id == "wrong_output_quantization":
            model["output"]["zero_point"] = 0
            write_json(manifest_path, manifest); update_checksum(temp_root, manifest_path); return
        if case_id == "wrong_preprocessing_profile":
            preprocessing["profile_name"] = "ZSCORE_ONLY"
            write_json(manifest_path, manifest); update_checksum(temp_root, manifest_path); return
        if case_id == "wrong_bpf_contract":
            preprocessing["bpf_lowcut_hz"] = 0.2
            write_json(manifest_path, manifest); update_checksum(temp_root, manifest_path); return
        if case_id == "wrong_zscore_stats":
            preprocessing["zscore_std"] = 1.0
            write_json(manifest_path, manifest); update_checksum(temp_root, manifest_path); return
        if case_id == "manifest_absolute_path":
            model["path"] = "/tmp/model.tflite"
            write_json(manifest_path, manifest); update_checksum(temp_root, manifest_path); return
        if case_id == "manifest_path_traversal":
            model["path"] = "../models/mmwave/mmwave_resp_int8_v0.1.0.tflite"
            write_json(manifest_path, manifest); update_checksum(temp_root, manifest_path); return
        if case_id == "missing_model":
            (out / "runtime_manifests" / "seed43_runtime_manifest.json").unlink()
            return
        if case_id == "model_binary_duplicated":
            source = temp_root / model["path"]
            duplicate = out / "duplicated_finalist_model.tflite"
            shutil.copy2(source, duplicate)
            return

        scenario_path = out / "scenario_results.json"
        scenario = load_json(scenario_path)
        valid = next(row for row in scenario["records"] if row["scenario_id"] == "A_NORMAL" and row["seed"] == 42)
        if case_id == "runtime_prediction_mismatch":
            pred_path = out / "runtime_prediction_identity.json"
            prediction = load_json(pred_path)
            row = prediction["rows"][0]
            row["runtime_class_index"] = (int(row["direct_class_index"]) + 1) % 3
            write_json(pred_path, prediction); update_checksum(temp_root, pred_path); return
        if case_id == "scenario_truth_forces_state":
            valid["mmwave_result"]["state"] = valid["mmwave_result"]["metadata"]["scenario_truth_class"]
            write_json(scenario_path, scenario); update_checksum(temp_root, scenario_path); return
        if case_id == "scenario_truth_forces_score":
            valid["mmwave_result"]["score"] = 1.0
            write_json(scenario_path, scenario); update_checksum(temp_root, scenario_path); return
        if case_id == "hidden_fallback":
            valid["mmwave_result"]["metadata"]["fallback_used"] = True
            write_json(scenario_path, scenario); update_checksum(temp_root, scenario_path); return
        if case_id == "missing_fallback_reason":
            fallback_path = out / "fallback_audit.json"
            fallback = load_json(fallback_path)
            fallback["records"][0]["fallback_used"] = True
            fallback["records"][0]["reason"] = None
            write_json(fallback_path, fallback); update_checksum(temp_root, fallback_path); return

        fault_map = {
            "invalid_shape": "E_INVALID_SHAPE",
            "nan": "F_NAN",
            "inf": "G_INF",
            "insufficient_history": "D_INSUFFICIENT_HISTORY",
            "stale_timestamp": "H_STALE",
            "provider_fault": "I_PROVIDER_SENSOR_FAULT",
            "read_exception": "J_READ_EXCEPTION",
            "timeout": "K_TIMEOUT",
            "sha_mismatch": "M_SHA_MISMATCH",
        }
        if case_id in fault_map:
            fault_path = out / "fault_timeout_stale_audit.json"
            fault = load_json(fault_path)
            row = next(item for item in fault["records"] if item["scenario_id"] == fault_map[case_id])
            row["valid"] = True
            if case_id == "stale_timestamp":
                row["stale_sensors"] = []
            else:
                row["error"] = None
            write_json(fault_path, fault); update_checksum(temp_root, fault_path); return
        if case_id == "risk_input_mismatch":
            risk_path = out / "risk_input_audit.json"
            risk = load_json(risk_path)
            risk["records"][0]["exact_inference_results_entering_risk"]["mmwave"]["score"] = 1.0
            write_json(risk_path, risk); update_checksum(temp_root, risk_path); return
        if case_id == "json_schema":
            json_path = out / "json_output_audit.json"
            audit = load_json(json_path)
            audit["records"][0]["schema_fields_present"] = False
            write_json(json_path, audit); update_checksum(temp_root, json_path); return
        if case_id == "json_nonfinite":
            json_path = out / "json_output_audit.json"
            audit = load_json(json_path)
            audit["records"][0]["nonfinite_probe"] = float("nan")
            write_json(json_path, audit, allow_nan=True); update_checksum(temp_root, json_path); return
        if case_id == "locked_test_access":
            locked_path = out / "locked_test_access_audit.json"
            locked = load_json(locked_path)
            locked["locked_test_inputs_loaded"] = True
            write_json(locked_path, locked); update_checksum(temp_root, locked_path); return
        if case_id == "checksum_corruption":
            summary_path = out / "m_b9_summary.json"
            summary = load_json(summary_path)
            summary["gate_status"] = "PASS"
            write_json(summary_path, summary)
            return
        raise RuntimeError(f"unhandled corruption case: {case_id}")

    temp_root = clone_workspace()
    try:
        mutate(temp_root)
        try:
            validate_m_b9_artifacts(temp_root, run_fresh=False, run_negative=False)
        except MB9ValidationError:
            return True
        return False
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def validate_m_b9_artifacts(
    root_dir: Path | None = None,
    *,
    run_fresh: bool = True,
    run_negative: bool = True,
) -> dict[str, Any]:
    root = Path(root_dir or ROOT_DIR).resolve()
    out_dir = root / OUT_DIR_REL
    loaded = _load_all_json(out_dir)
    serialized = json.dumps(loaded, ensure_ascii=False)
    if any(token in serialized for token in ("/Users/", "/private/", "file://")):
        raise MB9ValidationError("M-B9_ABSOLUTE_PATH_IN_EVIDENCE")
    stage = load_stage_artifacts(root)
    manifests = {seed: _assert_phase_manifest(root, seed, stage[seed]) for seed in SEEDS}
    authoritative_stats = load_json(root / "datasets/mmwave/manifests/M-B1_preprocessing_ablation/train_fit_statistics.json")["zscore_statistics"]["M-B1_D0_B1_Z1"]
    for seed, manifest in manifests.items():
        preprocessing = manifest["preprocessing"]
        if not np.isclose(float(preprocessing.get("zscore_mean")), float(authoritative_stats["mean"]), rtol=0, atol=0) or not np.isclose(float(preprocessing.get("zscore_std")), float(authoritative_stats["std"]), rtol=0, atol=0):
            raise MB9ValidationError(f"M-B9_PREPROCESSING_STATS_NOT_M-B1:{seed}")
    default_manifest_path = root / "models/model_manifest.json"
    default_manifest = load_json(default_manifest_path)
    if default_manifest["models"]["mmwave"]["model_id"] != "mmwave_resp_int8" or default_manifest["models"]["mmwave"]["validation_status"] != "BLOCKED":
        raise MB9ValidationError("M-B9_SHARED_DEFAULT_MANIFEST_CHANGED")
    checksums = _validate_saved_checksums(root, out_dir)
    stored_audit = _validate_stored_audits(loaded)
    stored_jsonl = _validate_scenario_jsonl_consistency(out_dir, loaded["scenario_results.json"]["records"])
    stored_prediction = _validate_stored_runtime_prediction_identity(loaded)
    experiment = loaded["experiment_contract.json"]
    environment = loaded["run_environment.json"]
    summary = loaded["m_b9_summary.json"]
    for value, name in ((experiment, "experiment_contract"), (environment, "run_environment"), (summary, "m_b9_summary")):
        if "formal_m_b8_latency_measurement_started" in value or "m_b8_latency_measurement_started" in value:
            raise MB9ValidationError(f"M-B9_AMBIGUOUS_M-B8_WORDING:{name}")
    if experiment.get("formal_m_b8_latency_measurement_rerun_during_m_b9") is not False or environment.get("formal_m_b8_latency_measurement_rerun_during_m_b9") is not False or summary.get("formal_m_b8_latency_measurement_rerun_during_m_b9") is not False:
        raise MB9ValidationError("M-B9_M8_RERUN_WORDING_CONTRACT")

    # Upstream M-B8 validator recursively revalidates M-B7…M-B0 and A5/A6.
    # Run it after local evidence gates so real corruption probes fail at the
    # intended M-B9 semantic boundary without masking the mutation upstream.
    upstream = validate_m_b8_artifacts(root_dir=root)
    if not upstream.get("validation_success"):
        raise MB9ValidationError("M-B9_BLOCKER_UPSTREAM_M-B8_VALIDATOR_FAILED")
    guard = PhaseBAccessGuard(root_dir=root)
    train = guard.get_model_selection_dataset("TRAIN")
    stats = fit_train_zscore_statistics(train["signals"], False, True)
    selection = select_validation_inputs(root)
    saved_selection = loaded["scenario_input_selection.json"]
    if saved_selection.get("locked_test_access") != 0 or saved_selection.get("validation_window_count") != 79:
        raise MB9ValidationError("M-B9_INPUT_SCOPE_OR_LOCKED_TEST")
    for label in LABELS:
        stored = next(item for item in saved_selection["selected"] if item["safenest_label"] == label)
        if stored["canonical_sample_index"] != selection["selected"][label]["canonical_sample_index"] or stored["split"] != "VALIDATION":
            raise MB9ValidationError(f"M-B9_SELECTION_PROVENANCE:{label}")
        signal = np.asarray(selection["selected"][label]["signal"], dtype=np.float64)
        if stored.get("canonical_signal_hash") != array_sha256(signal, np.float64):
            raise MB9ValidationError(f"M-B9_SELECTION_SIGNAL_HASH:{label}")
    fresh_identity = _fresh_runtime_identity(root, manifests, selection["selected"], stats)
    runtime_paths = {seed: root / RUNTIME_DIR_REL / f"seed{seed}_runtime_manifest.json" for seed in SEEDS}
    fresh_records = _fresh_scenarios(root, runtime_paths, selection["selected"]) if run_fresh else []
    scenario_audit = _validate_scenarios(fresh_records) if run_fresh else {"count": 0, "all_contracts_valid": True}
    fresh_risk = _validate_fresh_risk_records(fresh_records) if run_fresh else {"fresh_risk_engine_recomputation_exact": True, "fresh_risk_records_checked": 0}
    fresh_json = _validate_fresh_json_records(fresh_records) if run_fresh else {"fresh_json_exact": True, "fresh_json_records_checked": 0}
    fresh_closure = _compare_fresh_audits(root, loaded, fresh_records) if run_fresh else {
        "stored_vs_fresh_scenario_gate": False,
        "fresh_inference_result_gate": False,
        "fresh_fallback_gate": False,
        "fresh_fault_stale_timeout_gate": False,
        "fresh_risk_input_gate": False,
        "fresh_risk_engine_recomputation_gate": False,
        "fresh_json_gate": False,
    }
    negative_results = {case: _negative_case_detected(case, root) for case in NEGATIVE_CASES} if run_negative else {}
    if run_negative and not all(negative_results.values()):
        raise MB9ValidationError(f"M-B9_NEGATIVE_TEST_GAP:{[k for k, v in negative_results.items() if not v]}")
    if any("LOCKED_TEST" in json.dumps(value) and "access" in json.dumps(value).lower() for value in [loaded["experiment_contract.json"], loaded["runtime_manifest_contract.json"]]):
        # The contract is allowed to name the prohibition; only an access event
        # is forbidden.  This branch intentionally does nothing.
        pass
    return {
        "phase_id": "M-B9",
        "validation_success": True,
        "gate_status": "PASS_WITH_WARNINGS",
        "upstream_m_b8": upstream,
        "strict_finalist_identity": True,
        "runtime_preprocessing_identity": fresh_identity,
        "fresh_bounded_scenarios": scenario_audit,
        "fresh_risk_engine": fresh_risk,
        "fresh_json": fresh_json,
        "fresh_audit_closure": fresh_closure,
        "stored_audit_closure": stored_audit | stored_jsonl | stored_prediction,
        "saved_checksum_audit": checksums,
        "locked_test_accesses": 0,
        "negative_corruption_cases": negative_results,
        "shared_default_manifest_used_for_finalist_inference": False,
        "formal_m_b8_latency_measurement_rerun_during_m_b9": False,
        "findings": [{"classification": "NON-BLOCKING IMPROVEMENT", "code": "M-B9_MOCK_SCOPE_ONLY", "detail": "No production, Pi, MR60, clinical, or formal latency claim."}],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT_DIR))
    args = parser.parse_args()
    try:
        result = validate_m_b9_artifacts(Path(args.root).resolve(), run_fresh=True)
    except Exception as exc:
        print(f"M-B9 validation failed: {type(exc).__name__}: {exc}")
        return 1
    print("Standalone M-B9 Explicit-Finalist Mock E2E Runtime Validation Result:")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
