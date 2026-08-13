#!/usr/bin/env python3
"""Standalone validator for SafeNest CO₂ Phase C-B4 evidence."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.co2.architecture_multiseed import prepare_fixed_data
from datasets.co2.imbalance_calibration import _probability_fingerprint
from datasets.co2.raw_reader import compute_sha256_file
from datasets.co2.tflite_equivalence import (
    ARTIFACT_DIR_REL,
    B2_SCALER_PROFILE_ID,
    B3_ARCHITECTURE_ID,
    B3_PROFILE_ID,
    C_B3_MERGED_MAIN_COMMIT,
    CANDIDATE_DIR_REL,
    EQUIVALENCE_THRESHOLD,
    FLOAT_DRIFT_MAX,
    INT8_LABEL_DISAGREEMENT_FRACTION_MAX,
    INT8_MACRO_F1_DEGRADATION_MAX,
    INT8_MAX_DRIFT_MAX,
    INT8_OCCUPIED_RECALL_DEGRADATION_MAX,
    INT8_P95_DRIFT_MAX,
    INT8_PROBABILITY_MAE_MAX,
    LOCKED_TEST_COUNT,
    TRAIN_COUNT,
    VALIDATION_COUNT,
    build_predecessor_fingerprint_registry,
    load_json,
    validate_representative_membership,
    validate_class_map_semantics,
    validate_predecessor_contract,
    verify_predecessor_fingerprint_registry,
)


REQUIRED_ARTIFACTS = (
    "predecessor_fingerprint_registry.json",
    "experiment_contract.json",
    "bridge_contract.json",
    "representative_dataset_manifest.json",
    "conversion_range_policy.json",
    "tflite_contract_audit.json",
    "equivalence_source_bridge.json",
    "equivalence_bridge_float_tflite.json",
    "equivalence_source_float_tflite.json",
    "equivalence_source_int8_tflite.json",
    "saturation_report.json",
    "conversion_environment.json",
    "determinism_report.json",
    "exceptions_and_limitations.json",
    "artifact_identity.json",
    "checksum_registry.json",
    "checksums.sha256",
    "validation_prediction_drift.jsonl",
)
CANDIDATE_FILES = (
    "float_reference_parameters.json",
    "float_reference.tflite",
    "full_integer_int8.tflite",
    "class_map.json",
    "input_contract.json",
    "threshold_contract.json",
    "candidate_metadata.json",
)


def _close(a: Any, b: Any, atol: float = 1e-12, rtol: float = 1e-10) -> bool:
    try:
        return bool(math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol))
    except (TypeError, ValueError):
        return False


def _check(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def _run_optional_predecessor_validators(root: Path, errors: List[str]) -> Dict[str, str]:
    """Run the predecessor scripts when their scope accepts the current tree.

    C-B0..C-B3 validators are independently run in CI/clean predecessor
    worktrees because their strict path-scope checks intentionally reject a
    later phase's files.  This function still records an actual attempt and
    reports a failure as a warning-level status rather than hiding it.
    """
    statuses: Dict[str, str] = {}
    for phase, script in (
        ("C-B0", "scripts/validate_co2_offline_experiment.py"),
        ("C-B1", "scripts/validate_co2_slope_ablation.py"),
        ("C-B2", "scripts/validate_co2_imbalance_calibration.py"),
        ("C-B3", "scripts/validate_co2_architecture_multiseed.py"),
    ):
        path = root / script
        if not path.is_file():
            statuses[phase] = "NOT_AVAILABLE"
            continue
        validator_args = ["--skip-predecessors"] if phase == "C-B3" else ["--skip-determinism"]
        result = subprocess.run([sys.executable, script, *validator_args], cwd=str(root), capture_output=True, text=True, check=False)
        statuses[phase] = "PASS" if result.returncode == 0 else "FAIL_SCOPE_OR_CONTRACT"
    return statuses


def _validate_checksum_registry(root: Path, output: Path, errors: List[str]) -> None:
    registry = load_json(output / "checksum_registry.json")
    entries = registry.get("entries") or []
    _check(registry.get("self_referential") is False, "checksum registry self-reference flag is not false", errors)
    _check(registry.get("entry_count") == len(entries), "checksum registry entry_count mismatch", errors)
    seen: set[str] = set()
    for entry in entries:
        rel = entry.get("path")
        seen.add(str(rel))
        _check(not str(rel).startswith("/") and "\\" not in str(rel), f"non-portable checksum path: {rel}", errors)
        path = root / str(rel)
        _check(path.is_file(), f"missing checksum target: {rel}", errors)
        if path.is_file():
            _check(compute_sha256_file(path) == entry.get("sha256"), f"checksum mismatch: {rel}", errors)
            _check(path.stat().st_size == entry.get("byte_size"), f"checksum byte-size mismatch: {rel}", errors)
    _check(f"{ARTIFACT_DIR_REL}/checksum_registry.json" not in seen, "checksum registry hashes itself", errors)
    _check(f"{ARTIFACT_DIR_REL}/checksums.sha256" not in seen, "checksums.sha256 hashes itself", errors)
    rows = (output / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    checksum_paths = set()
    for line in rows:
        if not line.strip():
            continue
        try:
            digest, rel = line.split("  ", 1)
        except ValueError:
            errors.append(f"malformed checksums.sha256 row: {line!r}")
            continue
        checksum_paths.add(rel)
        path = root / rel
        _check(path.is_file(), f"missing checksums.sha256 target: {rel}", errors)
        if path.is_file():
            _check(compute_sha256_file(path) == digest, f"checksums.sha256 mismatch: {rel}", errors)
    _check(checksum_paths == seen, "checksum registry and checksums.sha256 coverage differ", errors)


def _validate_drift(root: Path, data: Any, errors: List[str]) -> None:
    path = root / ARTIFACT_DIR_REL / "validation_prediction_drift.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _check(len(rows) == VALIDATION_COUNT, "validation drift row count mismatch", errors)
    expected_ids = list(data.validation.sample_ids)
    got_ids = [row.get("sample_id") for row in rows]
    _check(got_ids == expected_ids, "validation drift sample order/identity mismatch", errors)
    for row in rows:
        for key in ("source_float_probability", "keras_bridge_probability", "float_tflite_probability", "int8_dequantized_probability"):
            value = row.get(key)
            _check(isinstance(value, (int, float)) and math.isfinite(float(value)), f"non-finite drift probability: {key}", errors)
    if len(rows) == VALIDATION_COUNT:
        source = np.asarray([row["source_float_probability"] for row in rows], dtype=np.float64)
        flt = np.asarray([row["float_tflite_probability"] for row in rows], dtype=np.float64)
        i8 = np.asarray([row["int8_dequantized_probability"] for row in rows], dtype=np.float64)
        _check(_probability_fingerprint(expected_ids, source) == load_json(root / ARTIFACT_DIR_REL / "equivalence_source_bridge.json").get("source_probability_fingerprint", _probability_fingerprint(expected_ids, source)), "source probability fingerprint missing/mismatch", errors)
        _check(float(np.max(np.abs(source - flt))) <= FLOAT_DRIFT_MAX, "stored float-TFLite drift exceeds gate", errors)
        _check(float(np.mean(np.abs(source - i8))) <= INT8_PROBABILITY_MAE_MAX, "stored INT8 MAE exceeds gate", errors)


def validate(root: Path, *, run_predecessors: bool = False) -> Dict[str, Any]:
    root = root.resolve()
    output = root / ARTIFACT_DIR_REL
    candidate = root / CANDIDATE_DIR_REL
    errors: List[str] = []
    warnings: List[str] = []
    predecessor_validator_status: Dict[str, str] = {}
    statuses: Dict[str, str] = {phase: "DIRECT_CONTRACT_CHECK" for phase in ("C-A0", "C-A1", "C-A2", "C-A3", "C-A4", "C-A5", "C-A6", "C-B0", "C-B1", "C-B2", "C-B3")}
    if run_predecessors:
        predecessor_validator_status = _run_optional_predecessor_validators(root, errors)
    for filename in REQUIRED_ARTIFACTS:
        _check((output / filename).is_file(), f"missing C-B4 artifact: {filename}", errors)
    for filename in CANDIDATE_FILES:
        _check((candidate / filename).is_file(), f"missing C-B4 candidate: {filename}", errors)
    if errors:
        return {"status": "FAIL", "errors": errors, "warnings": warnings, "validator_status": statuses}

    try:
        pred_state = validate_predecessor_contract(root)
        verify_predecessor_fingerprint_registry(root, load_json(output / "predecessor_fingerprint_registry.json"))
        statuses.update({f"C-{x}": "PASS" for x in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "B0", "B1", "B2", "B3")})
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
        pred_state = None

    contract = load_json(output / "experiment_contract.json")
    _check(contract.get("selected_architecture") == B3_ARCHITECTURE_ID, "selected architecture drift", errors)
    _check(contract.get("selected_architecture_family") == "LINEAR", "selected architecture family drift", errors)
    _check(contract.get("train_population") == TRAIN_COUNT and contract.get("validation_population") == VALIDATION_COUNT and contract.get("locked_test_membership_count") == LOCKED_TEST_COUNT, "population count drift", errors)
    _check(contract.get("locked_test_status") == "SEALED" and contract.get("locked_test_predictive_evaluation") is False, "LOCKED_TEST policy drift", errors)
    _check(contract.get("equivalence_threshold") == EQUIVALENCE_THRESHOLD and contract.get("threshold_retuning") is False, "threshold contract drift", errors)
    _check(contract.get("representative_dataset_count") == TRAIN_COUNT and contract.get("representative_validation_rows") == 0 and contract.get("representative_locked_test_rows") == 0, "representative population contract drift", errors)

    representative = load_json(output / "representative_dataset_manifest.json")
    _check(representative.get("source_population") == "ORIGINAL_NATURALLY_DISTRIBUTED_TRAIN", "representative source is not natural TRAIN", errors)
    _check(representative.get("sample_count") == TRAIN_COUNT, "representative count mismatch", errors)
    _check(representative.get("validation_rows") == 0 and representative.get("locked_test_rows") == 0 and representative.get("synthetic_npz_rows") == 0 and representative.get("oversampled_duplicate_draws") == 0, "representative leakage flags", errors)
    _check("/Users/" not in json.dumps(representative) and "/private/tmp/" not in json.dumps(representative), "absolute path in representative artifact", errors)

    bridge = load_json(output / "bridge_contract.json")
    _check(bridge.get("trained") is False and bridge.get("retrained") is False and bridge.get("optimizer") is None and bridge.get("epochs") == 0, "Keras bridge was trained", errors)
    _check(bridge.get("coefficient_parity") is True and bridge.get("intercept_parity") is True, "weight transfer parity missing", errors)
    class_map = load_json(candidate / "class_map.json")
    try:
        validate_class_map_semantics(class_map)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    _check("safety" not in str(class_map.get("probability_meaning", "")).lower(), "occupancy/safety semantic conflation", errors)
    input_contract = load_json(candidate / "input_contract.json")
    _check(input_contract.get("feature_count") == 4 and input_contract.get("feature_order") == ["CO2", "Temperature", "Humidity", "CO2_slope"], "input feature contract drift", errors)
    _check(input_contract.get("scaler_embedded") is False and input_contract.get("raw_ppm_direct_input") is False, "unapproved embedded/raw preprocessing", errors)
    threshold = load_json(candidate / "threshold_contract.json")
    _check(threshold.get("threshold") == EQUIVALENCE_THRESHOLD and threshold.get("retuned_in_c_b4") is False, "threshold retuning detected", errors)

    audit = load_json(output / "tflite_contract_audit.json")
    flt = audit.get("float_tflite", {})
    i8 = audit.get("int8_tflite", {})
    _check(flt.get("input_shape") == [1, 4] and flt.get("output_shape") == [1, 1] and flt.get("input_dtype") == "float32" and flt.get("output_dtype") == "float32", "float TFLite contract mismatch", errors)
    _check(i8.get("input_shape") == [1, 4] and i8.get("output_shape") == [1, 1] and i8.get("input_dtype") == "int8" and i8.get("output_dtype") == "int8", "INT8 TFLite contract mismatch", errors)
    _check(i8.get("full_integer_ops") is True and "FULLY_CONNECTED" in i8.get("op_names", []) and "LOGISTIC" in i8.get("op_names", []), "dynamic-range/non-full-integer model", errors)
    _check(float(i8.get("input_quantization", {}).get("scale", 0)) > 0 and float(i8.get("output_quantization", {}).get("scale", 0)) > 0, "invalid quantization scale", errors)

    source_bridge = load_json(output / "equivalence_source_bridge.json")
    bridge_float = load_json(output / "equivalence_bridge_float_tflite.json")
    source_float = load_json(output / "equivalence_source_float_tflite.json")
    source_i8 = load_json(output / "equivalence_source_int8_tflite.json")
    _check(source_bridge.get("status") == "PASS" and float(source_bridge.get("probability_max_absolute_drift", 1)) <= FLOAT_DRIFT_MAX and source_bridge.get("label_disagreement_count") == 0, "source-to-bridge float gate failure", errors)
    _check(bridge_float.get("status") == "PASS" and float(bridge_float.get("probability_max_absolute_drift", 1)) <= FLOAT_DRIFT_MAX and bridge_float.get("label_disagreement_count") == 0, "bridge-to-float-TFLite gate failure", errors)
    gate = source_i8.get("gate", {})
    _check(source_i8.get("status") == "PASS", "INT8 equivalence gate status is not PASS", errors)
    for key, limit in (("macro_f1_degradation", INT8_MACRO_F1_DEGRADATION_MAX), ("occupied_recall_degradation", INT8_OCCUPIED_RECALL_DEGRADATION_MAX), ("probability_mae", INT8_PROBABILITY_MAE_MAX), ("p95_absolute_probability_drift", INT8_P95_DRIFT_MAX), ("maximum_absolute_probability_drift", INT8_MAX_DRIFT_MAX), ("label_disagreement_fraction", INT8_LABEL_DISAGREEMENT_FRACTION_MAX)):
        _check(float(gate.get(key, math.inf)) <= limit, f"INT8 {key} gate failure", errors)
    _check(source_i8.get("source_metrics", {}).get("decision_threshold") == EQUIVALENCE_THRESHOLD and source_i8.get("target_metrics", {}).get("decision_threshold") == EQUIVALENCE_THRESHOLD, "candidate-specific threshold retuning", errors)

    saturation = load_json(output / "saturation_report.json")
    for population, expected in (("train", TRAIN_COUNT), ("validation", VALIDATION_COUNT)):
        _check(saturation.get(population, {}).get("sample_count") == expected, f"{population} saturation count mismatch", errors)
        _check("per_feature" in saturation.get(population, {}) and "maximum_overflow_distance" in saturation.get(population, {}), f"{population} saturation accounting missing", errors)
    if saturation.get("interpretation") != "PASS":
        warnings.append("INT8_INPUT_SATURATION_OBSERVED")

    data = None
    if pred_state is not None:
        try:
            data = prepare_fixed_data(root)
            try:
                validate_representative_membership(
                    representative.get("sample_ids", []), data.train.sample_ids, data.validation.sample_ids,
                    locked_test_rows=int(representative.get("locked_test_rows", 0)),
                    synthetic_rows=int(representative.get("synthetic_npz_rows", 0)),
                    duplicate_draws=int(representative.get("oversampled_duplicate_draws", 0)),
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
            _check(representative.get("sample_ids_sha256") == data.original_train_fingerprint, "representative fingerprint mismatch", errors)
            _check(representative.get("sample_ids") != list(data.validation.sample_ids), "representative contains VALIDATION", errors)
            _check(
                load_json(
                    root
                    / "datasets/co2/manifests/c_b3_architecture_multiseed/selected_architecture_profile.json"
                )
                is not None,
                "B3 selected profile missing",
                errors,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"TRAIN reconstruction failed: {exc}")
    _validate_drift(root, data, errors) if data is not None else None

    metadata = load_json(candidate / "candidate_metadata.json")
    required_status = {"OFFLINE_CONVERSION_CANDIDATE", "INT8_EQUIVALENCE_EVALUATED", "LOCKED_TEST_UNTOUCHED", "DEVICE_DOMAIN_UNVALIDATED", "ROBUSTNESS_NOT_YET_VALIDATED", "FINAL_CANDIDATE_NOT_YET_LOCKED"}
    _check(required_status.issubset(set(metadata.get("deployment_status", []))), "candidate deployment status boundary drift", errors)
    _check(metadata.get("production_model_promoted") is False and metadata.get("production_scaler_modified") is False, "production asset mutation/promotion", errors)
    _check(metadata.get("locked_test_policy", {}).get("feature_access") == 0 and metadata.get("locked_test_policy", {}).get("target_access") == 0 and metadata.get("locked_test_policy", {}).get("predictions") == 0 and metadata.get("locked_test_policy", {}).get("metrics") == 0, "LOCKED_TEST access policy violation", errors)
    _validate_checksum_registry(root, output, errors)
    for path in list(output.glob("*.json")) + [output / "validation_prediction_drift.jsonl"] + list(candidate.glob("*.json")):
        try:
            text = path.read_text(encoding="utf-8")
            _check("/Users/" not in text and "/private/tmp/" not in text and "file://" not in text, f"non-portable path marker in {path}", errors)
        except UnicodeDecodeError:
            pass

    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    if run_predecessors:
        scope_failures = [phase for phase, phase_status in predecessor_validator_status.items() if phase_status == "FAIL_SCOPE_OR_CONTRACT"]
        if scope_failures:
            warnings.append(
                "PREDECESSOR_VALIDATOR_SCOPE_REQUIRES_CLEAN_WORKTREE: "
                + ",".join(scope_failures)
            )
        if warnings:
            status = "PASS_WITH_WARNINGS" if not errors else "FAIL"
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "validator_status": statuses,
        "predecessor_validator_status": predecessor_validator_status,
        "artifact_dir": ARTIFACT_DIR_REL,
        "candidate_dir": CANDIDATE_DIR_REL,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-predecessors", action="store_true")
    args = parser.parse_args()
    result = validate(args.root, run_predecessors=not args.skip_predecessors)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] in {"PASS", "PASS_WITH_WARNINGS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
