#!/usr/bin/env python3
"""Standalone fail-closed validator for the SafeNest CO₂ C-B5 lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.co2.b5_robustness import (  # noqa: E402
    ARTIFACT_DIR_REL,
    CANDIDATE_DIR_REL,
    CLASS_MAP_SHA256,
    FLOAT_MODEL_SHA256,
    FLOAT_REFERENCE_SHA256,
    FREEZE_ID,
    INT8_MODEL_SHA256,
    LOCKED_TEST_COUNT,
    LOCKED_TEST_MEMBERSHIP_FINGERPRINT,
    PROTOCOL_ID,
    SCALER_FINGERPRINT,
    SLOPE_PROFILE,
    THRESHOLD,
    VALIDATION_COUNT,
    file_sha256,
    load_json,
    stable_sha256,
)


REQUIRED = (
    "robustness_protocol.json",
    "robustness_results.json",
    "host_latency_evidence.json",
    "pre_locked_test_candidate_freeze.json",
    "locked_test_evaluation.json",
    "exceptions_and_limitations.json",
    "run_environment.json",
    "final_candidate_lock.json",
    "checksum_registry.json",
    "checksums.sha256",
    "artifact_identity.json",
    "c_b5_run_summary.json",
)


def _check(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _validate_predecessor_identity(root: Path, errors: List[str]) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    for rel, expected, phase in (
        ("models/co2/candidates/c_b4/full_integer_int8.tflite", INT8_MODEL_SHA256, "C-B4"),
        ("models/co2/candidates/c_b4/float_reference.tflite", FLOAT_MODEL_SHA256, "C-B4"),
        ("models/co2/candidates/c_b4/float_reference_parameters.json", FLOAT_REFERENCE_SHA256, "C-B4"),
        ("models/co2/candidates/c_b4/class_map.json", CLASS_MAP_SHA256, "C-B4"),
    ):
        try:
            actual = file_sha256(root, rel)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            statuses[phase] = "FAIL"
            continue
        _check(actual == expected, f"{phase} artifact hash drift: {rel}", errors)
        statuses[phase] = "PASS" if actual == expected else "FAIL"
    try:
        scaler = load_json(root / "datasets/co2/manifests/c_b2_imbalance_calibration/preprocessing_scaler_evidence.json")
        _check(scaler.get("scaler_fingerprint") == SCALER_FINGERPRINT, "C-B2 scaler fingerprint drift", errors)
        statuses["C-B2"] = "PASS" if scaler.get("scaler_fingerprint") == SCALER_FINGERPRINT else "FAIL"
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
        statuses["C-B2"] = "FAIL"
    return statuses


def _validate_checksum_closure(root: Path, output: Path, errors: List[str]) -> None:
    lock = load_json(output / "final_candidate_lock.json")
    entries = lock.get("artifacts") or []
    _check(lock.get("final_lock_profile_id") == "CO2_B5_FINAL_OFFLINE_CANDIDATE_LOCK_001", "final lock profile drift", errors)
    _check(lock.get("closure_status") == "PASS", "final lock closure status is not PASS", errors)
    _check(lock.get("self_reference_policy", {}).get("final_lock_hashes_itself") is False, "final lock self-reference policy drift", errors)
    _check(lock.get("artifact_count") == len(entries), "final lock artifact count mismatch", errors)
    _check(lock.get("final_lock_sha256") == stable_sha256({k: v for k, v in lock.items() if k != "final_lock_sha256"}), "final lock checksum mismatch", errors)
    seen = set()
    for entry in entries:
        rel = str(entry.get("path"))
        seen.add(rel)
        _check(not rel.startswith("/") and "\\" not in rel, f"non-portable final-lock path: {rel}", errors)
        path = root / rel
        _check(path.is_file(), f"missing final-lock artifact: {rel}", errors)
        if path.is_file():
            _check(file_sha256(root, rel) == entry.get("sha256"), f"final-lock hash mismatch: {rel}", errors)
            _check(path.stat().st_size == entry.get("byte_size"), f"final-lock byte-size mismatch: {rel}", errors)
    _check(f"{ARTIFACT_DIR_REL}/final_candidate_lock.json" not in seen, "final lock hashes itself", errors)
    registry = load_json(output / "checksum_registry.json")
    reg_entries = registry.get("entries") or []
    _check(registry.get("self_referential") is False, "checksum registry self-reference flag drift", errors)
    _check(registry.get("entry_count") == len(reg_entries), "checksum registry entry count mismatch", errors)
    _check({str(e.get("path")) for e in reg_entries} == seen, "checksum registry coverage differs from final lock", errors)
    rows = set()
    for line in (output / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, rel = line.split("  ", 1)
        except ValueError:
            errors.append(f"malformed checksums.sha256 row: {line!r}")
            continue
        rows.add(rel)
        _check(file_sha256(root, rel) == digest, f"checksums.sha256 mismatch: {rel}", errors)
    _check(rows == seen, "checksums.sha256 coverage differs from final lock", errors)


def validate(root: Path, *, run_predecessors: bool = False) -> Dict[str, Any]:
    root = root.resolve()
    output = root / ARTIFACT_DIR_REL
    candidate = root / CANDIDATE_DIR_REL
    errors: List[str] = []
    warnings: List[str] = []
    for name in REQUIRED:
        _check((output / name).is_file(), f"missing C-B5 artifact: {name}", errors)
    _check((candidate / "final_candidate_metadata.json").is_file(), "missing C-B5 final candidate metadata", errors)
    if errors:
        return {"status": "FAIL", "errors": errors, "warnings": warnings}

    predecessor_status = _validate_predecessor_identity(root, errors)
    protocol = load_json(output / "robustness_protocol.json")
    robustness = load_json(output / "robustness_results.json")
    latency = load_json(output / "host_latency_evidence.json")
    freeze = load_json(output / "pre_locked_test_candidate_freeze.json")
    locked = load_json(output / "locked_test_evaluation.json")
    metadata = load_json(candidate / "final_candidate_metadata.json")

    _check(protocol.get("protocol_id") == PROTOCOL_ID, "robustness protocol identity drift", errors)
    _check(protocol.get("registration_status") == "PRE_REGISTERED_BEFORE_RESULTS", "protocol was not pre-registered", errors)
    _check(protocol.get("locked_test_used") is False and protocol.get("model_tuning_authorized") is False and protocol.get("threshold_tuning_authorized") is False, "robustness protocol permits forbidden selection", errors)
    _check(protocol.get("candidate", {}).get("model_sha256") == INT8_MODEL_SHA256, "protocol model hash drift", errors)
    _check(protocol.get("candidate", {}).get("threshold") == THRESHOLD, "protocol threshold drift", errors)
    _check(protocol.get("candidate", {}).get("feature_order") == ["CO2", "Temperature", "Humidity", "CO2_slope"], "protocol feature order drift", errors)
    _check(protocol.get("causal_procedure", {}).get("source_level_first") is True and protocol.get("causal_procedure", {}).get("silent_slope_imputation") is False, "causal source-level contract drift", errors)
    serialized_protocol = file_sha256(root, f"{ARTIFACT_DIR_REL}/robustness_protocol.json")

    _check(robustness.get("protocol_id") == PROTOCOL_ID, "robustness results protocol drift", errors)
    _check(robustness.get("locked_test_used") is False, "LOCKED_TEST used during robustness", errors)
    _check(robustness.get("locked_test_feature_access") == 0 and robustness.get("locked_test_target_access") == 0 and robustness.get("locked_test_predictions") == 0 and robustness.get("locked_test_probabilities") == 0 and robustness.get("locked_test_metrics") == 0, "LOCKED_TEST access appeared during robustness", errors)
    _check(robustness.get("validation_sample_count") == VALIDATION_COUNT, "validation count drift", errors)
    scenarios = robustness.get("results") or []
    _check(robustness.get("scenario_count") == len(scenarios) == 25, "robustness scenario count/grid drift", errors)
    _check(robustness.get("baseline_metrics", {}).get("sample_count") == VALIDATION_COUNT, "baseline validation metrics count drift", errors)
    _check(robustness.get("baseline_saturation", {}).get("saturated_element_count") == 3, "C-B4 validation saturation baseline drift", errors)
    for row in scenarios:
        _check(int(row.get("sample_count_available", -1)) <= int(row.get("sample_count_intended", -1)), f"availability exceeds intended for {row.get('scenario_id')}", errors)
        _check(float(row.get("availability_rate", -1.0)) <= 1.0, f"invalid availability for {row.get('scenario_id')}", errors)
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "recall_occupied"):
            value = row.get("metrics", {}).get(key)
            if row.get("sample_count_available", 0):
                _check(_finite(value), f"non-finite robustness metric {key} for {row.get('scenario_id')}", errors)
        _check(row.get("diagnostic_only") is (row.get("kind") != "baseline"), f"diagnostic flag drift for {row.get('scenario_id')}", errors)

    _check(latency.get("classification") == "HOST_MAC_LATENCY_SANITY_ONLY", "latency classification drift", errors)
    _check(latency.get("model_sha256") == INT8_MODEL_SHA256, "latency model hash drift", errors)
    _check(latency.get("warmup_iterations") == 100 and latency.get("timed_iterations") == 2000, "latency protocol count drift", errors)
    _check(latency.get("raspberry_pi_latency_claimed") is False and latency.get("production_realtime_claim") is False, "latency overclaim", errors)
    for key in ("mean", "median_p50", "p95", "p99", "minimum", "maximum"):
        _check(_finite(latency.get(key)) and float(latency[key]) >= 0.0, f"invalid latency statistic: {key}", errors)

    _check(freeze.get("freeze_profile_id") == FREEZE_ID and freeze.get("freeze_status") == "VALID_PRE_LOCKED_TEST", "pre-test freeze identity/status drift", errors)
    _check(freeze.get("freeze_sha256") == stable_sha256({k: v for k, v in freeze.items() if k != "freeze_sha256"}), "pre-test freeze checksum mismatch", errors)
    _check(freeze.get("candidate", {}).get("model_sha256") == INT8_MODEL_SHA256 and freeze.get("candidate", {}).get("threshold") == THRESHOLD and freeze.get("candidate", {}).get("scaler_fingerprint") == SCALER_FINGERPRINT, "pre-test candidate identity drift", errors)
    _check(freeze.get("evidence", {}).get("robustness_protocol_sha256") == serialized_protocol, "freeze protocol checksum mismatch", errors)
    _check(freeze.get("locked_test_prior_access") == {"feature_access": 0, "target_access": 0, "predictions": 0, "probabilities": 0, "metrics": 0}, "nonzero LOCKED_TEST access before freeze", errors)
    _check(all(bool(v) for v in freeze.get("decision_state", {}).values()), "not all candidate decisions frozen", errors)

    _check(locked.get("evaluation_profile_id") == "CO2_B5_LOCKED_TEST_EVALUATION_001", "LOCKED_TEST evaluation identity drift", errors)
    _check(locked.get("evaluation_count") == 1 and locked.get("evaluation_status") == "COMPLETED_ONE_TIME_UNPERTURBED", "LOCKED_TEST evaluation count/status drift", errors)
    _check(locked.get("eligible_sample_count") == LOCKED_TEST_COUNT, "LOCKED_TEST eligible count drift", errors)
    _check(locked.get("locked_test_membership_fingerprint") == LOCKED_TEST_MEMBERSHIP_FINGERPRINT, "LOCKED_TEST membership fingerprint drift", errors)
    _check(locked.get("candidate_sha256") == INT8_MODEL_SHA256 and locked.get("threshold") == THRESHOLD, "LOCKED_TEST candidate/threshold drift", errors)
    _check(locked.get("perturbation_sweeps") == 0, "LOCKED_TEST perturbation sweep detected", errors)
    _check(locked.get("post_test_tuning") == {"model_change": False, "scaler_change": False, "feature_change": False, "slope_change": False, "threshold_change": False}, "post-test tuning detected", errors)
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "recall_occupied", "roc_auc", "pr_auc_average_precision", "brier_score", "log_loss", "ece"):
        _check(_finite(locked.get("metrics", {}).get(key)), f"missing LOCKED_TEST metric: {key}", errors)

    _check(metadata.get("candidate_status") == "FINAL_OFFLINE_UCI_CANDIDATE_LOCKED", "final candidate status drift", errors)
    _check(metadata.get("model_sha256") == INT8_MODEL_SHA256 and metadata.get("threshold") == THRESHOLD and metadata.get("feature_order") == ["CO2", "Temperature", "Humidity", "CO2_slope"], "final candidate identity drift", errors)
    _check(metadata.get("device_domain_validation_status") == "NOT_YET_COMPLETE", "device-domain status overclaimed", errors)
    _check(metadata.get("post_test_tuning") == "NONE" and metadata.get("locked_test_evaluation_count") == 1, "final candidate post-test state drift", errors)

    _validate_checksum_closure(root, output, errors)
    if load_json(output / "exceptions_and_limitations.json").get("blockers"):
        errors.append("C-B5 limitations registry contains blockers")
    warnings.extend(["INT8_INPUT_SATURATION_OBSERVED", "HOST_MAC_LATENCY_SANITY_ONLY", "SCD40_DEVICE_DOMAIN_VALIDATION_NOT_YET_COMPLETE"])

    if run_predecessors:
        for phase, script in (("C-A0", "scripts/validate_co2_raw_inventory.py"), ("C-A1", "scripts/validate_co2_safe_reader.py"), ("C-A2", "scripts/validate_co2_temporal_blocks.py"), ("C-A3", "scripts/validate_co2_slope_feature.py"), ("C-A4", "scripts/validate_co2_target_semantics.py"), ("C-A5", "scripts/validate_co2_canonical_samples.py"), ("C-A6", "scripts/validate_co2_final_integrity.py"), ("C-B0", "scripts/validate_co2_offline_experiment.py"), ("C-B1", "scripts/validate_co2_slope_ablation.py"), ("C-B2", "scripts/validate_co2_imbalance_calibration.py"), ("C-B3", "scripts/validate_co2_architecture_multiseed.py"), ("C-B4", "scripts/validate_co2_tflite_equivalence.py")):
            path = root / script
            if not path.is_file():
                errors.append(f"missing predecessor validator: {script}")
                continue
            args = [sys.executable, str(path)]
            if phase == "C-B3":
                args.append("--skip-predecessors")
            elif phase in ("C-B0", "C-B1", "C-B2"):
                args.append("--skip-determinism")
            completed = subprocess.run(args, cwd=str(root), capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                errors.append(f"{phase} predecessor validator failed: exit={completed.returncode}")
            predecessor_status[phase] = "PASS" if completed.returncode == 0 else "FAIL"

    status = "FAIL" if errors else "PASS_WITH_WARNINGS"
    return {"status": status, "errors": errors, "warnings": warnings, "predecessor_status": predecessor_status, "artifact_count": load_json(output / "final_candidate_lock.json").get("artifact_count")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-predecessors", action="store_true")
    args = parser.parse_args()
    result = validate(args.root.resolve(), run_predecessors=args.run_predecessors)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
