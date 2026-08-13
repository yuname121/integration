#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/validate_co2_offline_experiment.py
Phase C-B0 — standalone offline experiment contract validator.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

repo_root_dir = Path(__file__).resolve().parent.parent
if str(repo_root_dir) not in sys.path:
    sys.path.insert(0, str(repo_root_dir))

from datasets.co2.canonical_samples import CANONICAL_SAMPLE_PROFILE_ID
from datasets.co2.offline_experiment import (
    A_SERIES_LOCK_PROFILE,
    A_SERIES_RELEASE_COMMIT,
    A_SERIES_RELEASE_TAG,
    EXPECTED_LOCKED_TEST_SEALED,
    EXPECTED_TRAIN_COMMON,
    EXPECTED_VALIDATION_COMMON,
    EXPECTED_WARMUP_CANONICAL,
    EXPERIMENT_CONTRACT_ID,
    MANIFEST_DIR_REL,
    assert_no_forbidden_path_markers,
    build_sample_universe_manifest,
    verify_a_series_artifact_lock,
    verify_a_series_release,
)
from datasets.co2.raw_reader import compute_sha256_file, get_repo_root
from datasets.co2.slope_feature import FEATURE_PROFILE_ID as SLOPE_PROFILE_ID
from datasets.co2.target_semantics import TARGET_PROFILE_ID as TARGET_PROFILE_ID

FORBIDDEN_PATH_MARKERS = ("/Users/", "file://", "~/", "/private/tmp/", "CloudDocs")
PROTECTED_SHARED = {
    "AGENTS.md",
    "README.md",
    "docs/README.md",
    "datasets/MANIFEST.json",
    "models/model_manifest.json",
    "docs/reports/model_inventory.json",
    "docs/reports/SENSOR_DATA_CONTRACT.md",
    "docs/reports/sensor_model_data_contract.json",
    "models/co2/co2_scaling_metadata_v0.1.0.json",
    "models/co2/co2_occupancy_int8_v0.1.0.tflite",
    "datasets/co2/processed/co2_occupancy_v1.npz",
    "sensors/co2/co2_adapter.py",
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_validator(script: str, repo_root: Path) -> bool:
    res = subprocess.run(
        ["python3", script],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    return res.returncode == 0


def derive_gate(errors: int, warnings: int) -> str:
    if errors > 0:
        return "FAIL"
    if warnings > 0:
        return "PASS_WITH_WARNINGS"
    return "PASS"


def validate_c_b0(repo_root: Path) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    errors: List[str] = []
    warnings: List[str] = []
    c_b0 = repo_root / MANIFEST_DIR_REL
    required = [
        "experiment_contract.json",
        "a_series_consumption_registry.json",
        "sample_universe_manifest.json",
        "feature_view_registry.json",
        "metric_contract.json",
        "leakage_audit.json",
        "preprocessing_fit_evidence.json",
        "reference_baseline_result.json",
        "generation_metadata.json",
        "exceptions_and_limitations.json",
        "run_environment.json",
        "checksums.sha256",
    ]
    for fname in required:
        if not (c_b0 / fname).exists():
            errors.append(f"Missing C-B0 artifact: {fname}")
    if errors:
        return False, errors, warnings, {}

    preds = {
        "c_a0": _run_validator("scripts/validate_co2_raw_inventory.py", repo_root),
        "c_a1": _run_validator("scripts/validate_co2_safe_reader.py", repo_root),
        "c_a2": _run_validator("scripts/validate_co2_temporal_blocks.py", repo_root),
        "c_a3": _run_validator("scripts/validate_co2_slope_feature.py", repo_root),
        "c_a4": _run_validator("scripts/validate_co2_target_semantics.py", repo_root),
        "c_a5": _run_validator("scripts/validate_co2_canonical_samples.py", repo_root),
        "c_a6": _run_validator("scripts/validate_co2_final_integrity.py", repo_root),
    }
    if not all(preds.values()):
        for k, ok in preds.items():
            if not ok:
                errors.append(f"Predecessor validator failed: {k}")

    release = verify_a_series_release(repo_root)
    if not release["matches_expected"]:
        errors.append("A_SERIES_RELEASE_PREREQUISITE_NOT_MET")
    try:
        lock = verify_a_series_artifact_lock(repo_root)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"A_SERIES_BASELINE_DRIFT: {exc}")
        lock = {}

    contract = _load_json(c_b0 / "experiment_contract.json")
    consumption = _load_json(c_b0 / "a_series_consumption_registry.json")
    universe = _load_json(c_b0 / "sample_universe_manifest.json")
    features = _load_json(c_b0 / "feature_view_registry.json")
    metrics = _load_json(c_b0 / "metric_contract.json")
    leakage = _load_json(c_b0 / "leakage_audit.json")
    preprocess = _load_json(c_b0 / "preprocessing_fit_evidence.json")
    baseline = _load_json(c_b0 / "reference_baseline_result.json")
    generation = _load_json(c_b0 / "generation_metadata.json")
    exceptions = _load_json(c_b0 / "exceptions_and_limitations.json")

    live_universe = build_sample_universe_manifest(repo_root)

    if contract.get("experiment_contract_id") != EXPERIMENT_CONTRACT_ID:
        errors.append("Unexpected experiment contract id")
    if contract.get("a_series_release_tag") != A_SERIES_RELEASE_TAG:
        errors.append("Experiment contract release tag mismatch")
    if contract.get("a_series_release_commit") != A_SERIES_RELEASE_COMMIT:
        errors.append("Experiment contract release commit mismatch")
    if contract.get("canonical_sample_profile_id") != CANONICAL_SAMPLE_PROFILE_ID:
        errors.append("Canonical sample profile mismatch")
    if contract.get("target_profile_id") != TARGET_PROFILE_ID:
        errors.append("Target profile mismatch")
    if contract.get("baseline_slope_profile_id") != SLOPE_PROFILE_ID:
        errors.append("Slope profile mismatch")
    if contract.get("a_series_artifact_lock_profile") != A_SERIES_LOCK_PROFILE:
        errors.append("Lock profile mismatch in contract")

    if consumption.get("a_series_release_tag") != A_SERIES_RELEASE_TAG:
        errors.append("Consumption registry tag mismatch")
    if consumption.get("baseline_drift_status") != "NONE":
        errors.append("A-series baseline drift reported")
    if lock and consumption.get("artifact_lock", {}).get("sha256") != lock.get("sha256"):
        errors.append("Consumption registry lock sha mismatch")

    if universe.get("b_series_common_train") != EXPECTED_TRAIN_COMMON:
        errors.append("TRAIN common universe count mismatch")
    if universe.get("b_series_common_validation") != EXPECTED_VALIDATION_COMMON:
        errors.append("VALIDATION common universe count mismatch")
    if universe.get("b_series_sealed_locked_test") != EXPECTED_LOCKED_TEST_SEALED:
        errors.append("LOCKED_TEST sealed count mismatch")
    if universe.get("canonical_warmup_records") != EXPECTED_WARMUP_CANONICAL:
        errors.append("Warm-up count mismatch")
    if universe.get("ordered_id_list_sha256") != live_universe.get("ordered_id_list_sha256"):
        errors.append("Sample universe fingerprint drift vs live recomputation")
    if live_universe["overlaps"]["train_validation"] != 0:
        errors.append("TRAIN/VALIDATION overlap")
    if live_universe["overlaps"]["train_locked_test"] != 0:
        errors.append("TRAIN/LOCKED_TEST overlap")
    if live_universe["overlaps"]["validation_locked_test"] != 0:
        errors.append("VALIDATION/LOCKED_TEST overlap")

    if features.get("final_feature_selection_performed") is not False:
        errors.append("Final feature selection must not be performed")
    if features.get("canonical_final_feature_set_claimed") is not False:
        errors.append("Canonical final feature set must not be claimed")
    if "Occupancy" not in features.get("feature_roles", {}):
        errors.append("Occupancy role missing")
    elif features["feature_roles"]["Occupancy"].get("may_be_model_input") is not False:
        errors.append("Occupancy must not be model input")
    for key in ("CO2", "Temperature", "Humidity", "CO2_slope", "Light", "HumidityRatio"):
        if key not in features.get("feature_roles", {}):
            errors.append(f"Missing feature role: {key}")
    if "HISTORICAL_COMPATIBILITY_REFERENCE" not in features.get("feature_views", {}):
        errors.append("Historical compatibility view missing")

    required_metrics = set(metrics.get("required_metrics", []))
    for m in (
        "accuracy",
        "balanced_accuracy",
        "precision_occupied",
        "recall_occupied",
        "f1_occupied",
        "macro_f1",
        "confusion_matrix",
    ):
        if m not in required_metrics:
            errors.append(f"Metric contract missing {m}")
    if metrics.get("positive_class") != "OCCUPIED":
        errors.append("Positive class must be OCCUPIED")
    if metrics.get("threshold_optimization_in_b0") is not False:
        errors.append("Threshold optimization must not occur in B0")

    if leakage.get("status") != "PASS":
        errors.append(f"Leakage audit failed: {leakage.get('errors')}")
    if leakage.get("locked_test_predictive_evaluation_in_b0") is not False:
        errors.append("LOCKED_TEST predictive evaluation flagged")

    if preprocess.get("fit_population") != "TRAIN":
        errors.append("Preprocessing fit population must be TRAIN")
    if preprocess.get("production_scaler_modified") is not False:
        errors.append("Production scaler modified")
    if preprocess.get("status") != "B0_EXPERIMENT_REFERENCE_ONLY":
        errors.append("B0 scaler status must be experiment-reference-only")
    if preprocess.get("train_sample_count") != EXPECTED_TRAIN_COMMON:
        errors.append("Scaler TRAIN sample count mismatch")

    if baseline.get("status") != "REFERENCE_BASELINE_ONLY":
        errors.append("Baseline status must be REFERENCE_BASELINE_ONLY")
    if baseline.get("candidate") is not False or baseline.get("deployable") is not False:
        errors.append("Baseline must not be candidate/deployable")
    if baseline.get("train_population_count") != EXPECTED_TRAIN_COMMON:
        errors.append("Baseline TRAIN population mismatch")
    if baseline.get("evaluation_population") != "VALIDATION":
        errors.append("Baseline must evaluate VALIDATION only")
    if baseline.get("locked_test_used") is not False:
        errors.append("Baseline used LOCKED_TEST")
    if baseline.get("threshold_optimization_performed") is not False:
        errors.append("Baseline threshold optimization not allowed")
    if baseline.get("complex_model_comparison_performed") is not False:
        errors.append("Complex model comparison not allowed")
    if "macro_f1" not in (baseline.get("metrics") or {}):
        errors.append("Baseline metrics incomplete")

    if generation.get("production_scaler_modified") is not False:
        errors.append("Generation claims production scaler modified")
    if generation.get("production_model_modified") is not False:
        errors.append("Generation claims production model modified")
    if generation.get("synthetic_npz_used_as_real_training_data") is not False:
        errors.append("Synthetic NPZ used as real training data")
    if generation.get("final_feature_selection_performed") is not False:
        errors.append("Final feature selection claimed in generation metadata")
    if generation.get("slope_ablation_performed") is not False:
        errors.append("Slope ablation claimed")
    if generation.get("locked_test_fit_usage") != 0 or generation.get("locked_test_tuning_usage") != 0:
        errors.append("LOCKED_TEST fit/tuning usage non-zero")

    # checksums
    for line in (c_b0 / "checksums.sha256").read_text(encoding="utf-8").strip().splitlines():
        digest, rel = line.split("  ", 1)
        if rel.endswith("checksums.sha256"):
            errors.append("checksums.sha256 must not hash itself")
        path = repo_root / rel
        if not path.exists():
            errors.append(f"Checksum path missing: {rel}")
        elif compute_sha256_file(path) != digest:
            errors.append(f"Checksum mismatch: {rel}")

    for fname in required:
        text = (c_b0 / fname).read_text(encoding="utf-8")
        for marker in FORBIDDEN_PATH_MARKERS:
            if marker in text:
                errors.append(f"Forbidden path marker {marker} in {fname}")
        errors.extend([f"{fname}: {e}" for e in assert_no_forbidden_path_markers(text)])

    # Git isolation
    try:
        diff = subprocess.run(
            ["git", "diff", "--name-only", "origin/main...HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        for path in [p.strip() for p in diff.stdout.splitlines() if p.strip()]:
            if path in PROTECTED_SHARED:
                errors.append(f"Unauthorized shared/production file modified: {path}")
            lower = path.lower()
            if path.startswith("datasets/mmwave/") or (
                "mmwave" in lower and path.startswith(("scripts/", "tests/", "docs/reports/", "models/"))
            ):
                errors.append(f"mmWave file in C-B0 branch: {path}")
            if path.startswith("datasets/thermal") or (
                "thermal" in lower and path.startswith(("scripts/", "tests/", "docs/reports/", "datasets/"))
            ):
                errors.append(f"Thermal file in C-B0 branch: {path}")
            if "occupancy+detection.zip" in path or (
                "raw_archives" in path and path.endswith((".zip", ".txt"))
            ):
                errors.append("Raw payload staged/modified")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Git isolation check skipped: {exc}")

    for item in exceptions.get("warnings", []):
        warnings.append(f"[{item.get('code')}] {item.get('description')}")
    if exceptions.get("blockers"):
        errors.append("Exception registry contains blockers")

    # dedupe
    deduped: List[str] = []
    seen = set()
    for e in errors:
        if e not in seen:
            seen.add(e)
            deduped.append(e)
    errors = deduped

    gate = derive_gate(len(errors), len(warnings))
    summary = {
        "gate_status": gate,
        "c_b0_merge_ready": "YES" if len(errors) == 0 else "NO",
        "c_b1_authorized_after_merge": "YES" if len(errors) == 0 else "NO",
        "train": universe.get("b_series_common_train"),
        "validation": universe.get("b_series_common_validation"),
        "locked_test": universe.get("b_series_sealed_locked_test"),
        "warmup": universe.get("canonical_warmup_records"),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "predecessors": preds,
        "release": release,
        "lock_sha256": lock.get("sha256") if lock else None,
    }
    return len(errors) == 0, errors, warnings, summary


def main() -> int:
    repo_root = get_repo_root()
    print(f"🔍 Validating Phase C-B0 CO₂ Offline Experiment in: {MANIFEST_DIR_REL}")
    ok, errors, warnings, summary = validate_c_b0(repo_root)
    print("\n--- C-B0 VALIDATOR RESULT ---")
    print(f"Gate Status:      {summary.get('gate_status', 'FAIL')}")
    print(f"Merge Ready:      {summary.get('c_b0_merge_ready', 'NO')}")
    print(f"C-B1 After Merge: {summary.get('c_b1_authorized_after_merge', 'NO')}")
    print(f"TRAIN:            {summary.get('train')}")
    print(f"VALIDATION:       {summary.get('validation')}")
    print(f"LOCKED_TEST:      {summary.get('locked_test')}")
    print(f"Warm-up:          {summary.get('warmup')}")
    print(f"Error Count:      {summary.get('error_count', len(errors))}")
    print(f"Warning Count:    {summary.get('warning_count', len(warnings))}")
    if errors:
        print("\nRecorded Errors:")
        for err in errors:
            print(f" ❌  {err}")
    if warnings:
        print("\nRecorded Warnings & Limitations:")
        for warn in warnings:
            print(f" ⚠️  {warn}")
    if ok:
        print("\n✅ SUCCESS: Phase C-B0 offline experiment contract is valid.")
        return 0
    print("\n❌ FAILURE: Phase C-B0 validation failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
