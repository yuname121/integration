#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone validator for SafeNest CO₂ Phase C-B3 evidence."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

repo_root_dir = Path(__file__).resolve().parent.parent
if str(repo_root_dir) not in sys.path:
    sys.path.insert(0, str(repo_root_dir))

from datasets.co2.architecture_multiseed import (
    ARCHITECTURE_IDS,
    ARTIFACT_DIR_REL,
    B2_POLICY_ID,
    B2_SCALER_PROFILE_ID,
    B2_THRESHOLD_PROTOCOL_ID,
    CB2_MERGED_MAIN_COMMIT,
    CB3Error,
    EXPECTED_RUN_COUNT,
    FIXED_FEATURES,
    LOCKED_TEST_COUNT,
    METRIC_NAMES,
    PredecessorFingerprintMismatch,
    SEEDS,
    TRAIN_COUNT,
    VALIDATION_COUNT,
    aggregate_architectures,
    architecture_registry,
    build_predecessor_fingerprint_registry,
    load_json,
    prepare_fixed_data,
    rank_architectures,
    stable_sha256,
    validate_architecture_registry,
    validate_feature_context,
    validate_predecessor_inputs,
    validate_seed_registry,
    verify_stored_predecessor_registry,
)
from datasets.co2.imbalance_calibration import (
    _probability_fingerprint,
    build_threshold_sweep,
    classification_metrics_at_threshold,
    expected_calibration_error,
    probability_quality_metrics,
)
from datasets.co2.raw_reader import compute_sha256_file, get_repo_root


REQUIRED_ARTIFACTS = (
    "predecessor_fingerprint_registry.json",
    "experiment_contract.json",
    "architecture_candidate_registry.json",
    "seed_registry.json",
    "fixed_comparison_universe_fingerprint.json",
    "fixed_feature_context_fingerprint.json",
    "preprocessing_parity_evidence.json",
    "oversampling_parity_evidence.json",
    "per_run_results.json",
    "validation_predictions.json",
    "threshold_sweep_results.json",
    "architecture_multiseed_aggregate.json",
    "default_vs_calibrated_comparison.json",
    "threshold_stability_summary.json",
    "architecture_complexity_summary.json",
    "architecture_ranking.json",
    "selected_architecture_profile.json",
    "leakage_audit.json",
    "determinism_report.json",
    "exceptions_and_limitations.json",
    "generation_metadata.json",
    "artifact_identity.json",
    "checksums.sha256",
)

ALLOWED_EXACT_PATHS = {
    "datasets/co2/architecture_multiseed.py",
    "scripts/audit_co2_architecture_multiseed.py",
    "scripts/validate_co2_architecture_multiseed.py",
    "tests/test_co2_architecture_multiseed.py",
}

# Later CO₂ phases are allowed to add their own namespace without weakening
# the cross-track checks below.  C-B3's original validator predated C-B5 and
# otherwise treated legitimate CO₂ robustness evidence as branch contamination.
C_B5_FORWARD_COMPATIBLE_PREFIXES = (
    "datasets/co2/b5_robustness.py",
    "datasets/co2/manifests/c_b5_robustness_final_lock/",
    "models/co2/candidates/c_b5/",
    "scripts/run_co2_b5.py",
    "scripts/validate_co2_b5.py",
    "tests/test_co2_b5.py",
)


def _allowed_path(path: str) -> bool:
    return (
        path in ALLOWED_EXACT_PATHS
        or path.startswith(f"{ARTIFACT_DIR_REL}/")
        or any(path == prefix or path.startswith(prefix) for prefix in C_B5_FORWARD_COMPATIBLE_PREFIXES)
    )


def _run_script(root: Path, script: str, *args: str) -> Tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, script, *args], cwd=str(root), capture_output=True, text=True, check=False
    )
    return result.returncode == 0, (result.stdout + "\n" + result.stderr).strip()


def _status_paths(root: Path) -> List[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=str(root), capture_output=True, text=True, check=False,
    )
    paths: List[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        payload = line[3:] if len(line) >= 3 else line
        if " -> " in payload:
            payload = payload.split(" -> ", 1)[1]
        paths.append(payload.strip())
    return paths


def _close(a: Any, b: Any, atol: float = 1e-13) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def _compare_metric_block(stored: Mapping[str, Any], live: Mapping[str, Any], context: str, errors: List[str]) -> None:
    for key in ("tn", "fp", "fn", "tp"):
        if int(stored.get(key, -1)) != int(live.get(key, -2)):
            errors.append(f"{context}: metric mismatch {key}")
    for key in (
        "accuracy", "balanced_accuracy", "precision_occupied", "recall_occupied", "f1_occupied",
        "precision_vacant", "recall_vacant", "f1_vacant", "macro_f1", "false_positive_rate",
        "false_negative_rate", "specificity", "sensitivity", "occupied_recall", "decision_threshold",
    ):
        if key in live and not _close(stored.get(key), live.get(key)):
            errors.append(f"{context}: metric mismatch {key}")
    if stored.get("confusion_matrix") != live.get("confusion_matrix"):
        errors.append(f"{context}: confusion matrix mismatch")


def _compare_summary(stored: Mapping[str, Any], live: Mapping[str, Any], context: str, errors: List[str]) -> None:
    for key in ("mean", "std", "min", "max", "worst_seed_value", "best_seed_value"):
        if not _close(stored.get(key), live.get(key)):
            errors.append(f"{context}: summary mismatch {key}")
    for key in ("min_seed", "max_seed", "worst_seed", "best_seed"):
        if stored.get(key) != live.get(key):
            errors.append(f"{context}: summary identity mismatch {key}")


def _validate_checksums(root: Path, output_dir: Path, errors: List[str]) -> None:
    checksum_rows = (output_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    expected = {name for name in REQUIRED_ARTIFACTS if name.endswith(".json")}
    seen: set[str] = set()
    for line in checksum_rows:
        if not line.strip():
            continue
        try:
            digest, rel = line.split("  ", 1)
        except ValueError:
            errors.append(f"Malformed checksum row: {line!r}")
            continue
        if rel.startswith("/") or "\\" in rel:
            errors.append(f"Non-portable checksum path: {rel}")
        path = root / rel
        seen.add(path.name)
        if not path.is_file():
            errors.append(f"Missing checksum path: {rel}")
        elif compute_sha256_file(path) != digest:
            errors.append(f"Checksum mismatch: {rel}")
    if seen != expected:
        errors.append("Checksum closure does not cover every C-B3 JSON artifact exactly")


def validate(root: Path, *, run_predecessors: bool = True) -> Dict[str, Any]:
    root = root.resolve()
    output_dir = root / ARTIFACT_DIR_REL
    errors: List[str] = []
    warnings: List[str] = []
    for filename in REQUIRED_ARTIFACTS:
        if not (output_dir / filename).is_file():
            errors.append(f"Missing C-B3 artifact: {filename}")
    if errors:
        return {"status": "FAIL", "errors": errors, "warnings": warnings, "artifact_dir": ARTIFACT_DIR_REL}

    artifacts = {name: load_json(output_dir / name) for name in REQUIRED_ARTIFACTS if name.endswith(".json")}
    predecessor_status = {"C-B0": "NOT_RUN", "C-B1": "NOT_RUN", "C-B2": "DIRECT_CONTRACT_CHECK"}
    if run_predecessors:
        b0_ok, b0_text = _run_script(root, "scripts/validate_co2_offline_experiment.py")
        predecessor_status["C-B0"] = "PASS" if b0_ok else "FAIL"
        if not b0_ok:
            errors.append("C-B0 predecessor validator failed: " + " | ".join(b0_text.splitlines()[-6:]))
        b1_ok, b1_text = _run_script(root, "scripts/validate_co2_slope_ablation.py", "--skip-determinism")
        predecessor_status["C-B1"] = "PASS" if b1_ok else "FAIL"
        if not b1_ok:
            errors.append("C-B1 predecessor validator failed: " + " | ".join(b1_text.splitlines()[-6:]))

    try:
        predecessor = validate_predecessor_inputs(root)
        verify_stored_predecessor_registry(root, artifacts["predecessor_fingerprint_registry.json"])
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
        predecessor = None

    if artifacts["predecessor_fingerprint_registry.json"].get("required_c_b2_merged_main_commit") != CB2_MERGED_MAIN_COMMIT:
        errors.append("C-B2 merged-main commit identity drift")
    try:
        validate_architecture_registry(artifacts["architecture_candidate_registry.json"])
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    try:
        validate_seed_registry(artifacts["seed_registry.json"])
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))

    universe = artifacts["fixed_comparison_universe_fingerprint.json"]
    if universe.get("train_count") != TRAIN_COUNT or universe.get("validation_count") != VALIDATION_COUNT or universe.get("locked_test_count") != LOCKED_TEST_COUNT:
        errors.append("Comparison-universe count mismatch")
    if universe.get("locked_test_status") != "SEALED":
        errors.append("LOCKED_TEST is not sealed")
    if universe.get("overlap_counts", {}).get("train_validation") != 0 or universe.get("overlap_counts", {}).get("train_locked_test") != 0 or universe.get("overlap_counts", {}).get("validation_locked_test") != 0:
        errors.append("Cross-split overlap")
    feature_context = artifacts["fixed_feature_context_fingerprint.json"]
    try:
        validate_feature_context(feature_context.get("feature_order", []))
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    if feature_context.get("feature_count") != 4 or feature_context.get("slope_profile_id") != "CO2_B1_SELECTED_SLOPE_PROFILE_001":
        errors.append("Fixed feature/slope context drift")
    if feature_context.get("target_fields_as_features") or feature_context.get("provenance_fields_as_features"):
        errors.append("Target/provenance feature leakage")

    data = None
    if not errors:
        try:
            data = prepare_fixed_data(root)
            if data.original_train_fingerprint != universe.get("train_ordered_id_sha256") or data.validation_fingerprint != universe.get("validation_ordered_id_sha256"):
                errors.append("Fixed comparison-universe fingerprint drift")
            if data.scaler_evidence.get("scaler_fingerprint") != artifacts["preprocessing_parity_evidence.json"].get("scaler_fingerprint"):
                errors.append("Scaler parity fingerprint mismatch")
            if data.oversample_plan.evidence.get("resampled_ordered_sample_ids_sha256") != artifacts["oversampling_parity_evidence.json"].get("resampled_ordered_sample_ids_sha256"):
                errors.append("Oversampling parity fingerprint mismatch")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Fixed data reconstruction failed: {exc}")

    runs_doc = artifacts["per_run_results.json"]
    runs = runs_doc.get("runs") or []
    if runs_doc.get("run_count") != EXPECTED_RUN_COUNT or len(runs) != EXPECTED_RUN_COUNT:
        errors.append("Expected exactly 20 architecture runs")
    run_by_id = {run.get("run_id"): run for run in runs}
    if len(run_by_id) != EXPECTED_RUN_COUNT:
        errors.append("Duplicate/missing run identities")
    for architecture_id in ARCHITECTURE_IDS:
        arch_seeds = sorted(int(run.get("seed")) for run in runs if run.get("architecture_id") == architecture_id)
        if arch_seeds != list(SEEDS):
            errors.append(f"{architecture_id}: seed coverage mismatch")
    predictions = artifacts["validation_predictions.json"]
    if predictions.get("sample_count") != VALIDATION_COUNT or predictions.get("locked_test_predictions") != 0 or predictions.get("locked_test_probabilities") != 0 or predictions.get("locked_test_metrics") != 0:
        errors.append("Validation prediction/LOCKED_TEST policy mismatch")
    if data is not None:
        if predictions.get("sample_ids") != data.validation.sample_ids or predictions.get("labels") != [int(x) for x in data.validation.labels.tolist()]:
            errors.append("Validation identity/label mutation")
    prediction_runs = predictions.get("runs") or {}
    sweeps = artifacts["threshold_sweep_results.json"].get("runs") or {}
    for run in runs:
        run_id = run.get("run_id")
        pred = prediction_runs.get(run_id)
        sweep = sweeps.get(run_id)
        if pred is None or sweep is None:
            errors.append(f"Missing prediction or threshold sweep for {run_id}")
            continue
        probabilities = np.asarray(pred.get("probabilities", []), dtype=np.float64)
        if probabilities.shape != (VALIDATION_COUNT,) or not np.isfinite(probabilities).all():
            errors.append(f"Invalid probability vector for {run_id}")
            continue
        if pred.get("probability_vector_sha256") != _probability_fingerprint(predictions["sample_ids"], probabilities):
            errors.append(f"Probability fingerprint mismatch for {run_id}")
        if pred.get("probability_vector_sha256") != run.get("validation_probability_vector_sha256"):
            errors.append(f"Probability/run registry mismatch for {run_id}")
        if data is not None:
            live_default, _ = classification_metrics_at_threshold(data.validation.labels, probabilities, 0.5)
            live_rows, live_ranking = build_threshold_sweep(y_validation=data.validation.labels, probabilities=probabilities, sample_ids=data.validation.sample_ids)
            selected_threshold = float(live_ranking[0]["threshold"])
            live_calibrated, _ = classification_metrics_at_threshold(data.validation.labels, probabilities, selected_threshold)
            live_quality = probability_quality_metrics(data.validation.labels, probabilities)
            live_ece = expected_calibration_error(data.validation.labels, probabilities)
            live_quality["expected_calibration_error"] = live_ece["expected_calibration_error"]
            _compare_metric_block(run.get("default_validation_metrics", {}), live_default, f"{run_id} default", errors)
            _compare_metric_block(run.get("calibrated_validation_metrics", {}), live_calibrated, f"{run_id} calibrated", errors)
            if run.get("selected_validation_threshold") != selected_threshold:
                errors.append(f"{run_id}: threshold selected outside inherited grid ranking")
            if run.get("threshold_numeric_inherited_from_b2") is not False:
                errors.append(f"{run_id}: B2 numeric 0.58 was forced")
            for metric, value in live_quality.items():
                if not _close(run.get("probability_quality_metrics", {}).get(metric), value):
                    errors.append(f"{run_id}: probability metric mismatch {metric}")
            if run.get("ece_diagnostic") != live_ece:
                errors.append(f"{run_id}: ECE diagnostic mismatch")
            stored_rows = sweep.get("rows") or []
            if [float(row.get("threshold")) for row in stored_rows] != [float(row["threshold"]) for row in live_rows]:
                errors.append(f"{run_id}: threshold grid mutation")
            if sweep.get("ranking") != live_ranking:
                errors.append(f"{run_id}: threshold ranking mismatch")
            if sweep.get("locked_test_threshold_evaluations") != 0 or sweep.get("population") != "VALIDATION":
                errors.append(f"{run_id}: threshold population policy mismatch")
        for key in ("locked_test_feature_access", "locked_test_target_access", "locked_test_predictions", "locked_test_probabilities", "locked_test_metrics"):
            if run.get(key) != 0:
                errors.append(f"{run_id}: nonzero {key}")
        for key in ("feature_selection_performed", "architecture_specific_sample_dropping", "architecture_specific_scaler", "architecture_specific_imbalance", "hyperparameter_search_performed", "early_stopping_used"):
            if run.get(key) is not False:
                errors.append(f"{run_id}: boundary flag {key} is not false")

    aggregate_doc = artifacts["architecture_multiseed_aggregate.json"]
    aggregates = aggregate_doc.get("architectures") or {}
    try:
        live_aggregates = aggregate_architectures(runs)
        if set(aggregates) != set(live_aggregates):
            errors.append("Architecture aggregate coverage mismatch")
        for architecture_id in ARCHITECTURE_IDS:
            stored_arch = aggregates.get(architecture_id, {})
            live_arch = live_aggregates[architecture_id]
            if stored_arch.get("seed_count") != 5:
                errors.append(f"{architecture_id}: aggregate seed count mismatch")
            for mode in ("calibrated_validation_metrics", "default_validation_metrics"):
                for metric in METRIC_NAMES:
                    _compare_summary(stored_arch.get(mode, {}).get(metric, {}), live_arch[mode][metric], f"{architecture_id} {mode} {metric}", errors)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Aggregate recomputation failed: {exc}")
        live_aggregates = {}

    ranking_doc = artifacts["architecture_ranking.json"]
    try:
        live_ranking = rank_architectures(live_aggregates)
        if ranking_doc.get("ranking") != live_ranking:
            errors.append("Architecture ranking differs from independent full-precision recomputation")
        selected = artifacts["selected_architecture_profile.json"]
        if selected.get("winning_architecture_id") != live_ranking[0]["architecture_id"]:
            errors.append("Selected architecture drift")
        if selected.get("deployment_status") != ["OFFLINE_VALIDATION_SELECTED", "MULTI_SEED_STABILITY_EVALUATED", "LOCKED_TEST_UNTOUCHED", "DEVICE_DOMAIN_UNVALIDATED", "PRODUCTION_ARTIFACT_NOT_CREATED"]:
            errors.append("Selected architecture status boundary drift")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Architecture ranking validation failed: {exc}")

    leakage = artifacts["leakage_audit.json"]
    zero_keys = (
        "train_validation_overlap", "train_locked_test_overlap", "validation_locked_test_overlap", "target_as_feature", "provenance_as_feature",
        "validation_in_scaler_fit", "locked_test_in_scaler_fit", "validation_in_oversampling", "locked_test_in_oversampling",
        "validation_used_for_model_fitting", "locked_test_used_for_model_fitting", "locked_test_feature_access", "locked_test_target_access",
        "locked_test_predictions", "locked_test_probability_outputs", "locked_test_metrics", "locked_test_threshold_calibration",
        "locked_test_model_selection", "architecture_specific_sample_dropping", "architecture_specific_feature_changes", "architecture_specific_scaler",
        "architecture_specific_imbalance_strategy", "class_weight_stacked_on_b2_oversampling", "architecture_hyperparameter_search", "early_stopping_tuning",
    )
    for key in zero_keys:
        if leakage.get(key) != 0:
            errors.append(f"Leakage audit nonzero: {key}")
    for key in ("production_model_modified", "production_scaler_modified", "a_series_locked_artifacts_modified", "b0_predecessor_artifacts_modified", "b1_predecessor_artifacts_modified", "b2_predecessor_artifacts_modified", "synthetic_fixture_used_as_real_training_data"):
        if leakage.get(key) is not False:
            errors.append(f"Leakage boundary flag {key} is not false")
    if leakage.get("threshold_selected_on") != "VALIDATION_ONLY" or leakage.get("locked_test_membership_count_verified") != LOCKED_TEST_COUNT:
        errors.append("Leakage threshold/LOCKED_TEST membership policy mismatch")

    generation = artifacts["generation_metadata.json"]
    for key in ("feature_selection_performed", "architecture_hyperparameter_search_performed", "early_stopping_tuning_performed", "imbalance_strategy_reselection_performed", "probability_recalibration_model_fitted", "production_model_modified", "production_scaler_modified", "a_series_locked_artifacts_modified", "b0_predecessor_artifacts_modified", "b1_predecessor_artifacts_modified", "b2_predecessor_artifacts_modified", "synthetic_npz_used_as_real_training_data"):
        if generation.get(key) is not False:
            errors.append(f"Generation boundary violation: {key}")
    if generation.get("completed_architecture_runs") != EXPECTED_RUN_COUNT or generation.get("locked_test_predictions") != 0 or generation.get("locked_test_metrics") != 0:
        errors.append("Generation run/LOCKED_TEST accounting mismatch")
    determinism = artifacts["determinism_report.json"]
    if determinism.get("data_pipeline_determinism") != "PASS":
        errors.append("Data-pipeline determinism did not pass")
    if determinism.get("model_run_reproducibility") != "PASS":
        warnings.append("Model-run reproducibility was not fully rerun/passed")

    identity = artifacts["artifact_identity.json"]
    if identity.get("artifact_json_count") != len(REQUIRED_ARTIFACTS) - 1 or identity.get("raw_payload_included") is not False or identity.get("production_model_created") is not False:
        errors.append("Artifact identity closure/boundary mismatch")
    _validate_checksums(root, output_dir, errors)
    for filename in REQUIRED_ARTIFACTS:
        if filename.endswith(".json"):
            forbidden = [x for x in ("/Users/", "file://", "~/", "/private/tmp/", "CloudDocs") if x in (output_dir / filename).read_text(encoding="utf-8")]
            if forbidden:
                errors.append(f"Non-portable path marker in {filename}: {forbidden}")
    if any(path.suffix.lower() in {".zip", ".txt", ".csv", ".npz", ".tflite"} for path in output_dir.iterdir() if path.name != "checksums.sha256"):
        errors.append("Raw/model payload included in C-B3 artifact directory")

    changed = set()
    diff = subprocess.run(["git", "diff", "--name-only", "origin/main...HEAD"], cwd=str(root), capture_output=True, text=True, check=False)
    changed.update(x for x in diff.stdout.splitlines() if x)
    changed.update(_status_paths(root))
    for path in sorted(changed):
        if not _allowed_path(path):
            errors.append(f"Unauthorized non-C-B3 path in branch/worktree: {path}")
        lower = path.lower()
        if "mmwave" in lower or "thermal" in lower or path.startswith(("shared/", "risk/", "integrated_node/", ".github/")):
            errors.append(f"Parallel-track contamination: {path}")
        if path.endswith((".zip", ".csv", ".npz", ".tflite")) or "raw_archives" in path:
            errors.append(f"Raw/model payload in diff: {path}")
    unique = subprocess.run(["git", "rev-list", "origin/main..HEAD"], cwd=str(root), capture_output=True, text=True, check=False).stdout.splitlines()
    for commit in [x for x in unique if x]:
        files = subprocess.run(["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit], cwd=str(root), capture_output=True, text=True, check=False).stdout.splitlines()
        for path in files:
            if path and not _allowed_path(path):
                errors.append(f"Branch-history contamination: {commit[:12]} {path}")
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=str(root), capture_output=True, text=True, check=False)
    if diff_check.returncode != 0 or diff_check.stdout.strip():
        errors.append("git diff --check failed")

    exceptions = artifacts["exceptions_and_limitations.json"]
    if exceptions.get("blockers"):
        errors.append("C-B3 exception registry contains blockers")
    for warning in exceptions.get("warnings", []):
        warnings.append(f"[{warning.get('code')}] {warning.get('description')}")
    errors = list(dict.fromkeys(errors))
    warnings = list(dict.fromkeys(warnings))
    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "artifact_dir": ARTIFACT_DIR_REL,
        "predecessor_status": predecessor_status,
        "c_b2_merged_main_ancestry": "PASS" if not any("C_B2_PREDECESSOR_NOT_MERGED" in e for e in errors) else "FAIL",
        "train": TRAIN_COUNT,
        "validation": VALIDATION_COUNT,
        "locked_test": LOCKED_TEST_COUNT,
        "architecture_count": len(ARCHITECTURE_IDS),
        "seed_count": len(SEEDS),
        "completed_architecture_runs": len(runs),
        "selected_architecture": artifacts["selected_architecture_profile.json"].get("winning_architecture_id"),
        "locked_test_predictions": leakage.get("locked_test_predictions"),
        "locked_test_metrics": leakage.get("locked_test_metrics"),
        "error_count": len(errors),
        "warning_count": len(warnings),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SafeNest CO2 Phase C-B3")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--skip-predecessors", action="store_true")
    args = parser.parse_args()
    result = validate(args.repo_root or get_repo_root(), run_predecessors=not args.skip_predecessors)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in ("PASS", "PASS_WITH_WARNINGS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
