#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone validator for Phase C-B2 imbalance/calibration evidence."""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

repo_root_dir = Path(__file__).resolve().parent.parent
if str(repo_root_dir) not in sys.path:
    sys.path.insert(0, str(repo_root_dir))

from datasets.co2.imbalance_calibration import (
    ARTIFACT_DIR_REL,
    AUTHORIZED_STRATEGIES,
    BALANCED_RANDOM_OVERSAMPLE,
    B1_MERGED_MAIN_COMMIT,
    B1_SELECTED_CANDIDATE_ID,
    B1_SELECTED_HISTORY_SECONDS,
    B1_SELECTED_METHOD,
    B1_SELECTED_PROFILE_ID,
    CALIBRATION_PROTOCOL_ID,
    CLASS_WEIGHT_BALANCED,
    DEFAULT_SEED,
    DEFAULT_THRESHOLD,
    ECE_BIN_COUNT,
    FEATURE_CONTEXT_ID,
    FIXED_FEATURES,
    FIXED_LOGISTIC_PARAMETERS,
    NATURAL_DISTRIBUTION,
    PROBE_PROFILE_ID,
    PRODUCTION_MODEL_REL,
    PRODUCTION_SCALER_REL,
    REFERENCE_THRESHOLD_RESULT_ID,
    SCALER_PROFILE_ID,
    SELECTED_IMBALANCE_POLICY_ID,
    STRATEGY_REGISTRY_ID,
    SYNTHETIC_FIXTURE_REL,
    TARGET_PROFILE_ID,
    THRESHOLD_COUNT,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
    _probability_fingerprint,
    assert_shared_scaler_fingerprints,
    build_balanced_oversample_plan,
    build_imbalance_strategy_registry,
    build_predecessor_fingerprint_registry,
    build_threshold_grid,
    classification_metrics_at_threshold,
    compute_balanced_class_weights,
    expected_calibration_error,
    fit_train_only_scaler,
    load_authorized_matrix,
    load_json,
    probability_quality_metrics,
    rank_imbalance_strategies,
    rank_threshold_rows,
    run_imbalance_calibration,
    validate_b1_selected_profile,
    validate_feature_context,
    validate_imbalance_registry,
    validate_logistic_parameter_contract,
    validate_population_contract,
    validate_probability_invariance,
    validate_probability_semantics,
    verify_oversample_evidence,
)
from datasets.co2.offline_experiment import (
    EXPECTED_LOCKED_TEST_SEALED,
    EXPECTED_TRAIN_COMMON,
    EXPECTED_VALIDATION_COMMON,
    EXPECTED_WARMUP_CANONICAL,
    _load_eligible_by_role,
    assert_no_forbidden_path_markers,
    build_sample_universe_manifest,
    verify_a_series_artifact_lock,
    verify_a_series_release,
)
from datasets.co2.raw_reader import compute_sha256_file, get_repo_root


REQUIRED_ARTIFACTS = (
    "predecessor_fingerprint_registry.json",
    "experiment_contract.json",
    "fixed_feature_context.json",
    "preprocessing_scaler_evidence.json",
    "imbalance_strategy_registry.json",
    "train_class_distribution.json",
    "class_weight_evidence.json",
    "balanced_sampling_evidence.json",
    "fixed_logistic_probe_contract.json",
    "probe_fit_evidence.json",
    "stage1_default_threshold_results.json",
    "imbalance_selection_decision.json",
    "threshold_calibration_protocol.json",
    "threshold_sweep_results.json",
    "reference_threshold_result.json",
    "calibration_diagnostics.json",
    "fp_fn_error_report.json",
    "occupancy_probability_semantic_contract.json",
    "validation_predictions.json",
    "leakage_audit.json",
    "exceptions_and_limitations.json",
    "generation_metadata.json",
    "artifact_identity.json",
    "checksums.sha256",
)

ALLOWED_EXACT_PATHS = {
    "datasets/co2/imbalance_calibration.py",
    "scripts/audit_co2_imbalance_calibration.py",
    "scripts/validate_co2_imbalance_calibration.py",
    "tests/test_co2_imbalance_calibration.py",
}

PROTECTED_SHARED = {
    "AGENTS.md",
    "README.md",
    "docs/README.md",
    "datasets/MANIFEST.json",
    "models/model_manifest.json",
    "docs/reports/model_inventory.json",
    "docs/reports/SENSOR_DATA_CONTRACT.md",
    "docs/reports/sensor_model_data_contract.json",
    PRODUCTION_SCALER_REL,
    PRODUCTION_MODEL_REL,
    SYNTHETIC_FIXTURE_REL,
}

# Scope ownership is deliberately narrower than “anything containing co2”.
# These are repository-local namespaces whose ownership is established by the
# active tree.  Shared/root paths are checked before these rules so a
# production asset or global contract cannot become allowed merely because it
# lives below a CO₂ directory.
C_B2_ARTIFACT_DRIFT = "C_B2_ARTIFACT_DRIFT"
C_B2_OWNED = "C_B2_OWNED"
CO2_SAME_TRACK = "CO2_SAME_TRACK"
MMWAVE_OTHER_TRACK = "MMWAVE_OTHER_TRACK"
THERMAL_OTHER_TRACK = "THERMAL_OTHER_TRACK"
INTEGRATION_OTHER_TRACK = "INTEGRATION_OTHER_TRACK"
SHARED_OR_UNAUTHORIZED = "SHARED_OR_UNAUTHORIZED"

_C_B2_ARTIFACT_PREFIX = f"{ARTIFACT_DIR_REL}/"
_CO2_LOCAL_PREFIXES = (
    "datasets/co2/",
    "models/co2/",
    "sensors/co2/",
    "benchmarks/co2/",
)
_CO2_LOCAL_PATTERNS = (
    "inference/co2_*.py",
    "scripts/co2_standalone/*",
    "scripts/*co2*.py",
    "tests/test_co2*.py",
    "docs/reports/co2/*",
    "docs/reports/co2_*.md",
)
_MMWAVE_PREFIXES = (
    "datasets/mmwave/",
    "models/mmwave/",
    "sensors/mmwave/",
    "devices/mmwave/",
    "benchmarks/mmwave/",
)
_MMWAVE_PATTERNS = (
    "scripts/*mmwave*.py",
    "tests/test_mmwave*.py",
    "docs/reports/mmwave/*",
    "docs/reports/mmwave_*.md",
)
_THERMAL_PREFIXES = (
    "datasets/thermal/",
    "models/thermal/",
    "models/thermal44/",
    "sensors/thermal/",
    "sensors/thermal44/",
    "devices/thermal/",
    "devices/thermal44/",
    "benchmarks/thermal/",
)
_THERMAL_PATTERNS = (
    "scripts/*thermal*.py",
    "tests/test_thermal*.py",
    "docs/reports/thermal/*",
    "docs/reports/thermal_*.md",
)
_INTEGRATION_PREFIXES = (
    "shared/",
    "risk/",
    "integrated_node/",
    "devices/",
    "ondevice_ai/",
    ".github/",
)


def _matches_path_rule(path: str, prefixes: Sequence[str], patterns: Sequence[str]) -> bool:
    return path.startswith(prefixes) or any(
        fnmatch.fnmatchcase(path, pattern) for pattern in patterns
    )


def classify_path_ownership(path: str) -> str:
    """Classify a changed path for predecessor isolation.

    C-B2 evidence is immutable even though it is same-track.  Later CO₂
    phases are allowed only after that protected namespace is checked, while
    cross-track and shared paths remain hard failures.
    """

    normalized = str(path).replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if (
        normalized.startswith("/")
        or normalized in PROTECTED_SHARED
    ):
        return SHARED_OR_UNAUTHORIZED
    if normalized.startswith(_C_B2_ARTIFACT_PREFIX):
        return C_B2_ARTIFACT_DRIFT
    if normalized in ALLOWED_EXACT_PATHS:
        return C_B2_OWNED
    if _matches_path_rule(normalized, _MMWAVE_PREFIXES, _MMWAVE_PATTERNS):
        return MMWAVE_OTHER_TRACK
    if _matches_path_rule(normalized, _THERMAL_PREFIXES, _THERMAL_PATTERNS):
        return THERMAL_OTHER_TRACK
    if _matches_path_rule(normalized, _CO2_LOCAL_PREFIXES, _CO2_LOCAL_PATTERNS):
        return CO2_SAME_TRACK
    if normalized.startswith(_INTEGRATION_PREFIXES):
        return INTEGRATION_OTHER_TRACK
    return SHARED_OR_UNAUTHORIZED


def audit_path_scope(
    scope_paths: Sequence[str],
    unique_commit_paths: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    """Audit working-tree and unique-commit paths without reading artifacts.

    The pure helper makes forward-compatibility behavior testable with small
    fixtures while the full validator still performs all C-B2 evidence and
    fingerprint checks independently.
    """

    errors: List[str] = []
    classifications: Dict[str, str] = {}
    same_track_paths: set[str] = set()
    cross_track_paths: set[str] = set()

    def record(path: str, *, commit: Optional[str] = None) -> None:
        normalized = str(path).replace("\\", "/")
        ownership = classify_path_ownership(normalized)
        classifications[normalized] = ownership
        if ownership == CO2_SAME_TRACK:
            same_track_paths.add(normalized)
            return
        if ownership in {C_B2_OWNED}:
            return
        if ownership == C_B2_ARTIFACT_DRIFT:
            errors.append(f"C_B2_ARTIFACT_DRIFT: {normalized}")
            return
        if ownership == MMWAVE_OTHER_TRACK:
            cross_track_paths.add(normalized)
            errors.append(
                f"PARALLEL_TRACK_BRANCH_CONTAMINATION: {commit[:12] + ' ' if commit else ''}{normalized}"
            )
            if commit is None:
                errors.append(f"mmWave contamination: {normalized}")
            return
        if ownership == THERMAL_OTHER_TRACK:
            cross_track_paths.add(normalized)
            errors.append(
                f"PARALLEL_TRACK_BRANCH_CONTAMINATION: {commit[:12] + ' ' if commit else ''}{normalized}"
            )
            if commit is None:
                errors.append(f"Thermal contamination: {normalized}")
            return
        if ownership == INTEGRATION_OTHER_TRACK:
            cross_track_paths.add(normalized)
            errors.append(
                f"PARALLEL_TRACK_BRANCH_CONTAMINATION: {commit[:12] + ' ' if commit else ''}{normalized}"
            )
            if commit is None:
                errors.append(f"Integration/shared contamination: {normalized}")
            return
        errors.append(f"Unauthorized non-C-B2 path in branch/worktree: {normalized}")

    for path in scope_paths:
        record(path)
        normalized = str(path).replace("\\", "/")
        if "raw_archives" in normalized or normalized.endswith((".zip", ".csv")):
            errors.append(f"Raw payload in diff: {normalized}")
    for commit, paths in (unique_commit_paths or {}).items():
        for path in paths:
            record(path, commit=commit)
            normalized = str(path).replace("\\", "/")
            if "raw_archives" in normalized or normalized.endswith((".zip", ".csv")):
                errors.append(f"Raw payload in branch history: {commit[:12]} {normalized}")

    return {
        "errors": list(dict.fromkeys(errors)),
        "path_ownership_classification": {
            path: classifications[path] for path in sorted(classifications)
        },
        "same_track_later_phase_paths": sorted(same_track_paths),
        "cross_track_contamination_paths": sorted(cross_track_paths),
    }


def _close(a: Any, b: Any, *, atol: float = 1e-14) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def _run_validator(repo_root: Path, script: str, *args: str) -> Tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, script, *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (result.stdout + "\n" + result.stderr).strip()
    return result.returncode == 0, combined


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )


def _extract_status_paths(status_text: str) -> List[str]:
    paths: List[str] = []
    for line in status_text.splitlines():
        if not line.strip():
            continue
        payload = line[3:] if len(line) >= 4 else line
        if " -> " in payload:
            payload = payload.split(" -> ", 1)[1]
        paths.append(payload.strip())
    return paths


def _allowed_c_b2_path(path: str) -> bool:
    return classify_path_ownership(path) == C_B2_OWNED


def _compare_metric_block(
    stored: Mapping[str, Any],
    live: Mapping[str, Any],
    errors: List[str],
    context: str,
) -> None:
    integer_keys = ("tn", "fp", "fn", "tp")
    float_keys = (
        "accuracy",
        "balanced_accuracy",
        "precision_vacant",
        "recall_vacant",
        "f1_vacant",
        "precision_occupied",
        "recall_occupied",
        "f1_occupied",
        "macro_f1",
        "false_positive_rate",
        "false_negative_rate",
        "specificity",
        "sensitivity",
    )
    for key in integer_keys:
        if int(stored.get(key, -1)) != int(live.get(key, -2)):
            errors.append(f"{context}: metric mismatch {key}")
    for key in float_keys:
        if not _close(stored.get(key), live.get(key)):
            errors.append(f"{context}: metric mismatch {key}")
    if stored.get("confusion_matrix") != live.get("confusion_matrix"):
        errors.append(f"{context}: confusion matrix mismatch")


def _artifact_hashes(directory: Path) -> Dict[str, str]:
    return {
        path.name: compute_sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def validate(
    repo_root: Path,
    *,
    rerun_determinism: bool = True,
    run_predecessors: bool = True,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    output_dir = repo_root / ARTIFACT_DIR_REL

    for name in REQUIRED_ARTIFACTS:
        if not (output_dir / name).is_file():
            errors.append(f"Missing C-B2 artifact: {name}")
    if errors:
        return {
            "status": "FAIL",
            "errors": errors,
            "warnings": warnings,
            "artifact_dir": ARTIFACT_DIR_REL,
        }

    # 1-3: independently validate C-B0/C-B1 and merged ancestry.
    predecessor_status = {"C-B0": "NOT_RUN", "C-B1": "NOT_RUN"}
    if run_predecessors:
        b0_ok, b0_output = _run_validator(
            repo_root, "scripts/validate_co2_offline_experiment.py"
        )
        predecessor_status["C-B0"] = "PASS" if b0_ok else "FAIL"
        if not b0_ok:
            errors.append(
                "C-B0 predecessor validator failed: "
                + " | ".join(b0_output.splitlines()[-8:])
            )
        b1_ok, b1_output = _run_validator(
            repo_root,
            "scripts/validate_co2_slope_ablation.py",
            "--skip-determinism",
        )
        predecessor_status["C-B1"] = "PASS" if b1_ok else "FAIL"
        if not b1_ok:
            errors.append(
                "C-B1 predecessor validator failed: "
                + " | ".join(b1_output.splitlines()[-8:])
            )

    ancestry = _git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        B1_MERGED_MAIN_COMMIT,
        "HEAD",
    )
    if ancestry.returncode != 0:
        errors.append("C-B1 merged-main commit is not in branch ancestry")

    release = verify_a_series_release(repo_root)
    if not release.get("matches_expected"):
        errors.append("A-series release anchor invalid")
    try:
        a_lock = verify_a_series_artifact_lock(repo_root)
        if a_lock.get("status") != "VERIFIED":
            errors.append("A-series artifact lock is not VERIFIED")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"A-series artifact lock validation failed: {exc}")
        a_lock = {}

    artifacts = {
        name: load_json(output_dir / name)
        for name in REQUIRED_ARTIFACTS
        if name.endswith(".json")
    }

    # 4, 9-10, 45: predecessor identity/fingerprint closure.
    try:
        live_predecessor = build_predecessor_fingerprint_registry(repo_root)
        stored_predecessor = artifacts["predecessor_fingerprint_registry.json"]
        if stored_predecessor != live_predecessor:
            errors.append("C_B0_OR_B1_PREDECESSOR_FINGERPRINT_MISMATCH")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Predecessor fingerprint validation failed: {exc}")

    try:
        selected_b1 = load_json(
            repo_root
            / "datasets/co2/manifests/c_b1_slope_method_history_ablation/selected_slope_profile.json"
        )
        validate_b1_selected_profile(selected_b1)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"B1 selected-slope validation failed: {exc}")

    # 5-8: population lock and overlaps, membership only for LOCKED_TEST.
    universe = build_sample_universe_manifest(repo_root)
    by_role = _load_eligible_by_role(repo_root)
    try:
        overlaps = validate_population_contract(
            by_role["TRAIN"], by_role["VALIDATION"], by_role["LOCKED_TEST"]
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Sample-universe validation failed: {exc}")
        overlaps = {}
    if universe.get("canonical_warmup_records") != EXPECTED_WARMUP_CANONICAL:
        errors.append("Canonical warm-up count drift")

    try:
        train, train_audit = load_authorized_matrix(
            repo_root=repo_root, split_role="TRAIN"
        )
        validation, validation_audit = load_authorized_matrix(
            repo_root=repo_root, split_role="VALIDATION"
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Authorized TRAIN/VALIDATION load failed: {exc}")
        train = validation = None
        train_audit = validation_audit = {}

    # 11: exact fixed feature context.
    feature_context = artifacts["fixed_feature_context.json"]
    try:
        validate_feature_context(feature_context.get("feature_order", []))
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    if feature_context.get("feature_context_id") != FEATURE_CONTEXT_ID:
        errors.append("Feature context identity mismatch")
    if feature_context.get("feature_count") != 4:
        errors.append("C-B2 feature count must be exactly four")
    if feature_context.get("slope_profile_id") != B1_SELECTED_PROFILE_ID:
        errors.append("B1 selected slope identity missing from feature context")
    if feature_context.get("slope_candidate_id") != B1_SELECTED_CANDIDATE_ID:
        errors.append("B1 selected slope candidate drift")
    if feature_context.get("slope_method") != B1_SELECTED_METHOD:
        errors.append("B1 selected slope method drift")
    if feature_context.get("minimum_history_seconds") != B1_SELECTED_HISTORY_SECONDS:
        errors.append("B1 selected slope history drift")

    # 12-13: original TRAIN-only scaler fit once and shared.
    scaler_evidence = artifacts["preprocessing_scaler_evidence.json"]
    if train is not None:
        try:
            _, live_scaler = fit_train_only_scaler(
                train,
                fit_population_fingerprint=universe["ordered_id_list_sha256"]["TRAIN"],
            )
            for key in (
                "scaler_profile_id",
                "classification",
                "feature_order",
                "fit_population",
                "fit_sample_count",
                "fit_population_fingerprint",
                "mean",
                "scale",
                "variance",
                "n_samples_seen",
                "scaler_fingerprint",
                "per_arm_scaler_fingerprint",
            ):
                if scaler_evidence.get(key) != live_scaler.get(key):
                    errors.append(f"Scaler evidence drift: {key}")
            assert_shared_scaler_fingerprints(scaler_evidence)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Scaler validation failed: {exc}")
    if scaler_evidence.get("validation_fit_rows") != 0:
        errors.append("Scaler includes VALIDATION")
    if scaler_evidence.get("locked_test_fit_rows") != 0:
        errors.append("Scaler includes LOCKED_TEST")
    if scaler_evidence.get("oversampled_fit_rows") != 0:
        errors.append("Scaler was fit after oversampling")

    # 14-22: exact strategies, weights, deterministic TRAIN-only oversampling.
    registry = artifacts["imbalance_strategy_registry.json"]
    try:
        validate_imbalance_registry(registry)
        live_registry = build_imbalance_strategy_registry()
        if registry != live_registry:
            errors.append("Imbalance strategy registry drift/post-hoc mutation")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Imbalance registry validation failed: {exc}")

    class_distribution = artifacts["train_class_distribution.json"]
    class_weight_evidence = artifacts["class_weight_evidence.json"]
    sampling_evidence = artifacts["balanced_sampling_evidence.json"]
    if train is not None:
        counts = {
            "VACANT": int(np.sum(train.labels == 0)),
            "OCCUPIED": int(np.sum(train.labels == 1)),
        }
        if class_distribution.get("class_counts") != counts:
            errors.append("TRAIN class-count evidence mismatch")
        try:
            weights = compute_balanced_class_weights(train.labels)
            expected_weights = {"VACANT": weights[0], "OCCUPIED": weights[1]}
            if class_weight_evidence.get("explicit_class_weights") != expected_weights:
                errors.append("Incorrect TRAIN-balanced class weights")
            verify_oversample_evidence(train.labels, train.sample_ids, sampling_evidence)
            plan = build_balanced_oversample_plan(train.labels, train.sample_ids)
            if plan.evidence["oversampled_class_counts"].get("VACANT") != plan.evidence[
                "oversampled_class_counts"
            ].get("OCCUPIED"):
                errors.append("Oversampling did not exactly balance classes")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Class-weight/oversampling validation failed: {exc}")
    if class_weight_evidence.get("derivation_population") != "TRAIN_ONLY":
        errors.append("Class weights were not TRAIN-derived only")
    if class_weight_evidence.get("validation_rows_used") != 0:
        errors.append("VALIDATION used for class weights")
    if class_weight_evidence.get("locked_test_rows_used") != 0:
        errors.append("LOCKED_TEST used for class weights")
    if sampling_evidence.get("source_population") != "TRAIN_ONLY":
        errors.append("Oversampling population is not TRAIN only")
    if sampling_evidence.get("majority_undersampling_count") != 0:
        errors.append("Majority undersampling detected")
    if sampling_evidence.get("seed") != DEFAULT_SEED:
        errors.append("Oversampling seed drift")
    if sampling_evidence.get("validation_rows_used") != 0:
        errors.append("VALIDATION used in oversampling")
    if sampling_evidence.get("locked_test_rows_used") != 0:
        errors.append("LOCKED_TEST used in oversampling")

    # 23-26: fixed logistic probe and no architecture/hyperparameter search.
    probe_contract = artifacts["fixed_logistic_probe_contract.json"]
    if probe_contract.get("probe_profile_id") != PROBE_PROFILE_ID:
        errors.append("Fixed logistic probe identity mismatch")
    try:
        validate_logistic_parameter_contract(probe_contract.get("parameters", {}))
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    for flag in (
        "architecture_search_performed",
        "hyperparameter_search_performed",
        "multi_seed_search_performed",
    ):
        if probe_contract.get(flag) is not False:
            errors.append(f"Unauthorized probe activity: {flag}")
    if probe_contract.get("production_model") is not False:
        errors.append("Reference probe labeled as production model")
    if probe_contract.get("final_architecture") is not False:
        errors.append("Reference probe labeled as final architecture")
    probe_fit = artifacts["probe_fit_evidence.json"]
    if set(probe_fit.get("per_strategy", {})) != set(AUTHORIZED_STRATEGIES):
        errors.append("Probe fit evidence missing strategy")
    for strategy_id, row in probe_fit.get("per_strategy", {}).items():
        try:
            validate_logistic_parameter_contract(row.get("parameters", {}))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{strategy_id}: {exc}")
        if row.get("fit_population") != "TRAIN_ONLY":
            errors.append(f"{strategy_id}: fit population not TRAIN only")
        if row.get("convergence_warning_count") != 0:
            errors.append(f"{strategy_id}: convergence warning present")

    # 26-28: Stage-1 fixed threshold, full metrics, and ranking.
    stage1 = artifacts["stage1_default_threshold_results.json"]
    candidates = stage1.get("candidates", {})
    if stage1.get("fixed_threshold") != DEFAULT_THRESHOLD:
        errors.append("Stage-1 threshold is not exactly 0.5")
    if stage1.get("candidate_count") != 3 or set(candidates) != set(AUTHORIZED_STRATEGIES):
        errors.append("Stage-1 does not contain exactly three arms")
    selected_policy = artifacts["imbalance_selection_decision.json"]
    try:
        live_ranking = rank_imbalance_strategies(list(candidates.values()))
        if selected_policy.get("ranking") != live_ranking:
            errors.append("Stage-1 ranking rule mismatch")
        if selected_policy.get("selected_strategy") != live_ranking[0]["strategy_id"]:
            errors.append("Selected imbalance arm mismatch")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Stage-1 ranking validation failed: {exc}")

    predictions = artifacts["validation_predictions.json"]
    stored_prediction_arms = predictions.get("stage1_default_threshold_arms", {})
    if validation is not None:
        for strategy_id in AUTHORIZED_STRATEGIES:
            block = stored_prediction_arms.get(strategy_id) or {}
            records = block.get("records") or []
            if len(records) != EXPECTED_VALIDATION_COMMON:
                errors.append(f"{strategy_id}: validation prediction count mismatch")
                continue
            sample_ids = [row.get("sample_id") for row in records]
            labels = np.asarray(
                [row.get("true_occupancy_label") for row in records], dtype=np.int64
            )
            probabilities = np.asarray(
                [row.get("occupancy_probability") for row in records], dtype=np.float64
            )
            stored_pred = np.asarray(
                [row.get("predicted_occupancy_label") for row in records], dtype=np.int64
            )
            if sample_ids != validation.sample_ids:
                errors.append(f"{strategy_id}: validation sample order mismatch")
            if not np.array_equal(labels, validation.labels):
                errors.append(f"{strategy_id}: validation target evidence mismatch")
            live_metrics, live_pred = classification_metrics_at_threshold(
                labels, probabilities, DEFAULT_THRESHOLD
            )
            if not np.array_equal(stored_pred, live_pred):
                errors.append(f"{strategy_id}: default-threshold predictions mismatch")
            if strategy_id in candidates:
                _compare_metric_block(
                    candidates[strategy_id]["metrics"],
                    live_metrics,
                    errors,
                    f"{strategy_id} Stage-1",
                )
                if candidates[strategy_id].get(
                    "validation_probability_vector_sha256"
                ) != _probability_fingerprint(sample_ids, probabilities):
                    errors.append(f"{strategy_id}: probability fingerprint mismatch")
                live_prob_metrics = probability_quality_metrics(labels, probabilities)
                live_ece = expected_calibration_error(labels, probabilities)
                live_prob_metrics["expected_calibration_error"] = live_ece[
                    "expected_calibration_error"
                ]
                for key, value in live_prob_metrics.items():
                    stored_value = candidates[strategy_id][
                        "probability_quality_metrics"
                    ].get(key)
                    if not _close(stored_value, value):
                        errors.append(f"{strategy_id}: probability metric mismatch {key}")

    if selected_policy.get("policy_id") != SELECTED_IMBALANCE_POLICY_ID:
        errors.append("Selected imbalance policy identity mismatch")
    if selected_policy.get("selection_threshold") != DEFAULT_THRESHOLD:
        errors.append("Selected imbalance policy did not use threshold 0.5")
    if selected_policy.get("deployment_status") != [
        "OFFLINE_VALIDATION_SELECTED",
        "ARCHITECTURE_GENERALIZATION_UNVERIFIED",
        "DEVICE_DOMAIN_UNVALIDATED",
    ]:
        errors.append("Selected imbalance policy deployment status overclaim")

    # 29-32: exact threshold protocol, independent sweep ranking, unchanged probs.
    protocol = artifacts["threshold_calibration_protocol.json"]
    if protocol.get("protocol_id") != CALIBRATION_PROTOCOL_ID:
        errors.append("Threshold calibration protocol identity mismatch")
    expected_grid = build_threshold_grid()
    if protocol.get("threshold_grid") != expected_grid:
        errors.append("Threshold grid mutation")
    if (
        protocol.get("threshold_grid_min") != THRESHOLD_MIN
        or protocol.get("threshold_grid_max") != THRESHOLD_MAX
        or protocol.get("threshold_grid_step") != THRESHOLD_STEP
        or protocol.get("threshold_count") != THRESHOLD_COUNT
    ):
        errors.append("Threshold grid contract fields invalid")
    if protocol.get("threshold_search_population") != "VALIDATION_ONLY":
        errors.append("Threshold search population is not VALIDATION only")
    if protocol.get("probability_recalibration_model_fitted") is not False:
        errors.append("Unauthorized probability recalibrator fitted")

    sweep = artifacts["threshold_sweep_results.json"]
    sweep_rows = sweep.get("rows", [])
    try:
        validate_probability_invariance(sweep_rows)
        live_threshold_ranking = rank_threshold_rows(sweep_rows)
        if sweep.get("ranking") != live_threshold_ranking:
            errors.append("Threshold ranking mismatch")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Threshold sweep validation failed: {exc}")
        live_threshold_ranking = []

    selected_strategy = selected_policy.get("selected_strategy")
    selected_default_records = (
        stored_prediction_arms.get(selected_strategy, {}).get("records", [])
    )
    if selected_default_records:
        selected_ids = [row["sample_id"] for row in selected_default_records]
        selected_labels = np.asarray(
            [row["true_occupancy_label"] for row in selected_default_records],
            dtype=np.int64,
        )
        selected_probabilities = np.asarray(
            [row["occupancy_probability"] for row in selected_default_records],
            dtype=np.float64,
        )
        if sweep.get("probability_vector_sha256") != _probability_fingerprint(
            selected_ids, selected_probabilities
        ):
            errors.append("Threshold sweep probability vector mismatch")
        for row in sweep_rows:
            live_metrics, _ = classification_metrics_at_threshold(
                selected_labels, selected_probabilities, float(row["threshold"])
            )
            _compare_metric_block(
                row["metrics"],
                live_metrics,
                errors,
                f"threshold {row['threshold']}",
            )

    reference = artifacts["reference_threshold_result.json"]
    if reference.get("result_id") != REFERENCE_THRESHOLD_RESULT_ID:
        errors.append("Reference threshold result identity mismatch")
    if live_threshold_ranking and reference.get("selected_reference_threshold") != live_threshold_ranking[0]["threshold"]:
        errors.append("Selected reference threshold does not match ranking")
    if reference.get("default_threshold") != DEFAULT_THRESHOLD:
        errors.append("Reference result default threshold drift")
    if reference.get("reference_threshold_production_final") is not False:
        errors.append("Reference threshold overclaimed as production-final")
    if reference.get("TRANSFER_TO_FUTURE_ARCHITECTURES") != "NOT_ASSUMED":
        errors.append("Reference threshold improperly transferred to future architectures")

    calibrated_block = predictions.get("selected_calibrated_threshold") or {}
    calibrated_records = calibrated_block.get("records") or []
    if selected_default_records and calibrated_records:
        if [row["sample_id"] for row in calibrated_records] != [
            row["sample_id"] for row in selected_default_records
        ]:
            errors.append("Calibrated prediction sample identity mutation")
        if [row["occupancy_probability"] for row in calibrated_records] != [
            row["occupancy_probability"] for row in selected_default_records
        ]:
            errors.append("Probability mutation across thresholds")

    # 33-38: FP/FN, calibration diagnostics, semantics, and no overclaim.
    fp_fn = artifacts["fp_fn_error_report.json"]
    if fp_fn.get("DOMAIN_FP_FN_COST_RATIO") != "UNSPECIFIED":
        errors.append("Fabricated/changed FP/FN domain cost ratio")
    if fp_fn.get("fabricated_weighted_safety_score") is not False:
        errors.append("Fabricated weighted safety score present")
    for strategy_id in AUTHORIZED_STRATEGIES:
        row = (fp_fn.get("stage1_default_threshold") or {}).get(strategy_id, {})
        if "fp" not in row or "fn" not in row:
            errors.append(f"Missing FP/FN accounting for {strategy_id}")

    calibration = artifacts["calibration_diagnostics.json"]
    ece_contract = calibration.get("ece_contract") or {}
    if ece_contract.get("bin_count") != ECE_BIN_COUNT:
        errors.append("ECE bin contract drift")
    if calibration.get("probability_recalibration_model_fitted") is not False:
        errors.append("Probability recalibration model unexpectedly fitted")
    for strategy_id, row in (calibration.get("per_stage1_arm") or {}).items():
        diagnostic = row.get("ece_diagnostic") or {}
        bins = diagnostic.get("bins") or []
        if len(bins) != ECE_BIN_COUNT:
            errors.append(f"{strategy_id}: missing calibration bins")
        if sum(int(b.get("sample_count", 0)) for b in bins) != EXPECTED_VALIDATION_COMMON:
            errors.append(f"{strategy_id}: calibration bins do not account for validation")

    semantics = artifacts["occupancy_probability_semantic_contract.json"]
    try:
        validate_probability_semantics(semantics)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    if semantics.get("target_profile_id") != TARGET_PROFILE_ID:
        errors.append("Occupancy target profile mismatch in semantic contract")
    if semantics.get("risk_logic_modified") is not False:
        errors.append("C-B2 modified risk logic")

    # 39-44: LOCKED_TEST zero-use, synthetic isolation, protected assets unchanged.
    leakage = artifacts["leakage_audit.json"]
    zero_keys = (
        "train_validation_overlap",
        "train_locked_test_overlap",
        "validation_locked_test_overlap",
        "target_as_feature",
        "provenance_as_feature",
        "validation_in_scaler_fit",
        "locked_test_in_scaler_fit",
        "validation_in_class_weight_derivation",
        "locked_test_in_class_weight_derivation",
        "validation_in_balanced_sampling",
        "locked_test_in_balanced_sampling",
        "validation_used_for_model_fitting",
        "locked_test_used_for_model_fitting",
        "locked_test_feature_access",
        "locked_test_target_access",
        "locked_test_threshold_access",
        "locked_test_predictions",
        "locked_test_probability_outputs",
        "locked_test_threshold_evaluations",
        "locked_test_metrics",
        "locked_test_fit_usage",
        "locked_test_tuning_usage",
    )
    for key in zero_keys:
        if leakage.get(key) != 0:
            errors.append(f"Leakage audit nonzero: {key}")
    if leakage.get("threshold_selected_on") != "VALIDATION_ONLY":
        errors.append("Threshold was not selected on VALIDATION only")
    if leakage.get("locked_test_membership_count_verified") != EXPECTED_LOCKED_TEST_SEALED:
        errors.append("LOCKED_TEST membership count not verified")
    for audit_name, live_audit in (
        ("train_matrix_load_audit", train_audit),
        ("validation_matrix_load_audit", validation_audit),
    ):
        stored_audit = leakage.get(audit_name) or {}
        if stored_audit != live_audit:
            errors.append(f"Matrix access audit drift: {audit_name}")
        if stored_audit.get("locked_test_feature_rows_decoded") != 0:
            errors.append(f"LOCKED_TEST features decoded in {audit_name}")
        if stored_audit.get("locked_test_target_rows_decoded") != 0:
            errors.append(f"LOCKED_TEST targets decoded in {audit_name}")

    generation = artifacts["generation_metadata.json"]
    false_flags = (
        "production_scaler_modified",
        "production_model_modified",
        "a_series_locked_artifacts_modified",
        "b0_predecessor_artifacts_modified",
        "b1_predecessor_artifacts_modified",
        "synthetic_npz_used_as_real_training_data",
        "architecture_comparison_performed",
        "multi_seed_comparison_performed",
        "final_feature_selection_performed",
        "probability_recalibration_model_fitted",
    )
    for key in false_flags:
        if generation.get(key) is not False:
            errors.append(f"Generation boundary violation: {key}")
    if generation.get("locked_test_predictions") != 0:
        errors.append("Generation reports LOCKED_TEST predictions")
    if generation.get("locked_test_metrics") != 0:
        errors.append("Generation reports LOCKED_TEST metrics")

    # 46-48: checksums, repository-relative paths, no raw payload.
    checksum_rows = (output_dir / "checksums.sha256").read_text(
        encoding="utf-8"
    ).splitlines()
    expected_checksum_files = {
        name for name in REQUIRED_ARTIFACTS if name.endswith(".json")
    }
    seen_checksum_files = set()
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
        path = repo_root / rel
        seen_checksum_files.add(path.name)
        if not path.is_file():
            errors.append(f"Checksum path missing: {rel}")
        elif compute_sha256_file(path) != digest:
            errors.append(f"Checksum mismatch: {rel}")
    if seen_checksum_files != expected_checksum_files:
        errors.append("Checksum closure does not cover every JSON artifact exactly")

    for name in REQUIRED_ARTIFACTS:
        path = output_dir / name
        text = path.read_text(encoding="utf-8")
        forbidden = assert_no_forbidden_path_markers(text)
        if forbidden:
            errors.append(f"Forbidden path marker in {name}: {forbidden}")
    if any(
        path.suffix.lower() in {".zip", ".txt", ".csv", ".npz", ".tflite"}
        for path in output_dir.iterdir()
        if path.name != "checksums.sha256"
    ):
        errors.append("Raw/model payload included in C-B2 artifact directory")

    identity = artifacts["artifact_identity.json"]
    if identity.get("raw_payload_included") is not False:
        errors.append("Artifact identity reports raw payload")
    if identity.get("reference_threshold_production_final") is not False:
        errors.append("Artifact identity overclaims production-final threshold")

    # 49: bit-identical complete regeneration.
    determinism_status = "SKIPPED"
    if rerun_determinism and not errors:
        before = _artifact_hashes(output_dir)
        try:
            run_imbalance_calibration(repo_root)
            after = _artifact_hashes(output_dir)
            if before != after:
                drift = sorted(
                    name
                    for name in set(before) | set(after)
                    if before.get(name) != after.get(name)
                )
                errors.append(f"Determinism mismatch: {drift}")
                determinism_status = "FAIL"
            else:
                determinism_status = "PASS_BIT_IDENTICAL"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Determinism regeneration failed: {exc}")
            determinism_status = "FAIL"

    # 50: C-B2 working-tree/history/PR-diff scope.  Later same-track CO₂
    # phases are valid predecessor context; only cross-track, shared, unknown,
    # or C-B2-artifact paths are rejected.
    changed = _git(repo_root, "diff", "--name-only", "origin/main...HEAD")
    status = _git(repo_root, "status", "--porcelain", "--untracked-files=all")
    scope_paths = set(changed.stdout.splitlines()) | set(
        _extract_status_paths(status.stdout)
    )
    scope_paths.discard("")
    unique_commits_result = _git(repo_root, "rev-list", "origin/main..HEAD")
    unique_commits = [x for x in unique_commits_result.stdout.splitlines() if x]
    unique_commit_paths: Dict[str, List[str]] = {}
    for commit in unique_commits:
        unique_commit_paths[commit] = _git(
            repo_root,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            commit,
        ).stdout.splitlines()
    scope_audit = audit_path_scope(sorted(scope_paths), unique_commit_paths)
    errors.extend(scope_audit["errors"])

    diff_check = _git(repo_root, "diff", "--check")
    if diff_check.returncode != 0 or diff_check.stdout.strip():
        errors.append(f"git diff --check failed: {diff_check.stdout.strip()}")

    exception_registry = artifacts["exceptions_and_limitations.json"]
    if exception_registry.get("blockers"):
        errors.append("C-B2 exception registry contains blockers")
    for warning in exception_registry.get("warnings", []):
        warnings.append(
            f"[{warning.get('code')}] {warning.get('description')}"
        )

    # Deduplicate repeated evidence failures for readable standalone output.
    errors = list(dict.fromkeys(errors))
    warnings = list(dict.fromkeys(warnings))
    final_status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "status": final_status,
        "errors": errors,
        "warnings": warnings,
        "artifact_dir": ARTIFACT_DIR_REL,
        "predecessor_status": predecessor_status,
        "b1_merged_main_ancestry": "PASS" if ancestry.returncode == 0 else "FAIL",
        "a_series_release": release,
        "a_series_lock_status": a_lock.get("status"),
        "train": len(by_role["TRAIN"]),
        "validation": len(by_role["VALIDATION"]),
        "locked_test": len(by_role["LOCKED_TEST"]),
        "locked_test_predictions": leakage.get("locked_test_predictions"),
        "locked_test_metrics": leakage.get("locked_test_metrics"),
        "selected_imbalance_strategy": selected_policy.get("selected_strategy"),
        "selected_reference_threshold": reference.get("selected_reference_threshold"),
        "reference_threshold_production_final": reference.get(
            "reference_threshold_production_final"
        ),
        "determinism": determinism_status,
        "unique_branch_commits": unique_commits,
        "changed_or_untracked_paths": sorted(scope_paths),
        "path_ownership_classification": scope_audit[
            "path_ownership_classification"
        ],
        "same_track_later_phase_paths": scope_audit[
            "same_track_later_phase_paths"
        ],
        "cross_track_contamination_paths": scope_audit[
            "cross_track_contamination_paths"
        ],
        "error_count": len(errors),
        "warning_count": len(warnings),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SafeNest CO2 Phase C-B2")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--skip-determinism", action="store_true")
    parser.add_argument("--skip-predecessors", action="store_true")
    args = parser.parse_args()
    root = (args.repo_root or get_repo_root()).resolve()
    result = validate(
        root,
        rerun_determinism=not args.skip_determinism,
        run_predecessors=not args.skip_predecessors,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in ("PASS", "PASS_WITH_WARNINGS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
