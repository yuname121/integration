#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SafeNest CO₂ Phase C-B3 controlled architecture/multi-seed evidence.

The implementation deliberately keeps the C-B2 data protocol immutable.  It
materialises the two open split matrices once, fits the C-B2 TRAIN-only scaler
once, reuses the approved C-B2 oversampled TRAIN multiset for every run, and
evaluates only VALIDATION.  The resulting files are evidence for offline
architecture selection; no production model is created here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from datasets.co2.imbalance_calibration import (
    BALANCED_RANDOM_OVERSAMPLE,
    B1_SELECTED_CANDIDATE_ID,
    B1_SELECTED_HISTORY_SECONDS,
    B1_SELECTED_METHOD,
    B1_SELECTED_PROFILE_ID,
    CALIBRATION_PROTOCOL_ID as B2_CALIBRATION_PROTOCOL_ID,
    ECE_BIN_COUNT,
    FIXED_FEATURES,
    _probability_fingerprint,
    _stable_json_sha256,
    build_balanced_oversample_plan,
    build_sample_universe_manifest,
    build_threshold_grid,
    build_threshold_sweep,
    classification_metrics_at_threshold,
    expected_calibration_error,
    fit_train_only_scaler,
    load_authorized_matrix,
    load_json,
    probability_quality_metrics,
    validate_b1_selected_profile,
    validate_population_contract,
    verify_oversample_evidence,
)
from datasets.co2.offline_experiment import (
    EXPECTED_LOCKED_TEST_SEALED,
    EXPECTED_TRAIN_COMMON,
    EXPECTED_VALIDATION_COMMON,
    _load_eligible_by_role,
    assert_no_forbidden_path_markers,
    verify_a_series_artifact_lock,
    verify_a_series_release,
)
from datasets.co2.raw_reader import compute_sha256_file, get_repo_root


PHASE_ID = "C-B3"
PHASE_NAME = "CO2_CONTROLLED_ARCHITECTURE_FAMILY_AND_MULTI_SEED_STABILITY"
ARTIFACT_DIR_REL = "datasets/co2/manifests/c_b3_architecture_multiseed"
B0_DIR_REL = "datasets/co2/manifests/c_b0_offline_experiment_contract"
B1_DIR_REL = "datasets/co2/manifests/c_b1_slope_method_history_ablation"
B2_DIR_REL = "datasets/co2/manifests/c_b2_imbalance_calibration"

CB2_MERGED_MAIN_COMMIT = "18bff3a106fbf7b4334391b90ebcd90e79e6d3b7"
B0_CONTRACT_ID = "CO2_B0_OFFLINE_EXPERIMENT_CONTRACT_001"
B1_SELECTED_PROFILE_ID = "CO2_B1_SELECTED_SLOPE_PROFILE_001"
B2_SCALER_PROFILE_ID = "CO2_B2_TRAIN_ONLY_STANDARD_SCALER_001"
B2_POLICY_ID = "CO2_B2_SELECTED_IMBALANCE_POLICY_001"
B2_THRESHOLD_PROTOCOL_ID = "CO2_B2_THRESHOLD_CALIBRATION_PROTOCOL_001"
B2_SEMANTIC_CONTRACT_ID = "CO2_B2_OCCUPANCY_PROBABILITY_SEMANTIC_001"
TARGET_PROFILE_ID = "CO2_OCCUPANCY_TARGET_PROFILE_001"
A_SERIES_RELEASE_TAG = "co2-a-series-raw-to-canonical"
A_SERIES_RELEASE_TARGET = "bfd860cad2bb8dafe35ef7600cfa931d7d2d554d"

TRAIN_COUNT = 8140
VALIDATION_COUNT = 2662
LOCKED_TEST_COUNT = 9749
POSITIVE_CLASS = "OCCUPIED"
NEGATIVE_CLASS = "VACANT"
DEFAULT_THRESHOLD = 0.50
SEEDS = (20260810, 20260811, 20260812, 20260813, 20260814)
EXPECTED_RUN_COUNT = 20
ARCHITECTURE_IDS = (
    "LINEAR_LOGISTIC",
    "TREE_RANDOM_FOREST",
    "TINY_MLP",
    "SMALL_MLP",
)
ARCHITECTURE_SIMPLICITY = {
    "LINEAR_LOGISTIC": 0,
    "TINY_MLP": 1,
    "SMALL_MLP": 2,
    "TREE_RANDOM_FOREST": 3,
}
METRIC_NAMES = (
    "macro_f1",
    "balanced_accuracy",
    "recall_occupied",
    "precision_occupied",
    "false_positive_rate",
    "false_negative_rate",
    "roc_auc",
    "pr_auc_average_precision",
    "brier_score",
    "expected_calibration_error",
)
LOWER_IS_BETTER = {
    "false_positive_rate": True,
    "false_negative_rate": True,
    "brier_score": True,
    "expected_calibration_error": True,
}

PRODUCTION_SCALER_REL = "models/co2/co2_scaling_metadata_v0.1.0.json"
PRODUCTION_MODEL_REL = "models/co2/co2_occupancy_int8_v0.1.0.tflite"
SYNTHETIC_FIXTURE_REL = "datasets/co2/processed/co2_occupancy_v1.npz"


class CB3Error(RuntimeError):
    """Base C-B3 contract error."""


class RequiredBackendUnavailable(CB3Error):
    """Raised when the preregistered backend grid cannot be executed."""


class PredecessorFingerprintMismatch(CB3Error):
    """Raised when immutable C-B0/C-B1/C-B2 evidence drifts."""


class LockedTestPolicyViolation(CB3Error):
    """Raised when a caller asks for sealed LOCKED_TEST predictive data."""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_sha256(payload: Any) -> str:
    return _stable_json_sha256(payload)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(root), capture_output=True, text=True, check=False
    )


def ensure_cb2_merged(root: Path) -> None:
    result = _git(root, "merge-base", "--is-ancestor", CB2_MERGED_MAIN_COMMIT, "HEAD")
    if result.returncode != 0:
        raise PredecessorFingerprintMismatch("C_B2_PREDECESSOR_NOT_MERGED")


def _verify_checksum_manifest(root: Path, directory_rel: str) -> None:
    checksum_path = root / directory_rel / "checksums.sha256"
    if not checksum_path.is_file():
        raise PredecessorFingerprintMismatch(f"Missing checksum manifest: {directory_rel}")
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, rel = line.split("  ", 1)
        except ValueError as exc:
            raise PredecessorFingerprintMismatch(
                f"Malformed checksum row in {directory_rel}: {line!r}"
            ) from exc
        path = root / rel
        if not path.is_file() or compute_sha256_file(path) != digest:
            raise PredecessorFingerprintMismatch(
                f"C_B0_B1_B2_PREDECESSOR_FINGERPRINT_MISMATCH: {rel}"
            )


def _predecessor_specs() -> Tuple[Tuple[str, str, str, str], ...]:
    return (
        (B0_DIR_REL, "experiment_contract.json", "C-B0", "B0_EXPERIMENT_CONTRACT"),
        (B0_DIR_REL, "sample_universe_manifest.json", "C-B0", "B0_COMPARISON_UNIVERSE"),
        (B0_DIR_REL, "metric_contract.json", "C-B0", "B0_METRIC_CONTRACT"),
        (B0_DIR_REL, "preprocessing_fit_evidence.json", "C-B0", "B0_PREPROCESSING_CONTRACT"),
        (B0_DIR_REL, "leakage_audit.json", "C-B0", "B0_LEAKAGE_CONTRACT"),
        (B1_DIR_REL, "selected_slope_profile.json", "C-B1", "B1_SELECTED_SLOPE_PROFILE"),
        (B1_DIR_REL, "candidate_feature_fingerprint_registry.json", "C-B1", "B1_FEATURE_FINGERPRINTS"),
        (B1_DIR_REL, "leakage_audit.json", "C-B1", "B1_LEAKAGE_AUDIT"),
        (B2_DIR_REL, "experiment_contract.json", "C-B2", "B2_EXPERIMENT_CONTRACT"),
        (B2_DIR_REL, "imbalance_selection_decision.json", "C-B2", "B2_SELECTED_IMBALANCE_POLICY"),
        (B2_DIR_REL, "preprocessing_scaler_evidence.json", "C-B2", "B2_SCALER_EVIDENCE"),
        (B2_DIR_REL, "balanced_sampling_evidence.json", "C-B2", "B2_OVERSAMPLING_EVIDENCE"),
        (B2_DIR_REL, "threshold_calibration_protocol.json", "C-B2", "B2_THRESHOLD_PROTOCOL"),
        (B2_DIR_REL, "occupancy_probability_semantic_contract.json", "C-B2", "B2_PROBABILITY_SEMANTICS"),
        (B2_DIR_REL, "leakage_audit.json", "C-B2", "B2_LEAKAGE_AUDIT"),
        (B2_DIR_REL, "predecessor_fingerprint_registry.json", "C-B2", "B2_PREDECESSOR_CLOSURE"),
        ("datasets/co2/manifests/c_a6_final_integrity_lock", "artifact_lock_manifest.json", "C-A6", "A_SERIES_ARTIFACT_LOCK"),
        (PRODUCTION_SCALER_REL.rsplit("/", 1)[0], PRODUCTION_SCALER_REL.rsplit("/", 1)[1], "PREEXISTING_PRODUCTION", "PRODUCTION_SCALER_READ_ONLY"),
        (PRODUCTION_MODEL_REL.rsplit("/", 1)[0], PRODUCTION_MODEL_REL.rsplit("/", 1)[1], "PREEXISTING_PRODUCTION", "PRODUCTION_MODEL_READ_ONLY"),
        (SYNTHETIC_FIXTURE_REL.rsplit("/", 1)[0], SYNTHETIC_FIXTURE_REL.rsplit("/", 1)[1], "PREEXISTING_FIXTURE", "SYNTHETIC_SMOKE_FIXTURE_EXCLUDED"),
    )


def validate_predecessor_inputs(root: Path) -> Dict[str, Any]:
    ensure_cb2_merged(root)
    for directory in (B0_DIR_REL, B1_DIR_REL, B2_DIR_REL):
        _verify_checksum_manifest(root, directory)

    b0 = load_json(root / B0_DIR_REL / "experiment_contract.json")
    universe = load_json(root / B0_DIR_REL / "sample_universe_manifest.json")
    if b0.get("experiment_contract_id") != B0_CONTRACT_ID:
        raise PredecessorFingerprintMismatch("C-B0 contract identity drift")
    if (
        universe.get("b_series_common_train") != TRAIN_COUNT
        or universe.get("b_series_common_validation") != VALIDATION_COUNT
        or universe.get("b_series_sealed_locked_test") != LOCKED_TEST_COUNT
    ):
        raise PredecessorFingerprintMismatch("C-B0 comparison universe drift")

    b1_profile = load_json(root / B1_DIR_REL / "selected_slope_profile.json")
    validate_b1_selected_profile(b1_profile)
    if b1_profile.get("selected_candidate_id") != B1_SELECTED_CANDIDATE_ID:
        raise PredecessorFingerprintMismatch("B1 selected slope drift")
    b1_leakage = load_json(root / B1_DIR_REL / "leakage_audit.json")
    if b1_leakage.get("status") != "PASS":
        raise PredecessorFingerprintMismatch("B1 leakage audit is not PASS")

    b2_contract = load_json(root / B2_DIR_REL / "experiment_contract.json")
    b2_policy = load_json(root / B2_DIR_REL / "imbalance_selection_decision.json")
    b2_scaler = load_json(root / B2_DIR_REL / "preprocessing_scaler_evidence.json")
    b2_sampling = load_json(root / B2_DIR_REL / "balanced_sampling_evidence.json")
    b2_threshold = load_json(root / B2_DIR_REL / "threshold_calibration_protocol.json")
    b2_semantics = load_json(root / B2_DIR_REL / "occupancy_probability_semantic_contract.json")
    b2_leakage = load_json(root / B2_DIR_REL / "leakage_audit.json")
    if b2_contract.get("phase") != "C-B2":
        raise PredecessorFingerprintMismatch("C-B2 contract phase drift")
    if b2_policy.get("selected_strategy") != BALANCED_RANDOM_OVERSAMPLE:
        raise PredecessorFingerprintMismatch("C-B2 selected imbalance policy drift")
    if b2_policy.get("policy_id") != B2_POLICY_ID:
        raise PredecessorFingerprintMismatch("C-B2 policy identity drift")
    if b2_scaler.get("scaler_profile_id") != B2_SCALER_PROFILE_ID:
        raise PredecessorFingerprintMismatch("C-B2 scaler profile drift")
    if b2_scaler.get("feature_order") != list(FIXED_FEATURES):
        raise PredecessorFingerprintMismatch("C-B2 scaler feature order drift")
    if b2_scaler.get("fit_sample_count") != TRAIN_COUNT:
        raise PredecessorFingerprintMismatch("C-B2 scaler TRAIN count drift")
    if b2_sampling.get("strategy_id") != BALANCED_RANDOM_OVERSAMPLE:
        raise PredecessorFingerprintMismatch("C-B2 oversampling strategy drift")
    if b2_sampling.get("validation_rows_used") != 0 or b2_sampling.get("locked_test_rows_used") != 0:
        raise PredecessorFingerprintMismatch("C-B2 oversampling leakage")
    if b2_threshold.get("protocol_id") != B2_THRESHOLD_PROTOCOL_ID:
        raise PredecessorFingerprintMismatch("C-B2 threshold protocol drift")
    if b2_threshold.get("threshold_search_population") != "VALIDATION_ONLY":
        raise PredecessorFingerprintMismatch("C-B2 threshold population drift")
    if b2_semantics.get("contract_id") != B2_SEMANTIC_CONTRACT_ID:
        raise PredecessorFingerprintMismatch("C-B2 probability semantic drift")
    if b2_semantics.get("risk_semantic") != "NONE" or b2_semantics.get("safety_semantic") != "NONE":
        raise PredecessorFingerprintMismatch("C-B2 occupancy/safety semantic conflation")
    if b2_leakage.get("status") != "PASS" or b2_leakage.get("locked_test_predictions") != 0:
        raise PredecessorFingerprintMismatch("C-B2 leakage audit is not sealed/pass")

    release = verify_a_series_release(root)
    lock = verify_a_series_artifact_lock(root)
    if not release.get("matches_expected") or release.get("resolved_commit") != A_SERIES_RELEASE_TARGET:
        raise PredecessorFingerprintMismatch("A-series release anchor invalid")
    if lock.get("status") != "VERIFIED":
        raise PredecessorFingerprintMismatch("A-series artifact lock invalid")
    return {
        "b0_contract": b0,
        "b0_universe": universe,
        "b1_profile": b1_profile,
        "b2_policy": b2_policy,
        "b2_scaler": b2_scaler,
        "b2_sampling": b2_sampling,
        "b2_threshold": b2_threshold,
        "b2_semantics": b2_semantics,
        "b2_leakage": b2_leakage,
        "a_series_release": release,
        "a_series_lock": lock,
    }


def build_predecessor_fingerprint_registry(root: Path) -> Dict[str, Any]:
    validate_predecessor_inputs(root)
    entries: List[Dict[str, Any]] = []
    for directory, filename, phase, role in _predecessor_specs():
        rel = f"{directory}/{filename}"
        path = root / rel
        if not path.is_file():
            raise PredecessorFingerprintMismatch(f"Missing predecessor artifact: {rel}")
        entries.append(
            {
                "path": rel,
                "phase": phase,
                "role": role,
                "byte_size": path.stat().st_size,
                "sha256": compute_sha256_file(path),
            }
        )
    entries.sort(key=lambda row: row["path"])
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "registry_id": "CO2_B3_PREDECESSOR_FINGERPRINT_REGISTRY_001",
        "required_c_b2_merged_main_commit": CB2_MERGED_MAIN_COMMIT,
        "a_series_release_tag": A_SERIES_RELEASE_TAG,
        "a_series_release_target": A_SERIES_RELEASE_TARGET,
        "b0_contract_id": B0_CONTRACT_ID,
        "b1_selected_slope_profile_id": B1_SELECTED_PROFILE_ID,
        "b2_policy_id": B2_POLICY_ID,
        "b2_scaler_profile_id": B2_SCALER_PROFILE_ID,
        "b2_threshold_protocol_id": B2_THRESHOLD_PROTOCOL_ID,
        "entry_count": len(entries),
        "entries": entries,
        "closure_fingerprint": stable_sha256(entries),
        "closure_status": "LOCKED",
        "mismatch_status": "C_B0_B1_B2_PREDECESSOR_FINGERPRINT_MISMATCH_ON_DRIFT",
    }


def verify_stored_predecessor_registry(root: Path, stored: Mapping[str, Any]) -> None:
    live = build_predecessor_fingerprint_registry(root)
    if dict(stored) != live:
        raise PredecessorFingerprintMismatch("C_B0_B1_B2_PREDECESSOR_FINGERPRINT_MISMATCH")


def architecture_registry() -> Dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "registry_id": "CO2_B3_ARCHITECTURE_CANDIDATE_REGISTRY_001",
        "registry_created_before_results": True,
        "candidate_count": 4,
        "candidate_ids": list(ARCHITECTURE_IDS),
        "fixed_feature_context": list(FIXED_FEATURES),
        "fixed_imbalance_policy": BALANCED_RANDOM_OVERSAMPLE,
        "fixed_scaler_profile_id": B2_SCALER_PROFILE_ID,
        "hyperparameter_search_performed": False,
        "early_stopping_tuning_performed": False,
        "candidates": [
            {
                "architecture_id": "LINEAR_LOGISTIC",
                "profile_id": "CO2_B3_LINEAR_LOGISTIC_001",
                "family": "LINEAR",
                "implementation": "sklearn.linear_model.LogisticRegression",
                "parameters": {
                    "penalty": "l2",
                    "C": 1.0,
                    "solver": "lbfgs",
                    "fit_intercept": True,
                    "max_iter": 2000,
                    "class_weight": None,
                    "random_state": "ARCHITECTURE_SEED",
                },
                "seed_affects": ["random_state_if_accepted"],
                "complexity": {"coefficient_count": 4, "intercept_count": 1},
                "deployment_status": "OFFLINE_DIAGNOSTIC_ONLY",
                "hyperparameter_search_performed": False,
            },
            {
                "architecture_id": "TREE_RANDOM_FOREST",
                "profile_id": "CO2_B3_RANDOM_FOREST_001",
                "family": "TREE",
                "implementation": "sklearn.ensemble.RandomForestClassifier",
                "parameters": {
                    "n_estimators": 200,
                    "max_depth": 8,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt",
                    "bootstrap": True,
                    "class_weight": None,
                    "n_jobs": 1,
                    "random_state": "ARCHITECTURE_SEED",
                },
                "seed_affects": ["bootstrap", "tree_rng"],
                "complexity": {
                    "tree_count": 200,
                    "depth_configuration": "max_depth=8",
                    "estimated_serialized_size_bytes": None,
                },
                "deployment_status": "OFFLINE_TREE_DIAGNOSTIC_NOT_TFLITE_VALIDATED",
                "hyperparameter_search_performed": False,
            },
            {
                "architecture_id": "TINY_MLP",
                "profile_id": "CO2_B3_TINY_MLP_001",
                "family": "MLP",
                "implementation": "tensorflow.keras",
                "parameters": {
                    "input_dimension": 4,
                    "layers": [
                        {"type": "Input", "shape": [4]},
                        {"type": "Dense", "units": 8, "activation": "relu"},
                        {"type": "Dense", "units": 1, "activation": "sigmoid"},
                    ],
                    "optimizer": "Adam",
                    "learning_rate": 0.001,
                    "loss": "binary_crossentropy",
                    "batch_size": 64,
                    "epochs": 100,
                    "early_stopping": False,
                    "shuffle": True,
                    "seed": "ARCHITECTURE_SEED",
                },
                "seed_affects": ["weight_initialization", "minibatch_order"],
                "complexity": {"trainable_parameter_count": 49},
                "deployment_status": "OFFLINE_TINYML_SCALE_ONLY",
                "hyperparameter_search_performed": False,
            },
            {
                "architecture_id": "SMALL_MLP",
                "profile_id": "CO2_B3_SMALL_MLP_001",
                "family": "MLP",
                "implementation": "tensorflow.keras",
                "parameters": {
                    "input_dimension": 4,
                    "layers": [
                        {"type": "Input", "shape": [4]},
                        {"type": "Dense", "units": 16, "activation": "relu"},
                        {"type": "Dense", "units": 8, "activation": "relu"},
                        {"type": "Dense", "units": 1, "activation": "sigmoid"},
                    ],
                    "optimizer": "Adam",
                    "learning_rate": 0.001,
                    "loss": "binary_crossentropy",
                    "batch_size": 64,
                    "epochs": 100,
                    "early_stopping": False,
                    "shuffle": True,
                    "seed": "ARCHITECTURE_SEED",
                },
                "seed_affects": ["weight_initialization", "minibatch_order"],
                "complexity": {"trainable_parameter_count": 225},
                "deployment_status": "OFFLINE_SMALL_MLP_ONLY",
                "hyperparameter_search_performed": False,
            },
        ],
        "selection_rule_id": "CO2_B3_ARCHITECTURE_RANKING_RULE_001",
    }


def seed_registry() -> Dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "registry_id": "CO2_B3_ARCHITECTURE_SEED_REGISTRY_001",
        "pre_registered_before_validation_results": True,
        "seed_count": len(SEEDS),
        "seeds": list(SEEDS),
        "post_hoc_seed_addition": False,
        "post_hoc_seed_removal": False,
        "seed_role": "ARCHITECTURE_STOCHASTICITY_ONLY",
        "fixed_data_rng_seed": 20260810,
    }


def validate_architecture_registry(registry: Mapping[str, Any]) -> None:
    if registry.get("candidate_count") != 4:
        raise CB3Error("Architecture registry count mismatch")
    if registry.get("candidate_ids") != list(ARCHITECTURE_IDS):
        raise CB3Error("Unauthorized architecture or architecture order")
    if registry.get("registry_created_before_results") is not True:
        raise CB3Error("Architecture registry was not preregistered")
    if registry.get("hyperparameter_search_performed") is not False:
        raise CB3Error("Architecture hyperparameter search was performed")
    expected = architecture_registry()
    if registry != expected:
        raise CB3Error("Architecture candidate definition drift")


def validate_seed_registry(registry: Mapping[str, Any]) -> None:
    if registry.get("seed_count") != 5 or registry.get("seeds") != list(SEEDS):
        raise CB3Error("Seed registry mismatch or post-hoc seed change")
    if registry.get("pre_registered_before_validation_results") is not True:
        raise CB3Error("Seed registry was not preregistered")
    if registry.get("post_hoc_seed_addition") is not False or registry.get("post_hoc_seed_removal") is not False:
        raise CB3Error("Post-hoc seed mutation")
    if registry != seed_registry():
        raise CB3Error("Seed registry definition drift")


def validate_feature_context(features: Sequence[str]) -> None:
    if tuple(features) != FIXED_FEATURES:
        raise CB3Error("Feature context must contain exactly the four fixed C-B2 features")


def validate_locked_test_access(role: str) -> None:
    if role == "LOCKED_TEST":
        raise LockedTestPolicyViolation("LOCKED_TEST_POLICY_VIOLATION")


@dataclass(frozen=True)
class PreparedData:
    train: Any
    validation: Any
    x_train_scaled: np.ndarray
    x_validation_scaled: np.ndarray
    scaler_evidence: Dict[str, Any]
    oversample_plan: Any
    universe: Dict[str, Any]
    train_load_audit: Dict[str, Any]
    validation_load_audit: Dict[str, Any]
    original_train_fingerprint: str
    validation_fingerprint: str
    transformed_train_fingerprint: str
    transformed_validation_fingerprint: str
    oversampled_train_fingerprint: str


def _array_fingerprint(sample_ids: Sequence[str], features: np.ndarray, labels: Optional[np.ndarray] = None) -> str:
    h = hashlib.sha256()
    h.update(("\n".join(sample_ids) + "\n").encode("utf-8"))
    h.update(np.asarray(features, dtype="<f8").tobytes(order="C"))
    if labels is not None:
        h.update(np.asarray(labels, dtype="<i8").tobytes(order="C"))
    return h.hexdigest()


def _ids_fingerprint(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(sample_ids) + "\n").encode("utf-8")).hexdigest()


def prepare_fixed_data(root: Path) -> PreparedData:
    predecessor = validate_predecessor_inputs(root)
    validate_feature_context(FIXED_FEATURES)
    by_role = _load_eligible_by_role(root)
    validate_population_contract(by_role["TRAIN"], by_role["VALIDATION"], by_role["LOCKED_TEST"])
    universe = build_sample_universe_manifest(root)
    train, train_audit = load_authorized_matrix(repo_root=root, split_role="TRAIN")
    validation, validation_audit = load_authorized_matrix(repo_root=root, split_role="VALIDATION")
    if train.sample_ids != by_role["TRAIN"] or validation.sample_ids != by_role["VALIDATION"]:
        raise CB3Error("Ordered comparison-universe identity mismatch")
    scaler, scaler_evidence = fit_train_only_scaler(
        train, fit_population_fingerprint=universe["ordered_id_list_sha256"]["TRAIN"]
    )
    stored_scaler = predecessor["b2_scaler"]
    if scaler_evidence["scaler_fingerprint"] != stored_scaler.get("scaler_fingerprint"):
        raise PredecessorFingerprintMismatch("C-B2 scaler parity mismatch")
    if scaler_evidence["mean"] != stored_scaler.get("mean") or scaler_evidence["scale"] != stored_scaler.get("scale"):
        raise PredecessorFingerprintMismatch("C-B2 scaler numeric parity mismatch")
    x_train_scaled = np.asarray(scaler.transform(train.features), dtype=np.float64)
    x_validation_scaled = np.asarray(scaler.transform(validation.features), dtype=np.float64)
    plan = build_balanced_oversample_plan(train.labels, train.sample_ids)
    verify_oversample_evidence(train.labels, train.sample_ids, predecessor["b2_sampling"])
    for key in (
        "seed",
        "appended_sequence_sha256",
        "resampled_ordered_sample_ids_sha256",
        "oversampled_class_counts",
        "majority_undersampling_count",
    ):
        if plan.evidence.get(key) != predecessor["b2_sampling"].get(key):
            raise PredecessorFingerprintMismatch(f"C-B2 oversampling parity mismatch: {key}")
    oversampled_ids = [train.sample_ids[int(i)] for i in plan.training_indices.tolist()]
    return PreparedData(
        train=train,
        validation=validation,
        x_train_scaled=x_train_scaled,
        x_validation_scaled=x_validation_scaled,
        scaler_evidence=scaler_evidence,
        oversample_plan=plan,
        universe=universe,
        train_load_audit=train_audit,
        validation_load_audit=validation_audit,
        original_train_fingerprint=universe["ordered_id_list_sha256"]["TRAIN"],
        validation_fingerprint=universe["ordered_id_list_sha256"]["VALIDATION"],
        transformed_train_fingerprint=_array_fingerprint(train.sample_ids, x_train_scaled, train.labels),
        transformed_validation_fingerprint=_array_fingerprint(validation.sample_ids, x_validation_scaled, validation.labels),
        oversampled_train_fingerprint=_ids_fingerprint(oversampled_ids),
    )


def build_fixed_comparison_universe(data: PreparedData) -> Dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "universe_id": "CO2_B3_FIXED_COMPARISON_UNIVERSE_001",
        "train_count": TRAIN_COUNT,
        "validation_count": VALIDATION_COUNT,
        "locked_test_count": LOCKED_TEST_COUNT,
        "locked_test_status": "SEALED",
        "train_ordered_id_sha256": data.original_train_fingerprint,
        "validation_ordered_id_sha256": data.validation_fingerprint,
        "locked_test_membership_fingerprint": data.universe["ordered_id_list_sha256"]["LOCKED_TEST"],
        "overlap_counts": data.universe["overlaps"],
        "warmup_records_outside_comparison_universe": data.universe["canonical_warmup_records"],
        "same_ordered_train_and_validation_identities_for_every_run": True,
        "architecture_specific_sample_population": False,
        "locked_test_feature_matrix_materialized": False,
        "locked_test_target_matrix_materialized": False,
    }


def build_fixed_feature_context() -> Dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "feature_context_id": "CO2_B3_FIXED_FEATURE_CONTEXT_001",
        "feature_order": list(FIXED_FEATURES),
        "feature_count": 4,
        "slope_profile_id": B1_SELECTED_PROFILE_ID,
        "slope_candidate_id": B1_SELECTED_CANDIDATE_ID,
        "slope_method": B1_SELECTED_METHOD,
        "minimum_history_seconds": B1_SELECTED_HISTORY_SECONDS,
        "target_profile_id": TARGET_PROFILE_ID,
        "excluded_features": ["Light", "HumidityRatio", "timestamp", "sample_id", "member_id", "split_role", "provenance_fields"],
        "target_fields_as_features": [],
        "provenance_fields_as_features": [],
        "final_feature_selection_performed": False,
        "architecture_specific_feature_changes": False,
    }


def build_preprocessing_parity(data: PreparedData, predecessor: Mapping[str, Any]) -> Dict[str, Any]:
    stored = predecessor["b2_scaler"]
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "parity_id": "CO2_B3_B2_SCALER_PARITY_001",
        "scaler_profile_id": B2_SCALER_PROFILE_ID,
        "implementation": "sklearn.preprocessing.StandardScaler",
        "fit_population": "ORIGINAL_NATURALLY_DISTRIBUTED_TRAIN_ONLY",
        "fit_sample_count": TRAIN_COUNT,
        "fit_once": True,
        "reused_for_every_architecture_and_seed": True,
        "feature_order": list(FIXED_FEATURES),
        "scaler_fingerprint": data.scaler_evidence["scaler_fingerprint"],
        "expected_c_b2_scaler_fingerprint": stored["scaler_fingerprint"],
        "mean": data.scaler_evidence["mean"],
        "scale": data.scaler_evidence["scale"],
        "variance": data.scaler_evidence["variance"],
        "validation_fit_rows": 0,
        "locked_test_fit_rows": 0,
        "oversampled_fit_rows": 0,
        "production_scaler_modified": False,
        "parity_status": "PASS",
    }


def build_oversampling_parity(data: PreparedData, predecessor: Mapping[str, Any]) -> Dict[str, Any]:
    stored = predecessor["b2_sampling"]
    evidence = data.oversample_plan.evidence
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "parity_id": "CO2_B3_B2_OVERSAMPLING_PARITY_001",
        "policy_id": B2_POLICY_ID,
        "strategy_id": BALANCED_RANDOM_OVERSAMPLE,
        "source_population": "TRAIN_ONLY",
        "source_sample_count": TRAIN_COUNT,
        "original_class_counts": evidence["original_class_counts"],
        "oversampled_class_counts": evidence["oversampled_class_counts"],
        "appended_minority_draw_count": evidence["appended_minority_draw_count"],
        "seed": evidence["seed"],
        "appended_sequence_sha256": evidence["appended_sequence_sha256"],
        "resampled_ordered_sample_ids_sha256": evidence["resampled_ordered_sample_ids_sha256"],
        "expected_c_b2_appended_sequence_sha256": stored["appended_sequence_sha256"],
        "expected_c_b2_resampled_ordered_sample_ids_sha256": stored["resampled_ordered_sample_ids_sha256"],
        "same_oversampled_train_multiset_for_all_architectures_and_seeds": True,
        "majority_undersampling_count": 0,
        "validation_rows_used": 0,
        "locked_test_rows_used": 0,
        "class_weight_stacked": False,
        "parity_status": "PASS",
    }


def _tensorflow() -> Any:
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RequiredBackendUnavailable("C_B3_REQUIRED_BACKEND_UNAVAILABLE: TensorFlow/Keras") from exc
    return tf


def configure_tensorflow(tf: Any) -> Dict[str, Any]:
    op_status = "SUPPORTED_ENABLED"
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as exc:  # noqa: BLE001
        op_status = f"UNAVAILABLE:{type(exc).__name__}"
    thread_status = "SUPPORTED"
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception as exc:  # noqa: BLE001
        thread_status = f"UNAVAILABLE:{type(exc).__name__}"
    return {
        "deterministic_ops": op_status,
        "single_thread_configuration": thread_status,
    }


def _seed_everything(tf: Any, seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)


def _candidate_by_id(registry: Mapping[str, Any], architecture_id: str) -> Mapping[str, Any]:
    for candidate in registry["candidates"]:
        if candidate["architecture_id"] == architecture_id:
            return candidate
    raise CB3Error(f"Unknown architecture: {architecture_id}")


def _fit_architecture(
    candidate: Mapping[str, Any],
    seed: int,
    data: PreparedData,
    tf: Any,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    architecture_id = candidate["architecture_id"]
    indices = data.oversample_plan.training_indices
    x_fit = data.x_train_scaled[indices]
    y_fit = data.train.labels[indices]
    x_val = data.x_validation_scaled
    if x_fit.shape[0] != 12828 or len(set(indices.tolist())) != TRAIN_COUNT:
        raise CB3Error("C-B2 oversampled TRAIN population changed")
    if architecture_id == "LINEAR_LOGISTIC":
        model = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", fit_intercept=True,
            max_iter=2000, class_weight=None, random_state=seed,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            with threadpool_limits(limits=1):
                model.fit(x_fit, y_fit)
        convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
        if convergence:
            raise CB3Error(f"LINEAR_LOGISTIC convergence failure: {convergence}")
        probabilities = np.asarray(model.predict_proba(x_val)[:, 1], dtype=np.float64)
        fit = {
            "fit_status": "SUCCESS",
            "convergence_status": "CONVERGED",
            "n_iter": [int(x) for x in model.n_iter_.tolist()],
            "classes": [int(x) for x in model.classes_.tolist()],
            "coefficient_count": int(model.coef_.size),
            "intercept_count": int(model.intercept_.size),
            "coefficient_feature_order": list(FIXED_FEATURES),
            "random_state": seed,
            "class_weight": None,
        }
        complexity = {"coefficient_count": int(model.coef_.size), "intercept_count": int(model.intercept_.size)}
    elif architecture_id == "TREE_RANDOM_FOREST":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=2,
            max_features="sqrt", bootstrap=True, class_weight=None,
            n_jobs=1, random_state=seed,
        )
        with threadpool_limits(limits=1):
            model.fit(x_fit, y_fit)
        probabilities = np.asarray(model.predict_proba(x_val)[:, 1], dtype=np.float64)
        fit = {
            "fit_status": "SUCCESS",
            "convergence_status": "COMPLETED",
            "tree_count": int(len(model.estimators_)),
            "max_depth": 8,
            "random_state": seed,
            "class_weight": None,
            "n_jobs": 1,
        }
        complexity = {
            "tree_count": int(len(model.estimators_)),
            "depth_configuration": "max_depth=8",
            "estimated_serialized_size_bytes": None,
        }
    elif architecture_id in ("TINY_MLP", "SMALL_MLP"):
        _seed_everything(tf, seed)
        if architecture_id == "TINY_MLP":
            hidden = (8,)
        else:
            hidden = (16, 8)
        model = tf.keras.Sequential(name=f"safenest_{architecture_id.lower()}")
        model.add(tf.keras.layers.Input(shape=(4,), name="input_4"))
        for layer_index, units in enumerate(hidden):
            model.add(tf.keras.layers.Dense(units, activation="relu", name=f"dense_{layer_index + 1}"))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="occupied_probability"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
        )
        history = model.fit(
            np.asarray(x_fit, dtype=np.float32), np.asarray(y_fit, dtype=np.float32),
            batch_size=64, epochs=100, shuffle=True, verbose=0,
        )
        probabilities = np.asarray(
            model.predict(np.asarray(x_val, dtype=np.float32), batch_size=256, verbose=0).reshape(-1),
            dtype=np.float64,
        )
        params = int(model.count_params())
        expected_params = 49 if architecture_id == "TINY_MLP" else 225
        if params != expected_params:
            raise CB3Error(f"{architecture_id} trainable parameter count mismatch")
        fit = {
            "fit_status": "SUCCESS",
            "convergence_status": "COMPLETED_FIXED_100_EPOCHS_NO_EARLY_STOPPING",
            "epochs_completed": 100,
            "batch_size": 64,
            "shuffle": True,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "loss": "binary_crossentropy",
            "trainable_parameter_count": params,
            "final_training_loss": float(history.history["loss"][-1]),
            "random_state": seed,
            "class_weight": None,
        }
        complexity = {"trainable_parameter_count": params}
        tf.keras.backend.clear_session()
    else:
        raise CB3Error(f"Unauthorized architecture: {architecture_id}")
    if probabilities.shape != (VALIDATION_COUNT,) or not np.isfinite(probabilities).all():
        raise CB3Error(f"Invalid VALIDATION probabilities for {architecture_id}/{seed}")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise CB3Error(f"Probability range violation for {architecture_id}/{seed}")
    return probabilities, fit, complexity


def _threshold_and_metrics(data: PreparedData, probabilities: np.ndarray) -> Dict[str, Any]:
    default_metrics, default_predictions = classification_metrics_at_threshold(
        data.validation.labels, probabilities, DEFAULT_THRESHOLD
    )
    threshold_rows, threshold_ranking = build_threshold_sweep(
        y_validation=data.validation.labels,
        probabilities=probabilities,
        sample_ids=data.validation.sample_ids,
        population_role="VALIDATION",
    )
    selected_threshold = float(threshold_ranking[0]["threshold"])
    selected_row = next(row for row in threshold_rows if float(row["threshold"]) == selected_threshold)
    calibrated_metrics, calibrated_predictions = classification_metrics_at_threshold(
        data.validation.labels, probabilities, selected_threshold
    )
    quality = probability_quality_metrics(data.validation.labels, probabilities)
    ece = expected_calibration_error(data.validation.labels, probabilities, n_bins=ECE_BIN_COUNT)
    quality["expected_calibration_error"] = ece["expected_calibration_error"]
    return {
        "default_threshold": DEFAULT_THRESHOLD,
        "default_metrics": default_metrics,
        "default_predictions": default_predictions,
        "selected_threshold": selected_threshold,
        "calibrated_metrics": calibrated_metrics,
        "calibrated_predictions": calibrated_predictions,
        "probability_quality_metrics": quality,
        "ece_diagnostic": ece,
        "threshold_rows": threshold_rows,
        "threshold_ranking": threshold_ranking,
        "probability_vector_sha256": _probability_fingerprint(data.validation.sample_ids, probabilities),
    }


def _run_grid(data: PreparedData, registry: Mapping[str, Any], tf: Any) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    probabilities: Dict[str, np.ndarray] = {}
    sweeps: Dict[str, Any] = {}
    complexity_by_architecture: Dict[str, List[Dict[str, Any]]] = {key: [] for key in ARCHITECTURE_IDS}
    for architecture_id in ARCHITECTURE_IDS:
        candidate = _candidate_by_id(registry, architecture_id)
        for seed in SEEDS:
            run_id = f"{architecture_id}__seed_{seed}"
            probs, fit, complexity = _fit_architecture(candidate, seed, data, tf)
            threshold = _threshold_and_metrics(data, probs)
            probabilities[run_id] = probs
            complexity_by_architecture[architecture_id].append(complexity)
            sweeps[run_id] = {
                "architecture_id": architecture_id,
                "seed": seed,
                "population": "VALIDATION",
                "probability_vector_sha256": threshold["probability_vector_sha256"],
                "probabilities_unchanged_across_thresholds": True,
                "rows": threshold["threshold_rows"],
                "ranking": threshold["threshold_ranking"],
                "locked_test_threshold_evaluations": 0,
            }
            calibrated = threshold["calibrated_metrics"]
            default = threshold["default_metrics"]
            run = {
                "run_id": run_id,
                "architecture_id": architecture_id,
                "architecture_family": candidate["family"],
                "architecture_profile_id": candidate["profile_id"],
                "seed": seed,
                "train_sample_fingerprint": data.original_train_fingerprint,
                "validation_sample_fingerprint": data.validation_fingerprint,
                "transformed_train_fingerprint": data.transformed_train_fingerprint,
                "transformed_validation_fingerprint": data.transformed_validation_fingerprint,
                "oversampling_fingerprint": data.oversampled_train_fingerprint,
                "preprocessing_fingerprint": data.scaler_evidence["scaler_fingerprint"],
                "oversampled_train_sample_count": int(data.oversample_plan.training_indices.size),
                "oversampled_train_unique_original_count": TRAIN_COUNT,
                "fit_population": "C_B2_FIXED_BALANCED_TRAIN_MULT ISET".replace(" ", ""),
                "fit_status": fit["fit_status"],
                "convergence_status": fit["convergence_status"],
                "training_duration_seconds": None,
                "training_duration_policy": "OMITTED_FOR_BIT_IDENTICAL_ARTIFACTS",
                "training_configuration": candidate["parameters"],
                "class_weight": None,
                "default_threshold": DEFAULT_THRESHOLD,
                "default_validation_metrics": default,
                "selected_validation_threshold": threshold["selected_threshold"],
                "calibrated_validation_metrics": calibrated,
                "threshold_protocol_id": B2_THRESHOLD_PROTOCOL_ID,
                "threshold_numeric_inherited_from_b2": False,
                "threshold_selected_on": "VALIDATION_ONLY",
                "threshold_ranking_top": threshold["threshold_ranking"][:3],
                "probability_quality_metrics": threshold["probability_quality_metrics"],
                "ece_diagnostic": threshold["ece_diagnostic"],
                "validation_probability_vector_sha256": threshold["probability_vector_sha256"],
                "complexity": complexity,
                "fit_diagnostics": fit,
                "locked_test_feature_access": 0,
                "locked_test_target_access": 0,
                "locked_test_predictions": 0,
                "locked_test_probabilities": 0,
                "locked_test_metrics": 0,
                "feature_selection_performed": False,
                "architecture_specific_sample_dropping": False,
                "architecture_specific_scaler": False,
                "architecture_specific_imbalance": False,
                "hyperparameter_search_performed": False,
                "early_stopping_used": False,
            }
            runs.append(run)
    if len(runs) != EXPECTED_RUN_COUNT:
        raise CB3Error(f"Expected {EXPECTED_RUN_COUNT} architecture runs, got {len(runs)}")
    runs.sort(key=lambda row: row["run_id"])
    return {
        "runs": runs,
        "probabilities": probabilities,
        "threshold_sweeps": sweeps,
        "complexity_by_architecture": complexity_by_architecture,
    }


def _metric_value(run: Mapping[str, Any], metric: str, calibrated: bool = True) -> float:
    if metric in METRIC_NAMES[:6]:
        block = run["calibrated_validation_metrics" if calibrated else "default_validation_metrics"]
        return float(block[metric])
    return float(run["probability_quality_metrics"][metric])


def summarize_metric(values: Sequence[float], seeds: Sequence[int], *, lower_is_better: bool = False) -> Dict[str, Any]:
    if len(values) != len(seeds) or len(values) < 2:
        raise CB3Error("Multi-seed summary requires at least two values")
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise CB3Error("Non-finite multi-seed metric")
    min_index = int(np.argmin(arr))
    max_index = int(np.argmax(arr))
    worst_index = max_index if lower_is_better else min_index
    best_index = min_index if lower_is_better else max_index
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "min": float(arr[min_index]),
        "max": float(arr[max_index]),
        "min_seed": int(seeds[min_index]),
        "max_seed": int(seeds[max_index]),
        "worst_seed": int(seeds[worst_index]),
        "worst_seed_value": float(arr[worst_index]),
        "best_seed": int(seeds[best_index]),
        "best_seed_value": float(arr[best_index]),
        "sample_standard_deviation": True,
    }


def aggregate_architectures(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_arch = {architecture_id: [] for architecture_id in ARCHITECTURE_IDS}
    for run in runs:
        by_arch[run["architecture_id"]].append(run)
    result: Dict[str, Any] = OrderedDict()
    for architecture_id in ARCHITECTURE_IDS:
        rows = sorted(by_arch[architecture_id], key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in rows] != list(SEEDS):
            raise CB3Error(f"Architecture {architecture_id} does not have exactly five preregistered seeds")
        calibrated: Dict[str, Any] = {}
        default: Dict[str, Any] = {}
        for metric in METRIC_NAMES:
            calibrated[metric] = summarize_metric(
                [_metric_value(row, metric, True) for row in rows],
                [int(row["seed"]) for row in rows],
                lower_is_better=LOWER_IS_BETTER.get(metric, False),
            )
            default[metric] = summarize_metric(
                [_metric_value(row, metric, False) for row in rows],
                [int(row["seed"]) for row in rows],
                lower_is_better=LOWER_IS_BETTER.get(metric, False),
            )
        thresholds = [float(row["selected_validation_threshold"]) for row in rows]
        threshold_summary = summarize_metric(thresholds, [int(row["seed"]) for row in rows])
        result[architecture_id] = {
            "architecture_id": architecture_id,
            "architecture_family": rows[0]["architecture_family"],
            "seed_count": len(rows),
            "seeds": [int(row["seed"]) for row in rows],
            "calibrated_validation_metrics": calibrated,
            "default_validation_metrics": default,
            "threshold_stability": {
                "values_by_seed": {str(row["seed"]): float(row["selected_validation_threshold"]) for row in rows},
                "mean": threshold_summary["mean"],
                "std": threshold_summary["std"],
                "min": threshold_summary["min"],
                "max": threshold_summary["max"],
                "min_seed": threshold_summary["min_seed"],
                "max_seed": threshold_summary["max_seed"],
            },
            "worst_seed_identity_by_macro_f1": calibrated["macro_f1"]["worst_seed"],
            "best_seed_identity_by_macro_f1": calibrated["macro_f1"]["best_seed"],
        }
    return result


def rank_architectures(aggregates: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if set(aggregates) != set(ARCHITECTURE_IDS):
        raise CB3Error("Architecture aggregate coverage mismatch")
    for architecture_id in ARCHITECTURE_IDS:
        if aggregates[architecture_id].get("seed_count") != len(SEEDS):
            raise CB3Error("Architecture aggregate seed count mismatch")

    def key(architecture_id: str) -> Tuple[Any, ...]:
        metrics = aggregates[architecture_id]["calibrated_validation_metrics"]
        return (
            -float(metrics["macro_f1"]["mean"]),
            -float(metrics["macro_f1"]["worst_seed_value"]),
            -float(metrics["recall_occupied"]["mean"]),
            float(metrics["macro_f1"]["std"]),
            -float(metrics["balanced_accuracy"]["mean"]),
            float(metrics["false_positive_rate"]["mean"]),
            ARCHITECTURE_SIMPLICITY[architecture_id],
            architecture_id,
        )

    ordered = sorted(ARCHITECTURE_IDS, key=key)
    rows: List[Dict[str, Any]] = []
    for rank, architecture_id in enumerate(ordered, 1):
        metrics = aggregates[architecture_id]["calibrated_validation_metrics"]
        rows.append(
            {
                "rank": rank,
                "architecture_id": architecture_id,
                "architecture_family": aggregates[architecture_id]["architecture_family"],
                "mean_calibrated_macro_f1": metrics["macro_f1"]["mean"],
                "worst_seed_calibrated_macro_f1": metrics["macro_f1"]["worst_seed_value"],
                "worst_seed_identity": metrics["macro_f1"]["worst_seed"],
                "mean_calibrated_occupied_recall": metrics["recall_occupied"]["mean"],
                "std_calibrated_macro_f1": metrics["macro_f1"]["std"],
                "mean_calibrated_balanced_accuracy": metrics["balanced_accuracy"]["mean"],
                "mean_calibrated_false_positive_rate": metrics["false_positive_rate"]["mean"],
                "simplicity_rank": ARCHITECTURE_SIMPLICITY[architecture_id],
            }
        )
    return rows


def build_validation_predictions(data: PreparedData, grid: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "population": "VALIDATION",
        "sample_count": VALIDATION_COUNT,
        "sample_ids": list(data.validation.sample_ids),
        "labels": [int(x) for x in data.validation.labels.tolist()],
        "runs": {
            run_id: {
                "architecture_id": next(row["architecture_id"] for row in grid["runs"] if row["run_id"] == run_id),
                "seed": next(int(row["seed"]) for row in grid["runs"] if row["run_id"] == run_id),
                "probabilities": [float(x) for x in probabilities.tolist()],
                "probability_vector_sha256": _probability_fingerprint(data.validation.sample_ids, probabilities),
            }
            for run_id, probabilities in sorted(grid["probabilities"].items())
        },
        "locked_test_predictions": 0,
        "locked_test_probabilities": 0,
        "locked_test_metrics": 0,
        "locked_test_feature_matrix_materialized": False,
        "locked_test_target_matrix_materialized": False,
    }


def build_leakage_audit(data: PreparedData, runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "status": "PASS",
        "train_validation_overlap": 0,
        "train_locked_test_overlap": 0,
        "validation_locked_test_overlap": 0,
        "target_as_feature": 0,
        "provenance_as_feature": 0,
        "validation_in_scaler_fit": 0,
        "locked_test_in_scaler_fit": 0,
        "validation_in_oversampling": 0,
        "locked_test_in_oversampling": 0,
        "validation_used_for_model_fitting": 0,
        "locked_test_used_for_model_fitting": 0,
        "locked_test_feature_access": 0,
        "locked_test_target_access": 0,
        "locked_test_predictions": 0,
        "locked_test_probability_outputs": 0,
        "locked_test_metrics": 0,
        "locked_test_threshold_calibration": 0,
        "locked_test_model_selection": 0,
        "locked_test_membership_count_verified": LOCKED_TEST_COUNT,
        "threshold_selected_on": "VALIDATION_ONLY",
        "architecture_specific_sample_dropping": 0,
        "architecture_specific_feature_changes": 0,
        "architecture_specific_scaler": 0,
        "architecture_specific_imbalance_strategy": 0,
        "class_weight_stacked_on_b2_oversampling": 0,
        "architecture_hyperparameter_search": 0,
        "early_stopping_tuning": 0,
        "synthetic_fixture_used_as_real_training_data": False,
        "production_model_modified": False,
        "production_scaler_modified": False,
        "a_series_locked_artifacts_modified": False,
        "b0_predecessor_artifacts_modified": False,
        "b1_predecessor_artifacts_modified": False,
        "b2_predecessor_artifacts_modified": False,
        "same_train_fingerprint_all_runs": len({row["train_sample_fingerprint"] for row in runs}) == 1,
        "same_validation_fingerprint_all_runs": len({row["validation_sample_fingerprint"] for row in runs}) == 1,
        "same_scaler_fingerprint_all_runs": len({row["preprocessing_fingerprint"] for row in runs}) == 1,
        "same_oversampling_fingerprint_all_runs": len({row["oversampling_fingerprint"] for row in runs}) == 1,
        "train_matrix_load_audit": data.train_load_audit,
        "validation_matrix_load_audit": data.validation_load_audit,
    }


def _compare_probabilities(first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    if set(first) != set(second):
        return {"status": "FAIL", "max_absolute_probability_difference": None, "differing_runs": sorted(set(first) ^ set(second))}
    max_diff = 0.0
    differing: List[str] = []
    for run_id in sorted(first):
        diff = float(np.max(np.abs(first[run_id] - second[run_id])))
        max_diff = max(max_diff, diff)
        if diff != 0.0:
            differing.append(run_id)
    return {
        "status": "PASS" if not differing else "FAIL",
        "max_absolute_probability_difference": max_diff,
        "differing_runs": differing,
        "compared_run_count": len(first),
    }


def _backend_environment(tf: Any, tf_config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "tensorflow_version": getattr(tf, "__version__", "unknown"),
        "keras_version": getattr(getattr(tf, "keras", None), "__version__", "unknown"),
        "thread_limit": 1,
        "tensorflow_deterministic_op_status": tf_config["deterministic_ops"],
        "tensorflow_single_thread_status": tf_config["single_thread_configuration"],
        "declared_dependency_contract": "requirements-mac.txt and requirements.txt",
    }


def run_architecture_multiseed(root: Optional[Path] = None, *, verify_repeat: bool = False) -> Dict[str, Any]:
    repo_root = (root or get_repo_root()).resolve()
    predecessor = validate_predecessor_inputs(repo_root)
    pred_registry = build_predecessor_fingerprint_registry(repo_root)
    registry = architecture_registry()
    validate_architecture_registry(registry)
    seeds = seed_registry()
    validate_seed_registry(seeds)
    tf = _tensorflow()
    tf_config = configure_tensorflow(tf)
    data = prepare_fixed_data(repo_root)
    first = _run_grid(data, registry, tf)
    second = _run_grid(data, registry, tf) if verify_repeat else None
    model_repro = (
        _compare_probabilities(first["probabilities"], second["probabilities"])
        if second is not None
        else {"status": "NOT_RUN", "max_absolute_probability_difference": None, "compared_run_count": 0, "differing_runs": []}
    )
    aggregates = aggregate_architectures(first["runs"])
    ranking = rank_architectures(aggregates)
    winner = ranking[0]["architecture_id"]
    predecessor_entries = pred_registry["entries"]
    feature_context = build_fixed_feature_context()
    universe = build_fixed_comparison_universe(data)
    preprocessing = build_preprocessing_parity(data, predecessor)
    oversampling = build_oversampling_parity(data, predecessor)
    predictions = build_validation_predictions(data, first)
    default_vs_calibrated = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "selection_uses": "CALIBRATED_VALIDATION_METRICS",
        "architectures": {
            architecture_id: {
                "default_threshold_multi_seed_results": aggregates[architecture_id]["default_validation_metrics"],
                "calibrated_threshold_multi_seed_results": aggregates[architecture_id]["calibrated_validation_metrics"],
            }
            for architecture_id in ARCHITECTURE_IDS
        },
    }
    threshold_stability = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "protocol_id": B2_THRESHOLD_PROTOCOL_ID,
        "numeric_b2_logistic_threshold_inherited": False,
        "architectures": {
            architecture_id: aggregates[architecture_id]["threshold_stability"]
            for architecture_id in ARCHITECTURE_IDS
        },
    }
    complexity = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "descriptive_only": True,
        "device_latency_claim": False,
        "architectures": {
            architecture_id: {
                "registered": _candidate_by_id(registry, architecture_id)["complexity"],
                "observed_per_seed": first["complexity_by_architecture"][architecture_id],
            }
            for architecture_id in ARCHITECTURE_IDS
        },
    }
    selected = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "profile_id": "CO2_B3_SELECTED_ARCHITECTURE_PROFILE_001",
        "winning_architecture_id": winner,
        "winning_architecture_family": aggregates[winner]["architecture_family"],
        "winning_architecture_configuration": _candidate_by_id(registry, winner),
        "architecture_registry_id": registry["registry_id"],
        "seed_registry_id": seeds["registry_id"],
        "seed_set": list(SEEDS),
        "selected_imbalance_policy_id": B2_POLICY_ID,
        "selected_imbalance_strategy": BALANCED_RANDOM_OVERSAMPLE,
        "selected_slope_profile_id": B1_SELECTED_PROFILE_ID,
        "preprocessing_profile_id": B2_SCALER_PROFILE_ID,
        "comparison_universe_id": universe["universe_id"],
        "comparison_universe_fingerprints": {
            "TRAIN": universe["train_ordered_id_sha256"],
            "VALIDATION": universe["validation_ordered_id_sha256"],
            "LOCKED_TEST": universe["locked_test_membership_fingerprint"],
        },
        "threshold_calibration_protocol_id": B2_THRESHOLD_PROTOCOL_ID,
        "architecture_ranking_rule_id": "CO2_B3_ARCHITECTURE_RANKING_RULE_001",
        "aggregate_metrics": aggregates[winner],
        "ranking": ranking,
        "threshold_stability": aggregates[winner]["threshold_stability"],
        "deployment_status": [
            "OFFLINE_VALIDATION_SELECTED",
            "MULTI_SEED_STABILITY_EVALUATED",
            "LOCKED_TEST_UNTOUCHED",
            "DEVICE_DOMAIN_UNVALIDATED",
            "PRODUCTION_ARTIFACT_NOT_CREATED",
        ],
        "offline_architecture_winner": True,
        "deployment_path_validated": False,
        "production_model": False,
        "production_threshold": False,
        "tflite_conversion": False,
        "int8_quantization": False,
    }
    leakage = build_leakage_audit(data, first["runs"])
    determinism = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "data_pipeline_determinism": "PASS",
        "model_run_reproducibility": model_repro["status"],
        "model_run_reproducibility_detail": model_repro,
        "repeat_execution_requested": bool(verify_repeat),
        "architecture_registry_fingerprint": stable_sha256(registry),
        "seed_registry_fingerprint": stable_sha256(seeds),
        "fixed_universe_fingerprint": stable_sha256(universe),
        "fixed_feature_context_fingerprint": stable_sha256(feature_context),
        "preprocessing_fingerprint": data.scaler_evidence["scaler_fingerprint"],
        "oversampling_fingerprint": data.oversampled_train_fingerprint,
        "selection_rule_fingerprint": stable_sha256(ranking),
        "selected_architecture": winner,
        "checksums_generated_deterministically": True,
        "neural_outputs_bit_identical_claimed": model_repro["status"] == "PASS",
    }
    exceptions = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "blockers": [],
        "warnings": [
            {"code": "DEVICE_UCI_CADENCE_DOMAIN_GAP", "description": "UCI occupancy cadence/domain equivalence to SCD40 is not validated."},
            {"code": "DEPLOYMENT_PATH_NOT_YET_VALIDATED", "description": "C-B3 selects an offline architecture candidate only; no TFLite, INT8, or Pi evidence is produced."},
            {"code": "TRAINING_DURATION_OMITTED_FOR_DETERMINISM", "description": "Wall-clock duration is diagnostic-only and omitted from persisted evidence to keep checksums bit-identical."},
        ],
        "deferred_shared_integration_update": "DEFERRED_SHARED_INTEGRATION_UPDATE",
        "production_model_created": False,
        "production_scaler_modified": False,
        "safety_threshold_calibrated": False,
        "clinical_claim": False,
    }
    generation = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "generator_script": "scripts/audit_co2_architecture_multiseed.py",
        "module": "datasets/co2/architecture_multiseed.py",
        "architecture_comparison_performed": True,
        "multi_seed_comparison_performed": True,
        "architecture_candidate_count": 4,
        "seed_count": 5,
        "expected_architecture_runs": EXPECTED_RUN_COUNT,
        "completed_architecture_runs": len(first["runs"]),
        "locked_test_predictions": 0,
        "locked_test_probabilities": 0,
        "locked_test_metrics": 0,
        "locked_test_threshold_calibration": 0,
        "locked_test_model_selection": 0,
        "feature_selection_performed": False,
        "architecture_hyperparameter_search_performed": False,
        "early_stopping_tuning_performed": False,
        "imbalance_strategy_reselection_performed": False,
        "probability_recalibration_model_fitted": False,
        "production_model_modified": False,
        "production_scaler_modified": False,
        "a_series_locked_artifacts_modified": False,
        "b0_predecessor_artifacts_modified": False,
        "b1_predecessor_artifacts_modified": False,
        "b2_predecessor_artifacts_modified": False,
        "synthetic_npz_used_as_real_training_data": False,
        "data_pipeline_determinism": "PASS",
        "model_run_reproducibility": model_repro["status"],
        "backend": _backend_environment(tf, tf_config),
        "generation_clock_policy": "OMITTED_FOR_BIT_IDENTICAL_RERUNS",
    }
    experiment_contract = {
        "manifest_version": "1.0",
        "schema": "SafeNest_CO2_C_B3_Architecture_MultiSeed_Contract",
        "phase": PHASE_ID,
        "phase_name": PHASE_NAME,
        "experiment_contract_id": "CO2_B3_ARCHITECTURE_MULTI_SEED_CONTRACT_001",
        "immediate_predecessor": "C-B2",
        "c_b2_required_merged_main_commit": CB2_MERGED_MAIN_COMMIT,
        "b0_contract_id": B0_CONTRACT_ID,
        "b1_selected_slope_profile_id": B1_SELECTED_PROFILE_ID,
        "b2_policy_id": B2_POLICY_ID,
        "b2_threshold_protocol_id": B2_THRESHOLD_PROTOCOL_ID,
        "target_profile_id": TARGET_PROFILE_ID,
        "positive_class": POSITIVE_CLASS,
        "negative_class": NEGATIVE_CLASS,
        "train_population": TRAIN_COUNT,
        "validation_population": VALIDATION_COUNT,
        "locked_test_membership_count": LOCKED_TEST_COUNT,
        "locked_test_status": "SEALED",
        "feature_context": list(FIXED_FEATURES),
        "architecture_registry_id": registry["registry_id"],
        "seed_registry_id": seeds["registry_id"],
        "architecture_search": "PREREGISTERED_FIXED_FAMILY_COMPARISON_NO_HYPERPARAMETER_SEARCH",
        "threshold_calibration": "B2_PROTOCOL_VALIDATION_ONLY_PER_ARCHITECTURE_SEED",
        "numeric_b2_logistic_threshold_inherited": False,
        "selection_metric": "CALIBRATED_VALIDATION_MACRO_F1_WITH_PREREGISTERED_TIE_BREAKS",
        "locked_test_predictive_evaluation": False,
        "production_model_promotion": False,
        "tflite_conversion": False,
        "int8_quantization": False,
    }
    artifacts = {
        "predecessor_fingerprint_registry.json": pred_registry,
        "experiment_contract.json": experiment_contract,
        "architecture_candidate_registry.json": registry,
        "seed_registry.json": seeds,
        "fixed_comparison_universe_fingerprint.json": universe,
        "fixed_feature_context_fingerprint.json": feature_context,
        "preprocessing_parity_evidence.json": preprocessing,
        "oversampling_parity_evidence.json": oversampling,
        "per_run_results.json": {"manifest_version": "1.0", "phase": PHASE_ID, "run_count": len(first["runs"]), "runs": first["runs"]},
        "validation_predictions.json": predictions,
        "threshold_sweep_results.json": {"manifest_version": "1.0", "phase": PHASE_ID, "protocol_id": B2_THRESHOLD_PROTOCOL_ID, "run_count": len(first["threshold_sweeps"]), "runs": first["threshold_sweeps"]},
        "architecture_multiseed_aggregate.json": {"manifest_version": "1.0", "phase": PHASE_ID, "architectures": aggregates},
        "default_vs_calibrated_comparison.json": default_vs_calibrated,
        "threshold_stability_summary.json": threshold_stability,
        "architecture_complexity_summary.json": complexity,
        "architecture_ranking.json": {"manifest_version": "1.0", "phase": PHASE_ID, "ranking_rule_id": "CO2_B3_ARCHITECTURE_RANKING_RULE_001", "ranking_rule": ["higher_mean_calibrated_macro_f1", "higher_worst_seed_calibrated_macro_f1", "higher_mean_calibrated_occupied_recall", "lower_std_calibrated_macro_f1", "higher_mean_calibrated_balanced_accuracy", "lower_mean_false_positive_rate", "simpler_architecture_LINEAR_THEN_TINY_THEN_SMALL_THEN_TREE", "lexicographically_smaller_architecture_id"], "ranking": ranking, "selected_architecture": winner},
        "selected_architecture_profile.json": selected,
        "leakage_audit.json": leakage,
        "determinism_report.json": determinism,
        "exceptions_and_limitations.json": exceptions,
        "generation_metadata.json": generation,
    }
    output_dir = repo_root / ARTIFACT_DIR_REL
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in sorted(artifacts.items()):
        path = output_dir / filename
        write_json(path, payload)
        forbidden = assert_no_forbidden_path_markers(path.read_text(encoding="utf-8"))
        if forbidden:
            raise CB3Error(f"Forbidden path marker in {filename}: {forbidden}")
    artifact_names = sorted(artifacts)
    identity = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "artifact_namespace": ARTIFACT_DIR_REL,
        "artifact_json_count": len(artifact_names) + 1,
        "artifact_json_files": artifact_names + ["artifact_identity.json"],
        "architecture_candidate_count": 4,
        "seed_count": 5,
        "expected_architecture_runs": EXPECTED_RUN_COUNT,
        "completed_architecture_runs": len(first["runs"]),
        "selected_architecture": winner,
        "selected_architecture_profile_id": selected["profile_id"],
        "locked_test_predictions": 0,
        "locked_test_probabilities": 0,
        "locked_test_metrics": 0,
        "raw_payload_included": False,
        "production_model_created": False,
        "production_scaler_modified": False,
    }
    write_json(output_dir / "artifact_identity.json", identity)
    checksum_lines = []
    for filename in identity["artifact_json_files"]:
        path = output_dir / filename
        checksum_lines.append(f"{compute_sha256_file(path)}  {ARTIFACT_DIR_REL}/{filename}")
    (output_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return {
        "artifact_dir": ARTIFACT_DIR_REL,
        "selected_architecture": winner,
        "architecture_ranking": ranking,
        "completed_architecture_runs": len(first["runs"]),
        "model_run_reproducibility": model_repro["status"],
        "backend": generation["backend"],
        "aggregate": aggregates,
        "checksums": {filename: compute_sha256_file(output_dir / filename) for filename in identity["artifact_json_files"]},
        "predecessor_entry_count": len(predecessor_entries),
    }


def load_c_b3_artifacts(root: Optional[Path] = None) -> Dict[str, Any]:
    directory = (root or get_repo_root()) / ARTIFACT_DIR_REL
    return {path.name: load_json(path) for path in sorted(directory.glob("*.json")) if path.is_file()}


__all__ = [
    "ARCHITECTURE_IDS",
    "ARTIFACT_DIR_REL",
    "BALANCED_RANDOM_OVERSAMPLE",
    "B2_POLICY_ID",
    "B2_SCALER_PROFILE_ID",
    "B2_THRESHOLD_PROTOCOL_ID",
    "CB2_MERGED_MAIN_COMMIT",
    "CB3Error",
    "EXPECTED_RUN_COUNT",
    "FIXED_FEATURES",
    "LOCKED_TEST_COUNT",
    "LockedTestPolicyViolation",
    "METRIC_NAMES",
    "PredecessorFingerprintMismatch",
    "SEEDS",
    "TRAIN_COUNT",
    "VALIDATION_COUNT",
    "aggregate_architectures",
    "architecture_registry",
    "build_predecessor_fingerprint_registry",
    "load_c_b3_artifacts",
    "rank_architectures",
    "run_architecture_multiseed",
    "seed_registry",
    "stable_sha256",
    "summarize_metric",
    "validate_architecture_registry",
    "validate_feature_context",
    "validate_locked_test_access",
    "validate_predecessor_inputs",
    "validate_seed_registry",
    "verify_stored_predecessor_registry",
]
