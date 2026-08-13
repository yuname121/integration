#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase C-B2 controlled class imbalance and threshold calibration.

This module deliberately fixes the feature context, preprocessing, and
probabilistic classifier so C-B2 varies only the imbalance intervention and,
after Stage 1 selection, the probability-to-class decision threshold.
LOCKED_TEST membership may be checked, but its feature/target rows are never
decoded, transformed, predicted, scored, or used for tuning here.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from datasets.co2.canonical_samples import make_canonical_sample_id
from datasets.co2.offline_experiment import (
    EXPECTED_LOCKED_TEST_SEALED,
    EXPECTED_TRAIN_COMMON,
    EXPECTED_VALIDATION_COMMON,
    EXPECTED_WARMUP_CANONICAL,
    MatrixBundle,
    _load_eligible_by_role,
    assert_no_forbidden_path_markers,
    build_sample_universe_manifest,
    compute_classification_metrics,
    ordered_id_list_sha256,
    verify_a_series_artifact_lock,
    verify_a_series_release,
)
from datasets.co2.raw_reader import (
    EXPECTED_ARCHIVE_REL_PATH,
    UCIOccupancyRawReader,
    compute_sha256_file,
    get_repo_root,
)
from datasets.co2.slope_feature import (
    STATUS_AVAILABLE,
    reconstruct_all_slope_features_with_params,
)


PHASE_ID = "C-B2"
PHASE_NAME = "CO2_CONTROLLED_IMBALANCE_AND_THRESHOLD_CALIBRATION"
ARTIFACT_DIR_REL = "datasets/co2/manifests/c_b2_imbalance_calibration"
B0_DIR_REL = "datasets/co2/manifests/c_b0_offline_experiment_contract"
B1_DIR_REL = "datasets/co2/manifests/c_b1_slope_method_history_ablation"

EXPERIMENT_CONTRACT_ID = "CO2_B2_IMBALANCE_CALIBRATION_CONTRACT_001"
FEATURE_CONTEXT_ID = "CO2_B2_FIXED_EXPERIMENT_FEATURE_CONTEXT_001"
SCALER_PROFILE_ID = "CO2_B2_TRAIN_ONLY_STANDARD_SCALER_001"
STRATEGY_REGISTRY_ID = "CO2_B2_IMBALANCE_STRATEGY_REGISTRY_001"
SELECTED_IMBALANCE_POLICY_ID = "CO2_B2_SELECTED_IMBALANCE_POLICY_001"
PROBE_PROFILE_ID = "B2_FIXED_LOGISTIC_PROBE_001"
CALIBRATION_PROTOCOL_ID = "CO2_B2_THRESHOLD_CALIBRATION_PROTOCOL_001"
REFERENCE_THRESHOLD_RESULT_ID = "CO2_B2_REFERENCE_THRESHOLD_RESULT_001"
PROBABILITY_SEMANTIC_CONTRACT_ID = "CO2_B2_OCCUPANCY_PROBABILITY_SEMANTIC_001"

B0_CONTRACT_ID = "CO2_B0_OFFLINE_EXPERIMENT_CONTRACT_001"
B1_SELECTED_PROFILE_ID = "CO2_B1_SELECTED_SLOPE_PROFILE_001"
B1_SELECTED_CANDIDATE_ID = "ENDPOINT_H150"
B1_SELECTED_METHOD = "ENDPOINT_DIFFERENCE"
B1_SELECTED_HISTORY_SECONDS = 150.0
B1_MERGED_MAIN_COMMIT = "d549cc890f40d9f0398ed346b744533625fdd775"
A3_SLOPE_PROFILE_ID = "CO2_SLOPE_FEATURE_PROFILE_001"
TARGET_PROFILE_ID = "CO2_OCCUPANCY_TARGET_PROFILE_001"
A_SERIES_RELEASE_TAG = "co2-a-series-raw-to-canonical"
A_SERIES_RELEASE_TARGET = "bfd860cad2bb8dafe35ef7600cfa931d7d2d554d"

DEFAULT_SEED = 20260810
DEFAULT_THRESHOLD = 0.50
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01
THRESHOLD_COUNT = 91
ECE_BIN_COUNT = 10

FIXED_FEATURES = ("CO2", "Temperature", "Humidity", "CO2_slope")
SOURCE_MEMBER_BY_ROLE = {
    "TRAIN": "datatraining.txt",
    "VALIDATION": "datatest.txt",
}
SEALED_LOCKED_TEST_MEMBER = "datatest2.txt"
NATURAL_DISTRIBUTION = "NATURAL_DISTRIBUTION"
CLASS_WEIGHT_BALANCED = "CLASS_WEIGHT_BALANCED"
BALANCED_RANDOM_OVERSAMPLE = "BALANCED_RANDOM_OVERSAMPLE"
AUTHORIZED_STRATEGIES = (
    NATURAL_DISTRIBUTION,
    CLASS_WEIGHT_BALANCED,
    BALANCED_RANDOM_OVERSAMPLE,
)
STRATEGY_SIMPLICITY = {
    NATURAL_DISTRIBUTION: 0,
    CLASS_WEIGHT_BALANCED: 1,
    BALANCED_RANDOM_OVERSAMPLE: 2,
}

PRODUCTION_SCALER_REL = "models/co2/co2_scaling_metadata_v0.1.0.json"
PRODUCTION_MODEL_REL = "models/co2/co2_occupancy_int8_v0.1.0.tflite"
SYNTHETIC_FIXTURE_REL = "datasets/co2/processed/co2_occupancy_v1.npz"

FIXED_LOGISTIC_PARAMETERS: Dict[str, Any] = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "fit_intercept": True,
    "max_iter": 2000,
    "random_state": DEFAULT_SEED,
}

B0_PREDECESSOR_FILES: Tuple[Tuple[str, str], ...] = (
    ("experiment_contract.json", "B0_EXPERIMENT_CONTRACT"),
    ("sample_universe_manifest.json", "B0_COMPARISON_UNIVERSE"),
    ("metric_contract.json", "B0_METRIC_CONTRACT"),
    ("feature_view_registry.json", "B0_FEATURE_VIEW_REGISTRY"),
    ("leakage_audit.json", "B0_LEAKAGE_POLICY"),
)

B1_PREDECESSOR_FILES: Tuple[Tuple[str, str], ...] = (
    ("predecessor_fingerprint_registry.json", "B1_PREDECESSOR_CLOSURE"),
    ("candidate_registry.json", "B1_SLOPE_CANDIDATE_REGISTRY"),
    ("selected_slope_profile.json", "B1_SELECTED_SLOPE_PROFILE"),
    ("selection_decision.json", "B1_CANDIDATE_SELECTION_RESULT"),
    ("validation_metric_results.json", "B1_CANDIDATE_RESULTS"),
    ("candidate_feature_fingerprint_registry.json", "B1_FEATURE_FINGERPRINTS"),
    ("leakage_audit.json", "B1_LEAKAGE_EVIDENCE"),
)


class CB2Error(RuntimeError):
    """Base C-B2 contract error."""


class LockedTestPolicyViolation(CB2Error):
    """Raised for any attempted predictive LOCKED_TEST access."""


class PredecessorFingerprintMismatch(CB2Error):
    """Raised when required C-B0/C-B1 evidence is missing or drifts."""


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_json_sha256(payload: Any) -> str:
    blob = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _probability_fingerprint(sample_ids: Sequence[str], probabilities: np.ndarray) -> str:
    arr = np.asarray(probabilities, dtype="<f8")
    if arr.ndim != 1 or arr.size != len(sample_ids):
        raise CB2Error("Probability fingerprint shape mismatch")
    h = hashlib.sha256()
    h.update(("\n".join(sample_ids) + "\n").encode("utf-8"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _verify_checksum_manifest(repo_root: Path, directory_rel: str) -> None:
    checksum_path = repo_root / directory_rel / "checksums.sha256"
    if not checksum_path.is_file():
        raise PredecessorFingerprintMismatch(f"Missing checksum manifest: {checksum_path}")
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, rel = line.split("  ", 1)
        except ValueError as exc:
            raise PredecessorFingerprintMismatch(
                f"Malformed checksum row in {directory_rel}: {line!r}"
            ) from exc
        path = repo_root / rel
        if not path.is_file() or compute_sha256_file(path) != digest:
            raise PredecessorFingerprintMismatch(
                f"C_B0_OR_B1_PREDECESSOR_FINGERPRINT_MISMATCH: {rel}"
            )


def validate_b1_selected_profile(profile: Mapping[str, Any]) -> None:
    checks = {
        "profile_id": B1_SELECTED_PROFILE_ID,
        "selected_candidate_id": B1_SELECTED_CANDIDATE_ID,
        "method": B1_SELECTED_METHOD,
        "minimum_history_seconds": B1_SELECTED_HISTORY_SECONDS,
        "locked_test_status": "SEALED",
    }
    for key, expected in checks.items():
        if profile.get(key) != expected:
            raise PredecessorFingerprintMismatch(
                f"B1_SELECTED_SLOPE_DRIFT: {key}={profile.get(key)!r}, expected={expected!r}"
            )
    if profile.get("a_series_baseline_retained") is not True:
        raise PredecessorFingerprintMismatch("B1 selected slope lost A-series parity")
    if profile.get("final_feature_selection") != "NOT_PERFORMED":
        raise PredecessorFingerprintMismatch("B1 final-feature boundary drift")


def _validate_predecessor_documents(repo_root: Path) -> None:
    _verify_checksum_manifest(repo_root, B0_DIR_REL)
    _verify_checksum_manifest(repo_root, B1_DIR_REL)

    b0_contract = load_json(repo_root / B0_DIR_REL / "experiment_contract.json")
    if b0_contract.get("experiment_contract_id") != B0_CONTRACT_ID:
        raise PredecessorFingerprintMismatch("C-B0 experiment contract identity drift")
    universe = load_json(repo_root / B0_DIR_REL / "sample_universe_manifest.json")
    if (
        universe.get("b_series_common_train") != EXPECTED_TRAIN_COMMON
        or universe.get("b_series_common_validation") != EXPECTED_VALIDATION_COMMON
        or universe.get("b_series_sealed_locked_test") != EXPECTED_LOCKED_TEST_SEALED
        or universe.get("canonical_warmup_records") != EXPECTED_WARMUP_CANONICAL
    ):
        raise PredecessorFingerprintMismatch("C-B0 comparison universe drift")

    selected = load_json(repo_root / B1_DIR_REL / "selected_slope_profile.json")
    validate_b1_selected_profile(selected)
    decision = load_json(repo_root / B1_DIR_REL / "selection_decision.json")
    parity = decision.get("endpoint_h150_a3_parity") or {}
    if parity.get("status") != "PASS" or parity.get("checked_samples") != 10802:
        raise PredecessorFingerprintMismatch("B1 ENDPOINT_H150 parity evidence invalid")
    if decision.get("winning_slope_candidate") != B1_SELECTED_CANDIDATE_ID:
        raise PredecessorFingerprintMismatch("B1 selection decision drift")
    leakage = load_json(repo_root / B1_DIR_REL / "leakage_audit.json")
    if leakage.get("status") != "PASS":
        raise PredecessorFingerprintMismatch("B1 leakage predecessor is not PASS")


def build_predecessor_fingerprint_registry(repo_root: Path) -> Dict[str, Any]:
    """Fingerprint every C-B0/C-B1 artifact C-B2 consumes."""
    _validate_predecessor_documents(repo_root)
    entries: List[Dict[str, Any]] = []
    for directory, phase, specs in (
        (B0_DIR_REL, "C-B0", B0_PREDECESSOR_FILES),
        (B1_DIR_REL, "C-B1", B1_PREDECESSOR_FILES),
    ):
        for filename, role in specs:
            rel = f"{directory}/{filename}"
            path = repo_root / rel
            if not path.is_file():
                raise PredecessorFingerprintMismatch(f"Missing predecessor: {rel}")
            entries.append(
                {
                    "path": rel,
                    "byte_size": path.stat().st_size,
                    "sha256": compute_sha256_file(path),
                    "role": role,
                    "phase": phase,
                }
            )

    protected = (
        (
            "datasets/co2/manifests/c_a6_final_integrity_lock/artifact_lock_manifest.json",
            "A_SERIES_ARTIFACT_LOCK",
            "C-A6",
        ),
        (PRODUCTION_SCALER_REL, "PRODUCTION_SCALER_READ_ONLY", "PREEXISTING_PRODUCTION"),
        (PRODUCTION_MODEL_REL, "PRODUCTION_MODEL_READ_ONLY", "PREEXISTING_PRODUCTION"),
        (SYNTHETIC_FIXTURE_REL, "SYNTHETIC_SMOKE_FIXTURE_EXCLUDED", "PREEXISTING_FIXTURE"),
    )
    for rel, role, phase in protected:
        path = repo_root / rel
        if not path.is_file():
            raise PredecessorFingerprintMismatch(f"Missing protected evidence: {rel}")
        entries.append(
            {
                "path": rel,
                "byte_size": path.stat().st_size,
                "sha256": compute_sha256_file(path),
                "role": role,
                "phase": phase,
            }
        )

    entries.sort(key=lambda row: row["path"])
    return {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "registry_id": "CO2_B2_PREDECESSOR_FINGERPRINT_REGISTRY_001",
        "required_b1_merged_main_commit": B1_MERGED_MAIN_COMMIT,
        "a_series_release_tag": A_SERIES_RELEASE_TAG,
        "a_series_release_target": A_SERIES_RELEASE_TARGET,
        "b0_contract_id": B0_CONTRACT_ID,
        "b1_selected_slope_profile_id": B1_SELECTED_PROFILE_ID,
        "entry_count": len(entries),
        "entries": entries,
        "closure_fingerprint": _stable_json_sha256(entries),
        "closure_status": "LOCKED",
        "mismatch_status": "C_B0_OR_B1_PREDECESSOR_FINGERPRINT_MISMATCH_ON_DRIFT",
    }


def verify_stored_predecessor_registry(
    repo_root: Path, stored: Mapping[str, Any]
) -> None:
    live = build_predecessor_fingerprint_registry(repo_root)
    if dict(stored) != live:
        raise PredecessorFingerprintMismatch(
            "C_B0_OR_B1_PREDECESSOR_FINGERPRINT_MISMATCH"
        )


def validate_population_contract(
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    locked_test_ids: Sequence[str],
) -> Dict[str, int]:
    if len(train_ids) != EXPECTED_TRAIN_COMMON:
        raise CB2Error(f"TRAIN count {len(train_ids)} != {EXPECTED_TRAIN_COMMON}")
    if len(validation_ids) != EXPECTED_VALIDATION_COMMON:
        raise CB2Error(
            f"VALIDATION count {len(validation_ids)} != {EXPECTED_VALIDATION_COMMON}"
        )
    if len(locked_test_ids) != EXPECTED_LOCKED_TEST_SEALED:
        raise CB2Error(
            f"LOCKED_TEST count {len(locked_test_ids)} != {EXPECTED_LOCKED_TEST_SEALED}"
        )
    overlap = {
        "train_validation": len(set(train_ids) & set(validation_ids)),
        "train_locked_test": len(set(train_ids) & set(locked_test_ids)),
        "validation_locked_test": len(set(validation_ids) & set(locked_test_ids)),
    }
    if any(overlap.values()):
        raise CB2Error(f"Cross-split overlap: {overlap}")
    if any(len(ids) != len(set(ids)) for ids in (train_ids, validation_ids, locked_test_ids)):
        raise CB2Error("Duplicate sample IDs within split")
    return overlap


def _feature_value(row: Mapping[str, Any], feature: str) -> float:
    field = {
        "CO2": "co2",
        "Temperature": "temperature",
        "Humidity": "humidity",
        "CO2_slope": "co2_slope",
    }.get(feature)
    if field is None:
        raise CB2Error(f"Unauthorized C-B2 feature: {feature}")
    value = row.get(field)
    if value is None or not math.isfinite(float(value)):
        raise CB2Error(f"Missing/non-finite {feature} for {row.get('canonical_sample_id')}")
    return float(value)


def validate_feature_context(features: Sequence[str]) -> None:
    if tuple(features) != FIXED_FEATURES:
        raise CB2Error(
            f"C-B2 feature context must be exactly {list(FIXED_FEATURES)}, got {list(features)}"
        )


def load_authorized_matrix(
    *,
    repo_root: Path,
    split_role: str,
    feature_names: Sequence[str] = FIXED_FEATURES,
) -> Tuple[MatrixBundle, Dict[str, Any]]:
    """Decode exactly one authorized raw member for TRAIN or VALIDATION.

    C-A2 fixes one whole UCI source member to each split. This loader asks the
    safe raw reader for only ``datatraining.txt`` or ``datatest.txt`` and
    reconstructs the B1-selected ENDPOINT_H150 slope for that member. The
    sealed ``datatest2.txt`` payload is never decompressed, parsed, or decoded.
    Hashing the enclosing archive is identity verification, not predictive
    access to the sealed member.
    """
    validate_feature_context(feature_names)
    if split_role not in ("TRAIN", "VALIDATION"):
        raise LockedTestPolicyViolation(
            "LOCKED_TEST_POLICY_VIOLATION: C-B2 predictive matrix access is limited to "
            "TRAIN and VALIDATION"
        )

    by_role = _load_eligible_by_role(repo_root)
    validate_population_contract(
        by_role["TRAIN"], by_role["VALIDATION"], by_role["LOCKED_TEST"]
    )
    ordered_ids = list(by_role[split_role])
    wanted = set(ordered_ids)
    rows: Dict[str, Tuple[List[float], int]] = {}
    member_name = SOURCE_MEMBER_BY_ROLE[split_role]
    reader = UCIOccupancyRawReader(repo_root=repo_root)
    observations = reader.read_all_observations(target_member=member_name)
    slopes = reconstruct_all_slope_features_with_params(
        observations,
        method=B1_SELECTED_METHOD,
        history_duration_seconds=B1_SELECTED_HISTORY_SECONDS,
        feature_contract_id=B1_SELECTED_PROFILE_ID,
    )
    if len(observations) != len(slopes):
        raise CB2Error("Authorized observation/slope reconstruction length mismatch")
    warmup_rows_excluded = 0
    for observation, slope in zip(observations, slopes):
        if observation.source_member_name != member_name:
            raise CB2Error("Safe reader returned a nonrequested source member")
        if slope.future_split_role != split_role:
            raise CB2Error("C-A2 split role mismatch during authorized reconstruction")
        sid = make_canonical_sample_id(observation)
        if slope.feature_status != STATUS_AVAILABLE or slope.co2_slope is None:
            warmup_rows_excluded += 1
            if sid in wanted:
                raise CB2Error(f"Selected slope unavailable for eligible sample {sid}")
            continue
        if sid not in wanted:
            raise CB2Error(f"Available authorized sample missing from B0 universe: {sid}")
        row = {
            "co2": observation.co2,
            "temperature": observation.temperature,
            "humidity": observation.humidity,
            "co2_slope": slope.co2_slope,
            "canonical_sample_id": sid,
        }
        feature_vector = [_feature_value(row, name) for name in feature_names]
        label = int(observation.occupancy)
        if label not in (0, 1):
            raise CB2Error(f"Unexpected occupancy label for {sid}: {label}")
        rows[sid] = (feature_vector, label)

    missing = [sid for sid in ordered_ids if sid not in rows]
    if missing:
        raise CB2Error(f"Authorized matrix materialization incomplete: {missing[:5]}")
    features = np.asarray([rows[sid][0] for sid in ordered_ids], dtype=np.float64)
    labels = np.asarray([rows[sid][1] for sid in ordered_ids], dtype=np.int64)
    if not np.isfinite(features).all():
        raise CB2Error("Non-finite C-B2 feature matrix")
    bundle = MatrixBundle(
        sample_ids=ordered_ids,
        features=features,
        labels=labels,
        feature_names=tuple(feature_names),
        split_role=split_role,
    )
    audit = {
        "requested_split": split_role,
        "source_archive": EXPECTED_ARCHIVE_REL_PATH,
        "archive_identity_hash_verification_only": True,
        "source_member_requested_and_decoded": member_name,
        "source_raw_rows_decoded": len(observations),
        "decoded_authorized_rows": len(ordered_ids),
        "authorized_warmup_rows_excluded": warmup_rows_excluded,
        "locked_test_member": SEALED_LOCKED_TEST_MEMBER,
        "locked_test_member_decompressed": False,
        "locked_test_member_parsed": False,
        "locked_test_feature_rows_decoded": 0,
        "locked_test_target_rows_decoded": 0,
        "canonical_sample_ids_recomputed_from_released_source_identity": True,
        "b1_selected_slope_reconstructed": B1_SELECTED_CANDIDATE_ID,
    }
    return bundle, audit


def build_imbalance_strategy_registry() -> Dict[str, Any]:
    strategies = [
        {
            "strategy_id": NATURAL_DISTRIBUTION,
            "training_population": "ORIGINAL_TRAIN_8140",
            "class_weight": "NONE",
            "sampling": "NONE",
            "participates_in_stage1_ranking": True,
        },
        {
            "strategy_id": CLASS_WEIGHT_BALANCED,
            "training_population": "ALL_ORIGINAL_TRAIN_8140",
            "class_weight": "EXPLICIT_BALANCED_FROM_TRAIN_ONLY",
            "sampling": "NONE",
            "participates_in_stage1_ranking": True,
        },
        {
            "strategy_id": BALANCED_RANDOM_OVERSAMPLE,
            "training_population": "ORIGINAL_TRAIN_PLUS_TRAIN_MINORITY_REPLACEMENT_DRAWS",
            "class_weight": "NONE",
            "sampling": "MINORITY_RANDOM_OVERSAMPLE_WITH_REPLACEMENT",
            "seed": DEFAULT_SEED,
            "participates_in_stage1_ranking": True,
        },
    ]
    registry = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "registry_id": STRATEGY_REGISTRY_ID,
        "registration_timing": "BEFORE_VALIDATION_EVALUATION",
        "strategy_count": len(strategies),
        "authorized_strategy_ids": list(AUTHORIZED_STRATEGIES),
        "strategies": strategies,
        "unauthorized_examples": [
            "SMOTE",
            "ADASYN",
            "MAJORITY_UNDERSAMPLING",
            "FOCAL_LOSS",
            "CUSTOM_ASYMMETRIC_LOSS",
            "MANUAL_DUPLICATION_RATIO",
            "COST_SENSITIVE_THRESHOLD",
        ],
        "stage1_threshold": DEFAULT_THRESHOLD,
        "ranking_rule": [
            "higher_validation_macro_f1",
            "higher_validation_occupied_recall",
            "higher_validation_balanced_accuracy",
            "lower_validation_false_positive_rate",
            "simpler_intervention_NATURAL_then_CLASS_WEIGHT_then_OVERSAMPLE",
            "lexicographically_smaller_strategy_id",
        ],
    }
    registry["registry_fingerprint"] = _stable_json_sha256(strategies)
    return registry


def validate_imbalance_registry(registry: Mapping[str, Any]) -> None:
    ids = [row.get("strategy_id") for row in registry.get("strategies", [])]
    if registry.get("strategy_count") != 3 or tuple(ids) != AUTHORIZED_STRATEGIES:
        raise CB2Error(f"Unauthorized or incomplete imbalance strategy registry: {ids}")
    if float(registry.get("stage1_threshold", -1.0)) != DEFAULT_THRESHOLD:
        raise CB2Error("Stage-1 threshold must be exactly 0.5")


def compute_balanced_class_weights(
    labels: np.ndarray,
    *,
    derivation_role: str = "TRAIN",
) -> Dict[int, float]:
    if derivation_role != "TRAIN":
        raise LockedTestPolicyViolation(
            f"Class weights must derive from TRAIN only, not {derivation_role}"
        )
    arr = np.asarray(labels, dtype=np.int64)
    counts = Counter(int(x) for x in arr.tolist())
    if set(counts) != {0, 1}:
        raise CB2Error(f"Expected both binary classes in TRAIN: {counts}")
    total = int(arr.size)
    return {
        0: float(total / (2 * counts[0])),
        1: float(total / (2 * counts[1])),
    }


def validate_class_weight_evidence(
    labels: np.ndarray, evidence: Mapping[str, Any]
) -> None:
    expected = compute_balanced_class_weights(labels, derivation_role="TRAIN")
    expected_named = {"VACANT": expected[0], "OCCUPIED": expected[1]}
    if evidence.get("derivation_population") != "TRAIN_ONLY":
        raise CB2Error("Class-weight derivation population is not TRAIN_ONLY")
    if evidence.get("explicit_class_weights") != expected_named:
        raise CB2Error("Incorrect class-weight calculation")
    if evidence.get("validation_rows_used") != 0:
        raise LockedTestPolicyViolation("VALIDATION-derived class weights rejected")
    if evidence.get("locked_test_rows_used") != 0:
        raise LockedTestPolicyViolation("LOCKED_TEST-derived class weights rejected")


@dataclass(frozen=True)
class OversamplePlan:
    training_indices: np.ndarray
    appended_indices: np.ndarray
    evidence: Dict[str, Any]


def build_balanced_oversample_plan(
    labels: np.ndarray,
    sample_ids: Sequence[str],
    *,
    seed: int = DEFAULT_SEED,
    source_role: str = "TRAIN",
) -> OversamplePlan:
    if source_role != "TRAIN":
        raise LockedTestPolicyViolation(
            f"Balanced sampling must use TRAIN only, not {source_role}"
        )
    if seed != DEFAULT_SEED:
        raise CB2Error(f"Oversampling seed drift: {seed} != {DEFAULT_SEED}")
    arr = np.asarray(labels, dtype=np.int64)
    if arr.ndim != 1 or arr.size != len(sample_ids):
        raise CB2Error("Oversampling label/sample-ID shape mismatch")
    counts = Counter(int(x) for x in arr.tolist())
    if set(counts) != {0, 1} or counts[0] == counts[1]:
        raise CB2Error(f"Oversampling expects two imbalanced TRAIN classes: {counts}")
    majority_label = max(counts, key=lambda label: (counts[label], -label))
    minority_label = 1 - majority_label
    majority_count = counts[majority_label]
    minority_indices = np.flatnonzero(arr == minority_label)
    deficit = majority_count - counts[minority_label]
    rng = np.random.default_rng(seed)
    appended = np.asarray(
        rng.choice(minority_indices, size=deficit, replace=True), dtype=np.int64
    )
    original = np.arange(arr.size, dtype=np.int64)
    training_indices = np.concatenate([original, appended])
    post_counts = Counter(int(x) for x in arr[training_indices].tolist())
    if post_counts[0] != post_counts[1]:
        raise CB2Error(f"Oversampling failed to balance classes: {post_counts}")
    if int(np.sum(arr[training_indices] == majority_label)) != majority_count:
        raise CB2Error("Majority undersampling detected")

    appended_ids = [sample_ids[int(i)] for i in appended.tolist()]
    multiplicity = Counter(sample_ids[int(i)] for i in training_indices.tolist())
    minority_multiplicity = [
        {"sample_id": sid, "multiplicity": multiplicity[sid]}
        for sid in sorted(sample_ids[i] for i in minority_indices.tolist())
    ]
    training_ids = [sample_ids[int(i)] for i in training_indices.tolist()]
    evidence = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "strategy_id": BALANCED_RANDOM_OVERSAMPLE,
        "source_population": "TRAIN_ONLY",
        "source_sample_count": int(arr.size),
        "source_unique_sample_count": len(set(sample_ids)),
        "seed": seed,
        "rng": "numpy.random.Generator(PCG64)",
        "method": "MINORITY_RANDOM_OVERSAMPLE_WITH_REPLACEMENT",
        "original_order_retained_first": True,
        "majority_label": majority_label,
        "minority_label": minority_label,
        "original_class_counts": {"VACANT": counts[0], "OCCUPIED": counts[1]},
        "appended_minority_draw_count": deficit,
        "appended_minority_sample_ids": appended_ids,
        "appended_sequence_sha256": ordered_id_list_sha256(appended_ids),
        "resampled_ordered_sample_ids_sha256": ordered_id_list_sha256(training_ids),
        "minority_sample_multiplicity": minority_multiplicity,
        "oversampled_class_counts": {
            "VACANT": post_counts[0],
            "OCCUPIED": post_counts[1],
        },
        "all_majority_examples_retained": True,
        "majority_undersampling_count": 0,
        "validation_rows_used": 0,
        "locked_test_rows_used": 0,
        "synthetic_interpolation": False,
        "deterministic": True,
    }
    return OversamplePlan(training_indices, appended, evidence)


def verify_oversample_evidence(
    labels: np.ndarray,
    sample_ids: Sequence[str],
    stored: Mapping[str, Any],
) -> None:
    live = build_balanced_oversample_plan(labels, sample_ids).evidence
    keys = (
        "seed",
        "appended_minority_sample_ids",
        "appended_sequence_sha256",
        "resampled_ordered_sample_ids_sha256",
        "minority_sample_multiplicity",
        "oversampled_class_counts",
        "majority_undersampling_count",
    )
    for key in keys:
        if stored.get(key) != live.get(key):
            raise CB2Error(f"Non-deterministic/tampered oversampling evidence: {key}")


def fit_train_only_scaler(
    train: MatrixBundle,
    *,
    fit_population_fingerprint: str,
) -> Tuple[StandardScaler, Dict[str, Any]]:
    if train.split_role != "TRAIN":
        raise CB2Error("Scaler fit must use original TRAIN only")
    if train.features.shape[0] != EXPECTED_TRAIN_COMMON:
        raise CB2Error("Scaler must fit exactly 8140 original TRAIN samples")
    validate_feature_context(train.feature_names)
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    with threadpool_limits(limits=1):
        scaler.fit(train.features)
    payload_for_fp = {
        "feature_order": list(train.feature_names),
        "fit_population_fingerprint": fit_population_fingerprint,
        "mean": [float(x) for x in scaler.mean_],
        "scale": [float(x) for x in scaler.scale_],
        "var": [float(x) for x in scaler.var_],
        "n_samples_seen": int(scaler.n_samples_seen_),
    }
    fingerprint = _stable_json_sha256(payload_for_fp)
    evidence = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "scaler_profile_id": SCALER_PROFILE_ID,
        "classification": "C_B2_EXPERIMENT_REFERENCE_ONLY",
        "implementation": "sklearn.preprocessing.StandardScaler",
        "feature_order": list(train.feature_names),
        "fit_population": "ORIGINAL_NATURALLY_DISTRIBUTED_TRAIN_ONLY",
        "fit_sample_count": int(train.features.shape[0]),
        "fit_population_fingerprint": fit_population_fingerprint,
        "validation_fit_rows": 0,
        "locked_test_fit_rows": 0,
        "oversampled_fit_rows": 0,
        "fit_once": True,
        "reused_for_every_imbalance_arm": True,
        "mean": payload_for_fp["mean"],
        "scale": payload_for_fp["scale"],
        "variance": payload_for_fp["var"],
        "n_samples_seen": payload_for_fp["n_samples_seen"],
        "scaler_fingerprint": fingerprint,
        "per_arm_scaler_fingerprint": {
            strategy: fingerprint for strategy in AUTHORIZED_STRATEGIES
        },
        "production_scaler_modified": False,
        "existing_production_scaler_status": "SCALER_FIT_LINEAGE_UNVERIFIED",
    }
    return scaler, evidence


def assert_shared_scaler_fingerprints(evidence: Mapping[str, Any]) -> None:
    expected = evidence.get("scaler_fingerprint")
    per_arm = evidence.get("per_arm_scaler_fingerprint") or {}
    if set(per_arm) != set(AUTHORIZED_STRATEGIES):
        raise CB2Error("Missing per-arm shared scaler evidence")
    if any(value != expected for value in per_arm.values()):
        raise CB2Error("Arm-specific scaler detected")


def build_logistic_probe(class_weight: Optional[Mapping[int, float]]) -> LogisticRegression:
    return LogisticRegression(
        penalty=FIXED_LOGISTIC_PARAMETERS["penalty"],
        C=FIXED_LOGISTIC_PARAMETERS["C"],
        solver=FIXED_LOGISTIC_PARAMETERS["solver"],
        fit_intercept=FIXED_LOGISTIC_PARAMETERS["fit_intercept"],
        max_iter=FIXED_LOGISTIC_PARAMETERS["max_iter"],
        class_weight=None if class_weight is None else dict(class_weight),
        random_state=FIXED_LOGISTIC_PARAMETERS["random_state"],
    )


def logistic_parameter_contract(model: LogisticRegression) -> Dict[str, Any]:
    params = model.get_params(deep=False)
    return {
        "penalty": params["penalty"],
        "C": float(params["C"]),
        "solver": params["solver"],
        "fit_intercept": bool(params["fit_intercept"]),
        "max_iter": int(params["max_iter"]),
        "random_state": int(params["random_state"]),
    }


def validate_logistic_parameter_contract(params: Mapping[str, Any]) -> None:
    if dict(params) != FIXED_LOGISTIC_PARAMETERS:
        raise CB2Error(
            f"B2 fixed logistic probe hyperparameter drift: {dict(params)}"
        )


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def classification_metrics_at_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Tuple[Dict[str, Any], np.ndarray]:
    y = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if y.shape != probs.shape or y.ndim != 1:
        raise CB2Error("Threshold metric shape mismatch")
    if not np.isfinite(probs).all() or np.any(probs < 0.0) or np.any(probs > 1.0):
        raise CB2Error("Invalid occupancy probabilities")
    if not 0.0 <= threshold <= 1.0:
        raise CB2Error("Decision threshold outside [0, 1]")
    pred = (probs >= float(threshold)).astype(np.int64)
    metrics = compute_classification_metrics(y, pred)
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    metrics.update(
        {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "false_positive_rate": _safe_div(fp, fp + tn),
            "false_negative_rate": _safe_div(fn, fn + tp),
            "specificity": metrics["recall_vacant"],
            "sensitivity": metrics["recall_occupied"],
            "occupied_recall": metrics["recall_occupied"],
            "decision_threshold": float(threshold),
            "positive_prediction_rule": "occupancy_probability_greater_than_or_equal_to_threshold",
        }
    )
    return metrics, pred


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = ECE_BIN_COUNT,
) -> Dict[str, Any]:
    if n_bins != ECE_BIN_COUNT:
        raise CB2Error("C-B2 ECE contract requires exactly 10 equal-width bins")
    y = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if y.shape != probs.shape or y.ndim != 1:
        raise CB2Error("ECE shape mismatch")
    bins: List[Dict[str, Any]] = []
    total_ece = 0.0
    for i in range(n_bins):
        lower = float(i / n_bins)
        upper = float((i + 1) / n_bins)
        if i == n_bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        count = int(mask.sum())
        if count:
            mean_probability = float(probs[mask].mean())
            empirical_frequency = float(y[mask].mean())
            contribution = float(
                (count / y.size) * abs(mean_probability - empirical_frequency)
            )
        else:
            mean_probability = None
            empirical_frequency = None
            contribution = 0.0
        total_ece += contribution
        bins.append(
            {
                "bin_index": i,
                "lower_bound": lower,
                "upper_bound": upper,
                "lower_inclusive": True,
                "upper_inclusive": i == n_bins - 1,
                "sample_count": count,
                "mean_predicted_occupied_probability": mean_probability,
                "empirical_occupied_frequency": empirical_frequency,
                "ece_contribution": contribution,
            }
        )
    if sum(row["sample_count"] for row in bins) != y.size:
        raise CB2Error("ECE bins do not account for all validation samples")
    return {
        "definition": "10_EQUAL_WIDTH_BINS_OVER_0_1",
        "bin_count": n_bins,
        "population_count": int(y.size),
        "bins": bins,
        "expected_calibration_error": float(total_ece),
    }


def probability_quality_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    return {
        "roc_auc": float(roc_auc_score(y, probs)),
        "pr_auc_average_precision": float(average_precision_score(y, probs)),
        "brier_score": float(brier_score_loss(y, probs)),
        "log_loss": float(log_loss(y, probs, labels=[0, 1])),
    }


def rank_imbalance_strategies(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    seen = {row.get("strategy_id") for row in candidate_rows}
    if seen != set(AUTHORIZED_STRATEGIES) or len(candidate_rows) != 3:
        raise CB2Error(f"Stage-1 ranking requires exactly three authorized arms: {seen}")

    def key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
        metrics = row["metrics"]
        sid = str(row["strategy_id"])
        return (
            -float(metrics["macro_f1"]),
            -float(metrics["recall_occupied"]),
            -float(metrics["balanced_accuracy"]),
            float(metrics["false_positive_rate"]),
            STRATEGY_SIMPLICITY[sid],
            sid,
        )

    ordered = sorted(candidate_rows, key=key)
    return [
        {
            "rank": rank,
            "strategy_id": row["strategy_id"],
            "validation_macro_f1": float(row["metrics"]["macro_f1"]),
            "validation_occupied_recall": float(row["metrics"]["recall_occupied"]),
            "validation_balanced_accuracy": float(
                row["metrics"]["balanced_accuracy"]
            ),
            "validation_false_positive_rate": float(
                row["metrics"]["false_positive_rate"]
            ),
            "intervention_simplicity_rank": STRATEGY_SIMPLICITY[row["strategy_id"]],
        }
        for rank, row in enumerate(ordered, 1)
    ]


def build_threshold_grid() -> List[float]:
    grid = [float(i / 100) for i in range(5, 96)]
    if (
        len(grid) != THRESHOLD_COUNT
        or grid[0] != THRESHOLD_MIN
        or grid[-1] != THRESHOLD_MAX
        or DEFAULT_THRESHOLD not in grid
    ):
        raise CB2Error("Threshold grid construction failure")
    return grid


def rank_threshold_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    expected_grid = build_threshold_grid()
    values = [float(row["threshold"]) for row in rows]
    if values != expected_grid:
        raise CB2Error("Threshold grid mutation or ordering drift")

    def key(row: Mapping[str, Any]) -> Tuple[float, ...]:
        metrics = row["metrics"]
        threshold = float(row["threshold"])
        return (
            -float(metrics["macro_f1"]),
            -float(metrics["recall_occupied"]),
            -float(metrics["balanced_accuracy"]),
            float(metrics["false_positive_rate"]),
            abs(threshold - DEFAULT_THRESHOLD),
            threshold,
        )

    ordered = sorted(rows, key=key)
    return [
        {
            "rank": rank,
            "threshold": float(row["threshold"]),
            "validation_macro_f1": float(row["metrics"]["macro_f1"]),
            "validation_occupied_recall": float(row["metrics"]["recall_occupied"]),
            "validation_balanced_accuracy": float(
                row["metrics"]["balanced_accuracy"]
            ),
            "validation_false_positive_rate": float(
                row["metrics"]["false_positive_rate"]
            ),
            "distance_from_default_threshold": abs(
                float(row["threshold"]) - DEFAULT_THRESHOLD
            ),
        }
        for rank, row in enumerate(ordered, 1)
    ]


def build_threshold_sweep(
    *,
    y_validation: np.ndarray,
    probabilities: np.ndarray,
    sample_ids: Sequence[str],
    population_role: str = "VALIDATION",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if population_role != "VALIDATION":
        raise LockedTestPolicyViolation(
            f"Threshold calibration population must be VALIDATION, not {population_role}"
        )
    probability_fp = _probability_fingerprint(sample_ids, probabilities)
    rows: List[Dict[str, Any]] = []
    for threshold in build_threshold_grid():
        metrics, _ = classification_metrics_at_threshold(
            y_validation, probabilities, threshold
        )
        rows.append(
            {
                "threshold": threshold,
                "metrics": metrics,
                "probability_vector_sha256": probability_fp,
            }
        )
    ranking = rank_threshold_rows(rows)
    return rows, ranking


def validate_probability_invariance(rows: Sequence[Mapping[str, Any]]) -> None:
    fingerprints = {row.get("probability_vector_sha256") for row in rows}
    if len(fingerprints) != 1 or None in fingerprints:
        raise CB2Error("Probability mutation detected across threshold evaluations")


def validate_probability_semantics(contract: Mapping[str, Any]) -> None:
    required = {
        "target": "ROOM_OCCUPANCY",
        "probability_class": "OCCUPIED",
        "safety_semantic": "NONE",
        "risk_semantic": "NONE",
        "decision_threshold_role": "OFFLINE_OCCUPANCY_CLASSIFICATION",
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise CB2Error(f"Occupancy/safety semantic conflation: {key}")


def validate_fp_fn_report(report: Mapping[str, Any]) -> None:
    if report.get("DOMAIN_FP_FN_COST_RATIO") != "UNSPECIFIED":
        raise CB2Error("Domain FP/FN cost ratio must remain UNSPECIFIED")
    if report.get("fabricated_weighted_safety_score") is not False:
        raise CB2Error("Fabricated weighted safety score rejected")
    rows = report.get("stage1_default_threshold") or {}
    if set(rows) != set(AUTHORIZED_STRATEGIES):
        raise CB2Error("FP/FN report strategy coverage mismatch")
    for strategy_id, row in rows.items():
        if "fp" not in row or "fn" not in row:
            raise CB2Error(f"Missing FP/FN accounting for {strategy_id}")


def validate_reference_threshold_claims(result: Mapping[str, Any]) -> None:
    if result.get("classification") != "REFERENCE_PROBE_THRESHOLD_ONLY":
        raise CB2Error("Reference threshold classification drift")
    if result.get("reference_threshold_production_final") is not False:
        raise CB2Error("Final-production-threshold overclaim rejected")
    if result.get("TRANSFER_TO_FUTURE_ARCHITECTURES") != "NOT_ASSUMED":
        raise CB2Error("Threshold transfer to future architectures is not allowed")


def _prediction_records(
    *,
    sample_ids: Sequence[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    strategy_id: str,
    threshold: float,
    threshold_id: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "sample_id": sid,
            "true_occupancy_label": int(y),
            "true_occupancy_class": "OCCUPIED" if int(y) == 1 else "VACANT",
            "occupancy_probability": float(prob),
            "predicted_occupancy_label": int(pred),
            "predicted_occupancy_class": "OCCUPIED" if int(pred) == 1 else "VACANT",
            "imbalance_arm": strategy_id,
            "threshold_identity": threshold_id,
            "decision_threshold": float(threshold),
        }
        for sid, y, prob, pred in zip(
            sample_ids,
            labels.tolist(),
            probabilities.tolist(),
            predictions.tolist(),
        )
    ]


def _fit_strategy(
    *,
    strategy_id: str,
    x_train_scaled: np.ndarray,
    y_train: np.ndarray,
    x_validation_scaled: np.ndarray,
    class_weights: Mapping[int, float],
    oversample_plan: OversamplePlan,
) -> Tuple[LogisticRegression, np.ndarray, Dict[str, Any]]:
    if strategy_id == NATURAL_DISTRIBUTION:
        indices = np.arange(y_train.size, dtype=np.int64)
        weight = None
    elif strategy_id == CLASS_WEIGHT_BALANCED:
        indices = np.arange(y_train.size, dtype=np.int64)
        weight = dict(class_weights)
    elif strategy_id == BALANCED_RANDOM_OVERSAMPLE:
        indices = oversample_plan.training_indices
        weight = None
    else:
        raise CB2Error(f"Unauthorized imbalance strategy: {strategy_id}")

    model = build_logistic_probe(weight)
    validate_logistic_parameter_contract(logistic_parameter_contract(model))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        with threadpool_limits(limits=1):
            model.fit(x_train_scaled[indices], y_train[indices])
    convergence_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    if convergence_messages:
        raise CB2Error(f"LOGISTIC_PROBE_CONVERGENCE_FAILURE: {convergence_messages}")
    probabilities = np.asarray(
        model.predict_proba(x_validation_scaled)[:, 1], dtype=np.float64
    )
    if not np.isfinite(probabilities).all():
        raise CB2Error(f"Non-finite probabilities for {strategy_id}")
    fit_evidence = {
        "strategy_id": strategy_id,
        "fit_population": "TRAIN_ONLY",
        "fit_row_count": int(indices.size),
        "fit_unique_original_row_count": int(len(set(indices.tolist()))),
        "class_weight": (
            None
            if weight is None
            else {"VACANT": float(weight[0]), "OCCUPIED": float(weight[1])}
        ),
        "parameters": logistic_parameter_contract(model),
        "coefficient_feature_order": list(FIXED_FEATURES),
        "coefficients": [float(x) for x in model.coef_[0]],
        "intercept": [float(x) for x in model.intercept_],
        "classes": [int(x) for x in model.classes_.tolist()],
        "n_iter": [int(x) for x in model.n_iter_.tolist()],
        "convergence_warning_count": 0,
        "architecture_search_performed": False,
        "multi_seed_search_performed": False,
    }
    return model, probabilities, fit_evidence


def _threshold_is_strictly_better(
    calibrated: Mapping[str, Any], default: Mapping[str, Any]
) -> bool:
    def key(row: Mapping[str, Any]) -> Tuple[float, ...]:
        metrics = row["metrics"]
        threshold = float(row["threshold"])
        return (
            -float(metrics["macro_f1"]),
            -float(metrics["recall_occupied"]),
            -float(metrics["balanced_accuracy"]),
            float(metrics["false_positive_rate"]),
            abs(threshold - DEFAULT_THRESHOLD),
            threshold,
        )

    return key(calibrated) < key(default)


def run_imbalance_calibration(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root or get_repo_root()
    output_dir = root / ARTIFACT_DIR_REL
    output_dir.mkdir(parents=True, exist_ok=True)

    predecessor = build_predecessor_fingerprint_registry(root)
    release = verify_a_series_release(root)
    lock = verify_a_series_artifact_lock(root)
    if not release.get("matches_expected") or release.get("resolved_commit") != A_SERIES_RELEASE_TARGET:
        raise PredecessorFingerprintMismatch("A-series release anchor invalid")
    if lock.get("status") != "VERIFIED":
        raise PredecessorFingerprintMismatch("A-series artifact lock invalid")

    universe = build_sample_universe_manifest(root)
    by_role = _load_eligible_by_role(root)
    overlap = validate_population_contract(
        by_role["TRAIN"], by_role["VALIDATION"], by_role["LOCKED_TEST"]
    )
    train, train_load_audit = load_authorized_matrix(repo_root=root, split_role="TRAIN")
    validation, validation_load_audit = load_authorized_matrix(
        repo_root=root, split_role="VALIDATION"
    )
    if train.sample_ids != by_role["TRAIN"] or validation.sample_ids != by_role["VALIDATION"]:
        raise CB2Error("Ordered B0 sample-universe mismatch")

    scaler, scaler_evidence = fit_train_only_scaler(
        train,
        fit_population_fingerprint=universe["ordered_id_list_sha256"]["TRAIN"],
    )
    assert_shared_scaler_fingerprints(scaler_evidence)
    x_train_scaled = np.asarray(scaler.transform(train.features), dtype=np.float64)
    x_validation_scaled = np.asarray(
        scaler.transform(validation.features), dtype=np.float64
    )

    counts = Counter(int(x) for x in train.labels.tolist())
    class_weights = compute_balanced_class_weights(train.labels)
    oversample_plan = build_balanced_oversample_plan(train.labels, train.sample_ids)
    strategy_registry = build_imbalance_strategy_registry()
    validate_imbalance_registry(strategy_registry)

    stage1_candidates: List[Dict[str, Any]] = []
    fit_evidence: Dict[str, Any] = {}
    arm_probabilities: Dict[str, np.ndarray] = {}
    arm_predictions: Dict[str, np.ndarray] = {}
    calibration_by_arm: Dict[str, Any] = {}
    prediction_arms: Dict[str, Any] = {}

    for strategy_id in AUTHORIZED_STRATEGIES:
        _, probabilities, fit_doc = _fit_strategy(
            strategy_id=strategy_id,
            x_train_scaled=x_train_scaled,
            y_train=train.labels,
            x_validation_scaled=x_validation_scaled,
            class_weights=class_weights,
            oversample_plan=oversample_plan,
        )
        metrics, predictions = classification_metrics_at_threshold(
            validation.labels, probabilities, DEFAULT_THRESHOLD
        )
        probability_metrics = probability_quality_metrics(validation.labels, probabilities)
        calibration = expected_calibration_error(validation.labels, probabilities)
        probability_metrics["expected_calibration_error"] = calibration[
            "expected_calibration_error"
        ]
        probability_fp = _probability_fingerprint(validation.sample_ids, probabilities)
        candidate = {
            "strategy_id": strategy_id,
            "stage": "STAGE_1_IMBALANCE_COMPARISON",
            "decision_threshold": DEFAULT_THRESHOLD,
            "evaluation_population": "VALIDATION",
            "evaluation_sample_count": len(validation.sample_ids),
            "evaluation_population_fingerprint": universe["ordered_id_list_sha256"][
                "VALIDATION"
            ],
            "scaler_fingerprint": scaler_evidence["scaler_fingerprint"],
            "metrics": metrics,
            "probability_quality_metrics": probability_metrics,
            "validation_probability_vector_sha256": probability_fp,
            "locked_test_predictions": 0,
            "locked_test_metrics": 0,
        }
        stage1_candidates.append(candidate)
        fit_evidence[strategy_id] = fit_doc
        arm_probabilities[strategy_id] = probabilities
        arm_predictions[strategy_id] = predictions
        calibration_by_arm[strategy_id] = {
            "probability_quality_metrics": probability_metrics,
            "ece_diagnostic": calibration,
            "probability_vector_sha256": probability_fp,
        }
        records = _prediction_records(
            sample_ids=validation.sample_ids,
            labels=validation.labels,
            probabilities=probabilities,
            predictions=predictions,
            strategy_id=strategy_id,
            threshold=DEFAULT_THRESHOLD,
            threshold_id="DEFAULT_THRESHOLD_0_50",
        )
        prediction_arms[strategy_id] = {
            "threshold": DEFAULT_THRESHOLD,
            "count": len(records),
            "records": records,
        }

    stage1_ranking = rank_imbalance_strategies(stage1_candidates)
    selected_strategy = stage1_ranking[0]["strategy_id"]
    selected_stage1 = next(
        row for row in stage1_candidates if row["strategy_id"] == selected_strategy
    )
    selected_probabilities = arm_probabilities[selected_strategy]
    threshold_rows, threshold_ranking = build_threshold_sweep(
        y_validation=validation.labels,
        probabilities=selected_probabilities,
        sample_ids=validation.sample_ids,
    )
    validate_probability_invariance(threshold_rows)
    selected_threshold = float(threshold_ranking[0]["threshold"])
    selected_threshold_row = next(
        row for row in threshold_rows if float(row["threshold"]) == selected_threshold
    )
    default_threshold_row = next(
        row for row in threshold_rows if float(row["threshold"]) == DEFAULT_THRESHOLD
    )
    calibrated_metrics = selected_threshold_row["metrics"]
    calibrated_predictions = classification_metrics_at_threshold(
        validation.labels, selected_probabilities, selected_threshold
    )[1]
    calibrated_records = _prediction_records(
        sample_ids=validation.sample_ids,
        labels=validation.labels,
        probabilities=selected_probabilities,
        predictions=calibrated_predictions,
        strategy_id=selected_strategy,
        threshold=selected_threshold,
        threshold_id="SELECTED_REFERENCE_THRESHOLD",
    )

    default_metrics = default_threshold_row["metrics"]
    deltas = {
        "delta_macro_f1": float(calibrated_metrics["macro_f1"])
        - float(default_metrics["macro_f1"]),
        "delta_balanced_accuracy": float(calibrated_metrics["balanced_accuracy"])
        - float(default_metrics["balanced_accuracy"]),
        "delta_occupied_recall": float(calibrated_metrics["recall_occupied"])
        - float(default_metrics["recall_occupied"]),
        "delta_fp": int(calibrated_metrics["fp"]) - int(default_metrics["fp"]),
        "delta_fn": int(calibrated_metrics["fn"]) - int(default_metrics["fn"]),
        "delta_false_positive_rate": float(calibrated_metrics["false_positive_rate"])
        - float(default_metrics["false_positive_rate"]),
        "delta_false_negative_rate": float(calibrated_metrics["false_negative_rate"])
        - float(default_metrics["false_negative_rate"]),
    }
    incremental_established = (
        selected_threshold != DEFAULT_THRESHOLD
        and _threshold_is_strictly_better(selected_threshold_row, default_threshold_row)
    )
    incremental_status = (
        "THRESHOLD_CALIBRATION_INCREMENTAL_VALUE_ESTABLISHED"
        if incremental_established
        else "THRESHOLD_CALIBRATION_INCREMENTAL_VALUE_NOT_ESTABLISHED"
    )

    class_distribution = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "population": "ORIGINAL_NATURALLY_DISTRIBUTED_TRAIN",
        "population_count": len(train.sample_ids),
        "population_fingerprint": universe["ordered_id_list_sha256"]["TRAIN"],
        "class_counts": {"VACANT": counts[0], "OCCUPIED": counts[1]},
        "negative_class": "VACANT",
        "positive_class": "OCCUPIED",
        "majority_class": "VACANT" if counts[0] > counts[1] else "OCCUPIED",
        "minority_class": "OCCUPIED" if counts[1] < counts[0] else "VACANT",
        "majority_to_minority_ratio": float(max(counts.values()) / min(counts.values())),
        "validation_used": 0,
        "locked_test_used": 0,
    }
    class_weight_evidence = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "strategy_id": CLASS_WEIGHT_BALANCED,
        "formula": "N_train / (N_classes * N_class_c)",
        "derivation_population": "TRAIN_ONLY",
        "n_train": len(train.sample_ids),
        "n_classes": 2,
        "class_counts": {"VACANT": counts[0], "OCCUPIED": counts[1]},
        "explicit_class_weights": {
            "VACANT": class_weights[0],
            "OCCUPIED": class_weights[1],
        },
        "all_original_train_rows_retained": True,
        "validation_rows_used": 0,
        "locked_test_rows_used": 0,
    }

    experiment_contract = {
        "manifest_version": "1.0",
        "schema": "SafeNest_CO2_C_B2_Imbalance_Calibration_Contract",
        "phase": PHASE_ID,
        "phase_name": PHASE_NAME,
        "experiment_contract_id": EXPERIMENT_CONTRACT_ID,
        "immediate_predecessor": "C-B1",
        "b1_required_merged_commit": B1_MERGED_MAIN_COMMIT,
        "b0_contract_id": B0_CONTRACT_ID,
        "b1_selected_slope_profile_id": B1_SELECTED_PROFILE_ID,
        "target_profile_id": TARGET_PROFILE_ID,
        "positive_class": "OCCUPIED",
        "negative_class": "VACANT",
        "train_population": EXPECTED_TRAIN_COMMON,
        "validation_population": EXPECTED_VALIDATION_COMMON,
        "locked_test_membership_count": EXPECTED_LOCKED_TEST_SEALED,
        "locked_test_status": "SEALED",
        "stage1_variable": "IMBALANCE_STRATEGY_ONLY",
        "stage1_threshold": DEFAULT_THRESHOLD,
        "stage2_variable": "DECISION_THRESHOLD_ONLY",
        "stage2_population": "VALIDATION_ONLY",
        "architecture_search": "NOT_PERFORMED",
        "multi_seed_search": "NOT_PERFORMED",
        "feature_selection": "NOT_PERFORMED",
        "probability_recalibration_model": "NOT_FITTED",
        "production_model_promotion": "PROHIBITED",
        "next_phase": "C-B3_ARCHITECTURE_MULTI_SEED",
    }
    feature_context = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "feature_context_id": FEATURE_CONTEXT_ID,
        "classification": "B2_FIXED_EXPERIMENT_FEATURE_CONTEXT",
        "feature_order": list(FIXED_FEATURES),
        "feature_count": len(FIXED_FEATURES),
        "slope_profile_id": B1_SELECTED_PROFILE_ID,
        "slope_candidate_id": B1_SELECTED_CANDIDATE_ID,
        "slope_method": B1_SELECTED_METHOD,
        "minimum_history_seconds": B1_SELECTED_HISTORY_SECONDS,
        "slope_numeric_source": "C_A3_CANONICAL_VALUES_WITH_B1_EXACT_PARITY_PASS",
        "excluded_features": ["Light", "HumidityRatio"],
        "target_fields_as_features": [],
        "provenance_fields_as_features": [],
        "final_feature_selection_performed": False,
    }
    fixed_probe_contract = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "probe_profile_id": PROBE_PROFILE_ID,
        "implementation": "sklearn.linear_model.LogisticRegression",
        "parameters": FIXED_LOGISTIC_PARAMETERS,
        "class_weight": "STRATEGY_DEPENDENT_ONLY",
        "dependency": "scikit-learn",
        "dependency_version": sklearn.__version__,
        "classification": "REFERENCE_PROBABILISTIC_PROBE_ONLY",
        "production_model": False,
        "final_architecture": False,
        "tflite_candidate": False,
        "architecture_search_performed": False,
        "hyperparameter_search_performed": False,
        "multi_seed_search_performed": False,
    }
    stage1_results = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "stage": "STAGE_1_IMBALANCE_STRATEGY_COMPARISON",
        "fixed_threshold": DEFAULT_THRESHOLD,
        "evaluation_population": "VALIDATION",
        "candidate_count": len(stage1_candidates),
        "candidates": {row["strategy_id"]: row for row in stage1_candidates},
        "probability_metrics_are_threshold_independent": True,
        "locked_test_predictions": 0,
        "locked_test_metrics": 0,
    }
    selected_policy = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "policy_id": SELECTED_IMBALANCE_POLICY_ID,
        "selected_strategy": selected_strategy,
        "fixed_probe_identity": PROBE_PROFILE_ID,
        "train_population_fingerprint": universe["ordered_id_list_sha256"]["TRAIN"],
        "validation_population_fingerprint": universe["ordered_id_list_sha256"][
            "VALIDATION"
        ],
        "class_counts": class_distribution["class_counts"],
        "explicit_class_weights_if_applicable": (
            class_weight_evidence["explicit_class_weights"]
            if selected_strategy == CLASS_WEIGHT_BALANCED
            else None
        ),
        "oversampling_parameters_if_applicable": (
            {
                "method": oversample_plan.evidence["method"],
                "seed": DEFAULT_SEED,
                "oversampled_class_counts": oversample_plan.evidence[
                    "oversampled_class_counts"
                ],
            }
            if selected_strategy == BALANCED_RANDOM_OVERSAMPLE
            else None
        ),
        "seed": DEFAULT_SEED,
        "preprocessing_fingerprint": scaler_evidence["scaler_fingerprint"],
        "selection_metric": "VALIDATION_MACRO_F1",
        "selection_threshold": DEFAULT_THRESHOLD,
        "complete_ranking_rule": strategy_registry["ranking_rule"],
        "ranking": stage1_ranking,
        "stage1_candidate_results": stage1_results["candidates"],
        "deployment_status": [
            "OFFLINE_VALIDATION_SELECTED",
            "ARCHITECTURE_GENERALIZATION_UNVERIFIED",
            "DEVICE_DOMAIN_UNVALIDATED",
        ],
    }
    threshold_protocol = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "protocol_id": CALIBRATION_PROTOCOL_ID,
        "threshold_search_population": "VALIDATION_ONLY",
        "threshold_grid": build_threshold_grid(),
        "threshold_grid_min": THRESHOLD_MIN,
        "threshold_grid_max": THRESHOLD_MAX,
        "threshold_grid_step": THRESHOLD_STEP,
        "threshold_count": THRESHOLD_COUNT,
        "threshold_grid_fingerprint": _stable_json_sha256(build_threshold_grid()),
        "positive_class": "OCCUPIED",
        "default_threshold": DEFAULT_THRESHOLD,
        "classification_rule": "probability_greater_than_or_equal_to_threshold_is_OCCUPIED",
        "required_metrics": [
            "macro_f1",
            "balanced_accuracy",
            "recall_occupied",
            "precision_occupied",
            "recall_vacant",
            "fp",
            "fn",
            "false_positive_rate",
            "false_negative_rate",
            "confusion_matrix",
        ],
        "fp_fn_reporting": "BOTH_REQUIRED_SEPARATELY",
        "ranking_rule": [
            "higher_validation_macro_f1",
            "higher_validation_occupied_recall",
            "higher_validation_balanced_accuracy",
            "lower_validation_false_positive_rate",
            "threshold_closer_to_0_50",
            "lower_numeric_threshold",
        ],
        "locked_test_fit_tuning_or_evaluation": "PROHIBITED",
        "probability_recalibration_model_fitted": False,
        "future_architecture_use": "REUSE_PROTOCOL_NOT_LOGISTIC_NUMERIC_THRESHOLD",
    }
    threshold_sweep = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "protocol_id": CALIBRATION_PROTOCOL_ID,
        "selected_imbalance_strategy": selected_strategy,
        "population": "VALIDATION",
        "population_count": len(validation.sample_ids),
        "probability_vector_sha256": _probability_fingerprint(
            validation.sample_ids, selected_probabilities
        ),
        "probabilities_unchanged_across_thresholds": True,
        "probability_quality_metrics_threshold_independent": selected_stage1[
            "probability_quality_metrics"
        ],
        "threshold_count": len(threshold_rows),
        "rows": threshold_rows,
        "ranking": threshold_ranking,
        "locked_test_threshold_evaluations": 0,
    }
    reference_threshold = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "result_id": REFERENCE_THRESHOLD_RESULT_ID,
        "selected_imbalance_strategy": selected_strategy,
        "fixed_logistic_probe": PROBE_PROFILE_ID,
        "default_threshold": DEFAULT_THRESHOLD,
        "selected_reference_threshold": selected_threshold,
        "default_threshold_metrics": default_metrics,
        "calibrated_threshold_metrics": calibrated_metrics,
        "metric_deltas": deltas,
        "fp_change": deltas["delta_fp"],
        "fn_change": deltas["delta_fn"],
        "threshold_grid_fingerprint": threshold_protocol["threshold_grid_fingerprint"],
        "calibration_protocol_id": CALIBRATION_PROTOCOL_ID,
        "incremental_evidence_status": incremental_status,
        "classification": "REFERENCE_PROBE_THRESHOLD_ONLY",
        "reference_threshold_production_final": False,
        "TRANSFER_TO_FUTURE_ARCHITECTURES": "NOT_ASSUMED",
        "device_domain_validation": "NOT_PERFORMED",
    }
    calibration_diagnostics = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "ece_contract": {
            "bin_count": ECE_BIN_COUNT,
            "binning": "EQUAL_WIDTH",
            "range": [0.0, 1.0],
            "selection_criterion": False,
        },
        "per_stage1_arm": calibration_by_arm,
        "probability_quality_metrics_are_threshold_independent": True,
        "probability_recalibration_model_fitted": False,
    }
    fp_fn_report = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "DOMAIN_FP_FN_COST_RATIO": "UNSPECIFIED",
        "fabricated_weighted_safety_score": False,
        "stage1_default_threshold": {
            row["strategy_id"]: {
                "fp": row["metrics"]["fp"],
                "fn": row["metrics"]["fn"],
                "false_positive_rate": row["metrics"]["false_positive_rate"],
                "false_negative_rate": row["metrics"]["false_negative_rate"],
                "occupied_recall": row["metrics"]["recall_occupied"],
                "specificity": row["metrics"]["specificity"],
            }
            for row in stage1_candidates
        },
        "selected_strategy_default_vs_calibrated": {
            "strategy_id": selected_strategy,
            "default_threshold": DEFAULT_THRESHOLD,
            "calibrated_threshold": selected_threshold,
            "default": {
                "fp": default_metrics["fp"],
                "fn": default_metrics["fn"],
                "false_positive_rate": default_metrics["false_positive_rate"],
                "false_negative_rate": default_metrics["false_negative_rate"],
                "occupied_recall": default_metrics["recall_occupied"],
                "specificity": default_metrics["specificity"],
            },
            "calibrated": {
                "fp": calibrated_metrics["fp"],
                "fn": calibrated_metrics["fn"],
                "false_positive_rate": calibrated_metrics["false_positive_rate"],
                "false_negative_rate": calibrated_metrics["false_negative_rate"],
                "occupied_recall": calibrated_metrics["recall_occupied"],
                "specificity": calibrated_metrics["specificity"],
            },
            "deltas": deltas,
        },
    }
    probability_semantics = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "contract_id": PROBABILITY_SEMANTIC_CONTRACT_ID,
        "target": "ROOM_OCCUPANCY",
        "target_profile_id": TARGET_PROFILE_ID,
        "probability_class": "OCCUPIED",
        "probability_range": [0.0, 1.0],
        "probability_interpretation": "P(Occupancy=OCCUPIED | fixed B2 model features)",
        "safety_semantic": "NONE",
        "risk_semantic": "NONE",
        "co2_danger_probability": False,
        "ventilation_danger_probability": False,
        "fatality_probability": False,
        "multisensor_risk_score": False,
        "decision_threshold_role": "OFFLINE_OCCUPANCY_CLASSIFICATION",
        "risk_logic_modified": False,
    }
    validate_probability_semantics(probability_semantics)
    leakage_audit = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "train_validation_overlap": overlap["train_validation"],
        "train_locked_test_overlap": overlap["train_locked_test"],
        "validation_locked_test_overlap": overlap["validation_locked_test"],
        "target_as_feature": 0,
        "provenance_as_feature": 0,
        "validation_in_scaler_fit": 0,
        "locked_test_in_scaler_fit": 0,
        "validation_in_class_weight_derivation": 0,
        "locked_test_in_class_weight_derivation": 0,
        "validation_in_balanced_sampling": 0,
        "locked_test_in_balanced_sampling": 0,
        "validation_used_for_model_fitting": 0,
        "locked_test_used_for_model_fitting": 0,
        "threshold_selected_on": "VALIDATION_ONLY",
        "locked_test_feature_access": 0,
        "locked_test_target_access": 0,
        "locked_test_threshold_access": 0,
        "locked_test_predictions": 0,
        "locked_test_probability_outputs": 0,
        "locked_test_threshold_evaluations": 0,
        "locked_test_metrics": 0,
        "locked_test_fit_usage": 0,
        "locked_test_tuning_usage": 0,
        "locked_test_membership_count_verified": len(by_role["LOCKED_TEST"]),
        "train_matrix_load_audit": train_load_audit,
        "validation_matrix_load_audit": validation_load_audit,
        "synthetic_fixture_used_as_real_training_data": False,
        "status": "PASS",
    }
    exceptions = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "warnings": [
            {
                "code": "DEVICE_UCI_CADENCE_DOMAIN_GAP",
                "description": "C-B2 remains OFFLINE_UCI; SCD40 cadence/domain is unvalidated.",
            },
            {
                "code": "REFERENCE_PROBE_ONLY",
                "description": "The logistic probe and its numeric threshold are not production-final.",
            },
            {
                "code": "DOMAIN_FP_FN_COST_RATIO_UNSPECIFIED",
                "description": "FP and FN are reported separately; no domain cost ratio is invented.",
            },
        ],
        "blockers": [],
        "device_validation": "NOT_PERFORMED",
        "real_scd40_validation": "NOT_PERFORMED",
        "raspberry_pi_validation": "NOT_PERFORMED",
    }
    prediction_artifact = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "population": "VALIDATION_ONLY",
        "target": "ROOM_OCCUPANCY",
        "probability_class": "OCCUPIED",
        "safety_semantic": "NONE",
        "stage1_default_threshold_arms": prediction_arms,
        "selected_calibrated_threshold": {
            "strategy_id": selected_strategy,
            "threshold": selected_threshold,
            "count": len(calibrated_records),
            "records": calibrated_records,
        },
        "locked_test_predictions": 0,
        "locked_test_probability_outputs": 0,
    }
    generation = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "generator_script": "scripts/audit_co2_imbalance_calibration.py",
        "module": "datasets/co2/imbalance_calibration.py",
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "seed": DEFAULT_SEED,
        "thread_limit": 1,
        "generation_clock_policy": "OMITTED_FOR_BIT_IDENTICAL_RERUNS",
        "determinism_required": True,
        "determinism_validation": "BIT_IDENTICAL_REGENERATION_ENFORCED_BY_STANDALONE_VALIDATOR",
        "production_scaler_modified": False,
        "production_model_modified": False,
        "a_series_locked_artifacts_modified": False,
        "b0_predecessor_artifacts_modified": False,
        "b1_predecessor_artifacts_modified": False,
        "synthetic_npz_used_as_real_training_data": False,
        "architecture_comparison_performed": False,
        "multi_seed_comparison_performed": False,
        "final_feature_selection_performed": False,
        "probability_recalibration_model_fitted": False,
        "locked_test_predictions": 0,
        "locked_test_metrics": 0,
    }
    probe_fit_artifact = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "probe_profile_id": PROBE_PROFILE_ID,
        "fit_population": "TRAIN_ONLY",
        "evaluation_population": "VALIDATION_ONLY",
        "per_strategy": fit_evidence,
    }

    artifacts: Dict[str, Mapping[str, Any]] = {
        "predecessor_fingerprint_registry.json": predecessor,
        "experiment_contract.json": experiment_contract,
        "fixed_feature_context.json": feature_context,
        "preprocessing_scaler_evidence.json": scaler_evidence,
        "imbalance_strategy_registry.json": strategy_registry,
        "train_class_distribution.json": class_distribution,
        "class_weight_evidence.json": class_weight_evidence,
        "balanced_sampling_evidence.json": oversample_plan.evidence,
        "fixed_logistic_probe_contract.json": fixed_probe_contract,
        "probe_fit_evidence.json": probe_fit_artifact,
        "stage1_default_threshold_results.json": stage1_results,
        "imbalance_selection_decision.json": selected_policy,
        "threshold_calibration_protocol.json": threshold_protocol,
        "threshold_sweep_results.json": threshold_sweep,
        "reference_threshold_result.json": reference_threshold,
        "calibration_diagnostics.json": calibration_diagnostics,
        "fp_fn_error_report.json": fp_fn_report,
        "occupancy_probability_semantic_contract.json": probability_semantics,
        "validation_predictions.json": prediction_artifact,
        "leakage_audit.json": leakage_audit,
        "exceptions_and_limitations.json": exceptions,
        "generation_metadata.json": generation,
    }

    for filename, payload in sorted(artifacts.items()):
        path = output_dir / filename
        write_json(path, payload)
        forbidden = assert_no_forbidden_path_markers(path.read_text(encoding="utf-8"))
        if forbidden:
            raise CB2Error(f"Forbidden path marker in {filename}: {forbidden}")

    artifact_names = sorted(artifacts)
    identity = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "artifact_namespace": ARTIFACT_DIR_REL,
        "artifact_json_count": len(artifacts) + 1,
        "artifact_json_files": artifact_names + ["artifact_identity.json"],
        "selected_imbalance_policy_id": SELECTED_IMBALANCE_POLICY_ID,
        "selected_strategy": selected_strategy,
        "threshold_calibration_protocol_id": CALIBRATION_PROTOCOL_ID,
        "reference_threshold_result_id": REFERENCE_THRESHOLD_RESULT_ID,
        "reference_threshold_production_final": False,
        "locked_test_predictions": 0,
        "locked_test_metrics": 0,
        "raw_payload_included": False,
    }
    write_json(output_dir / "artifact_identity.json", identity)

    checksum_lines: List[str] = []
    for filename in sorted(identity["artifact_json_files"]):
        path = output_dir / filename
        rel = f"{ARTIFACT_DIR_REL}/{filename}"
        checksum_lines.append(f"{compute_sha256_file(path)}  {rel}")
    (output_dir / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )

    return {
        "artifact_dir": ARTIFACT_DIR_REL,
        "selected_strategy": selected_strategy,
        "selected_reference_threshold": selected_threshold,
        "stage1_selected_metrics": selected_stage1["metrics"],
        "calibrated_metrics": calibrated_metrics,
        "incremental_status": incremental_status,
        "train_class_counts": class_distribution["class_counts"],
        "class_weights": class_weight_evidence["explicit_class_weights"],
        "oversampled_class_counts": oversample_plan.evidence[
            "oversampled_class_counts"
        ],
        "scikit_learn_version": sklearn.__version__,
    }


def load_b2_artifacts(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root or get_repo_root()
    directory = root / ARTIFACT_DIR_REL
    return {
        path.name: load_json(path)
        for path in sorted(directory.glob("*.json"))
        if path.is_file()
    }
