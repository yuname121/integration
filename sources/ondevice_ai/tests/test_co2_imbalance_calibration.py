#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused contract and rejection tests for SafeNest CO2 Phase C-B2."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from datasets.co2.imbalance_calibration import (
    ARTIFACT_DIR_REL,
    AUTHORIZED_STRATEGIES,
    B0_DIR_REL,
    B1_DIR_REL,
    BALANCED_RANDOM_OVERSAMPLE,
    CB2Error,
    CLASS_WEIGHT_BALANCED,
    DEFAULT_SEED,
    DEFAULT_THRESHOLD,
    FIXED_FEATURES,
    FIXED_LOGISTIC_PARAMETERS,
    LockedTestPolicyViolation,
    NATURAL_DISTRIBUTION,
    PredecessorFingerprintMismatch,
    _probability_fingerprint,
    assert_shared_scaler_fingerprints,
    build_balanced_oversample_plan,
    build_imbalance_strategy_registry,
    build_predecessor_fingerprint_registry,
    build_threshold_grid,
    build_threshold_sweep,
    compute_balanced_class_weights,
    expected_calibration_error,
    fit_train_only_scaler,
    load_authorized_matrix,
    load_json,
    rank_imbalance_strategies,
    rank_threshold_rows,
    run_imbalance_calibration,
    validate_b1_selected_profile,
    validate_class_weight_evidence,
    validate_feature_context,
    validate_fp_fn_report,
    validate_imbalance_registry,
    validate_logistic_parameter_contract,
    validate_population_contract,
    validate_probability_invariance,
    validate_probability_semantics,
    validate_reference_threshold_claims,
    verify_oversample_evidence,
    verify_stored_predecessor_registry,
)
from datasets.co2.offline_experiment import (
    EXPECTED_LOCKED_TEST_SEALED,
    EXPECTED_TRAIN_COMMON,
    EXPECTED_VALIDATION_COMMON,
    MatrixBundle,
    assert_no_forbidden_path_markers,
)
from datasets.co2.raw_reader import compute_sha256_file, get_repo_root
from scripts.validate_co2_imbalance_calibration import (
    C_B2_ARTIFACT_DRIFT,
    C_B2_OWNED,
    CO2_SAME_TRACK,
    INTEGRATION_OTHER_TRACK,
    MMWAVE_OTHER_TRACK,
    SHARED_OR_UNAUTHORIZED,
    THERMAL_OTHER_TRACK,
    audit_path_scope,
    classify_path_ownership,
    validate as validate_c_b2,
)


def _bundle(role: str, n: int = 4) -> MatrixBundle:
    return MatrixBundle(
        sample_ids=[f"s{i}" for i in range(n)],
        features=np.arange(n * 4, dtype=np.float64).reshape(n, 4),
        labels=np.asarray([i % 2 for i in range(n)], dtype=np.int64),
        feature_names=FIXED_FEATURES,
        split_role=role,
    )


def _metric_row(
    strategy_id: str,
    *,
    macro_f1: float,
    recall: float,
    balanced: float,
    fpr: float,
) -> dict:
    return {
        "strategy_id": strategy_id,
        "metrics": {
            "macro_f1": macro_f1,
            "recall_occupied": recall,
            "balanced_accuracy": balanced,
            "false_positive_rate": fpr,
        },
    }


def _threshold_rows() -> list:
    return [
        {
            "threshold": threshold,
            "metrics": {
                "macro_f1": 0.5,
                "recall_occupied": 0.5,
                "balanced_accuracy": 0.5,
                "false_positive_rate": 0.5,
            },
        }
        for threshold in build_threshold_grid()
    ]


def test_missing_b1_predecessor_rejected(tmp_path: Path):
    root = get_repo_root()
    for directory in (B0_DIR_REL, B1_DIR_REL):
        destination = tmp_path / directory
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root / directory, destination)
    (tmp_path / B1_DIR_REL / "selected_slope_profile.json").unlink()
    with pytest.raises(PredecessorFingerprintMismatch):
        build_predecessor_fingerprint_registry(tmp_path)


def test_b1_selected_slope_drift_rejected():
    root = get_repo_root()
    profile = load_json(root / B1_DIR_REL / "selected_slope_profile.json")
    profile["selected_candidate_id"] = "ENDPOINT_H120"
    with pytest.raises(PredecessorFingerprintMismatch, match="B1_SELECTED_SLOPE_DRIFT"):
        validate_b1_selected_profile(profile)


def test_wrong_train_count_rejected():
    train = [f"t{i}" for i in range(EXPECTED_TRAIN_COMMON - 1)]
    validation = [f"v{i}" for i in range(EXPECTED_VALIDATION_COMMON)]
    locked = [f"l{i}" for i in range(EXPECTED_LOCKED_TEST_SEALED)]
    with pytest.raises(CB2Error, match="TRAIN count"):
        validate_population_contract(train, validation, locked)


def test_wrong_validation_count_rejected():
    train = [f"t{i}" for i in range(EXPECTED_TRAIN_COMMON)]
    validation = [f"v{i}" for i in range(EXPECTED_VALIDATION_COMMON - 1)]
    locked = [f"l{i}" for i in range(EXPECTED_LOCKED_TEST_SEALED)]
    with pytest.raises(CB2Error, match="VALIDATION count"):
        validate_population_contract(train, validation, locked)


def test_cross_split_overlap_rejected():
    train = [f"t{i}" for i in range(EXPECTED_TRAIN_COMMON)]
    validation = [f"v{i}" for i in range(EXPECTED_VALIDATION_COMMON)]
    validation[0] = train[0]
    locked = [f"l{i}" for i in range(EXPECTED_LOCKED_TEST_SEALED)]
    with pytest.raises(CB2Error, match="Cross-split overlap"):
        validate_population_contract(train, validation, locked)


def test_locked_test_predictive_access_rejected():
    with pytest.raises(LockedTestPolicyViolation, match="LOCKED_TEST_POLICY_VIOLATION"):
        load_authorized_matrix(repo_root=get_repo_root(), split_role="LOCKED_TEST")


def test_target_leakage_rejected():
    with pytest.raises(CB2Error, match="feature context"):
        validate_feature_context(["CO2", "Temperature", "Humidity", "Occupancy"])


def test_provenance_leakage_rejected():
    with pytest.raises(CB2Error, match="feature context"):
        validate_feature_context(
            ["CO2", "Temperature", "Humidity", "canonical_sample_id"]
        )


def test_scaler_fit_including_validation_rejected():
    with pytest.raises(CB2Error, match="original TRAIN only"):
        fit_train_only_scaler(_bundle("VALIDATION"), fit_population_fingerprint="x")


def test_scaler_fit_including_locked_test_rejected():
    with pytest.raises(CB2Error, match="original TRAIN only"):
        fit_train_only_scaler(_bundle("LOCKED_TEST"), fit_population_fingerprint="x")


def test_arm_specific_scaler_rejected():
    evidence = {
        "scaler_fingerprint": "same",
        "per_arm_scaler_fingerprint": {
            NATURAL_DISTRIBUTION: "same",
            CLASS_WEIGHT_BALANCED: "different",
            BALANCED_RANDOM_OVERSAMPLE: "same",
        },
    }
    with pytest.raises(CB2Error, match="Arm-specific scaler"):
        assert_shared_scaler_fingerprints(evidence)


def test_unauthorized_imbalance_strategy_rejected():
    registry = build_imbalance_strategy_registry()
    registry["strategies"].append({"strategy_id": "SMOTE"})
    registry["strategy_count"] = 4
    with pytest.raises(CB2Error, match="Unauthorized"):
        validate_imbalance_registry(registry)


def test_incorrect_class_weight_calculation_rejected():
    labels = np.asarray([0, 0, 0, 1], dtype=np.int64)
    evidence = {
        "derivation_population": "TRAIN_ONLY",
        "explicit_class_weights": {"VACANT": 1.0, "OCCUPIED": 1.0},
        "validation_rows_used": 0,
        "locked_test_rows_used": 0,
    }
    with pytest.raises(CB2Error, match="Incorrect class-weight"):
        validate_class_weight_evidence(labels, evidence)


def test_validation_derived_class_weights_rejected():
    labels = np.asarray([0, 0, 0, 1], dtype=np.int64)
    with pytest.raises(LockedTestPolicyViolation, match="TRAIN only"):
        compute_balanced_class_weights(labels, derivation_role="VALIDATION")


def test_nondeterministic_oversampling_detected():
    labels = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)
    sample_ids = [f"s{i}" for i in range(labels.size)]
    stored = copy.deepcopy(build_balanced_oversample_plan(labels, sample_ids).evidence)
    stored["appended_minority_sample_ids"] = list(
        reversed(stored["appended_minority_sample_ids"])
    )
    with pytest.raises(CB2Error, match="Non-deterministic"):
        verify_oversample_evidence(labels, sample_ids, stored)


def test_oversampling_using_validation_rejected():
    labels = np.asarray([0, 0, 0, 1], dtype=np.int64)
    with pytest.raises(LockedTestPolicyViolation, match="TRAIN only"):
        build_balanced_oversample_plan(
            labels, ["a", "b", "c", "d"], source_role="VALIDATION"
        )


def test_majority_undersampling_rejected():
    labels = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)
    sample_ids = [f"s{i}" for i in range(labels.size)]
    stored = copy.deepcopy(build_balanced_oversample_plan(labels, sample_ids).evidence)
    stored["majority_undersampling_count"] = 1
    with pytest.raises(CB2Error, match="tampered oversampling"):
        verify_oversample_evidence(labels, sample_ids, stored)


def test_reference_probe_hyperparameter_drift_rejected():
    params = dict(FIXED_LOGISTIC_PARAMETERS)
    params["C"] = 10.0
    with pytest.raises(CB2Error, match="hyperparameter drift"):
        validate_logistic_parameter_contract(params)


def test_stage1_threshold_not_half_rejected():
    registry = build_imbalance_strategy_registry()
    registry["stage1_threshold"] = 0.4
    with pytest.raises(CB2Error, match="exactly 0.5"):
        validate_imbalance_registry(registry)


def test_imbalance_ranking_rule_correctness():
    rows = [
        _metric_row(
            NATURAL_DISTRIBUTION,
            macro_f1=0.7,
            recall=0.9,
            balanced=0.8,
            fpr=0.1,
        ),
        _metric_row(
            CLASS_WEIGHT_BALANCED,
            macro_f1=0.9,
            recall=0.5,
            balanced=0.5,
            fpr=0.5,
        ),
        _metric_row(
            BALANCED_RANDOM_OVERSAMPLE,
            macro_f1=0.8,
            recall=0.99,
            balanced=0.99,
            fpr=0.01,
        ),
    ]
    assert rank_imbalance_strategies(rows)[0]["strategy_id"] == CLASS_WEIGHT_BALANCED


def test_imbalance_tie_break_prefers_simpler_intervention():
    rows = [
        _metric_row(
            strategy_id,
            macro_f1=0.8,
            recall=0.8,
            balanced=0.8,
            fpr=0.1,
        )
        for strategy_id in reversed(AUTHORIZED_STRATEGIES)
    ]
    ranking = rank_imbalance_strategies(rows)
    assert [row["strategy_id"] for row in ranking] == list(AUTHORIZED_STRATEGIES)


def test_threshold_grid_mutation_rejected():
    rows = _threshold_rows()
    rows[0]["threshold"] = 0.04
    with pytest.raises(CB2Error, match="grid mutation"):
        rank_threshold_rows(rows)


def test_threshold_ranking_rule_correctness():
    rows = _threshold_rows()
    target = next(row for row in rows if row["threshold"] == 0.6)
    target["metrics"]["macro_f1"] = 0.9
    assert rank_threshold_rows(rows)[0]["threshold"] == 0.6


def test_threshold_tie_break_prefers_closest_then_lower():
    rows = _threshold_rows()
    ranking = rank_threshold_rows(rows)
    assert ranking[0]["threshold"] == 0.5
    rank_by_threshold = {row["threshold"]: row["rank"] for row in ranking}
    assert rank_by_threshold[0.49] < rank_by_threshold[0.51]


def test_locked_test_threshold_tuning_rejected():
    y = np.asarray([0, 1], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.9], dtype=np.float64)
    with pytest.raises(LockedTestPolicyViolation, match="VALIDATION"):
        build_threshold_sweep(
            y_validation=y,
            probabilities=probabilities,
            sample_ids=["a", "b"],
            population_role="LOCKED_TEST",
        )


def test_probability_mutation_across_thresholds_rejected():
    rows = [
        {"probability_vector_sha256": "a"},
        {"probability_vector_sha256": "b"},
    ]
    with pytest.raises(CB2Error, match="Probability mutation"):
        validate_probability_invariance(rows)


def test_missing_fp_fn_accounting_rejected():
    report = {
        "DOMAIN_FP_FN_COST_RATIO": "UNSPECIFIED",
        "fabricated_weighted_safety_score": False,
        "stage1_default_threshold": {
            strategy_id: {"fp": 1, "fn": 2} for strategy_id in AUTHORIZED_STRATEGIES
        },
    }
    del report["stage1_default_threshold"][NATURAL_DISTRIBUTION]["fn"]
    with pytest.raises(CB2Error, match="Missing FP/FN"):
        validate_fp_fn_report(report)


def test_occupancy_safety_semantic_conflation_rejected():
    contract = {
        "target": "ROOM_OCCUPANCY",
        "probability_class": "OCCUPIED",
        "safety_semantic": "CO2_DANGER",
        "risk_semantic": "NONE",
        "decision_threshold_role": "OFFLINE_OCCUPANCY_CLASSIFICATION",
    }
    with pytest.raises(CB2Error, match="semantic conflation"):
        validate_probability_semantics(contract)


def test_final_production_threshold_overclaim_rejected():
    result = {
        "classification": "REFERENCE_PROBE_THRESHOLD_ONLY",
        "reference_threshold_production_final": True,
        "TRANSFER_TO_FUTURE_ARCHITECTURES": "NOT_ASSUMED",
    }
    with pytest.raises(CB2Error, match="overclaim"):
        validate_reference_threshold_claims(result)


def test_predecessor_fingerprint_mismatch_rejected():
    root = get_repo_root()
    stored = build_predecessor_fingerprint_registry(root)
    stored["entries"][0]["sha256"] = "0" * 64
    with pytest.raises(PredecessorFingerprintMismatch, match="FINGERPRINT_MISMATCH"):
        verify_stored_predecessor_registry(root, stored)


def test_absolute_path_rejected():
    assert assert_no_forbidden_path_markers('{"path":"/Users/example/private"}')
    assert assert_no_forbidden_path_markers('{"path":"datasets/co2/relative.json"}') == []


def test_clean_historical_c_b2_validator_remains_pass_with_warnings():
    result = validate_c_b2(
        get_repo_root(), rerun_determinism=False, run_predecessors=False
    )
    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["errors"] == []


def test_later_co2_file_is_same_track_not_contamination():
    audit = audit_path_scope(["datasets/co2/tflite_equivalence.py"])
    assert audit["errors"] == []
    assert audit["path_ownership_classification"]["datasets/co2/tflite_equivalence.py"] == CO2_SAME_TRACK
    assert audit["same_track_later_phase_paths"] == ["datasets/co2/tflite_equivalence.py"]


def test_later_co2_manifest_is_same_track_not_contamination():
    path = "datasets/co2/manifests/c_b5_robustness/robustness_report.json"
    audit = audit_path_scope([path])
    assert audit["errors"] == []
    assert audit["path_ownership_classification"][path] == CO2_SAME_TRACK


def test_later_co2_unique_commit_is_allowed():
    path = "models/co2/candidates/c_b5/robust_candidate.tflite"
    audit = audit_path_scope([], {"c0ffee1234567890": [path]})
    assert audit["errors"] == []
    assert audit["path_ownership_classification"][path] == CO2_SAME_TRACK


def test_future_co2_script_test_and_report_namespaces_are_allowed():
    paths = [
        "scripts/validate_co2_c_b6.py",
        "tests/test_co2_c_b6.py",
        "docs/reports/co2_c_b6.md",
        "inference/co2_c_b6.py",
    ]
    audit = audit_path_scope(paths)
    assert audit["errors"] == []
    assert all(
        audit["path_ownership_classification"][path] == CO2_SAME_TRACK
        for path in paths
    )


@pytest.mark.parametrize(
    ("path", "ownership"),
    [
        ("datasets/mmwave/future_model.json", MMWAVE_OTHER_TRACK),
        ("devices/mmwave/src/provider.py", MMWAVE_OTHER_TRACK),
        ("datasets/thermal/future_frame.json", THERMAL_OTHER_TRACK),
        ("devices/thermal44/src/provider.py", THERMAL_OTHER_TRACK),
        ("shared/contracts/sensor.py", INTEGRATION_OTHER_TRACK),
        ("integrated_node/fusion.py", INTEGRATION_OTHER_TRACK),
    ],
)
def test_other_track_paths_are_rejected(path, ownership):
    audit = audit_path_scope([path])
    assert audit["path_ownership_classification"][path] == ownership
    assert any(
        "PARALLEL_TRACK_BRANCH_CONTAMINATION" in error
        for error in audit["errors"]
    )


def test_other_track_unique_commit_is_rejected():
    path = "scripts/run_mmwave_future.py"
    audit = audit_path_scope([], {"deadbeef12345678": [path]})
    assert any(
        "PARALLEL_TRACK_BRANCH_CONTAMINATION" in error
        for error in audit["errors"]
    )


def test_unauthorized_shared_root_path_is_rejected():
    path = "README.md"
    audit = audit_path_scope([path])
    assert audit["path_ownership_classification"][path] == SHARED_OR_UNAUTHORIZED
    assert any("Unauthorized non-C-B2 path" in error for error in audit["errors"])


def test_c_b2_locked_artifact_mutation_is_rejected_even_same_track():
    path = f"{ARTIFACT_DIR_REL}/reference_threshold_result.json"
    audit = audit_path_scope([path])
    assert audit["path_ownership_classification"][path] == C_B2_ARTIFACT_DRIFT
    assert any("C_B2_ARTIFACT_DRIFT" in error for error in audit["errors"])


def test_c_b2_tooling_path_is_owned_by_corrective_validator():
    path = "scripts/validate_co2_imbalance_calibration.py"
    audit = audit_path_scope([path], {"cafebabe12345678": [path]})
    assert audit["errors"] == []
    assert audit["path_ownership_classification"][path] == C_B2_OWNED


def test_production_co2_asset_remains_protected():
    path = "models/co2/co2_occupancy_int8_v0.1.0.tflite"
    audit = audit_path_scope([path])
    assert audit["path_ownership_classification"][path] == SHARED_OR_UNAUTHORIZED
    assert any("Unauthorized non-C-B2 path" in error for error in audit["errors"])


def test_ece_contract_is_ten_equal_width_bins():
    y = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.0, 0.1, 0.9, 1.0], dtype=np.float64)
    diagnostic = expected_calibration_error(y, probabilities)
    assert diagnostic["bin_count"] == 10
    assert len(diagnostic["bins"]) == 10
    assert sum(row["sample_count"] for row in diagnostic["bins"]) == 4


def test_artifacts_enforce_locked_test_and_scope_boundaries():
    root = get_repo_root()
    directory = root / ARTIFACT_DIR_REL
    leakage = load_json(directory / "leakage_audit.json")
    generation = load_json(directory / "generation_metadata.json")
    reference = load_json(directory / "reference_threshold_result.json")
    assert leakage["locked_test_feature_access"] == 0
    assert leakage["locked_test_target_access"] == 0
    assert leakage["locked_test_predictions"] == 0
    assert leakage["locked_test_metrics"] == 0
    assert generation["architecture_comparison_performed"] is False
    assert generation["multi_seed_comparison_performed"] is False
    assert generation["final_feature_selection_performed"] is False
    validate_reference_threshold_claims(reference)


def test_deterministic_regeneration_is_bit_identical():
    root = get_repo_root()
    directory = root / ARTIFACT_DIR_REL
    before = {
        path.name: compute_sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }
    run_imbalance_calibration(root)
    after = {
        path.name: compute_sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }
    assert before == after


def test_standalone_validator_recomputes_c_b2_contract():
    result = validate_c_b2(
        get_repo_root(), rerun_determinism=False, run_predecessors=False
    )
    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["errors"] == []
    assert result["locked_test_predictions"] == 0
    assert result["locked_test_metrics"] == 0
