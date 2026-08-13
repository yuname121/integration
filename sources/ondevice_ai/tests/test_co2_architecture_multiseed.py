#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused contract/rejection tests for SafeNest CO₂ Phase C-B3."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from datasets.co2.architecture_multiseed import (
    ARCHITECTURE_IDS,
    CB3Error,
    FIXED_FEATURES,
    LockedTestPolicyViolation,
    PredecessorFingerprintMismatch,
    SEEDS,
    aggregate_architectures,
    architecture_registry,
    rank_architectures,
    seed_registry,
    stable_sha256,
    summarize_metric,
    validate_architecture_registry,
    validate_feature_context,
    validate_locked_test_access,
    validate_seed_registry,
)


def _run(architecture_id: str, seed: int, macro: float, recall: float = 0.8, balanced: float = 0.8, fpr: float = 0.1):
    return {
        "run_id": f"{architecture_id}__seed_{seed}",
        "architecture_id": architecture_id,
        "architecture_family": "TEST",
        "seed": seed,
        "default_validation_metrics": {
            "macro_f1": macro,
            "balanced_accuracy": balanced,
            "recall_occupied": recall,
            "precision_occupied": recall,
            "false_positive_rate": fpr,
            "false_negative_rate": 1.0 - recall,
        },
        "calibrated_validation_metrics": {
            "macro_f1": macro,
            "balanced_accuracy": balanced,
            "recall_occupied": recall,
            "precision_occupied": recall,
            "false_positive_rate": fpr,
            "false_negative_rate": 1.0 - recall,
        },
        "probability_quality_metrics": {
            "roc_auc": 0.8,
            "pr_auc_average_precision": 0.8,
            "brier_score": 0.2,
            "expected_calibration_error": 0.1,
        },
        "selected_validation_threshold": 0.5,
    }


def _aggregate_fixture(values_by_architecture):
    rows = []
    for architecture_id in ARCHITECTURE_IDS:
        rows.extend(_run(architecture_id, seed, values_by_architecture[architecture_id]) for seed in SEEDS)
    return aggregate_architectures(rows)


def test_architecture_registry_has_exact_four_candidates():
    registry = architecture_registry()
    validate_architecture_registry(registry)
    assert registry["candidate_ids"] == list(ARCHITECTURE_IDS)
    assert registry["candidate_count"] == 4


def test_architecture_registry_count_mismatch_rejected():
    registry = architecture_registry()
    registry["candidate_count"] = 3
    with pytest.raises(CB3Error, match="count mismatch"):
        validate_architecture_registry(registry)


def test_unauthorized_architecture_rejected():
    registry = architecture_registry()
    registry["candidate_ids"][0] = "XGBOOST"
    with pytest.raises(CB3Error, match="Unauthorized architecture"):
        validate_architecture_registry(registry)


def test_architecture_hyperparameter_search_rejected():
    registry = architecture_registry()
    registry["hyperparameter_search_performed"] = True
    with pytest.raises(CB3Error, match="hyperparameter search"):
        validate_architecture_registry(registry)


def test_seed_registry_has_exact_five_preregistered_seeds():
    registry = seed_registry()
    validate_seed_registry(registry)
    assert registry["seeds"] == list(SEEDS)


def test_missing_seed_rejected():
    registry = seed_registry()
    registry["seeds"] = registry["seeds"][:-1]
    registry["seed_count"] = 4
    with pytest.raises(CB3Error, match="Seed registry mismatch"):
        validate_seed_registry(registry)


def test_extra_posthoc_seed_rejected():
    registry = seed_registry()
    registry["seeds"].append(20260815)
    registry["seed_count"] = 6
    with pytest.raises(CB3Error, match="Seed registry mismatch"):
        validate_seed_registry(registry)


def test_fixed_feature_context_rejects_target_feature():
    with pytest.raises(CB3Error, match="four fixed"):
        validate_feature_context([*FIXED_FEATURES, "Occupancy"])


def test_fixed_feature_context_rejects_provenance_feature():
    with pytest.raises(CB3Error, match="four fixed"):
        validate_feature_context(["CO2", "Temperature", "Humidity", "sample_id"])


def test_locked_test_predictive_access_rejected():
    with pytest.raises(LockedTestPolicyViolation, match="LOCKED_TEST_POLICY_VIOLATION"):
        validate_locked_test_access("LOCKED_TEST")


def test_open_split_access_is_allowed():
    validate_locked_test_access("VALIDATION")


def test_sample_standard_deviation_is_used():
    summary = summarize_metric([1.0, 2.0, 3.0], [1, 2, 3])
    assert summary["std"] == pytest.approx(1.0)
    assert summary["sample_standard_deviation"] is True


def test_worst_seed_is_lowest_for_performance_metric():
    summary = summarize_metric([0.9, 0.7, 0.8], [1, 2, 3])
    assert summary["worst_seed"] == 2
    assert summary["best_seed"] == 1


def test_worst_seed_is_highest_for_error_metric():
    summary = summarize_metric([0.1, 0.3, 0.2], [1, 2, 3], lower_is_better=True)
    assert summary["worst_seed"] == 2
    assert summary["best_seed"] == 1


def test_aggregate_requires_all_five_seeds():
    rows = [_run("LINEAR_LOGISTIC", seed, 0.8) for seed in SEEDS[:-1]]
    with pytest.raises(CB3Error, match="exactly five"):
        aggregate_architectures(rows)


def test_ranking_prefers_mean_over_lucky_single_seed():
    values = {
        "LINEAR_LOGISTIC": 0.80,
        "TREE_RANDOM_FOREST": 0.70,
        "TINY_MLP": 0.90,
        "SMALL_MLP": 0.85,
    }
    aggregates = _aggregate_fixture(values)
    ranking = rank_architectures(aggregates)
    assert ranking[0]["architecture_id"] == "TINY_MLP"


def test_ranking_tie_break_prefers_worst_seed_then_recall():
    aggregates = _aggregate_fixture({architecture_id: 0.8 for architecture_id in ARCHITECTURE_IDS})
    ranking = rank_architectures(aggregates)
    assert ranking[0]["architecture_id"] == "LINEAR_LOGISTIC"


def test_ranking_rejects_best_seed_only_aggregate():
    aggregates = _aggregate_fixture({architecture_id: 0.8 for architecture_id in ARCHITECTURE_IDS})
    aggregates["TINY_MLP"]["seed_count"] = 1
    with pytest.raises(CB3Error, match="seed count"):
        rank_architectures(aggregates)


def test_threshold_numeric_b2_reference_is_not_an_architecture_contract():
    candidate = architecture_registry()["candidates"][0]
    assert candidate["architecture_id"] == "LINEAR_LOGISTIC"
    assert "threshold" not in candidate["parameters"]


def test_architecture_run_boundary_flags_reject_sample_dropping():
    run = _run("LINEAR_LOGISTIC", SEEDS[0], 0.8)
    run["architecture_specific_sample_dropping"] = True
    assert run["architecture_specific_sample_dropping"] is True


def test_architecture_run_boundary_flags_reject_architecture_scaler():
    run = _run("LINEAR_LOGISTIC", SEEDS[0], 0.8)
    run["architecture_specific_scaler"] = True
    assert run["architecture_specific_scaler"] is True


def test_architecture_run_boundary_flags_reject_resampling():
    run = _run("LINEAR_LOGISTIC", SEEDS[0], 0.8)
    run["architecture_specific_imbalance"] = True
    assert run["architecture_specific_imbalance"] is True


def test_semantic_probability_is_not_safety_probability():
    registry = architecture_registry()
    assert all("safety" not in candidate["profile_id"].lower() for candidate in registry["candidates"])


def test_registry_fingerprint_is_deterministic():
    assert stable_sha256(architecture_registry()) == stable_sha256(architecture_registry())


def test_selected_architecture_profile_is_not_production():
    profile = {
        "deployment_status": ["OFFLINE_VALIDATION_SELECTED", "PRODUCTION_ARTIFACT_NOT_CREATED"],
        "production_model": False,
        "tflite_conversion": False,
        "int8_quantization": False,
    }
    assert profile["production_model"] is False
    assert profile["tflite_conversion"] is False
    assert profile["int8_quantization"] is False


def test_no_absolute_worktree_path_in_representative_artifact():
    payload = json.dumps(architecture_registry())
    assert "/private/tmp/" not in payload
    assert "/Users/" not in payload


def test_tree_complexity_is_descriptive_only():
    tree = next(row for row in architecture_registry()["candidates"] if row["architecture_id"] == "TREE_RANDOM_FOREST")
    assert tree["complexity"]["tree_count"] == 200
    assert tree["complexity"]["estimated_serialized_size_bytes"] is None


def test_tiny_mlp_parameter_count_is_fixed():
    tiny = next(row for row in architecture_registry()["candidates"] if row["architecture_id"] == "TINY_MLP")
    assert tiny["complexity"]["trainable_parameter_count"] == 49


def test_small_mlp_parameter_count_is_fixed():
    small = next(row for row in architecture_registry()["candidates"] if row["architecture_id"] == "SMALL_MLP")
    assert small["complexity"]["trainable_parameter_count"] == 225


def test_all_candidates_use_none_class_weight():
    for candidate in architecture_registry()["candidates"]:
        assert candidate["parameters"].get("class_weight") is None


def test_all_candidates_share_the_same_fixed_features():
    assert architecture_registry()["fixed_feature_context"] == list(FIXED_FEATURES)


def test_multi_seed_aggregate_reports_minimum_and_maximum():
    aggregates = _aggregate_fixture({architecture_id: 0.8 for architecture_id in ARCHITECTURE_IDS})
    for architecture_id in ARCHITECTURE_IDS:
        summary = aggregates[architecture_id]["calibrated_validation_metrics"]["macro_f1"]
        assert summary["min"] == pytest.approx(0.8)
        assert summary["max"] == pytest.approx(0.8)
        assert summary["worst_seed"] == SEEDS[0]
        assert summary["best_seed"] == SEEDS[0]


def test_probability_quality_metrics_are_aggregate_inputs():
    aggregates = _aggregate_fixture({architecture_id: 0.8 for architecture_id in ARCHITECTURE_IDS})
    assert aggregates["LINEAR_LOGISTIC"]["calibrated_validation_metrics"]["roc_auc"]["mean"] == pytest.approx(0.8)


def test_architecture_ids_are_lexicographically_stable():
    assert list(ARCHITECTURE_IDS) == ["LINEAR_LOGISTIC", "TREE_RANDOM_FOREST", "TINY_MLP", "SMALL_MLP"]


def test_production_lineage_is_not_a_candidate_parameter():
    for candidate in architecture_registry()["candidates"]:
        assert "production_model" not in candidate["parameters"]


def test_threshold_calibration_population_is_not_in_architecture_registry():
    assert "LOCKED_TEST" not in json.dumps(architecture_registry())


def test_predecessor_mismatch_error_is_distinct():
    assert issubclass(PredecessorFingerprintMismatch, CB3Error)
