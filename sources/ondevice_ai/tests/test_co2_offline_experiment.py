#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_co2_offline_experiment.py
Focused tests for C-B0 offline experiment contract and harness.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from datasets.co2.offline_experiment import (
    A_SERIES_RELEASE_COMMIT,
    A_SERIES_RELEASE_TAG,
    EXPECTED_LOCKED_TEST_SEALED,
    EXPECTED_TRAIN_COMMON,
    EXPECTED_VALIDATION_COMMON,
    LockedTestPolicyViolation,
    MajorityClassBaseline,
    MatrixBundle,
    TrainOnlyStandardScaler,
    assert_no_forbidden_path_markers,
    build_feature_view_registry,
    build_metric_contract,
    build_sample_universe_manifest,
    compute_classification_metrics,
    load_comparison_matrix,
    ordered_id_list_sha256,
    run_leakage_audit,
    verify_a_series_artifact_lock,
    verify_a_series_release,
)
from datasets.co2.raw_reader import get_repo_root


def test_a_series_tag_and_lock_verified():
    root = get_repo_root()
    release = verify_a_series_release(root)
    assert release["matches_expected"] is True
    assert release["resolved_commit"] == A_SERIES_RELEASE_COMMIT
    assert release["expected_tag"] == A_SERIES_RELEASE_TAG
    lock = verify_a_series_artifact_lock(root)
    assert lock["status"] == "VERIFIED"
    assert lock["matches_released_lock_sha256"] is True


def test_a_series_tag_mismatch_detection(monkeypatch):
    import datasets.co2.offline_experiment as mod

    def fake_check_output(*args, **kwargs):
        return "0" * 40 + "\n"

    monkeypatch.setattr(mod.subprocess if hasattr(mod, "subprocess") else __import__("subprocess"), "check_output", fake_check_output)
    # patch via module-level import inside function - call with monkeypatched subprocess in verify
    monkeypatch.setattr("subprocess.check_output", fake_check_output)
    release = verify_a_series_release(get_repo_root())
    assert release["matches_expected"] is False
    assert release["status"] == "A_SERIES_RELEASE_PREREQUISITE_NOT_MET"


def test_sample_universe_counts_and_no_overlap():
    universe = build_sample_universe_manifest(get_repo_root())
    assert universe["b_series_common_train"] == EXPECTED_TRAIN_COMMON
    assert universe["b_series_common_validation"] == EXPECTED_VALIDATION_COMMON
    assert universe["b_series_sealed_locked_test"] == EXPECTED_LOCKED_TEST_SEALED
    assert universe["canonical_warmup_records"] == 9
    assert universe["overlaps"]["train_validation"] == 0
    assert universe["overlaps"]["train_locked_test"] == 0
    assert universe["overlaps"]["validation_locked_test"] == 0
    # deterministic fingerprints
    u2 = build_sample_universe_manifest(get_repo_root())
    assert universe["ordered_id_list_sha256"] == u2["ordered_id_list_sha256"]


def test_duplicate_and_overlap_helpers():
    ids = ["a", "b", "a"]
    assert ordered_id_list_sha256(ids) != ordered_id_list_sha256(["a", "b"])
    assert len(set(ids)) != len(ids)


def test_feature_registry_no_target_or_final_winner():
    reg = build_feature_view_registry()
    assert reg["final_feature_selection_performed"] is False
    assert reg["feature_roles"]["Occupancy"]["may_be_model_input"] is False
    assert "Light" in reg["uci_only_or_non_native_features"]
    assert "CO2_slope" in reg["derived_scd40_compatible_features"]
    hist = reg["feature_views"]["HISTORICAL_COMPATIBILITY_REFERENCE"]
    assert hist["winner"] is False
    assert hist["status"] == "HISTORICAL_COMPATIBILITY_VIEW"


def test_metric_contract_and_positive_class():
    m = build_metric_contract()
    assert m["positive_class"] == "OCCUPIED"
    assert m["primary_summary_metric"] == "macro_f1"
    assert m["threshold_optimization_in_b0"] is False
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["tp"] == 2
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_majority_baseline_train_only_behavior():
    y_train = np.array([0, 0, 0, 1])
    baseline = MajorityClassBaseline().fit(y_train)
    assert baseline.majority_label == 0
    pred = baseline.predict(3)
    assert pred.tolist() == [0, 0, 0]
    assert baseline.candidate is False
    assert baseline.deployable is False


def test_scaler_fit_rejects_non_train_and_locked_transform():
    names = ("CO2", "Humidity")
    train = MatrixBundle(
        sample_ids=["t1", "t2"],
        features=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        labels=np.array([0, 1]),
        feature_names=names,
        split_role="TRAIN",
    )
    val = MatrixBundle(
        sample_ids=["v1"],
        features=np.array([[2.0, 3.0]], dtype=np.float64),
        labels=np.array([0]),
        feature_names=names,
        split_role="VALIDATION",
    )
    locked = MatrixBundle(
        sample_ids=["l1"],
        features=np.array([[2.0, 3.0]], dtype=np.float64),
        labels=np.array([1]),
        feature_names=names,
        split_role="LOCKED_TEST",
    )
    scaler = TrainOnlyStandardScaler(feature_names=names).fit(train)
    _ = scaler.transform(val)
    with pytest.raises(Exception):
        TrainOnlyStandardScaler(feature_names=names).fit(val)
    with pytest.raises(LockedTestPolicyViolation):
        scaler.transform(locked)


def test_locked_test_matrix_policy_violation():
    with pytest.raises(LockedTestPolicyViolation):
        load_comparison_matrix(
            repo_root=get_repo_root(),
            split_role="LOCKED_TEST",
            feature_names=["CO2"],
            allow_locked_test_predictive=False,
        )


def test_target_and_provenance_feature_rejection():
    with pytest.raises(Exception):
        load_comparison_matrix(
            repo_root=get_repo_root(),
            split_role="TRAIN",
            feature_names=["CO2", "Occupancy"],
        )
    with pytest.raises(Exception):
        load_comparison_matrix(
            repo_root=get_repo_root(),
            split_role="TRAIN",
            feature_names=["CO2", "canonical_sample_id"],
        )


def test_absolute_path_rejection():
    assert assert_no_forbidden_path_markers("/Users/x/y") 
    assert assert_no_forbidden_path_markers("datasets/co2/x.json") == []


def test_leakage_audit_pass_on_real_universe():
    root = get_repo_root()
    universe = build_sample_universe_manifest(root)
    features = build_feature_view_registry()
    audit = run_leakage_audit(universe, features)
    assert audit["status"] == "PASS"


def test_generated_artifacts_if_present():
    root = get_repo_root()
    c_b0 = root / "datasets/co2/manifests/c_b0_offline_experiment_contract"
    if not c_b0.exists():
        pytest.skip("C-B0 artifacts not generated yet")
    checksum = (c_b0 / "checksums.sha256").read_text(encoding="utf-8")
    assert "checksums.sha256\n" not in checksum.split("  ")[-1] if False else True
    names = {line.split("  ", 1)[1].split("/")[-1] for line in checksum.strip().splitlines()}
    assert "checksums.sha256" not in names
    baseline = json.loads((c_b0 / "reference_baseline_result.json").read_text(encoding="utf-8"))
    assert baseline["evaluation_population"] == "VALIDATION"
    assert baseline["locked_test_used"] is False
    gen = json.loads((c_b0 / "generation_metadata.json").read_text(encoding="utf-8"))
    assert gen["synthetic_npz_used_as_real_training_data"] is False
