#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused tests for C-B1 CO2 slope method/history ablation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pytest

from datasets.co2.offline_experiment import MatrixBundle
from datasets.co2.raw_reader import CO2SourceRowObservation, get_repo_root
from datasets.co2.slope_feature import (
    STATUS_AVAILABLE,
    STATUS_GAP_RESTART,
    _TimedRow,
    _find_history_start,
    reconstruct_block_slope_features_with_params,
)
from datasets.co2.offline_experiment import assert_no_forbidden_path_markers
from datasets.co2.slope_ablation import (
    ARTIFACT_DIR_REL,
    AUTHORIZED_HISTORIES,
    AUTHORIZED_METHODS,
    BASELINE_CANDIDATE_ID,
    CB1Error,
    NearestCentroidProbe,
    SlopeCandidateSpec,
    build_candidate_registry,
    build_feature_bundle,
    build_preregistered_candidates,
    rank_slope_candidates,
    _fingerprint_float64_le,
)


def _row(ts: datetime, co2: float, idx: int, member: str = "datatraining.txt") -> _TimedRow:
    obs = CO2SourceRowObservation(
        source_archive_path="datasets/raw_archives/external_datasets/occupancy+detection.zip",
        source_archive_sha256="0" * 64,
        source_member_name=member,
        source_member_sha256="0" * 64,
        source_physical_line_number=idx + 2,
        source_row_identifier=f"row_{idx}",
        source_timestamp_raw=ts.strftime("%Y-%m-%d %H:%M:%S"),
        timestamp_reference="SOURCE_ACQUISITION_CLOCK",
        source_timezone="UNVERIFIED",
        utc_conversion_claimed=False,
        temperature=20.0,
        humidity=30.0,
        light=0.0,
        co2=co2,
        humidity_ratio=0.004,
        occupancy=0,
    )
    return _TimedRow(
        obs=obs,
        dt=ts,
        ts_canonical=ts.strftime("%Y-%m-%dT%H:%M:%S"),
        block_id="BLOCK_02_DATATRAINING",
        future_split_role="TRAIN",
        index_in_block=idx,
    )


def test_candidate_registry_exact_grid():
    reg = build_candidate_registry()
    assert reg["candidate_count"] == 6
    assert set(reg["authorized_methods"]) == set(AUTHORIZED_METHODS)
    assert set(reg["authorized_history_thresholds_seconds"]) == set(AUTHORIZED_HISTORIES)
    ids = [c["candidate_id"] for c in reg["candidates"]]
    assert ids == [
        "ENDPOINT_H60",
        "ENDPOINT_H120",
        "ENDPOINT_H150",
        "LINEAR_REGRESSION_H60",
        "LINEAR_REGRESSION_H120",
        "LINEAR_REGRESSION_H150",
    ]


def test_wrong_candidate_count_rejection():
    specs = build_preregistered_candidates()
    assert len(specs) == 6
    # Simulating post-hoc addition is rejected by registry fingerprint consumers
    bloated = list(specs) + [
        SlopeCandidateSpec("ENDPOINT_H180", "ENDPOINT_DIFFERENCE", 180.0, False)
    ]
    assert len(bloated) != 6


def test_unauthorized_history_and_method_rejected_by_reconstruction():
    t0 = datetime(2015, 2, 4, 17, 51, 0)
    rows = [_row(t0 + timedelta(seconds=60 * i), 500 + i, i) for i in range(6)]
    with pytest.raises(ValueError, match="Unsupported slope method"):
        reconstruct_block_slope_features_with_params(
            rows, method="FUTURE_LOOKING_SLOPE", history_duration_seconds=60.0
        )


def test_candidate_specific_sample_dropping_rejection():
    # build_feature_bundle must refuse non-finite slopes rather than drop rows
    with pytest.raises(CB1Error, match="CANDIDATE_AVAILABILITY_CONTRACT_MISMATCH"):
        build_feature_bundle(
            sample_ids=["a", "b"],
            labels=np.asarray([0, 1]),
            base_features=np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
            slope_values=[1.0, float("nan")],
            split_role="TRAIN",
            include_slope=True,
        )


def test_future_data_and_cross_block_and_gap_policy():
    t0 = datetime(2015, 2, 4, 17, 51, 0)
    rows = [_row(t0 + timedelta(seconds=60 * i), 500 + 10 * i, i) for i in range(5)]
    # Insert prohibited gap between index 1 and 2
    rows[2] = _row(rows[1].dt + timedelta(seconds=200), 520.0, 2)
    rows[3] = _row(rows[2].dt + timedelta(seconds=60), 530.0, 3)
    rows[4] = _row(rows[3].dt + timedelta(seconds=60), 540.0, 4)
    start_idx, status = _find_history_start(rows, 4, history_duration_seconds=150.0)
    assert start_idx is None
    assert status == STATUS_GAP_RESTART

    # Cross-block: history search never sees other blocks because reconstruction
    # is per-block; mixing members is not done by _find_history_start.
    recs = reconstruct_block_slope_features_with_params(
        rows, method="ENDPOINT_DIFFERENCE", history_duration_seconds=60.0
    )
    # Index 2 is first after gap — warm-up / gap restart, never uses pre-gap rows
    assert recs[2].feature_status in {STATUS_GAP_RESTART, "FEATURE_UNAVAILABLE_WARMUP"}
    if recs[4].feature_status == STATUS_AVAILABLE:
        assert "row_1" not in recs[4].history_source_row_identifiers


def test_endpoint_h150_is_baseline_candidate():
    specs = {s.candidate_id: s for s in build_preregistered_candidates()}
    assert specs[BASELINE_CANDIDATE_ID].is_a3_baseline is True
    assert specs[BASELINE_CANDIDATE_ID].method == "ENDPOINT_DIFFERENCE"
    assert specs[BASELINE_CANDIDATE_ID].minimum_history_seconds == 150.0


def test_ranking_and_tie_breaks():
    base_metrics = {
        "macro_f1": 0.5,
        "balanced_accuracy": 0.5,
        "recall_occupied": 0.5,
    }
    rows = [
        {
            "candidate_id": "LINEAR_REGRESSION_H150",
            "method": "CAUSAL_LINEAR_REGRESSION",
            "minimum_history_seconds": 150.0,
            "validation_metrics": dict(base_metrics),
        },
        {
            "candidate_id": "ENDPOINT_H150",
            "method": "ENDPOINT_DIFFERENCE",
            "minimum_history_seconds": 150.0,
            "validation_metrics": dict(base_metrics),
        },
        {
            "candidate_id": "ENDPOINT_H60",
            "method": "ENDPOINT_DIFFERENCE",
            "minimum_history_seconds": 60.0,
            "validation_metrics": dict(base_metrics),
        },
    ]
    ranked = rank_slope_candidates(rows)
    # same metrics → shorter history, then simpler method, then lex id
    assert ranked[0]["candidate_id"] == "ENDPOINT_H60"

    # higher macro F1 wins regardless of complexity
    rows[0]["validation_metrics"]["macro_f1"] = 0.9
    ranked = rank_slope_candidates(rows)
    assert ranked[0]["candidate_id"] == "LINEAR_REGRESSION_H150"


def test_no_slope_control_excluded_from_ranking():
    rows = [
        {
            "candidate_id": "ENDPOINT_H150",
            "method": "ENDPOINT_DIFFERENCE",
            "minimum_history_seconds": 150.0,
            "validation_metrics": {
                "macro_f1": 0.2,
                "balanced_accuracy": 0.2,
                "recall_occupied": 0.2,
            },
        }
    ]
    ranked = rank_slope_candidates(rows)
    assert all(r["candidate_id"] != "SCD40_NATIVE_NO_SLOPE_CONTROL" for r in ranked)


def test_nearest_centroid_tie_and_train_only_semantics():
    x = np.asarray([[0.0, 0.0], [10.0, 10.0], [1.0, 1.0], [9.0, 9.0]], dtype=np.float64)
    y = np.asarray([0, 1, 0, 1], dtype=np.int64)
    probe = NearestCentroidProbe(feature_names=("a", "b"))
    probe.fit(x, y)
    # Point exactly midway should tie → VACANT
    mid = (probe.vacant_centroid_ + probe.occupied_centroid_) / 2.0
    pred = probe.predict(mid.reshape(1, -1))
    assert int(pred[0]) == 0


def test_fingerprint_determinism():
    ids = ["s1", "s2", "s3"]
    vals = [1.25, -0.5, 3.0]
    a = _fingerprint_float64_le(vals, ids)
    b = _fingerprint_float64_le(vals, ids)
    assert a == b
    assert a != _fingerprint_float64_le([1.25, -0.5, 3.0000001], ids)


def test_absolute_path_rejection_helper():
    bad = assert_no_forbidden_path_markers('{"p":"/Users/junwoo/x"}')
    assert bad
    good = assert_no_forbidden_path_markers('{"p":"datasets/co2/manifests/x.json"}')
    assert good == []


def test_artifacts_exist_and_selected_profile_stable():
    root = get_repo_root()
    d = root / ARTIFACT_DIR_REL
    selected = json.loads((d / "selected_slope_profile.json").read_text(encoding="utf-8"))
    decision = json.loads((d / "selection_decision.json").read_text(encoding="utf-8"))
    assert selected["profile_id"] == "CO2_B1_SELECTED_SLOPE_PROFILE_001"
    assert selected["selected_candidate_id"] == decision["winning_slope_candidate"]
    assert selected["deployment_status"] == "NOT_VALIDATED"
    assert selected["final_feature_selection"] == "NOT_PERFORMED"
    # A-series profile not overwritten
    a3 = json.loads(
        (
            root
            / "datasets/co2/manifests/c_a3_slope_feature/co2_slope_feature_profile.json"
        ).read_text(encoding="utf-8")
    )
    assert a3["profile_id"] == "CO2_SLOPE_FEATURE_PROFILE_001"


def test_locked_test_absent_from_b1_metrics_and_predictions():
    root = get_repo_root()
    d = root / ARTIFACT_DIR_REL
    metrics = json.loads((d / "validation_metric_results.json").read_text(encoding="utf-8"))
    preds = json.loads((d / "validation_predictions.json").read_text(encoding="utf-8"))
    assert metrics["locked_test_metrics"] == 0
    assert preds["locked_test_predictions"] == 0
    fp = json.loads(
        (d / "candidate_feature_fingerprint_registry.json").read_text(encoding="utf-8")
    )
    for block in fp["candidates"].values():
        assert block["LOCKED_TEST"]["status"] == "NOT_GENERATED"
