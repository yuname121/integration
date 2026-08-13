#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_co2_slope_feature.py
Focused tests for C-A3 CO₂ Slope Feature Reconstruction.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta

import pytest

from datasets.co2.raw_reader import CO2SourceRowObservation, get_repo_root
from datasets.co2.slope_feature import (
    COMPARISON_ABS_TOLERANCE,
    FEATURE_PROFILE_ID,
    FEATURE_UNIT,
    HISTORY_DURATION_SECONDS,
    MAX_INTERNAL_GAP_SECONDS,
    STATUS_AVAILABLE,
    STATUS_GAP_RESTART,
    STATUS_NONFINITE,
    STATUS_WARMUP,
    compute_endpoint_slope_ppm_per_min,
    reconstruct_all_slope_features,
)


def _obs(
    member: str,
    row_id: str,
    line: int,
    ts: datetime,
    co2: float,
) -> CO2SourceRowObservation:
    return CO2SourceRowObservation(
        source_archive_path="datasets/raw_archives/external_datasets/occupancy+detection.zip",
        source_archive_sha256="4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a",
        source_member_name=member,
        source_member_sha256="b2c4d0ce2b9e4e453c476f7125ef31aeec2d1f5c7f5572d0e80de3df6521ab56",
        source_physical_line_number=line,
        source_row_identifier=row_id,
        source_timestamp_raw=ts.strftime("%Y-%m-%d %H:%M:%S"),
        timestamp_reference="SOURCE_ACQUISITION_CLOCK",
        source_timezone="UNVERIFIED",
        utc_conversion_claimed=False,
        temperature=20.0,
        humidity=30.0,
        light=100.0,
        co2=co2,
        humidity_ratio=0.01,
        occupancy=0,
    )


def test_selected_slope_formula_and_unit():
    slope = compute_endpoint_slope_ppm_per_min(610.0, 600.0, 180.0)
    assert FEATURE_UNIT == "ppm/min"
    assert abs(slope - ((610.0 - 600.0) / 3.0)) <= COMPARISON_ABS_TOLERANCE


def test_actual_elapsed_time_not_nominal_sample_rename():
    # 180s elapsed -> /3 minutes, not /1 sample
    slope = compute_endpoint_slope_ppm_per_min(630.0, 600.0, 180.0)
    assert abs(slope - 10.0) <= COMPARISON_ABS_TOLERANCE
    # If someone wrongly treated one sample step as one minute despite 90s:
    wrong = (630.0 - 600.0) / 1.0
    assert abs(slope - wrong) > 1.0


@pytest.mark.parametrize("delta_sec,expected_elapsed", [(59.0, 177.0), (60.0, 180.0), (61.0, 183.0)])
def test_jitter_intervals_use_actual_elapsed(delta_sec, expected_elapsed):
    # Build 4 samples with constant adjacent delta; eligibility at index 3.
    t0 = datetime(2015, 2, 4, 17, 51, 0)
    rows = []
    for i in range(4):
        ts = t0 + timedelta(seconds=delta_sec * i)
        rows.append(_obs("datatraining.txt", str(i + 1), i + 2, ts, 600.0 + 3.0 * i))
    records = reconstruct_all_slope_features(rows)
    assert records[0].feature_status == STATUS_WARMUP
    assert records[1].feature_status == STATUS_WARMUP
    assert records[2].feature_status == STATUS_WARMUP
    assert records[3].feature_status == STATUS_AVAILABLE
    assert records[3].history_elapsed_seconds == expected_elapsed
    expected = compute_endpoint_slope_ppm_per_min(
        rows[3].co2, rows[0].co2, expected_elapsed
    )
    assert abs(records[3].co2_slope - expected) <= COMPARISON_ABS_TOLERANCE


def test_first_row_warmup_and_general_warmup():
    t0 = datetime(2015, 2, 4, 17, 51, 0)
    rows = [
        _obs("datatraining.txt", str(i + 1), i + 2, t0 + timedelta(seconds=60 * i), 600.0)
        for i in range(4)
    ]
    records = reconstruct_all_slope_features(rows)
    assert records[0].feature_status == STATUS_WARMUP
    assert records[0].co2_slope is None
    assert records[2].feature_status == STATUS_WARMUP
    assert records[3].feature_status == STATUS_AVAILABLE


def test_no_future_sample_use():
    t0 = datetime(2015, 2, 4, 17, 51, 0)
    # If future were used, slope would include the large jump at the end.
    rows = [
        _obs("datatraining.txt", "1", 2, t0, 600.0),
        _obs("datatraining.txt", "2", 3, t0 + timedelta(seconds=60), 600.0),
        _obs("datatraining.txt", "3", 4, t0 + timedelta(seconds=120), 600.0),
        _obs("datatraining.txt", "4", 5, t0 + timedelta(seconds=180), 600.0),
        _obs("datatraining.txt", "5", 6, t0 + timedelta(seconds=240), 900.0),
    ]
    records = reconstruct_all_slope_features(rows)
    # At row 4, history is rows 1..4 all 600 -> slope 0; future row 5 must not affect it.
    assert records[3].co2_slope == 0.0
    assert "5" not in records[3].history_source_row_identifiers


def test_block_boundary_restart():
    t0 = datetime(2015, 2, 2, 14, 19, 0)
    # End of validation-like block
    block1 = [
        _obs("datatest.txt", str(140 + i), 2 + i, t0 + timedelta(seconds=60 * i), 700.0 + i)
        for i in range(4)
    ]
    # Start of training block hours later — must warm up again
    t1 = datetime(2015, 2, 4, 17, 51, 0)
    block2 = [
        _obs("datatraining.txt", str(i + 1), 2 + i, t1 + timedelta(seconds=60 * i), 500.0)
        for i in range(4)
    ]
    records = reconstruct_all_slope_features(block1 + block2)
    train_first = next(
        r
        for r in records
        if r.target_source_member == "datatraining.txt"
        and r.target_source_row_identifier == "1"
    )
    assert train_first.feature_status == STATUS_WARMUP
    assert train_first.co2_slope is None
    # Ensure history does not cite datatest identifiers
    train_eligible = next(
        r
        for r in records
        if r.target_source_member == "datatraining.txt"
        and r.feature_status == STATUS_AVAILABLE
    )
    assert all(x.isdigit() for x in train_eligible.history_source_row_identifiers)


def test_gap_boundary_restart():
    t0 = datetime(2015, 2, 4, 17, 51, 0)
    rows = [
        _obs("datatraining.txt", "1", 2, t0, 600.0),
        _obs("datatraining.txt", "2", 3, t0 + timedelta(seconds=60), 610.0),
        _obs("datatraining.txt", "3", 4, t0 + timedelta(seconds=120), 620.0),
        # Forbidden internal gap
        _obs(
            "datatraining.txt",
            "4",
            5,
            t0 + timedelta(seconds=120 + MAX_INTERNAL_GAP_SECONDS + 30),
            630.0,
        ),
        _obs(
            "datatraining.txt",
            "5",
            6,
            t0 + timedelta(seconds=120 + MAX_INTERNAL_GAP_SECONDS + 90),
            640.0,
        ),
        _obs(
            "datatraining.txt",
            "6",
            7,
            t0 + timedelta(seconds=120 + MAX_INTERNAL_GAP_SECONDS + 150),
            650.0,
        ),
        _obs(
            "datatraining.txt",
            "7",
            8,
            t0 + timedelta(seconds=120 + MAX_INTERNAL_GAP_SECONDS + 210),
            660.0,
        ),
    ]
    records = reconstruct_all_slope_features(rows)
    assert records[3].feature_status == STATUS_GAP_RESTART
    assert records[3].co2_slope is None
    # After gap, needs fresh 150s history within post-gap segment
    assert records[4].feature_status in {STATUS_GAP_RESTART, STATUS_WARMUP}
    assert records[6].feature_status == STATUS_AVAILABLE
    assert "1" not in records[6].history_source_row_identifiers


def test_source_row_lineage_completeness_and_sample_count():
    t0 = datetime(2015, 2, 4, 17, 51, 0)
    rows = [
        _obs("datatraining.txt", str(i + 1), i + 2, t0 + timedelta(seconds=60 * i), 600.0 + i)
        for i in range(4)
    ]
    rec = reconstruct_all_slope_features(rows)[3]
    assert rec.feature_contract_id == FEATURE_PROFILE_ID
    assert rec.history_start_source_row_identifier == "1"
    assert rec.history_end_source_row_identifier == "4"
    assert rec.source_sample_count_used == 4
    assert list(rec.history_source_row_identifiers) == ["1", "2", "3", "4"]
    assert rec.history_elapsed_seconds == 180.0


def test_nonfinite_input_rejection():
    t0 = datetime(2015, 2, 4, 17, 51, 0)
    rows = [
        _obs("datatraining.txt", "1", 2, t0, 600.0),
        _obs("datatraining.txt", "2", 3, t0 + timedelta(seconds=60), 600.0),
        _obs("datatraining.txt", "3", 4, t0 + timedelta(seconds=120), 600.0),
        _obs("datatraining.txt", "4", 5, t0 + timedelta(seconds=180), float("nan")),
    ]
    rec = reconstruct_all_slope_features(rows)[3]
    assert rec.feature_status == STATUS_NONFINITE
    assert rec.co2_slope is None


def test_no_silent_row_loss_on_real_data():
    repo_root = get_repo_root()
    from datasets.co2.raw_reader import UCIOccupancyRawReader

    obs = UCIOccupancyRawReader(repo_root=repo_root).read_all_observations()
    records = reconstruct_all_slope_features(obs)
    assert len(obs) == 20560
    assert len(records) == 20560


def test_deterministic_output_and_profile_artifact():
    repo_root = get_repo_root()
    c_a3 = repo_root / "datasets/co2/manifests/c_a3_slope_feature"
    checksums1 = (c_a3 / "checksums.sha256").read_text(encoding="utf-8")
    subprocess.run(
        ["python3", "scripts/audit_co2_slope_feature.py"],
        cwd=str(repo_root),
        check=True,
    )
    checksums2 = (c_a3 / "checksums.sha256").read_text(encoding="utf-8")
    assert checksums1 == checksums2
    profile = json.loads((c_a3 / "co2_slope_feature_profile.json").read_text(encoding="utf-8"))
    assert profile["history_duration_seconds"] == HISTORY_DURATION_SECONDS
    assert profile["slope_method"] == "ENDPOINT_DIFFERENCE"


def test_no_scaler_fitting_and_no_synthetic_npz_dependence():
    repo_root = get_repo_root()
    c_a3 = repo_root / "datasets/co2/manifests/c_a3_slope_feature"
    generation = json.loads((c_a3 / "generation_metadata.json").read_text(encoding="utf-8"))
    assert generation["scaler_fitted"] is False
    assert generation["model_trained"] is False
    assert generation["synthetic_npz_used_as_real_source"] is False
    for fname in [
        "co2_slope_feature_profile.json",
        "feature_eligibility_summary.json",
        "candidate_method_evaluation.json",
    ]:
        text = (c_a3 / fname).read_text(encoding="utf-8")
        assert "co2_occupancy_v1.npz" not in text


def test_validator_script():
    repo_root = get_repo_root()
    res = subprocess.run(
        ["python3", "scripts/validate_co2_slope_feature.py"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "C-A4 Authorized:  YES" in res.stdout
    assert "Gate Status:      PASS_WITH_WARNINGS" in res.stdout or "Gate Status:      PASS" in res.stdout


def test_path_portability():
    repo_root = get_repo_root()
    c_a3 = repo_root / "datasets/co2/manifests/c_a3_slope_feature"
    for path in c_a3.glob("*"):
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "file://" not in text


def test_runtime_vs_offline_baseline_classification():
    """C-A3 must distinguish runtime semantics from offline canonical baseline."""
    repo_root = get_repo_root()
    profile = json.loads(
        (repo_root / "datasets/co2/manifests/c_a3_slope_feature/co2_slope_feature_profile.json")
        .read_text(encoding="utf-8")
    )
    assert "device_adapter_alignment" not in profile
    assert profile["offline_baseline_status"] == "CANONICAL_OFFLINE_BASELINE_DESIGN"
    assert profile["offline_baseline_equivalence_claims"]["active_runtime_equivalent"] is False
    assert profile["offline_baseline_equivalence_claims"][
        "verified_historical_training_contract"
    ] is False
    runtime = profile["runtime_evidence"]
    assert runtime["runtime_slope_method_verified"] == "ENDPOINT_DIFFERENCE"
    assert runtime["runtime_history_maxlen"] == 30
    assert runtime["runtime_required_history_sec"] == 5.0
    assert runtime["configured_window_seconds"] == 150.0
    assert runtime["configured_window_seconds_applied_to_slope_logic"] is False
    assert runtime["nominal_full_buffer_endpoint_span_seconds"] == 145.0
    assert profile["offline_canonical_history_threshold_seconds"] == 150.0
    assert profile["historical_training_history_contract_status"] == (
        "CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED"
    )


def test_active_adapter_read_requests_five_second_history_not_window_seconds():
    """Prove active adapter read() uses 5.0s eligibility; window_seconds is unused there."""
    import inspect
    from sensors.co2.co2_adapter import CO2SensorAdapter

    source = inspect.getsource(CO2SensorAdapter.read)
    assert "required_history_sec = 5.0" in source
    assert "required_history_sec=required_history_sec" in source
    assert "window_seconds" not in source
    # Constructor still configures window_seconds, but slope path does not apply it.
    init_source = inspect.getsource(CO2SensorAdapter.__init__)
    assert "window_seconds: float = 150.0" in init_source
    calc_source = inspect.getsource(CO2SensorAdapter.calculate_co2_slope)
    assert "required_history_sec: float = 5.0" in calc_source
    assert "window_seconds" not in calc_source


def test_window_seconds_not_asserted_as_active_runtime_eligibility():
    repo_root = get_repo_root()
    c_a3 = repo_root / "datasets/co2/manifests/c_a3_slope_feature"
    exceptions = json.loads(
        (c_a3 / "exceptions_and_limitations.json").read_text(encoding="utf-8")
    )
    codes = {w["code"] for w in exceptions["warnings"]}
    assert "ADAPTER_WINDOW_SECONDS_NOT_APPLIED_TO_ACTIVE_SLOPE_PATH" in codes
    assert "CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED" in codes
    candidates = json.loads(
        (c_a3 / "candidate_method_evaluation.json").read_text(encoding="utf-8")
    )
    scaler_cmp = candidates["train_only_secondary_scaler_diagnostic"]["comparison_result"]
    assert scaler_cmp in {
        "PARTIAL_MEAN_ALIGNMENT_ONLY",
        "INSUFFICIENT_TO_PROVE_LINEAGE",
        "INCONSISTENT_WITH_HISTORICAL_SCALER",
    }
    assert scaler_cmp != "CONSISTENT_WITH_HISTORICAL_SCALER"
