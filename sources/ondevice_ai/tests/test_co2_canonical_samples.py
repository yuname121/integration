#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_co2_canonical_samples.py
Focused tests for C-A5 canonical sample provenance and split materialization.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime

import pytest

from datasets.co2.canonical_samples import (
    CANONICAL_SAMPLE_PROFILE_ID,
    EXPECTED_SLOPE_ELIGIBLE,
    EXPECTED_TOTAL_SAMPLES,
    EXPECTED_WARMUP,
    build_canonical_sample_profile,
    build_predecessor_fingerprint_registry,
    make_canonical_sample_id,
    materialize_canonical_samples,
    verify_predecessor_fingerprints,
)
from datasets.co2.raw_reader import CO2SourceRowObservation, get_repo_root
from datasets.co2.slope_feature import STATUS_AVAILABLE, STATUS_WARMUP


def _obs(member: str, row_id: str, line: int, occupancy: int = 0, co2: float = 600.0, minute: int = 0):
    ts = datetime(2015, 2, 4, 17, 51 + minute, 0)
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
        occupancy=occupancy,
    )


def test_stable_canonical_sample_id_and_profile():
    a = _obs("datatraining.txt", "1", 2)
    b = _obs("datatraining.txt", "1", 2)
    assert make_canonical_sample_id(a) == make_canonical_sample_id(b)
    assert make_canonical_sample_id(a).startswith("co2cs_")
    profile = build_canonical_sample_profile()
    assert profile["profile_id"] == CANONICAL_SAMPLE_PROFILE_ID
    assert profile["access_semantics"]["scaler_fit_authorized_roles"] == ["TRAIN"]
    assert profile["access_semantics"]["locked_test_authorized_for_fitting"] is False
    assert profile["a_series_release_status"] == "DEFERRED_UNTIL_C-A6"


def test_deterministic_ordering_and_warmup_preservation():
    rows = [
        _obs("datatraining.txt", str(i + 1), i + 2, co2=600.0 + i, minute=i)
        for i in range(4)
    ]
    samples = materialize_canonical_samples(rows)
    assert [s.source_row_identifier for s in samples] == ["1", "2", "3", "4"]
    assert samples[0].co2_slope_status == STATUS_WARMUP
    assert samples[0].co2_slope is None
    assert samples[0].model_eligible_for_slope_complete_view is False
    assert samples[0].model_eligibility_exclusion_reason == STATUS_WARMUP
    assert samples[3].co2_slope_status == STATUS_AVAILABLE
    assert samples[3].model_eligible_for_slope_complete_view is True


def test_member_order_is_c_a2_chronological_not_alpha():
    rows = [
        _obs("datatest2.txt", "1", 2, minute=0),
        _obs("datatraining.txt", "1", 2, minute=0),
        _obs("datatest.txt", "140", 2, minute=0),
    ]
    samples = materialize_canonical_samples(rows)
    assert [s.source_member_name for s in samples] == [
        "datatest.txt",
        "datatraining.txt",
        "datatest2.txt",
    ]


def test_split_and_target_and_scaler_rules_on_fixture():
    rows = [
        _obs("datatraining.txt", "1", 2, occupancy=1),
        _obs("datatest.txt", "140", 2, occupancy=0),
        _obs("datatest2.txt", "1", 2, occupancy=1),
    ]
    samples = materialize_canonical_samples(rows)
    by_member = {s.source_member_name: s for s in samples}
    assert by_member["datatraining.txt"].future_split_role == "TRAIN"
    assert by_member["datatraining.txt"].scaler_fit_authorized is True
    assert by_member["datatest.txt"].future_split_role == "VALIDATION"
    assert by_member["datatest.txt"].scaler_fit_authorized is False
    assert by_member["datatest2.txt"].future_split_role == "LOCKED_TEST"
    assert by_member["datatest2.txt"].locked_test_fit_authorized is False
    assert by_member["datatest2.txt"].locked_test_tuning_authorized is False
    assert by_member["datatraining.txt"].occupancy_canonical_class == "OCCUPIED"
    assert by_member["datatest.txt"].occupancy_canonical_class == "VACANT"


def test_real_data_counts_and_artifacts():
    repo_root = get_repo_root()
    from datasets.co2.raw_reader import UCIOccupancyRawReader

    obs = UCIOccupancyRawReader(repo_root=repo_root).read_all_observations()
    samples = materialize_canonical_samples(obs)
    assert len(samples) == EXPECTED_TOTAL_SAMPLES
    assert len({s.canonical_sample_id for s in samples}) == EXPECTED_TOTAL_SAMPLES
    assert sum(1 for s in samples if s.model_eligible_for_slope_complete_view) == EXPECTED_SLOPE_ELIGIBLE
    assert sum(1 for s in samples if s.co2_slope_status == STATUS_WARMUP) == EXPECTED_WARMUP
    c_a5 = repo_root / "datasets/co2/manifests/c_a5_canonical_samples"
    integrity = json.loads((c_a5 / "materialization_integrity_summary.json").read_text(encoding="utf-8"))
    assert integrity["one_to_one_ok"] is True
    assert integrity["missing_source_mappings"] == 0
    assert integrity["duplicate_canonical_ids"] == 0


def test_predecessor_fingerprint_mismatch_detection():
    repo_root = get_repo_root()
    registry = build_predecessor_fingerprint_registry(repo_root)
    assert verify_predecessor_fingerprints(registry, repo_root) == []
    bad = json.loads(json.dumps(registry))
    bad["phases"]["C-A4"][0]["sha256"] = "0" * 64
    errs = verify_predecessor_fingerprints(bad, repo_root)
    assert errs and "fingerprint mismatch" in errs[0]


def test_synthetic_npz_isolation_and_path_portability():
    repo_root = get_repo_root()
    c_a5 = repo_root / "datasets/co2/manifests/c_a5_canonical_samples"
    generation = json.loads((c_a5 / "generation_metadata.json").read_text(encoding="utf-8"))
    assert generation["synthetic_npz_used_as_real_source"] is False
    assert generation["scaler_fitted"] is False
    assert generation["model_trained"] is False
    profile_text = (c_a5 / "canonical_sample_profile.json").read_text(encoding="utf-8")
    assert "SYNTHETIC_SMOKE_FIXTURE" in profile_text
    for path in c_a5.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "file://" not in text


def test_deterministic_generation_and_validator():
    repo_root = get_repo_root()
    c_a5 = repo_root / "datasets/co2/manifests/c_a5_canonical_samples"
    checksums1 = (c_a5 / "checksums.sha256").read_text(encoding="utf-8")
    subprocess.run(
        ["python3", "scripts/audit_co2_canonical_samples.py"],
        cwd=str(repo_root),
        check=True,
    )
    checksums2 = (c_a5 / "checksums.sha256").read_text(encoding="utf-8")
    assert checksums1 == checksums2
    res = subprocess.run(
        ["python3", "scripts/validate_co2_canonical_samples.py"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "C-A6 Authorized:  YES" in res.stdout
