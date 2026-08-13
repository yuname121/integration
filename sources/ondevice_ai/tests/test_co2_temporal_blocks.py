#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_co2_temporal_blocks.py
Focused unit and integration tests for C-A2 CO₂ Temporal Blocks & Split Contract.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# Ensure repo root is on sys.path
repo_root_dir = Path(__file__).resolve().parent.parent
if str(repo_root_dir) not in sys.path:
    sys.path.insert(0, str(repo_root_dir))

import pytest
from datasets.co2.raw_reader import UCIOccupancyRawReader, get_repo_root


def test_co2_temporal_blocks_predecessor_usage():
    repo_root = get_repo_root()
    reader = UCIOccupancyRawReader(repo_root=repo_root)
    obs = reader.read_all_observations()
    assert len(obs) == 20560


def test_co2_temporal_blocks_timestamp_canonicalization():
    repo_root = get_repo_root()
    c_a2_dir = repo_root / "datasets/co2/manifests/c_a2_temporal_blocks"
    profile = json.loads((c_a2_dir / "timestamp_cadence_profile.json").read_text(encoding="utf-8"))

    sem = profile["timestamp_semantics"]
    assert sem["timestamp_reference"] == "SOURCE_ACQUISITION_CLOCK"
    assert sem["source_timezone"] == "UNVERIFIED"
    assert sem["utc_conversion_claimed"] is False
    assert sem["canonical_format"] == "YYYY-MM-DDTHH:MM:SS"


def test_co2_temporal_blocks_cadence_profile():
    repo_root = get_repo_root()
    c_a2_dir = repo_root / "datasets/co2/manifests/c_a2_temporal_blocks"
    profile = json.loads((c_a2_dir / "timestamp_cadence_profile.json").read_text(encoding="utf-8"))

    cad = profile["sampling_cadence"]
    assert cad["observed_dominant_interval_seconds"] == 60.0
    assert cad["observed_delta_range_seconds"] == [59.0, 61.0]

    timeline = profile["per_member_timeline"]
    assert len(timeline) == 3
    for m_name, meta in timeline.items():
        assert meta["reversals"] == 0
        assert meta["duplicates"] == 0
        assert meta["row_count"] in [2665, 8143, 9752]


def test_co2_temporal_blocks_reconstruction():
    repo_root = get_repo_root()
    c_a2_dir = repo_root / "datasets/co2/manifests/c_a2_temporal_blocks"
    blocks_manifest = json.loads((c_a2_dir / "temporal_blocks_manifest.json").read_text(encoding="utf-8"))

    assert blocks_manifest["total_source_rows_read"] == 20560
    assert blocks_manifest["total_temporal_blocks"] == 3
    assert blocks_manifest["total_rows_assigned_to_blocks"] == 20560
    assert blocks_manifest["rows_omitted"] == 0
    assert blocks_manifest["duplicate_block_membership_count"] == 0

    blocks = blocks_manifest["blocks"]
    assert blocks[0]["block_id"] == "BLOCK_01_DATATEST"
    assert blocks[0]["row_count"] == 2665
    assert blocks[1]["block_id"] == "BLOCK_02_DATATRAINING"
    assert blocks[1]["row_count"] == 8143
    assert blocks[2]["block_id"] == "BLOCK_03_DATATEST2"
    assert blocks[2]["row_count"] == 9752


def test_co2_temporal_blocks_inter_block_gaps():
    repo_root = get_repo_root()
    c_a2_dir = repo_root / "datasets/co2/manifests/c_a2_temporal_blocks"
    blocks_manifest = json.loads((c_a2_dir / "temporal_blocks_manifest.json").read_text(encoding="utf-8"))

    blocks = blocks_manifest["blocks"]
    assert blocks[0]["preceding_gap_seconds"] is None
    assert blocks[1]["preceding_gap_seconds"] == 25680.0  # 7.13 hours
    assert blocks[2]["preceding_gap_seconds"] == 105300.0  # 29.25 hours


def test_co2_temporal_blocks_group_split_contract():
    repo_root = get_repo_root()
    c_a2_dir = repo_root / "datasets/co2/manifests/c_a2_temporal_blocks"
    contract = json.loads((c_a2_dir / "grouping_split_contract.json").read_text(encoding="utf-8"))

    assert contract["strongest_defensible_grouping_unit"] == "TEMPORAL_ACQUISITION_BLOCK"
    assert contract["group_independence_status"] == "GROUP_INDEPENDENCE_NOT_VERIFIABLE"
    assert contract["random_row_wise_split_policy"]["allowed"] is False
    assert contract["scaler_fit_scope_rule"] == "MUST_FIT_ON_TRAIN_ONLY"
    assert contract["feature_history_cross_block_rule"] == "DERIVED_TEMPORAL_FEATURES_MUST_NOT_CROSS_BLOCK_BOUNDARIES"

    splits = contract["future_split_assignments"]
    assert splits["TRAIN"]["assigned_block_id"] == "BLOCK_02_DATATRAINING"
    assert splits["VALIDATION"]["assigned_block_id"] == "BLOCK_01_DATATEST"
    assert splits["LOCKED_TEST"]["assigned_block_id"] == "BLOCK_03_DATATEST2"


def test_co2_temporal_blocks_validator_script():
    repo_root = get_repo_root()
    cmd = ["python3", "scripts/validate_co2_temporal_blocks.py"]
    res = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)

    assert res.returncode == 0, f"validate_co2_temporal_blocks.py failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "Gate Status:      PASS_WITH_WARNINGS" in res.stdout or "Gate Status:      PASS" in res.stdout
    assert "C-A3 Authorized:  YES" in res.stdout


def test_co2_temporal_blocks_determinism():
    repo_root = get_repo_root()
    c_a2_dir = repo_root / "datasets/co2/manifests/c_a2_temporal_blocks"
    checksums1 = (c_a2_dir / "checksums.sha256").read_text(encoding="utf-8")

    # Re-run audit script
    subprocess.run(["python3", "scripts/audit_co2_temporal_blocks.py"], cwd=str(repo_root), check=True)
    checksums2 = (c_a2_dir / "checksums.sha256").read_text(encoding="utf-8")

    assert checksums1 == checksums2, "Audit output must be 100% deterministic"


def test_co2_temporal_blocks_path_portability():
    repo_root = get_repo_root()
    c_a2_dir = repo_root / "datasets/co2/manifests/c_a2_temporal_blocks"

    for fname in ["timestamp_cadence_profile.json", "temporal_blocks_manifest.json", "grouping_split_contract.json"]:
        text = (c_a2_dir / fname).read_text(encoding="utf-8")
        assert "/Users/" not in text, f"Found forbidden '/Users/' in {fname}"
        assert "file://" not in text, f"Found forbidden 'file://' in {fname}"


def test_co2_temporal_blocks_synthetic_npz_isolation():
    repo_root = get_repo_root()
    c_a2_dir = repo_root / "datasets/co2/manifests/c_a2_temporal_blocks"
    contract_text = (c_a2_dir / "grouping_split_contract.json").read_text(encoding="utf-8")

    assert "co2_occupancy_v1.npz" not in contract_text
