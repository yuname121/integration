#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_co2_safe_reader.py
Focused unit and integration tests for C-A1 Safe CO₂ Raw Reader and Contract.
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

from datasets.co2.raw_reader import (
    UCIOccupancyRawReader,
    CO2SourceRowObservation,
    ArchiveIntegrityError,
    SchemaValidationError,
    get_repo_root,
)


def test_co2_raw_reader_archive_verification():
    repo_root = get_repo_root()
    reader = UCIOccupancyRawReader(repo_root=repo_root)
    size, sha256 = reader.verify_archive()

    assert size == 335713, f"Expected 335713 bytes, got {size}"
    assert sha256 == "4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a"


def test_co2_raw_reader_corrupt_archive_rejection(tmp_path):
    fake_archive = tmp_path / "fake.zip"
    fake_archive.write_bytes(b"not a real zip payload")

    reader = UCIOccupancyRawReader(repo_root=tmp_path, archive_rel_path="fake.zip")
    with pytest.raises(ArchiveIntegrityError):
        reader.verify_archive()


def test_co2_raw_reader_zero_row_loss_and_counts():
    repo_root = get_repo_root()
    reader = UCIOccupancyRawReader(repo_root=repo_root)
    obs_list = reader.read_all_observations()

    assert len(obs_list) == 20560, f"Expected 20560 total rows, got {len(obs_list)}"

    counts = {"datatest.txt": 0, "datatest2.txt": 0, "datatraining.txt": 0}
    occ_0 = 0
    occ_1 = 0

    for obs in obs_list:
        counts[obs.source_member_name] += 1
        if obs.occupancy == 0:
            occ_0 += 1
        elif obs.occupancy == 1:
            occ_1 += 1

    assert counts["datatest.txt"] == 2665
    assert counts["datatest2.txt"] == 9752
    assert counts["datatraining.txt"] == 8143

    assert occ_0 == 15810, f"Expected 15810 vacant rows, got {occ_0}"
    assert occ_1 == 4750, f"Expected 4750 occupied rows, got {occ_1}"


def test_co2_raw_reader_schema_mismatch_resolution():
    repo_root = get_repo_root()
    reader = UCIOccupancyRawReader(repo_root=repo_root)
    # Check first observation of datatraining.txt
    obs = next(reader.iter_observations(target_member="datatraining.txt"))

    assert obs.source_row_identifier == "1"
    assert obs.source_timestamp_raw == "2015-02-04 17:51:00"
    assert obs.temperature == 23.18
    assert obs.humidity == 27.272
    assert obs.light == 426.0
    assert obs.co2 == 721.25
    assert obs.humidity_ratio == pytest.approx(0.00479298817650529)
    assert obs.occupancy == 1


def test_co2_raw_reader_provenance_completeness():
    repo_root = get_repo_root()
    reader = UCIOccupancyRawReader(repo_root=repo_root)
    obs_list = reader.read_all_observations()

    for idx, obs in enumerate(obs_list, 1):
        assert isinstance(obs, CO2SourceRowObservation)
        assert obs.source_archive_path == "datasets/raw_archives/external_datasets/occupancy+detection.zip"
        assert obs.source_archive_sha256 == "4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a"
        assert obs.source_member_name in ["datatest.txt", "datatest2.txt", "datatraining.txt"]
        assert len(obs.source_member_sha256) == 64
        assert obs.source_physical_line_number >= 2
        assert obs.source_row_identifier != ""
        assert obs.source_timestamp_raw != ""
        assert obs.timestamp_reference == "SOURCE_ACQUISITION_CLOCK"
        assert obs.source_timezone == "UNVERIFIED"
        assert obs.utc_conversion_claimed is False


def test_co2_raw_reader_unnormalized_measurements():
    repo_root = get_repo_root()
    reader = UCIOccupancyRawReader(repo_root=repo_root)
    obs = next(reader.iter_observations(target_member="datatest.txt"))

    # Ensure measurements are raw floats, not normalized Z-scores or slope values
    assert 15.0 <= obs.temperature <= 30.0
    assert 10.0 <= obs.humidity <= 60.0
    assert 0.0 <= obs.light <= 2000.0
    assert 300.0 <= obs.co2 <= 2500.0
    assert 0.001 <= obs.humidity_ratio <= 0.01
    assert not hasattr(obs, "co2_slope")  # Must NOT derive CO2_slope in C-A1


def test_co2_raw_reader_manifest_completeness():
    repo_root = get_repo_root()
    c_a1_dir = repo_root / "datasets/co2/manifests/c_a1_safe_reader"

    required_files = [
        "source_schema_profile.json",
        "source_row_provenance_contract.json",
        "reader_validation_summary.json",
        "checksums.sha256",
    ]

    for fname in required_files:
        fpath = c_a1_dir / fname
        assert fpath.exists(), f"Missing C-A1 manifest: {fname}"

    summary = json.loads((c_a1_dir / "reader_validation_summary.json").read_text(encoding="utf-8"))
    assert summary["total_observations_read"] == 20560
    assert summary["silent_row_loss"] == 0
    assert summary["validation_status"] == "PASS"


def test_co2_raw_reader_validator_script():
    repo_root = get_repo_root()
    cmd = ["python3", "scripts/validate_co2_safe_reader.py"]
    res = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)

    assert res.returncode == 0, f"validate_co2_safe_reader.py failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "Gate Status:      PASS_WITH_WARNINGS" in res.stdout or "Gate Status:      PASS" in res.stdout
    assert "C-A2 Authorized:  YES" in res.stdout


def test_co2_raw_reader_determinism():
    repo_root = get_repo_root()
    reader1 = UCIOccupancyRawReader(repo_root=repo_root)
    obs1 = reader1.read_all_observations()

    reader2 = UCIOccupancyRawReader(repo_root=repo_root)
    obs2 = reader2.read_all_observations()

    assert len(obs1) == len(obs2)
    for o1, o2 in zip(obs1, obs2):
        assert o1 == o2


def test_co2_raw_reader_path_portability():
    repo_root = get_repo_root()
    c_a1_dir = repo_root / "datasets/co2/manifests/c_a1_safe_reader"

    for fname in ["source_schema_profile.json", "source_row_provenance_contract.json", "reader_validation_summary.json"]:
        text = (c_a1_dir / fname).read_text(encoding="utf-8")
        assert "/Users/" not in text, f"Found forbidden '/Users/' path in {fname}"
        assert "file://" not in text, f"Found forbidden 'file://' path in {fname}"


def test_co2_raw_reader_synthetic_npz_isolation():
    repo_root = get_repo_root()
    reader = UCIOccupancyRawReader(repo_root=repo_root)

    # Reader source archive must be external zip, not synthetic npz
    assert reader.archive_rel_path.endswith(".zip")
    assert "co2_occupancy_v1.npz" not in reader.archive_rel_path
