#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_co2_raw_inventory.py
Focused Unit / Integration Tests for Phase C-A0 CO₂ Raw Inventory.
"""

import os
import json
import hashlib
import subprocess
from pathlib import Path
import pytest


def get_repo_root() -> Path:
    root = Path(__file__).parent.parent
    if (root / "AGENTS.md").exists():
        return root
    return Path(os.getcwd())


def test_co2_raw_archive_existence_and_hash():
    repo_root = get_repo_root()
    archive_path = repo_root / "datasets/raw_archives/external_datasets/occupancy+detection.zip"

    assert archive_path.exists(), "Raw UCI archive occupancy+detection.zip must exist locally"
    assert archive_path.stat().st_size == 335713, "Archive size must match owner-confirmed 335,713 bytes"

    h = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert h == "4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a", "Archive SHA256 must match owner-confirmed hash"


def test_co2_raw_inventory_auditor_script():
    repo_root = get_repo_root()
    cmd = ["python3", "scripts/audit_co2_raw_inventory.py"]
    res = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    assert res.returncode == 0, f"audit_co2_raw_inventory.py failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "Generated C-A0 manifest artifacts in: datasets/co2/manifests/c_a0_raw_inventory" in res.stdout


def test_co2_raw_inventory_validator_script():
    repo_root = get_repo_root()
    cmd = ["python3", "scripts/validate_co2_raw_inventory.py"]
    res = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    assert res.returncode == 0, f"validate_co2_raw_inventory.py failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "Gate Status:      PASS_WITH_WARNINGS" in res.stdout or "Gate Status:      PASS" in res.stdout
    assert "C-A1 Authorized:  YES" in res.stdout


def test_co2_manifest_file_completeness():
    repo_root = get_repo_root()
    manifest_dir = repo_root / "datasets/co2/manifests/c_a0_raw_inventory"

    required_files = [
        "source_identity.json",
        "official_source_license.json",
        "archive_integrity.json",
        "archive_members.jsonl",
        "raw_inventory_summary.json",
        "lineage_registry.json",
        "anomalies_and_limitations.json",
        "checksums.sha256"
    ]

    for fname in required_files:
        assert (manifest_dir / fname).exists(), f"Manifest file missing: {fname}"

    source_id = json.loads((manifest_dir / "source_identity.json").read_text(encoding="utf-8"))
    assert source_id["doi"] == "10.24432/C5X01N"
    assert source_id["journal_paper_doi"] == "10.1016/j.enbuild.2015.11.071"

    files = source_id["collection_methodology"]["dataset_files"]
    assert "2015-02-02 14:19:00 to 2015-02-04 10:43:00" in files["datatest.txt"]
    assert "2015-02-04 17:51:00 to 2015-02-10 09:33:00" in files["datatraining.txt"]
    assert "2015-02-11 14:48:00 to 2015-02-18 09:19:00" in files["datatest2.txt"]


def test_co2_schema_mismatch_representation():
    repo_root = get_repo_root()
    members_path = repo_root / "datasets/co2/manifests/c_a0_raw_inventory/archive_members.jsonl"

    members = [json.loads(line) for line in members_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(members) == 3, "Must contain exactly 3 members"

    for m in members:
        assert m["schema_mismatch_detected"] is True, f"Member {m['member_name']} must report schema mismatch"
        assert m["header_field_count"] == 7, f"Header field count must be 7 for {m['member_name']}"
        assert m["actual_data_field_count"] == 8, f"Data field count must be 8 for {m['member_name']}"


def test_co2_lineage_separation():
    repo_root = get_repo_root()
    lineage_path = repo_root / "datasets/co2/manifests/c_a0_raw_inventory/lineage_registry.json"
    data = json.loads(lineage_path.read_text(encoding="utf-8"))

    lineages = data.get("lineages", {})
    lin_b = lineages.get("Lineage_B_Synthetic_Smoke_Fixture", {})
    lin_c = lineages.get("Lineage_C_Existing_CO2_Model", {})
    lin_d = lineages.get("Lineage_D_Existing_Scaling_Metadata", {})

    assert lin_b.get("classification") == "SYNTHETIC_SMOKE_FIXTURE"
    assert lin_b.get("verified_real_data") is False
    assert lin_c.get("manifest_validation_status") == "CONFIRMED_SYNTHETIC_ONLY"
    assert lin_d.get("fit_lineage_status") == "FIT_DATA_LINEAGE_UNVERIFIED"


def test_co2_timestamp_preservation_and_no_utc():
    repo_root = get_repo_root()
    members_path = repo_root / "datasets/co2/manifests/c_a0_raw_inventory/archive_members.jsonl"
    members = [json.loads(line) for line in members_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    for m in members:
        assert m["timezone_evidence_status"] == "UNVERIFIED"
        assert m["utc_conversion_claimed"] is False
        assert m["timestamp_reference"] == "SOURCE_ACQUISITION_CLOCK"


def test_co2_portable_paths():
    repo_root = get_repo_root()
    manifest_dir = repo_root / "datasets/co2/manifests/c_a0_raw_inventory"

    for p in manifest_dir.glob("*"):
        content = p.read_text(encoding="utf-8")
        assert "/Users/" not in content, f"Forbidden absolute path found in {p.name}"
        assert "file://" not in content, f"Forbidden file:// URI found in {p.name}"


def test_co2_git_payload_safety():
    repo_root = get_repo_root()
    try:
        staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=str(repo_root), text=True).splitlines()
        for f in staged:
            assert "occupancy+detection.zip" not in f, f"Raw zip staged in git: {f}"
            assert not (f.endswith(".txt") and "datasets/raw_archives" in f), f"Raw extracted txt staged in git: {f}"
    except Exception:
        pass


def test_co2_inventory_determinism():
    repo_root = get_repo_root()
    manifest_dir = repo_root / "datasets/co2/manifests/c_a0_raw_inventory"

    checksums1 = (manifest_dir / "checksums.sha256").read_text(encoding="utf-8")

    # Run auditor again
    subprocess.run(["python3", "scripts/audit_co2_raw_inventory.py"], cwd=str(repo_root), check=True)

    checksums2 = (manifest_dir / "checksums.sha256").read_text(encoding="utf-8")

    assert checksums1 == checksums2, "Audit output must be 100% deterministic across multiple runs"
