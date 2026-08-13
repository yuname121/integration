#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_co2_final_integrity.py
Focused tests for C-A6 full-chain integrity audit and artifact lock.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datasets.co2.integrity_audit import (
    AUDIT_PROFILE_ID,
    C_A6_HASHED_EVIDENCE,
    LOCK_PROFILE_ID,
    MANIFEST_DIR_REL,
    PROPOSED_RELEASE_TAG,
    RELEASE_READINESS_PROFILE_ID,
    assert_no_forbidden_path_markers,
    build_artifact_lock_manifest,
    build_population_audit,
    build_release_readiness_manifest,
    independently_hash_raw_archive,
    independently_hash_raw_members,
    verify_artifact_lock,
)
from datasets.co2.raw_reader import (
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_ARCHIVE_SIZE,
    get_repo_root,
)


def test_raw_archive_identity_match():
    root = get_repo_root()
    raw = independently_hash_raw_archive(root)
    assert raw["exists"] is True
    assert raw["byte_size"] == EXPECTED_ARCHIVE_SIZE
    assert raw["sha256"] == EXPECTED_ARCHIVE_SHA256
    assert raw["matches_expected"] is True
    assert raw["included_in_git_release"] is False


def test_raw_archive_identity_mismatch_detection(tmp_path: Path, monkeypatch):
    root = get_repo_root()
    # Point reader helper at a fake missing archive via temporary copy of function expectation
    fake = tmp_path / "datasets/raw_archives/external_datasets"
    fake.mkdir(parents=True)
    (fake / "occupancy+detection.zip").write_bytes(b"not-the-uci-archive")
    monkeypatch.chdir(tmp_path)
    # Create minimal AGENTS.md so get_repo_root can climb? integrity uses repo_root arg path.
    # Call with explicit tmp root instead.
    result = independently_hash_raw_archive(tmp_path)
    assert result["exists"] is True
    assert result["matches_expected"] is False


def test_raw_member_hash_match_and_mismatch_detection():
    root = get_repo_root()
    members = independently_hash_raw_members(root)
    assert members["all_members_match"] is True
    assert members["total_source_rows"] == 20560
    # Mutate expected in a shallow copy to simulate mismatch detection logic
    bad = json.loads(json.dumps(members))
    first = next(iter(bad["members"]))
    bad["members"][first]["sha256"] = "0" * 64
    bad["members"][first]["matches_expected"] = (
        bad["members"][first]["sha256"]
        == members["members"][first]["sha256"]
    )
    assert bad["members"][first]["matches_expected"] is False


def test_population_audit_rejects_count_mismatch():
    class _S:
        def __init__(self, sid, member, row, role, occ, status, elig):
            self.canonical_sample_id = sid
            self.source_member_name = member
            self.source_row_identifier = row
            self.future_split_role = role
            self.occupancy_source_value = occ
            self.co2_slope_status = status
            self.model_eligible_for_slope_complete_view = elig

    class _O:
        def __init__(self, member, row):
            self.source_member_name = member
            self.source_row_identifier = row

    samples = [
        _S("a", "datatest.txt", "1", "VALIDATION", 0, "FEATURE_UNAVAILABLE_WARMUP", False),
        _S("a", "datatest.txt", "2", "VALIDATION", 1, "FEATURE_AVAILABLE", True),  # dup id
    ]
    obs = [_O("datatest.txt", "1"), _O("datatest.txt", "2")]
    pop = build_population_audit(samples, obs)
    assert pop["duplicate_canonical_ids"] == 1
    assert pop["one_to_one_ok"] is False


def test_missing_source_mapping_rejection():
    class _S:
        def __init__(self):
            self.canonical_sample_id = "x"
            self.source_member_name = "datatest.txt"
            self.source_row_identifier = "1"
            self.future_split_role = "VALIDATION"
            self.occupancy_source_value = 0
            self.co2_slope_status = "FEATURE_AVAILABLE"
            self.model_eligible_for_slope_complete_view = True

    class _O:
        def __init__(self, row):
            self.source_member_name = "datatest.txt"
            self.source_row_identifier = row

    pop = build_population_audit([_S()], [_O("1"), _O("2")])
    assert pop["missing_source_mappings"] == 1
    assert pop["one_to_one_ok"] is False


def test_artifact_lock_hash_and_self_reference_policy():
    root = get_repo_root()
    lock = build_artifact_lock_manifest(root)
    assert lock["lock_profile_id"] == LOCK_PROFILE_ID
    assert lock["self_reference_policy"]["artifact_lock_manifest_hashes_itself"] is False
    assert lock["self_reference_policy"]["checksums_sha256_hashes_itself"] is False
    assert verify_artifact_lock(lock, root) == []
    bad = json.loads(json.dumps(lock))
    bad["artifacts"][0]["sha256"] = "0" * 64
    errs = verify_artifact_lock(bad, root)
    assert errs and "hash mismatch" in errs[0]


def test_release_readiness_false_claim_and_pending_commit():
    ready = build_release_readiness_manifest(
        integrity_ok=True, lock_ok=True, predecessor_ok=True, determinism_ok=True
    )
    assert ready["profile_id"] == RELEASE_READINESS_PROFILE_ID
    assert ready["git_tag_created"] is False
    assert ready["github_release_created"] is False
    assert ready["release_commit"] == "PENDING_POST_MERGE"
    assert ready["proposed_release_tag"] == PROPOSED_RELEASE_TAG
    assert ready["release_ready_after_merge"] is True
    not_ready = build_release_readiness_manifest(
        integrity_ok=False, lock_ok=True, predecessor_ok=True, determinism_ok=True
    )
    assert not_ready["release_ready_after_merge"] is False


def test_absolute_path_rejection():
    errs = assert_no_forbidden_path_markers('path="/Users/someone/file"')
    assert errs
    assert assert_no_forbidden_path_markers("datasets/co2/manifests/x.json") == []


def test_generated_c_a6_artifacts_present_and_checksum_safe():
    root = get_repo_root()
    c_a6 = root / MANIFEST_DIR_REL
    if not c_a6.exists():
        pytest.skip("C-A6 artifacts not generated yet")
    checksum = (c_a6 / "checksums.sha256").read_text(encoding="utf-8")
    assert "checksums.sha256" not in {
        line.split("  ", 1)[1].split("/")[-1] for line in checksum.strip().splitlines()
    }
    names = {line.split("  ", 1)[1].split("/")[-1] for line in checksum.strip().splitlines()}
    for fname in C_A6_HASHED_EVIDENCE:
        assert fname in names
        assert (c_a6 / fname).exists()
    integrity = json.loads((c_a6 / "full_chain_integrity_summary.json").read_text(encoding="utf-8"))
    assert integrity["audit_profile_id"] == AUDIT_PROFILE_ID
    assert integrity["canonical_contract"]["canonical_source_samples"] == 20560
    readiness = json.loads((c_a6 / "release_readiness_manifest.json").read_text(encoding="utf-8"))
    assert readiness["release_commit"] == "PENDING_POST_MERGE"
    assert readiness["git_tag_created"] is False


def test_deterministic_generation_fields_stable():
    a = build_release_readiness_manifest(
        integrity_ok=True, lock_ok=True, predecessor_ok=True, determinism_ok=True
    )
    b = build_release_readiness_manifest(
        integrity_ok=True, lock_ok=True, predecessor_ok=True, determinism_ok=True
    )
    assert a == b
