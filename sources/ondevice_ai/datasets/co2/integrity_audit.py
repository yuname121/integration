#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datasets/co2/integrity_audit.py
Phase C-A6 — CO₂ Final Raw-to-Canonical Integrity Audit and Artifact Lock helpers.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets.co2.canonical_samples import (
    CANONICAL_SAMPLE_PROFILE_ID,
    EXPECTED_SLOPE_ELIGIBLE,
    EXPECTED_TOTAL_SAMPLES,
    EXPECTED_WARMUP,
    materialize_canonical_samples,
    verify_predecessor_fingerprints,
)
from datasets.co2.raw_reader import (
    EXPECTED_ARCHIVE_REL_PATH,
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_ARCHIVE_SIZE,
    EXPECTED_MEMBER_METADATA,
    UCIOccupancyRawReader,
    compute_sha256_file,
    get_repo_root,
)
from datasets.co2.slope_feature import (
    FEATURE_PROFILE_ID as SLOPE_PROFILE_ID,
    STATUS_AVAILABLE,
    STATUS_WARMUP,
)
from datasets.co2.target_semantics import TARGET_PROFILE_ID as OCCUPANCY_TARGET_PROFILE_ID

LOCK_PROFILE_ID = "CO2_A_SERIES_ARTIFACT_LOCK_PROFILE_001"
AUDIT_PROFILE_ID = "CO2_A_SERIES_FULL_CHAIN_AUDIT_PROFILE_001"
RELEASE_READINESS_PROFILE_ID = "CO2_A_SERIES_RELEASE_READINESS_PROFILE_001"
MANIFEST_DIR_REL = "datasets/co2/manifests/c_a6_final_integrity_lock"
PROPOSED_RELEASE_TAG = "co2-a-series-raw-to-canonical"

EXPECTED_SPLIT = {
    "TRAIN": {"canonical": 8143, "eligible": 8140, "warmup": 3, "vacant": 6414, "occupied": 1729},
    "VALIDATION": {"canonical": 2665, "eligible": 2662, "warmup": 3, "vacant": 1693, "occupied": 972},
    "LOCKED_TEST": {"canonical": 9752, "eligible": 9749, "warmup": 3, "vacant": 7703, "occupied": 2049},
}
EXPECTED_OCC_0 = 15810
EXPECTED_OCC_1 = 4750

# Artifacts locked for A-series release (excludes C-A6 lock/checksum self files).
A_SERIES_LOCK_ARTIFACTS: List[Tuple[str, str, str]] = [
    # phase, role, path
    ("C-A0", "source_identity", "datasets/co2/manifests/c_a0_raw_inventory/source_identity.json"),
    ("C-A0", "official_source_license", "datasets/co2/manifests/c_a0_raw_inventory/official_source_license.json"),
    ("C-A0", "archive_integrity", "datasets/co2/manifests/c_a0_raw_inventory/archive_integrity.json"),
    ("C-A0", "archive_members", "datasets/co2/manifests/c_a0_raw_inventory/archive_members.jsonl"),
    ("C-A0", "raw_inventory_summary", "datasets/co2/manifests/c_a0_raw_inventory/raw_inventory_summary.json"),
    ("C-A0", "lineage_registry", "datasets/co2/manifests/c_a0_raw_inventory/lineage_registry.json"),
    ("C-A0", "anomalies_and_limitations", "datasets/co2/manifests/c_a0_raw_inventory/anomalies_and_limitations.json"),
    ("C-A0", "checksums", "datasets/co2/manifests/c_a0_raw_inventory/checksums.sha256"),
    ("C-A1", "source_schema_profile", "datasets/co2/manifests/c_a1_safe_reader/source_schema_profile.json"),
    ("C-A1", "source_row_provenance_contract", "datasets/co2/manifests/c_a1_safe_reader/source_row_provenance_contract.json"),
    ("C-A1", "reader_validation_summary", "datasets/co2/manifests/c_a1_safe_reader/reader_validation_summary.json"),
    ("C-A1", "checksums", "datasets/co2/manifests/c_a1_safe_reader/checksums.sha256"),
    ("C-A2", "temporal_blocks_manifest", "datasets/co2/manifests/c_a2_temporal_blocks/temporal_blocks_manifest.json"),
    ("C-A2", "grouping_split_contract", "datasets/co2/manifests/c_a2_temporal_blocks/grouping_split_contract.json"),
    ("C-A2", "timestamp_cadence_profile", "datasets/co2/manifests/c_a2_temporal_blocks/timestamp_cadence_profile.json"),
    ("C-A2", "checksums", "datasets/co2/manifests/c_a2_temporal_blocks/checksums.sha256"),
    ("C-A3", "co2_slope_feature_profile", "datasets/co2/manifests/c_a3_slope_feature/co2_slope_feature_profile.json"),
    ("C-A3", "feature_eligibility_summary", "datasets/co2/manifests/c_a3_slope_feature/feature_eligibility_summary.json"),
    ("C-A3", "source_row_feature_lineage_contract", "datasets/co2/manifests/c_a3_slope_feature/source_row_feature_lineage_contract.json"),
    ("C-A3", "checksums", "datasets/co2/manifests/c_a3_slope_feature/checksums.sha256"),
    ("C-A4", "occupancy_target_profile", "datasets/co2/manifests/c_a4_target_semantics/occupancy_target_profile.json"),
    ("C-A4", "target_integrity_summary", "datasets/co2/manifests/c_a4_target_semantics/target_integrity_summary.json"),
    ("C-A4", "occupancy_safety_separation_contract", "datasets/co2/manifests/c_a4_target_semantics/occupancy_safety_separation_contract.json"),
    ("C-A4", "checksums", "datasets/co2/manifests/c_a4_target_semantics/checksums.sha256"),
    ("C-A5", "canonical_sample_profile", "datasets/co2/manifests/c_a5_canonical_samples/canonical_sample_profile.json"),
    ("C-A5", "predecessor_fingerprint_registry", "datasets/co2/manifests/c_a5_canonical_samples/predecessor_fingerprint_registry.json"),
    ("C-A5", "split_membership_manifest", "datasets/co2/manifests/c_a5_canonical_samples/split_membership_manifest.json"),
    ("C-A5", "feature_availability_manifest", "datasets/co2/manifests/c_a5_canonical_samples/feature_availability_manifest.json"),
    ("C-A5", "materialization_integrity_summary", "datasets/co2/manifests/c_a5_canonical_samples/materialization_integrity_summary.json"),
    ("C-A5", "canonical_source_samples", "datasets/co2/manifests/c_a5_canonical_samples/canonical_source_samples.jsonl"),
    ("C-A5", "model_eligible_sample_ids", "datasets/co2/manifests/c_a5_canonical_samples/model_eligible_sample_ids.jsonl"),
    ("C-A5", "artifact_identity", "datasets/co2/manifests/c_a5_canonical_samples/artifact_identity.json"),
    ("C-A5", "checksums", "datasets/co2/manifests/c_a5_canonical_samples/checksums.sha256"),
]

C_A6_HASHED_EVIDENCE = [
    "full_chain_integrity_summary.json",
    "full_chain_audit_manifest.json",
    "predecessor_fingerprint_closure.json",
    "release_readiness_manifest.json",
    "exceptions_and_limitations.json",
    "generation_metadata.json",
    "release_notes_draft.md",
    "artifact_lock_manifest.json",
]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def independently_hash_raw_archive(repo_root: Path) -> Dict[str, Any]:
    path = repo_root / EXPECTED_ARCHIVE_REL_PATH
    if not path.exists():
        return {
            "exists": False,
            "path": EXPECTED_ARCHIVE_REL_PATH,
            "byte_size": None,
            "sha256": None,
            "matches_expected": False,
            "git_visibility_status": "GIT_IGNORED_RAW_ARCHIVE",
            "read_only_status": "READ_ONLY_SOURCE_EVIDENCE",
        }
    data = path.read_bytes()
    digest = _sha256_bytes(data)
    return {
        "exists": True,
        "path": EXPECTED_ARCHIVE_REL_PATH,
        "byte_size": len(data),
        "sha256": digest,
        "matches_expected": (
            len(data) == EXPECTED_ARCHIVE_SIZE and digest == EXPECTED_ARCHIVE_SHA256
        ),
        "git_visibility_status": "GIT_IGNORED_RAW_ARCHIVE",
        "read_only_status": "READ_ONLY_SOURCE_EVIDENCE",
        "included_in_git_release": False,
    }


def independently_hash_raw_members(repo_root: Path) -> Dict[str, Any]:
    path = repo_root / EXPECTED_ARCHIVE_REL_PATH
    members: Dict[str, Any] = {}
    with zipfile.ZipFile(path, "r") as zf:
        for name, expected in EXPECTED_MEMBER_METADATA.items():
            raw = zf.read(name)  # exact uncompressed bytes; no text decode
            digest = _sha256_bytes(raw)
            # data rows = non-empty lines minus header
            lines = raw.splitlines()
            row_count = max(0, len(lines) - 1)
            members[name] = {
                "byte_size": len(raw),
                "sha256": digest,
                "rows": row_count,
                "matches_expected": (
                    len(raw) == expected["size"]
                    and digest == expected["sha256"]
                    and row_count == expected["rows"]
                ),
            }
    total_rows = sum(int(v["rows"]) for v in members.values())
    return {
        "members": members,
        "total_source_rows": total_rows,
        "all_members_match": all(v["matches_expected"] for v in members.values())
        and total_rows == EXPECTED_TOTAL_SAMPLES,
    }


def build_population_audit(samples: Sequence[Any], observations: Sequence[Any]) -> Dict[str, Any]:
    ids = [s.canonical_sample_id for s in samples]
    obs_keys = {(o.source_member_name, o.source_row_identifier) for o in observations}
    sample_keys = {(s.source_member_name, s.source_row_identifier) for s in samples}
    missing = sorted(obs_keys - sample_keys)
    extras = sorted(sample_keys - obs_keys)
    by_role: Dict[str, Dict[str, int]] = {}
    for role, exp in EXPECTED_SPLIT.items():
        role_samples = [s for s in samples if s.future_split_role == role]
        by_role[role] = {
            "canonical": len(role_samples),
            "eligible": sum(1 for s in role_samples if s.model_eligible_for_slope_complete_view),
            "warmup": sum(1 for s in role_samples if s.co2_slope_status == STATUS_WARMUP),
            "vacant": sum(1 for s in role_samples if s.occupancy_source_value == 0),
            "occupied": sum(1 for s in role_samples if s.occupancy_source_value == 1),
            "matches_expected": (
                len(role_samples) == exp["canonical"]
                and sum(1 for s in role_samples if s.model_eligible_for_slope_complete_view)
                == exp["eligible"]
                and sum(1 for s in role_samples if s.co2_slope_status == STATUS_WARMUP)
                == exp["warmup"]
                and sum(1 for s in role_samples if s.occupancy_source_value == 0) == exp["vacant"]
                and sum(1 for s in role_samples if s.occupancy_source_value == 1)
                == exp["occupied"]
            ),
        }
    occ0 = sum(1 for s in samples if s.occupancy_source_value == 0)
    occ1 = sum(1 for s in samples if s.occupancy_source_value == 1)
    eligible = sum(1 for s in samples if s.model_eligible_for_slope_complete_view)
    warmup = sum(1 for s in samples if s.co2_slope_status == STATUS_WARMUP)
    return {
        "source_observation_count": len(observations),
        "canonical_source_sample_count": len(samples),
        "model_eligible_sample_count": eligible,
        "warmup_sample_count": warmup,
        "missing_source_mappings": len(missing),
        "duplicate_source_mappings": len(samples) - len(sample_keys),
        "extra_canonical_mappings": len(extras),
        "duplicate_canonical_ids": len(ids) - len(set(ids)),
        "occupancy_0_count": occ0,
        "occupancy_1_count": occ1,
        "by_role": by_role,
        "one_to_one_ok": (
            len(observations) == len(samples) == EXPECTED_TOTAL_SAMPLES
            and not missing
            and not extras
            and len(ids) == len(set(ids))
            and eligible == EXPECTED_SLOPE_ELIGIBLE
            and warmup == EXPECTED_WARMUP
            and occ0 == EXPECTED_OCC_0
            and occ1 == EXPECTED_OCC_1
            and all(v["matches_expected"] for v in by_role.values())
        ),
    }


def select_round_trip_indices(samples: Sequence[Any]) -> List[Dict[str, Any]]:
    """Select representative indices for full-chain round-trip audit."""
    by_member: Dict[str, List[int]] = {
        "datatest.txt": [],
        "datatraining.txt": [],
        "datatest2.txt": [],
    }
    for i, s in enumerate(samples):
        by_member[s.source_member_name].append(i)

    picks: List[Tuple[str, int]] = []
    picks.append(("first_canonical_row", 0))
    picks.append(("last_canonical_row", len(samples) - 1))
    for member, label in (
        ("datatest.txt", "BLOCK_01"),
        ("datatraining.txt", "BLOCK_02"),
        ("datatest2.txt", "BLOCK_03"),
    ):
        idxs = by_member[member]
        picks.append((f"first_row_{label}", idxs[0]))
        picks.append((f"last_row_{label}", idxs[-1]))
        # first slope-eligible in block
        first_elig = next(
            i for i in idxs if samples[i].co2_slope_status == STATUS_AVAILABLE
        )
        picks.append((f"first_slope_eligible_{label}", first_elig))
        # warm-up rows in block
        for j, i in enumerate(idxs[:3]):
            if samples[i].co2_slope_status == STATUS_WARMUP:
                picks.append((f"warmup_{label}_{j}", i))

    # interior samples by role
    for role, label in (
        ("TRAIN", "train_interior"),
        ("VALIDATION", "validation_interior"),
        ("LOCKED_TEST", "locked_test_integrity_only"),
    ):
        role_idxs = [i for i, s in enumerate(samples) if s.future_split_role == role]
        picks.append((label, role_idxs[len(role_idxs) // 2]))

    # VACANT / OCCUPIED examples (TRAIN)
    vacant = next(
        i
        for i, s in enumerate(samples)
        if s.future_split_role == "TRAIN" and s.occupancy_source_value == 0
        and s.co2_slope_status == STATUS_AVAILABLE
    )
    occupied = next(
        i
        for i, s in enumerate(samples)
        if s.future_split_role == "TRAIN" and s.occupancy_source_value == 1
        and s.co2_slope_status == STATUS_AVAILABLE
    )
    picks.append(("vacant_example", vacant))
    picks.append(("occupied_example", occupied))

    # de-dup while preserving order
    seen = set()
    unique: List[Dict[str, Any]] = []
    for case_id, idx in picks:
        key = (case_id, idx)
        if key in seen:
            continue
        seen.add(key)
        unique.append({"case_id": case_id, "canonical_sample_index": idx})
    return unique


def run_round_trip_cases(
    samples: Sequence[Any],
    observations_by_key: Dict[Tuple[str, str], Any],
    cases: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for case in cases:
        idx = int(case["canonical_sample_index"])
        s = samples[idx]
        obs = observations_by_key[(s.source_member_name, s.source_row_identifier)]
        ok = (
            s.source_member_name == obs.source_member_name
            and s.source_physical_line_number == obs.source_physical_line_number
            and s.source_row_identifier == obs.source_row_identifier
            and s.source_timestamp_raw == obs.source_timestamp_raw
            and float(s.temperature) == float(obs.temperature)
            and float(s.humidity) == float(obs.humidity)
            and float(s.light) == float(obs.light)
            and float(s.co2) == float(obs.co2)
            and float(s.humidity_ratio) == float(obs.humidity_ratio)
            and int(s.occupancy_source_value) == int(obs.occupancy)
            and s.source_archive_sha256 == EXPECTED_ARCHIVE_SHA256
        )
        results.append(
            {
                "case_id": case["case_id"],
                "canonical_sample_index": idx,
                "canonical_sample_id": s.canonical_sample_id,
                "source_member_name": s.source_member_name,
                "source_physical_line_number": s.source_physical_line_number,
                "source_row_identifier": s.source_row_identifier,
                "source_timestamp_raw": s.source_timestamp_raw,
                "temporal_block_id": s.temporal_block_id,
                "future_split_role": s.future_split_role,
                "co2_slope_status": s.co2_slope_status,
                "co2_slope": s.co2_slope,
                "occupancy_source_value": s.occupancy_source_value,
                "occupancy_canonical_class": s.occupancy_canonical_class,
                "round_trip_ok": ok,
            }
        )
    return results


def verify_stored_canonical_against_live(
    repo_root: Path,
    samples: Sequence[Any],
) -> Dict[str, Any]:
    jsonl = (
        repo_root
        / "datasets/co2/manifests/c_a5_canonical_samples/canonical_source_samples.jsonl"
    )
    stored = [
        json.loads(line)
        for line in jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mismatches: List[str] = []
    if len(stored) != len(samples):
        mismatches.append(f"count live={len(samples)} stored={len(stored)}")
        return {"ok": False, "mismatches": mismatches, "compared": 0}
    compared = 0
    for live_s, stored_s in zip(samples, stored):
        compared += 1
        if live_s.canonical_sample_id != stored_s["canonical_sample_id"]:
            mismatches.append("canonical_sample_id mismatch")
            break
        if live_s.future_split_role != stored_s["future_split_role"]:
            mismatches.append("split role mismatch")
            break
        if live_s.occupancy_source_value != stored_s["occupancy_source_value"]:
            mismatches.append("target mismatch")
            break
        if live_s.co2_slope_status != stored_s["co2_slope_status"]:
            mismatches.append("slope status mismatch")
            break
        if live_s.co2_slope_status == STATUS_AVAILABLE:
            if live_s.co2_slope != stored_s["co2_slope"]:
                mismatches.append("slope value mismatch")
                break
        elif stored_s["co2_slope"] is not None:
            mismatches.append("warmup slope must be null")
            break
    return {"ok": not mismatches, "mismatches": mismatches, "compared": compared}


def build_predecessor_fingerprint_closure(repo_root: Path) -> Dict[str, Any]:
    c_a5_reg = _load_json(
        repo_root
        / "datasets/co2/manifests/c_a5_canonical_samples/predecessor_fingerprint_registry.json"
    )
    errors = verify_predecessor_fingerprints(c_a5_reg, repo_root)
    # Also fingerprint C-A5 itself for C-A6 closure
    c_a5_files = [
        "datasets/co2/manifests/c_a5_canonical_samples/canonical_sample_profile.json",
        "datasets/co2/manifests/c_a5_canonical_samples/predecessor_fingerprint_registry.json",
        "datasets/co2/manifests/c_a5_canonical_samples/split_membership_manifest.json",
        "datasets/co2/manifests/c_a5_canonical_samples/feature_availability_manifest.json",
        "datasets/co2/manifests/c_a5_canonical_samples/materialization_integrity_summary.json",
        "datasets/co2/manifests/c_a5_canonical_samples/canonical_source_samples.jsonl",
        "datasets/co2/manifests/c_a5_canonical_samples/model_eligible_sample_ids.jsonl",
        "datasets/co2/manifests/c_a5_canonical_samples/artifact_identity.json",
        "datasets/co2/manifests/c_a5_canonical_samples/checksums.sha256",
    ]
    c_a5_entries = []
    for rel in c_a5_files:
        path = repo_root / rel
        c_a5_entries.append(
            {
                "path": rel,
                "sha256": compute_sha256_file(path),
                "byte_size": path.stat().st_size,
            }
        )
    return {
        "manifest_version": "1.0",
        "consumer_phase": "C-A6",
        "audit_profile_id": AUDIT_PROFILE_ID,
        "c_a5_predecessor_registry_path": (
            "datasets/co2/manifests/c_a5_canonical_samples/predecessor_fingerprint_registry.json"
        ),
        "c_a5_predecessor_fingerprint_errors": errors,
        "c_a5_predecessor_fingerprints_ok": len(errors) == 0,
        "c_a5_artifact_fingerprints": c_a5_entries,
        "status": "LOCKED" if not errors else "PREDECESSOR_FINGERPRINT_MISMATCH",
    }


def build_artifact_lock_manifest(repo_root: Path) -> Dict[str, Any]:
    artifacts = []
    for phase, role, rel in A_SERIES_LOCK_ARTIFACTS:
        path = repo_root / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing lock target: {rel}")
        artifacts.append(
            {
                "path": rel,
                "artifact_role": role,
                "artifact_phase": phase,
                "byte_size": path.stat().st_size,
                "sha256": compute_sha256_file(path),
                "schema_or_profile_id": None,
                "release_inclusion_status": "INCLUDED_IN_A_SERIES_LOCK",
            }
        )
    # Attach known profile IDs
    profile_map = {
        "datasets/co2/manifests/c_a1_safe_reader/source_schema_profile.json": (
            "C-A1_UCI_OCCUPANCY_SCHEMA_PROFILE_001"
        ),
        "datasets/co2/manifests/c_a3_slope_feature/co2_slope_feature_profile.json": SLOPE_PROFILE_ID,
        "datasets/co2/manifests/c_a4_target_semantics/occupancy_target_profile.json": (
            OCCUPANCY_TARGET_PROFILE_ID
        ),
        "datasets/co2/manifests/c_a5_canonical_samples/canonical_sample_profile.json": (
            CANONICAL_SAMPLE_PROFILE_ID
        ),
    }
    for art in artifacts:
        if art["path"] in profile_map:
            art["schema_or_profile_id"] = profile_map[art["path"]]

    raw = independently_hash_raw_archive(repo_root)
    return {
        "manifest_version": "1.0",
        "lock_profile_id": LOCK_PROFILE_ID,
        "track": "CO2",
        "milestone": "A_SERIES_RAW_TO_CANONICAL",
        "phases_covered": ["C-A0", "C-A1", "C-A2", "C-A3", "C-A4", "C-A5", "C-A6"],
        "self_reference_policy": {
            "artifact_lock_manifest_hashes_itself": False,
            "checksums_sha256_hashes_itself": False,
            "checksums_sha256_hashes_artifact_lock_manifest": True,
            "artifact_lock_excludes": [
                f"{MANIFEST_DIR_REL}/artifact_lock_manifest.json",
                f"{MANIFEST_DIR_REL}/checksums.sha256",
            ],
        },
        "raw_archive_lock": raw,
        "locked_artifact_count": len(artifacts),
        "artifacts": artifacts,
        "note": (
            "Raw ZIP is recorded by identity only and remains GIT_IGNORED; "
            "it is not included in the Git release payload."
        ),
    }


def verify_artifact_lock(lock: Dict[str, Any], repo_root: Path) -> List[str]:
    errors: List[str] = []
    for art in lock.get("artifacts", []):
        rel = art["path"]
        path = repo_root / rel
        if not path.exists():
            errors.append(f"Locked artifact missing: {rel}")
            continue
        if compute_sha256_file(path) != art["sha256"]:
            errors.append(f"Artifact lock hash mismatch: {rel}")
        if path.stat().st_size != art["byte_size"]:
            errors.append(f"Artifact lock size mismatch: {rel}")
    raw = lock.get("raw_archive_lock", {})
    live = independently_hash_raw_archive(repo_root)
    if live.get("sha256") != raw.get("sha256") or live.get("byte_size") != raw.get("byte_size"):
        errors.append("Raw archive lock identity mismatch")
    if raw.get("included_in_git_release") is not False:
        errors.append("Raw archive must not be included in Git release")
    # self-reference safety
    policy = lock.get("self_reference_policy", {})
    if policy.get("artifact_lock_manifest_hashes_itself") is not False:
        errors.append("Self-referential lock policy invalid")
    if policy.get("checksums_sha256_hashes_itself") is not False:
        errors.append("Self-referential checksum policy invalid")
    return errors


def build_exceptions_registry() -> Dict[str, Any]:
    warnings = [
        {
            "code": "SOURCE_TIMEZONE_UNVERIFIED",
            "severity": "WARNING",
            "description": "Source timestamps remain timezone-naive SOURCE_ACQUISITION_CLOCK.",
        },
        {
            "code": "GROUP_INDEPENDENCE_NOT_VERIFIABLE",
            "severity": "WARNING",
            "description": "All temporal blocks originate from a single office room.",
        },
        {
            "code": "MODEL_TRAINING_LINEAGE_UNVERIFIED",
            "severity": "WARNING",
            "description": "Existing TFLite training lineage remains unverified; A-series does not promote it.",
        },
        {
            "code": "SCALER_FIT_LINEAGE_UNVERIFIED",
            "severity": "WARNING",
            "description": "Existing scaling metadata lineage remains unverified; no scaler fitted in C-A0..C-A6.",
        },
        {
            "code": "CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED",
            "severity": "WARNING",
            "description": "150s offline baseline remains CANONICAL_OFFLINE_BASELINE_DESIGN, not verified historical training.",
        },
        {
            "code": "DEVICE_UCI_CADENCE_DOMAIN_GAP",
            "severity": "WARNING",
            "description": "SCD40 runtime cadence vs UCI ~60s source cadence remains out of A-series scope.",
        },
        {
            "code": "SAFETY_RULE_CONTRACT_OUT_OF_SCOPE",
            "severity": "WARNING",
            "description": "CO2 safety-threshold contracts are out of A-series release scope.",
        },
        {
            "code": "SENSOR_HEALTH_CONTRACT_OUT_OF_SCOPE",
            "severity": "WARNING",
            "description": "SCD40/sensor-health contracts are out of A-series release scope.",
        },
        {
            "code": "MULTISENSOR_RISK_CONTRACT_OUT_OF_SCOPE",
            "severity": "WARNING",
            "description": "Multisensor risk fusion is out of A-series release scope.",
        },
        {
            "code": "DEFERRED_SHARED_INTEGRATION_UPDATE",
            "severity": "WARNING",
            "description": "Shared inventory/contract refresh deferred to a separate integration commit.",
        },
        {
            "code": "RELEASE_TAG_DEFERRED_UNTIL_C_A6_MERGE",
            "severity": "WARNING",
            "description": "Git tag and GitHub Release remain deferred until C-A6 merges to canonical main.",
        },
    ]
    return {
        "manifest_version": "1.0",
        "phase": "C-A6",
        "warnings": warnings,
        "blockers": [],
    }


def build_release_readiness_manifest(
    *,
    integrity_ok: bool,
    lock_ok: bool,
    predecessor_ok: bool,
    determinism_ok: bool,
) -> Dict[str, Any]:
    ready = integrity_ok and lock_ok and predecessor_ok and determinism_ok
    return {
        "manifest_version": "1.0",
        "profile_id": RELEASE_READINESS_PROFILE_ID,
        "track": "CO2",
        "milestone": "A_SERIES_RAW_TO_CANONICAL",
        "phases": ["C-A0", "C-A1", "C-A2", "C-A3", "C-A4", "C-A5", "C-A6"],
        "source_identity_status": "LOCKED",
        "license_status": "VERIFIED",
        "raw_integrity_status": "LOCKED" if integrity_ok else "FAIL",
        "reader_status": "LOCKED",
        "timeline_status": "LOCKED",
        "split_status": "LOCKED",
        "feature_status": "LOCKED_WITH_HISTORY_LINEAGE_WARNING",
        "target_status": "LOCKED",
        "canonical_materialization_status": "LOCKED" if integrity_ok else "FAIL",
        "artifact_lock_status": "PASS" if lock_ok else "FAIL",
        "determinism_status": "PASS" if determinism_ok else "FAIL",
        "regression_status": "PENDING_VALIDATOR_EXECUTION",
        "parallel_isolation_status": "REQUIRED_AT_CLOSEOUT",
        "release_ready_after_merge": ready,
        "release_target_policy": "C_A6_MERGE_COMMIT_ON_CANONICAL_MAIN",
        "release_commit": "PENDING_POST_MERGE",
        "git_tag_created": False,
        "github_release_created": False,
        "proposed_release_tag": PROPOSED_RELEASE_TAG,
        "a_series_phase_status": "C-A0_THROUGH_C-A6_COMPLETE" if ready else "BLOCKED",
        "release_scope": "CO2_REAL_RAW_TO_CANONICAL_RECONSTRUCTION_MILESTONE",
        "explicit_exclusions": [
            "CO2_MODEL_REAL_DATA_VALIDATED",
            "CO2_MODEL_DEPLOYMENT_READY",
            "SCD40_DEVICE_VALIDATED",
            "CO2_SAFETY_THRESHOLDS_CALIBRATED",
            "MULTISENSOR_INTEGRATION_VALIDATED",
            "RASPBERRY_PI_PERFORMANCE_VALIDATED",
        ],
        "status_label": (
            "CO2_A_SERIES_RELEASE_READY_AFTER_MERGE" if ready else "NOT_READY"
        ),
    }


def build_release_notes_draft() -> str:
    return """# CO₂ A-Series Release Notes Draft (Raw-to-Canonical)

> Draft only. Do **not** publish a GitHub Release from the C-A6 feature branch.
> Tag/release target policy: exact C-A6 merge commit on canonical `main`.

## Milestone

CO₂ real raw-to-canonical reconstruction milestone complete (C-A0 through C-A6).

Proposed tag: `co2-a-series-raw-to-canonical`

## Source identity

- Dataset: UCI Occupancy Detection (UCI Dataset ID 357)
- UCI dataset DOI: `10.24432/C5X01N`
- Journal publication DOI (separate): `10.1016/j.enbuild.2015.11.071`
- License: CC-BY-4.0 (verified)
- Raw archive path: `datasets/raw_archives/external_datasets/occupancy+detection.zip`
- Raw archive size: 335713 bytes
- Raw archive SHA-256: `4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a`
- Raw payload: **not** included in Git release; materialize separately under the approved provenance contract.

## Reconstruction counts

- Source rows / canonical source samples: 20560
- Model-eligible (CO2_slope available): 20551
- Warm-up preserved: 9
- Temporal blocks: BLOCK_01_DATATEST, BLOCK_02_DATATRAINING, BLOCK_03_DATATEST2
- Split roles: TRAIN / VALIDATION / LOCKED_TEST (immutable C-A2 block assignment)
- Target: Occupancy 0=VACANT (15810), 1=OCCUPIED (4750); derivation NONE
- Slope profile: `CO2_SLOPE_FEATURE_PROFILE_001` / ENDPOINT_DIFFERENCE / ppm/min

## Artifact lock

C-A0..C-A6 machine-readable artifacts are checksum-locked under
`datasets/co2/manifests/c_a6_final_integrity_lock/`.

## Explicit non-claims

This release does **not** mean:

- CO₂ model real-data validated
- CO₂ model deployment-ready
- SCD40 device validated
- CO₂ safety thresholds calibrated
- Multisensor integration validated
- Raspberry Pi performance validated

Existing model/scaler lineage remains `MODEL_TRAINING_LINEAGE_UNVERIFIED` /
`SCALER_FIT_LINEAGE_UNVERIFIED` / `CONFIRMED_SYNTHETIC_ONLY` where applicable.

## Major limitations retained

- SOURCE_TIMEZONE_UNVERIFIED
- GROUP_INDEPENDENCE_NOT_VERIFIABLE
- CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED
- DEVICE_UCI_CADENCE_DOMAIN_GAP
- SAFETY_RULE_CONTRACT_OUT_OF_SCOPE
- SENSOR_HEALTH_CONTRACT_OUT_OF_SCOPE
- MULTISENSOR_RISK_CONTRACT_OUT_OF_SCOPE

## Next phase

After tag/release on the exact C-A6 merge commit: begin C-B0 offline real-data model comparison against this locked baseline.
"""


def assert_no_forbidden_path_markers(text: str) -> List[str]:
    errors = []
    for marker in ("/Users/", "file://", "~/"):
        if marker in text:
            errors.append(f"Forbidden path marker: {marker}")
    return errors
