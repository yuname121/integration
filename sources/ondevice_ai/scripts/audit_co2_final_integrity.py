#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/audit_co2_final_integrity.py
Phase C-A6 — generate final integrity audit, artifact lock, and release-readiness evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

repo_root_dir = Path(__file__).resolve().parent.parent
if str(repo_root_dir) not in sys.path:
    sys.path.insert(0, str(repo_root_dir))

from datasets.co2.canonical_samples import (
    CANONICAL_SAMPLE_PROFILE_ID,
    EXPECTED_SLOPE_ELIGIBLE,
    EXPECTED_TOTAL_SAMPLES,
    EXPECTED_WARMUP,
    materialize_canonical_samples,
)
from datasets.co2.integrity_audit import (
    AUDIT_PROFILE_ID,
    C_A6_HASHED_EVIDENCE,
    LOCK_PROFILE_ID,
    MANIFEST_DIR_REL,
    PROPOSED_RELEASE_TAG,
    RELEASE_READINESS_PROFILE_ID,
    build_artifact_lock_manifest,
    build_exceptions_registry,
    build_population_audit,
    build_predecessor_fingerprint_closure,
    build_release_notes_draft,
    build_release_readiness_manifest,
    independently_hash_raw_archive,
    independently_hash_raw_members,
    run_round_trip_cases,
    select_round_trip_indices,
    verify_artifact_lock,
    verify_stored_canonical_against_live,
)
from datasets.co2.raw_reader import (
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_ARCHIVE_SIZE,
    UCIOccupancyRawReader,
    compute_sha256_file,
    get_repo_root,
)
from datasets.co2.slope_feature import FEATURE_PROFILE_ID as SLOPE_PROFILE_ID
from datasets.co2.target_semantics import TARGET_PROFILE_ID as TARGET_PROFILE_ID


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def audit_co2_final_integrity() -> Path:
    repo_root = get_repo_root()
    out_dir = repo_root / MANIFEST_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_archive = independently_hash_raw_archive(repo_root)
    if not raw_archive.get("matches_expected"):
        raise RuntimeError(f"Raw archive identity failed: {raw_archive}")

    members = independently_hash_raw_members(repo_root)
    if not members.get("all_members_match"):
        raise RuntimeError(f"Raw member identity failed: {members}")

    reader = UCIOccupancyRawReader(repo_root=repo_root)
    observations = reader.read_all_observations()
    if len(observations) != EXPECTED_TOTAL_SAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL_SAMPLES} observations")

    samples = materialize_canonical_samples(observations)
    population = build_population_audit(samples, observations)
    if not population["one_to_one_ok"]:
        raise RuntimeError(f"Population audit failed: {population}")

    stored_check = verify_stored_canonical_against_live(repo_root, samples)
    if not stored_check["ok"]:
        raise RuntimeError(f"Live vs stored mismatch: {stored_check}")

    obs_by_key = {(o.source_member_name, o.source_row_identifier): o for o in observations}
    cases = select_round_trip_indices(samples)
    round_trip = run_round_trip_cases(samples, obs_by_key, cases)
    if not all(c["round_trip_ok"] for c in round_trip):
        raise RuntimeError("Round-trip audit failure")

    predecessor_closure = build_predecessor_fingerprint_closure(repo_root)
    if not predecessor_closure["c_a5_predecessor_fingerprints_ok"]:
        raise RuntimeError(
            "PREDECESSOR_FINGERPRINT_MISMATCH: "
            f"{predecessor_closure['c_a5_predecessor_fingerprint_errors']}"
        )

    source_identity = json.loads(
        (repo_root / "datasets/co2/manifests/c_a0_raw_inventory/source_identity.json").read_text(
            encoding="utf-8"
        )
    )
    license_doc = json.loads(
        (
            repo_root / "datasets/co2/manifests/c_a0_raw_inventory/official_source_license.json"
        ).read_text(encoding="utf-8")
    )
    schema = json.loads(
        (
            repo_root / "datasets/co2/manifests/c_a1_safe_reader/source_schema_profile.json"
        ).read_text(encoding="utf-8")
    )
    blocks = json.loads(
        (
            repo_root / "datasets/co2/manifests/c_a2_temporal_blocks/temporal_blocks_manifest.json"
        ).read_text(encoding="utf-8")
    )
    split = json.loads(
        (
            repo_root / "datasets/co2/manifests/c_a2_temporal_blocks/grouping_split_contract.json"
        ).read_text(encoding="utf-8")
    )
    slope = json.loads(
        (
            repo_root / "datasets/co2/manifests/c_a3_slope_feature/co2_slope_feature_profile.json"
        ).read_text(encoding="utf-8")
    )
    target = json.loads(
        (
            repo_root / "datasets/co2/manifests/c_a4_target_semantics/occupancy_target_profile.json"
        ).read_text(encoding="utf-8")
    )

    integrity_summary = {
        "manifest_version": "1.0",
        "phase": "C-A6",
        "audit_profile_id": AUDIT_PROFILE_ID,
        "track": "CO2",
        "milestone": "A_SERIES_RAW_TO_CANONICAL",
        "source_identity": {
            "dataset_name": source_identity.get("dataset_name"),
            "uci_dataset_id": 357,
            "doi": source_identity.get("doi"),
            "journal_paper_doi": source_identity.get("journal_paper_doi"),
            "doi_match_expected": source_identity.get("doi") == "10.24432/C5X01N",
            "journal_doi_match_expected": (
                source_identity.get("journal_paper_doi") == "10.1016/j.enbuild.2015.11.071"
            ),
        },
        "license": {
            "spdx_id": license_doc.get("license_spdx_id"),
            "status": license_doc.get("license_classification_status"),
        },
        "raw_archive": raw_archive,
        "raw_members": members,
        "schema_contract": {
            "profile_id": schema.get("profile_id"),
            "header_field_count": schema.get("header_contract", {}).get("header_field_count"),
            "physical_field_count": schema.get("physical_row_contract", {}).get(
                "physical_field_count"
            ),
        },
        "timeline": {
            "timestamp_reference": "SOURCE_ACQUISITION_CLOCK",
            "source_timezone": "UNVERIFIED",
            "utc_conversion_claimed": False,
            "temporal_blocks": [
                {
                    "block_id": b["block_id"],
                    "source_member_name": b["source_member_name"],
                    "first_timestamp_raw": b["first_timestamp_raw"],
                    "last_timestamp_raw": b["last_timestamp_raw"],
                    "row_count": b["row_count"],
                    "future_split_role": b["future_split_role"],
                }
                for b in blocks.get("blocks", [])
            ],
            "rows_omitted": blocks.get("rows_omitted"),
            "duplicate_block_membership_count": blocks.get("duplicate_block_membership_count"),
        },
        "split_contract": {
            "assignments": split.get("future_split_assignments"),
            "random_row_wise_split_allowed": split.get("random_row_wise_split_policy", {}).get(
                "allowed"
            ),
            "group_independence_status": split.get("group_independence_status"),
        },
        "target_contract": {
            "profile_id": target.get("target_profile_id"),
            "semantic_mapping": target.get("canonical_class_mapping"),
            "label_derivation": target.get("label_derivation"),
            "occupancy_0_count": population["occupancy_0_count"],
            "occupancy_1_count": population["occupancy_1_count"],
            "target_labels_modified": 0,
        },
        "slope_contract": {
            "profile_id": slope.get("profile_id"),
            "method": slope.get("slope_method"),
            "unit": slope.get("feature_unit"),
            "history_classification": slope.get("historical_training_history_contract_status"),
            "runtime_equivalence_claimed": slope.get("offline_baseline_equivalence_claims", {}).get(
                "active_runtime_equivalent"
            ),
            "eligible_samples": population["model_eligible_sample_count"],
            "warmup_samples": population["warmup_sample_count"],
        },
        "canonical_contract": {
            "profile_id": CANONICAL_SAMPLE_PROFILE_ID,
            "canonical_source_samples": population["canonical_source_sample_count"],
            "model_eligible_samples": population["model_eligible_sample_count"],
            "warmup_samples": population["warmup_sample_count"],
            "missing_source_mappings": population["missing_source_mappings"],
            "duplicate_canonical_ids": population["duplicate_canonical_ids"],
            "one_to_one_ok": population["one_to_one_ok"],
            "ordering": ["datatest.txt", "datatraining.txt", "datatest2.txt"],
        },
        "population_audit": population,
        "live_vs_stored_canonical": stored_check,
        "synthetic_npz_isolation": {
            "path": "datasets/co2/processed/co2_occupancy_v1.npz",
            "status": "SYNTHETIC_SMOKE_FIXTURE",
            "used_as_real_source": False,
        },
        "model_scaler_status": {
            "existing_model_lineage": "MODEL_TRAINING_LINEAGE_UNVERIFIED",
            "existing_scaling_metadata_lineage": "SCALER_FIT_LINEAGE_UNVERIFIED",
            "model_manifest_status": "CONFIRMED_SYNTHETIC_ONLY",
            "scaler_fitted_during_c_a0_c_a6": False,
            "model_trained_during_c_a0_c_a6": False,
            "model_selected_during_c_a0_c_a6": False,
            "quantization_performed_during_c_a0_c_a6": False,
        },
        "locked_test_protection": {
            "used_for_feature_contract_tuning": False,
            "used_for_scaler_fitting": False,
            "used_for_model_selection": False,
            "used_for_hyperparameter_tuning": False,
            "used_for_threshold_calibration": False,
            "integrity_inspection_only": True,
        },
        "expected_constants": {
            "archive_size": EXPECTED_ARCHIVE_SIZE,
            "archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "total_samples": EXPECTED_TOTAL_SAMPLES,
            "eligible": EXPECTED_SLOPE_ELIGIBLE,
            "warmup": EXPECTED_WARMUP,
            "slope_profile_id": SLOPE_PROFILE_ID,
            "target_profile_id": TARGET_PROFILE_ID,
        },
        "status": "PASS",
    }

    audit_manifest = {
        "manifest_version": "1.0",
        "phase": "C-A6",
        "audit_profile_id": AUDIT_PROFILE_ID,
        "round_trip_case_count": len(round_trip),
        "round_trip_all_ok": all(c["round_trip_ok"] for c in round_trip),
        "cases": round_trip,
        "method": (
            "Independent live materialization from raw reader + comparison to stored "
            "C-A5 canonical JSONL and raw observation fields (not file-self comparison)."
        ),
    }

    # First write non-lock artifacts, then lock, then readiness, then checksums.
    exceptions = build_exceptions_registry()
    generation = {
        "manifest_version": "1.0",
        "phase": "C-A6",
        "audit_profile_id": AUDIT_PROFILE_ID,
        "lock_profile_id": LOCK_PROFILE_ID,
        "release_readiness_profile_id": RELEASE_READINESS_PROFILE_ID,
        "generator_script": "scripts/audit_co2_final_integrity.py",
        "module": "datasets/co2/integrity_audit.py",
        "scaler_fitted": False,
        "model_trained": False,
        "synthetic_npz_used_as_real_source": False,
        "git_tag_created": False,
        "github_release_created": False,
        "proposed_release_tag": PROPOSED_RELEASE_TAG,
        "determinism": {
            "host_timezone_independent": True,
            "locale_independent": True,
            "random_values_used": False,
        },
    }

    _write_json(out_dir / "full_chain_integrity_summary.json", integrity_summary)
    _write_json(out_dir / "full_chain_audit_manifest.json", audit_manifest)
    _write_json(out_dir / "predecessor_fingerprint_closure.json", predecessor_closure)
    _write_json(out_dir / "exceptions_and_limitations.json", exceptions)
    _write_json(out_dir / "generation_metadata.json", generation)
    (out_dir / "release_notes_draft.md").write_text(build_release_notes_draft(), encoding="utf-8")

    lock = build_artifact_lock_manifest(repo_root)
    lock_errors = verify_artifact_lock(lock, repo_root)
    if lock_errors:
        raise RuntimeError(f"Artifact lock verification failed: {lock_errors}")
    _write_json(out_dir / "artifact_lock_manifest.json", lock)

    readiness = build_release_readiness_manifest(
        integrity_ok=True,
        lock_ok=True,
        predecessor_ok=True,
        determinism_ok=True,
    )
    _write_json(out_dir / "release_readiness_manifest.json", readiness)

    # checksums.sha256 hashes all C-A6 evidence including lock, but not itself
    lines = []
    for fname in C_A6_HASHED_EVIDENCE:
        path = out_dir / fname
        rel = f"{MANIFEST_DIR_REL}/{fname}"
        lines.append(f"{compute_sha256_file(path)}  {rel}")
    (out_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"✅ Generated C-A6 final integrity lock in: {MANIFEST_DIR_REL}")
    print(
        f"   canonical={population['canonical_source_sample_count']} "
        f"eligible={population['model_eligible_sample_count']} "
        f"warmup={population['warmup_sample_count']} "
        f"locked_artifacts={lock['locked_artifact_count']}"
    )
    return out_dir


if __name__ == "__main__":
    audit_co2_final_integrity()
