#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/validate_co2_final_integrity.py
Phase C-A6 — standalone full-chain integrity / artifact-lock / release-readiness validator.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    EXPECTED_OCC_0,
    EXPECTED_OCC_1,
    LOCK_PROFILE_ID,
    MANIFEST_DIR_REL,
    PROPOSED_RELEASE_TAG,
    RELEASE_READINESS_PROFILE_ID,
    assert_no_forbidden_path_markers,
    build_population_audit,
    independently_hash_raw_archive,
    independently_hash_raw_members,
    verify_artifact_lock,
    verify_predecessor_fingerprints,
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

FORBIDDEN_PATH_MARKERS = ("/Users/", "file://", "~/")
PROTECTED_SHARED = {
    "AGENTS.md",
    "README.md",
    "docs/README.md",
    "datasets/MANIFEST.json",
    "models/model_manifest.json",
    "docs/reports/model_inventory.json",
    "docs/reports/SENSOR_DATA_CONTRACT.md",
    "docs/reports/sensor_model_data_contract.json",
    "models/co2/co2_scaling_metadata_v0.1.0.json",
    "datasets/co2/processed/co2_occupancy_v1.npz",
    "sensors/co2/co2_adapter.py",
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_validator(script: str, repo_root: Path) -> bool:
    res = subprocess.run(
        ["python3", script],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    return res.returncode == 0


def derive_gate(errors: int, warnings: int) -> Tuple[str, str]:
    if errors > 0:
        return "FAIL", "NO"
    if warnings > 0:
        return "PASS_WITH_WARNINGS", "YES"
    return "PASS", "YES"


def validate_c_a6(repo_root: Path) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    errors: List[str] = []
    warnings: List[str] = []
    c_a6 = repo_root / MANIFEST_DIR_REL
    required = [
        "full_chain_integrity_summary.json",
        "full_chain_audit_manifest.json",
        "artifact_lock_manifest.json",
        "predecessor_fingerprint_closure.json",
        "release_readiness_manifest.json",
        "exceptions_and_limitations.json",
        "generation_metadata.json",
        "release_notes_draft.md",
        "checksums.sha256",
    ]
    for fname in required:
        if not (c_a6 / fname).exists():
            errors.append(f"Missing C-A6 artifact: {fname}")
    if errors:
        return False, errors, warnings, {}

    preds = {
        "c_a0": _run_validator("scripts/validate_co2_raw_inventory.py", repo_root),
        "c_a1": _run_validator("scripts/validate_co2_safe_reader.py", repo_root),
        "c_a2": _run_validator("scripts/validate_co2_temporal_blocks.py", repo_root),
        "c_a3": _run_validator("scripts/validate_co2_slope_feature.py", repo_root),
        "c_a4": _run_validator("scripts/validate_co2_target_semantics.py", repo_root),
        "c_a5": _run_validator("scripts/validate_co2_canonical_samples.py", repo_root),
    }
    if not all(preds.values()):
        for k, ok in preds.items():
            if not ok:
                errors.append(f"Predecessor validator failed: {k}")

    integrity = _load_json(c_a6 / "full_chain_integrity_summary.json")
    audit = _load_json(c_a6 / "full_chain_audit_manifest.json")
    lock = _load_json(c_a6 / "artifact_lock_manifest.json")
    closure = _load_json(c_a6 / "predecessor_fingerprint_closure.json")
    readiness = _load_json(c_a6 / "release_readiness_manifest.json")
    generation = _load_json(c_a6 / "generation_metadata.json")
    exceptions = _load_json(c_a6 / "exceptions_and_limitations.json")

    # Independent live checks
    raw = independently_hash_raw_archive(repo_root)
    if not raw.get("matches_expected"):
        errors.append("Raw archive size/hash mismatch")
    members = independently_hash_raw_members(repo_root)
    if not members.get("all_members_match"):
        errors.append("Raw member hash/size/row mismatch")

    observations = UCIOccupancyRawReader(repo_root=repo_root).read_all_observations()
    samples = materialize_canonical_samples(observations)
    population = build_population_audit(samples, observations)
    if not population["one_to_one_ok"]:
        errors.append("Population one-to-one integrity failed")
    if population["canonical_source_sample_count"] != EXPECTED_TOTAL_SAMPLES:
        errors.append("Canonical sample count mismatch")
    if population["model_eligible_sample_count"] != EXPECTED_SLOPE_ELIGIBLE:
        errors.append("Model-eligible count mismatch")
    if population["warmup_sample_count"] != EXPECTED_WARMUP:
        errors.append("Warm-up count mismatch")
    if population["occupancy_0_count"] != EXPECTED_OCC_0:
        errors.append("Occupancy 0 count mismatch")
    if population["occupancy_1_count"] != EXPECTED_OCC_1:
        errors.append("Occupancy 1 count mismatch")
    if population["missing_source_mappings"] != 0:
        errors.append("Missing source mappings")
    if population["duplicate_canonical_ids"] != 0:
        errors.append("Duplicate canonical IDs")

    # Manifest consistency
    if integrity.get("audit_profile_id") != AUDIT_PROFILE_ID:
        errors.append("Unexpected audit profile id")
    if lock.get("lock_profile_id") != LOCK_PROFILE_ID:
        errors.append("Unexpected lock profile id")
    if readiness.get("profile_id") != RELEASE_READINESS_PROFILE_ID:
        errors.append("Unexpected release-readiness profile id")
    src = integrity.get("source_identity", {})
    if src.get("doi") != "10.24432/C5X01N":
        errors.append("UCI dataset DOI incorrect")
    if src.get("journal_paper_doi") != "10.1016/j.enbuild.2015.11.071":
        errors.append("Journal DOI incorrect")
    if integrity.get("license", {}).get("status") != "VERIFIED":
        errors.append("License status not verified")
    schema = integrity.get("schema_contract", {})
    if schema.get("header_field_count") != 7 or schema.get("physical_field_count") != 8:
        errors.append("Schema contract mismatch")
    timeline = integrity.get("timeline", {})
    if timeline.get("timestamp_reference") != "SOURCE_ACQUISITION_CLOCK":
        errors.append("Timestamp reference mismatch")
    if timeline.get("source_timezone") != "UNVERIFIED":
        errors.append("Source timezone must remain UNVERIFIED")
    if timeline.get("utc_conversion_claimed") is not False:
        errors.append("UTC conversion must not be claimed")
    split = integrity.get("split_contract", {})
    if split.get("random_row_wise_split_allowed") is not False:
        errors.append("Random row-wise split must be prohibited")
    if split.get("group_independence_status") != "GROUP_INDEPENDENCE_NOT_VERIFIABLE":
        errors.append("Group independence claim upgraded illegally")
    slope = integrity.get("slope_contract", {})
    if slope.get("profile_id") != SLOPE_PROFILE_ID:
        errors.append("Slope profile mismatch")
    if slope.get("runtime_equivalence_claimed") is not False:
        errors.append("Runtime equivalence must not be claimed")
    target = integrity.get("target_contract", {})
    if target.get("profile_id") != TARGET_PROFILE_ID:
        errors.append("Target profile mismatch")
    if target.get("target_labels_modified") != 0:
        errors.append("Target labels modified")
    canon = integrity.get("canonical_contract", {})
    if canon.get("profile_id") != CANONICAL_SAMPLE_PROFILE_ID:
        errors.append("Canonical sample profile mismatch")
    if integrity.get("synthetic_npz_isolation", {}).get("used_as_real_source") is not False:
        errors.append("Synthetic NPZ used as real source")
    model = integrity.get("model_scaler_status", {})
    if model.get("scaler_fitted_during_c_a0_c_a6") is not False:
        errors.append("Scaler fitted during A-series")
    if model.get("model_trained_during_c_a0_c_a6") is not False:
        errors.append("Model trained during A-series")
    ltp = integrity.get("locked_test_protection", {})
    for key in (
        "used_for_feature_contract_tuning",
        "used_for_scaler_fitting",
        "used_for_model_selection",
        "used_for_hyperparameter_tuning",
        "used_for_threshold_calibration",
    ):
        if ltp.get(key) is not False:
            errors.append(f"LOCKED_TEST protection violated: {key}")

    if not audit.get("round_trip_all_ok"):
        errors.append("Round-trip audit not all ok")
    if int(audit.get("round_trip_case_count", 0)) < 10:
        errors.append("Insufficient round-trip cases")

    if not closure.get("c_a5_predecessor_fingerprints_ok"):
        errors.append("PREDECESSOR_FINGERPRINT_MISMATCH")
    # Live re-check C-A5 registry
    c_a5_reg = _load_json(
        repo_root
        / "datasets/co2/manifests/c_a5_canonical_samples/predecessor_fingerprint_registry.json"
    )
    fp_errors = verify_predecessor_fingerprints(c_a5_reg, repo_root)
    if fp_errors:
        errors.extend(fp_errors)

    lock_errors = verify_artifact_lock(lock, repo_root)
    errors.extend(lock_errors)
    if lock.get("self_reference_policy", {}).get("artifact_lock_manifest_hashes_itself") is not False:
        errors.append("Self-referential checksum defect in lock policy")
    if lock.get("self_reference_policy", {}).get("checksums_sha256_hashes_itself") is not False:
        errors.append("checksums.sha256 must not hash itself")

    # Release-readiness conservatism
    if readiness.get("git_tag_created") is not False:
        errors.append("release-readiness false claim: git_tag_created")
    if readiness.get("github_release_created") is not False:
        errors.append("release-readiness false claim: github_release_created")
    if readiness.get("release_commit") != "PENDING_POST_MERGE":
        errors.append("release_commit must remain PENDING_POST_MERGE on feature branch")
    if readiness.get("release_target_policy") != "C_A6_MERGE_COMMIT_ON_CANONICAL_MAIN":
        errors.append("release_target_policy mismatch")
    if readiness.get("proposed_release_tag") != PROPOSED_RELEASE_TAG:
        errors.append("proposed release tag mismatch")
    if generation.get("git_tag_created") is not False:
        errors.append("generation metadata claims tag created")
    if generation.get("scaler_fitted") is not False:
        errors.append("generation metadata claims scaler fitted")
    if generation.get("model_trained") is not False:
        errors.append("generation metadata claims model trained")

    # Checksums: hash evidence including lock; never hash checksums itself
    checksum_paths = set()
    for line in (c_a6 / "checksums.sha256").read_text(encoding="utf-8").strip().splitlines():
        digest, rel = line.split("  ", 1)
        checksum_paths.add(Path(rel).name)
        if rel.endswith("checksums.sha256"):
            errors.append("checksums.sha256 must not hash itself")
        path = repo_root / rel
        if not path.exists():
            errors.append(f"Checksum path missing: {rel}")
        elif compute_sha256_file(path) != digest:
            errors.append(f"Checksum mismatch: {rel}")
    for fname in C_A6_HASHED_EVIDENCE:
        if fname not in checksum_paths:
            errors.append(f"Missing checksum entry for {fname}")
    if "checksums.sha256" in checksum_paths:
        errors.append("Self-referential checksums.sha256 entry present")

    # Path portability
    for fname in required:
        text = (c_a6 / fname).read_text(encoding="utf-8")
        for marker in FORBIDDEN_PATH_MARKERS:
            if marker in text:
                errors.append(f"Forbidden path marker {marker} in {fname}")
        errors.extend([f"{fname}: {e}" for e in assert_no_forbidden_path_markers(text)])

    # Git isolation (branch unique files)
    try:
        diff = subprocess.run(
            ["git", "diff", "--name-only", "origin/main...HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        for path in [p.strip() for p in diff.stdout.splitlines() if p.strip()]:
            if path in PROTECTED_SHARED:
                errors.append(f"Unauthorized shared file modified: {path}")
            lower = path.lower()
            if path.startswith("datasets/mmwave/") or (
                "mmwave" in lower
                and path.startswith(("scripts/", "tests/", "docs/reports/", "models/"))
            ):
                errors.append(f"mmWave file in C-A6 branch: {path}")
            if path.startswith("datasets/thermal") or (
                "thermal" in lower
                and path.startswith(("scripts/", "tests/", "docs/reports/", "datasets/"))
            ):
                errors.append(f"Thermal file in C-A6 branch: {path}")
            if "occupancy+detection.zip" in path or path.endswith(".txt") and "raw_archives" in path:
                errors.append("Raw payload staged/modified")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Git isolation check skipped: {exc}")

    for item in exceptions.get("warnings", []):
        warnings.append(f"[{item.get('code')}] {item.get('description')}")
    if exceptions.get("blockers"):
        errors.append("Exception registry contains blockers")

    # Deduplicate errors while preserving order
    deduped: List[str] = []
    seen = set()
    for e in errors:
        if e not in seen:
            seen.add(e)
            deduped.append(e)
    errors = deduped

    gate, merge_ready = derive_gate(len(errors), len(warnings))
    summary = {
        "gate_status": gate,
        "c_a6_merge_ready": merge_ready,
        "release_ready_after_merge": bool(readiness.get("release_ready_after_merge"))
        and len(errors) == 0,
        "canonical_source_samples": population["canonical_source_sample_count"],
        "slope_eligible": population["model_eligible_sample_count"],
        "warmup_unavailable": population["warmup_sample_count"],
        "locked_artifact_count": lock.get("locked_artifact_count"),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "predecessors": preds,
        "raw_archive_size": raw.get("byte_size"),
        "raw_archive_sha256": raw.get("sha256"),
    }
    return len(errors) == 0, errors, warnings, summary


def main() -> int:
    repo_root = get_repo_root()
    print(f"🔍 Validating Phase C-A6 CO₂ Final Integrity in: {MANIFEST_DIR_REL}")
    ok, errors, warnings, summary = validate_c_a6(repo_root)
    print("\n--- C-A6 VALIDATOR RESULT ---")
    print(f"Gate Status:      {summary.get('gate_status', 'FAIL')}")
    print(f"Merge Ready:      {summary.get('c_a6_merge_ready', 'NO')}")
    print(f"Release After Merge: {summary.get('release_ready_after_merge')}")
    print(f"Canonical Samples:{summary.get('canonical_source_samples')}")
    print(f"Slope Eligible:   {summary.get('slope_eligible')}")
    print(f"Warm-up Rows:     {summary.get('warmup_unavailable')}")
    print(f"Locked Artifacts: {summary.get('locked_artifact_count')}")
    print(f"Error Count:      {summary.get('error_count', len(errors))}")
    print(f"Warning Count:    {summary.get('warning_count', len(warnings))}")
    if errors:
        print("\nRecorded Errors:")
        for err in errors:
            print(f" ❌  {err}")
    if warnings:
        print("\nRecorded Warnings & Limitations:")
        for warn in warnings:
            print(f" ⚠️  {warn}")
    if ok:
        print("\n✅ SUCCESS: Phase C-A6 final integrity lock is valid.")
        return 0
    print("\n❌ FAILURE: Phase C-A6 validation failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
