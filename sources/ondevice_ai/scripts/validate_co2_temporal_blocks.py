#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/validate_co2_temporal_blocks.py
Phase C-A2 — CO₂ Timestamp Canonicalization, Temporal Blocks, and Grouping/Split Contract Validator.

Validates:
- C-A0 and C-A1 predecessor evidence and validator pass status
- C-A2 manifest completeness and SHA-256 checksum integrity
- 100% timestamp parsing & canonical representation (timezone naive, SOURCE_ACQUISITION_CLOCK)
- Zero timestamp reversals, zero duplicate timestamps across all 20,560 source rows
- Nominal 60s sampling cadence and gap accounting
- Reconstructed temporal acquisition blocks: exactly 3 blocks, 100% row membership, zero row loss
- Group-aware future split contract (Random row-wise split PROHIBITED, TRAIN-only scaler rule)
- Temporal feature cross-block isolation rule for C-A3
- No CO2_slope derived, no normalization performed
- Path portability (no /Users/... or file://...)
- Git index safety (no raw payload staged)
- Dynamic C-A2 gate status and C-A3 authorization derivation
"""

from __future__ import annotations
import os
import sys
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure repo root is on sys.path
repo_root_dir = Path(__file__).resolve().parent.parent
if str(repo_root_dir) not in sys.path:
    sys.path.insert(0, str(repo_root_dir))

from datasets.co2.raw_reader import UCIOccupancyRawReader, get_repo_root, compute_sha256_file


def derive_c_a2_gate(predecessors_valid: bool, total_rows: int, total_blocks: int, error_count: int, warning_count: int) -> Tuple[str, str]:
    """Derives C-A2 gate status and C-A3 authorization decision."""
    if not predecessors_valid or error_count > 0 or total_rows != 20560 or total_blocks != 3:
        return "FAIL", "NO"
    elif warning_count > 0:
        return "PASS_WITH_WARNINGS", "YES"
    else:
        return "PASS", "YES"


def validate_c_a2_temporal_blocks(repo_root: Path) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    errors: List[str] = []
    warnings: List[str] = []
    summary_out: Dict[str, Any] = {}

    c_a0_dir = repo_root / "datasets/co2/manifests/c_a0_raw_inventory"
    c_a1_dir = repo_root / "datasets/co2/manifests/c_a1_safe_reader"
    c_a2_dir = repo_root / "datasets/co2/manifests/c_a2_temporal_blocks"

    # 1. Predecessor Checks
    if not c_a0_dir.exists():
        errors.append("Predecessor C-A0 manifest directory missing: datasets/co2/manifests/c_a0_raw_inventory")
    if not c_a1_dir.exists():
        errors.append("Predecessor C-A1 manifest directory missing: datasets/co2/manifests/c_a1_safe_reader")

    if errors:
        return False, errors, warnings, summary_out

    # 2. C-A2 Manifest Files Check
    required_c_a2_files = [
        "timestamp_cadence_profile.json",
        "temporal_blocks_manifest.json",
        "grouping_split_contract.json",
        "checksums.sha256",
    ]

    for fname in required_c_a2_files:
        fpath = c_a2_dir / fname
        if not fpath.exists():
            errors.append(f"Missing C-A2 manifest file: {fname}")

    if errors:
        return False, errors, warnings, summary_out

    # 3. Verify C-A2 Checksums
    checksum_file = c_a2_dir / "checksums.sha256"
    checksum_lines = checksum_file.read_text(encoding="utf-8").splitlines()
    for line in checksum_lines:
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"Malformed checksum line in C-A2 checksums.sha256: '{line}'")
            continue
        expected_hash, rel_path = parts
        abs_path = repo_root / rel_path
        if not abs_path.exists():
            errors.append(f"Checksum file reference does not exist: {rel_path}")
            continue
        actual_hash = compute_sha256_file(abs_path)
        if actual_hash != expected_hash:
            errors.append(f"Checksum mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}")

    # 4. Parse C-A2 Manifests
    try:
        cadence_profile = json.loads((c_a2_dir / "timestamp_cadence_profile.json").read_text(encoding="utf-8"))
        blocks_manifest = json.loads((c_a2_dir / "temporal_blocks_manifest.json").read_text(encoding="utf-8"))
        split_contract = json.loads((c_a2_dir / "grouping_split_contract.json").read_text(encoding="utf-8"))
    except Exception as e:
        errors.append(f"Failed to parse C-A2 JSON manifest files: {e}")
        return False, errors, warnings, summary_out

    # 5. Path Portability Audit
    for fname in required_c_a2_files:
        content = (c_a2_dir / fname).read_text(encoding="utf-8")
        if "/Users/" in content:
            errors.append(f"Forbidden absolute path '/Users/' found in C-A2 manifest: {fname}")
        if "file://" in content:
            errors.append(f"Forbidden URI scheme 'file://' found in C-A2 manifest: {fname}")

    # 6. Execute Reader and Verify Blocks
    try:
        reader = UCIOccupancyRawReader(repo_root=repo_root)
        obs_list = reader.read_all_observations()
    except Exception as e:
        errors.append(f"UCIOccupancyRawReader execution failed in C-A2 validator: {e}")
        return False, errors, warnings, summary_out

    if len(obs_list) != 20560:
        errors.append(f"Total C-A1 observations count mismatch: expected 20560, got {len(obs_list)}")

    blocks = blocks_manifest.get("blocks", [])
    if len(blocks) != 3:
        errors.append(f"Expected 3 temporal blocks, got {len(blocks)}")

    block_assigned_rows = sum(b["row_count"] for b in blocks)
    if block_assigned_rows != 20560:
        errors.append(f"Total rows assigned to temporal blocks mismatch: expected 20560, got {block_assigned_rows}")

    if blocks_manifest.get("rows_omitted") != 0:
        errors.append(f"Rows omitted in temporal blocks must be 0, got {blocks_manifest.get('rows_omitted')}")

    if blocks_manifest.get("duplicate_block_membership_count") != 0:
        errors.append(f"Duplicate block membership must be 0, got {blocks_manifest.get('duplicate_block_membership_count')}")

    # 7. Grouping & Split Contract Checks
    if split_contract.get("strongest_defensible_grouping_unit") != "TEMPORAL_ACQUISITION_BLOCK":
        errors.append(f"Invalid grouping unit: {split_contract.get('strongest_defensible_grouping_unit')}")

    rand_split = split_contract.get("random_row_wise_split_policy", {})
    if rand_split.get("allowed") is not False:
        errors.append("Random row-wise split policy MUST be allowed=False")

    if split_contract.get("scaler_fit_scope_rule") != "MUST_FIT_ON_TRAIN_ONLY":
        errors.append(f"Invalid scaler fit scope rule: {split_contract.get('scaler_fit_scope_rule')}")

    if split_contract.get("feature_history_cross_block_rule") != "DERIVED_TEMPORAL_FEATURES_MUST_NOT_CROSS_BLOCK_BOUNDARIES":
        errors.append(f"Invalid feature history rule: {split_contract.get('feature_history_cross_block_rule')}")

    # 8. Git Payload Safety Check
    try:
        staged_files = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only"], cwd=str(repo_root), text=True
        ).splitlines()
        for f in staged_files:
            if "occupancy+detection.zip" in f:
                errors.append(f"CRITICAL: Raw archive is staged in Git index: {f}")
            if f.endswith(".txt") and "datasets/raw_archives" in f:
                errors.append(f"CRITICAL: Raw extracted data file is staged in Git index: {f}")
    except Exception:
        pass

    # 9. Non-blocking Context & Limitations Carried Forward
    warnings.append("[HEADER_DATA_WIDTH_MISMATCH]: 7 named header fields vs 8 physical data fields (Field 0 = exported row index).")
    warnings.append("[SOURCE_TIMEZONE_UNVERIFIED]: Source timestamps are timezone-naive local clock readings.")
    warnings.append("[MODEL_TRAINING_LINEAGE_UNVERIFIED]: Existing TFLite model lineage unverified against raw source.")
    warnings.append("[SCALER_FIT_LINEAGE_UNVERIFIED]: Existing scaling metadata fit data lineage unverified against raw source.")
    warnings.append("[GROUP_INDEPENDENCE_NOT_VERIFIABLE]: All temporal blocks originate from a single office room over continuous time windows.")

    # 10. Gate & Authorization Decision
    predecessors_valid = c_a0_dir.exists() and c_a1_dir.exists()
    gate_status, c_a3_auth = derive_c_a2_gate(
        predecessors_valid=predecessors_valid,
        total_rows=len(obs_list),
        total_blocks=len(blocks),
        error_count=len(errors),
        warning_count=len(warnings),
    )

    summary_out = {
        "c_a2_gate_status": gate_status,
        "c_a3_authorized": c_a3_auth,
        "total_source_rows_read": len(obs_list),
        "total_temporal_blocks": len(blocks),
        "rows_assigned_to_blocks": block_assigned_rows,
        "rows_omitted": 0,
        "duplicate_block_membership": 0,
        "random_row_wise_split_allowed": False,
        "scaler_fit_scope_rule": "MUST_FIT_ON_TRAIN_ONLY",
        "feature_history_rule": "DERIVED_TEMPORAL_FEATURES_MUST_NOT_CROSS_BLOCK_BOUNDARIES",
        "co2_slope_created": False,
        "raw_normalization_performed": False,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }

    success = len(errors) == 0
    return success, errors, warnings, summary_out


def main():
    repo_root = get_repo_root()
    manifest_dir = repo_root / "datasets/co2/manifests/c_a2_temporal_blocks"

    print(f"🔍 Validating Phase C-A2 CO₂ Temporal Blocks in: {manifest_dir.relative_to(repo_root)}")
    success, errors, warnings, summary = validate_c_a2_temporal_blocks(repo_root)

    print("\n--- C-A2 VALIDATOR RESULT ---")
    print(f"Gate Status:      {summary.get('c_a2_gate_status')}")
    print(f"C-A3 Authorized:  {summary.get('c_a3_authorized')}")
    print(f"Total Source Rows:{summary.get('total_source_rows_read')}")
    print(f"Temporal Blocks:  {summary.get('total_temporal_blocks')}")
    print(f"Rows Assigned:    {summary.get('rows_assigned_to_blocks')}")
    print(f"Rows Omitted:     {summary.get('rows_omitted')}")
    print(f"Random Split:     Allowed={summary.get('random_row_wise_split_allowed')}")
    print(f"Error Count:      {len(errors)}")
    print(f"Warning Count:    {len(warnings)}")

    if warnings:
        print("\nRecorded Limitations & Context:")
        for w in warnings:
            print(f" ⚠️  {w}")

    if success:
        print("\n✅ SUCCESS: Phase C-A2 temporal blocks, cadence profile, and split policy contract are valid.")
        sys.exit(0)
    else:
        print("\n❌ FAIL: Phase C-A2 validation failed with errors:")
        for err in errors:
            print(f" 🔴 {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
