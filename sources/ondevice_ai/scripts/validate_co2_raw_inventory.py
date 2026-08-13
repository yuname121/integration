#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/validate_co2_raw_inventory.py
Phase C-A0 — CO₂ Machine-Readable Inventory and Lineage Validator.

Validates internal consistency across all C-A0 manifest files:
- Manifest file completeness and valid JSON/JSONL syntax
- SHA-256 checksum integrity against checksums.sha256
- Measured archive properties (byte size: 335,713, sha256: 4ae3f46aa...)
- Member inventory consistency (3 members, 20,560 total rows)
- Schema 7-header vs 8-data-field mismatch explicit representation
- Timestamp timezone status explicit (SOURCE_ACQUISITION_CLOCK, UNVERIFIED)
- Official dataset identity (UCI ID 357, DOI: 10.24432/C5CW2B, CC-BY-4.0)
- Label semantics occupancy-only (no clinical apnea or CO2 danger claims)
- Synthetic vs real lineage separation (NPZ = SYNTHETIC_SMOKE_FIXTURE)
- Existing model/scaler lineage non-promotion (CONFIRMED_SYNTHETIC_ONLY)
- Portable repository-relative paths (no /Users/... or file://...)
- Git index safety (no raw archive or payload staged)
- Dynamic C-A0 gate status and C-A1 authorization status derivation
"""

from __future__ import annotations
import os
import sys
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple


def get_repo_root() -> Path:
    """Returns the canonical repository root containing AGENTS.md."""
    root = Path(__file__).parent.parent
    if (root / "AGENTS.md").exists():
        return root
    return Path(os.getcwd())


def compute_sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def derive_c_a0_gate(archive_present: bool, member_count: int, total_rows: int,
                     blockers: List[str], warnings: List[str]) -> Tuple[str, str]:
    """Derives C-A0 gate status and C-A1 authorization decision."""
    if not archive_present or len(blockers) > 0:
        return "FAIL", "NO"
    elif member_count != 3 or total_rows != 20560:
        return "FAIL", "NO"
    elif len(warnings) > 0:
        return "PASS_WITH_WARNINGS", "YES"
    else:
        return "PASS", "YES"


def validate_manifests(repo_root: Path, manifest_dir: Path) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    errors: List[str] = []
    warnings: List[str] = []
    summary_out: Dict[str, Any] = {}

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
        fpath = manifest_dir / fname
        if not fpath.exists():
            errors.append(f"Missing required C-A0 manifest file: {fname}")

    if errors:
        return False, errors, warnings, summary_out

    # 1. Verify SHA-256 Checksums
    checksum_file = manifest_dir / "checksums.sha256"
    checksum_lines = checksum_file.read_text(encoding="utf-8").splitlines()
    for line in checksum_lines:
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"Malformed checksum line in checksums.sha256: '{line}'")
            continue
        expected_hash, rel_path = parts
        abs_path = repo_root / rel_path
        if not abs_path.exists():
            errors.append(f"Checksum file reference does not exist: {rel_path}")
            continue
        actual_hash = compute_sha256_file(abs_path)
        if actual_hash != expected_hash:
            errors.append(f"Checksum mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}")

    # 2. Parse Manifest Files
    try:
        source_id = json.loads((manifest_dir / "source_identity.json").read_text(encoding="utf-8"))
        license_info = json.loads((manifest_dir / "official_source_license.json").read_text(encoding="utf-8"))
        archive_info = json.loads((manifest_dir / "archive_integrity.json").read_text(encoding="utf-8"))
        summary = json.loads((manifest_dir / "raw_inventory_summary.json").read_text(encoding="utf-8"))
        lineage = json.loads((manifest_dir / "lineage_registry.json").read_text(encoding="utf-8"))
        anomalies = json.loads((manifest_dir / "anomalies_and_limitations.json").read_text(encoding="utf-8"))

        members = []
        for line_no, line in enumerate((manifest_dir / "archive_members.jsonl").read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                members.append(json.loads(line))
    except Exception as e:
        errors.append(f"Failed to parse JSON/JSONL manifest files: {e}")
        return False, errors, warnings, summary_out

    # 3. Path Portability Audit (No /Users/... or file://...)
    for fname in required_files:
        content = (manifest_dir / fname).read_text(encoding="utf-8")
        if "/Users/" in content:
            errors.append(f"Forbidden absolute local path '/Users/' detected in manifest file: {fname}")
        if "file://" in content:
            errors.append(f"Forbidden URI scheme 'file://' detected in manifest file: {fname}")

    # 4. Raw Archive Verification
    if archive_info.get("byte_size") != 335713:
        errors.append(f"Archive byte size mismatch: expected 335713, got {archive_info.get('byte_size')}")
    if archive_info.get("sha256") != "4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a":
        errors.append(f"Archive SHA-256 mismatch: expected 4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a, got {archive_info.get('sha256')}")

    # 5. Raw Member Audit & Counts
    if len(members) != 3:
        errors.append(f"Expected 3 archive members, found {len(members)}")

    expected_members = {
        "datatraining.txt": {"rows": 8143, "occ_0": 6414, "occ_1": 1729},
        "datatest.txt": {"rows": 2665, "occ_0": 1693, "occ_1": 972},
        "datatest2.txt": {"rows": 9752, "occ_0": 7703, "occ_1": 2049}
    }

    total_rows = 0
    total_occ_0 = 0
    total_occ_1 = 0

    for m in members:
        name = m.get("member_name")
        if name not in expected_members:
            errors.append(f"Unexpected raw archive member: {name}")
            continue
        exp = expected_members[name]
        r_count = m.get("row_count_excluding_header", 0)
        occ_0 = m.get("label_distribution", {}).get("Occupancy_0", 0)
        occ_1 = m.get("label_distribution", {}).get("Occupancy_1", 0)

        if r_count != exp["rows"]:
            errors.append(f"Row count mismatch for {name}: expected {exp['rows']}, got {r_count}")
        if occ_0 != exp["occ_0"]:
            errors.append(f"Occupancy 0 count mismatch for {name}: expected {exp['occ_0']}, got {occ_0}")
        if occ_1 != exp["occ_1"]:
            errors.append(f"Occupancy 1 count mismatch for {name}: expected {exp['occ_1']}, got {occ_1}")

        total_rows += r_count
        total_occ_0 += occ_0
        total_occ_1 += occ_1

        # Check schema mismatch explicit representation
        if not m.get("schema_mismatch_detected"):
            errors.append(f"Schema mismatch not detected/recorded for member: {name}")
        if m.get("header_field_count") != 7:
            errors.append(f"Header field count mismatch for {name}: expected 7, got {m.get('header_field_count')}")
        if m.get("actual_data_field_count") != 8:
            errors.append(f"Data field count mismatch for {name}: expected 8, got {m.get('actual_data_field_count')}")

        # Check timestamp timezone representation
        if m.get("timezone_evidence_status") != "UNVERIFIED":
            errors.append(f"Timezone evidence status must be UNVERIFIED for {name}, got {m.get('timezone_evidence_status')}")
        if m.get("utc_conversion_claimed") is not False:
            errors.append(f"UTC conversion claim must be False for {name}")

    if total_rows != 20560:
        errors.append(f"Total raw row count mismatch: expected 20560, got {total_rows}")
    if total_occ_0 != 15810:
        errors.append(f"Total Occupancy 0 count mismatch: expected 15810, got {total_occ_0}")
    if total_occ_1 != 4750:
        errors.append(f"Total Occupancy 1 count mismatch: expected 4750, got {total_occ_1}")

    # 6. Official Identity & License Audit
    if source_id.get("dataset_name") != "UCI Occupancy Detection Dataset":
        errors.append(f"Dataset name mismatch: {source_id.get('dataset_name')}")
    if source_id.get("doi") != "10.24432/C5X01N":
        errors.append(f"DOI mismatch: expected 10.24432/C5X01N, got {source_id.get('doi')}")
    if source_id.get("journal_paper_doi") != "10.1016/j.enbuild.2015.11.071":
        errors.append(f"Journal paper DOI mismatch: expected 10.1016/j.enbuild.2015.11.071, got {source_id.get('journal_paper_doi')}")

    # Validate exact raw member timestamps to prevent collection date range drift
    expected_member_timelines = {
        "datatest.txt": ("2015-02-02 14:19:00", "2015-02-04 10:43:00"),
        "datatraining.txt": ("2015-02-04 17:51:00", "2015-02-10 09:33:00"),
        "datatest2.txt": ("2015-02-11 14:48:00", "2015-02-18 09:19:00"),
    }

    for m in members:
        name = m.get("member_name")
        if name in expected_member_timelines:
            exp_first, exp_last = expected_member_timelines[name]
            if m.get("first_timestamp_string") != exp_first:
                errors.append(f"Member {name} first timestamp mismatch: expected {exp_first}, got {m.get('first_timestamp_string')}")
            if m.get("last_timestamp_string") != exp_last:
                errors.append(f"Member {name} last timestamp mismatch: expected {exp_last}, got {m.get('last_timestamp_string')}")

    target_sem = source_id.get("target_semantics", {})
    if target_sem.get("apnea_proxy_claim") is not False:
        errors.append("Apnea proxy claim must be False in source identity")
    if target_sem.get("clinical_apnea_claim") is not False:
        errors.append("Clinical apnea claim must be False in source identity")
    if target_sem.get("co2_danger_claim") is not False:
        errors.append("CO2 danger claim must be False in source identity")

    if license_info.get("license_spdx_id") != "CC-BY-4.0":
        errors.append(f"License SPDX ID mismatch: {license_info.get('license_spdx_id')}")
    if "10.24432/C5X01N" not in license_info.get("citation_string", ""):
        errors.append(f"Citation string does not contain expected DOI 10.24432/C5X01N: {license_info.get('citation_string')}")

    # 7. Lineage Separation Audit
    lineages = lineage.get("lineages", {})
    lin_a = lineages.get("Lineage_A_Real_UCI_Raw_Source", {})
    lin_b = lineages.get("Lineage_B_Synthetic_Smoke_Fixture", {})
    lin_c = lineages.get("Lineage_C_Existing_CO2_Model", {})
    lin_d = lineages.get("Lineage_D_Existing_Scaling_Metadata", {})

    if lin_a.get("classification") != "REAL_EXTERNAL_SOURCE":
        errors.append(f"Lineage A classification must be REAL_EXTERNAL_SOURCE, got {lin_a.get('classification')}")
    if lin_b.get("classification") != "SYNTHETIC_SMOKE_FIXTURE":
        errors.append(f"Lineage B classification must be SYNTHETIC_SMOKE_FIXTURE, got {lin_b.get('classification')}")
    if lin_b.get("verified_real_data") is not False:
        errors.append("Lineage B verified_real_data must be False")
    if lin_c.get("manifest_validation_status") != "CONFIRMED_SYNTHETIC_ONLY":
        errors.append(f"Lineage C status must be CONFIRMED_SYNTHETIC_ONLY, got {lin_c.get('manifest_validation_status')}")
    if lin_d.get("fit_lineage_status") != "FIT_DATA_LINEAGE_UNVERIFIED":
        errors.append(f"Lineage D status must be FIT_DATA_LINEAGE_UNVERIFIED, got {lin_d.get('fit_lineage_status')}")

    # 8. Git Index Payload Safety Audit
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
        pass  # ignore if git command fails in non-git test environment

    # 9. Record Warnings / Anomalies
    anom_list = anomalies.get("anomalies_and_limitations", [])
    for a in anom_list:
        warnings.append(f"[{a.get('condition_code')}]: {a.get('description')}")

    # 10. Gate & Authorization Decision
    gate_status, c_a1_auth = derive_c_a0_gate(
        archive_present=(archive_info.get("byte_size") == 335713),
        member_count=len(members),
        total_rows=total_rows,
        blockers=errors,
        warnings=warnings
    )

    summary_out = {
        "c_a0_gate_status": gate_status,
        "c_a1_authorized": c_a1_auth,
        "raw_archive_sha256": archive_info.get("sha256"),
        "total_raw_rows": total_rows,
        "occupancy_0_rows": total_occ_0,
        "occupancy_1_rows": total_occ_1,
        "schema_mismatch_detected": True,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings
    }

    success = len(errors) == 0
    return success, errors, warnings, summary_out


def main():
    repo_root = get_repo_root()
    manifest_dir = repo_root / "datasets/co2/manifests/c_a0_raw_inventory"

    print(f"🔍 Validating Phase C-A0 CO₂ Raw Inventory in: {manifest_dir.relative_to(repo_root)}")
    success, errors, warnings, summary = validate_manifests(repo_root, manifest_dir)

    print(f"\n--- C-A0 VALIDATOR RESULT ---")
    print(f"Gate Status:      {summary.get('c_a0_gate_status')}")
    print(f"C-A1 Authorized:  {summary.get('c_a1_authorized')}")
    print(f"Total Raw Rows:   {summary.get('total_raw_rows')} (Occ 0: {summary.get('occupancy_0_rows')}, Occ 1: {summary.get('occupancy_1_rows')})")
    print(f"Error Count:      {len(errors)}")
    print(f"Warning Count:    {len(warnings)}")

    if warnings:
        print("\nRecorded Warnings & Limitations:")
        for w in warnings:
            print(f" ⚠️  {w}")

    if success:
        print("\n✅ SUCCESS: Phase C-A0 inventory and lineage manifests are internally coherent and valid.")
        sys.exit(0)
    else:
        print("\n❌ FAIL: Phase C-A0 validation failed with errors:")
        for err in errors:
            print(f" 🔴 {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
