#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/validate_co2_canonical_samples.py
Phase C-A5 — canonical sample provenance / split materialization validator.
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
    verify_predecessor_fingerprints,
)
from datasets.co2.raw_reader import UCIOccupancyRawReader, compute_sha256_file, get_repo_root
from datasets.co2.slope_feature import FEATURE_PROFILE_ID as SLOPE_PROFILE_ID
from datasets.co2.slope_feature import STATUS_AVAILABLE, STATUS_WARMUP
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


def derive_gate(
    predecessors_valid: bool,
    total: int,
    errors: int,
    warnings: int,
    one_to_one: bool,
) -> Tuple[str, str]:
    if not predecessors_valid or errors > 0 or total != EXPECTED_TOTAL_SAMPLES or not one_to_one:
        return "FAIL", "NO"
    if warnings > 0:
        return "PASS_WITH_WARNINGS", "YES"
    return "PASS", "YES"


def validate_c_a5(repo_root: Path) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    errors: List[str] = []
    warnings: List[str] = []
    c_a5 = repo_root / "datasets/co2/manifests/c_a5_canonical_samples"
    required = [
        "canonical_sample_profile.json",
        "predecessor_fingerprint_registry.json",
        "split_membership_manifest.json",
        "feature_availability_manifest.json",
        "materialization_integrity_summary.json",
        "exceptions_and_limitations.json",
        "generation_metadata.json",
        "artifact_identity.json",
        "canonical_source_samples.jsonl",
        "model_eligible_sample_ids.jsonl",
        "checksums.sha256",
    ]
    for fname in required:
        if not (c_a5 / fname).exists():
            errors.append(f"Missing C-A5 artifact: {fname}")
    if errors:
        return False, errors, warnings, {}

    preds = {
        "c_a0": _run_validator("scripts/validate_co2_raw_inventory.py", repo_root),
        "c_a1": _run_validator("scripts/validate_co2_safe_reader.py", repo_root),
        "c_a2": _run_validator("scripts/validate_co2_temporal_blocks.py", repo_root),
        "c_a3": _run_validator("scripts/validate_co2_slope_feature.py", repo_root),
        "c_a4": _run_validator("scripts/validate_co2_target_semantics.py", repo_root),
    }
    if not all(preds.values()):
        for k, ok in preds.items():
            if not ok:
                errors.append(f"Predecessor validator failed: {k}")

    profile = _load_json(c_a5 / "canonical_sample_profile.json")
    fingerprints = _load_json(c_a5 / "predecessor_fingerprint_registry.json")
    split = _load_json(c_a5 / "split_membership_manifest.json")
    availability = _load_json(c_a5 / "feature_availability_manifest.json")
    integrity = _load_json(c_a5 / "materialization_integrity_summary.json")
    generation = _load_json(c_a5 / "generation_metadata.json")
    exceptions = _load_json(c_a5 / "exceptions_and_limitations.json")

    if profile.get("profile_id") != CANONICAL_SAMPLE_PROFILE_ID:
        errors.append("Unexpected canonical sample profile_id")
    if profile.get("inherited_profiles", {}).get("slope_feature_profile_id") != SLOPE_PROFILE_ID:
        errors.append("Inherited slope profile mismatch")
    if profile.get("inherited_profiles", {}).get("occupancy_target_profile_id") != TARGET_PROFILE_ID:
        errors.append("Inherited target profile mismatch")
    if profile.get("a_series_release_status") != "DEFERRED_UNTIL_C-A6":
        errors.append("A-series release must be deferred until C-A6")
    access = profile.get("access_semantics", {})
    if access.get("scaler_fit_authorized_roles") != ["TRAIN"]:
        errors.append("Scaler-fit must be TRAIN only")
    if access.get("locked_test_authorized_for_fitting") is not False:
        errors.append("LOCKED_TEST fitting must be unauthorized")
    if access.get("locked_test_authorized_for_tuning") is not False:
        errors.append("LOCKED_TEST tuning must be unauthorized")
    if split.get("random_row_wise_split") is not False:
        errors.append("Random row-wise split must be absent")

    fp_errors = verify_predecessor_fingerprints(fingerprints, repo_root)
    errors.extend(fp_errors)

    if integrity.get("canonical_source_sample_count") != EXPECTED_TOTAL_SAMPLES:
        errors.append("Canonical sample count must be 20560")
    if integrity.get("missing_source_mappings") != 0:
        errors.append("Missing source mappings")
    if integrity.get("duplicate_canonical_ids") != 0:
        errors.append("Duplicate canonical IDs")
    if not integrity.get("one_to_one_ok"):
        errors.append("One-to-one integrity failed")
    if availability.get("co2_slope_eligible") != EXPECTED_SLOPE_ELIGIBLE:
        errors.append("Slope-eligible count mismatch")
    if availability.get("co2_slope_unavailable") != EXPECTED_WARMUP:
        errors.append("Warm-up unavailable count mismatch")

    expected_role = {
        "TRAIN": (8143, 8140, 3, 6414, 1729),
        "VALIDATION": (2665, 2662, 3, 1693, 972),
        "LOCKED_TEST": (9752, 9749, 3, 7703, 2049),
    }
    for role, (n, elig, warm, v0, v1) in expected_role.items():
        row = split["by_role"][role]
        if (
            row["canonical_source_samples"] != n
            or row["slope_eligible_samples"] != elig
            or row["warmup_unavailable_samples"] != warm
            or row["vacant_count"] != v0
            or row["occupied_count"] != v1
        ):
            errors.append(f"Split membership mismatch for {role}")

    if generation.get("scaler_fitted") is not False:
        errors.append("Scaler must not be fitted")
    if generation.get("model_trained") is not False:
        errors.append("Model must not be trained")
    if generation.get("synthetic_npz_used_as_real_source") is not False:
        errors.append("Synthetic NPZ must not be real source")

    # Live reconstruction cross-check against JSONL
    observations = UCIOccupancyRawReader(repo_root=repo_root).read_all_observations()
    live = materialize_canonical_samples(observations)
    jsonl_path = c_a5 / "canonical_source_samples.jsonl"
    stored = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(stored) != EXPECTED_TOTAL_SAMPLES:
        errors.append(f"JSONL row count {len(stored)} != 20560")
    if len(live) != len(stored):
        errors.append("Live vs stored sample count mismatch")
    else:
        for live_s, stored_s in zip(live, stored):
            if live_s.canonical_sample_id != stored_s["canonical_sample_id"]:
                errors.append("Canonical sample ID order/value mismatch")
                break
            if live_s.occupancy_source_value != stored_s["occupancy_source_value"]:
                errors.append("Target label drift vs C-A4")
                break
            if live_s.future_split_role != stored_s["future_split_role"]:
                errors.append("Split role drift vs C-A2")
                break
            if live_s.co2_slope_status != stored_s["co2_slope_status"]:
                errors.append("Slope status drift vs C-A3")
                break
            if live_s.co2_slope_status == STATUS_AVAILABLE:
                if live_s.co2_slope != stored_s["co2_slope"]:
                    errors.append("Slope value drift vs C-A3")
                    break
            elif stored_s["co2_slope"] is not None:
                errors.append("Unavailable slope must be null")
                break
            if live_s.co2_slope_status == STATUS_WARMUP and stored_s[
                "model_eligibility_exclusion_reason"
            ] != STATUS_WARMUP:
                errors.append("Warm-up exclusion reason missing")
                break

    eligible_stored = [
        json.loads(line)
        for line in (c_a5 / "model_eligible_sample_ids.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(eligible_stored) != EXPECTED_SLOPE_ELIGIBLE:
        errors.append("Model-eligible ID count mismatch")

    # Checksums + path portability
    for line in (c_a5 / "checksums.sha256").read_text(encoding="utf-8").strip().splitlines():
        digest, rel = line.split("  ", 1)
        path = repo_root / rel
        if not path.exists():
            errors.append(f"Checksum path missing: {rel}")
        elif compute_sha256_file(path) != digest:
            errors.append(f"Checksum mismatch: {rel}")
    for fname in required:
        text = (c_a5 / fname).read_text(encoding="utf-8")
        for marker in FORBIDDEN_PATH_MARKERS:
            if marker in text:
                errors.append(f"Forbidden path marker {marker} in {fname}")

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
            if path.startswith("datasets/mmwave/") or "mmwave" in lower and path.startswith(
                ("scripts/", "tests/", "docs/reports/", "models/")
            ):
                errors.append(f"mmWave file in C-A5 branch: {path}")
            if path.startswith("datasets/thermal") or (
                "thermal" in lower
                and path.startswith(("scripts/", "tests/", "docs/reports/", "datasets/"))
            ):
                errors.append(f"Thermal file in C-A5 branch: {path}")
            if "occupancy+detection.zip" in path:
                errors.append("Raw payload staged/modified")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Git isolation check skipped: {exc}")

    for item in exceptions.get("warnings", []):
        warnings.append(f"[{item.get('code')}] {item.get('description')}")
    if exceptions.get("blockers"):
        errors.append("Exception registry contains blockers")

    gate, auth = derive_gate(
        predecessors_valid=all(preds.values()),
        total=int(integrity.get("canonical_source_sample_count", 0)),
        errors=len(errors),
        warnings=len(warnings),
        one_to_one=bool(integrity.get("one_to_one_ok")),
    )
    summary = {
        "gate_status": gate,
        "c_a6_authorized": auth,
        "canonical_source_samples": integrity.get("canonical_source_sample_count"),
        "slope_eligible": availability.get("co2_slope_eligible"),
        "warmup_unavailable": availability.get("co2_slope_unavailable"),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "predecessors": preds,
    }
    return len(errors) == 0, errors, warnings, summary


def main() -> int:
    repo_root = get_repo_root()
    print(
        "🔍 Validating Phase C-A5 CO₂ Canonical Samples in: "
        "datasets/co2/manifests/c_a5_canonical_samples"
    )
    ok, errors, warnings, summary = validate_c_a5(repo_root)
    print("\n--- C-A5 VALIDATOR RESULT ---")
    print(f"Gate Status:      {summary.get('gate_status', 'FAIL')}")
    print(f"C-A6 Authorized:  {summary.get('c_a6_authorized', 'NO')}")
    print(f"Canonical Samples:{summary.get('canonical_source_samples')}")
    print(f"Slope Eligible:   {summary.get('slope_eligible')}")
    print(f"Warm-up Rows:     {summary.get('warmup_unavailable')}")
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
        print("\n✅ SUCCESS: Phase C-A5 canonical sample contract is valid.")
        return 0
    print("\n❌ FAILURE: Phase C-A5 validation failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
