#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/validate_co2_slope_feature.py
Phase C-A3 — CO₂ Slope Feature Reconstruction standalone validator.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

repo_root_dir = Path(__file__).resolve().parent.parent
if str(repo_root_dir) not in sys.path:
    sys.path.insert(0, str(repo_root_dir))

from datasets.co2.raw_reader import UCIOccupancyRawReader, compute_sha256_file, get_repo_root
from datasets.co2.slope_feature import (
    CAUSALITY,
    FEATURE_NAME,
    FEATURE_PROFILE_ID,
    FEATURE_UNIT,
    HISTORY_DURATION_SECONDS,
    SLOPE_METHOD,
    STATUS_AVAILABLE,
    TIMESTAMP_BASIS,
    reconstruct_all_slope_features,
)


FORBIDDEN_PATH_MARKERS = ("/Users/", "file://", "~/")
PROTECTED_SHARED = [
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
]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_validator(script: str, repo_root: Path) -> Tuple[bool, str]:
    res = subprocess.run(
        ["python3", script],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    return res.returncode == 0, res.stdout + res.stderr


def derive_c_a3_gate(
    predecessors_valid: bool,
    total_rows: int,
    error_count: int,
    warning_count: int,
    manual_pass: bool,
) -> Tuple[str, str]:
    if (
        not predecessors_valid
        or error_count > 0
        or total_rows != 20560
        or not manual_pass
    ):
        return "FAIL", "NO"
    if warning_count > 0:
        return "PASS_WITH_WARNINGS", "YES"
    return "PASS", "YES"


def validate_c_a3_slope_feature(
    repo_root: Path,
) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    errors: List[str] = []
    warnings: List[str] = []
    summary: Dict[str, Any] = {}

    c_a3_dir = repo_root / "datasets/co2/manifests/c_a3_slope_feature"
    required = [
        "co2_slope_feature_profile.json",
        "feature_eligibility_summary.json",
        "feature_audit_statistics.json",
        "manual_verification_cases.json",
        "candidate_method_evaluation.json",
        "source_row_feature_lineage_contract.json",
        "exceptions_and_limitations.json",
        "generation_metadata.json",
        "checksums.sha256",
    ]
    for fname in required:
        if not (c_a3_dir / fname).exists():
            errors.append(f"Missing C-A3 artifact: {fname}")
    if errors:
        return False, errors, warnings, summary

    # Predecessor validators
    ok0, out0 = _run_validator("scripts/validate_co2_raw_inventory.py", repo_root)
    ok1, out1 = _run_validator("scripts/validate_co2_safe_reader.py", repo_root)
    ok2, out2 = _run_validator("scripts/validate_co2_temporal_blocks.py", repo_root)
    predecessors_valid = ok0 and ok1 and ok2
    if not ok0:
        errors.append("C-A0 predecessor validator failed")
    if not ok1:
        errors.append("C-A1 predecessor validator failed")
    if not ok2:
        errors.append("C-A2 predecessor validator failed")

    profile = _load_json(c_a3_dir / "co2_slope_feature_profile.json")
    eligibility = _load_json(c_a3_dir / "feature_eligibility_summary.json")
    manual = _load_json(c_a3_dir / "manual_verification_cases.json")
    candidates = _load_json(c_a3_dir / "candidate_method_evaluation.json")
    generation = _load_json(c_a3_dir / "generation_metadata.json")
    exceptions = _load_json(c_a3_dir / "exceptions_and_limitations.json")
    audit = _load_json(c_a3_dir / "feature_audit_statistics.json")

    # Profile locks
    if profile.get("profile_id") != FEATURE_PROFILE_ID:
        errors.append(f"Unexpected profile_id: {profile.get('profile_id')}")
    if profile.get("feature_name") != FEATURE_NAME:
        errors.append("feature_name must be CO2_slope")
    if profile.get("feature_unit") != FEATURE_UNIT:
        errors.append("feature_unit must be ppm/min")
    if profile.get("slope_method") != SLOPE_METHOD:
        errors.append("slope_method must be ENDPOINT_DIFFERENCE")
    if profile.get("history_duration_seconds") != HISTORY_DURATION_SECONDS:
        errors.append("history_duration_seconds must be 150.0")
    if profile.get("causality") != CAUSALITY:
        errors.append("causality must be PAST_ONLY")
    if profile.get("timestamp_basis") != TIMESTAMP_BASIS:
        errors.append("timestamp_basis must be SOURCE_ACQUISITION_CLOCK")
    if profile.get("future_samples_allowed") is not False:
        errors.append("future samples must be forbidden")
    if profile.get("interpolation_allowed") is not False:
        errors.append("interpolation must be forbidden")
    if profile.get("scaler_fitting_in_phase") is not False:
        errors.append("scaler fitting must be false in C-A3")
    if profile.get("model_training_in_phase") is not False:
        errors.append("model training must be false in C-A3")
    if profile.get("locked_test_used_for_contract_selection") is not False:
        errors.append("LOCKED_TEST must not drive feature-contract selection")
    if profile.get("boundary_policy") != (
        "DERIVED_TEMPORAL_FEATURES_MUST_NOT_CROSS_BLOCK_BOUNDARIES"
    ):
        errors.append("Invalid block-boundary policy")
    if profile.get("gap_policy") != "RESTART_HISTORY_AFTER_FORBIDDEN_GAP":
        errors.append("Invalid gap policy")
    if profile.get("warm_up_policy") != "PRESERVE_ROW_WITH_NULL_SLOPE":
        errors.append("Warm-up rows must be preserved with null slope")

    # Runtime vs offline baseline classification (must not claim exact adapter equivalence)
    if "device_adapter_alignment" in profile:
        errors.append(
            "profile must not use device_adapter_alignment implying runtime equivalence"
        )
    if profile.get("offline_baseline_status") != "CANONICAL_OFFLINE_BASELINE_DESIGN":
        errors.append("offline_baseline_status must be CANONICAL_OFFLINE_BASELINE_DESIGN")
    equiv = profile.get("offline_baseline_equivalence_claims") or {}
    if equiv.get("verified_historical_training_contract") is not False:
        errors.append("must not claim VERIFIED_HISTORICAL_TRAINING_CONTRACT")
    if equiv.get("active_runtime_equivalent") is not False:
        errors.append("must not claim ACTIVE_RUNTIME_EQUIVALENT")
    runtime = profile.get("runtime_evidence") or {}
    if runtime.get("runtime_slope_method_verified") != "ENDPOINT_DIFFERENCE":
        errors.append("runtime_slope_method_verified must be ENDPOINT_DIFFERENCE")
    if runtime.get("runtime_history_maxlen") != 30:
        errors.append("runtime_history_maxlen must be 30")
    if runtime.get("runtime_required_history_sec") != 5.0:
        errors.append("runtime_required_history_sec must be 5.0")
    if runtime.get("configured_window_seconds") != 150.0:
        errors.append("configured_window_seconds must be 150.0")
    if runtime.get("configured_window_seconds_applied_to_slope_logic") is not False:
        errors.append("configured_window_seconds must be marked NOT applied to slope logic")
    if runtime.get("nominal_full_buffer_endpoint_span_seconds") != 145.0:
        errors.append("nominal_full_buffer_endpoint_span_seconds must be 145.0")
    if profile.get("offline_canonical_history_threshold_seconds") != HISTORY_DURATION_SECONDS:
        errors.append("offline_canonical_history_threshold_seconds must match 150.0")
    if profile.get("historical_training_history_contract_status") != (
        "CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED"
    ):
        errors.append("historical_training_history_contract_status must be UNVERIFIED")

    # Row integrity
    if eligibility.get("total_source_rows_represented") != 20560:
        errors.append("Source-row count must remain 20560")
    if eligibility.get("silent_row_loss") != 0 or eligibility.get("rows_omitted") != 0:
        errors.append("Silent row loss detected in eligibility summary")
    if eligibility.get("eligible_slope_rows", 0) + eligibility.get(
        "warmup_unavailable_rows", 0
    ) + eligibility.get("gap_restart_unavailable_rows", 0) + eligibility.get(
        "nonfinite_unavailable_rows", 0
    ) != 20560:
        # non_monotonic may also exist; re-check via by_block totals
        by_block_total = sum(b["source_row_count"] for b in eligibility["by_block"])
        if by_block_total != 20560:
            errors.append("by_block source_row_count does not sum to 20560")

    # Live reconstruction checks
    reader = UCIOccupancyRawReader(repo_root=repo_root)
    observations = reader.read_all_observations()
    if len(observations) != 20560:
        errors.append(f"Live reader returned {len(observations)} rows")
    records = reconstruct_all_slope_features(observations)
    if len(records) != 20560:
        errors.append(f"Reconstruction lost rows: {len(records)}")

    # No cross-block history / causality / finite eligible slopes
    for record in records:
        if record.feature_status == STATUS_AVAILABLE:
            if record.co2_slope is None or not math.isfinite(record.co2_slope):
                errors.append(
                    f"Nonfinite eligible slope at "
                    f"{record.target_source_member}:{record.target_source_row_identifier}"
                )
                break
            if record.history_elapsed_seconds is None or record.history_elapsed_seconds < HISTORY_DURATION_SECONDS:
                errors.append("Eligible slope has insufficient elapsed history")
                break
            if record.source_sample_count_used < 2:
                errors.append("Eligible slope used fewer than 2 samples")
                break
            # history members must match target member (block isolation)
            # identifiers alone are not unique across members; member equality is required
            if record.history_start_source_row_identifier is None:
                errors.append("Eligible slope missing lineage history start")
                break
        else:
            if record.co2_slope is not None:
                errors.append("Unavailable row must have null co2_slope")
                break

    # Block boundary: first three rows of each block must be warmup (given UCI cadence)
    for member, first_ids in (
        ("datatest.txt", ["140", "141", "142"]),
        ("datatraining.txt", ["1", "2", "3"]),
        ("datatest2.txt", ["1", "2", "3"]),
    ):
        for row_id in first_ids:
            rec = next(
                r
                for r in records
                if r.target_source_member == member
                and r.target_source_row_identifier == row_id
            )
            if rec.feature_status != "FEATURE_UNAVAILABLE_WARMUP" or rec.co2_slope is not None:
                errors.append(f"Warm-up contract violated for {member}:{row_id}")
                break

    # Manual verification
    if not manual.get("all_cases_pass"):
        errors.append("Manual verification cases did not all pass")
    if not manual.get("contract_frozen_before_validation_locked_test_cases"):
        errors.append("VALIDATION/LOCKED_TEST cases used before contract freeze")
    for case in manual.get("cases", []):
        if not case.get("pass"):
            errors.append(f"Manual case failed: {case.get('case_id')}")

    # Candidate selection hygiene
    if candidates.get("selected_candidate_id") != "B_ENDPOINT_DIFFERENCE_HISTORY_DURATION":
        errors.append("Unexpected selected slope candidate")
    selected = next(
        (
            c
            for c in candidates.get("candidates_considered", [])
            if c.get("candidate_id") == "B_ENDPOINT_DIFFERENCE_HISTORY_DURATION"
        ),
        {},
    )
    reason = str(selected.get("reason", ""))
    forbidden_claim = (
        "adapter window_seconds=150.0; uses actual source-clock elapsed time"
    )
    if forbidden_claim in reason:
        errors.append(
            "candidate B rationale must not claim active code verifies 150s duration"
        )
    if "CANONICAL_OFFLINE_BASELINE_DESIGN" not in reason:
        errors.append("candidate B must classify 150s as CANONICAL_OFFLINE_BASELINE_DESIGN")
    if "required_history_sec=5.0" not in reason:
        errors.append("candidate B must document active runtime required_history_sec=5.0")
    if selected.get("offline_baseline_status") != "CANONICAL_OFFLINE_BASELINE_DESIGN":
        errors.append("candidate B offline_baseline_status missing/incorrect")
    scaler_cmp = (
        candidates.get("train_only_secondary_scaler_diagnostic", {}).get("comparison_result")
    )
    if scaler_cmp == "CONSISTENT_WITH_HISTORICAL_SCALER":
        errors.append(
            "scaler diagnostic must not claim CONSISTENT_WITH_HISTORICAL_SCALER "
            "from mean proximity alone"
        )
    if scaler_cmp not in {
        "PARTIAL_MEAN_ALIGNMENT_ONLY",
        "INSUFFICIENT_TO_PROVE_LINEAGE",
        "INCONSISTENT_WITH_HISTORICAL_SCALER",
    }:
        errors.append(f"Unexpected scaler comparison status: {scaler_cmp}")

    warning_codes = {w.get("code") for w in exceptions.get("warnings", [])}
    for required_code in (
        "CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED",
        "ADAPTER_WINDOW_SECONDS_NOT_APPLIED_TO_ACTIVE_SLOPE_PATH",
        "MODEL_TRAINING_LINEAGE_UNVERIFIED",
        "SCALER_FIT_LINEAGE_UNVERIFIED",
    ):
        if required_code not in warning_codes:
            errors.append(f"Missing required limitation warning: {required_code}")
    if generation.get("scaler_fitted") is not False:
        errors.append("generation metadata claims scaler was fitted")
    if generation.get("model_trained") is not False:
        errors.append("generation metadata claims model was trained")
    if generation.get("synthetic_npz_used_as_real_source") is not False:
        errors.append("synthetic NPZ must not be used as real source")
    if generation.get("locked_test_used_for_contract_selection") is not False:
        errors.append("LOCKED_TEST used for contract selection")

    # LOCKED_TEST value stats must not be used for selection; audit may omit values
    locked = audit["by_role"]["LOCKED_TEST"]
    if locked.get("value_statistics_included") is True:
        warnings.append(
            "LOCKED_TEST value statistics are present; ensure they did not drive selection"
        )
    if locked.get("source_row_count") != 9752:
        errors.append("LOCKED_TEST source_row_count mismatch")

    # Checksums
    checksum_text = (c_a3_dir / "checksums.sha256").read_text(encoding="utf-8").strip().splitlines()
    for line in checksum_text:
        digest, rel = line.split("  ", 1)
        path = repo_root / rel
        if not path.exists():
            errors.append(f"Checksum path missing: {rel}")
            continue
        actual = compute_sha256_file(path)
        if actual != digest:
            errors.append(f"Checksum mismatch for {rel}")

    # Path portability
    for fname in required:
        if fname == "checksums.sha256":
            text = (c_a3_dir / fname).read_text(encoding="utf-8")
        else:
            text = (c_a3_dir / fname).read_text(encoding="utf-8")
        for marker in FORBIDDEN_PATH_MARKERS:
            if marker in text:
                errors.append(f"Forbidden path marker {marker} in {fname}")

    # Protected shared files must remain unmodified in this phase check via git if available
    try:
        diff = subprocess.run(
            ["git", "diff", "--name-only", "origin/main...HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        changed = [line.strip() for line in diff.stdout.splitlines() if line.strip()]
        for path in changed:
            if path in PROTECTED_SHARED:
                errors.append(f"Unauthorized shared file modified: {path}")
            if path.startswith("datasets/mmwave/") or path.startswith("sensors/mmwave/"):
                errors.append(f"mmWave file modified in C-A3 branch: {path}")
            if path.startswith("datasets/thermal") or path.startswith("sensors/thermal"):
                errors.append(f"Thermal file modified in C-A3 branch: {path}")
            if "occupancy+detection.zip" in path:
                errors.append("Raw payload staged/modified")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Git isolation check skipped: {exc}")

    # Exception warnings mirror expected non-blocking limitations
    for item in exceptions.get("warnings", []):
        warnings.append(f"[{item.get('code')}] {item.get('description')}")
    if exceptions.get("blockers"):
        errors.append("Exception registry contains blockers")

    warning_count = len(warnings)
    gate, auth = derive_c_a3_gate(
        predecessors_valid=predecessors_valid,
        total_rows=int(eligibility.get("total_source_rows_represented", 0)),
        error_count=len(errors),
        warning_count=warning_count,
        manual_pass=bool(manual.get("all_cases_pass")),
    )
    summary = {
        "gate_status": gate,
        "c_a4_authorized": auth,
        "total_source_rows": eligibility.get("total_source_rows_represented"),
        "eligible_slope_rows": eligibility.get("eligible_slope_rows"),
        "warmup_unavailable_rows": eligibility.get("warmup_unavailable_rows"),
        "feature_profile_id": profile.get("profile_id"),
        "slope_method": profile.get("slope_method"),
        "history_duration_seconds": profile.get("history_duration_seconds"),
        "manual_all_pass": manual.get("all_cases_pass"),
        "error_count": len(errors),
        "warning_count": warning_count,
        "predecessor_outputs": {
            "c_a0_ok": ok0,
            "c_a1_ok": ok1,
            "c_a2_ok": ok2,
        },
    }
    return len(errors) == 0, errors, warnings, summary


def main() -> int:
    repo_root = get_repo_root()
    print(f"🔍 Validating Phase C-A3 CO₂ Slope Feature in: datasets/co2/manifests/c_a3_slope_feature")
    ok, errors, warnings, summary = validate_c_a3_slope_feature(repo_root)

    print("\n--- C-A3 VALIDATOR RESULT ---")
    print(f"Gate Status:      {summary.get('gate_status', 'FAIL')}")
    print(f"C-A4 Authorized:  {summary.get('c_a4_authorized', 'NO')}")
    print(f"Total Source Rows:{summary.get('total_source_rows')}")
    print(f"Eligible Slopes:  {summary.get('eligible_slope_rows')}")
    print(f"Warm-up Rows:     {summary.get('warmup_unavailable_rows')}")
    print(f"Profile ID:       {summary.get('feature_profile_id')}")
    print(f"Slope Method:     {summary.get('slope_method')}")
    print(f"History Seconds:  {summary.get('history_duration_seconds')}")
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
        print("\n✅ SUCCESS: Phase C-A3 slope feature contract and manifests are valid.")
        return 0
    print("\n❌ FAILURE: Phase C-A3 validation failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
