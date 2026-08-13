#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/validate_co2_target_semantics.py
Phase C-A4 — occupancy target semantics standalone validator.
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

from datasets.co2.raw_reader import UCIOccupancyRawReader, compute_sha256_file, get_repo_root
from datasets.co2.target_semantics import (
    CANONICAL_CLASS_MAPPING,
    EXPECTED_OCC_0,
    EXPECTED_OCC_1,
    EXPECTED_TOTAL_ROWS,
    NEGATIVE_CLASS_NAME,
    POSITIVE_CLASS_NAME,
    SOURCE_ALLOWED_VALUES,
    TARGET_PROFILE_ID,
    reconstruct_all_occupancy_targets,
    summarize_target_integrity,
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
    "sensors/co2/co2_adapter.py",
]


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


def derive_c_a4_gate(
    predecessors_valid: bool,
    total_rows: int,
    error_count: int,
    warning_count: int,
    integrity_ok: bool,
) -> Tuple[str, str]:
    if (
        not predecessors_valid
        or error_count > 0
        or total_rows != EXPECTED_TOTAL_ROWS
        or not integrity_ok
    ):
        return "FAIL", "NO"
    if warning_count > 0:
        return "PASS_WITH_WARNINGS", "YES"
    return "PASS", "YES"


def validate_c_a4_target_semantics(
    repo_root: Path,
) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    errors: List[str] = []
    warnings: List[str] = []
    summary: Dict[str, Any] = {}

    c_a4_dir = repo_root / "datasets/co2/manifests/c_a4_target_semantics"
    required = [
        "occupancy_target_profile.json",
        "feature_target_role_registry.json",
        "occupancy_safety_separation_contract.json",
        "target_integrity_summary.json",
        "label_transition_audit.json",
        "exceptions_and_limitations.json",
        "generation_metadata.json",
        "checksums.sha256",
    ]
    for fname in required:
        if not (c_a4_dir / fname).exists():
            errors.append(f"Missing C-A4 artifact: {fname}")
    if errors:
        return False, errors, warnings, summary

    ok0 = _run_validator("scripts/validate_co2_raw_inventory.py", repo_root)
    ok1 = _run_validator("scripts/validate_co2_safe_reader.py", repo_root)
    ok2 = _run_validator("scripts/validate_co2_temporal_blocks.py", repo_root)
    ok3 = _run_validator("scripts/validate_co2_slope_feature.py", repo_root)
    predecessors_valid = ok0 and ok1 and ok2 and ok3
    if not ok0:
        errors.append("C-A0 predecessor validator failed")
    if not ok1:
        errors.append("C-A1 predecessor validator failed")
    if not ok2:
        errors.append("C-A2 predecessor validator failed")
    if not ok3:
        errors.append("C-A3 predecessor validator failed")

    profile = _load_json(c_a4_dir / "occupancy_target_profile.json")
    roles = _load_json(c_a4_dir / "feature_target_role_registry.json")
    separation = _load_json(c_a4_dir / "occupancy_safety_separation_contract.json")
    integrity = _load_json(c_a4_dir / "target_integrity_summary.json")
    transitions = _load_json(c_a4_dir / "label_transition_audit.json")
    generation = _load_json(c_a4_dir / "generation_metadata.json")
    exceptions = _load_json(c_a4_dir / "exceptions_and_limitations.json")

    if profile.get("target_profile_id") != TARGET_PROFILE_ID:
        errors.append("Unexpected target_profile_id")
    if profile.get("source_field") != "Occupancy":
        errors.append("source_field must be Occupancy")
    if profile.get("source_allowed_values") != list(SOURCE_ALLOWED_VALUES):
        errors.append("source_allowed_values must be [0, 1]")
    if profile.get("canonical_class_mapping") != CANONICAL_CLASS_MAPPING:
        errors.append("canonical_class_mapping mismatch")
    if profile.get("positive_class", {}).get("semantic_name") != POSITIVE_CLASS_NAME:
        errors.append("positive class must be OCCUPIED")
    if profile.get("negative_class", {}).get("semantic_name") != NEGATIVE_CLASS_NAME:
        errors.append("negative class must be VACANT")
    if profile.get("label_derivation") != "NONE":
        errors.append("label_derivation must be NONE")
    if profile.get("threshold_based_relabeling") != "PROHIBITED":
        errors.append("threshold_based_relabeling must be PROHIBITED")
    if profile.get("is_safety_state") is not False:
        errors.append("occupancy target must not be marked safety state")
    if profile.get("is_model_prediction") is not False:
        errors.append("source target must not be marked model prediction")
    if profile.get("occupancy_means_dangerous_co2") is not False:
        errors.append("occupancy must not claim CO2 danger")
    if profile.get("co2_ppm_may_modify_label") is not False:
        errors.append("CO2 ppm must not modify occupancy label")
    if profile.get("co2_slope_may_modify_label") is not False:
        errors.append("CO2_slope must not modify occupancy label")
    if profile.get("locked_test_used_for_contract_selection") is not False:
        errors.append("LOCKED_TEST must not drive contract design")

    role_by_name = {f["field_name"]: f for f in roles.get("fields", [])}
    expected_roles = {
        "Temperature": "MEASURED_FEATURE",
        "Humidity": "MEASURED_FEATURE",
        "Light": "MEASURED_FEATURE",
        "CO2": "MEASURED_FEATURE",
        "HumidityRatio": "MEASURED_FEATURE",
        "CO2_slope": "DERIVED_FEATURE",
        "Occupancy": "SOURCE_TARGET_LABEL",
    }
    for name, role in expected_roles.items():
        if role_by_name.get(name, {}).get("role") != role:
            errors.append(f"Role mismatch for {name}")
        if role_by_name.get(name, {}).get("may_modify_occupancy_label") is not False:
            errors.append(f"{name} must not modify occupancy label")

    concept_ids = {c["concept_id"] for c in separation.get("concepts", [])}
    for required_concept in (
        "UCI_OCCUPANCY_LABEL",
        "MEASURED_CO2_PPM",
        "DERIVED_CO2_SLOPE",
        "FUTURE_OCCUPANCY_PROBABILITY",
        "RULE_BASED_CO2_SAFETY_STATE",
        "SENSOR_HEALTH_STATE",
        "MULTISENSOR_RISK_SCORE",
    ):
        if required_concept not in concept_ids:
            errors.append(f"Missing separation concept: {required_concept}")
    for concept in separation.get("concepts", []):
        if concept["concept_id"] != "UCI_OCCUPANCY_LABEL" and concept.get(
            "allowed_to_modify_occupancy_label"
        ):
            errors.append(
                f"{concept['concept_id']} must not be allowed to modify occupancy label"
            )
        if concept["concept_id"] == "UCI_OCCUPANCY_LABEL" and not concept.get(
            "used_as_model_target"
        ):
            errors.append("UCI occupancy must be the model target concept")
        if concept["concept_id"] == "RULE_BASED_CO2_SAFETY_STATE" and concept.get(
            "used_as_model_target"
        ):
            errors.append("safety state must not be model target")

    if integrity.get("total_source_rows") != EXPECTED_TOTAL_ROWS:
        errors.append("total_source_rows must be 20560")
    if integrity.get("occupancy_0_count") != EXPECTED_OCC_0:
        errors.append("Occupancy 0 count mismatch")
    if integrity.get("occupancy_1_count") != EXPECTED_OCC_1:
        errors.append("Occupancy 1 count mismatch")
    if integrity.get("unexpected_labels") != 0:
        errors.append("unexpected labels present")
    if integrity.get("missing_target_labels") != 0:
        errors.append("missing target labels present")
    if integrity.get("modified_target_labels") != 0:
        errors.append("modified target labels present")
    if integrity.get("derived_reconstructed_labels") != 0:
        errors.append("derived labels present")

    by_role = integrity.get("by_future_split_role", {})
    expected_role_counts = {
        "TRAIN": (8143, 6414, 1729),
        "VALIDATION": (2665, 1693, 972),
        "LOCKED_TEST": (9752, 7703, 2049),
    }
    for role, (n, n0, n1) in expected_role_counts.items():
        row = by_role.get(role, {})
        if (
            row.get("source_row_count") != n
            or row.get("occupancy_0_count") != n0
            or row.get("occupancy_1_count") != n1
        ):
            errors.append(f"Per-role target count mismatch for {role}")

    if transitions.get("labels_modified") is not False:
        errors.append("transition audit must not modify labels")
    if transitions.get("label_smoothing_applied") is not False:
        errors.append("label smoothing prohibited")

    if generation.get("scaler_fitted") is not False:
        errors.append("scaler must not be fitted")
    if generation.get("model_trained") is not False:
        errors.append("model must not be trained")
    if generation.get("synthetic_npz_used_as_real_label_source") is not False:
        errors.append("synthetic NPZ must not be real label source")
    if generation.get("class_balancing_performed") is not False:
        errors.append("class balancing prohibited in C-A4")
    if generation.get("threshold_based_relabeling_performed") is not False:
        errors.append("threshold relabeling prohibited")

    # Live reconstruction integrity
    observations = UCIOccupancyRawReader(repo_root=repo_root).read_all_observations()
    targets = reconstruct_all_occupancy_targets(observations)
    live = summarize_target_integrity(observations, targets)
    if not live["counts_match_predecessor_expectation"]:
        errors.append("Live target integrity check failed")
    obs_by_key = {
        (obs.source_member_name, obs.source_row_identifier): obs for obs in observations
    }
    for tgt in targets:
        obs = obs_by_key.get((tgt.target_source_member, tgt.target_source_row_identifier))
        if obs is None or obs.occupancy != tgt.occupancy_source_value:
            errors.append("Silent label modification detected")
            break
        if tgt.occupancy_semantic_name != CANONICAL_CLASS_MAPPING[str(obs.occupancy)]:
            errors.append("Semantic mapping mismatch")
            break

    # Checksums + path portability
    for line in (c_a4_dir / "checksums.sha256").read_text(encoding="utf-8").strip().splitlines():
        digest, rel = line.split("  ", 1)
        path = repo_root / rel
        if not path.exists():
            errors.append(f"Checksum path missing: {rel}")
        elif compute_sha256_file(path) != digest:
            errors.append(f"Checksum mismatch for {rel}")
    for fname in required:
        text = (c_a4_dir / fname).read_text(encoding="utf-8")
        for marker in FORBIDDEN_PATH_MARKERS:
            if marker in text:
                errors.append(f"Forbidden path marker {marker} in {fname}")

    # Git isolation against protected / other tracks
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
            lower = path.lower()
            if (
                path.startswith("datasets/mmwave/")
                or "/mmwave" in lower
                or "mmwave_m_b" in lower
                or path.startswith("scripts/mmwave")
                or path.startswith("scripts/run_mmwave")
                or path.startswith("tests/test_mmwave")
                or "m-b" in lower and "mmwave" in lower
            ):
                errors.append(f"mmWave file modified in C-A4 branch: {path}")
            if (
                path.startswith("datasets/thermal")
                or "/thermal" in lower
                or path.startswith("scripts/generate_thermal")
                or path.startswith("scripts/validate_thermal")
                or path.startswith("tests/test_thermal")
                or "t-a" in lower and "thermal" in lower
            ):
                errors.append(f"Thermal file modified in C-A4 branch: {path}")
            if "occupancy+detection.zip" in path:
                errors.append("Raw payload staged/modified")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Git isolation check skipped: {exc}")

    for item in exceptions.get("warnings", []):
        warnings.append(f"[{item.get('code')}] {item.get('description')}")
    if exceptions.get("blockers"):
        errors.append("Exception registry contains blockers")

    integrity_ok = bool(integrity.get("counts_match_predecessor_expectation"))
    gate, auth = derive_c_a4_gate(
        predecessors_valid=predecessors_valid,
        total_rows=int(integrity.get("total_source_rows", 0)),
        error_count=len(errors),
        warning_count=len(warnings),
        integrity_ok=integrity_ok,
    )
    summary = {
        "gate_status": gate,
        "c_a5_authorized": auth,
        "total_source_rows": integrity.get("total_source_rows"),
        "occupancy_0_count": integrity.get("occupancy_0_count"),
        "occupancy_1_count": integrity.get("occupancy_1_count"),
        "target_profile_id": profile.get("target_profile_id"),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "predecessor_ok": {
            "c_a0": ok0,
            "c_a1": ok1,
            "c_a2": ok2,
            "c_a3": ok3,
        },
    }
    return len(errors) == 0, errors, warnings, summary


def main() -> int:
    repo_root = get_repo_root()
    print(
        "🔍 Validating Phase C-A4 CO₂ Target Semantics in: "
        "datasets/co2/manifests/c_a4_target_semantics"
    )
    ok, errors, warnings, summary = validate_c_a4_target_semantics(repo_root)
    print("\n--- C-A4 VALIDATOR RESULT ---")
    print(f"Gate Status:      {summary.get('gate_status', 'FAIL')}")
    print(f"C-A5 Authorized:  {summary.get('c_a5_authorized', 'NO')}")
    print(f"Total Source Rows:{summary.get('total_source_rows')}")
    print(f"Occupancy 0:      {summary.get('occupancy_0_count')}")
    print(f"Occupancy 1:      {summary.get('occupancy_1_count')}")
    print(f"Profile ID:       {summary.get('target_profile_id')}")
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
        print("\n✅ SUCCESS: Phase C-A4 occupancy target semantics are valid.")
        return 0
    print("\n❌ FAILURE: Phase C-A4 validation failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
