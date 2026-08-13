#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/audit_co2_target_semantics.py
Phase C-A4 — generate deterministic occupancy target semantic evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

repo_root_dir = Path(__file__).resolve().parent.parent
if str(repo_root_dir) not in sys.path:
    sys.path.insert(0, str(repo_root_dir))

from datasets.co2.raw_reader import UCIOccupancyRawReader, compute_sha256_file, get_repo_root
from datasets.co2.target_semantics import (
    EXPECTED_OCC_0,
    EXPECTED_OCC_1,
    EXPECTED_TOTAL_ROWS,
    TARGET_PROFILE_ID,
    audit_label_transitions,
    build_feature_target_role_registry,
    build_occupancy_safety_separation_contract,
    build_occupancy_target_profile,
    reconstruct_all_occupancy_targets,
    summarize_target_integrity,
)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_exceptions_registry() -> Dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "phase": "C-A4",
        "warnings": [
            {
                "code": "SOURCE_TIMEZONE_UNVERIFIED",
                "severity": "WARNING",
                "description": (
                    "Source timestamps remain timezone-naive SOURCE_ACQUISITION_CLOCK readings."
                ),
            },
            {
                "code": "GROUP_INDEPENDENCE_NOT_VERIFIABLE",
                "severity": "WARNING",
                "description": (
                    "All temporal blocks originate from a single office room."
                ),
            },
            {
                "code": "MODEL_TRAINING_LINEAGE_UNVERIFIED",
                "severity": "WARNING",
                "description": (
                    "Existing TFLite training provenance remains unverified; C-A4 does not "
                    "promote model status."
                ),
            },
            {
                "code": "SCALER_FIT_LINEAGE_UNVERIFIED",
                "severity": "WARNING",
                "description": (
                    "Existing scaling metadata fit lineage remains unverified; C-A4 does not "
                    "fit or overwrite scaler statistics."
                ),
            },
            {
                "code": "CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED",
                "severity": "WARNING",
                "description": (
                    "Inherited from C-A3: historical training slope-history duration remains "
                    "unverified; offline 150s baseline is CANONICAL_OFFLINE_BASELINE_DESIGN."
                ),
            },
            {
                "code": "DEVICE_UCI_CADENCE_DOMAIN_GAP",
                "severity": "WARNING",
                "description": (
                    "Inherited domain gap between SCD40 runtime cadence and UCI ~60s source "
                    "cadence remains out of C-A4 scope."
                ),
            },
            {
                "code": "SAFETY_RULE_CONTRACT_OUT_OF_SCOPE",
                "severity": "WARNING",
                "description": (
                    "Active risk/adapter CO2>1500 safety scoring is documented for separation "
                    "only; DEFERRED_SAFETY_RULE_CONTRACT."
                ),
            },
            {
                "code": "SENSOR_HEALTH_CONTRACT_OUT_OF_SCOPE",
                "severity": "WARNING",
                "description": "SCD40/sensor-health contracts are out of C-A4 scope.",
            },
            {
                "code": "MULTISENSOR_RISK_CONTRACT_OUT_OF_SCOPE",
                "severity": "WARNING",
                "description": "Multisensor SafeNest risk fusion is out of C-A4 scope.",
            },
            {
                "code": "DEFERRED_SHARED_INTEGRATION_UPDATE",
                "severity": "WARNING",
                "description": (
                    "Shared SENSOR_DATA_CONTRACT / model_manifest / MANIFEST updates deferred "
                    "to a later approved integration commit."
                ),
            },
        ],
        "blockers": [],
    }


def audit_co2_target_semantics() -> Path:
    repo_root = get_repo_root()
    out_dir = repo_root / "datasets/co2/manifests/c_a4_target_semantics"
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = UCIOccupancyRawReader(repo_root=repo_root)
    observations = reader.read_all_observations()
    if len(observations) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL_ROWS} rows, got {len(observations)}")

    targets = reconstruct_all_occupancy_targets(observations)
    integrity = summarize_target_integrity(observations, targets)
    if not integrity["counts_match_predecessor_expectation"]:
        raise RuntimeError(f"Target integrity mismatch: {integrity}")
    if integrity["occupancy_0_count"] != EXPECTED_OCC_0:
        raise RuntimeError("Occupancy 0 count mismatch")
    if integrity["occupancy_1_count"] != EXPECTED_OCC_1:
        raise RuntimeError("Occupancy 1 count mismatch")

    profile = build_occupancy_target_profile()
    roles = build_feature_target_role_registry()
    separation = build_occupancy_safety_separation_contract()
    transitions = audit_label_transitions(observations)
    exceptions = build_exceptions_registry()
    generation = {
        "manifest_version": "1.0",
        "phase": "C-A4",
        "target_profile_id": TARGET_PROFILE_ID,
        "generator_script": "scripts/audit_co2_target_semantics.py",
        "semantics_module": "datasets/co2/target_semantics.py",
        "total_source_rows": len(observations),
        "total_target_records": len(targets),
        "scaler_fitted": False,
        "model_trained": False,
        "class_balancing_performed": False,
        "threshold_based_relabeling_performed": False,
        "synthetic_npz_used_as_real_label_source": False,
        "locked_test_used_for_contract_selection": False,
        "co2_adapter_modified": False,
        "safety_thresholds_modified": False,
        "determinism": {
            "host_timezone_independent": True,
            "locale_independent": True,
            "random_values_used": False,
        },
    }

    _write_json(out_dir / "occupancy_target_profile.json", profile)
    _write_json(out_dir / "feature_target_role_registry.json", roles)
    _write_json(out_dir / "occupancy_safety_separation_contract.json", separation)
    _write_json(out_dir / "target_integrity_summary.json", integrity)
    _write_json(out_dir / "label_transition_audit.json", transitions)
    _write_json(out_dir / "exceptions_and_limitations.json", exceptions)
    _write_json(out_dir / "generation_metadata.json", generation)

    checksum_files = [
        "occupancy_target_profile.json",
        "feature_target_role_registry.json",
        "occupancy_safety_separation_contract.json",
        "target_integrity_summary.json",
        "label_transition_audit.json",
        "exceptions_and_limitations.json",
        "generation_metadata.json",
    ]
    lines = []
    for fname in checksum_files:
        rel = f"datasets/co2/manifests/c_a4_target_semantics/{fname}"
        lines.append(f"{compute_sha256_file(out_dir / fname)}  {rel}")
    (out_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"✅ Generated C-A4 target semantics manifests in: {out_dir.relative_to(repo_root)}")
    print(
        f"   rows={integrity['total_source_rows']} "
        f"occ0={integrity['occupancy_0_count']} "
        f"occ1={integrity['occupancy_1_count']}"
    )
    return out_dir


if __name__ == "__main__":
    audit_co2_target_semantics()
