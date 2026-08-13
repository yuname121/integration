#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/audit_co2_canonical_samples.py
Phase C-A5 — materialize canonical sample provenance and split membership evidence.
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
    build_canonical_sample_profile,
    build_predecessor_fingerprint_registry,
    materialize_canonical_samples,
    summarize_feature_availability,
    summarize_materialization_integrity,
    summarize_split_membership,
    write_jsonl,
)
from datasets.co2.raw_reader import UCIOccupancyRawReader, compute_sha256_file, get_repo_root


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_exceptions_registry() -> Dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "phase": "C-A5",
        "warnings": [
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
                "description": "Existing TFLite training lineage remains unverified; C-A5 does not promote it.",
            },
            {
                "code": "SCALER_FIT_LINEAGE_UNVERIFIED",
                "severity": "WARNING",
                "description": "Existing scaling metadata lineage remains unverified; C-A5 does not fit scalers.",
            },
            {
                "code": "CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED",
                "severity": "WARNING",
                "description": "Inherited C-A3 history-duration lineage limitation remains.",
            },
            {
                "code": "DEVICE_UCI_CADENCE_DOMAIN_GAP",
                "severity": "WARNING",
                "description": "Inherited SCD40 vs UCI cadence domain gap remains out of C-A5 scope.",
            },
            {
                "code": "DEFERRED_SHARED_INTEGRATION_UPDATE",
                "severity": "WARNING",
                "description": "Shared inventory/contract refresh deferred to later integration commit.",
            },
            {
                "code": "A_SERIES_RELEASE_DEFERRED_UNTIL_C_A6",
                "severity": "WARNING",
                "description": "CO₂ A-series release/tag is deferred until C-A6 final integrity lock.",
            },
        ],
        "blockers": [],
    }


def audit_co2_canonical_samples() -> Path:
    repo_root = get_repo_root()
    out_dir = repo_root / "datasets/co2/manifests/c_a5_canonical_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = UCIOccupancyRawReader(repo_root=repo_root)
    observations = reader.read_all_observations()
    if len(observations) != EXPECTED_TOTAL_SAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL_SAMPLES} observations")

    samples = materialize_canonical_samples(observations)
    if len(samples) != EXPECTED_TOTAL_SAMPLES:
        raise RuntimeError("Canonical sample count mismatch")

    integrity = summarize_materialization_integrity(observations, samples)
    if not integrity["one_to_one_ok"]:
        raise RuntimeError(f"One-to-one integrity failed: {integrity}")

    availability = summarize_feature_availability(samples)
    if availability["co2_slope_eligible"] != EXPECTED_SLOPE_ELIGIBLE:
        raise RuntimeError(
            f"Expected {EXPECTED_SLOPE_ELIGIBLE} eligible, got {availability['co2_slope_eligible']}"
        )
    if availability["co2_slope_unavailable"] != EXPECTED_WARMUP:
        raise RuntimeError(
            f"Expected {EXPECTED_WARMUP} unavailable, got {availability['co2_slope_unavailable']}"
        )

    split = summarize_split_membership(samples)
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
            raise RuntimeError(f"Split summary mismatch for {role}: {row}")

    profile = build_canonical_sample_profile()
    fingerprints = build_predecessor_fingerprint_registry(repo_root)
    exceptions = build_exceptions_registry()
    generation = {
        "manifest_version": "1.0",
        "phase": "C-A5",
        "profile_id": CANONICAL_SAMPLE_PROFILE_ID,
        "generator_script": "scripts/audit_co2_canonical_samples.py",
        "module": "datasets/co2/canonical_samples.py",
        "canonical_source_samples": len(samples),
        "model_eligible_samples": availability["co2_slope_eligible"],
        "warmup_unavailable_samples": availability["co2_slope_unavailable"],
        "scaler_fitted": False,
        "model_trained": False,
        "class_balancing_performed": False,
        "feature_selection_performed": False,
        "synthetic_npz_used_as_real_source": False,
        "a_series_release_status": "DEFERRED_UNTIL_C-A6",
        "determinism": {
            "host_timezone_independent": True,
            "locale_independent": True,
            "random_values_used": False,
        },
    }

    samples_path = out_dir / "canonical_source_samples.jsonl"
    write_jsonl(samples_path, [s.to_dict() for s in samples])

    eligible_ids_path = out_dir / "model_eligible_sample_ids.jsonl"
    write_jsonl(
        eligible_ids_path,
        [
            {
                "canonical_sample_id": s.canonical_sample_id,
                "canonical_sample_index": s.canonical_sample_index,
                "future_split_role": s.future_split_role,
            }
            for s in samples
            if s.model_eligible_for_slope_complete_view
        ],
    )

    _write_json(out_dir / "canonical_sample_profile.json", profile)
    _write_json(out_dir / "predecessor_fingerprint_registry.json", fingerprints)
    _write_json(out_dir / "split_membership_manifest.json", split)
    _write_json(out_dir / "feature_availability_manifest.json", availability)
    _write_json(out_dir / "materialization_integrity_summary.json", integrity)
    _write_json(out_dir / "exceptions_and_limitations.json", exceptions)
    _write_json(out_dir / "generation_metadata.json", generation)

    artifact_identity = {
        "manifest_version": "1.0",
        "profile_id": CANONICAL_SAMPLE_PROFILE_ID,
        "artifacts": [],
    }
    checksum_files = [
        "canonical_sample_profile.json",
        "predecessor_fingerprint_registry.json",
        "split_membership_manifest.json",
        "feature_availability_manifest.json",
        "materialization_integrity_summary.json",
        "exceptions_and_limitations.json",
        "generation_metadata.json",
        "canonical_source_samples.jsonl",
        "model_eligible_sample_ids.jsonl",
    ]
    lines = []
    for fname in checksum_files:
        path = out_dir / fname
        rel = f"datasets/co2/manifests/c_a5_canonical_samples/{fname}"
        digest = compute_sha256_file(path)
        size = path.stat().st_size
        lines.append(f"{digest}  {rel}")
        artifact_identity["artifacts"].append(
            {
                "path": rel,
                "byte_size": size,
                "sha256": digest,
            }
        )
    _write_json(out_dir / "artifact_identity.json", artifact_identity)
    # include artifact_identity in checksums after writing
    aid_rel = "datasets/co2/manifests/c_a5_canonical_samples/artifact_identity.json"
    lines.append(f"{compute_sha256_file(out_dir / 'artifact_identity.json')}  {aid_rel}")
    (out_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"✅ Generated C-A5 canonical samples in: {out_dir.relative_to(repo_root)}")
    print(
        f"   canonical={len(samples)} eligible={availability['co2_slope_eligible']} "
        f"warmup={availability['co2_slope_unavailable']}"
    )
    return out_dir


if __name__ == "__main__":
    audit_co2_canonical_samples()
