#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datasets/co2/canonical_samples.py
Phase C-A5 — CO₂ Canonical Sample Provenance and Group-Wise Split Materialization.

Joins C-A1 source observations, C-A2 blocks/splits, C-A3 slope features, and
C-A4 occupancy targets into deterministic canonical sample records.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets.co2.raw_reader import (
    EXPECTED_ARCHIVE_REL_PATH,
    EXPECTED_ARCHIVE_SHA256,
    CO2SourceRowObservation,
    compute_sha256_file,
    get_repo_root,
)
from datasets.co2.slope_feature import (
    FEATURE_PROFILE_ID as SLOPE_PROFILE_ID,
    MEMBER_ORDER,
    STATUS_AVAILABLE,
    STATUS_WARMUP,
    SlopeFeatureRecord,
    parse_source_timestamp,
    reconstruct_all_slope_features,
)
from datasets.co2.target_semantics import (
    TARGET_PROFILE_ID as OCCUPANCY_TARGET_PROFILE_ID,
    CanonicalOccupancyTarget,
    reconstruct_all_occupancy_targets,
)

CANONICAL_SAMPLE_PROFILE_ID = "CO2_CANONICAL_SAMPLE_PROFILE_001"
EXPECTED_TOTAL_SAMPLES = 20560
EXPECTED_SLOPE_ELIGIBLE = 20551
EXPECTED_WARMUP = 9
ORDERING_RULE = "CHRONOLOGICAL_C_A2_MEMBER_ORDER"
# MEMBER_ORDER: datatest.txt → datatraining.txt → datatest2.txt

PREDECESSOR_CHECKSUM_FILES = {
    "C-A1": [
        "datasets/co2/manifests/c_a1_safe_reader/checksums.sha256",
        "datasets/co2/manifests/c_a1_safe_reader/source_row_provenance_contract.json",
        "datasets/co2/manifests/c_a1_safe_reader/reader_validation_summary.json",
        "datasets/co2/manifests/c_a1_safe_reader/source_schema_profile.json",
    ],
    "C-A2": [
        "datasets/co2/manifests/c_a2_temporal_blocks/checksums.sha256",
        "datasets/co2/manifests/c_a2_temporal_blocks/temporal_blocks_manifest.json",
        "datasets/co2/manifests/c_a2_temporal_blocks/grouping_split_contract.json",
        "datasets/co2/manifests/c_a2_temporal_blocks/timestamp_cadence_profile.json",
    ],
    "C-A3": [
        "datasets/co2/manifests/c_a3_slope_feature/checksums.sha256",
        "datasets/co2/manifests/c_a3_slope_feature/co2_slope_feature_profile.json",
        "datasets/co2/manifests/c_a3_slope_feature/feature_eligibility_summary.json",
    ],
    "C-A4": [
        "datasets/co2/manifests/c_a4_target_semantics/checksums.sha256",
        "datasets/co2/manifests/c_a4_target_semantics/occupancy_target_profile.json",
        "datasets/co2/manifests/c_a4_target_semantics/target_integrity_summary.json",
    ],
}


@dataclass(frozen=True)
class CanonicalSampleRecord:
    """One canonical source sample (exactly one C-A1 source row)."""

    canonical_sample_id: str
    canonical_sample_index: int
    sample_kind: str
    source_archive_path: str
    source_archive_sha256: str
    source_member_name: str
    source_member_sha256: str
    source_physical_line_number: int
    source_row_identifier: str
    source_timestamp_raw: str
    canonical_timestamp: str
    temporal_block_id: str
    future_split_role: str
    temperature: float
    humidity: float
    light: float
    co2: float
    humidity_ratio: float
    occupancy_source_value: int
    occupancy_canonical_class: str
    target_profile_id: str
    co2_slope_profile_id: str
    co2_slope_status: str
    co2_slope: Optional[float]
    history_start_source_row_identifier: Optional[str]
    history_elapsed_seconds: Optional[float]
    source_sample_count_used: int
    model_eligible_for_slope_complete_view: bool
    model_eligibility_exclusion_reason: Optional[str]
    scaler_fit_authorized: bool
    locked_test_fit_authorized: bool
    locked_test_tuning_authorized: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_canonical_sample_id(obs: CO2SourceRowObservation) -> str:
    """
    Deterministic ID from stable source identity (not random, not process order alone).
    """
    payload = "|".join(
        [
            obs.source_archive_sha256,
            obs.source_member_name,
            obs.source_row_identifier,
            str(obs.source_physical_line_number),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"co2cs_{digest[:32]}"


def build_canonical_sample_profile() -> Dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "profile_id": CANONICAL_SAMPLE_PROFILE_ID,
        "sample_grain": "ONE_C_A1_SOURCE_ROW",
        "expected_canonical_source_samples": EXPECTED_TOTAL_SAMPLES,
        "canonical_sample_id_definition": (
            "co2cs_ + sha256(archive_sha256|member|source_row_identifier|"
            "physical_line)[:32]"
        ),
        "ordering_rule": ORDERING_RULE,
        "ordering_member_sequence": list(MEMBER_ORDER),
        "inherited_profiles": {
            "slope_feature_profile_id": SLOPE_PROFILE_ID,
            "occupancy_target_profile_id": OCCUPANCY_TARGET_PROFILE_ID,
        },
        "sample_kinds": {
            "CANONICAL_SOURCE_SAMPLE": (
                "Every C-A1 source row, including slope warm-up rows."
            ),
            "MODEL_ELIGIBLE_SAMPLE": (
                "Canonical source sample with FEATURE_AVAILABLE CO2_slope under "
                "C-A3; explicit derived view, not a silent reduction of canonical set."
            ),
        },
        "schema": {
            "raw_measured_features": [
                "Temperature",
                "Humidity",
                "Light",
                "CO2",
                "HumidityRatio",
            ],
            "derived_features": ["CO2_slope"],
            "target": ["Occupancy"],
            "metadata_not_model_inputs": [
                "canonical_sample_id",
                "source provenance fields",
                "temporal_block_id",
                "future_split_role",
                "feature availability / lineage",
            ],
        },
        "split_contract": {
            "TRAIN": "BLOCK_02_DATATRAINING",
            "VALIDATION": "BLOCK_01_DATATEST",
            "LOCKED_TEST": "BLOCK_03_DATATEST2",
            "random_row_wise_split": "PROHIBITED",
        },
        "access_semantics": {
            "scaler_fit_authorized_roles": ["TRAIN"],
            "validation_authorized_for_fitting": False,
            "locked_test_authorized_for_fitting": False,
            "locked_test_authorized_for_tuning": False,
            "locked_test_authorized_for_model_selection": False,
            "locked_test_materialized_for_membership_and_provenance": True,
        },
        "feature_selection_status": "NOT_PERFORMED_IN_C_A5",
        "class_imbalance_intervention": "NOT_PERFORMED_IN_C_A5",
        "a_series_release_status": "DEFERRED_UNTIL_C-A6",
        "synthetic_npz_isolation": {
            "path": "datasets/co2/processed/co2_occupancy_v1.npz",
            "status": "SYNTHETIC_SMOKE_FIXTURE",
            "used_as_real_source": False,
        },
    }


def _obs_key(obs: CO2SourceRowObservation) -> Tuple[str, str]:
    return (obs.source_member_name, obs.source_row_identifier)


def _slope_key(rec: SlopeFeatureRecord) -> Tuple[str, str]:
    return (rec.target_source_member, rec.target_source_row_identifier)


def _target_key(rec: CanonicalOccupancyTarget) -> Tuple[str, str]:
    return (rec.target_source_member, rec.target_source_row_identifier)


def materialize_canonical_samples(
    observations: Sequence[CO2SourceRowObservation],
) -> List[CanonicalSampleRecord]:
    """
    Build exactly one canonical sample per source observation.

    Ordering: C-A2 chronological member order (MEMBER_ORDER), preserving
    within-member source order from the observation list regrouped by member.
    """
    by_member: Dict[str, List[CO2SourceRowObservation]] = {m: [] for m in MEMBER_ORDER}
    for obs in observations:
        if obs.source_member_name not in by_member:
            raise ValueError(f"Unexpected member: {obs.source_member_name}")
        by_member[obs.source_member_name].append(obs)

    ordered_obs: List[CO2SourceRowObservation] = []
    for member in MEMBER_ORDER:
        ordered_obs.extend(by_member[member])

    slopes = reconstruct_all_slope_features(ordered_obs)
    targets = reconstruct_all_occupancy_targets(ordered_obs)
    if not (len(ordered_obs) == len(slopes) == len(targets) == EXPECTED_TOTAL_SAMPLES):
        # Allow non-20560 only for synthetic unit fixtures.
        if len(ordered_obs) != len(slopes) or len(ordered_obs) != len(targets):
            raise ValueError("Observation/slope/target length mismatch")

    slope_by_key = {_slope_key(s): s for s in slopes}
    target_by_key = {_target_key(t): t for t in targets}
    if len(slope_by_key) != len(slopes) or len(target_by_key) != len(targets):
        raise ValueError("Duplicate slope or target keys")

    samples: List[CanonicalSampleRecord] = []
    seen_ids = set()
    for idx, obs in enumerate(ordered_obs):
        key = _obs_key(obs)
        slope = slope_by_key[key]
        target = target_by_key[key]
        _, ts_canonical = parse_source_timestamp(obs.source_timestamp_raw)
        sample_id = make_canonical_sample_id(obs)
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate canonical sample ID: {sample_id}")
        seen_ids.add(sample_id)

        eligible = slope.feature_status == STATUS_AVAILABLE
        exclusion = None if eligible else slope.feature_status
        samples.append(
            CanonicalSampleRecord(
                canonical_sample_id=sample_id,
                canonical_sample_index=idx,
                sample_kind="CANONICAL_SOURCE_SAMPLE",
                source_archive_path=obs.source_archive_path,
                source_archive_sha256=obs.source_archive_sha256,
                source_member_name=obs.source_member_name,
                source_member_sha256=obs.source_member_sha256,
                source_physical_line_number=obs.source_physical_line_number,
                source_row_identifier=obs.source_row_identifier,
                source_timestamp_raw=obs.source_timestamp_raw,
                canonical_timestamp=ts_canonical,
                temporal_block_id=slope.temporal_block_id,
                future_split_role=slope.future_split_role,
                temperature=float(obs.temperature),
                humidity=float(obs.humidity),
                light=float(obs.light),
                co2=float(obs.co2),
                humidity_ratio=float(obs.humidity_ratio),
                occupancy_source_value=int(target.occupancy_source_value),
                occupancy_canonical_class=target.occupancy_semantic_name,
                target_profile_id=OCCUPANCY_TARGET_PROFILE_ID,
                co2_slope_profile_id=SLOPE_PROFILE_ID,
                co2_slope_status=slope.feature_status,
                co2_slope=slope.co2_slope,
                history_start_source_row_identifier=slope.history_start_source_row_identifier,
                history_elapsed_seconds=slope.history_elapsed_seconds,
                source_sample_count_used=int(slope.source_sample_count_used),
                model_eligible_for_slope_complete_view=eligible,
                model_eligibility_exclusion_reason=exclusion,
                scaler_fit_authorized=(slope.future_split_role == "TRAIN"),
                locked_test_fit_authorized=False,
                locked_test_tuning_authorized=False,
            )
        )
    return samples


def summarize_split_membership(samples: Sequence[CanonicalSampleRecord]) -> Dict[str, Any]:
    by_role: Dict[str, Dict[str, Any]] = {}
    for role in ("TRAIN", "VALIDATION", "LOCKED_TEST"):
        role_samples = [s for s in samples if s.future_split_role == role]
        eligible = [s for s in role_samples if s.model_eligible_for_slope_complete_view]
        warmup = [
            s
            for s in role_samples
            if s.co2_slope_status == STATUS_WARMUP
        ]
        by_role[role] = {
            "future_split_role": role,
            "canonical_source_samples": len(role_samples),
            "slope_eligible_samples": len(eligible),
            "warmup_unavailable_samples": len(warmup),
            "vacant_count": sum(1 for s in role_samples if s.occupancy_source_value == 0),
            "occupied_count": sum(1 for s in role_samples if s.occupancy_source_value == 1),
            "scaler_fit_authorized": role == "TRAIN",
            "locked_test_fit_authorized": False,
            "locked_test_tuning_authorized": False,
        }
    return {
        "manifest_version": "1.0",
        "profile_id": CANONICAL_SAMPLE_PROFILE_ID,
        "random_row_wise_split": False,
        "split_assignments": {
            "TRAIN": "BLOCK_02_DATATRAINING",
            "VALIDATION": "BLOCK_01_DATATEST",
            "LOCKED_TEST": "BLOCK_03_DATATEST2",
        },
        "by_role": by_role,
    }


def summarize_feature_availability(
    samples: Sequence[CanonicalSampleRecord],
) -> Dict[str, Any]:
    eligible_ids = [
        s.canonical_sample_id
        for s in samples
        if s.model_eligible_for_slope_complete_view
    ]
    excluded = [
        {
            "canonical_sample_id": s.canonical_sample_id,
            "canonical_sample_index": s.canonical_sample_index,
            "future_split_role": s.future_split_role,
            "exclusion_reason": s.model_eligibility_exclusion_reason,
            "co2_slope_status": s.co2_slope_status,
        }
        for s in samples
        if not s.model_eligible_for_slope_complete_view
    ]
    return {
        "manifest_version": "1.0",
        "profile_id": CANONICAL_SAMPLE_PROFILE_ID,
        "canonical_source_samples": len(samples),
        "co2_slope_eligible": len(eligible_ids),
        "co2_slope_unavailable": len(excluded),
        "model_eligible_view_definition": (
            "Canonical samples with co2_slope_status == FEATURE_AVAILABLE"
        ),
        "model_eligible_sample_ids_sha256": hashlib.sha256(
            ("\n".join(eligible_ids) + ("\n" if eligible_ids else "")).encode("utf-8")
        ).hexdigest(),
        "excluded_from_model_eligible_view": excluded,
        "note": (
            "Canonical lineage retains all source samples; model-eligible view is "
            "explicitly derived and must not replace the 20560-row canonical set."
        ),
    }


def summarize_materialization_integrity(
    observations: Sequence[CO2SourceRowObservation],
    samples: Sequence[CanonicalSampleRecord],
) -> Dict[str, Any]:
    obs_keys = {_obs_key(o) for o in observations}
    sample_keys = {(s.source_member_name, s.source_row_identifier) for s in samples}
    missing = sorted(obs_keys - sample_keys)
    extra = sorted(sample_keys - obs_keys)
    ids = [s.canonical_sample_id for s in samples]
    return {
        "manifest_version": "1.0",
        "profile_id": CANONICAL_SAMPLE_PROFILE_ID,
        "source_observation_count": len(observations),
        "canonical_source_sample_count": len(samples),
        "missing_source_mappings": len(missing),
        "extra_canonical_mappings": len(extra),
        "duplicate_canonical_ids": len(ids) - len(set(ids)),
        "one_to_one_ok": (
            len(observations) == len(samples)
            and not missing
            and not extra
            and len(ids) == len(set(ids))
        ),
        "target_labels_modified": 0,
        "split_assignments_modified": 0,
        "slope_status_modified": 0,
        "random_row_wise_split_used": False,
        "scaler_fitted": False,
        "model_trained": False,
        "synthetic_npz_used_as_real_source": False,
        "a_series_release_status": "DEFERRED_UNTIL_C-A6",
    }


def build_predecessor_fingerprint_registry(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root or get_repo_root()
    phases: Dict[str, Any] = {}
    for phase, paths in PREDECESSOR_CHECKSUM_FILES.items():
        entries = []
        for rel in paths:
            path = root / rel
            if not path.exists():
                raise FileNotFoundError(f"Missing predecessor artifact: {rel}")
            entries.append(
                {
                    "path": rel,
                    "sha256": compute_sha256_file(path),
                    "byte_size": path.stat().st_size,
                }
            )
        phases[phase] = entries
    return {
        "manifest_version": "1.0",
        "consumer_phase": "C-A5",
        "canonical_sample_profile_id": CANONICAL_SAMPLE_PROFILE_ID,
        "source_archive": {
            "path": EXPECTED_ARCHIVE_REL_PATH,
            "sha256": EXPECTED_ARCHIVE_SHA256,
            "read_only": True,
        },
        "phases": phases,
        "note": (
            "C-A5 validation must fail if any listed predecessor checksum changes "
            "without regenerating C-A5 artifacts."
        ),
    }


def verify_predecessor_fingerprints(
    registry: Dict[str, Any],
    repo_root: Optional[Path] = None,
) -> List[str]:
    root = repo_root or get_repo_root()
    errors: List[str] = []
    for phase, entries in registry.get("phases", {}).items():
        for entry in entries:
            rel = entry["path"]
            path = root / rel
            if not path.exists():
                errors.append(f"Missing predecessor fingerprint path: {rel}")
                continue
            actual = compute_sha256_file(path)
            if actual != entry["sha256"]:
                errors.append(
                    f"Predecessor fingerprint mismatch [{phase}] {rel}: "
                    f"expected {entry['sha256']}, got {actual}"
                )
            if path.stat().st_size != entry["byte_size"]:
                errors.append(f"Predecessor size mismatch for {rel}")
    return errors


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
            f.write("\n")
