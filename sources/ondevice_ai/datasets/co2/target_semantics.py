#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datasets/co2/target_semantics.py
Phase C-A4 — CO₂ Occupancy Label Semantics, Safety Separation, and Canonical Target Contract.

Defines deterministic source-target semantics for UCI Occupancy Detection labels.
Does not train models, fit scalers, or embed safety thresholds into the target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from datasets.co2.raw_reader import CO2SourceRowObservation
from datasets.co2.slope_feature import MEMBER_ORDER, MEMBER_TO_BLOCK

TARGET_PROFILE_ID = "CO2_OCCUPANCY_TARGET_PROFILE_001"
TARGET_NAME = "Occupancy"
SOURCE_FIELD = "Occupancy"
SOURCE_ALLOWED_VALUES = (0, 1)
CANONICAL_DTYPE = "int64"
NEGATIVE_CLASS_VALUE = 0
POSITIVE_CLASS_VALUE = 1
NEGATIVE_CLASS_NAME = "VACANT"
POSITIVE_CLASS_NAME = "OCCUPIED"
CANONICAL_CLASS_MAPPING = {
    "0": NEGATIVE_CLASS_NAME,
    "1": POSITIVE_CLASS_NAME,
}
EXPECTED_TOTAL_ROWS = 20560
EXPECTED_OCC_0 = 15810
EXPECTED_OCC_1 = 4750

# Existing SafeNest risk code inspects CO2>1500 for safety scoring. C-A4 documents
# separation only and must never embed this threshold into occupancy target mapping.
DOCUMENTED_OUT_OF_SCOPE_CO2_SAFETY_THRESHOLD_PPM = 1500.0


class TargetSemanticsError(ValueError):
    """Raised when a source occupancy label violates the C-A4 contract."""


@dataclass(frozen=True)
class CanonicalOccupancyTarget:
    """One source-row occupancy target with preserved source value + semantic name."""

    target_source_member: str
    target_source_row_identifier: str
    target_physical_line: int
    temporal_block_id: str
    future_split_role: str
    occupancy_source_value: int
    occupancy_semantic_name: str
    target_profile_id: str
    label_derivation: str
    mapping_type: str
    assignment_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_occupancy_target_profile() -> Dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "target_profile_id": TARGET_PROFILE_ID,
        "target_name": TARGET_NAME,
        "source_field": SOURCE_FIELD,
        "source_semantics": (
            "Room occupancy state as defined by the original UCI Occupancy Detection "
            "dataset for a single office room. Occupancy=1 means the room was labeled "
            "occupied in the source; Occupancy=0 means vacant/unoccupied in the source."
        ),
        "source_allowed_values": list(SOURCE_ALLOWED_VALUES),
        "canonical_dtype": CANONICAL_DTYPE,
        "canonical_class_mapping": dict(CANONICAL_CLASS_MAPPING),
        "negative_class": {
            "source_value": NEGATIVE_CLASS_VALUE,
            "semantic_name": NEGATIVE_CLASS_NAME,
        },
        "positive_class": {
            "source_value": POSITIVE_CLASS_VALUE,
            "semantic_name": POSITIVE_CLASS_NAME,
        },
        "semantic_naming_basis": (
            "Aligned with existing models/model_manifest.json CO2 class_map "
            "{0: VACANT, 1: OCCUPIED}; source integer values remain authoritative."
        ),
        "is_source_label": True,
        "is_model_prediction": False,
        "is_safety_state": False,
        "is_sensor_health_state": False,
        "is_multisensor_risk_state": False,
        "label_derivation": "NONE",
        "mapping_type": "IDENTITY_SOURCE_PRESERVATION",
        "threshold_based_relabeling": "PROHIBITED",
        "class_balancing_at_label_definition_stage": "PROHIBITED",
        "feature_driven_relabeling": "PROHIBITED",
        "co2_ppm_may_modify_label": False,
        "co2_slope_may_modify_label": False,
        "safety_threshold_may_modify_label": False,
        "locked_test_used_for_contract_selection": False,
        "occupancy_means_dangerous_co2": False,
        "vacant_means_safe_co2": False,
        "high_co2_implies_occupied": False,
        "low_co2_implies_vacant": False,
        "documented_out_of_scope_safety_threshold_ppm": (
            DOCUMENTED_OUT_OF_SCOPE_CO2_SAFETY_THRESHOLD_PPM
        ),
        "documented_out_of_scope_safety_threshold_note": (
            "Active SafeNest code references CO2 > 1500 ppm for risk/safety scoring. "
            "C-A4 inspects this only to enforce semantic separation and does not "
            "embed or calibrate that threshold into the occupancy target contract."
        ),
    }


def map_occupancy_source_value(value: Any) -> CanonicalOccupancyTarget:
    """Validate and map a raw occupancy source value for contract/unit tests."""
    if value is None:
        raise TargetSemanticsError("missing occupancy target label")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TargetSemanticsError(f"invalid occupancy label: {value!r}")
    if value not in SOURCE_ALLOWED_VALUES:
        raise TargetSemanticsError(f"unexpected occupancy label: {value}")
    return CanonicalOccupancyTarget(
        target_source_member="__fixture__",
        target_source_row_identifier="__fixture__",
        target_physical_line=0,
        temporal_block_id="__fixture__",
        future_split_role="__fixture__",
        occupancy_source_value=value,
        occupancy_semantic_name=CANONICAL_CLASS_MAPPING[str(value)],
        target_profile_id=TARGET_PROFILE_ID,
        label_derivation="NONE",
        mapping_type="IDENTITY_SOURCE_PRESERVATION",
        assignment_status="ASSIGNED_FROM_SOURCE",
    )


def occupancy_semantic_name(source_value: int) -> str:
    if source_value not in SOURCE_ALLOWED_VALUES:
        raise TargetSemanticsError(f"unexpected occupancy label: {source_value}")
    return CANONICAL_CLASS_MAPPING[str(source_value)]


def build_canonical_target_from_observation(
    obs: CO2SourceRowObservation,
) -> CanonicalOccupancyTarget:
    if obs.source_member_name not in MEMBER_TO_BLOCK:
        raise TargetSemanticsError(f"unknown source member: {obs.source_member_name}")
    block_id, role = MEMBER_TO_BLOCK[obs.source_member_name]
    value = obs.occupancy
    if not isinstance(value, int) or isinstance(value, bool):
        raise TargetSemanticsError(
            f"non-int occupancy at {obs.source_member_name}:{obs.source_row_identifier}"
        )
    if value not in SOURCE_ALLOWED_VALUES:
        raise TargetSemanticsError(
            f"unexpected occupancy {value} at "
            f"{obs.source_member_name}:{obs.source_row_identifier}"
        )
    return CanonicalOccupancyTarget(
        target_source_member=obs.source_member_name,
        target_source_row_identifier=obs.source_row_identifier,
        target_physical_line=obs.source_physical_line_number,
        temporal_block_id=block_id,
        future_split_role=role,
        occupancy_source_value=value,
        occupancy_semantic_name=occupancy_semantic_name(value),
        target_profile_id=TARGET_PROFILE_ID,
        label_derivation="NONE",
        mapping_type="IDENTITY_SOURCE_PRESERVATION",
        assignment_status="ASSIGNED_FROM_SOURCE",
    )


def reconstruct_all_occupancy_targets(
    observations: Sequence[CO2SourceRowObservation],
) -> List[CanonicalOccupancyTarget]:
    """Preserve source order: datatest, datatraining, datatest2."""
    by_member: Dict[str, List[CO2SourceRowObservation]] = {m: [] for m in MEMBER_ORDER}
    for obs in observations:
        if obs.source_member_name not in by_member:
            raise TargetSemanticsError(f"unexpected member {obs.source_member_name}")
        by_member[obs.source_member_name].append(obs)
    targets: List[CanonicalOccupancyTarget] = []
    for member in MEMBER_ORDER:
        for obs in by_member[member]:
            targets.append(build_canonical_target_from_observation(obs))
    return targets


def build_feature_target_role_registry() -> Dict[str, Any]:
    def feature(name: str, role: str, notes: str = "") -> Dict[str, Any]:
        return {
            "field_name": name,
            "role": role,
            "is_model_target": role == "SOURCE_TARGET_LABEL",
            "may_modify_occupancy_label": False,
            "notes": notes,
        }

    return {
        "manifest_version": "1.0",
        "registry_id": "CO2_FEATURE_TARGET_ROLE_REGISTRY_001",
        "target_profile_id": TARGET_PROFILE_ID,
        "fields": [
            feature("Temperature", "MEASURED_FEATURE"),
            feature("Humidity", "MEASURED_FEATURE"),
            feature("Light", "MEASURED_FEATURE"),
            feature("CO2", "MEASURED_FEATURE", "Measured ppm; not an occupancy label."),
            feature("HumidityRatio", "MEASURED_FEATURE"),
            feature(
                "CO2_slope",
                "DERIVED_FEATURE",
                "C-A3 derived ppm/min feature; never a target label.",
            ),
            feature(
                "Occupancy",
                "SOURCE_TARGET_LABEL",
                "Original UCI source label preserved by C-A4.",
            ),
        ],
        "future_model_input_note": (
            "Existing model input order [CO2_slope, Humidity, CO2] is a later "
            "feature-selection/runtime concern; C-A4 does not perform feature selection."
        ),
    }


def build_occupancy_safety_separation_contract() -> Dict[str, Any]:
    concepts = [
        {
            "concept_id": "UCI_OCCUPANCY_LABEL",
            "semantic_domain": "SOURCE_OCCUPANCY_CLASSIFICATION",
            "source_or_derived": "SOURCE",
            "produced_in_current_phase": True,
            "allowed_to_modify_occupancy_label": False,
            "used_as_model_target": True,
            "used_as_safety_rule": False,
            "notes": "Authoritative C-A4 target.",
        },
        {
            "concept_id": "MEASURED_CO2_PPM",
            "semantic_domain": "PHYSICAL_MEASUREMENT",
            "source_or_derived": "SOURCE",
            "produced_in_current_phase": False,
            "allowed_to_modify_occupancy_label": False,
            "used_as_model_target": False,
            "used_as_safety_rule": False,
            "notes": "Measured feature only in C-A4 scope.",
        },
        {
            "concept_id": "DERIVED_CO2_SLOPE",
            "semantic_domain": "DERIVED_TEMPORAL_FEATURE",
            "source_or_derived": "DERIVED",
            "produced_in_current_phase": False,
            "allowed_to_modify_occupancy_label": False,
            "used_as_model_target": False,
            "used_as_safety_rule": False,
            "notes": "Inherited from C-A3; feature not target.",
        },
        {
            "concept_id": "FUTURE_OCCUPANCY_PROBABILITY",
            "semantic_domain": "MODEL_OUTPUT",
            "source_or_derived": "DERIVED",
            "produced_in_current_phase": False,
            "allowed_to_modify_occupancy_label": False,
            "used_as_model_target": False,
            "used_as_safety_rule": False,
            "notes": "Deferred to later model phases; not created in C-A4.",
        },
        {
            "concept_id": "RULE_BASED_CO2_SAFETY_STATE",
            "semantic_domain": "SAFETY_RISK_RULE",
            "source_or_derived": "DERIVED",
            "produced_in_current_phase": False,
            "allowed_to_modify_occupancy_label": False,
            "used_as_model_target": False,
            "used_as_safety_rule": True,
            "notes": (
                "Existing risk/adapter code may use CO2>1500; out of C-A4 scope "
                "(DEFERRED_SAFETY_RULE_CONTRACT)."
            ),
        },
        {
            "concept_id": "SENSOR_HEALTH_STATE",
            "semantic_domain": "SENSOR_HEALTH",
            "source_or_derived": "DERIVED",
            "produced_in_current_phase": False,
            "allowed_to_modify_occupancy_label": False,
            "used_as_model_target": False,
            "used_as_safety_rule": False,
            "notes": "SENSOR_HEALTH_CONTRACT_OUT_OF_SCOPE for C-A4.",
        },
        {
            "concept_id": "MULTISENSOR_RISK_SCORE",
            "semantic_domain": "MULTISENSOR_RISK",
            "source_or_derived": "DERIVED",
            "produced_in_current_phase": False,
            "allowed_to_modify_occupancy_label": False,
            "used_as_model_target": False,
            "used_as_safety_rule": True,
            "notes": "MULTISENSOR_RISK_CONTRACT_OUT_OF_SCOPE for C-A4.",
        },
    ]
    return {
        "manifest_version": "1.0",
        "contract_id": "CO2_OCCUPANCY_SAFETY_SEPARATION_CONTRACT_001",
        "target_profile_id": TARGET_PROFILE_ID,
        "invariants": [
            "Occupancy==1 DOES NOT MEAN dangerous CO2 exposure",
            "Occupancy==0 DOES NOT MEAN safe CO2 environment",
            "CO2 above any safety threshold DOES NOT AUTOMATICALLY MEAN Occupancy==1",
            "CO2 at or below any safety threshold DOES NOT AUTOMATICALLY MEAN Occupancy==0",
            "CO2_slope IS A FEATURE NOT A TARGET",
            "occupancy model prediction IS NOT SafeNest emergency risk",
        ],
        "concepts": concepts,
        "threshold_based_relabeling": "PROHIBITED",
        "deferred_contracts": [
            "DEFERRED_SAFETY_RULE_CONTRACT",
            "SENSOR_HEALTH_CONTRACT_OUT_OF_SCOPE",
            "MULTISENSOR_RISK_CONTRACT_OUT_OF_SCOPE",
        ],
    }


def summarize_target_integrity(
    observations: Sequence[CO2SourceRowObservation],
    targets: Sequence[CanonicalOccupancyTarget],
) -> Dict[str, Any]:
    if len(observations) != len(targets):
        raise TargetSemanticsError("observation/target count mismatch")

    obs_by_key = {
        (obs.source_member_name, obs.source_row_identifier): obs for obs in observations
    }
    if len(obs_by_key) != len(observations):
        raise TargetSemanticsError("duplicate observation keys detected")

    modified = 0
    for tgt in targets:
        key = (tgt.target_source_member, tgt.target_source_row_identifier)
        if key not in obs_by_key:
            modified += 1
            continue
        obs = obs_by_key[key]
        if obs.occupancy != tgt.occupancy_source_value:
            modified += 1
        if obs.source_physical_line_number != tgt.target_physical_line:
            modified += 1

    occ0 = sum(1 for t in targets if t.occupancy_source_value == 0)
    occ1 = sum(1 for t in targets if t.occupancy_source_value == 1)
    unexpected = sum(
        1 for t in targets if t.occupancy_source_value not in SOURCE_ALLOWED_VALUES
    )

    by_role: Dict[str, Dict[str, Any]] = {}
    for role in ("TRAIN", "VALIDATION", "LOCKED_TEST"):
        role_targets = [t for t in targets if t.future_split_role == role]
        by_role[role] = {
            "future_split_role": role,
            "source_row_count": len(role_targets),
            "occupancy_0_count": sum(
                1 for t in role_targets if t.occupancy_source_value == 0
            ),
            "occupancy_1_count": sum(
                1 for t in role_targets if t.occupancy_source_value == 1
            ),
        }

    return {
        "manifest_version": "1.0",
        "target_profile_id": TARGET_PROFILE_ID,
        "total_source_rows": len(targets),
        "occupancy_0_count": occ0,
        "occupancy_1_count": occ1,
        "unexpected_labels": unexpected,
        "missing_target_labels": 0,
        "modified_target_labels": modified,
        "derived_reconstructed_labels": 0,
        "expected_total_source_rows": EXPECTED_TOTAL_ROWS,
        "expected_occupancy_0_count": EXPECTED_OCC_0,
        "expected_occupancy_1_count": EXPECTED_OCC_1,
        "counts_match_predecessor_expectation": (
            len(targets) == EXPECTED_TOTAL_ROWS
            and occ0 == EXPECTED_OCC_0
            and occ1 == EXPECTED_OCC_1
            and unexpected == 0
            and modified == 0
        ),
        "by_future_split_role": by_role,
        "statistics_role": "TARGET_INTEGRITY_ONLY",
        "not_model_performance": True,
    }


def audit_label_transitions(
    observations: Sequence[CO2SourceRowObservation],
) -> Dict[str, Any]:
    """Descriptive per-block transition provenance; never used to alter labels."""
    blocks: Dict[str, List[int]] = {}
    member_meta: Dict[str, Tuple[str, str]] = {}
    for obs in observations:
        block_id, role = MEMBER_TO_BLOCK[obs.source_member_name]
        blocks.setdefault(block_id, []).append(int(obs.occupancy))
        member_meta[block_id] = (obs.source_member_name, role)

    per_block = []
    for block_id in sorted(blocks.keys()):
        labels = blocks[block_id]
        transitions_0_to_1 = 0
        transitions_1_to_0 = 0
        run_lengths: List[int] = []
        if labels:
            current = labels[0]
            run = 1
            for nxt in labels[1:]:
                if nxt == current:
                    run += 1
                else:
                    run_lengths.append(run)
                    if current == 0 and nxt == 1:
                        transitions_0_to_1 += 1
                    elif current == 1 and nxt == 0:
                        transitions_1_to_0 += 1
                    current = nxt
                    run = 1
            run_lengths.append(run)
        member, role = member_meta[block_id]
        per_block.append(
            {
                "temporal_block_id": block_id,
                "source_member_name": member,
                "future_split_role": role,
                "row_count": len(labels),
                "occupancy_0_count": sum(1 for x in labels if x == 0),
                "occupancy_1_count": sum(1 for x in labels if x == 1),
                "transitions_0_to_1": transitions_0_to_1,
                "transitions_1_to_0": transitions_1_to_0,
                "run_count": len(run_lengths),
                "max_run_length": max(run_lengths) if run_lengths else 0,
                "min_run_length": min(run_lengths) if run_lengths else 0,
            }
        )

    return {
        "manifest_version": "1.0",
        "target_profile_id": TARGET_PROFILE_ID,
        "audit_role": "DESCRIPTIVE_PROVENANCE_ONLY",
        "label_smoothing_applied": False,
        "debounce_applied": False,
        "hysteresis_applied": False,
        "labels_modified": False,
        "per_block": per_block,
    }


def assert_features_cannot_relabel(
    occupancy_source_value: int,
    co2_ppm: float,
    co2_slope: Optional[float] = None,
    humidity: Optional[float] = None,
) -> int:
    """
    Explicit helper proving measured/derived features do not alter the target.

    Returns the unchanged occupancy source value.
    """
    _ = (co2_ppm, co2_slope, humidity)
    if occupancy_source_value not in SOURCE_ALLOWED_VALUES:
        raise TargetSemanticsError(f"unexpected occupancy label: {occupancy_source_value}")
    return occupancy_source_value
