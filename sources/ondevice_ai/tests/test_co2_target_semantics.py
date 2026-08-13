#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_co2_target_semantics.py
Focused tests for C-A4 occupancy target semantics and safety separation.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime

import pytest

from datasets.co2.raw_reader import CO2SourceRowObservation, get_repo_root
from datasets.co2.target_semantics import (
    CANONICAL_CLASS_MAPPING,
    DOCUMENTED_OUT_OF_SCOPE_CO2_SAFETY_THRESHOLD_PPM,
    EXPECTED_OCC_0,
    EXPECTED_OCC_1,
    EXPECTED_TOTAL_ROWS,
    NEGATIVE_CLASS_NAME,
    POSITIVE_CLASS_NAME,
    TARGET_PROFILE_ID,
    TargetSemanticsError,
    assert_features_cannot_relabel,
    build_canonical_target_from_observation,
    build_feature_target_role_registry,
    build_occupancy_safety_separation_contract,
    build_occupancy_target_profile,
    map_occupancy_source_value,
    occupancy_semantic_name,
    reconstruct_all_occupancy_targets,
    summarize_target_integrity,
)


def _obs(member: str, row_id: str, line: int, occupancy: int, co2: float = 600.0):
    return CO2SourceRowObservation(
        source_archive_path="datasets/raw_archives/external_datasets/occupancy+detection.zip",
        source_archive_sha256="4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a",
        source_member_name=member,
        source_member_sha256="b2c4d0ce2b9e4e453c476f7125ef31aeec2d1f5c7f5572d0e80de3df6521ab56",
        source_physical_line_number=line,
        source_row_identifier=row_id,
        source_timestamp_raw="2015-02-04 17:51:00",
        timestamp_reference="SOURCE_ACQUISITION_CLOCK",
        source_timezone="UNVERIFIED",
        utc_conversion_claimed=False,
        temperature=20.0,
        humidity=30.0,
        light=100.0,
        co2=co2,
        humidity_ratio=0.01,
        occupancy=occupancy,
    )


def test_occupancy_0_and_1_preservation():
    assert map_occupancy_source_value(0).occupancy_source_value == 0
    assert map_occupancy_source_value(0).occupancy_semantic_name == NEGATIVE_CLASS_NAME
    assert map_occupancy_source_value(1).occupancy_source_value == 1
    assert map_occupancy_source_value(1).occupancy_semantic_name == POSITIVE_CLASS_NAME


def test_invalid_and_missing_target_rejection():
    with pytest.raises(TargetSemanticsError):
        map_occupancy_source_value(2)
    with pytest.raises(TargetSemanticsError):
        map_occupancy_source_value(-1)
    with pytest.raises(TargetSemanticsError):
        map_occupancy_source_value(None)
    with pytest.raises(TargetSemanticsError):
        map_occupancy_source_value(1.5)
    with pytest.raises(TargetSemanticsError):
        map_occupancy_source_value(True)


def test_deterministic_canonical_class_mapping_and_source_preservation():
    profile = build_occupancy_target_profile()
    assert profile["canonical_class_mapping"] == CANONICAL_CLASS_MAPPING
    assert profile["canonical_class_mapping"]["0"] == "VACANT"
    assert profile["canonical_class_mapping"]["1"] == "OCCUPIED"
    assert occupancy_semantic_name(0) == "VACANT"
    assert occupancy_semantic_name(1) == "OCCUPIED"
    tgt = build_canonical_target_from_observation(_obs("datatraining.txt", "1", 2, 1))
    assert tgt.occupancy_source_value == 1
    assert tgt.occupancy_semantic_name == "OCCUPIED"
    assert tgt.label_derivation == "NONE"


def test_co2_and_slope_cannot_relabel_occupancy():
    # Extreme CO2 and slope must leave source occupancy unchanged.
    assert assert_features_cannot_relabel(0, co2_ppm=5000.0, co2_slope=100.0) == 0
    assert assert_features_cannot_relabel(1, co2_ppm=200.0, co2_slope=-50.0) == 1
    high = _obs("datatraining.txt", "9", 10, 0, co2=DOCUMENTED_OUT_OF_SCOPE_CO2_SAFETY_THRESHOLD_PPM + 100)
    tgt = build_canonical_target_from_observation(high)
    assert tgt.occupancy_source_value == 0
    assert tgt.occupancy_semantic_name == "VACANT"


def test_high_co2_threshold_not_embedded_as_target_logic():
    profile = build_occupancy_target_profile()
    assert profile["threshold_based_relabeling"] == "PROHIBITED"
    assert profile["safety_threshold_may_modify_label"] is False
    assert profile["documented_out_of_scope_safety_threshold_ppm"] == 1500.0
    # Mapping function ignores CO2 entirely.
    assert map_occupancy_source_value(0).occupancy_source_value == 0


def test_occupancy_not_safety_and_roles_separated():
    profile = build_occupancy_target_profile()
    assert profile["is_safety_state"] is False
    assert profile["is_model_prediction"] is False
    assert profile["is_source_label"] is True
    roles = {f["field_name"]: f["role"] for f in build_feature_target_role_registry()["fields"]}
    assert roles["CO2"] == "MEASURED_FEATURE"
    assert roles["CO2_slope"] == "DERIVED_FEATURE"
    assert roles["Occupancy"] == "SOURCE_TARGET_LABEL"
    assert roles["Humidity"] == "MEASURED_FEATURE"
    assert roles["Temperature"] == "MEASURED_FEATURE"
    assert roles["Light"] == "MEASURED_FEATURE"
    assert roles["HumidityRatio"] == "MEASURED_FEATURE"


def test_source_target_distinct_from_future_prediction_and_risk():
    sep = build_occupancy_safety_separation_contract()
    by_id = {c["concept_id"]: c for c in sep["concepts"]}
    assert by_id["UCI_OCCUPANCY_LABEL"]["used_as_model_target"] is True
    assert by_id["FUTURE_OCCUPANCY_PROBABILITY"]["produced_in_current_phase"] is False
    assert by_id["RULE_BASED_CO2_SAFETY_STATE"]["used_as_model_target"] is False
    assert by_id["SENSOR_HEALTH_STATE"]["allowed_to_modify_occupancy_label"] is False
    assert by_id["MULTISENSOR_RISK_SCORE"]["allowed_to_modify_occupancy_label"] is False
    assert by_id["DERIVED_CO2_SLOPE"]["used_as_model_target"] is False


def test_real_data_target_counts_preserved():
    repo_root = get_repo_root()
    from datasets.co2.raw_reader import UCIOccupancyRawReader

    obs = UCIOccupancyRawReader(repo_root=repo_root).read_all_observations()
    targets = reconstruct_all_occupancy_targets(obs)
    integrity = summarize_target_integrity(obs, targets)
    assert integrity["total_source_rows"] == EXPECTED_TOTAL_ROWS
    assert integrity["occupancy_0_count"] == EXPECTED_OCC_0
    assert integrity["occupancy_1_count"] == EXPECTED_OCC_1
    assert integrity["unexpected_labels"] == 0
    assert integrity["modified_target_labels"] == 0
    assert integrity["by_future_split_role"]["TRAIN"]["source_row_count"] == 8143
    assert integrity["by_future_split_role"]["VALIDATION"]["source_row_count"] == 2665
    assert integrity["by_future_split_role"]["LOCKED_TEST"]["source_row_count"] == 9752


def test_no_balancing_scaler_or_synthetic_npz_dependence():
    repo_root = get_repo_root()
    c_a4 = repo_root / "datasets/co2/manifests/c_a4_target_semantics"
    generation = json.loads((c_a4 / "generation_metadata.json").read_text(encoding="utf-8"))
    assert generation["class_balancing_performed"] is False
    assert generation["scaler_fitted"] is False
    assert generation["model_trained"] is False
    assert generation["synthetic_npz_used_as_real_label_source"] is False
    for fname in [
        "occupancy_target_profile.json",
        "feature_target_role_registry.json",
        "occupancy_safety_separation_contract.json",
    ]:
        assert "co2_occupancy_v1.npz" not in (c_a4 / fname).read_text(encoding="utf-8")


def test_deterministic_manifest_generation_and_path_portability():
    repo_root = get_repo_root()
    c_a4 = repo_root / "datasets/co2/manifests/c_a4_target_semantics"
    checksums1 = (c_a4 / "checksums.sha256").read_text(encoding="utf-8")
    subprocess.run(
        ["python3", "scripts/audit_co2_target_semantics.py"],
        cwd=str(repo_root),
        check=True,
    )
    checksums2 = (c_a4 / "checksums.sha256").read_text(encoding="utf-8")
    assert checksums1 == checksums2
    for path in c_a4.glob("*"):
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "file://" not in text
    profile = json.loads((c_a4 / "occupancy_target_profile.json").read_text(encoding="utf-8"))
    assert profile["target_profile_id"] == TARGET_PROFILE_ID


def test_validator_script():
    repo_root = get_repo_root()
    res = subprocess.run(
        ["python3", "scripts/validate_co2_target_semantics.py"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "C-A5 Authorized:  YES" in res.stdout
    assert "Gate Status:      PASS_WITH_WARNINGS" in res.stdout or "Gate Status:      PASS" in res.stdout
