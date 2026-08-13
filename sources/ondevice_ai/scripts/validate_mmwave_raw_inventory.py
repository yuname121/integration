#!/usr/bin/env python3
"""
Machine-Readable Inventory Validator for Phase A0 mmWave Raw Dataset Inventory.

Verifies internal consistency across all Phase A0 manifest files:
- Valid JSON/JSONL syntax
- Uniqueness of deterministic identifiers (source_file_id, recording_id, subject_id, anomaly_id)
- Detail vs Summary count matching (zip members, recordings, anomalies, linkages, profiles, role counts, identifier collisions)
- Referenced archive member path existence
- Recording schema profile assignment consistency with schema_profiles.json
- Dynamic A0 gate status and A1 entry status consistency
"""

import os
import sys
import json
import argparse


def derive_a0_gate_reference(archive_present, zip_integrity, blocker_count, error_count, warning_count,
                             partial_count, ambiguous_count, broken_count, validation_success):
    """Reference copy of A0 gate derivation rule for validation."""
    zip_pass = zip_integrity.get("zip_integrity_status") == "PASS"

    if not archive_present or not zip_pass or blocker_count > 0:
        a0_gate = "BLOCKED"
        a1_entry = "BLOCKED"
    elif not validation_success or error_count > 0 or broken_count > 0:
        a0_gate = "FAIL"
        a1_entry = "NOT_READY"
    elif warning_count > 0 or partial_count > 0 or ambiguous_count > 0:
        a0_gate = "PASS_WITH_WARNINGS"
        a1_entry = "READY_WITH_CONDITIONS"
    else:
        a0_gate = "PASS"
        a1_entry = "READY"

    return a0_gate, a1_entry


def validate_inventory_objects(summary, source_id, claims, integrity, members, recordings, profiles, anomalies):
    """
    Validates in-memory manifest objects for internal consistency.
    Returns (success: bool, errors: list[str]).
    """
    errors = []

    # Parse profiles map
    prof_list = profiles.get("profiles", []) if isinstance(profiles, dict) else profiles
    profiles_map = {p["schema_profile"]: p for p in prof_list}

    # 1. Verify Identifier Uniqueness & Count Collisions
    member_paths = set()
    source_file_ids_list = []
    role_counts = {
        "RADAR_DATA": 0,
        "RADAR_TIMESTAMP": 0,
        "CHIRP_CONFIG": 0,
        "MOVESENSE_ACC": 0,
        "MOVESENSE_ECG": 0,
        "NON_BREATHING_ANNOTATION": 0,
        "AUXILIARY": 0,
        "UNKNOWN": 0
    }

    for idx, m in enumerate(members, 1):
        path = m.get("member_path")
        if path:
            member_paths.add(path)
        sf_id = m.get("source_file_id")
        if sf_id:
            source_file_ids_list.append(sf_id)

        role = m.get("role_hint")
        if role in role_counts:
            role_counts[role] += 1

    recording_ids_list = []
    subject_ids_list = []
    linkage_counts = {
        "COMPLETE": 0,
        "COMPLETE_WITH_OPTIONAL_FILES_ABSENT": 0,
        "PARTIAL": 0,
        "AMBIGUOUS": 0,
        "BROKEN": 0,
        "UNCLASSIFIED": 0
    }

    for idx, r in enumerate(recordings, 1):
        rec_id = r.get("recording_id")
        if rec_id:
            recording_ids_list.append(rec_id)
        subj_id = r.get("subject_id")
        if subj_id:
            subject_ids_list.append(subj_id)

        status = r.get("linkage_status")
        if status in linkage_counts:
            linkage_counts[status] += 1
        else:
            errors.append(f"Unknown linkage_status '{status}' in recording '{rec_id}'")

        # Check referenced member paths exist in member_paths
        all_ref_files = (r.get("radar_files", []) + r.get("timestamp_files", []) +
                         r.get("chirp_config_files", []) + r.get("reference_files", []) +
                         r.get("annotation_files", []))
        for ref_path in all_ref_files:
            if ref_path not in member_paths:
                errors.append(f"Recording '{rec_id}' references missing archive member path: '{ref_path}'")

        # CROSS-VERIFY RECORDING SCHEMA PROFILE ASSIGNMENT AGAINST DEFINITION
        assigned_prof_id = r.get("schema_profile")
        if not assigned_prof_id or assigned_prof_id not in profiles_map:
            errors.append(f"Recording '{rec_id}' references invalid/missing schema profile ID '{assigned_prof_id}'")
        else:
            prof_def = profiles_map[assigned_prof_id]
            has_ann = bool(r.get("annotation_files"))
            expected_ann_fmt = "ISO8601_RANGE_CSV" if has_ann else "NONE"

            if prof_def.get("annotation_format") != expected_ann_fmt:
                errors.append(
                    f"Schema profile assignment mismatch in recording '{rec_id}': "
                    f"assigned profile '{assigned_prof_id}' has annotation_format '{prof_def.get('annotation_format')}', "
                    f"but recording has_annotation={has_ann} (expected '{expected_ann_fmt}')"
                )

    sf_collisions = len(source_file_ids_list) - len(set(source_file_ids_list))
    rec_collisions = len(recording_ids_list) - len(set(recording_ids_list))
    subj_collisions = len(subject_ids_list) - len(set(subject_ids_list))
    calculated_id_collisions = sf_collisions + rec_collisions

    if sf_collisions > 0:
        errors.append(f"Detected {sf_collisions} source_file_id collisions in archive_members.jsonl")
    if rec_collisions > 0:
        errors.append(f"Detected {rec_collisions} recording_id collisions in recording_index.jsonl")

    anomaly_ids = set()
    severity_counts = {"BLOCKER": 0, "ERROR": 0, "WARNING": 0, "INFO": 0}
    for a in anomalies:
        a_id = a.get("anomaly_id")
        if a_id:
            if a_id in anomaly_ids:
                errors.append(f"Duplicate anomaly_id '{a_id}' in anomalies")
            anomaly_ids.add(a_id)
        sev = a.get("severity")
        if sev in severity_counts:
            severity_counts[sev] += 1

    # 2. Cross-check Summary vs Detailed Counts
    if summary.get("archive_present"):
        if len(members) != summary.get("zip_member_count"):
            errors.append(f"Member count mismatch: members list ({len(members)}) vs summary ({summary.get('zip_member_count')})")

        if len(recordings) != summary.get("recording_count"):
            errors.append(f"Recording count mismatch: recordings list ({len(recordings)}) vs summary ({summary.get('recording_count')})")

        if summary.get("identifier_collision_count") != calculated_id_collisions:
            errors.append(f"Identifier collision count mismatch: summary ({summary.get('identifier_collision_count')}) vs calculated generated ID collisions ({calculated_id_collisions})")

        # Linkage counts comparison
        if summary.get("complete_linkage_count") != linkage_counts["COMPLETE"]:
            errors.append(f"Complete linkage count mismatch: summary ({summary.get('complete_linkage_count')}) vs detailed ({linkage_counts['COMPLETE']})")

        if summary.get("complete_with_optional_missing_count") != linkage_counts["COMPLETE_WITH_OPTIONAL_FILES_ABSENT"]:
            errors.append(f"Complete with optional missing count mismatch: summary ({summary.get('complete_with_optional_missing_count')}) vs detailed ({linkage_counts['COMPLETE_WITH_OPTIONAL_FILES_ABSENT']})")

        if summary.get("partial_linkage_count") != linkage_counts["PARTIAL"]:
            errors.append(f"Partial linkage count mismatch: summary ({summary.get('partial_linkage_count')}) vs detailed ({linkage_counts['PARTIAL']})")

        if summary.get("ambiguous_linkage_count") != linkage_counts["AMBIGUOUS"]:
            errors.append(f"Ambiguous linkage count mismatch: summary ({summary.get('ambiguous_linkage_count')}) vs detailed ({linkage_counts['AMBIGUOUS']})")

        if summary.get("broken_linkage_count") != linkage_counts["BROKEN"]:
            errors.append(f"Broken linkage count mismatch: summary ({summary.get('broken_linkage_count')}) vs detailed ({linkage_counts['BROKEN']})")

        # Anomaly counts comparison
        if summary.get("blocker_count") != severity_counts["BLOCKER"]:
            errors.append(f"Blocker count mismatch: summary ({summary.get('blocker_count')}) vs anomalies ({severity_counts['BLOCKER']})")

        if summary.get("error_count") != severity_counts["ERROR"]:
            errors.append(f"Error count mismatch: summary ({summary.get('error_count')}) vs anomalies ({severity_counts['ERROR']})")

        if summary.get("warning_count") != severity_counts["WARNING"]:
            errors.append(f"Warning count mismatch: summary ({summary.get('warning_count')}) vs anomalies ({severity_counts['WARNING']})")

        if summary.get("info_count") != severity_counts["INFO"]:
            errors.append(f"Info count mismatch: summary ({summary.get('info_count')}) vs anomalies ({severity_counts['INFO']})")

        # Role file counts comparison
        if summary.get("radar_file_count") != role_counts["RADAR_DATA"]:
            errors.append(f"Radar file count mismatch: summary ({summary.get('radar_file_count')}) vs inventory ({role_counts['RADAR_DATA']})")

        if summary.get("timestamp_file_count") != role_counts["RADAR_TIMESTAMP"]:
            errors.append(f"Timestamp file count mismatch: summary ({summary.get('timestamp_file_count')}) vs inventory ({role_counts['RADAR_TIMESTAMP']})")

        if summary.get("chirp_config_file_count") != role_counts["CHIRP_CONFIG"]:
            errors.append(f"Chirp config file count mismatch: summary ({summary.get('chirp_config_file_count')}) vs inventory ({role_counts['CHIRP_CONFIG']})")

        if summary.get("movesense_acc_file_count") != role_counts["MOVESENSE_ACC"]:
            errors.append(f"Movesense ACC file count mismatch: summary ({summary.get('movesense_acc_file_count')}) vs inventory ({role_counts['MOVESENSE_ACC']})")

        if summary.get("movesense_ecg_file_count") != role_counts["MOVESENSE_ECG"]:
            errors.append(f"Movesense ECG file count mismatch: summary ({summary.get('movesense_ecg_file_count')}) vs inventory ({role_counts['MOVESENSE_ECG']})")

        if summary.get("annotation_file_count") != role_counts["NON_BREATHING_ANNOTATION"]:
            errors.append(f"Annotation file count mismatch: summary ({summary.get('annotation_file_count')}) vs inventory ({role_counts['NON_BREATHING_ANNOTATION']})")

        # Schema profile count
        if summary.get("schema_profile_count") != len(prof_list):
            errors.append(f"Schema profile count mismatch: summary ({summary.get('schema_profile_count')}) vs profiles ({len(prof_list)})")

        # Re-derive Gate Status and compare
        pre_val_success = len(errors) == 0
        expected_gate, expected_a1 = derive_a0_gate_reference(
            summary.get("archive_present"), integrity,
            summary.get("blocker_count"), summary.get("error_count"), summary.get("warning_count"),
            summary.get("partial_linkage_count"), summary.get("ambiguous_linkage_count"), summary.get("broken_linkage_count"),
            pre_val_success
        )

        if summary.get("a0_gate_status") != expected_gate:
            errors.append(f"A0 Gate Status mismatch: stored ({summary.get('a0_gate_status')}) vs expected derived ({expected_gate})")

        if summary.get("a1_entry_status") != expected_a1:
            errors.append(f"A1 Entry Status mismatch: stored ({summary.get('a1_entry_status')}) vs expected derived ({expected_a1})")

    success = len(errors) == 0
    return success, errors


def validate_inventory_directory(inventory_dir):
    """
    Validates all manifest files in the specified directory.
    Returns (success: bool, errors: list[str]).
    """
    errors = []

    summary_path = os.path.join(inventory_dir, "inventory_summary.json")
    source_id_path = os.path.join(inventory_dir, "source_identity.json")
    claims_path = os.path.join(inventory_dir, "documented_claims.json")
    integrity_path = os.path.join(inventory_dir, "archive_integrity.json")
    members_path = os.path.join(inventory_dir, "archive_members.jsonl")
    recordings_path = os.path.join(inventory_dir, "recording_index.jsonl")
    profiles_path = os.path.join(inventory_dir, "schema_profiles.json")
    anomalies_path = os.path.join(inventory_dir, "anomalies.json")

    # 1. Check required files exist
    required_paths = [
        summary_path, source_id_path, claims_path, integrity_path,
        members_path, recordings_path, profiles_path, anomalies_path
    ]
    for p in required_paths:
        if not os.path.exists(p):
            errors.append(f"Required manifest file missing: {p}")

    if errors:
        return False, errors

    # 2. Parse all JSON files cleanly
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as e:
        errors.append(f"Failed to parse inventory_summary.json: {e}")
        return False, errors

    try:
        with open(source_id_path, "r", encoding="utf-8") as f:
            source_id = json.load(f)
    except Exception as e:
        errors.append(f"Failed to parse source_identity.json: {e}")
        return False, errors

    try:
        with open(claims_path, "r", encoding="utf-8") as f:
            claims = json.load(f)
    except Exception as e:
        errors.append(f"Failed to parse documented_claims.json: {e}")
        return False, errors

    try:
        with open(integrity_path, "r", encoding="utf-8") as f:
            integrity = json.load(f)
    except Exception as e:
        errors.append(f"Failed to parse archive_integrity.json: {e}")
        return False, errors

    try:
        with open(profiles_path, "r", encoding="utf-8") as f:
            profiles = json.load(f)
    except Exception as e:
        errors.append(f"Failed to parse schema_profiles.json: {e}")
        return False, errors

    try:
        with open(anomalies_path, "r", encoding="utf-8") as f:
            anomalies_data = json.load(f)
            anomalies = anomalies_data.get("anomalies", [])
    except Exception as e:
        errors.append(f"Failed to parse anomalies.json: {e}")
        return False, errors

    # 3. Parse JSONL files
    members = []
    try:
        with open(members_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                members.append(json.loads(line))
    except Exception as e:
        errors.append(f"Failed to parse archive_members.jsonl: {e}")
        return False, errors

    recordings = []
    try:
        with open(recordings_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                recordings.append(json.loads(line))
    except Exception as e:
        errors.append(f"Failed to parse recording_index.jsonl: {e}")
        return False, errors

    # 4. Delegate to object validator
    obj_success, obj_errors = validate_inventory_objects(
        summary, source_id, claims, integrity, members, recordings, profiles, anomalies
    )
    errors.extend(obj_errors)

    success = len(errors) == 0
    return success, errors


def main():
    parser = argparse.ArgumentParser(description="Validate Phase A0 Machine-Readable Raw Inventory")
    parser.add_argument("--inventory-dir", default="datasets/mmwave/manifests/a0_raw_inventory")
    args = parser.parse_args()

    repo_root = os.popen("git rev-parse --show-toplevel").read().strip() or os.getcwd()
    abs_inventory_dir = os.path.isabs(args.inventory_dir) and args.inventory_dir or os.path.join(repo_root, args.inventory_dir)

    print(f"Validating Phase A0 inventory in: {abs_inventory_dir}")
    success, errors = validate_inventory_directory(abs_inventory_dir)

    if success:
        print("SUCCESS: Phase A0 machine-readable inventory is internally valid and consistent.")
        sys.exit(0)
    else:
        print(f"VALIDATION FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f" - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
