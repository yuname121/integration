#!/usr/bin/env python3
"""Shared in-memory and standalone Phase A5 validator."""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:  # Script execution and package-style unit-test imports are both supported.
    from mmwave_subject_split import (
        REQUIRED_A4_FIELDS, SPLIT_PROFILE_ID, SPLITS, SPLIT_SEED,
        calculate_split_counts, cross_split_duplicate_hashes, measure_inventory,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by package import mode
    from scripts.mmwave_subject_split import (
        REQUIRED_A4_FIELDS, SPLIT_PROFILE_ID, SPLITS, SPLIT_SEED,
        calculate_split_counts, cross_split_duplicate_hashes, measure_inventory,
    )


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path("datasets/mmwave/manifests/a5_subject_split")
SPLIT_OUTPUT = Path("datasets/mmwave/splits/mmwave_real_subject_split_v1.json")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contains_local_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(contains_local_path(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_local_path(item) for item in value)
    return isinstance(value, str) and (value.startswith(("/Users/", "file://", "~/")) or "SafeNest_V6/ondevice_ai" in value)


def derive_gate(validation_success: bool, warnings_present: bool = True) -> tuple[str, str]:
    if not validation_success:
        return "FAIL", "NOT_READY"
    return ("PASS_WITH_WARNINGS" if warnings_present else "PASS"), ("READY_WITH_CONDITIONS" if warnings_present else "READY")


def validate_checksums(output: Path, split_output: Path) -> list[str]:
    errors: list[str] = []
    checksum_path = output / "checksums.sha256"
    expected_paths = {path.relative_to(ROOT).as_posix() for path in output.iterdir() if path.is_file() and path.name != "checksums.sha256"}
    expected_paths.add(split_output.relative_to(ROOT).as_posix())
    observed: set[str] = set()
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        observed.add(rel)
        path = ROOT / rel
        if not path.is_file() or sha256_file(path) != digest:
            errors.append(f"Checksum mismatch: {rel}")
    if observed != expected_paths:
        errors.append("Checksum coverage does not exactly match A5 artifacts")
    return errors


def validate_a5(
    inventory: list[dict[str, Any]], a4_windows: list[dict[str, Any]], profile: dict[str, Any],
    subjects: list[dict[str, Any]], recordings: list[dict[str, Any]], windows: list[dict[str, Any]],
    schema: dict[str, Any], balance: dict[str, Any], exceptions: dict[str, Any],
    split_contract: dict[str, Any], *, verify_checksum_file: bool = False,
    output: Path | None = None, split_output: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    inventory_subjects = {row["subject_id"] for row in inventory}
    inventory_recordings = {row["recording_id"] for row in inventory}
    subject_ids = [row.get("subject_id") for row in subjects]
    recording_ids = [row.get("recording_id") for row in recordings]
    window_ids = [row.get("window_id") for row in windows]

    inventory_evidence = measure_inventory(inventory)
    if len(subject_ids) != len(set(subject_ids)) or set(subject_ids) != inventory_subjects:
        errors.append("Subject manifest must uniquely and exactly cover A0 subjects")
    if len(recording_ids) != len(set(recording_ids)) or set(recording_ids) != inventory_recordings:
        errors.append("Recording manifest must uniquely and exactly cover A0 recordings")
    expected_windows = {row["window_id"] for row in a4_windows}
    if len(window_ids) != len(set(window_ids)) or set(window_ids) != expected_windows:
        errors.append("Pilot manifest must uniquely and exactly cover A4 windows")

    if profile.get("profile_id") != SPLIT_PROFILE_ID or profile.get("split_seed") != SPLIT_SEED:
        errors.append("Split profile ID or seed mismatch")
    if profile.get("split_unit") != "SUBJECT" or profile.get("locked_test_policy") != "NO_MODEL_SELECTION_ACCESS":
        errors.append("Subject split or locked-test policy mismatch")
    if any(row.get("split") not in SPLITS for row in subjects + recordings + windows):
        errors.append("Invalid split enum")
    actual_counts = Counter(row["split"] for row in subjects)
    if dict(actual_counts) != calculate_split_counts(len(inventory_subjects)):
        errors.append("Subject counts do not match deterministic 70/15/15 rounding")

    split_sets = {split: {row["subject_id"] for row in subjects if row["split"] == split} for split in SPLITS}
    if any(split_sets[left] & split_sets[right] for index, left in enumerate(SPLITS) for right in SPLITS[index + 1:]):
        errors.append("Cross-split subject leakage")
    if set().union(*split_sets.values()) != inventory_subjects:
        errors.append("Split union does not cover the full subject roster")
    subject_split = {row["subject_id"]: row["split"] for row in subjects}

    for row in recordings:
        if row["subject_id"] not in subject_split or row["split"] != subject_split.get(row["subject_id"]):
            errors.append(f"Recording inheritance mismatch: {row['recording_id']}")
        if row.get("synthetic") is not False:
            errors.append(f"Real recording not marked synthetic=false: {row['recording_id']}")

    a4_by_window = {row["window_id"]: row for row in a4_windows}
    for row in windows:
        source = a4_by_window[row["window_id"]]
        if row["subject_id"] not in subject_split or row["split"] != subject_split.get(row["subject_id"]):
            errors.append(f"Window inheritance mismatch: {row['window_id']}")
        for field in REQUIRED_A4_FIELDS:
            if row.get(field) != source.get(field):
                errors.append(f"A4 semantic field changed for {row['window_id']}: {field}")
        clean = row["assignment_status"] == "ASSIGNED" and row["safenest_label"] is not None
        expected = {
            "training_eligible": clean and row["split"] == "TRAIN",
            "validation_eligible": clean and row["split"] == "VALIDATION",
            "locked_test_evaluation_eligible": clean and row["split"] == "LOCKED_TEST",
            "supervised_training_eligible": clean and row["split"] == "TRAIN",
        }
        if any(row.get(field) != value for field, value in expected.items()):
            errors.append(f"Eligibility mismatch: {row['window_id']}")
        if row["split"] == "LOCKED_TEST" and row.get("training_eligible"):
            errors.append(f"Locked-test training leakage: {row['window_id']}")
        if row["assignment_status"] == "AMBIGUOUS" and any(row.get(field) for field in expected):
            errors.append(f"Ambiguous pure-class eligibility: {row['window_id']}")
        if not row.get("pilot_signal_sha256"):
            errors.append(f"Missing pilot signal hash: {row['window_id']}")
        if row.get("timestamp_reference") != "COMMON_ACQUISITION_COMPUTER_CLOCK":
            errors.append(f"Timestamp reference mismatch: {row['window_id']}")
        if row.get("source_timezone") != "UNVERIFIED" or row.get("utc_conversion_claimed") is not False:
            errors.append(f"Unverified source timestamp was represented as UTC: {row['window_id']}")
    if cross_split_duplicate_hashes(windows):
        errors.append("Cross-split exact pilot signal duplicate")

    required_schema = {
        "archive_sha256", "source_radar_member", "source_timestamp_member", "subject_id", "recording_id",
        "phase_profile", "selected_range_bin_index", "selected_virtual_channels", "timeline_profile",
        "window_id", "original_annotation_type", "safenest_label", "mapping_type", "assignment_status",
        "split", "split_profile_id", "synthetic", "quality_flags", "future_npz_sample_index",
        "timestamp_reference", "source_timezone", "utc_conversion_claimed",
    }
    if not required_schema.issubset(set(schema.get("required_future_sample_fields", []))):
        errors.append("Provenance schema is incomplete")
    if split_contract.get("profile_id") != SPLIT_PROFILE_ID or split_contract.get("subject_split_map") != subject_split:
        errors.append("A6 subject split lookup contract mismatch")
    if split_contract.get("timestamp_contract") != schema.get("timestamp_contract"):
        errors.append("A6 timestamp contract does not match the provenance schema")
    if balance.get("a4_pilot_only", {}).get("scope") != "A4_PILOT_ONLY_NOT_FULL_CLASS_BALANCE":
        errors.append("Pilot class statistics scope is not explicit")
    if balance.get("full_dataset_source_metadata", {}).get("recordings_per_subject") != inventory_evidence["recording_count_per_subject"]:
        errors.append("Recordings-per-subject balance was not derived from A0 evidence")
    if exceptions.get("blockers") or exceptions.get("errors"):
        errors.append("A5 exception registry contains blocking/error entries")
    if any(contains_local_path(value) for value in (profile, subjects, recordings, windows, schema, balance, exceptions, split_contract)):
        errors.append("Canonical A5 artifacts contain a local absolute/version-wrapper path")
    if verify_checksum_file:
        errors.extend(validate_checksums(output or ROOT / OUTPUT, split_output or ROOT / SPLIT_OUTPUT))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--split", type=Path, default=SPLIT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    split_output = args.split if args.split.is_absolute() else ROOT / args.split
    summary = load_json(output / "a5_summary.json")
    errors = validate_a5(
        load_jsonl(ROOT / "datasets/mmwave/manifests/a0_raw_inventory/recording_index.jsonl"),
        load_jsonl(ROOT / "datasets/mmwave/manifests/a4_label_pilot/window_label_manifest.jsonl"),
        load_json(output / "split_profile.json"), load_jsonl(output / "subject_split_manifest.jsonl"),
        load_jsonl(output / "recording_split_manifest.jsonl"), load_jsonl(output / "pilot_window_split_manifest.jsonl"),
        load_json(output / "provenance_schema.json"), load_json(output / "split_balance_report.json"),
        load_json(output / "exceptions.json"), load_json(split_output), verify_checksum_file=True,
        output=output, split_output=split_output,
    )
    if summary.get("validation_success") is not True or summary.get("a5_gate_status") != "PASS_WITH_WARNINGS" or summary.get("a6_entry_status") != "READY_WITH_CONDITIONS":
        errors.append("Final summary is not coupled to a passing A5 gate")
    gate, readiness = derive_gate(not errors)
    result = {"validation_status": "PASS" if not errors else "FAIL", "a5_gate_status": gate, "a6_entry_status": readiness, "errors": errors}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
