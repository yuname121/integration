#!/usr/bin/env python3
"""Deterministic Phase A5 subject split and provenance construction."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import math
from typing import Any, Iterable


SPLIT_PROFILE_ID = "MMWAVE_SUBJECT_SPLIT_PROFILE_001"
SPLITS = ("TRAIN", "VALIDATION", "LOCKED_TEST")
TARGET_RATIOS = {"TRAIN": 0.70, "VALIDATION": 0.15, "LOCKED_TEST": 0.15}
SPLIT_SEED = 20260808
REQUIRED_A4_FIELDS = {
    "window_id", "recording_id", "subject_id", "mapping_type",
    "mapping_rule_id", "assignment_status", "safenest_label", "safenest_label_id",
}


class SubjectSplitError(ValueError):
    """Raised when an A5 input or invariant is invalid."""


def _scalar(value: Any) -> Any:
    return value.get("value") if isinstance(value, dict) and "value" in value else value


def deterministic_assignment_key(subject_id: str, seed: int = SPLIT_SEED) -> str:
    return hashlib.sha256(f"{seed}:{subject_id}".encode("utf-8")).hexdigest()


def calculate_split_counts(total: int, ratios: dict[str, float] | None = None) -> dict[str, int]:
    """Largest-remainder allocation; ties follow TRAIN, VALIDATION, LOCKED_TEST."""
    selected = dict(ratios or TARGET_RATIOS)
    if tuple(selected) != SPLITS or total <= 0 or not math.isclose(sum(selected.values()), 1.0):
        raise SubjectSplitError("Ratios must define TRAIN/VALIDATION/LOCKED_TEST and sum to 1")
    quotas = {name: total * selected[name] for name in SPLITS}
    counts = {name: math.floor(quotas[name]) for name in SPLITS}
    remaining = total - sum(counts.values())
    priority = {name: index for index, name in enumerate(SPLITS)}
    order = sorted(SPLITS, key=lambda name: (-(quotas[name] - counts[name]), priority[name]))
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def build_subject_catalog(recordings: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(recordings)
    recording_ids = [row.get("recording_id") for row in rows]
    if not rows:
        raise SubjectSplitError("Recording inventory is empty")
    if None in recording_ids or len(recording_ids) != len(set(recording_ids)):
        raise SubjectSplitError("A0 recording IDs must be present and unique")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("subject_id"):
            raise SubjectSplitError("Every recording must contain subject_id")
        grouped[str(row["subject_id"])].append(row)

    catalog: list[dict[str, Any]] = []
    for subject_id in sorted(grouped):
        subject_rows = sorted(grouped[subject_id], key=lambda row: row["recording_id"])
        identity_fields = ("dataset_id", "archive_id", "source_subject_id")
        if any(len({row.get(field) for row in subject_rows}) != 1 for field in identity_fields):
            raise SubjectSplitError(f"Inconsistent identity fields for {subject_id}")
        catalog.append({
            "dataset_id": subject_rows[0]["dataset_id"],
            "archive_id": subject_rows[0]["archive_id"],
            "subject_id": subject_id,
            "source_subject_id": subject_rows[0].get("source_subject_id"),
            "recording_count": len(subject_rows),
            "recording_ids": [row["recording_id"] for row in subject_rows],
            "session_ids": sorted({str(row.get("session_id")) for row in subject_rows}),
            "posture_counts": dict(sorted(Counter(str(_scalar(row.get("posture"))) for row in subject_rows).items())),
            "condition_counts": dict(sorted(Counter(str(_scalar(row.get("activity_or_test"))) for row in subject_rows).items())),
            "annotation_bearing_recording_count": sum(bool(row.get("annotation_files")) for row in subject_rows),
            "available_metadata": {
                "recording_design": True,
                "participant_demographics": False,
                "participant_metadata_status": "NOT_AVAILABLE_LOCALLY",
            },
        })
    return catalog


def measure_inventory(recordings: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Derive roster cardinalities from evidence instead of dataset constants."""
    rows = list(recordings)
    subject_ids = [row.get("subject_id") for row in rows]
    recording_ids = [row.get("recording_id") for row in rows]
    if not rows or None in subject_ids or None in recording_ids:
        raise SubjectSplitError("Inventory measurements require identified recordings")
    if len(recording_ids) != len(set(recording_ids)):
        raise SubjectSplitError("Inventory measurements require unique recording IDs")
    per_subject = Counter(subject_ids)
    distribution = Counter(per_subject.values())
    return {
        "subject_count": len(per_subject),
        "recording_count": len(rows),
        "unique_recording_id_count": len(set(recording_ids)),
        "recording_count_per_subject": {
            "minimum": min(per_subject.values()),
            "maximum": max(per_subject.values()),
            "distribution": {str(count): subjects for count, subjects in sorted(distribution.items())},
        },
    }


def assign_subject_splits(
    subject_catalog: Iterable[dict[str, Any]], seed: int = SPLIT_SEED,
    target_ratios: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    subjects = [deepcopy(row) for row in subject_catalog]
    subject_ids = [row.get("subject_id") for row in subjects]
    if None in subject_ids or len(subject_ids) != len(set(subject_ids)):
        raise SubjectSplitError("Subject catalog IDs must be present and unique")
    counts = calculate_split_counts(len(subjects), target_ratios)
    ordered = sorted(subjects, key=lambda row: (deterministic_assignment_key(row["subject_id"], seed), row["subject_id"]))
    assignment: dict[str, str] = {}
    cursor = 0
    for split in SPLITS:
        for row in ordered[cursor:cursor + counts[split]]:
            assignment[row["subject_id"]] = split
        cursor += counts[split]
    output = []
    for row in sorted(subjects, key=lambda item: item["subject_id"]):
        row.update({
            "split": assignment[row["subject_id"]],
            "split_profile_id": SPLIT_PROFILE_ID,
            "deterministic_assignment_key": deterministic_assignment_key(row["subject_id"], seed),
            "assignment_reason": "DETERMINISTIC_SHA256_SUBJECT_ALLOCATION_IDENTICAL_RECORDING_DESIGN",
            "stratification_features": {
                "posture_counts": row["posture_counts"],
                "condition_counts": row["condition_counts"],
                "annotation_bearing_recording_count": row["annotation_bearing_recording_count"],
            },
            "warnings": ["PARTICIPANT_DEMOGRAPHICS_UNAVAILABLE"],
        })
        output.append(row)
    return output


def build_recording_split_manifest(
    recordings: Iterable[dict[str, Any]], subject_manifest: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    subjects = list(subject_manifest)
    split_by_subject = {row["subject_id"]: row["split"] for row in subjects}
    if len(split_by_subject) != len(subjects) or any(value not in SPLITS for value in split_by_subject.values()):
        raise SubjectSplitError("Subject manifest is duplicate or has an invalid split")
    rows = list(recordings)
    ids = [row.get("recording_id") for row in rows]
    if None in ids or len(ids) != len(set(ids)):
        raise SubjectSplitError("Recording IDs must be present and unique")
    output = []
    for row in sorted(rows, key=lambda item: item["recording_id"]):
        subject_id = row.get("subject_id")
        if subject_id not in split_by_subject:
            raise SubjectSplitError(f"Unknown recording subject: {subject_id}")
        radar = list(row.get("radar_files", []))
        timestamps = list(row.get("timestamp_files", []))
        output.append({
            "dataset_id": row["dataset_id"], "archive_id": row["archive_id"],
            "recording_id": row["recording_id"], "subject_id": subject_id,
            "session_id": row.get("session_id"), "split": split_by_subject[subject_id],
            "split_profile_id": SPLIT_PROFILE_ID, "posture": _scalar(row.get("posture")),
            "source_test_condition": _scalar(row.get("activity_or_test")),
            "source_recording_path": row.get("source_recording_path"),
            "source_radar_member": radar[0] if len(radar) == 1 else None,
            "source_timestamp_member": timestamps[0] if len(timestamps) == 1 else None,
            "annotation_present": bool(row.get("annotation_files")),
            "schema_profile": row.get("schema_profile"), "quality_status": row.get("quality_status"),
            "data_origin_type": "REAL_HUMAN_SUBJECT_RECORDING", "synthetic": False,
        })
    return output


def attach_pilot_window_provenance(
    a4_windows: Iterable[dict[str, Any]], subject_manifest: Iterable[dict[str, Any]],
    recording_manifest: Iterable[dict[str, Any]] | None = None,
    signal_hashes: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    subjects = list(subject_manifest)
    split_by_subject = {row["subject_id"]: row["split"] for row in subjects}
    recording_by_id = {row["recording_id"]: row for row in (recording_manifest or [])}
    windows = list(a4_windows)
    ids = [row.get("window_id") for row in windows]
    if None in ids or len(ids) != len(set(ids)):
        raise SubjectSplitError("Window IDs must be present and unique")
    output = []
    for source in sorted(windows, key=lambda row: row["window_id"]):
        missing = REQUIRED_A4_FIELDS.difference(source)
        if missing:
            raise SubjectSplitError(f"A4 window {source.get('window_id')} missing {sorted(missing)}")
        subject_id = source["subject_id"]
        if subject_id not in split_by_subject:
            raise SubjectSplitError(f"Unknown window subject: {subject_id}")
        if recording_by_id and source["recording_id"] not in recording_by_id:
            raise SubjectSplitError(f"Unknown window recording: {source['recording_id']}")
        clean = source["assignment_status"] == "ASSIGNED" and source["safenest_label"] is not None
        split = split_by_subject[subject_id]
        row = deepcopy(source)
        row.update({
            "split": split, "split_profile_id": SPLIT_PROFILE_ID,
            "label_eligible": clean,
            "training_eligible": clean and split == "TRAIN",
            "validation_eligible": clean and split == "VALIDATION",
            "locked_test_evaluation_eligible": clean and split == "LOCKED_TEST",
            "supervised_training_eligible": clean and split == "TRAIN",
            "data_origin_type": "REAL_HUMAN_SUBJECT_RECORDING", "synthetic": False,
            "pilot_signal_sha256": (signal_hashes or {}).get(source["window_id"]),
            "timestamp_reference": "COMMON_ACQUISITION_COMPUTER_CLOCK",
            "source_timezone": "UNVERIFIED",
            "utc_conversion_claimed": False,
        })
        output.append(row)
    return output


def cross_split_duplicate_hashes(windows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in windows:
        if row.get("pilot_signal_sha256"):
            by_hash[row["pilot_signal_sha256"]].append(row)
    return [
        {"sha256": digest, "window_ids": sorted(row["window_id"] for row in rows),
         "splits": sorted({row["split"] for row in rows})}
        for digest, rows in sorted(by_hash.items()) if len({row["split"] for row in rows}) > 1
    ]


def summarize_counts(subjects: list[dict[str, Any]], recordings: list[dict[str, Any]], windows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "subject_counts": dict(Counter(row["split"] for row in subjects)),
        "recording_counts": dict(Counter(row["split"] for row in recordings)),
        "pilot_window_counts": dict(Counter(row["split"] for row in windows)),
        "pilot_assignment_status_counts": dict(Counter(row["assignment_status"] for row in windows)),
        "pilot_label_counts": dict(Counter(str(row["safenest_label"]) for row in windows)),
        "eligibility_counts": {
            "training_eligible": sum(row["training_eligible"] for row in windows),
            "validation_eligible": sum(row["validation_eligible"] for row in windows),
            "locked_test_evaluation_eligible": sum(row["locked_test_evaluation_eligible"] for row in windows),
        },
    }
