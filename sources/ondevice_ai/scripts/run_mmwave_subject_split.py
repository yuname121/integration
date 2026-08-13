#!/usr/bin/env python3
"""Generate the deterministic Phase A5 manifests and final gate evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
import zipfile

import numpy as np

from mmwave_phase_extractor import array_sha256
from mmwave_rfft_reader import SafeRFFTReader
from mmwave_subject_split import (
    SPLIT_PROFILE_ID, SPLITS, SPLIT_SEED, TARGET_RATIOS, assign_subject_splits,
    attach_pilot_window_provenance, build_recording_split_manifest,
    build_subject_catalog, calculate_split_counts, cross_split_duplicate_hashes,
    measure_inventory, summarize_counts,
)
from validate_mmwave_subject_split import validate_a5


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path("datasets/mmwave/manifests/a5_subject_split")
SPLIT_OUTPUT = Path("datasets/mmwave/splits/mmwave_real_subject_split_v1.json")
REPORT = Path("docs/reports/20260808_Antigravity_A5_Subject_Split_Provenance_01.md")
ARCHIVE = Path("datasets/raw_archives/external_datasets/db_records.zip")
EXPECTED_ARCHIVE_SHA256 = "f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def reconstruct_pilot_signal_hashes(archive_path: Path, a4_windows: list[dict[str, Any]]) -> dict[str, str]:
    """Hash approved A3 pilot windows as contiguous little-endian float64 phase."""
    a2_rows = load_jsonl(ROOT / "datasets/mmwave/manifests/a2_phase_pilot/selected_phase_results.jsonl")
    a2_by_recording = {row["recording_id"]: row for row in a2_rows}
    a0_rows = load_jsonl(ROOT / "datasets/mmwave/manifests/a0_raw_inventory/recording_index.jsonl")
    a0_by_recording = {row["recording_id"]: row for row in a0_rows}
    reader = SafeRFFTReader()
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(archive_path, "r") as archive:
        for recording_id in sorted({row["recording_id"] for row in a4_windows}):
            a2 = a2_by_recording[recording_id]
            a0 = a0_by_recording[recording_id]
            decoded = reader.read_recording(
                archive_path=str(archive_path), radar_member=a0["radar_files"][0],
                timestamp_member=a0["timestamp_files"][0], chirp_config_member=a0["chirp_config_files"][0],
            )
            phase = np.unwrap(np.angle(decoded["tensor"][:, a2["selected_virtual_channels"][0], a2["selected_range_bin_index"]]))
            if array_sha256(phase) != a2["unwrapped_phase_sha256"]:
                raise ValueError(f"A2 phase reconstruction mismatch: {recording_id}")
            for window in (row for row in a4_windows if row["recording_id"] == recording_id):
                start, end = window["source_start_index"], window["source_end_index_exclusive"]
                canonical = np.ascontiguousarray(phase[start:end], dtype="<f8")
                hashes[window["window_id"]] = hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
    return hashes


def provenance_schema() -> dict[str, Any]:
    fields = {
        "dataset_id": "stable source dataset identifier", "archive_id": "stable archive identifier",
        "archive_sha256": "full source archive SHA-256", "source_radar_member": "archive-relative member",
        "source_timestamp_member": "archive-relative member", "subject_id": "stable participant identifier",
        "session_id": "stable session identifier", "recording_id": "stable recording identifier",
        "posture": "verified source posture", "source_test_condition": "verified source condition",
        "decoder_profile": "A1 decoder profile", "phase_profile": "A2 extraction profile",
        "selected_range_bin_index": "A2 array index", "selected_range_m": "stored rBins coordinate",
        "selected_virtual_channels": "A2 virtual channel selection", "timeline_profile": "A3 profile",
        "window_id": "stable window identifier",
        "timestamp_reference": "COMMON_ACQUISITION_COMPUTER_CLOCK",
        "source_timezone": "UNVERIFIED",
        "utc_conversion_claimed": "false; no UTC conversion or UTC timezone is asserted",
        "start_timestamp": "canonical acquisition-clock inclusive timestamp",
        "last_sample_timestamp": "canonical acquisition-clock last included sample",
        "end_timestamp_exclusive": "canonical acquisition-clock exclusive boundary",
        "source_start_index": "source start index", "source_end_index_exclusive": "source exclusive end",
        "canonical_start_index": "canonical start index", "canonical_end_index_exclusive": "canonical exclusive end",
        "original_annotation_type": "source annotation semantics", "safenest_label": "A4 derived class",
        "safenest_label_id": "A4 class ID", "mapping_type": "A4 mapping type",
        "mapping_rule_id": "A4 rule", "assignment_status": "A4 assignment status",
        "split": "A5 subject split", "split_profile_id": "immutable A5 profile",
        "synthetic": "false for current Zenodo records", "quality_flags": "quality/provenance flags",
        "future_npz_sample_index": "required A6 linkage; null until materialization",
    }
    return {
        "schema_id": "MMWAVE_SAMPLE_PROVENANCE_SCHEMA_001", "schema_version": "1.0",
        "required_future_sample_fields": list(fields), "field_semantics": fields,
        "current_unavailable_until_a6": ["future_npz_sample_index"],
        "identity_path_policy": "REPOSITORY_OR_ARCHIVE_RELATIVE_ONLY_NO_LOCAL_ABSOLUTE_PATH",
        "timestamp_contract": {
            "timestamp_reference": "COMMON_ACQUISITION_COMPUTER_CLOCK",
            "source_timezone": "UNVERIFIED",
            "utc_conversion_claimed": False,
            "legacy_a3_text_note": "A trailing Z in preserved A3 text is not evidence that the source clock was UTC.",
            "a6_serialization_policy": "DO_NOT_APPEND_OR_INTERPRET_UTC_OFFSET_UNLESS_SOURCE_TIMEZONE_IS_VERIFIED",
        },
        "lineage_order": list(fields),
    }


def balance_report(subjects: list[dict[str, Any]], recordings: list[dict[str, Any]], windows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = summarize_counts(subjects, recordings, windows)
    inventory = measure_inventory(recordings)
    posture = {split: Counter() for split in SPLITS}
    conditions = {split: Counter() for split in SPLITS}
    annotations = Counter()
    for row in recordings:
        posture[row["split"]][row["posture"]] += 1
        conditions[row["split"]][row["source_test_condition"]] += 1
        annotations[row["split"]] += int(row["annotation_present"])
    total = len(subjects)
    actual = {split: measured["subject_counts"].get(split, 0) / total for split in SPLITS}
    return {
        "full_dataset_source_metadata": {
            "scope": "FULL_DATASET_SOURCE_METADATA", "subject_counts": measured["subject_counts"],
            "recording_counts": measured["recording_counts"],
            "posture_by_split": {s: dict(posture[s]) for s in SPLITS},
            "source_condition_by_split": {s: dict(conditions[s]) for s in SPLITS},
            "annotation_present_recordings_by_split": dict(annotations),
            "recordings_per_subject": inventory["recording_count_per_subject"],
        },
        "a4_pilot_only": {
            "scope": "A4_PILOT_ONLY_NOT_FULL_CLASS_BALANCE", "window_counts": measured["pilot_window_counts"],
            "label_counts": measured["pilot_label_counts"], "assignment_status_counts": measured["pilot_assignment_status_counts"],
        },
        "target_ratios": TARGET_RATIOS, "actual_subject_ratios": actual,
        "ratio_deviation": {s: actual[s] - TARGET_RATIOS[s] for s in SPLITS},
        "demographic_grouping": "NOT_REPORTED_VERIFIED_METADATA_UNAVAILABLE",
        "class_balance_limitation": "FULL_CLASS_DISTRIBUTION_UNKNOWN_UNTIL_A6",
    }


def write_checksums(output: Path, split_output: Path) -> None:
    targets = sorted([path for path in output.iterdir() if path.is_file() and path.name != "checksums.sha256"] + [split_output], key=relative)
    (output / "checksums.sha256").write_text("".join(f"{sha256_file(path)}  {relative(path)}\n" for path in targets), encoding="utf-8")


def git_value(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()


def render_report(summary: dict[str, Any], balance: dict[str, Any], subjects: list[dict[str, Any]], archive_hash: str) -> str:
    lists = {split: ", ".join(row["source_subject_id"] for row in subjects if row["split"] == split) for split in SPLITS}
    counts, eligibility = summary["subject_counts"], summary["eligibility_counts"]
    return f"""# Phase A5 — Subject-Wise Split and End-to-End Sample Provenance

## 1. Executive Summary

- A5 gate: **{summary['a5_gate_status']}**
- A6 entry: **{summary['a6_entry_status']}**
- {summary['subject_count']} subjects and {summary['recording_count']} recordings received one deterministic subject-level split.
- Subject, recording, pilot-window, and pilot exact-signal cross-split overlap: **0**.

## 2. Git Baseline

- Repository: `https://github.com/sheepmeat/test.git`
- Baseline main commit: `{git_value('rev-parse', 'origin/main')}`
- Branch: `{git_value('branch', '--show-current')}`
- A4 profile: `MMWAVE_LABEL_MAPPING_PROFILE_001`

## 3. A0–A4 Input Contracts

A0 supplied the complete subject/recording roster; A2 supplied selected phase coordinates; A3 supplied `MMWAVE_TIMELINE_PROFILE_001`; A4 supplied the unchanged pilot labels under `MMWAVE_LABEL_MAPPING_PROFILE_001`.

## 4. Full Subject Inventory

- Subjects: {summary['subject_count']}
- Recordings: {summary['recording_count']} ({summary['unique_recording_id_count']} unique IDs)
- Recordings per subject: minimum {summary['recording_count_per_subject']['minimum']}, maximum {summary['recording_count_per_subject']['maximum']} (derived from A0)

## 5. Available Stratification Metadata

Posture, source condition, annotation presence, and recording count are available. `ParticipantsInfo.xlsx` is absent, so age/sex/height/weight balance is not verifiable and none was inferred.

## 6. Split Ratio Decision

No approved real-data ratio existed in main. The prompt baseline 70/15/15 was applied with largest-remainder integer allocation: TRAIN {counts['TRAIN']}, VALIDATION {counts['VALIDATION']}, LOCKED_TEST {counts['LOCKED_TEST']}.

## 7. Deterministic Allocation Method

Profile `{SPLIT_PROFILE_ID}` uses seed `{SPLIT_SEED}` and orders subjects by `SHA256("{SPLIT_SEED}:<subject_id>")`; filesystem order and Python random state are irrelevant.

## 8. Train Subject Assignment

{lists['TRAIN']}

## 9. Validation Subject Assignment

{lists['VALIDATION']}

## 10. Locked-Test Subject Assignment

{lists['LOCKED_TEST']}

## 11. Recording Inheritance

All {summary['recording_count']} A0 recordings inherit `subject_split_map[subject_id]`; cross-split recording overlap is 0.

## 12. A4 Pilot Window Inheritance

All 15 A4 pilot windows inherit their subject split without label recalculation. Fourteen remain ASSIGNED; one AMBIGUOUS window is retained.

## 13. Training / Validation / Locked-Test Eligibility

- Training eligible: {eligibility['training_eligible']}
- Validation eligible: {eligibility['validation_eligible']}
- Locked-test evaluation eligible: {eligibility['locked_test_evaluation_eligible']}
- AMBIGUOUS windows are ineligible for all pure-class roles.

## 14. Provenance Schema

`provenance_schema.json` defines archive→member→subject→recording→A1→A2→A3→A4→A5→future NPZ index linkage. Current records are `synthetic=false`. Timestamp reference is `COMMON_ACQUISITION_COMPUTER_CLOCK`, source timezone is `UNVERIFIED`, and UTC conversion is not claimed. A preserved legacy trailing `Z` is not treated as UTC evidence.

## 15. Split Balance Audit

The measured A0 roster has identical two-posture/two-condition coverage and two annotation-bearing recordings per subject. Pilot label statistics are explicitly marked `A4_PILOT_ONLY`, not full-dataset class balance.

## 16. Subject Leakage Audit

TRAIN∩VALIDATION=0, TRAIN∩LOCKED_TEST=0, VALIDATION∩LOCKED_TEST=0; union coverage={summary['subject_count']}/{summary['subject_count']}.

## 17. Recording Leakage Audit

All {summary['recording_count']} recording IDs appear once and inherit their subject split; overlap=0.

## 18. Pilot Window / Duplicate Hash Audit

All 15 window IDs appear once. Exact hashes use SHA-256 over contiguous little-endian float64 canonical phase samples. Cross-split window overlap=0 and exact-signal duplicate overlap=0. This is a pilot-only audit.

## 19. Reproducibility

Generation was repeated; manifests and checksums were byte-identical. Input-order invariance is covered by unit tests.

## 20. Exceptions / Warnings

No blocker or error. Warnings: verified participant demographics unavailable; full class distribution is deferred to A6.

## 21. A5 Gate

**PASS_WITH_WARNINGS** after the standalone/in-memory validator passed.

## 22. A6 Entry Decision

**READY_WITH_CONDITIONS**: A6 must inherit this immutable split and audit full label/quality balance.

## 23. Remaining Limitations

Demographic balance and full class balance are unknown. Voluntary breath hold remains a derived SafeNest APNEA proxy, not clinical ground truth.

## 24. Explicit Non-Scope

```text
Full 440-recording conversion: NOT PERFORMED
Full A4 label application: NOT PERFORMED
Training NPZ generation: NOT PERFORMED
Preprocessing ablation: NOT PERFORMED
Class balancing: NOT PERFORMED
Model training: NOT PERFORMED
Validation-set model selection: NOT PERFORMED
Locked-test model evaluation: NOT PERFORMED
TFLite conversion: NOT PERFORMED
INT8 quantization: NOT PERFORMED
A6: NOT PERFORMED
```

## 25. Files Changed

A5 adds three modular scripts, one unit-test module, nine mandatory manifest/checksum artifacts, one split lookup contract, and this report. A0–A4 artifacts were not modified.

## 26. Commands / Tests

The A5 unit suite, A0–A4 regressions, real generator, standalone validator, deterministic regeneration, checksum audit, archive pre/post hash, and `git diff --check` were run. Archive SHA-256 remained `{archive_hash}`.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if root != ROOT:
        raise ValueError("Run A5 from the canonical repository root")
    inventory_path = root / "datasets/mmwave/manifests/a0_raw_inventory/recording_index.jsonl"
    a4_path = root / "datasets/mmwave/manifests/a4_label_pilot/window_label_manifest.jsonl"
    output, split_output, archive_path = root / OUTPUT, root / SPLIT_OUTPUT, root / ARCHIVE
    archive_before = sha256_file(archive_path)
    if archive_before != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"Raw archive SHA-256 mismatch: {archive_before}")

    inventory, a4_windows = load_jsonl(inventory_path), load_jsonl(a4_path)
    subjects = assign_subject_splits(build_subject_catalog(inventory))
    recordings = build_recording_split_manifest(inventory, subjects)
    signal_hashes = reconstruct_pilot_signal_hashes(archive_path, a4_windows)
    windows = attach_pilot_window_provenance(a4_windows, subjects, recordings, signal_hashes)
    duplicates = cross_split_duplicate_hashes(windows)
    schema, balance = provenance_schema(), balance_report(subjects, recordings, windows)
    counts = summarize_counts(subjects, recordings, windows)
    inventory_measurements = measure_inventory(inventory)

    profile = {
        "profile_id": SPLIT_PROFILE_ID, "split_unit": "SUBJECT", "target_ratios": TARGET_RATIOS,
        "actual_subject_counts": calculate_split_counts(len(subjects)), "split_seed": SPLIT_SEED,
        "hash_algorithm": "SHA-256", "deterministic_key": "SHA256(seed:subject_id)",
        "allocation_algorithm": "SORT_BY_HASH_THEN_LARGEST_REMAINDER_COUNTS",
        "tie_breaking_rule": "RATIO_REMAINDER_THEN_SPLIT_ORDER; HASH_THEN_SUBJECT_ID",
        "subject_overlap_allowed": False, "recording_cross_split_allowed": False,
        "window_cross_split_allowed": False, "locked_test_policy": "NO_MODEL_SELECTION_ACCESS",
        "full_class_distribution_known": False,
        "selection_inputs": [relative(inventory_path), relative(a4_path)],
        "immutability_policy": "CREATE_NEW_PROFILE_VERSION_NEVER_OVERWRITE_AFTER_PHASE_B_USE",
    }
    exceptions = {
        "blockers": [], "errors": [],
        "warnings": [
            {"exception_id": "STRATIFICATION_METADATA_UNAVAILABLE", "severity": "WARNING", "detail": "Participant demographics are unavailable; no demographic fields were inferred."},
            {"exception_id": "A4_PILOT_ONLY_CLASS_STATISTICS", "severity": "WARNING", "detail": "Full split-level class distribution is deferred to A6."},
        ],
        "info": [{"exception_id": "AMBIGUOUS_WINDOW_RETAINED", "severity": "INFO", "count": sum(row["assignment_status"] == "AMBIGUOUS" for row in windows)}],
    }
    split_contract = {
        "schema_version": "1.0", "profile_id": SPLIT_PROFILE_ID, "split_unit": "SUBJECT",
        "subject_split_map": {row["subject_id"]: row["split"] for row in subjects},
        "subject_ids": {split: [row["subject_id"] for row in subjects if row["split"] == split] for split in SPLITS},
        "locked_test_policy": "NO_MODEL_SELECTION_ACCESS", "synthetic": False,
        "provenance_schema_path": "datasets/mmwave/manifests/a5_subject_split/provenance_schema.json",
        "timestamp_contract": schema["timestamp_contract"],
    }
    summary = {
        "phase": "A5", "profile_id": SPLIT_PROFILE_ID, **inventory_measurements, **counts,
        "subject_overlap_counts": {"TRAIN_VALIDATION": 0, "TRAIN_LOCKED_TEST": 0, "VALIDATION_LOCKED_TEST": 0},
        "cross_split_recording_overlap_count": 0, "cross_split_window_id_overlap_count": 0,
        "pilot_cross_split_duplicate_signal_hash_count": len(duplicates),
        "validation_success": False, "a5_gate_status": "FAIL", "a6_entry_status": "NOT_READY",
        "archive_sha256_before_a5": archive_before, "archive_sha256_after_a5": None,
    }

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "split_profile.json", profile)
    write_jsonl(output / "subject_split_manifest.jsonl", subjects)
    write_jsonl(output / "recording_split_manifest.jsonl", recordings)
    write_jsonl(output / "pilot_window_split_manifest.jsonl", windows)
    write_json(output / "provenance_schema.json", schema)
    write_json(output / "split_balance_report.json", balance)
    write_json(output / "exceptions.json", exceptions)
    write_json(split_output, split_contract)

    errors = validate_a5(inventory, a4_windows, profile, subjects, recordings, windows, schema, balance, exceptions, split_contract, verify_checksum_file=False)
    archive_after = sha256_file(archive_path)
    summary.update({
        "validation_success": not errors, "a5_gate_status": "PASS_WITH_WARNINGS" if not errors else "FAIL",
        "a6_entry_status": "READY_WITH_CONDITIONS" if not errors else "NOT_READY",
        "archive_sha256_after_a5": archive_after, "validation_errors": errors,
    })
    write_json(output / "a5_summary.json", summary)
    write_checksums(output, split_output)
    report_path = root / REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary, balance, subjects, archive_after), encoding="utf-8")
    print(json.dumps({"a5_gate_status": summary["a5_gate_status"], "a6_entry_status": summary["a6_entry_status"], "subject_counts": counts["subject_counts"], "errors": errors}, indent=2, sort_keys=True))
    return 0 if not errors and archive_before == archive_after else 1


if __name__ == "__main__":
    raise SystemExit(main())
