#!/usr/bin/env python3
"""Cross-validate Phase A1 pilot artifacts against the authoritative A0 index."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


def _json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_timestamp_counts(
    archive_path: Path,
    records: list[dict[str, Any]],
    *,
    max_timestamp_bytes: int = 1024 * 1024,
) -> dict[str, int]:
    """Boundedly count linked timestamp rows without opening rFFT members."""
    counts: dict[str, int] = {}
    with zipfile.ZipFile(archive_path, "r") as archive:
        for record in sorted(records, key=lambda item: item["recording_id"]):
            timestamp_members = record.get("timestamp_files", [])
            if len(timestamp_members) != 1:
                raise ValueError(
                    f"{record['recording_id']} has {len(timestamp_members)} timestamp members"
                )
            member = timestamp_members[0]
            info = archive.getinfo(member)
            if info.file_size > max_timestamp_bytes:
                raise ValueError(
                    f"timestamp member {member} exceeds {max_timestamp_bytes} bytes"
                )
            with archive.open(info, "r") as stream:
                raw = stream.read(max_timestamp_bytes + 1)
            if len(raw) > max_timestamp_bytes:
                raise ValueError(
                    f"timestamp member {member} exceeds {max_timestamp_bytes} bytes"
                )
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"timestamp member {member} is not UTF-8") from exc
            count = sum(bool(line.strip()) for line in text.splitlines())
            if count < 2:
                raise ValueError(
                    f"timestamp member {member} has only {count} non-empty rows"
                )
            counts[record["recording_id"]] = count
    if len(counts) != len(records):
        raise ValueError("timestamp pre-scan did not cover every A0 recording")
    return counts


def derive_validated_gate(
    *,
    results: list[dict[str, Any]],
    exceptions: list[dict[str, Any]],
    validation_success: bool,
) -> tuple[str, str]:
    """Derive A1/A2 state only after shared validation has run."""
    successful = [
        item
        for item in results
        if item.get("payload_decode_status", "").startswith("SUCCESS")
        and not item.get("errors")
    ]
    failures = [item for item in results if item not in successful]
    axes_verified = bool(successful) and all(
        item.get("frame_axis") == 0
        and item.get("antenna_axis") == 1
        and item.get("range_bin_axis") == 2
        for item in successful
    )
    blocker_count = sum(item.get("severity") == "BLOCKER" for item in exceptions)
    error_count = sum(item.get("severity") == "ERROR" for item in exceptions)
    warning_count = sum(item.get("severity") == "WARNING" for item in exceptions)
    if blocker_count:
        return "BLOCKED", "BLOCKED"
    if not validation_success or failures or error_count or not axes_verified:
        return "FAIL", "NOT_READY"
    if warning_count:
        return "PASS_WITH_WARNINGS", "READY_WITH_CONDITIONS"
    return "PASS", "READY"


def validate_documents(
    *,
    a0_records: list[dict[str, Any]],
    selection_doc: dict[str, Any],
    results: list[dict[str, Any]],
    profiles_doc: dict[str, Any],
    exceptions_doc: dict[str, Any],
    summary: dict[str, Any],
    live_archive_sha256: str | None = None,
    observed_timestamp_counts: dict[str, int] | None = None,
    enforce_gate_fields: bool = True,
) -> list[str]:
    """Validate fully in-memory A1 documents before a gate is finalized."""
    errors: list[str] = []
    a0_by_id = {item["recording_id"]: item for item in a0_records}
    selections = selection_doc.get("recordings", [])
    profiles = profiles_doc.get("profiles", [])
    exceptions = exceptions_doc.get("exceptions", [])

    selection_ids = [item.get("recording_id") for item in selections]
    result_ids = [item.get("recording_id") for item in results]
    if len(selection_ids) != len(set(selection_ids)):
        errors.append("pilot selection contains duplicate recording IDs")
    if len(result_ids) != len(set(result_ids)):
        errors.append("decode results contain duplicate recording IDs")
    if set(selection_ids) != set(result_ids):
        errors.append("every and only selected pilot recording must have a decode result")
    if selection_ids != sorted(selection_ids):
        errors.append("pilot selection ordering is not deterministic recording-ID order")
    if result_ids != sorted(result_ids):
        errors.append("decode result ordering is not deterministic recording-ID order")

    required_selection_fields = {
        "recording_id",
        "subject_id",
        "source_recording_path",
        "posture",
        "activity_or_test",
        "a0_schema_profile",
        "annotation_present",
        "timestamp_count_prescan",
        "selection_reason",
    }
    for item in selections:
        missing = required_selection_fields - set(item)
        if missing:
            errors.append(
                f"selection {item.get('recording_id')} missing fields: {sorted(missing)}"
            )
        source = a0_by_id.get(item.get("recording_id"))
        if source is None:
            errors.append(f"selection references unknown A0 recording: {item.get('recording_id')}")
            continue
        comparisons = {
            "subject_id": source["subject_id"],
            "source_recording_path": source["source_recording_path"],
            "posture": source["posture"]["value"],
            "activity_or_test": source["activity_or_test"]["value"],
            "a0_schema_profile": source["schema_profile"],
            "annotation_present": bool(source.get("annotation_files")),
        }
        for key, expected in comparisons.items():
            if item.get(key) != expected:
                errors.append(
                    f"selection {item['recording_id']} {key} differs from A0: "
                    f"{item.get(key)!r} != {expected!r}"
                )

    profile_ids = [profile.get("decoder_profile_id") for profile in profiles]
    if len(profile_ids) != len(set(profile_ids)):
        errors.append("decoder profile IDs are not unique")
    profile_id_set = set(profile_ids)
    valid_alignment = {
        "EXACT_ALIGNMENT",
        "OFF_BY_ONE",
        "FRAME_COUNT_MISMATCH",
        "TIMESTAMP_PARSE_FAILURE",
        "DECODE_FAILURE",
    }
    successful: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    selection_by_id = {item["recording_id"]: item for item in selections}
    for result in results:
        record_id = result.get("recording_id")
        source = a0_by_id.get(record_id)
        if source is None:
            errors.append(f"decode result references unknown A0 recording: {record_id}")
            continue
        expected_members = {
            "radar_member": source["radar_files"][0],
            "timestamp_member": source["timestamp_files"][0],
            "chirp_config_member": source["chirp_config_files"][0],
        }
        for key, expected in expected_members.items():
            if result.get(key) != expected:
                errors.append(f"{record_id} {key} does not match A0 linkage")
        status = result.get("payload_decode_status", "")
        if status.startswith("SUCCESS") and not result.get("errors"):
            successful.append(result)
            numeric_nonnegative = (
                "compressed_size_bytes",
                "decompressed_size_bytes",
                "compression_ratio",
                "frame_count",
                "timestamp_count",
                "duplicate_timestamp_count",
                "backward_timestamp_count",
                "large_gap_count",
            )
            for key in numeric_nonnegative:
                value = result.get(key)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                    errors.append(f"{record_id} has invalid numeric {key}: {value!r}")
            for key in ("frame_axis", "antenna_axis", "range_bin_axis"):
                if not isinstance(result.get(key), int):
                    errors.append(f"{record_id} unresolved axis is not explicit integer: {key}")
            shape = result.get("shape")
            if not (
                isinstance(shape, list)
                and len(shape) == 3
                and all(isinstance(value, int) and value > 0 for value in shape)
            ):
                errors.append(f"{record_id} has invalid shape: {shape!r}")
            else:
                if result["frame_count"] != shape[result["frame_axis"]]:
                    errors.append(f"{record_id} frame count does not match decoded shape")
            expected_difference = result["frame_count"] - result["timestamp_count"]
            if result.get("frame_timestamp_difference") != expected_difference:
                errors.append(f"{record_id} frame/timestamp difference is inconsistent")
            selected_prescan = selection_by_id.get(record_id, {}).get(
                "timestamp_count_prescan"
            )
            if result.get("timestamp_count") != selected_prescan:
                errors.append(
                    f"{record_id} decoded timestamp count differs from bounded pre-scan"
                )
            if result.get("decoder_profile_id") not in profile_id_set:
                errors.append(f"{record_id} references missing decoder profile")
            if result.get("arbitrary_object_execution") is not False:
                errors.append(f"{record_id} does not explicitly prohibit object execution")
        else:
            failures.append(result)
            if not result.get("errors"):
                errors.append(f"failed result {record_id} has no preserved error")
        if result.get("alignment_status") not in valid_alignment:
            errors.append(f"{record_id} has invalid alignment status")

    profile_counts = Counter(
        result.get("decoder_profile_id") for result in successful
    )
    for profile in profiles:
        profile_id = profile["decoder_profile_id"]
        if profile.get("recording_count") != profile_counts[profile_id]:
            errors.append(f"profile {profile_id} recording count mismatch")
        supported = sorted(
            {
                result["a0_schema_profile"]
                for result in successful
                if result["decoder_profile_id"] == profile_id
            }
        )
        if profile.get("supported_a0_schema_profiles") != supported:
            errors.append(f"profile {profile_id} A0-profile mapping mismatch")
        if profile.get("safe_decoder") is not True:
            errors.append(f"profile {profile_id} is not explicitly safe")
        if "remaining_unknowns" not in profile:
            errors.append(f"profile {profile_id} omits remaining_unknowns")

    prescan = selection_doc.get("timestamp_prescan", {})
    strata = prescan.get("strata", [])
    if prescan.get("scanned_recording_count") != len(a0_records):
        errors.append("timestamp pre-scan does not cover every A0 recording")
    available_counts = {
        item.get("timestamp_count")
        for item in strata
        if isinstance(item.get("timestamp_count"), int)
    }
    if sum(item.get("recording_count", 0) for item in strata) != len(a0_records):
        errors.append("timestamp pre-scan stratum totals do not match A0 recording count")
    if observed_timestamp_counts is not None:
        if set(observed_timestamp_counts) != set(a0_by_id):
            errors.append("live timestamp pre-scan IDs do not match the A0 index")
        live_frequency = Counter(observed_timestamp_counts.values())
        documented_frequency = {
            item.get("timestamp_count"): item.get("recording_count") for item in strata
        }
        if dict(sorted(live_frequency.items())) != documented_frequency:
            errors.append("timestamp pre-scan strata differ from live bounded ZIP scan")
        for item in selections:
            if item.get("timestamp_count_prescan") != observed_timestamp_counts.get(
                item["recording_id"]
            ):
                errors.append(
                    f"{item['recording_id']} selected timestamp count differs from live scan"
                )
    selected_prescan_counts = {
        item.get("timestamp_count_prescan") for item in selections
    }
    if not available_counts.issubset(selected_prescan_counts):
        errors.append("pilot selection does not cover every timestamp-count stratum")
    decoded_frame_counts = {item.get("frame_count") for item in successful}
    if not available_counts.issubset(decoded_frame_counts):
        errors.append("pilot decode does not prove every timestamp-count stratum")
    required_known_strata = {400, 500, 600}.intersection(available_counts)
    if not required_known_strata.issubset(decoded_frame_counts):
        errors.append("pilot decode omits an available 400/500/600-frame stratum")

    expected_counts = {
        "pilot_recording_count": len(selections),
        "pilot_subject_count": len({item["subject_id"] for item in selections}),
        "decode_success_count": len(successful),
        "decode_warning_count": sum(bool(item.get("warnings")) for item in successful),
        "decode_failure_count": len(failures),
        "decoder_profile_count": len(profiles),
        "exact_frame_timestamp_alignment_count": sum(
            item.get("alignment_status") == "EXACT_ALIGNMENT" for item in successful
        ),
        "alignment_mismatch_count": sum(
            item.get("alignment_status") != "EXACT_ALIGNMENT" for item in successful
        ),
        "blocker_count": sum(item.get("severity") == "BLOCKER" for item in exceptions),
        "error_count": sum(item.get("severity") == "ERROR" for item in exceptions),
        "warning_count": sum(item.get("severity") == "WARNING" for item in exceptions),
        "info_count": sum(item.get("severity") == "INFO" for item in exceptions),
    }
    for key, expected in expected_counts.items():
        if summary.get(key) != expected:
            errors.append(f"summary {key} mismatch: {summary.get(key)!r} != {expected!r}")
    expected_structural_summary = {
        "unique_shapes": [
            list(shape) for shape in sorted({tuple(item["shape"]) for item in successful})
        ],
        "unique_dtypes": sorted({item["dtype"] for item in successful}),
        "unique_radar_header_signatures": sorted(
            {item["radar_header_signature"] for item in successful}
        ),
        "complex_representation_verified": bool(successful)
        and all(item["is_complex"] for item in successful),
        "frame_axis_verified": bool(successful)
        and all(item["frame_axis"] == 0 for item in successful),
        "antenna_axis_verified": bool(successful)
        and all(item["antenna_axis"] == 1 for item in successful),
        "range_bin_axis_verified": bool(successful)
        and all(item["range_bin_axis"] == 2 for item in successful),
        "timestamp_prescan_recording_count": len(a0_records),
        "available_timestamp_count_strata": sorted(available_counts),
    }
    for key, expected in expected_structural_summary.items():
        if summary.get(key) != expected:
            errors.append(f"summary {key} mismatch: {summary.get(key)!r} != {expected!r}")

    selected_profiles = {item["a0_schema_profile"] for item in selections}
    if selected_profiles != {"SCHEMA_PROFILE_001", "SCHEMA_PROFILE_002"}:
        errors.append("pilot does not cover both A0 schema profiles")
    if {item["posture"] for item in selections} != {"Lying", "Sitting"}:
        errors.append("pilot does not cover both postures")
    if {item["activity_or_test"] for item in selections} != {"Rest", "Post-exercise"}:
        errors.append("pilot does not cover rest and post-exercise")
    if {item["annotation_present"] for item in selections} != {False, True}:
        errors.append("pilot does not cover annotation presence and absence")

    if summary.get("archive_unchanged_after_a1") is not True:
        errors.append("summary does not confirm archive immutability")
    if summary.get("arbitrary_object_execution_performed") is not False:
        errors.append("summary does not confirm zero arbitrary object execution")
    if summary.get("unsafe_deserialization_required") is not False:
        errors.append("summary claims unsafe deserialization is required")
    if live_archive_sha256 is not None:
        if live_archive_sha256 != summary.get("archive_sha256_before_a1"):
            errors.append("live archive hash differs from A1 pre-hash")
        if live_archive_sha256 != summary.get("archive_sha256_after_a1"):
            errors.append("live archive hash differs from A1 post-hash")

    valid_severity = {"INFO", "WARNING", "ERROR", "BLOCKER"}
    valid_category = {
        "DECOMPRESSION",
        "SERIALIZATION",
        "UNSAFE_SERIALIZATION",
        "ARRAY_SHAPE",
        "DTYPE",
        "COMPLEX_ENCODING",
        "AXIS_SEMANTICS",
        "CHIRP_CONFIG",
        "FRAME_COUNT",
        "TIMESTAMP_ALIGNMENT",
        "SCHEMA_VARIANT",
        "A0_CONTRADICTION",
        "UNKNOWN",
    }
    exception_ids = [item.get("exception_id") for item in exceptions]
    if len(exception_ids) != len(set(exception_ids)):
        errors.append("exception IDs are not unique")
    for item in exceptions:
        if item.get("severity") not in valid_severity:
            errors.append(f"invalid exception severity: {item.get('severity')}")
        if item.get("category") not in valid_category:
            errors.append(f"invalid exception category: {item.get('category')}")
        unknown = set(item.get("affected_recording_ids", [])) - set(result_ids)
        if unknown:
            errors.append(f"exception {item.get('exception_id')} references unknown pilots")

    if enforce_gate_fields:
        content_errors = list(errors)
        validation_success = not content_errors
        if summary.get("validation_success") is not validation_success:
            errors.append(
                "summary validation_success does not match in-memory validator result"
            )
        if summary.get("validation_error_count") != len(content_errors):
            errors.append("summary validation_error_count mismatch")
        if summary.get("validation_errors") != content_errors:
            errors.append("summary validation_errors do not match validator output")
        expected_a1, expected_a2 = derive_validated_gate(
            results=results,
            exceptions=exceptions,
            validation_success=validation_success,
        )
        if summary.get("a1_gate_status") != expected_a1:
            errors.append("A1 gate is not derived from validator result")
        if summary.get("a2_entry_status") != expected_a2:
            errors.append("A2 entry status is not derived from validator result")

    return errors


def validate(
    *,
    a0_dir: Path,
    a1_dir: Path,
    archive_path: Path | None = None,
) -> list[str]:
    """Load committed artifacts and run the same validator used by the runner."""
    errors: list[str] = []
    required = {
        "pilot_selection.json",
        "pilot_decode_results.jsonl",
        "decoder_profiles.json",
        "exceptions.json",
        "a1_summary.json",
    }
    for name in sorted(required):
        if not (a1_dir / name).is_file():
            errors.append(f"missing A1 artifact: {name}")
    if errors:
        return errors
    a0_records = _jsonl(a0_dir / "recording_index.jsonl")
    observed_timestamp_counts = None
    if archive_path is not None:
        try:
            observed_timestamp_counts = scan_timestamp_counts(archive_path, a0_records)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            return [f"live bounded timestamp pre-scan failed: {exc}"]
    return validate_documents(
        a0_records=a0_records,
        selection_doc=_json(a1_dir / "pilot_selection.json"),
        results=_jsonl(a1_dir / "pilot_decode_results.jsonl"),
        profiles_doc=_json(a1_dir / "decoder_profiles.json"),
        exceptions_doc=_json(a1_dir / "exceptions.json"),
        summary=_json(a1_dir / "a1_summary.json"),
        live_archive_sha256=_sha256(archive_path) if archive_path is not None else None,
        observed_timestamp_counts=observed_timestamp_counts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--a0-dir",
        type=Path,
        default=root / "datasets/mmwave/manifests/a0_raw_inventory",
    )
    parser.add_argument(
        "--a1-dir",
        type=Path,
        default=root / "datasets/mmwave/manifests/a1_rfft_pilot",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=root / "datasets/raw_archives/external_datasets/db_records.zip",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    problems = validate(
        a0_dir=args.a0_dir.resolve(),
        a1_dir=args.a1_dir.resolve(),
        archive_path=args.archive.resolve(),
    )
    if problems:
        print("A1 manifest validation: FAIL")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)
    print("A1 manifest validation: PASS")
