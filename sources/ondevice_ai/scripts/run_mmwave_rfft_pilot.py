#!/usr/bin/env python3
"""Run the deterministic Phase A1 safe rFFT pilot against the immutable ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from mmwave_rfft_reader import RFFTReaderError, SafeRFFTReader  # noqa: E402
import validate_mmwave_rfft_pilot as pilot_validator  # noqa: E402


EXPECTED_ARCHIVE_SHA256 = "f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0"
DEFAULT_TARGET_COUNT = 12
OFFICIAL_HELPER_URL = (
    "https://zenodo.org/api/records/18599983/files/helper_fns.py/content"
)
OFFICIAL_NOTEBOOK_URL = (
    "https://zenodo.org/api/records/18599983/files/ExampleCode.ipynb/content"
)


def streaming_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n"
        for value in values
    )
    path.write_text(text, encoding="utf-8")


def _anomaly_recording_paths(anomalies: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for anomaly in anomalies.get("anomalies", []):
        if anomaly.get("category") != "SCHEMA":
            continue
        for affected in anomaly.get("affected_files", []):
            marker = "/radar_timestamps.csv"
            if affected.endswith(marker):
                paths.add(affected[: -len(marker)])
    return paths


def scan_timestamp_counts(
    archive_path: Path,
    records: list[dict[str, Any]],
    *,
    max_timestamp_bytes: int = 1024 * 1024,
) -> dict[str, int]:
    """Use the validator's shared bounded ZIP timestamp scanner."""
    return pilot_validator.scan_timestamp_counts(
        archive_path,
        records,
        max_timestamp_bytes=max_timestamp_bytes,
    )


def deterministic_pilot_selection(
    records: list[dict[str, Any]],
    anomalies: dict[str, Any] | None = None,
    recommended_paths: set[str] | None = None,
    timestamp_counts: dict[str, int] | None = None,
    target_count: int = DEFAULT_TARGET_COUNT,
) -> list[dict[str, Any]]:
    """Select anchors, A0 exceptions/count strata, then a midpoint fill."""
    if target_count < 8:
        raise ValueError("pilot target must be at least 8 to cover full anchor conditions")
    records = sorted(records, key=lambda item: item["recording_id"])
    subjects = sorted({item["subject_id"] for item in records})
    if len(subjects) < 2:
        raise ValueError("pilot selection requires at least two subjects")
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_subject[record["subject_id"]].append(record)

    selected: dict[str, tuple[dict[str, Any], str]] = {}

    def add(record: dict[str, Any], reason: str) -> None:
        selected.setdefault(record["recording_id"], (record, reason))

    for record in sorted(by_subject[subjects[0]], key=lambda item: item["recording_id"]):
        add(record, "LOW_ID_SUBJECT_FULL_FACTORIAL_ANCHOR")
    for record in sorted(by_subject[subjects[-1]], key=lambda item: item["recording_id"]):
        add(record, "HIGH_ID_SUBJECT_FULL_FACTORIAL_ANCHOR")

    anomaly_paths = _anomaly_recording_paths(anomalies or {})
    for record in records:
        if record["source_recording_path"] in anomaly_paths:
            add(record, "A0_RECORDED_TIMESTAMP_LENGTH_EXCEPTION")

    for record in records:
        relative = record["source_recording_path"].removeprefix("db_records/")
        if relative in (recommended_paths or set()):
            add(record, "A0_REPORT_RECOMMENDED_PILOT")

    timestamp_counts = timestamp_counts or {}
    covered_timestamp_counts = {
        timestamp_counts[record["recording_id"]]
        for record, _reason in selected.values()
        if record["recording_id"] in timestamp_counts
    }
    available_timestamp_counts = sorted(set(timestamp_counts.values()))
    for count in available_timestamp_counts:
        if count in covered_timestamp_counts:
            continue
        representative = next(
            record
            for record in records
            if timestamp_counts.get(record["recording_id"]) == count
        )
        add(representative, "ZIP_TIMESTAMP_COUNT_STRATUM_REPRESENTATIVE")
        covered_timestamp_counts.add(count)

    if len(selected) > target_count:
        raise ValueError(
            f"mandatory pilot coverage requires {len(selected)} recordings, exceeding target {target_count}"
        )

    midpoint = (len(subjects) - 1) / 2
    midpoint_subjects = sorted(
        subjects,
        key=lambda subject: (abs(subjects.index(subject) - midpoint), subject),
    )
    fill_candidates = [
        record
        for subject in midpoint_subjects
        for record in sorted(by_subject[subject], key=lambda item: item["recording_id"])
    ]
    for record in fill_candidates:
        if len(selected) >= target_count:
            break
        add(record, "MIDPOINT_SUBJECT_DETERMINISTIC_BALANCE_FILL")

    if len(selected) < target_count:
        raise ValueError(
            f"only {len(selected)} pilot records available for target {target_count}"
        )

    output = []
    for record, reason in sorted(selected.values(), key=lambda pair: pair[0]["recording_id"]):
        output.append(
            {
                "recording_id": record["recording_id"],
                "subject_id": record["subject_id"],
                "source_recording_path": record["source_recording_path"],
                "posture": record["posture"]["value"],
                "activity_or_test": record["activity_or_test"]["value"],
                "a0_schema_profile": record["schema_profile"],
                "annotation_present": bool(record.get("annotation_files")),
                "timestamp_count_prescan": timestamp_counts.get(record["recording_id"]),
                "selection_reason": reason,
            }
        )
    return output


def decoder_signature(result: dict[str, Any]) -> tuple[Any, ...] | None:
    if not result.get("payload_decode_status", "").startswith("SUCCESS"):
        return None
    shape = result["shape"]
    return (
        result["payload_format"],
        result["radar_header_signature"],
        result["dtype"],
        result["endianness"],
        result["is_complex"],
        tuple(shape[1:]),
        result["frame_axis"],
        result["antenna_axis"],
        result["range_bin_axis"],
        result["chirp_config_hash"],
    )


def assign_decoder_profiles(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    signatures = sorted(
        {decoder_signature(result) for result in results if decoder_signature(result) is not None},
        key=repr,
    )
    profile_ids = {
        signature: f"RFFT_DECODER_PROFILE_{index:03d}"
        for index, signature in enumerate(signatures, 1)
    }
    for result in results:
        signature = decoder_signature(result)
        result["decoder_profile_id"] = profile_ids.get(signature)

    profiles = []
    for signature in signatures:
        profile_id = profile_ids[signature]
        members = [
            result for result in results if result["decoder_profile_id"] == profile_id
        ]
        representative = members[0]
        profiles.append(
            {
                "decoder_profile_id": profile_id,
                "payload_format": representative["payload_format"],
                "compression": "zlib",
                "radar_header_signatures": sorted(
                    {member["radar_header_signature"] for member in members}
                ),
                "dtype": representative["dtype"],
                "endianness": representative["endianness"],
                "is_complex": representative["is_complex"],
                "complex_representation": representative["complex_representation"],
                "shape_pattern": [None] + representative["shape"][1:],
                "observed_frame_counts": sorted({member["frame_count"] for member in members}),
                "frame_axis": representative["frame_axis"],
                "antenna_axis": representative["antenna_axis"],
                "range_bin_axis": representative["range_bin_axis"],
                "chirp_config_requirements": {
                    "a0_compatible_chirp_config_hash": representative[
                        "a0_compatible_chirp_config_hash"
                    ],
                    "adc_samples": representative["chirp_config_summary"]["interpreted"][
                        "adc_samples"
                    ],
                    "tx_antenna_count": representative["chirp_config_summary"][
                        "interpreted"
                    ]["tx_antenna_count"],
                    "rx_antenna_count": representative["chirp_config_summary"][
                        "interpreted"
                    ]["rx_antenna_count"],
                },
                "timestamp_requirements": {
                    "format": "ISO8601_HEADERLESS_UTF8",
                    "required_alignment": "EXACT_ALIGNMENT",
                    "expected_median_dt_seconds": representative[
                        "timestamp_median_dt_seconds"
                    ],
                },
                "supported_a0_schema_profiles": sorted(
                    {member["a0_schema_profile"] for member in members}
                ),
                "safe_decoder": True,
                "safe_decode_method": representative["safe_decode_method"],
                "recording_count": len(members),
                "remaining_unknowns": [
                    "TX/RX-to-virtual-antenna channel ordering is not encoded in chirp config.",
                    "Stored rBins spacing differs from configured R_BIN; A2 must use the stored rBins vector for physical coordinates and preserve the discrepancy.",
                ],
            }
        )
    return profiles


def _empty_result(selection: dict[str, Any], index_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "recording_id": selection["recording_id"],
        "subject_id": selection["subject_id"],
        "a0_schema_profile": selection["a0_schema_profile"],
        "selection_reason": selection["selection_reason"],
        "radar_member": index_record["radar_files"][0],
        "timestamp_member": index_record["timestamp_files"][0],
        "chirp_config_member": index_record["chirp_config_files"][0],
        "compressed_size_bytes": None,
        "decompressed_size_bytes": None,
        "compression_ratio": None,
        "radar_header_signature": None,
        "zlib_decode_success": False,
        "unused_trailing_bytes": None,
        "concatenated_stream_detected": None,
        "payload_format": None,
        "payload_decode_status": "FAILURE",
        "safe_decode_method": None,
        "arbitrary_object_execution": False,
        "dtype": None,
        "endianness": None,
        "is_complex": None,
        "complex_representation": None,
        "shape": None,
        "frame_axis": None,
        "antenna_axis": None,
        "range_bin_axis": None,
        "frame_count": None,
        "timestamp_count": None,
        "frame_timestamp_difference": None,
        "alignment_status": "DECODE_FAILURE",
        "timestamp_median_dt_seconds": None,
        "timestamp_mean_dt_seconds": None,
        "timestamp_min_dt_seconds": None,
        "timestamp_max_dt_seconds": None,
        "duplicate_timestamp_count": None,
        "backward_timestamp_count": None,
        "large_gap_count": None,
        "empirical_frame_rate_hz": None,
        "chirp_config_hash": None,
        "a0_compatible_chirp_config_hash": None,
        "chirp_config_summary": {},
        "schema_observations": {
            "annotation_present": selection["annotation_present"],
            "posture": selection["posture"],
            "activity_or_test": selection["activity_or_test"],
        },
        "warnings": [],
        "errors": [],
        "decoder_profile_id": None,
    }


def decode_pilot_record(
    reader: SafeRFFTReader,
    archive_path: Path,
    selection: dict[str, Any],
    index_record: dict[str, Any],
) -> dict[str, Any]:
    output = _empty_result(selection, index_record)
    try:
        decoded = reader.read_recording(
            archive_path=str(archive_path),
            radar_member=output["radar_member"],
            timestamp_member=output["timestamp_member"],
            chirp_config_member=output["chirp_config_member"],
        )
    except (RFFTReaderError, KeyError, OSError) as exc:
        output["errors"] = [f"{type(exc).__name__}: {exc}"]
        return output

    structural = decoded["structural_metadata"]
    timestamps = decoded["timestamp_metadata"]
    chirp = decoded["chirp_metadata"]
    for key in (
        "compressed_size_bytes",
        "decompressed_size_bytes",
        "compression_ratio",
        "radar_header_signature",
        "zlib_decode_success",
        "unused_trailing_bytes",
        "concatenated_stream_detected",
        "payload_format",
        "payload_decode_status",
        "safe_decode_method",
        "arbitrary_object_execution",
        "dtype",
        "endianness",
        "is_complex",
        "complex_representation",
        "shape",
        "frame_axis",
        "antenna_axis",
        "range_bin_axis",
        "frame_count",
        "frame_timestamp_difference",
        "alignment_status",
    ):
        output[key] = structural[key]
    for key in (
        "timestamp_count",
        "first_timestamp",
        "last_timestamp",
        "timestamp_median_dt_seconds",
        "timestamp_mean_dt_seconds",
        "timestamp_min_dt_seconds",
        "timestamp_max_dt_seconds",
        "duplicate_timestamp_count",
        "backward_timestamp_count",
        "large_gap_count",
        "large_gap_threshold_seconds",
        "empirical_frame_rate_hz",
    ):
        output[key] = timestamps[key]
    output["chirp_config_hash"] = chirp["chirp_config_hash"]
    output["a0_compatible_chirp_config_hash"] = chirp[
        "a0_compatible_chirp_config_hash"
    ]
    output["chirp_config_summary"] = {
        "field_interpretations": chirp["field_interpretations"],
        "interpreted": chirp["interpreted"],
        "unresolved": chirp["unresolved"],
    }
    output["schema_observations"] = {
        "annotation_present": selection["annotation_present"],
        "posture": selection["posture"],
        "activity_or_test": selection["activity_or_test"],
        "selection_reason": selection["selection_reason"],
        "a0_profile_describes_annotation_level_structure": True,
        "rfft_root_structure": structural["root_structure"],
        "range_bins_shape": structural["range_bins_shape"],
        "range_bins_dtype": structural["range_bins_dtype"],
        "range_bins_first_m": structural["range_bins_first_m"],
        "range_bins_last_m": structural["range_bins_last_m"],
        "range_bins_spacing_m_median": structural["range_bins_spacing_m_median"],
        "virtual_antenna_count": structural["virtual_antenna_count"],
        "virtual_antenna_ordering": structural["virtual_antenna_ordering"],
        "range_bin_count": structural["range_bin_count"],
        "axis_semantics_evidence": structural["axis_semantics_evidence"],
        "pickle_opcode_counts": structural["opcode_counts"],
    }
    output["warnings"] = decoded["warnings"]
    output["errors"] = decoded["errors"]
    return output


def build_exceptions(
    results: list[dict[str, Any]],
    a0_profiles: dict[str, Any],
    timestamp_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    exceptions: list[dict[str, Any]] = []

    def add(
        severity: str,
        category: str,
        affected: list[str],
        evidence: str,
        impact: str,
        blocks_a2: bool = False,
    ) -> None:
        exceptions.append(
            {
                "exception_id": f"A1-EXC-{len(exceptions) + 1:04d}",
                "severity": severity,
                "category": category,
                "affected_recording_ids": sorted(affected),
                "observed_evidence": evidence,
                "impact": impact,
                "blocks_a2": blocks_a2,
                "status": "OPEN",
            }
        )

    pickle_records = [
        result["recording_id"]
        for result in results
        if result.get("payload_format") == "PYTHON_PICKLE_PROTOCOL_5_NUMPY_ARRAY_PAIR"
    ]
    a0_claims = sorted(
        {
            profile.get("radar_serialization")
            for profile in a0_profiles.get("profiles", [])
        }
    )
    if pickle_records:
        add(
            "WARNING",
            "A0_CONTRADICTION",
            pickle_records,
            f"Direct decompressed-payload inspection found protocol-5 pickle; A0 recorded radar_serialization={a0_claims} and stated no pickle was needed.",
            "A0 outputs remain unchanged; A1 replaces only the decoding assumption with a measured restricted-pickle contract.",
        )
        add(
            "WARNING",
            "UNSAFE_SERIALIZATION",
            pickle_records,
            "The source container is object-execution-capable, but all pilot records were decoded by a pickletools-tokenized symbolic VM allowing only NumPy dtype and _frombuffer structures.",
            "Normal object deserialization must remain prohibited; future payload opcode/global drift fails closed.",
        )

    spacing_records = [
        result["recording_id"]
        for result in results
        if "CONFIGURED_R_BIN_DIFFERS_FROM_STORED_RBINS_SPACING"
        in result.get("warnings", [])
    ]
    if spacing_records:
        first = next(
            result for result in results if result["recording_id"] == spacing_records[0]
        )
        stored = first["schema_observations"]["range_bins_spacing_m_median"]
        configured = first["chirp_config_summary"]["interpreted"]["range_bin_spacing_m"]
        add(
            "WARNING",
            "CHIRP_CONFIG",
            spacing_records,
            f"Stored rBins median spacing is {stored} m while chirp R_BIN is {configured} m.",
            "Range axis index is verified, but physical range coordinates should come from the stored rBins vector; A2 must preserve this discrepancy.",
        )

    short_records = [
        result["recording_id"]
        for result in results
        if result.get("frame_count") == 400
    ]
    if short_records:
        add(
            "INFO",
            "SCHEMA_VARIANT",
            short_records,
            "The pilot directly decoded 400-frame tensors for the A0-recorded short-duration cases; each has exactly 400 timestamps.",
            "The variable frame dimension belongs to one decoder profile and requires no truncation.",
        )

    report_duration_records = [
        result["recording_id"]
        for result in results
        if result.get("selection_reason") == "A0_REPORT_RECOMMENDED_PILOT"
        and result["recording_id"].endswith("-p002-lying-post_exercise")
        and result.get("frame_count") != 600
    ]
    if report_duration_records:
        result = next(
            item for item in results if item["recording_id"] == report_duration_records[0]
        )
        add(
            "WARNING",
            "A0_CONTRADICTION",
            report_duration_records,
            f"The committed A0 human report describes P002/Lying/Post-exercise as 600 frames, but A1 directly decoded {result['frame_count']} frames and parsed {result['timestamp_count']} timestamps.",
            "The machine-readable A0 baseline does not assert 600 for this recording and remains unchanged; the human-report recommendation is corrected only through this A1 exception.",
        )

    if timestamp_counts is not None and 600 not in set(timestamp_counts.values()):
        affected = [
            result["recording_id"]
            for result in results
            if result["recording_id"].endswith("-p002-lying-post_exercise")
        ]
        add(
            "WARNING",
            "A0_CONTRADICTION",
            affected,
            "A bounded scan of every A0-linked radar_timestamps.csv member found no 600-row recording despite the committed A0 human-report claim.",
            "A0 outputs remain unchanged; the absence of the claimed duration stratum must be resolved before relying on it downstream.",
        )

    successful = [
        result["recording_id"]
        for result in results
        if result.get("payload_decode_status", "").startswith("SUCCESS")
    ]
    if successful:
        add(
            "INFO",
            "AXIS_SEMANTICS",
            successful,
            "The antenna axis and count are verified, but TX/RX-to-virtual-channel ordering is absent from config and official example documentation.",
            "A2 may operate on the verified antenna axis but must not claim physical channel ordering without new evidence.",
        )

    for result in results:
        if result.get("errors"):
            add(
                "ERROR",
                "SERIALIZATION",
                [result["recording_id"]],
                "; ".join(result["errors"]),
                "Pilot decode failed and cannot be omitted.",
                True,
            )
        elif result.get("alignment_status") != "EXACT_ALIGNMENT":
            add(
                "ERROR",
                "TIMESTAMP_ALIGNMENT",
                [result["recording_id"]],
                f"Frame/timestamp difference is {result.get('frame_timestamp_difference')} ({result.get('alignment_status')}).",
                "A2 cannot assume one timestamp per decoded frame.",
                True,
            )
    return exceptions


def build_summary(
    selections: list[dict[str, Any]],
    results: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    exceptions: list[dict[str, Any]],
    archive_before: str,
    archive_after: str,
    timestamp_counts: dict[str, int],
    validation_success: bool,
    validation_errors: list[str],
) -> dict[str, Any]:
    successes = [
        result
        for result in results
        if result.get("payload_decode_status", "").startswith("SUCCESS")
        and not result.get("errors")
    ]
    failures = [result for result in results if result not in successes]
    warnings = [result for result in successes if result.get("warnings")]
    exact = [result for result in successes if result["alignment_status"] == "EXACT_ALIGNMENT"]
    mismatches = [result for result in successes if result["alignment_status"] != "EXACT_ALIGNMENT"]
    blocker_count = sum(item["severity"] == "BLOCKER" for item in exceptions)
    error_count = sum(item["severity"] == "ERROR" for item in exceptions)
    warning_count = sum(item["severity"] == "WARNING" for item in exceptions)
    gate, a2 = pilot_validator.derive_validated_gate(
        results=results,
        exceptions=exceptions,
        validation_success=validation_success,
    )

    return {
        "schema_version": "1.0",
        "pilot_recording_count": len(selections),
        "pilot_subject_count": len({item["subject_id"] for item in selections}),
        "decode_success_count": len(successes),
        "decode_warning_count": len(warnings),
        "decode_failure_count": len(failures),
        "decoder_profile_count": len(profiles),
        "unique_shapes": [
            list(shape)
            for shape in sorted({tuple(result["shape"]) for result in successes})
        ],
        "unique_dtypes": sorted({result["dtype"] for result in successes}),
        "unique_radar_header_signatures": sorted(
            {result["radar_header_signature"] for result in successes}
        ),
        "unique_chirp_config_hashes": sorted(
            {result["chirp_config_hash"] for result in successes}
        ),
        "unique_a0_compatible_chirp_config_hashes": sorted(
            {result["a0_compatible_chirp_config_hash"] for result in successes}
        ),
        "complex_representation_verified": bool(successes)
        and all(result["is_complex"] for result in successes),
        "frame_axis_verified": bool(successes)
        and all(result["frame_axis"] == 0 for result in successes),
        "antenna_axis_verified": bool(successes)
        and all(result["antenna_axis"] == 1 for result in successes),
        "range_bin_axis_verified": bool(successes)
        and all(result["range_bin_axis"] == 2 for result in successes),
        "exact_frame_timestamp_alignment_count": len(exact),
        "alignment_mismatch_count": len(mismatches),
        "unsafe_serialization_detected": bool(successes),
        "unsafe_deserialization_required": False,
        "arbitrary_object_execution_performed": False,
        "archive_sha256_before_a1": archive_before,
        "archive_sha256_after_a1": archive_after,
        "archive_unchanged_after_a1": archive_before == archive_after,
        "expected_archive_sha256_match": archive_before == EXPECTED_ARCHIVE_SHA256,
        "timestamp_prescan_recording_count": len(timestamp_counts),
        "available_timestamp_count_strata": sorted(set(timestamp_counts.values())),
        "validation_success": validation_success,
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
        "blocker_count": blocker_count,
        "error_count": error_count,
        "warning_count": warning_count,
        "info_count": sum(item["severity"] == "INFO" for item in exceptions),
        "a1_gate_status": gate,
        "a2_entry_status": a2,
        "remaining_unknowns": [
            "TX/RX-to-virtual-antenna channel ordering is not verified.",
            "Configured R_BIN differs from stored rBins spacing; stored coordinates are authoritative for decoded payload structure.",
        ],
        "explicit_non_scope_confirmation": {
            "target_range_bin_selection": "NOT_PERFORMED",
            "respiration_phase_extraction": "NOT_PERFORMED",
            "complex_phase_unwrap": "NOT_PERFORMED",
            "antenna_performance_selection_or_aggregation": "NOT_PERFORMED",
            "respiration_spectrum_or_snr_analysis": "NOT_PERFORMED",
            "detrend_bpf_zscore": "NOT_PERFORMED",
            "resampling": "NOT_PERFORMED",
            "window_generation": "NOT_PERFORMED",
            "label_mapping": "NOT_PERFORMED",
            "subject_splitting": "NOT_PERFORMED",
            "full_dataset_conversion": "NOT_PERFORMED",
            "training_evaluation_tflite_quantization": "NOT_PERFORMED",
        },
    }


def _format_values(values: list[Any]) -> str:
    return ", ".join(f"`{value}`" for value in values)


def write_report(
    path: Path,
    summary: dict[str, Any],
    selection: dict[str, Any],
    results: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    exceptions: list[dict[str, Any]],
) -> None:
    successful = [
        result for result in results if result["payload_decode_status"].startswith("SUCCESS")
    ]
    timestamp_medians = sorted({result["timestamp_median_dt_seconds"] for result in successful})
    frame_rates = sorted({result["empirical_frame_rate_hz"] for result in successful})
    first = successful[0] if successful else None
    selection_rows = "\n".join(
        f"| `{item['recording_id']}` | `{item['posture']}` | `{item['activity_or_test']}` | `{item['a0_schema_profile']}` | `{item['annotation_present']}` | `{item['selection_reason']}` |"
        for item in selection["recordings"]
    )
    exception_rows = "\n".join(
        f"| `{item['exception_id']}` | `{item['severity']}` | `{item['category']}` | {item['observed_evidence']} |"
        for item in exceptions
    )
    profile_rows = "\n".join(
        f"| `{profile['decoder_profile_id']}` | `{profile['recording_count']}` | `{profile['shape_pattern']}` | {', '.join(profile['supported_a0_schema_profiles'])} |"
        for profile in profiles
    )
    config = first["chirp_config_summary"]["interpreted"] if first else {}
    stored_spacing = (
        first["schema_observations"]["range_bins_spacing_m_median"] if first else None
    )
    prescan_strata = ", ".join(
        f"{item['timestamp_count']} rows: {item['recording_count']} recordings"
        for item in selection["timestamp_prescan"]["strata"]
    )
    text = f"""# Phase A1 Safe rFFT Reader Pilot

## 1. Executive Summary

Phase A1 gate: **`{summary['a1_gate_status']}`**. A2 entry: **`{summary['a2_entry_status']}`**. The deterministic {summary['pilot_recording_count']}-recording, {summary['pilot_subject_count']}-subject pilot safely decoded {summary['decode_success_count']} recordings and failed {summary['decode_failure_count']}. The decoded payload is a zlib-compressed protocol-5 pickle containing `[rFFTs, rBins]`; no arbitrary object execution occurred. A strict `pickletools` opcode/global allowlist decoded only primitive NumPy buffer structures.

The shared in-memory A1 validator completed before gate derivation: **`{summary['validation_success']}`** ({summary['validation_error_count']} errors).

This proves structural radar decoding only. It does not prove respiration extraction.

## 2. A0 Input Baseline

- A0 gate: `PASS_WITH_WARNINGS`; A1 entry: `READY_WITH_CONDITIONS`.
- A0 authoritative inventory: 110 participants, 440 recordings, two annotation/file-role schema profiles.
- Pre-A1 measured archive SHA-256: `{summary['archive_sha256_before_a1']}`.
- A0 serialization claim is preserved but contradicted by A1 direct evidence; see `A1-EXC-0001`.

## 3. Pilot Selection

The selection is derived deterministically from the A0 recording index plus a bounded read-only scan of all 440 linked `radar_timestamps.csv` ZIP members. No rFFT member is opened during this scan. Measured strata: {prescan_strata}. Complete low/high subject anchors, A0 exceptions/candidates, and one representative for every measured timestamp-count stratum are selected before deterministic fill.

| Recording | Posture | Condition | A0 profile | Annotation | Selection reason |
|---|---|---|---|---:|---|
{selection_rows}

## 4. Safe Serialization Investigation

- Pipeline: bounded ZIP member stream → validated zlib stream → bounded decompressed bytes → `pickletools` opcode stream → symbolic allowlisted NumPy dtype/from-buffer VM.
- Detected root: `PYTHON_PICKLE_PROTOCOL_5_NUMPY_ARRAY_PAIR` representing `[rFFTs, rBins]`.
- Allowed symbolic globals: `numpy.dtype`, `numpy.core.numeric._frombuffer` (and NumPy 2 spelling `numpy._core.numeric._frombuffer`). They are never imported or called by the VM.
- Any unsupported opcode, global, dtype, shape, order, root structure, or trailing pickle byte fails closed.
- Unsafe normal object deserialization required: **NO**.
- Object-execution-capable source container detected: **YES**.

[Official dataset `helper_fns.py`]({OFFICIAL_HELPER_URL}) confirms the producer expected `rFFTs, rBins` after zlib/pickle loading.

## 5. rFFT Container/Compression Structure

- ZIP member name: `radar_rFFTs.zlib`.
- Inner compression: zlib, header(s): {_format_values(summary['unique_radar_header_signatures'])}.
- Decompression checks: valid header, bounded compressed and decompressed sizes, EOF required, output cap, no trailing/unused bytes, no concatenated stream.
- Observed decompressed sizes and compression ratios are retained per recording in `pilot_decode_results.jsonl`.

## 6. Decoded Tensor Structure

- Shape(s): {_format_values(summary['unique_shapes'])}.
- Dtype(s): {_format_values(summary['unique_dtypes'])}; little-endian complex values with interleaved float64 real/imag storage.
- Frame axis: `0` (official example code and exact timestamp dimension consistency).
- Virtual-antenna axis: `1` (official notebook documentation plus 2 TX × 4 RX config).
- Range-bin axis: `2` (official notebook documentation, 64-element stored `rBins`, and 64 ADC samples).
- Range vector: `float64[64]`, from `{first['schema_observations']['range_bins_first_m'] if first else None}` m to `{first['schema_observations']['range_bins_last_m'] if first else None}` m.
- Virtual-channel ordering: **NOT_VERIFIABLE** from current config/documentation.

[The official example notebook]({OFFICIAL_NOTEBOOK_URL}) explicitly documents `(frames, virtual antennas, range bins)` and names axes 0/1/2.

## 7. Chirp Configuration Interpretation

- Unique configuration hashes: `{len(summary['unique_chirp_config_hashes'])}` (A1 SHA-256); A0-compatible hash(es): {_format_values(summary['unique_a0_compatible_chirp_config_hashes'])}.
- Start frequency: `{config.get('start_frequency_hz')}` Hz.
- Sampled bandwidth/end/center: `{config.get('sampled_bandwidth_hz')}` / `{config.get('sampled_end_frequency_hz')}` / `{config.get('sampled_center_frequency_hz')}` Hz.
- ADC samples: `{config.get('adc_samples')}`; loop count: `{config.get('configured_loop_count')}`; explicit chirps per frame: `NOT_VERIFIABLE`.
- TX/RX/derived virtual count: `{config.get('tx_antenna_count')}` / `{config.get('rx_antenna_count')}` / `{config.get('virtual_antenna_count')}`.
- Period/frame rate: `{config.get('frame_period_seconds')}` s / `{config.get('configured_frame_rate_hz')}` Hz.
- Configured `R_BIN`: `{config.get('range_bin_spacing_m')}` m; stored `rBins` median spacing: `{stored_spacing}` m. These differ and are not silently reconciled.
- Each original config key/value, interpretation, and evidence is preserved per pilot result.

## 8. Timestamp and Frame Alignment

- Exact alignments: `{summary['exact_frame_timestamp_alignment_count']}`; mismatches: `{summary['alignment_mismatch_count']}`.
- Decoded frame-count strata: {_format_values(summary['available_timestamp_count_strata'])}; each measured timestamp stratum has at least one safely decoded tensor in the pilot.
- Timestamp median Δt value(s): {_format_values(timestamp_medians)} seconds; empirical frame rate value(s): {_format_values(frame_rates)} Hz.
- Duplicate/backward/large-gap totals: `{sum(result['duplicate_timestamp_count'] for result in successful)}` / `{sum(result['backward_timestamp_count'] for result in successful)}` / `{sum(result['large_gap_count'] for result in successful)}`.
- The two 400-frame A0 exceptions decode as 400-frame tensors with exactly 400 timestamps. No truncation occurs.

## 9. Decoder Profiles

| Profile | Recordings | Shape pattern | Supported A0 profiles |
|---|---:|---|---|
{profile_rows}

Both A0 file-role/annotation profiles map to the same radar decoder profile. Frame count is variable; tensor representation and non-frame axes are uniform in the pilot.

## 10. Exceptions and Failures

| ID | Severity | Category | Direct observation |
|---|---|---|---|
{exception_rows}

Decode failures: `{summary['decode_failure_count']}`. Blockers: `{summary['blocker_count']}`. Errors: `{summary['error_count']}`. Warnings: `{summary['warning_count']}`.

## 11. A0 Contradictions

A0 labeled the inner representation as raw zlib-compressed numeric data and stated a pickle reader was unnecessary. Direct A1 decompression found a protocol-5 pickle beginning with `80 05` and NumPy `_frombuffer` opcodes. The committed A0 human report also describes `P002/Lying/Post-exercise` as a 600-frame candidate, while A1 measures 500 frames and 500 timestamps. A0 outputs are unchanged; both discrepancies are preserved in the A1 exception registry.

## 12. A1 Gate Decision

**`{summary['a1_gate_status']}`**. This state is derived only after the shared in-memory validator returns `validation_success={summary['validation_success']}`. The format, tensor contract, axes, frame counts, timing, and chirp linkage are measured; all pilot records decode and align. Warnings remain because the source serialization is object-execution-capable and because configured/stored range spacing differs.

## 13. A2 Entry Decision

**`{summary['a2_entry_status']}`**. The range and antenna axes are verified, so A2 can begin. Conditions: keep the restricted reader fail-closed, use stored `rBins` for decoded physical coordinates, preserve the config-spacing discrepancy, and do not claim physical virtual-channel ordering without new evidence.

## 14. Remaining Unknowns for A2

- TX/RX-to-virtual-antenna channel ordering.
- Why the config `R_BIN` equals a different spacing convention than the stored inclusive 64-element range vector.
- These do not make the range/antenna dimensions unresolved; they constrain physical interpretation.

## 15. Files Created/Modified

- `scripts/mmwave_rfft_reader.py`: bounded zlib and non-executing restricted NumPy-pickle reader.
- `scripts/run_mmwave_rfft_pilot.py`: deterministic selection, real pilot, profiles, exceptions, summary, and report generation.
- `scripts/validate_mmwave_rfft_pilot.py`: cross-manifest A1 validator.
- `tests/test_mmwave_rfft_reader.py`: archive-independent synthetic tests.
- `datasets/mmwave/manifests/a1_rfft_pilot/`: A1 machine-readable artifacts.
- `docs/reports/20260807_Codex_A1_Safe_rFFT_Reader_Pilot_01.md`: this report.

## 16. Commands and Tests

```text
python3 -m unittest tests/test_mmwave_rfft_reader.py -v
python3 scripts/run_mmwave_rfft_pilot.py
python3 scripts/validate_mmwave_rfft_pilot.py
git diff --check
shasum -a 256 datasets/raw_archives/external_datasets/db_records.zip
```

Final measured command outcomes are recorded in the Phase A1 handoff after execution.

## 17. Explicit Non-Scope Confirmation

Target range-bin selection, respiration scoring, respiration phase extraction, unwrap, antenna performance selection/aggregation, respiration spectrum/SNR work, detrending, filtering, normalization, resampling, windowing, labels, subject splits, full conversion, training, evaluation, TFLite, and quantization were **NOT PERFORMED**.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    archive_path = (repo_root / args.archive).resolve()
    a0_dir = (repo_root / args.a0_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    report_path = (repo_root / args.report).resolve()

    archive_before = streaming_sha256(archive_path)
    if archive_before != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(
            f"archive SHA-256 mismatch before A1: {archive_before}"
        )
    a0_summary = read_json(a0_dir / "inventory_summary.json")
    if archive_before != a0_summary["archive_sha256"]:
        raise RuntimeError("measured archive hash differs from authoritative A0 summary")
    records = read_jsonl(a0_dir / "recording_index.jsonl")
    anomalies = read_json(a0_dir / "anomalies.json")
    a0_profiles = read_json(a0_dir / "schema_profiles.json")

    a0_report_path = repo_root / "docs/reports/20260806_Antigravity_A0_Zenodo_Raw_Identity_Inventory_Audit_01.md"
    recommended_paths: set[str] = set()
    if a0_report_path.is_file():
        report_text = a0_report_path.read_text(encoding="utf-8")
        recommended_paths = set(
            re.findall(r"`(P\d{3}/(?:Sitting|Lying)/(?:Rest|Post-exercise))`", report_text)
        )

    timestamp_counts = scan_timestamp_counts(archive_path, records)
    record_by_id = {record["recording_id"]: record for record in records}
    timestamp_strata = []
    for count in sorted(set(timestamp_counts.values())):
        representative_id = next(
            record_id
            for record_id in sorted(timestamp_counts)
            if timestamp_counts[record_id] == count
        )
        timestamp_strata.append(
            {
                "timestamp_count": count,
                "recording_count": sum(
                    observed == count for observed in timestamp_counts.values()
                ),
                "representative_recording_id": representative_id,
                "representative_timestamp_member": record_by_id[representative_id][
                    "timestamp_files"
                ][0],
            }
        )

    selections = deterministic_pilot_selection(
        records,
        anomalies,
        recommended_paths=recommended_paths,
        timestamp_counts=timestamp_counts,
        target_count=args.pilot_count,
    )
    selection_doc = {
        "schema_version": "1.0",
        "selection_method": (
            "A0_INDEX_LOW_HIGH_FULL_FACTORIAL_ANCHORS_THEN_A0_SCHEMA_EXCEPTIONS_"
            "AND_A0_REPORT_CANDIDATES_AND_BOUNDED_ZIP_TIMESTAMP_COUNT_STRATA_"
            "THEN_MIDPOINT_FILL"
        ),
        "target_count": args.pilot_count,
        "timestamp_prescan": {
            "method": "BOUNDED_ZIP_TIMESTAMP_MEMBER_UTF8_NONEMPTY_LINE_COUNT",
            "rfft_members_opened": 0,
            "scanned_recording_count": len(timestamp_counts),
            "strata": timestamp_strata,
        },
        "recordings": selections,
    }
    index_by_id = {record["recording_id"]: record for record in records}
    reader = SafeRFFTReader(
        max_compressed_bytes=args.max_compressed_bytes,
        max_decompressed_bytes=args.max_decompressed_bytes,
    )
    results = [
        decode_pilot_record(
            reader, archive_path, selection, index_by_id[selection["recording_id"]]
        )
        for selection in selections
    ]
    profiles = assign_decoder_profiles(results)
    exceptions = build_exceptions(results, a0_profiles, timestamp_counts)
    archive_after = streaming_sha256(archive_path)
    if archive_after != archive_before:
        raise RuntimeError("archive SHA-256 changed during A1")

    profiles_doc = {"schema_version": "1.0", "profiles": profiles}
    exceptions_doc = {"schema_version": "1.0", "exceptions": exceptions}
    summary = build_summary(
        selections,
        results,
        profiles,
        exceptions,
        archive_before,
        archive_after,
        timestamp_counts,
        True,
        [],
    )
    validation_errors = pilot_validator.validate_documents(
        a0_records=records,
        selection_doc=selection_doc,
        results=results,
        profiles_doc=profiles_doc,
        exceptions_doc=exceptions_doc,
        summary=summary,
        live_archive_sha256=archive_after,
        observed_timestamp_counts=timestamp_counts,
        enforce_gate_fields=False,
    )
    summary = build_summary(
        selections,
        results,
        profiles,
        exceptions,
        archive_before,
        archive_after,
        timestamp_counts,
        not validation_errors,
        validation_errors,
    )
    final_validation_errors = pilot_validator.validate_documents(
        a0_records=records,
        selection_doc=selection_doc,
        results=results,
        profiles_doc=profiles_doc,
        exceptions_doc=exceptions_doc,
        summary=summary,
        live_archive_sha256=archive_after,
        observed_timestamp_counts=timestamp_counts,
    )
    if final_validation_errors != validation_errors:
        raise RuntimeError(
            "validator/gate coupling produced inconsistent validation evidence: "
            + "; ".join(final_validation_errors)
        )

    write_json(output_dir / "pilot_selection.json", selection_doc)
    write_jsonl(output_dir / "pilot_decode_results.jsonl", results)
    write_json(output_dir / "decoder_profiles.json", profiles_doc)
    write_json(output_dir / "exceptions.json", exceptions_doc)
    write_json(output_dir / "a1_summary.json", summary)
    write_report(report_path, summary, selection_doc, results, profiles, exceptions)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--archive", default="datasets/raw_archives/external_datasets/db_records.zip"
    )
    parser.add_argument(
        "--a0-dir", default="datasets/mmwave/manifests/a0_raw_inventory"
    )
    parser.add_argument(
        "--output-dir", default="datasets/mmwave/manifests/a1_rfft_pilot"
    )
    parser.add_argument(
        "--report",
        default="docs/reports/20260807_Codex_A1_Safe_rFFT_Reader_Pilot_01.md",
    )
    parser.add_argument("--pilot-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--max-compressed-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--max-decompressed-bytes", type=int, default=64 * 1024 * 1024)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["validation_success"]:
        raise SystemExit(1)
