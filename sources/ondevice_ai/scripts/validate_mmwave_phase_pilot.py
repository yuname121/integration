#!/usr/bin/env python3
"""Shared in-memory and standalone validator for Phase A2 manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def derive_gate(
    *, validation_success: bool, failure_count: int, warning_count: int, blocked: bool = False
) -> tuple[str, str]:
    if blocked:
        return "BLOCKED", "BLOCKED"
    if not validation_success or failure_count:
        return "FAIL", "NOT_READY"
    if warning_count:
        return "PASS_WITH_WARNINGS", "READY_WITH_CONDITIONS"
    return "PASS", "READY"


def validate_manifests(
    *,
    pilot_selection: dict[str, Any],
    candidate_results: list[dict[str, Any]],
    selected_results: list[dict[str, Any]],
    search_region: dict[str, Any],
    profiles_doc: dict[str, Any],
    exceptions_doc: dict[str, Any],
    summary: dict[str, Any] | None = None,
    valid_decoded_recording_ids: set[str] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    pilot = pilot_selection.get("recordings", [])
    pilot_ids = [row.get("recording_id") for row in pilot]
    candidate_ids = [row.get("recording_id") for row in candidate_results]
    selected_ids = [row.get("recording_id") for row in selected_results]
    if len(pilot_ids) != len(set(pilot_ids)):
        errors.append("DUPLICATE_PILOT_RECORDING_ID")
    if len(candidate_ids) != len(set(candidate_ids)):
        errors.append("DUPLICATE_CANDIDATE_RECORDING_ID")
    if len(selected_ids) != len(set(selected_ids)):
        errors.append("DUPLICATE_SELECTED_RECORDING_ID")
    if set(candidate_ids) != set(pilot_ids):
        errors.append("CANDIDATE_RESULTS_DO_NOT_COVER_PILOT")
    if set(selected_ids) != set(pilot_ids):
        errors.append("SELECTED_RESULTS_DO_NOT_COVER_PILOT")

    profile_ids = {row.get("profile_id") for row in profiles_doc.get("profiles", [])}
    eligible = set(search_region.get("eligible_bin_indices", []))
    stored_count = int(search_region.get("stored_rbins_count", 0))
    if not eligible or any(not isinstance(index, int) for index in eligible):
        errors.append("INVALID_SEARCH_REGION")
    for result in selected_results:
        rid = result.get("recording_id", "UNKNOWN")
        if valid_decoded_recording_ids is not None and rid not in valid_decoded_recording_ids:
            errors.append(f"{rid}:NOT_A1_DECODED")
        selected_bin = result.get("selected_range_bin_index")
        if not isinstance(selected_bin, int) or selected_bin < 0 or selected_bin >= stored_count:
            errors.append(f"{rid}:SELECTED_BIN_OUT_OF_STORED_RBINS")
        elif selected_bin not in eligible:
            errors.append(f"{rid}:SELECTED_BIN_OUTSIDE_SEARCH_REGION")
        channels = result.get("selected_virtual_channels", [])
        if not channels or any(not isinstance(channel, int) or channel < 0 or channel > 7 for channel in channels):
            errors.append(f"{rid}:INVALID_VIRTUAL_CHANNEL")
        if result.get("selection_used_labels") is not False:
            errors.append(f"{rid}:LABEL_INDEPENDENCE_NOT_PROVEN")
        if result.get("selected_extraction_profile") not in profile_ids:
            errors.append(f"{rid}:UNKNOWN_EXTRACTION_PROFILE")
        phase_length = result.get("canonical_phase_length")
        if phase_length != result.get("frame_count") or phase_length != result.get("timestamp_count"):
            errors.append(f"{rid}:CANONICAL_PHASE_LENGTH_MISMATCH")
        nonfinite = result.get("nonfinite_phase_count")
        if not isinstance(nonfinite, int) or nonfinite < 0:
            errors.append(f"{rid}:INVALID_NONFINITE_COUNT")
        elif nonfinite and "NONFINITE_CANONICAL_SIGNAL" not in result.get("errors", []):
            errors.append(f"{rid}:UNEXPLAINED_NONFINITE_PHASE")
        quality = result.get("quality_status")
        if quality == "FAILURE" and not result.get("errors"):
            errors.append(f"{rid}:FAILURE_WITHOUT_ERROR")
        if quality != "FAILURE" and result.get("errors"):
            errors.append(f"{rid}:ERROR_WITH_NONFAILURE_STATUS")

    if len(candidate_results) != len(selected_results):
        errors.append("STRATEGY_RECORD_COUNT_MISMATCH")
    exception_rows = exceptions_doc.get("exceptions", [])
    if summary is not None:
        if summary.get("pilot_recording_count") != len(pilot):
            errors.append("SUMMARY_PILOT_COUNT_MISMATCH")
        if summary.get("candidate_strategy_recording_count") != len(candidate_results):
            errors.append("SUMMARY_STRATEGY_COUNT_MISMATCH")
        if summary.get("exception_count") != len(exception_rows):
            errors.append("SUMMARY_EXCEPTION_COUNT_MISMATCH")
        expected_success = len([row for row in selected_results if row.get("quality_status") != "FAILURE"])
        if summary.get("extraction_success_count") != expected_success:
            errors.append("SUMMARY_SUCCESS_COUNT_MISMATCH")

    success = not errors
    if summary is not None:
        failure_count = len([row for row in selected_results if row.get("quality_status") == "FAILURE"])
        warning_count = len([row for row in selected_results if row.get("warnings")]) + len(
            [row for row in exception_rows if row.get("severity") == "WARNING"]
        )
        expected_gate, expected_a3 = derive_gate(
            validation_success=success, failure_count=failure_count, warning_count=warning_count
        )
        if summary.get("validation_success") != success:
            errors.append("SUMMARY_VALIDATION_STATUS_MISMATCH")
        if summary.get("a2_gate_status") != expected_gate:
            errors.append("SUMMARY_GATE_INCONSISTENT_WITH_VALIDATION")
        if summary.get("a3_entry_status") != expected_a3:
            errors.append("SUMMARY_A3_STATUS_INCONSISTENT_WITH_VALIDATION")
        success = not errors

    return {
        "schema_version": "1.0",
        "validation_success": success,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "validated_pilot_count": len(pilot),
        "validated_candidate_count": len(candidate_results),
        "validated_selected_count": len(selected_results),
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir", type=Path,
        default=Path("datasets/mmwave/manifests/a2_phase_pilot")
    )
    args = parser.parse_args()
    directory = args.manifest_dir
    selected = _read_jsonl(directory / "selected_phase_results.jsonl")
    result = validate_manifests(
        pilot_selection=_read_json(directory / "pilot_selection.json"),
        candidate_results=_read_jsonl(directory / "candidate_strategy_results.jsonl"),
        selected_results=selected,
        search_region=_read_json(directory / "search_region.json"),
        profiles_doc=_read_json(directory / "extraction_profiles.json"),
        exceptions_doc=_read_json(directory / "exceptions.json"),
        summary=_read_json(directory / "a2_summary.json"),
        valid_decoded_recording_ids={row["recording_id"] for row in selected if row.get("a1_decode_contract_verified")},
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["validation_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
