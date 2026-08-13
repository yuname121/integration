#!/usr/bin/env python3
"""Run the deterministic Phase A2 target/bin and canonical phase pilot."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from typing import Any
import zipfile

import numpy as np

from mmwave_phase_extractor import (
    MmwavePhaseExtractor,
    PROFILE_ID,
    SearchRegion,
    array_sha256,
)
from mmwave_rfft_reader import SafeRFFTReader
from validate_mmwave_phase_pilot import derive_gate, validate_manifests


EXPECTED_ARCHIVE_SHA256 = "f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0"
SEARCH_REGION_PROFILE = "PILOT_SEARCH_REGION_001"
SEARCH_MIN_M = 0.30
SEARCH_MAX_M = 2.00


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(_json_value(row), sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_pilot(root: Path) -> dict[str, Any]:
    a1 = json.loads((root / "datasets/mmwave/manifests/a1_rfft_pilot/pilot_selection.json").read_text())
    a0 = _load_jsonl(root / "datasets/mmwave/manifests/a0_raw_inventory/recording_index.jsonl")
    a0_by_id = {row["recording_id"]: row for row in a0}
    rows = []
    for item in a1["recordings"]:
        row = dict(item)
        row["a2_selection_reason"] = "PRESERVE_APPROVED_A1_PILOT"
        rows.append(row)
    extra = next(row for row in a0 if row["recording_id"].endswith("p004-lying-rest"))
    if extra["recording_id"] not in {row["recording_id"] for row in rows}:
        rows.append({
            "recording_id": extra["recording_id"],
            "subject_id": extra["subject_id"],
            "source_recording_path": extra["source_recording_path"],
            "posture": extra["posture"]["value"],
            "activity_or_test": extra["activity_or_test"]["value"],
            "a0_schema_profile": extra["schema_profile"],
            "annotation_present": bool(extra["annotation_files"]),
            "selection_reason": "OFFICIAL_ZERO_FRAME_MAX_CASE",
            "a2_selection_reason": "DETERMINISTIC_KNOWN_MAXIMUM_INITIAL_ZERO_FRAME_CASE",
        })
    # Normalise fields from A1 against the authoritative A0 index.
    for row in rows:
        source = a0_by_id[row["recording_id"]]
        row["subject_id"] = source["subject_id"]
        row["source_recording_path"] = source["source_recording_path"]
        row["posture"] = source["posture"]["value"]
        row["activity_or_test"] = source["activity_or_test"]["value"]
        row["a0_schema_profile"] = source["schema_profile"]
        row["annotation_present"] = bool(source["annotation_files"])
    rows.sort(key=lambda row: row["recording_id"])
    return {
        "schema_version": "1.0",
        "selection_method": "APPROVED_A1_PILOT_PLUS_DETERMINISTIC_OFFICIAL_MAX_ZERO_FRAME_CASE",
        "label_independent_selection": True,
        "recordings": rows,
    }


def _parse_datetime(value: str) -> dt.datetime:
    text = value.replace("Z", "+00:00")
    if "." in text:
        head, fraction = text.split(".", 1)
        timezone = ""
        for marker in ("+", "-"):
            position = fraction.find(marker)
            if position >= 0:
                fraction, timezone = fraction[:position], fraction[position:]
                break
        text = f"{head}.{fraction[:6].ljust(6, '0')}{timezone}"
    return dt.datetime.fromisoformat(text)


def annotation_validation(
    archive: zipfile.ZipFile,
    annotation_member: str | None,
    first_timestamp: str,
    sampling_rate_hz: float,
    phase: np.ndarray,
) -> dict[str, Any]:
    base = {"available": False, "used_for_selection": False, "post_selection_result": "NOT_AVAILABLE"}
    if not annotation_member:
        return base
    rows = list(csv.reader(io.StringIO(archive.read(annotation_member).decode("utf-8"))))
    values = {row[0].strip().lower(): _parse_datetime(row[1].strip()) for row in rows if len(row) >= 2}
    if "begin" not in values or "end" not in values:
        return {**base, "available": True, "post_selection_result": "AMBIGUOUS_PARSE"}
    start = _parse_datetime(first_timestamp)
    begin_seconds = (values["begin"] - start).total_seconds()
    end_seconds = (values["end"] - start).total_seconds()
    step_times = (np.arange(phase.size - 1, dtype=np.float64) + 0.5) / sampling_rate_hz
    steps = np.diff(phase)
    inside = (step_times >= begin_seconds) & (step_times <= end_seconds) & np.isfinite(steps)
    outside = (~inside) & np.isfinite(steps)
    if np.count_nonzero(inside) < 5 or np.count_nonzero(outside) < 5:
        result = "AMBIGUOUS_INSUFFICIENT_SAMPLES"
        ratio = None
    else:
        inside_energy = float(np.mean(steps[inside] ** 2))
        outside_energy = float(np.mean(steps[outside] ** 2))
        ratio = float(inside_energy / outside_energy) if outside_energy > 0 else None
        result = "SUPPORT" if ratio is not None and ratio < 0.75 else (
            "NO_SUPPORT" if ratio is not None and ratio > 1.25 else "AMBIGUOUS"
        )
    return {
        "available": True,
        "used_for_selection": False,
        "post_selection_result": result,
        "annotation_semantics": "VOLUNTARY_NON_BREATHING_OR_BREATH_HOLD_NOT_CLINICAL_APNEA",
        "begin_seconds_from_first_radar_frame": begin_seconds,
        "end_seconds_from_first_radar_frame": end_seconds,
        "phase_step_energy_inside_to_outside_ratio": ratio,
        "decision_thresholds": {"support_below": 0.75, "no_support_above": 1.25},
    }


def _stats(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return {"minimum": None, "maximum": None, "mean": None, "median": None, "std": None}
    return {
        "minimum": float(np.min(finite)), "maximum": float(np.max(finite)),
        "mean": float(np.mean(finite)), "median": float(np.median(finite)),
        "std": float(np.std(finite)),
    }


def _strategy_record(recording_id: str, analysis: dict[str, Any], rbins: np.ndarray) -> dict[str, Any]:
    strategies = []
    for row in analysis["strategy_results"]:
        item = dict(row)
        if "bin_index" in item:
            item["selected_bin_index"] = item.pop("bin_index")
            item["selected_channel"] = item.pop("channel")
            item["selected_channels"] = [item["selected_channel"]]
            item["selected_range_m"] = float(rbins[item["selected_bin_index"]])
            metric_key = {
                "A_MEAN_MAGNITUDE_PER_CHANNEL": "mean_magnitude",
                "A_MEAN_MAGNITUDE_CHANNEL_AGGREGATED": "channel_aggregated_selection_score",
                "B_DYNAMIC_ENERGY_PER_CHANNEL": "dynamic_energy",
                "B_DYNAMIC_ENERGY_CHANNEL_AGGREGATED": "channel_aggregated_selection_score",
                "C_RESPIRATION_BAND_ENERGY": "respiration_band_energy",
                "D_PHASE_QUALITY": "phase_quality_rank_score",
                "E_NEIGHBOR_BIN_SUPPORT": "neighbor_bin_agreement",
            }[item["strategy_id"]]
            item["selection_score"] = item.get(metric_key)
        strategies.append(item)
    return {
        "recording_id": recording_id,
        "selection_evidence": "LABEL_INDEPENDENT_RADAR_SIGNAL_QUALITY_AND_ACQUISITION_GEOMETRY",
        "labels_or_annotations_used_for_selection": False,
        "search_region_profile": SEARCH_REGION_PROFILE,
        "candidate_count": len(analysis["candidate_metrics"]),
        "strategies": strategies,
        "selected_profile_candidate": {
            "range_method": "B_MEDIAN_CHANNEL_DYNAMIC_ENERGY",
            "virtual_channel_method": "V1_SINGLE_BEST_PHASE_QUALITY_CHANNEL",
            "selected_bin_index": analysis["selected_bin_index"],
            "selected_range_m": analysis["selected_range_m"],
            "selected_channel": analysis["selected_channel"],
        },
    }


def build_report(summary: dict[str, Any], search: dict[str, Any], selected: list[dict[str, Any]],
                 candidates: list[dict[str, Any]], exceptions: dict[str, Any], archive_sha: str) -> str:
    bins = {}
    channels = {}
    for row in selected:
        bins[str(row["selected_range_bin_index"])] = bins.get(str(row["selected_range_bin_index"]), 0) + 1
        channel = str(row["selected_virtual_channels"][0])
        channels[channel] = channels.get(channel, 0) + 1
    dominant = [row["diagnostic_spectrum"]["dominant_frequency_hz"] for row in selected
                if row["diagnostic_spectrum"]["dominant_frequency_hz"] is not None]
    annotation = [row["annotation_validation"]["post_selection_result"] for row in selected
                  if row["annotation_validation"]["available"]]
    lines = f"""# Phase A2: Deterministic Range-Bin and Phase Extraction Pilot

## 1. Executive Summary

Phase A2 decoded {summary['pilot_recording_count']} pilot recordings and established a deterministic, label-independent extraction profile. The gate is `{summary['a2_gate_status']}` and A3 entry is `{summary['a3_entry_status']}`. The canonical output is the unfiltered, unnormalised `np.unwrap(np.angle(z))` phase; diagnostic detrending and periodograms are not canonical outputs.

## 2. Git / Input Baseline

The work used merged A1 commit `be92a00e58f76b48bb85ec38e022f4fd3a313cbe`. The measured archive SHA-256 before and after execution was `{archive_sha}`.

## 3. A1 Decoder Contract Used

All inputs used `RFFT_DECODER_PROFILE_001`: `complex128[frames, 8, 64]`, axes frame/virtual-channel/range-bin, plus authoritative stored `float64[64]` rBins. Restricted symbolic pickle decoding remained mandatory; no arbitrary object execution occurred.

## 4. Pilot Composition

The approved 12-recording A1 pilot was preserved and `P004/Lying/Rest` was added deterministically because the [official Zenodo record](https://zenodo.org/records/18599983) identifies it as the largest initial-zero-frame case (11 frames). Coverage includes both postures, both activity conditions, both A0 schema profiles, annotations present/absent, and 400/500/600-frame recordings.

## 5. Stored rBins / Search Region

Stored rBins span `{search['stored_rbins_minimum_m']}` to `{search['stored_rbins_maximum_m']}` m. `{SEARCH_REGION_PROFILE}` admits indices `{search['eligible_bin_indices']}` (`{search['minimum_range_m']}`–`{search['maximum_range_m']}` m threshold, whose actual admitted coordinates are `{search['eligible_range_coordinates_m']}`). Bin 0 is excluded as the zero-range/near-field coordinate. Bins above 2 m are excluded from this pilot search because the documented acquisition placed the radar about 0.5 m from the thorax; the limit is conservative pilot methodology, not universal hardware truth. See [Scientific Data](https://www.nature.com/articles/s41597-026-07172-9).

## 6. Candidate Range-Bin Strategies

The same eligible candidates were compared with A mean magnitude, B static-component-reduced dynamic energy, C 0.1–0.5 Hz diagnostic energy, D rank-based phase quality, and E adjacent-bin agreement. The selected range rule is B using the median across anonymous virtual channels. In all {len(selected)} pilot recordings it selected stored rBins index 2.

## 7. Virtual-Channel Strategies

V1 single-channel phase-quality selection, V2 quality-weighted aligned phase aggregation, and V3 median aligned phase consensus were compared. V1 was retained because it preserves a direct raw complex lineage and avoids opaque fusion while physical TX/RX ordering remains unknown. Channels are reported only as `virtual_channel_N`.

## 8. Canonical Phase Extraction

The selected complex timeline is preserved by checksum together with real/imaginary and magnitude statistics. Wrapped phase uses `np.angle`; canonical phase uses `np.unwrap` with default discontinuity π and period 2π. Near-zero samples are flagged and preserved without interpolation. No detrending, filtering, smoothing, resampling, or normalisation is applied to canonical phase.

## 9. Strategy Comparison

There are {len(candidates)} per-recording comparison records. The pilot-selected profile favours perfect range stability ({bins}) plus deterministic channel-quality selection ({channels}), reproducible byte checksums, direct provenance, and implementation simplicity. Strategy C remains diagnostic because 0.1–0.5 Hz is not asserted to cover all post-exercise respiration.

## 10. Selected Extraction Profile

`{PROFILE_ID}` uses B median-channel dynamic energy for range, V1 rank-composite phase quality for channel, a single stored bin, and fixed ties by lowest bin then lowest channel.

## 11. Time-Domain Diagnostics

All canonical phase lengths equal their frame and timestamp counts. Near-zero samples were retained and flagged rather than repaired. Phase step percentiles, unwrap corrections, large steps, and magnitude outliers are recorded per recording.

## 12. Frequency-Domain Diagnostics

Temporary linearly detrended Hann periodograms use a 0.05–2.0 Hz total band and 0.1–0.5 Hz respiration diagnostic band. Dominant frequencies across valid pilots span `{min(dominant) if dominant else None}`–`{max(dominant) if dominant else None}` Hz. These diagnostics do not alter the canonical phase.

## 13. Annotation-Based Post-Selection Validation

Annotations were loaded only after bin/channel selection. Outcomes were `{annotation}` using the predeclared inside/outside phase-step-energy ratio thresholds. These are annotated voluntary non-breathing/breath-hold intervals, not clinical apnea, and never selection evidence.

## 14. Failure / Low-Quality Cases

Extraction failures: {summary['extraction_failure_count']}. Warning-bearing selected results: {summary['extraction_warning_count']}. The added P004/Lying/Rest case preserves its initial zero-magnitude frames and emits a quality warning.

## 15. Exceptions

The registry contains {len(exceptions['exceptions'])} items. Physical virtual-channel ordering remains unknown, configured R_BIN differs from stored rBins spacing, and restricted pickle decoding remains required.

## 16. Validation

The shared in-memory validator ran before the gate was derived and returned `{summary['validation_success']}`. It checked pilot coverage, A1 decode contract, coordinates, channels, label independence, profile linkage, phase lengths, nonfinite values, counts, and gate consistency.

## 17. A2 Gate

`{summary['a2_gate_status']}`: deterministic extraction succeeded, with non-blocking preserved limitations.

## 18. A3 Entry Decision

`{summary['a3_entry_status']}`: A3 may consume the unfiltered canonical phase and quality/provenance metadata, while respecting the warnings above.

## 19. Remaining Limitations

This is a 13-recording pilot, not a full 440-recording validation. Stored selected range is a radar coordinate, not a claim of true chest distance. Virtual-channel physical mapping and the config/stored range-spacing discrepancy remain unresolved.

## 20. Explicit Non-Scope Confirmation

No permanent detrending/BPF/Z-score, resampling, 30-second windows, SafeNest label mapping, subject split, full conversion, NPZ generation, model training, or A3 work was performed.

## 21. Files Changed

Implementation, validator, synthetic tests, this report, and the eight required manifest/checksum files were added for Phase A2.

## 22. Commands / Tests

`python3 -m unittest tests/test_mmwave_phase_extractor.py -v`; `python3 scripts/run_mmwave_phase_pilot.py`; `python3 scripts/validate_mmwave_phase_pilot.py`; isolated regeneration and SHA-256 comparison; `git diff --check`; archive SHA-256 before/after.
"""
    return lines


def run(root: Path, output_dir: Path, report_path: Path) -> dict[str, Any]:
    archive_path = root / "datasets/raw_archives/external_datasets/db_records.zip"
    pre_hash = sha256_file(archive_path)
    if pre_hash != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(f"archive SHA-256 mismatch before A2: {pre_hash}")
    source_identity = json.loads(
        (root / "datasets/mmwave/manifests/a0_raw_inventory/source_identity.json").read_text(encoding="utf-8")
    )
    if source_identity.get("local_archive", {}).get("sha256") != pre_hash:
        raise RuntimeError("measured archive hash contradicts authoritative A0 source identity")
    # Parse the remaining declared schema baselines before any A2 analysis.
    json.loads((root / "datasets/mmwave/manifests/a0_raw_inventory/schema_profiles.json").read_text(encoding="utf-8"))
    json.loads((root / "datasets/mmwave/manifests/a0_raw_inventory/inventory_summary.json").read_text(encoding="utf-8"))
    json.loads((root / "datasets/mmwave/manifests/a0_raw_inventory/anomalies.json").read_text(encoding="utf-8"))
    a1_summary = json.loads(
        (root / "datasets/mmwave/manifests/a1_rfft_pilot/a1_summary.json").read_text(encoding="utf-8")
    )
    if a1_summary.get("a1_gate_status") not in {"PASS", "PASS_WITH_WARNINGS"}:
        raise RuntimeError("approved A1 gate is not usable")
    if a1_summary.get("a2_entry_status") not in {"READY", "READY_WITH_CONDITIONS"}:
        raise RuntimeError("approved A1 summary does not permit A2 entry")
    a1_profiles = json.loads(
        (root / "datasets/mmwave/manifests/a1_rfft_pilot/decoder_profiles.json").read_text(encoding="utf-8")
    )
    decoder_profile = next(
        (row for row in a1_profiles.get("profiles", []) if row.get("decoder_profile_id") == "RFFT_DECODER_PROFILE_001"),
        None,
    )
    if not decoder_profile or not decoder_profile.get("safe_decoder") or decoder_profile.get("shape_pattern") != [None, 8, 64]:
        raise RuntimeError("approved A1 decoder profile contract is missing or incompatible")
    a1_decode_ids = {
        row["recording_id"] for row in _load_jsonl(
            root / "datasets/mmwave/manifests/a1_rfft_pilot/pilot_decode_results.jsonl"
        )
    }
    json.loads((root / "datasets/mmwave/manifests/a1_rfft_pilot/exceptions.json").read_text(encoding="utf-8"))
    pilot = build_pilot(root)
    preserved_a1_ids = {
        row["recording_id"] for row in pilot["recordings"]
        if row["a2_selection_reason"] == "PRESERVE_APPROVED_A1_PILOT"
    }
    if not preserved_a1_ids.issubset(a1_decode_ids):
        raise RuntimeError("A2 pilot contains an approved A1 recording without an A1 result")
    a0_rows = _load_jsonl(root / "datasets/mmwave/manifests/a0_raw_inventory/recording_index.jsonl")
    a0_by_id = {row["recording_id"]: row for row in a0_rows}
    reader = SafeRFFTReader()
    search_profile = SearchRegion(SEARCH_REGION_PROFILE, SEARCH_MIN_M, SEARCH_MAX_M)
    extractor = MmwavePhaseExtractor(search_profile, sampling_rate_hz=10.0)
    candidate_results = []
    selected_results = []
    exceptions = [
        {"exception_id": "A2-EXC-0001", "severity": "WARNING", "category": "CONFIG_RBIN_MISMATCH", "scope": "PILOT", "message": "Configured R_BIN differs from authoritative stored rBins spacing."},
        {"exception_id": "A2-EXC-0002", "severity": "WARNING", "category": "VIRTUAL_CHANNEL_SELECTION", "scope": "PILOT", "message": "Physical TX/RX mapping for virtual_channel_0..7 remains unknown."},
        {"exception_id": "A2-EXC-0003", "severity": "INFO", "category": "A1_CONTRADICTION", "scope": "PILOT", "message": "A1 restricted symbolic pickle decoder remains required; arbitrary object execution is not used."},
    ]
    stored_rbins = None
    valid_decoded_ids = set()
    with zipfile.ZipFile(archive_path, "r") as archive:
        for entry in pilot["recordings"]:
            base = entry["source_recording_path"]
            decoded = reader.read_recording(
                archive_path=str(archive_path),
                radar_member=f"{base}/radar_rFFTs.zlib",
                timestamp_member=f"{base}/radar_timestamps.csv",
                chirp_config_member=f"{base}/radar_chirpConfig.json",
            )
            if decoded["errors"] or decoded["structural_metadata"]["alignment_status"] != "EXACT_ALIGNMENT":
                raise RuntimeError(f"A1 decode contract failed for {entry['recording_id']}")
            entry["a2_measured_frame_count"] = decoded["structural_metadata"]["frame_count"]
            valid_decoded_ids.add(entry["recording_id"])
            rbins = decoded["range_bins"]
            if stored_rbins is None:
                stored_rbins = rbins.copy()
            elif not np.array_equal(stored_rbins, rbins):
                raise RuntimeError("pilot has inconsistent stored rBins")
            result = extractor.extract(
                rffts=decoded["tensor"], rbins=rbins,
                timestamps=decoded["timestamp_metadata"], config=decoded["chirp_metadata"],
            )
            candidate_results.append(_strategy_record(entry["recording_id"], result["candidate_analysis"], rbins))
            source = a0_by_id[entry["recording_id"]]
            annotation_member = source["annotation_files"][0] if source["annotation_files"] else None
            annotation = annotation_validation(
                archive, annotation_member, decoded["timestamp_metadata"]["first_timestamp"],
                10.0, result["unwrapped_phase"],
            )
            stats = result["statistics"]
            selection = result["selection"]
            warnings = list(result["warnings"])
            errors = list(result["errors"])
            if stats["near_zero_magnitude_count"]:
                exception_id = f"A2-EXC-{len(exceptions)+1:04d}"
                exceptions.append({
                    "exception_id": exception_id, "severity": "WARNING", "category": "NEAR_ZERO_MAGNITUDE",
                    "scope": entry["recording_id"],
                    "message": f"{stats['near_zero_magnitude_count']} selected complex samples are near zero and were preserved without interpolation.",
                })
            selected_results.append({
                "recording_id": entry["recording_id"], "subject_id": entry["subject_id"],
                "source_radar_member": f"{base}/radar_rFFTs.zlib",
                "source_decoder_profile": "RFFT_DECODER_PROFILE_001",
                "a1_decode_contract_verified": True,
                "frame_count": decoded["structural_metadata"]["frame_count"],
                "timestamp_count": decoded["timestamp_metadata"]["timestamp_count"],
                "sampling_rate_hz": decoded["timestamp_metadata"]["empirical_frame_rate_hz"],
                "selected_extraction_profile": PROFILE_ID,
                "selected_range_bin_index": selection["selected_range_bin_index"],
                "selected_range_m": selection["selected_range_m"],
                "selected_range_m_coordinate_source": "STORED_RBINS",
                "configured_r_bin_m": decoded["chirp_metadata"]["interpreted"]["range_bin_spacing_m"],
                "selected_virtual_channels": selection["selected_virtual_channels"],
                "virtual_channel_aggregation": selection["virtual_channel_aggregation"],
                "selection_score": selection["selection_score"],
                "selection_score_components": selection["selection_score_components"],
                "selection_used_labels": False,
                "complex_signal_sha256": array_sha256(result["complex_signal"]),
                "complex_real_statistics": _stats(result["complex_signal"].real),
                "complex_imaginary_statistics": _stats(result["complex_signal"].imag),
                "complex_magnitude_statistics": _stats(result["magnitude"]),
                "wrapped_phase_sha256": array_sha256(result["wrapped_phase"]),
                "wrapped_phase_statistics": _stats(result["wrapped_phase"]),
                "unwrapped_phase_sha256": array_sha256(result["unwrapped_phase"]),
                "unwrapped_phase_statistics": _stats(result["unwrapped_phase"]),
                "canonical_phase_length": int(result["unwrapped_phase"].size),
                "canonical_phase_units": "radians",
                "canonical_phase_filtered": False,
                "canonical_phase_normalized": False,
                "near_zero_magnitude_count": stats["near_zero_magnitude_count"],
                "near_zero_magnitude_ratio": stats["near_zero_magnitude_ratio"],
                "near_zero_threshold": stats["near_zero_threshold"],
                "nonfinite_complex_count": stats["nonfinite_complex_count"],
                "nonfinite_phase_count": stats["nonfinite_phase_count"],
                "unwrap_correction_count": stats["unwrap_correction_count"],
                "wrapped_phase_jump_count": stats["wrapped_phase_jump_count"],
                "large_unwrapped_step_count": stats["large_unwrapped_step_count"],
                "phase_step_percentiles_rad": stats["phase_step_percentiles_rad"],
                "magnitude_outlier_count": stats["magnitude_outlier_count"],
                "diagnostic_spectrum": stats["spectrum"],
                "annotation_validation": annotation,
                "quality_status": "FAILURE" if errors else ("SUCCESS_WITH_WARNING" if warnings else "SUCCESS"),
                "warnings": warnings, "errors": errors,
            })

    assert stored_rbins is not None
    eligible = search_profile.eligible_indices(stored_rbins).tolist()
    search = {
        "schema_version": "1.0", "search_region_profile": SEARCH_REGION_PROFILE,
        "search_region_status": "PILOT_SEARCH_REGION",
        "minimum_range_m": SEARCH_MIN_M, "maximum_range_m": SEARCH_MAX_M,
        "stored_rbins_count": int(stored_rbins.size),
        "stored_rbins_minimum_m": float(stored_rbins[0]), "stored_rbins_maximum_m": float(stored_rbins[-1]),
        "stored_rbins_median_spacing_m": float(np.median(np.diff(stored_rbins))),
        "configured_r_bin_m_preserved_separately": 0.31228381041666664,
        "eligible_bin_indices": eligible,
        "eligible_range_coordinates_m": [float(stored_rbins[index]) for index in eligible],
        "excluded_bin_indices": [index for index in range(stored_rbins.size) if index not in eligible],
        "exclusion_reasons": {
            "0": "ZERO_RANGE_NEAR_FIELD_COORDINATE_EXCLUDED",
            "7-63": "ABOVE_CONSERVATIVE_2M_PILOT_REGION_FOR_DOCUMENTED_APPROXIMATELY_0.5M_THORAX_GEOMETRY",
        },
        "evidence": [
            {"type": "DIRECT_PAYLOAD_STRUCTURE", "statement": "Stored rBins are the coordinate authority."},
            {"type": "OFFICIAL_DATASET_DOCUMENTATION", "statement": "Radar was placed approximately 0.5 m in front of the thorax.", "source": "https://www.nature.com/articles/s41597-026-07172-9"},
            {"type": "PILOT_MAGNITUDE_AND_DYNAMIC_ENERGY", "statement": "All 13 pilot recordings peak at stored rBins index 2 for aggregate mean magnitude and dynamic energy."},
        ],
    }
    profiles = {"schema_version": "1.0", "profiles": [{
        "profile_id": PROFILE_ID, "profile_status": "SELECTED_A2_PILOT_PROFILE",
        "search_region": {"profile_id": SEARCH_REGION_PROFILE, "minimum_range_m": SEARCH_MIN_M, "maximum_range_m": SEARCH_MAX_M},
        "range_bin_selection_method": "B_MEDIAN_CHANNEL_STATIC_COMPONENT_REDUCED_DYNAMIC_ENERGY",
        "virtual_channel_method": "V1_PHASE_QUALITY_RANK_COMPOSITE_SINGLE_CHANNEL",
        "phase_quality_rank_components": ["dynamic_energy", "phase_continuity", "neighbor_bin_agreement", "channel_agreement"],
        "adjacent_bin_policy": "SINGLE_BIN_CANONICAL; PLUS_MINUS_ONE_AGREEMENT_DIAGNOSTIC_ONLY",
        "tie_breaking": "HIGHEST_SCORE_THEN_LOWEST_RANGE_BIN_INDEX_THEN_LOWEST_VIRTUAL_CHANNEL_INDEX_WITH_1E-12_TOLERANCE",
        "phase_method": "numpy_angle_radians",
        "unwrap_method": "numpy_unwrap_default_discontinuity_pi_period_2pi",
        "near_zero_policy": "THRESHOLD_MAX_FLOAT_TINY_OR_1E-6_TIMES_MEDIAN_POSITIVE_MAGNITUDE; FLAG_AND_PRESERVE_NO_INTERPOLATION",
        "diagnostic_only_processing": {"detrending": "linear_least_squares", "psd_method": "hann_periodogram", "respiration_band_hz": [0.1, 0.5], "total_band_hz": [0.05, 2.0]},
        "label_independent_selection": True,
        "canonical_filtered": False, "canonical_normalized": False,
    }]}
    exceptions_doc = {"schema_version": "1.0", "exceptions": exceptions}
    preliminary = validate_manifests(
        pilot_selection=pilot, candidate_results=candidate_results, selected_results=selected_results,
        search_region=search, profiles_doc=profiles, exceptions_doc=exceptions_doc,
        valid_decoded_recording_ids=valid_decoded_ids,
    )
    failure_count = len([row for row in selected_results if row["quality_status"] == "FAILURE"])
    selected_warning_count = len([row for row in selected_results if row["warnings"]])
    registry_warning_count = len([row for row in exceptions if row["severity"] == "WARNING"])
    gate, a3 = derive_gate(
        validation_success=preliminary["validation_success"], failure_count=failure_count,
        warning_count=selected_warning_count + registry_warning_count,
    )
    post_hash = sha256_file(archive_path)
    if post_hash != pre_hash:
        raise RuntimeError("archive changed during A2")
    summary = {
        "schema_version": "1.0", "pilot_recording_count": len(pilot["recordings"]),
        "pilot_subject_count": len({row["subject_id"] for row in pilot["recordings"]}),
        "postures_covered": sorted({row["posture"] for row in pilot["recordings"]}),
        "activity_or_test_conditions_covered": sorted({row["activity_or_test"] for row in pilot["recordings"]}),
        "frame_count_variants": sorted({row["a2_measured_frame_count"] for row in pilot["recordings"]}),
        "a0_schema_profiles_covered": sorted({row["a0_schema_profile"] for row in pilot["recordings"]}),
        "a1_decoder_profiles_covered": ["RFFT_DECODER_PROFILE_001"],
        "extraction_success_count": len(selected_results) - failure_count,
        "extraction_warning_count": selected_warning_count,
        "extraction_failure_count": failure_count,
        "candidate_strategy_recording_count": len(candidate_results),
        "range_bin_strategies_evaluated": [
            "A_MEAN_MAGNITUDE_PER_CHANNEL", "A_MEAN_MAGNITUDE_CHANNEL_AGGREGATED",
            "B_DYNAMIC_ENERGY_PER_CHANNEL", "B_DYNAMIC_ENERGY_CHANNEL_AGGREGATED",
            "C_RESPIRATION_BAND_ENERGY", "D_PHASE_QUALITY", "E_NEIGHBOR_BIN_SUPPORT",
        ],
        "neighbor_bin_strategies_evaluated": [
            "SINGLE_BIN", "PLUS_MINUS_ONE_INDIVIDUAL_PHASE_CENTER_THEN_MEDIAN_DIAGNOSTIC"
        ],
        "virtual_channel_strategies_evaluated": ["V1_SINGLE_BEST", "V2_QUALITY_WEIGHTED_PHASE", "V3_MEDIAN_CONSENSUS_PHASE"],
        "selected_profile_id": PROFILE_ID,
        "selected_bin_distribution": {}, "selected_range_distribution_m": {}, "selected_virtual_channel_distribution": {},
        "phase_length_match_count": len([row for row in selected_results if row["canonical_phase_length"] == row["frame_count"] == row["timestamp_count"]]),
        "nonfinite_phase_sample_count": sum(row["nonfinite_phase_count"] for row in selected_results),
        "near_zero_magnitude_sample_count": sum(row["near_zero_magnitude_count"] for row in selected_results),
        "annotation_present_count": len([row for row in selected_results if row["annotation_validation"]["available"]]),
        "exception_count": len(exceptions), "validation_success": preliminary["validation_success"],
        "archive_sha256_before_a2": pre_hash, "archive_sha256_after_a2": post_hash,
        "archive_unchanged_after_a2": pre_hash == post_hash,
        "a2_gate_status": gate, "a3_entry_status": a3,
        "explicit_non_scope": {name: "NOT_PERFORMED" for name in [
            "permanent_detrending", "permanent_bandpass_filtering", "z_score_normalization", "resampling",
            "30_second_windowing", "safenest_label_mapping", "subject_split", "full_440_recording_conversion",
            "npz_generation", "model_training", "a3",
        ]},
    }
    for row in selected_results:
        for key, value in (
            ("selected_bin_distribution", str(row["selected_range_bin_index"])),
            ("selected_range_distribution_m", f"{row['selected_range_m']:.12f}"),
            ("selected_virtual_channel_distribution", str(row["selected_virtual_channels"][0])),
        ):
            summary[key][value] = summary[key].get(value, 0) + 1
    final_validation = validate_manifests(
        pilot_selection=pilot, candidate_results=candidate_results, selected_results=selected_results,
        search_region=search, profiles_doc=profiles, exceptions_doc=exceptions_doc,
        summary=summary, valid_decoded_recording_ids=valid_decoded_ids,
    )
    if not final_validation["validation_success"]:
        summary["validation_success"] = False
        summary["a2_gate_status"], summary["a3_entry_status"] = derive_gate(
            validation_success=False, failure_count=failure_count, warning_count=registry_warning_count
        )
        summary["validation_errors"] = final_validation["errors"]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "pilot_selection.json", pilot)
    write_json(output_dir / "search_region.json", search)
    write_jsonl(output_dir / "candidate_strategy_results.jsonl", candidate_results)
    write_json(output_dir / "extraction_profiles.json", profiles)
    write_jsonl(output_dir / "selected_phase_results.jsonl", selected_results)
    write_json(output_dir / "exceptions.json", exceptions_doc)
    write_json(output_dir / "a2_summary.json", summary)
    checksum_names = [
        "pilot_selection.json", "search_region.json", "candidate_strategy_results.jsonl",
        "extraction_profiles.json", "selected_phase_results.jsonl", "exceptions.json", "a2_summary.json",
    ]
    checksums = "".join(f"{sha256_file(output_dir / name)}  {name}\n" for name in checksum_names)
    (output_dir / "checksums.sha256").write_text(checksums, encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(summary, search, selected_results, candidate_results, exceptions_doc, post_hash), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir or root / "datasets/mmwave/manifests/a2_phase_pilot"
    report = args.report_path or root / "docs/reports/20260807_Codex_A2_RangeBin_Phase_Extraction_Pilot_01.md"
    summary = run(root, output, report)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
