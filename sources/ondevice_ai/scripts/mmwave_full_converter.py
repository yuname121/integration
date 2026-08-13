#!/usr/bin/env python3
"""SafeNest Phase A6 — Full mmWave Real-Data Conversion & Provenance Module.

This module orchestrates the full conversion of all recordings in the authoritative
A0 raw inventory through the approved Phase A1–A5 contracts into deterministic
canonical phase signals, machine-readable provenance records, and immutable split inheritance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence
import zipfile

import numpy as np

from mmwave_label_mapper import (
    LabelMappingProfile,
    extract_movesense_respiration_rate,
    map_window_label,
    parse_annotation_file,
)
from mmwave_phase_extractor import (
    PROFILE_ID as PHASE_EXTRACTION_PROFILE_ID,
    MmwavePhaseExtractor,
    SearchRegion,
)
from mmwave_rfft_reader import SafeRFFTReader
from mmwave_timeline import TimelineProfile, process_recording_timeline

PROFILE_ID = "MMWAVE_FULL_CONVERSION_PROFILE_001"
SEARCH_REGION_ID = "MMWAVE_SEARCH_REGION_001"


class FullConversionError(Exception):
    """Raised when full conversion or provenance assembly fails."""


@dataclass(frozen=True)
class FullConversionProfile:
    """Deterministic configuration for Phase A6 full conversion."""

    profile_id: str = PROFILE_ID
    a1_decoder_profile: str = "RFFT_DECODER_PROFILE_001"
    a2_extraction_profile: str = "MMWAVE_PHASE_EXTRACTION_PROFILE_001"
    a3_timeline_profile: str = "MMWAVE_TIMELINE_PROFILE_001"
    a4_label_profile: str = "MMWAVE_LABEL_MAPPING_PROFILE_001"
    a5_split_profile: str = "MMWAVE_SUBJECT_SPLIT_PROFILE_001"
    split_source: str = "datasets/mmwave/splits/mmwave_real_subject_split_v1.json"
    split_recomputed: bool = False
    timestamp_reference: str = "COMMON_ACQUISITION_COMPUTER_CLOCK"
    source_timezone: str = "UNVERIFIED"
    utc_conversion_claimed: bool = False
    canonical_signal: str = "UNFILTERED_UNNORMALIZED_PHASE"
    search_region_min_m: float = 0.3
    search_region_max_m: float = 1.91

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "a1_decoder_profile": self.a1_decoder_profile,
            "a2_extraction_profile": self.a2_extraction_profile,
            "a3_timeline_profile": self.a3_timeline_profile,
            "a4_label_profile": self.a4_label_profile,
            "a5_split_profile": self.a5_split_profile,
            "split_source": self.split_source,
            "split_recomputed": self.split_recomputed,
            "timestamp_reference": self.timestamp_reference,
            "source_timezone": self.source_timezone,
            "utc_conversion_claimed": self.utc_conversion_claimed,
            "canonical_signal": self.canonical_signal,
            "search_region": {
                "region_id": SEARCH_REGION_ID,
                "minimum_range_m": self.search_region_min_m,
                "maximum_range_m": self.search_region_max_m,
            },
        }


def compute_canonical_signal_hash(canonical_phase: np.ndarray) -> str:
    """Compute deterministic SHA-256 over contiguous float64 phase values."""
    arr = np.ascontiguousarray(canonical_phase, dtype=np.float64)
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def load_authoritative_a0_inventory(root_dir: Path) -> list[dict[str, Any]]:
    """Load recording records from authoritative A0 inventory manifest."""
    a0_manifest = root_dir / "datasets/mmwave/manifests/a0_raw_inventory/recording_index.jsonl"
    if not a0_manifest.is_file():
        raise FullConversionError(f"Authoritative A0 inventory manifest not found: {a0_manifest}")

    recordings = []
    with open(a0_manifest, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recordings.append(json.loads(line))
    return sorted(recordings, key=lambda r: (r.get("subject_id", ""), r.get("recording_id", "")))


def load_authoritative_a5_splits(root_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load subject and recording split maps from authoritative A5 artifacts."""
    a5_split_json = root_dir / "datasets/mmwave/splits/mmwave_real_subject_split_v1.json"
    rec_split_jsonl = root_dir / "datasets/mmwave/manifests/a5_subject_split/recording_split_manifest.jsonl"

    if not a5_split_json.is_file():
        raise FullConversionError(f"Authoritative A5 split JSON not found: {a5_split_json}")
    if not rec_split_jsonl.is_file():
        raise FullConversionError(f"Authoritative A5 recording split manifest not found: {rec_split_jsonl}")

    split_data = json.loads(a5_split_json.read_text(encoding="utf-8"))
    subject_split_map = split_data.get("subject_split_map", {})

    recording_split_map = {}
    with open(rec_split_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                rec_id = rec["recording_id"]
                rec_split = rec["split"]
                rec_subj = rec["subject_id"]

                expected_split = subject_split_map.get(rec_subj)
                if expected_split is None:
                    raise FullConversionError(f"A5 recording split contains unknown subject: {rec_subj}")
                if rec_split != expected_split:
                    raise FullConversionError(
                        f"A5 artifact split conflict for recording {rec_id}: recording manifest split '{rec_split}' != subject map split '{expected_split}'"
                    )
                recording_split_map[rec_id] = rec_split

    return subject_split_map, recording_split_map


def process_single_recording(
    rec_record: dict[str, Any],
    zip_archive: zipfile.ZipFile,
    subject_split: str,
    profile: FullConversionProfile = FullConversionProfile(),
) -> dict[str, Any]:
    """Process a single A0 recording through A1–A5 contracts into A6 canonical outputs."""
    rec_id = rec_record["recording_id"]
    subj_id = rec_record["subject_id"]
    source_path = rec_record["source_recording_path"]

    cond_obj = rec_record.get("activity_or_test") or rec_record.get("source_test_condition") or rec_record.get("source_explicit_condition")
    if isinstance(cond_obj, dict):
        condition = cond_obj.get("value", "UNKNOWN")
    elif isinstance(cond_obj, str):
        condition = cond_obj
    else:
        condition = "UNKNOWN"

    posture_obj = rec_record.get("posture") or rec_record.get("source_explicit_posture")
    if isinstance(posture_obj, dict):
        posture = posture_obj.get("value", "UNKNOWN")
    elif isinstance(posture_obj, str):
        posture = posture_obj
    else:
        posture = "UNKNOWN"

    radar_member = f"{source_path}/radar_rFFTs.zlib"
    timestamp_member = f"{source_path}/radar_timestamps.csv"
    chirp_config_member = f"{source_path}/radar_chirpConfig.json"
    ann_member = f"{source_path}/non_breathing_ts.csv"
    acc_member = f"{source_path}/movesense_acc.csv"

    # 1. Preflight check for member presence
    zip_members = set(zip_archive.namelist())
    if radar_member not in zip_members or timestamp_member not in zip_members or chirp_config_member not in zip_members:
        return {
            "recording_id": rec_id,
            "subject_id": subj_id,
            "status": "BLOCKED_MISSING_SOURCE",
            "error": f"Required member missing for {rec_id}",
            "windows": [],
            "provenance": [],
            "phase_slices": [],
        }

    # 2. A1 Safe Decode
    reader = SafeRFFTReader()
    try:
        a1_res = reader.read_recording(
            archive_path=zip_archive.filename,
            radar_member=radar_member,
            timestamp_member=timestamp_member,
            chirp_config_member=chirp_config_member,
        )
    except Exception as exc:
        return {
            "recording_id": rec_id,
            "subject_id": subj_id,
            "status": "FAILED_DECODE",
            "error": str(exc),
            "windows": [],
            "provenance": [],
            "phase_slices": [],
        }

    rffts = a1_res["tensor"]
    rbins = a1_res["range_bins"]

    # 3. A2 Phase Extraction
    search_region = SearchRegion(SEARCH_REGION_ID, profile.search_region_min_m, profile.search_region_max_m)
    extractor = MmwavePhaseExtractor(search_region)
    try:
        a2_res = extractor.extract(rffts=rffts, rbins=rbins)
    except Exception as exc:
        return {
            "recording_id": rec_id,
            "subject_id": subj_id,
            "status": "FAILED_PHASE_EXTRACTION",
            "error": str(exc),
            "windows": [],
            "provenance": [],
            "phase_slices": [],
        }

    canonical_phase = a2_res["unwrapped_phase"]

    # 4. A3 Timeline & 30-second Windows
    try:
        ts_bytes = zip_archive.read(timestamp_member)
        timestamps_raw = ts_bytes.decode("utf-8")
    except Exception as exc:
        return {
            "recording_id": rec_id,
            "subject_id": subj_id,
            "status": "FAILED_TIMELINE",
            "error": f"Failed reading radar timestamps: {exc}",
            "windows": [],
            "provenance": [],
            "phase_slices": [],
        }

    tl_profile = TimelineProfile()
    rec_tl_summary, a3_windows, exceptions = process_recording_timeline(
        phase=canonical_phase,
        timestamps_raw=timestamps_raw,
        recording_id=rec_id,
        subject_id=subj_id,
        extraction_profile_id=profile.a2_extraction_profile,
        profile=tl_profile,
    )
    exceptions = list(exceptions)

    def failed_label_evidence_result(status: str, message: str) -> dict[str, Any]:
        """Return an auditable failed result without emitting potentially mislabelled windows."""
        return {
            "recording_id": rec_id,
            "subject_id": subj_id,
            "split": subject_split,
            "status": status,
            "error": message,
            "frame_count": int(rffts.shape[0]),
            "tensor_shape": list(rffts.shape),
            "range_bin_count": int(rffts.shape[2]),
            "virtual_channel_count": int(rffts.shape[1]),
            "selected_range_bin_index": a2_res["selection"]["selected_range_bin_index"],
            "selected_stored_rbin_coordinate": a2_res["selection"]["selected_range_m"],
            "selected_virtual_channel": a2_res["selection"]["selected_virtual_channels"][0],
            "timeline_summary": rec_tl_summary,
            "window_count": 0,
            "annotation_file_count": 1,
            "annotation_event_count": 0,
            "windows": [],
            "provenance": [],
            "phase_slices": [],
            "exceptions": exceptions,
        }

    # Read optional annotation and Movesense ACC members
    events = []
    if ann_member in zip_members:
        try:
            ann_bytes = zip_archive.read(ann_member)
            radar_t0 = rec_tl_summary["first_timestamp"]
            events = parse_annotation_file(ann_bytes, radar_t0)
        except Exception as exc:
            message = f"Failed to read or parse annotation evidence: {exc}"
            exceptions.append(
                {
                    "category": "ANNOTATION_PARSE_FAILED",
                    "message": message,
                    "recording_id": rec_id,
                    "severity": "ERROR",
                    "source_member": ann_member,
                }
            )
            return failed_label_evidence_result("FAILED_ANNOTATION_PARSE", message)
        if not events:
            message = "Annotation member was present but produced no valid non-breathing event"
            exceptions.append(
                {
                    "category": "ANNOTATION_EVENT_MISSING",
                    "message": message,
                    "recording_id": rec_id,
                    "severity": "ERROR",
                    "source_member": ann_member,
                }
            )
            return failed_label_evidence_result("FAILED_ANNOTATION_PARSE", message)

    acc_bytes = None
    if acc_member in zip_members:
        try:
            acc_bytes = zip_archive.read(acc_member)
        except Exception as exc:
            acc_bytes = None
            exceptions.append(
                {
                    "category": "REFERENCE_ACC_READ_FAILED",
                    "message": f"Failed to read optional Movesense ACC reference: {exc}",
                    "recording_id": rec_id,
                    "severity": "WARNING",
                    "source_member": acc_member,
                }
            )

    # 5. A4 Derived Label Mapping & A5 Split Inheritance
    lbl_profile = LabelMappingProfile()
    final_windows = []
    provenance_records = []
    phase_slices = []

    for win in a3_windows:
        win_idx = win["window_index"]
        win_start_sec = win_idx * 30.0
        win_end_sec = win_start_sec + 30.0

        movesense_rr_info = None
        if acc_bytes is not None and rec_tl_summary.get("first_timestamp"):
            radar_t0 = rec_tl_summary["first_timestamp"]
            try:
                movesense_rr_info = extract_movesense_respiration_rate(
                    acc_bytes,
                    radar_t0,
                    win_start_sec,
                    win_end_sec,
                    search_band_hz=lbl_profile.movesense_rr_search_band_hz,
                )
            except Exception as exc:
                exceptions.append(
                    {
                        "category": "REFERENCE_RESPIRATION_EXTRACTION_FAILED",
                        "message": f"Failed to extract optional Movesense respiration reference: {exc}",
                        "recording_id": rec_id,
                        "window_id": win["window_id"],
                        "severity": "WARNING",
                        "source_member": acc_member,
                    }
                )

        mapped_win = map_window_label(
            window_record=win,
            events=events,
            source_condition=condition,
            posture=posture,
            profile=lbl_profile,
            movesense_rr_info=movesense_rr_info,
        )

        # Strip trailing 'Z' from newly formatted window timestamps for consistency with naive clock
        mapped_win["start_timestamp"] = mapped_win["start_timestamp"].rstrip("Z")
        mapped_win["last_sample_timestamp"] = mapped_win["last_sample_timestamp"].rstrip("Z")
        mapped_win["end_timestamp_exclusive"] = mapped_win["end_timestamp_exclusive"].rstrip("Z")

        # Enforce exact A5 split inheritance
        mapped_win["split"] = subject_split
        mapped_win["split_profile_id"] = profile.a5_split_profile

        # Define explicit eligibility flags
        assign_status = mapped_win["assignment_status"]

        training_eligible = (subject_split == "TRAIN") and (assign_status == "ASSIGNED")
        validation_eligible = (subject_split == "VALIDATION") and (assign_status == "ASSIGNED")
        locked_test_evaluation_eligible = (subject_split == "LOCKED_TEST") and (assign_status == "ASSIGNED")

        # Hard safety constraints
        if subject_split == "LOCKED_TEST":
            training_eligible = False

        if assign_status == "AMBIGUOUS":
            training_eligible = False
            validation_eligible = False
            locked_test_evaluation_eligible = False

        mapped_win["training_eligible"] = training_eligible
        mapped_win["validation_eligible"] = validation_eligible
        mapped_win["locked_test_evaluation_eligible"] = locked_test_evaluation_eligible

        # Calculate canonical phase slice for this window (300 samples)
        c_start = win["canonical_start_index"]
        c_end = win["canonical_end_index_exclusive"]
        win_phase_slice = np.ascontiguousarray(canonical_phase[c_start:c_end], dtype=np.float64)
        phase_hash = compute_canonical_signal_hash(win_phase_slice)
        mapped_win["canonical_signal_hash"] = phase_hash

        # Real signal quality audit measurements
        has_nan = bool(np.isnan(win_phase_slice).any())
        has_inf = bool(np.isinf(win_phase_slice).any())
        is_exact_constant = bool(np.all(win_phase_slice == win_phase_slice[0]))
        std_val = float(np.std(win_phase_slice))
        is_near_constant = bool(std_val < 1e-6)

        mapped_win["signal_quality_metrics"] = {
            "has_nan": has_nan,
            "has_inf": has_inf,
            "is_exact_constant": is_exact_constant,
            "is_near_constant": is_near_constant,
            "std_dev": round(std_val, 6),
            "mean_val": round(float(np.mean(win_phase_slice)), 6),
        }

        # Extract naive acquisition-clock ISO timestamps for A6 provenance
        start_ts_naive = mapped_win["start_timestamp"]
        last_ts_naive = mapped_win["last_sample_timestamp"]
        end_excl_ts_naive = mapped_win["end_timestamp_exclusive"]

        prov_row = {
            "window_id": win["window_id"],
            "recording_id": rec_id,
            "subject_id": subj_id,
            "archive_identifier": rec_record.get("archive_id", "db_records.zip"),
            "archive_sha256": rec_record.get("archive_sha256", "f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0"),
            "source_radar_member": radar_member,
            "source_timestamp_member": timestamp_member,
            "a1_decoder_profile": profile.a1_decoder_profile,
            "a2_extraction_profile": profile.a2_extraction_profile,
            "selected_range_bin_index": a2_res["selection"]["selected_range_bin_index"],
            "selected_stored_rbin_coordinate": a2_res["selection"]["selected_range_m"],
            "selected_virtual_channel": a2_res["selection"]["selected_virtual_channels"][0],
            "a3_timeline_profile": profile.a3_timeline_profile,
            "source_start_index": win["source_start_index"],
            "source_end_index_exclusive": win["source_end_index_exclusive"],
            "canonical_start_index": win["canonical_start_index"],
            "canonical_end_index_exclusive": win["canonical_end_index_exclusive"],
            "acquisition_clock_start_timestamp": start_ts_naive,
            "acquisition_clock_last_sample_timestamp": last_ts_naive,
            "acquisition_clock_end_timestamp_exclusive": end_excl_ts_naive,
            "timestamp_reference": profile.timestamp_reference,
            "source_timezone": profile.source_timezone,
            "utc_conversion_claimed": profile.utc_conversion_claimed,
            "original_annotation_semantics": mapped_win["original_annotation_type"],
            "safenest_label": mapped_win["safenest_label"],
            "safenest_label_id": mapped_win["safenest_label_id"],
            "mapping_type": mapped_win["mapping_type"],
            "mapping_rule_id": mapped_win["mapping_rule_id"],
            "assignment_status": mapped_win["assignment_status"],
            "split": subject_split,
            "split_profile_id": profile.a5_split_profile,
            "synthetic": False,
            "canonical_signal_hash": phase_hash,
            "quality_flags": win["quality_flags"],
            "training_eligible": training_eligible,
            "validation_eligible": validation_eligible,
            "locked_test_evaluation_eligible": locked_test_evaluation_eligible,
            "future_npz_sample_index": None,  # Set to None/null until Phase B training NPZ creation
        }

        final_windows.append(mapped_win)
        provenance_records.append(prov_row)
        phase_slices.append(win_phase_slice)

    status = "SUCCESS_WITH_WARNINGS" if exceptions else "SUCCESS"

    return {
        "recording_id": rec_id,
        "subject_id": subj_id,
        "split": subject_split,
        "status": status,
        "frame_count": int(rffts.shape[0]),
        "tensor_shape": list(rffts.shape),
        "range_bin_count": int(rffts.shape[2]),
        "virtual_channel_count": int(rffts.shape[1]),
        "selected_range_bin_index": a2_res["selection"]["selected_range_bin_index"],
        "selected_stored_rbin_coordinate": a2_res["selection"]["selected_range_m"],
        "selected_virtual_channel": a2_res["selection"]["selected_virtual_channels"][0],
        "timeline_summary": rec_tl_summary,
        "window_count": len(final_windows),
        "annotation_file_count": int(ann_member in zip_members),
        "annotation_event_count": len(events),
        "windows": final_windows,
        "provenance": provenance_records,
        "phase_slices": phase_slices,
        "exceptions": exceptions,
    }


__all__ = [
    "PROFILE_ID",
    "SEARCH_REGION_ID",
    "FullConversionError",
    "FullConversionProfile",
    "compute_canonical_signal_hash",
    "load_authoritative_a0_inventory",
    "load_authoritative_a5_splits",
    "process_single_recording",
]
