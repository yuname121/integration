#!/usr/bin/env python3
"""Run the deterministic Phase A3 timeline reconstruction and 30-second window pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from mmwave_phase_extractor import array_sha256
from mmwave_rfft_reader import SafeRFFTReader
from mmwave_timeline import (
    PROFILE_ID,
    TimelineProfile,
    process_recording_timeline,
)
from validate_mmwave_timeline_pilot import derive_gate, validate_manifests


EXPECTED_ARCHIVE_SHA256 = "f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0"


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


def build_a3_pilot_selection(a2_pilot: dict[str, Any]) -> dict[str, Any]:
    recordings = []
    for item in a2_pilot["recordings"]:
        rec = dict(item)
        rec["a3_selection_reason"] = "PRESERVE_APPROVED_A2_PILOT"
        recordings.append(rec)
    return {
        "label_independent_selection": True,
        "recordings": recordings,
    }


def write_checksums(manifest_dir: Path) -> None:
    manifest_files = sorted(
        [
            p
            for p in manifest_dir.glob("*")
            if p.is_file() and p.name != "checksums.sha256"
        ]
    )
    lines = []
    for p in manifest_files:
        h = sha256_file(p)
        lines.append(f"{h}  {p.name}\n")
    (manifest_dir / "checksums.sha256").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase A3 timeline pilot.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root directory",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    archive_path = root / "datasets/raw_archives/external_datasets/db_records.zip"
    if not archive_path.exists():
        raise FileNotFoundError(f"Raw archive missing: {archive_path}")

    initial_archive_sha256 = sha256_file(archive_path)
    if initial_archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(
            f"Archive SHA256 mismatch: got {initial_archive_sha256}, expected {EXPECTED_ARCHIVE_SHA256}"
        )

    a2_dir = root / "datasets/mmwave/manifests/a2_phase_pilot"
    a2_pilot = json.loads((a2_dir / "pilot_selection.json").read_text(encoding="utf-8"))
    a2_results = _load_jsonl(a2_dir / "selected_phase_results.jsonl")
    a2_by_id = {row["recording_id"]: row for row in a2_results}

    profile = TimelineProfile()

    reader = SafeRFFTReader()
    rec_results: list[dict[str, Any]] = []
    all_windows: list[dict[str, Any]] = []
    all_exceptions: list[dict[str, Any]] = []

    a3_pilot_sel = build_a3_pilot_selection(a2_pilot)

    for item in a3_pilot_sel["recordings"]:
        rec_id = item["recording_id"]
        subj_id = item["subject_id"]
        source_path = item["source_recording_path"]

        a2_row = a2_by_id[rec_id]
        bin_idx = a2_row["selected_range_bin_index"]
        chan_idx = a2_row["selected_virtual_channels"][0]

        radar_member = source_path + "/radar_rFFTs.zlib"
        ts_member = source_path + "/radar_timestamps.csv"
        cfg_member = source_path + "/radar_chirpConfig.json"

        decomp = reader.read_recording(
            archive_path=str(archive_path),
            radar_member=radar_member,
            timestamp_member=ts_member,
            chirp_config_member=cfg_member,
        )

        tensor = decomp["tensor"]
        complex_sig = tensor[:, chan_idx, bin_idx]
        wrapped = np.angle(complex_sig)
        unwrapped = np.unwrap(wrapped)

        unwrapped_sha256 = array_sha256(unwrapped)
        if unwrapped_sha256 != a2_row["unwrapped_phase_sha256"]:
            raise ValueError(
                f"Recording {rec_id} unwrapped phase SHA256 mismatch with A2: "
                f"reconstructed={unwrapped_sha256} != a2={a2_row['unwrapped_phase_sha256']}"
            )

        with open(archive_path, "rb") as zf:
            import zipfile
            with zipfile.ZipFile(zf, "r") as z:
                ts_raw = z.read(ts_member)

        rec_res, windows, exceptions = process_recording_timeline(
            phase=unwrapped,
            timestamps_raw=ts_raw,
            recording_id=rec_id,
            subject_id=subj_id,
            extraction_profile_id=a2_row["selected_extraction_profile"],
            profile=profile,
        )

        rec_results.append(rec_res)
        all_windows.extend(windows)
        all_exceptions.extend(exceptions)

    final_archive_sha256 = sha256_file(archive_path)
    archive_unchanged = (initial_archive_sha256 == final_archive_sha256)

    # Compile summary metrics
    total_wins = len(all_windows)
    valid_wins = sum(1 for w in all_windows if w["timeline_valid"])
    invalid_wins = total_wins - valid_wins

    total_dropped_tail = sum(r["dropped_tail_samples"] for r in rec_results)
    total_interpolated = sum(r["interpolated_sample_count"] for r in rec_results)
    resampling_performed_count = sum(1 for r in rec_results if r["resampling_performed"])

    subjects = sorted(list({r["subject_id"] for r in rec_results}))
    postures = sorted(list({item["posture"] for item in a3_pilot_sel["recordings"]}))
    activities = sorted(list({item["activity_or_test"] for item in a3_pilot_sel["recordings"]}))
    frame_variants = sorted(list({r["source_sample_count"] for r in rec_results}))

    # Derive initial gate status assuming clean structural validation
    gate_status, a4_entry_status = derive_gate(True, all_exceptions, rec_results)

    explicit_non_scope = {
        "safenest_label_mapping": "NOT_PERFORMED",
        "breath_hold_apnea_mapping": "NOT_PERFORMED",
        "subject_split": "NOT_PERFORMED",
        "full_440_recording_conversion": "NOT_PERFORMED",
        "training_npz_generation": "NOT_PERFORMED",
        "permanent_bandpass_filtering": "NOT_PERFORMED",
        "z_score_normalization": "NOT_PERFORMED",
        "model_training": "NOT_PERFORMED",
        "a4": "NOT_PERFORMED",
    }

    summary = {
        "schema_version": "1.0",
        "selected_profile_id": profile.profile_id,
        "a2_extraction_profile": "MMWAVE_PHASE_EXTRACTION_PROFILE_001",
        "a3_gate_status": gate_status,
        "a4_entry_status": a4_entry_status,
        "validation_success": True,
        "pilot_recording_count": len(rec_results),
        "pilot_subject_count": len(subjects),
        "postures_covered": postures,
        "activity_or_test_conditions_covered": activities,
        "frame_count_variants": frame_variants,
        "total_window_count": total_wins,
        "total_valid_window_count": valid_wins,
        "total_invalid_window_count": invalid_wins,
        "total_dropped_tail_samples": total_dropped_tail,
        "total_interpolated_samples": total_interpolated,
        "resampling_performed_count": resampling_performed_count,
        "exception_count": len(all_exceptions),
        "archive_sha256_before_a3": initial_archive_sha256,
        "archive_sha256_after_a3": final_archive_sha256,
        "archive_unchanged_after_a3": archive_unchanged,
        "explicit_non_scope": explicit_non_scope,
    }

    val_success, val_errors = validate_manifests(
        a2_pilot=a2_pilot,
        profile=profile.to_dict(),
        rec_results=rec_results,
        windows=all_windows,
        exceptions=all_exceptions,
        summary=summary,
    )

    if not val_success or val_errors:
        gate_status, a4_entry_status = derive_gate(False, all_exceptions, rec_results)
        summary["a3_gate_status"] = gate_status
        summary["a4_entry_status"] = a4_entry_status
        summary["validation_success"] = False

    manifest_dir = root / "datasets/mmwave/manifests/a3_timeline_pilot"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    write_json(manifest_dir / "pilot_selection.json", a3_pilot_sel)
    write_json(manifest_dir / "timeline_profile.json", profile.to_dict())
    write_jsonl(manifest_dir / "recording_timeline_results.jsonl", rec_results)
    write_jsonl(manifest_dir / "window_manifest.jsonl", all_windows)
    write_json(manifest_dir / "exceptions.json", all_exceptions)
    write_json(manifest_dir / "a3_summary.json", summary)

    write_checksums(manifest_dir)

    print(f"Phase A3 pilot execution completed successfully.")
    print(f"Gate Status: {gate_status}")
    print(f"A4 Entry Status: {a4_entry_status}")
    print(f"Recordings: {len(rec_results)}, Windows: {total_wins}, Dropped Tail Samples: {total_dropped_tail}")


if __name__ == "__main__":
    main()
