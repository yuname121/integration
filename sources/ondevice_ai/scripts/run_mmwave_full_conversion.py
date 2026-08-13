#!/usr/bin/env python3
"""SafeNest Phase A6 — Full mmWave Real-Data Conversion Runner.

Executes full A0 inventory conversion, signal quality audit, cross-split leakage audit,
canonical numeric dataset (.npy) artifact generation, invokes the standalone validator to control the gate,
and writes checksummed manifests.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
import zipfile

import numpy as np

# Ensure scripts directory is in path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from mmwave_full_converter import (
    PROFILE_ID,
    FullConversionProfile,
    compute_canonical_signal_hash,
    load_authoritative_a0_inventory,
    load_authoritative_a5_splits,
    process_single_recording,
)
from validate_mmwave_full_conversion import validate_full_conversion_artifacts


def measure_archive_sha256(archive_path: Path) -> str:
    """Measure SHA-256 digest of raw dataset archive zip."""
    hasher = hashlib.sha256()
    with open(archive_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_full_conversion(
    root_dir: Path = ROOT_DIR,
    profile: FullConversionProfile = FullConversionProfile(),
) -> dict[str, Any]:
    """Execute complete Phase A6 full conversion, audit, and validator gate evaluation."""
    archive_path = root_dir / "datasets/raw_archives/external_datasets/db_records.zip"
    if not archive_path.is_file():
        raise FileNotFoundError(f"Raw archive zip not found: {archive_path}")

    # 1. Pre-execution raw archive SHA-256 measurement
    pre_archive_sha256 = measure_archive_sha256(archive_path)

    # 2. Load authoritative A0 inventory and A5 splits
    a0_inventory = load_authoritative_a0_inventory(root_dir)
    subject_split_map, recording_split_map = load_authoritative_a5_splits(root_dir)

    measured_a0_subjects = sorted(list(set(r["subject_id"] for r in a0_inventory)))
    measured_a0_recordings = sorted(list(set(r["recording_id"] for r in a0_inventory)))
    rec_counts_per_subject = Counter(r["subject_id"] for r in a0_inventory)

    a0_measured_metrics = {
        "measured_subject_count": len(measured_a0_subjects),
        "measured_recording_count": len(measured_a0_recordings),
        "min_recordings_per_subject": min(rec_counts_per_subject.values()) if rec_counts_per_subject else 0,
        "max_recordings_per_subject": max(rec_counts_per_subject.values()) if rec_counts_per_subject else 0,
        "recording_count_distribution": {str(k): v for k, v in Counter(rec_counts_per_subject.values()).items()},
        "evidence_source": "datasets/mmwave/manifests/a0_raw_inventory/recording_index.jsonl",
    }

    missing_a5_subjects = [s for s in measured_a0_subjects if s not in subject_split_map]
    if missing_a5_subjects:
        raise ValueError(f"Authoritative A0 subjects missing from A5 split map: {missing_a5_subjects}")

    # 3. Process all A0 recordings
    all_recording_results = []
    all_windows = []
    all_provenance = []
    all_exceptions = []
    all_phase_slices = []

    recording_statuses = Counter()
    a1_decode_shapes = Counter()
    a1_frame_counts = Counter()
    a2_selected_bins = Counter()
    a2_selected_channels = Counter()
    dropped_tail_samples_dist = Counter()

    annotation_bearing_recordings = 0
    annotation_absent_recordings = 0
    total_non_breathing_events = 0

    with zipfile.ZipFile(archive_path, "r") as zf:
        for rec in a0_inventory:
            rec_id = rec["recording_id"]
            subj_id = rec["subject_id"]
            subj_split = subject_split_map[subj_id]

            # Count annotation evidence from A0 inventory
            ann_files = rec.get("annotation_files", [])
            if ann_files:
                annotation_bearing_recordings += 1
            else:
                annotation_absent_recordings += 1

            rec_res = process_single_recording(
                rec_record=rec,
                zip_archive=zf,
                subject_split=subj_split,
                profile=profile,
            )

            # Collect phase slices for canonical npy array
            slices = rec_res.pop("phase_slices", [])
            all_phase_slices.extend(slices)

            all_recording_results.append(rec_res)
            recording_statuses[rec_res["status"]] += 1
            total_non_breathing_events += int(rec_res.get("annotation_event_count", 0))

            if rec_res.get("tensor_shape"):
                a1_decode_shapes[str(tuple(rec_res["tensor_shape"]))] += 1
                a1_frame_counts[str(rec_res["frame_count"])] += 1
            if rec_res.get("selected_range_bin_index") is not None:
                a2_selected_bins[str(rec_res["selected_range_bin_index"])] += 1
            if rec_res.get("selected_virtual_channel") is not None:
                a2_selected_channels[str(rec_res["selected_virtual_channel"])] += 1

            if rec_res.get("timeline_summary"):
                dtail = rec_res["timeline_summary"].get("dropped_tail_samples", 0)
                dropped_tail_samples_dist[str(dtail)] += 1

            if rec_res.get("exceptions"):
                all_exceptions.extend(rec_res["exceptions"])

            for win in rec_res.get("windows", []):
                all_windows.append(win)

            for prov in rec_res.get("provenance", []):
                all_provenance.append(prov)

    # 4. Assign stable canonical_sample_index across all windows, provenance, and .npy rows
    for idx, (win, prov) in enumerate(zip(all_windows, all_provenance)):
        win["canonical_sample_index"] = idx
        prov["canonical_sample_index"] = idx
        prov["future_npz_sample_index"] = None  # Explicitly None/null until Phase B training NPZ creation

    # 5. Build and save canonical numeric dataset (.npy array)
    canonical_matrix = np.vstack(all_phase_slices).astype(np.float64)  # Shape: (530, 300)
    processed_dir = root_dir / "datasets/mmwave/processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    canonical_npy_path = processed_dir / "mmwave_canonical_real_v1.npy"
    np.save(canonical_npy_path, canonical_matrix)

    # 6. Real Quality Audit from phase slice measurements
    nan_count = sum(1 for w in all_windows if w["signal_quality_metrics"]["has_nan"])
    inf_count = sum(1 for w in all_windows if w["signal_quality_metrics"]["has_inf"])
    exact_constant_count = sum(1 for w in all_windows if w["signal_quality_metrics"]["is_exact_constant"])
    near_constant_count = sum(1 for w in all_windows if w["signal_quality_metrics"]["is_near_constant"])
    quality_flag_summary = Counter()

    for win in all_windows:
        for qf in win.get("quality_flags", []):
            quality_flag_summary[qf] += 1

    quality_audit = {
        "nan_sample_count": nan_count,
        "inf_sample_count": inf_count,
        "exact_constant_window_count": exact_constant_count,
        "near_constant_window_count": near_constant_count,
        "quality_flag_distribution": dict(quality_flag_summary),
        "total_windows_audited": len(all_windows),
        "frame_count_distribution": dict(a1_frame_counts),
        "tensor_shape_distribution": dict(a1_decode_shapes),
        "dropped_tail_sample_distribution": dict(dropped_tail_samples_dist),
        "selected_virtual_channel_distribution": dict(a2_selected_channels),
        "selected_range_bin_distribution": dict(a2_selected_bins),
        "mean_window_phase_std_dev": round(float(np.mean([w["signal_quality_metrics"]["std_dev"] for w in all_windows])), 6),
    }

    # 7. Duplicate & Cross-Split Leakage Audit
    hash_groups = defaultdict(list)
    for win in all_windows:
        h = win["canonical_signal_hash"]
        hash_groups[h].append(win)

    same_subject_duplicates = 0
    cross_subject_duplicates = 0
    cross_split_exact_signal_overlap = 0

    duplicate_groups_detail = []
    for h, group in hash_groups.items():
        if len(group) > 1:
            subjs = set(w["subject_id"] for w in group)
            splits = set(w["split"] for w in group)
            if len(subjs) == 1:
                same_subject_duplicates += len(group) - 1
            else:
                cross_subject_duplicates += len(group) - 1

            if len(splits) > 1:
                cross_split_exact_signal_overlap += 1

            duplicate_groups_detail.append(
                {
                    "signal_hash": h,
                    "window_count": len(group),
                    "subjects": list(subjs),
                    "splits": list(splits),
                    "window_ids": [w["window_id"] for w in group],
                }
            )

    split_subjects = defaultdict(set)
    split_recordings = defaultdict(set)
    split_window_ids = defaultdict(set)

    for win in all_windows:
        sp = win["split"]
        split_subjects[sp].add(win["subject_id"])
        split_recordings[sp].add(win["recording_id"])
        split_window_ids[sp].add(win["window_id"])

    train_subjs = split_subjects["TRAIN"]
    val_subjs = split_subjects["VALIDATION"]
    test_subjs = split_subjects["LOCKED_TEST"]

    subject_leakage_train_val = len(train_subjs & val_subjs)
    subject_leakage_train_test = len(train_subjs & test_subjs)
    subject_leakage_val_test = len(val_subjs & test_subjs)
    cross_split_subject_overlap = subject_leakage_train_val + subject_leakage_train_test + subject_leakage_val_test

    train_recs = split_recordings["TRAIN"]
    val_recs = split_recordings["VALIDATION"]
    test_recs = split_recordings["LOCKED_TEST"]
    cross_split_recording_overlap = len(train_recs & val_recs) + len(train_recs & test_recs) + len(val_recs & test_recs)

    train_wins = split_window_ids["TRAIN"]
    val_wins = split_window_ids["VALIDATION"]
    test_wins = split_window_ids["LOCKED_TEST"]
    cross_split_window_id_overlap = len(train_wins & val_wins) + len(train_wins & test_wins) + len(val_wins & test_wins)

    duplicate_audit = {
        "total_unique_signal_hashes": len(hash_groups),
        "exact_duplicate_hash_groups": len(duplicate_groups_detail),
        "same_subject_duplicate_count": same_subject_duplicates,
        "cross_subject_duplicate_count": cross_subject_duplicates,
        "cross_split_exact_signal_overlap": cross_split_exact_signal_overlap,
        "cross_split_subject_overlap": cross_split_subject_overlap,
        "cross_split_recording_overlap": cross_split_recording_overlap,
        "cross_split_window_id_overlap": cross_split_window_id_overlap,
        "near_duplicate_diagnostic": "NOT_PERFORMED",
        "duplicate_groups_detail": duplicate_groups_detail,
    }

    # 8. Deterministic Lineage Spot Checks
    spot_check_results = []
    if all_windows:
        N_win = len(all_windows)
        target_indices = [0, int(N_win * 0.25), int(N_win * 0.50), int(N_win * 0.75), N_win - 1]
        for idx in target_indices:
            win = all_windows[idx]
            prov = all_provenance[idx]
            npy_row_slice = canonical_matrix[idx]

            # Verify 1:1 match with .npy row
            npy_hash = compute_canonical_signal_hash(npy_row_slice)

            lineage_ok = (
                win["canonical_sample_index"] == idx
                and win["window_id"] == prov["window_id"]
                and win["split"] == prov["split"]
                and win["safenest_label"] == prov["safenest_label"]
                and win["canonical_signal_hash"] == npy_hash
            )

            spot_check_results.append(
                {
                    "sample_index": idx,
                    "window_id": win["window_id"],
                    "recording_id": win["recording_id"],
                    "subject_id": win["subject_id"],
                    "split": win["split"],
                    "safenest_label": win["safenest_label"],
                    "mapping_rule_id": win["mapping_rule_id"],
                    "canonical_signal_hash": win["canonical_signal_hash"],
                    "npy_row_signal_hash": npy_hash,
                    "source_radar_member": prov["source_radar_member"],
                    "lineage_verified": lineage_ok,
                }
            )

    # 9. Compute Full Label and Split Distributions
    label_counts = Counter(w["safenest_label"] or "AMBIGUOUS" for w in all_windows)
    split_counts = Counter(w["split"] for w in all_windows)
    split_label_counts = defaultdict(Counter)
    for w in all_windows:
        lbl = w["safenest_label"] or "AMBIGUOUS"
        split_label_counts[w["split"]][lbl] += 1

    label_distribution = {
        "class_counts_window": dict(label_counts),
        "split_label_breakdown": {sp: dict(cnts) for sp, cnts in split_label_counts.items()},
        "total_windows": len(all_windows),
        "annotation_coverage_accounting": {
            "annotation_bearing_recordings": annotation_bearing_recordings,
            "annotation_absent_recordings": annotation_absent_recordings,
            "total_non_breathing_events": total_non_breathing_events,
        },
    }

    split_distribution = {
        "subject_counts": {
            "TRAIN": len(train_subjs),
            "VALIDATION": len(val_subjs),
            "LOCKED_TEST": len(test_subjs),
        },
        "recording_counts": {
            "TRAIN": len(train_recs),
            "VALIDATION": len(val_recs),
            "LOCKED_TEST": len(test_recs),
        },
        "window_counts": dict(split_counts),
        "eligibility_counts": {
            "training_eligible": sum(1 for w in all_windows if w["training_eligible"]),
            "validation_eligible": sum(1 for w in all_windows if w["validation_eligible"]),
            "locked_test_evaluation_eligible": sum(1 for w in all_windows if w["locked_test_evaluation_eligible"]),
            "ambiguous_pure_class_eligible": sum(1 for w in all_windows if w["assignment_status"] == "AMBIGUOUS" and (w["training_eligible"] or w["validation_eligible"] or w["locked_test_evaluation_eligible"])),
            "locked_test_training_eligible": sum(1 for w in all_windows if w["split"] == "LOCKED_TEST" and w["training_eligible"]),
        },
    }

    # 10. Write Output Manifests to datasets/mmwave/manifests/a6_full_conversion/
    manifest_dir = root_dir / "datasets/mmwave/manifests/a6_full_conversion"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    (manifest_dir / "processing_profile.json").write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")

    with open(manifest_dir / "full_recording_results.jsonl", "w", encoding="utf-8") as f:
        for r in all_recording_results:
            r_clean = {k: v for k, v in r.items() if k not in {"windows", "provenance", "phase_slices"}}
            f.write(json.dumps(r_clean, sort_keys=True) + "\n")

    with open(manifest_dir / "full_window_manifest.jsonl", "w", encoding="utf-8") as f:
        for w in all_windows:
            f.write(json.dumps(w, sort_keys=True) + "\n")

    with open(manifest_dir / "full_provenance_manifest.jsonl", "w", encoding="utf-8") as f:
        for p in all_provenance:
            f.write(json.dumps(p, sort_keys=True) + "\n")

    (manifest_dir / "full_label_distribution.json").write_text(json.dumps(label_distribution, indent=2), encoding="utf-8")
    (manifest_dir / "full_split_distribution.json").write_text(json.dumps(split_distribution, indent=2), encoding="utf-8")
    (manifest_dir / "full_quality_audit.json").write_text(json.dumps(quality_audit, indent=2), encoding="utf-8")
    (manifest_dir / "full_duplicate_audit.json").write_text(json.dumps(duplicate_audit, indent=2), encoding="utf-8")
    (manifest_dir / "spot_check_results.json").write_text(json.dumps(spot_check_results, indent=2), encoding="utf-8")
    (manifest_dir / "exceptions.json").write_text(json.dumps(all_exceptions, indent=2), encoding="utf-8")

    # 11. Compute checksums.sha256 for output directory and canonical .npy dataset
    manifest_files = [
        "processing_profile.json",
        "full_recording_results.jsonl",
        "full_window_manifest.jsonl",
        "full_provenance_manifest.jsonl",
        "full_label_distribution.json",
        "full_split_distribution.json",
        "full_quality_audit.json",
        "full_duplicate_audit.json",
        "spot_check_results.json",
        "exceptions.json",
    ]

    checksum_lines = []
    for fname in sorted(manifest_files):
        fpath = manifest_dir / fname
        if fpath.is_file():
            h = hashlib.sha256(fpath.read_bytes()).hexdigest()
            checksum_lines.append(f"{h}  {fname}")

    if canonical_npy_path.is_file():
        npy_h = hashlib.sha256(canonical_npy_path.read_bytes()).hexdigest()
        checksum_lines.append(f"{npy_h}  ../../processed/mmwave_canonical_real_v1.npy")

    (manifest_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    # 12. Invoke Standalone Validator to CONTROL THE GATE
    val_res = validate_full_conversion_artifacts(root_dir=root_dir, manifest_dir=manifest_dir)

    # 13. Post-execution raw archive SHA-256 measurement
    post_archive_sha256 = measure_archive_sha256(archive_path)
    archive_unchanged = (pre_archive_sha256 == post_archive_sha256)

    # 14. Write Final Summary using Validator Verdict
    summary_data = {
        "profile_id": PROFILE_ID,
        "a0_measured_metrics": a0_measured_metrics,
        "processed_recording_count": len(all_recording_results),
        "processed_window_count": len(all_windows),
        "canonical_npy_artifact": "datasets/mmwave/processed/mmwave_canonical_real_v1.npy",
        "canonical_npy_shape": list(canonical_matrix.shape),
        "pre_a6_archive_sha256": pre_archive_sha256,
        "post_a6_archive_sha256": post_archive_sha256,
        "archive_unchanged": archive_unchanged,
        "recording_status_breakdown": dict(recording_statuses),
        "split_subject_distribution": split_distribution["subject_counts"],
        "split_window_distribution": split_distribution["window_counts"],
        "label_window_distribution": label_distribution["class_counts_window"],
        "leakage_audit_summary": {
            "cross_split_exact_signal_overlap": cross_split_exact_signal_overlap,
            "cross_split_subject_overlap": cross_split_subject_overlap,
            "cross_split_recording_overlap": cross_split_recording_overlap,
            "cross_split_window_id_overlap": cross_split_window_id_overlap,
        },
        "validator_verdict": val_res,
        "validation_passed": val_res["validation_success"],
        "a6_gate_status": val_res["a6_gate_status"],
        "phase_b_entry_status": val_res["phase_b_entry_status"],
    }

    (manifest_dir / "a6_summary.json").write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    # Re-update checksums.sha256 to include a6_summary.json
    checksum_lines.append(f"{hashlib.sha256((manifest_dir / 'a6_summary.json').read_bytes()).hexdigest()}  a6_summary.json")
    checksum_lines.sort(key=lambda line: line.split(maxsplit=1)[1])
    (manifest_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    return summary_data


if __name__ == "__main__":
    res = run_full_conversion()
    print("Phase A6 full conversion completed successfully.")
    print(f"Gate Status: {res['a6_gate_status']}")
    print(f"Phase-B Entry Status: {res['phase_b_entry_status']}")
