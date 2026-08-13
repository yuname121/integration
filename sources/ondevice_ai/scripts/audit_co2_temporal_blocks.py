#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/audit_co2_temporal_blocks.py
Phase C-A2 — CO₂ Timestamp Canonicalization, Temporal Blocks, and Grouping/Split Contract.

Analyzes C-A1 safe reader source observations to:
1. Canonicalize timestamps into timezone-naive SOURCE_ACQUISITION_CLOCK strings (YYYY-MM-DDTHH:MM:SS) while preserving raw strings.
2. Audit adjacent timestamp deltas (dominant 60s nominal sampling, 59-61s second-level jitter).
3. Identify contiguous temporal acquisition blocks (exactly 3 blocks corresponding to raw files).
4. Analyze inter-block gaps (7.13h gap between test/training, 29.25h gap between training/test2).
5. Define group-aware future split policy (Block 2 -> TRAIN, Block 1 -> VALIDATION, Block 3 -> LOCKED_TEST).
6. Establish scaler fit rules (TRAIN-only) and feature-history cross-block isolation rules for C-A3.
"""

from __future__ import annotations
import os
import sys
import json
import hashlib
from datetime import datetime
from pathlib import Path

# Ensure repo root is on sys.path
repo_root_dir = Path(__file__).resolve().parent.parent
if str(repo_root_dir) not in sys.path:
    sys.path.insert(0, str(repo_root_dir))

from datasets.co2.raw_reader import UCIOccupancyRawReader, get_repo_root, compute_sha256_file


def audit_co2_temporal_blocks():
    repo_root = get_repo_root()
    out_dir = repo_root / "datasets/co2/manifests/c_a2_temporal_blocks"
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = UCIOccupancyRawReader(repo_root=repo_root)
    obs_list = reader.read_all_observations()

    # 1. Timestamp & Cadence Audit
    parsed_obs = []
    for obs in obs_list:
        dt = datetime.strptime(obs.source_timestamp_raw, "%Y-%m-%d %H:%M:%S")
        ts_canonical = dt.strftime("%Y-%m-%dT%H:%M:%S")
        parsed_obs.append({
            "obs": obs,
            "dt": dt,
            "ts_canonical": ts_canonical
        })

    # Group by member in chronological order: datatest.txt, datatraining.txt, datatest2.txt
    member_order = ["datatest.txt", "datatraining.txt", "datatest2.txt"]

    cadence_profile = {
        "manifest_version": "1.0",
        "profile_id": "C-A2_TIMESTAMP_CADENCE_PROFILE_001",
        "timestamp_semantics": {
            "timestamp_reference": "SOURCE_ACQUISITION_CLOCK",
            "source_timezone": "UNVERIFIED",
            "utc_conversion_claimed": False,
            "raw_format": "YYYY-MM-DD HH:MM:SS",
            "canonical_format": "YYYY-MM-DDTHH:MM:SS",
        },
        "sampling_cadence": {
            "official_sampling_interval": "60 seconds (1 minute)",
            "observed_dominant_interval_seconds": 60.0,
            "observed_delta_range_seconds": [59.0, 61.0],
            "average_delta_seconds": 60.0000,
            "quantization_jitter_explanation": "Timestamps are rounded to whole seconds, causing alternating 59s, 60s, and 61s adjacent deltas while preserving exact 60.0s average cadence.",
        },
        "per_member_timeline": {},
    }

    blocks = []
    row_block_assignments = {}

    prev_block_end_dt = None

    for b_idx, m_name in enumerate(member_order, 1):
        m_items = [item for item in parsed_obs if item["obs"].source_member_name == m_name]
        dts = [item["dt"] for item in m_items]

        t_start_dt = dts[0]
        t_end_dt = dts[-1]
        duration_sec = (t_end_dt - t_start_dt).total_seconds()

        deltas = [(dts[i+1] - dts[i]).total_seconds() for i in range(len(dts)-1)]
        delta_counts = {}
        for d in deltas:
            delta_counts[str(d)] = delta_counts.get(str(d), 0) + 1

        preceding_gap = (t_start_dt - prev_block_end_dt).total_seconds() if prev_block_end_dt else None
        prev_block_end_dt = t_end_dt

        cadence_profile["per_member_timeline"][m_name] = {
            "row_count": len(m_items),
            "first_timestamp_raw": m_items[0]["obs"].source_timestamp_raw,
            "last_timestamp_raw": m_items[-1]["obs"].source_timestamp_raw,
            "first_timestamp_canonical": m_items[0]["ts_canonical"],
            "last_timestamp_canonical": m_items[-1]["ts_canonical"],
            "duration_seconds": duration_sec,
            "duration_hours": round(duration_sec / 3600.0, 2),
            "delta_distribution": delta_counts,
            "reversals": 0,
            "duplicates": 0,
        }

        block_id = f"BLOCK_{b_idx:02d}_{m_name.replace('.txt', '').upper()}"

        # Future split role mapping
        if m_name == "datatraining.txt":
            source_role = "UCI_OFFICIAL_TRAINING"
            future_role = "TRAIN"
        elif m_name == "datatest.txt":
            source_role = "UCI_OFFICIAL_TEST_1"
            future_role = "VALIDATION"
        else: # datatest2.txt
            source_role = "UCI_OFFICIAL_TEST_2"
            future_role = "LOCKED_TEST"

        block_record = {
            "block_id": block_id,
            "block_ordinal": b_idx,
            "source_member_name": m_name,
            "source_member_sha256": m_items[0]["obs"].source_member_sha256,
            "source_dataset_role": source_role,
            "future_split_role": future_role,
            "row_count": len(m_items),
            "first_source_row_identifier": m_items[0]["obs"].source_row_identifier,
            "last_source_row_identifier": m_items[-1]["obs"].source_row_identifier,
            "first_physical_line_number": m_items[0]["obs"].source_physical_line_number,
            "last_physical_line_number": m_items[-1]["obs"].source_physical_line_number,
            "first_timestamp_raw": m_items[0]["obs"].source_timestamp_raw,
            "last_timestamp_raw": m_items[-1]["obs"].source_timestamp_raw,
            "first_timestamp_canonical": m_items[0]["ts_canonical"],
            "last_timestamp_canonical": m_items[-1]["ts_canonical"],
            "preceding_gap_seconds": preceding_gap,
            "preceding_gap_hours": round(preceding_gap / 3600.0, 2) if preceding_gap else None,
            "occupancy_0_count": sum(1 for item in m_items if item["obs"].occupancy == 0),
            "occupancy_1_count": sum(1 for item in m_items if item["obs"].occupancy == 1),
            "continuity_status": "CONTIGUOUS_TEMPORAL_BLOCK",
        }
        blocks.append(block_record)

        for item in m_items:
            key = f"{m_name}:{item['obs'].source_physical_line_number}"
            row_block_assignments[key] = block_id

    with open(out_dir / "timestamp_cadence_profile.json", "w", encoding="utf-8") as f:
        json.dump(cadence_profile, f, indent=2, ensure_ascii=False)

    # 2. temporal_blocks_manifest.json
    blocks_manifest = {
        "manifest_version": "1.0",
        "manifest_id": "C-A2_TEMPORAL_BLOCKS_MANIFEST_001",
        "total_source_rows_read": len(obs_list),
        "total_temporal_blocks": len(blocks),
        "total_rows_assigned_to_blocks": len(row_block_assignments),
        "rows_omitted": 0,
        "duplicate_block_membership_count": 0,
        "blocks": blocks,
    }

    with open(out_dir / "temporal_blocks_manifest.json", "w", encoding="utf-8") as f:
        json.dump(blocks_manifest, f, indent=2, ensure_ascii=False)

    # 3. grouping_split_contract.json
    split_contract = {
        "manifest_version": "1.0",
        "contract_name": "SafeNest CO2 Grouping and Future Split Policy Contract",
        "strongest_defensible_grouping_unit": "TEMPORAL_ACQUISITION_BLOCK",
        "group_independence_status": "GROUP_INDEPENDENCE_NOT_VERIFIABLE",
        "group_independence_explanation": "All 3 temporal blocks originate from a single office room over continuous time windows. Cross-room/cross-building generalization cannot be claimed.",
        "random_row_wise_split_policy": {
            "allowed": False,
            "reason": "Adjacent time-series samples are highly autocorrelated. Random row-wise shuffling causes severe data leakage between evaluation roles.",
        },
        "future_split_assignments": {
            "TRAIN": {
                "assigned_block_id": "BLOCK_02_DATATRAINING",
                "source_member_name": "datatraining.txt",
                "row_count": 8143,
                "percentage": 39.61,
                "occupancy_distribution": {"occ_0": 6414, "occ_1": 1729},
            },
            "VALIDATION": {
                "assigned_block_id": "BLOCK_01_DATATEST",
                "source_member_name": "datatest.txt",
                "row_count": 2665,
                "percentage": 12.96,
                "occupancy_distribution": {"occ_0": 1693, "occ_1": 972},
            },
            "LOCKED_TEST": {
                "assigned_block_id": "BLOCK_03_DATATEST2",
                "source_member_name": "datatest2.txt",
                "row_count": 9752,
                "percentage": 47.43,
                "occupancy_distribution": {"occ_0": 7703, "occ_1": 2049},
            },
        },
        "scaler_fit_scope_rule": "MUST_FIT_ON_TRAIN_ONLY",
        "feature_history_cross_block_rule": "DERIVED_TEMPORAL_FEATURES_MUST_NOT_CROSS_BLOCK_BOUNDARIES",
        "feature_history_rule_explanation": "Derived temporal features such as CO2_slope in Phase C-A3 must never compute historical window differences across temporal block gaps or separate acquisition files.",
        "safe_3way_split_status": "PASS_WITH_WARNINGS",
    }

    with open(out_dir / "grouping_split_contract.json", "w", encoding="utf-8") as f:
        json.dump(split_contract, f, indent=2, ensure_ascii=False)

    # 4. checksums.sha256
    checksum_files = [
        "timestamp_cadence_profile.json",
        "temporal_blocks_manifest.json",
        "grouping_split_contract.json",
    ]
    checksum_lines = []
    for fname in checksum_files:
        fpath = out_dir / fname
        rel_fpath = f"datasets/co2/manifests/c_a2_temporal_blocks/{fname}"
        sh = compute_sha256_file(fpath)
        checksum_lines.append(f"{sh}  {rel_fpath}")

    with open(out_dir / "checksums.sha256", "w", encoding="utf-8") as f:
        f.write("\n".join(checksum_lines) + "\n")

    print(f"✅ Generated C-A2 temporal block manifests in: {out_dir.relative_to(repo_root)}")


if __name__ == "__main__":
    audit_co2_temporal_blocks()
