#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/audit_co2_slope_feature.py
Phase C-A3 — CO₂ Slope Feature Reconstruction evidence generator.

Produces deterministic manifests under:
  datasets/co2/manifests/c_a3_slope_feature/
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

repo_root_dir = Path(__file__).resolve().parent.parent
if str(repo_root_dir) not in sys.path:
    sys.path.insert(0, str(repo_root_dir))

from datasets.co2.raw_reader import UCIOccupancyRawReader, compute_sha256_file, get_repo_root
from datasets.co2.slope_feature import (
    COMPARISON_ABS_TOLERANCE,
    FEATURE_PROFILE_ID,
    HISTORY_DURATION_SECONDS,
    MEMBER_ORDER,
    STATUS_AVAILABLE,
    STATUS_WARMUP,
    build_feature_profile,
    compute_endpoint_slope_ppm_per_min,
    compute_slope_audit_statistics,
    parse_source_timestamp,
    reconstruct_all_slope_features,
    summarize_eligibility,
)


HISTORICAL_SCALER_MEAN_SLOPE = 0.011184156631440968
HISTORICAL_SCALER_SCALE_SLOPE = 4.373409389136896


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _independent_endpoint_slope(
    co2_now: float,
    co2_start: float,
    t_now: datetime,
    t_start: datetime,
) -> float:
    """Independent recomputation (not calling production reconstruct helpers)."""
    elapsed = (t_now - t_start).total_seconds()
    return (float(co2_now) - float(co2_start)) / (float(elapsed) / 60.0)


def _find_record(records, member: str, row_id: str):
    for record in records:
        if (
            record.target_source_member == member
            and record.target_source_row_identifier == row_id
        ):
            return record
    raise KeyError(f"Missing record {member}:{row_id}")


def _pick_delta_case(observations, member: str, target_delta: float):
    member_obs = [o for o in observations if o.source_member_name == member]
    for i in range(3, len(member_obs)):
        prev = member_obs[i - 1]
        cur = member_obs[i]
        t_prev, _ = parse_source_timestamp(prev.source_timestamp_raw)
        t_cur, _ = parse_source_timestamp(cur.source_timestamp_raw)
        if abs((t_cur - t_prev).total_seconds() - target_delta) < 1e-9:
            return cur.source_row_identifier
    raise RuntimeError(f"No {target_delta}s delta case in {member}")


def build_manual_verification_cases(observations, records) -> Dict[str, Any]:
    """Independently recompute expected slopes for representative cases."""
    cases: List[Dict[str, Any]] = []

    def add_case(
        case_id: str,
        category: str,
        member: str,
        row_id: str,
        expect_available: bool,
        notes: str,
    ) -> None:
        record = _find_record(records, member, row_id)
        expected_slope: Optional[float] = None
        independent_method = "NULL_UNAVAILABLE"

        if expect_available:
            # Independent walk-back using only raw timestamps/CO2 from the same member.
            member_obs = [o for o in observations if o.source_member_name == member]
            idx = next(
                i
                for i, o in enumerate(member_obs)
                if o.source_row_identifier == row_id
            )
            t_now, _ = parse_source_timestamp(member_obs[idx].source_timestamp_raw)
            history_idx = None
            for k in range(idx - 1, -1, -1):
                t_k, _ = parse_source_timestamp(member_obs[k].source_timestamp_raw)
                if (t_now - t_k).total_seconds() >= HISTORY_DURATION_SECONDS:
                    history_idx = k
                    break
            if history_idx is None:
                raise AssertionError(f"Expected available history for {case_id}")
            t_start, _ = parse_source_timestamp(
                member_obs[history_idx].source_timestamp_raw
            )
            expected_slope = _independent_endpoint_slope(
                member_obs[idx].co2,
                member_obs[history_idx].co2,
                t_now,
                t_start,
            )
            # Cross-check helper formula separately from reconstruction.
            helper_slope = compute_endpoint_slope_ppm_per_min(
                float(member_obs[idx].co2),
                float(member_obs[history_idx].co2),
                float((t_now - t_start).total_seconds()),
            )
            if abs(helper_slope - expected_slope) > COMPARISON_ABS_TOLERANCE:
                raise AssertionError("Independent and helper slopes diverge")
            independent_method = (
                "INDEPENDENT_ENDPOINT_DIFFERENCE_OVER_SOURCE_CLOCK_ELAPSED_MINUTES"
            )

        actual = record.co2_slope
        if expect_available:
            assert actual is not None
            abs_err = abs(float(actual) - float(expected_slope))
            passed = abs_err <= COMPARISON_ABS_TOLERANCE
        else:
            abs_err = 0.0 if actual is None else math.inf
            passed = actual is None and record.feature_status == STATUS_WARMUP

        cases.append(
            {
                "case_id": case_id,
                "category": category,
                "target_source_member": member,
                "target_source_row_identifier": row_id,
                "temporal_block_id": record.temporal_block_id,
                "future_split_role": record.future_split_role,
                "feature_status_actual": record.feature_status,
                "expect_available": expect_available,
                "independent_method": independent_method,
                "expected_co2_slope": expected_slope,
                "actual_co2_slope": actual,
                "absolute_error": abs_err if expect_available else None,
                "tolerance": COMPARISON_ABS_TOLERANCE,
                "pass": passed,
                "history_elapsed_seconds": record.history_elapsed_seconds,
                "history_source_row_identifiers": list(
                    record.history_source_row_identifiers
                ),
                "notes": notes,
            }
        )

    # 1) First row of TRAIN block
    add_case(
        "CASE_01_BLOCK_FIRST_TRAIN",
        "first_row_of_temporal_block",
        "datatraining.txt",
        "1",
        False,
        "First row of BLOCK_02_DATATRAINING has no past history in-block.",
    )
    # 2) Warm-up row (index 2 / row id 3)
    add_case(
        "CASE_02_WARMUP_TRAIN",
        "warmup_row",
        "datatraining.txt",
        "3",
        False,
        "Third row spans ~120s < 150s history duration.",
    )
    # 3) First eligible after warm-up (row id 4)
    add_case(
        "CASE_03_FIRST_ELIGIBLE_TRAIN",
        "first_eligible_after_warmup",
        "datatraining.txt",
        "4",
        True,
        "Fourth TRAIN row first reaches >=150s same-block history.",
    )
    # 4) Normal interior TRAIN row
    add_case(
        "CASE_04_INTERIOR_TRAIN",
        "normal_interior_row",
        "datatraining.txt",
        "500",
        True,
        "Interior TRAIN row with full in-block history.",
    )
    # 5) 59-second adjacent delta case (TRAIN)
    row_59 = _pick_delta_case(observations, "datatraining.txt", 59.0)
    add_case(
        "CASE_05_DELTA_59S_TRAIN",
        "adjacent_delta_59s",
        "datatraining.txt",
        row_59,
        True,
        "Eligible row whose immediate previous adjacent delta is 59s.",
    )
    # 6) 61-second adjacent delta case (TRAIN)
    row_61 = _pick_delta_case(observations, "datatraining.txt", 61.0)
    add_case(
        "CASE_06_DELTA_61S_TRAIN",
        "adjacent_delta_61s",
        "datatraining.txt",
        row_61,
        True,
        "Eligible row whose immediate previous adjacent delta is 61s.",
    )
    # 7) Block-boundary restart (first VALIDATION row)
    add_case(
        "CASE_07_BOUNDARY_RESTART_VALIDATION",
        "block_boundary_restart",
        "datatest.txt",
        "140",
        False,
        "First VALIDATION/block row must not borrow TRAIN/prior-block history.",
    )
    # 8) One eligible row from each future role (contract already frozen)
    add_case(
        "CASE_08_ELIGIBLE_VALIDATION",
        "per_role_validation",
        "datatest.txt",
        "143",
        True,
        "First eligible VALIDATION row after warm-up (post-freeze sanity).",
    )
    add_case(
        "CASE_09_ELIGIBLE_TRAIN_ROLE",
        "per_role_train",
        "datatraining.txt",
        "100",
        True,
        "Eligible TRAIN role verification.",
    )
    add_case(
        "CASE_10_ELIGIBLE_LOCKED_TEST",
        "per_role_locked_test",
        "datatest2.txt",
        "4",
        True,
        "Eligible LOCKED_TEST row for lineage/integrity only; not used for formula selection.",
    )

    all_pass = all(c["pass"] for c in cases)
    return {
        "manifest_version": "1.0",
        "feature_contract_id": FEATURE_PROFILE_ID,
        "verification_policy": (
            "Independent source-clock endpoint recomputation; production reconstruct "
            "is not merely called twice."
        ),
        "comparison_abs_tolerance": COMPARISON_ABS_TOLERANCE,
        "contract_frozen_before_validation_locked_test_cases": True,
        "all_cases_pass": all_pass,
        "case_count": len(cases),
        "cases": cases,
    }


def build_candidate_evaluation(train_records) -> Dict[str, Any]:
    """Record candidate evaluation evidence (no label/model metrics)."""
    # TRAIN-only descriptive comparison already computed externally during selection;
    # persist the locked rationale.
    train_slopes = [
        float(r.co2_slope)
        for r in train_records
        if r.feature_status == STATUS_AVAILABLE and r.co2_slope is not None
    ]
    mean = sum(train_slopes) / len(train_slopes)
    var = sum((x - mean) ** 2 for x in train_slopes) / len(train_slopes)
    stdev = math.sqrt(var)
    mean_diff = abs(mean - HISTORICAL_SCALER_MEAN_SLOPE)
    scale_diff = abs(stdev - HISTORICAL_SCALER_SCALE_SLOPE)
    # Mean proximity alone is not enough to claim scaler consistency; stdev/scale
    # divergence keeps lineage non-authoritative.
    if mean_diff < 0.01 and scale_diff < 0.5:
        scaler_cmp = "CONSISTENT_WITH_HISTORICAL_SCALER"
    elif mean_diff < 0.01:
        scaler_cmp = "PARTIAL_MEAN_ALIGNMENT_ONLY"
    else:
        scaler_cmp = "INSUFFICIENT_TO_PROVE_LINEAGE"

    return {
        "manifest_version": "1.0",
        "selection_forbidden_signals": [
            "occupancy_label_separation",
            "classifier_accuracy",
            "LOCKED_TEST_model_metrics",
        ],
        "candidates_considered": [
            {
                "candidate_id": "A_PREVIOUS_SAMPLE_RATE",
                "concept": "(CO2_now - CO2_previous) / elapsed_minutes",
                "status": "REJECTED",
                "reason": (
                    "Active deployment adapter retains a multi-sample history window "
                    "and computes endpoint difference against the oldest retained "
                    "sample, not solely the previous sample."
                ),
            },
            {
                "candidate_id": "B_ENDPOINT_DIFFERENCE_HISTORY_DURATION",
                "concept": "(CO2_now - CO2_history_start) / elapsed_minutes",
                "history_duration_seconds": HISTORY_DURATION_SECONDS,
                "status": "SELECTED",
                "offline_baseline_status": "CANONICAL_OFFLINE_BASELINE_DESIGN",
                "reason": (
                    "Endpoint-difference method is VERIFIED in sensors/co2/co2_adapter.py. "
                    "The 150s offline history threshold is a CANONICAL_OFFLINE_BASELINE_DESIGN "
                    "derived from configured/intended window_seconds=150.0, NOT from active "
                    "runtime eligibility (required_history_sec=5.0) and NOT a verified "
                    "historical training duration. Active runtime uses deque(maxlen=30) with "
                    "growing span and ~145s nominal full-buffer endpoint span; window_seconds "
                    "is configured but not applied to the active slope path. Offline UCI "
                    "cadence yields typical selected endpoints of ~179-181s. Causal, "
                    "block-isolated, actual source-clock elapsed time."
                ),
            },
            {
                "candidate_id": "C_LINEAR_REGRESSION_HISTORY_WINDOW",
                "concept": "OLS slope of CO2 vs elapsed_minutes over history window",
                "status": "REJECTED",
                "reason": (
                    "No active repository evidence implements regression slope for "
                    "CO2_slope; adapter and SENSOR_DATA_CONTRACT document endpoint "
                    "difference only."
                ),
            },
        ],
        "selected_candidate_id": "B_ENDPOINT_DIFFERENCE_HISTORY_DURATION",
        "evidence_basis": [
            "sensors/co2/co2_adapter.py calculate_co2_slope (ENDPOINT_DIFFERENCE verified; required_history_sec=5.0 on read())",
            "sensors/co2/co2_adapter.py window_seconds=150.0 CONFIGURED_BUT_NOT_APPLIED to slope eligibility/endpoint",
            "docs/reports/SENSOR_DATA_CONTRACT.md Slope Calculation row (method evidence; history duration not runtime-proof)",
            "docs/reports/sensor_model_data_contract.json history_maxlen/sampling",
            "models/co2/co2_scaling_metadata_v0.1.0.json feature name presence only",
            "C-A2 cadence profile (59–61s jitter requires actual elapsed time)",
            "roadmap C-B1 retains history-length / slope-method ablation; C-C retains SCD40 domain alignment",
        ],
        "train_only_secondary_scaler_diagnostic": {
            "historical_scaler_mean_co2_slope": HISTORICAL_SCALER_MEAN_SLOPE,
            "historical_scaler_scale_co2_slope": HISTORICAL_SCALER_SCALE_SLOPE,
            "reconstructed_train_mean": mean,
            "reconstructed_train_stdev": stdev,
            "abs_mean_difference": mean_diff,
            "abs_stdev_scale_difference": scale_diff,
            "comparison_result": scaler_cmp,
            "authoritative_for_formula_lock": False,
            "authoritative_for_history_lineage": False,
            "note": (
                "Secondary non-authoritative diagnostic only. Mean proximity without "
                "matching scale/stdev is PARTIAL_MEAN_ALIGNMENT_ONLY / insufficient to "
                "prove historical training history lineage. Method lock uses verified "
                "endpoint-difference evidence; 150s offline threshold is an explicit "
                "canonical baseline design, not scaler matching."
            ),
        },
    }


def build_exceptions_registry(scaler_cmp: str) -> Dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "phase": "C-A3",
        "warnings": [
            {
                "code": "SOURCE_TIMEZONE_UNVERIFIED",
                "severity": "WARNING",
                "description": (
                    "Source timestamps remain timezone-naive SOURCE_ACQUISITION_CLOCK "
                    "readings; UTC conversion is not claimed."
                ),
            },
            {
                "code": "MODEL_TRAINING_LINEAGE_UNVERIFIED",
                "severity": "WARNING",
                "description": (
                    "Existing TFLite training script/dataset provenance remains unverified."
                ),
            },
            {
                "code": "SCALER_FIT_LINEAGE_UNVERIFIED",
                "severity": "WARNING",
                "description": (
                    "Existing scaling metadata fit lineage remains unverified; C-A3 "
                    "does not fit or overwrite scaler statistics."
                ),
            },
            {
                "code": "GROUP_INDEPENDENCE_NOT_VERIFIABLE",
                "severity": "WARNING",
                "description": (
                    "All temporal blocks originate from a single office room."
                ),
            },
            {
                "code": "CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED",
                "severity": "WARNING",
                "description": (
                    "Historical training history duration/formula for CO2_slope remains "
                    "unverified. C-A3 locks ENDPOINT_DIFFERENCE as method and retains "
                    "150s as CANONICAL_OFFLINE_BASELINE_DESIGN only, not as "
                    "VERIFIED_HISTORICAL_TRAINING_CONTRACT or ACTIVE_RUNTIME_EQUIVALENT."
                ),
            },
            {
                "code": "ADAPTER_WINDOW_SECONDS_NOT_APPLIED_TO_ACTIVE_SLOPE_PATH",
                "severity": "WARNING",
                "description": (
                    "sensors/co2/co2_adapter.py configures window_seconds=150.0 but the "
                    "active read()/calculate_co2_slope path uses required_history_sec=5.0 "
                    "and deque(maxlen=30) oldest-to-current endpoints. window_seconds is "
                    "CONFIGURED_BUT_NOT_APPLIED to slope eligibility or endpoint selection. "
                    "Nominal full-buffer span at 0.2 Hz is ~145s, not a fixed 150s contract."
                ),
            },
            {
                "code": "DEVICE_UCI_CADENCE_DOMAIN_GAP",
                "severity": "WARNING",
                "description": (
                    "Active SCD40 adapter samples at 0.2 Hz with maxlen=30, while UCI "
                    "source cadence is ~60s. Offline C-A3 150s threshold therefore yields "
                    "typical selected endpoints of ~179-181s, distinct from runtime "
                    "eligibility (~5s) and full-buffer span (~145s). C-B1/C-C remain "
                    "responsible for history ablation and device-domain alignment."
                ),
            },
            {
                "code": f"HISTORICAL_SCALER_COMPARISON_{scaler_cmp}",
                "severity": "WARNING",
                "description": (
                    f"TRAIN reconstructed slope vs historical scaler diagnostic: {scaler_cmp} "
                    "(non-authoritative; does not prove history lineage)."
                ),
            },
        ],
        "blockers": [],
        "deferred_shared_integration_updates": [
            {
                "code": "DEFERRED_SHARED_INTEGRATION_UPDATE",
                "targets": [
                    "docs/reports/SENSOR_DATA_CONTRACT.md",
                    "docs/reports/sensor_model_data_contract.json",
                    "models/model_manifest.json",
                    "datasets/MANIFEST.json",
                ],
                "reason": (
                    "Shared inventory/contract refresh belongs to a later approved "
                    "integration commit after C-A3 review."
                ),
            }
        ],
    }


def build_lineage_contract() -> Dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "feature_contract_id": FEATURE_PROFILE_ID,
        "lineage_reconstruction_rule": (
            "Every FEATURE_AVAILABLE slope is fully determined by the target source "
            "row plus the contiguous same-block past window from history_start through "
            "target, using SOURCE_ACQUISITION_CLOCK timestamps and raw CO2 values."
        ),
        "required_lineage_fields": [
            "target_source_member",
            "target_source_row_identifier",
            "target_physical_line",
            "target_timestamp_raw",
            "temporal_block_id",
            "feature_name",
            "feature_contract_id",
            "history_start_source_row_identifier",
            "history_end_source_row_identifier",
            "history_start_timestamp",
            "history_end_timestamp",
            "history_elapsed_seconds",
            "source_sample_count_used",
            "slope_method",
            "slope_unit",
            "feature_status",
            "co2_slope",
        ],
        "compact_evidence_strategy": (
            "Full per-row lineage is regenerable via "
            "datasets.co2.slope_feature.reconstruct_all_slope_features; C-A3 persists "
            "compact eligibility summary, manual verification lineages, and checksums "
            "instead of a giant redundant row dump."
        ),
        "absolute_paths_forbidden": True,
        "cross_block_history_forbidden": True,
        "future_sample_use_forbidden": True,
    }


def audit_co2_slope_feature() -> Path:
    repo_root = get_repo_root()
    out_dir = repo_root / "datasets/co2/manifests/c_a3_slope_feature"
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = UCIOccupancyRawReader(repo_root=repo_root)
    observations = reader.read_all_observations()
    if len(observations) != 20560:
        raise RuntimeError(f"Expected 20560 source rows, got {len(observations)}")

    records = reconstruct_all_slope_features(observations)
    if len(records) != 20560:
        raise RuntimeError(f"Silent row loss detected: {len(records)} records")

    profile = build_feature_profile()
    eligibility = summarize_eligibility(records)
    audit_stats = compute_slope_audit_statistics(
        records, include_value_stats_for_locked_test=False
    )
    manual = build_manual_verification_cases(observations, records)
    if not manual["all_cases_pass"]:
        failed = [c["case_id"] for c in manual["cases"] if not c["pass"]]
        raise RuntimeError(f"Manual verification failed: {failed}")

    train_records = [r for r in records if r.future_split_role == "TRAIN"]
    candidate_eval = build_candidate_evaluation(train_records)
    scaler_cmp = candidate_eval["train_only_secondary_scaler_diagnostic"][
        "comparison_result"
    ]
    exceptions = build_exceptions_registry(scaler_cmp)
    lineage_contract = build_lineage_contract()

    generation_meta = {
        "manifest_version": "1.0",
        "phase": "C-A3",
        "feature_contract_id": FEATURE_PROFILE_ID,
        "generator_script": "scripts/audit_co2_slope_feature.py",
        "reconstruction_module": "datasets/co2/slope_feature.py",
        "total_source_rows": len(observations),
        "total_feature_records": len(records),
        "member_order": MEMBER_ORDER,
        "synthetic_npz_used_as_real_source": False,
        "scaler_fitted": False,
        "model_trained": False,
        "occupancy_labels_used_for_contract_selection": False,
        "locked_test_used_for_contract_selection": False,
        "determinism": {
            "host_timezone_independent": True,
            "locale_independent": True,
            "random_values_used": False,
        },
    }

    _write_json(out_dir / "co2_slope_feature_profile.json", profile)
    _write_json(out_dir / "feature_eligibility_summary.json", eligibility)
    _write_json(out_dir / "feature_audit_statistics.json", audit_stats)
    _write_json(out_dir / "manual_verification_cases.json", manual)
    _write_json(out_dir / "candidate_method_evaluation.json", candidate_eval)
    _write_json(out_dir / "source_row_feature_lineage_contract.json", lineage_contract)
    _write_json(out_dir / "exceptions_and_limitations.json", exceptions)
    _write_json(out_dir / "generation_metadata.json", generation_meta)

    checksum_files = [
        "co2_slope_feature_profile.json",
        "feature_eligibility_summary.json",
        "feature_audit_statistics.json",
        "manual_verification_cases.json",
        "candidate_method_evaluation.json",
        "source_row_feature_lineage_contract.json",
        "exceptions_and_limitations.json",
        "generation_metadata.json",
    ]
    checksum_lines = []
    for fname in checksum_files:
        rel = f"datasets/co2/manifests/c_a3_slope_feature/{fname}"
        checksum_lines.append(f"{compute_sha256_file(out_dir / fname)}  {rel}")
    (out_dir / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )

    print(f"✅ Generated C-A3 slope feature manifests in: {out_dir.relative_to(repo_root)}")
    print(
        f"   rows={eligibility['total_source_rows_represented']} "
        f"eligible={eligibility['eligible_slope_rows']} "
        f"warmup={eligibility['warmup_unavailable_rows']}"
    )
    return out_dir


if __name__ == "__main__":
    audit_co2_slope_feature()
