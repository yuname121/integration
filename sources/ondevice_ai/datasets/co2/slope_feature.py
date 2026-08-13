#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datasets/co2/slope_feature.py
Phase C-A3 — CO₂ Slope Feature Reconstruction and Source-Row Feature Lineage.

Deterministic, causal, block-isolated reconstruction of CO2_slope in ppm/min
from C-A1 source rows under the C-A2 temporal acquisition block contract.

Selected contract (locked):
  profile_id: CO2_SLOPE_FEATURE_PROFILE_001
  method: ENDPOINT_DIFFERENCE (verified runtime method)
  formula: (co2_now - co2_history_start) / (elapsed_seconds / 60.0)
  history_duration_seconds: 150.0 (CANONICAL_OFFLINE_BASELINE_DESIGN; not active-runtime equivalent)
  timestamp_basis: SOURCE_ACQUISITION_CLOCK
  causality: PAST_ONLY
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets.co2.raw_reader import CO2SourceRowObservation

FEATURE_PROFILE_ID = "CO2_SLOPE_FEATURE_PROFILE_001"
FEATURE_NAME = "CO2_slope"
FEATURE_UNIT = "ppm/min"
SLOPE_METHOD = "ENDPOINT_DIFFERENCE"
HISTORY_DURATION_SECONDS = 150.0
HISTORY_DURATION_MINUTES = 2.5
MINIMUM_SOURCE_SAMPLES = 2
MINIMUM_ELAPSED_SECONDS = 150.0
MAX_INTERNAL_GAP_SECONDS = 90.0
CALCULATION_DTYPE = "float64"
COMPARISON_ABS_TOLERANCE = 1e-12
TIMESTAMP_BASIS = "SOURCE_ACQUISITION_CLOCK"
CAUSALITY = "PAST_ONLY"

STATUS_AVAILABLE = "FEATURE_AVAILABLE"
STATUS_WARMUP = "FEATURE_UNAVAILABLE_WARMUP"
STATUS_GAP_RESTART = "FEATURE_UNAVAILABLE_GAP_RESTART"
STATUS_NONFINITE = "FEATURE_UNAVAILABLE_NONFINITE_INPUT"
STATUS_NON_MONOTONIC = "FEATURE_UNAVAILABLE_NON_MONOTONIC_TIMESTAMP"

MEMBER_TO_BLOCK = {
    "datatest.txt": ("BLOCK_01_DATATEST", "VALIDATION"),
    "datatraining.txt": ("BLOCK_02_DATATRAINING", "TRAIN"),
    "datatest2.txt": ("BLOCK_03_DATATEST2", "LOCKED_TEST"),
}

MEMBER_ORDER = ["datatest.txt", "datatraining.txt", "datatest2.txt"]


def build_feature_profile() -> Dict[str, Any]:
    """Return the locked machine-readable CO2_slope feature profile."""
    return {
        "manifest_version": "1.0",
        "profile_id": FEATURE_PROFILE_ID,
        "feature_name": FEATURE_NAME,
        "feature_unit": FEATURE_UNIT,
        "slope_method": SLOPE_METHOD,
        "formula": (
            "(co2_now - co2_history_start) / (elapsed_source_clock_seconds / 60.0)"
        ),
        "formula_description": (
            "Endpoint difference over the earliest same-block past observation whose "
            "source-clock age is at least history_duration_seconds; elapsed time uses "
            "actual SOURCE_ACQUISITION_CLOCK deltas (not a fixed 60.0s assumption)."
        ),
        "causality": CAUSALITY,
        "timestamp_basis": TIMESTAMP_BASIS,
        "source_timezone": "UNVERIFIED",
        "utc_conversion_claimed": False,
        "history_duration_seconds": HISTORY_DURATION_SECONDS,
        "history_duration_minutes": HISTORY_DURATION_MINUTES,
        "minimum_source_samples": MINIMUM_SOURCE_SAMPLES,
        "minimum_elapsed_seconds": MINIMUM_ELAPSED_SECONDS,
        "nominal_source_cadence_seconds": 60.0,
        "nominal_history_sample_span_at_source_cadence": 3,
        "offline_baseline_status": "CANONICAL_OFFLINE_BASELINE_DESIGN",
        "offline_baseline_equivalence_claims": {
            "verified_historical_training_contract": False,
            "active_runtime_equivalent": False,
        },
        "runtime_evidence": {
            "evidence_path": "sensors/co2/co2_adapter.py",
            "runtime_slope_method_verified": "ENDPOINT_DIFFERENCE",
            "runtime_history_maxlen": 30,
            "runtime_required_history_sec": 5.0,
            "configured_window_seconds": 150.0,
            "configured_window_seconds_applied_to_slope_logic": False,
            "nominal_full_buffer_endpoint_span_seconds": 145.0,
            "nominal_full_buffer_span_notes": (
                "At sample_rate_hz=0.2, a full maxlen=30 deque spans approximately "
                "29 intervals x 5s = 145s from oldest retained to current, not a "
                "guaranteed fixed 150s window. Active read() eligibility uses "
                "required_history_sec=5.0; window_seconds is configured but not "
                "applied to slope eligibility or endpoint selection."
            ),
        },
        "offline_canonical_history_threshold_seconds": HISTORY_DURATION_SECONDS,
        "offline_effective_endpoint_span_due_to_source_cadence": {
            "typical_seconds_range": [179.0, 181.0],
            "explanation": (
                "UCI nominal cadence is ~60s. Under the past-sample-only rule that "
                "selects the earliest same-block observation with age >= 150s, the "
                "selected endpoint is normally three intervals back (~179-181s)."
            ),
        },
        "historical_training_history_contract_status": "CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED",
        "offline_baseline_derivation": (
            "150s is retained as a CANONICAL_OFFLINE_BASELINE_DESIGN derived from the "
            "configured/intended adapter window_seconds value. It is NOT verified as "
            "the historical training feature duration and is NOT active-runtime "
            "equivalent. Endpoint-difference method is the verified runtime method; "
            "history-length ablation remains C-B1; SCD40 domain alignment remains C-C."
        ),
        "boundary_policy": "DERIVED_TEMPORAL_FEATURES_MUST_NOT_CROSS_BLOCK_BOUNDARIES",
        "gap_policy": "RESTART_HISTORY_AFTER_FORBIDDEN_GAP",
        "max_internal_gap_seconds": MAX_INTERNAL_GAP_SECONDS,
        "warm_up_policy": "PRESERVE_ROW_WITH_NULL_SLOPE",
        "warm_up_status": STATUS_WARMUP,
        "gap_restart_status": STATUS_GAP_RESTART,
        "nonfinite_policy": "FAIL_CLOSED_STATUS_NO_CANONICAL_SLOPE",
        "output_type": "float64_or_null",
        "calculation_precision": CALCULATION_DTYPE,
        "comparison_abs_tolerance": COMPARISON_ABS_TOLERANCE,
        "interpolation_allowed": False,
        "future_samples_allowed": False,
        "centered_window_allowed": False,
        "scaler_fitting_in_phase": False,
        "model_training_in_phase": False,
        "occupancy_label_used_for_contract_selection": False,
        "locked_test_used_for_contract_selection": False,
    }


@dataclass(frozen=True)
class SlopeFeatureRecord:
    """One source-row feature eligibility/lineage record."""

    target_source_member: str
    target_source_row_identifier: str
    target_physical_line: int
    target_timestamp_raw: str
    target_timestamp_canonical: str
    temporal_block_id: str
    future_split_role: str
    feature_name: str
    feature_contract_id: str
    history_start_source_row_identifier: Optional[str]
    history_end_source_row_identifier: Optional[str]
    history_start_physical_line: Optional[int]
    history_end_physical_line: Optional[int]
    history_start_timestamp_raw: Optional[str]
    history_end_timestamp_raw: Optional[str]
    history_start_timestamp_canonical: Optional[str]
    history_end_timestamp_canonical: Optional[str]
    history_elapsed_seconds: Optional[float]
    source_sample_count_used: int
    history_source_row_identifiers: Tuple[str, ...]
    slope_method: str
    slope_unit: str
    feature_status: str
    co2_slope: Optional[float]
    co2_now: float
    co2_history_start: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["history_source_row_identifiers"] = list(self.history_source_row_identifiers)
        return payload


@dataclass(frozen=True)
class _TimedRow:
    obs: CO2SourceRowObservation
    dt: datetime
    ts_canonical: str
    block_id: str
    future_split_role: str
    index_in_block: int


def parse_source_timestamp(raw: str) -> Tuple[datetime, str]:
    dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    return dt, dt.strftime("%Y-%m-%dT%H:%M:%S")


def _is_finite_number(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def compute_endpoint_slope_ppm_per_min(
    co2_now: float,
    co2_history_start: float,
    elapsed_seconds: float,
) -> float:
    """Independently usable endpoint-difference slope in ppm/min (float64)."""
    if not _is_finite_number(co2_now) or not _is_finite_number(co2_history_start):
        raise ValueError("CO2 values must be finite")
    if not _is_finite_number(elapsed_seconds) or elapsed_seconds <= 0.0:
        raise ValueError("elapsed_seconds must be a positive finite value")
    elapsed_minutes = float(elapsed_seconds) / 60.0
    return float((float(co2_now) - float(co2_history_start)) / elapsed_minutes)


def compute_causal_linear_regression_slope_ppm_per_min(
    co2_values: Sequence[float],
    elapsed_seconds_from_anchor: Sequence[float],
) -> float:
    """
    Fit CO2 = intercept + slope * elapsed_minutes on causal history support.

    elapsed_seconds_from_anchor must be actual SOURCE_ACQUISITION_CLOCK deltas
    from the history anchor (not sample_count * nominal period). Returns slope
    in ppm/min as float64. Requires at least two distinct timestamps.
    """
    if len(co2_values) != len(elapsed_seconds_from_anchor):
        raise ValueError("co2_values and elapsed_seconds_from_anchor length mismatch")
    if len(co2_values) < 2:
        raise ValueError("linear regression requires at least two samples")
    xs = [float(v) / 60.0 for v in elapsed_seconds_from_anchor]
    ys = [float(v) for v in co2_values]
    if any(not _is_finite_number(v) for v in xs + ys):
        raise ValueError("non-finite regression inputs")
    if len({x for x in xs}) < 2:
        raise ValueError("linear regression requires at least two distinct timestamps")
    n = float(len(xs))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0.0:
        raise ValueError("zero time variance")
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = float(cov_xy / var_x)
    if not _is_finite_number(slope):
        raise ValueError("non-finite regression slope")
    return slope


def _group_observations_by_block(
    observations: Sequence[CO2SourceRowObservation],
) -> Dict[str, List[_TimedRow]]:
    grouped: Dict[str, List[_TimedRow]] = {m: [] for m in MEMBER_ORDER}
    for obs in observations:
        if obs.source_member_name not in MEMBER_TO_BLOCK:
            raise ValueError(f"Unexpected source member: {obs.source_member_name}")
        block_id, role = MEMBER_TO_BLOCK[obs.source_member_name]
        dt, ts_canonical = parse_source_timestamp(obs.source_timestamp_raw)
        grouped[obs.source_member_name].append(
            _TimedRow(
                obs=obs,
                dt=dt,
                ts_canonical=ts_canonical,
                block_id=block_id,
                future_split_role=role,
                index_in_block=len(grouped[obs.source_member_name]),
            )
        )
    return grouped


def _find_history_start(
    block_rows: Sequence[_TimedRow],
    target_index: int,
    *,
    history_duration_seconds: float = HISTORY_DURATION_SECONDS,
) -> Tuple[Optional[int], str]:
    """
    Walk backward within the same block only.

    Returns (history_start_index, unavailable_status_if_none).
    Crossing a forbidden internal gap truncates usable history.

    history_duration_seconds is a minimum-history threshold: the first past
    sample (walking backward from the current row) whose actual source-clock
    age is at least this threshold becomes the history anchor.
    """
    if target_index <= 0:
        return None, STATUS_WARMUP

    crossed_gap = False
    for k in range(target_index - 1, -1, -1):
        delta = (block_rows[k + 1].dt - block_rows[k].dt).total_seconds()
        if delta <= 0.0:
            return None, STATUS_NON_MONOTONIC
        if delta > MAX_INTERNAL_GAP_SECONDS:
            crossed_gap = True
            break
        elapsed = (block_rows[target_index].dt - block_rows[k].dt).total_seconds()
        if elapsed >= float(history_duration_seconds):
            return k, STATUS_AVAILABLE

    if crossed_gap:
        return None, STATUS_GAP_RESTART
    return None, STATUS_WARMUP


def reconstruct_block_slope_features_with_params(
    block_rows: Sequence[_TimedRow],
    *,
    method: str = SLOPE_METHOD,
    history_duration_seconds: float = HISTORY_DURATION_SECONDS,
    feature_contract_id: str = FEATURE_PROFILE_ID,
) -> List[SlopeFeatureRecord]:
    """
    Reconstruct CO2_slope for one temporal acquisition block.

    Preserves C-A3 causal / gap / restart semantics. `history_duration_seconds`
    is a minimum-history threshold using actual source-clock elapsed time.
    Supported methods:
      - ENDPOINT_DIFFERENCE
      - CAUSAL_LINEAR_REGRESSION
    """
    if method not in ("ENDPOINT_DIFFERENCE", "CAUSAL_LINEAR_REGRESSION"):
        raise ValueError(f"Unsupported slope method: {method}")

    records: List[SlopeFeatureRecord] = []
    for i, row in enumerate(block_rows):
        obs = row.obs
        co2_now = float(obs.co2)

        if not _is_finite_number(co2_now):
            records.append(
                SlopeFeatureRecord(
                    target_source_member=obs.source_member_name,
                    target_source_row_identifier=obs.source_row_identifier,
                    target_physical_line=obs.source_physical_line_number,
                    target_timestamp_raw=obs.source_timestamp_raw,
                    target_timestamp_canonical=row.ts_canonical,
                    temporal_block_id=row.block_id,
                    future_split_role=row.future_split_role,
                    feature_name=FEATURE_NAME,
                    feature_contract_id=feature_contract_id,
                    history_start_source_row_identifier=None,
                    history_end_source_row_identifier=None,
                    history_start_physical_line=None,
                    history_end_physical_line=None,
                    history_start_timestamp_raw=None,
                    history_end_timestamp_raw=None,
                    history_start_timestamp_canonical=None,
                    history_end_timestamp_canonical=None,
                    history_elapsed_seconds=None,
                    source_sample_count_used=0,
                    history_source_row_identifiers=tuple(),
                    slope_method=method,
                    slope_unit=FEATURE_UNIT,
                    feature_status=STATUS_NONFINITE,
                    co2_slope=None,
                    co2_now=co2_now,
                    co2_history_start=None,
                )
            )
            continue

        history_start_idx, unavailable_status = _find_history_start(
            block_rows,
            i,
            history_duration_seconds=float(history_duration_seconds),
        )
        if history_start_idx is None:
            records.append(
                SlopeFeatureRecord(
                    target_source_member=obs.source_member_name,
                    target_source_row_identifier=obs.source_row_identifier,
                    target_physical_line=obs.source_physical_line_number,
                    target_timestamp_raw=obs.source_timestamp_raw,
                    target_timestamp_canonical=row.ts_canonical,
                    temporal_block_id=row.block_id,
                    future_split_role=row.future_split_role,
                    feature_name=FEATURE_NAME,
                    feature_contract_id=feature_contract_id,
                    history_start_source_row_identifier=None,
                    history_end_source_row_identifier=None,
                    history_start_physical_line=None,
                    history_end_physical_line=None,
                    history_start_timestamp_raw=None,
                    history_end_timestamp_raw=None,
                    history_start_timestamp_canonical=None,
                    history_end_timestamp_canonical=None,
                    history_elapsed_seconds=None,
                    source_sample_count_used=1,
                    history_source_row_identifiers=(obs.source_row_identifier,),
                    slope_method=method,
                    slope_unit=FEATURE_UNIT,
                    feature_status=unavailable_status,
                    co2_slope=None,
                    co2_now=co2_now,
                    co2_history_start=None,
                )
            )
            continue

        window = block_rows[history_start_idx : i + 1]
        if any(not _is_finite_number(float(w.obs.co2)) for w in window):
            records.append(
                SlopeFeatureRecord(
                    target_source_member=obs.source_member_name,
                    target_source_row_identifier=obs.source_row_identifier,
                    target_physical_line=obs.source_physical_line_number,
                    target_timestamp_raw=obs.source_timestamp_raw,
                    target_timestamp_canonical=row.ts_canonical,
                    temporal_block_id=row.block_id,
                    future_split_role=row.future_split_role,
                    feature_name=FEATURE_NAME,
                    feature_contract_id=feature_contract_id,
                    history_start_source_row_identifier=None,
                    history_end_source_row_identifier=None,
                    history_start_physical_line=None,
                    history_end_physical_line=None,
                    history_start_timestamp_raw=None,
                    history_end_timestamp_raw=None,
                    history_start_timestamp_canonical=None,
                    history_end_timestamp_canonical=None,
                    history_elapsed_seconds=None,
                    source_sample_count_used=0,
                    history_source_row_identifiers=tuple(),
                    slope_method=method,
                    slope_unit=FEATURE_UNIT,
                    feature_status=STATUS_NONFINITE,
                    co2_slope=None,
                    co2_now=co2_now,
                    co2_history_start=None,
                )
            )
            continue

        start_row = block_rows[history_start_idx]
        co2_start = float(start_row.obs.co2)
        elapsed_seconds = float((row.dt - start_row.dt).total_seconds())
        if method == "ENDPOINT_DIFFERENCE":
            slope = compute_endpoint_slope_ppm_per_min(co2_now, co2_start, elapsed_seconds)
        else:
            co2_vals = [float(w.obs.co2) for w in window]
            elapsed_vals = [
                float((w.dt - start_row.dt).total_seconds()) for w in window
            ]
            slope = compute_causal_linear_regression_slope_ppm_per_min(
                co2_vals, elapsed_vals
            )
        if not _is_finite_number(slope):
            raise ValueError(
                f"Nonfinite slope produced for {obs.source_member_name}:"
                f"{obs.source_row_identifier}"
            )

        records.append(
            SlopeFeatureRecord(
                target_source_member=obs.source_member_name,
                target_source_row_identifier=obs.source_row_identifier,
                target_physical_line=obs.source_physical_line_number,
                target_timestamp_raw=obs.source_timestamp_raw,
                target_timestamp_canonical=row.ts_canonical,
                temporal_block_id=row.block_id,
                future_split_role=row.future_split_role,
                feature_name=FEATURE_NAME,
                feature_contract_id=feature_contract_id,
                history_start_source_row_identifier=start_row.obs.source_row_identifier,
                history_end_source_row_identifier=obs.source_row_identifier,
                history_start_physical_line=start_row.obs.source_physical_line_number,
                history_end_physical_line=obs.source_physical_line_number,
                history_start_timestamp_raw=start_row.obs.source_timestamp_raw,
                history_end_timestamp_raw=obs.source_timestamp_raw,
                history_start_timestamp_canonical=start_row.ts_canonical,
                history_end_timestamp_canonical=row.ts_canonical,
                history_elapsed_seconds=elapsed_seconds,
                source_sample_count_used=len(window),
                history_source_row_identifiers=tuple(
                    w.obs.source_row_identifier for w in window
                ),
                slope_method=method,
                slope_unit=FEATURE_UNIT,
                feature_status=STATUS_AVAILABLE,
                co2_slope=slope,
                co2_now=co2_now,
                co2_history_start=co2_start,
            )
        )
    return records


def reconstruct_block_slope_features(
    block_rows: Sequence[_TimedRow],
) -> List[SlopeFeatureRecord]:
    """Reconstruct CO2_slope for one temporal acquisition block (C-A3 baseline)."""
    return reconstruct_block_slope_features_with_params(block_rows)


def reconstruct_all_slope_features_with_params(
    observations: Sequence[CO2SourceRowObservation],
    *,
    method: str = SLOPE_METHOD,
    history_duration_seconds: float = HISTORY_DURATION_SECONDS,
    feature_contract_id: str = FEATURE_PROFILE_ID,
) -> List[SlopeFeatureRecord]:
    """Reconstruct CO2_slope for all source rows under explicit candidate params."""
    grouped = _group_observations_by_block(observations)
    all_records: List[SlopeFeatureRecord] = []
    for member in MEMBER_ORDER:
        all_records.extend(
            reconstruct_block_slope_features_with_params(
                grouped[member],
                method=method,
                history_duration_seconds=history_duration_seconds,
                feature_contract_id=feature_contract_id,
            )
        )
    return all_records


def reconstruct_all_slope_features(
    observations: Sequence[CO2SourceRowObservation],
) -> List[SlopeFeatureRecord]:
    """
    Reconstruct CO2_slope for all source rows.

    Preserves C-A1 member order (datatest, datatraining, datatest2) and
    within-member source order. Never crosses temporal block boundaries.
    """
    grouped = _group_observations_by_block(observations)
    all_records: List[SlopeFeatureRecord] = []
    for member in MEMBER_ORDER:
        all_records.extend(reconstruct_block_slope_features(grouped[member]))
    return all_records


def summarize_eligibility(records: Sequence[SlopeFeatureRecord]) -> Dict[str, Any]:
    """Deterministic eligibility/warm-up summary by block and role."""
    by_block: Dict[str, Dict[str, Any]] = {}
    for record in records:
        block = by_block.setdefault(
            record.temporal_block_id,
            {
                "temporal_block_id": record.temporal_block_id,
                "future_split_role": record.future_split_role,
                "source_member_name": record.target_source_member,
                "source_row_count": 0,
                "eligible_slope_count": 0,
                "warmup_unavailable_count": 0,
                "gap_restart_unavailable_count": 0,
                "nonfinite_unavailable_count": 0,
                "non_monotonic_unavailable_count": 0,
                "null_slope_count": 0,
                "finite_slope_count": 0,
                "nonfinite_slope_output_count": 0,
            },
        )
        block["source_row_count"] += 1
        if record.feature_status == STATUS_AVAILABLE:
            block["eligible_slope_count"] += 1
            if record.co2_slope is None or not math.isfinite(record.co2_slope):
                block["nonfinite_slope_output_count"] += 1
            else:
                block["finite_slope_count"] += 1
        else:
            block["null_slope_count"] += 1
            if record.feature_status == STATUS_WARMUP:
                block["warmup_unavailable_count"] += 1
            elif record.feature_status == STATUS_GAP_RESTART:
                block["gap_restart_unavailable_count"] += 1
            elif record.feature_status == STATUS_NONFINITE:
                block["nonfinite_unavailable_count"] += 1
            elif record.feature_status == STATUS_NON_MONOTONIC:
                block["non_monotonic_unavailable_count"] += 1

    total_rows = len(records)
    eligible = sum(1 for r in records if r.feature_status == STATUS_AVAILABLE)
    warmup = sum(1 for r in records if r.feature_status == STATUS_WARMUP)
    return {
        "manifest_version": "1.0",
        "feature_contract_id": FEATURE_PROFILE_ID,
        "total_source_rows_represented": total_rows,
        "eligible_slope_rows": eligible,
        "warmup_unavailable_rows": warmup,
        "gap_restart_unavailable_rows": sum(
            1 for r in records if r.feature_status == STATUS_GAP_RESTART
        ),
        "nonfinite_unavailable_rows": sum(
            1 for r in records if r.feature_status == STATUS_NONFINITE
        ),
        "silent_row_loss": 0,
        "rows_omitted": 0,
        "by_block": [by_block[k] for k in sorted(by_block.keys())],
    }


def compute_slope_audit_statistics(
    records: Sequence[SlopeFeatureRecord],
    *,
    include_value_stats_for_locked_test: bool = False,
) -> Dict[str, Any]:
    """
    Descriptive slope audit statistics by future split role.

    LOCKED_TEST value statistics are optional; by default only eligibility
    integrity is reported for LOCKED_TEST to avoid feature-definition feedback.
    """

    def _role_stats(role: str, allow_values: bool) -> Dict[str, Any]:
        role_records = [r for r in records if r.future_split_role == role]
        eligible = [r for r in role_records if r.feature_status == STATUS_AVAILABLE]
        slopes = [float(r.co2_slope) for r in eligible if r.co2_slope is not None]
        finite = [s for s in slopes if math.isfinite(s)]
        payload: Dict[str, Any] = {
            "future_split_role": role,
            "source_row_count": len(role_records),
            "eligible_slope_count": len(eligible),
            "warmup_unavailable_count": sum(
                1 for r in role_records if r.feature_status == STATUS_WARMUP
            ),
            "unavailable_count": len(role_records) - len(eligible),
            "finite_count": len(finite),
            "nonfinite_count": len(slopes) - len(finite),
        }
        if allow_values and finite:
            mean = sum(finite) / len(finite)
            var = sum((x - mean) ** 2 for x in finite) / len(finite)
            sorted_vals = sorted(finite)
            mid = len(sorted_vals) // 2
            if len(sorted_vals) % 2:
                median = sorted_vals[mid]
            else:
                median = 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])
            payload.update(
                {
                    "minimum": float(min(finite)),
                    "maximum": float(max(finite)),
                    "mean": float(mean),
                    "median": float(median),
                    "standard_deviation": float(math.sqrt(var)),
                    "value_statistics_included": True,
                }
            )
        else:
            payload["value_statistics_included"] = False
            payload["value_statistics_omission_reason"] = (
                "LOCKED_TEST descriptive value inspection deferred from feature-definition feedback"
                if role == "LOCKED_TEST" and not allow_values
                else "no eligible finite slopes"
            )
        return payload

    return {
        "manifest_version": "1.0",
        "feature_contract_id": FEATURE_PROFILE_ID,
        "statistics_role": "DESCRIPTIVE_FEATURE_AUDIT_ONLY",
        "not_model_performance": True,
        "by_role": {
            "TRAIN": _role_stats("TRAIN", True),
            "VALIDATION": _role_stats("VALIDATION", True),
            "LOCKED_TEST": _role_stats("LOCKED_TEST", include_value_stats_for_locked_test),
        },
    }
