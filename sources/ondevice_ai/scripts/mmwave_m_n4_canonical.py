#!/usr/bin/env python3
"""M-N4 canonical R2 input transform, subject split, and freeze helpers.

This module is the executable counterpart of
config/mmwave/m_n4_canonical_input_dataset_contract.json.
It does not train, does not reopen M-N3 research, and must not inspect
NEW_MODEL_HELDOUT_TEST performance.

Integration copy: ROOT is sources/ondevice_ai/. Public A6 freeze inputs and
standalone sensors/ drivers are intentionally not imported. Default CLI is
the contract self-check only.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ID = "MMWAVE_MR60_COMPAT_INPUT_DATASET_V1"
SPLIT_PROFILE_ID = "MMWAVE_MR60_COMPAT_SUBJECT_SPLIT_V1"
A4_PROFILE_ID = "MMWAVE_LABEL_MAPPING_PROFILE_001"
A2_PROFILE_ID = "MMWAVE_PHASE_EXTRACTION_PROFILE_001"
A1_PROFILE_ID = "RFFT_DECODER_PROFILE_001"
A6_WINDOW_MANIFEST = ROOT / "datasets/mmwave/manifests/a6_full_conversion/full_window_manifest.jsonl"
A6_PROVENANCE = ROOT / "datasets/mmwave/manifests/a6_full_conversion/full_provenance_manifest.jsonl"

RATE_HZ = 8.0
WINDOW_SECONDS = 30.0
SAMPLE_COUNT = 240
UPDATE_ADVANCE_TOLERANCE_MS = 8.0
GAP_FLOOR_S = 0.40
GAP_MULTIPLE = 4.0
MIN_INTERVALS_FOR_MEDIAN = 8
MAD_EPSILON = 1e-6
EDGE_HOLD_MAX_SECONDS = 0.250
DT_MIN_S = 1e-6
SPLIT_SEED = 20260818
SPLIT_NAMESPACE = CONTRACT_ID
SPLITS = ("TRAIN", "VAL", "NEW_MODEL_HELDOUT_TEST")
TARGET_RATIOS = {"TRAIN": 0.70, "VAL": 0.15, "NEW_MODEL_HELDOUT_TEST": 0.15}
CLASS_TO_ID = {"NORMAL": 0, "RAPID_OR_ABNORMAL": 1, "APNEA": 2}

# Same-subject limited device reference reserved for M-N7. Not unseen-person GT.
MR60_HELDOUT_REFERENCE = [
    "LEGACY_2026-07-28_empty_v2_360s",
    "LEGACY_2026-07-25_occupied_d09_60s",
    "LEGACY_2026-07-25_occupied_front_d06_60s",
]
MR60_MN2_MN3_DEVELOPMENT_REFERENCE = [
    "LEGACY_2026-07-25_occupied_d06_v1_360s",
    "LEGACY_2026-07-25_occupied_d09_v1_360s",
    "LEGACY_2026-07-28_occupied_d09_v2_360s",
    "LEGACY_2026-08-01_occupied_d09_v120_31min",
    "LEGACY_2026-08-01_empty_v120_30min",
    "LEGACY_2026-07-25_empty_gate_v1_360s",
    "M-C0-PILOT-DESKWORK-001",
    "LEGACY_2026-07-26_breath_paced_15rpm",
    "LEGACY_2026-07-28_breath_paced_12rpm_explicit_v2_attempt03",
    "LEGACY_2026-07-26_breath_paced_20rpm_deep",
]


class CanonicalContractError(ValueError):
    """Raised when a candidate window cannot be formed under the frozen contract."""


@dataclass(frozen=True)
class CanonicalWindow:
    values: np.ndarray
    t_grid_s: np.ndarray
    t_start_s: float
    mad: float
    collapsed: bool
    n_phase_events: int
    n_derivative_samples: int
    median_update_dt_s: float
    gap_threshold_s: float
    notes: tuple[str, ...]


def canonical_grid(t_start_s: float) -> np.ndarray:
    return t_start_s + np.arange(SAMPLE_COUNT, dtype=np.float64) / RATE_HZ


def window_mad(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=np.float64)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    return float(np.median(np.abs(finite - median)))


def apply_s1(values: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """Window-local MAD: divide-only, no centering.

    normalized = r / MAD. Not (r - median(r)) / MAD.
    MAD < 1e-6 is a numerical guard against divide-by-near-zero, not an occupancy threshold.
    """
    y = np.asarray(values, dtype=np.float64)
    mad = window_mad(y)
    if mad < MAD_EPSILON:
        return np.zeros(y.shape, dtype=np.float32), mad, True
    return (y / mad).astype(np.float32), mad, False


def phase_update_estimate_ms(ts_monotonic_ms: float, phase_age_ms: float) -> float:
    return float(ts_monotonic_ms) - float(phase_age_ms)


def accept_phase_events(
    ts: np.ndarray,
    phase: np.ndarray,
    phase_age_ms: np.ndarray | None,
    *,
    production: bool,
    timestamps_are_seconds: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Accept genuine phase-update events.

    Public native frames: timestamps_are_seconds=True, phase_age_ms=None.
    Production MR60: timestamps_are_seconds=False (ms), phase_age_ms required.
    """
    t_raw = np.asarray(ts, dtype=np.float64)
    x = np.asarray(phase, dtype=np.float64)
    if t_raw.size != x.size or t_raw.size == 0:
        raise CanonicalContractError("PHASE_EVENT_INPUT_LENGTH_MISMATCH")
    if production:
        if phase_age_ms is None:
            raise CanonicalContractError("PRODUCTION_FRESHNESS_UNAVAILABLE")
        age = np.asarray(phase_age_ms, dtype=np.float64)
        if age.size != t_raw.size or not np.all(np.isfinite(age)):
            raise CanonicalContractError("PRODUCTION_FRESHNESS_UNAVAILABLE")
        t_ms = t_raw - age
        note = "PHASE_UPDATE_ESTIMATE_TS_MINUS_AGE"
    elif phase_age_ms is not None:
        age = np.asarray(phase_age_ms, dtype=np.float64)
        if age.size != t_raw.size or not np.all(np.isfinite(age)):
            raise CanonicalContractError("DEVELOPMENT_FRESHNESS_UNAVAILABLE")
        t_ms = t_raw - age
        note = "PHASE_UPDATE_ESTIMATE_TS_MINUS_AGE"
    else:
        t_ms = t_raw * 1000.0 if timestamps_are_seconds else t_raw
        note = "PUBLIC_NATIVE_FRAME_TIME" if timestamps_are_seconds else "LEGACY_ROW_TS_NOT_FOR_PRODUCTION"

    keep_t: list[float] = []
    keep_x: list[float] = []
    last_accepted_update_estimate_ms: float | None = None
    n_repub = 0
    for i in range(t_ms.size):
        if not math.isfinite(t_ms[i]) or not math.isfinite(x[i]):
            continue
        # 8 ms is compared to the last accepted update estimate, not the previous row.
        if (
            last_accepted_update_estimate_ms is not None
            and t_ms[i] <= last_accepted_update_estimate_ms + UPDATE_ADVANCE_TOLERANCE_MS
        ):
            n_repub += 1
            continue
        keep_t.append(float(t_ms[i]) / 1000.0)
        keep_x.append(float(x[i]))
        last_accepted_update_estimate_ms = float(t_ms[i])
    t = np.asarray(keep_t, dtype=np.float64)
    p = np.asarray(keep_x, dtype=np.float64)
    if t.size >= 2:
        order = np.argsort(t, kind="mergesort")
        t, p = t[order], p[order]
    return t, p, {"n_republications": n_repub, "notes": [note], "n_events": int(t.size)}


def r2_on_events(phase: np.ndarray, t_update_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R2[i] = (x[i]-x[i-1]) / (t[i]-t[i-1]); timestamp is t[i]. First event has no sample."""
    x = np.asarray(phase, dtype=np.float64)
    t = np.asarray(t_update_s, dtype=np.float64)
    if x.size < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    dt = np.diff(t)
    dy = np.diff(x)
    good = dt > DT_MIN_S
    deriv = np.where(good, dy / dt, np.nan)
    td = t[1:]
    finite = np.isfinite(deriv)
    return deriv[finite], td[finite]


def completed_window_gap_threshold(interval_s: np.ndarray) -> tuple[float, float]:
    """Return (gap_threshold_s, median_update_dt_s) from the completed window."""
    dt = np.asarray(interval_s, dtype=np.float64)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size < MIN_INTERVALS_FOR_MEDIAN:
        raise CanonicalContractError("TOO_FEW_UPDATE_INTERVALS_FOR_GAP_MEDIAN")
    median_dt = float(np.median(dt))
    return max(GAP_FLOOR_S, GAP_MULTIPLE * median_dt), median_dt


def form_canonical_window(
    t_update_s: np.ndarray,
    phase: np.ndarray,
    t_start_s: float,
    *,
    boot_ids: np.ndarray | None = None,
) -> CanonicalWindow:
    t = np.asarray(t_update_s, dtype=np.float64)
    x = np.asarray(phase, dtype=np.float64)
    t_end = t_start_s + WINDOW_SECONDS
    if boot_ids is not None:
        boots = np.asarray(boot_ids)
        if boots.size != t.size:
            raise CanonicalContractError("BOOT_ID_LENGTH_MISMATCH")
        in_win = (t >= t_start_s) & (t <= t_end)
        if len({str(b) for b in boots[in_win]}) > 1:
            raise CanonicalContractError("BOOT_BOUNDARY_CROSSED")

    # Keep accepted events in [t_start - 0.250 s, t_start) as derivative left-hand
    # context so the first in-window event can form R2. Not an extra tensor channel.
    in_span = (t >= t_start_s - EDGE_HOLD_MAX_SECONDS) & (t <= t_end + EDGE_HOLD_MAX_SECONDS)
    t_seg, x_seg = t[in_span], x[in_span]
    if t_seg.size < MIN_INTERVALS_FOR_MEDIAN + 1:
        raise CanonicalContractError("TOO_FEW_PHASE_EVENTS")

    inside = (t_seg >= t_start_s) & (t_seg <= t_end)
    t_in = t_seg[inside]
    if t_in.size < MIN_INTERVALS_FOR_MEDIAN + 1:
        raise CanonicalContractError("TOO_FEW_PHASE_EVENTS_IN_WINDOW")
    intervals = np.diff(t_in)
    gap_thr, median_dt = completed_window_gap_threshold(intervals)
    if np.any(intervals > gap_thr):
        raise CanonicalContractError("WINDOW_CONTAINS_LARGE_GAP")

    deriv, td = r2_on_events(x_seg, t_seg)
    if deriv.size < 4:
        raise CanonicalContractError("TOO_FEW_DERIVATIVE_SAMPLES")
    grid = canonical_grid(t_start_s)
    if td[0] - grid[0] > EDGE_HOLD_MAX_SECONDS or grid[-1] - td[-1] > EDGE_HOLD_MAX_SECONDS:
        raise CanonicalContractError("EDGE_HOLD_EXCEEDED")
    resampled = np.interp(grid, td, deriv)
    if not np.all(np.isfinite(resampled)):
        raise CanonicalContractError("GRID_NONFINITE")
    scaled, mad, collapsed = apply_s1(resampled)
    if scaled.shape != (SAMPLE_COUNT,):
        raise CanonicalContractError("SAMPLE_COUNT_MISMATCH")
    return CanonicalWindow(
        values=scaled,
        t_grid_s=grid,
        t_start_s=t_start_s,
        mad=mad,
        collapsed=collapsed,
        n_phase_events=int(t_in.size),
        n_derivative_samples=int(deriv.size),
        median_update_dt_s=median_dt,
        gap_threshold_s=gap_thr,
        notes=("S1_AFTER_RESAMPLE", "N0_NO_SMOOTHING"),
    )


def deterministic_assignment_key(subject_id: str) -> str:
    return hashlib.sha256(f"{SPLIT_NAMESPACE}:{SPLIT_SEED}:{subject_id}".encode("utf-8")).hexdigest()


def calculate_split_counts(total: int) -> dict[str, int]:
    quotas = {name: total * TARGET_RATIOS[name] for name in SPLITS}
    counts = {name: math.floor(quotas[name]) for name in SPLITS}
    remaining = total - sum(counts.values())
    priority = {name: index for index, name in enumerate(SPLITS)}
    order = sorted(SPLITS, key=lambda name: (-(quotas[name] - counts[name]), priority[name]))
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def assign_subject_splits(subject_ids: list[str]) -> dict[str, str]:
    unique = sorted(set(subject_ids))
    if len(unique) != len(subject_ids) and len(unique) != len(set(subject_ids)):
        pass
    unique = sorted(set(subject_ids))
    counts = calculate_split_counts(len(unique))
    ordered = sorted(unique, key=lambda sid: (deterministic_assignment_key(sid), sid))
    assignment: dict[str, str] = {}
    cursor = 0
    for split in SPLITS:
        for sid in ordered[cursor : cursor + counts[split]]:
            assignment[sid] = split
        cursor += counts[split]
    return assignment


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def canonical_from_public_native(elapsed_s: np.ndarray, phase: np.ndarray, t_start_s: float) -> CanonicalWindow:
    t_acc, x_acc, _meta = accept_phase_events(
        elapsed_s, phase, None, production=False, timestamps_are_seconds=True
    )
    return form_canonical_window(t_acc, x_acc, t_start_s)


def freeze_split_and_index(out_dir: Path) -> dict[str, Any]:
    windows = _load_jsonl(A6_WINDOW_MANIFEST)
    subject_ids = sorted({row["subject_id"] for row in windows})
    if len(subject_ids) != 110:
        raise CanonicalContractError(f"expected 110 public subjects, got {len(subject_ids)}")
    assignment = assign_subject_splits(subject_ids)
    sets = {split: {sid for sid, sp in assignment.items() if sp == split} for split in SPLITS}
    if sets["TRAIN"] & sets["VAL"] or sets["TRAIN"] & sets["NEW_MODEL_HELDOUT_TEST"] or sets["VAL"] & sets["NEW_MODEL_HELDOUT_TEST"]:
        raise CanonicalContractError("SUBJECT_OVERLAP")

    index_rows = []
    struct: dict[str, Counter] = {split: Counter() for split in SPLITS}
    eligible: dict[str, Counter] = {split: Counter() for split in SPLITS}
    recs: dict[str, set[str]] = {split: set() for split in SPLITS}
    for row in windows:
        split = assignment[row["subject_id"]]
        recs[split].add(row["recording_id"])
        label = row.get("safenest_label")
        status = row.get("assignment_status")
        struct[split][label if label else status] += 1
        supervised = status == "ASSIGNED" and label in CLASS_TO_ID
        if supervised:
            eligible[split][label] += 1
        index_rows.append({
            "window_id": row["window_id"],
            "recording_id": row["recording_id"],
            "subject_id": row["subject_id"],
            "split": split,
            "split_profile_id": SPLIT_PROFILE_ID,
            "contract_id": CONTRACT_ID,
            "window_index": row["window_index"],
            "source_start_index": row["source_start_index"],
            "source_end_index_exclusive": row["source_end_index_exclusive"],
            "duration_seconds": row["duration_seconds"],
            "start_timestamp": row["start_timestamp"],
            "end_timestamp_exclusive": row["end_timestamp_exclusive"],
            "assignment_status": status,
            "safenest_label": label,
            "safenest_label_id": row.get("safenest_label_id"),
            "mapping_rule_id": row["mapping_rule_id"],
            "mapping_type": row["mapping_type"],
            "supervised_eligible": supervised,
            "a4_profile_id": A4_PROFILE_ID,
            "a2_profile_id": A2_PROFILE_ID,
            "heldout_performance_inspected": False,
            "team_mr60_supervised": False,
        })

    for split in ("TRAIN", "VAL"):
        for cls in CLASS_TO_ID:
            if eligible[split][cls] == 0:
                raise CanonicalContractError(f"MISSING_CLASS_{split}_{cls}")

    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = ROOT / "datasets/mmwave/splits/mmwave_mr60_compat_subject_split_v1.json"
    split_doc = {
        "profile_id": SPLIT_PROFILE_ID,
        "contract_id": CONTRACT_ID,
        "split_unit": "SUBJECT",
        "algorithm": "SHA256(namespace:seed:subject_id) then largest-remainder counts",
        "namespace": SPLIT_NAMESPACE,
        "seed": SPLIT_SEED,
        "target_ratios": TARGET_RATIOS,
        "actual_subject_counts": {split: len(sets[split]) for split in SPLITS},
        "subject_overlap_allowed": False,
        "heldout_name": "NEW_MODEL_HELDOUT_TEST",
        "heldout_is_project_wide_pristine": False,
        "heldout_policy": "NO_M_N5_MODEL_SELECTION_ACCESS",
        "historical_a5_split_copied": False,
        "subject_ids": {split: sorted(sets[split]) for split in SPLITS},
        "assignment_keys": {
            sid: deterministic_assignment_key(sid) for sid in subject_ids
        },
    }
    split_path.write_text(json.dumps(split_doc, indent=2) + "\n")
    index_path = out_dir / "window_index.jsonl"
    with index_path.open("w") as handle:
        for row in sorted(index_rows, key=lambda r: r["window_id"]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "contract_id": CONTRACT_ID,
        "public_subjects": 110,
        "public_recordings": 440,
        "public_windows": len(windows),
        "window_strategy": "A6_30S_NONOVERLAP_BOUNDARIES_REUSED_SIGNAL_RECOMPUTED",
        "subject_counts": {split: len(sets[split]) for split in SPLITS},
        "recording_counts": {split: len(recs[split]) for split in SPLITS},
        "window_counts": {split: sum(struct[split].values()) for split in SPLITS},
        "label_counts_including_ambiguous": {split: dict(struct[split]) for split in SPLITS},
        "supervised_eligible_window_counts": {split: dict(eligible[split]) for split in SPLITS},
        "supervised_eligible_totals": {split: sum(eligible[split].values()) for split in SPLITS},
        "subject_overlap": 0,
        "heldout_performance_inspected": False,
        "team_mr60_supervised_training": "DISALLOWED",
        "mr60_heldout_reference": MR60_HELDOUT_REFERENCE,
        "mr60_heldout_meaning": "SAME_SUBJECT_LIMITED_DEVICE_REFERENCE",
        "mr60_mn2_mn3_development_reference": MR60_MN2_MN3_DEVELOPMENT_REFERENCE,
        "split_path": str(split_path.relative_to(ROOT)),
        "index_path": str(index_path.relative_to(ROOT)),
    }
    summary_path = out_dir / "split_structural_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def contract_self_check() -> list[str]:
    errors: list[str] = []
    if int(round(WINDOW_SECONDS * RATE_HZ)) != SAMPLE_COUNT:
        errors.append("30s*8Hz != 240")
    grid = canonical_grid(0.0)
    if grid.size != 240 or abs(grid[-1] - 29.875) > 1e-12:
        errors.append("grid endpoint mismatch")
    zeros, mad, collapsed = apply_s1(np.zeros(240))
    if not collapsed or mad != 0.0 or not np.all(zeros == 0):
        errors.append("constant signal must collapse")
    # Tiny noise below epsilon
    tiny = np.full(240, 1e-12)
    _, mad_tiny, coll_tiny = apply_s1(tiny)
    if not coll_tiny or not (mad_tiny < MAD_EPSILON):
        errors.append("near-zero MAD must collapse")
    # Gap reject
    t = np.concatenate([np.linspace(0, 10, 80), np.linspace(12, 30, 80)])
    x = np.sin(2 * np.pi * 0.3 * t)
    try:
        form_canonical_window(t, x, 0.0)
        errors.append("1s gap should reject")
    except CanonicalContractError as exc:
        if str(exc) != "WINDOW_CONTAINS_LARGE_GAP":
            errors.append(f"unexpected gap error {exc}")
    # Boot reject
    t = np.linspace(0, 30, 240)
    x = np.sin(2 * np.pi * 0.25 * t)
    boots = np.array(["a"] * 120 + ["b"] * 120)
    try:
        form_canonical_window(t, x, 0.0, boot_ids=boots)
        errors.append("boot crossing should reject")
    except CanonicalContractError as exc:
        if str(exc) != "BOOT_BOUNDARY_CROSSED":
            errors.append(f"unexpected boot error {exc}")
    # Happy path 10 Hz native public-like
    t = np.arange(0, 30.0, 0.1)
    x = np.sin(2 * np.pi * 0.25 * t)
    win = form_canonical_window(t, x, 0.0)
    if win.values.shape != (240,) or win.values.dtype != np.float32:
        errors.append("happy-path shape/dtype")
    if not np.all(np.isfinite(win.values)):
        errors.append("happy-path nonfinite")
    # Production requires freshness.
    t = np.arange(0, 30000, 100, dtype=np.float64)
    x = np.sin(2 * np.pi * 0.25 * t / 1000.0)
    try:
        accept_phase_events(t, x, None, production=True, timestamps_are_seconds=False)
        errors.append("production missing age should fail")
    except CanonicalContractError as exc:
        if str(exc) != "PRODUCTION_FRESHNESS_UNAVAILABLE":
            errors.append(f"unexpected production error {exc}")
    # Equal numeric phase may still be a new sample.
    t_eq = np.array([0.0, 0.12, 0.24, 0.36, 0.48] + list(np.arange(0.60, 30.0, 0.12)))
    x_eq = np.concatenate([np.zeros(4), np.sin(2 * np.pi * 0.25 * t_eq[4:])])
    t_acc, x_acc, meta = accept_phase_events(
        t_eq, x_eq, None, production=False, timestamps_are_seconds=True
    )
    if t_acc.size < 5 or meta["n_republications"] != 0:
        errors.append("equal phase must not be auto-deduplicated")
    return errors


def main() -> int:
    errors = contract_self_check()
    print(
        json.dumps(
            {
                "contract_id": CONTRACT_ID,
                "root": str(ROOT.name),
                "self_check_errors": errors,
                "freeze_split_and_index": "SKIPPED_PUBLIC_A6_NOT_IMPORTED",
            },
            indent=2,
        )
    )
    if errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
