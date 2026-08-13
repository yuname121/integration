#!/usr/bin/env python3
"""Validator for Phase A3 timeline and window manifest artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


FORBIDDEN_LABEL_FIELDS = {
    "label",
    "labels",
    "target",
    "targets",
    "class",
    "annotation",
    "NORMAL",
    "RAPID_OR_ABNORMAL",
    "APNEA",
    "apnea",
    "normal",
    "rapid",
    "abnormal",
}


def derive_gate(
    validation_success: bool,
    exceptions: list[dict[str, Any]],
    rec_results: list[dict[str, Any]],
) -> tuple[str, str]:
    """Derive A3 gate and A4 entry status cleanly."""
    if not validation_success:
        return "FAIL", "NOT_READY"

    has_blocker = any(item.get("severity") == "BLOCKER" for item in exceptions)
    has_error = any(
        item.get("severity") == "ERROR" for item in exceptions
    ) or any(len(r.get("errors", [])) > 0 for r in rec_results)

    if has_blocker:
        return "BLOCKED", "BLOCKED"
    if has_error:
        return "FAIL", "NOT_READY"

    has_warning = any(
        item.get("severity") == "WARNING" for item in exceptions
    ) or any(len(r.get("warnings", [])) > 0 for r in rec_results)

    if has_warning:
        return "PASS_WITH_WARNINGS", "READY_WITH_CONDITIONS"

    return "PASS", "READY"


def validate_manifests(
    a2_pilot: dict[str, Any],
    profile: dict[str, Any],
    rec_results: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    exceptions: list[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Validate Phase A3 manifests against all 15 structural and timing invariants."""
    errors: list[str] = []

    a2_recs = {r["recording_id"] for r in a2_pilot.get("recordings", [])}
    a3_recs = {r["recording_id"] for r in rec_results}

    # 1. Every A3 recording references an A2 pilot recording
    missing_a2 = a3_recs - a2_recs
    if missing_a2:
        errors.append(f"A3 recordings not in A2 pilot: {sorted(list(missing_a2))}")

    # 2. Every window references a valid recording
    win_recs = {w["recording_id"] for w in windows}
    orphan_wins = win_recs - a3_recs
    if orphan_wins:
        errors.append(f"Window manifest contains orphan recordings: {sorted(list(orphan_wins))}")

    # 3. Phase/timestamp lengths agree
    for r in rec_results:
        if r["source_sample_count"] != r["source_timestamp_count"]:
            errors.append(
                f"Recording {r['recording_id']} phase/timestamp count mismatch: "
                f"sample_count={r['source_sample_count']} != timestamp_count={r['source_timestamp_count']}"
            )

    # 4 & 5. Monotonicity and duplicate/backward counts
    for r in rec_results:
        if r["backward_timestamp_count"] > 0 and r["quality_status"] == "SUCCESS":
            errors.append(f"Recording {r['recording_id']} has backward timestamps but status is SUCCESS")

    # 6. Target sampling rate validation
    expected_rate = profile.get("target_sampling_rate_hz", 10.0)
    dt_exp = profile.get("expected_dt_seconds", 0.1)
    for r in rec_results:
        emp_rate = r.get("empirical_sampling_rate_hz", 0.0)
        if abs(emp_rate - expected_rate) > 0.05 and not r.get("resampling_performed", False):
            errors.append(
                f"Recording {r['recording_id']} empirical sampling rate {emp_rate} Hz "
                f"differs from target {expected_rate} Hz without resampling"
            )
        if r.get("resampling_performed", False):
            expected_grid_count = int(math.floor(r["duration_seconds"] / dt_exp)) + 1
            if r["canonical_sample_count"] != expected_grid_count:
                errors.append(
                    f"Recording {r['recording_id']} resampled canonical sample count {r['canonical_sample_count']} "
                    f"!= expected grid count {expected_grid_count}"
                )

    # 7. All valid canonical windows have exactly 300 samples
    expected_samples = profile.get("window", {}).get("samples", 300)
    for w in windows:
        if w["sample_count"] != expected_samples:
            errors.append(f"Window {w['window_id']} sample count {w['sample_count']} != {expected_samples}")
        if "last_sample_timestamp" not in w or "end_timestamp_exclusive" not in w:
            errors.append(f"Window {w['window_id']} missing required timestamp fields (last_sample_timestamp or end_timestamp_exclusive)")

    # 8. Window IDs are unique
    win_ids = [w["window_id"] for w in windows]
    if len(win_ids) != len(set(win_ids)):
        errors.append(f"Duplicate window IDs found in manifest: total={len(win_ids)}, unique={len(set(win_ids))}")

    # 9. Window boundaries are deterministic and non-overlapping if zero overlap specified
    overlap = profile.get("window", {}).get("overlap_samples", 0)
    stride = profile.get("window", {}).get("stride_samples", 300)
    if overlap == 0:
        by_rec: dict[str, list[dict[str, Any]]] = {}
        for w in windows:
            by_rec.setdefault(w["recording_id"], []).append(w)
        for rec_id, rec_wins in by_rec.items():
            sorted_wins = sorted(rec_wins, key=lambda x: x["window_index"])
            for idx in range(len(sorted_wins)):
                w = sorted_wins[idx]
                if w["window_index"] != idx:
                    errors.append(f"Window index gap in {rec_id}: index={w['window_index']} != {idx}")
                expected_start = idx * stride
                expected_end = expected_start + expected_samples
                if w["canonical_start_index"] != expected_start or w["canonical_end_index_exclusive"] != expected_end:
                    errors.append(
                        f"Window {w['window_id']} boundary mismatch: "
                        f"got [{w['canonical_start_index']}, {w['canonical_end_index_exclusive']}), "
                        f"expected [{expected_start}, {expected_end})"
                    )

    # 10. No window crosses a prohibited large gap
    for w in windows:
        if w.get("large_gap_count", 0) > 0 and w["timeline_valid"] is True:
            errors.append(f"Window {w['window_id']} crosses large gap but is marked timeline_valid=True")

    # 11. No label fields were introduced
    for w in windows:
        found_forbidden = FORBIDDEN_LABEL_FIELDS.intersection(w.keys())
        if found_forbidden:
            errors.append(f"Window {w['window_id']} contains forbidden label fields: {sorted(list(found_forbidden))}")

    # 12. Interpolation counts match
    for w in windows:
        if w["interpolated_sample_count"] > w["sample_count"]:
            errors.append(f"Window {w['window_id']} interpolated count exceeds sample count")

    # 13. Dropped-tail counts match recording lengths
    for r in rec_results:
        expected_dropped = r["canonical_sample_count"] - (r["window_count"] * expected_samples)
        if r["dropped_tail_samples"] != expected_dropped:
            errors.append(
                f"Recording {r['recording_id']} dropped tail count mismatch: "
                f"got {r['dropped_tail_samples']}, expected {expected_dropped}"
            )

    # 14. Summary counts match detailed manifests
    if summary.get("pilot_recording_count") != len(rec_results):
        errors.append(f"Summary pilot_recording_count mismatch: {summary.get('pilot_recording_count')} != {len(rec_results)}")
    if summary.get("total_window_count") != len(windows):
        errors.append(f"Summary total_window_count mismatch: {summary.get('total_window_count')} != {len(windows)}")
    if summary.get("total_dropped_tail_samples") != sum(r["dropped_tail_samples"] for r in rec_results):
        errors.append("Summary total_dropped_tail_samples mismatch")

    # 15. Gate state check
    gate, ready = derive_gate(len(errors) == 0, exceptions, rec_results)
    if summary.get("a3_gate_status") != gate:
        errors.append(f"Summary gate status mismatch: summary={summary.get('a3_gate_status')} != calculated={gate}")
    if summary.get("a4_entry_status") != ready:
        errors.append(f"Summary A4 entry status mismatch: summary={summary.get('a4_entry_status')} != calculated={ready}")

    return len(errors) == 0, errors


def main() -> None:
    root = Path(".")
    manifest_dir = root / "datasets/mmwave/manifests/a3_timeline_pilot"
    a2_dir = root / "datasets/mmwave/manifests/a2_phase_pilot"

    if not manifest_dir.exists():
        print(f"Error: manifest directory {manifest_dir} does not exist")
        raise SystemExit(1)

    a2_pilot = json.loads((a2_dir / "pilot_selection.json").read_text(encoding="utf-8"))
    profile = json.loads((manifest_dir / "timeline_profile.json").read_text(encoding="utf-8"))
    rec_results = [
        json.loads(line)
        for line in (manifest_dir / "recording_timeline_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    windows = [
        json.loads(line)
        for line in (manifest_dir / "window_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    exceptions = json.loads((manifest_dir / "exceptions.json").read_text(encoding="utf-8"))
    summary = json.loads((manifest_dir / "a3_summary.json").read_text(encoding="utf-8"))

    success, errors = validate_manifests(a2_pilot, profile, rec_results, windows, exceptions, summary)
    gate, ready = derive_gate(success, exceptions, rec_results)

    print(f"Validation Success: {success}")
    print(f"Derived A3 Gate: {gate}")
    print(f"Derived A4 Entry Status: {ready}")

    if errors:
        print("\nValidation Errors:")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
