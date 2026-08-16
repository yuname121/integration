#!/usr/bin/env python3
"""Read-only RP-X0 diagnostic inspection of collected sensor evidence.

Classification: RP_X0_DIAGNOSTIC_ANALYSIS

This script never writes into data/mmwave, data/co2, or data/thermal.
It never talks to TCP :9000 or UDP :5005. It is not part of production runtime.
Thresholds printed here are descriptive only and are not model contracts.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PHASE_STALE_MS = 100
FRESH_PHASE_WINDOW_S = 30.0
FRESH_PHASE_SAMPLE_TARGET = 300


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only RP-X0 inspection of mmWave JSONL, CO2 JSONL, or Thermal NPZ"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    mmwave = sub.add_parser("mmwave", help="cadence/provenance of mmWave JSONL")
    mmwave.add_argument("paths", nargs="+", type=Path)
    co2 = sub.add_parser("co2", help="provenance/cadence of CO2 JSONL")
    co2.add_argument("paths", nargs="+", type=Path)
    thermal = sub.add_parser("thermal", help="inspect Thermal NPZ files")
    thermal.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if args.command == "mmwave":
        print(json.dumps(inspect_mmwave(load_jsonl(args.paths)), indent=2, sort_keys=True))
        return 0
    if args.command == "co2":
        print(json.dumps(inspect_co2(load_jsonl(args.paths)), indent=2, sort_keys=True))
        return 0
    print(json.dumps(inspect_thermal(expand_npz(args.paths)), indent=2, sort_keys=True, default=str))
    return 0


def load_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        target = path.expanduser().resolve()
        if not target.is_file():
            raise SystemExit(f"not a file: {target}")
        for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{target}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise SystemExit(f"{target}:{line_number}: expected object")
            rows.append(row)
    return rows


def expand_npz(paths: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        target = path.expanduser().resolve()
        if target.is_file() and target.suffix == ".npz":
            found.append(target)
        elif target.is_dir():
            found.extend(sorted(item for item in target.glob("*.npz") if item.is_file()))
        else:
            raise SystemExit(f"not an NPZ file or directory: {target}")
    return found


def inspect_mmwave(rows: list[dict[str, Any]]) -> dict[str, Any]:
    transport_seq = [row.get("sequence") for row in rows]
    receive = [row.get("receive_monotonic") for row in rows]
    uptime = [row.get("source_uptime_ms") for row in rows]
    nested = [row.get("mmwave") if isinstance(row.get("mmwave"), dict) else None for row in rows]
    nested_seq = [item.get("seq") if item else None for item in nested]
    phases = [item.get("breath_phase") if item and "breath_phase" in item else None for item in nested]
    ages = [item.get("phase_age_ms") if item and "phase_age_ms" in item else None for item in nested]
    firmware = [item.get("firmware_version") if item else None for item in nested]
    config_hash = [item.get("config_hash") if item else None for item in nested]
    schema = [item.get("schema_version") if item else None for item in nested]
    present_phase = sum(1 for value in phases if value is not None)
    numeric_ages = [int(value) for value in ages if isinstance(value, int)]
    fresh = sum(1 for value in numeric_ages if value < PHASE_STALE_MS)
    stale = sum(1 for value in numeric_ages if value >= PHASE_STALE_MS)
    repeated_phase_runs = _longest_run(phases)
    duration_s = _span(receive)
    transport_dt = _deltas(receive)
    esp_dt_ms = _deltas(uptime)
    return {
        "classification": "RP_X0_DIAGNOSTIC_ANALYSIS",
        "packet_count": len(rows),
        "duration_s": duration_s,
        "transport_interarrival_s": _summary(transport_dt),
        "esp_monotonic_interval_ms": _summary(esp_dt_ms),
        "transport_sequence": _sequence_report(transport_seq),
        "nested_mmwave_sequence": _sequence_report(nested_seq),
        "breath_phase_availability_ratio": _ratio(present_phase, len(rows)),
        "phase_age_ms": _summary(numeric_ages),
        "fresh_phase_count_age_lt_ms": {"threshold_ms": PHASE_STALE_MS, "fresh": fresh, "stale": stale},
        "repeated_phase_longest_run": repeated_phase_runs,
        "effective_fresh_phase_rate_hz": (
            None if duration_s in (None, 0) else round(fresh / duration_s, 6)
        ),
        "feasibility_note": {
            "30s_window": FRESH_PHASE_WINDOW_S,
            "300_sample_target": FRESH_PHASE_SAMPLE_TARGET,
            "not_a_model_contract": True,
        },
        "firmware_versions": _unique(firmware),
        "config_hashes": _unique(config_hash),
        "schema_versions": _unique(schema),
        "identity_changes_during_run": _identity_changed(nested),
    }


def inspect_co2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    transport_seq = [row.get("sequence") for row in rows]
    receive = [row.get("receive_monotonic") for row in rows]
    event_ids = [row.get("co2_measurement_event_id") for row in rows]
    event_ms = [row.get("co2_measurement_monotonic_ms") for row in rows]
    values = [row.get("co2_ppm") for row in rows]
    changes = 0
    for previous, current in zip(values, values[1:]):
        if previous != current:
            changes += 1
    return {
        "classification": "RP_X0_DIAGNOSTIC_ANALYSIS",
        "transport_packet_count": len(rows),
        "event_id_present_count": sum(1 for value in event_ids if value is not None),
        "unique_event_ids": len({(row.get("device_id"), row.get("boot_id"), row.get("co2_measurement_event_id")) for row in rows}),
        "co2_measurement_monotonic_ms_present_count": sum(1 for value in event_ms if value is not None),
        "co2_value_change_count": changes,
        "co2_ppm_summary": _summary([float(value) for value in values if isinstance(value, (int, float))]),
        "receive_interval_s": _summary(_deltas(receive)),
        "transport_sequence": _sequence_report(transport_seq),
        "questions_for_real_collection": [
            "How many transport packets arrived vs physical measurement events?",
            "Are event IDs present?",
            "Is co2_measurement_monotonic_ms present?",
            "How often does CO2 value actually change?",
            "Does the 60-second fallback activate?",
            "Are repeated transport publications being collapsed?",
            "Are gaps visible?",
        ],
        "production_co2_slope_not_calculated": True,
    }


def inspect_thermal(paths: list[Path]) -> dict[str, Any]:
    import numpy as np

    files = []
    total_frames = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            frames = saved["frames"] if "frames" in saved.files else None
            sequences = saved["frame_sequences"].tolist() if "frame_sequences" in saved.files else []
            timestamps = saved["timestamps"].tolist() if "timestamps" in saved.files else []
            receive = saved["receive_monotonic"].tolist() if "receive_monotonic" in saved.files else []
            uptime = saved["source_uptime_ms"].tolist() if "source_uptime_ms" in saved.files else []
            frame_count = int(frames.shape[0]) if frames is not None else 0
            total_frames += frame_count
            files.append(
                {
                    "path": str(path),
                    "keys": list(saved.files),
                    "frame_count": frame_count,
                    "frames_shape": list(frames.shape) if frames is not None else None,
                    "dtype": str(frames.dtype) if frames is not None else None,
                    "frame_sequences": sequences,
                    "timestamp_count": len(timestamps),
                    "receive_monotonic_count": len(receive),
                    "source_uptime_ms_count": len(uptime),
                    "min_raw": int(frames.min()) if frames is not None and frames.size else None,
                    "max_raw": int(frames.max()) if frames is not None and frames.size else None,
                    "sequence_discontinuities": _sequence_report(sequences),
                }
            )
    return {
        "classification": "RP_X0_DIAGNOSTIC_ANALYSIS",
        "file_count": len(paths),
        "frame_count": total_frames,
        "files": files,
        "not_fed_to_t_b5": True,
    }


def _sequence_report(values: list[Any]) -> dict[str, Any]:
    numeric = [int(value) for value in values if isinstance(value, int)]
    gaps = 0
    duplicates = 0
    out_of_order = 0
    seen: set[int] = set()
    previous = None
    for value in numeric:
        if value in seen:
            duplicates += 1
        seen.add(value)
        if previous is not None:
            if value < previous:
                out_of_order += 1
            elif value > previous + 1:
                gaps += value - previous - 1
        previous = value
    return {
        "count": len(numeric),
        "missing": len(values) - len(numeric),
        "gaps": gaps,
        "duplicate_values": duplicates,
        "out_of_order": out_of_order,
        "first": numeric[0] if numeric else None,
        "last": numeric[-1] if numeric else None,
    }


def _deltas(values: list[Any]) -> list[float]:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return [later - earlier for earlier, later in zip(numeric, numeric[1:])]


def _span(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if len(numeric) < 2:
        return None
    return numeric[-1] - numeric[0]


def _summary(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return {
        "count": len(finite),
        "min": min(finite),
        "max": max(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
    }


def _unique(values: list[Any]) -> list[Any]:
    seen: list[Any] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _identity_changed(nested: list[dict[str, Any] | None]) -> bool:
    identities = []
    for item in nested:
        if not item:
            continue
        identity = (
            item.get("firmware_version"),
            item.get("config_hash"),
            item.get("schema_version"),
        )
        if identity not in identities:
            identities.append(identity)
    return len(identities) > 1


def _longest_run(values: list[Any]) -> dict[str, Any] | None:
    longest = 0
    current = 0
    previous = object()
    value = None
    for item in values:
        if item == previous:
            current += 1
        else:
            current = 1
            previous = item
        if current > longest:
            longest = current
            value = item
    if not values:
        return None
    return {"value": value, "length": longest, "distinct_values": len(Counter(map(_freeze, values)))}


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


if __name__ == "__main__":
    sys.exit(main())
