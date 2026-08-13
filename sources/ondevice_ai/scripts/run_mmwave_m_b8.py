#!/usr/bin/env python3
"""Run SafeNest M-B8 only with explicit smoke or formally guarded benchmark mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from mmwave_m_b8_benchmark import (  # noqa: E402
    ARCHITECTURE_ID,
    BENCHMARK_METRICS,
    FORMAL_SEED_ORDERS,
    FROZEN_SEEDS,
    MANIFEST_RELATIVE,
    REPORT_RELATIVE,
    REQUIRED_OUTPUT_FILENAMES,
    BenchmarkEnvironmentBlocked,
    TFLiteBenchmarkSession,
    benchmark_confirmation_invoke,
    benchmark_seed_series,
    build_benchmark_input_evidence,
    build_complete_evidence,
    build_memory_observation,
    build_static_evidence,
    capture_machine_environment,
    file_sha256,
    find_known_safenest_workloads,
    make_run_index,
    prepare_benchmark_inputs,
    quantize_model_input,
    render_report,
    require_idle_stabilization,
    write_deterministic_npz,
    write_json,
)


def _assert_no_new_workload(stage: str) -> None:
    active = find_known_safenest_workloads()
    if active:
        raise BenchmarkEnvironmentBlocked(
            f"FORMAL_BENCHMARK_ENVIRONMENT_DISTURBED_{stage}: "
            + json.dumps(active, sort_keys=True)
        )


def run_smoke_checks(root_dir: Path = ROOT_DIR) -> Dict[str, Any]:
    """Exercise one frozen functional path without reading a timing clock or writing evidence."""
    static = build_static_evidence(root_dir)
    inputs = prepare_benchmark_inputs(root_dir)
    seed = FROZEN_SEEDS[0]
    artifact = static["artifacts"][seed]
    session = TFLiteBenchmarkSession(root_dir / artifact["relative_path"])
    quantized = quantize_model_input(
        inputs["model_inputs"][0], session.input_scale, session.input_zero_point
    )
    session.set_input(quantized)
    session.invoke()
    output = session.output_int8()
    if output.shape != (1, 3) or output.dtype != np.int8:
        raise RuntimeError(f"M-B8 smoke output mismatch: shape={output.shape}, dtype={output.dtype}")
    return {
        "smoke_only": True,
        "formal_latency_measurement_started": False,
        "seed": seed,
        "architecture": ARCHITECTURE_ID,
        "validation_input_cycle_size": inputs["input_cycle_size"],
        "output_shape": [int(value) for value in output.shape],
        "output_dtype": output.dtype.name,
        "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        "locked_test_access_attempts": 0,
    }


def run_formal_benchmark(root_dir: Path = ROOT_DIR) -> Dict[str, Any]:
    """Perform the full M-B8 benchmark only after all required idle guards pass."""
    # All loading, identity validation, architecture inspection, and preprocessing
    # setup happen before the first idle gate, never inside a timed interval.
    static = build_static_evidence(root_dir)
    inputs = prepare_benchmark_inputs(root_dir)
    run_index = make_run_index(static["artifacts"])
    raw_arrays: Dict[str, np.ndarray] = {}
    series_idle_conditions = []
    memory_series: Dict[str, Any] = {}

    for series_number, seed_order in enumerate(FORMAL_SEED_ORDERS, 1):
        # Mandatory user/contract gate: detect C-B2/other SafeNest workloads
        # and then observe a continuous 30-second idle period immediately before
        # every formal series.
        idle = require_idle_stabilization()
        idle["benchmark_stage"] = "FORMAL"
        idle["series"] = series_number
        idle["seed_order"] = list(seed_order)
        series_idle_conditions.append(idle)
        memory_series[f"formal_series_{series_number:02d}"] = {}

        for seed in seed_order:
            measurements, memory = benchmark_seed_series(
                root_dir,
                seed,
                inputs["canonical_inputs"],
                inputs["model_inputs"],
                inputs["zscore_stats"],
                warmup_iterations=100,
                measured_iterations=1000,
            )
            for metric in BENCHMARK_METRICS:
                entry = next(
                    item
                    for item in run_index["formal_runs"]
                    if item["series"] == series_number
                    and item["seed"] == seed
                    and item["metric"] == metric
                )
                raw_arrays[entry["raw_array_key"]] = measurements[metric]
            memory_series[f"formal_series_{series_number:02d}"][str(seed)] = memory
            _assert_no_new_workload(f"AFTER_FORMAL_SERIES_{series_number:02d}_SEED_{seed}")

    # A short independent confirmation series is also preceded by the same
    # workload/idle guard and is limited to invoke-only timing by contract.
    confirmation_idle = require_idle_stabilization()
    confirmation_idle["benchmark_stage"] = "CONFIRMATION"
    confirmation_idle["series"] = 1
    confirmation_idle["seed_order"] = list(FROZEN_SEEDS)
    series_idle_conditions.append(confirmation_idle)
    memory_series["confirmation_series_01"] = {}
    for seed in FROZEN_SEEDS:
        samples, metadata = benchmark_confirmation_invoke(root_dir, seed, inputs["model_inputs"])
        entry = next(
            item for item in run_index["confirmation_runs"] if item["seed"] == seed
        )
        raw_arrays[entry["raw_array_key"]] = samples
        memory_series["confirmation_series_01"][str(seed)] = metadata
        _assert_no_new_workload(f"AFTER_CONFIRMATION_SEED_{seed}")

    environment = capture_machine_environment(series_idle_conditions)
    environment.update(build_benchmark_input_evidence(inputs, static["artifacts"]))
    memory = build_memory_observation(memory_series)
    evidence = build_complete_evidence(static, run_index, raw_arrays, environment, memory)
    return evidence


def write_m_b8_artifacts(root_dir: Path = ROOT_DIR) -> Dict[str, Any]:
    """Run a guarded formal benchmark, then persist its evidence and report."""
    evidence = run_formal_benchmark(root_dir)
    manifest_dir = root_dir / MANIFEST_RELATIVE
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_OUTPUT_FILENAMES:
        target = manifest_dir / filename
        value = evidence[filename]
        if filename.endswith(".npz"):
            write_deterministic_npz(target, value)
        else:
            write_json(target, value)
    checksum_lines = [
        f"{file_sha256(manifest_dir / filename)}  {filename}"
        for filename in REQUIRED_OUTPUT_FILENAMES
    ]
    (manifest_dir / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    report_path = root_dir / REPORT_RELATIVE
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(evidence), encoding="utf-8")
    return evidence["m_b8_summary.json"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Run one non-timing functional smoke check and write no M-B8 evidence.",
    )
    mode.add_argument(
        "--formal",
        action="store_true",
        help="Run formal timing only after workload absence and each 30-second stabilization gate.",
    )
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(run_smoke_checks(ROOT_DIR), indent=2, sort_keys=True))
        return
    try:
        summary = write_m_b8_artifacts(ROOT_DIR)
    except BenchmarkEnvironmentBlocked as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
    print("SafeNest M-B8 formal benchmark complete")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
