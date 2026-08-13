#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmarks/benchmark_thermal.py
SafeNest Thermal Model Latency & Memory Benchmark Script
"""

import sys
from pathlib import Path
import json
import platform
import statistics
import time
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.thermal_interpreter import ThermalInterpreter

WARMUP_RUNS = 50
MEASURE_RUNS = 1000

def percentile(values, q):
    return float(np.percentile(np.asarray(values), q))

def main():
    runner = ThermalInterpreter(project_root=PROJECT_ROOT)
    frame = np.zeros((62, 80), dtype=np.float32)

    print("🔥 Warming up interpreter (50 runs)...")
    for _ in range(WARMUP_RUNS):
        runner.predict(frame)

    print("📊 Measuring inference latency (1000 runs)...")
    latencies = []
    for _ in range(MEASURE_RUNS):
        started = time.perf_counter()
        runner.predict(frame)
        latencies.append((time.perf_counter() - started) * 1000.0)

    result = {
        "model_id": runner.model_meta["model_id"],
        "model_version": runner.model_meta["version"],
        "model_path": str(runner.model_meta["path"]),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "warmup_runs": WARMUP_RUNS,
        "measure_runs": MEASURE_RUNS,
        "latency_ms": {
            "mean": float(statistics.mean(latencies)),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "max": float(max(latencies))
        }
    }

    # Save to thermal_latest.json
    output_latest = PROJECT_ROOT / "benchmarks/thermal_latest.json"
    output_latest.parent.mkdir(parents=True, exist_ok=True)
    output_latest.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save to thermal_mac_20260725.json
    output_mac = PROJECT_ROOT / "benchmarks/thermal_mac_20260725.json"
    output_mac.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== Benchmark Summary ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n✅ Benchmark JSON saved to: {output_mac.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
