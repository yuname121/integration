#!/usr/bin/env python3
"""Build a compact, reproducible SafeNest v4 source/model delivery archive."""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
OUTPUT = PROJECT_ROOT / "releases" / "SafeNest_v4.0_commercialization_package.zip"
CORE_INCLUDES = (
    "README.md", "walkthrough.md", "requirements-mac.txt", "requirements-pi.txt",
    "docs/ONDEVICE_AI_COMPLETION_AUDIT_V4.md",
    "docs/TEAM_SENSOR_AI_INTEGRATION_PROMPTS_V4.md",
    "inference/__init__.py", "inference/thermal_interpreter.py", "inference/infer_pi_thermal.py",
    "inference/co2_interpreter.py", "inference/mmwave_interpreter.py", "inference/model_registry.py",
    "risk/__init__.py", "risk/risk_rules.py", "risk/risk_engine.py", "risk/risk_config.json",
    "integrated_node/run_demo.py", "integrated_node/virtual_sensor_streamer.py",
    "integrated_node/safenest_integrated_plotter.py", "integrated_node/safenest_risk_engine.py",
    "adapters/__init__.py", "adapters/mmwave_stream_adapter.py", "adapters/mmwave_csv_adapter.py",
    "models/model_manifest.json", "models/thermal/thermal_fall_int8_v0.1.0.tflite",
    "models/co2/co2_occupancy_int8_v0.1.0.tflite", "models/co2/co2_scaling_metadata_v0.1.0.json",
    "models/mmwave/mmwave_resp_int8_v0.1.0.tflite", "models/mmwave/sensor_stats_metadata_v0.1.0.json",
    "models/mmwave/source_sensor_stats_metadata_20260713.json", "models/mmwave/IMPORT_PROVENANCE_20260725.md",
    "benchmarks/benchmark_thermal.py", "scripts/build_v4_archive.py", "scripts/test_thermal_tflite.py",
    "thermal_prep.py", "thermal_train.py",
)


def archive_inputs() -> tuple[str, ...]:
    tests = tuple(
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "tests").glob("test_*.py"))
    )
    return tuple(sorted(set(CORE_INCLUDES + tests)))


def main() -> None:
    includes = archive_inputs()
    missing = [name for name in includes if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"archive inputs missing: {missing}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    checksums = []
    with zipfile.ZipFile(temporary_output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in includes:
            print(f"packing {name}", flush=True)
            payload = (ROOT / name).read_bytes()
            checksums.append(f"{hashlib.sha256(payload).hexdigest()}  {name}")
            info = zipfile.ZipInfo(f"SafeNest_v4.0/{name}", date_time=(2026, 7, 28, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload)
        sums_info = zipfile.ZipInfo(
            "SafeNest_v4.0/SHA256SUMS.txt", date_time=(2026, 7, 28, 0, 0, 0)
        )
        sums_info.compress_type = zipfile.ZIP_DEFLATED
        sums_info.external_attr = 0o644 << 16
        archive.writestr(sums_info, "\n".join(checksums) + "\n")
    temporary_output.replace(OUTPUT)
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    (OUTPUT.with_suffix(OUTPUT.suffix + ".sha256")).write_text(
        f"{digest}  {OUTPUT.name}\n", encoding="utf-8"
    )
    print(f"{OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
