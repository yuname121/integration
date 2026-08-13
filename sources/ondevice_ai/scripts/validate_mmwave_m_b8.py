#!/usr/bin/env python3
"""Standalone fail-closed validator for SafeNest M-B8 benchmark evidence.

It never reruns a wall-clock benchmark.  Instead it independently inspects
the frozen artifacts and recomputes all reported latency summaries directly
from saved positive integer nanosecond samples.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from mmwave_m_b8_benchmark import (  # noqa: E402
    BENCHMARK_METRICS,
    DELEGATE_RUNTIME_MODE,
    FORMAL_SEED_ORDERS,
    FORMAL_SERIES_COUNT,
    FROZEN_SEEDS,
    MANIFEST_RELATIVE,
    MEMORY_MEASUREMENT_TYPE,
    MEMORY_METHOD,
    MINIMUM_IDLE_SECONDS,
    NUM_THREADS,
    PERCENTILE_METHOD,
    REQUIRED_OUTPUT_FILENAMES,
    build_complete_evidence,
    build_benchmark_input_evidence,
    build_static_evidence,
    cross_seed_latency_summary,
    make_run_index,
    prepare_benchmark_inputs,
    summarize_raw_samples,
)
from validate_mmwave_m_b7 import validate_m_b7_artifacts  # noqa: E402


class MB8ValidationError(Exception):
    """Raised when M-B8 benchmark evidence cannot be independently proven."""


REQUIRED_MB8_ARTIFACTS = set(REQUIRED_OUTPUT_FILENAMES) | {"checksums.sha256"}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MB8ValidationError(f"Malformed JSON artifact {path.name}: {exc}") from exc


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise MB8ValidationError(f"{label} mismatch against independent validation")


def _validate_checksums(manifest_dir: Path) -> None:
    path = manifest_dir / "checksums.sha256"
    seen: set = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise MB8ValidationError(f"Unable to read checksums.sha256: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        pieces = line.split(maxsplit=1)
        if len(pieces) != 2:
            raise MB8ValidationError(f"Malformed checksum line {line_number}")
        digest, relative = pieces[0].lower(), pieces[1].strip()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise MB8ValidationError(f"Malformed checksum digest at line {line_number}")
        rel_path = Path(relative)
        if (
            rel_path.is_absolute()
            or ".." in rel_path.parts
            or "\\" in relative
            or relative.startswith("~")
            or "file://" in relative
        ):
            raise MB8ValidationError(f"Checksum path traversal/absolute path at line {line_number}")
        if relative in seen:
            raise MB8ValidationError(f"Duplicate checksum target: {relative}")
        seen.add(relative)
        target = manifest_dir / relative
        if target.parent.resolve() != manifest_dir.resolve() or not target.is_file():
            raise MB8ValidationError(f"Checksum target missing or escaping manifest: {relative}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise MB8ValidationError(f"Checksum mismatch for {relative}")
    expected = REQUIRED_MB8_ARTIFACTS - {"checksums.sha256"}
    if seen != expected:
        raise MB8ValidationError(
            f"Checksum coverage mismatch: missing={sorted(expected - seen)}, unexpected={sorted(seen - expected)}"
        )


def _validate_machine_paths(manifest_dir: Path, root_dir: Path) -> None:
    forbidden = ("/Users/", "file://", str(root_dir.resolve()))
    for path in manifest_dir.iterdir():
        if path.suffix != ".json":
            continue
        content = path.read_text(encoding="utf-8")
        if any(token and token in content for token in forbidden):
            raise MB8ValidationError(f"Absolute local path found in {path.name}")


def _validate_input_identity(root_dir: Path, artifact: Dict[str, Any]) -> None:
    rows = artifact.get("inputs")
    if not isinstance(rows, list) or artifact.get("total_inputs") != len(rows) or len(rows) < 30:
        raise MB8ValidationError("Incomplete M-B8 input identity inventory")
    seen: set = set()
    for row in rows:
        relative = row.get("repository_relative_path")
        digest = row.get("measured_sha256")
        if not isinstance(relative, str) or not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise MB8ValidationError("Malformed input identity row")
        rel_path = Path(relative)
        if rel_path.is_absolute() or ".." in rel_path.parts or relative in seen:
            raise MB8ValidationError("Unsafe or duplicate input identity path")
        seen.add(relative)
        target = root_dir / relative
        if not target.is_file():
            raise MB8ValidationError(f"Input identity target missing: {relative}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise MB8ValidationError(f"Input identity SHA mismatch: {relative}")


def _validate_benchmark_environment(
    root_dir: Path, environment: Dict[str, Any], artifacts: Dict[int, Dict[str, Any]]
) -> None:
    if environment.get("phase_id") != "M-B8":
        raise MB8ValidationError("Benchmark environment phase mismatch")
    if environment.get("thread_configuration", {}).get("num_threads") != NUM_THREADS:
        raise MB8ValidationError("Benchmark environment thread-count mismatch")
    if environment.get("delegate_runtime_mode") != DELEGATE_RUNTIME_MODE:
        raise MB8ValidationError("Benchmark environment delegate-runtime mismatch")
    if not environment.get("formal_benchmark_environment_ready"):
        raise MB8ValidationError("Formal benchmark environment was not marked ready")
    conditions = environment.get("known_safenest_workload_checks")
    if not isinstance(conditions, list) or len(conditions) != FORMAL_SERIES_COUNT + 1:
        raise MB8ValidationError("Idle-condition evidence count mismatch")
    expected_stages = [("FORMAL", 1), ("FORMAL", 2), ("FORMAL", 3), ("CONFIRMATION", 1)]
    for condition, (stage, series) in zip(conditions, expected_stages):
        if condition.get("benchmark_stage") != stage or condition.get("series") != series:
            raise MB8ValidationError("Idle-condition stage/order mismatch")
        if condition.get("known_safenest_workloads") != []:
            raise MB8ValidationError("Known SafeNest workload present at formal benchmark")
        if float(condition.get("observed_idle_seconds", 0.0)) < MINIMUM_IDLE_SECONDS:
            raise MB8ValidationError("Required 30-second idle stabilization was not satisfied")
        if condition.get("required_idle_seconds") != MINIMUM_IDLE_SECONDS:
            raise MB8ValidationError("Idle stabilization policy mismatch")
    for series, order in enumerate(FORMAL_SEED_ORDERS, 1):
        condition = conditions[series - 1]
        if condition.get("seed_order") != list(order):
            raise MB8ValidationError("Formal seed-order evidence mismatch")
    try:
        inputs = prepare_benchmark_inputs(root_dir)
    except Exception as exc:
        raise MB8ValidationError(f"Benchmark input provenance cannot be reconstructed: {exc}") from exc
    expected_identity = build_benchmark_input_evidence(inputs, artifacts)
    for field, expected in expected_identity.items():
        if environment.get(field) != expected:
            raise MB8ValidationError(f"Benchmark input identity mismatch: {field}")


def _load_raw_arrays(path: Path, run_index: Dict[str, Any]) -> Dict[str, np.ndarray]:
    expected_entries = list(run_index["formal_runs"]) + list(run_index["confirmation_runs"])
    expected_keys = {entry["raw_array_key"] for entry in expected_entries}
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_keys:
                raise MB8ValidationError("Raw latency NPZ schema/key mismatch")
            arrays: Dict[str, np.ndarray] = {}
            for entry in expected_entries:
                key = entry["raw_array_key"]
                value = archive[key]
                if value.dtype != np.dtype("int64") or value.ndim != 1:
                    raise MB8ValidationError(f"Raw latency dtype/shape mismatch: {key}")
                if value.size != int(entry["measured_iterations"]):
                    raise MB8ValidationError(f"Raw latency sample count mismatch: {key}")
                if np.any(value <= 0):
                    raise MB8ValidationError(f"Invalid non-positive raw latency: {key}")
                arrays[key] = value.copy()
            return arrays
    except MB8ValidationError:
        raise
    except Exception as exc:
        raise MB8ValidationError(f"Malformed latency_raw_samples.npz: {exc}") from exc


def _validate_memory(memory: Dict[str, Any]) -> None:
    if memory.get("measurement_type") != MEMORY_MEASUREMENT_TYPE:
        raise MB8ValidationError("False/unsupported memory measurement semantics")
    if memory.get("method") != MEMORY_METHOD:
        raise MB8ValidationError("Memory measurement method mismatch")
    semantics = str(memory.get("semantics", ""))
    if "PROCESS_LEVEL_RSS" not in semantics or "NOT_TFLITE_ARENA" not in semantics:
        raise MB8ValidationError("Memory observation is not clearly labeled as a process proxy")
    # The required proxy disclaimer intentionally contains the words
    # "MODEL_RAM_REQUIREMENT" in a negation.  Reject an affirmative claim,
    # while accepting that explicit disclaimer.
    if semantics.strip().startswith("MODEL_RAM_REQUIREMENT"):
        raise MB8ValidationError("Process RSS proxy falsely claimed as model RAM requirement")
    observations = memory.get("formal_series_observations")
    if not isinstance(observations, dict) or not observations:
        raise MB8ValidationError("Memory observation evidence missing")
    if not isinstance(memory.get("limitations"), list) or not memory["limitations"]:
        raise MB8ValidationError("Memory proxy limitations missing")
    peak = memory.get("peak_during_benchmark")
    if not isinstance(peak, dict) or peak.get("measurement_type") != (
        "PEAK_MODEL_MEMORY_NOT_RELIABLY_MEASURABLE_ON_CURRENT_MAC_RUNTIME"
    ):
        raise MB8ValidationError("Peak-memory limitation is not explicitly recorded")

    expected_series = {f"formal_series_{number:02d}" for number in range(1, FORMAL_SERIES_COUNT + 1)}
    expected_series.add("confirmation_series_01")
    if set(observations) != expected_series:
        raise MB8ValidationError("Memory-series coverage mismatch")

    def validate_snapshot(snapshot: Any, label: str) -> None:
        if not isinstance(snapshot, dict):
            raise MB8ValidationError(f"Missing RSS snapshot: {label}")
        if snapshot.get("measurement_type") != MEMORY_MEASUREMENT_TYPE:
            raise MB8ValidationError(f"RSS snapshot type mismatch: {label}")
        if snapshot.get("method") != MEMORY_METHOD:
            raise MB8ValidationError(f"RSS snapshot method mismatch: {label}")
        rss = snapshot.get("rss_bytes")
        if not isinstance(rss, int) or rss <= 0:
            raise MB8ValidationError(f"Invalid RSS snapshot bytes: {label}")

    expected_seed_keys = {str(seed) for seed in FROZEN_SEEDS}
    for series in range(1, FORMAL_SERIES_COUNT + 1):
        records = observations[f"formal_series_{series:02d}"]
        if not isinstance(records, dict) or set(records) != expected_seed_keys:
            raise MB8ValidationError("Formal memory seed coverage mismatch")
        for seed in FROZEN_SEEDS:
            record = records[str(seed)]
            if not isinstance(record, dict):
                raise MB8ValidationError("Malformed formal memory record")
            validate_snapshot(record.get("before_interpreter"), f"series{series}/seed{seed}/before")
            validate_snapshot(record.get("after_allocation"), f"series{series}/seed{seed}/allocation")
            for field in ("after_warmup", "after_measured_metrics"):
                samples = record.get(field)
                if not isinstance(samples, dict) or set(samples) != set(BENCHMARK_METRICS):
                    raise MB8ValidationError(f"Memory {field} metric coverage mismatch")
                for metric in BENCHMARK_METRICS:
                    validate_snapshot(samples[metric], f"series{series}/seed{seed}/{field}/{metric}")
    confirmation = observations["confirmation_series_01"]
    if not isinstance(confirmation, dict) or set(confirmation) != expected_seed_keys:
        raise MB8ValidationError("Confirmation memory seed coverage mismatch")


def validate_m_b8_artifacts(
    root_dir: Path = ROOT_DIR,
    manifest_dir: Optional[Path] = None,
    *,
    verify_upstream: bool = True,
) -> Dict[str, Any]:
    """Validate persisted M-B8 evidence without rerunning timing loops."""
    root_dir = Path(root_dir)
    if manifest_dir is None:
        manifest_dir = root_dir / MANIFEST_RELATIVE
    manifest_dir = Path(manifest_dir)
    if not manifest_dir.is_dir():
        raise MB8ValidationError(f"M-B8 manifest directory missing: {manifest_dir}")
    missing = sorted(name for name in REQUIRED_MB8_ARTIFACTS if not (manifest_dir / name).is_file())
    if missing:
        raise MB8ValidationError(f"Required M-B8 artifacts missing: {missing}")
    _validate_machine_paths(manifest_dir, root_dir)

    if verify_upstream:
        upstream = validate_m_b7_artifacts(root_dir=root_dir)
        if not upstream.get("validation_success"):
            raise MB8ValidationError("Upstream M-B7/A5/A6 validation did not pass")

    try:
        static = build_static_evidence(root_dir)
    except Exception as exc:
        raise MB8ValidationError(f"Frozen artifact/contract verification failed: {exc}") from exc

    json_artifacts = {
        filename: _load_json(manifest_dir / filename)
        for filename in REQUIRED_OUTPUT_FILENAMES
        if filename.endswith(".json")
    }
    _validate_input_identity(root_dir, json_artifacts["input_identity.json"])
    _require_equal(
        json_artifacts["input_identity.json"], static["input_identity.json"], "input identity"
    )
    _require_equal(
        json_artifacts["experiment_contract.json"],
        static["experiment_contract.json"],
        "experiment contract",
    )
    _require_equal(
        json_artifacts["benchmark_contract.json"],
        static["benchmark_contract.json"],
        "benchmark contract",
    )
    _require_equal(
        json_artifacts["artifact_footprint.json"],
        static["artifact_footprint.json"],
        "strict-INT8 artifact structure/footprint",
    )
    _require_equal(
        json_artifacts["locked_test_access_audit.json"],
        static["locked_test_access_audit.json"],
        "LOCKED_TEST audit",
    )
    _require_equal(
        json_artifacts["run_environment.json"], static["run_environment.json"], "run environment"
    )
    _validate_benchmark_environment(
        root_dir, json_artifacts["benchmark_environment.json"], static["artifacts"]
    )

    expected_index = make_run_index(static["artifacts"])
    run_index = json_artifacts["benchmark_run_index.json"]
    _require_equal(run_index, expected_index, "benchmark run-index/warm-up/count/order/thread metadata")
    raw_arrays = _load_raw_arrays(manifest_dir / "latency_raw_samples.npz", run_index)
    try:
        recomputed_latency = summarize_raw_samples(run_index, raw_arrays)
    except Exception as exc:
        raise MB8ValidationError(f"Raw latency validation failed: {exc}") from exc
    _require_equal(
        json_artifacts["latency_summary.json"], recomputed_latency, "latency summary statistic/unit"
    )
    recomputed_cross = cross_seed_latency_summary(recomputed_latency)
    _require_equal(
        json_artifacts["cross_seed_latency_summary.json"], recomputed_cross, "cross-seed latency summary"
    )
    _validate_memory(json_artifacts["memory_observation.json"])
    expected = build_complete_evidence(
        static,
        run_index,
        raw_arrays,
        json_artifacts["benchmark_environment.json"],
        json_artifacts["memory_observation.json"],
    )
    _require_equal(
        json_artifacts["exceptions.json"], expected["exceptions.json"], "exception classification"
    )
    _require_equal(
        json_artifacts["m_b8_summary.json"], expected["m_b8_summary.json"], "derived M-B8 summary"
    )
    _validate_checksums(manifest_dir)
    return {
        "validation_success": True,
        "m_b8_gate_status": json_artifacts["m_b8_summary.json"]["gate_status"],
        "m_b9_entry_status": "READY_AFTER_INDEPENDENT_REVIEW",
        "independently_measured": {
            "upstream_m_b7_a5_a6_verified": bool(verify_upstream),
            "input_identity_verified": True,
            "strict_int8_artifact_identity_structure_verified": True,
            "raw_latency_schema_and_counts_verified": True,
            "positive_nanosecond_samples_verified": True,
            "latency_statistics_and_units_recomputed": True,
            "percentile_method_verified": PERCENTILE_METHOD,
            "warmup_and_series_contract_verified": True,
            "benchmark_order_verified": True,
            "environment_idle_guard_verified": True,
            "artifact_footprint_parameter_count_verified": True,
            "memory_proxy_semantics_verified": True,
            "locked_test_access_blocked": True,
            "hardened_checksums_verified": True,
        },
    }


def main() -> None:
    result = validate_m_b8_artifacts()
    measured = result["independently_measured"]
    print("Standalone M-B8 Mac Latency & Footprint Validation Result:")
    print(f"Validation Success: {result['validation_success']}")
    print(f"M-B8 Gate Status: {result['m_b8_gate_status']}")
    print(f"M-B9 Entry Status: {result['m_b9_entry_status']}")
    print(f"Raw latency statistics recomputed: {measured['latency_statistics_and_units_recomputed']}")
    print(f"Benchmark idle guard verified: {measured['environment_idle_guard_verified']}")
    print(f"Actual strict-INT8 artifact gate: {measured['strict_int8_artifact_identity_structure_verified']}")
    print(f"LOCKED_TEST guard verified: {measured['locked_test_access_blocked']}")
    print(f"Hardened checksums verified: {measured['hardened_checksums_verified']}")


if __name__ == "__main__":
    main()
