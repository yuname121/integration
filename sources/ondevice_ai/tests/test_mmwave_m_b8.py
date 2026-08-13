"""Focused fail-closed corruption tests for SafeNest M-B8 evidence validation.

The fixtures use synthetic positive integer nanosecond arrays.  They exercise
the evidence/validator contract and never execute a wall-clock benchmark.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from mmwave_m_b8_benchmark import (  # noqa: E402
    BENCHMARK_METRICS,
    BenchmarkEnvironmentBlocked,
    DELEGATE_RUNTIME_MODE,
    FORMAL_SEED_ORDERS,
    FROZEN_SEEDS,
    MEMORY_MEASUREMENT_TYPE,
    MEMORY_METHOD,
    REQUIRED_OUTPUT_FILENAMES,
    build_complete_evidence,
    build_benchmark_input_evidence,
    build_static_evidence,
    classify_known_safenest_workload,
    make_run_index,
    prepare_benchmark_inputs,
    render_report,
    require_idle_stabilization,
    write_deterministic_npz,
    write_json,
)
from validate_mmwave_m_b8 import MB8ValidationError, validate_m_b8_artifacts  # noqa: E402


class TestMmwaveMB8(unittest.TestCase):
    """Ensure checksum-closed M-B8 evidence still fails on scientific corruption."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.static = build_static_evidence(ROOT_DIR)
        cls.benchmark_inputs = prepare_benchmark_inputs(ROOT_DIR)
        cls.run_index = make_run_index(cls.static["artifacts"])
        cls.raw_arrays = cls._synthetic_raw_arrays(cls.run_index)
        cls.environment = cls._synthetic_environment(cls.benchmark_inputs, cls.static["artifacts"])
        cls.memory = cls._synthetic_memory()
        cls.evidence = build_complete_evidence(
            cls.static,
            cls.run_index,
            cls.raw_arrays,
            cls.environment,
            cls.memory,
        )
        cls.fixture_dir = tempfile.TemporaryDirectory(prefix="safenest_m_b8_fixture_")
        cls.fixture_manifest = Path(cls.fixture_dir.name) / "M-B8_mac_latency_footprint"
        cls._write_evidence(cls.fixture_manifest, cls.evidence)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixture_dir.cleanup()

    @staticmethod
    def _synthetic_raw_arrays(run_index: Dict[str, Any]) -> Dict[str, np.ndarray]:
        arrays: Dict[str, np.ndarray] = {}
        entries = list(run_index["formal_runs"]) + list(run_index["confirmation_runs"])
        for ordinal, entry in enumerate(entries, 1):
            base = 100_000 + ordinal * 1_000
            arrays[entry["raw_array_key"]] = np.arange(
                base,
                base + int(entry["measured_iterations"]),
                dtype=np.int64,
            )
        return arrays

    @staticmethod
    def _synthetic_environment(
        inputs: Dict[str, Any], artifacts: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        conditions = []
        for series, order in enumerate(FORMAL_SEED_ORDERS, 1):
            conditions.append(
                {
                    "policy": "KNOWN_SAFENEST_WORKLOAD_ABSENT_CONTINUOUS_IDLE_STABILIZATION",
                    "required_idle_seconds": 30.0,
                    "observed_idle_seconds": 30.0,
                    "started_utc": "2026-08-11T00:00:00+00:00",
                    "ended_utc": "2026-08-11T00:00:30+00:00",
                    "known_safenest_workloads": [],
                    "load_indicator_samples": [],
                    "benchmark_stage": "FORMAL",
                    "series": series,
                    "seed_order": list(order),
                }
            )
        conditions.append(
            {
                "policy": "KNOWN_SAFENEST_WORKLOAD_ABSENT_CONTINUOUS_IDLE_STABILIZATION",
                "required_idle_seconds": 30.0,
                "observed_idle_seconds": 30.0,
                "started_utc": "2026-08-11T00:10:00+00:00",
                "ended_utc": "2026-08-11T00:10:30+00:00",
                "known_safenest_workloads": [],
                "load_indicator_samples": [],
                "benchmark_stage": "CONFIRMATION",
                "series": 1,
                "seed_order": list(FROZEN_SEEDS),
            }
        )
        return {
            "phase_id": "M-B8",
            "thread_configuration": {"num_threads": 1},
            "delegate_runtime_mode": DELEGATE_RUNTIME_MODE,
            "formal_benchmark_environment_ready": True,
            "known_safenest_workload_checks": conditions,
            "machine_model_identifier": "TEST_MAC",
            **build_benchmark_input_evidence(inputs, artifacts),
        }

    @staticmethod
    def _synthetic_memory() -> Dict[str, Any]:
        def snapshot() -> Dict[str, Any]:
            return {
                "measurement_type": MEMORY_MEASUREMENT_TYPE,
                "method": MEMORY_METHOD,
                "rss_bytes": 1,
                "timestamp_utc": "2026-08-11T00:00:00+00:00",
            }

        formal_observations: Dict[str, Any] = {}
        for series in range(1, 4):
            formal_observations[f"formal_series_{series:02d}"] = {
                str(seed): {
                    "before_interpreter": snapshot(),
                    "after_allocation": snapshot(),
                    "after_warmup": {metric: snapshot() for metric in BENCHMARK_METRICS},
                    "after_measured_metrics": {
                        metric: snapshot() for metric in BENCHMARK_METRICS
                    },
                    "interpreter_structure": {},
                }
                for seed in FROZEN_SEEDS
            }
        formal_observations["confirmation_series_01"] = {
            str(seed): {"interpreter_structure": {}} for seed in FROZEN_SEEDS
        }
        return {
            "phase_id": "M-B8",
            "measurement_type": MEMORY_MEASUREMENT_TYPE,
            "method": MEMORY_METHOD,
            "semantics": "PROCESS_LEVEL_RSS_OBSERVATION_NOT_TFLITE_ARENA_OR_MODEL_RAM_REQUIREMENT",
            "formal_series_observations": formal_observations,
            "peak_during_benchmark": {
                "measurement_type": "PEAK_MODEL_MEMORY_NOT_RELIABLY_MEASURABLE_ON_CURRENT_MAC_RUNTIME",
                "reason": "Synthetic fixture uses a snapshot-only proxy.",
            },
            "limitations": ["PROCESS_RSS_PROXY is not exact TFLite arena evidence."],
        }

    @classmethod
    def _write_evidence(cls, manifest: Path, evidence: Dict[str, Any]) -> None:
        manifest.mkdir(parents=True, exist_ok=True)
        for filename in REQUIRED_OUTPUT_FILENAMES:
            target = manifest / filename
            if filename.endswith(".npz"):
                write_deterministic_npz(target, evidence[filename])
            else:
                write_json(target, evidence[filename])
        cls._refresh_checksums_at(manifest)

    @staticmethod
    def _refresh_checksums_at(manifest: Path) -> None:
        lines = []
        for filename in REQUIRED_OUTPUT_FILENAMES:
            digest = hashlib.sha256((manifest / filename).read_bytes()).hexdigest()
            lines.append(f"{digest}  {filename}")
        (manifest / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="safenest_m_b8_test_")
        self.manifest = Path(self.temp_dir.name) / "M-B8_mac_latency_footprint"
        shutil.copytree(self.fixture_manifest, self.manifest)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _load_json(self, filename: str) -> Dict[str, Any]:
        return json.loads((self.manifest / filename).read_text(encoding="utf-8"))

    def _write_json(self, filename: str, value: Dict[str, Any]) -> None:
        write_json(self.manifest / filename, value)

    def _refresh_checksums(self) -> None:
        self._refresh_checksums_at(self.manifest)

    def _corrupt_json(self, filename: str, change: Callable[[Dict[str, Any]], None]) -> None:
        value = self._load_json(filename)
        change(value)
        self._write_json(filename, value)
        self._refresh_checksums()

    def _corrupt_npz(self, change: Callable[[Dict[str, np.ndarray]], None]) -> None:
        path = self.manifest / "latency_raw_samples.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        change(arrays)
        write_deterministic_npz(path, arrays)
        self._refresh_checksums()

    def _assert_rejected(self) -> None:
        with self.assertRaises(MB8ValidationError):
            validate_m_b8_artifacts(
                root_dir=ROOT_DIR,
                manifest_dir=self.manifest,
                verify_upstream=False,
            )

    def test_00_valid_synthetic_evidence_passes_without_benchmark_rerun(self) -> None:
        result = validate_m_b8_artifacts(
            root_dir=ROOT_DIR,
            manifest_dir=self.manifest,
            verify_upstream=False,
        )
        self.assertTrue(result["validation_success"])

    def test_01_rejects_wrong_model_sha(self) -> None:
        self._corrupt_json(
            "artifact_footprint.json",
            lambda value: value["strict_int8_artifacts"]["42"].__setitem__("sha256", "0" * 64),
        )
        self._assert_rejected()

    def test_02_rejects_wrong_artifact_byte_size(self) -> None:
        self._corrupt_json(
            "artifact_footprint.json",
            lambda value: value["strict_int8_artifacts"]["43"].__setitem__("bytes", 1),
        )
        self._assert_rejected()

    def test_03_rejects_wrong_input_dtype_evidence(self) -> None:
        self._corrupt_json(
            "artifact_footprint.json",
            lambda value: value["strict_int8_artifacts"]["44"].__setitem__("input_dtype", "float32"),
        )
        self._assert_rejected()

    def test_04_rejects_wrong_operator_evidence(self) -> None:
        def change(value: Dict[str, Any]) -> None:
            value["strict_int8_artifacts"]["42"]["op_types"].append("FlexFake")
            value["strict_int8_artifacts"]["42"]["select_tf_ops_count"] = 1

        self._corrupt_json("artifact_footprint.json", change)
        self._assert_rejected()

    def test_05_rejects_raw_latency_sample_count_corruption(self) -> None:
        self._corrupt_npz(lambda arrays: arrays.__setitem__(next(iter(arrays)), next(iter(arrays.values()))[:-1]))
        self._assert_rejected()

    def test_06_rejects_negative_latency(self) -> None:
        def change(arrays: Dict[str, np.ndarray]) -> None:
            key = next(iter(arrays))
            arrays[key][0] = -1

        self._corrupt_npz(change)
        self._assert_rejected()

    def test_07_rejects_latency_unit_conversion_corruption(self) -> None:
        def change(value: Dict[str, Any]) -> None:
            stats = value["per_seed"]["42"]["TFLITE_INVOKE_ONLY"]["pooled_formal"]["statistics_ms"]
            stats["mean"] += 1.0

        self._corrupt_json("latency_summary.json", change)
        self._assert_rejected()

    def test_08_rejects_mean_corruption(self) -> None:
        def change(value: Dict[str, Any]) -> None:
            value["per_seed"]["42"]["TFLITE_INVOKE_ONLY"]["pooled_formal"]["statistics_ns"]["mean"] += 1.0

        self._corrupt_json("latency_summary.json", change)
        self._assert_rejected()

    def test_09_rejects_median_corruption(self) -> None:
        def change(value: Dict[str, Any]) -> None:
            value["per_seed"]["42"]["TFLITE_INVOKE_ONLY"]["pooled_formal"]["statistics_ns"]["median"] += 1.0

        self._corrupt_json("latency_summary.json", change)
        self._assert_rejected()

    def test_10_rejects_p95_corruption(self) -> None:
        def change(value: Dict[str, Any]) -> None:
            value["per_seed"]["42"]["TFLITE_INVOKE_ONLY"]["pooled_formal"]["statistics_ns"]["p95"] += 1.0

        self._corrupt_json("latency_summary.json", change)
        self._assert_rejected()

    def test_11_rejects_p99_corruption(self) -> None:
        def change(value: Dict[str, Any]) -> None:
            value["per_seed"]["42"]["TFLITE_INVOKE_ONLY"]["pooled_formal"]["statistics_ns"]["p99"] += 1.0

        self._corrupt_json("latency_summary.json", change)
        self._assert_rejected()

    def test_12_rejects_percentile_method_mismatch(self) -> None:
        self._corrupt_json(
            "latency_summary.json",
            lambda value: value.__setitem__("numpy_percentile_method", "nearest"),
        )
        self._assert_rejected()

    def test_13_rejects_warmup_count_corruption(self) -> None:
        self._corrupt_json(
            "benchmark_run_index.json",
            lambda value: value["formal_runs"][0].__setitem__("warmup_iterations", 99),
        )
        self._assert_rejected()

    def test_14_rejects_measured_iteration_count_corruption(self) -> None:
        self._corrupt_json(
            "benchmark_run_index.json",
            lambda value: value["formal_runs"][0].__setitem__("measured_iterations", 999),
        )
        self._assert_rejected()

    def test_15_rejects_benchmark_order_corruption(self) -> None:
        def change(value: Dict[str, Any]) -> None:
            value["formal_seed_order_by_series"][0] = [44, 43, 42]

        self._corrupt_json("benchmark_run_index.json", change)
        self._assert_rejected()

    def test_16_rejects_thread_count_metadata_corruption(self) -> None:
        self._corrupt_json(
            "benchmark_run_index.json",
            lambda value: value["formal_runs"][0].__setitem__("thread_count", 2),
        )
        self._assert_rejected()

    def test_17_rejects_file_size_corruption(self) -> None:
        self._corrupt_json(
            "artifact_footprint.json",
            lambda value: value["strict_int8_artifacts"]["42"].__setitem__("bytes", 22081),
        )
        self._assert_rejected()

    def test_18_rejects_parameter_count_corruption(self) -> None:
        self._corrupt_json(
            "artifact_footprint.json",
            lambda value: value.__setitem__("parameter_count", 0),
        )
        self._assert_rejected()

    def test_19_rejects_false_memory_semantics(self) -> None:
        self._corrupt_json(
            "memory_observation.json",
            lambda value: value.__setitem__("semantics", "MODEL_RAM_REQUIREMENT"),
        )
        self._assert_rejected()

    def test_20_rejects_locked_test_access_insertion(self) -> None:
        self._corrupt_json(
            "locked_test_access_audit.json",
            lambda value: value.__setitem__("prediction_access_attempts", 1),
        )
        self._assert_rejected()

    def test_21_rejects_insufficient_idle_stabilization(self) -> None:
        def change(value: Dict[str, Any]) -> None:
            value["known_safenest_workload_checks"][0]["observed_idle_seconds"] = 29.9

        self._corrupt_json("benchmark_environment.json", change)
        self._assert_rejected()

    def test_22_rejects_input_identity_corruption(self) -> None:
        self._corrupt_json(
            "input_identity.json",
            lambda value: value["inputs"][0].__setitem__("measured_sha256", "f" * 64),
        )
        self._assert_rejected()

    def test_23_rejects_validation_tensor_identity_corruption(self) -> None:
        self._corrupt_json(
            "benchmark_environment.json",
            lambda value: value.__setitem__("m_b6_model_ready_float32_tensor_sha256", "0" * 64),
        )
        self._assert_rejected()

    def test_24_run_index_records_actual_strict_int8_provenance(self) -> None:
        entry = self.run_index["formal_runs"][0]
        artifact = self.static["artifacts"][entry["seed"]]
        self.assertEqual(entry["model_sha256"], artifact["sha256"])
        self.assertEqual(entry["model_relative_path"], artifact["relative_path"])
        self.assertEqual(entry["model_bytes"], artifact["bytes"])

    def test_25_idle_guard_requires_continuous_30_seconds_without_workload(self) -> None:
        clock = {"seconds": 0.0}

        def monotonic() -> float:
            return clock["seconds"]

        def sleeper(seconds: float) -> None:
            clock["seconds"] += seconds

        condition = require_idle_stabilization(
            minimum_seconds=30.0,
            poll_seconds=10.0,
            detector=lambda: [],
            sleeper=sleeper,
            monotonic_clock=monotonic,
        )
        self.assertGreaterEqual(condition["observed_idle_seconds"], 30.0)
        self.assertEqual(condition["known_safenest_workloads"], [])

    def test_26_idle_guard_blocks_known_safenest_workload_before_timing(self) -> None:
        with self.assertRaises(BenchmarkEnvironmentBlocked):
            require_idle_stabilization(
                detector=lambda: [{"pid": 123, "matched_pattern": "co2_c_b"}],
                sleeper=lambda _: None,
                monotonic_clock=lambda: 0.0,
            )

    def test_27_workload_detector_ignores_observer_shell_but_keeps_co2_worker(self) -> None:
        self.assertIsNone(
            classify_known_safenest_workload(
                "/bin/zsh", "-lc ps -axo command | rg run_mmwave_m_b8.py --formal"
            )
        )
        self.assertEqual(
            classify_known_safenest_workload(
                "/usr/bin/python3", "scripts/validate_co2_final_integrity.py"
            ),
            "validate_co2",
        )

    def test_28_report_renders_all_required_latency_sections_from_evidence(self) -> None:
        report = render_report(self.evidence)
        self.assertIn("Raw-sample provenance", report)
        self.assertIn("Per-seed preprocessing and quantization latency", report)
        self.assertIn("Mac-development reference comparison", report)
        self.assertIn("M-B8 benchmarks this specific Mac environment only", report)

    def test_29_rejects_malformed_checksum(self) -> None:
        path = self.manifest / "checksums.sha256"
        path.write_text(path.read_text(encoding="utf-8") + "bad-checksum\n", encoding="utf-8")
        self._assert_rejected()

    def test_30_rejects_checksum_path_traversal(self) -> None:
        path = self.manifest / "checksums.sha256"
        lines = path.read_text(encoding="utf-8").splitlines()
        digest = lines[0].split(maxsplit=1)[0]
        lines[0] = f"{digest}  ../escaped.json"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._assert_rejected()


if __name__ == "__main__":
    unittest.main()
