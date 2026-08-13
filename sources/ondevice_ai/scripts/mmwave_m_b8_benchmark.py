#!/usr/bin/env python3
"""SafeNest M-B8 macOS offline latency and footprint benchmark helpers.

This module has no training, conversion, model-selection, or LOCKED_TEST path.
Formal timing is intentionally exposed only through the runner's explicit
``--formal`` mode, which enforces an idle/stabilization gate before every
formal series.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from mmwave_m_b1_preprocessing import (  # noqa: E402
    compute_tensor_fingerprint,
    fit_train_zscore_statistics,
    transform_signals,
)
from mmwave_m_b3_architecture import build_model_by_id  # noqa: E402
from mmwave_phase_b_access import PhaseBAccessGuard  # noqa: E402


PHASE_ID = "M-B8"
ARCHITECTURE_ID = "M-B3_CONV1D_GAP_BASELINE"
CALIBRATION_PROFILE_ID = "M-B5_CAL_CLASS_BALANCED_120"
PREPROCESSING_PROFILE_ID = "M-B1_D0_B1_Z1"
PREPROCESSING_PROFILE_NAME = "BPF_ZSCORE"
IMBALANCE_STRATEGY_ID = "M-B2_CE_UNWEIGHTED"
FROZEN_SEEDS: Tuple[int, ...] = (42, 43, 44)
NUM_THREADS = 1
WARMUP_ITERATIONS = 100
FORMAL_MEASURED_ITERATIONS = 1000
FORMAL_SERIES_COUNT = 3
CONFIRMATION_MEASURED_ITERATIONS = 300
PERCENTILE_METHOD = "linear"
MINIMUM_IDLE_SECONDS = 30.0
INPUT_CYCLE_SIZE = 79

METRIC_TFLITE_INVOKE_ONLY = "TFLITE_INVOKE_ONLY"
METRIC_PREPROCESSING_ONLY = "PREPROCESSING_ONLY"
METRIC_QUANTIZATION_ONLY = "QUANTIZATION_ONLY"
METRIC_PIPELINE = "PREPROCESSING_QUANTIZATION_INVOKE"
BENCHMARK_METRICS: Tuple[str, ...] = (
    METRIC_TFLITE_INVOKE_ONLY,
    METRIC_PREPROCESSING_ONLY,
    METRIC_QUANTIZATION_ONLY,
    METRIC_PIPELINE,
)
FORMAL_SEED_ORDERS: Tuple[Tuple[int, ...], ...] = (
    (42, 43, 44),
    (43, 44, 42),
    (44, 42, 43),
)
DELEGATE_RUNTIME_MODE = "DEFAULT_TFLITE_CPU_RUNTIME_WITH_AUTOMATIC_XNNPACK_CPU_DELEGATE_OBSERVED"
MEMORY_MEASUREMENT_TYPE = "PROCESS_RSS_PROXY"
MEMORY_METHOD = "PS_RSS_KIB_PROCESS_PROXY"

MANIFEST_RELATIVE = Path("datasets/mmwave/manifests/M-B8_mac_latency_footprint")
REPORT_RELATIVE = Path("docs/reports/20260811_Codex_M-B8_Mac_Latency_Footprint_01.md")
REQUIRED_OUTPUT_FILENAMES: Tuple[str, ...] = (
    "input_identity.json",
    "experiment_contract.json",
    "benchmark_environment.json",
    "benchmark_contract.json",
    "benchmark_run_index.json",
    "latency_raw_samples.npz",
    "latency_summary.json",
    "cross_seed_latency_summary.json",
    "artifact_footprint.json",
    "memory_observation.json",
    "locked_test_access_audit.json",
    "run_environment.json",
    "exceptions.json",
    "m_b8_summary.json",
)

INPUT_IDENTITY_PATHS: Tuple[Tuple[str, str], ...] = (
    ("requirements-mac.txt", "Pinned macOS dependency contract"),
    ("datasets/mmwave/manifests/M-B0_evaluation_protocol/m_b0_summary.json", "M-B0 validator-passed summary"),
    ("datasets/mmwave/manifests/M-B1_preprocessing_ablation/m_b1_summary.json", "M-B1 validator-passed summary"),
    ("datasets/mmwave/manifests/M-B1_preprocessing_ablation/selected_preprocessing_profile.json", "Frozen M-B1 preprocessing"),
    ("datasets/mmwave/manifests/M-B1_preprocessing_ablation/train_fit_statistics.json", "Frozen M-B1 TRAIN-fit statistics"),
    ("datasets/mmwave/manifests/M-B1_preprocessing_ablation/preprocessing_fingerprints.json", "Frozen M-B1 tensor fingerprints"),
    ("datasets/mmwave/manifests/M-B1_preprocessing_ablation/checksums.sha256", "M-B1 checksum closure"),
    ("datasets/mmwave/manifests/M-B3_architecture_comparison/m_b3_summary.json", "M-B3 validator-passed summary"),
    ("datasets/mmwave/manifests/M-B3_architecture_comparison/architecture_profiles.json", "Frozen M-B3 architecture definitions"),
    ("datasets/mmwave/manifests/M-B3_architecture_comparison/checksums.sha256", "M-B3 checksum closure"),
    ("datasets/mmwave/manifests/M-B4_multiseed_stability/m_b4_summary.json", "M-B4 validator-passed summary"),
    ("datasets/mmwave/manifests/M-B4_multiseed_stability/primary_float_finalist.json", "Frozen M-B4 primary finalist"),
    ("datasets/mmwave/manifests/M-B4_multiseed_stability/seed_weights.npz", "Frozen M-B4 three-seed weights"),
    ("datasets/mmwave/manifests/M-B4_multiseed_stability/checksums.sha256", "M-B4 checksum closure"),
    ("datasets/mmwave/manifests/M-B5_representative_calibration/m_b5_summary.json", "M-B5 validator-passed summary"),
    ("datasets/mmwave/manifests/M-B5_representative_calibration/selected_calibration_profile.json", "Frozen M-B5 calibration selection"),
    ("datasets/mmwave/manifests/M-B5_representative_calibration/checksums.sha256", "M-B5 checksum closure"),
    ("datasets/mmwave/manifests/M-B6_stage_equivalence/m_b6_summary.json", "M-B6 validator-passed summary"),
    ("datasets/mmwave/manifests/M-B6_stage_equivalence/stage_artifact_manifest.json", "M-B6 strict-INT8 artifact identities"),
    ("datasets/mmwave/manifests/M-B6_stage_equivalence/per_seed_stage_metrics.json", "M-B6 Stage-C evidence"),
    ("datasets/mmwave/manifests/M-B6_stage_equivalence/checksums.sha256", "M-B6 checksum closure"),
    ("datasets/mmwave/manifests/M-B7_perturbation_robustness/m_b7_summary.json", "M-B7 validator-passed summary"),
    ("datasets/mmwave/manifests/M-B7_perturbation_robustness/checksums.sha256", "M-B7 checksum closure"),
    ("datasets/mmwave/manifests/a5_subject_split/subject_split_manifest.jsonl", "Immutable A5 subject assignment"),
    ("datasets/mmwave/splits/mmwave_real_subject_split_v1.json", "Immutable A5 split lookup"),
    ("datasets/mmwave/processed/mmwave_canonical_real_v1.npy", "A6 canonical numeric matrix"),
    ("datasets/mmwave/manifests/a6_full_conversion/full_window_manifest.jsonl", "A6 canonical window manifest"),
    ("datasets/mmwave/manifests/a6_full_conversion/full_provenance_manifest.jsonl", "A6 provenance manifest"),
    ("datasets/mmwave/manifests/a6_full_conversion/checksums.sha256", "A6 checksum closure"),
)


class MB8Error(RuntimeError):
    """Base M-B8 error."""


class BenchmarkEnvironmentBlocked(MB8Error):
    """Raised when a formal benchmark cannot establish an idle environment."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(array: np.ndarray, dtype: Optional[np.dtype] = None) -> str:
    value = np.asarray(array, dtype=dtype) if dtype is not None else np.asarray(array)
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _round(value: float, digits: int = 9) -> float:
    return round(float(value), digits)


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_deterministic_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    """Write stable NPZ bytes without changing raw sample values or dtypes."""
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def raw_array_key(stage: str, series: int, seed: int, metric: str) -> str:
    return f"{stage.lower()}__series_{series:02d}__seed_{seed}__{metric}__ns"


def run_id(stage: str, series: int, seed: int, metric: str) -> str:
    return f"{stage.upper()}_SERIES_{series:02d}_SEED_{seed}_{metric}"


def benchmark_statistics(samples_ns: np.ndarray) -> Dict[str, Any]:
    """Compute contract statistics from positive integer nanosecond samples."""
    values = np.asarray(samples_ns)
    if values.ndim != 1 or values.size == 0:
        raise MB8Error("Latency samples must be a non-empty 1D array")
    if not np.issubdtype(values.dtype, np.integer):
        raise MB8Error("Raw latency samples must use an integer nanosecond dtype")
    if np.any(values <= 0):
        raise MB8Error("Latency samples must all be positive nanoseconds")
    float_values = values.astype(np.float64)
    percentiles = np.percentile(float_values, [50, 90, 95, 99], method=PERCENTILE_METHOD)
    stats_ns = {
        "count": int(values.size),
        "mean": float(np.mean(float_values)),
        "median": float(np.median(float_values)),
        "std": float(np.std(float_values)),
        "p50": float(percentiles[0]),
        "p90": float(percentiles[1]),
        "p95": float(percentiles[2]),
        "p99": float(percentiles[3]),
        "min": int(np.min(values)),
        "max": int(np.max(values)),
    }
    stats_ms = {
        key: (_round(value / 1_000_000.0, 9) if key != "count" else value)
        for key, value in stats_ns.items()
    }
    return {
        "raw_unit": "ns",
        "summary_unit": "ms",
        "numpy_percentile_method": PERCENTILE_METHOD,
        "all_valid_samples_included": True,
        "statistics_ns": stats_ns,
        "statistics_ms": stats_ms,
        "coefficient_of_variation": _round(stats_ns["std"] / stats_ns["mean"], 12),
    }


def quantize_model_input(model_input: np.ndarray, input_scale: float, input_zero_point: int) -> np.ndarray:
    """Apply the exact strict-INT8 quantization used by M-B6/M-B7."""
    source = np.asarray(model_input, dtype=np.float32)
    if source.shape == (300,):
        source = source.reshape(1, 300, 1)
    elif source.shape == (300, 1):
        source = source.reshape(1, 300, 1)
    if source.shape != (1, 300, 1):
        raise MB8Error(f"Expected model input shape (1,300,1), got {source.shape}")
    quantized_raw = np.round(source / input_scale + input_zero_point)
    return np.clip(quantized_raw, -128, 127).astype(np.int8)


class TFLiteBenchmarkSession:
    """One strict-INT8 interpreter reused for all warm-up and timed work in a series."""

    def __init__(self, model_path: Path, num_threads: int = NUM_THREADS) -> None:
        self.model_path = Path(model_path)
        self.model_bytes = self.model_path.read_bytes()
        self.sha256 = hashlib.sha256(self.model_bytes).hexdigest()
        self.num_threads = int(num_threads)
        if self.num_threads <= 0:
            raise MB8Error("num_threads must be positive")
        self.interpreter = tf.lite.Interpreter(
            model_content=self.model_bytes,
            num_threads=self.num_threads,
        )
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.input_index = int(self.input_detail["index"])
        self.output_index = int(self.output_detail["index"])
        self.input_scale = float(self.input_detail["quantization"][0])
        self.input_zero_point = int(self.input_detail["quantization"][1])
        self.output_scale = float(self.output_detail["quantization"][0])
        self.output_zero_point = int(self.output_detail["quantization"][1])
        self.op_types = [entry["op_name"] for entry in self.interpreter._get_ops_details()]

    def structure(self, relative_path: str) -> Dict[str, Any]:
        return {
            "relative_path": relative_path,
            "bytes": len(self.model_bytes),
            "sha256": self.sha256,
            "input_dtype": self.input_detail["dtype"].__name__,
            "output_dtype": self.output_detail["dtype"].__name__,
            "input_shape": [int(v) for v in self.input_detail["shape"]],
            "output_shape": [int(v) for v in self.output_detail["shape"]],
            "input_scale": self.input_scale,
            "input_zero_point": self.input_zero_point,
            "output_scale": self.output_scale,
            "output_zero_point": self.output_zero_point,
            "op_types": list(self.op_types),
            "select_tf_ops_count": sum(
                1 for name in self.op_types if "Flex" in name or "Select" in name
            ),
            "thread_count": self.num_threads,
            "delegate_runtime_mode": DELEGATE_RUNTIME_MODE,
        }

    def set_input(self, quantized_input: np.ndarray) -> None:
        value = np.asarray(quantized_input, dtype=np.int8)
        if value.shape != (1, 300, 1):
            raise MB8Error(f"Expected quantized input (1,300,1), got {value.shape}")
        self.interpreter.set_tensor(self.input_index, value)

    def invoke(self) -> None:
        self.interpreter.invoke()

    def output_int8(self) -> np.ndarray:
        return np.asarray(self.interpreter.get_tensor(self.output_index), dtype=np.int8)


def load_json(root_dir: Path, relative_path: str) -> Any:
    return json.loads((Path(root_dir) / relative_path).read_text(encoding="utf-8"))


def verify_frozen_contracts(root_dir: Path) -> Dict[str, Any]:
    """Load and independently cross-check the immutable Phase-B selection state."""
    selected_preprocessing = load_json(
        root_dir,
        "datasets/mmwave/manifests/M-B1_preprocessing_ablation/selected_preprocessing_profile.json",
    )
    selected_imbalance = load_json(
        root_dir,
        "datasets/mmwave/manifests/M-B2_class_imbalance/selected_imbalance_strategy.json",
    )
    primary = load_json(
        root_dir,
        "datasets/mmwave/manifests/M-B4_multiseed_stability/primary_float_finalist.json",
    )
    calibration = load_json(
        root_dir,
        "datasets/mmwave/manifests/M-B5_representative_calibration/selected_calibration_profile.json",
    )
    mb6 = load_json(root_dir, "datasets/mmwave/manifests/M-B6_stage_equivalence/m_b6_summary.json")
    mb7 = load_json(root_dir, "datasets/mmwave/manifests/M-B7_perturbation_robustness/m_b7_summary.json")
    if selected_preprocessing.get("selected_profile_id") != PREPROCESSING_PROFILE_ID:
        raise MB8Error("Frozen M-B1 preprocessing profile mismatch")
    if selected_preprocessing.get("selected_profile_name") != PREPROCESSING_PROFILE_NAME:
        raise MB8Error("Frozen M-B1 preprocessing name mismatch")
    if selected_imbalance.get("selected_strategy_id") != IMBALANCE_STRATEGY_ID:
        raise MB8Error("Frozen M-B2 imbalance strategy mismatch")
    if primary.get("primary_stable_float_finalist") != ARCHITECTURE_ID:
        raise MB8Error("Frozen M-B4 architecture mismatch")
    if calibration.get("selected_calibration_profile") != CALIBRATION_PROFILE_ID:
        raise MB8Error("Frozen M-B5 calibration profile mismatch")
    if mb6.get("frozen_weight_seeds") != list(FROZEN_SEEDS):
        raise MB8Error("Frozen M-B6 seed set mismatch")
    if mb7.get("frozen_seeds") != list(FROZEN_SEEDS):
        raise MB8Error("Frozen M-B7 seed set mismatch")
    return {
        "preprocessing": selected_preprocessing,
        "imbalance": selected_imbalance,
        "primary": primary,
        "calibration": calibration,
        "m_b6": mb6,
        "m_b7": mb7,
    }


def load_frozen_artifacts(root_dir: Path, num_threads: int = NUM_THREADS) -> Dict[int, Dict[str, Any]]:
    """Inspect exact M-B6-qualified strict-INT8 files without invoking them."""
    manifest = load_json(
        root_dir, "datasets/mmwave/manifests/M-B6_stage_equivalence/stage_artifact_manifest.json"
    )["artifacts"]
    artifacts: Dict[int, Dict[str, Any]] = {}
    for seed in FROZEN_SEEDS:
        key = f"{ARCHITECTURE_ID}_seed_{seed}_stage_c"
        metadata = manifest.get(key)
        if not metadata:
            raise MB8Error(f"M-B6 stage artifact missing for seed {seed}")
        relative_path = metadata["relative_path"]
        model_path = Path(root_dir) / relative_path
        session = TFLiteBenchmarkSession(model_path, num_threads=num_threads)
        structure = session.structure(relative_path)
        if structure["sha256"] != metadata.get("sha256") or structure["bytes"] != metadata.get("bytes"):
            raise MB8Error(f"Strict-INT8 artifact identity mismatch for seed {seed}")
        if (
            structure["input_dtype"] != "int8"
            or structure["output_dtype"] != "int8"
            or structure["input_shape"] != [1, 300, 1]
            or structure["output_shape"] != [1, 3]
            or structure["select_tf_ops_count"] != 0
        ):
            raise MB8Error(f"Strict-INT8 runtime structure mismatch for seed {seed}")
        artifacts[seed] = structure
    return artifacts


def verify_parameter_count() -> Dict[str, Any]:
    """Independently instantiate the frozen architecture only to inspect its parameter count."""
    model = build_model_by_id(ARCHITECTURE_ID, input_shape=(300, 1))
    return {
        "architecture": ARCHITECTURE_ID,
        "parameter_count": int(model.count_params()),
        "verification_method": "FRESH_FROZEN_ARCHITECTURE_CONSTRUCTION_COUNT_PARAMS",
    }


def prepare_benchmark_inputs(root_dir: Path) -> Dict[str, Any]:
    """Prepare 79 deterministic VALIDATION inputs outside all timed intervals."""
    guard = PhaseBAccessGuard(root_dir=Path(root_dir))
    train_data = guard.get_model_selection_dataset("TRAIN")
    validation_data = guard.get_model_selection_dataset("VALIDATION")
    if len(train_data["windows"]) != 327 or len(validation_data["windows"]) != INPUT_CYCLE_SIZE:
        raise MB8Error("Authoritative pure-class TRAIN/VALIDATION counts mismatch")
    subjects = sorted({window["subject_id"] for window in validation_data["windows"]})
    if len(subjects) != 17:
        raise MB8Error("Authoritative VALIDATION subject count mismatch")
    zscore_stats = fit_train_zscore_statistics(train_data["signals"], detrend=False, bpf=True)
    stored_stats = load_json(
        root_dir, "datasets/mmwave/manifests/M-B1_preprocessing_ablation/train_fit_statistics.json"
    )["zscore_statistics"][PREPROCESSING_PROFILE_ID]
    if zscore_stats["mean"] != stored_stats["mean"] or zscore_stats["std"] != stored_stats["std"]:
        raise MB8Error("M-B1 TRAIN-fitted Z-score statistics mismatch")
    # M-B1 fingerprints the preprocessing tensor in float64.  M-B6 then
    # converts that exact frozen-preprocessor result to float32 before the
    # strict-INT8 input quantizer.  Preserve and verify both representations
    # rather than comparing the float32 model-ready tensor to M-B1's float64
    # provenance hash.
    preprocessed_inputs_float64 = transform_signals(
        validation_data["signals"],
        detrend=False,
        bpf=True,
        zscore=True,
        zscore_stats=zscore_stats,
    )
    expected_fingerprint = load_json(
        root_dir,
        "datasets/mmwave/manifests/M-B1_preprocessing_ablation/preprocessing_fingerprints.json",
    )["fingerprints"][PREPROCESSING_PROFILE_ID]["validation_tensor_sha256"]
    if compute_tensor_fingerprint(preprocessed_inputs_float64) != expected_fingerprint:
        raise MB8Error("M-B1 VALIDATION preprocessing tensor fingerprint mismatch")
    model_inputs = preprocessed_inputs_float64.astype(np.float32)
    identity_rows = [
        {
            "canonical_sample_index": int(window["canonical_sample_index"]),
            "window_id": window["window_id"],
            "subject_id": window["subject_id"],
            "recording_id": window["recording_id"],
            "split": window["split"],
        }
        for window in validation_data["windows"]
    ]
    if any(row["split"] != "VALIDATION" for row in identity_rows):
        raise MB8Error("Benchmark input policy attempted non-VALIDATION access")
    return {
        "canonical_inputs": np.asarray(validation_data["signals"], dtype=np.float64),
        "model_inputs": model_inputs,
        "zscore_stats": zscore_stats,
        "canonical_validation_tensor_sha256": array_sha256(validation_data["signals"], np.float64),
        "m_b1_preprocessed_validation_tensor_sha256": expected_fingerprint,
        "m_b6_model_ready_float32_tensor_sha256": array_sha256(model_inputs, np.float32),
        "m_b1_preprocessed_tensor_shape": [int(value) for value in preprocessed_inputs_float64.shape],
        "m_b1_preprocessed_tensor_dtype": preprocessed_inputs_float64.dtype.name,
        "model_ready_tensor_shape": [int(value) for value in model_inputs.shape],
        "model_ready_tensor_dtype": model_inputs.dtype.name,
        "input_cycle_identity_sha256": hashlib.sha256(
            json.dumps(identity_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "input_cycle_size": len(identity_rows),
        "validation_subject_count": len(subjects),
        "validation_window_count": len(identity_rows),
        "input_policy": "AUTHORITATIVE_PURE_CLASS_VALIDATION_DETERMINISTIC_CYCLING",
    }


def build_quantized_input_identities(
    artifacts: Dict[int, Dict[str, Any]], model_inputs: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """Hash each seed's precomputed strict-INT8 invoke cycle outside timing."""
    identities: Dict[str, Dict[str, Any]] = {}
    for seed in FROZEN_SEEDS:
        artifact = artifacts[seed]
        quantized_cycle = np.stack(
            [
                quantize_model_input(
                    value, artifact["input_scale"], artifact["input_zero_point"]
                )
                for value in model_inputs
            ],
            axis=0,
        )
        identities[str(seed)] = {
            "model_sha256": artifact["sha256"],
            "input_scale": artifact["input_scale"],
            "input_zero_point": artifact["input_zero_point"],
            "quantized_cycle_sha256": array_sha256(quantized_cycle, np.int8),
            "quantized_cycle_shape": [int(value) for value in quantized_cycle.shape],
            "quantized_cycle_dtype": quantized_cycle.dtype.name,
            "precomputed_outside_timed_interval": True,
        }
    return identities


def build_benchmark_input_evidence(
    inputs: Dict[str, Any], artifacts: Dict[int, Dict[str, Any]]
) -> Dict[str, Any]:
    """Return the persisted identity record for all in-memory benchmark inputs."""
    return {
        "input_cycle_identity_sha256": inputs["input_cycle_identity_sha256"],
        "input_cycle_size": inputs["input_cycle_size"],
        "canonical_validation_tensor_sha256": inputs["canonical_validation_tensor_sha256"],
        "m_b1_preprocessed_validation_tensor_sha256": inputs[
            "m_b1_preprocessed_validation_tensor_sha256"
        ],
        "m_b6_model_ready_float32_tensor_sha256": inputs[
            "m_b6_model_ready_float32_tensor_sha256"
        ],
        "m_b1_preprocessed_tensor_shape": inputs["m_b1_preprocessed_tensor_shape"],
        "m_b1_preprocessed_tensor_dtype": inputs["m_b1_preprocessed_tensor_dtype"],
        "model_ready_tensor_shape": inputs["model_ready_tensor_shape"],
        "model_ready_tensor_dtype": inputs["model_ready_tensor_dtype"],
        "precomputed_strict_int8_input_cycles": build_quantized_input_identities(
            artifacts, inputs["model_inputs"]
        ),
    }


def build_input_identity(root_dir: Path, artifacts: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    inputs: List[Dict[str, str]] = []
    for relative_path, role in INPUT_IDENTITY_PATHS:
        path = Path(root_dir) / relative_path
        if not path.is_file():
            raise MB8Error(f"Required M-B8 input missing: {relative_path}")
        inputs.append(
            {
                "repository_relative_path": relative_path,
                "measured_sha256": file_sha256(path),
                "evidence_role": role,
            }
        )
    for seed in FROZEN_SEEDS:
        artifact = artifacts[seed]
        inputs.append(
            {
                "repository_relative_path": artifact["relative_path"],
                "measured_sha256": artifact["sha256"],
                "evidence_role": f"Frozen M-B6/M-B7 strict-INT8 finalist for seed {seed}",
            }
        )
    return {"phase_id": PHASE_ID, "total_inputs": len(inputs), "inputs": inputs}


def benchmark_contract() -> Dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "scientific_scope": ["MAC_OFFLINE_LATENCY", "OFFLINE_MODEL_FOOTPRINT"],
        "architecture": ARCHITECTURE_ID,
        "frozen_seeds": list(FROZEN_SEEDS),
        "frozen_preprocessing_profile": PREPROCESSING_PROFILE_ID,
        "frozen_preprocessing_name": PREPROCESSING_PROFILE_NAME,
        "frozen_imbalance_strategy": IMBALANCE_STRATEGY_ID,
        "frozen_calibration_profile": CALIBRATION_PROFILE_ID,
        "thread_configuration": {"primary_num_threads": NUM_THREADS, "exploratory_configurations": []},
        "delegate_runtime_mode": DELEGATE_RUNTIME_MODE,
        "clock": {"name": "time.perf_counter_ns", "raw_unit": "ns", "monotonic": True},
        "warmup_iterations": WARMUP_ITERATIONS,
        "formal_measured_iterations_per_series": FORMAL_MEASURED_ITERATIONS,
        "formal_series_count": FORMAL_SERIES_COUNT,
        "formal_seed_order_by_series": [list(order) for order in FORMAL_SEED_ORDERS],
        "confirmation": {
            "warmup_iterations": WARMUP_ITERATIONS,
            "measured_iterations": CONFIRMATION_MEASURED_ITERATIONS,
            "metric": METRIC_TFLITE_INVOKE_ONLY,
            "median_difference_warning_ratio": 0.2,
        },
        "input_policy": {
            "source_split": "VALIDATION",
            "source_window_count": INPUT_CYCLE_SIZE,
            "input_cycle": "DETERMINISTIC_ROUND_ROBIN_79_WINDOWS",
            "m_b1_preprocessing_provenance_dtype": "float64",
            "m_b6_model_ready_dtype": "float32",
            "locked_test_access": "PROHIBITED",
            "file_loading_in_timed_region": False,
            "dataset_loading_in_timed_region": False,
            "interpreter_construction_in_timed_region": False,
            "window_acquisition_in_timed_region": False,
        },
        "metrics": {
            METRIC_TFLITE_INVOKE_ONLY: {
                "timed_operations": ["interpreter.invoke"],
                "excluded_operations": ["set_tensor", "output_dequantization", "argmax", "input_quantization"],
                "quantized_inputs_precomputed_outside_timed_region": True,
            },
            METRIC_PREPROCESSING_ONLY: {
                "timed_operations": ["BPF", "TRAIN_FITTED_ZSCORE", "FLOAT64_TO_FLOAT32_MODEL_READY_CAST"],
                "excluded_operations": ["quantization", "set_tensor", "interpreter.invoke", "output_dequantization", "argmax"],
            },
            METRIC_QUANTIZATION_ONLY: {
                "timed_operations": ["INT8_INPUT_QUANTIZATION"],
                "excluded_operations": ["BPF", "Z_SCORE", "set_tensor", "interpreter.invoke", "output_dequantization", "argmax"],
            },
            METRIC_PIPELINE: {
                "timed_operations": ["BPF", "TRAIN_FITTED_ZSCORE", "FLOAT64_TO_FLOAT32_MODEL_READY_CAST", "INT8_INPUT_QUANTIZATION", "set_tensor", "interpreter.invoke"],
                "excluded_operations": ["output_dequantization", "argmax", "InferenceResult_construction"],
                "canonical_window_already_resident_in_memory": True,
            },
        },
        "percentile": {"method": PERCENTILE_METHOD, "all_valid_samples_included": True},
        "outlier_policy": "NO_VALID_TIMING_OUTLIERS_REMOVED",
        "idle_stabilization": {
            "required_seconds": MINIMUM_IDLE_SECONDS,
            "known_safenest_workload_must_be_absent": True,
            "environment_disturbed_policy": "ABORT_SERIES_AND_REQUIRE_NEW_IDLE_STABILIZATION",
        },
        "window_semantics": {
            "window_acquisition_duration_seconds": 30.0,
            "compute_latency_excludes_window_acquisition": True,
            "claim_term": "MAC_STEADY_STATE_INFERENCE_LATENCY",
        },
        "model_trainings": 0,
        "model_conversions": 0,
        "float_tflite_reference": {"measured": False, "reason": "STRICT_INT8_PRIMARY_SCOPE_NOT_EXPANDED"},
    }


def build_experiment_contract(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "scientific_scope": "MAC_OFFLINE_LATENCY_AND_FOOTPRINT",
        "evaluation_population": "AUTHORITATIVE_PURE_CLASS_VALIDATION_INPUTS_ONLY",
        "validation_pure_class_windows": INPUT_CYCLE_SIZE,
        "validation_subjects": 17,
        "architecture": ARCHITECTURE_ID,
        "frozen_training_seeds": list(FROZEN_SEEDS),
        "frozen_preprocessing_profile": PREPROCESSING_PROFILE_ID,
        "frozen_imbalance_strategy": IMBALANCE_STRATEGY_ID,
        "frozen_calibration_profile": CALIBRATION_PROFILE_ID,
        "input_identity_count": inputs["total_inputs"],
        "model_trainings": 0,
        "model_conversions": 0,
        "locked_test_performance_access": 0,
        "locked_test_policy": "ZERO_PERFORMANCE_PREDICTION_LABEL_ACCESS",
        "claim_limits": [
            "NOT_RASPBERRY_PI_LATENCY",
            "NOT_REAL_SENSOR_LATENCY",
            "NOT_SENSOR_TO_ALARM_LATENCY",
            "NOT_PRODUCTION_REALTIME_VALIDATED",
            "NOT_MR60_RUNTIME_VALIDATED",
        ],
    }


def artifact_footprint(artifacts: Dict[int, Dict[str, Any]], parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "architecture": ARCHITECTURE_ID,
        "parameter_count": parameters["parameter_count"],
        "parameter_count_verification_method": parameters["verification_method"],
        "strict_int8_artifacts": {str(seed): artifacts[seed] for seed in FROZEN_SEEDS},
        "float_tflite_reference": {
            "measured": False,
            "reason": "STRICT_INT8_PRIMARY_SCOPE_NOT_EXPANDED",
        },
        "file_size_is_not_memory_claim": True,
    }


def _sysctl_value(name: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=False
        )
        value = result.stdout.strip()
        return value or None
    except OSError:
        return None


def _sw_vers_value(flag: str) -> Optional[str]:
    try:
        result = subprocess.run(["sw_vers", flag], capture_output=True, text=True, check=False)
        value = result.stdout.strip()
        return value or None
    except OSError:
        return None


def _power_source() -> str:
    try:
        result = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=False)
        text = result.stdout
        if "AC Power" in text:
            return "AC"
        if "Battery Power" in text:
            return "BATTERY"
    except OSError:
        pass
    return "UNKNOWN"


def capture_machine_environment(idle_conditions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    memsize = _sysctl_value("hw.memsize")
    return {
        "phase_id": PHASE_ID,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "operating_system": platform.system(),
        "macos_version": _sw_vers_value("-productVersion"),
        "macos_build": _sw_vers_value("-buildVersion"),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "machine_model_identifier": _sysctl_value("hw.model"),
        "chip_identifier": _sysctl_value("machdep.cpu.brand_string"),
        "logical_cpu_count": int(_sysctl_value("hw.logicalcpu") or os.cpu_count() or 0),
        "physical_cpu_count": int(_sysctl_value("hw.physicalcpu") or 0),
        "performance_core_count": int(_sysctl_value("hw.perflevel0.physicalcpu") or 0),
        "efficiency_core_count": int(_sysctl_value("hw.perflevel1.physicalcpu") or 0),
        "total_ram_bytes": int(memsize) if memsize else None,
        "power_source": _power_source(),
        "process_architecture": platform.machine(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "thread_configuration": {"num_threads": NUM_THREADS},
        "delegate_runtime_mode": DELEGATE_RUNTIME_MODE,
        "known_safenest_workload_checks": list(idle_conditions),
        "formal_benchmark_environment_ready": all(
            condition.get("known_safenest_workloads") == []
            and float(condition.get("observed_idle_seconds", 0.0)) >= MINIMUM_IDLE_SECONDS
            for condition in idle_conditions
        ),
        "relevant_environment_variables": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")
        },
    }


def run_environment(root_dir: Path) -> Dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "process_architecture": platform.machine(),
        "requirements_mac_sha256": file_sha256(Path(root_dir) / "requirements-mac.txt"),
    }


def process_rss_proxy() -> Dict[str, Any]:
    """Capture current process RSS through macOS ps; never call it a TFLite arena size."""
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            check=False,
        )
        rss_kib = int(result.stdout.strip())
        return {
            "measurement_type": MEMORY_MEASUREMENT_TYPE,
            "method": MEMORY_METHOD,
            "rss_bytes": rss_kib * 1024,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    except (OSError, ValueError):
        return {
            "measurement_type": "PEAK_MODEL_MEMORY_NOT_RELIABLY_MEASURABLE_ON_CURRENT_MAC_RUNTIME",
            "method": "UNAVAILABLE",
            "rss_bytes": None,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }


SAFE_NEST_WORKLOAD_PATTERNS: Tuple[str, ...] = (
    "train_mmwave",
    "run_mmwave",
    "validate_mmwave",
    "train_co2",
    "run_co2",
    "validate_co2",
    "co2_c_b",
    "train_thermal",
    "run_thermal",
    "validate_thermal",
    "thermal_t_",
    "pytest",
    "unittest",
)


def classify_known_safenest_workload(command: str, args: str) -> Optional[str]:
    """Identify actual SafeNest Python/test jobs without matching observer shells.

    A shell command that merely *mentions* ``run_mmwave`` (for example a
    process-status query) is not a workload.  SafeNest training/evaluation
    jobs in this repository execute under Python or pytest, so constrain the
    detector to those executable families before inspecting their arguments.
    """
    executable = Path(command).name.lower()
    is_python = "python" in executable
    is_pytest = "pytest" in executable or executable in {"py.test", "pytest"}
    if not (is_python or is_pytest):
        return None
    lowered = args.lower()
    if is_pytest:
        return "pytest"
    return next(
        (pattern for pattern in SAFE_NEST_WORKLOAD_PATTERNS if pattern in lowered), None
    )


def _ancestor_pids() -> set:
    ancestors = {os.getpid()}
    current = os.getppid()
    while current > 1 and current not in ancestors:
        ancestors.add(current)
        try:
            result = subprocess.run(
                ["ps", "-o", "ppid=", "-p", str(current)],
                capture_output=True,
                text=True,
                check=False,
            )
            next_parent = int(result.stdout.strip())
        except (OSError, ValueError):
            break
        if next_parent == current:
            break
        current = next_parent
    return ancestors


def find_known_safenest_workloads() -> List[Dict[str, Any]]:
    """Return sanitized evidence of other known SafeNest CPU workloads.

    Current process and its shell ancestors are excluded so an explicit formal
    benchmark does not block itself.  Full command lines are hashed rather than
    persisted to keep machine-specific paths out of evidence.
    """
    excluded = _ancestor_pids()
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,comm=,args="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    matches: List[Dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        if pid in excluded:
            continue
        command = parts[2]
        args = parts[3]
        matched_pattern = classify_known_safenest_workload(command, args)
        if matched_pattern:
            matches.append(
                {
                    "pid": pid,
                    "ppid": ppid,
                    "matched_pattern": matched_pattern,
                    "command_sha256": hashlib.sha256(args.encode("utf-8")).hexdigest(),
                }
            )
    return sorted(matches, key=lambda item: item["pid"])


def load_indicator() -> Dict[str, Any]:
    try:
        first, five, fifteen = os.getloadavg()
        return {"load_average_1m": first, "load_average_5m": five, "load_average_15m": fifteen}
    except (AttributeError, OSError):
        return {"load_average_1m": None, "load_average_5m": None, "load_average_15m": None}


def require_idle_stabilization(
    minimum_seconds: float = MINIMUM_IDLE_SECONDS,
    poll_seconds: float = 1.0,
    detector: Callable[[], List[Dict[str, Any]]] = find_known_safenest_workloads,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> Dict[str, Any]:
    """Require continuous known-workload absence for the full stabilization interval."""
    if minimum_seconds < 0 or poll_seconds <= 0:
        raise MB8Error("Invalid idle stabilization configuration")
    initial = detector()
    if initial:
        raise BenchmarkEnvironmentBlocked(
            "FORMAL_BENCHMARK_BLOCKED_KNOWN_SAFENEST_WORKLOAD_ACTIVE: "
            + json.dumps(initial, sort_keys=True)
        )
    started_wall = dt.datetime.now(dt.timezone.utc).isoformat()
    started_mono = monotonic_clock()
    samples: List[Dict[str, Any]] = []
    while True:
        elapsed = monotonic_clock() - started_mono
        active = detector()
        if active:
            raise BenchmarkEnvironmentBlocked(
                "FORMAL_BENCHMARK_ENVIRONMENT_DISTURBED_DURING_STABILIZATION: "
                + json.dumps(active, sort_keys=True)
            )
        samples.append({"elapsed_seconds": _round(elapsed, 6), **load_indicator()})
        if elapsed >= minimum_seconds:
            break
        sleeper(min(poll_seconds, max(0.0, minimum_seconds - elapsed)))
    observed = monotonic_clock() - started_mono
    return {
        "policy": "KNOWN_SAFENEST_WORKLOAD_ABSENT_CONTINUOUS_IDLE_STABILIZATION",
        "required_idle_seconds": minimum_seconds,
        "observed_idle_seconds": _round(observed, 6),
        "started_utc": started_wall,
        "ended_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "known_safenest_workloads": [],
        "load_indicator_samples": samples,
    }


def _warmup_invoke(session: TFLiteBenchmarkSession, quantized_inputs: np.ndarray, count: int) -> None:
    for index in range(count):
        session.set_input(quantized_inputs[index % len(quantized_inputs)])
        session.invoke()


def _warmup_preprocessing(canonical_inputs: np.ndarray, zscore_stats: Dict[str, float], count: int) -> None:
    for index in range(count):
        transform_signals(
            canonical_inputs[index % len(canonical_inputs)].reshape(1, 300),
            detrend=False,
            bpf=True,
            zscore=True,
            zscore_stats=zscore_stats,
        ).astype(np.float32)


def _warmup_quantization(model_inputs: np.ndarray, session: TFLiteBenchmarkSession, count: int) -> None:
    for index in range(count):
        quantize_model_input(
            model_inputs[index % len(model_inputs)], session.input_scale, session.input_zero_point
        )


def _warmup_pipeline(
    canonical_inputs: np.ndarray,
    zscore_stats: Dict[str, float],
    session: TFLiteBenchmarkSession,
    count: int,
) -> None:
    for index in range(count):
        preprocessed = transform_signals(
            canonical_inputs[index % len(canonical_inputs)].reshape(1, 300),
            detrend=False,
            bpf=True,
            zscore=True,
            zscore_stats=zscore_stats,
        )[0].astype(np.float32)
        session.set_input(quantize_model_input(preprocessed, session.input_scale, session.input_zero_point))
        session.invoke()


def measure_metric(
    metric: str,
    session: TFLiteBenchmarkSession,
    canonical_inputs: np.ndarray,
    model_inputs: np.ndarray,
    quantized_inputs: np.ndarray,
    zscore_stats: Dict[str, float],
    warmup_iterations: int,
    measured_iterations: int,
    after_warmup_capture: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """Warm and time one precisely defined metric, returning positive int64 nanoseconds."""
    if measured_iterations <= 0:
        raise MB8Error("Measured iteration count must be positive")
    if metric == METRIC_TFLITE_INVOKE_ONLY:
        _warmup_invoke(session, quantized_inputs, warmup_iterations)
        if after_warmup_capture is not None:
            after_warmup_capture(metric)
        samples = np.empty(measured_iterations, dtype=np.int64)
        for index in range(measured_iterations):
            session.set_input(quantized_inputs[index % len(quantized_inputs)])
            started = time.perf_counter_ns()
            session.invoke()
            samples[index] = time.perf_counter_ns() - started
    elif metric == METRIC_PREPROCESSING_ONLY:
        _warmup_preprocessing(canonical_inputs, zscore_stats, warmup_iterations)
        if after_warmup_capture is not None:
            after_warmup_capture(metric)
        samples = np.empty(measured_iterations, dtype=np.int64)
        for index in range(measured_iterations):
            started = time.perf_counter_ns()
            transform_signals(
                canonical_inputs[index % len(canonical_inputs)].reshape(1, 300),
                detrend=False,
                bpf=True,
                zscore=True,
                zscore_stats=zscore_stats,
            ).astype(np.float32)
            samples[index] = time.perf_counter_ns() - started
    elif metric == METRIC_QUANTIZATION_ONLY:
        _warmup_quantization(model_inputs, session, warmup_iterations)
        if after_warmup_capture is not None:
            after_warmup_capture(metric)
        samples = np.empty(measured_iterations, dtype=np.int64)
        for index in range(measured_iterations):
            started = time.perf_counter_ns()
            quantize_model_input(
                model_inputs[index % len(model_inputs)], session.input_scale, session.input_zero_point
            )
            samples[index] = time.perf_counter_ns() - started
    elif metric == METRIC_PIPELINE:
        _warmup_pipeline(canonical_inputs, zscore_stats, session, warmup_iterations)
        if after_warmup_capture is not None:
            after_warmup_capture(metric)
        samples = np.empty(measured_iterations, dtype=np.int64)
        for index in range(measured_iterations):
            started = time.perf_counter_ns()
            preprocessed = transform_signals(
                canonical_inputs[index % len(canonical_inputs)].reshape(1, 300),
                detrend=False,
                bpf=True,
                zscore=True,
                zscore_stats=zscore_stats,
            )[0].astype(np.float32)
            session.set_input(
                quantize_model_input(preprocessed, session.input_scale, session.input_zero_point)
            )
            session.invoke()
            samples[index] = time.perf_counter_ns() - started
    else:
        raise MB8Error(f"Unknown M-B8 metric: {metric}")
    if np.any(samples <= 0):
        raise MB8Error(f"Invalid non-positive latency sample in {metric}")
    return samples


def benchmark_seed_series(
    root_dir: Path,
    seed: int,
    canonical_inputs: np.ndarray,
    model_inputs: np.ndarray,
    zscore_stats: Dict[str, float],
    warmup_iterations: int,
    measured_iterations: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Run one seed's full formal metric set with one interpreter reuse."""
    stage_manifest = load_json(
        root_dir, "datasets/mmwave/manifests/M-B6_stage_equivalence/stage_artifact_manifest.json"
    )["artifacts"]
    artifact = stage_manifest[f"{ARCHITECTURE_ID}_seed_{seed}_stage_c"]
    before = process_rss_proxy()
    session = TFLiteBenchmarkSession(Path(root_dir) / artifact["relative_path"], num_threads=NUM_THREADS)
    after_allocation = process_rss_proxy()
    quantized_inputs = np.stack(
        [quantize_model_input(value, session.input_scale, session.input_zero_point) for value in model_inputs],
        axis=0,
    )
    results: Dict[str, np.ndarray] = {}
    after_warmup: Dict[str, Dict[str, Any]] = {}
    after_measured_metrics: Dict[str, Dict[str, Any]] = {}
    for metric in BENCHMARK_METRICS:
        results[metric] = measure_metric(
            metric,
            session,
            canonical_inputs,
            model_inputs,
            quantized_inputs,
            zscore_stats,
            warmup_iterations,
            measured_iterations,
            after_warmup_capture=lambda metric_name: after_warmup.__setitem__(
                metric_name, process_rss_proxy()
            ),
        )
        after_measured_metrics[metric] = process_rss_proxy()
    return results, {
        "before_interpreter": before,
        "after_allocation": after_allocation,
        "after_warmup": after_warmup,
        "after_measured_metrics": after_measured_metrics,
        "interpreter_structure": session.structure(artifact["relative_path"]),
    }


def benchmark_confirmation_invoke(
    root_dir: Path,
    seed: int,
    model_inputs: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run the pre-registered small invoke-only confirmation series."""
    stage_manifest = load_json(
        root_dir, "datasets/mmwave/manifests/M-B6_stage_equivalence/stage_artifact_manifest.json"
    )["artifacts"]
    artifact = stage_manifest[f"{ARCHITECTURE_ID}_seed_{seed}_stage_c"]
    session = TFLiteBenchmarkSession(Path(root_dir) / artifact["relative_path"], num_threads=NUM_THREADS)
    quantized_inputs = np.stack(
        [quantize_model_input(value, session.input_scale, session.input_zero_point) for value in model_inputs],
        axis=0,
    )
    samples = measure_metric(
        METRIC_TFLITE_INVOKE_ONLY,
        session,
        np.empty((0, 300), dtype=np.float64),
        model_inputs,
        quantized_inputs,
        {},
        WARMUP_ITERATIONS,
        CONFIRMATION_MEASURED_ITERATIONS,
    )
    return samples, {"interpreter_structure": session.structure(artifact["relative_path"])}


def make_run_index(artifacts: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Return the immutable raw-sample provenance schema/order."""
    if set(artifacts) != set(FROZEN_SEEDS):
        raise MB8Error("Run-index artifact provenance must cover exactly all frozen seeds")

    def model_provenance(seed: int) -> Dict[str, Any]:
        artifact = artifacts[seed]
        return {
            "model_relative_path": artifact["relative_path"],
            "model_sha256": artifact["sha256"],
            "model_bytes": artifact["bytes"],
            "metric_definition_reference": "benchmark_contract.json.metrics",
        }

    formal_runs: List[Dict[str, Any]] = []
    for series, order in enumerate(FORMAL_SEED_ORDERS, 1):
        for position, seed in enumerate(order, 1):
            for metric in BENCHMARK_METRICS:
                formal_runs.append(
                    {
                        "run_id": run_id("FORMAL", series, seed, metric),
                        "stage": "FORMAL",
                        "series": series,
                        "seed": seed,
                        "metric": metric,
                        "raw_array_key": raw_array_key("FORMAL", series, seed, metric),
                        "warmup_iterations": WARMUP_ITERATIONS,
                        "measured_iterations": FORMAL_MEASURED_ITERATIONS,
                        "thread_count": NUM_THREADS,
                        "delegate_runtime_mode": DELEGATE_RUNTIME_MODE,
                        "input_cycle_policy": "DETERMINISTIC_ROUND_ROBIN_79_WINDOWS",
                        "input_cycle_size": INPUT_CYCLE_SIZE,
                        "series_seed_order": list(order),
                        "seed_position_in_series": position,
                        "warmup_samples_stored": False,
                        **model_provenance(seed),
                    }
                )
    confirmation_runs = [
        {
            "run_id": run_id("CONFIRMATION", 1, seed, METRIC_TFLITE_INVOKE_ONLY),
            "stage": "CONFIRMATION",
            "series": 1,
            "seed": seed,
            "metric": METRIC_TFLITE_INVOKE_ONLY,
            "raw_array_key": raw_array_key("CONFIRMATION", 1, seed, METRIC_TFLITE_INVOKE_ONLY),
            "warmup_iterations": WARMUP_ITERATIONS,
            "measured_iterations": CONFIRMATION_MEASURED_ITERATIONS,
            "thread_count": NUM_THREADS,
            "delegate_runtime_mode": DELEGATE_RUNTIME_MODE,
            "input_cycle_policy": "DETERMINISTIC_ROUND_ROBIN_79_WINDOWS",
            "input_cycle_size": INPUT_CYCLE_SIZE,
            "series_seed_order": list(FROZEN_SEEDS),
            "seed_position_in_series": position,
            "warmup_samples_stored": False,
            **model_provenance(seed),
        }
        for position, seed in enumerate(FROZEN_SEEDS, 1)
    ]
    return {
        "phase_id": PHASE_ID,
        "raw_sample_unit": "ns",
        "formal_seed_order_by_series": [list(order) for order in FORMAL_SEED_ORDERS],
        "formal_runs": formal_runs,
        "confirmation_runs": confirmation_runs,
    }


def summarize_raw_samples(run_index: Dict[str, Any], raw_arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Recompute every persisted summary from raw integer nanosecond arrays."""
    expected_runs = list(run_index["formal_runs"]) + list(run_index["confirmation_runs"])
    expected_keys = {entry["raw_array_key"] for entry in expected_runs}
    if set(raw_arrays) != expected_keys:
        raise MB8Error("Raw latency array keys do not match benchmark run index")
    per_run: Dict[str, Any] = {}
    per_seed: Dict[str, Dict[str, Any]] = {str(seed): {} for seed in FROZEN_SEEDS}
    for entry in expected_runs:
        array = np.asarray(raw_arrays[entry["raw_array_key"]])
        if array.size != int(entry["measured_iterations"]):
            raise MB8Error(f"Raw sample count mismatch for {entry['run_id']}")
        stats = benchmark_statistics(array)
        per_run[entry["run_id"]] = {
            "raw_array_key": entry["raw_array_key"],
            "stage": entry["stage"],
            "series": entry["series"],
            "seed": entry["seed"],
            "metric": entry["metric"],
            **stats,
        }
    for seed in FROZEN_SEEDS:
        seed_key = str(seed)
        for metric in BENCHMARK_METRICS:
            formal_entries = [
                entry
                for entry in run_index["formal_runs"]
                if entry["seed"] == seed and entry["metric"] == metric
            ]
            ordered_arrays = [np.asarray(raw_arrays[entry["raw_array_key"]]) for entry in formal_entries]
            per_seed[seed_key][metric] = {
                "pooled_formal": benchmark_statistics(np.concatenate(ordered_arrays)),
                "per_series": {
                    str(entry["series"]): per_run[entry["run_id"]]
                    for entry in formal_entries
                },
            }
        confirmation_entry = next(
            entry for entry in run_index["confirmation_runs"] if entry["seed"] == seed
        )
        per_seed[seed_key]["confirmation_invoke_only"] = per_run[confirmation_entry["run_id"]]
    return {
        "phase_id": PHASE_ID,
        "raw_sample_unit": "ns",
        "summary_unit": "ms",
        "numpy_percentile_method": PERCENTILE_METHOD,
        "per_run": per_run,
        "per_seed": per_seed,
    }


def cross_seed_latency_summary(latency_summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for metric in BENCHMARK_METRICS:
        per_seed_stats = {
            str(seed): latency_summary["per_seed"][str(seed)][metric]["pooled_formal"]
            for seed in FROZEN_SEEDS
        }
        medians = np.asarray(
            [per_seed_stats[str(seed)]["statistics_ns"]["median"] for seed in FROZEN_SEEDS],
            dtype=np.float64,
        )
        p99s = np.asarray(
            [per_seed_stats[str(seed)]["statistics_ns"]["p99"] for seed in FROZEN_SEEDS],
            dtype=np.float64,
        )
        metrics[metric] = {
            "per_seed_pooled_statistics": per_seed_stats,
            "mean_of_seed_medians_ns": float(np.mean(medians)),
            "minimum_seed_median_ns": float(np.min(medians)),
            "maximum_seed_median_ns": float(np.max(medians)),
            "maximum_seed_p99_ns": float(np.max(p99s)),
            "minimum_seed_p99_ns": float(np.min(p99s)),
            "median_relative_spread": float((np.max(medians) - np.min(medians)) / np.min(medians)),
        }
    return {
        "phase_id": PHASE_ID,
        "cross_seed_metrics": metrics,
        "seed_selection_performed": False,
        "interpretation": "SEED_RUNTIME_DIFFERENCES_ARE_DESCRIPTIVE_ONLY",
    }


def confirmation_stability(latency_summary: Dict[str, Any]) -> Dict[str, Any]:
    findings: Dict[str, Any] = {}
    warnings: List[int] = []
    for seed in FROZEN_SEEDS:
        seed_key = str(seed)
        primary = latency_summary["per_seed"][seed_key][METRIC_TFLITE_INVOKE_ONLY]["pooled_formal"]
        confirmation = latency_summary["per_seed"][seed_key]["confirmation_invoke_only"]
        primary_median = primary["statistics_ns"]["median"]
        confirmation_median = confirmation["statistics_ns"]["median"]
        median_difference = abs(confirmation_median - primary_median) / primary_median
        p95_ratio = confirmation["statistics_ns"]["p95"] / primary["statistics_ns"]["p95"]
        warning = median_difference > 0.2
        if warning:
            warnings.append(seed)
        findings[seed_key] = {
            "primary_pooled_invoke_statistics": primary,
            "confirmation_invoke_statistics": confirmation,
            "median_ratio_confirmation_to_primary": float(confirmation_median / primary_median),
            "median_difference_ratio": float(median_difference),
            "p95_ratio_confirmation_to_primary": float(p95_ratio),
            "instability_warning_threshold": 0.2,
            "environment_instability_warning": warning,
        }
    return {
        "phase_id": PHASE_ID,
        "pre_registered_median_difference_warning_ratio": 0.2,
        "per_seed": findings,
        "warning_seeds": warnings,
    }


def build_memory_observation(series_memory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "measurement_type": MEMORY_MEASUREMENT_TYPE,
        "method": MEMORY_METHOD,
        "semantics": "PROCESS_LEVEL_RSS_OBSERVATION_NOT_TFLITE_ARENA_OR_MODEL_RAM_REQUIREMENT",
        "formal_series_observations": series_memory,
        "peak_during_benchmark": {
            "measurement_type": "PEAK_MODEL_MEMORY_NOT_RELIABLY_MEASURABLE_ON_CURRENT_MAC_RUNTIME",
            "reason": "The available ps RSS interface provides snapshots, not a reliable continuous process peak.",
        },
        "limitations": [
            "RSS includes Python, TensorFlow, allocator state, and other process allocations.",
            "PROCESS_RSS_PROXY must not be interpreted as exact TFLite arena bytes.",
        ],
    }


def build_static_evidence(root_dir: Path) -> Dict[str, Any]:
    contracts = verify_frozen_contracts(root_dir)
    artifacts = load_frozen_artifacts(root_dir)
    parameters = verify_parameter_count()
    if parameters["parameter_count"] != 9315:
        raise MB8Error("Frozen architecture parameter count does not equal authoritative 9315")
    identity = build_input_identity(root_dir, artifacts)
    return {
        "contracts": contracts,
        "artifacts": artifacts,
        "parameters": parameters,
        "input_identity.json": identity,
        "experiment_contract.json": build_experiment_contract(identity),
        "benchmark_contract.json": benchmark_contract(),
        "artifact_footprint.json": artifact_footprint(artifacts, parameters),
        "locked_test_access_audit.json": {
            "phase_id": PHASE_ID,
            "performance_access_attempts": 0,
            "prediction_access_attempts": 0,
            "label_access_attempts": 0,
            "lock_preserved": True,
            "benchmark_input_split": "VALIDATION",
            "locked_test_inputs_loaded": False,
        },
        "run_environment.json": run_environment(root_dir),
    }


def build_complete_evidence(
    static: Dict[str, Any],
    run_index: Dict[str, Any],
    raw_arrays: Dict[str, np.ndarray],
    environment: Dict[str, Any],
    memory: Dict[str, Any],
) -> Dict[str, Any]:
    latency_summary = summarize_raw_samples(run_index, raw_arrays)
    cross_seed = cross_seed_latency_summary(latency_summary)
    confirmation = confirmation_stability(latency_summary)
    invoke_cross = cross_seed["cross_seed_metrics"][METRIC_TFLITE_INVOKE_ONLY]
    pipeline_cross = cross_seed["cross_seed_metrics"][METRIC_PIPELINE]
    reference = {
        "median_under_5ms_all_seeds": all(
            latency_summary["per_seed"][str(seed)][METRIC_TFLITE_INVOKE_ONLY]["pooled_formal"]
            ["statistics_ms"]["median"] < 5.0
            for seed in FROZEN_SEEDS
        ),
        "p99_under_15ms_all_seeds": all(
            latency_summary["per_seed"][str(seed)][METRIC_TFLITE_INVOKE_ONLY]["pooled_formal"]
            ["statistics_ms"]["p99"] < 15.0
            for seed in FROZEN_SEEDS
        ),
        "interpretation": "MAC_DEVELOPMENT_REFERENCE_ONLY",
    }
    warnings = []
    if confirmation["warning_seeds"]:
        warnings.append("BENCHMARK_ENVIRONMENT_INSTABILITY_WARNING")
    findings = [
        {
            "classification": "NON-BLOCKING IMPROVEMENT",
            "code": "MAC_SPECIFIC_RUNTIME_EVIDENCE_ONLY",
            "detail": "M-B8 measures this Mac environment only, not Raspberry Pi, MR60, or sensor-to-alarm latency.",
        },
        {
            "classification": "NON-BLOCKING IMPROVEMENT",
            "code": "PROCESS_RSS_PROXY_LIMITATION",
            "detail": "Process RSS is observational and is not exact TFLite arena or model RAM evidence.",
        },
    ]
    if confirmation["warning_seeds"]:
        findings.append(
            {
                "classification": "REQUIRED REFINEMENT",
                "code": "BENCHMARK_ENVIRONMENT_INSTABILITY_WARNING",
                "detail": f"Confirmation invoke median differed by more than 20% for seeds {confirmation['warning_seeds']}.",
            }
        )
    summary = {
        "phase_id": PHASE_ID,
        "gate_status": "PASS_WITH_WARNINGS",
        "next_phase_authorized": False,
        "scientific_scope": "MAC_OFFLINE_LATENCY_AND_FOOTPRINT",
        "architecture": ARCHITECTURE_ID,
        "frozen_seeds": list(FROZEN_SEEDS),
        "calibration_profile": CALIBRATION_PROFILE_ID,
        "model_trainings": 0,
        "model_conversions": 0,
        "validation_windows": INPUT_CYCLE_SIZE,
        "validation_subjects": 17,
        "locked_test_access_attempts": 0,
        "formal_series_count": FORMAL_SERIES_COUNT,
        "measured_iterations_per_series": FORMAL_MEASURED_ITERATIONS,
        "warmup_iterations": WARMUP_ITERATIONS,
        "confirmation_measured_iterations": CONFIRMATION_MEASURED_ITERATIONS,
        "mac_development_reference": reference,
        "cross_seed_invoke": invoke_cross,
        "cross_seed_pipeline": pipeline_cross,
        "confirmation_warning_seeds": confirmation["warning_seeds"],
        "warnings": warnings + [finding["code"] for finding in findings],
        "blockers": [],
    }
    return {
        "input_identity.json": static["input_identity.json"],
        "experiment_contract.json": static["experiment_contract.json"],
        "benchmark_environment.json": environment,
        "benchmark_contract.json": static["benchmark_contract.json"],
        "benchmark_run_index.json": run_index,
        "latency_raw_samples.npz": raw_arrays,
        "latency_summary.json": latency_summary,
        "cross_seed_latency_summary.json": cross_seed,
        "artifact_footprint.json": static["artifact_footprint.json"],
        "memory_observation.json": memory,
        "locked_test_access_audit.json": static["locked_test_access_audit.json"],
        "run_environment.json": static["run_environment.json"],
        "exceptions.json": {"phase_id": PHASE_ID, "findings": findings, "blocker_count": 0},
        "m_b8_summary.json": summary,
        "_confirmation_stability": confirmation,
    }


def render_report(evidence: Dict[str, Any]) -> str:
    summary = evidence["m_b8_summary.json"]
    environment = evidence["benchmark_environment.json"]
    latency = evidence["latency_summary.json"]
    cross = evidence["cross_seed_latency_summary.json"]["cross_seed_metrics"]
    footprint = evidence["artifact_footprint.json"]
    confirmation = evidence["_confirmation_stability"]
    lines = [
        "# SafeNest mmWave M-B8 — macOS Offline Latency & Footprint",
        "",
        f"- Scope: `{summary['scientific_scope']}`",
        f"- Machine: `{environment.get('machine_model_identifier')}` / `{environment.get('chip_identifier')}` / macOS `{environment.get('macos_version')}`",
        f"- Runtime: TensorFlow `{environment.get('tensorflow_version')}`, `num_threads={NUM_THREADS}`, `{DELEGATE_RUNTIME_MODE}`",
        f"- Inputs: {INPUT_CYCLE_SIZE} deterministic VALIDATION windows; LOCKED_TEST access `0`",
        f"- Formal policy: {FORMAL_SERIES_COUNT} rotated series × {FORMAL_MEASURED_ITERATIONS} samples/seed/metric after {WARMUP_ITERATIONS} warm-ups",
        "",
        "## Timing definitions",
        "",
        "- `TFLITE_INVOKE_ONLY`: `set_tensor` is outside the timed interval; only `interpreter.invoke()` is timed.",
        "- `PREPROCESSING_ONLY`: frozen M-B1 BPF + TRAIN-fitted Z-score and the M-B6 model-ready `float32` cast on an in-memory canonical 300-sample window.",
        "- `QUANTIZATION_ONLY`: frozen strict-INT8 input quantization only.",
        "- `PREPROCESSING_QUANTIZATION_INVOKE`: preprocessing, quantization, `set_tensor`, and invoke; output dequantization/argmax excluded.",
        "",
        "## Raw-sample provenance",
        "",
        f"- `latency_raw_samples.npz` contains {len(latency['per_run'])} positive integer-nanosecond arrays; warm-up samples are excluded.",
        "- `benchmark_run_index.json` binds every array to its seed, strict-INT8 model path/SHA/bytes, metric, thread count, delegate/runtime mode, warm-up count, series, and deterministic 79-window cycle.",
        "- All primary summaries use every valid measured sample with NumPy percentile method `linear`; no latency outliers were removed.",
        "",
        "## Per-seed strict-INT8 latency",
        "",
        "| Seed | Invoke median ms | Invoke P95 ms | Invoke P99 ms | Pipeline median ms | Pipeline P95 ms | Pipeline P99 ms |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in FROZEN_SEEDS:
        per_seed = latency["per_seed"][str(seed)]
        invoke = per_seed[METRIC_TFLITE_INVOKE_ONLY]["pooled_formal"]["statistics_ms"]
        pipeline = per_seed[METRIC_PIPELINE]["pooled_formal"]["statistics_ms"]
        lines.append(
            f"| {seed} | {invoke['median']:.9f} | {invoke['p95']:.9f} | {invoke['p99']:.9f} | "
            f"{pipeline['median']:.9f} | {pipeline['p95']:.9f} | {pipeline['p99']:.9f} |"
        )
    lines.extend(
        [
            "",
            "## Per-seed preprocessing and quantization latency",
            "",
            "| Seed | Preprocessing median ms | Preprocessing P95 ms | Preprocessing P99 ms | Quantization median ms | Quantization P95 ms | Quantization P99 ms |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in FROZEN_SEEDS:
        per_seed = latency["per_seed"][str(seed)]
        preprocessing = per_seed[METRIC_PREPROCESSING_ONLY]["pooled_formal"]["statistics_ms"]
        quantization = per_seed[METRIC_QUANTIZATION_ONLY]["pooled_formal"]["statistics_ms"]
        lines.append(
            f"| {seed} | {preprocessing['median']:.9f} | {preprocessing['p95']:.9f} | {preprocessing['p99']:.9f} | "
            f"{quantization['median']:.9f} | {quantization['p95']:.9f} | {quantization['p99']:.9f} |"
        )
    invoke_cross = cross[METRIC_TFLITE_INVOKE_ONLY]
    pipeline_cross = cross[METRIC_PIPELINE]
    lines.extend(
        [
            "",
            "## Cross-seed runtime summary",
            "",
            f"- Mean of invoke medians: `{invoke_cross['mean_of_seed_medians_ns'] / 1_000_000.0:.9f} ms`",
            f"- Invoke median relative spread: `{invoke_cross['median_relative_spread']:.9f}`",
            f"- Maximum invoke P99: `{invoke_cross['maximum_seed_p99_ns'] / 1_000_000.0:.9f} ms`",
            f"- Maximum pipeline P99: `{pipeline_cross['maximum_seed_p99_ns'] / 1_000_000.0:.9f} ms`",
            "",
            "## Mac-development reference comparison",
            "",
            f"- All invoke medians below 5 ms: `{summary['mac_development_reference']['median_under_5ms_all_seeds']}`",
            f"- All invoke P99 values below 15 ms: `{summary['mac_development_reference']['p99_under_15ms_all_seeds']}`",
            f"- Interpretation: `{summary['mac_development_reference']['interpretation']}`; these are not deployment or hardware acceptance criteria.",
            "",
            "## Static footprint and memory observation",
            "",
            f"- Parameter count: `{footprint['parameter_count']}`",
            "- Strict-INT8 file bytes: "
            + ", ".join(
                f"seed{seed}={footprint['strict_int8_artifacts'][str(seed)]['bytes']}"
                for seed in FROZEN_SEEDS
            ),
            f"- Memory method: `{evidence['memory_observation.json']['measurement_type']}` / `{evidence['memory_observation.json']['method']}`",
            "- Process RSS is an observational proxy, not a TFLite arena or model-RAM claim.",
            f"- Peak memory status: `{evidence['memory_observation.json']['peak_during_benchmark']['measurement_type']}`",
            "",
            "## Confirmation stability",
            "",
        ]
    )
    for seed in FROZEN_SEEDS:
        item = confirmation["per_seed"][str(seed)]
        lines.append(
            f"- seed{seed}: median difference `{item['median_difference_ratio']:.6f}`; "
            f"P95 ratio `{item['p95_ratio_confirmation_to_primary']:.6f}`; "
            f"warning `{item['environment_instability_warning']}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "The model consumes a 30-second observation window, but that window-acquisition duration is not CPU model-inference time.",
            "These values describe steady-state offline compute on this specific Mac only. `<5 ms` and `P99 <15 ms` are `MAC_DEVELOPMENT_REFERENCE_ONLY`, not Raspberry Pi, real-sensor, sensor-to-alarm, MR60, or production-real-time claims.",
            "M-B8 benchmarks this specific Mac environment only; it performs no model training, model conversion, seed selection, or LOCKED_TEST access.",
            "",
        ]
    )
    return "\n".join(lines)
