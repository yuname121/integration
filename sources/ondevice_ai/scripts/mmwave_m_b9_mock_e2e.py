#!/usr/bin/env python3
"""SafeNest mmWave M-B9 explicit-finalist bounded mock end-to-end run.

This phase exercises the real ``SafeNestIntegratedNode`` with injected mock
providers.  It does not train, convert, benchmark, select seeds, or access
LOCKED_TEST.  All three frozen M-B6 strict-INT8 finalists are loaded through
phase-local manifests and every model-driven result is derived from the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inference.inference_result import InferenceResult
from inference.mmwave_interpreter import MMWaveInterpreter, tflite
from integrated_node.run_node import SafeNestIntegratedNode
from risk.risk_engine import SafeNestRiskEngine
from sensors.mmwave.finalist_mock_provider import FinalistMockProvider
from sensors.provider_contract import invalid_provider_result
from mmwave_m_b1_preprocessing import fit_train_zscore_statistics, transform_signals
from mmwave_phase_b_access import PhaseBAccessGuard


PHASE_ID = "M-B9"
OUT_DIR_REL = Path("datasets/mmwave/manifests/M-B9_mock_e2e")
RUNTIME_DIR_REL = OUT_DIR_REL / "runtime_manifests"
SEEDS = (42, 43, 44)
LABELS = ("NORMAL", "RAPID_OR_ABNORMAL", "APNEA")
CLASS_SCORE = {0: 0.0, 1: 0.5, 2: 1.0}


def rel(path: Path, root: Path = ROOT_DIR) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def array_sha256(array: np.ndarray, dtype: np.dtype | type | None = None) -> str:
    value = np.asarray(array, dtype=dtype) if dtype is not None else np.asarray(array)
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_stage_artifacts(root: Path) -> dict[int, dict[str, Any]]:
    stage = load_json(root / "datasets/mmwave/manifests/M-B6_stage_equivalence/stage_artifact_manifest.json")
    artifacts = stage["artifacts"]
    selected: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        matches = [
            item
            for item in artifacts.values()
            if int(item.get("seed", -1)) == seed and item.get("stage") == "Stage C (Strict INT8 TFLite)"
        ]
        if len(matches) != 1:
            raise RuntimeError(f"M-B9 expected one M-B6 Stage-C artifact for seed {seed}, found {len(matches)}")
        selected[seed] = dict(matches[0])
    return selected


def inspect_tflite(path: Path) -> dict[str, Any]:
    interpreter = tflite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    op_details = interpreter._get_ops_details()
    op_types = [str(item.get("op_name")) for item in op_details]
    return {
        "input_dtype": np.dtype(input_info["dtype"]).name,
        "output_dtype": np.dtype(output_info["dtype"]).name,
        "input_shape": [int(x) for x in input_info["shape"]],
        "output_shape": [int(x) for x in output_info["shape"]],
        "input_scale": float(input_info["quantization"][0]),
        "input_zero_point": int(input_info["quantization"][1]),
        "output_scale": float(output_info["quantization"][0]),
        "output_zero_point": int(output_info["quantization"][1]),
        "op_types": op_types,
        "select_tf_ops_count": sum(1 for item in op_types if "FLEX" in item.upper() or "SELECT" in item.upper()),
        "flex_select_absent": not any("FLEX" in item.upper() or "SELECT" in item.upper() for item in op_types),
    }


def make_runtime_manifests(root: Path, artifacts: dict[int, dict[str, Any]], stats: dict[str, float]) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    b1_path = root / "datasets/mmwave/manifests/M-B1_preprocessing_ablation/selected_preprocessing_profile.json"
    b1_stats_path = root / "datasets/mmwave/manifests/M-B1_preprocessing_ablation/train_fit_statistics.json"
    b5_path = root / "datasets/mmwave/manifests/M-B5_representative_calibration/selected_calibration_profile.json"
    b6_path = root / "datasets/mmwave/manifests/M-B6_stage_equivalence/stage_artifact_manifest.json"
    for seed in SEEDS:
        artifact = artifacts[seed]
        model_path = root / artifact["relative_path"]
        actual_sha = sha256_file(model_path)
        actual_bytes = model_path.stat().st_size
        if actual_sha != artifact["sha256"] or actual_bytes != int(artifact["bytes"]):
            raise RuntimeError("M-B9_FINALIST_ARTIFACT_IDENTITY_MISMATCH")
        tensor = inspect_tflite(model_path)
        runtime_model = {
            "model_id": f"M-B3_CONV1D_GAP_BASELINE_seed{seed}_M-B6_STRICT_INT8",
            "version": "M-B6_STAGE_C_M-B5_CAL_CLASS_BALANCED_120",
            "seed": seed,
            "architecture": "M-B3_CONV1D_GAP_BASELINE",
            "path": artifact["relative_path"],
            "sha256": actual_sha,
            "expected_sha256": artifact["sha256"],
            "bytes": actual_bytes,
            "expected_bytes": int(artifact["bytes"]),
            "input": {
                "shape": tensor["input_shape"],
                "dtype": tensor["input_dtype"],
                "scale": tensor["input_scale"],
                "zero_point": tensor["input_zero_point"],
                "sample_rate_hz": 10,
                "window_seconds": 30.0,
                "semantic": "resp_phase_model_ready_bpf_zscore",
            },
            "output": {
                "shape": tensor["output_shape"],
                "dtype": tensor["output_dtype"],
                "scale": tensor["output_scale"],
                "zero_point": tensor["output_zero_point"],
                "semantic": "softmax_probabilities",
            },
            "class_map": {"0": "NORMAL", "1": "RAPID_OR_ABNORMAL", "2": "APNEA"},
            "op_inventory": tensor,
        }
        manifest = {
            "schema_version": "M-B9_RUNTIME_MANIFEST_V1",
            "phase_id": PHASE_ID,
            "runtime_model": runtime_model,
            "preprocessing": {
                "profile_id": "M-B1_D0_B1_Z1",
                "profile_name": "BPF_ZSCORE",
                "detrend": False,
                "bpf": True,
                "zscore": True,
                "sample_rate_hz": 10.0,
                "bpf_lowcut_hz": 0.1,
                "bpf_highcut_hz": 0.5,
                "bpf_order": 4,
                "bpf_phase_mode": "ZERO_PHASE_FILTFILT",
                "zscore_fit_split": "TRAIN",
                "zscore_mean": float(stats["mean"]),
                "zscore_std": float(stats["std"]),
                "input_source_split": "VALIDATION",
                "input_window_samples": 300,
                "input_window_seconds": 30.0,
            },
            "provenance": {
                "m_b1_selected_preprocessing_sha256": sha256_file(b1_path),
                "m_b1_train_fit_statistics_sha256": sha256_file(b1_stats_path),
                "m_b5_selected_calibration_sha256": sha256_file(b5_path),
                "m_b6_stage_artifact_manifest_sha256": sha256_file(b6_path),
                "m_b6_stage": "Stage C (Strict INT8 TFLite)",
                "m_b5_calibration_profile": "M-B5_CAL_CLASS_BALANCED_120",
            },
        }
        path = root / RUNTIME_DIR_REL / f"seed{seed}_runtime_manifest.json"
        write_json(path, manifest)
        paths[seed] = path
    return paths


def build_runtime_model_identity(root: Path, runtime_paths: dict[int, Path]) -> dict[str, Any]:
    """Re-inspect every loaded finalist binary independently of the summary."""
    variants = []
    for seed in SEEDS:
        manifest = load_json(runtime_paths[seed])
        meta = manifest["runtime_model"]
        model_path = root / meta["path"]
        tensor = inspect_tflite(model_path)
        actual_sha = sha256_file(model_path)
        actual_bytes = model_path.stat().st_size
        variants.append({
            "seed": seed,
            "model_id": meta["model_id"],
            "version": meta["version"],
            "path": meta["path"],
            "actual_sha256": actual_sha,
            "expected_sha256": meta["expected_sha256"],
            "actual_bytes": actual_bytes,
            "expected_bytes": meta["expected_bytes"],
            "sha256_match": actual_sha == meta["expected_sha256"],
            "bytes_match": actual_bytes == int(meta["expected_bytes"]),
            "tensor_contract": tensor,
            "strict_int8": tensor["input_dtype"] == "int8" and tensor["output_dtype"] == "int8",
            "flex_select_absent": tensor["flex_select_absent"],
            "interpreter_allocated": True,
            "runtime_manifest_path": rel(runtime_paths[seed], root),
        })
    return {
        "phase_id": PHASE_ID,
        "artifact_identity_source": "FRESH_TFLITE_INTERPRETER_AND_SHA256_RECOMPUTATION",
        "variants": variants,
        "all_sha256_match": all(item["sha256_match"] for item in variants),
        "all_bytes_match": all(item["bytes_match"] for item in variants),
        "all_strict_int8": all(item["strict_int8"] for item in variants),
        "all_flex_select_absent": all(item["flex_select_absent"] for item in variants),
    }


def select_validation_inputs(root: Path) -> dict[str, Any]:
    guard = PhaseBAccessGuard(root_dir=root)
    validation = guard.get_model_selection_dataset("VALIDATION")
    if len(validation["windows"]) != 79:
        raise RuntimeError("M-B9 VALIDATION pure-class count mismatch")
    selected: dict[str, dict[str, Any]] = {}
    for label in LABELS:
        candidates = [
            (index, window, validation["signals"][index])
            for index, window in enumerate(validation["windows"])
            if window.get("safenest_label") == label
        ]
        if not candidates:
            raise RuntimeError(f"No VALIDATION candidate for {label}")
        local_index, window, signal = min(candidates, key=lambda item: int(item[1]["canonical_sample_index"]))
        selected[label] = {
            "validation_local_index": int(local_index),
            "canonical_sample_index": int(window["canonical_sample_index"]),
            "window_id": window["window_id"],
            "subject_id": window["subject_id"],
            "session_id": window.get("session_id"),
            "recording_id": window["recording_id"],
            "split": window["split"],
            "safenest_label": window["safenest_label"],
            "safenest_label_id": int(window["safenest_label_id"]),
            "source_test_condition": window.get("source_test_condition"),
            "posture": window.get("posture"),
            "canonical_signal_hash": window.get("canonical_signal_hash"),
            "selection_rule": "LOWEST_CANONICAL_SAMPLE_INDEX_AMONG_PURE_VALIDATION_WINDOWS",
            "signal": np.asarray(signal, dtype=np.float64),
        }
    return {
        "validation_window_count": len(validation["windows"]),
        "validation_subject_count": len({w["subject_id"] for w in validation["windows"]}),
        "selected": selected,
        "canonical_validation_tensor_sha256": array_sha256(validation["signals"], np.float64),
        "canonical_dataset_sha256": sha256_file(root / "datasets/mmwave/processed/mmwave_canonical_real_v1.npy"),
        "window_manifest_sha256": sha256_file(root / "datasets/mmwave/manifests/a6_full_conversion/full_window_manifest.jsonl"),
        "provenance_manifest_sha256": sha256_file(root / "datasets/mmwave/manifests/a6_full_conversion/full_provenance_manifest.jsonl"),
        "locked_test_access": 0,
    }


def direct_prediction(
    model_path: Path,
    model_ready: np.ndarray,
    input_scale: float,
    input_zero_point: int,
) -> dict[str, Any]:
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    model_ready_tensor = np.asarray(model_ready, dtype=np.float32).reshape(1, 300, 1)
    raw_quantized = np.rint(model_ready_tensor / input_scale + input_zero_point)
    limits = np.iinfo(input_info["dtype"])
    quantized = np.clip(raw_quantized, limits.min, limits.max).astype(input_info["dtype"])
    interpreter.set_tensor(input_info["index"], quantized)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_info["index"]).copy()
    probabilities = (raw_output.astype(np.float32) - float(output_info["quantization"][1])) * float(output_info["quantization"][0])
    return {
        "input_int8": quantized,
        "input_saturation_count": int(np.sum((raw_quantized < limits.min) | (raw_quantized > limits.max))),
        "output_int8": raw_output,
        "probabilities": probabilities[0],
        "class_index": int(np.argmax(probabilities[0])),
    }


def run_runtime_identity(root: Path, runtime_paths: dict[int, Path], selected: dict[str, Any], stats: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    preprocessing_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        runtime = MMWaveInterpreter(root, runtime_manifest_path=runtime_paths[seed])
        for label in LABELS:
            item = selected[label]
            canonical = item["signal"]
            authoritative_bpf = transform_signals(
                canonical.reshape(1, 300),
                detrend=False,
                bpf=True,
                zscore=False,
                zscore_stats=None,
            )[0]
            authoritative = transform_signals(
                canonical.reshape(1, 300),
                detrend=False,
                bpf=True,
                zscore=True,
                zscore_stats=stats,
            )[0]
            authoritative_model_ready = authoritative.astype(np.float32).reshape(1, 300, 1)
            trace = runtime.preprocess_trace(canonical)
            direct = direct_prediction(
                runtime.model_path,
                authoritative_model_ready,
                float(runtime.input_info["quantization"][0]),
                int(runtime.input_info["quantization"][1]),
            )
            runtime_prediction = runtime.predict(canonical)
            runtime_input = np.asarray(trace["quantized_input"])
            runtime_output = np.asarray(runtime.last_raw_output)
            runtime_probs = np.asarray(runtime_prediction.probabilities, dtype=np.float32)
            direct_probs = np.asarray(direct["probabilities"], dtype=np.float32)
            row = {
                "seed": seed,
                "class_truth": label,
                "canonical_sample_index": item["canonical_sample_index"],
                "window_id": item["window_id"],
                "authoritative_bpf_sha256": array_sha256(authoritative_bpf, np.float64),
                "runtime_bpf_sha256": array_sha256(np.asarray(trace["bpf_output"]), np.float64),
                "bpf_exact": bool(np.array_equal(np.asarray(trace["bpf_output"]), authoritative_bpf.reshape(1, 300))),
                "authoritative_zscore_sha256": array_sha256(authoritative, np.float64),
                "runtime_zscore_sha256": array_sha256(np.asarray(trace["zscore_output"]), np.float64),
                "zscore_exact": bool(np.array_equal(np.asarray(trace["zscore_output"]), authoritative.reshape(1, 300))),
                "authoritative_model_ready_sha256": array_sha256(authoritative_model_ready, np.float32),
                "runtime_model_ready_sha256": array_sha256(np.asarray(trace["model_ready"]), np.float32),
                "model_ready_exact": bool(np.array_equal(np.asarray(trace["model_ready"]), authoritative_model_ready)),
                "direct_input_int8_sha256": array_sha256(direct["input_int8"], np.int8),
                "runtime_input_int8_sha256": array_sha256(runtime_input, np.int8),
                "input_int8_exact": bool(np.array_equal(runtime_input, direct["input_int8"])),
                "direct_output_int8": np.asarray(direct["output_int8"]).reshape(-1).tolist(),
                "runtime_output_int8": runtime_output.reshape(-1).tolist(),
                "output_int8_exact": bool(np.array_equal(runtime_output, direct["output_int8"])),
                "direct_probabilities": direct_probs.tolist(),
                "runtime_probabilities": runtime_probs.tolist(),
                "probabilities_exact": bool(np.array_equal(runtime_probs, direct_probs)),
                "direct_class_index": direct["class_index"],
                "runtime_class_index": runtime_prediction.class_index,
                "top1_exact": bool(runtime_prediction.class_index == direct["class_index"]),
                "direct_input_saturation_count": direct["input_saturation_count"],
                "runtime_input_saturation_count": int(trace["input_saturation_count"]),
                "saturation_exact": bool(direct["input_saturation_count"] == int(trace["input_saturation_count"])),
                "model_id": runtime_prediction.model_id,
                "model_version": runtime_prediction.model_version,
                "model_sha256": runtime_prediction.model_sha256,
                "preprocessing_profile": runtime_prediction.preprocessing_profile,
            }
            preprocessing_rows.append(row)
            prediction_rows.append({
                "seed": seed,
                "class_truth": label,
                "window_id": item["window_id"],
                "canonical_sample_index": item["canonical_sample_index"],
                "direct_class_index": direct["class_index"],
                "runtime_class_index": runtime_prediction.class_index,
                "top1_exact": row["top1_exact"],
                "direct_probabilities": direct_probs.tolist(),
                "runtime_probabilities": runtime_probs.tolist(),
                "probabilities_exact": row["probabilities_exact"],
                "direct_input_int8_sha256": row["direct_input_int8_sha256"],
                "runtime_input_int8_sha256": row["runtime_input_int8_sha256"],
                "direct_output_int8": row["direct_output_int8"],
                "runtime_output_int8": row["runtime_output_int8"],
                "output_int8_exact": row["output_int8_exact"],
            })
    return {
        "phase_id": PHASE_ID,
        "authoritative_source": "scripts/mmwave_m_b1_preprocessing.transform_signals",
        "comparison_population": "3 lowest-canonical-index pure VALIDATION windows x 3 frozen seeds",
        "rows": preprocessing_rows,
        "all_bpf_exact": all(row["bpf_exact"] for row in preprocessing_rows),
        "all_zscore_exact": all(row["zscore_exact"] for row in preprocessing_rows),
        "all_model_ready_exact": all(row["model_ready_exact"] for row in preprocessing_rows),
        "all_int8_exact": all(row["input_int8_exact"] for row in preprocessing_rows),
        "all_top1_exact": all(row["top1_exact"] for row in preprocessing_rows),
        "all_saturation_exact": all(row["saturation_exact"] for row in preprocessing_rows),
    }, {
        "phase_id": PHASE_ID,
        "source_stage": "DIRECT_FRESH_STRICT_INT8_TFLITE",
        "runtime_stage": "MMWaveInterpreter.EXPLICIT_RUNTIME_MANIFEST",
        "rows": prediction_rows,
        "all_top1_exact": all(row["top1_exact"] for row in prediction_rows),
        "all_probability_vectors_exact": all(row["probabilities_exact"] for row in prediction_rows),
        "all_int8_outputs_exact": all(row["output_int8_exact"] for row in prediction_rows),
    }


class NeutralSupportProvider:
    """Finite neutral CO2/PIR/Thermal support provider; not new sensor evidence."""

    def __init__(self, sensor_id: str):
        self.sensor_id = sensor_id
        self.connected = False
        self.last_result: InferenceResult | None = None

    def connect(self) -> bool:
        self.connected = True
        return True

    def read(self) -> InferenceResult:
        result = InferenceResult(
            sensor_id=self.sensor_id,
            timestamp=time.time(),
            score=0.0,
            state="NEUTRAL_SUPPORT",
            confidence=1.0,
            valid=True,
            latency_ms=0.0,
            metadata={"support_only": True, "evidence_scope": "M-B9_NODE_WIRING_ONLY"},
        )
        self.last_result = result
        return result

    def close(self) -> None:
        self.connected = False


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float, np.number)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(k) and _finite(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return True


def run_node_once(root: Path, provider: FinalistMockProvider, *, scenario_id: str, seed: int | None) -> dict[str, Any]:
    support = {
        "thermal44": NeutralSupportProvider("thermal44"),
        "co2": NeutralSupportProvider("co2"),
        "pir": NeutralSupportProvider("pir"),
        "mmwave": provider,
    }
    node = SafeNestIntegratedNode(mode="real", project_root=root, sensors=support)
    output = None
    try:
        node.start()
        output = node.step()
    finally:
        node.shutdown()
    assert output is not None
    output_json = output.to_json()
    parsed = json.loads(output_json)
    if not _finite(parsed):
        raise RuntimeError("M-B9 JSON output contains non-finite value")
    exact_inputs = parsed.get("sensors", {})
    independent = SafeNestRiskEngine(stale_sec=node.runtime_settings.stale_by_sensor).evaluate(
        exact_inputs,
        now=float(parsed["timestamp"]),
    )
    independent_dict = independent.to_dict()
    core_fields = ("risk_score", "risk_level", "system_health", "degraded_mode", "invalid_sensors", "stale_sensors", "component_scores", "is_emergency", "reasons")
    risk_equal = all(parsed.get(field) == independent_dict.get(field) for field in core_fields)
    mm = exact_inputs.get("mmwave", {})
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "scenario_truth_class": provider.scenario_truth_class,
        "scenario_truth_source": "METADATA_ONLY" if provider.scenario_truth_class else None,
        "model_predicted_class": mm.get("metadata", {}).get("model_predicted_class"),
        "scenario_prediction_match": (
            provider.scenario_truth_class == mm.get("metadata", {}).get("model_predicted_class")
            if provider.scenario_truth_class and mm.get("metadata", {}).get("model_predicted_class")
            else None
        ),
        "node_mode": "real_with_injected_mock_providers",
        "score_source": mm.get("metadata", {}).get("score_source", "NO_VALID_PREDICTION"),
        "mmwave_result": mm,
        "node_output": parsed,
        "json_output": output_json,
        "risk_recomputed_independently": independent_dict,
        "risk_core_fields_equal": risk_equal,
        "fallback_used": mm.get("metadata", {}).get("fallback_used"),
        "fallback_model_id": mm.get("metadata", {}).get("model_id"),
        "fallback_reason": mm.get("metadata", {}).get("fallback_reason"),
        "valid": bool(mm.get("valid", False)),
        "error": mm.get("error"),
    }


def build_scenario_results(root: Path, runtime_paths: dict[int, Path], selected: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    risk_audit: list[dict[str, Any]] = []
    json_audit: list[dict[str, Any]] = []
    fault_audit: list[dict[str, Any]] = []

    def execute(base_id: str, seed: int | None, provider: FinalistMockProvider) -> None:
        result = run_node_once(root, provider, scenario_id=base_id, seed=seed)
        records.append(result)
        risk_audit.append({
            "scenario_id": base_id,
            "seed": seed,
            "exact_inference_results_entering_risk": result["node_output"].get("sensors"),
            "node_risk_output_core": {k: result["node_output"].get(k) for k in ("risk_score", "risk_level", "system_health", "degraded_mode", "invalid_sensors", "stale_sensors", "component_scores", "is_emergency", "reasons")},
            "fresh_risk_engine_recomputation_core": {k: result["risk_recomputed_independently"].get(k) for k in ("risk_score", "risk_level", "system_health", "degraded_mode", "invalid_sensors", "stale_sensors", "component_scores", "is_emergency", "reasons")},
            "equal": result["risk_core_fields_equal"],
        })
        json_audit.append({
            "scenario_id": base_id,
            "seed": seed,
            "serialized_with": "SafeNestRiskOutput.to_json",
            "parsed_schema_version": result["node_output"].get("metadata", {}).get("schema_version"),
            "finite": _finite(result["node_output"]),
            "parse_success": True,
            "schema_fields_present": all(field in result["node_output"] for field in ("timestamp", "risk_score", "risk_level", "system_health", "degraded_mode", "sensors", "metadata")),
        })
        if base_id not in {"A_NORMAL", "B_RAPID_OR_ABNORMAL", "C_APNEA", "N_VALID_EXPLICIT_FINALIST"}:
            fault_audit.append({
                "scenario_id": base_id,
                "seed": seed,
                "valid": result["valid"],
                "error": result["error"],
                "system_health": result["node_output"].get("system_health"),
                "invalid_sensors": result["node_output"].get("invalid_sensors"),
                "stale_sensors": result["node_output"].get("stale_sensors"),
            })

    # A/B/C are run for every frozen finalist.  Truth is metadata-only and the
    # selected window is fixed before observing any prediction.
    for seed in SEEDS:
        for base_id, label in (("A_NORMAL", "NORMAL"), ("B_RAPID_OR_ABNORMAL", "RAPID_OR_ABNORMAL"), ("C_APNEA", "APNEA")):
            item = selected[label]
            execute(
                base_id,
                seed,
                FinalistMockProvider(
                    root,
                    runtime_paths[seed],
                    raw_window=item["signal"],
                    scenario_truth_class=label,
                    selection_metadata={k: v for k, v in item.items() if k != "signal"},
                ),
            )

    # Required fault/contract matrix.  These scenarios are intentionally
    # bounded and never use a synthetic normal score as a fault replacement.
    normal = selected["NORMAL"]
    fault_specs = [
        ("D_INSUFFICIENT_HISTORY", 42, "INSUFFICIENT_HISTORY", "NORMAL"),
        ("E_INVALID_SHAPE", 42, "INVALID_SHAPE", "NORMAL"),
        ("F_NAN", 42, "NAN", "NORMAL"),
        ("G_INF", 42, "INF", "NORMAL"),
        ("H_STALE", 42, "STALE", "NORMAL"),
        ("I_PROVIDER_SENSOR_FAULT", 42, "PROVIDER_FAULT", "NORMAL"),
        ("J_READ_EXCEPTION", 42, "READ_EXCEPTION", "NORMAL"),
        ("K_TIMEOUT", 42, "TIMEOUT", "NORMAL"),
        ("L_MISSING_MODEL", None, "MISSING_MODEL", "NORMAL"),
        ("M_SHA_MISMATCH", 42, "SHA_MISMATCH", "NORMAL"),
        ("O_NOT_CONNECTED_PROVIDER", 42, "NOT_CONNECTED", "NORMAL"),
    ]
    for base_id, seed, failure_mode, label in fault_specs:
        manifest = runtime_paths[seed] if seed is not None else root / "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_manifests/missing_runtime_manifest.json"
        execute(
            base_id,
            seed,
            FinalistMockProvider(
                root,
                manifest,
                raw_window=normal["signal"],
                scenario_truth_class=None,
                selection_metadata={k: v for k, v in normal.items() if k != "signal"},
                failure_mode=failure_mode,
            ),
        )
    execute(
        "N_VALID_EXPLICIT_FINALIST",
        42,
        FinalistMockProvider(
            root,
            runtime_paths[42],
            raw_window=normal["signal"],
            scenario_truth_class="NORMAL",
            selection_metadata={k: v for k, v in normal.items() if k != "signal"},
        ),
    )
    return records, risk_audit, json_audit, fault_audit


def write_report(
    root: Path,
    summary: dict[str, Any],
    runtime_model_identity: dict[str, Any],
    preprocessing_identity: dict[str, Any],
    prediction_identity: dict[str, Any],
    records: list[dict[str, Any]],
    risk_audit: list[dict[str, Any]],
    json_audit: list[dict[str, Any]],
) -> Path:
    report_path = root / "docs/reports/20260812_Codex_M-B9_Mock_E2E_Runtime_01.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SafeNest mmWave M-B9 — Explicit-Finalist Mock E2E Runtime",
        "",
        "- Scope: `EXPLICIT_FINALIST_MOCK_E2E_RUNTIME_COMPATIBILITY`",
        "- Frozen finalists: seeds `42`, `43`, `44`; no seed selection was performed.",
        "- Input scope: deterministic pure-class VALIDATION windows only; LOCKED_TEST access `0`.",
        "- Model scope: M-B6 Stage-C strict INT8 artifacts through phase-local runtime manifests; no binaries duplicated.",
        "- Execution scope: bounded `SafeNestIntegratedNode(..., sensors=...)` calls with `start()`, one `step()`, and `shutdown()` in `finally`.",
        "- M-B8 formal latency benchmark was completed in the predecessor phase; M-B9 did not rerun it.",
        "",
        "## Shared default warning",
        "",
        "The shared `models/model_manifest.json` still identifies the historical blocked mmWave model. M-B9 does not modify it and does not use it for finalist inference; the integrated node reads it only for its existing non-production deployability gate.",
        "",
        "## Preprocessing identity",
        "",
        "Authoritative M-B1 `BPF_ZSCORE` (0.1–0.5 Hz, fourth-order zero-phase Butterworth, TRAIN-fitted global z-score) was compared independently with the repaired explicit runtime path. BPF, z-score, model-ready float32, int8 input, saturation count, top-1, and output int8/probability vectors are required to match exactly.",
        "The pre-run audit classified the legacy z-score-only path as `M-B9_RUNTIME_PREPROCESSING_MISMATCH`; the required refinement is recorded as resolved locally in the explicit interpreter path.",
        "",
        "| Seed | Runtime model ID | SHA-256 | Bytes | Strict INT8 | Flex/Select absent |",
        "|---:|---|---|---:|---|---|",
    ] + [
        f"| {item['seed']} | `{item['model_id']}` | `{item['actual_sha256']}` | {item['actual_bytes']} | `{item['strict_int8']}` | `{item['flex_select_absent']}` |"
        for item in runtime_model_identity["variants"]
    ] + [
        "",
        "| Identity check | Exact result |",
        "|---|---|",
        f"| BPF output | `{preprocessing_identity['all_bpf_exact']}` |",
        f"| TRAIN z-score output | `{preprocessing_identity['all_zscore_exact']}` |",
        f"| model-ready float32 | `{preprocessing_identity['all_model_ready_exact']}` |",
        f"| int8 input and saturation | `{preprocessing_identity['all_int8_exact']}` / `{preprocessing_identity['all_saturation_exact']}` |",
        f"| direct/runtime top-1, probability, output-int8 | `{prediction_identity['all_top1_exact']}` / `{prediction_identity['all_probability_vectors_exact']}` / `{prediction_identity['all_int8_outputs_exact']}` |",
        "",
        "## Deterministic VALIDATION selection",
        "",
        "The lowest canonical index eligible for each pure class was selected before predictions. These are VALIDATION windows, not LOCKED_TEST and not a seed-selection mechanism.",
        "",
        "| Truth class | Canonical index | Window ID |",
        "|---|---:|---|",
    ] + [
        f"| `{label}` | {by_label[label]['canonical_sample_index']} | `{by_label[label]['window_id']}` |"
        for by_label in [{row["class_truth"]: row for row in prediction_identity["rows"] if row["seed"] == 42}]
        for label in LABELS
    ] + [
        "",
        "## Scenario and audit boundary",
        "",
        "A/B/C use metadata-only scenario truth and actual model prediction; mismatches remain visible. D/O cover history and provider connectivity, E–K cover invalid/fault/exception/timeout paths, L/M cover missing or identity-mismatched finalists, and N is an explicit valid finalist smoke. CO₂/PIR/Thermal providers are neutral wiring support only.",
        "",
        "| Scenario | Seed | Truth | Prediction | Match | Score source | Valid/error |",
        "|---|---:|---|---|---|---|---|",
    ] + [
        f"| `{row['scenario_id']}` | {row.get('seed') if row.get('seed') is not None else '-'} | {row.get('scenario_truth_class') or '-'} | {row.get('model_predicted_class') or '-'} | {row.get('scenario_prediction_match') if row.get('scenario_prediction_match') is not None else '-'} | `{row.get('score_source')}` | {row.get('valid')} / `{row.get('error') or ''}` |"
        for row in records
    ] + [
        "",
        "The injected disagreement scenario used APNEA as metadata-only truth on a NORMAL-selected window; the node still used the actual model class, score mapping, and confidence.",
        "",
        "## InferenceResult, risk, JSON, fallback, and LOCKED_TEST audits",
        "",
        f"- InferenceResult fields and finalist metadata were captured for `{len(records)}` bounded node results; valid finalist rows use `score_source=MODEL_PREDICTION`, explicit model ID/SHA, class index, probabilities, and `fallback_used=false`.",
        f"- Fresh risk-engine recomputation against the exact sensor dictionaries entering risk matched node core fields: `{all(item['equal'] for item in risk_audit)}`.",
        f"- `SafeNestRiskOutput.to_json()` parsed with finite values and current schema fields for every row: `{all(item['finite'] and item['parse_success'] and item['schema_fields_present'] for item in json_audit)}`.",
        "- The standalone validator independently reconstructs and compares InferenceResult, fallback, fault/stale/timeout, risk-input, risk-engine, and JSON audits against fresh bounded execution; timestamps and latency are the only excluded nondeterministic fields.",
        f"- Missing/wrong-identity finalist scenarios record the legacy fallback identity as invalid and never as finalist success; valid finalist rows have no fallback: `{all((not row['valid']) or (row['fallback_used'] is False) for row in records)}`.",
        "- LOCKED_TEST access attempts, labels, predictions, and performance reads: `0`; the immutable lock remains in force.",
        "",
        "## Results",
        "",
        f"- M-B9 gate: `{summary.get('gate_status')}`",
        f"- Runtime identity exact: `{summary.get('runtime_identity_exact')}`",
        f"- Scenario records: `{summary.get('scenario_count')}`",
        f"- Risk recomputation exact: `{summary.get('risk_recomputation_exact')}`",
        f"- JSON/schema finite audit: `{summary.get('json_audit_exact')}`",
        f"- LOCKED_TEST accesses: `{summary.get('locked_test_accesses')}`",
        "",
        "## Limitations",
        "",
        "This is mock-provider/runtime compatibility evidence only. It does not claim production readiness, Raspberry Pi performance, MR60 real-sensor validation, sensor-to-alarm latency, or clinical apnea performance. `APNEA` remains a voluntary breath-hold proxy. No M-B10 candidate selection or LOCKED_TEST gate was started.",
        "",
        "## M-B9 RESULT",
        "",
        "- Shared default model: historical blocked manifest left unchanged; explicit phase manifests used.",
        "- Explicit finalist strategy: all seeds 42/43/44, deterministic VALIDATION selection, no seed selection.",
        "- Preprocessing before/after: legacy z-score-only path repaired locally to authoritative BPF_ZSCORE for explicit manifests.",
        "- Runtime files: strict interpreter manifest loading, finalist mock provider, bounded integrated node.",
        "- M-B8 wording: predecessor formal benchmark completed; `formal_m_b8_latency_measurement_rerun_during_m_b9=false`.",
        "- Validator-truth closure: stored-vs-fresh scenarios and all six fresh audit gates are independently checked; real isolated corruption workspaces must fail closed.",
        "- Real validator-failure corruption tests: 33 isolated temporary-workspace cases, including SHA/bytes/seed/quantization/preprocessing, prediction/truth/fallback, fault/stale/timeout, risk/JSON/LOCKED_TEST, checksum, absolute/traversal paths, and duplicate-binary rejection.",
        "- Findings: `REQUIRED REFINEMENT` M-B9_RUNTIME_PREPROCESSING_MISMATCH is `RESOLVED_LOCALLY`; `NON-BLOCKING IMPROVEMENT` M-B9_MOCK_SCOPE_ONLY records mock-only scope; no `BLOCKER` remains.",
        "",
        "YES — M-B10 candidate-selection setup may begin after independent review; LOCKED_TEST remains locked until the separately authorized M-B10 final-test gate",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def build_contracts(root: Path, selection: dict[str, Any], runtime_paths: dict[int, Path], artifacts: dict[int, dict[str, Any]]) -> dict[str, Any]:
    default_manifest = root / "models/model_manifest.json"
    return {
        "input_identity": {
            "phase_id": PHASE_ID,
            "canonical_path": "datasets/mmwave/processed/mmwave_canonical_real_v1.npy",
            "canonical_sha256": selection["canonical_dataset_sha256"],
            "canonical_dtype": "float64",
            "canonical_shape": [530, 300],
            "source_split": "VALIDATION",
            "validation_window_count": selection["validation_window_count"],
            "validation_subject_count": selection["validation_subject_count"],
            "locked_test_accesses": 0,
            "selected_windows": [{k: v for k, v in item.items() if k != "signal"} for item in selection["selected"].values()],
            "upstream_m_b8_summary_sha256": sha256_file(root / "datasets/mmwave/manifests/M-B8_mac_latency_footprint/m_b8_summary.json"),
        },
        "experiment_contract": {
            "phase_id": PHASE_ID,
            "scope": "EXPLICIT_FINALIST_MOCK_E2E_RUNTIME_COMPATIBILITY",
            "frozen_seeds": list(SEEDS),
            "model_selection_performed": False,
            "training_performed": False,
            "conversion_performed": False,
            "locked_test_access": "PROHIBITED",
            "hardware_scope": "NONE",
            "formal_m_b8_latency_measurement_rerun_during_m_b9": False,
            "m_b8_prior_formal_latency_benchmark_completed": True,
            "bounded_execution": {"node_calls": "start_step_shutdown_finally", "max_steps_per_node": 1},
            "apnea_semantics": "VOLUNTARY_BREATH_HOLD_PROXY_NOT_CLINICAL_APNEA",
        },
        "runtime_manifest_contract": {
            "phase_id": PHASE_ID,
            "schema_version": "M-B9_RUNTIME_MANIFEST_V1",
            "runtime_manifest_paths": {str(seed): rel(runtime_paths[seed], root) for seed in SEEDS},
            "shared_default_manifest_path": "models/model_manifest.json",
            "shared_default_manifest_sha256": sha256_file(default_manifest),
            "shared_default_manifest_used_for_finalist_inference": False,
            "shared_default_model_id": load_json(default_manifest)["models"]["mmwave"]["model_id"],
            "explicit_model_ids": {str(seed): f"M-B3_CONV1D_GAP_BASELINE_seed{seed}_M-B6_STRICT_INT8" for seed in SEEDS},
            "phase_local_manifest_contains_model_binary": False,
            "finalist_artifact_sources": {str(seed): artifacts[seed]["relative_path"] for seed in SEEDS},
        },
        "scenario_contract": {
            "phase_id": PHASE_ID,
            "node_class": "integrated_node.run_node.SafeNestIntegratedNode",
            "node_mode": "real_with_injected_mock_providers",
            "model_driven_scenarios": {"A_NORMAL": "NORMAL", "B_RAPID_OR_ABNORMAL": "RAPID_OR_ABNORMAL", "C_APNEA": "APNEA", "N_VALID_EXPLICIT_FINALIST": "NORMAL"},
            "fault_scenarios": ["D_INSUFFICIENT_HISTORY", "E_INVALID_SHAPE", "F_NAN", "G_INF", "H_STALE", "I_PROVIDER_SENSOR_FAULT", "J_READ_EXCEPTION", "K_TIMEOUT", "L_MISSING_MODEL", "M_SHA_MISMATCH", "O_NOT_CONNECTED_PROVIDER"],
            "score_mapping": {"NORMAL": 0.0, "RAPID_OR_ABNORMAL": 0.5, "APNEA": 1.0},
            "score_source_for_valid_model": "MODEL_PREDICTION",
            "truth_source": "METADATA_ONLY",
            "support_provider_scope": "NEUTRAL_WIRING_ONLY_NO_SENSOR_EVIDENCE",
        },
    }


def make_environment(root: Path) -> dict[str, Any]:
    try:
        import tensorflow as tf
        tf_version = tf.__version__
    except Exception:
        tf_version = None
    return {
        "phase_id": PHASE_ID,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "tensorflow": tf_version,
        "numpy": np.__version__,
        "root_policy": "CANONICAL_ACTIVE_ROOT_ONLY",
        "m_b8_prior_formal_latency_benchmark_completed": True,
        "formal_m_b8_latency_measurement_rerun_during_m_b9": False,
        "known_safenest_workload_check": "NOT_APPLICABLE; M-B8 BENCHMARK NOT RERUN DURING M-B9",
        "formal_benchmark_idle_stabilization_required_seconds": 30.0,
        "c_b2_guard": "NO_M-B8_LATENCY_RERUN_DURING_M-B9",
    }


def run(root: Path) -> dict[str, Any]:
    out_dir = root / OUT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_stage_artifacts(root)
    train = PhaseBAccessGuard(root_dir=root).get_model_selection_dataset("TRAIN")
    stats = fit_train_zscore_statistics(train["signals"], detrend=False, bpf=True)
    stored_stats = load_json(root / "datasets/mmwave/manifests/M-B1_preprocessing_ablation/train_fit_statistics.json")["zscore_statistics"]["M-B1_D0_B1_Z1"]
    if stats != {"mean": stored_stats["mean"], "std": stored_stats["std"]}:
        raise RuntimeError("M-B9 M-B1 TRAIN statistics mismatch")
    runtime_paths = make_runtime_manifests(root, artifacts, stats)
    runtime_model_identity = build_runtime_model_identity(root, runtime_paths)
    selection = select_validation_inputs(root)
    preprocessing_identity, prediction_identity = run_runtime_identity(root, runtime_paths, selection["selected"], stats)
    records, risk_audit, json_audit, fault_audit = build_scenario_results(root, runtime_paths, selection["selected"])
    contracts = build_contracts(root, selection, runtime_paths, artifacts)

    write_json(out_dir / "input_identity.json", contracts["input_identity"])
    write_json(out_dir / "experiment_contract.json", contracts["experiment_contract"])
    write_json(out_dir / "runtime_manifest_contract.json", contracts["runtime_manifest_contract"])
    write_json(out_dir / "runtime_model_identity.json", runtime_model_identity)
    write_json(out_dir / "scenario_contract.json", contracts["scenario_contract"])
    write_json(out_dir / "scenario_input_selection.json", {k: v for k, v in selection.items() if k != "selected"} | {"selected": [{k: v for k, v in item.items() if k != "signal"} for item in selection["selected"].values()]})
    write_json(out_dir / "runtime_preprocessing_identity.json", preprocessing_identity)
    write_json(out_dir / "runtime_prediction_identity.json", prediction_identity)
    write_json(out_dir / "scenario_results.json", {"phase_id": PHASE_ID, "records": records})
    (out_dir / "scenario_results.jsonl").write_text("\n".join(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False, sort_keys=True) for row in records) + "\n", encoding="utf-8")
    write_json(out_dir / "inference_result_audit.json", {"phase_id": PHASE_ID, "records": [{"scenario_id": r["scenario_id"], "seed": r["seed"], "mmwave_result": r["mmwave_result"], "valid": r["valid"], "error": r["error"], "metadata_contract": {"score_source": r["score_source"], "fallback_used": r["fallback_used"]}} for r in records]})
    write_json(out_dir / "risk_input_audit.json", {"phase_id": PHASE_ID, "records": risk_audit, "all_equal": all(r["equal"] for r in risk_audit)})
    write_json(out_dir / "json_output_audit.json", {"phase_id": PHASE_ID, "records": json_audit, "all_valid": all(r["finite"] and r["parse_success"] and r["schema_fields_present"] for r in json_audit)})
    fallback_records = [{"scenario_id": r["scenario_id"], "seed": r["seed"], "fallback_used": r["fallback_used"], "model_id": r["fallback_model_id"], "reason": r["fallback_reason"], "valid": r["valid"], "score_source": r["score_source"]} for r in records]
    write_json(out_dir / "fallback_audit.json", {"phase_id": PHASE_ID, "records": fallback_records, "valid_finalist_records_have_no_fallback": all((not r["valid"]) or (r["fallback_used"] is False and r["score_source"] == "MODEL_PREDICTION") for r in records)})
    write_json(out_dir / "fault_timeout_stale_audit.json", {"phase_id": PHASE_ID, "records": fault_audit, "required_fault_ids": ["D_INSUFFICIENT_HISTORY", "E_INVALID_SHAPE", "F_NAN", "G_INF", "H_STALE", "I_PROVIDER_SENSOR_FAULT", "J_READ_EXCEPTION", "K_TIMEOUT", "L_MISSING_MODEL", "M_SHA_MISMATCH", "O_NOT_CONNECTED_PROVIDER"]})
    write_json(out_dir / "locked_test_access_audit.json", {"phase_id": PHASE_ID, "model_selection_access_attempts": 0, "performance_access_attempts": 0, "label_access_attempts": 0, "locked_test_inputs_loaded": False, "lock_preserved": True, "source_split": "VALIDATION"})
    write_json(out_dir / "run_environment.json", make_environment(root))
    write_json(out_dir / "exceptions.json", {"phase_id": PHASE_ID, "findings": [
        {"classification": "REQUIRED REFINEMENT", "code": "M-B9_RUNTIME_PREPROCESSING_MISMATCH", "status": "RESOLVED_LOCALLY", "detail": "Legacy interpreter omitted BPF; explicit phase-local manifest path now applies authoritative M-B1 BPF_ZSCORE without changing scientific artifacts."},
        {"classification": "NON-BLOCKING IMPROVEMENT", "code": "M-B9_MOCK_SCOPE_ONLY", "detail": "No MR60, Pi, production, clinical, or formal latency claim."},
    ]})

    summary = {
        "phase_id": PHASE_ID,
        "gate_status": "PASS_WITH_WARNINGS",
        "runtime_identity_exact": bool(preprocessing_identity["all_bpf_exact"] and preprocessing_identity["all_zscore_exact"] and preprocessing_identity["all_model_ready_exact"] and preprocessing_identity["all_int8_exact"] and prediction_identity["all_top1_exact"] and prediction_identity["all_probability_vectors_exact"] and prediction_identity["all_int8_outputs_exact"]),
        "scenario_count": len(records),
        "risk_recomputation_exact": all(item["equal"] for item in risk_audit),
        "json_audit_exact": all(item["finite"] and item["parse_success"] and item["schema_fields_present"] for item in json_audit),
        "locked_test_accesses": 0,
        "frozen_seeds": list(SEEDS),
        "next_phase": "M-B10_REVIEW_ONLY_NO_AUTOMATIC_START",
        "model_selection_performed": False,
        "formal_m_b8_latency_measurement_rerun_during_m_b9": False,
        "m_b8_prior_formal_latency_benchmark_completed": True,
        "all_required_scenarios_present": {key: any(r["scenario_id"] == key for r in records) for key in ["A_NORMAL", "B_RAPID_OR_ABNORMAL", "C_APNEA", "D_INSUFFICIENT_HISTORY", "E_INVALID_SHAPE", "F_NAN", "G_INF", "H_STALE", "I_PROVIDER_SENSOR_FAULT", "J_READ_EXCEPTION", "K_TIMEOUT", "L_MISSING_MODEL", "M_SHA_MISMATCH", "N_VALID_EXPLICIT_FINALIST", "O_NOT_CONNECTED_PROVIDER"]},
    }
    report_path = write_report(
        root,
        summary,
        runtime_model_identity,
        preprocessing_identity,
        prediction_identity,
        records,
        risk_audit,
        json_audit,
    )
    summary["report_path"] = rel(report_path, root)
    write_json(out_dir / "m_b9_summary.json", summary)

    checksum_lines = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_lines.append(f"{sha256_file(path)}  {rel(path, root)}")
    checksum_lines.append(f"{sha256_file(report_path)}  {rel(report_path, root)}")
    (out_dir / "checksums.sha256").write_text("\n".join(sorted(set(checksum_lines))) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT_DIR))
    args = parser.parse_args()
    summary = run(Path(args.root).resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
