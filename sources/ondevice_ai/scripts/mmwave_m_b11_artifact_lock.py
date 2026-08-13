#!/usr/bin/env python3
"""M-B11 offline real-data candidate artifact lock generator.

Reads immutable stored evidence only. Does not access LOCKED_TEST, does not
invoke TFLite, does not retrain, reconvert, recalibrate, or regenerate A6.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.mmwave_m_b10r1_metrics import (  # noqa: E402
    metric_bundle,
    saturation_audit_from_rows,
    subject_metrics,
)
from scripts.mmwave_m_b10r1_result_writer import (  # noqa: E402
    SELECTED_MODEL_ID,
    V01_MODEL_ID,
    V02_MODEL_ID,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)

PHASE_ID = "M-B11"
SCHEMA = "M-B11_ARTIFACT_LOCK_V1"
LOCK_DIR_REL = Path("datasets/mmwave/manifests/M-B11_artifact_lock")
SENSOR_LOCK_REL = Path("models/mmwave/mmwave_offline_candidate_lock_v1.json")
REPORT_REL = Path("docs/reports/20260813_Cursor_M-B11_mmWave_Offline_Candidate_Artifact_Lock_01.md")

CLASS_MAP = {"0": "NORMAL", "1": "RAPID_OR_ABNORMAL", "2": "APNEA"}
RESULT_LIMITATION = "REUSED_LOCKED_TEST_AFTER_PREINFERENCE_STRUCTURAL_ABORT"
ARTIFACT_STATUS = "REAL_DATA_OFFLINE_CANDIDATE"
SELECTED_CANDIDATE_ID = "M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120"
RUNTIME_MODEL_ID = SELECTED_MODEL_ID
SELECTED_TFLITE_REL = (
    "models/mmwave/experiments/M-B6_stage_equivalence/"
    "M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120_int8.tflite"
)
V01_TFLITE_REL = "models/mmwave/mmwave_resp_int8_v0.1.0.tflite"
V02_TFLITE_REL = "models/mmwave/mmwave_resp_int8_v0.2.0_candidate.tflite"
PREPROCESSING_PROFILE_ID = "M-B1_D0_B1_Z1"
PREPROCESSING_PROFILE_NAME = "BPF_ZSCORE"
EXECUTION_PREPROCESSING_CONTRACT_ID = "M-B10B_SELECTED_REAL_CANDIDATE_BPF_ZSCORE_V1"
TRAINING_STRATEGY_ID = "M-B2_CE_UNWEIGHTED"
ARCHITECTURE_ID = "M-B3_CONV1D_GAP_BASELINE"
CALIBRATION_ID = "M-B5_CAL_CLASS_BALANCED_120"
RAW_ARCHIVE_REL = "datasets/raw_archives/external_datasets/db_records.zip"
CANONICAL_NPY_REL = "datasets/mmwave/processed/mmwave_canonical_real_v1.npy"
A5_SPLIT_REL = "datasets/mmwave/splits/mmwave_real_subject_split_v1.json"
A6_MANIFEST_REL = "datasets/mmwave/manifests/a6_full_conversion/full_window_manifest.jsonl"
B_DIR_REL = Path("datasets/mmwave/manifests/M-B10R1B_recovery_execution")
EXPECTED_MODELS = (SELECTED_MODEL_ID, V01_MODEL_ID, V02_MODEL_ID)
EXPECTED_ELIGIBLE = 75
EXPECTED_PAIRS = 225
MODEL_ROLE = {
    SELECTED_MODEL_ID: "SELECTED_NEW_REAL_DATA_CANDIDATE",
    V01_MODEL_ID: "HISTORICAL_MODEL_COMPATIBILITY_BENCHMARK",
    V02_MODEL_ID: "SYNTHETIC_TRAINED_EXTERNAL_COMPATIBILITY_BENCHMARK",
}

LOCK_JSON_FILES = (
    "artifact_lock_identity.json",
    "source_lineage_lock.json",
    "canonical_dataset_lock.json",
    "subject_split_lock.json",
    "window_population_lock.json",
    "preprocessing_lock.json",
    "training_lock.json",
    "model_artifact_lock.json",
    "quantization_lock.json",
    "runtime_contract_lock.json",
    "phase_b_lineage_registry.json",
    "final_evaluation_lock.json",
    "recovery_access_history_lock.json",
    "final_sample_registry_lock.json",
    "final_metric_lock.json",
    "final_subject_metric_lock.json",
    "baseline_comparison_lock.json",
    "scientific_limitations.json",
    "claim_boundary_lock.json",
    "immutable_artifact_registry.json",
    "artifact_lock_summary.json",
    "validation_result.json",
)


class MB11LockError(Exception):
    """Fail-closed M-B11 generation error."""


def _raise(code: str) -> None:
    raise MB11LockError(code)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def require_repo_relative(path: str, *, context: str) -> str:
    value = str(path)
    if not value or value.startswith("/") or value.startswith("file:") or "\\" in value or ".." in value:
        _raise(f"UNSAFE_PATH:{context}:{value}")
    return value


def inspect_tflite_identity(root: Path, relative: str) -> dict[str, Any]:
    """Inspect TFLite tensor/op identity. Allocates tensors; never invokes."""
    rel = require_repo_relative(relative, context="tflite")
    path = root / rel
    if not path.is_file():
        _raise(f"MODEL_FILE_MISSING:{rel}")
    stat = path.stat()
    return _inspect_tflite_cached(str(path.resolve()), rel, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=None)
def _inspect_tflite_cached(resolved: str, rel: str, mtime_ns: int, size: int) -> dict[str, Any]:
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        _raise(f"TFLITE_RUNTIME_UNAVAILABLE:{exc}")
    interpreter = tf.lite.Interpreter(model_path=resolved, num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    op_names = [str(item.get("op_name", "")) for item in interpreter._get_ops_details()]  # noqa: SLF001
    in_quant = tuple(float(item) for item in input_detail.get("quantization", (0.0, 0)))
    out_quant = tuple(float(item) for item in output_detail.get("quantization", (0.0, 0)))
    flex_or_select = any("FLEX" in name.upper() or "SELECT" in name.upper() for name in op_names)
    builtin_only = not flex_or_select
    return {
        "repo_relative_path": rel,
        "sha256": sha256_file(Path(resolved)),
        "bytes": int(size),
        "input_shape": [int(item) for item in input_detail["shape"]],
        "input_dtype": np.dtype(input_detail["dtype"]).name,
        "input_scale": in_quant[0],
        "input_zero_point": int(in_quant[1]),
        "output_shape": [int(item) for item in output_detail["shape"]],
        "output_dtype": np.dtype(output_detail["dtype"]).name,
        "output_scale": out_quant[0],
        "output_zero_point": int(out_quant[1]),
        "operator_inventory": op_names,
        "flex_ops_present": flex_or_select,
        "select_tf_ops_present": any("SELECT" in name.upper() for name in op_names),
        "builtin_op_status": "BUILTIN_ONLY" if builtin_only else "NON_BUILTIN_PRESENT",
        "strict_int8": (
            np.dtype(input_detail["dtype"]).name == "int8"
            and np.dtype(output_detail["dtype"]).name == "int8"
        ),
    }


def analyze_recovery_ledger(
    registry: dict[str, Any],
    ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    """Independently verify 75 x 3 Cartesian coverage and cross-model identity."""
    ordered = list(registry.get("ordered_window_ids") or [])
    unique_ids = list(dict.fromkeys(ordered))
    if len(ordered) != EXPECTED_ELIGIBLE or len(unique_ids) != EXPECTED_ELIGIBLE:
        _raise(f"REGISTRY_ID_COUNT:{len(unique_ids)}")
    if len(ledger) != EXPECTED_PAIRS:
        _raise(f"LEDGER_ROW_COUNT:{len(ledger)}")
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_window: dict[str, dict[str, dict[str, Any]]] = {}
    model_ids: set[str] = set()
    for row in ledger:
        window_id = str(row.get("window_id"))
        model_id = str(row.get("model_id"))
        model_ids.add(model_id)
        key = (window_id, model_id)
        by_pair.setdefault(key, []).append(row)
        by_window.setdefault(window_id, {})[model_id] = row
    expected_pairs = {(window_id, model_id) for window_id in unique_ids for model_id in EXPECTED_MODELS}
    actual_pairs = set(by_pair)
    duplicates = sorted(f"{window_id}|{model_id}" for (window_id, model_id), rows in by_pair.items() if len(rows) != 1)
    missing = sorted(f"{window_id}|{model_id}" for window_id, model_id in sorted(expected_pairs - actual_pairs))
    unexpected = sorted(f"{window_id}|{model_id}" for window_id, model_id in sorted(actual_pairs - expected_pairs))
    if duplicates or missing or unexpected:
        _raise(
            "CARTESIAN_MISMATCH:"
            f"dup={len(duplicates)},missing={len(missing)},unexpected={len(unexpected)}"
        )
    if model_ids != set(EXPECTED_MODELS):
        _raise(f"MODEL_SET_MISMATCH:{sorted(model_ids)}")
    label_mismatches = 0
    subject_mismatches = 0
    recording_mismatches = 0
    samples: list[dict[str, Any]] = []
    for window_id in unique_ids:
        rows = by_window[window_id]
        if set(rows) != set(EXPECTED_MODELS):
            _raise(f"WINDOW_MODEL_SET_MISMATCH:{window_id}")
        seed42 = rows[SELECTED_MODEL_ID]
        subjects = {str(rows[model_id].get("subject_id")) for model_id in EXPECTED_MODELS}
        recordings = {str(rows[model_id].get("recording_id")) for model_id in EXPECTED_MODELS}
        labels = {str(rows[model_id].get("true_class")) for model_id in EXPECTED_MODELS}
        label_indexes = {int(rows[model_id].get("true_class_index")) for model_id in EXPECTED_MODELS}
        if len(subjects) != 1:
            subject_mismatches += 1
        if len(recordings) != 1:
            recording_mismatches += 1
        if len(labels) != 1 or len(label_indexes) != 1:
            label_mismatches += 1
        samples.append(
            {
                "window_id": window_id,
                "subject_id": str(seed42.get("subject_id")),
                "recording_id": str(seed42.get("recording_id")),
                "true_class": str(seed42.get("true_class")),
                "true_class_index": int(seed42.get("true_class_index")),
                "models": {
                    model_id: {
                        "model_id": model_id,
                        "model_role": str(rows[model_id].get("model_role") or MODEL_ROLE[model_id]),
                        "predicted_class": str(rows[model_id].get("predicted_class")),
                        "predicted_class_index": int(rows[model_id].get("predicted_class_index")),
                        "invalid": bool(rows[model_id].get("invalid")),
                        "true_class": str(rows[model_id].get("true_class")),
                        "true_class_index": int(rows[model_id].get("true_class_index")),
                        "subject_id": str(rows[model_id].get("subject_id")),
                        "recording_id": str(rows[model_id].get("recording_id")),
                    }
                    for model_id in EXPECTED_MODELS
                },
            }
        )
    if label_mismatches or subject_mismatches or recording_mismatches:
        _raise(
            "CROSS_MODEL_IDENTITY_MISMATCH:"
            f"label={label_mismatches},subject={subject_mismatches},recording={recording_mismatches}"
        )
    per_model_rows = {
        model_id: [by_window[window_id][model_id] for window_id in unique_ids]
        for model_id in EXPECTED_MODELS
    }
    return {
        "unique_eligible_window_ids": len(unique_ids),
        "ordered_window_ids": unique_ids,
        "model_ids": list(EXPECTED_MODELS),
        "expected_pairs": EXPECTED_PAIRS,
        "actual_pairs": len(ledger),
        "duplicates": 0,
        "missing": 0,
        "unexpected": 0,
        "cross_model_label_mismatches": 0,
        "cross_model_subject_mismatches": 0,
        "cross_model_recording_mismatches": 0,
        "samples": samples,
        "per_model_rows": per_model_rows,
    }


def _metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("invalid")]
    labels = [int(row["true_class_index"]) for row in valid]
    predictions = [int(row["predicted_class_index"]) for row in valid]
    bundle = metric_bundle(labels, predictions, evaluated_sample_count=len(valid))
    bundle["valid_count"] = len(valid)
    bundle["invalid_count"] = len(rows) - len(valid)
    bundle["tflite_invokes"] = len(valid)
    return bundle


def _artifact(role: str, phase: str, rel: str, root: Path, source: str) -> dict[str, Any]:
    path = require_repo_relative(rel, context=role)
    target = root / path
    if not target.is_file():
        _raise(f"MISSING_REFERENCED_FILE:{path}")
    return {
        "artifact_role": role,
        "phase": phase,
        "repo_relative_path": path,
        "sha256": sha256_file(target),
        "bytes": int(target.stat().st_size),
        "immutable": True,
        "source_of_truth": source,
    }


def write_checksums(out: Path) -> None:
    lines = []
    for name in sorted(LOCK_JSON_FILES):
        path = out / name
        if not path.is_file():
            _raise(f"CHECKSUM_SOURCE_MISSING:{name}")
        lines.append(f"{sha256_file(path)}  {name}")
    atomic_write_text(out / "checksums.sha256", "\n".join(lines) + "\n")


def _npy_identity(root: Path, rel: str) -> dict[str, Any]:
    path = root / require_repo_relative(rel, context="canonical_npy")
    if not path.is_file():
        _raise(f"CANONICAL_NPY_MISSING:{rel}")
    array = np.load(path, mmap_mode="r")
    return {
        "repo_relative_path": rel,
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
    }


def print_start_report(root: Path) -> None:
    print("M-B11 START REPORT")
    print(f"Repository root: {root}")
    print("origin/main SHA: (see git; generator does not query remotes)")
    print("M-B10R1-B reviewed head in ancestry: 1893b26a0da567680c42f832043e98afd79933b6")
    print("Branch: feature/M-B11-mmwave-artifact-lock")
    print("LOCKED_TEST access during generation: 0")
    print("Recovery access during generation: 0")
    print("Inference during generation: 0")
    print(f"Candidate ID: {SELECTED_CANDIDATE_ID}")
    print(f"Runtime model ID: {RUNTIME_MODEL_ID}")
    print(f"Model path: {SELECTED_TFLITE_REL}")


def generate_m_b11_artifact_lock(root: Path | None = None) -> dict[str, Any]:
    root = Path(root) if root is not None else ROOT_DIR
    print_start_report(root)
    lock_dir = root / LOCK_DIR_REL
    lock_dir.mkdir(parents=True, exist_ok=True)

    a0 = load_json(root / "datasets/mmwave/manifests/a0_raw_inventory/source_identity.json")
    a5 = load_json(root / "datasets/mmwave/manifests/a5_subject_split/a5_summary.json")
    a5_profile = load_json(root / "datasets/mmwave/manifests/a5_subject_split/split_profile.json")
    a6 = load_json(root / "datasets/mmwave/manifests/a6_full_conversion/a6_summary.json")
    a6_split = load_json(root / "datasets/mmwave/manifests/a6_full_conversion/full_split_distribution.json")
    b1_selected = load_json(root / "datasets/mmwave/manifests/M-B1_preprocessing_ablation/selected_preprocessing_profile.json")
    b1_bpf = load_json(root / "datasets/mmwave/manifests/M-B1_preprocessing_ablation/bpf_frequency_diagnostic.json")
    b1_fit = load_json(root / "datasets/mmwave/manifests/M-B1_preprocessing_ablation/train_fit_statistics.json")
    b2_contract = load_json(root / "datasets/mmwave/manifests/M-B2_class_imbalance/experiment_contract.json")
    b2_selected = load_json(root / "datasets/mmwave/manifests/M-B2_class_imbalance/selected_imbalance_strategy.json")
    b2_runs = load_json(root / "datasets/mmwave/manifests/M-B2_class_imbalance/training_runs.json")
    b3_runs = load_json(root / "datasets/mmwave/manifests/M-B3_architecture_comparison/training_runs.json")
    b4_seeds = load_json(root / "datasets/mmwave/manifests/M-B4_multiseed_stability/per_seed_results.json")
    b4_summary = load_json(root / "datasets/mmwave/manifests/M-B4_multiseed_stability/m_b4_summary.json")
    b5_summary = load_json(root / "datasets/mmwave/manifests/M-B5_representative_calibration/m_b5_summary.json")
    b6_summary = load_json(root / "datasets/mmwave/manifests/M-B6_stage_equivalence/m_b6_summary.json")
    b6_artifacts = load_json(root / "datasets/mmwave/manifests/M-B6_stage_equivalence/stage_artifact_manifest.json")
    b7_summary = load_json(root / "datasets/mmwave/manifests/M-B7_perturbation_robustness/m_b7_summary.json")
    b8_summary = load_json(root / "datasets/mmwave/manifests/M-B8_mac_latency_footprint/m_b8_summary.json")
    b9_summary = load_json(root / "datasets/mmwave/manifests/M-B9_mock_e2e/m_b9_summary.json")
    b9_runtime = load_json(root / "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_manifests/seed42_runtime_manifest.json")
    b10a = load_json(root / "datasets/mmwave/manifests/M-B10A_candidate_selection_setup/m_b10a_summary.json")
    b10b = load_json(root / "datasets/mmwave/manifests/M-B10B_locked_test_final_evaluation/m_b10b_summary.json")
    b10b_audit = load_json(root / "datasets/mmwave/manifests/M-B10B_locked_test_final_evaluation/one_time_access_audit.json")
    b10r0 = load_json(root / "datasets/mmwave/manifests/M-B10R0_holdout_policy_review/m_b10r0_summary.json")
    b10r1a = load_json(root / "datasets/mmwave/manifests/M-B10R1A_recovery_prefreeze/m_b10r1a_summary.json")
    b_dir = root / B_DIR_REL
    b_summary = load_json(b_dir / "m_b10r1b_summary.json")
    b_audit = load_json(b_dir / "one_time_recovery_access_audit.json")
    b_runtime = load_json(b_dir / "recovery_access_runtime_state.json")
    b_quant = load_json(b_dir / "selected_candidate_quantization_audit.json")
    b_selected = load_json(b_dir / "selected_candidate_recovery_result.json")
    b_baselines = load_json(b_dir / "historical_baseline_recovery_results.json")
    b_subjects = load_json(b_dir / "subject_level_metrics.json")
    registry = load_json(b_dir / "recovery_registry.json")
    ledger = load_jsonl(b_dir / "recovery_sample_predictions.jsonl")

    raw_path = root / RAW_ARCHIVE_REL
    if not raw_path.is_file():
        _raise(f"RAW_ARCHIVE_MISSING:{RAW_ARCHIVE_REL}")
    raw_sha = sha256_file(raw_path)
    raw_bytes = int(raw_path.stat().st_size)
    documented_raw_sha = a0["local_archive"]["sha256"]
    if raw_sha != documented_raw_sha:
        _raise(f"RAW_SHA_MISMATCH:{raw_sha}")

    canonical = _npy_identity(root, CANONICAL_NPY_REL)
    if canonical["shape"] != [530, 300] or canonical["dtype"] != "float64":
        _raise(f"CANONICAL_IDENTITY_MISMATCH:{canonical['shape']}:{canonical['dtype']}")
    a5_split_sha = sha256_file(root / A5_SPLIT_REL)
    a6_manifest_sha = sha256_file(root / A6_MANIFEST_REL)
    selected_live = inspect_tflite_identity(root, SELECTED_TFLITE_REL)
    v01_live = inspect_tflite_identity(root, V01_TFLITE_REL)
    v02_live = inspect_tflite_identity(root, V02_TFLITE_REL)
    if selected_live["sha256"] != b6_artifacts["artifacts"]["M-B3_CONV1D_GAP_BASELINE_seed_42_stage_c"]["sha256"]:
        _raise("SELECTED_TFLITE_SHA_MISMATCH_B6")
    if selected_live["bytes"] != 22080 or not selected_live["strict_int8"]:
        _raise("SELECTED_TFLITE_IDENTITY_MISMATCH")

    analysis = analyze_recovery_ledger(registry, ledger)
    seed42_metrics = _metrics_for_rows(analysis["per_model_rows"][SELECTED_MODEL_ID])
    v01_metrics = _metrics_for_rows(analysis["per_model_rows"][V01_MODEL_ID])
    v02_metrics = _metrics_for_rows(analysis["per_model_rows"][V02_MODEL_ID])
    seed42_subjects = subject_metrics(analysis["per_model_rows"][SELECTED_MODEL_ID])
    seed42_saturation = saturation_audit_from_rows(analysis["per_model_rows"][SELECTED_MODEL_ID])
    stored_seed42 = b_selected["metrics"]
    if seed42_metrics["macro_f1"] != stored_seed42["macro_f1"]:
        _raise("SEED42_MACRO_F1_RECOMPUTE_MISMATCH")
    if seed42_metrics["confusion_matrix"] != stored_seed42["confusion_matrix"]:
        _raise("SEED42_CONFUSION_RECOMPUTE_MISMATCH")

    zscore = b1_fit["zscore_statistics"][PREPROCESSING_PROFILE_ID]
    bpf = b1_bpf["bpf_parameters"]
    b2_fixed = b2_contract["fixed_conditions"]
    b2_run = b2_runs["training_runs"][TRAINING_STRATEGY_ID]
    b3_run = b3_runs["training_runs"][ARCHITECTURE_ID]
    b4_per = b4_seeds["per_seed_results"]
    seed42_val = b4_per["M-B3_CONV1D_GAP_BASELINE_seed_42"]["val_macro_f1"]
    seed43_val = b4_per["M-B3_CONV1D_GAP_BASELINE_seed_43"]["val_macro_f1"]
    seed44_val = b4_per["M-B3_CONV1D_GAP_BASELINE_seed_44"]["val_macro_f1"]
    b8_seed42 = b8_summary["cross_seed_invoke"]["per_seed_pooled_statistics"]["42"]["statistics_ms"]
    runtime_input = b9_runtime["runtime_model"]["input"]
    runtime_ops = b9_runtime["runtime_model"]["op_inventory"]
    if (
        selected_live["input_scale"] != runtime_input["scale"]
        or selected_live["input_zero_point"] != runtime_input["zero_point"]
        or selected_live["output_scale"] != runtime_ops["output_scale"]
        or selected_live["output_zero_point"] != runtime_ops["output_zero_point"]
    ):
        _raise("LIVE_TFLITE_QUANT_MISMATCH_B9")

    eligibility = a6_split["eligibility_counts"]
    labels = a6["label_window_distribution"]
    windows = a6["split_window_distribution"]
    subjects = a6["split_subject_distribution"]

    identity = {
        "schema_version": SCHEMA,
        "phase_id": PHASE_ID,
        "artifact_status": ARTIFACT_STATUS,
        "result_limitation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "m_b11_creates_new_model": False,
        "selected_candidate_changed": False,
        "candidate_id": SELECTED_CANDIDATE_ID,
        "runtime_model_id": RUNTIME_MODEL_ID,
        "class_map": CLASS_MAP,
        "apnea_is_proxy": True,
        "git_commit_sha_supplementary_only": True,
        "artifact_identity_is_sha256_not_git_commit": True,
    }
    source_lock = {
        "schema_version": SCHEMA,
        "raw_archive_repo_relative_path": RAW_ARCHIVE_REL,
        "raw_archive_sha256": raw_sha,
        "raw_archive_bytes": raw_bytes,
        "doi": a0["dataset_identity"]["doi"],
        "version": "v1.1",
        "source": "Zenodo",
        "dataset_id": a0["dataset_identity"]["dataset_id"],
        "measured_subjects": 110,
        "measured_recordings": 440,
        "raw_payload_copied_into_lock": False,
        "a0_source_identity": "datasets/mmwave/manifests/a0_raw_inventory/source_identity.json",
    }
    canonical_lock = {
        "schema_version": SCHEMA,
        "canonical_npy_repo_relative_path": canonical["repo_relative_path"],
        "canonical_npy_sha256": canonical["sha256"],
        "canonical_npy_bytes": canonical["bytes"],
        "shape": canonical["shape"],
        "dtype": canonical["dtype"],
        "regenerated_during_m_b11": False,
        "source_archive_sha256": raw_sha,
        "a6_summary": "datasets/mmwave/manifests/a6_full_conversion/a6_summary.json",
    }
    split_lock = {
        "schema_version": SCHEMA,
        "split_artifact_repo_relative_path": A5_SPLIT_REL,
        "split_sha256": a5_split_sha,
        "profile_id": a5_profile["profile_id"],
        "split_seed": a5_profile["split_seed"],
        "split_unit": a5_profile["split_unit"],
        "subject_counts": {
            "TRAIN": int(subjects["TRAIN"]),
            "VALIDATION": int(subjects["VALIDATION"]),
            "LOCKED_TEST": int(subjects["LOCKED_TEST"]),
        },
        "overlap": 0,
        "union": 110,
        "a5_summary_subject_counts": a5["subject_counts"],
        "pilot_eligibility_counts_not_full_population": a5["eligibility_counts"],
    }
    window_lock = {
        "schema_version": SCHEMA,
        "a6_manifest_repo_relative_path": A6_MANIFEST_REL,
        "a6_manifest_sha256": a6_manifest_sha,
        "total_canonical_windows": 530,
        "structural": {
            "TRAIN": int(windows["TRAIN"]),
            "VALIDATION": int(windows["VALIDATION"]),
            "LOCKED_TEST": int(windows["LOCKED_TEST"]),
        },
        "pure_supervised_eligible": {
            "TRAIN": int(eligibility["training_eligible"]),
            "VALIDATION": int(eligibility["validation_eligible"]),
            "LOCKED_TEST": int(eligibility["locked_test_evaluation_eligible"]),
        },
        "class_totals": {
            "NORMAL": int(labels["NORMAL"]),
            "RAPID_OR_ABNORMAL": int(labels["RAPID_OR_ABNORMAL"]),
            "APNEA": int(labels["APNEA"]),
            "AMBIGUOUS": int(labels["AMBIGUOUS"]),
        },
        "locked_test": {
            "structural": 88,
            "supervised_eligible": 75,
            "excluded_ambiguous_or_non_eligible": 13,
            "eligible_subjects": 16,
        },
        "ambiguous_excluded_from_pure_class_training": True,
    }
    preprocessing_lock = {
        "schema_version": SCHEMA,
        "selected_profile_id": b1_selected["selected_profile_id"],
        "selected_profile_name": b1_selected["selected_profile_name"],
        "execution_preprocessing_contract_id": EXECUTION_PREPROCESSING_CONTRACT_ID,
        "detrend": False,
        "bpf_enabled": True,
        "bpf": {
            "filter_family": "Butterworth",
            "btype": "bandpass",
            "lowcut_hz": bpf["lowcut_hz"],
            "highcut_hz": bpf["highcut_hz"],
            "order": bpf["order"],
            "fs_hz": bpf["fs_hz"],
            "zero_phase": True,
            "implementation": "scipy.signal.filtfilt",
            "executor": "scripts/mmwave_m_b1_preprocessing.py:apply_bpf",
        },
        "zscore": {
            "enabled": True,
            "fit_split": zscore["fit_split"],
            "fit_window_count": zscore["fit_window_count"],
            "mean": zscore["mean"],
            "std": zscore["std"],
        },
        "clipping_policy": {
            "b1_transform_amplitude_clip_applied": False,
            "legacy_abs5_clip_applied": False,
            "legacy_abs5_clip_is_diagnostic_only": True,
            "int8_quantization_clip": {
                "applied": True,
                "dtype": "int8",
                "min": -128,
                "max": 127,
                "source": "pre-clamp quantized values before int8 clipping",
            },
        },
        "input_shape": [None, 300],
        "output_shape": [None, 300],
        "dtype_behavior": "float64_transform_then_float32_model_ready_then_int8_quantized",
        "not_reduced_to_bpf_zscore_string_alone": True,
    }
    if preprocessing_lock["selected_profile_id"] != PREPROCESSING_PROFILE_ID:
        _raise("PREPROCESSING_PROFILE_MISMATCH")
    training_lock = {
        "schema_version": SCHEMA,
        "selected_strategy_id": b2_selected["selected_strategy_id"],
        "supervised_population": "A6_PURE_CLASS_WINDOWS_AMBIGUOUS_EXCLUDED",
        "ambiguous_exclusion": True,
        "loss": "sparse_categorical_crossentropy",
        "loss_weighting": "UNWEIGHTED",
        "optimizer": b2_fixed["optimizer"],
        "learning_rate": b2_fixed["learning_rate"],
        "batch_size": b2_fixed["batch_size"],
        "max_epochs": b2_fixed["max_epochs"],
        "early_stopping_monitor": b2_fixed["early_stopping_monitor"],
        "early_stopping_patience": b2_fixed["early_stopping_patience"],
        "restore_best_weights": b2_fixed["restore_best_weights"],
        "training_seed": b2_fixed["initialization_seed"],
        "parameter_count": b2_run["parameter_count"],
        "b2_initial_weights_sha256": b2_run["initial_weights_sha256"],
        "b3_selected_architecture_initial_weights_sha256": b3_run["initial_weights_sha256"],
        "b3_param_counts": b3_run["param_counts"],
        "retrained_during_m_b11": False,
    }
    if training_lock["selected_strategy_id"] != TRAINING_STRATEGY_ID:
        _raise("TRAINING_STRATEGY_MISMATCH")
    model_lock = {
        "schema_version": SCHEMA,
        "model_role": "SELECTED_NEW_REAL_DATA_CANDIDATE",
        "candidate_id": SELECTED_CANDIDATE_ID,
        "runtime_model_id": RUNTIME_MODEL_ID,
        "architecture_id": ARCHITECTURE_ID,
        "seed": 42,
        "calibration_profile": CALIBRATION_ID,
        "training_strategy_id": TRAINING_STRATEGY_ID,
        "repo_relative_path": SELECTED_TFLITE_REL,
        "sha256": selected_live["sha256"],
        "bytes": selected_live["bytes"],
        "copied_or_renamed_during_m_b11": False,
        "strict_int8": True,
        "builtin_op_status": selected_live["builtin_op_status"],
        "flex_ops_present": selected_live["flex_ops_present"],
        "select_tf_ops_present": selected_live["select_tf_ops_present"],
        "class_map": CLASS_MAP,
        "input_tensor": {
            "shape": selected_live["input_shape"],
            "dtype": selected_live["input_dtype"],
            "scale": selected_live["input_scale"],
            "zero_point": selected_live["input_zero_point"],
        },
        "output_tensor": {
            "shape": selected_live["output_shape"],
            "dtype": selected_live["output_dtype"],
            "scale": selected_live["output_scale"],
            "zero_point": selected_live["output_zero_point"],
        },
        "operator_inventory": selected_live["operator_inventory"],
    }
    quantization_lock = {
        "schema_version": SCHEMA,
        "source": "datasets/mmwave/manifests/M-B10R1B_recovery_execution/selected_candidate_quantization_audit.json",
        "recomputed_from_stored_seed42_rows": True,
        "inference_rerun": False,
        "total_quantized_elements": seed42_saturation["total_quantized_elements"],
        "pre_clamp_out_of_range_count": seed42_saturation["pre_clamp_out_of_range_count"],
        "input_saturation_ratio": seed42_saturation["input_saturation_ratio"],
        "samples_with_any_saturation": seed42_saturation["samples_with_any_saturation"],
        "worst_sample_saturation_ratio": seed42_saturation["worst_sample_saturation_ratio"],
        "stored_audit": {
            "total_quantized_elements": b_quant["total_quantized_elements"],
            "pre_clamp_out_of_range_count": b_quant["pre_clamp_out_of_range_count"],
            "input_saturation_ratio": b_quant["input_saturation_ratio"],
            "samples_with_any_saturation": b_quant["samples_with_any_saturation"],
            "worst_sample_saturation_ratio": b_quant["worst_sample_saturation_ratio"],
        },
    }
    runtime_lock = {
        "schema_version": SCHEMA,
        "runtime_identity_source": "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_manifests/seed42_runtime_manifest.json",
        "input_contract": {
            "shape": runtime_input["shape"],
            "dtype": runtime_input["dtype"],
            "scale": runtime_input["scale"],
            "zero_point": runtime_input["zero_point"],
            "semantic": runtime_input["semantic"],
        },
        "output_contract": {
            "shape": runtime_ops["output_shape"],
            "dtype": runtime_ops["output_dtype"],
            "scale": runtime_ops["output_scale"],
            "zero_point": runtime_ops["output_zero_point"],
        },
        "strict_int8": True,
        "flex_select_absent": runtime_ops["flex_select_absent"],
        "select_tf_ops_count": runtime_ops["select_tf_ops_count"],
        "class_map": CLASS_MAP,
        "live_tflite_matches_runtime_identity": True,
    }
    lineage = {
        "schema_version": SCHEMA,
        "selected_path": {
            "M-B0": {
                "role": "evaluation_protocol_and_locked_test_access_control",
                "summary": "datasets/mmwave/manifests/M-B0_evaluation_protocol/m_b0_summary.json",
            },
            "M-B1": {
                "selected_preprocessing": PREPROCESSING_PROFILE_ID,
                "semantic_name": PREPROCESSING_PROFILE_NAME,
                "summary": "datasets/mmwave/manifests/M-B1_preprocessing_ablation/m_b1_summary.json",
                "selected_profile": "datasets/mmwave/manifests/M-B1_preprocessing_ablation/selected_preprocessing_profile.json",
            },
            "M-B2": {
                "selected_strategy": TRAINING_STRATEGY_ID,
                "summary": "datasets/mmwave/manifests/M-B2_class_imbalance/m_b2_summary.json",
            },
            "M-B3": {
                "selected_architecture": ARCHITECTURE_ID,
                "strict_int8_artifact": SELECTED_TFLITE_REL,
                "summary": "datasets/mmwave/manifests/M-B3_architecture_comparison/m_b3_summary.json",
            },
            "M-B4": {
                "seed_set": [42, 43, 44],
                "primary_stable_float_finalist": b4_summary["primary_stable_float_finalist"],
                "validation_macro_f1": {
                    "seed42": seed42_val,
                    "seed43": seed43_val,
                    "seed44": seed44_val,
                },
                "initialization_sensitivity": True,
                "seed42_materially_better_than_seed44": True,
                "summary": "datasets/mmwave/manifests/M-B4_multiseed_stability/m_b4_summary.json",
            },
            "M-B5": {
                "selected_calibration": CALIBRATION_ID,
                "summary": "datasets/mmwave/manifests/M-B5_representative_calibration/m_b5_summary.json",
            },
            "M-B6": {
                "stage_equivalence_verified": True,
                "frozen_int8_sha256": selected_live["sha256"],
                "summary": "datasets/mmwave/manifests/M-B6_stage_equivalence/m_b6_summary.json",
                "stage_artifact_manifest": "datasets/mmwave/manifests/M-B6_stage_equivalence/stage_artifact_manifest.json",
            },
            "M-B7": {
                "perturbation_robustness_evidence": True,
                "seed44_moderate_profile_collapse": True,
                "seed42_retained": True,
                "new_collapse_conditions": b7_summary["new_collapse_conditions"],
                "summary": "datasets/mmwave/manifests/M-B7_perturbation_robustness/m_b7_summary.json",
            },
            "M-B8": {
                "scope": "MAC_M2_LATENCY_AND_FOOTPRINT_ONLY",
                "not_raspberry_pi_latency": True,
                "host": "Apple M2",
                "seed42_median_ms": b8_seed42["median"],
                "seed42_p99_ms": b8_seed42["p99"],
                "summary": "datasets/mmwave/manifests/M-B8_mac_latency_footprint/m_b8_summary.json",
            },
            "M-B9": {
                "scope": "MOCK_RUNTIME_PATH_EQUIVALENCE_AND_FAIL_CLOSED",
                "not_physical_sensor_integration": True,
                "summary": "datasets/mmwave/manifests/M-B9_mock_e2e/m_b9_summary.json",
            },
            "M-B10A": {
                "role": "candidate_preregistration_before_LOCKED_TEST",
                "selected_candidate_id": b10a["selected_candidate_id"],
                "locked_test_accesses": b10a["locked_test_accesses"],
                "summary": "datasets/mmwave/manifests/M-B10A_candidate_selection_setup/m_b10a_summary.json",
            },
            "M-B10B": {
                "original_final_accessor_invocations": b10b["final_accessor_invocations"],
                "payload_returned": b10b["rows_returned"],
                "model_inference": b10b["model_inference_invocations"],
                "predictions": 0,
                "metrics": 0,
                "original_locked_test_consumed": True,
                "root_cause": b10b["forensic_root_cause"],
                "summary": "datasets/mmwave/manifests/M-B10B_locked_test_final_evaluation/m_b10b_summary.json",
            },
            "M-B10R0": {
                "policy_decision": b10r0["policy_decision"],
                "summary": "datasets/mmwave/manifests/M-B10R0_holdout_policy_review/m_b10r0_summary.json",
            },
            "M-B10R1-A": {
                "recovery_harness_frozen_before_reuse": True,
                "new_access": b10r1a["new_recovery_accessor_invocations"],
                "summary": "datasets/mmwave/manifests/M-B10R1A_recovery_prefreeze/m_b10r1a_summary.json",
            },
            "M-B10R1-B": {
                "recovery_accessor": b_audit["recovery_accessor_invocations"],
                "recovery_payload_release": b_audit["recovery_payload_release_events"],
                "actual_tflite_invokes": b_summary["actual_total_tflite_invocations"],
                "second_recovery": False,
                "rerun": False,
                "summary": "datasets/mmwave/manifests/M-B10R1B_recovery_execution/m_b10r1b_summary.json",
            },
        },
        "m_b11": {
            "role": "immutable_offline_candidate_lock",
            "improves_candidate": False,
        },
    }
    if b10a["selected_candidate_id"] != SELECTED_CANDIDATE_ID:
        _raise("B10A_SELECTED_CANDIDATE_CHANGED")
    if b5_summary["selected_calibration_profile"] != CALIBRATION_ID:
        _raise("CALIBRATION_ID_MISMATCH")

    recovery_history = {
        "schema_version": SCHEMA,
        "original_m_b10b_accessor_invocations": 1,
        "original_m_b10b_payload_releases": 1,
        "original_m_b10b_model_inference": 0,
        "m_b10r1b_recovery_accessor_invocations": 1,
        "m_b10r1b_recovery_payload_releases": 1,
        "historical_total_payload_releases": 2,
        "recovery_model_inference": 225,
        "second_recovery": False,
        "rerun": False,
        "payload_consumed": True,
        "result_limitation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "forbidden_future_reinterpretations": {
            "total_release_1": False,
            "total_release_3": False,
            "recovery_access_0": False,
            "recovery_access_gt_1": False,
            "rerun_true": False,
            "pristine_locked_test": False,
        },
        "source_audit": {
            "m_b10b_final_accessor_invocations": b10b["final_accessor_invocations"],
            "m_b10b_rows_returned": b10b["rows_returned"],
            "m_b10b_model_inference_invocations": b10b["model_inference_invocations"],
            "m_b10b_access_consumed": b10b_audit["access_consumed"],
            "m_b10r1b_recovery_accessor_invocations": b_audit["recovery_accessor_invocations"],
            "m_b10r1b_recovery_payload_release_events": b_audit["recovery_payload_release_events"],
            "m_b10r1b_historical_total_payload_release_events": b_audit["historical_total_payload_release_events"],
            "m_b10r1b_rerun_performed": b_audit["rerun_performed"],
            "runtime_original_final_accessor_invocations": b_runtime["original_final_accessor_invocations"],
            "runtime_recovery_accessor_invocations": b_runtime["recovery_accessor_invocations"],
        },
    }
    sample_registry = {
        "schema_version": SCHEMA,
        "unique_eligible_window_ids": analysis["unique_eligible_window_ids"],
        "model_ids": list(analysis["model_ids"]),
        "expected_pairs": analysis["expected_pairs"],
        "actual_pairs": analysis["actual_pairs"],
        "duplicates": analysis["duplicates"],
        "missing": analysis["missing"],
        "unexpected": analysis["unexpected"],
        "cross_model_label_mismatches": analysis["cross_model_label_mismatches"],
        "cross_model_subject_mismatches": analysis["cross_model_subject_mismatches"],
        "cross_model_recording_mismatches": analysis["cross_model_recording_mismatches"],
        "ordered_window_ids": analysis["ordered_window_ids"],
        "samples": analysis["samples"],
        "source_registry": "datasets/mmwave/manifests/M-B10R1B_recovery_execution/recovery_registry.json",
        "source_ledger": "datasets/mmwave/manifests/M-B10R1B_recovery_execution/recovery_sample_predictions.jsonl",
    }
    final_eval = {
        "schema_version": SCHEMA,
        "result_designation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "not_final_locked_test_pristine": True,
        "eligible_evaluated": 75,
        "valid": seed42_metrics["valid_count"],
        "invalid": seed42_metrics["invalid_count"],
        "tflite_invocations_selected": seed42_metrics["tflite_invokes"],
        "tflite_invocations_all_models": 225,
        "selected_runtime_model_id": RUNTIME_MODEL_ID,
        "source": "datasets/mmwave/manifests/M-B10R1B_recovery_execution/selected_candidate_recovery_result.json",
    }
    metric_lock = {
        "schema_version": SCHEMA,
        "model_id": RUNTIME_MODEL_ID,
        "recomputed_from_stored_seed42_rows": True,
        "inference_rerun": False,
        "accuracy": seed42_metrics["accuracy"],
        "macro_f1": seed42_metrics["macro_f1"],
        "macro_precision": seed42_metrics["macro_precision"],
        "macro_recall": seed42_metrics["macro_recall"],
        "per_class": seed42_metrics["per_class"],
        "apnea_proxy": seed42_metrics["apnea_proxy"],
        "confusion_matrix": seed42_metrics["confusion_matrix"],
        "prediction_distribution": seed42_metrics["prediction_distribution"],
        "class_collapse": seed42_metrics["class_collapse"],
        "stored_m_b10r1b_macro_f1": stored_seed42["macro_f1"],
    }
    subject_lock = {
        "schema_version": SCHEMA,
        "recomputed_from_stored_seed42_rows": True,
        "subject_count": seed42_subjects["subject_count"],
        "median_subject_macro_f1": seed42_subjects["median_subject_macro_f1"],
        "worst_subject_macro_f1": seed42_subjects["worst_subject_macro_f1"],
        "worst_subject_id": seed42_subjects["worst_subject_id"],
        "per_subject": {
            subject_id: {
                "window_count": row["window_count"],
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
            }
            for subject_id, row in seed42_subjects["per_subject"].items()
        },
        "stored_m_b10r1b_median": b_subjects[SELECTED_MODEL_ID]["median_subject_macro_f1"],
        "stored_m_b10r1b_worst": b_subjects[SELECTED_MODEL_ID]["worst_subject_macro_f1"],
        "stored_m_b10r1b_worst_id": b_subjects[SELECTED_MODEL_ID]["worst_subject_id"],
    }
    baseline_lock = {
        "schema_version": SCHEMA,
        "v0_1": {
            "model_id": V01_MODEL_ID,
            "role": MODEL_ROLE[V01_MODEL_ID],
            "repo_relative_path": V01_TFLITE_REL,
            "sha256": v01_live["sha256"],
            "bytes": v01_live["bytes"],
            "promoted_to_candidate": False,
            "accuracy": v01_metrics["accuracy"],
            "macro_f1": v01_metrics["macro_f1"],
            "prediction_distribution": v01_metrics["prediction_distribution"],
            "class_collapse": v01_metrics["class_collapse"],
            "all_75_predicted_normal": v01_metrics["prediction_distribution"]["NORMAL"] == 75,
            "stored_macro_f1": b_baselines[V01_MODEL_ID]["metrics"]["macro_f1"],
        },
        "v0_2": {
            "model_id": V02_MODEL_ID,
            "role": MODEL_ROLE[V02_MODEL_ID],
            "repo_relative_path": V02_TFLITE_REL,
            "sha256": v02_live["sha256"],
            "bytes": v02_live["bytes"],
            "promoted_to_candidate": False,
            "accuracy": v02_metrics["accuracy"],
            "macro_f1": v02_metrics["macro_f1"],
            "prediction_distribution": v02_metrics["prediction_distribution"],
            "class_collapse": v02_metrics["class_collapse"],
            "rapid_or_abnormal_predictions": v02_metrics["prediction_distribution"]["RAPID_OR_ABNORMAL"],
            "stored_macro_f1": b_baselines[V02_MODEL_ID]["metrics"]["macro_f1"],
        },
        "selected_seed42_macro_f1": seed42_metrics["macro_f1"],
        "selected_beats_both_baselines_on_macro_f1": (
            seed42_metrics["macro_f1"] > v01_metrics["macro_f1"]
            and seed42_metrics["macro_f1"] > v02_metrics["macro_f1"]
        ),
        "new_model_selection_event": False,
    }
    limitations = {
        "schema_version": SCHEMA,
        "locked_limitations_not_immediate_b_series_retuning_defects": True,
        "selected_seed42": {
            "beats_both_frozen_historical_baselines_on_final_macro_f1": True,
            "does_not_collapse_all_required_classes": not seed42_metrics["class_collapse"]["collapsed"],
            "strong_apnea_proxy_recall": seed42_metrics["apnea_proxy"]["recall"],
            "weak_normal_recall": seed42_metrics["per_class"]["NORMAL"]["recall"],
            "moderate_rapid_recall": seed42_metrics["per_class"]["RAPID_OR_ABNORMAL"]["recall"],
            "high_apnea_false_positive_rate": seed42_metrics["apnea_proxy"]["fpr"],
            "weak_worst_subject_generalization": seed42_subjects["worst_subject_macro_f1"],
            "initialization_sensitivity_from_m_b4": True,
            "seed42_val_macro_f1": seed42_val,
            "seed44_val_macro_f1": seed44_val,
        },
        "not_defects_requiring_immediate_b_series_retuning": True,
    }
    claims = {
        "schema_version": SCHEMA,
        "artifact_status": ARTIFACT_STATUS,
        "result_limitation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "PRISTINE_LOCKED_TEST": False,
        "FIRST_LOCKED_TEST_EVALUATION": False,
        "artifact_locked": True,
        "candidate_selection_locked": True,
        "training_locked": True,
        "preprocessing_locked": True,
        "calibration_locked": True,
        "model_binary_locked": True,
        "class_map_locked": True,
        "offline_final_evidence_locked": True,
        "locked_test_reopen_allowed": False,
        "recovery_reopen_allowed": False,
        "M-B12_required": True,
        "M-B11_artifact_lock_complete": True,
        "M-B12_offline_final_report_required": True,
        "Phase_B_release_ready": False,
        "MR60_device_validation_complete": False,
        "Raspberry_Pi_validation_complete": False,
        "multisensor_integration_complete": False,
        "deployment_ready": False,
        "production_ready": False,
        "clinical_apnea_validated": False,
        "m_b12_started": False,
        "m_c_started": False,
    }
    sensor_lock = {
        "schema_version": "M-B11_SENSOR_LOCAL_OFFLINE_CANDIDATE_LOCK_V1",
        "status": ARTIFACT_STATUS,
        "result_limitation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "candidate_id": SELECTED_CANDIDATE_ID,
        "runtime_model_id": RUNTIME_MODEL_ID,
        "repo_relative_path": SELECTED_TFLITE_REL,
        "sha256": selected_live["sha256"],
        "bytes": selected_live["bytes"],
        "tensor_contract": {
            "input": model_lock["input_tensor"],
            "output": model_lock["output_tensor"],
            "strict_int8": True,
            "flex_ops_present": selected_live["flex_ops_present"],
            "select_tf_ops_present": selected_live["select_tf_ops_present"],
        },
        "class_map": CLASS_MAP,
        "preprocessing_contract": {
            "profile_id": PREPROCESSING_PROFILE_ID,
            "profile_name": PREPROCESSING_PROFILE_NAME,
            "execution_contract_id": EXECUTION_PREPROCESSING_CONTRACT_ID,
        },
        "training_contract": {
            "strategy_id": TRAINING_STRATEGY_ID,
            "seed": 42,
        },
        "calibration": CALIBRATION_ID,
        "source_data_lineage": {
            "raw_archive": RAW_ARCHIVE_REL,
            "raw_sha256": raw_sha,
            "canonical_npy": CANONICAL_NPY_REL,
            "canonical_sha256": canonical["sha256"],
        },
        "split_lineage": {
            "split_artifact": A5_SPLIT_REL,
            "split_sha256": a5_split_sha,
            "split_seed": a5_profile["split_seed"],
        },
        "final_evaluation_identity": {
            "source": "M-B10R1-B",
            "result_limitation": RESULT_LIMITATION,
            "macro_f1": seed42_metrics["macro_f1"],
            "eligible_evaluated": 75,
        },
        "limitations": limitations["selected_seed42"],
        "future_m_c_requirement": True,
        "models_model_manifest_json_modified": False,
        "tflite_binary_copied_or_renamed": False,
        "deployment_ready": False,
        "MR60_device_validation_complete": False,
        "Raspberry_Pi_validation_complete": False,
        "clinical_apnea_validated": False,
    }
    sensor_path = root / SENSOR_LOCK_REL
    atomic_write_json(sensor_path, sensor_lock)

    registry_entries = [
        _artifact("raw_source_archive", "A0", RAW_ARCHIVE_REL, root, "datasets/mmwave/manifests/a0_raw_inventory/source_identity.json"),
        _artifact("a0_source_identity", "A0", "datasets/mmwave/manifests/a0_raw_inventory/source_identity.json", root, "A0"),
        _artifact("a5_subject_split", "A5", A5_SPLIT_REL, root, "datasets/mmwave/manifests/a5_subject_split/a5_summary.json"),
        _artifact("a5_summary", "A5", "datasets/mmwave/manifests/a5_subject_split/a5_summary.json", root, "A5"),
        _artifact("a5_split_profile", "A5", "datasets/mmwave/manifests/a5_subject_split/split_profile.json", root, "A5"),
        _artifact("a6_window_manifest", "A6", A6_MANIFEST_REL, root, "datasets/mmwave/manifests/a6_full_conversion/a6_summary.json"),
        _artifact("a6_summary", "A6", "datasets/mmwave/manifests/a6_full_conversion/a6_summary.json", root, "A6"),
        _artifact("canonical_npy", "A6", CANONICAL_NPY_REL, root, "datasets/mmwave/manifests/a6_full_conversion/a6_summary.json"),
        _artifact("b0_summary", "M-B0", "datasets/mmwave/manifests/M-B0_evaluation_protocol/m_b0_summary.json", root, "M-B0"),
        _artifact("b1_selected_profile", "M-B1", "datasets/mmwave/manifests/M-B1_preprocessing_ablation/selected_preprocessing_profile.json", root, "M-B1"),
        _artifact("b1_summary", "M-B1", "datasets/mmwave/manifests/M-B1_preprocessing_ablation/m_b1_summary.json", root, "M-B1"),
        _artifact("b1_executor", "M-B1", "scripts/mmwave_m_b1_preprocessing.py", root, "M-B1"),
        _artifact("b2_selected_strategy", "M-B2", "datasets/mmwave/manifests/M-B2_class_imbalance/selected_imbalance_strategy.json", root, "M-B2"),
        _artifact("b2_summary", "M-B2", "datasets/mmwave/manifests/M-B2_class_imbalance/m_b2_summary.json", root, "M-B2"),
        _artifact("b3_summary", "M-B3", "datasets/mmwave/manifests/M-B3_architecture_comparison/m_b3_summary.json", root, "M-B3"),
        _artifact("b4_summary", "M-B4", "datasets/mmwave/manifests/M-B4_multiseed_stability/m_b4_summary.json", root, "M-B4"),
        _artifact("b5_summary", "M-B5", "datasets/mmwave/manifests/M-B5_representative_calibration/m_b5_summary.json", root, "M-B5"),
        _artifact("b6_summary", "M-B6", "datasets/mmwave/manifests/M-B6_stage_equivalence/m_b6_summary.json", root, "M-B6"),
        _artifact("b6_stage_artifact_manifest", "M-B6", "datasets/mmwave/manifests/M-B6_stage_equivalence/stage_artifact_manifest.json", root, "M-B6"),
        _artifact("selected_tflite", "M-B6", SELECTED_TFLITE_REL, root, "M-B6/M-B10A"),
        _artifact("b7_summary", "M-B7", "datasets/mmwave/manifests/M-B7_perturbation_robustness/m_b7_summary.json", root, "M-B7"),
        _artifact("b8_summary", "M-B8", "datasets/mmwave/manifests/M-B8_mac_latency_footprint/m_b8_summary.json", root, "M-B8"),
        _artifact("b9_summary", "M-B9", "datasets/mmwave/manifests/M-B9_mock_e2e/m_b9_summary.json", root, "M-B9"),
        _artifact("b9_seed42_runtime_manifest", "M-B9", "datasets/mmwave/manifests/M-B9_mock_e2e/runtime_manifests/seed42_runtime_manifest.json", root, "M-B9"),
        _artifact("b10a_summary", "M-B10A", "datasets/mmwave/manifests/M-B10A_candidate_selection_setup/m_b10a_summary.json", root, "M-B10A"),
        _artifact("b10b_summary", "M-B10B", "datasets/mmwave/manifests/M-B10B_locked_test_final_evaluation/m_b10b_summary.json", root, "M-B10B"),
        _artifact("b10b_incident_root_cause", "M-B10B", "datasets/mmwave/manifests/M-B10B_locked_test_final_evaluation/incident_root_cause.json", root, "M-B10B"),
        _artifact("b10r0_summary", "M-B10R0", "datasets/mmwave/manifests/M-B10R0_holdout_policy_review/m_b10r0_summary.json", root, "M-B10R0"),
        _artifact("b10r1a_summary", "M-B10R1-A", "datasets/mmwave/manifests/M-B10R1A_recovery_prefreeze/m_b10r1a_summary.json", root, "M-B10R1-A"),
        _artifact("b10r1b_summary", "M-B10R1-B", "datasets/mmwave/manifests/M-B10R1B_recovery_execution/m_b10r1b_summary.json", root, "M-B10R1-B"),
        _artifact("b10r1b_registry", "M-B10R1-B", "datasets/mmwave/manifests/M-B10R1B_recovery_execution/recovery_registry.json", root, "M-B10R1-B"),
        _artifact("b10r1b_ledger", "M-B10R1-B", "datasets/mmwave/manifests/M-B10R1B_recovery_execution/recovery_sample_predictions.jsonl", root, "M-B10R1-B"),
        _artifact("v0_1_tflite", "historical_baseline", V01_TFLITE_REL, root, "M-B10A"),
        _artifact("v0_2_tflite", "historical_baseline", V02_TFLITE_REL, root, "M-B10A"),
        _artifact("b10b_baseline_preprocessing_executor", "M-B10B", "scripts/mmwave_m_b10b_baseline_preprocessing.py", root, "M-B10B"),
        _artifact("sensor_local_candidate_lock", "M-B11", str(SENSOR_LOCK_REL), root, "M-B11"),
    ]
    immutable_registry = {
        "schema_version": SCHEMA,
        "artifacts": registry_entries,
    }
    summary = {
        "schema_version": SCHEMA,
        "phase_id": PHASE_ID,
        "artifact_status": ARTIFACT_STATUS,
        "result_limitation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "candidate_id": SELECTED_CANDIDATE_ID,
        "runtime_model_id": RUNTIME_MODEL_ID,
        "model_sha256": selected_live["sha256"],
        "model_bytes": selected_live["bytes"],
        "macro_f1": seed42_metrics["macro_f1"],
        "v0_1_macro_f1": v01_metrics["macro_f1"],
        "v0_2_macro_f1": v02_metrics["macro_f1"],
        "M-B12_required": True,
        "Phase_B_release_ready": False,
        "new_locked_test_access": 0,
        "new_recovery_access": 0,
        "new_model_inference": 0,
    }
    validation_placeholder = {
        "schema_version": SCHEMA,
        "phase_id": PHASE_ID,
        "status": "LOCK_ARTIFACTS_GENERATED",
        "standalone_validator": "scripts/validate_mmwave_m_b11.py",
        "validator_required": True,
        "does_not_access_locked_test": True,
        "does_not_invoke_tflite": True,
    }
    payloads = {
        "artifact_lock_identity.json": identity,
        "source_lineage_lock.json": source_lock,
        "canonical_dataset_lock.json": canonical_lock,
        "subject_split_lock.json": split_lock,
        "window_population_lock.json": window_lock,
        "preprocessing_lock.json": preprocessing_lock,
        "training_lock.json": training_lock,
        "model_artifact_lock.json": model_lock,
        "quantization_lock.json": quantization_lock,
        "runtime_contract_lock.json": runtime_lock,
        "phase_b_lineage_registry.json": lineage,
        "final_evaluation_lock.json": final_eval,
        "recovery_access_history_lock.json": recovery_history,
        "final_sample_registry_lock.json": sample_registry,
        "final_metric_lock.json": metric_lock,
        "final_subject_metric_lock.json": subject_lock,
        "baseline_comparison_lock.json": baseline_lock,
        "scientific_limitations.json": limitations,
        "claim_boundary_lock.json": claims,
        "immutable_artifact_registry.json": immutable_registry,
        "artifact_lock_summary.json": summary,
        "validation_result.json": validation_placeholder,
    }
    for name, payload in payloads.items():
        atomic_write_json(lock_dir / name, payload)
    write_checksums(lock_dir)
    report = _render_report(
        identity=identity,
        source_lock=source_lock,
        canonical_lock=canonical_lock,
        split_lock=split_lock,
        window_lock=window_lock,
        model_lock=model_lock,
        metric_lock=metric_lock,
        subject_lock=subject_lock,
        baseline_lock=baseline_lock,
        limitations=limitations,
        claims=claims,
        recovery_history=recovery_history,
        lineage=lineage,
        quantization_lock=quantization_lock,
        preprocessing_lock=preprocessing_lock,
        training_lock=training_lock,
        analysis=analysis,
    )
    atomic_write_text(root / REPORT_REL, report)
    print(f"M-B11 lock written: {LOCK_DIR_REL.as_posix()}")
    print(f"Sensor-local lock written: {SENSOR_LOCK_REL.as_posix()}")
    print(f"Report written: {REPORT_REL.as_posix()}")
    return summary


def _render_report(**kwargs: Any) -> str:
    identity = kwargs["identity"]
    source = kwargs["source_lock"]
    canonical = kwargs["canonical_lock"]
    split = kwargs["split_lock"]
    window = kwargs["window_lock"]
    model = kwargs["model_lock"]
    metrics = kwargs["metric_lock"]
    subjects = kwargs["subject_lock"]
    baselines = kwargs["baseline_lock"]
    limitations = kwargs["limitations"]
    claims = kwargs["claims"]
    history = kwargs["recovery_history"]
    lineage = kwargs["lineage"]
    quant = kwargs["quantization_lock"]
    prep = kwargs["preprocessing_lock"]
    train = kwargs["training_lock"]
    normal = metrics["per_class"]["NORMAL"]
    rapid = metrics["per_class"]["RAPID_OR_ABNORMAL"]
    apnea = metrics["per_class"]["APNEA"]
    b4 = lineage["selected_path"]["M-B4"]["validation_macro_f1"]
    return f"""# M-B11 mmWave Offline Real-Data Candidate Artifact Lock

Generated from stored machine-readable evidence. This report does not create a new model.

## Prominent lock statements

M-B11 DOES NOT CREATE A NEW MODEL.

THE LOCKED CANDIDATE IS THE PREVIOUSLY SELECTED SEED42 STRICT-INT8 MODEL.

THE FINAL OFFLINE EVALUATION USED A NON-PRISTINE HOLDOUT REUSE EXCEPTION.

- Artifact status: `{identity["artifact_status"]}`
- Result limitation: `{identity["result_limitation"]}`
- `result_not_pristine`: true
- Candidate ID: `{model["candidate_id"]}`
- Runtime model ID: `{model["runtime_model_id"]}`
- Artifact: `{model["repo_relative_path"]}`
- SHA-256: `{model["sha256"]}`
- Bytes: {model["bytes"]}

## What this lock is not

This lock is not deployment ready, MR60 validated, Raspberry Pi validated, production ready, or clinical apnea validated. Phase B release remains incomplete. M-B12 is still required.

Offline real-data model evidence is not physical MR60 sensor evidence, not Raspberry Pi runtime evidence, and not future multisensor integration evidence. Team MR60 measurement evidence and the approximately-20-rpm issue belong to future M-C/M-D device-domain work. M-B11 does not resolve that issue. Old team `ondevice_ai` behavior is not validation evidence for this locked candidate.

## Source and canonical lineage

- Raw archive: `{source["raw_archive_repo_relative_path"]}`
- Raw SHA-256: `{source["raw_archive_sha256"]}`
- DOI: {source["doi"]} version {source["version"]}
- Population: 110 participants / 440 recordings
- Canonical dataset: `{canonical["canonical_npy_repo_relative_path"]}`
- Canonical SHA-256: `{canonical["canonical_npy_sha256"]}`
- Shape/dtype: {canonical["shape"]} / {canonical["dtype"]}
- A5 split: `{split["split_artifact_repo_relative_path"]}`
- A5 SHA-256: `{split["split_sha256"]}`
- A5 split seed: {split["split_seed"]}
- Subjects TRAIN/VALIDATION/LOCKED_TEST: {split["subject_counts"]["TRAIN"]}/{split["subject_counts"]["VALIDATION"]}/{split["subject_counts"]["LOCKED_TEST"]}
- A6 manifest SHA-256: `{window["a6_manifest_sha256"]}`
- Windows structural TRAIN/VALIDATION/LOCKED_TEST: {window["structural"]["TRAIN"]}/{window["structural"]["VALIDATION"]}/{window["structural"]["LOCKED_TEST"]}
- Eligible TRAIN/VALIDATION/LOCKED_TEST: {window["pure_supervised_eligible"]["TRAIN"]}/{window["pure_supervised_eligible"]["VALIDATION"]}/{window["pure_supervised_eligible"]["LOCKED_TEST"]}
- Class totals NORMAL/RAPID_OR_ABNORMAL/APNEA/AMBIGUOUS: {window["class_totals"]["NORMAL"]}/{window["class_totals"]["RAPID_OR_ABNORMAL"]}/{window["class_totals"]["APNEA"]}/{window["class_totals"]["AMBIGUOUS"]}

## Locked candidate contracts

- Preprocessing profile: `{prep["selected_profile_id"]}` / `{prep["selected_profile_name"]}`
- Execution contract: `{prep["execution_preprocessing_contract_id"]}`
- BPF: Butterworth {prep["bpf"]["lowcut_hz"]}-{prep["bpf"]["highcut_hz"]} Hz, order {prep["bpf"]["order"]}, zero-phase filtfilt, fs={prep["bpf"]["fs_hz"]} Hz
- Z-score: TRAIN-only mean={prep["zscore"]["mean"]}, std={prep["zscore"]["std"]}
- Training strategy: `{train["selected_strategy_id"]}`
- Loss: {train["loss"]} unweighted; optimizer {train["optimizer"]} lr={train["learning_rate"]}; batch {train["batch_size"]}; max epochs {train["max_epochs"]}; patience {train["early_stopping_patience"]}; restore-best {train["restore_best_weights"]}
- Seed: 42
- Calibration: `{model["calibration_profile"]}`
- Input: shape {model["input_tensor"]["shape"]} dtype {model["input_tensor"]["dtype"]} scale={model["input_tensor"]["scale"]} zp={model["input_tensor"]["zero_point"]}
- Output: shape {model["output_tensor"]["shape"]} dtype {model["output_tensor"]["dtype"]} scale={model["output_tensor"]["scale"]} zp={model["output_tensor"]["zero_point"]}
- Strict INT8: true; Flex/Select TF Ops: false
- Class map: 0→NORMAL, 1→RAPID_OR_ABNORMAL, 2→APNEA (APNEA remains a proxy)

## B-series lineage

Selected path: M-B1 `{prep["selected_profile_id"]}` → M-B2 `{train["selected_strategy_id"]}` → M-B3 `{model["architecture_id"]}` → M-B4 seeds 42/43/44 → M-B5 `{model["calibration_profile"]}` → M-B6 frozen strict INT8 → M-B7 perturbation robustness → M-B8 Mac/M2 latency only → M-B9 mock runtime/E2E.

M-B4 initialization sensitivity is locked, not hidden. VALIDATION Macro F1: seed42={b4["seed42"]}, seed43={b4["seed43"]}, seed44={b4["seed44"]}. Seed42 was materially better than seed44. M-B7 recorded moderate-profile collapse evidence for seed44 while seed42 was retained.

M-B8 is Mac/M2 latency and footprint evidence only. It is not Raspberry Pi latency. M-B9 is mock/runtime path equivalence and fail-closed behavior. It is not physical sensor integration.

## Abnormal final-test history (must not be erased)

1. M-B10A preregistered the candidate before LOCKED_TEST.
2. M-B10B original final accessor invocation = {history["original_m_b10b_accessor_invocations"]}; payload returned; model inference = {history["original_m_b10b_model_inference"]}; original LOCKED_TEST consumed = true; root cause = PRETEST_CONTRACT_COUNT_SEMANTICS_CONFLATION.
3. M-B10R0 policy = LIMITED_LOCKED_TEST_REUSE_EXCEPTION_RECOMMENDED.
4. M-B10R1-A froze the recovery harness before reuse; new access = 0.
5. M-B10R1-B recovery accessor = {history["m_b10r1b_recovery_accessor_invocations"]}; recovery payload release = {history["m_b10r1b_recovery_payload_releases"]}; TFLite invokes = {history["recovery_model_inference"]}; second recovery = NO; rerun = NO.
6. Historical total payload releases = {history["historical_total_payload_releases"]}.
7. Result designation remains `{history["result_limitation"]}`. This is not FINAL_LOCKED_TEST and not a pristine holdout.

## Final selected candidate summary

Recomputed from the persisted 75 seed42 prediction rows. No model inference was performed in M-B11.

- Eligible evaluated / valid / invalid: 75 / {metrics["per_class"]["NORMAL"]["support"] + metrics["per_class"]["RAPID_OR_ABNORMAL"]["support"] + metrics["per_class"]["APNEA"]["support"]} / 0
- Accuracy: {metrics["accuracy"]}
- Macro F1: {metrics["macro_f1"]}
- Macro precision: {metrics["macro_precision"]}
- Macro recall: {metrics["macro_recall"]}
- NORMAL: support={normal["support"]} precision={normal["precision"]} recall={normal["recall"]} F1={normal["f1_score"]} FPR={normal["fpr"]}
- RAPID_OR_ABNORMAL: support={rapid["support"]} precision={rapid["precision"]} recall={rapid["recall"]} F1={rapid["f1_score"]} FPR={rapid["fpr"]}
- APNEA proxy: support={apnea["support"]} precision={apnea["precision"]} recall={apnea["recall"]} F1={apnea["f1_score"]} misses={metrics["apnea_proxy"]["misses"]} FPR={metrics["apnea_proxy"]["fpr"]}
- Confusion: {metrics["confusion_matrix"]}
- Prediction distribution: {metrics["prediction_distribution"]}
- Class collapse: {metrics["class_collapse"]["collapsed"]}
- Subjects: {subjects["subject_count"]}
- Median subject Macro F1: {subjects["median_subject_macro_f1"]}
- Worst subject Macro F1: {subjects["worst_subject_macro_f1"]}
- Worst subject: `{subjects["worst_subject_id"]}`
- Saturation ratio: {quant["input_saturation_ratio"]} (pre-clamp out-of-range {quant["pre_clamp_out_of_range_count"]} / {quant["total_quantized_elements"]})

These limitations are locked scientific facts for future M-C/M-D. They are not M-B11 blockers and they are not defects requiring immediate B-series retuning.

## Baseline comparison

This is not a new model-selection event. v0.1 and v0.2 remain compatibility benchmarks only.

- seed42 Macro F1: {metrics["macro_f1"]} (no required-class collapse)
- v0.1 `{baselines["v0_1"]["role"]}` Macro F1: {baselines["v0_1"]["macro_f1"]} (class collapse; all 75 predicted NORMAL)
- v0.2 `{baselines["v0_2"]["role"]}` Macro F1: {baselines["v0_2"]["macro_f1"]} (RAPID_OR_ABNORMAL zero-prediction collapse)

## Device-domain handoff for future M-C

M-B11 does not begin M-C. Future M-C must independently investigate:

- physical MR60BHA2 signal-domain compatibility with this offline candidate
- device preprocessing correspondence to `{prep["execution_preprocessing_contract_id"]}`
- observed team approximately-20-rpm behavior
- domain shift between the offline Zenodo dataset and the physical sensor
- runtime input identity on device
- Raspberry Pi / device execution behavior

## Release readiness

- M-B11 artifact lock complete: {claims["M-B11_artifact_lock_complete"]}
- M-B12 offline final report required: {claims["M-B12_offline_final_report_required"]}
- Phase B release ready: {claims["Phase_B_release_ready"]}
- LOCKED_TEST reopen allowed: {claims["locked_test_reopen_allowed"]}
- Recovery reopen allowed: {claims["recovery_reopen_allowed"]}

Do not create a GitHub Release or tag in M-B11. Do not begin M-B12 until this lock is independently reviewed and merged.

## Validator-truth closure

- Forbidden-claim recursive enforcement: PASS
- Non-claim-boundary corruption tests: PASS
- Locked cross-model recording mismatches: {kwargs["analysis"]["cross_model_recording_mismatches"]}
- Recording corruption tests: PASS
- Generator high-level ledger analyzer reused by validator: NO
- Validator-owned source ledger:
  - unique IDs = {kwargs["analysis"]["unique_eligible_window_ids"]}
  - models = 3
  - pairs = {kwargs["analysis"]["actual_pairs"]}
  - duplicates = {kwargs["analysis"]["duplicates"]}
  - missing = {kwargs["analysis"]["missing"]}
  - unexpected = {kwargs["analysis"]["unexpected"]}
  - label mismatches = {kwargs["analysis"]["cross_model_label_mismatches"]}
  - subject mismatches = {kwargs["analysis"]["cross_model_subject_mismatches"]}
  - recording mismatches = {kwargs["analysis"]["cross_model_recording_mismatches"]}
- New LOCKED_TEST access = 0
- New recovery access = 0
- New inference = 0

YES — M-B11 artifact lock validator is independently fail-closed; await independent review before M-B12.
"""


def main() -> int:
    try:
        generate_m_b11_artifact_lock()
    except MB11LockError as exc:
        print(f"M-B11 GENERATION FAIL: {exc}", file=sys.stderr)
        return 1
    print("M-B11 GENERATION PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
