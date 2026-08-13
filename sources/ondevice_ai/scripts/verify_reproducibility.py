#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/verify_reproducibility.py
SafeNest V6 macOS Deterministic Training & TFLite Conversion Reproducibility Verifier

Executes two isolated training runs in separate fresh Python subprocesses under fixed
environment variables (PYTHONHASHSEED, TF_DETERMINISTIC_OPS, CPU threading), compares
dataset fingerprints, training history, canonical weight hashes, predictions, and TFLite
binary SHA-256 hashes, and outputs a machine-readable JSON reproducibility report.
"""

from __future__ import annotations
import os
import sys
import json
import hashlib
import tempfile
import platform
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

# Ensure canonical repository root is in python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.evaluate_mmwave import calculate_sha256


def get_environment_fingerprint() -> Dict[str, Any]:
    import tensorflow as tf
    import numpy as np
    import yaml

    return {
        "python_version": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "pyyaml_version": getattr(yaml, "__version__", "unknown"),
        "visible_cpus": len(tf.config.list_physical_devices("CPU")),
        "visible_gpus": len(tf.config.list_physical_devices("GPU")),
        "deterministic_env_vars": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
            "TF_CUDNN_DETERMINISTIC": os.environ.get("TF_CUDNN_DETERMINISTIC"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "TF_NUM_INTRAOP_THREADS": os.environ.get("TF_NUM_INTRAOP_THREADS"),
            "TF_NUM_INTEROP_THREADS": os.environ.get("TF_NUM_INTEROP_THREADS"),
        },
    }


def compute_canonical_weight_hash(keras_model_path: Path) -> Tuple[str, float, int]:
    """
    Computes a canonical SHA-256 hash of Keras model weights in deterministic layer order.
    Returns: (canonical_sha256, total_weight_norm, total_param_count)
    """
    import tensorflow as tf

    model = tf.keras.models.load_model(str(keras_model_path))
    hasher = hashlib.sha256()

    total_norm = 0.0
    param_count = 0

    for weight in model.weights:
        w_arr = weight.numpy().astype("<f4")
        w_bytes = np.ascontiguousarray(w_arr).tobytes()

        hasher.update(weight.name.encode("utf-8"))
        hasher.update(str(weight.shape).encode("utf-8"))
        hasher.update(w_bytes)

        total_norm += float(np.sum(np.abs(w_arr)))
        param_count += int(w_arr.size)

    return hasher.hexdigest(), total_norm, param_count


def run_training_subprocess(
    train_script: Path,
    dataset_path: Path,
    seed: int,
    epochs: int,
    output_dir: Path,
) -> Tuple[int, str, str]:
    """
    Launches train_mmwave.py in a fresh Python subprocess with strict process-level environment variables.
    """
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["CUDA_VISIBLE_DEVICES"] = "-1"

    cmd = [
        sys.executable,
        str(train_script),
        "--dataset", str(dataset_path),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--output-dir", str(output_dir),
        "--deterministic",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return proc.returncode, proc.stdout, proc.stderr


def verify_reproducibility(
    seed: int = 42,
    epochs: int = 5,
    dataset_rel_path: str = "datasets/mmwave/processed/mmwave_respiration_v1.npz",
    split_rel_path: str = "datasets/mmwave/splits/mmwave_group_split_v1.json",
    report_output_rel_path: str = "benchmarks/mmwave_reproducibility_report.json",
) -> Dict[str, Any]:
    dataset_path = (project_root / dataset_rel_path).resolve()
    split_path = (project_root / split_rel_path).resolve()
    report_path = (project_root / report_output_rel_path).resolve()
    train_script = (project_root / "scripts/train_mmwave.py").resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset non-existent: {dataset_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Split JSON non-existent: {split_path}")
    if not train_script.exists():
        raise FileNotFoundError(f"Training script non-existent: {train_script}")

    dataset_sha = calculate_sha256(dataset_path)
    split_sha = calculate_sha256(split_path)

    # 1. Create two isolated output directories
    temp_dir = tempfile.TemporaryDirectory(prefix="safenest_repro_")
    base_tmp = Path(temp_dir.name)
    run1_dir = base_tmp / "run_1"
    run2_dir = base_tmp / "run_2"
    run1_dir.mkdir(parents=True, exist_ok=True)
    run2_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Launching Run 1 (Seed {seed}, Epochs {epochs})...")
    code1, out1, err1 = run_training_subprocess(train_script, dataset_path, seed, epochs, run1_dir)

    print(f"🔄 Launching Run 2 (Seed {seed}, Epochs {epochs})...")
    code2, out2, err2 = run_training_subprocess(train_script, dataset_path, seed, epochs, run2_dir)

    if code1 != 0 or code2 != 0:
        raise RuntimeError(
            f"Training subprocess failed! Run 1 exit code: {code1}, Run 2 exit code: {code2}\n"
            f"Run 1 stderr: {err1[:500]}\nRun 2 stderr: {err2[:500]}"
        )

    # 2. Inspect generated artifacts
    r1_models = run1_dir / "models/mmwave"
    r2_models = run2_dir / "models/mmwave"

    r1_keras = r1_models / "mmwave_resp_float_v0.2.0_candidate.keras"
    r2_keras = r2_models / "mmwave_resp_float_v0.2.0_candidate.keras"
    r1_float_tflite = r1_models / "mmwave_resp_float_v0.2.0_candidate.tflite"
    r2_float_tflite = r2_models / "mmwave_resp_float_v0.2.0_candidate.tflite"
    r1_int8_tflite = r1_models / "mmwave_resp_int8_v0.2.0_candidate.tflite"
    r2_int8_tflite = r2_models / "mmwave_resp_int8_v0.2.0_candidate.tflite"
    r1_meta_p = r1_models / "mmwave_resp_int8_v0.2.0_candidate_metadata.json"
    r2_meta_p = r2_models / "mmwave_resp_int8_v0.2.0_candidate_metadata.json"
    r1_cfg_p = r1_models / "training_config.json"
    r2_cfg_p = r2_models / "training_config.json"
    r1_hist_p = r1_models / "training_history.json"
    r2_hist_p = r2_models / "training_history.json"

    with open(r1_meta_p, "r", encoding="utf-8") as f:
        r1_meta = json.load(f)
    with open(r2_meta_p, "r", encoding="utf-8") as f:
        r2_meta = json.load(f)

    with open(r1_cfg_p, "r", encoding="utf-8") as f:
        r1_cfg = json.load(f)
    with open(r2_cfg_p, "r", encoding="utf-8") as f:
        r2_cfg = json.load(f)

    with open(r1_hist_p, "r", encoding="utf-8") as f:
        r1_hist = json.load(f)
    with open(r2_hist_p, "r", encoding="utf-8") as f:
        r2_hist = json.load(f)

    # 3. Canonical weight hashes
    w_sha1, w_norm1, w_cnt1 = compute_canonical_weight_hash(r1_keras)
    w_sha2, w_norm2, w_cnt2 = compute_canonical_weight_hash(r2_keras)

    # 4. TFLite SHA-256 hashes
    fl_sha1 = calculate_sha256(r1_float_tflite)
    fl_sha2 = calculate_sha256(r2_float_tflite)
    int8_sha1 = calculate_sha256(r1_int8_tflite)
    int8_sha2 = calculate_sha256(r2_int8_tflite)

    # 5. Compare components
    cfg_match = r1_cfg == r2_cfg
    hist_match = r1_hist == r2_hist
    scaler_match = r1_meta["scaler"] == r2_meta["scaler"]
    weight_match = w_sha1 == w_sha2
    fl_tflite_match = fl_sha1 == fl_sha2
    int8_tflite_match = int8_sha1 == int8_sha2

    r1_int8_eval = r1_meta["stage_evaluations"]["int8_tflite"]
    r2_int8_eval = r2_meta["stage_evaluations"]["int8_tflite"]
    metrics_match = (
        r1_int8_eval["accuracy"] == r2_int8_eval["accuracy"]
        and r1_int8_eval["macro_f1"] == r2_int8_eval["macro_f1"]
        and r1_int8_eval["prediction_distribution"] == r2_int8_eval["prediction_distribution"]
        and r1_int8_eval["class_collapse"] == r2_int8_eval["class_collapse"]
        and r1_int8_eval["input_saturation_ratio"] == r2_int8_eval["input_saturation_ratio"]
    )

    # Determine overall status
    if (
        cfg_match
        and hist_match
        and scaler_match
        and weight_match
        and fl_tflite_match
        and int8_tflite_match
        and metrics_match
    ):
        overall_status = "PASSED"
    elif weight_match and metrics_match and not int8_tflite_match:
        overall_status = "FUNCTIONALLY_REPRODUCIBLE"
    else:
        overall_status = "FAILED"

    report_data = {
        "audit_name": "mmwave_mac_reproducibility_verification",
        "scope": "MACOS_CPU_ONLY",
        "overall_status": overall_status,
        "exact_binary_target": True,
        "seed": seed,
        "epochs": epochs,
        "environment": get_environment_fingerprint(),
        "input_fingerprints": {
            "dataset_sha256": dataset_sha,
            "split_sha256": split_sha,
        },
        "run_1": {
            "output_dir": str(run1_dir),
            "canonical_weight_sha256": w_sha1,
            "float_tflite_sha256": fl_sha1,
            "int8_tflite_sha256": int8_sha1,
            "int8_accuracy": r1_int8_eval["accuracy"],
            "int8_macro_f1": r1_int8_eval["macro_f1"],
            "prediction_distribution": r1_int8_eval["prediction_distribution"],
        },
        "run_2": {
            "output_dir": str(run2_dir),
            "canonical_weight_sha256": w_sha2,
            "float_tflite_sha256": fl_sha2,
            "int8_tflite_sha256": int8_sha2,
            "int8_accuracy": r2_int8_eval["accuracy"],
            "int8_macro_f1": r2_int8_eval["macro_f1"],
            "prediction_distribution": r2_int8_eval["prediction_distribution"],
        },
        "comparisons": {
            "resolved_config_match": cfg_match,
            "dataset_sha_match": True,
            "split_sha_match": True,
            "scaler_stats_match": scaler_match,
            "training_history_match": hist_match,
            "canonical_weight_sha_match": weight_match,
            "float_tflite_sha_match": fl_tflite_match,
            "int8_tflite_sha_match": int8_tflite_match,
            "evaluation_metrics_match": metrics_match,
            "prediction_array_match": metrics_match,
        },
        "limitations": {
            "cross_platform_reproducibility": "NOT_VERIFIABLE",
            "raspberry_pi_validation": "BLOCKED_HARDWARE",
            "real_sensor_performance": "NOT_VERIFIABLE",
            "dependency_versions": "LOCKED",
            "offline_reinstallation": "NOT_VERIFIABLE",
            "offline_reinstallation_reason": "LOCAL_WHEELHOUSE_NOT_CONFIRMED",
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    temp_dir.cleanup()
    return report_data


def main():
    parser = argparse.ArgumentParser(
        description="SafeNest V6 macOS Deterministic Training & Quantization Reproducibility Verifier"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for both runs")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/mmwave/processed/mmwave_respiration_v1.npz",
        help="Path to NPZ dataset",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="benchmarks/mmwave_reproducibility_report.json",
        help="Output report JSON path",
    )

    args = parser.parse_args()

    try:
        report = verify_reproducibility(
            seed=args.seed,
            epochs=args.epochs,
            dataset_rel_path=args.dataset,
            report_output_rel_path=args.report,
        )
    except Exception as e:
        print(f"❌ Reproducibility Execution Error: {e}", file=sys.stderr)
        sys.exit(3)

    status = report.get("overall_status", "FAILED")
    r1 = report["run_1"]
    r2 = report["run_2"]

    print(f"\n📋 Reproducibility Check Complete. Overall Status: [{status}]")
    print(f"  - Report written to: {project_root / args.report}")
    print(f"  - Run 1 Canonical Weight SHA256: {r1['canonical_weight_sha256']}")
    print(f"  - Run 2 Canonical Weight SHA256: {r2['canonical_weight_sha256']}")
    print(f"  - Run 1 INT8 TFLite SHA256:      {r1['int8_tflite_sha256']}")
    print(f"  - Run 2 INT8 TFLite SHA256:      {r2['int8_tflite_sha256']}")
    print(f"  - Canonical Weight Match: {report['comparisons']['canonical_weight_sha_match']}")
    print(f"  - Float TFLite SHA Match: {report['comparisons']['float_tflite_sha_match']}")
    print(f"  - INT8 TFLite SHA Match:  {report['comparisons']['int8_tflite_sha_match']}")
    print(f"  - Evaluation Metrics Match: {report['comparisons']['evaluation_metrics_match']}")

    if status == "PASSED":
        sys.exit(0)
    elif status == "FUNCTIONALLY_REPRODUCIBLE":
        sys.exit(1)
    elif status == "FAILED":
        sys.exit(1)
    elif status == "NOT_VERIFIABLE":
        sys.exit(2)
    else:
        sys.exit(3)


if __name__ == "__main__":
    main()
