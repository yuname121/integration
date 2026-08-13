#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/audit_co2_offline_experiment.py
Phase C-B0 — generate offline experiment contract and harness evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

repo_root_dir = Path(__file__).resolve().parent.parent
if str(repo_root_dir) not in sys.path:
    sys.path.insert(0, str(repo_root_dir))

from datasets.co2.offline_experiment import (
    DEFAULT_SEED,
    EXPERIMENT_CONTRACT_ID,
    MANIFEST_DIR_REL,
    PHASE_ID,
    MajorityClassBaseline,
    TrainOnlyStandardScaler,
    build_a_series_consumption_registry,
    build_environment_metadata,
    build_exceptions_registry,
    build_experiment_contract,
    build_feature_view_registry,
    build_metric_contract,
    build_sample_universe_manifest,
    compute_classification_metrics,
    load_comparison_matrix,
    ordered_id_list_sha256,
    run_leakage_audit,
)
from datasets.co2.raw_reader import compute_sha256_file, get_repo_root

C_B0_HASHED = [
    "experiment_contract.json",
    "a_series_consumption_registry.json",
    "sample_universe_manifest.json",
    "feature_view_registry.json",
    "metric_contract.json",
    "leakage_audit.json",
    "preprocessing_fit_evidence.json",
    "reference_baseline_result.json",
    "generation_metadata.json",
    "exceptions_and_limitations.json",
    "run_environment.json",
]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def audit_co2_offline_experiment() -> Path:
    np.random.seed(DEFAULT_SEED)
    repo_root = get_repo_root()
    out_dir = repo_root / MANIFEST_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    consumption = build_a_series_consumption_registry(repo_root)
    universe = build_sample_universe_manifest(repo_root)
    features = build_feature_view_registry()
    metrics = build_metric_contract()
    contract = build_experiment_contract()
    env = build_environment_metadata()

    # Harness validation uses HISTORICAL_COMPATIBILITY_REFERENCE feature order
    view_id = "HISTORICAL_COMPATIBILITY_REFERENCE"
    feature_names = features["feature_views"][view_id]["features"]
    train = load_comparison_matrix(
        repo_root=repo_root, split_role="TRAIN", feature_names=feature_names
    )
    val = load_comparison_matrix(
        repo_root=repo_root, split_role="VALIDATION", feature_names=feature_names
    )

    scaler = TrainOnlyStandardScaler(feature_names=tuple(feature_names)).fit(train)
    x_train = scaler.transform(train)
    x_val = scaler.transform(val)
    # Matrices are used to prove finite preprocess path; majority baseline ignores features.
    if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():
        raise RuntimeError("Non-finite preprocessed features")

    baseline = MajorityClassBaseline().fit(train.labels)
    y_pred = baseline.predict(len(val.labels))
    metric_values = compute_classification_metrics(val.labels, y_pred)

    preprocess_meta = scaler.to_metadata(
        fit_population_fingerprint=universe["ordered_id_list_sha256"]["TRAIN"],
        feature_view_id=view_id,
    )
    baseline_result = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "baseline_id": "MAJORITY_CLASS_BASELINE",
        "status": "REFERENCE_BASELINE_ONLY",
        "candidate": False,
        "deployable": False,
        "train_population_count": len(train.sample_ids),
        "train_population_fingerprint": universe["ordered_id_list_sha256"]["TRAIN"],
        "evaluation_population": "VALIDATION",
        "evaluation_population_count": len(val.sample_ids),
        "evaluation_population_fingerprint": universe["ordered_id_list_sha256"]["VALIDATION"],
        "majority_label": int(baseline.majority_label),
        "majority_class_name": "OCCUPIED" if baseline.majority_label == 1 else "VACANT",
        "feature_view_id": view_id,
        "uses_features": False,
        "metrics": metric_values,
        "locked_test_used": False,
        "threshold_optimization_performed": False,
        "complex_model_comparison_performed": False,
        "prediction_id_list_sha256": ordered_id_list_sha256(
            [f"{sid}:{int(pred)}" for sid, pred in zip(val.sample_ids, y_pred.tolist())]
        ),
    }

    leakage = run_leakage_audit(universe, features)
    if leakage["status"] != "PASS":
        raise RuntimeError(f"Leakage audit failed: {leakage['errors']}")

    exceptions = build_exceptions_registry(sklearn_skipped=not env["sklearn_available"])
    generation = {
        "manifest_version": "1.0",
        "phase": PHASE_ID,
        "experiment_contract_id": EXPERIMENT_CONTRACT_ID,
        "generator_script": "scripts/audit_co2_offline_experiment.py",
        "module": "datasets/co2/offline_experiment.py",
        "seed": DEFAULT_SEED,
        "production_scaler_modified": False,
        "production_model_modified": False,
        "synthetic_npz_used_as_real_training_data": False,
        "final_feature_selection_performed": False,
        "slope_ablation_performed": False,
        "complex_model_comparison_performed": False,
        "locked_test_fit_usage": 0,
        "locked_test_tuning_usage": 0,
        "optional_linear_baseline": env["optional_linear_baseline_status"],
        "determinism": {
            "data_pipeline_determinism_required": True,
            "random_values_used_for_manifests": False,
            "host_timezone_independent": True,
        },
    }

    _write_json(out_dir / "experiment_contract.json", contract)
    _write_json(out_dir / "a_series_consumption_registry.json", consumption)
    _write_json(out_dir / "sample_universe_manifest.json", universe)
    _write_json(out_dir / "feature_view_registry.json", features)
    _write_json(out_dir / "metric_contract.json", metrics)
    _write_json(out_dir / "leakage_audit.json", leakage)
    _write_json(out_dir / "preprocessing_fit_evidence.json", preprocess_meta)
    _write_json(out_dir / "reference_baseline_result.json", baseline_result)
    _write_json(out_dir / "generation_metadata.json", generation)
    _write_json(out_dir / "exceptions_and_limitations.json", exceptions)
    _write_json(out_dir / "run_environment.json", env)

    lines = []
    for fname in C_B0_HASHED:
        path = out_dir / fname
        rel = f"{MANIFEST_DIR_REL}/{fname}"
        lines.append(f"{compute_sha256_file(path)}  {rel}")
    (out_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"✅ Generated C-B0 offline experiment contract in: {MANIFEST_DIR_REL}")
    print(
        f"   train={universe['b_series_common_train']} "
        f"val={universe['b_series_common_validation']} "
        f"locked={universe['b_series_sealed_locked_test']} "
        f"warmup={universe['canonical_warmup_records']}"
    )
    print(
        f"   majority_label={baseline_result['majority_class_name']} "
        f"macro_f1={metric_values['macro_f1']:.6f}"
    )
    return out_dir


if __name__ == "__main__":
    audit_co2_offline_experiment()
