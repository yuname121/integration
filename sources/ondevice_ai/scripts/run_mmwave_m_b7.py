#!/usr/bin/env python3
"""Generate SafeNest mmWave M-B7 deterministic perturbation evidence."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from mmwave_m_b1_preprocessing import (
    compute_tensor_fingerprint,
    fit_train_zscore_statistics,
    transform_signals,
)
from mmwave_m_b2_imbalance import LABEL_NAMES
from mmwave_m_b7_perturbation import (
    ALL_PROFILE_ORDER,
    CLEAN_PROFILE_ID,
    FROZEN_SEEDS,
    GLOBAL_PERTURBATION_SEED,
    PERTURBATION_PROFILE_ORDER,
    PROFILE_DEFINITIONS,
    StrictInt8Runner,
    aggregate_cross_seed,
    compute_run_metrics,
    generate_profile_sample,
    perturbation_profile_contract,
    subject_level_metrics,
)
from mmwave_phase_b_access import PhaseBAccessGuard


MANIFEST_RELATIVE = Path("datasets/mmwave/manifests/M-B7_perturbation_robustness")
REPORT_RELATIVE = Path("docs/reports/20260811_Codex_M-B7_Perturbation_Robustness_01.md")
ARCHITECTURE_ID = "M-B3_CONV1D_GAP_BASELINE"
CALIBRATION_PROFILE_ID = "M-B5_CAL_CLASS_BALANCED_120"


REQUIRED_OUTPUT_FILENAMES = (
    "input_identity.json",
    "experiment_contract.json",
    "perturbation_profile_contract.json",
    "clean_baseline_results.json",
    "perturbation_runs.json",
    "perturbation_results.json",
    "cross_seed_robustness_summary.json",
    "subject_level_robustness.json",
    "prediction_changes.jsonl",
    "perturbation_sample_index.jsonl",
    "perturbation_fidelity_audit.json",
    "preprocessing_attenuation_audit.json",
    "quantization_diagnostics.json",
    "fallback_recommendations.json",
    "locked_test_access_audit.json",
    "run_environment.json",
    "exceptions.json",
    "determinism_audit.json",
    "prediction_vectors.npz",
    "m_b7_summary.json",
)


INPUT_IDENTITY_PATHS: Tuple[Tuple[str, str], ...] = (
    ("requirements-mac.txt", "Pinned macOS dependency contract"),
    ("datasets/mmwave/manifests/M-B0_evaluation_protocol/m_b0_summary.json", "M-B0 validator-passed summary"),
    ("datasets/mmwave/manifests/M-B0_evaluation_protocol/evaluation_contract.json", "M-B0 evaluation contract"),
    ("datasets/mmwave/manifests/M-B0_evaluation_protocol/checksums.sha256", "M-B0 checksum closure"),
    ("datasets/mmwave/manifests/M-B1_preprocessing_ablation/m_b1_summary.json", "M-B1 validator-passed summary"),
    ("datasets/mmwave/manifests/M-B1_preprocessing_ablation/selected_preprocessing_profile.json", "Frozen M-B1 preprocessing"),
    ("datasets/mmwave/manifests/M-B1_preprocessing_ablation/train_fit_statistics.json", "Frozen TRAIN-fit Z-score statistics"),
    ("datasets/mmwave/manifests/M-B1_preprocessing_ablation/preprocessing_fingerprints.json", "Frozen preprocessing tensor fingerprints"),
    ("datasets/mmwave/manifests/M-B1_preprocessing_ablation/checksums.sha256", "M-B1 checksum closure"),
    ("datasets/mmwave/manifests/M-B2_class_imbalance/m_b2_summary.json", "M-B2 validator-passed summary"),
    ("datasets/mmwave/manifests/M-B2_class_imbalance/selected_imbalance_strategy.json", "Frozen M-B2 imbalance strategy"),
    ("datasets/mmwave/manifests/M-B2_class_imbalance/checksums.sha256", "M-B2 checksum closure"),
    ("datasets/mmwave/manifests/M-B3_architecture_comparison/m_b3_summary.json", "M-B3 validator-passed summary"),
    ("datasets/mmwave/manifests/M-B3_architecture_comparison/architecture_profiles.json", "Frozen architecture profiles"),
    ("datasets/mmwave/manifests/M-B3_architecture_comparison/selected_architecture_shortlist.json", "Frozen M-B3 shortlist"),
    ("datasets/mmwave/manifests/M-B3_architecture_comparison/checksums.sha256", "M-B3 checksum closure"),
    ("datasets/mmwave/manifests/M-B4_multiseed_stability/m_b4_summary.json", "M-B4 validator-passed summary"),
    ("datasets/mmwave/manifests/M-B4_multiseed_stability/primary_float_finalist.json", "Frozen M-B4 primary finalist"),
    ("datasets/mmwave/manifests/M-B4_multiseed_stability/seed_weights.npz", "Frozen M-B4 three-seed weights"),
    ("datasets/mmwave/manifests/M-B4_multiseed_stability/per_seed_results.json", "Frozen M-B4 per-seed evidence"),
    ("datasets/mmwave/manifests/M-B4_multiseed_stability/checksums.sha256", "M-B4 checksum closure"),
    ("datasets/mmwave/manifests/M-B5_representative_calibration/m_b5_summary.json", "M-B5 validator-passed summary"),
    ("datasets/mmwave/manifests/M-B5_representative_calibration/selected_calibration_profile.json", "Frozen M-B5 calibration selection"),
    ("datasets/mmwave/manifests/M-B5_representative_calibration/representative_dataset_indices.json", "Frozen representative indices"),
    ("datasets/mmwave/manifests/M-B5_representative_calibration/tflite_artifact_manifest.json", "M-B5 strict-INT8 identities"),
    ("datasets/mmwave/manifests/M-B5_representative_calibration/checksums.sha256", "M-B5 checksum closure"),
    ("datasets/mmwave/manifests/M-B6_stage_equivalence/m_b6_summary.json", "M-B6 validator-passed summary"),
    ("datasets/mmwave/manifests/M-B6_stage_equivalence/experiment_contract.json", "M-B6 experiment contract"),
    ("datasets/mmwave/manifests/M-B6_stage_equivalence/stage_artifact_manifest.json", "M-B6 stage artifact identities"),
    ("datasets/mmwave/manifests/M-B6_stage_equivalence/per_seed_stage_metrics.json", "M-B6 per-seed Stage-C metrics"),
    ("datasets/mmwave/manifests/M-B6_stage_equivalence/int8_tflite_predictions.npz", "M-B6 clean strict-INT8 top-1 vectors"),
    ("datasets/mmwave/manifests/M-B6_stage_equivalence/quantization_diagnostics.json", "M-B6 clean quantization diagnostics"),
    ("datasets/mmwave/manifests/M-B6_stage_equivalence/checksums.sha256", "M-B6 checksum closure"),
    ("datasets/mmwave/manifests/a5_subject_split/subject_split_manifest.jsonl", "Immutable A5 subject assignment"),
    ("datasets/mmwave/splits/mmwave_real_subject_split_v1.json", "Immutable A5 split lookup"),
    ("datasets/mmwave/processed/mmwave_canonical_real_v1.npy", "A6 canonical numeric matrix"),
    ("datasets/mmwave/manifests/a6_full_conversion/full_window_manifest.jsonl", "A6 canonical window manifest"),
    ("datasets/mmwave/manifests/a6_full_conversion/full_provenance_manifest.jsonl", "A6 provenance manifest"),
    ("datasets/mmwave/manifests/a6_full_conversion/checksums.sha256", "A6 checksum closure"),
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def write_deterministic_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    """Write stable NPZ bytes (fixed member order, timestamp, permissions)."""
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def array_key(seed: int, profile_id: str, field: str) -> str:
    return f"seed_{seed}__{profile_id}__{field}"


def build_input_identity(root_dir: Path, model_paths: Dict[int, str]) -> Dict[str, Any]:
    inputs: List[Dict[str, Any]] = []
    for relative, role in INPUT_IDENTITY_PATHS:
        path = root_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required M-B7 input missing: {relative}")
        inputs.append(
            {
                "repository_relative_path": relative,
                "measured_sha256": file_sha256(path),
                "evidence_role": role,
            }
        )
    for seed in FROZEN_SEEDS:
        relative = model_paths[seed]
        path = root_dir / relative
        inputs.append(
            {
                "repository_relative_path": relative,
                "measured_sha256": file_sha256(path),
                "evidence_role": f"Frozen M-B6-qualified strict-INT8 artifact for seed {seed}",
            }
        )
    return {"phase_id": "M-B7", "total_inputs": len(inputs), "inputs": inputs}


def load_frozen_contracts(root_dir: Path) -> Dict[str, Any]:
    def load(relative: str) -> Any:
        return json.loads((root_dir / relative).read_text(encoding="utf-8"))

    selected_preprocessing = load(
        "datasets/mmwave/manifests/M-B1_preprocessing_ablation/selected_preprocessing_profile.json"
    )
    selected_imbalance = load(
        "datasets/mmwave/manifests/M-B2_class_imbalance/selected_imbalance_strategy.json"
    )
    primary = load("datasets/mmwave/manifests/M-B4_multiseed_stability/primary_float_finalist.json")
    selected_calibration = load(
        "datasets/mmwave/manifests/M-B5_representative_calibration/selected_calibration_profile.json"
    )
    mb6_summary = load("datasets/mmwave/manifests/M-B6_stage_equivalence/m_b6_summary.json")
    if selected_preprocessing.get("selected_profile_name") != "BPF_ZSCORE":
        raise RuntimeError("Frozen M-B1 preprocessing is not BPF_ZSCORE")
    if selected_imbalance.get("selected_strategy_name") != "CE_UNWEIGHTED":
        raise RuntimeError("Frozen M-B2 imbalance strategy is not CE_UNWEIGHTED")
    if primary.get("primary_stable_float_finalist") != ARCHITECTURE_ID:
        raise RuntimeError("Frozen M-B4 primary architecture mismatch")
    if selected_calibration.get("selected_calibration_profile") != CALIBRATION_PROFILE_ID:
        raise RuntimeError("Frozen M-B5 calibration profile mismatch")
    if mb6_summary.get("frozen_weight_seeds") != list(FROZEN_SEEDS):
        raise RuntimeError("Frozen M-B6 seed set mismatch")
    return {
        "preprocessing": selected_preprocessing,
        "imbalance": selected_imbalance,
        "primary": primary,
        "calibration": selected_calibration,
        "m_b6_summary": mb6_summary,
    }


def load_model_runners(root_dir: Path) -> Tuple[Dict[int, StrictInt8Runner], Dict[int, str], Dict[str, Any]]:
    stage_manifest_path = root_dir / "datasets/mmwave/manifests/M-B6_stage_equivalence/stage_artifact_manifest.json"
    stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))["artifacts"]
    runners: Dict[int, StrictInt8Runner] = {}
    model_paths: Dict[int, str] = {}
    structures: Dict[str, Any] = {}
    for seed in FROZEN_SEEDS:
        key = f"{ARCHITECTURE_ID}_seed_{seed}_stage_c"
        metadata = stage_manifest[key]
        relative = metadata["relative_path"]
        runner = StrictInt8Runner(root_dir / relative)
        structure = runner.structure()
        structure["relative_path"] = relative
        if structure["sha256"] != metadata["sha256"] or structure["bytes"] != metadata["bytes"]:
            raise RuntimeError(f"M-B7_STRICT_INT8_ARTIFACT_IDENTITY_MISMATCH seed {seed}")
        if (
            structure["input_dtype"] != "int8"
            or structure["output_dtype"] != "int8"
            or structure["select_tf_ops_count"] != 0
        ):
            raise RuntimeError(f"M-B7_STRICT_INT8_RUNTIME_CONTRACT_MISMATCH seed {seed}")
        runners[seed] = runner
        model_paths[seed] = relative
        structures[str(seed)] = structure
    return runners, model_paths, structures


def _stats(values: Sequence[float], digits: int = 9) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": round(float(np.mean(arr)), digits),
        "median": round(float(np.median(arr)), digits),
        "min": round(float(np.min(arr)), digits),
        "max": round(float(np.max(arr)), digits),
    }


def build_profile_audits(
    sample_rows: Sequence[Dict[str, Any]],
    determinism_details: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    fidelity: Dict[str, Any] = {}
    attenuation: Dict[str, Any] = {}
    fallback: Dict[str, Any] = {}
    for profile_id in PERTURBATION_PROFILE_ORDER:
        rows = [row for row in sample_rows if row["profile_id"] == profile_id]
        valid_rows = [
            row
            for row in rows
            if row["validity_status"] == "INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION"
        ]
        family = PROFILE_DEFINITIONS[profile_id]["family"]
        entry: Dict[str, Any] = {
            "family": family,
            "sample_count": len(rows),
            "valid_sample_count": len(valid_rows),
            "deterministic_regeneration_verified": determinism_details[profile_id][
                "all_samples_regenerated_identically"
            ],
        }
        if family == "GAUSSIAN_NOISE":
            entry["target_snr_db"] = PROFILE_DEFINITIONS[profile_id]["target_snr_db"]
            entry["achieved_snr_db"] = _stats(
                [float(row["parameters"]["achieved_snr_db"]) for row in valid_rows], 9
            )
            entry["noise_fingerprint_count"] = len(
                {row["parameters"]["noise_sha256_float64"] for row in valid_rows}
            )
        elif family == "AMPLITUDE_SCALING":
            entry["scale"] = PROFILE_DEFINITIONS[profile_id]["scale"]
            entry["maximum_formula_error"] = max(
                float(row["parameters"]["max_abs_formula_error"]) for row in valid_rows
            )
        elif family == "BASELINE_DRIFT":
            entry["frequency_hz"] = PROFILE_DEFINITIONS[profile_id]["frequency_hz"]
            entry["amplitude_multiplier"] = PROFILE_DEFINITIONS[profile_id][
                "amplitude_rms_multiplier"
            ]
            entry["drift_pre_b1_rms"] = _stats(
                [float(row["parameters"]["drift_rms"]) for row in valid_rows], 9
            )
        elif family == "CONTIGUOUS_DROPOUT":
            entry["duration_samples"] = PROFILE_DEFINITIONS[profile_id]["duration_samples"]
            entry["all_masks_have_exact_duration"] = all(
                sum(row["parameters"]["dropout_mask"])
                == PROFILE_DEFINITIONS[profile_id]["duration_samples"]
                for row in valid_rows
            )
            entry["mask_fingerprint_count"] = len(
                {row["parameters"]["dropout_mask_sha256_uint8"] for row in valid_rows}
            )
        elif family == "MISSING_FRAME":
            entry["missing_count"] = PROFILE_DEFINITIONS[profile_id]["missing_count"]
            entry["all_removed_counts_exact"] = all(
                row["parameters"]["removed_count"]
                == PROFILE_DEFINITIONS[profile_id]["missing_count"]
                for row in rows
            )
            entry["total_interpolated_count"] = sum(
                int(row["parameters"]["interpolated_count"]) for row in rows
            )
            entry["total_rejected_count"] = sum(
                int(row["parameters"]["rejected_count"]) for row in rows
            )
        elif family == "MOTION_BURST":
            entry["duration_samples"] = PROFILE_DEFINITIONS[profile_id]["duration_samples"]
            entry["std_multiplier"] = PROFILE_DEFINITIONS[profile_id]["std_multiplier"]
            ratios = [
                abs(float(row["parameters"]["signed_amplitude"]))
                / float(row["parameters"]["window_std"])
                for row in valid_rows
            ]
            entry["achieved_abs_amplitude_to_std_ratio"] = _stats(ratios, 9)
        elif family == "COMBINED":
            entry["application_order"] = PROFILE_DEFINITIONS[profile_id]["application_order"]
            entry["application_order_verified_all_samples"] = all(
                row["parameters"]["application_order"]
                == PROFILE_DEFINITIONS[profile_id]["application_order"]
                for row in valid_rows
            )
            entry["gaussian_achieved_snr_db"] = _stats(
                [float(row["parameters"]["gaussian"]["achieved_snr_db"]) for row in valid_rows],
                9,
            )
            entry["all_dropout_masks_have_exact_duration"] = all(
                sum(row["parameters"]["dropout_mask"])
                == PROFILE_DEFINITIONS[profile_id]["dropout_duration_samples"]
                for row in valid_rows
            )
        fidelity[profile_id] = entry

        pre_values = [
            float(row["pre_b1_delta_rms"])
            for row in valid_rows
            if row["pre_b1_delta_rms"] is not None
        ]
        post_values = [
            float(row["post_b1_delta_rms"])
            for row in valid_rows
            if row["post_b1_delta_rms"] is not None
        ]
        ratios = [
            float(row["preprocessing_attenuation_ratio"])
            for row in valid_rows
            if row["preprocessing_attenuation_ratio"] is not None
        ]
        attenuation[profile_id] = {
            "injection_domain": PROFILE_DEFINITIONS[profile_id]["injection_domain"],
            "pre_b1_delta_rms": _stats(pre_values, 9),
            "post_b1_delta_rms": _stats(post_values, 9),
            "post_to_pre_rms_ratio": _stats(ratios, 9),
            "interpretation_limit": (
                "Post-B1 companion; pre-B1 attenuation is not applicable."
                if not pre_values
                else "Ratio quantifies frozen BPF/Z-score attenuation or amplification; it is not real-sensor robustness."
            ),
        }

        reasons = Counter(
            reason for row in rows for reason in row.get("invalid_reason_codes", [])
        )
        fallback[profile_id] = {
            "condition_recommendation": (
                "INFERENCE_VALID_FOR_OFFLINE_STRESS_EVALUATION"
                if len(valid_rows) == len(rows)
                else "INVALID_OR_FALLBACK_RECOMMENDED"
            ),
            "total_samples": len(rows),
            "valid_samples": len(valid_rows),
            "invalid_or_fallback_samples": len(rows) - len(valid_rows),
            "reason_counts": dict(sorted(reasons.items())),
            "runtime_policy_changed": False,
        }
    return (
        {"phase_id": "M-B7", "profiles": fidelity},
        {"phase_id": "M-B7", "profiles": attenuation},
        {
            "phase_id": "M-B7",
            "recommendation_scope": "LATER_INTEGRATION_GUIDANCE_ONLY_NO_RUNTIME_POLICY_CHANGE",
            "profiles": fallback,
        },
    )


def compute_m_b7_evidence(root_dir: Path = ROOT_DIR) -> Dict[str, Any]:
    """Recompute complete M-B7 evidence in memory without writing files."""
    frozen = load_frozen_contracts(root_dir)
    guard = PhaseBAccessGuard(root_dir=root_dir)
    train_data = guard.get_model_selection_dataset("TRAIN")
    validation_data = guard.get_model_selection_dataset("VALIDATION")
    if len(train_data["windows"]) != 327 or len(validation_data["windows"]) != 79:
        raise RuntimeError("Authoritative pure-class TRAIN/VALIDATION count mismatch")
    validation_subjects = sorted({window["subject_id"] for window in validation_data["windows"]})
    if len(validation_subjects) != 17:
        raise RuntimeError("Authoritative VALIDATION subject count mismatch")

    zscore_stats = fit_train_zscore_statistics(train_data["signals"], detrend=False, bpf=True)
    stored_zstats = json.loads(
        (
            root_dir
            / "datasets/mmwave/manifests/M-B1_preprocessing_ablation/train_fit_statistics.json"
        ).read_text(encoding="utf-8")
    )["zscore_statistics"]["M-B1_D0_B1_Z1"]
    if zscore_stats["mean"] != stored_zstats["mean"] or zscore_stats["std"] != stored_zstats["std"]:
        raise RuntimeError("Frozen M-B1 TRAIN-fit statistics mismatch")
    clean_b1 = transform_signals(
        validation_data["signals"],
        detrend=False,
        bpf=True,
        zscore=True,
        zscore_stats=zscore_stats,
    )
    expected_fingerprint = json.loads(
        (
            root_dir
            / "datasets/mmwave/manifests/M-B1_preprocessing_ablation/preprocessing_fingerprints.json"
        ).read_text(encoding="utf-8")
    )["fingerprints"]["M-B1_D0_B1_Z1"]["validation_tensor_sha256"]
    if compute_tensor_fingerprint(clean_b1) != expected_fingerprint:
        raise RuntimeError("Frozen M-B1 VALIDATION tensor identity mismatch")

    runners, model_paths, model_structures = load_model_runners(root_dir)
    input_identity = build_input_identity(root_dir, model_paths)
    experiment_contract = {
        "phase_id": "M-B7",
        "scientific_scope": "OFFLINE_REAL_DATA_PERTURBATION_ROBUSTNESS",
        "eval_population": "AUTHORITATIVE_PURE_CLASS_VALIDATION_ONLY",
        "architecture": ARCHITECTURE_ID,
        "frozen_training_seeds": list(FROZEN_SEEDS),
        "frozen_preprocessing_profile": "M-B1_D0_B1_Z1",
        "frozen_preprocessing_name": "BPF_ZSCORE",
        "frozen_imbalance_strategy": "M-B2_CE_UNWEIGHTED",
        "frozen_calibration_profile": CALIBRATION_PROFILE_ID,
        "train_pure_class_windows": len(train_data["windows"]),
        "validation_pure_class_windows": len(validation_data["windows"]),
        "validation_subjects": len(validation_subjects),
        "locked_test_performance_access": 0,
        "model_trainings": 0,
        "model_conversions": 0,
        "global_perturbation_seed": GLOBAL_PERTURBATION_SEED,
        "clean_profile_count": 1,
        "perturbation_profile_count": len(PERTURBATION_PROFILE_ORDER),
        "total_inference_profile_count": len(ALL_PROFILE_ORDER),
        "expected_strict_int8_invocations": len(ALL_PROFILE_ORDER)
        * len(validation_data["windows"])
        * len(FROZEN_SEEDS),
        "locked_test_policy": "ZERO_PERFORMANCE_OR_PREDICTION_ACCESS",
        "claim_limits": [
            "NOT_REAL_SENSOR_ROBUSTNESS",
            "NOT_MR60_VALIDATION",
            "NOT_RASPBERRY_PI_VALIDATION",
            "NOT_CLINICAL_APNEA_DETECTION",
        ],
    }

    sample_rows: List[Dict[str, Any]] = []
    generated: Dict[str, List[Dict[str, Any]]] = {profile_id: [] for profile_id in ALL_PROFILE_ORDER}
    for profile_id in ALL_PROFILE_ORDER:
        for index, window in enumerate(validation_data["windows"]):
            item = generate_profile_sample(
                profile_id,
                validation_data["signals"][index],
                clean_b1[index],
                window,
                zscore_stats,
            )
            generated[profile_id].append(item)
            sample_rows.append(item["evidence"])

    determinism_profiles: Dict[str, Any] = {}
    for profile_id in ALL_PROFILE_ORDER:
        definition = PROFILE_DEFINITIONS[profile_id]
        if not definition["stochastic"]:
            determinism_profiles[profile_id] = {
                "stochastic": False,
                "samples_regenerated": 0,
                "all_samples_regenerated_identically": True,
                "numeric_fingerprint_example": None,
            }
            continue
        matches: List[bool] = []
        for index, window in enumerate(validation_data["windows"]):
            replay = generate_profile_sample(
                profile_id,
                validation_data["signals"][index],
                clean_b1[index],
                window,
                zscore_stats,
            )
            original = generated[profile_id][index]
            matches.append(
                replay["evidence"] == original["evidence"]
                and (
                    (replay["model_input"] is None and original["model_input"] is None)
                    or np.array_equal(replay["model_input"], original["model_input"])
                )
            )
        example = generated[profile_id][0]["evidence"]
        determinism_profiles[profile_id] = {
            "stochastic": True,
            "samples_regenerated": len(matches),
            "all_samples_regenerated_identically": bool(all(matches)),
            "numeric_fingerprint_example": {
                "canonical_sample_index": example["canonical_sample_index"],
                "model_input_sha256_float32": example["model_input_sha256_float32"],
                "perturbed_canonical_sha256_float64": example[
                    "perturbed_canonical_sha256_float64"
                ],
            },
        }
    determinism_audit = {
        "phase_id": "M-B7",
        "global_perturbation_seed": GLOBAL_PERTURBATION_SEED,
        "profiles": determinism_profiles,
        "all_stochastic_profiles_regenerated_identically": all(
            value["all_samples_regenerated_identically"]
            for value in determinism_profiles.values()
        ),
    }

    y_true = np.asarray(
        [window["safenest_label_id"] for window in validation_data["windows"]], dtype=int
    )
    window_ids = [window["window_id"] for window in validation_data["windows"]]
    m_b6_predictions = np.load(
        root_dir / "datasets/mmwave/manifests/M-B6_stage_equivalence/int8_tflite_predictions.npz"
    )
    m_b6_metrics = json.loads(
        (
            root_dir / "datasets/mmwave/manifests/M-B6_stage_equivalence/per_seed_stage_metrics.json"
        ).read_text(encoding="utf-8")
    )["per_seed_stage_metrics"]
    m_b6_quant = json.loads(
        (
            root_dir / "datasets/mmwave/manifests/M-B6_stage_equivalence/quantization_diagnostics.json"
        ).read_text(encoding="utf-8")
    )["quantization_diagnostics"]

    arrays: Dict[str, np.ndarray] = {}
    clean_baseline: Dict[str, Any] = {
        "phase_id": "M-B7",
        "m_b6_clean_identity_verified": True,
        "per_seed": {},
    }
    clean_runtime: Dict[int, Dict[str, np.ndarray]] = {}
    for seed in FROZEN_SEEDS:
        clean_inference = runners[seed].infer(clean_b1.astype(np.float32))
        clean_runtime[seed] = clean_inference
        run_key = f"{ARCHITECTURE_ID}_seed_{seed}"
        expected_predictions = np.asarray(m_b6_predictions[run_key], dtype=int)
        metrics = compute_run_metrics(
            y_true,
            clean_inference["predictions"],
            clean_inference["probabilities"],
            clean_inference["saturation_counts"],
            clean_inference["output_endpoint_counts"],
            clean_inference["predictions"],
            clean_inference["probabilities"],
            window_ids,
        )
        stage_c = m_b6_metrics[run_key]["stage_c_int8_tflite"]
        quant_c = m_b6_quant[run_key]
        identity_checks = {
            "top1_vector_equal": bool(
                np.array_equal(clean_inference["predictions"], expected_predictions)
            ),
            "macro_f1_equal": metrics["macro_f1"] == stage_c["macro_f1"],
            "accuracy_equal": metrics["accuracy"] == stage_c["accuracy"],
            "per_class_equal": metrics["per_class"] == stage_c["class_metrics"],
            "input_saturation_equal": metrics["quantization"]["input_saturation_ratio"]
            == quant_c["input_saturation_ratio"],
            "output_endpoint_equal": metrics["quantization"]["output_endpoint_ratio"]
            == quant_c["output_endpoint_ratio"],
        }
        if not all(identity_checks.values()):
            raise RuntimeError(
                f"M-B7_CLEAN_BASELINE_IDENTITY_MISMATCH seed {seed}: {identity_checks}"
            )
        clean_baseline["per_seed"][str(seed)] = {
            "model_artifact": model_structures[str(seed)],
            "m_b6_identity_checks": identity_checks,
            "metrics": metrics,
            "top1_predictions": clean_inference["predictions"].astype(int).tolist(),
            "dequantized_probabilities": clean_inference["probabilities"].tolist(),
        }
        for field in (
            "predictions",
            "probabilities",
            "saturation_counts",
            "output_endpoint_counts",
        ):
            arrays[array_key(seed, CLEAN_PROFILE_ID, field)] = clean_inference[field]

    perturbation_results: Dict[str, Any] = {
        profile_id: {
            "profile_id": profile_id,
            "family": PROFILE_DEFINITIONS[profile_id]["family"],
            "injection_domain": PROFILE_DEFINITIONS[profile_id]["injection_domain"],
            "per_seed": {},
        }
        for profile_id in PERTURBATION_PROFILE_ORDER
    }
    run_records: Dict[str, Any] = {}
    subject_results: Dict[str, Any] = {
        CLEAN_PROFILE_ID: {"per_seed": {}},
        **{profile_id: {"per_seed": {}} for profile_id in PERTURBATION_PROFILE_ORDER},
    }
    quantization_results: Dict[str, Any] = {}
    prediction_changes: List[Dict[str, Any]] = []

    for seed in FROZEN_SEEDS:
        clean_pred = clean_runtime[seed]["predictions"]
        clean_prob = clean_runtime[seed]["probabilities"]
        subject_results[CLEAN_PROFILE_ID]["per_seed"][str(seed)] = subject_level_metrics(
            validation_data["windows"], clean_pred, clean_pred
        )
        clean_metrics = clean_baseline["per_seed"][str(seed)]["metrics"]
        quantization_results[f"seed_{seed}__{CLEAN_PROFILE_ID}"] = clean_metrics[
            "quantization"
        ]

        for profile_id in PERTURBATION_PROFILE_ORDER:
            profile_items = generated[profile_id]
            valid_indices = [index for index, item in enumerate(profile_items) if item["valid"]]
            invalid_indices = [index for index, item in enumerate(profile_items) if not item["valid"]]
            full_predictions = np.full(len(validation_data["windows"]), -1, dtype=np.int16)
            full_probabilities = np.full(
                (len(validation_data["windows"]), len(LABEL_NAMES)), np.nan, dtype=np.float32
            )
            full_saturation = np.zeros(len(validation_data["windows"]), dtype=np.int32)
            full_endpoints = np.zeros(len(validation_data["windows"]), dtype=np.int32)
            if valid_indices:
                model_inputs = np.stack(
                    [profile_items[index]["model_input"] for index in valid_indices], axis=0
                )
                inference = runners[seed].infer(model_inputs)
                full_predictions[valid_indices] = inference["predictions"]
                full_probabilities[valid_indices] = inference["probabilities"]
                full_saturation[valid_indices] = inference["saturation_counts"]
                full_endpoints[valid_indices] = inference["output_endpoint_counts"]
                sliced_windows = [validation_data["windows"][index] for index in valid_indices]
                metrics = compute_run_metrics(
                    y_true[valid_indices],
                    full_predictions[valid_indices],
                    full_probabilities[valid_indices],
                    full_saturation[valid_indices],
                    full_endpoints[valid_indices],
                    clean_pred[valid_indices],
                    clean_prob[valid_indices],
                    [window_ids[index] for index in valid_indices],
                )
                subject = subject_level_metrics(
                    sliced_windows,
                    full_predictions[valid_indices],
                    clean_pred[valid_indices],
                )
            else:
                raise RuntimeError(f"No valid samples for profile {profile_id}")

            metrics["valid_sample_count"] = len(valid_indices)
            metrics["invalid_or_fallback_sample_count"] = len(invalid_indices)
            perturbation_results[profile_id]["per_seed"][str(seed)] = metrics
            subject_results[profile_id]["per_seed"][str(seed)] = subject
            quantization_results[f"seed_{seed}__{profile_id}"] = metrics["quantization"]
            for field, array in (
                ("predictions", full_predictions),
                ("probabilities", full_probabilities),
                ("saturation_counts", full_saturation),
                ("output_endpoint_counts", full_endpoints),
                (
                    "valid_mask",
                    np.asarray([index in valid_indices for index in range(len(full_predictions))], dtype=np.uint8),
                ),
            ):
                arrays[array_key(seed, profile_id, field)] = array

            run_id = f"seed_{seed}__{profile_id}"
            run_records[run_id] = {
                "seed": seed,
                "profile_id": profile_id,
                "model_sha256": runners[seed].sha256,
                "model_relative_path": model_paths[seed],
                "valid_sample_count": len(valid_indices),
                "invalid_sample_count": len(invalid_indices),
                "prediction_array_key": array_key(seed, profile_id, "predictions"),
                "probability_array_key": array_key(seed, profile_id, "probabilities"),
                "strict_int8_runtime_failures": 0,
            }
            for index in valid_indices:
                if int(full_predictions[index]) != int(clean_pred[index]):
                    row_evidence = profile_items[index]["evidence"]
                    prediction_changes.append(
                        {
                            "canonical_sample_index": row_evidence["canonical_sample_index"],
                            "window_id": row_evidence["window_id"],
                            "subject_id": row_evidence["subject_id"],
                            "recording_id": row_evidence["recording_id"],
                            "true_class": row_evidence["true_class"],
                            "true_label": row_evidence["true_label"],
                            "seed": seed,
                            "profile_id": profile_id,
                            "perturbation_parameters": row_evidence["parameters"],
                            "derived_rng_seed": row_evidence["derived_rng_seed"],
                            "perturbation_fingerprint": row_evidence[
                                "model_input_sha256_float32"
                            ],
                            "clean_prediction": int(clean_pred[index]),
                            "perturbed_prediction": int(full_predictions[index]),
                            "clean_output_probabilities": clean_prob[index].tolist(),
                            "perturbed_dequantized_probabilities": full_probabilities[
                                index
                            ].tolist(),
                            "clean_confidence": round(float(np.max(clean_prob[index])), 9),
                            "perturbed_confidence": round(
                                float(np.max(full_probabilities[index])), 9
                            ),
                        }
                    )

    perturbation_results_payload = {
        "phase_id": "M-B7",
        "profiles": perturbation_results,
    }
    cross_seed = aggregate_cross_seed(perturbation_results)
    fidelity, attenuation, fallback = build_profile_audits(
        sample_rows, determinism_profiles
    )

    cross_profiles = cross_seed["profiles"]
    worst = {
        "macro_f1_degradation": max(
            PERTURBATION_PROFILE_ORDER,
            key=lambda profile_id: cross_profiles[profile_id]["macro_f1_degradation"]["max"],
        ),
        "per_class_recall_degradation": max(
            PERTURBATION_PROFILE_ORDER,
            key=lambda profile_id: cross_profiles[profile_id][
                "maximum_positive_per_class_recall_degradation"
            ]["max"],
        ),
        "top1_agreement": min(
            PERTURBATION_PROFILE_ORDER,
            key=lambda profile_id: cross_profiles[profile_id]["top1_agreement"]["minimum"],
        ),
        "saturation": max(
            PERTURBATION_PROFILE_ORDER,
            key=lambda profile_id: cross_profiles[profile_id]["input_saturation"]["max"],
        ),
        "confidence_degradation": min(
            PERTURBATION_PROFILE_ORDER,
            key=lambda profile_id: cross_profiles[profile_id]["confidence_change"]["worst"],
        ),
    }
    new_collapse_conditions = [
        {
            "profile_id": profile_id,
            "affected_seeds": cross_profiles[profile_id]["collapse"]["affected_seeds"],
        }
        for profile_id in PERTURBATION_PROFILE_ORDER
        if cross_profiles[profile_id]["collapse"]["total"] > 0
    ]
    exceptions_registry: List[Dict[str, Any]] = [
        {
            "classification": "NON-BLOCKING IMPROVEMENT",
            "code": "INITIALIZATION_SEED_SENSITIVITY_PRESERVED",
            "detail": "M-B4 initialization sensitivity remains authoritative; M-B7 reports every seed separately and does not select a best seed.",
        },
        {
            "classification": "NON-BLOCKING IMPROVEMENT",
            "code": "TIMESTAMP_JITTER_OPTIONAL_PROFILE_NOT_ADDED",
            "detail": "A3 resampling machinery exists, but no timestamp-jitter magnitude was preregistered in the fixed 17-profile matrix. No post-hoc profile was added.",
        },
    ]
    for collapse in new_collapse_conditions:
        exceptions_registry.append(
            {
                "classification": "REQUIRED REFINEMENT",
                "code": "PERTURBATION_CLASS_COLLAPSE_OBSERVED",
                "detail": (
                    f"Scientific offline stress finding for {collapse['profile_id']} on seeds "
                    f"{collapse['affected_seeds']}; it is not an implementation blocker."
                ),
            }
        )

    clean_summary = {
        str(seed): {
            "macro_f1": clean_baseline["per_seed"][str(seed)]["metrics"]["macro_f1"],
            "accuracy": clean_baseline["per_seed"][str(seed)]["metrics"]["accuracy"],
        }
        for seed in FROZEN_SEEDS
    }
    summary = {
        "phase_id": "M-B7",
        "gate_status": "PASS_WITH_WARNINGS",
        "next_phase_authorized": False,
        "scientific_scope": "OFFLINE_REAL_DATA_PERTURBATION_ROBUSTNESS",
        "m_b6_clean_identity_verified": True,
        "architecture": ARCHITECTURE_ID,
        "frozen_seeds": list(FROZEN_SEEDS),
        "calibration_profile": CALIBRATION_PROFILE_ID,
        "model_trainings": 0,
        "model_conversions": 0,
        "validation_windows": len(validation_data["windows"]),
        "validation_subjects": len(validation_subjects),
        "locked_test_access_attempts": 0,
        "perturbation_seed": GLOBAL_PERTURBATION_SEED,
        "clean_baseline": clean_summary,
        "profile_count": len(PERTURBATION_PROFILE_ORDER),
        "total_strict_int8_invocations": len(ALL_PROFILE_ORDER)
        * len(validation_data["windows"])
        * len(FROZEN_SEEDS),
        "worst_profiles": worst,
        "new_collapse_conditions": new_collapse_conditions,
        "invalid_or_fallback_sample_count": sum(
            value["invalid_or_fallback_samples"] for value in fallback["profiles"].values()
        ),
        "warnings": [entry["code"] for entry in exceptions_registry],
        "blockers": [],
    }

    return {
        "input_identity.json": input_identity,
        "experiment_contract.json": experiment_contract,
        "perturbation_profile_contract.json": perturbation_profile_contract(),
        "clean_baseline_results.json": clean_baseline,
        "perturbation_runs.json": {
            "phase_id": "M-B7",
            "model_artifacts": model_structures,
            "runs": run_records,
        },
        "perturbation_results.json": perturbation_results_payload,
        "cross_seed_robustness_summary.json": cross_seed,
        "subject_level_robustness.json": {
            "phase_id": "M-B7",
            "validation_subject_count": len(validation_subjects),
            "profiles": subject_results,
        },
        "prediction_changes.jsonl": prediction_changes,
        "perturbation_sample_index.jsonl": sample_rows,
        "perturbation_fidelity_audit.json": fidelity,
        "preprocessing_attenuation_audit.json": attenuation,
        "quantization_diagnostics.json": {
            "phase_id": "M-B7",
            "diagnostic_scope": "PRE_CLAMP_INPUT_SATURATION_AND_OUTPUT_ENDPOINT_OCCUPANCY",
            "runs": quantization_results,
        },
        "fallback_recommendations.json": fallback,
        "locked_test_access_audit.json": {
            "phase_id": "M-B7",
            "performance_access_attempts": 0,
            "prediction_access_attempts": 0,
            "label_access_attempts": 0,
            "lock_preserved": True,
            "evaluated_split": "VALIDATION",
            "locked_test_predictions_generated": False,
        },
        "run_environment.json": {
            "phase_id": "M-B7",
            "python_version": sys.version.split()[0],
            "tensorflow_version": tf.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "requirements_mac_sha256": file_sha256(root_dir / "requirements-mac.txt"),
        },
        "exceptions.json": {
            "phase_id": "M-B7",
            "findings": exceptions_registry,
            "blocker_count": 0,
        },
        "determinism_audit.json": determinism_audit,
        "prediction_vectors.npz": arrays,
        "m_b7_summary.json": summary,
        "_internal": {
            "zscore_stats": zscore_stats,
            "validation_data": validation_data,
            "frozen": frozen,
        },
    }


def render_report(evidence: Dict[str, Any]) -> str:
    summary = evidence["m_b7_summary.json"]
    perturbation_results = evidence["perturbation_results.json"]["profiles"]
    cross_seed = evidence["cross_seed_robustness_summary.json"]["profiles"]
    attenuation = evidence["preprocessing_attenuation_audit.json"]["profiles"]
    fidelity = evidence["perturbation_fidelity_audit.json"]["profiles"]
    fallback = evidence["fallback_recommendations.json"]["profiles"]

    clean_rows = []
    for seed in FROZEN_SEEDS:
        clean = evidence["clean_baseline_results.json"]["per_seed"][str(seed)]["metrics"]
        clean_rows.append(
            f"| {seed} | {clean['macro_f1']:.6f} | {clean['accuracy']:.6f} | "
            f"{clean['prediction_distribution']['NORMAL']} / "
            f"{clean['prediction_distribution']['RAPID_OR_ABNORMAL']} / "
            f"{clean['prediction_distribution']['APNEA']} | "
            f"{clean['quantization']['input_saturation_ratio']:.9f} | "
            f"{clean['quantization']['output_endpoint_ratio']:.9f} |"
        )

    profile_rows = []
    for profile_id in PERTURBATION_PROFILE_ORDER:
        cross = cross_seed[profile_id]
        profile_rows.append(
            f"| `{profile_id}` | {cross['macro_f1_degradation']['max']:.6f} "
            f"(seed {cross['macro_f1_degradation']['worst_seed']}) | "
            f"{cross['maximum_positive_per_class_recall_degradation']['max']:.6f} | "
            f"{cross['top1_agreement']['minimum']:.6f} | "
            f"{cross['input_saturation']['max']:.9f} | "
            f"{cross['confidence_change']['worst']:.6f} | "
            f"{cross['collapse']['total']} |"
        )

    per_seed_rows = []
    for profile_id in PERTURBATION_PROFILE_ORDER:
        for seed in FROZEN_SEEDS:
            record = perturbation_results[profile_id]["per_seed"][str(seed)]
            per_seed_rows.append(
                f"| `{profile_id}` | {seed} | {record['macro_f1']:.6f} | "
                f"{record['accuracy']:.6f} | "
                f"{record['relative_to_clean']['positive_macro_f1_degradation']:.6f} | "
                f"{record['relative_to_clean']['maximum_positive_per_class_recall_degradation']:.6f} | "
                f"{record['relative_to_clean']['top1_agreement']:.6f} | "
                f"{record['confidence']['all_predictions']['mean']:.6f} | "
                f"{str(record['class_collapse_state']['collapsed']).lower()} |"
            )

    attenuation_rows = []
    for profile_id in PERTURBATION_PROFILE_ORDER:
        item = attenuation[profile_id]
        ratio = item["post_to_pre_rms_ratio"]
        attenuation_rows.append(
            f"| `{profile_id}` | `{item['injection_domain']}` | "
            f"{item['pre_b1_delta_rms']['mean'] if item['pre_b1_delta_rms']['mean'] is not None else 'N/A'} | "
            f"{item['post_b1_delta_rms']['mean'] if item['post_b1_delta_rms']['mean'] is not None else 'N/A'} | "
            f"{ratio['mean'] if ratio['mean'] is not None else 'N/A'} |"
        )

    fidelity_rows = []
    for profile_id in PERTURBATION_PROFILE_ORDER:
        item = fidelity[profile_id]
        if "achieved_snr_db" in item:
            magnitude = f"achieved SNR mean {item['achieved_snr_db']['mean']:.6f} dB"
        elif "gaussian_achieved_snr_db" in item:
            magnitude = (
                f"Gaussian achieved SNR mean "
                f"{item['gaussian_achieved_snr_db']['mean']:.6f} dB"
            )
        elif "scale" in item:
            magnitude = f"scale {item['scale']:.2f}; max formula error {item['maximum_formula_error']}"
        elif "frequency_hz" in item:
            magnitude = f"{item['frequency_hz']:.2f} Hz; amplitude multiplier {item['amplitude_multiplier']:.2f}"
        elif item["family"] == "CONTIGUOUS_DROPOUT":
            magnitude = f"{item['duration_samples']} samples; exact masks={item['all_masks_have_exact_duration']}"
        elif item["family"] == "MISSING_FRAME":
            magnitude = f"{item['missing_count']} removed; rejected={item['total_rejected_count']}"
        else:
            magnitude = f"{item.get('duration_samples', 'fixed')} samples; deterministic"
        fidelity_rows.append(
            f"| `{profile_id}` | {magnitude} | "
            f"{item['deterministic_regeneration_verified']} |"
        )

    fallback_rows = [
        f"| `{profile_id}` | `{fallback[profile_id]['condition_recommendation']}` | "
        f"{fallback[profile_id]['invalid_or_fallback_samples']} |"
        for profile_id in PERTURBATION_PROFILE_ORDER
    ]

    collapse_text = (
        json.dumps(summary["new_collapse_conditions"], ensure_ascii=False)
        if summary["new_collapse_conditions"]
        else "None observed."
    )
    return f"""# SafeNest mmWave M-B7 — Deterministic Input-Perturbation Robustness

- Phase: `M-B7`
- Scope: `OFFLINE_REAL_DATA_PERTURBATION_ROBUSTNESS`
- Frozen architecture: `{ARCHITECTURE_ID}`
- Frozen seeds: `{list(FROZEN_SEEDS)}`
- Frozen calibration: `{CALIBRATION_PROFILE_ID}`
- Evaluation population: 79 pure-class VALIDATION windows / 17 subjects
- LOCKED_TEST performance access: `0`
- Model trainings: `0`
- Model conversions: `0`

## Clean M-B6 identity

Fresh strict-INT8 clean inference reproduced the M-B6 top-1 vectors, per-class metrics,
Macro F1, accuracy, input saturation, and output endpoint ratio for all three seeds.

| Seed | Macro F1 | Accuracy | Prediction distribution N/R/A | Input saturation | Output endpoint |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(clean_rows)}

## Preregistered perturbations and cross-seed worst cases

| Profile | Worst F1 degradation | Worst recall degradation | Minimum Top-1 | Max saturation | Worst confidence change | Collapsed seeds |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(profile_rows)}

## Every seed and profile

| Profile | Seed | Macro F1 | Accuracy | F1 degradation | Max recall degradation | Top-1 vs clean | Mean confidence | Collapse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(per_seed_rows)}

Softmax confidence is reported only as the maximum dequantized output value; it is
not interpreted as a calibrated probability.

## Perturbation fidelity

| Profile | Independently regenerable magnitude evidence | Replay identical |
|---|---|---:|
{chr(10).join(fidelity_rows)}

## Frozen preprocessing attenuation

| Profile | Injection domain | Mean pre-B1 delta RMS | Mean post-B1 delta RMS | Mean post/pre ratio |
|---|---|---:|---:|---:|
{chr(10).join(attenuation_rows)}

Pre-B1 amplitude and drift behavior must not be interpreted as model invariance when
the frozen BPF/Z-score stage attenuates the injected signal. Post-B1 Gaussian profiles
separately probe model-input robustness.

## Subject-level and class findings

All 17 fixed VALIDATION subjects are retained in `subject_level_robustness.json`, with
per-class TP/FP/TN/FN, precision, recall, F1, prediction distribution, and clean deltas
for every seed/profile. This is one fixed subject set, not subject-split cross-validation.

New collapse conditions: {collapse_text}

## Fallback recommendations

| Profile | Recommendation | Invalid/fallback samples |
|---|---|---:|
{chr(10).join(fallback_rows)}

These are recommendations for later integration work only; M-B7 does not change a
runtime or risk policy.

## Worst conditions

- Macro-F1 degradation: `{summary['worst_profiles']['macro_f1_degradation']}`
- Per-class recall degradation: `{summary['worst_profiles']['per_class_recall_degradation']}`
- Top-1 agreement: `{summary['worst_profiles']['top1_agreement']}`
- Saturation: `{summary['worst_profiles']['saturation']}`
- Confidence degradation: `{summary['worst_profiles']['confidence_degradation']}`

## Limitations and claim boundary

This experiment injects deterministic synthetic perturbations into real, canonical
VALIDATION windows. It does not measure MR60 hardware, Raspberry Pi execution, a live
acquisition path, deployment readiness, or clinical apnea. SafeNest `APNEA` remains a
voluntary breath-hold proxy. Timestamp jitter was not added because the fixed matrix did
not preregister a magnitude; missing-frame damage does use the approved A3 timeline and
resampling contract.
"""


def write_m_b7_artifacts(root_dir: Path = ROOT_DIR) -> Dict[str, Any]:
    evidence = compute_m_b7_evidence(root_dir)
    manifest_dir = root_dir / MANIFEST_RELATIVE
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_OUTPUT_FILENAMES:
        value = evidence[filename]
        target = manifest_dir / filename
        if filename.endswith(".jsonl"):
            write_jsonl(target, value)
        elif filename.endswith(".npz"):
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
    return evidence["m_b7_summary.json"]


def main() -> None:
    summary = write_m_b7_artifacts(ROOT_DIR)
    print("SafeNest M-B7 generation complete")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
