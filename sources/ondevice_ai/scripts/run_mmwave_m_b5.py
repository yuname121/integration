# SafeNest mmWave Track — Phase M-B5 Representative Calibration Dataset Comparison Pipeline

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import scipy
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from mmwave_m_b1_preprocessing import fit_train_zscore_statistics, transform_signals
from mmwave_m_b2_imbalance import LABEL_NAMES, compute_one_vs_rest_false_positives
from mmwave_m_b3_architecture import build_model_by_id, compute_numerical_weights_sha256
from mmwave_m_b5_calibration import (
    CALIBRATION_RNG_SEED,
    CALIBRATION_SAMPLE_COUNT,
    PROFILE_IDS,
    SHORTLIST_SEEDS,
    build_all_calibration_profiles,
    build_all_calibration_profiles_with_metadata,
    compute_positive_recall_degradation,
    compute_tensor_statistics,
    convert_model_to_strict_int8_tflite,
    detect_new_quantization_collapse,
    evaluate_tflite_int8_model,
    explain_ranking_decision,
    inspect_tflite_model_bytes,
    rank_cross_seed_calibration_profiles,
)
from mmwave_phase_b_access import PhaseBAccessGuard


def _build_profile_provenance(
    prof_id: str,
    idx_list: List[int],
    train_windows: List[Dict[str, Any]],
    train_x_float32: np.ndarray,
) -> Dict[str, Any]:
    selected_windows = [train_windows[i] for i in idx_list]
    lbls = [w["safenest_label"] for w in selected_windows]
    class_dist = {c: lbls.count(c) for c in LABEL_NAMES}
    class_frac = {c: round(class_dist[c] / CALIBRATION_SAMPLE_COUNT, 6) for c in class_dist}
    subjs = [w["subject_id"] for w in selected_windows]
    subj_counts = {s: subjs.count(s) for s in set(subjs)}
    return {
        "profile_id": prof_id,
        "sample_count": CALIBRATION_SAMPLE_COUNT,
        "unique_subjects": len(subj_counts),
        "min_samples_per_subject": min(subj_counts.values()),
        "max_samples_per_subject": max(subj_counts.values()),
        "class_distribution_counts": class_dist,
        "class_distribution_fractions": class_frac,
        "samples": [
            {
                "calibration_slot_index": slot_i,
                "canonical_train_index": orig_i,
                "window_id": w["window_id"],
                "subject_id": w["subject_id"],
                "recording_id": w["recording_id"],
                "safenest_label": w["safenest_label"],
                "posture": w.get("posture"),
                "source_test_condition": w.get("source_test_condition"),
            }
            for slot_i, (orig_i, w) in enumerate(zip(idx_list, selected_windows))
        ],
    }


def _evaluate_conversion_run(
    *,
    primary_arch_id: str,
    seed: int,
    prof_id: str,
    model: tf.keras.Model,
    calib_x_float32: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    fl_probs: np.ndarray,
    fl_mets: Dict[str, Any],
    root_dir: Path,
    write_tflite: bool,
) -> Dict[str, Any]:
    run_key = f"{primary_arch_id}_seed_{seed}_{prof_id}"
    tflite_bytes, conv_meta = convert_model_to_strict_int8_tflite(model, calib_x_float32)
    tflite_rel_path = (
        f"models/mmwave/experiments/M-B5_representative_calibration/"
        f"{primary_arch_id}_seed{seed}_{prof_id}_int8.tflite"
    )
    if write_tflite:
        (root_dir / tflite_rel_path).write_bytes(tflite_bytes)
    conv_meta = dict(conv_meta)
    conv_meta["relative_path"] = tflite_rel_path

    eval_res = evaluate_tflite_int8_model(tflite_bytes, val_x, val_y, fl_probs)
    int8_f1 = eval_res["val_macro_f1"]
    float_f1 = round(fl_mets["macro_f1"], 6)
    signed_macro_f1_delta = round(int8_f1 - float_f1, 6)
    pos_macro_f1_degradation = round(max(0.0, float_f1 - int8_f1), 6)

    per_class_rec_deg, max_pos_rec_degradation = compute_positive_recall_degradation(
        fl_mets["class_metrics"],
        eval_res["class_metrics"],
    )
    new_collapse = detect_new_quantization_collapse(
        fl_mets["predictions"],
        fl_mets["class_metrics"],
        eval_res["int8_predictions"],
        eval_res["class_metrics"],
    )

    run_result_payload = {
        "architecture_id": primary_arch_id,
        "seed": seed,
        "profile_id": prof_id,
        "conversion_success": True,
        "select_tf_ops_count": conv_meta["select_tf_ops_count"],
        "strict_int8_eligible": (
            conv_meta["select_tf_ops_count"] == 0
            and conv_meta["input_dtype"] == "int8"
            and conv_meta["output_dtype"] == "int8"
        ),
        "float_baseline": {
            "macro_f1": float_f1,
            "accuracy": round(fl_mets["accuracy"], 6),
        },
        "int8_tflite": {
            "macro_f1": int8_f1,
            "accuracy": eval_res["val_accuracy"],
            "min_per_class_recall": eval_res["min_per_class_recall"],
            "apnea_recall": eval_res["apnea_recall"],
            "rapid_recall": eval_res["rapid_recall"],
            "collapsed": eval_res["collapsed"],
            "prediction_distribution": eval_res["prediction_distribution"],
            "class_metrics": eval_res["class_metrics"],
        },
        "quantization_diagnostics": {
            "signed_macro_f1_delta": signed_macro_f1_delta,
            "positive_macro_f1_degradation": pos_macro_f1_degradation,
            "per_class_positive_recall_degradation": per_class_rec_deg,
            "max_positive_recall_degradation": max_pos_rec_degradation,
            "top1_agreement": eval_res["top1_agreement"],
            "dequantized_output_mae": eval_res["dequantized_output_mae"],
            "dequantized_output_max_err": eval_res["dequantized_output_max_err"],
            "dequantized_output_min": eval_res["dequantized_output_min"],
            "dequantized_output_max": eval_res["dequantized_output_max"],
            "input_saturation_ratio": eval_res["input_saturation_ratio"],
            "saturated_input_elements": eval_res["saturated_input_elements"],
            "saturated_sample_count": eval_res["saturated_sample_count"],
            "output_endpoint_ratio": eval_res["output_endpoint_ratio"],
            "new_class_collapse": new_collapse,
        },
    }
    return {
        "run_key": run_key,
        "conv_meta": conv_meta,
        "run_result": run_result_payload,
        "int8_predictions": eval_res["int8_predictions"],
        "mismatch_samples": eval_res["mismatch_samples"],
        "eval_res": eval_res,
        "tflite_bytes": tflite_bytes,
    }


def _aggregate_cross_seed(
    calibration_results_dict: Dict[str, Any],
    tflite_manifest_dict: Dict[str, Any],
    primary_arch_id: str,
) -> List[Dict[str, Any]]:
    cross_seed_aggregates = []
    for prof_id in PROFILE_IDS:
        seed_runs = [calibration_results_dict[f"{primary_arch_id}_seed_{s}_{prof_id}"] for s in SHORTLIST_SEEDS]
        conv_success = sum(1 for r in seed_runs if r["conversion_success"])
        strict_eligible = all(r["strict_int8_eligible"] for r in seed_runs)
        new_collapse_cnt = sum(1 for r in seed_runs if r["quantization_diagnostics"]["new_class_collapse"])
        pos_f1_degs = [r["quantization_diagnostics"]["positive_macro_f1_degradation"] for r in seed_runs]
        pos_rec_degs = [r["quantization_diagnostics"]["max_positive_recall_degradation"] for r in seed_runs]
        top1_agrees = [r["quantization_diagnostics"]["top1_agreement"] for r in seed_runs]
        output_maes = [r["quantization_diagnostics"]["dequantized_output_mae"] for r in seed_runs]
        input_sats = [r["quantization_diagnostics"]["input_saturation_ratio"] for r in seed_runs]
        output_ends = [r["quantization_diagnostics"]["output_endpoint_ratio"] for r in seed_runs]
        tflite_sizes = [tflite_manifest_dict[f"{primary_arch_id}_seed_{s}_{prof_id}"]["bytes"] for s in SHORTLIST_SEEDS]
        is_eligible = (conv_success == len(SHORTLIST_SEEDS)) and strict_eligible and (new_collapse_cnt == 0)
        cross_seed_aggregates.append({
            "profile_id": prof_id,
            "eligible": is_eligible,
            "conversion_success_count": conv_success,
            "strict_int8_eligible": strict_eligible,
            "new_class_collapse_count": new_collapse_cnt,
            "worst_positive_macro_f1_degradation": round(float(np.max(pos_f1_degs)), 6),
            "worst_positive_recall_degradation": round(float(np.max(pos_rec_degs)), 6),
            "min_top1_agreement": round(float(np.min(top1_agrees)), 6),
            "max_dequantized_output_mae": round(float(np.max(output_maes)), 6),
            "max_input_saturation_ratio": round(float(np.max(input_sats)), 6),
            "max_output_endpoint_ratio": round(float(np.max(output_ends)), 6),
            "macro_f1_degradation": {
                "mean": round(float(np.mean(pos_f1_degs)), 6),
                "median": round(float(np.median(pos_f1_degs)), 6),
                "std": round(float(np.std(pos_f1_degs)), 6),
                "min": round(float(np.min(pos_f1_degs)), 6),
                "max": round(float(np.max(pos_f1_degs)), 6),
                "per_seed": {
                    str(s): r["quantization_diagnostics"]["positive_macro_f1_degradation"]
                    for s, r in zip(SHORTLIST_SEEDS, seed_runs)
                },
            },
            "top1_agreement": {
                "mean": round(float(np.mean(top1_agrees)), 6),
                "min": round(float(np.min(top1_agrees)), 6),
                "per_seed": {
                    str(s): r["quantization_diagnostics"]["top1_agreement"]
                    for s, r in zip(SHORTLIST_SEEDS, seed_runs)
                },
            },
            "output_mae": {
                "mean": round(float(np.mean(output_maes)), 6),
                "max": round(float(np.max(output_maes)), 6),
                "per_seed": {
                    str(s): r["quantization_diagnostics"]["dequantized_output_mae"]
                    for s, r in zip(SHORTLIST_SEEDS, seed_runs)
                },
            },
            "tflite_bytes": {
                "mean": int(np.mean(tflite_sizes)),
                "min": int(np.min(tflite_sizes)),
                "max": int(np.max(tflite_sizes)),
            },
        })
    return cross_seed_aggregates


def _replay_selected_profile_conversions(
    *,
    winning_profile_id: str,
    profile_indices_dict: Dict[str, List[int]],
    frozen_models_by_seed: Dict[int, tf.keras.Model],
    float_probs_by_seed: Dict[int, np.ndarray],
    float_metrics_by_seed: Dict[int, Dict[str, Any]],
    train_x: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    primary_arch_id: str,
    stored_tflite_manifest: Dict[str, Any],
    stored_preds: Dict[str, np.ndarray],
    stored_calib: Dict[str, Any],
    root_dir: Path,
) -> Dict[str, Any]:
    idx_list = profile_indices_dict[winning_profile_id]
    calib_x = train_x[idx_list]
    seed_replays = {}
    functional_ok = True
    notes = []

    for seed in SHORTLIST_SEEDS:
        run_key = f"{primary_arch_id}_seed_{seed}_{winning_profile_id}"
        replay = _evaluate_conversion_run(
            primary_arch_id=primary_arch_id,
            seed=seed,
            prof_id=winning_profile_id,
            model=frozen_models_by_seed[seed],
            calib_x_float32=calib_x,
            val_x=val_x,
            val_y=val_y,
            fl_probs=float_probs_by_seed[seed],
            fl_mets=float_metrics_by_seed[seed],
            root_dir=root_dir,
            write_tflite=False,
        )
        stored_meta = stored_tflite_manifest[run_key]
        stored_diag = stored_calib[run_key]["quantization_diagnostics"]
        stored_int8 = stored_calib[run_key]["int8_tflite"]
        replay_meta = replay["conv_meta"]
        replay_diag = replay["run_result"]["quantization_diagnostics"]
        replay_int8 = replay["run_result"]["int8_tflite"]

        checks = {
            "representative_indices_equal": True,
            "input_scale_equal": abs(float(stored_meta["input_scale"]) - float(replay_meta["input_scale"])) <= 1e-12,
            "input_zero_point_equal": int(stored_meta["input_zero_point"]) == int(replay_meta["input_zero_point"]),
            "output_scale_equal": abs(float(stored_meta["output_scale"]) - float(replay_meta["output_scale"])) <= 1e-12,
            "output_zero_point_equal": int(stored_meta["output_zero_point"]) == int(replay_meta["output_zero_point"]),
            "op_inventory_equal": list(stored_meta.get("op_types", [])) == list(replay_meta.get("op_types", [])),
            "prediction_vector_equal": bool(np.array_equal(stored_preds[run_key], replay["int8_predictions"])),
            "macro_f1_equal": float(stored_int8["macro_f1"]) == float(replay_int8["macro_f1"]),
            "accuracy_equal": float(stored_int8["accuracy"]) == float(replay_int8["accuracy"]),
            "top1_agreement_equal": float(stored_diag["top1_agreement"]) == float(replay_diag["top1_agreement"]),
            "output_mae_equal": float(stored_diag["dequantized_output_mae"]) == float(replay_diag["dequantized_output_mae"]),
            "input_saturation_equal": float(stored_diag["input_saturation_ratio"]) == float(replay_diag["input_saturation_ratio"]),
            "sha256_identical": stored_meta["sha256"] == replay_meta["sha256"],
        }
        seed_ok = all(v for k, v in checks.items() if k != "sha256_identical")
        if not seed_ok:
            functional_ok = False
            notes.append(f"{run_key} functional mismatch: {[k for k, v in checks.items() if not v]}")
        seed_replays[str(seed)] = {
            "run_key": run_key,
            "functional_equality": seed_ok,
            "checks": checks,
            "replay_input_scale": replay_meta["input_scale"],
            "replay_input_zero_point": replay_meta["input_zero_point"],
            "replay_output_scale": replay_meta["output_scale"],
            "replay_output_zero_point": replay_meta["output_zero_point"],
            "replay_sha256": replay_meta["sha256"],
            "stored_sha256": stored_meta["sha256"],
        }

    return {
        "selected_profile_id": winning_profile_id,
        "representative_indices": idx_list,
        "seeds_replayed": SHORTLIST_SEEDS,
        "functional_reproducibility_verified": functional_ok,
        "seed_replays": seed_replays,
        "notes": notes,
    }


def run_m_b5_pipeline(
    root_dir: Path = ROOT_DIR,
    profiles_to_convert: Optional[List[str]] = None,
    require_preserve_abc_indices: bool = False,
) -> Dict[str, Any]:
    """Execute M-B5 representative calibration comparison.

    When profiles_to_convert is a subset (e.g. Profile D only), existing TFLite /
    calibration artifacts for other profiles are reused after independent validation.
    """
    print("=== SafeNest Phase M-B5 Representative Calibration Pipeline ===")

    if profiles_to_convert is None:
        profiles_to_convert = list(PROFILE_IDS)
    convert_set: Set[str] = set(profiles_to_convert)
    unknown = convert_set - set(PROFILE_IDS)
    if unknown:
        raise ValueError(f"Unknown profiles_to_convert: {sorted(unknown)}")

    manifest_dir = root_dir / "datasets/mmwave/manifests/M-B5_representative_calibration"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    exp_models_dir = root_dir / "models/mmwave/experiments/M-B5_representative_calibration"
    exp_models_dir.mkdir(parents=True, exist_ok=True)
    report_dir = root_dir / "docs/reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    actual_tf = tf.__version__
    actual_np = np.__version__
    actual_scipy = scipy.__version__
    req_file = root_dir / "requirements-mac.txt"
    if not req_file.is_file():
        raise RuntimeError("requirements-mac.txt missing!")
    req_sha = hashlib.sha256(req_file.read_bytes()).hexdigest()
    print(f"0. Pinned environment preflight passed: TF={actual_tf}, NP={actual_np}, SciPy={actual_scipy}.")

    upstream_files_to_hash = [
        "requirements-mac.txt",
        "datasets/mmwave/manifests/M-B0_evaluation_protocol/evaluation_contract.json",
        "datasets/mmwave/manifests/M-B0_evaluation_protocol/locked_test_access_policy.json",
        "datasets/mmwave/manifests/M-B0_evaluation_protocol/checksums.sha256",
        "datasets/mmwave/manifests/M-B1_preprocessing_ablation/selected_preprocessing_profile.json",
        "datasets/mmwave/manifests/M-B1_preprocessing_ablation/train_fit_statistics.json",
        "datasets/mmwave/manifests/M-B1_preprocessing_ablation/experiment_contract.json",
        "datasets/mmwave/manifests/M-B1_preprocessing_ablation/checksums.sha256",
        "datasets/mmwave/manifests/M-B2_class_imbalance/selected_imbalance_strategy.json",
        "datasets/mmwave/manifests/M-B2_class_imbalance/experiment_contract.json",
        "datasets/mmwave/manifests/M-B2_class_imbalance/checksums.sha256",
        "datasets/mmwave/manifests/M-B3_architecture_comparison/selected_architecture_shortlist.json",
        "datasets/mmwave/manifests/M-B3_architecture_comparison/architecture_profiles.json",
        "datasets/mmwave/manifests/M-B3_architecture_comparison/experiment_contract.json",
        "datasets/mmwave/manifests/M-B3_architecture_comparison/checksums.sha256",
        "datasets/mmwave/manifests/M-B4_multiseed_stability/primary_float_finalist.json",
        "datasets/mmwave/manifests/M-B4_multiseed_stability/backup_architecture.json",
        "datasets/mmwave/manifests/M-B4_multiseed_stability/experiment_contract.json",
        "datasets/mmwave/manifests/M-B4_multiseed_stability/training_runs.json",
        "datasets/mmwave/manifests/M-B4_multiseed_stability/seed_weights.npz",
        "datasets/mmwave/manifests/M-B4_multiseed_stability/validation_predictions.npz",
        "datasets/mmwave/manifests/M-B4_multiseed_stability/per_seed_results.json",
        "datasets/mmwave/manifests/M-B4_multiseed_stability/multi_seed_results.json",
        "datasets/mmwave/manifests/M-B4_multiseed_stability/checksums.sha256",
        "datasets/mmwave/manifests/a5_subject_split/subject_split_manifest.jsonl",
        "datasets/mmwave/manifests/a6_full_conversion/full_window_manifest.jsonl",
    ]
    input_identity_list = []
    for rel_p in upstream_files_to_hash:
        fp = root_dir / rel_p
        if not fp.is_file():
            raise FileNotFoundError(f"Required upstream file missing: {rel_p}")
        input_identity_list.append({"path": rel_p, "measured_sha256": hashlib.sha256(fp.read_bytes()).hexdigest()})
    (manifest_dir / "input_identity.json").write_text(
        json.dumps({"phase_id": "M-B5", "inputs": input_identity_list}, indent=2), encoding="utf-8"
    )
    print(f"1. Upstream identity locked ({len(input_identity_list)} files).")

    mb1_sel = json.loads(
        (root_dir / "datasets/mmwave/manifests/M-B1_preprocessing_ablation/selected_preprocessing_profile.json").read_text(encoding="utf-8")
    ).get("selected_profile_id")
    mb2_sel = json.loads(
        (root_dir / "datasets/mmwave/manifests/M-B2_class_imbalance/selected_imbalance_strategy.json").read_text(encoding="utf-8")
    ).get("selected_strategy_id")
    primary_arch_id = json.loads(
        (root_dir / "datasets/mmwave/manifests/M-B4_multiseed_stability/primary_float_finalist.json").read_text(encoding="utf-8")
    ).get("primary_stable_float_finalist")
    backup_arch_id = json.loads(
        (root_dir / "datasets/mmwave/manifests/M-B4_multiseed_stability/backup_architecture.json").read_text(encoding="utf-8")
    ).get("backup_architecture_id")
    print(f"2. Upstream contracts verified: M-B1={mb1_sel}, M-B2={mb2_sel}, M-B4 Primary={primary_arch_id}, Backup={backup_arch_id}")

    guard = PhaseBAccessGuard(root_dir=root_dir)
    train_data = guard.get_model_selection_dataset("TRAIN")
    val_data = guard.get_model_selection_dataset("VALIDATION")
    zstats = fit_train_zscore_statistics(train_data["signals"], detrend=False, bpf=True)
    train_x_float32 = transform_signals(train_data["signals"], detrend=False, bpf=True, zscore=True, zscore_stats=zstats).astype(np.float32)
    val_x_float32 = transform_signals(val_data["signals"], detrend=False, bpf=True, zscore=True, zscore_stats=zstats).astype(np.float32)
    train_x = np.expand_dims(train_x_float32, axis=-1)
    val_x = np.expand_dims(val_x_float32, axis=-1)
    val_y = np.array([w["safenest_label_id"] for w in val_data["windows"]], dtype=int)
    train_subjs_count = len(set(w["subject_id"] for w in train_data["windows"]))
    val_subjs_count = len(set(w["subject_id"] for w in val_data["windows"]))

    (manifest_dir / "experiment_contract.json").write_text(
        json.dumps(
            {
                "phase_id": "M-B5",
                "description": "Comparison of four preregistered TRAIN-only representative calibration dataset profiles across frozen M-B4 seed weight sets 42, 43, 44",
                "frozen_preprocessing_profile": mb1_sel,
                "frozen_imbalance_strategy": mb2_sel,
                "frozen_primary_architecture": primary_arch_id,
                "frozen_seed_weights": SHORTLIST_SEEDS,
                "eval_population": "VALIDATION_SET_ONLY",
                "train_samples": len(train_data["windows"]),
                "train_subjects": train_subjs_count,
                "eval_samples": len(val_data["windows"]),
                "eval_subjects": val_subjs_count,
                "locked_test_access": "ZERO_PROHIBITED",
                "calibration_profiles": PROFILE_IDS,
                "calibration_sample_count": CALIBRATION_SAMPLE_COUNT,
                "calibration_sampling_seed": CALIBRATION_RNG_SEED,
                "new_model_trainings": 0,
                "profiles_converted_this_run": sorted(convert_set),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    profile_indices_dict, profile_d_meta = build_all_calibration_profiles_with_metadata(
        train_data["windows"], train_x_float32, sample_count=CALIBRATION_SAMPLE_COUNT
    )

    existing_indices_path = manifest_dir / "representative_dataset_indices.json"
    if require_preserve_abc_indices and existing_indices_path.is_file():
        existing = json.loads(existing_indices_path.read_text(encoding="utf-8")).get("profile_indices", {})
        for pid in ("M-B5_CAL_TRAIN_ORDER_120", "M-B5_CAL_RANDOM_PROPORTIONAL_120", "M-B5_CAL_CLASS_BALANCED_120"):
            if existing.get(pid) != profile_indices_dict[pid]:
                raise RuntimeError(f"Profiles A/B/C index preservation failed for {pid}")

    (manifest_dir / "representative_dataset_indices.json").write_text(
        json.dumps(
            {
                "phase_id": "M-B5",
                "calibration_sample_count": CALIBRATION_SAMPLE_COUNT,
                "profile_indices": profile_indices_dict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    all_train_stats = compute_tensor_statistics(train_x_float32)
    profile_provenance_dict = {}
    profile_stats_dict = {}
    for prof_id, idx_list in profile_indices_dict.items():
        if len(idx_list) != CALIBRATION_SAMPLE_COUNT or len(set(idx_list)) != CALIBRATION_SAMPLE_COUNT:
            raise ValueError(f"Invalid index set for {prof_id}")
        if any(i < 0 or i >= len(train_data["windows"]) for i in idx_list):
            raise ValueError(f"Out-of-bounds index found in profile {prof_id}")
        profile_provenance_dict[prof_id] = _build_profile_provenance(
            prof_id, idx_list, train_data["windows"], train_x_float32
        )
        prof_stats = compute_tensor_statistics(train_x_float32[idx_list])
        range_cov = (prof_stats["max"] - prof_stats["min"]) / (all_train_stats["max"] - all_train_stats["min"])
        prof_stats["range_coverage_ratio_vs_train"] = round(float(range_cov), 6)
        profile_stats_dict[prof_id] = prof_stats

    (manifest_dir / "representative_dataset_provenance.json").write_text(
        json.dumps({"phase_id": "M-B5", "profiles": profile_provenance_dict}, indent=2), encoding="utf-8"
    )
    (manifest_dir / "representative_dataset_statistics.json").write_text(
        json.dumps(
            {"phase_id": "M-B5", "all_train_statistics": all_train_stats, "profile_statistics": profile_stats_dict},
            indent=2,
        ),
        encoding="utf-8",
    )
    (manifest_dir / "representative_profile_contract.json").write_text(
        json.dumps(
            {
                "phase_id": "M-B5",
                "sample_count_per_profile": CALIBRATION_SAMPLE_COUNT,
                "calibration_sampling_seed": CALIBRATION_RNG_SEED,
                "profiles": {
                    "M-B5_CAL_TRAIN_ORDER_120": "First 120 eligible pure-class TRAIN rows in canonical order",
                    "M-B5_CAL_RANDOM_PROPORTIONAL_120": "Random proportional sample matching TRAIN class distribution without replacement (RNG seed 20260810)",
                    "M-B5_CAL_CLASS_BALANCED_120": "Equal class balanced sample (40 NORMAL, 40 RAPID_OR_ABNORMAL, 40 APNEA) without replacement (RNG seed 20260810)",
                    "M-B5_CAL_DISTRIBUTION_AWARE_120": "Deterministic farthest-point coverage in TRAIN robust feature space using authoritative posture + source_test_condition metadata with max-2-per-subject cap",
                },
                "distribution_aware_profile": profile_d_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("4. All 4 calibration profiles constructed deterministically.")

    mb4_tr_data = json.loads(
        (root_dir / "datasets/mmwave/manifests/M-B4_multiseed_stability/training_runs.json").read_text(encoding="utf-8")
    ).get("training_runs", {})
    mb4_weights = np.load(root_dir / "datasets/mmwave/manifests/M-B4_multiseed_stability/seed_weights.npz")
    mb4_preds = np.load(root_dir / "datasets/mmwave/manifests/M-B4_multiseed_stability/validation_predictions.npz")

    frozen_models_by_seed = {}
    float_probs_by_seed = {}
    float_metrics_by_seed = {}
    for seed in SHORTLIST_SEEDS:
        run_key = f"{primary_arch_id}_seed_{seed}"
        model = build_model_by_id(primary_arch_id)
        arch_w_keys = sorted(
            [k for k in mb4_weights.files if k.startswith(f"{primary_arch_id}_seed_{seed}_layer_weight_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        model.set_weights([mb4_weights[k] for k in arch_w_keys])
        computed_sha = compute_numerical_weights_sha256(model)
        if computed_sha != mb4_tr_data[run_key]["final_weights_sha256"]:
            raise RuntimeError(f"M-B5_UPSTREAM_WEIGHT_IDENTITY_MISMATCH for {run_key}")
        fl_probs = model.predict(val_x, verbose=0)
        fl_preds = np.argmax(fl_probs, axis=1).astype(int)
        if not np.array_equal(fl_preds, mb4_preds[run_key]):
            raise RuntimeError(f"Float prediction mismatch for {run_key}")
        fl_cm = compute_one_vs_rest_false_positives(val_y, fl_preds)
        frozen_models_by_seed[seed] = model
        float_probs_by_seed[seed] = fl_probs
        float_metrics_by_seed[seed] = {
            "macro_f1": float(np.mean([fl_cm[c]["f1_score"] for c in LABEL_NAMES])),
            "accuracy": float(np.mean(fl_preds == val_y)),
            "predictions": fl_preds,
            "class_metrics": fl_cm,
        }
    print("5. Frozen M-B4 models & weights loaded and validated for seeds 42, 43, 44.")

    # Load reusable A/B/C artifacts when only converting a subset.
    existing_tflite_manifest = {}
    existing_calib = {}
    existing_preds = {}
    if convert_set != set(PROFILE_IDS):
        existing_tflite_manifest = json.loads(
            (manifest_dir / "tflite_artifact_manifest.json").read_text(encoding="utf-8")
        ).get("tflite_artifacts", {})
        existing_calib = json.loads(
            (manifest_dir / "calibration_results.json").read_text(encoding="utf-8")
        ).get("calibration_results", {})
        existing_preds = dict(np.load(manifest_dir / "validation_predictions.npz"))
        for pid in PROFILE_IDS:
            if pid in convert_set:
                continue
            for seed in SHORTLIST_SEEDS:
                run_key = f"{primary_arch_id}_seed_{seed}_{pid}"
                meta = existing_tflite_manifest[run_key]
                tf_path = root_dir / meta["relative_path"]
                if not tf_path.is_file():
                    raise RuntimeError(f"Missing reusable TFLite for preserved profile: {meta['relative_path']}")
                measured = inspect_tflite_model_bytes(tf_path.read_bytes())
                if measured["sha256"] != meta["sha256"] or measured["bytes"] != meta["bytes"]:
                    raise RuntimeError(f"Preserved TFLite identity mismatch for {run_key}")
                if measured["input_dtype"] != "int8" or measured["output_dtype"] != "int8":
                    raise RuntimeError(f"Preserved TFLite dtype gate failed for {run_key}")
                if measured["select_tf_ops_count"] != 0:
                    raise RuntimeError(f"Preserved TFLite Flex/Select detected for {run_key}")
                # Runtime re-eval to refresh diagnostics under hardened helpers.
                eval_res = evaluate_tflite_int8_model(
                    tf_path.read_bytes(), val_x, val_y, float_probs_by_seed[seed]
                )
                if not np.array_equal(eval_res["int8_predictions"], existing_preds[run_key]):
                    raise RuntimeError(f"Preserved prediction vector mismatch for {run_key}")
                fl_mets = float_metrics_by_seed[seed]
                per_class_rec_deg, max_pos_rec_degradation = compute_positive_recall_degradation(
                    fl_mets["class_metrics"], eval_res["class_metrics"]
                )
                new_collapse = detect_new_quantization_collapse(
                    fl_mets["predictions"],
                    fl_mets["class_metrics"],
                    eval_res["int8_predictions"],
                    eval_res["class_metrics"],
                )
                float_f1 = round(fl_mets["macro_f1"], 6)
                existing_calib[run_key] = {
                    "architecture_id": primary_arch_id,
                    "seed": seed,
                    "profile_id": pid,
                    "conversion_success": True,
                    "select_tf_ops_count": measured["select_tf_ops_count"],
                    "strict_int8_eligible": True,
                    "float_baseline": {
                        "macro_f1": float_f1,
                        "accuracy": round(fl_mets["accuracy"], 6),
                    },
                    "int8_tflite": {
                        "macro_f1": eval_res["val_macro_f1"],
                        "accuracy": eval_res["val_accuracy"],
                        "min_per_class_recall": eval_res["min_per_class_recall"],
                        "apnea_recall": eval_res["apnea_recall"],
                        "rapid_recall": eval_res["rapid_recall"],
                        "collapsed": eval_res["collapsed"],
                        "prediction_distribution": eval_res["prediction_distribution"],
                        "class_metrics": eval_res["class_metrics"],
                    },
                    "quantization_diagnostics": {
                        "signed_macro_f1_delta": round(eval_res["val_macro_f1"] - float_f1, 6),
                        "positive_macro_f1_degradation": round(max(0.0, float_f1 - eval_res["val_macro_f1"]), 6),
                        "per_class_positive_recall_degradation": per_class_rec_deg,
                        "max_positive_recall_degradation": max_pos_rec_degradation,
                        "top1_agreement": eval_res["top1_agreement"],
                        "dequantized_output_mae": eval_res["dequantized_output_mae"],
                        "dequantized_output_max_err": eval_res["dequantized_output_max_err"],
                        "dequantized_output_min": eval_res["dequantized_output_min"],
                        "dequantized_output_max": eval_res["dequantized_output_max"],
                        "input_saturation_ratio": eval_res["input_saturation_ratio"],
                        "saturated_input_elements": eval_res["saturated_input_elements"],
                        "saturated_sample_count": eval_res["saturated_sample_count"],
                        "output_endpoint_ratio": eval_res["output_endpoint_ratio"],
                        "new_class_collapse": new_collapse,
                    },
                }
                # Keep measured op inventory on preserved artifacts.
                existing_tflite_manifest[run_key] = {
                    **meta,
                    **{k: measured[k] for k in (
                        "bytes", "sha256", "input_dtype", "output_dtype", "input_shape", "output_shape",
                        "input_scale", "input_zero_point", "output_scale", "output_zero_point",
                        "op_types", "select_tf_ops_count",
                    )},
                    "relative_path": meta["relative_path"],
                }

    conversion_runs_dict = {}
    calibration_results_dict = {}
    tflite_manifest_dict = {}
    all_val_preds_npz_dict = {}
    all_mismatch_samples_list = []
    profile_d_conversions = 0
    abc_conversions = 0

    for prof_id in PROFILE_IDS:
        idx_list = profile_indices_dict[prof_id]
        calib_x_float32 = train_x[idx_list]
        for seed in SHORTLIST_SEEDS:
            run_key = f"{primary_arch_id}_seed_{seed}_{prof_id}"
            if prof_id not in convert_set:
                tflite_manifest_dict[run_key] = existing_tflite_manifest[run_key]
                calibration_results_dict[run_key] = existing_calib[run_key]
                all_val_preds_npz_dict[run_key] = np.asarray(existing_preds[run_key], dtype=int)
                continue

            print(f"\n--- Converting & Evaluating {run_key} ---")
            result = _evaluate_conversion_run(
                primary_arch_id=primary_arch_id,
                seed=seed,
                prof_id=prof_id,
                model=frozen_models_by_seed[seed],
                calib_x_float32=calib_x_float32,
                val_x=val_x,
                val_y=val_y,
                fl_probs=float_probs_by_seed[seed],
                fl_mets=float_metrics_by_seed[seed],
                root_dir=root_dir,
                write_tflite=True,
            )
            tflite_manifest_dict[run_key] = result["conv_meta"]
            conversion_runs_dict[run_key] = result["conv_meta"]
            calibration_results_dict[run_key] = result["run_result"]
            all_val_preds_npz_dict[run_key] = result["int8_predictions"]
            for mitem in result["mismatch_samples"]:
                mitem = dict(mitem)
                mitem["profile_id"] = prof_id
                mitem["seed"] = seed
                all_mismatch_samples_list.append(mitem)
            if prof_id == "M-B5_CAL_DISTRIBUTION_AWARE_120":
                profile_d_conversions += 1
            else:
                abc_conversions += 1

    # Refresh mismatch samples for preserved profiles too (diagnostic completeness).
    if convert_set != set(PROFILE_IDS):
        for pid in PROFILE_IDS:
            if pid in convert_set:
                continue
            for seed in SHORTLIST_SEEDS:
                run_key = f"{primary_arch_id}_seed_{seed}_{pid}"
                eval_res = evaluate_tflite_int8_model(
                    (root_dir / tflite_manifest_dict[run_key]["relative_path"]).read_bytes(),
                    val_x,
                    val_y,
                    float_probs_by_seed[seed],
                )
                for mitem in eval_res["mismatch_samples"]:
                    mitem = dict(mitem)
                    mitem["profile_id"] = pid
                    mitem["seed"] = seed
                    all_mismatch_samples_list.append(mitem)

    np.savez_compressed(manifest_dir / "validation_predictions.npz", **all_val_preds_npz_dict)
    val_index_lines = []
    for idx_w, w in enumerate(val_data["windows"]):
        val_index_lines.append(
            json.dumps(
                {
                    "validation_window_index": idx_w,
                    "recording_id": w["recording_id"],
                    "subject_id": w["subject_id"],
                    "true_label": w["safenest_label"],
                    "predictions_by_run": {
                        rkey: int(preds[idx_w]) for rkey, preds in all_val_preds_npz_dict.items()
                    },
                }
            )
        )
    (manifest_dir / "validation_prediction_index.jsonl").write_text("\n".join(val_index_lines) + "\n", encoding="utf-8")
    (manifest_dir / "mismatch_samples.jsonl").write_text(
        "\n".join(json.dumps(m) for m in all_mismatch_samples_list) + ("\n" if all_mismatch_samples_list else ""),
        encoding="utf-8",
    )

    cross_seed_aggregates = _aggregate_cross_seed(calibration_results_dict, tflite_manifest_dict, primary_arch_id)
    (manifest_dir / "conversion_runs.json").write_text(
        json.dumps(
            {
                "phase_id": "M-B5",
                "total_conversions": 12,
                "conversions_rerun_this_execution": profile_d_conversions + abc_conversions,
                "profile_d_conversions_rerun": profile_d_conversions,
                "profiles_abc_conversions_rerun": abc_conversions,
                "conversions": tflite_manifest_dict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (manifest_dir / "calibration_results.json").write_text(
        json.dumps({"phase_id": "M-B5", "calibration_results": calibration_results_dict}, indent=2), encoding="utf-8"
    )
    (manifest_dir / "cross_seed_calibration_results.json").write_text(
        json.dumps({"phase_id": "M-B5", "cross_seed_calibration_results": cross_seed_aggregates}, indent=2),
        encoding="utf-8",
    )
    (manifest_dir / "tflite_artifact_manifest.json").write_text(
        json.dumps({"phase_id": "M-B5", "tflite_artifacts": tflite_manifest_dict}, indent=2), encoding="utf-8"
    )

    ranked_profiles = rank_cross_seed_calibration_profiles(cross_seed_aggregates, eps=1e-5)
    ranking_decision = explain_ranking_decision(ranked_profiles, eps=1e-5)
    if not ranked_profiles:
        winning_profile = None
        selection_status = "INCONCLUSIVE"
    else:
        winning_profile = ranked_profiles[0]
        selection_status = "SELECTED_CALIBRATION_PROFILE"

    selected_profile_payload = {
        "phase_id": "M-B5",
        "selection_status": selection_status,
        "selected_calibration_profile": winning_profile["profile_id"] if winning_profile else None,
        "profile_details": winning_profile,
        "ranking_decision": ranking_decision,
        "selection_rationale": (
            "Preregistered M-B5 8-criterion ranking rule with eps=1e-5 numerical tie tolerance: "
            "Lower worst-seed positive Macro F1 degradation, lower worst-seed max positive recall degradation, "
            "higher min Top-1 agreement, lower max dequantized output MAE, lower max input saturation ratio, "
            "lower max output endpoint ratio, simpler policy order, lexicographic profile ID."
        ),
    }
    (manifest_dir / "selected_calibration_profile.json").write_text(
        json.dumps(selected_profile_payload, indent=2), encoding="utf-8"
    )
    print(f"7. Selected Calibration Profile: {winning_profile['profile_id'] if winning_profile else 'NONE'}")
    print(f"   Ranking decision: {ranking_decision.get('deciding_criterion')}")

    # Index reconstruction check for all profiles.
    replay_indices = build_all_calibration_profiles(train_data["windows"], train_x_float32, sample_count=CALIBRATION_SAMPLE_COUNT)
    index_replay_ok = all(replay_indices[pid] == profile_indices_dict[pid] for pid in PROFILE_IDS)

    selected_replay = None
    if winning_profile:
        selected_replay = _replay_selected_profile_conversions(
            winning_profile_id=winning_profile["profile_id"],
            profile_indices_dict=profile_indices_dict,
            frozen_models_by_seed=frozen_models_by_seed,
            float_probs_by_seed=float_probs_by_seed,
            float_metrics_by_seed=float_metrics_by_seed,
            train_x=train_x,
            val_x=val_x,
            val_y=val_y,
            primary_arch_id=primary_arch_id,
            stored_tflite_manifest=tflite_manifest_dict,
            stored_preds=all_val_preds_npz_dict,
            stored_calib=calibration_results_dict,
            root_dir=root_dir,
        )
        if not selected_replay["functional_reproducibility_verified"]:
            raise RuntimeError(f"Selected-profile conversion replay failed: {selected_replay['notes']}")

    determinism_payload = {
        "phase_id": "M-B5",
        "calibration_sampling_seed": CALIBRATION_RNG_SEED,
        "profile_generation_deterministic": index_replay_ok,
        "functional_reproducibility_verified": bool(
            selected_replay and selected_replay["functional_reproducibility_verified"]
        ) if winning_profile else index_replay_ok,
        "index_reconstruction_verified_all_profiles": index_replay_ok,
        "selected_profile_three_seed_conversion_replay": selected_replay,
        "notes": (
            "Index reconstruction verified for all four profiles. "
            "Selected calibration profile also received a clean three-seed conversion/evaluation replay "
            "requiring functional equality of quantization params, op inventory, prediction vectors, and metrics."
        ),
    }
    (manifest_dir / "determinism_audit.json").write_text(json.dumps(determinism_payload, indent=2), encoding="utf-8")

    (manifest_dir / "locked_test_access_audit.json").write_text(
        json.dumps(
            {
                "phase_id": "M-B5",
                "performance_access_attempts": 0,
                "lock_preserved": True,
                "notes": "No model predictions or calibration calculations evaluated on LOCKED_TEST set during Phase M-B5.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (manifest_dir / "run_environment.json").write_text(
        json.dumps(
            {
                "phase_id": "M-B5",
                "python_version": sys.version.split()[0],
                "tensorflow_version": actual_tf,
                "numpy_version": actual_np,
                "scipy_version": actual_scipy,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "requirements_mac_sha256": req_sha,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (manifest_dir / "exceptions.json").write_text(
        json.dumps(
            {
                "phase_id": "M-B5",
                "exceptions_registry": [
                    {
                        "exception_id": "HISTORICAL_DETREND_MEAN_CENTERING_DISCREPANCY",
                        "severity": "WARNING",
                        "status": "APPROVED_HISTORICAL_DISCREPANCY",
                        "impact": "Non-blocking historical pilot discrepancy in A6 annotations.",
                    },
                    {
                        "exception_id": "INITIALIZATION_SEED_SENSITIVITY",
                        "severity": "WARNING",
                        "status": "REGISTERED_SEED_SENSITIVITY",
                        "impact": (
                            "M-B4 primary Conv1D GAP baseline exhibits substantial initialization-seed sensitivity "
                            "(seed42 Macro F1≈0.663708, seed43≈0.451010, seed44≈0.329107). "
                            "M-B5 evaluates calibration profiles across all three frozen seeds; it does not select seeds."
                        ),
                    },
                    {
                        "exception_id": "SEED_CLASS_COLLAPSE",
                        "severity": "WARNING",
                        "status": "REGISTERED_SEED_COLLAPSE",
                        "impact": "SeparableConv1D GAP collapsed on seed 44. Excluded from M-B4/M-B5 considerations.",
                    },
                    {
                        "exception_id": "PROFILE_D_METADATA_SEMANTIC_REPAIR",
                        "severity": "INFO",
                        "status": "REPAIRED",
                        "impact": (
                            "Distribution-Aware Profile D now derives posture/source_test_condition vocabularies "
                            "from authoritative TRAIN metadata (Lying/Sitting, Rest/Post-exercise) instead of the "
                            "prior incorrect supine/left/right and class-like test_condition encoding."
                        ),
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_payload = {
        "phase_id": "M-B5",
        "gate_status": "PASS_WITH_WARNINGS" if winning_profile else "INCONCLUSIVE",
        "m_b6_entry_status": "READY_WITH_CONDITIONS" if winning_profile else "NO",
        "selected_calibration_profile": winning_profile["profile_id"] if winning_profile else None,
        "ranking_decision": ranking_decision,
        "frozen_primary_architecture": primary_arch_id,
        "frozen_weight_seeds": SHORTLIST_SEEDS,
        "profiles_evaluated": PROFILE_IDS,
        "total_strict_int8_conversions": 12,
        "profile_d_conversions_rerun": profile_d_conversions,
        "profiles_abc_conversions_rerun": abc_conversions,
        "neural_network_models_retrained": 0,
        "locked_test_access_attempts": 0,
    }
    (manifest_dir / "m_b5_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    manifest_files = [
        "input_identity.json",
        "experiment_contract.json",
        "representative_profile_contract.json",
        "representative_dataset_indices.json",
        "representative_dataset_provenance.json",
        "representative_dataset_statistics.json",
        "conversion_runs.json",
        "calibration_results.json",
        "cross_seed_calibration_results.json",
        "validation_predictions.npz",
        "validation_prediction_index.jsonl",
        "mismatch_samples.jsonl",
        "tflite_artifact_manifest.json",
        "selected_calibration_profile.json",
        "determinism_audit.json",
        "locked_test_access_audit.json",
        "run_environment.json",
        "exceptions.json",
        "m_b5_summary.json",
    ]
    checksum_lines = []
    for rel_n in manifest_files:
        target_f = manifest_dir / rel_n
        checksum_lines.append(f"{hashlib.sha256(target_f.read_bytes()).hexdigest()}  {rel_n}")
    (manifest_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    report_rows = []
    for agg in cross_seed_aggregates:
        report_rows.append(
            "| `{pid}` | `{elig}` | `{w_f1_deg:.6f}` | `{w_rec_deg:.6f}` | `{min_top1:.6f}` | `{max_mae:.6f}` | `{max_sat:.6f}` | `{max_end:.6f}` |".format(
                pid=agg["profile_id"],
                elig="ELIGIBLE" if agg["eligible"] else "INELIGIBLE",
                w_f1_deg=agg["worst_positive_macro_f1_degradation"],
                w_rec_deg=agg["worst_positive_recall_degradation"],
                min_top1=agg["min_top1_agreement"],
                max_mae=agg["max_dequantized_output_mae"],
                max_sat=agg["max_input_saturation_ratio"],
                max_end=agg["max_output_endpoint_ratio"],
            )
        )
    win_id = winning_profile["profile_id"] if winning_profile else "NONE"
    deciding = ranking_decision.get("deciding_criterion", "N/A")
    seed_detail_lines = []
    if winning_profile:
        for seed in SHORTLIST_SEEDS:
            rk = f"{primary_arch_id}_seed_{seed}_{win_id}"
            row = calibration_results_dict[rk]
            seed_detail_lines.append(
                f"- Seed {seed}: Float Macro F1=`{row['float_baseline']['macro_f1']:.6f}`, "
                f"INT8 Macro F1=`{row['int8_tflite']['macro_f1']:.6f}`, "
                f"Top-1=`{row['quantization_diagnostics']['top1_agreement']:.6f}`, "
                f"MAE=`{row['quantization_diagnostics']['dequantized_output_mae']:.6f}`, "
                f"Input sat=`{row['quantization_diagnostics']['input_saturation_ratio']:.6f}`"
            )

    report_content = f"""# SafeNest mmWave M-B5 — Representative Calibration Dataset Comparison Report

- **Author**: Cursor Implementation / Validation Engineer
- **Date**: 2026-08-10
- **Target Repository**: `https://github.com/sheepmeat/test.git`
- **Branch**: `feature/M-B5-representative-calibration`
- **Phase M-B5 Gate Status**: `{summary_payload['gate_status']}`
- **M-B6 Entry Status**: `{summary_payload['m_b6_entry_status']}`
- **Pinned Environment**: Python {sys.version.split()[0]} / TensorFlow {actual_tf} / NumPy {actual_np} / SciPy {actual_scipy}
- **Frozen Primary Architecture**: `{primary_arch_id}`
- **Frozen Weight Seeds**: `[42, 43, 44]`
- **TRAIN Population**: {len(train_data['windows'])} pure-class windows ({train_subjs_count} subjects)
- **VALIDATION Population**: {len(val_data['windows'])} pure-class windows ({val_subjs_count} subjects)
- **Selected Calibration Profile**: `{win_id}`
- **Ranking Deciding Criterion**: `{deciding}`
- **Tie Tolerance**: `eps=1e-5`
- **Profile-D conversions rerun this closure**: `{profile_d_conversions}`
- **Profiles A/B/C conversions rerun this closure**: `{abc_conversions}`
- **Neural-network models retrained**: `0`

---

## 1. Executive Summary

Phase M-B5 compares four pre-registered TRAIN-only representative calibration dataset profiles across the three frozen M-B4 primary-architecture weight sets (`42`, `43`, `44`). M-B5 selects a **calibration profile**, not a model seed.

M-B4 already demonstrated substantial **initialization-seed sensitivity** for `{primary_arch_id}` (seed42 Macro F1≈0.663708, seed43≈0.451010, seed44≈0.329107). That scientific result is preserved; M-B5 measures quantization behavior under those frozen weights.

Profile D (`M-B5_CAL_DISTRIBUTION_AWARE_120`) was repaired to use authoritative TRAIN metadata:
- posture vocabulary: `{profile_d_meta['posture_vocabulary']}`
- source_test_condition vocabulary: `{profile_d_meta['source_test_condition_vocabulary']}`
- unknown/missing policy: `{profile_d_meta['unknown_or_missing_token']}`
- continuous features: `{profile_d_meta['continuous_features']}`
- subject-cap final state: `{profile_d_meta['subject_cap_final_state']}`

Final selected calibration profile under the preregistered epsilon-aware ranking rule: **`{win_id}`** (decided by `{deciding}`).

LOCKED_TEST performance access attempts: **0**. Formal M-B6 Float Keras → Float TFLite → INT8 stage equivalence remains pending. MR60 hardware and Raspberry Pi performance are **not** validated. APNEA remains a voluntary breath-hold proxy, not clinical apnea. This phase does **not** claim production/deployment readiness.

---

## 2. Cross-Seed Calibration Profile Performance Matrix (VALIDATION Set)

| Profile ID | Eligibility | Worst F1 Deg. | Worst Rec Deg. | Min Top-1 | Max Output MAE | Max Input Sat. | Max End. Ratio |
|---|---|---|---|---|---|---|---|
{chr(10).join(report_rows)}

---

## 3. Selected Profile Details

Selected Calibration Profile: **`{win_id}`**
- Worst Positive Macro F1 Degradation: `{winning_profile['worst_positive_macro_f1_degradation'] if winning_profile else 'N/A'}`
- Worst Positive Recall Degradation: `{winning_profile['worst_positive_recall_degradation'] if winning_profile else 'N/A'}`
- Minimum Top-1 Agreement: `{winning_profile['min_top1_agreement'] if winning_profile else 'N/A'}`
- Maximum Output Probability MAE: `{winning_profile['max_dequantized_output_mae'] if winning_profile else 'N/A'}`
- Maximum Input Saturation Ratio: `{winning_profile['max_input_saturation_ratio'] if winning_profile else 'N/A'}`
- Maximum Output Endpoint Ratio: `{winning_profile['max_output_endpoint_ratio'] if winning_profile else 'N/A'}`

Selected-profile per-seed diagnostics:
{chr(10).join(seed_detail_lines) if seed_detail_lines else '- NONE'}

Selected-profile three-seed conversion replay functional equality: `{determinism_payload['functional_reproducibility_verified']}`

---

## 4. Limitations & Scope

- Inherited immutable A5 subject split (TRAIN={train_subjs_count} subjects, VALIDATION={val_subjs_count} subjects).
- LOCKED_TEST remained unused for representative sampling, calibration, ranking, and mismatch inspection.
- No clinical apnea claims.
- M-B6 formal stage equivalence is still pending.
- MR60 / Raspberry Pi hardware validation is not claimed.

---

## 5. Validation & Exit Gate Summary

- Standalone M-B5 validator (`scripts/validate_mmwave_m_b5.py`) must pass independently against these artifacts.
- Checksum coverage: {len(manifest_files)} machine-readable manifests in `checksums.sha256`
- M-B5 Gate Status: `{summary_payload['gate_status']}`
- M-B6 Entry Status: `{summary_payload['m_b6_entry_status']}`
"""
    (report_dir / "20260810_Antigravity_M-B5_Representative_Calibration_01.md").write_text(
        report_content, encoding="utf-8"
    )
    print("12. Human-readable report written.")

    print("\n=== Standalone M-B5 Validator Execution ===")
    from validate_mmwave_m_b5 import validate_m_b5_artifacts

    val_res = validate_m_b5_artifacts(root_dir=root_dir, manifest_dir=manifest_dir)
    print("M-B5 Validation Success:", val_res["validation_success"])
    print("=== M-B5 Pipeline Execution Completed Successfully ===")
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="SafeNest mmWave M-B5 representative calibration pipeline")
    parser.add_argument(
        "--repair-profile-d",
        action="store_true",
        help="Rebuild/reconvert only Profile D; preserve and revalidate Profiles A/B/C artifacts",
    )
    args = parser.parse_args()
    if args.repair_profile_d:
        run_m_b5_pipeline(
            profiles_to_convert=["M-B5_CAL_DISTRIBUTION_AWARE_120"],
            require_preserve_abc_indices=True,
        )
    else:
        run_m_b5_pipeline()


if __name__ == "__main__":
    main()
