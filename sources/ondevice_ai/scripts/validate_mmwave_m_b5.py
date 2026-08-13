# SafeNest mmWave Track — Phase M-B5 Standalone Validator (Hardened Evidence-Truth)

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import scipy
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from mmwave_m_b1_preprocessing import fit_train_zscore_statistics, transform_signals
from mmwave_m_b2_imbalance import LABEL_NAMES, compute_one_vs_rest_false_positives
from mmwave_m_b3_architecture import build_model_by_id, compute_numerical_weights_sha256
from mmwave_m_b5_calibration import (
    CALIBRATION_SAMPLE_COUNT,
    PROFILE_IDS,
    SHORTLIST_SEEDS,
    build_all_calibration_profiles,
    build_all_calibration_profiles_with_metadata,
    compute_positive_recall_degradation,
    detect_new_quantization_collapse,
    evaluate_tflite_int8_model,
    explain_ranking_decision,
    inspect_tflite_model_bytes,
    rank_cross_seed_calibration_profiles,
)
from mmwave_phase_b_access import PhaseBAccessGuard
from validate_mmwave_full_conversion import validate_full_conversion_artifacts
from validate_mmwave_m_b0 import validate_m_b0_artifacts
from validate_mmwave_m_b1 import validate_m_b1_artifacts
from validate_mmwave_m_b2 import validate_m_b2_artifacts
from validate_mmwave_m_b3 import validate_m_b3_artifacts
from validate_mmwave_m_b4 import validate_m_b4_artifacts


class MB5ValidationError(Exception):
    """Raised when Phase M-B5 validation fails."""


REQUIRED_MB5_ARTIFACTS = {
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
    "checksums.sha256",
}


def _round6(x: float) -> float:
    return round(float(x), 6)


def validate_m_b5_artifacts(
    root_dir: Path = ROOT_DIR,
    manifest_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Independently validate all Phase M-B5 calibration comparison artifacts without retraining."""
    if manifest_dir is None:
        manifest_dir = root_dir / "datasets/mmwave/manifests/M-B5_representative_calibration"

    if not manifest_dir.is_dir():
        raise MB5ValidationError(f"M-B5 manifest directory missing: {manifest_dir}")

    guard = PhaseBAccessGuard(root_dir=root_dir)

    actual_tf = tf.__version__
    actual_np = np.__version__
    actual_scipy = scipy.__version__
    req_file = root_dir / "requirements-mac.txt"
    if not req_file.is_file():
        raise MB5ValidationError("requirements-mac.txt missing!")
    req_sha = hashlib.sha256(req_file.read_bytes()).hexdigest()

    env_file = manifest_dir / "run_environment.json"
    if not env_file.is_file():
        raise MB5ValidationError("run_environment.json missing!")
    env_data = json.loads(env_file.read_text(encoding="utf-8"))
    if (
        env_data.get("tensorflow_version") != actual_tf
        or env_data.get("numpy_version") != actual_np
        or env_data.get("scipy_version") != actual_scipy
    ):
        raise MB5ValidationError("Environment mismatch versus run_environment.json")
    if env_data.get("requirements_mac_sha256") != req_sha:
        raise MB5ValidationError("requirements-mac.txt SHA-256 mismatch!")

    # Upstream validators (M-B0..M-B4 + A5/A6).
    for name, fn in (
        ("M-B0", validate_m_b0_artifacts),
        ("M-B1", validate_m_b1_artifacts),
        ("M-B2", validate_m_b2_artifacts),
        ("M-B3", validate_m_b3_artifacts),
        ("M-B4", validate_m_b4_artifacts),
        ("M-A6", validate_full_conversion_artifacts),
    ):
        res = fn(root_dir=root_dir)
        if not res.get("validation_success"):
            raise MB5ValidationError(f"Upstream {name} validation failed: {res}")

    # A5 subject-split standalone gate (artifact-coupled).
    from validate_mmwave_subject_split import load_json, load_jsonl, validate_a5

    a5_manifest_dir = root_dir / "datasets/mmwave/manifests/a5_subject_split"
    a5_split_file = root_dir / "datasets/mmwave/splits/mmwave_real_subject_split_v1.json"
    a5_summary = load_json(a5_manifest_dir / "a5_summary.json")
    a5_errors = validate_a5(
        load_jsonl(root_dir / "datasets/mmwave/manifests/a0_raw_inventory/recording_index.jsonl"),
        load_jsonl(root_dir / "datasets/mmwave/manifests/a4_label_pilot/window_label_manifest.jsonl"),
        load_json(a5_manifest_dir / "split_profile.json"),
        load_jsonl(a5_manifest_dir / "subject_split_manifest.jsonl"),
        load_jsonl(a5_manifest_dir / "recording_split_manifest.jsonl"),
        load_jsonl(a5_manifest_dir / "pilot_window_split_manifest.jsonl"),
        load_json(a5_manifest_dir / "provenance_schema.json"),
        load_json(a5_manifest_dir / "split_balance_report.json"),
        load_json(a5_manifest_dir / "exceptions.json"),
        load_json(a5_split_file),
        verify_checksum_file=True,
        output=a5_manifest_dir,
        split_output=a5_split_file,
    )
    if a5_errors:
        raise MB5ValidationError(f"Upstream M-A5 validation failed: {a5_errors}")
    if (
        a5_summary.get("validation_success") is not True
        or a5_summary.get("a5_gate_status") != "PASS_WITH_WARNINGS"
        or a5_summary.get("a6_entry_status") != "READY_WITH_CONDITIONS"
    ):
        raise MB5ValidationError("Upstream M-A5 summary gate coupling failed")

    input_identity_file = manifest_dir / "input_identity.json"
    if not input_identity_file.is_file():
        raise MB5ValidationError("input_identity.json missing!")
    inputs_list = json.loads(input_identity_file.read_text(encoding="utf-8")).get("inputs", [])
    if len(inputs_list) < 25:
        raise MB5ValidationError(f"input_identity.json must contain at least 25 upstream files, got {len(inputs_list)}")
    for input_item in inputs_list:
        rel_p = input_item.get("path")
        exp_sha = input_item.get("measured_sha256")
        full_p = root_dir / rel_p
        if not full_p.is_file():
            raise MB5ValidationError(f"Upstream identity file missing from checkout: {rel_p}")
        act_sha = hashlib.sha256(full_p.read_bytes()).hexdigest()
        if act_sha != exp_sha:
            raise MB5ValidationError(f"Upstream identity SHA mismatch for '{rel_p}'")

    train_data = guard.get_model_selection_dataset("TRAIN")
    val_data = guard.get_model_selection_dataset("VALIDATION")
    act_train_subjs = len(set(w["subject_id"] for w in train_data["windows"]))
    act_val_subjs = len(set(w["subject_id"] for w in val_data["windows"]))
    if len(train_data["windows"]) != 327 or act_train_subjs != 77:
        raise MB5ValidationError(
            f"TRAIN population mismatch: windows={len(train_data['windows'])}, subjects={act_train_subjs}"
        )
    if len(val_data["windows"]) != 79 or act_val_subjs != 17:
        raise MB5ValidationError(
            f"VALIDATION population mismatch: windows={len(val_data['windows'])}, subjects={act_val_subjs}"
        )

    exp_contract_data = json.loads((manifest_dir / "experiment_contract.json").read_text(encoding="utf-8"))
    if exp_contract_data.get("train_subjects") != act_train_subjs:
        raise MB5ValidationError("TRAIN subject count mismatch in experiment_contract.json")
    if exp_contract_data.get("eval_subjects") != act_val_subjs:
        raise MB5ValidationError("VALIDATION subject count mismatch in experiment_contract.json")
    if exp_contract_data.get("new_model_trainings", -1) != 0:
        raise MB5ValidationError("M-B5 must record new_model_trainings=0")

    # Pure-class / split integrity for representative indices.
    for w in train_data["windows"]:
        if w.get("safenest_label") == "AMBIGUOUS":
            raise MB5ValidationError("TRAIN model-selection population unexpectedly contains AMBIGUOUS")
        if w.get("split") != "TRAIN":
            raise MB5ValidationError("Non-TRAIN window leaked into TRAIN model-selection population")

    mb4_pri_arch = json.loads(
        (root_dir / "datasets/mmwave/manifests/M-B4_multiseed_stability/primary_float_finalist.json").read_text(encoding="utf-8")
    ).get("primary_stable_float_finalist")
    if mb4_pri_arch != "M-B3_CONV1D_GAP_BASELINE":
        raise MB5ValidationError(f"M-B4 primary float finalist mismatch: {mb4_pri_arch}")

    mb4_tr_data = json.loads(
        (root_dir / "datasets/mmwave/manifests/M-B4_multiseed_stability/training_runs.json").read_text(encoding="utf-8")
    ).get("training_runs", {})
    mb4_weights = np.load(root_dir / "datasets/mmwave/manifests/M-B4_multiseed_stability/seed_weights.npz")
    mb4_preds = np.load(root_dir / "datasets/mmwave/manifests/M-B4_multiseed_stability/validation_predictions.npz")

    zstats = fit_train_zscore_statistics(train_data["signals"], detrend=False, bpf=True)
    train_x_float32 = transform_signals(train_data["signals"], detrend=False, bpf=True, zscore=True, zscore_stats=zstats).astype(np.float32)
    val_x_float32 = transform_signals(val_data["signals"], detrend=False, bpf=True, zscore=True, zscore_stats=zstats).astype(np.float32)
    val_x = np.expand_dims(val_x_float32, axis=-1)
    val_y = np.array([w["safenest_label_id"] for w in val_data["windows"]], dtype=int)

    float_probs_by_seed = {}
    float_preds_by_seed = {}
    float_metrics_by_seed = {}
    for seed in SHORTLIST_SEEDS:
        run_key = f"{mb4_pri_arch}_seed_{seed}"
        model = build_model_by_id(mb4_pri_arch)
        arch_w_keys = sorted(
            [k for k in mb4_weights.files if k.startswith(f"{mb4_pri_arch}_seed_{seed}_layer_weight_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        if not arch_w_keys:
            raise MB5ValidationError(f"No stored weights found for {run_key}")
        model.set_weights([mb4_weights[k] for k in arch_w_keys])
        computed_sha = compute_numerical_weights_sha256(model)
        if computed_sha != mb4_tr_data[run_key]["final_weights_sha256"]:
            raise MB5ValidationError(f"M-B5_UPSTREAM_WEIGHT_IDENTITY_MISMATCH for {run_key}")
        fl_probs = model.predict(val_x, verbose=0)
        fl_preds = np.argmax(fl_probs, axis=1).astype(int)
        if not np.array_equal(fl_preds, mb4_preds[run_key]):
            raise MB5ValidationError(f"Float prediction vector mismatch for {run_key}")
        fl_cm = compute_one_vs_rest_false_positives(val_y, fl_preds)
        float_probs_by_seed[seed] = fl_probs
        float_preds_by_seed[seed] = fl_preds
        float_metrics_by_seed[seed] = fl_cm

    # Representative profile reconstruction including Profile-D metadata contract.
    stored_indices_dict = json.loads(
        (manifest_dir / "representative_dataset_indices.json").read_text(encoding="utf-8")
    ).get("profile_indices", {})
    recomputed_indices_dict, profile_d_meta = build_all_calibration_profiles_with_metadata(
        train_data["windows"], train_x_float32, sample_count=CALIBRATION_SAMPLE_COUNT
    )
    for prof_id in PROFILE_IDS:
        stored_idx = stored_indices_dict.get(prof_id)
        recomputed_idx = recomputed_indices_dict[prof_id]
        if stored_idx is None:
            raise MB5ValidationError(f"Profile '{prof_id}' missing from representative_dataset_indices.json")
        if len(stored_idx) != CALIBRATION_SAMPLE_COUNT:
            raise MB5ValidationError(f"Profile '{prof_id}' stored index count mismatch")
        if len(set(stored_idx)) != CALIBRATION_SAMPLE_COUNT:
            raise MB5ValidationError(f"Duplicate index found in stored profile '{prof_id}'")
        if any(i < 0 or i >= len(train_data["windows"]) for i in stored_idx):
            raise MB5ValidationError(f"Out-of-bounds index found in stored profile '{prof_id}'")
        for i in stored_idx:
            w = train_data["windows"][i]
            if w.get("split") != "TRAIN":
                raise MB5ValidationError(f"Non-TRAIN sample in profile {prof_id}")
            if w.get("safenest_label") == "AMBIGUOUS":
                raise MB5ValidationError(f"AMBIGUOUS sample in profile {prof_id}")
        if stored_idx != recomputed_idx:
            raise MB5ValidationError(f"M-B5_PROFILE_NONDETERMINISTIC: Recomputed indices for '{prof_id}' do not match stored indices")

    contract = json.loads((manifest_dir / "representative_profile_contract.json").read_text(encoding="utf-8"))
    d_contract = contract.get("distribution_aware_profile")
    if not isinstance(d_contract, dict):
        raise MB5ValidationError("representative_profile_contract.json missing distribution_aware_profile metadata")
    if d_contract.get("posture_vocabulary") != profile_d_meta["posture_vocabulary"]:
        raise MB5ValidationError(
            f"Profile-D posture vocabulary mismatch: stored={d_contract.get('posture_vocabulary')}, "
            f"expected={profile_d_meta['posture_vocabulary']}"
        )
    if d_contract.get("source_test_condition_vocabulary") != profile_d_meta["source_test_condition_vocabulary"]:
        raise MB5ValidationError(
            f"Profile-D source_test_condition vocabulary mismatch: stored={d_contract.get('source_test_condition_vocabulary')}, "
            f"expected={profile_d_meta['source_test_condition_vocabulary']}"
        )
    bad_postures = {"supine", "left", "right"}
    if any(str(v).lower() in bad_postures for v in d_contract.get("posture_vocabulary", [])):
        raise MB5ValidationError("Profile-D still encodes incorrect supine/left/right posture vocabulary")
    if "Lying" not in d_contract.get("posture_vocabulary", []) or "Sitting" not in d_contract.get("posture_vocabulary", []):
        # Runtime evidence wins; only enforce when TRAIN actually contains these values.
        if set(profile_d_meta["posture_vocabulary"]) >= {"Lying", "Sitting"}:
            raise MB5ValidationError("Profile-D posture vocabulary missing authoritative Lying/Sitting values")
    if d_contract.get("snr_available") is not False or d_contract.get("snr_source") != "NOT_AVAILABLE":
        raise MB5ValidationError("Profile-D must record snr_available=false / snr_source=NOT_AVAILABLE unless authoritative SNR exists")

    # Actual TFLite gates + independent VALIDATION recomputation.
    tflite_manifest_data = json.loads(
        (manifest_dir / "tflite_artifact_manifest.json").read_text(encoding="utf-8")
    ).get("tflite_artifacts", {})
    val_preds_npz = np.load(manifest_dir / "validation_predictions.npz")
    calib_res_data = json.loads(
        (manifest_dir / "calibration_results.json").read_text(encoding="utf-8")
    ).get("calibration_results", {})

    recomputed_calib_results = {}
    for prof_id in PROFILE_IDS:
        for seed in SHORTLIST_SEEDS:
            run_key = f"{mb4_pri_arch}_seed_{seed}_{prof_id}"
            if run_key not in tflite_manifest_data:
                raise MB5ValidationError(f"TFLite artifact entry '{run_key}' missing")
            tmeta = tflite_manifest_data[run_key]
            rel_p = tmeta.get("relative_path")
            full_tf_path = root_dir / rel_p
            if not full_tf_path.is_file():
                raise MB5ValidationError(f"Strict INT8 TFLite file missing: {rel_p}")

            tflite_bytes = full_tf_path.read_bytes()
            measured = inspect_tflite_model_bytes(tflite_bytes)
            if measured["bytes"] != full_tf_path.stat().st_size:
                raise MB5ValidationError(f"TFLite byte-size inconsistency for {run_key}")
            if measured["sha256"] != tmeta.get("sha256") or measured["bytes"] != tmeta.get("bytes"):
                raise MB5ValidationError(f"TFLite SHA/bytes mismatch versus measured artifact for {run_key}")
            if measured["input_dtype"] != "int8":
                raise MB5ValidationError(f"Actual TFLite input dtype mismatch for {run_key}: {measured['input_dtype']}")
            if measured["output_dtype"] != "int8":
                raise MB5ValidationError(f"Actual TFLite output dtype mismatch for {run_key}: {measured['output_dtype']}")
            if measured["select_tf_ops_count"] != 0:
                raise MB5ValidationError(f"M-B5_SELECT_TF_OPS_DETECTED for {run_key}")
            if list(tmeta.get("op_types", [])) != list(measured["op_types"]):
                raise MB5ValidationError(f"Actual operator inventory mismatch for {run_key}")
            if tmeta.get("input_dtype") != "int8" or tmeta.get("output_dtype") != "int8":
                raise MB5ValidationError(f"Manifest dtype strings disagree with strict INT8 for {run_key}")

            fl_probs = float_probs_by_seed[seed]
            fl_preds = float_preds_by_seed[seed]
            fl_cm = float_metrics_by_seed[seed]
            try:
                eval_res = evaluate_tflite_int8_model(tflite_bytes, val_x, val_y, fl_probs)
            except Exception as e:
                raise MB5ValidationError(f"M-B5_INT8_RUNTIME_FAILURE for {run_key}: {e}") from e

            if len(eval_res["int8_predictions"]) != 79:
                raise MB5ValidationError(f"INT8 prediction length != 79 for {run_key}")
            if run_key not in val_preds_npz.files:
                raise MB5ValidationError(f"Predictions for '{run_key}' missing from validation_predictions.npz")
            if not np.array_equal(eval_res["int8_predictions"], val_preds_npz[run_key]):
                raise MB5ValidationError(f"Stored validation prediction vector mismatch for {run_key}")

            # Independent saturation recompute from measured quantization params.
            in_scale = measured["input_scale"]
            in_zp = measured["input_zero_point"]
            sat_elems = 0
            sat_samples = 0
            total_elems = 0
            for i in range(len(val_x)):
                q_raw = np.round(val_x[i : i + 1] / in_scale + in_zp)
                sat_mask = (q_raw < -128) | (q_raw > 127)
                sat_cnt = int(np.sum(sat_mask))
                sat_elems += sat_cnt
                total_elems += q_raw.size
                if sat_cnt > 0:
                    sat_samples += 1
            sat_ratio = _round6(sat_elems / total_elems) if total_elems else 0.0
            if sat_ratio != eval_res["input_saturation_ratio"]:
                raise MB5ValidationError(f"Independent input saturation mismatch for {run_key}")

            per_class_rec_deg, max_pos_rec_degradation = compute_positive_recall_degradation(
                fl_cm, eval_res["class_metrics"]
            )
            new_collapse = detect_new_quantization_collapse(
                fl_preds, fl_cm, eval_res["int8_predictions"], eval_res["class_metrics"]
            )

            fl_macro_f1 = _round6(float(np.mean([fl_cm[c]["f1_score"] for c in LABEL_NAMES])))
            pos_f1_deg = _round6(max(0.0, fl_macro_f1 - eval_res["val_macro_f1"]))

            art_res = calib_res_data.get(run_key, {})
            art_diag = art_res.get("quantization_diagnostics", {})
            if _round6(art_res.get("int8_tflite", {}).get("macro_f1", -1.0)) != eval_res["val_macro_f1"]:
                raise MB5ValidationError(f"Macro F1 mismatch for {run_key}")
            if _round6(art_res.get("int8_tflite", {}).get("accuracy", -1.0)) != eval_res["val_accuracy"]:
                raise MB5ValidationError(f"Accuracy mismatch for {run_key}")
            if _round6(art_diag.get("positive_macro_f1_degradation", -1.0)) != pos_f1_deg:
                raise MB5ValidationError(f"Positive Macro F1 degradation mismatch for {run_key}")
            if art_diag.get("per_class_positive_recall_degradation") != per_class_rec_deg:
                raise MB5ValidationError(f"Per-class recall-degradation corruption for {run_key}")
            if _round6(art_diag.get("max_positive_recall_degradation", -1.0)) != max_pos_rec_degradation:
                raise MB5ValidationError(f"Max recall-degradation corruption for {run_key}")
            if bool(art_diag.get("new_class_collapse")) != bool(new_collapse):
                raise MB5ValidationError(f"New-collapse flag corruption for {run_key}")
            if _round6(art_diag.get("top1_agreement", -1.0)) != eval_res["top1_agreement"]:
                raise MB5ValidationError(f"Top-1 agreement mismatch for {run_key}")
            if _round6(art_diag.get("dequantized_output_mae", -1.0)) != eval_res["dequantized_output_mae"]:
                raise MB5ValidationError(f"Output MAE mismatch for {run_key}")
            if _round6(art_diag.get("input_saturation_ratio", -1.0)) != eval_res["input_saturation_ratio"]:
                raise MB5ValidationError(f"Input saturation ratio mismatch for {run_key}")
            if int(art_diag.get("saturated_sample_count", -1)) != int(eval_res["saturated_sample_count"]):
                raise MB5ValidationError(f"Saturated sample count mismatch for {run_key}")
            if _round6(art_diag.get("output_endpoint_ratio", -1.0)) != eval_res["output_endpoint_ratio"]:
                raise MB5ValidationError(f"Output endpoint ratio mismatch for {run_key}")

            recomputed_calib_results[run_key] = {
                "architecture_id": mb4_pri_arch,
                "seed": seed,
                "profile_id": prof_id,
                "conversion_success": True,
                "select_tf_ops_count": measured["select_tf_ops_count"],
                "strict_int8_eligible": True,
                "float_baseline": {"macro_f1": fl_macro_f1},
                "int8_tflite": {
                    "macro_f1": eval_res["val_macro_f1"],
                    "accuracy": eval_res["val_accuracy"],
                    "collapsed": eval_res["collapsed"],
                },
                "quantization_diagnostics": {
                    "positive_macro_f1_degradation": pos_f1_deg,
                    "per_class_positive_recall_degradation": per_class_rec_deg,
                    "max_positive_recall_degradation": max_pos_rec_degradation,
                    "top1_agreement": eval_res["top1_agreement"],
                    "dequantized_output_mae": eval_res["dequantized_output_mae"],
                    "input_saturation_ratio": eval_res["input_saturation_ratio"],
                    "output_endpoint_ratio": eval_res["output_endpoint_ratio"],
                    "new_class_collapse": new_collapse,
                },
            }

    cross_seed_data = json.loads(
        (manifest_dir / "cross_seed_calibration_results.json").read_text(encoding="utf-8")
    ).get("cross_seed_calibration_results", [])
    recomputed_cross_seed = []
    for prof_id in PROFILE_IDS:
        seed_runs = [recomputed_calib_results[f"{mb4_pri_arch}_seed_{s}_{prof_id}"] for s in SHORTLIST_SEEDS]
        conv_success = sum(1 for r in seed_runs if r["conversion_success"])
        strict_eligible = all(r["strict_int8_eligible"] for r in seed_runs)
        new_collapse_cnt = sum(1 for r in seed_runs if r["quantization_diagnostics"]["new_class_collapse"])
        pos_f1_degs = [r["quantization_diagnostics"]["positive_macro_f1_degradation"] for r in seed_runs]
        pos_rec_degs = [r["quantization_diagnostics"]["max_positive_recall_degradation"] for r in seed_runs]
        top1_agrees = [r["quantization_diagnostics"]["top1_agreement"] for r in seed_runs]
        output_maes = [r["quantization_diagnostics"]["dequantized_output_mae"] for r in seed_runs]
        input_sats = [r["quantization_diagnostics"]["input_saturation_ratio"] for r in seed_runs]
        output_ends = [r["quantization_diagnostics"]["output_endpoint_ratio"] for r in seed_runs]
        is_eligible = (conv_success == len(SHORTLIST_SEEDS)) and strict_eligible and (new_collapse_cnt == 0)
        recomputed_cross_seed.append({
            "profile_id": prof_id,
            "eligible": is_eligible,
            "conversion_success_count": conv_success,
            "strict_int8_eligible": strict_eligible,
            "new_class_collapse_count": new_collapse_cnt,
            "worst_positive_macro_f1_degradation": _round6(float(np.max(pos_f1_degs))),
            "worst_positive_recall_degradation": _round6(float(np.max(pos_rec_degs))),
            "min_top1_agreement": _round6(float(np.min(top1_agrees))),
            "max_dequantized_output_mae": _round6(float(np.max(output_maes))),
            "max_input_saturation_ratio": _round6(float(np.max(input_sats))),
            "max_output_endpoint_ratio": _round6(float(np.max(output_ends))),
        })

    if len(cross_seed_data) != len(recomputed_cross_seed):
        raise MB5ValidationError("Cross-seed aggregate count mismatch")
    for art_m, calc_m in zip(cross_seed_data, recomputed_cross_seed):
        if art_m.get("profile_id") != calc_m["profile_id"]:
            raise MB5ValidationError("Cross-seed aggregate profile ID mismatch")
        for fld in (
            "eligible",
            "conversion_success_count",
            "strict_int8_eligible",
            "new_class_collapse_count",
            "worst_positive_macro_f1_degradation",
            "worst_positive_recall_degradation",
            "min_top1_agreement",
            "max_dequantized_output_mae",
            "max_input_saturation_ratio",
            "max_output_endpoint_ratio",
        ):
            if art_m.get(fld) != calc_m.get(fld):
                raise MB5ValidationError(
                    f"Cross-seed aggregate field '{fld}' mismatch for {art_m.get('profile_id')}: "
                    f"manifest={art_m.get(fld)}, calc={calc_m.get(fld)}"
                )

    ranked_recomputed = rank_cross_seed_calibration_profiles(recomputed_cross_seed, eps=1e-5)
    ranking_decision = explain_ranking_decision(ranked_recomputed, eps=1e-5)
    exp_winner_id = ranked_recomputed[0]["profile_id"] if ranked_recomputed else None
    sel_data = json.loads((manifest_dir / "selected_calibration_profile.json").read_text(encoding="utf-8"))
    act_winner_id = sel_data.get("selected_calibration_profile")
    if act_winner_id != exp_winner_id:
        raise MB5ValidationError(f"Calibration profile selection mismatch: expected {exp_winner_id}, got {act_winner_id}")
    if ranked_recomputed and sel_data.get("ranking_decision", {}).get("deciding_criterion") != ranking_decision.get("deciding_criterion"):
        raise MB5ValidationError("Selected-profile ranking decision criterion mismatch")

    # Determinism audit must include selected-profile three-seed conversion replay evidence.
    det = json.loads((manifest_dir / "determinism_audit.json").read_text(encoding="utf-8"))
    if not det.get("profile_generation_deterministic"):
        raise MB5ValidationError("determinism_audit.json reports non-deterministic profile generation")
    if exp_winner_id:
        replay = det.get("selected_profile_three_seed_conversion_replay")
        if not isinstance(replay, dict):
            raise MB5ValidationError("determinism_audit.json missing selected_profile_three_seed_conversion_replay")
        if replay.get("selected_profile_id") != exp_winner_id:
            raise MB5ValidationError("Selected-profile replay profile_id mismatch")
        if not replay.get("functional_reproducibility_verified"):
            raise MB5ValidationError("Selected-profile three-seed conversion replay failed functional equality")
        for seed in SHORTLIST_SEEDS:
            seed_item = replay.get("seed_replays", {}).get(str(seed))
            if not seed_item or not seed_item.get("functional_equality"):
                raise MB5ValidationError(f"Selected-profile replay missing/failed for seed {seed}")
            checks = seed_item.get("checks", {})
            for required in (
                "prediction_vector_equal",
                "input_scale_equal",
                "input_zero_point_equal",
                "output_scale_equal",
                "output_zero_point_equal",
                "op_inventory_equal",
                "macro_f1_equal",
                "top1_agreement_equal",
                "output_mae_equal",
                "input_saturation_equal",
            ):
                if not checks.get(required):
                    raise MB5ValidationError(f"Selected-profile replay check '{required}' failed for seed {seed}")

    locked_data = json.loads((manifest_dir / "locked_test_access_audit.json").read_text(encoding="utf-8"))
    if locked_data.get("performance_access_attempts", -1) != 0 or not locked_data.get("lock_preserved"):
        raise MB5ValidationError("LOCKED_TEST_ACCESS_VIOLATION detected!")

    checksums_file = manifest_dir / "checksums.sha256"
    if not checksums_file.is_file():
        raise MB5ValidationError("checksums.sha256 missing")
    seen_entries = set()
    for line_num, line in enumerate(checksums_file.read_text(encoding="utf-8").splitlines(), 1):
        line_str = line.strip()
        if not line_str:
            continue
        parts = line_str.split(maxsplit=1)
        if len(parts) != 2:
            raise MB5ValidationError(f"Malformed checksum line {line_num}")
        digest, rel_name = parts[0].strip(), parts[1].strip()
        if not re.fullmatch(r"^[0-9a-fA-F]{64}$", digest):
            raise MB5ValidationError(f"Invalid SHA-256 digest format at line {line_num}: '{digest}'")
        if rel_name.startswith("/") or rel_name.startswith("\\") or "file://" in rel_name or "~" in rel_name or ".." in rel_name:
            raise MB5ValidationError(f"Path traversal or absolute path in checksums.sha256 line {line_num}: '{rel_name}'")
        if rel_name in seen_entries:
            raise MB5ValidationError(f"Duplicate file entry in checksums.sha256 at line {line_num}: '{rel_name}'")
        seen_entries.add(rel_name)
        target_f = (manifest_dir / rel_name).resolve()
        if not target_f.is_file():
            raise MB5ValidationError(f"Checksum target file missing: '{rel_name}'")
        if target_f.parent != manifest_dir.resolve():
            raise MB5ValidationError(f"Checksum target file escapes manifest directory: '{rel_name}'")
        actual_hash = hashlib.sha256(target_f.read_bytes()).hexdigest()
        if actual_hash != digest.lower():
            raise MB5ValidationError(f"Checksum mismatch for '{rel_name}'")

    missing_required = (REQUIRED_MB5_ARTIFACTS - {"checksums.sha256"}) - seen_entries
    if missing_required:
        raise MB5ValidationError(f"checksums.sha256 missing required M-B5 artifacts: {missing_required}")

    for manifest_f in manifest_dir.glob("*"):
        if manifest_f.suffix in (".json", ".jsonl"):
            content_str = manifest_f.read_text(encoding="utf-8")
            if "/Users/" in content_str or "file://" in content_str:
                raise MB5ValidationError(f"Absolute local path found in machine-readable artifact {manifest_f.name}")

    summary = json.loads((manifest_dir / "m_b5_summary.json").read_text(encoding="utf-8"))
    if summary.get("selected_calibration_profile") != act_winner_id:
        raise MB5ValidationError("m_b5_summary.json selected profile disagrees with selected_calibration_profile.json")
    if summary.get("neural_network_models_retrained", -1) != 0:
        raise MB5ValidationError("m_b5_summary.json must record neural_network_models_retrained=0")

    return {
        "validation_success": True,
        "m_b5_gate_status": "PASS_WITH_WARNINGS" if exp_winner_id else "INCONCLUSIVE",
        "m_b6_entry_status": "READY_WITH_CONDITIONS" if exp_winner_id else "NO",
        "independently_measured": {
            "pinned_environment_verified": True,
            "upstream_identity_chain_verified": True,
            "m_b0_gate_verified": True,
            "m_b1_gate_verified": True,
            "m_b2_gate_verified": True,
            "m_b3_gate_verified": True,
            "m_b4_gate_verified": True,
            "m_a5_gate_verified": True,
            "m_a6_gate_verified": True,
            "primary_float_finalist": mb4_pri_arch,
            "frozen_weight_seeds": SHORTLIST_SEEDS,
            "selected_calibration_profile": act_winner_id,
            "ranking_decision": ranking_decision,
            "profiles_evaluated": PROFILE_IDS,
            "strict_int8_conversions_verified": 12,
            "independent_recall_degradation_gate": True,
            "independent_new_collapse_gate": True,
            "actual_tflite_dtype_gate": True,
            "actual_tflite_operator_inventory_gate": True,
            "selected_profile_three_seed_conversion_replay": True if exp_winner_id else False,
            "locked_test_access_blocked": True,
            "hardened_checksum_verification": True,
            "train_windows": 327,
            "train_subjects": 77,
            "validation_windows": 79,
            "validation_subjects": 17,
        },
    }


def main() -> None:
    res = validate_m_b5_artifacts()
    print("Standalone M-B5 Representative Calibration Validation Result:")
    print(f"Validation Success: {res['validation_success']}")
    print(f"M-B5 Gate Status: {res['m_b5_gate_status']}")
    print(f"M-B6 Entry Status: {res['m_b6_entry_status']}")
    print(f"Primary Float Finalist: {res['independently_measured']['primary_float_finalist']}")
    print(f"Selected Calibration Profile: {res['independently_measured']['selected_calibration_profile']}")
    print(f"Ranking Decision: {res['independently_measured']['ranking_decision'].get('deciding_criterion')}")
    print(f"LOCKED_TEST Guard Verified: {res['independently_measured']['locked_test_access_blocked']}")
    print(f"Hardened Checksums Verified: {res['independently_measured']['hardened_checksum_verification']}")


if __name__ == "__main__":
    main()
