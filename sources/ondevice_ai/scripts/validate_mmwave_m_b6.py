# SafeNest mmWave Track — Phase M-B6 Standalone Validator (Hardened Evidence-Truth)

import hashlib
import json
import os
import re
import sys
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional

import scipy
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from mmwave_m_b1_preprocessing import fit_train_zscore_statistics, transform_signals
from mmwave_m_b2_imbalance import LABEL_NAMES, compute_one_vs_rest_false_positives, compute_subject_level_diagnostics
from mmwave_m_b3_architecture import build_model_by_id, compute_numerical_weights_sha256
from mmwave_m_b6_equivalence import (
    SHORTLIST_SEEDS,
    compute_pairwise_equivalence,
    evaluate_tflite_float32_model,
    evaluate_tflite_int8_model_full,
)
from mmwave_phase_b_access import PhaseBAccessGuard
from validate_mmwave_full_conversion import validate_full_conversion_artifacts
from validate_mmwave_m_b0 import validate_m_b0_artifacts
from validate_mmwave_m_b1 import validate_m_b1_artifacts
from validate_mmwave_m_b2 import validate_m_b2_artifacts
from validate_mmwave_m_b3 import validate_m_b3_artifacts
from validate_mmwave_m_b4 import validate_m_b4_artifacts
from validate_mmwave_m_b5 import validate_m_b5_artifacts
from validate_mmwave_subject_split import validate_a5


class MB6ValidationError(Exception):
    """Raised when Phase M-B6 validation fails."""
    pass


REQUIRED_MB6_ARTIFACTS = {
    "input_identity.json",
    "experiment_contract.json",
    "stage_artifact_manifest.json",
    "stage_conversion_runs.json",
    "keras_predictions.npz",
    "float_tflite_predictions.npz",
    "int8_tflite_predictions.npz",
    "validation_prediction_index.jsonl",
    "per_seed_stage_metrics.json",
    "pairwise_equivalence_metrics.json",
    "cross_seed_equivalence_summary.json",
    "subject_level_stage_metrics.json",
    "mismatch_samples.jsonl",
    "quantization_diagnostics.json",
    "class_collapse_transition_audit.json",
    "locked_test_access_audit.json",
    "run_environment.json",
    "exceptions.json",
    "m_b6_summary.json",
    "checksums.sha256",
}


def inspect_tflite_structure(tflite_bytes: bytes) -> Dict[str, Any]:
    """Instantiate TFLite interpreter and inspect dtypes, shapes, and op inventory."""
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()

    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]

    op_details = interpreter._get_ops_details()
    op_types = [op["op_name"] for op in op_details]
    select_tf_ops_count = sum(1 for t in op_types if "Flex" in t or "Select" in t)

    return {
        "input_dtype": str(in_details["dtype"].__name__),
        "output_dtype": str(out_details["dtype"].__name__),
        "input_shape": [int(x) for x in in_details["shape"]],
        "output_shape": [int(x) for x in out_details["shape"]],
        "input_scale": float(in_details["quantization"][0]),
        "input_zero_point": int(in_details["quantization"][1]),
        "output_scale": float(out_details["quantization"][0]),
        "output_zero_point": int(out_details["quantization"][1]),
        "op_types": op_types,
        "select_tf_ops_count": select_tf_ops_count,
    }


def validate_m_b6_artifacts(
    root_dir: Path = ROOT_DIR,
    manifest_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Independently validate all Phase M-B6 stage-equivalence artifacts without retraining."""
    if manifest_dir is None:
        manifest_dir = root_dir / "datasets/mmwave/manifests/M-B6_stage_equivalence"

    if not manifest_dir.is_dir():
        raise MB6ValidationError(f"M-B6 manifest directory missing: {manifest_dir}")

    guard = PhaseBAccessGuard(root_dir=root_dir)

    # 1. Verify Pinned Environment
    actual_tf = tf.__version__
    actual_np = np.__version__
    actual_scipy = scipy.__version__

    req_file = root_dir / "requirements-mac.txt"
    if not req_file.is_file():
        raise MB6ValidationError("requirements-mac.txt missing!")
    req_sha = hashlib.sha256(req_file.read_bytes()).hexdigest()

    env_file = manifest_dir / "run_environment.json"
    if not env_file.is_file():
        raise MB6ValidationError("run_environment.json missing!")
    env_data = json.loads(env_file.read_text(encoding="utf-8"))

    if env_data.get("tensorflow_version") != actual_tf or env_data.get("numpy_version") != actual_np or env_data.get("scipy_version") != actual_scipy:
        raise MB6ValidationError(
            f"Environment mismatch: manifest TF/NP/SciPy={env_data.get('tensorflow_version')}/{env_data.get('numpy_version')}/{env_data.get('scipy_version')}, actual={actual_tf}/{actual_np}/{actual_scipy}"
        )
    if env_data.get("requirements_mac_sha256") != req_sha:
        raise MB6ValidationError("requirements-mac.txt SHA-256 mismatch!")

    # 2. Invoke Upstream Standalone Validators (M-B0..M-B5, A5, A6)
    if not validate_m_b0_artifacts(root_dir=root_dir).get("validation_success"):
        raise MB6ValidationError("Upstream M-B0 validation failed!")
    if not validate_m_b1_artifacts(root_dir=root_dir).get("validation_success"):
        raise MB6ValidationError("Upstream M-B1 validation failed!")
    if not validate_m_b2_artifacts(root_dir=root_dir).get("validation_success"):
        raise MB6ValidationError("Upstream M-B2 validation failed!")
    if not validate_m_b3_artifacts(root_dir=root_dir).get("validation_success"):
        raise MB6ValidationError("Upstream M-B3 validation failed!")
    if not validate_m_b4_artifacts(root_dir=root_dir).get("validation_success"):
        raise MB6ValidationError("Upstream M-B4 validation failed!")
    if not validate_m_b5_artifacts(root_dir=root_dir).get("validation_success"):
        raise MB6ValidationError("Upstream M-B5 validation failed!")

    # Section I: Invoke authoritative A5 validator
    a5_dir = root_dir / "datasets/mmwave/manifests/a5_subject_split"
    def load_json_helper(p):
        return json.loads(p.read_text(encoding="utf-8"))
    def load_jsonl_helper(p):
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    split_contract_file = root_dir / "datasets/mmwave/splits/mmwave_real_subject_split_v1.json"
    a5_errors = validate_a5(
        load_jsonl_helper(root_dir / "datasets/mmwave/manifests/a0_raw_inventory/recording_index.jsonl"),
        load_jsonl_helper(root_dir / "datasets/mmwave/manifests/a4_label_pilot/window_label_manifest.jsonl"),
        load_json_helper(a5_dir / "split_profile.json"),
        load_jsonl_helper(a5_dir / "subject_split_manifest.jsonl"),
        load_jsonl_helper(a5_dir / "recording_split_manifest.jsonl"),
        load_jsonl_helper(a5_dir / "pilot_window_split_manifest.jsonl"),
        load_json_helper(a5_dir / "provenance_schema.json"),
        load_json_helper(a5_dir / "split_balance_report.json"),
        load_json_helper(a5_dir / "exceptions.json"),
        load_json_helper(split_contract_file),
        verify_checksum_file=True,
        output=a5_dir,
        split_output=split_contract_file,
    )
    if a5_errors:
        raise MB6ValidationError(f"Upstream A5 validation errors detected: {a5_errors}")

    if not validate_full_conversion_artifacts(root_dir=root_dir).get("validation_success"):
        raise MB6ValidationError("Upstream A6 validation failed!")

    # 3. Verify Upstream Input Identity Chain
    input_identity_file = manifest_dir / "input_identity.json"
    if not input_identity_file.is_file():
        raise MB6ValidationError("input_identity.json missing!")
    input_identity_data = json.loads(input_identity_file.read_text(encoding="utf-8"))
    inputs_list = input_identity_data.get("inputs", [])

    if len(inputs_list) < 25:
        raise MB6ValidationError(f"input_identity.json must contain at least 25 upstream files, got {len(inputs_list)}")

    for input_item in inputs_list:
        rel_p = input_item.get("path")
        exp_sha = input_item.get("measured_sha256")
        if not rel_p or not exp_sha:
            raise MB6ValidationError(f"Malformed input_identity item: {input_item}")
        full_p = root_dir / rel_p
        if not full_p.is_file():
            raise MB6ValidationError(f"Upstream identity file missing from checkout: {rel_p}")
        act_sha = hashlib.sha256(full_p.read_bytes()).hexdigest()
        if act_sha != exp_sha:
            raise MB6ValidationError(f"Upstream identity SHA mismatch for '{rel_p}': expected {exp_sha}, got {act_sha}")

    # 4. Verify Datasets & Authoritative Subject Counts
    train_data = guard.get_model_selection_dataset("TRAIN")
    val_data = guard.get_model_selection_dataset("VALIDATION")

    act_train_subjs = len(set(w["subject_id"] for w in train_data["windows"]))
    act_val_subjs = len(set(w["subject_id"] for w in val_data["windows"]))

    exp_contract_file = manifest_dir / "experiment_contract.json"
    if not exp_contract_file.is_file():
        raise MB6ValidationError("experiment_contract.json missing!")
    exp_contract_data = json.loads(exp_contract_file.read_text(encoding="utf-8"))

    if exp_contract_data.get("train_subjects") != act_train_subjs or act_train_subjs != 77:
        raise MB6ValidationError(f"TRAIN subject count mismatch: manifest={exp_contract_data.get('train_subjects')}, actual={act_train_subjs}")
    if exp_contract_data.get("eval_subjects") != act_val_subjs or act_val_subjs != 17:
        raise MB6ValidationError(f"VALIDATION subject count mismatch: manifest={exp_contract_data.get('eval_subjects')}, actual={act_val_subjs}")

    val_y = np.array([w["safenest_label_id"] for w in val_data["windows"]], dtype=int)

    # 5. Verify Frozen M-B4 Primary Architecture & Seed Weights
    mb4_pri_file = root_dir / "datasets/mmwave/manifests/M-B4_multiseed_stability/primary_float_finalist.json"
    mb4_pri_arch = json.loads(mb4_pri_file.read_text(encoding="utf-8")).get("primary_stable_float_finalist")
    if mb4_pri_arch != "M-B3_CONV1D_GAP_BASELINE":
        raise MB6ValidationError(f"M-B4 primary float finalist mismatch: expected M-B3_CONV1D_GAP_BASELINE, got {mb4_pri_arch}")

    mb4_tr_file = root_dir / "datasets/mmwave/manifests/M-B4_multiseed_stability/training_runs.json"
    mb4_tr_data = json.loads(mb4_tr_file.read_text(encoding="utf-8")).get("training_runs", {})

    mb4_weights_file = root_dir / "datasets/mmwave/manifests/M-B4_multiseed_stability/seed_weights.npz"
    mb4_weights = np.load(mb4_weights_file)

    # Section H: Prove M-B5 selected INT8 reuse identity
    mb5_sel_file = root_dir / "datasets/mmwave/manifests/M-B5_representative_calibration/selected_calibration_profile.json"
    mb5_sel_prof = json.loads(mb5_sel_file.read_text(encoding="utf-8")).get("selected_calibration_profile")
    if mb5_sel_prof != "M-B5_CAL_CLASS_BALANCED_120":
        raise MB6ValidationError(f"M-B5 selected calibration profile mismatch: expected M-B5_CAL_CLASS_BALANCED_120, got {mb5_sel_prof}")

    mb5_art_file = root_dir / "datasets/mmwave/manifests/M-B5_representative_calibration/tflite_artifact_manifest.json"
    mb5_artifacts_data = json.loads(mb5_art_file.read_text(encoding="utf-8")).get("tflite_artifacts", {})

    zstats = fit_train_zscore_statistics(train_data["signals"], detrend=False, bpf=True)
    val_x_float32 = transform_signals(val_data["signals"], detrend=False, bpf=True, zscore=True, zscore_stats=zstats).astype(np.float32)
    val_x_3d = np.expand_dims(val_x_float32, axis=-1)

    stage_artifacts_file = manifest_dir / "stage_artifact_manifest.json"
    if not stage_artifacts_file.is_file():
        raise MB6ValidationError("stage_artifact_manifest.json missing!")
    stage_artifacts_data = json.loads(stage_artifacts_file.read_text(encoding="utf-8")).get("artifacts", {})

    keras_preds_npz = np.load(manifest_dir / "keras_predictions.npz")
    float_tflite_preds_npz = np.load(manifest_dir / "float_tflite_predictions.npz")
    int8_tflite_preds_npz = np.load(manifest_dir / "int8_tflite_predictions.npz")

    per_seed_metrics_file = manifest_dir / "per_seed_stage_metrics.json"
    if not per_seed_metrics_file.is_file():
        raise MB6ValidationError("per_seed_stage_metrics.json missing!")
    per_seed_metrics_manifest = json.loads(per_seed_metrics_file.read_text(encoding="utf-8")).get("per_seed_stage_metrics", {})

    pairwise_file = manifest_dir / "pairwise_equivalence_metrics.json"
    if not pairwise_file.is_file():
        raise MB6ValidationError("pairwise_equivalence_metrics.json missing!")
    pairwise_data = json.loads(pairwise_file.read_text(encoding="utf-8")).get("pairwise_equivalence", {})

    quant_diag_file = manifest_dir / "quantization_diagnostics.json"
    if not quant_diag_file.is_file():
        raise MB6ValidationError("quantization_diagnostics.json missing!")
    quant_diag_manifest = json.loads(quant_diag_file.read_text(encoding="utf-8")).get("quantization_diagnostics", {})

    recomputed_pairwise_dict = {}
    recomputed_per_seed_metrics = {}
    recomputed_quant_diag = {}
    recomputed_collapse_transitions = {}

    for seed in SHORTLIST_SEEDS:
        run_key = f"{mb4_pri_arch}_seed_{seed}"
        if run_key not in mb4_tr_data:
            raise MB6ValidationError(f"Training run '{run_key}' missing from M-B4 training_runs.json")

        # --- Stage A Verification ---
        model = build_model_by_id(mb4_pri_arch)
        arch_w_keys = sorted(
            [k for k in mb4_weights.files if k.startswith(f"{mb4_pri_arch}_seed_{seed}_layer_weight_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        arch_w_list = [mb4_weights[k] for k in arch_w_keys]
        model.set_weights(arch_w_list)

        computed_sha = compute_numerical_weights_sha256(model)
        exp_sha = mb4_tr_data[run_key]["final_weights_sha256"]
        if computed_sha != exp_sha:
            raise MB6ValidationError(f"M-B6_UPSTREAM_FLOAT_IDENTITY_MISMATCH for seed {seed}: computed ({computed_sha}) != M-B4 ({exp_sha})")

        probs_a = model.predict(val_x_3d, verbose=0).astype(np.float32)
        preds_a = np.argmax(probs_a, axis=1).astype(int)

        if not np.array_equal(preds_a, keras_preds_npz[run_key]):
            raise MB6ValidationError(f"Keras prediction vector mismatch for {run_key}")

        cm_a = compute_one_vs_rest_false_positives(val_y, preds_a)
        f1_a = float(np.mean([cm_a[c]["f1_score"] for c in LABEL_NAMES]))
        acc_a = float(np.mean(preds_a == val_y))
        col_a = (cm_a["APNEA"]["recall"] == 0.0) or (cm_a["RAPID_OR_ABNORMAL"]["recall"] == 0.0) or (len(np.unique(preds_a)) < 3)

        # --- Stage B (Float TFLite) Verification ---
        b_key = f"{run_key}_stage_b"
        if b_key not in stage_artifacts_data:
            raise MB6ValidationError(f"Stage B artifact entry missing for {b_key}")
        b_meta = stage_artifacts_data[b_key]
        b_file_path = root_dir / b_meta["relative_path"]

        if not b_file_path.is_file():
            raise MB6ValidationError(f"Float TFLite file missing: {b_meta['relative_path']}")
        if b_file_path.stat().st_size != b_meta["bytes"]:
            raise MB6ValidationError(f"Float TFLite size mismatch for {run_key}")
        if hashlib.sha256(b_file_path.read_bytes()).hexdigest() != b_meta["sha256"]:
            raise MB6ValidationError(f"Float TFLite SHA mismatch for {run_key}")

        # Section G: Inspect actual TFLite structure
        b_struct = inspect_tflite_structure(b_file_path.read_bytes())
        if b_struct["input_dtype"] != "float32" or b_struct["output_dtype"] != "float32":
            raise MB6ValidationError(f"Float TFLite actual dtype mismatch for {run_key}: input={b_struct['input_dtype']}, output={b_struct['output_dtype']}")
        if b_struct["select_tf_ops_count"] > 0 or b_meta.get("select_tf_ops_count", 0) > 0:
            raise MB6ValidationError(f"Select TF Ops detected in Float TFLite for {run_key}")

        if b_meta["input_dtype"] != b_struct["input_dtype"] or b_meta["output_dtype"] != b_struct["output_dtype"]:
            raise MB6ValidationError(f"Stage B manifest vs actual dtype mismatch for {run_key}")

        preds_b, probs_b = evaluate_tflite_float32_model(b_file_path.read_bytes(), val_x_3d)
        if not np.array_equal(preds_b, float_tflite_preds_npz[run_key]):
            raise MB6ValidationError(f"Float TFLite prediction vector mismatch for {run_key}")

        cm_b = compute_one_vs_rest_false_positives(val_y, preds_b)
        f1_b = float(np.mean([cm_b[c]["f1_score"] for c in LABEL_NAMES]))
        acc_b = float(np.mean(preds_b == val_y))
        col_b = (cm_b["APNEA"]["recall"] == 0.0) or (cm_b["RAPID_OR_ABNORMAL"]["recall"] == 0.0) or (len(np.unique(preds_b)) < 3)

        # --- Stage C (Strict INT8 TFLite) Verification ---
        c_key = f"{run_key}_stage_c"
        if c_key not in stage_artifacts_data:
            raise MB6ValidationError(f"Stage C artifact entry missing for {c_key}")
        c_meta = stage_artifacts_data[c_key]
        c_file_path = root_dir / c_meta["relative_path"]

        if not c_file_path.is_file():
            raise MB6ValidationError(f"Strict INT8 TFLite file missing: {c_meta['relative_path']}")
        if c_file_path.stat().st_size != c_meta["bytes"]:
            raise MB6ValidationError(f"Strict INT8 file size mismatch for {run_key}")
        if hashlib.sha256(c_file_path.read_bytes()).hexdigest() != c_meta["sha256"]:
            raise MB6ValidationError(f"Strict INT8 SHA mismatch for {run_key}")

        # Section G: Inspect actual TFLite structure
        c_struct = inspect_tflite_structure(c_file_path.read_bytes())
        if c_struct["input_dtype"] != "int8" or c_struct["output_dtype"] != "int8":
            raise MB6ValidationError(f"Strict INT8 actual dtype mismatch for {run_key}: input={c_struct['input_dtype']}, output={c_struct['output_dtype']}")
        if c_struct["select_tf_ops_count"] > 0 or c_meta.get("select_tf_ops_count", 0) > 0:
            raise MB6ValidationError(f"Select TF Ops detected in Strict INT8 structure for {run_key}")

        if c_meta["input_dtype"] != c_struct["input_dtype"] or c_meta["output_dtype"] != c_struct["output_dtype"]:
            raise MB6ValidationError(f"Stage C manifest vs actual dtype mismatch for {run_key}")

        # Section H: Prove M-B5 selected INT8 reuse identity
        mb5_key = f"{mb4_pri_arch}_seed_{seed}_{mb5_sel_prof}"
        if mb5_key not in mb5_artifacts_data:
            raise MB6ValidationError(f"M-B5 artifact key '{mb5_key}' missing from M-B5 manifest")
        mb5_art_meta = mb5_artifacts_data[mb5_key]

        if c_meta["sha256"] != mb5_art_meta["sha256"]:
            raise MB6ValidationError(f"M-B5 selected INT8 SHA mismatch for {run_key}: M-B6={c_meta['sha256']}, M-B5={mb5_art_meta['sha256']}")
        if c_meta["bytes"] != mb5_art_meta["bytes"]:
            raise MB6ValidationError(f"M-B5 selected INT8 byte size mismatch for {run_key}")
        if not c_meta.get("m_b5_selected_int8_reused"):
            raise MB6ValidationError(f"m_b5_selected_int8_reused flag must be True for {run_key}")

        eval_c = evaluate_tflite_int8_model_full(c_file_path.read_bytes(), val_x_3d, val_y, float_probs=probs_a)
        preds_c = eval_c["predictions"]
        probs_c = eval_c["probabilities"]

        if not np.array_equal(preds_c, int8_tflite_preds_npz[run_key]):
            raise MB6ValidationError(f"Strict INT8 prediction vector mismatch for {run_key}")

        cm_c = eval_c["class_metrics"]
        f1_c = eval_c["macro_f1"]
        acc_c = eval_c["accuracy"]
        col_c = (cm_c["APNEA"]["recall"] == 0.0) or (cm_c["RAPID_OR_ABNORMAL"]["recall"] == 0.0) or (len(np.unique(preds_c)) < 3)

        # Section C: Per-seed stage metrics independent validation
        calc_per_seed = {
            "seed": seed,
            "stage_a_float_keras": {"macro_f1": f1_a, "accuracy": acc_a, "collapsed": col_a, "class_metrics": cm_a},
            "stage_b_float_tflite": {"macro_f1": f1_b, "accuracy": acc_b, "collapsed": col_b, "class_metrics": cm_b},
            "stage_c_int8_tflite": {"macro_f1": f1_c, "accuracy": acc_c, "collapsed": col_c, "class_metrics": cm_c},
        }
        recomputed_per_seed_metrics[run_key] = calc_per_seed

        manifest_ps = per_seed_metrics_manifest.get(run_key, {})
        for stg_id in ("stage_a_float_keras", "stage_b_float_tflite", "stage_c_int8_tflite"):
            m_stg = manifest_ps.get(stg_id, {})
            c_stg = calc_per_seed[stg_id]
            for fld in ("macro_f1", "accuracy", "collapsed"):
                if m_stg.get(fld) != c_stg.get(fld):
                    raise MB6ValidationError(f"per_seed_stage_metrics '{stg_id}.{fld}' mismatch for {run_key}: manifest={m_stg.get(fld)}, calc={c_stg.get(fld)}")

            m_cm_map = m_stg.get("class_metrics", {})
            c_cm_map = c_stg["class_metrics"]
            for cname in LABEL_NAMES:
                m_cm = m_cm_map.get(cname, {})
                c_cm = c_cm_map.get(cname, {})
                for fld in ("support", "tp", "fp", "tn", "fn", "precision", "recall", "f1_score", "fpr"):
                    if m_cm.get(fld) != c_cm.get(fld):
                        raise MB6ValidationError(f"per_seed_stage_metrics '{stg_id}.class_metrics.{cname}.{fld}' mismatch for {run_key}: manifest={m_cm.get(fld)}, calc={c_cm.get(fld)}")

        # Section E: Class collapse transition audit
        new_col_ab = (not col_a) and col_b
        new_col_bc = (not col_b) and col_c
        new_col_ac = (not col_a) and col_c
        calc_collapse_entry = {
            "seed": seed,
            "stage_a_collapsed": col_a,
            "stage_b_collapsed": col_b,
            "stage_c_collapsed": col_c,
            "new_collapse_a_to_b": new_col_ab,
            "new_collapse_b_to_c": new_col_bc,
            "new_collapse_a_to_c": new_col_ac,
            "transition_label": f"A({col_a}) -> B({col_b}) -> C({col_c})",
        }
        recomputed_collapse_transitions[run_key] = calc_collapse_entry

        # Section F: Quantization diagnostics verification
        calc_qd = {
            "seed": seed,
            "input_scale": eval_c["input_scale"],
            "input_zero_point": eval_c["input_zero_point"],
            "output_scale": eval_c["output_scale"],
            "output_zero_point": eval_c["output_zero_point"],
            "input_saturation_ratio": eval_c["input_saturation_ratio"],
            "saturated_sample_count": eval_c["saturated_sample_count"],
            "output_endpoint_ratio": eval_c["output_endpoint_ratio"],
        }
        recomputed_quant_diag[run_key] = calc_qd

        manifest_qd = quant_diag_manifest.get(run_key, {})
        for fld in ("input_scale", "input_zero_point", "output_scale", "output_zero_point", "input_saturation_ratio", "saturated_sample_count", "output_endpoint_ratio"):
            if manifest_qd.get(fld) != calc_qd.get(fld):
                raise MB6ValidationError(f"quantization_diagnostics field '{fld}' mismatch for {run_key}: manifest={manifest_qd.get(fld)}, calc={calc_qd.get(fld)}")

        # --- Recompute Pairwise Equivalence ---
        pair_a_b = compute_pairwise_equivalence(preds_a, probs_a, preds_b, probs_b, val_y, val_data["windows"], "Stage A (Float Keras)", "Stage B (Float TFLite)")
        pair_b_c = compute_pairwise_equivalence(preds_b, probs_b, preds_c, probs_c, val_y, val_data["windows"], "Stage B (Float TFLite)", "Stage C (Strict INT8 TFLite)")
        pair_a_c = compute_pairwise_equivalence(preds_a, probs_a, preds_c, probs_c, val_y, val_data["windows"], "Stage A (Float Keras)", "Stage C (Strict INT8 TFLite)")

        recomputed_pairwise_dict[run_key] = {
            "seed": seed,
            "a_to_b": {k: v for k, v in pair_a_b.items() if k != "mismatch_samples"},
            "b_to_c": {k: v for k, v in pair_b_c.items() if k != "mismatch_samples"},
            "a_to_c": {k: v for k, v in pair_a_c.items() if k != "mismatch_samples"},
        }

        # Compare 1:1 against pairwise_equivalence_metrics.json
        art_pair = pairwise_data.get(run_key, {})
        for p_sub in ("a_to_b", "b_to_c", "a_to_c"):
            art_sub = art_pair.get(p_sub, {})
            calc_sub = recomputed_pairwise_dict[run_key][p_sub]
            for fld in ("top1_agreement", "mismatch_count", "output_probability_mae", "output_probability_rmse", "positive_macro_f1_degradation", "max_positive_recall_degradation"):
                if art_sub.get(fld) != calc_sub.get(fld):
                    raise MB6ValidationError(f"Pairwise field '{p_sub}.{fld}' mismatch for {run_key}: manifest={art_sub.get(fld)}, calc={calc_sub.get(fld)}")

    # Section D: Independently validate cross-seed summaries
    cross_seed_file = manifest_dir / "cross_seed_equivalence_summary.json"
    if not cross_seed_file.is_file():
        raise MB6ValidationError("cross_seed_equivalence_summary.json missing!")
    cross_seed_manifest = json.loads(cross_seed_file.read_text(encoding="utf-8"))

    for pair_k in ("a_to_b", "b_to_c", "a_to_c"):
        runs = [recomputed_pairwise_dict[f"{mb4_pri_arch}_seed_{s}"][pair_k] for s in SHORTLIST_SEEDS]
        top1_agrees = [r["top1_agreement"] for r in runs]
        f1_degs = [r["positive_macro_f1_degradation"] for r in runs]
        rec_degs = [r["max_positive_recall_degradation"] for r in runs]
        maes = [r["output_probability_mae"] for r in runs]

        worst_seed_idx = int(np.argmax(f1_degs))
        worst_seed = SHORTLIST_SEEDS[worst_seed_idx]

        calc_summary = {
            "min_top1_agreement": round(float(np.min(top1_agrees)), 6),
            "mean_top1_agreement": round(float(np.mean(top1_agrees)), 6),
            "max_top1_agreement": round(float(np.max(top1_agrees)), 6),
            "worst_macro_f1_degradation": round(float(np.max(f1_degs)), 6),
            "mean_macro_f1_degradation": round(float(np.mean(f1_degs)), 6),
            "worst_recall_degradation": round(float(np.max(rec_degs)), 6),
            "mean_recall_degradation": round(float(np.mean(rec_degs)), 6),
            "maximum_output_probability_mae": round(float(np.max(maes)), 6),
            "mean_output_probability_mae": round(float(np.mean(maes)), 6),
            "worst_seed": worst_seed,
            "per_seed": {str(s): runs[i] for i, s in enumerate(SHORTLIST_SEEDS)},
        }

        m_summary = cross_seed_manifest.get(f"cross_seed_{pair_k}", {})
        for fld in ("min_top1_agreement", "mean_top1_agreement", "max_top1_agreement", "worst_macro_f1_degradation", "mean_macro_f1_degradation", "worst_recall_degradation", "mean_recall_degradation", "maximum_output_probability_mae", "mean_output_probability_mae", "worst_seed"):
            if m_summary.get(fld) != calc_summary.get(fld):
                raise MB6ValidationError(f"cross_seed_equivalence_summary field 'cross_seed_{pair_k}.{fld}' mismatch: manifest={m_summary.get(fld)}, calc={calc_summary.get(fld)}")

    # Section E: Class collapse transition audit verification
    collapse_file = manifest_dir / "class_collapse_transition_audit.json"
    if not collapse_file.is_file():
        raise MB6ValidationError("class_collapse_transition_audit.json missing!")
    collapse_data = json.loads(collapse_file.read_text(encoding="utf-8")).get("class_collapse_transitions", {})

    for seed in SHORTLIST_SEEDS:
        run_key = f"{mb4_pri_arch}_seed_{seed}"
        m_entry = collapse_data.get(run_key, {})
        c_entry = recomputed_collapse_transitions[run_key]
        for fld in ("stage_a_collapsed", "stage_b_collapsed", "stage_c_collapsed", "new_collapse_a_to_b", "new_collapse_b_to_c", "new_collapse_a_to_c"):
            if m_entry.get(fld) != c_entry.get(fld):
                raise MB6ValidationError(f"class_collapse_transition_audit field '{fld}' mismatch for {run_key}: manifest={m_entry.get(fld)}, calc={c_entry.get(fld)}")
        if c_entry.get("new_collapse_a_to_b") or c_entry.get("new_collapse_b_to_c") or c_entry.get("new_collapse_a_to_c"):
            raise MB6ValidationError(f"M-B6_NEW_CLASS_COLLAPSE detected for {run_key}!")

    # 6. Verify Subject-Level Metrics (3 seeds x 17 subjects x 3 stages = 153 entries)
    subj_file = manifest_dir / "subject_level_stage_metrics.json"
    if not subj_file.is_file():
        raise MB6ValidationError("subject_level_stage_metrics.json missing!")
    subj_data = json.loads(subj_file.read_text(encoding="utf-8")).get("subject_level_stage_metrics", {})

    for seed in SHORTLIST_SEEDS:
        run_key = f"{mb4_pri_arch}_seed_{seed}"
        if run_key not in subj_data:
            raise MB6ValidationError(f"Subject metrics missing for {run_key}")

        preds_a = keras_preds_npz[run_key]
        preds_b = float_tflite_preds_npz[run_key]
        preds_c = int8_tflite_preds_npz[run_key]

        calc_diag_a = compute_subject_level_diagnostics(val_data["windows"], preds_a)
        calc_diag_b = compute_subject_level_diagnostics(val_data["windows"], preds_b)
        calc_diag_c = compute_subject_level_diagnostics(val_data["windows"], preds_c)

        art_stages = subj_data[run_key]
        for stg_k, calc_d in (("stage_a", calc_diag_a), ("stage_b", calc_diag_b), ("stage_c", calc_diag_c)):
            art_per_s = art_stages.get(stg_k, {}).get("per_subject", {})
            calc_per_s = calc_d.get("per_subject", {})

            if len(art_per_s) != 17 or len(calc_per_s) != 17:
                raise MB6ValidationError(f"Subject count mismatch for {run_key} {stg_k}: art={len(art_per_s)}, calc={len(calc_per_s)}")

            for sid, calc_s in calc_per_s.items():
                art_s = art_per_s.get(sid, {})
                for k_stat in ("window_count", "accuracy", "subject_macro_f1", "apnea_fp", "apnea_fn", "rapid_fp", "rapid_fn", "prediction_distribution"):
                    if art_s.get(k_stat) != calc_s.get(k_stat):
                        raise MB6ValidationError(f"Subject {sid} stat '{k_stat}' mismatch for {run_key} {stg_k}: manifest={art_s.get(k_stat)}, calc={calc_s.get(k_stat)}")

                # Full per-class subject metrics comparison
                art_cm_map = art_s.get("class_metrics", {})
                calc_cm_map = calc_s.get("class_metrics", {})

                for cname in LABEL_NAMES:
                    art_cm = art_cm_map.get(cname, {})
                    calc_cm = calc_cm_map.get(cname, {})
                    for fld in ("support", "tp", "fp", "tn", "fn", "recall", "precision", "f1"):
                        if art_cm.get(fld) != calc_cm.get(fld):
                            raise MB6ValidationError(f"Subject {sid} class_metrics field '{cname}.{fld}' mismatch for {run_key} {stg_k}: manifest={art_cm.get(fld)}, calc={calc_cm.get(fld)}")

    # 8. Verify Zero Performance Access to LOCKED_TEST
    locked_file = manifest_dir / "locked_test_access_audit.json"
    if not locked_file.is_file():
        raise MB6ValidationError("locked_test_access_audit.json missing!")
    locked_data = json.loads(locked_file.read_text(encoding="utf-8"))

    if locked_data.get("performance_access_attempts", -1) != 0 or not locked_data.get("lock_preserved"):
        raise MB6ValidationError("LOCKED_TEST_ACCESS_VIOLATION detected!")

    # 9. HARDENED CHECKSUM MANIFEST VALIDATION
    checksums_file = manifest_dir / "checksums.sha256"
    if not checksums_file.is_file():
        raise MB6ValidationError(f"checksums.sha256 missing: {checksums_file}")

    raw_lines = checksums_file.read_text(encoding="utf-8").splitlines()
    seen_entries = set()

    for line_num, line in enumerate(raw_lines, 1):
        line_str = line.strip()
        if not line_str:
            continue

        parts = line_str.split(maxsplit=1)
        if len(parts) != 2:
            raise MB6ValidationError(f"Malformed checksum line {line_num} in checksums.sha256: '{line}'")

        digest, rel_name = parts[0].strip(), parts[1].strip()

        if not re.fullmatch(r"^[0-9a-fA-F]{64}$", digest):
            raise MB6ValidationError(f"Invalid SHA-256 digest format at line {line_num}: '{digest}'")

        if rel_name.startswith("/") or rel_name.startswith("\\") or "file://" in rel_name or "~" in rel_name or ".." in rel_name:
            raise MB6ValidationError(f"Path traversal or absolute path in checksums.sha256 line {line_num}: '{rel_name}'")

        if rel_name in seen_entries:
            raise MB6ValidationError(f"Duplicate file entry in checksums.sha256 at line {line_num}: '{rel_name}'")
        seen_entries.add(rel_name)

        target_f = (manifest_dir / rel_name).resolve()
        if not target_f.is_file():
            raise MB6ValidationError(f"Checksum target file missing: '{rel_name}'")
        if target_f.parent != manifest_dir.resolve():
            raise MB6ValidationError(f"Checksum target file escapes manifest directory: '{rel_name}'")

        actual_hash = hashlib.sha256(target_f.read_bytes()).hexdigest()
        if actual_hash != digest.lower():
            raise MB6ValidationError(f"Checksum mismatch for '{rel_name}': expected {digest}, got {actual_hash}")

    missing_required = (REQUIRED_MB6_ARTIFACTS - {"checksums.sha256"}) - seen_entries
    if missing_required:
        raise MB6ValidationError(f"checksums.sha256 missing required M-B6 artifacts: {missing_required}")

    # 10. Verify No Local Absolute Paths in JSON/JSONL Manifests
    for manifest_f in manifest_dir.glob("*"):
        if manifest_f.suffix in (".json", ".jsonl"):
            content_str = manifest_f.read_text(encoding="utf-8")
            if "/Users/" in content_str or "file://" in content_str:
                raise MB6ValidationError(f"Absolute local path found in machine-readable artifact {manifest_f.name}")

    return {
        "validation_success": True,
        "m_b6_gate_status": "PASS_WITH_WARNINGS",
        "m_b7_entry_status": "READY_WITH_CONDITIONS",
        "independently_measured": {
            "pinned_environment_verified": True,
            "upstream_identity_chain_verified": True,
            "m_b0_to_b5_gates_verified": True,
            "a5_a6_gates_verified": True,
            "primary_float_finalist": mb4_pri_arch,
            "selected_calibration_profile": mb5_sel_prof,
            "frozen_weight_seeds": SHORTLIST_SEEDS,
            "stage_a_b_c_execution_verified": True,
            "pairwise_equivalence_verified": True,
            "subject_level_equivalence_verified": True,
            "per_seed_stage_metrics_verified": True,
            "cross_seed_summaries_verified": True,
            "class_collapse_transitions_verified": True,
            "quantization_diagnostics_verified": True,
            "actual_tflite_structure_verified": True,
            "m_b5_selected_int8_reuse_proven": True,
            "zero_new_class_collapses": True,
            "locked_test_access_blocked": True,
            "hardened_checksum_verification": True,
        },
    }


def main() -> None:
    res = validate_m_b6_artifacts()
    print("Standalone M-B6 Stage-Equivalence Validation Result:")
    print(f"Validation Success: {res['validation_success']}")
    print(f"M-B6 Gate Status: {res['m_b6_gate_status']}")
    print(f"M-B7 Entry Status: {res['m_b7_entry_status']}")
    print(f"Primary Float Finalist: {res['independently_measured']['primary_float_finalist']}")
    print(f"Selected Calibration Profile: {res['independently_measured']['selected_calibration_profile']}")
    print(f"M-B5 INT8 Reuse Identity Proven: {res['independently_measured']['m_b5_selected_int8_reuse_proven']}")
    print(f"LOCKED_TEST Guard Verified: {res['independently_measured']['locked_test_access_blocked']}")
    print(f"Hardened Checksums Verified: {res['independently_measured']['hardened_checksum_verification']}")


if __name__ == "__main__":
    main()
