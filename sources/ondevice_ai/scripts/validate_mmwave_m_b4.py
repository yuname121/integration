# SafeNest mmWave Track — Phase M-B4 Standalone Validator (Hardened Evidence-Truth)

import hashlib
import json
import os
import re
import sys
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional

import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from mmwave_m_b2_imbalance import LABEL_NAMES, compute_one_vs_rest_false_positives, compute_subject_level_diagnostics
from mmwave_m_b3_architecture import (
    build_model_by_id,
    compute_numerical_weights_sha256,
    reset_seeds,
)
from mmwave_m_b4_multiseed import SEEDS, rank_multiseed_architectures
from mmwave_phase_b_access import PhaseBAccessGuard
from validate_mmwave_m_b0 import validate_m_b0_artifacts
from validate_mmwave_m_b1 import validate_m_b1_artifacts
from validate_mmwave_m_b2 import validate_m_b2_artifacts
from validate_mmwave_m_b3 import validate_m_b3_artifacts


class MB4ValidationError(Exception):
    """Raised when Phase M-B4 validation fails."""
    pass


REQUIRED_MB4_ARTIFACTS = {
    "input_identity.json",
    "experiment_contract.json",
    "seed_plan.json",
    "seed42_reuse_audit.json",
    "training_runs.json",
    "seed_weights.npz",
    "validation_predictions.npz",
    "validation_prediction_index.jsonl",
    "per_seed_results.json",
    "multi_seed_results.json",
    "subject_level_seed_metrics.json",
    "primary_float_finalist.json",
    "backup_architecture.json",
    "locked_test_access_audit.json",
    "run_environment.json",
    "exceptions.json",
    "m_b4_summary.json",
    "checksums.sha256",
}


def validate_m_b4_artifacts(
    root_dir: Path = ROOT_DIR,
    manifest_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Independently validate all Phase M-B4 multi-seed stability artifacts without retraining."""
    if manifest_dir is None:
        manifest_dir = root_dir / "datasets/mmwave/manifests/M-B4_multiseed_stability"

    if not manifest_dir.is_dir():
        raise MB4ValidationError(f"M-B4 manifest directory missing: {manifest_dir}")

    guard = PhaseBAccessGuard(root_dir=root_dir)

    # 1. Verify Pinned Environment (TF, NumPy, SciPy)
    import scipy

    actual_tf = tf.__version__
    actual_np = np.__version__
    actual_scipy = scipy.__version__

    req_file = root_dir / "requirements-mac.txt"
    if not req_file.is_file():
        raise MB4ValidationError("requirements-mac.txt missing!")
    req_sha = hashlib.sha256(req_file.read_bytes()).hexdigest()

    env_file = manifest_dir / "run_environment.json"
    if not env_file.is_file():
        raise MB4ValidationError("run_environment.json missing!")
    env_data = json.loads(env_file.read_text(encoding="utf-8"))

    if env_data.get("tensorflow_version") != actual_tf or env_data.get("numpy_version") != actual_np or env_data.get("scipy_version") != actual_scipy:
        raise MB4ValidationError(
            f"Environment mismatch: manifest TF/NP/SciPy={env_data.get('tensorflow_version')}/{env_data.get('numpy_version')}/{env_data.get('scipy_version')}, actual={actual_tf}/{actual_np}/{actual_scipy}"
        )
    if env_data.get("requirements_mac_sha256") != req_sha:
        raise MB4ValidationError("requirements-mac.txt SHA-256 mismatch!")

    # 2. Invoke Upstream Standalone Validators (M-B0, M-B1, M-B2, M-B3)
    mb0_res = validate_m_b0_artifacts(root_dir=root_dir)
    if not mb0_res.get("validation_success"):
        raise MB4ValidationError(f"Upstream M-B0 validation failed: {mb0_res}")

    mb1_res = validate_m_b1_artifacts(root_dir=root_dir)
    if not mb1_res.get("validation_success"):
        raise MB4ValidationError(f"Upstream M-B1 validation failed: {mb1_res}")

    mb2_res = validate_m_b2_artifacts(root_dir=root_dir)
    if not mb2_res.get("validation_success"):
        raise MB4ValidationError(f"Upstream M-B2 validation failed: {mb2_res}")

    mb3_res = validate_m_b3_artifacts(root_dir=root_dir)
    if not mb3_res.get("validation_success"):
        raise MB4ValidationError(f"Upstream M-B3 validation failed: {mb3_res}")

    # 3. Verify Upstream Input Identity Chain
    input_identity_file = manifest_dir / "input_identity.json"
    if not input_identity_file.is_file():
        raise MB4ValidationError("input_identity.json missing!")
    input_identity_data = json.loads(input_identity_file.read_text(encoding="utf-8"))
    inputs_list = input_identity_data.get("inputs", [])

    if len(inputs_list) < 19:
        raise MB4ValidationError(f"input_identity.json must contain at least 19 upstream files, got {len(inputs_list)}")

    for input_item in inputs_list:
        rel_p = input_item.get("path")
        exp_sha = input_item.get("measured_sha256")
        if not rel_p or not exp_sha:
            raise MB4ValidationError(f"Malformed input_identity item: {input_item}")
        full_p = root_dir / rel_p
        if not full_p.is_file():
            raise MB4ValidationError(f"Upstream identity file missing from checkout: {rel_p}")
        act_sha = hashlib.sha256(full_p.read_bytes()).hexdigest()
        if act_sha != exp_sha:
            raise MB4ValidationError(f"Upstream identity SHA mismatch for '{rel_p}': expected {exp_sha}, got {act_sha}")

    # Verify Seed Plan
    seed_plan_file = manifest_dir / "seed_plan.json"
    if not seed_plan_file.is_file():
        raise MB4ValidationError("seed_plan.json missing!")
    seed_plan_data = json.loads(seed_plan_file.read_text(encoding="utf-8"))
    if seed_plan_data.get("seeds") != SEEDS:
        raise MB4ValidationError(f"Seed plan mismatch: expected {SEEDS}, got {seed_plan_data.get('seeds')}")

    # 4. Verify Datasets & Authoritative Subject Counts
    train_data = guard.get_model_selection_dataset("TRAIN")
    val_data = guard.get_model_selection_dataset("VALIDATION")

    act_train_subjs = len(set(w["subject_id"] for w in train_data["windows"]))
    act_val_subjs = len(set(w["subject_id"] for w in val_data["windows"]))

    exp_contract_file = manifest_dir / "experiment_contract.json"
    if not exp_contract_file.is_file():
        raise MB4ValidationError("experiment_contract.json missing!")
    exp_contract_data = json.loads(exp_contract_file.read_text(encoding="utf-8"))

    if exp_contract_data.get("train_subjects") != act_train_subjs:
        raise MB4ValidationError(f"TRAIN subject count mismatch: manifest={exp_contract_data.get('train_subjects')}, actual={act_train_subjs}")
    if exp_contract_data.get("eval_subjects") != act_val_subjs:
        raise MB4ValidationError(f"VALIDATION subject count mismatch: manifest={exp_contract_data.get('eval_subjects')}, actual={act_val_subjs}")

    val_y = np.array([w["safenest_label_id"] for w in val_data["windows"]], dtype=int)

    # 5. Verify Seed 42 Reuse against Authoritative M-B3 Evidence
    seed42_audit_file = manifest_dir / "seed42_reuse_audit.json"
    if not seed42_audit_file.is_file():
        raise MB4ValidationError("seed42_reuse_audit.json missing!")
    seed42_audit_data = json.loads(seed42_audit_file.read_text(encoding="utf-8"))

    mb3_tr_file = root_dir / "datasets/mmwave/manifests/M-B3_architecture_comparison/training_runs.json"
    if not mb3_tr_file.is_file():
        raise MB4ValidationError("M-B3 training_runs.json missing!")
    mb3_tr_data = json.loads(mb3_tr_file.read_text(encoding="utf-8")).get("training_runs", {})

    mb3_preds_file = root_dir / "datasets/mmwave/manifests/M-B3_architecture_comparison/validation_predictions.npz"
    if not mb3_preds_file.is_file():
        raise MB4ValidationError("M-B3 validation_predictions.npz missing!")
    mb3_preds = np.load(mb3_preds_file)

    mb3_res_file = root_dir / "datasets/mmwave/manifests/M-B3_architecture_comparison/architecture_results.json"
    if not mb3_res_file.is_file():
        raise MB4ValidationError("M-B3 architecture_results.json missing!")
    mb3_res_data = json.loads(mb3_res_file.read_text(encoding="utf-8")).get("results", {})

    mb4_tr_file = manifest_dir / "training_runs.json"
    if not mb4_tr_file.is_file():
        raise MB4ValidationError("M-B4 training_runs.json missing!")
    mb4_tr_data = json.loads(mb4_tr_file.read_text(encoding="utf-8")).get("training_runs", {})

    mb4_preds_file = manifest_dir / "validation_predictions.npz"
    if not mb4_preds_file.is_file():
        raise MB4ValidationError("M-B4 validation_predictions.npz missing!")
    mb4_preds = np.load(mb4_preds_file)

    shortlist_ids = ["M-B3_CONV1D_GAP_BASELINE", "M-B3_SEPARABLECONV1D_GAP"]

    for aid in shortlist_ids:
        run_key_42 = f"{aid}_seed_42"

        # Check M-B4 training_runs.json for seed 42
        if run_key_42 not in mb4_tr_data:
            raise MB4ValidationError(f"Seed 42 training run '{run_key_42}' missing from M-B4 training_runs.json")
        mb4_run_42 = mb4_tr_data[run_key_42]
        if mb4_run_42.get("seed") != 42:
            raise MB4ValidationError(f"Declared seed for '{run_key_42}' is not 42: got {mb4_run_42.get('seed')}")
        if mb4_run_42.get("architecture_id") != aid:
            raise MB4ValidationError(f"Architecture ID mismatch for '{run_key_42}': got {mb4_run_42.get('architecture_id')}")

        # Check M-B3 authoritative training run
        if aid not in mb3_tr_data:
            raise MB4ValidationError(f"M-B3 training run for '{aid}' missing from M-B3 training_runs.json")
        mb3_run = mb3_tr_data[aid]

        # 1. Initial weights SHA-256 equality
        if mb4_run_42.get("initial_weights_sha256") != mb3_run.get("initial_weights_sha256"):
            raise MB4ValidationError(
                f"Seed 42 initial weights SHA mismatch for {aid}: M-B3={mb3_run.get('initial_weights_sha256')}, M-B4={mb4_run_42.get('initial_weights_sha256')}"
            )

        # 2. Final numerical weights SHA-256 equality
        if mb4_run_42.get("final_weights_sha256") != mb3_run.get("final_weights_sha256"):
            raise MB4ValidationError(
                f"Seed 42 final weights SHA mismatch for {aid}: M-B3={mb3_run.get('final_weights_sha256')}, M-B4={mb4_run_42.get('final_weights_sha256')}"
            )

        # 3. Exact prediction vector equality
        if aid not in mb3_preds.files or run_key_42 not in mb4_preds.files:
            raise MB4ValidationError(f"Prediction array missing for {aid} in M-B3 or M-B4 prediction NPZ files")
        pred_3 = mb3_preds[aid]
        pred_4 = mb4_preds[run_key_42]
        if not np.array_equal(pred_3, pred_4):
            raise MB4ValidationError(f"Seed 42 prediction vector mismatch between M-B3 and M-B4 for {aid}")

        # 4. Recomputed Macro F1 and Accuracy equality against M-B3 architecture_results.json
        cm_seed42 = compute_one_vs_rest_false_positives(val_y, pred_4)
        f1_calc_42 = float(np.mean([cm_seed42[c]["f1_score"] for c in LABEL_NAMES]))
        acc_calc_42 = float(np.mean(pred_4 == val_y))
        dist_calc_42 = {c: int(np.sum(pred_4 == idx)) for idx, c in enumerate(LABEL_NAMES)}

        mb3_arch_res = mb3_res_data.get(aid, {})
        exp_f1_3 = mb3_arch_res.get("float_macro_f1")
        exp_acc_3 = mb3_arch_res.get("float_accuracy")

        if round(f1_calc_42, 6) != round(exp_f1_3, 6):
            raise MB4ValidationError(f"Seed 42 recomputed Macro F1 mismatch for {aid}: calc={f1_calc_42}, M-B3={exp_f1_3}")
        if round(acc_calc_42, 6) != round(exp_acc_3, 6):
            raise MB4ValidationError(f"Seed 42 recomputed Accuracy mismatch for {aid}: calc={acc_calc_42}, M-B3={exp_acc_3}")

    # 6. Verify Seed 43/44 Initialization Identity
    tr_file = manifest_dir / "training_runs.json"
    if not tr_file.is_file():
        raise MB4ValidationError("training_runs.json missing!")
    tr_data = json.loads(tr_file.read_text(encoding="utf-8")).get("training_runs", {})

    for aid in shortlist_ids:
        for seed in [43, 44]:
            run_key = f"{aid}_seed_{seed}"
            if run_key not in tr_data:
                raise MB4ValidationError(f"Training run {run_key} missing from training_runs.json")

            reset_seeds(seed)
            m_init = build_model_by_id(aid)
            computed_init_sha = compute_numerical_weights_sha256(m_init)
            exp_init_sha = tr_data[run_key]["initial_weights_sha256"]

            if computed_init_sha != exp_init_sha:
                raise MB4ValidationError(
                    f"INITIALIZATION SHA MISMATCH for {run_key}: computed ({computed_init_sha}) != training_runs.json ({exp_init_sha})"
                )

    # 7. Model Weight Reconstruction & Numerical Weight Lineage Verification
    weights_npz_file = manifest_dir / "seed_weights.npz"
    preds_npz_file = manifest_dir / "validation_predictions.npz"
    per_seed_file = manifest_dir / "per_seed_results.json"
    multi_seed_file = manifest_dir / "multi_seed_results.json"

    for req_f, fpath in [
        ("seed_weights.npz", weights_npz_file),
        ("validation_predictions.npz", preds_npz_file),
        ("per_seed_results.json", per_seed_file),
        ("multi_seed_results.json", multi_seed_file),
    ]:
        if not fpath.is_file():
            raise MB4ValidationError(f"Required artifact missing: {req_f}")

    seed_weights_npz = np.load(weights_npz_file)
    val_preds_npz = np.load(preds_npz_file)
    per_seed_data = json.loads(per_seed_file.read_text(encoding="utf-8")).get("per_seed_results", {})
    multi_seed_data = json.loads(multi_seed_file.read_text(encoding="utf-8")).get("multi_seed_results", [])

    for aid in shortlist_ids:
        for seed in SEEDS:
            run_key = f"{aid}_seed_{seed}"
            m_rebuilt = build_model_by_id(aid)
            arch_w_keys = sorted(
                [k for k in seed_weights_npz.files if k.startswith(f"{aid}_seed_{seed}_layer_weight_")],
                key=lambda x: int(x.split("_")[-1]),
            )
            if not arch_w_keys:
                raise MB4ValidationError(f"No stored weights found in seed_weights.npz for {run_key}")
            arch_w_list = [seed_weights_npz[k] for k in arch_w_keys]
            m_rebuilt.set_weights(arch_w_list)

            rebuilt_weight_sha = compute_numerical_weights_sha256(m_rebuilt)
            exp_weight_sha = tr_data[run_key]["final_weights_sha256"]
            if rebuilt_weight_sha != exp_weight_sha:
                raise MB4ValidationError(
                    f"LINEAGE MISMATCH for {run_key}: stored NPZ weight SHA ({rebuilt_weight_sha}) != training_runs.json final_weights_sha256 ({exp_weight_sha})"
                )

    # 8. Fully Recompute & Validate Every Per-Seed Result
    recomputed_per_seed_dict = {}

    for aid in shortlist_ids:
        for seed in SEEDS:
            run_key = f"{aid}_seed_{seed}"
            if run_key not in val_preds_npz:
                raise MB4ValidationError(f"Predictions for {run_key} missing from validation_predictions.npz")
            preds = val_preds_npz[run_key]
            if len(preds) != len(val_y):
                raise MB4ValidationError(f"Prediction count mismatch for {run_key}: got {len(preds)}, expected {len(val_y)}")

            cm = compute_one_vs_rest_false_positives(val_y, preds)
            macro_f1 = float(np.mean([cm[c]["f1_score"] for c in LABEL_NAMES]))
            accuracy = float(np.mean(preds == val_y))
            min_rec = float(min(cm[c]["recall"] for c in LABEL_NAMES))
            apnea_rec = cm["APNEA"]["recall"]
            rapid_rec = cm["RAPID_OR_ABNORMAL"]["recall"]

            collapsed = (apnea_rec == 0.0) or (rapid_rec == 0.0) or (len(np.unique(preds)) < 3)
            pred_dist = {c: int(np.sum(preds == idx)) for idx, c in enumerate(LABEL_NAMES)}

            recomputed_per_seed_dict[run_key] = {
                "val_macro_f1": round(macro_f1, 6),
                "val_accuracy": round(accuracy, 6),
                "min_per_class_recall": round(min_rec, 6),
                "apnea_recall": round(apnea_rec, 6),
                "rapid_recall": round(rapid_rec, 6),
                "collapsed": collapsed,
                "prediction_distribution": pred_dist,
                "class_metrics": cm,
            }

            art_seed = per_seed_data.get(run_key, {})
            if round(art_seed.get("val_macro_f1", 0.0), 6) != round(macro_f1, 6):
                raise MB4ValidationError(f"Per-seed Macro F1 mismatch for {run_key}: manifest={art_seed.get('val_macro_f1')}, calc={macro_f1}")
            if round(art_seed.get("val_accuracy", 0.0), 6) != round(accuracy, 6):
                raise MB4ValidationError(f"Per-seed Accuracy mismatch for {run_key}: manifest={art_seed.get('val_accuracy')}, calc={accuracy}")
            if art_seed.get("collapsed") != collapsed:
                raise MB4ValidationError(f"Per-seed Collapsed flag mismatch for {run_key}: manifest={art_seed.get('collapsed')}, calc={collapsed}")
            if art_seed.get("prediction_distribution") != pred_dist:
                raise MB4ValidationError(f"Per-seed Prediction distribution mismatch for {run_key}")

            art_cm = art_seed.get("class_metrics", {})
            for cname in LABEL_NAMES:
                for k_metric in ("tp", "fp", "tn", "fn", "precision", "recall", "f1_score", "fpr", "support"):
                    if art_cm.get(cname, {}).get(k_metric) != cm.get(cname, {}).get(k_metric):
                        raise MB4ValidationError(
                            f"Per-seed class metric mismatch for {run_key} {cname} '{k_metric}': manifest={art_cm.get(cname, {}).get(k_metric)}, calc={cm.get(cname, {}).get(k_metric)}"
                        )

    # 9. Fully Recompute & Validate Multi-Seed Aggregates
    recomputed_multi_seed = []

    for aid in shortlist_ids:
        seed_runs = [recomputed_per_seed_dict[f"{aid}_seed_{s}"] for s in SEEDS]
        f1_vals = [r["val_macro_f1"] for r in seed_runs]
        acc_vals = [r["val_accuracy"] for r in seed_runs]
        min_rec_vals = [r["min_per_class_recall"] for r in seed_runs]

        worst_idx = int(np.argmin(f1_vals))
        worst_seed = SEEDS[worst_idx]
        collapsed_cnt = sum(1 for r in seed_runs if r["collapsed"])

        per_class_aggregates = {}
        for cname in LABEL_NAMES:
            c_recs = [r["class_metrics"][cname]["recall"] for r in seed_runs]
            c_f1s = [r["class_metrics"][cname]["f1_score"] for r in seed_runs]
            worst_c_idx = int(np.argmin(c_recs))
            per_class_aggregates[cname] = {
                "recall_mean": round(float(np.mean(c_recs)), 6),
                "recall_std": round(float(np.std(c_recs)), 6),
                "recall_min": round(float(np.min(c_recs)), 6),
                "worst_seed_id": SEEDS[worst_c_idx],
                "f1_mean": round(float(np.mean(c_f1s)), 6),
                "f1_std": round(float(np.std(c_f1s)), 6),
                "f1_min": round(float(np.min(c_f1s)), 6),
            }

        total_p = tr_data[f"{aid}_seed_42"]["param_counts"]["total_params"]
        mb3_conv_file = root_dir / "datasets/mmwave/manifests/M-B3_architecture_comparison/conversion_compatibility.json"
        int8_sz = json.loads(mb3_conv_file.read_text(encoding="utf-8")).get("conversion_compatibility", {}).get(aid, {}).get("strict_int8", {}).get("file_bytes")

        recomputed_multi_seed.append({
            "architecture_id": aid,
            "seeds_evaluated": SEEDS,
            "total_params": total_p,
            "strict_int8_bytes": int8_sz,
            "collapsed_seed_count": collapsed_cnt,
            "apnea_zero_recall_seeds": sum(1 for r in seed_runs if r["apnea_recall"] == 0.0),
            "rapid_zero_recall_seeds": sum(1 for r in seed_runs if r["rapid_recall"] == 0.0),
            "macro_f1": {
                "mean": round(float(np.mean(f1_vals)), 6),
                "median": round(float(np.median(f1_vals)), 6),
                "std": round(float(np.std(f1_vals)), 6),
                "min": round(float(np.min(f1_vals)), 6),
                "max": round(float(np.max(f1_vals)), 6),
                "worst_seed_val": round(float(np.min(f1_vals)), 6),
                "worst_seed_id": worst_seed,
                "per_seed": {str(s): r["val_macro_f1"] for s, r in zip(SEEDS, seed_runs)},
            },
            "accuracy": {
                "mean": round(float(np.mean(acc_vals)), 6),
                "std": round(float(np.std(acc_vals)), 6),
                "min": round(float(np.min(acc_vals)), 6),
                "per_seed": {str(s): r["val_accuracy"] for s, r in zip(SEEDS, seed_runs)},
            },
            "min_per_class_recall": {
                "mean": round(float(np.mean(min_rec_vals)), 6),
                "std": round(float(np.std(min_rec_vals)), 6),
                "min": round(float(np.min(min_rec_vals)), 6),
                "worst_seed_val": round(float(np.min(min_rec_vals)), 6),
            },
            "per_class": per_class_aggregates,
        })

    # Compare recomputed multi-seed aggregates against multi_seed_results.json
    if len(multi_seed_data) != len(recomputed_multi_seed):
        raise MB4ValidationError(f"Multi-seed aggregate count mismatch: manifest={len(multi_seed_data)}, recomputed={len(recomputed_multi_seed)}")

    for art_m, calc_m in zip(multi_seed_data, recomputed_multi_seed):
        aid = art_m.get("architecture_id")
        if aid != calc_m["architecture_id"]:
            raise MB4ValidationError(f"Multi-seed aggregate architecture ID mismatch: manifest={aid}, calc={calc_m['architecture_id']}")

        for k in ("collapsed_seed_count", "apnea_zero_recall_seeds", "rapid_zero_recall_seeds"):
            if art_m.get(k) != calc_m.get(k):
                raise MB4ValidationError(f"Multi-seed aggregate field '{k}' mismatch for {aid}: manifest={art_m.get(k)}, calc={calc_m.get(k)}")

        for sub_k in ("macro_f1", "accuracy", "min_per_class_recall"):
            art_sub = art_m.get(sub_k, {})
            calc_sub = calc_m.get(sub_k, {})
            for stat_k, stat_val in calc_sub.items():
                if art_sub.get(stat_k) != stat_val:
                    raise MB4ValidationError(f"Multi-seed '{sub_k}.{stat_k}' mismatch for {aid}: manifest={art_sub.get(stat_k)}, calc={stat_val}")

        art_pc = art_m.get("per_class", {})
        calc_pc = calc_m.get("per_class", {})
        for cname in LABEL_NAMES:
            if art_pc.get(cname) != calc_pc.get(cname):
                raise MB4ValidationError(f"Multi-seed per-class aggregate mismatch for {aid} {cname}: manifest={art_pc.get(cname)}, calc={calc_pc.get(cname)}")

    # 10. Finalist Selection Validation using Recomputed Aggregates
    ranked_recomputed = rank_multiseed_architectures(recomputed_multi_seed, eps=1e-5)
    exp_winner = ranked_recomputed[0]["architecture_id"] if ranked_recomputed and ranked_recomputed[0]["collapsed_seed_count"] == 0 else None
    exp_backup = ranked_recomputed[1]["architecture_id"] if len(ranked_recomputed) > 1 and ranked_recomputed[1]["collapsed_seed_count"] == 0 else None

    primary_file = manifest_dir / "primary_float_finalist.json"
    backup_file = manifest_dir / "backup_architecture.json"

    if not primary_file.is_file() or not backup_file.is_file():
        raise MB4ValidationError("primary_float_finalist.json or backup_architecture.json missing!")

    act_winner = json.loads(primary_file.read_text(encoding="utf-8")).get("primary_stable_float_finalist")
    act_backup = json.loads(backup_file.read_text(encoding="utf-8")).get("backup_architecture_id")

    if act_winner != exp_winner:
        raise MB4ValidationError(f"Primary finalist selection mismatch: expected {exp_winner}, got {act_winner}")
    if act_backup != exp_backup:
        raise MB4ValidationError(f"Backup architecture selection mismatch: expected {exp_backup}, got {act_backup}")

    # 11. Fully Validate Subject-Level Seed Metrics (17 Subjects × 6 Runs x 3 Classes)
    subj_file = manifest_dir / "subject_level_seed_metrics.json"
    if not subj_file.is_file():
        raise MB4ValidationError("subject_level_seed_metrics.json missing!")
    subj_data = json.loads(subj_file.read_text(encoding="utf-8"))

    if subj_data.get("subject_split_variation") != "NOT_PERFORMED_IN_M-B4":
        raise MB4ValidationError("subject_level_seed_metrics.json must record subject_split_variation='NOT_PERFORMED_IN_M-B4'")
    if subj_data.get("stability_type") != "INITIALIZATION_SEED_STABILITY":
        raise MB4ValidationError("subject_level_seed_metrics.json must record stability_type='INITIALIZATION_SEED_STABILITY'")

    art_subj_runs = subj_data.get("subject_diagnostics_by_run", {})

    for aid in shortlist_ids:
        for seed in SEEDS:
            run_key = f"{aid}_seed_{seed}"
            if run_key not in art_subj_runs:
                raise MB4ValidationError(f"Subject diagnostics run key {run_key} missing from subject_level_seed_metrics.json")

            val_preds = val_preds_npz[run_key]
            calc_subj_diag = compute_subject_level_diagnostics(val_data["windows"], val_preds)

            art_run_subj = art_subj_runs[run_key]
            art_per_s = art_run_subj.get("per_subject", {})
            calc_per_s = calc_subj_diag.get("per_subject", {})

            if len(art_per_s) != 17 or len(calc_per_s) != 17:
                raise MB4ValidationError(f"Subject count mismatch for {run_key}: art={len(art_per_s)}, calc={len(calc_per_s)}")

            for sid, calc_s in calc_per_s.items():
                art_s = art_per_s.get(sid, {})
                for k_stat in ("window_count", "accuracy", "subject_macro_f1", "apnea_fp", "apnea_fn", "rapid_fp", "rapid_fn", "prediction_distribution"):
                    if art_s.get(k_stat) != calc_s.get(k_stat):
                        raise MB4ValidationError(f"Subject {sid} metric '{k_stat}' mismatch for {run_key}: manifest={art_s.get(k_stat)}, calc={calc_s.get(k_stat)}")

                # Full per-class subject metrics comparison (support, tp, fp, tn, fn, recall, precision, f1)
                art_cm_map = art_s.get("class_metrics", {})
                calc_cm_map = calc_s.get("class_metrics", {})

                for cname in LABEL_NAMES:
                    if cname not in art_cm_map or cname not in calc_cm_map:
                        raise MB4ValidationError(f"Subject {sid} missing class_metrics for '{cname}' in {run_key}")

                    art_cm = art_cm_map[cname]
                    calc_cm = calc_cm_map[cname]

                    for fld in ("support", "tp", "fp", "tn", "fn", "recall", "precision", "f1"):
                        if art_cm.get(fld) != calc_cm.get(fld):
                            raise MB4ValidationError(
                                f"Subject {sid} class_metrics field '{cname}.{fld}' mismatch for {run_key}: manifest={art_cm.get(fld)}, calc={calc_cm.get(fld)}"
                            )

    # 12. Verify Zero Performance Access to LOCKED_TEST
    locked_file = manifest_dir / "locked_test_access_audit.json"
    if not locked_file.is_file():
        raise MB4ValidationError("locked_test_access_audit.json missing!")
    locked_data = json.loads(locked_file.read_text(encoding="utf-8"))

    if locked_data.get("performance_access_attempts", -1) != 0 or not locked_data.get("lock_preserved"):
        raise MB4ValidationError("LOCKED_TEST performance access violation detected!")

    # 13. HARDENED CHECKSUM MANIFEST VALIDATION
    checksums_file = manifest_dir / "checksums.sha256"
    if not checksums_file.is_file():
        raise MB4ValidationError(f"checksums.sha256 missing: {checksums_file}")

    raw_lines = checksums_file.read_text(encoding="utf-8").splitlines()
    seen_entries = set()

    for line_num, line in enumerate(raw_lines, 1):
        line_str = line.strip()
        if not line_str:
            continue

        parts = line_str.split(maxsplit=1)
        if len(parts) != 2:
            raise MB4ValidationError(f"Malformed checksum line {line_num} in checksums.sha256: '{line}'")

        digest, rel_name = parts[0].strip(), parts[1].strip()

        if not re.fullmatch(r"^[0-9a-fA-F]{64}$", digest):
            raise MB4ValidationError(f"Invalid SHA-256 digest format at line {line_num}: '{digest}'")

        if rel_name.startswith("/") or rel_name.startswith("\\") or "file://" in rel_name or "~" in rel_name or ".." in rel_name:
            raise MB4ValidationError(f"Path traversal or absolute path in checksums.sha256 line {line_num}: '{rel_name}'")

        if rel_name in seen_entries:
            raise MB4ValidationError(f"Duplicate file entry in checksums.sha256 at line {line_num}: '{rel_name}'")
        seen_entries.add(rel_name)

        target_f = (manifest_dir / rel_name).resolve()
        if not target_f.is_file():
            raise MB4ValidationError(f"Checksum target file missing: '{rel_name}'")
        if target_f.parent != manifest_dir.resolve():
            raise MB4ValidationError(f"Checksum target file escapes manifest directory: '{rel_name}'")

        actual_hash = hashlib.sha256(target_f.read_bytes()).hexdigest()
        if actual_hash != digest.lower():
            raise MB4ValidationError(f"Checksum mismatch for '{rel_name}': expected {digest}, got {actual_hash}")

    missing_required = (REQUIRED_MB4_ARTIFACTS - {"checksums.sha256"}) - seen_entries
    if missing_required:
        raise MB4ValidationError(f"checksums.sha256 missing required M-B4 artifacts: {missing_required}")

    # 14. Verify No Local Absolute Paths in JSON/JSONL Manifests
    for manifest_f in manifest_dir.glob("*"):
        if manifest_f.suffix in (".json", ".jsonl"):
            content_str = manifest_f.read_text(encoding="utf-8")
            if "/Users/" in content_str or "file://" in content_str:
                raise MB4ValidationError(f"Absolute local path found in machine-readable artifact {manifest_f.name}")

    return {
        "validation_success": True,
        "m_b4_gate_status": "PASS_WITH_WARNINGS",
        "m_b5_entry_status": "READY_WITH_CONDITIONS",
        "independently_measured": {
            "pinned_environment_verified": True,
            "upstream_identity_chain_verified": True,
            "m_b0_gate_verified": True,
            "m_b1_gate_verified": True,
            "m_b2_gate_verified": True,
            "m_b3_gate_verified": True,
            "seed_plan_verified": SEEDS,
            "seed42_reuse_verified": True,
            "seed43_44_initialization_sha_verified": True,
            "train_subjects_verified": act_train_subjs,
            "eval_subjects_verified": act_val_subjs,
            "primary_stable_float_finalist": act_winner,
            "backup_architecture": act_backup,
            "locked_test_access_blocked": True,
            "hardened_checksum_verification": True,
        },
    }


def main() -> None:
    res = validate_m_b4_artifacts()
    print("Standalone M-B4 Multi-Seed Stability Validation Result:")
    print(f"Validation Success: {res['validation_success']}")
    print(f"M-B4 Gate Status: {res['m_b4_gate_status']}")
    print(f"M-B5 Entry Status: {res['m_b5_entry_status']}")
    print(f"Primary Stable Float Finalist: {res['independently_measured']['primary_stable_float_finalist']}")
    print(f"Backup Architecture: {res['independently_measured']['backup_architecture']}")
    print(f"LOCKED_TEST Guard Verified: {res['independently_measured']['locked_test_access_blocked']}")
    print(f"Hardened Checksums Verified: {res['independently_measured']['hardened_checksum_verification']}")


if __name__ == "__main__":
    main()
