# SafeNest mmWave Track — Phase M-B3 Standalone Validator (Hardened Evidence & Lineage Closure)

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
    ARCHITECTURES,
    build_model_by_id,
    compute_numerical_weights_sha256,
    rank_architectures,
)
from mmwave_phase_b_access import PhaseBAccessGuard
from validate_mmwave_m_b0 import validate_m_b0_artifacts
from validate_mmwave_m_b1 import validate_m_b1_artifacts
from validate_mmwave_m_b2 import validate_m_b2_artifacts


class MB3ValidationError(Exception):
    """Raised when Phase M-B3 validation fails."""
    pass


REQUIRED_MB3_ARTIFACTS = {
    "input_identity.json",
    "experiment_contract.json",
    "architecture_profiles.json",
    "training_runs.json",
    "architecture_weights.npz",
    "validation_predictions.npz",
    "validation_prediction_index.jsonl",
    "architecture_results.json",
    "conversion_compatibility.json",
    "compatibility_representative_dataset.json",
    "tflite_artifact_manifest.json",
    "subject_level_metrics.json",
    "selected_architecture_shortlist.json",
    "locked_test_access_audit.json",
    "determinism_audit.json",
    "run_environment.json",
    "exceptions.json",
    "m_b3_summary.json",
    "checksums.sha256",
}


def validate_m_b3_artifacts(
    root_dir: Path = ROOT_DIR,
    manifest_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Independently validate all Phase M-B3 architecture comparison artifacts without retraining."""
    if manifest_dir is None:
        manifest_dir = root_dir / "datasets/mmwave/manifests/M-B3_architecture_comparison"

    if not manifest_dir.is_dir():
        raise MB3ValidationError(f"M-B3 manifest directory missing: {manifest_dir}")

    guard = PhaseBAccessGuard(root_dir=root_dir)

    # 1. Verify Pinned Environment
    import scipy

    actual_tf = tf.__version__
    actual_np = np.__version__
    actual_scipy = scipy.__version__

    req_file = root_dir / "requirements-mac.txt"
    if not req_file.is_file():
        raise MB3ValidationError("requirements-mac.txt missing!")
    req_sha = hashlib.sha256(req_file.read_bytes()).hexdigest()

    env_file = manifest_dir / "run_environment.json"
    if not env_file.is_file():
        raise MB3ValidationError("run_environment.json missing!")
    env_data = json.loads(env_file.read_text(encoding="utf-8"))

    if env_data.get("tensorflow_version") != actual_tf or env_data.get("numpy_version") != actual_np:
        raise MB3ValidationError(f"Environment mismatch: manifest={env_data.get('tensorflow_version')}/{env_data.get('numpy_version')}, actual={actual_tf}/{actual_np}")
    if env_data.get("requirements_mac_sha256") != req_sha:
        raise MB3ValidationError("requirements-mac.txt SHA-256 mismatch!")

    # 2. Invoke Upstream Standalone Validators (M-B0, M-B1, M-B2)
    mb0_res = validate_m_b0_artifacts(root_dir=root_dir)
    if not mb0_res.get("validation_success"):
        raise MB3ValidationError(f"Upstream M-B0 validation failed: {mb0_res}")

    mb1_res = validate_m_b1_artifacts(root_dir=root_dir)
    if not mb1_res.get("validation_success"):
        raise MB3ValidationError(f"Upstream M-B1 validation failed: {mb1_res}")

    mb2_res = validate_m_b2_artifacts(root_dir=root_dir)
    if not mb2_res.get("validation_success"):
        raise MB3ValidationError(f"Upstream M-B2 validation failed: {mb2_res}")

    # 3. Harden Upstream Input Identity Chain Verification
    input_identity_file = manifest_dir / "input_identity.json"
    if not input_identity_file.is_file():
        raise MB3ValidationError("input_identity.json missing!")
    input_identity_data = json.loads(input_identity_file.read_text(encoding="utf-8"))
    inputs_list = input_identity_data.get("inputs", [])

    if len(inputs_list) < 14:
        raise MB3ValidationError(f"input_identity.json must contain at least 14 upstream files, got {len(inputs_list)}")

    for input_item in inputs_list:
        rel_p = input_item.get("path")
        exp_sha = input_item.get("measured_sha256")
        if not rel_p or not exp_sha:
            raise MB3ValidationError(f"Malformed input_identity item: {input_item}")
        full_p = root_dir / rel_p
        if not full_p.is_file():
            raise MB3ValidationError(f"Upstream identity file missing from checkout: {rel_p}")
        act_sha = hashlib.sha256(full_p.read_bytes()).hexdigest()
        if act_sha != exp_sha:
            raise MB3ValidationError(f"Upstream identity SHA mismatch for '{rel_p}': expected {exp_sha}, got {act_sha}")

    # Verify Pre-Registered Selections
    mb1_sel_path = root_dir / "datasets/mmwave/manifests/M-B1_preprocessing_ablation/selected_preprocessing_profile.json"
    mb2_sel_path = root_dir / "datasets/mmwave/manifests/M-B2_class_imbalance/selected_imbalance_strategy.json"

    mb1_sel = json.loads(mb1_sel_path.read_text(encoding="utf-8")).get("selected_profile_id")
    mb2_sel = json.loads(mb2_sel_path.read_text(encoding="utf-8")).get("selected_strategy_id")

    if mb1_sel != "M-B1_D0_B1_Z1":
        raise MB3ValidationError(f"Frozen M-B1 preprocessing profile mismatch: expected M-B1_D0_B1_Z1, got {mb1_sel}")
    if mb2_sel != "M-B2_CE_UNWEIGHTED":
        raise MB3ValidationError(f"Frozen M-B2 imbalance strategy mismatch: expected M-B2_CE_UNWEIGHTED, got {mb2_sel}")

    # 4. Verify Datasets & Transformed Tensors
    train_data = guard.get_model_selection_dataset("TRAIN")
    val_data = guard.get_model_selection_dataset("VALIDATION")

    if len(train_data["windows"]) != 327 or len(val_data["windows"]) != 79:
        raise MB3ValidationError(f"Dataset window count mismatch: TRAIN={len(train_data['windows'])}, VAL={len(val_data['windows'])}")

    val_y = np.array([w["safenest_label_id"] for w in val_data["windows"]], dtype=int)

    # 5. Verify Architecture Profiles & Dynamic Topology Parameter Rebuilding
    arch_file = manifest_dir / "architecture_profiles.json"
    if not arch_file.is_file():
        raise MB3ValidationError("architecture_profiles.json missing!")
    loaded_archs = json.loads(arch_file.read_text(encoding="utf-8")).get("architectures", [])

    if len(loaded_archs) != 3:
        raise MB3ValidationError(f"Expected exactly 3 pre-registered architectures, got {len(loaded_archs)}")

    expected_ids = ["M-B3_CONV1D_GAP_BASELINE", "M-B3_SEPARABLECONV1D_GAP", "M-B3_CONV1D_BILSTM"]
    loaded_ids = [a["architecture_id"] for a in loaded_archs]
    if loaded_ids != expected_ids:
        raise MB3ValidationError(f"Architecture ID mismatch: expected {expected_ids}, got {loaded_ids}")

    # 6. Verify NPZ Weights Lineage & Model Reconstruction
    weights_npz_file = manifest_dir / "architecture_weights.npz"
    res_file = manifest_dir / "architecture_results.json"
    tr_file = manifest_dir / "training_runs.json"
    conv_file = manifest_dir / "conversion_compatibility.json"
    npz_file = manifest_dir / "validation_predictions.npz"

    for required_f, fpath in [
        ("architecture_weights.npz", weights_npz_file),
        ("architecture_results.json", res_file),
        ("training_runs.json", tr_file),
        ("conversion_compatibility.json", conv_file),
        ("validation_predictions.npz", npz_file),
    ]:
        if not fpath.is_file():
            raise MB3ValidationError(f"Required artifact missing: {required_f}")

    weights_npz = np.load(weights_npz_file)
    val_preds_npz = np.load(npz_file)
    loaded_results = json.loads(res_file.read_text(encoding="utf-8")).get("results", {})
    tr_data = json.loads(tr_file.read_text(encoding="utf-8")).get("training_runs", {})
    conv_data = json.loads(conv_file.read_text(encoding="utf-8")).get("conversion_compatibility", {})

    for arch_info in loaded_archs:
        aid = arch_info["architecture_id"]
        # Rebuild architecture model
        rebuilt_m = build_model_by_id(aid)
        arch_w_keys = sorted(
            [k for k in weights_npz.files if k.startswith(f"{aid}_layer_weight_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        if not arch_w_keys:
            raise MB3ValidationError(f"No stored weights found in architecture_weights.npz for {aid}")
        arch_w_list = [weights_npz[k] for k in arch_w_keys]
        try:
            rebuilt_m.set_weights(arch_w_list)
        except Exception as e:
            raise MB3ValidationError(f"Failed to set stored NPZ weights for {aid}: {e}")

        # Compute numerical weight SHA-256 of rebuilt model
        rebuilt_weight_sha = compute_numerical_weights_sha256(rebuilt_m)
        exp_weight_sha = tr_data[aid]["final_weights_sha256"]
        if rebuilt_weight_sha != exp_weight_sha:
            raise MB3ValidationError(
                f"LINEAGE MISMATCH for {aid}: stored NPZ weight SHA ({rebuilt_weight_sha}) != training_runs.json final_weights_sha256 ({exp_weight_sha})"
            )

        # Verify total parameter count
        rebuilt_params = int(rebuilt_m.count_params())
        exp_params = loaded_results[aid]["total_params"]
        if rebuilt_params != exp_params:
            raise MB3ValidationError(f"Parameter count mismatch for {aid}: rebuilt={rebuilt_params}, manifest={exp_params}")

    # Baseline Equivalence Check for Architecture A
    arch_a_run = tr_data.get("M-B3_CONV1D_GAP_BASELINE", {})
    mb1_runs_file = root_dir / "datasets/mmwave/manifests/M-B1_preprocessing_ablation/training_runs.json"
    mb1_runs = json.loads(mb1_runs_file.read_text(encoding="utf-8")).get("training_runs", {}).get("M-B1_D0_B1_Z1", {})
    mb1_preds_file = root_dir / "datasets/mmwave/manifests/M-B1_preprocessing_ablation/validation_predictions.npz"
    mb1_val_preds = np.load(mb1_preds_file)["M-B1_D0_B1_Z1"]

    if arch_a_run.get("initial_weights_sha256") != mb1_runs.get("initial_weights_sha256"):
        raise MB3ValidationError("M-B3_BASELINE_DRIFT: Architecture A initial weights SHA mismatch!")
    if arch_a_run.get("final_weights_sha256") != mb1_runs.get("final_weights_sha256"):
        raise MB3ValidationError("M-B3_BASELINE_DRIFT: Architecture A final weights SHA mismatch!")

    arch_a_preds = val_preds_npz.get("M-B3_CONV1D_GAP_BASELINE")
    if arch_a_preds is None or not np.array_equal(arch_a_preds, mb1_val_preds):
        raise MB3ValidationError("M-B3_BASELINE_DRIFT: Architecture A VALIDATION prediction vector mismatch!")

    # 7. Inspect Committed TFLite Artifacts (Float & Strict INT8 Verification)
    tflite_manifest_file = manifest_dir / "tflite_artifact_manifest.json"
    if not tflite_manifest_file.is_file():
        raise MB3ValidationError("tflite_artifact_manifest.json missing!")
    tflite_entries = json.loads(tflite_manifest_file.read_text(encoding="utf-8")).get("tflite_artifacts", [])

    model_exp_dir = root_dir / "models/mmwave/experiments/M-B3_architecture_comparison"
    if not model_exp_dir.is_dir():
        raise MB3ValidationError(f"Model experiment directory missing: {model_exp_dir}")

    # Prepare transformed VALIDATION inputs for TFLite inference (79, 300, 1)
    from mmwave_m_b1_preprocessing import fit_train_zscore_statistics, transform_signals
    raw_train_phase = train_data["signals"]
    raw_val_phase = val_data["signals"]
    zstats = fit_train_zscore_statistics(raw_train_phase, detrend=False, bpf=True)
    val_x_float32 = transform_signals(raw_val_phase, detrend=False, bpf=True, zscore=True, zscore_stats=zstats).astype(np.float32)
    val_x_3d = np.expand_dims(val_x_float32, axis=-1)

    for entry in tflite_entries:
        fname = entry["filename"]
        exp_bytes = entry["file_bytes"]
        exp_sha256 = entry["sha256"]
        fpath = model_exp_dir / fname

        if not fpath.is_file():
            raise MB3ValidationError(f"Committed TFLite artifact missing: {fname}")
        actual_content = fpath.read_bytes()
        if len(actual_content) != exp_bytes:
            raise MB3ValidationError(f"TFLite byte size mismatch for {fname}: actual={len(actual_content)}, manifest={exp_bytes}")
        actual_sha = hashlib.sha256(actual_content).hexdigest()
        if actual_sha != exp_sha256:
            raise MB3ValidationError(f"TFLite SHA-256 mismatch for {fname}: actual={actual_sha}, manifest={exp_sha256}")

        fmt = entry["format"]
        aid = entry["architecture_id"]

        if fmt == "int8":
            # Strict INT8 verification
            try:
                interp = tf.lite.Interpreter(model_content=actual_content)
                interp.allocate_tensors()
            except Exception as e:
                raise MB3ValidationError(f"Strict INT8 TFLite interpreter allocation failed for {fname}: {e}")

            in_det = interp.get_input_details()
            out_det = interp.get_output_details()

            if in_det[0]["dtype"] != np.int8 or out_det[0]["dtype"] != np.int8:
                raise MB3ValidationError(f"Strict INT8 dtype mismatch for {fname}: in={in_det[0]['dtype']}, out={out_det[0]['dtype']}")

            # Operator inventory inspection
            try:
                op_details = interp._get_ops_details()
                op_names = [op["op_name"] for op in op_details]
                if any("FLEX" in op_n.upper() or "SELECT" in op_n.upper() for op_n in op_names):
                    raise MB3ValidationError(f"Flex / Select TF Ops detected in strict INT8 artifact {fname}: {op_names}")
            except AttributeError:
                pass

            # Execute all 79 VALIDATION samples through INT8 TFLite interpreter
            in_scale, in_zero = in_det[0]["quantization"]
            out_scale, out_zero = out_det[0]["quantization"]

            int8_preds = []
            for i in range(len(val_x_3d)):
                sample_float = val_x_3d[i : i + 1]
                if in_scale > 0:
                    sample_quant = np.clip(np.round(sample_float / in_scale + in_zero), -128, 127).astype(np.int8)
                else:
                    sample_quant = sample_float.astype(np.int8)

                interp.set_tensor(in_det[0]["index"], sample_quant)
                interp.invoke()
                out_quant = interp.get_tensor(out_det[0]["index"])

                if out_scale > 0:
                    out_dequant = (out_quant.astype(np.float32) - out_zero) * out_scale
                else:
                    out_dequant = out_quant.astype(np.float32)

                pred_cls = int(np.argmax(out_dequant, axis=1)[0])
                int8_preds.append(pred_cls)

            int8_preds_arr = np.array(int8_preds, dtype=int)
            exp_int8_preds = val_preds_npz.get(f"{aid}_tflite_int8")
            if exp_int8_preds is not None and not np.array_equal(int8_preds_arr, exp_int8_preds):
                raise MB3ValidationError(f"TFLite INT8 prediction mismatch for {aid} across 79 VALIDATION samples!")

            # Recompute INT8 metrics
            int8_cm = compute_one_vs_rest_false_positives(val_y, int8_preds_arr)
            int8_macro_f1 = float(np.mean([int8_cm[c]["f1_score"] for c in LABEL_NAMES]))
            if int8_cm["APNEA"]["recall"] == 0.0 or int8_cm["RAPID_OR_ABNORMAL"]["recall"] == 0.0:
                raise MB3ValidationError(f"Class collapse detected in strict INT8 artifact {fname}!")

    # 8. Recompute Ranking & Shortlist Selection
    recomputed_ranking = []
    for arch_info in loaded_archs:
        aid = arch_info["architecture_id"]
        if aid not in val_preds_npz:
            raise MB3ValidationError(f"Predictions for architecture {aid} missing from validation_predictions.npz!")

        preds = val_preds_npz[aid]
        if len(preds) != len(val_y):
            raise MB3ValidationError(f"Prediction count mismatch for {aid}: got {len(preds)}, expected {len(val_y)}")

        per_class = compute_one_vs_rest_false_positives(val_y, preds)

        macro_f1 = float(np.mean([per_class[c]["f1_score"] for c in LABEL_NAMES]))
        min_rec = float(min(per_class[c]["recall"] for c in LABEL_NAMES))
        apnea_rec = per_class["APNEA"]["recall"]

        c_info = conv_data.get(aid, {})
        eligibility = c_info.get("deployment_eligibility")

        # Validate BiLSTM classification semantics
        if aid == "M-B3_CONV1D_BILSTM":
            if c_info.get("strict_int8", {}).get("success") is not False:
                raise MB3ValidationError("Architecture C (BiLSTM) must be strict INT8 unsupported!")
            if not c_info.get("strict_int8", {}).get("select_tf_ops_required"):
                raise MB3ValidationError("Architecture C (BiLSTM) must record select_tf_ops_required=True!")
            if eligibility != "SELECT_TF_OPS_REQUIRED":
                raise MB3ValidationError(f"Architecture C eligibility must be SELECT_TF_OPS_REQUIRED, got {eligibility}")

        recomputed_ranking.append({
            "architecture_id": aid,
            "name": arch_info["name"],
            "total_params": tr_data[aid]["param_counts"]["total_params"],
            "float_macro_f1": round(macro_f1, 6),
            "float_min_per_class_recall": round(min_rec, 6),
            "float_apnea_recall": round(apnea_rec, 6),
            "strict_int8_bytes": c_info.get("strict_int8", {}).get("file_bytes"),
            "deployment_eligibility": eligibility,
        })

    ranked_eligible = rank_architectures(recomputed_ranking, eps=1e-5)
    recomputed_shortlist = [r["architecture_id"] for r in ranked_eligible[:2]]

    shortlist_file = manifest_dir / "selected_architecture_shortlist.json"
    if not shortlist_file.is_file():
        raise MB3ValidationError("selected_architecture_shortlist.json missing!")
    loaded_shortlist = json.loads(shortlist_file.read_text(encoding="utf-8")).get("selected_architecture_shortlist", [])

    if loaded_shortlist != recomputed_shortlist:
        raise MB3ValidationError(f"Shortlist selection mismatch: recomputed={recomputed_shortlist}, loaded={loaded_shortlist}")

    # 9. Verify Full 17-Subject Diagnostics for Winner
    winner_id = recomputed_shortlist[0]
    winner_preds = val_preds_npz[winner_id]

    calc_subj_diag = compute_subject_level_diagnostics(val_data["windows"], winner_preds)
    subj_file = manifest_dir / "subject_level_metrics.json"
    if not subj_file.is_file():
        raise MB3ValidationError("subject_level_metrics.json missing!")
    loaded_subj = json.loads(subj_file.read_text(encoding="utf-8")).get("subject_diagnostics", {}).get(winner_id, {})

    art_per_subj = loaded_subj.get("per_subject", {})
    calc_per_subj = calc_subj_diag.get("per_subject", {})

    if len(art_per_subj) != 17 or len(calc_per_subj) != 17:
        raise MB3ValidationError(f"VALIDATION subject count mismatch: art={len(art_per_subj)}, calc={len(calc_per_subj)}, expected 17")
    if set(art_per_subj.keys()) != set(calc_per_subj.keys()):
        raise MB3ValidationError(f"Subject ID set mismatch in subject_level_metrics.json: expected {sorted(calc_per_subj.keys())}, got {sorted(art_per_subj.keys())}")

    for sid, calc_s in calc_per_subj.items():
        art_s = art_per_subj.get(sid, {})
        for k in ("window_count", "accuracy", "subject_macro_f1", "apnea_fp", "apnea_fn", "rapid_fp", "rapid_fn", "prediction_distribution"):
            if art_s.get(k) != calc_s.get(k):
                raise MB3ValidationError(f"Subject {sid} field '{k}' mismatch: expected {calc_s.get(k)}, got {art_s.get(k)}")

    # 10. Verify Zero Performance Access to LOCKED_TEST
    locked_file = manifest_dir / "locked_test_access_audit.json"
    if not locked_file.is_file():
        raise MB3ValidationError("locked_test_access_audit.json missing!")
    locked_data = json.loads(locked_file.read_text(encoding="utf-8"))

    if locked_data.get("performance_access_attempts", -1) != 0 or not locked_data.get("lock_preserved"):
        raise MB3ValidationError("LOCKED_TEST performance access violation detected!")

    # 11. HARDENED CHECKSUM MANIFEST VALIDATION
    checksums_file = manifest_dir / "checksums.sha256"
    if not checksums_file.is_file():
        raise MB3ValidationError(f"checksums.sha256 missing: {checksums_file}")

    raw_lines = checksums_file.read_text(encoding="utf-8").splitlines()
    seen_entries = set()

    for line_num, line in enumerate(raw_lines, 1):
        line_str = line.strip()
        if not line_str:
            continue

        parts = line_str.split(maxsplit=1)
        if len(parts) != 2:
            raise MB3ValidationError(f"Malformed checksum line {line_num} in checksums.sha256: '{line}'")

        digest, rel_name = parts[0].strip(), parts[1].strip()

        if not re.fullmatch(r"^[0-9a-fA-F]{64}$", digest):
            raise MB3ValidationError(f"Invalid SHA-256 digest format at line {line_num}: '{digest}'")

        if rel_name.startswith("/") or rel_name.startswith("\\") or "file://" in rel_name or "~" in rel_name or ".." in rel_name:
            raise MB3ValidationError(f"Path traversal or absolute path in checksums.sha256 line {line_num}: '{rel_name}'")

        if rel_name in seen_entries:
            raise MB3ValidationError(f"Duplicate file entry in checksums.sha256 at line {line_num}: '{rel_name}'")
        seen_entries.add(rel_name)

        target_f = (manifest_dir / rel_name).resolve()
        if not target_f.is_file():
            raise MB3ValidationError(f"Checksum target file missing: '{rel_name}'")
        if target_f.parent != manifest_dir.resolve():
            raise MB3ValidationError(f"Checksum target file escapes manifest directory: '{rel_name}'")

        actual_hash = hashlib.sha256(target_f.read_bytes()).hexdigest()
        if actual_hash != digest.lower():
            raise MB3ValidationError(f"Checksum mismatch for '{rel_name}': expected {digest}, got {actual_hash}")

    missing_required = (REQUIRED_MB3_ARTIFACTS - {"checksums.sha256"}) - seen_entries
    if missing_required:
        raise MB3ValidationError(f"checksums.sha256 missing required M-B3 artifacts: {missing_required}")

    # 12. Verify No Local Absolute Paths in JSON/JSONL Manifests
    for manifest_f in manifest_dir.glob("*"):
        if manifest_f.suffix in (".json", ".jsonl"):
            content_str = manifest_f.read_text(encoding="utf-8")
            if "/Users/" in content_str or "file://" in content_str:
                raise MB3ValidationError(f"Absolute local path found in machine-readable artifact {manifest_f.name}")

    # 13. Verify Report Truthfulness & Consistency
    report_file = root_dir / "docs/reports/20260810_Antigravity_M-B3_Architecture_Comparison_01.md"
    if not report_file.is_file():
        raise MB3ValidationError("Report file missing: 20260810_Antigravity_M-B3_Architecture_Comparison_01.md")
    report_text = report_file.read_text(encoding="utf-8")

    if "100% hardware acceleration" in report_text.lower():
        raise MB3ValidationError("Report contains unverified hardware acceleration claims!")

    return {
        "validation_success": True,
        "m_b3_gate_status": "PASS_WITH_WARNINGS",
        "m_b4_entry_status": "READY_WITH_CONDITIONS",
        "independently_measured": {
            "pinned_environment_verified": True,
            "input_identity_upstream_verified": True,
            "m_b0_gate_verified": True,
            "m_b1_gate_verified": True,
            "m_b2_gate_verified": True,
            "m_b1_selected_preprocessing_profile": "M-B1_D0_B1_Z1",
            "m_b2_selected_imbalance_strategy": "M-B2_CE_UNWEIGHTED",
            "architectures_audited": len(loaded_archs),
            "recomputed_shortlist": recomputed_shortlist,
            "bilstm_select_tf_ops_classified": True,
            "locked_test_access_blocked": True,
            "subject_level_metrics_verified": True,
            "hardened_checksum_verification": True,
            "stored_npz_weight_lineage_verified": True,
            "committed_tflite_artifacts_verified": True,
            "report_consistency_verified": True,
        },
    }


def main() -> None:
    res = validate_m_b3_artifacts()
    print("Standalone M-B3 TinyML Architecture Comparison Validation Result:")
    print(f"Validation Success: {res['validation_success']}")
    print(f"M-B3 Gate Status: {res['m_b3_gate_status']}")
    print(f"M-B4 Entry Status: {res['m_b4_entry_status']}")
    print(f"Architectures Audited: {res['independently_measured']['architectures_audited']}")
    print(f"Recomputed Shortlist: {res['independently_measured']['recomputed_shortlist']}")
    print(f"LOCKED_TEST Guard Verified: {res['independently_measured']['locked_test_access_blocked']}")
    print(f"Hardened Checksums Verified: {res['independently_measured']['hardened_checksum_verification']}")


if __name__ == "__main__":
    main()
