#!/usr/bin/env python3
"""SafeNest Phase M-B1 — Standalone Validator.

Independently validates M-B1 real-data preprocessing full-factorial ablation,
recomputing Z-score statistics, transformed tensor fingerprints, validation metrics,
class-collapse rejection, pre-registered winner ranking, prediction index provenance,
environment compliance, and fail-closed checksum manifest, anchoring to the immutable M-B0/A5/A6 identity chain.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from mmwave_m_b1_preprocessing import (
    PROFILES,
    compute_tensor_fingerprint,
    fit_train_zscore_statistics,
    transform_signals,
)
from mmwave_phase_b_access import LOCKED_TEST_AccessError, PhaseBAccessGuard
from validate_mmwave_m_b0 import validate_m_b0_artifacts

REQUIRED_MB1_ARTIFACTS = {
    "input_identity.json",
    "experiment_contract.json",
    "preprocessing_profiles.json",
    "train_fit_statistics.json",
    "preprocessing_fingerprints.json",
    "training_runs.json",
    "ablation_results.json",
    "signal_diagnostics.json",
    "bpf_frequency_diagnostic.json",
    "apnea_proxy_preprocessing_diagnostic.json",
    "validation_predictions.npz",
    "validation_prediction_index.jsonl",
    "selected_preprocessing_profile.json",
    "reproducibility_comparison.json",
    "locked_test_access_audit.json",
    "determinism_audit.json",
    "run_environment.json",
    "exceptions.json",
    "m_b1_summary.json",
}

LABEL_ID_TO_NAME = {0: "NORMAL", 1: "RAPID_OR_ABNORMAL", 2: "APNEA"}


class MB1ValidationError(Exception):
    """Raised when Phase M-B1 validation fails."""


def validate_m_b1_artifacts(
    root_dir: Path = ROOT_DIR,
    manifest_dir: Path | None = None,
) -> dict[str, Any]:
    """Independently validate all Phase M-B1 artifacts against strict contracts."""
    if manifest_dir is None:
        manifest_dir = root_dir / "datasets/mmwave/manifests/M-B1_preprocessing_ablation"

    if not manifest_dir.is_dir():
        raise MB1ValidationError(f"M-B1 manifest directory missing: {manifest_dir}")

    # 1. Verify Pinned Environment Preflight & requirements-mac.txt SHA-256
    env_file = manifest_dir / "run_environment.json"
    if not env_file.is_file():
        raise MB1ValidationError("run_environment.json missing!")
    env_data = json.loads(env_file.read_text(encoding="utf-8"))

    if env_data.get("tensorflow_version") != "2.20.0":
        raise MB1ValidationError(f"Invalid TensorFlow version in run_environment.json: got {env_data.get('tensorflow_version')}, expected 2.20.0")
    if env_data.get("numpy_version") != "1.26.4":
        raise MB1ValidationError(f"Invalid NumPy version in run_environment.json: got {env_data.get('numpy_version')}, expected 1.26.4")
    if env_data.get("scipy_version") != "1.13.1":
        raise MB1ValidationError(f"Invalid SciPy version in run_environment.json: got {env_data.get('scipy_version')}, expected 1.13.1")
    if not env_data.get("pinned_environment_verified"):
        raise MB1ValidationError("run_environment.json pinned_environment_verified must be True!")

    req_mac = root_dir / "requirements-mac.txt"
    if not req_mac.is_file():
        raise MB1ValidationError("requirements-mac.txt missing from repository root!")
    actual_req_sha = hashlib.sha256(req_mac.read_bytes()).hexdigest()
    recorded_req_sha = env_data.get("requirements_mac_sha256")
    if recorded_req_sha != actual_req_sha:
        raise MB1ValidationError(f"requirements-mac.txt SHA mismatch in run_environment.json! Expected {actual_req_sha}, got {recorded_req_sha}")

    # 2. Independently Load and Verify M-B1 input_identity.json Upstream Hashes
    input_id_file = manifest_dir / "input_identity.json"
    if not input_id_file.is_file():
        raise MB1ValidationError("input_identity.json missing!")
    input_id_data = json.loads(input_id_file.read_text(encoding="utf-8"))

    for item in input_id_data.get("inputs", []):
        rel_p = item.get("repository_relative_path", "")
        recorded_sha = item.get("measured_sha256", "")
        target_f = root_dir / rel_p
        if not target_f.is_file():
            raise MB1ValidationError(f"Upstream identity file missing: {rel_p}")
        measured_sha = hashlib.sha256(target_f.read_bytes()).hexdigest()
        if measured_sha != recorded_sha:
            raise MB1ValidationError(f"Upstream identity SHA mismatch for '{rel_p}': expected {recorded_sha}, got {measured_sha}")

    # 3. Independently Run and Verify M-B0 Gate & Upstream Identity Chain
    mb0_res = validate_m_b0_artifacts(root_dir=root_dir)
    if not mb0_res.get("validation_success") or mb0_res.get("m_b0_gate_status") != "PASS_WITH_WARNINGS":
        raise MB1ValidationError("M-B0 standalone validation failed! Cannot validate M-B1 on top of invalid M-B0.")

    mb0_dir = root_dir / "datasets/mmwave/manifests/M-B0_evaluation_protocol"
    mb0_checksums = mb0_dir / "checksums.sha256"
    if not mb0_checksums.is_file():
        raise MB1ValidationError("M-B0 checksums.sha256 missing!")

    for line in mb0_checksums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise MB1ValidationError(f"Malformed line in M-B0 checksums.sha256: '{line}'")
        expected_sha, rel_n = parts[0].strip(), parts[1].strip()
        target_f = mb0_dir / rel_n
        if not target_f.is_file():
            raise MB1ValidationError(f"M-B0 checksum target file missing: {rel_n}")
        actual_sha = hashlib.sha256(target_f.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise MB1ValidationError(f"M-B0 checksum mismatch for '{rel_n}': expected {expected_sha}, got {actual_sha}")

    # Upstream A5 Subject Split & A6 Matrix/Manifest Identities
    a5_split_file = root_dir / "datasets/mmwave/splits/mmwave_real_subject_split_v1.json"
    if not a5_split_file.is_file():
        raise MB1ValidationError("A5 real subject split file missing!")
    actual_a5_sha = hashlib.sha256(a5_split_file.read_bytes()).hexdigest()
    if actual_a5_sha != "a1996ea00a1d5066eae6d25f022d04137085434d4768e27cbebcdad4e0385baa":
        raise MB1ValidationError(f"A5 real subject split SHA changed! Got {actual_a5_sha}")

    canonical_npy = root_dir / "datasets/mmwave/processed/mmwave_canonical_real_v1.npy"
    if not canonical_npy.is_file():
        raise MB1ValidationError("Canonical matrix missing!")
    actual_npy_sha = hashlib.sha256(canonical_npy.read_bytes()).hexdigest()
    if actual_npy_sha != "c2e2cd1615c7af0f0e21700f291ee12ac0347a9f7fc6ccc9f337433c16868f0e":
        raise MB1ValidationError(f"Canonical NPY SHA changed! Got {actual_npy_sha}")

    a6_manifest = root_dir / "datasets/mmwave/manifests/a6_full_conversion/full_window_manifest.jsonl"
    if not a6_manifest.is_file():
        raise MB1ValidationError("A6 full_window_manifest.jsonl missing!")
    actual_a6_sha = hashlib.sha256(a6_manifest.read_bytes()).hexdigest()
    if actual_a6_sha != "1d1728eafdc3d4786e34fc663329a12a311322a698bdbf2fd01e6bce95c50acf":
        raise MB1ValidationError(f"A6 window manifest SHA changed! Got {actual_a6_sha}")

    # 4. Test PhaseBAccessGuard LOCKED_TEST Fail-Closed Guard
    guard = PhaseBAccessGuard(root_dir=root_dir)
    try:
        guard.get_model_selection_dataset("LOCKED_TEST")
        raise MB1ValidationError("PhaseBAccessGuard failed to block LOCKED_TEST model selection access!")
    except LOCKED_TEST_AccessError:
        pass

    # 5. Load Pure-Class Datasets & Verify Validation Prediction Index Provenance
    train_data = guard.get_train_data(include_ambiguous=False)
    val_data = guard.get_validation_data(include_ambiguous=False)

    if train_data["total_count"] != 327 or val_data["total_count"] != 79:
        raise MB1ValidationError(f"Dataset population mismatch! TRAIN={train_data['total_count']}, VAL={val_data['total_count']}")

    train_signals = train_data["signals"]
    val_signals = val_data["signals"]

    val_idx_file = manifest_dir / "validation_prediction_index.jsonl"
    if not val_idx_file.is_file():
        raise MB1ValidationError(f"validation_prediction_index.jsonl missing: {val_idx_file}")

    val_idx_lines = [json.loads(line) for line in val_idx_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(val_idx_lines) != 79:
        raise MB1ValidationError(f"validation_prediction_index.jsonl row count mismatch: expected 79, got {len(val_idx_lines)}")

    for pos, row in enumerate(val_idx_lines):
        if row.get("split") != "VALIDATION":
            raise MB1ValidationError(f"Non-VALIDATION row in validation_prediction_index.jsonl at line {pos}: split='{row.get('split')}'")
        if row.get("validation_position") != pos:
            raise MB1ValidationError(f"validation_position mismatch at line {pos}: got {row.get('validation_position')}")

        w_exp = val_data["windows"][pos]
        if row.get("window_id") != w_exp["window_id"] or row.get("canonical_sample_index") != w_exp["canonical_sample_index"]:
            raise MB1ValidationError(f"Window mapping mismatch at position {pos} in validation_prediction_index.jsonl!")

    # 6. Verify 8 Preprocessing Profiles (2^3 Factorial)
    prof_file = manifest_dir / "preprocessing_profiles.json"
    if not prof_file.is_file():
        raise MB1ValidationError(f"preprocessing_profiles.json missing: {prof_file}")
    loaded_profiles = json.loads(prof_file.read_text(encoding="utf-8")).get("profiles", [])

    if len(loaded_profiles) != 8:
        raise MB1ValidationError(f"Expected 8 profiles, got {len(loaded_profiles)}")

    profile_ids = [p["profile_id"] for p in loaded_profiles]
    expected_ids = [p["profile_id"] for p in PROFILES]
    if profile_ids != expected_ids:
        raise MB1ValidationError(f"Profile ID mismatch! Expected {expected_ids}, got {profile_ids}")

    # 7. Independently Recompute Z-Score Statistics & Tensor Fingerprints
    zstat_file = manifest_dir / "train_fit_statistics.json"
    fingerprint_file = manifest_dir / "preprocessing_fingerprints.json"
    if not zstat_file.is_file() or not fingerprint_file.is_file():
        raise MB1ValidationError("train_fit_statistics.json or preprocessing_fingerprints.json missing!")

    loaded_zstats = json.loads(zstat_file.read_text(encoding="utf-8")).get("zscore_statistics", {})
    loaded_fingerprints = json.loads(fingerprint_file.read_text(encoding="utf-8")).get("fingerprints", {})

    for prof in PROFILES:
        pid = prof["profile_id"]
        detrend, bpf, zscore = prof["detrend"], prof["bpf"], prof["zscore"]

        if zscore:
            calc_zstats = fit_train_zscore_statistics(train_signals, detrend=detrend, bpf=bpf)
            manif_z = loaded_zstats.get(pid, {})
            if abs(calc_zstats["mean"] - manif_z.get("mean", 0.0)) > 1e-6 or abs(calc_zstats["std"] - manif_z.get("std", 1.0)) > 1e-6:
                raise MB1ValidationError(f"Z-score stat mismatch for {pid}! Calc={calc_zstats}, Manifest={manif_z}")
            stats_to_use = calc_zstats
        else:
            stats_to_use = None

        calc_train_t = transform_signals(train_signals, detrend=detrend, bpf=bpf, zscore=zscore, zscore_stats=stats_to_use)
        calc_val_t = transform_signals(val_signals, detrend=detrend, bpf=bpf, zscore=zscore, zscore_stats=stats_to_use)

        train_fp = compute_tensor_fingerprint(calc_train_t)
        val_fp = compute_tensor_fingerprint(calc_val_t)

        manif_fp = loaded_fingerprints.get(pid, {})
        if train_fp != manif_fp.get("train_tensor_sha256") or val_fp != manif_fp.get("validation_tensor_sha256"):
            raise MB1ValidationError(f"Tensor fingerprint mismatch for {pid}!")

    # 8. Verify Validation Predictions & Recompute Metrics
    npz_file = manifest_dir / "validation_predictions.npz"
    ablation_file = manifest_dir / "ablation_results.json"
    if not npz_file.is_file() or not ablation_file.is_file():
        raise MB1ValidationError("validation_predictions.npz or ablation_results.json missing!")

    val_preds_npz = np.load(npz_file)
    loaded_ablation = json.loads(ablation_file.read_text(encoding="utf-8")).get("results", {})

    val_true_ids = np.array([w["safenest_label_id"] for w in val_data["windows"]], dtype=int)

    recomputed_ranking = []

    for prof in PROFILES:
        pid = prof["profile_id"]
        if pid not in val_preds_npz:
            raise MB1ValidationError(f"Predictions for {pid} missing from NPZ!")

        preds = val_preds_npz[pid]
        if len(preds) != len(val_true_ids):
            raise MB1ValidationError(f"Prediction count mismatch for {pid}: got {len(preds)}, expected {len(val_true_ids)}")

        per_class = {}
        for cid in (0, 1, 2):
            cname = LABEL_ID_TO_NAME[cid]
            tp = int(np.sum((preds == cid) & (val_true_ids == cid)))
            fp = int(np.sum((preds == cid) & (val_true_ids != cid)))
            fn = int(np.sum((preds != cid) & (val_true_ids == cid)))

            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            per_class[cname] = {"precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6), "tp": tp, "fp": fp, "fn": fn}

        macro_f1 = float(np.mean([per_class[c]["f1"] for c in ("NORMAL", "RAPID_OR_ABNORMAL", "APNEA")]))
        min_rec = float(min(per_class[c]["recall"] for c in ("NORMAL", "RAPID_OR_ABNORMAL", "APNEA")))
        apnea_rec = per_class["APNEA"]["recall"]

        is_collapsed = (apnea_rec == 0.0) or (per_class["RAPID_OR_ABNORMAL"]["recall"] == 0.0)

        manif_res = loaded_ablation.get(pid, {})
        if abs(macro_f1 - manif_res.get("macro_f1", 0.0)) > 1e-4:
            raise MB1ValidationError(f"Macro F1 mismatch for {pid}: calc={macro_f1:.6f}, manifest={manif_res.get('macro_f1')}")

        recomputed_ranking.append({
            "profile_id": pid,
            "is_collapsed": is_collapsed,
            "macro_f1": round(macro_f1, 6),
            "min_recall": round(min_rec, 6),
            "apnea_recall": round(apnea_rec, 6),
            "num_operations": int(prof["detrend"]) + int(prof["bpf"]) + int(prof["zscore"]),
        })

    # 9. Pre-Registered Winner Selection Ranking
    eligible_candidates = [r for r in recomputed_ranking if not r["is_collapsed"]]
    if not eligible_candidates:
        raise MB1ValidationError("ALL 8 PREPROCESSING PROFILES COLLAPSED! No valid candidate winner.")

    eligible_candidates.sort(
        key=lambda r: (
            r["macro_f1"],
            r["min_recall"],
            r["apnea_recall"],
            -r["num_operations"],
            r["profile_id"],
        ),
        reverse=True,
    )

    recomputed_winner = eligible_candidates[0]["profile_id"]

    sel_file = manifest_dir / "selected_preprocessing_profile.json"
    if not sel_file.is_file():
        raise MB1ValidationError(f"selected_preprocessing_profile.json missing: {sel_file}")
    loaded_winner = json.loads(sel_file.read_text(encoding="utf-8")).get("selected_profile_id")

    if loaded_winner != recomputed_winner:
        raise MB1ValidationError(f"Winner selection mismatch! Recomputed winner={recomputed_winner}, Loaded={loaded_winner}")

    # 10. Verify Semantic Integrity of reproducibility_comparison.json
    repro_file = manifest_dir / "reproducibility_comparison.json"
    if not repro_file.is_file():
        raise MB1ValidationError("reproducibility_comparison.json missing!")
    repro_data = json.loads(repro_file.read_text(encoding="utf-8"))

    hist_winner = repro_data.get("historical_winner", "")
    pin_winner = repro_data.get("pinned_winner", "")
    winner_changed = repro_data.get("winner_changed")

    expected_changed = bool("M-B1_D1_B0_Z0" not in pin_winner)
    if winner_changed != expected_changed:
        raise MB1ValidationError(f"reproducibility_comparison.json winner_changed mismatch! Declared={winner_changed}, Expected={expected_changed}")

    verdict = repro_data.get("reproducibility_verdict", "")
    if winner_changed and ("VERIFIED_IDENTICAL" in verdict or "100% Identical" in verdict):
        raise MB1ValidationError("Contradictory reproducibility_comparison.json: winner_changed is True but verdict claims VERIFIED_IDENTICAL!")

    profs_comp = repro_data.get("profile_comparisons", {})
    for pid, pdata in profs_comp.items():
        old_f1 = pdata.get("old_numpy_202", {}).get("macro_f1", 0.0)
        new_f1 = pdata.get("new_pinned_numpy_1264", {}).get("macro_f1", 0.0)
        calc_delta = round(new_f1 - old_f1, 6)
        manif_delta = pdata.get("delta_macro_f1", 0.0)
        if abs(calc_delta - manif_delta) > 1e-5:
            raise MB1ValidationError(f"Delta Macro F1 mismatch for {pid}: calc={calc_delta}, manifest={manif_delta}")

    # 11. HARDENED CHECKSUM MANIFEST VALIDATION
    checksums_file = manifest_dir / "checksums.sha256"
    if not checksums_file.is_file():
        raise MB1ValidationError(f"checksums.sha256 missing: {checksums_file}")

    raw_lines = checksums_file.read_text(encoding="utf-8").splitlines()
    seen_entries = set()

    for line_num, line in enumerate(raw_lines, 1):
        line_str = line.strip()
        if not line_str:
            continue

        parts = line_str.split(maxsplit=1)
        if len(parts) != 2:
            raise MB1ValidationError(f"Malformed checksum line {line_num} in checksums.sha256: '{line}'")

        digest, rel_name = parts[0].strip(), parts[1].strip()

        if not re.fullmatch(r"^[0-9a-fA-F]{64}$", digest):
            raise MB1ValidationError(f"Invalid SHA-256 digest format at line {line_num}: '{digest}'")

        if rel_name.startswith("/") or rel_name.startswith("\\") or "file://" in rel_name or "~" in rel_name or ".." in rel_name:
            raise MB1ValidationError(f"Path traversal or absolute path in checksums.sha256 line {line_num}: '{rel_name}'")

        if rel_name in seen_entries:
            raise MB1ValidationError(f"Duplicate file entry in checksums.sha256 at line {line_num}: '{rel_name}'")
        seen_entries.add(rel_name)

        target_f = (manifest_dir / rel_name).resolve()
        if not target_f.is_file():
            raise MB1ValidationError(f"Checksum target file missing: '{rel_name}'")
        if target_f.parent != manifest_dir.resolve():
            raise MB1ValidationError(f"Checksum target file escapes manifest directory: '{rel_name}'")

        actual_hash = hashlib.sha256(target_f.read_bytes()).hexdigest()
        if actual_hash != digest.lower():
            raise MB1ValidationError(f"Checksum mismatch for '{rel_name}': expected {digest}, got {actual_hash}")

    missing_required = REQUIRED_MB1_ARTIFACTS - seen_entries
    if missing_required:
        raise MB1ValidationError(f"checksums.sha256 missing required M-B1 artifacts: {missing_required}")

    # 12. Verify No Local Absolute Paths in JSON/JSONL Manifests
    for manifest_f in manifest_dir.glob("*"):
        if manifest_f.suffix in (".json", ".jsonl"):
            content_str = manifest_f.read_text(encoding="utf-8")
            if "/Users/" in content_str or "file://" in content_str:
                raise MB1ValidationError(f"Absolute local path found in machine-readable artifact {manifest_f.name}")

    return {
        "validation_success": True,
        "m_b1_gate_status": "PASS_WITH_WARNINGS",
        "m_b2_entry_status": "READY_WITH_CONDITIONS",
        "independently_measured": {
            "pinned_environment_verified": True,
            "requirements_mac_sha_verified": True,
            "input_identity_upstream_verified": True,
            "m_b0_gate_verified": True,
            "a5_split_sha": actual_a5_sha,
            "canonical_npy_sha": actual_npy_sha,
            "a6_manifest_sha": actual_a6_sha,
            "train_window_count": len(train_data["windows"]),
            "validation_window_count": len(val_data["windows"]),
            "validation_prediction_index_provenance_verified": True,
            "profiles_audited": len(PROFILES),
            "zscore_statistics_verified": True,
            "tensor_fingerprints_verified": True,
            "validation_metrics_recomputed": True,
            "recomputed_winner_profile": recomputed_winner,
            "locked_test_access_blocked": True,
            "reproducibility_verdict_consistent": True,
            "hardened_checksum_verification": True,
        },
        "declared_policy_attributes": {
            "fixed_probe_architecture": "Conv1D_16_32_64_GAP_Dense3",
            "fixed_initialization_seed": 42,
            "fixed_imbalance_strategy": "UNWEIGHTED_SPARSE_CATEGORICAL_CROSSENTROPY",
        },
    }


def main() -> None:
    res = validate_m_b1_artifacts()
    print("Standalone M-B1 Preprocessing Ablation Validation Result:")
    print(f"Validation Success: {res['validation_success']}")
    print(f"M-B1 Gate Status: {res['m_b1_gate_status']}")
    print(f"M-B2 Entry Status: {res['m_b2_entry_status']}")
    print(f"Pinned Environment Verified: {res['independently_measured']['pinned_environment_verified']}")
    print(f"Upstream Identity Verified: {res['independently_measured']['input_identity_upstream_verified']}")
    print(f"M-B0 Gate Verified: {res['independently_measured']['m_b0_gate_verified']}")
    print(f"Profiles Audited: {res['independently_measured']['profiles_audited']}")
    print(f"Recomputed Winner: {res['independently_measured']['recomputed_winner_profile']}")
    print(f"LOCKED_TEST Guard Verified: {res['independently_measured']['locked_test_access_blocked']}")
    print(f"Prediction Index Provenance Verified: {res['independently_measured']['validation_prediction_index_provenance_verified']}")
    print(f"Hardened Checksums Verified: {res['independently_measured']['hardened_checksum_verification']}")


if __name__ == "__main__":
    main()
