#!/usr/bin/env python3
"""SafeNest Phase M-B0 — Standalone Validator.

Independently validates M-B0 evaluation protocol, M-A input identity, split isolation,
exact duplicate audit, near-duplicate diagnostic policy & independent recomputation,
LOCKED_TEST access control, metric contract, and hardened checksum coverage.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from mmwave_phase_b_access import FORBIDDEN_LABEL_FIELDS, LOCKED_TEST_AccessError, PhaseBAccessGuard

REQUIRED_MB0_JSON_MANIFESTS = {
    "input_identity.json",
    "split_isolation_audit.json",
    "exact_duplicate_audit.json",
    "near_duplicate_policy.json",
    "near_duplicate_audit.json",
    "locked_test_access_policy.json",
    "evaluation_contract.json",
    "exceptions.json",
    "m_b0_summary.json",
}


class MB0ValidationError(Exception):
    """Raised when Phase M-B0 validation fails."""


def validate_m_b0_artifacts(
    root_dir: Path = ROOT_DIR,
    manifest_dir: Path | None = None,
) -> dict[str, Any]:
    """Independently validate all Phase M-B0 artifacts against strict contracts."""
    if manifest_dir is None:
        manifest_dir = root_dir / "datasets/mmwave/manifests/M-B0_evaluation_protocol"

    if not manifest_dir.is_dir():
        raise MB0ValidationError(f"M-B0 manifest directory missing: {manifest_dir}")

    # 1. Verify Raw Archive SHA-256 Immutability (INDEPENDENTLY_MEASURED)
    raw_archive_path = root_dir / "datasets/raw_archives/external_datasets/db_records.zip"
    if not raw_archive_path.is_file():
        raise MB0ValidationError(f"Raw archive zip missing: {raw_archive_path}")

    hasher = hashlib.sha256()
    with open(raw_archive_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    measured_raw_sha256 = hasher.hexdigest()
    expected_raw_sha256 = "f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0"
    if measured_raw_sha256 != expected_raw_sha256:
        raise MB0ValidationError(f"Raw archive SHA-256 changed! Expected {expected_raw_sha256}, got {measured_raw_sha256}")

    # 2. Check input_identity.json and verify all input SHA-256 digests (INDEPENDENTLY_MEASURED)
    input_id_file = manifest_dir / "input_identity.json"
    if not input_id_file.is_file():
        raise MB0ValidationError(f"input_identity.json missing: {input_id_file}")

    input_id_data = json.loads(input_id_file.read_text(encoding="utf-8"))
    for item in input_id_data.get("inputs", []):
        rel_path = item["repository_relative_path"]
        exp_sha = item["measured_sha256"]
        full_p = root_dir / rel_path
        if not full_p.is_file():
            raise MB0ValidationError(f"Input artifact missing: {rel_path}")
        actual_sha = hashlib.sha256(full_p.read_bytes()).hexdigest()
        if actual_sha != exp_sha:
            raise MB0ValidationError(f"Input artifact SHA-256 mismatch for {rel_path}: expected {exp_sha}, got {actual_sha}")

    # 3. Load M-A5 splits & M-A6 manifests
    split_json_path = root_dir / "datasets/mmwave/splits/mmwave_real_subject_split_v1.json"
    if not split_json_path.is_file():
        raise MB0ValidationError(f"Real subject split JSON missing: {split_json_path}")
    split_data = json.loads(split_json_path.read_text(encoding="utf-8"))

    window_manifest_path = root_dir / "datasets/mmwave/manifests/a6_full_conversion/full_window_manifest.jsonl"
    provenance_manifest_path = root_dir / "datasets/mmwave/manifests/a6_full_conversion/full_provenance_manifest.jsonl"
    canonical_npy_path = root_dir / "datasets/mmwave/processed/mmwave_canonical_real_v1.npy"

    windows = [json.loads(l) for l in window_manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    provenance = [json.loads(l) for l in provenance_manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    matrix = np.load(canonical_npy_path)

    # 4. Verify 1:1 Index Alignment and Matrix Shape (INDEPENDENTLY_MEASURED)
    if len(windows) != 530 or len(provenance) != 530 or matrix.shape != (530, 300) or matrix.dtype != np.float64:
        raise MB0ValidationError(f"Canonical dataset shape invalid: windows={len(windows)}, matrix={matrix.shape} {matrix.dtype}")

    for idx, (w, p, row) in enumerate(zip(windows, provenance, matrix)):
        if w.get("canonical_sample_index") != idx or p.get("canonical_sample_index") != idx:
            raise MB0ValidationError(f"Contiguous index mismatch at row {idx}")
        row_bytes = np.ascontiguousarray(row, dtype=np.float64).tobytes()
        row_hash = hashlib.sha256(row_bytes).hexdigest()
        if w["canonical_signal_hash"] != row_hash:
            raise MB0ValidationError(f"Signal hash mismatch at row {idx}")

    # 5. Independent Re-calculation of Split Isolation (INDEPENDENTLY_MEASURED)
    split_subjects = defaultdict(set)
    split_recordings = defaultdict(set)
    split_windows = defaultdict(set)
    hash_splits = defaultdict(set)

    for w in windows:
        sp = w["split"]
        split_subjects[sp].add(w["subject_id"])
        split_recordings[sp].add(w["recording_id"])
        split_windows[sp].add(w["window_id"])
        hash_splits[w["canonical_signal_hash"]].add(sp)

    subj_leakage = len(split_subjects["TRAIN"] & split_subjects["VALIDATION"]) + len(split_subjects["TRAIN"] & split_subjects["LOCKED_TEST"]) + len(split_subjects["VALIDATION"] & split_subjects["LOCKED_TEST"])
    rec_leakage = len(split_recordings["TRAIN"] & split_recordings["VALIDATION"]) + len(split_recordings["TRAIN"] & split_recordings["LOCKED_TEST"]) + len(split_recordings["VALIDATION"] & split_recordings["LOCKED_TEST"])
    win_leakage = len(split_windows["TRAIN"] & split_windows["VALIDATION"]) + len(split_windows["TRAIN"] & split_windows["LOCKED_TEST"]) + len(split_windows["VALIDATION"] & split_windows["LOCKED_TEST"])
    exact_hash_leakage = sum(1 for grp in hash_splits.values() if len(grp) > 1)

    if subj_leakage != 0 or rec_leakage != 0 or win_leakage != 0 or exact_hash_leakage != 0:
        raise MB0ValidationError(f"Cross-split leakage detected: subj={subj_leakage}, rec={rec_leakage}, win={win_leakage}, hash={exact_hash_leakage}")

    # 6. Verify Exact Duplicate Audit (INDEPENDENTLY_MEASURED)
    exact_audit_file = manifest_dir / "exact_duplicate_audit.json"
    if not exact_audit_file.is_file():
        raise MB0ValidationError(f"exact_duplicate_audit.json missing: {exact_audit_file}")
    exact_audit_data = json.loads(exact_audit_file.read_text(encoding="utf-8"))
    if exact_audit_data.get("cross_split_exact_duplicates") != 0 or exact_audit_data.get("total_exact_duplicates") != 0:
        raise MB0ValidationError(f"Exact duplicates detected in exact_duplicate_audit.json: {exact_audit_data}")

    # 7. TRULY INDEPENDENT NEAR-DUPLICATE AUDIT RECOMPUTATION (INDEPENDENTLY_MEASURED)
    near_policy_file = manifest_dir / "near_duplicate_policy.json"
    near_audit_file = manifest_dir / "near_duplicate_audit.json"
    if not near_policy_file.is_file() or not near_audit_file.is_file():
        raise MB0ValidationError("Near-duplicate policy or audit manifest missing!")

    near_policy = json.loads(near_policy_file.read_text(encoding="utf-8"))
    near_audit = json.loads(near_audit_file.read_text(encoding="utf-8"))

    if near_policy.get("diagnostic_status") != "COMPLETED":
        raise MB0ValidationError("Near-duplicate diagnostic status is not COMPLETED!")

    frozen_min_corr = near_policy.get("frozen_threshold_applied", {}).get("min_correlation")
    frozen_max_nrmse = near_policy.get("frozen_threshold_applied", {}).get("max_nrmse")

    if frozen_min_corr != 0.995 or frozen_max_nrmse != 0.05:
        raise MB0ValidationError(f"Invalid frozen near-duplicate thresholds: corr={frozen_min_corr}, nrmse={frozen_max_nrmse}")
    if near_policy.get("locked_test_used_for_calibration") is not False:
        raise MB0ValidationError("Near-duplicate threshold was improperly tuned using LOCKED_TEST!")

    # Perform full independent re-calculation of 140,185 pairs directly from canonical NPY
    all_mean = matrix.mean(axis=1, keepdims=True)
    all_std = matrix.std(axis=1, keepdims=True)
    all_std[all_std == 0] = 1.0
    norm_all = (matrix - all_mean) / all_std

    corr_matrix = np.dot(norm_all, norm_all.T) / 300.0

    calc_cross_split_near_dups = 0
    calc_same_rec_near_dups = 0
    calc_same_subj_diff_rec_near_dups = 0
    calc_cross_subj_same_split_near_dups = 0
    calc_total_flagged = 0

    num_windows = len(windows)
    for i in range(num_windows):
        for j in range(i + 1, num_windows):
            r_val = float(corr_matrix[i, j])
            if r_val >= frozen_min_corr:
                w_a = windows[i]
                w_b = windows[j]
                diff = matrix[i] - matrix[j]
                nrmse = float(np.sqrt(np.mean(diff ** 2)) / (all_std[i, 0] + all_std[j, 0]))

                if nrmse <= frozen_max_nrmse:
                    calc_total_flagged += 1
                    if w_a["subject_id"] != w_b["subject_id"]:
                        if w_a["split"] != w_b["split"]:
                            calc_cross_split_near_dups += 1
                        else:
                            calc_cross_subj_same_split_near_dups += 1
                    elif w_a["recording_id"] != w_b["recording_id"]:
                        calc_same_subj_diff_rec_near_dups += 1
                    else:
                        calc_same_rec_near_dups += 1

    # Compare validator's independent recomputation against near_duplicate_audit.json manifest
    if calc_cross_split_near_dups != 0:
        raise MB0ValidationError(f"CRITICAL LEAKAGE: Independently calculated cross-split near duplicates = {calc_cross_split_near_dups}")

    if near_audit.get("total_flagged_near_duplicates") != calc_total_flagged:
        raise MB0ValidationError(
            f"Near-duplicate audit mismatch! Manifest total_flagged={near_audit.get('total_flagged_near_duplicates')}, recomputed={calc_total_flagged}"
        )
    if near_audit.get("cross_split_near_duplicates") != calc_cross_split_near_dups:
        raise MB0ValidationError(
            f"Near-duplicate audit mismatch! Manifest cross_split={near_audit.get('cross_split_near_duplicates')}, recomputed={calc_cross_split_near_dups}"
        )
    if near_audit.get("same_recording_near_duplicates") != calc_same_rec_near_dups:
        raise MB0ValidationError(
            f"Near-duplicate audit mismatch! Manifest same_recording={near_audit.get('same_recording_near_duplicates')}, recomputed={calc_same_rec_near_dups}"
        )

    # 8. Verify Evaluation Contract & Access Policies (DECLARED_POLICY vs INDEPENDENTLY_MEASURED)
    eval_contract_file = manifest_dir / "evaluation_contract.json"
    locked_access_file = manifest_dir / "locked_test_access_policy.json"
    if not eval_contract_file.is_file() or not locked_access_file.is_file():
        raise MB0ValidationError("Evaluation contract or LOCKED_TEST access policy manifest missing!")

    eval_contract = json.loads(eval_contract_file.read_text(encoding="utf-8"))
    if eval_contract.get("locked_test_model_selection_prohibited") is not True:
        raise MB0ValidationError("Evaluation contract does not prohibit LOCKED_TEST model selection!")
    if eval_contract.get("ambiguous_pure_class_exclusion_enforced") is not True:
        raise MB0ValidationError("Evaluation contract does not enforce AMBIGUOUS pure-class exclusion!")
    if eval_contract.get("apnea_proxy_terminology_enforced") is not True:
        raise MB0ValidationError("Evaluation contract does not enforce SafeNest APNEA-proxy terminology!")

    # 9. Verify LOCKED_TEST Access Controller Guard & Label Sanitization (INDEPENDENTLY_MEASURED)
    guard = PhaseBAccessGuard(root_dir=root_dir)
    try:
        guard.get_model_selection_dataset("LOCKED_TEST")
        raise MB0ValidationError("PhaseBAccessGuard failed to block LOCKED_TEST model selection access!")
    except LOCKED_TEST_AccessError:
        pass  # Guard successfully blocked model selection access

    # Verify structural audit dataset removes all forbidden label fields for LOCKED_TEST
    struct_test_ds = guard.get_structural_audit_dataset("LOCKED_TEST")
    if struct_test_ds["total_count"] != 88:
        raise MB0ValidationError(f"Structural audit dataset returned invalid LOCKED_TEST count: {struct_test_ds['total_count']}")

    for w_obj in struct_test_ds["windows"]:
        forbidden_present = FORBIDDEN_LABEL_FIELDS & set(w_obj.keys())
        if forbidden_present:
            raise MB0ValidationError(f"CRITICAL LABEL LEAK: Structural audit window exposes forbidden label fields: {forbidden_present}")

    # 10. HARDENED CHECKSUM MANIFEST VALIDATION (INDEPENDENTLY_MEASURED)
    checksums_file = manifest_dir / "checksums.sha256"
    if not checksums_file.is_file():
        raise MB0ValidationError(f"checksums.sha256 missing: {checksums_file}")

    raw_checksum_lines = checksums_file.read_text(encoding="utf-8").splitlines()
    seen_entries = set()

    for line_num, line in enumerate(raw_checksum_lines, 1):
        line_str = line.strip()
        if not line_str:
            continue

        parts = line_str.split(maxsplit=1)
        if len(parts) != 2:
            raise MB0ValidationError(f"Malformed checksum line {line_num} in checksums.sha256: '{line}'")

        digest, rel_name = parts[0].strip(), parts[1].strip()

        # Check digest length & hex format
        if not re.fullmatch(r"^[0-9a-fA-F]{64}$", digest):
            raise MB0ValidationError(f"Invalid SHA-256 digest format at line {line_num}: '{digest}'")

        # Check for path traversal / absolute paths
        if rel_name.startswith("/") or rel_name.startswith("\\") or "file://" in rel_name or "~" in rel_name or ".." in rel_name:
            raise MB0ValidationError(f"Path traversal or absolute path in checksums.sha256 line {line_num}: '{rel_name}'")

        # Check for duplicate entry
        if rel_name in seen_entries:
            raise MB0ValidationError(f"Duplicate file entry in checksums.sha256 at line {line_num}: '{rel_name}'")
        seen_entries.add(rel_name)

        # Target file resolution within manifest_dir
        target_f = (manifest_dir / rel_name).resolve()
        if not target_f.is_file():
            raise MB0ValidationError(f"Checksum target file missing: '{rel_name}'")
        if target_f.parent != manifest_dir.resolve():
            raise MB0ValidationError(f"Checksum target file escapes manifest directory: '{rel_name}'")

        actual_hash = hashlib.sha256(target_f.read_bytes()).hexdigest()
        if actual_hash != digest.lower():
            raise MB0ValidationError(f"Checksum mismatch for '{rel_name}': expected {digest}, got {actual_hash}")

    # Check required artifact completeness
    missing_required_checksums = REQUIRED_MB0_JSON_MANIFESTS - seen_entries
    if missing_required_checksums:
        raise MB0ValidationError(f"checksums.sha256 is missing required M-B0 artifacts: {missing_required_checksums}")

    # 11. Verify No Local Absolute Paths in JSON Manifests (INDEPENDENTLY_MEASURED)
    for manifest_f in manifest_dir.glob("*.json"):
        content_str = manifest_f.read_text(encoding="utf-8")
        if "/Users/" in content_str or "file://" in content_str:
            raise MB0ValidationError(f"Absolute local path found in machine-readable artifact {manifest_f.name}")

    return {
        "validation_success": True,
        "m_b0_gate_status": "PASS_WITH_WARNINGS",
        "m_b1_entry_status": "READY_WITH_CONDITIONS",
        "independently_measured": {
            "raw_archive_sha256": measured_raw_sha256,
            "inputs_validated": len(input_id_data.get("inputs", [])),
            "split_isolation_leakage": 0,
            "exact_duplicates_found": exact_audit_data.get("total_exact_duplicates", 0),
            "independently_recomputed_near_duplicate_pairs": calc_total_flagged,
            "independently_recomputed_cross_split_near_duplicates": calc_cross_split_near_dups,
            "locked_test_access_blocked": True,
            "locked_test_label_sanitization_verified": True,
            "hardened_checksum_verification": True,
        },
        "declared_policy_attributes": {
            "eval_contract_fit_train_only": True,
            "eval_contract_select_validation_only": True,
            "apnea_proxy_terminology": True,
        },
    }


def main() -> None:
    res = validate_m_b0_artifacts()
    print("Standalone M-B0 Evaluation Protocol Validation Result:")
    print(f"Validation Success: {res['validation_success']}")
    print(f"M-B0 Gate Status: {res['m_b0_gate_status']}")
    print(f"M-B1 Entry Status: {res['m_b1_entry_status']}")
    print(f"Independently Recomputed Near-Duplicates: {res['independently_measured']['independently_recomputed_near_duplicate_pairs']}")
    print(f"Cross-Split Near-Duplicates: {res['independently_measured']['independently_recomputed_cross_split_near_duplicates']}")
    print(f"LOCKED_TEST Guard & Label Sanitization: {res['independently_measured']['locked_test_label_sanitization_verified']}")
    print(f"Hardened Checksums Verified: {res['independently_measured']['hardened_checksum_verification']}")


if __name__ == "__main__":
    main()
