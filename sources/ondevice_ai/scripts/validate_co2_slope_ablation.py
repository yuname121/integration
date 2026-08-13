#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/validate_co2_slope_ablation.py
Standalone C-B1 validator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.co2.offline_experiment import (
    EXPECTED_LOCKED_TEST_SEALED,
    EXPECTED_TRAIN_COMMON,
    EXPECTED_VALIDATION_COMMON,
    assert_no_forbidden_path_markers,
    build_sample_universe_manifest,
)
from datasets.co2.raw_reader import compute_sha256_file, get_repo_root
from datasets.co2.slope_ablation import (
    A3_SLOPE_PROFILE_ID,
    A_SERIES_LOCK_PROFILE,
    A_SERIES_TAG,
    A_SERIES_TARGET,
    ARTIFACT_DIR_REL,
    AUTHORIZED_HISTORIES,
    AUTHORIZED_METHODS,
    BASELINE_CANDIDATE_ID,
    B0_DIR_REL,
    B0_PREDECESSOR_FILES,
    EXPECTED_LOCK_SHA256,
    FIXED_BASE_FEATURES,
    NO_SLOPE_CONTROL_ID,
    PRODUCTION_SCALER_REL,
    SELECTED_SLOPE_PROFILE_ID,
    build_candidate_registry,
    build_predecessor_fingerprint_registry,
    rank_slope_candidates,
    run_slope_ablation,
)

REQUIRED_ARTIFACTS = [
    "predecessor_fingerprint_registry.json",
    "slope_ablation_contract.json",
    "candidate_registry.json",
    "candidate_feature_fingerprint_registry.json",
    "candidate_diagnostics.json",
    "fixed_probe_contract.json",
    "preprocessing_evidence.json",
    "probe_fit_evidence.json",
    "validation_metric_results.json",
    "validation_predictions.json",
    "no_slope_control_result.json",
    "selection_decision.json",
    "selected_slope_profile.json",
    "leakage_audit.json",
    "exceptions_and_limitations.json",
    "generation_metadata.json",
    "candidate_availability_summary.json",
    "artifact_identity.json",
    "checksums.sha256",
]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(repo_root: Path, *, rerun_determinism: bool = True) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    out = repo_root / ARTIFACT_DIR_REL

    for name in REQUIRED_ARTIFACTS:
        if not (out / name).is_file():
            errors.append(f"missing artifact: {name}")
    if errors:
        return {"status": "FAIL", "errors": errors, "warnings": warnings}

    # Predecessor / A-series
    try:
        pred = build_predecessor_fingerprint_registry(repo_root)
        stored_pred = _load(out / "predecessor_fingerprint_registry.json")
        if stored_pred.get("a_series_artifact_lock_sha256") != EXPECTED_LOCK_SHA256:
            errors.append("A-series lock sha mismatch")
        if stored_pred.get("a_series_release_tag") != A_SERIES_TAG:
            errors.append("A-series tag mismatch")
        if stored_pred.get("a_series_release_target") != A_SERIES_TARGET:
            errors.append("A-series target mismatch")
        if stored_pred.get("file_sha256") != pred["file_sha256"]:
            errors.append("C_B0_PREDECESSOR_FINGERPRINT_MISMATCH")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"predecessor registry failure: {exc}")

    universe = build_sample_universe_manifest(repo_root)
    if universe["b_series_common_train"] != EXPECTED_TRAIN_COMMON:
        errors.append("TRAIN count != 8140")
    if universe["b_series_common_validation"] != EXPECTED_VALIDATION_COMMON:
        errors.append("VALIDATION count != 2662")
    if universe["b_series_sealed_locked_test"] != EXPECTED_LOCKED_TEST_SEALED:
        errors.append("LOCKED_TEST count != 9749")

    registry = _load(out / "candidate_registry.json")
    if registry.get("candidate_count") != 6:
        errors.append("candidate count != 6")
    methods = {c["method"] for c in registry["candidates"]}
    histories = {float(c["minimum_history_seconds"]) for c in registry["candidates"]}
    if methods != set(AUTHORIZED_METHODS):
        errors.append(f"unauthorized methods: {methods}")
    if histories != set(AUTHORIZED_HISTORIES):
        errors.append(f"unauthorized histories: {histories}")
    live_reg = build_candidate_registry()
    if registry.get("registry_fingerprint") != live_reg["registry_fingerprint"]:
        errors.append("candidate registry fingerprint drift / post-hoc change")

    avail = _load(out / "candidate_availability_summary.json")
    for cid, row in avail["per_candidate"].items():
        if row["train_available"] != EXPECTED_TRAIN_COMMON:
            errors.append(f"{cid} TRAIN availability != 8140")
        if row["validation_available"] != EXPECTED_VALIDATION_COMMON:
            errors.append(f"{cid} VALIDATION availability != 2662")
        if row["train_unavailable"] or row["validation_unavailable"]:
            errors.append(f"{cid} candidate-specific dropping detected")
        if row["train_non_finite"] or row["validation_non_finite"]:
            errors.append(f"{cid} non-finite slope values")

    decision = _load(out / "selection_decision.json")
    parity = decision.get("endpoint_h150_a3_parity") or {}
    if parity.get("status") != "PASS":
        errors.append("ENDPOINT_H150 C-A3 parity failure")
    if parity.get("status_mismatches") not in (0, None):
        errors.append("C-A3 status mismatches != 0")
    if parity.get("value_mismatches") not in (0, None):
        errors.append("C-A3 value mismatches != 0")

    metrics = _load(out / "validation_metric_results.json")
    if metrics.get("locked_test_metrics") != 0:
        errors.append("LOCKED_TEST metrics present")
    preds = _load(out / "validation_predictions.json")
    if preds.get("locked_test_predictions") != 0:
        errors.append("LOCKED_TEST predictions present")

    # Reconstruct ranking independently
    cand_rows = []
    for cid, m in metrics["candidates"].items():
        spec = next(c for c in registry["candidates"] if c["candidate_id"] == cid)
        cand_rows.append(
            {
                "candidate_id": cid,
                "method": spec["method"],
                "minimum_history_seconds": spec["minimum_history_seconds"],
                "validation_metrics": m,
            }
        )
    recomputed = rank_slope_candidates(cand_rows)
    if recomputed != decision["ranking"]:
        errors.append("ranking does not match independently recomputed rule")
    if decision["winning_slope_candidate"] != recomputed[0]["candidate_id"]:
        errors.append("selected winner mismatch vs recomputed ranking")
    if NO_SLOPE_CONTROL_ID in {r["candidate_id"] for r in decision["ranking"]}:
        errors.append("no-slope control entered slope ranking")

    selected = _load(out / "selected_slope_profile.json")
    if selected.get("profile_id") != SELECTED_SLOPE_PROFILE_ID:
        errors.append("selected profile id mismatch")
    if selected.get("locked_test_status") != "SEALED":
        errors.append("selected profile LOCKED_TEST not sealed")
    if selected.get("deployment_status") != "NOT_VALIDATED":
        errors.append("selected profile deployment status incorrect")
    if selected.get("final_feature_selection") != "NOT_PERFORMED":
        errors.append("final feature selection incorrectly marked")
    if selected.get("a_series_baseline_profile") != A3_SLOPE_PROFILE_ID:
        errors.append("A-series baseline identity missing")

    # Feature context / leakage
    contract = _load(out / "slope_ablation_contract.json")
    feats = contract["feature_context"]["slope_candidate_features"]
    if feats != list(FIXED_BASE_FEATURES) + ["candidate_CO2_slope"]:
        errors.append("fixed feature context mismatch")
    if contract["feature_context"].get("light_excluded") is not True:
        errors.append("Light not excluded")
    leakage = _load(out / "leakage_audit.json")
    for key in (
        "train_validation_overlap",
        "train_locked_test_overlap",
        "validation_locked_test_overlap",
        "target_included_as_feature",
        "provenance_included_as_feature",
        "locked_test_metrics",
        "locked_test_predictions",
        "locked_test_candidate_values_in_result_artifacts",
    ):
        if leakage.get(key) not in (0, False):
            errors.append(f"leakage audit failed: {key}")
    if leakage.get("validation_used_to_fit_scaler") is not False:
        errors.append("VALIDATION scaler fit leakage")
    if leakage.get("validation_used_to_fit_probe") is not False:
        errors.append("VALIDATION probe fit leakage")
    if leakage.get("candidate_grid_changed_after_evaluation") is not False:
        errors.append("candidate grid changed after evaluation")

    # A-series / production unchanged
    a3 = repo_root / "datasets/co2/manifests/c_a3_slope_feature/co2_slope_feature_profile.json"
    a3_doc = _load(a3)
    if a3_doc.get("profile_id") != A3_SLOPE_PROFILE_ID:
        errors.append("A-series slope profile identity damaged")
    prod_scaler = repo_root / PRODUCTION_SCALER_REL
    if not prod_scaler.is_file():
        warnings.append("production scaler metadata missing (unexpected)")
    # Ensure B1 did not overwrite A3 profile path content vs predecessor fingerprint
    pred_files = _load(out / "predecessor_fingerprint_registry.json")["file_sha256"]
    a3_rel = "datasets/co2/manifests/c_a3_slope_feature/co2_slope_feature_profile.json"
    if compute_sha256_file(a3) != pred_files.get(a3_rel):
        errors.append("A-series slope profile modified")

    # Portable paths + checksums
    checksum_path = out / "checksums.sha256"
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        path = repo_root / rel
        if not path.is_file():
            errors.append(f"checksum path missing: {rel}")
            continue
        if compute_sha256_file(path) != digest:
            errors.append(f"checksum mismatch: {rel}")
        bad = assert_no_forbidden_path_markers(path.read_text(encoding="utf-8"))
        if bad:
            errors.append(f"absolute/forbidden path in {rel}: {bad}")

    # Fingerprints exist for TRAIN/VAL only
    fps = _load(out / "candidate_feature_fingerprint_registry.json")
    for cid, block in fps["candidates"].items():
        if "LOCKED_TEST" in block and block["LOCKED_TEST"].get("status") != "NOT_GENERATED":
            errors.append(f"LOCKED_TEST fingerprint present for {cid}")
        if block["TRAIN"]["count"] != EXPECTED_TRAIN_COMMON:
            errors.append(f"{cid} TRAIN fp count")
        if block["VALIDATION"]["count"] != EXPECTED_VALIDATION_COMMON:
            errors.append(f"{cid} VAL fp count")

    # Determinism regeneration
    if rerun_determinism:
        before = {
            p.name: compute_sha256_file(p)
            for p in out.iterdir()
            if p.is_file()
        }
        run_slope_ablation(repo_root)
        after = {
            p.name: compute_sha256_file(p)
            for p in out.iterdir()
            if p.is_file()
        }
        if before != after:
            drift = sorted(
                k for k in set(before) | set(after) if before.get(k) != after.get(k)
            )
            errors.append(f"determinism mismatch: {drift}")

    # Contamination markers in B1 artifacts
    for path in out.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        for marker in ("mmwave", "thermal44", "/Users/", "file://"):
            if marker in text and marker in ("/Users/", "file://"):
                errors.append(f"forbidden marker {marker} in {path.name}")

    status = "PASS" if not errors else "FAIL"
    if status == "PASS" and warnings:
        status = "PASS_WITH_WARNINGS"
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "artifact_dir": ARTIFACT_DIR_REL,
        "winner": decision.get("winning_slope_candidate"),
        "a_series_lock_profile": A_SERIES_LOCK_PROFILE,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate C-B1 CO2 slope ablation")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--skip-determinism", action="store_true")
    args = parser.parse_args()
    root = args.repo_root or get_repo_root()
    result = validate(root, rerun_determinism=not args.skip_determinism)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in ("PASS", "PASS_WITH_WARNINGS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
