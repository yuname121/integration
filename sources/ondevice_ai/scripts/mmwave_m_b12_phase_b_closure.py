#!/usr/bin/env python3
"""M-B12 Phase-B offline final report and intermediate-release-readiness closure.

Summarizes and validates existing locked evidence only.
Does not access LOCKED_TEST, invoke TFLite, train, convert, calibrate,
retune, or reselect the candidate. Does not create a git tag or GitHub Release.
Does not begin M-C.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.mmwave_m_b10r1_result_writer import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from scripts.mmwave_m_b11_artifact_lock import (  # noqa: E402
    A5_SPLIT_REL,
    A6_MANIFEST_REL,
    ARCHITECTURE_ID,
    ARTIFACT_STATUS,
    CALIBRATION_ID,
    CANONICAL_NPY_REL,
    CLASS_MAP,
    EXECUTION_PREPROCESSING_CONTRACT_ID,
    EXPECTED_ELIGIBLE,
    EXPECTED_PAIRS,
    LOCK_DIR_REL as M_B11_DIR_REL,
    LOCK_JSON_FILES as M_B11_JSON_FILES,
    PREPROCESSING_PROFILE_ID,
    PREPROCESSING_PROFILE_NAME,
    RAW_ARCHIVE_REL,
    RESULT_LIMITATION,
    RUNTIME_MODEL_ID,
    SELECTED_CANDIDATE_ID,
    SELECTED_TFLITE_REL,
    SENSOR_LOCK_REL,
    TRAINING_STRATEGY_ID,
    load_json,
    require_repo_relative,
)

PHASE_ID = "M-B12"
SCHEMA = "M-B12_PHASE_B_OFFLINE_FINAL_V1"
CLOSURE_DIR_REL = Path("datasets/mmwave/manifests/M-B12_phase_b_offline_final")
REPORT_REL = Path("docs/reports/20260813_Cursor_M-B12_mmWave_Phase_B_Offline_Final_Report_01.md")
STATUS_LABEL = "PHASE_B_OFFLINE_INTERMEDIATE_RELEASE_READY_AFTER_MERGE"
PROPOSED_TAG = "mmwave-phase-b-offline-candidate"
EXPECTED_MODEL_SHA = "6dff6aaa72c79d76715d40cf7e32bb1e6cd9b2c2e3ac78eaf2fda737561430c5"
EXPECTED_MODEL_BYTES = 22080
EXPECTED_MACRO_F1 = 0.494836
EXPECTED_V01_F1 = 0.166667
EXPECTED_V02_F1 = 0.391074
EXPECTED_MODELS = 3
EXPECTED_HISTORICAL_RELEASES = 2
EXPECTED_RECOVERY_INFERENCE = 225

CLOSURE_JSON_FILES = (
    "phase_b_closure_identity.json",
    "predecessor_gate.json",
    "source_and_population_summary.json",
    "selected_path_lineage.json",
    "locked_candidate_summary.json",
    "final_evaluation_summary.json",
    "scientific_limitations.json",
    "claim_boundary.json",
    "release_readiness_manifest.json",
    "device_domain_handoff.json",
    "immutable_evidence_registry.json",
    "phase_b_required_role_registry.json",
    "final_report_identity.json",
    "phase_b_closure_summary.json",
    "validation_result.json",
)

# Exact role strings from datasets/mmwave/manifests/M-B11_artifact_lock/immutable_artifact_registry.json
REQUIRED_M11_REGISTRY_ROLES: tuple[tuple[str, str, str], ...] = (
    ("raw_source_archive", "A0", "A/source lineage"),
    ("a0_source_identity", "A0", "A/source lineage"),
    ("a5_subject_split", "A5", "A5 split"),
    ("a5_summary", "A5", "A5 split"),
    ("a5_split_profile", "A5", "A5 split"),
    ("a6_window_manifest", "A6", "A6 conversion/window manifest"),
    ("a6_summary", "A6", "A6 conversion/window manifest"),
    ("canonical_npy", "A6", "A6 conversion/window manifest"),
    ("b0_summary", "M-B0", "M-B0"),
    ("b1_selected_profile", "M-B1", "M-B1"),
    ("b1_summary", "M-B1", "M-B1"),
    ("b1_executor", "M-B1", "M-B1"),
    ("b2_selected_strategy", "M-B2", "M-B2"),
    ("b2_summary", "M-B2", "M-B2"),
    ("b3_summary", "M-B3", "M-B3"),
    ("b4_summary", "M-B4", "M-B4"),
    ("b5_summary", "M-B5", "M-B5"),
    ("b6_summary", "M-B6", "M-B6"),
    ("b6_stage_artifact_manifest", "M-B6", "M-B6"),
    ("selected_tflite", "M-B6", "selected seed42 model artifact"),
    ("b7_summary", "M-B7", "M-B7"),
    ("b8_summary", "M-B8", "M-B8"),
    ("b9_summary", "M-B9", "M-B9"),
    ("b9_seed42_runtime_manifest", "M-B9", "M-B9"),
    ("b10a_summary", "M-B10A", "M-B10A"),
    ("b10b_summary", "M-B10B", "M-B10B summary"),
    ("b10b_incident_root_cause", "M-B10B", "M-B10B incident/root cause"),
    ("b10r0_summary", "M-B10R0", "M-B10R0"),
    ("b10r1a_summary", "M-B10R1-A", "M-B10R1-A"),
    ("b10r1b_summary", "M-B10R1-B", "M-B10R1-B"),
    ("b10r1b_registry", "M-B10R1-B", "final recovery registry"),
    ("b10r1b_ledger", "M-B10R1-B", "final recovery ledger"),
    ("v0_1_tflite", "historical_baseline", "historical v0.1 baseline artifact"),
    ("v0_2_tflite", "historical_baseline", "historical v0.2 baseline artifact"),
    ("b10b_baseline_preprocessing_executor", "M-B10B", "M-B10B"),
    ("sensor_local_candidate_lock", "M-B11", "M-B11 sensor-local lock"),
)

REQUIRED_M11_LOCK_FILE_ROLES: tuple[tuple[str, str, str, str], ...] = (
    (
        "m_b11_lock_identity",
        "M-B11",
        "M-B11 lock identity/summary",
        "datasets/mmwave/manifests/M-B11_artifact_lock/artifact_lock_identity.json",
    ),
    (
        "m_b11_lock_summary",
        "M-B11",
        "M-B11 lock identity/summary",
        "datasets/mmwave/manifests/M-B11_artifact_lock/artifact_lock_summary.json",
    ),
)

MACHINE_FACTS_BEGIN = "<!-- MACHINE_VERIFIED_FINAL_FACTS -->"
MACHINE_FACTS_END = "<!-- END_MACHINE_VERIFIED_FINAL_FACTS -->"


class MB12ClosureError(Exception):
    """Fail-closed M-B12 generation error."""


def _raise(code: str) -> None:
    raise MB12ClosureError(code)


def _require_equal(actual: Any, expected: Any, code: str) -> None:
    if actual != expected:
        _raise(f"{code}:{actual}!={expected}")


def write_checksums(out: Path) -> None:
    lines = []
    for name in sorted(CLOSURE_JSON_FILES):
        path = out / name
        if not path.is_file():
            _raise(f"CHECKSUM_SOURCE_MISSING:{name}")
        lines.append(f"{sha256_file(path)}  {name}")
    atomic_write_text(out / "checksums.sha256", "\n".join(lines) + "\n")


def _load_m_b11(root: Path) -> dict[str, Any]:
    lock_dir = root / M_B11_DIR_REL
    if not lock_dir.is_dir():
        _raise("M_B11_LOCK_DIR_MISSING")
    return {name: load_json(lock_dir / name) for name in M_B11_JSON_FILES}


def _verify_live_sha(root: Path, rel: str, expected: str, code: str) -> str:
    path = root / require_repo_relative(rel, context=code)
    if not path.is_file():
        _raise(f"{code}_MISSING:{rel}")
    live = sha256_file(path)
    _require_equal(live, expected, code)
    return live


def print_start_report(root: Path) -> None:
    print("M-B12 START REPORT")
    print(f"Repository root: {root}")
    print("origin/main SHA: (see git; generator does not query remotes)")
    print("M-B11 merge commit expected in ancestry: ec0ab23a156686f25456fe43c7abe732e77a5acc")
    print("Branch: feature/M-B12-mmwave-phase-b-closure")
    print("LOCKED_TEST access during generation: 0")
    print("Recovery access during generation: 0")
    print("Inference during generation: 0")
    print("Training/conversion/calibration/reselection: 0")
    print("Git tag created: NO")
    print("GitHub Release created: NO")
    print("M-C started: NO")
    print(f"Candidate ID: {SELECTED_CANDIDATE_ID}")
    print(f"Runtime model ID: {RUNTIME_MODEL_ID}")
    print(f"Model path: {SELECTED_TFLITE_REL}")
    print(f"Result limitation: {RESULT_LIMITATION}")


def generate_m_b12_closure(root: Path | None = None) -> dict[str, Any]:
    root = Path(root) if root is not None else ROOT_DIR
    print_start_report(root)
    locks = _load_m_b11(root)
    identity = locks["artifact_lock_identity.json"]
    source = locks["source_lineage_lock.json"]
    canonical = locks["canonical_dataset_lock.json"]
    split = locks["subject_split_lock.json"]
    window = locks["window_population_lock.json"]
    prep = locks["preprocessing_lock.json"]
    train = locks["training_lock.json"]
    model = locks["model_artifact_lock.json"]
    quant = locks["quantization_lock.json"]
    runtime = locks["runtime_contract_lock.json"]
    lineage = locks["phase_b_lineage_registry.json"]
    final_eval = locks["final_evaluation_lock.json"]
    history = locks["recovery_access_history_lock.json"]
    registry = locks["final_sample_registry_lock.json"]
    metrics = locks["final_metric_lock.json"]
    subjects = locks["final_subject_metric_lock.json"]
    baselines = locks["baseline_comparison_lock.json"]
    limitations = locks["scientific_limitations.json"]
    claims = locks["claim_boundary_lock.json"]
    m11_summary = locks["artifact_lock_summary.json"]

    _require_equal(identity.get("artifact_status"), ARTIFACT_STATUS, "M11_STATUS")
    _require_equal(identity.get("result_limitation"), RESULT_LIMITATION, "M11_LIMITATION")
    _require_equal(identity.get("candidate_id"), SELECTED_CANDIDATE_ID, "M11_CANDIDATE")
    _require_equal(identity.get("runtime_model_id"), RUNTIME_MODEL_ID, "M11_RUNTIME")
    _require_equal(identity.get("m_b11_creates_new_model"), False, "M11_NEW_MODEL")
    if identity.get("result_not_pristine") is not True:
        _raise("M11_RESULT_PRISTINE")
    _require_equal(model.get("sha256"), EXPECTED_MODEL_SHA, "M11_MODEL_SHA")
    _require_equal(model.get("bytes"), EXPECTED_MODEL_BYTES, "M11_MODEL_BYTES")
    _require_equal(model.get("seed"), 42, "M11_SEED")
    _require_equal(metrics.get("macro_f1"), EXPECTED_MACRO_F1, "M11_MACRO_F1")
    _require_equal(baselines["v0_1"].get("macro_f1"), EXPECTED_V01_F1, "M11_V01_F1")
    _require_equal(baselines["v0_2"].get("macro_f1"), EXPECTED_V02_F1, "M11_V02_F1")
    _require_equal(registry.get("unique_eligible_window_ids"), EXPECTED_ELIGIBLE, "M11_UNIQUE")
    _require_equal(registry.get("actual_pairs"), EXPECTED_PAIRS, "M11_PAIRS")
    _require_equal(registry.get("duplicates"), 0, "M11_DUP")
    _require_equal(registry.get("missing"), 0, "M11_MISSING")
    _require_equal(registry.get("unexpected"), 0, "M11_UNEXPECTED")
    _require_equal(registry.get("cross_model_label_mismatches"), 0, "M11_LABEL")
    _require_equal(registry.get("cross_model_subject_mismatches"), 0, "M11_SUBJECT")
    _require_equal(registry.get("cross_model_recording_mismatches"), 0, "M11_RECORDING")
    _require_equal(history.get("historical_total_payload_releases"), EXPECTED_HISTORICAL_RELEASES, "M11_HIST")
    _require_equal(history.get("recovery_model_inference"), EXPECTED_RECOVERY_INFERENCE, "M11_INFER")
    if history.get("rerun") is not False or history.get("second_recovery") is not False:
        _raise("M11_RERUN_OR_SECOND")
    if claims.get("deployment_ready") is not False:
        _raise("M11_DEPLOYMENT_TRUE")
    if claims.get("clinical_apnea_validated") is not False:
        _raise("M11_CLINICAL_TRUE")
    if m11_summary.get("new_locked_test_access") != 0 or m11_summary.get("new_model_inference") != 0:
        _raise("M11_NEW_ACCESS")

    live_model_sha = _verify_live_sha(root, SELECTED_TFLITE_REL, EXPECTED_MODEL_SHA, "LIVE_MODEL_SHA")
    live_model_bytes = int((root / SELECTED_TFLITE_REL).stat().st_size)
    _require_equal(live_model_bytes, EXPECTED_MODEL_BYTES, "LIVE_MODEL_BYTES")
    _verify_live_sha(root, RAW_ARCHIVE_REL, source["raw_archive_sha256"], "LIVE_RAW_SHA")
    _verify_live_sha(root, CANONICAL_NPY_REL, canonical["canonical_npy_sha256"], "LIVE_NPY_SHA")
    _verify_live_sha(root, A5_SPLIT_REL, split["split_sha256"], "LIVE_A5_SHA")
    _verify_live_sha(root, A6_MANIFEST_REL, window["a6_manifest_sha256"], "LIVE_A6_SHA")
    sensor = load_json(root / SENSOR_LOCK_REL)
    _require_equal(sensor.get("sha256"), EXPECTED_MODEL_SHA, "SENSOR_SHA")
    _require_equal(sensor.get("status"), ARTIFACT_STATUS, "SENSOR_STATUS")
    m11_checksums = root / M_B11_DIR_REL / "checksums.sha256"
    if not m11_checksums.is_file():
        _raise("M11_CHECKSUMS_MISSING")
    m11_checksums_sha = sha256_file(m11_checksums)
    m11_registry_sha = sha256_file(root / M_B11_DIR_REL / "immutable_artifact_registry.json")

    selected_path = dict(lineage.get("selected_path") or {})
    a_series = {
        "A0": {
            "role": "raw_source_inventory",
            "summary": "datasets/mmwave/manifests/a0_raw_inventory/inventory_summary.json",
            "source_identity": "datasets/mmwave/manifests/a0_raw_inventory/source_identity.json",
            "doi": source["doi"],
            "raw_archive_sha256": source["raw_archive_sha256"],
        },
        "A1": {
            "role": "rfft_pilot",
            "summary": "datasets/mmwave/manifests/a1_rfft_pilot/a1_summary.json",
        },
        "A2": {
            "role": "phase_pilot",
            "summary": "datasets/mmwave/manifests/a2_phase_pilot/a2_summary.json",
        },
        "A3": {
            "role": "timeline_pilot",
            "summary": "datasets/mmwave/manifests/a3_timeline_pilot/a3_summary.json",
        },
        "A4": {
            "role": "label_pilot_apnea_proxy_not_clinical_apnea",
            "summary": "datasets/mmwave/manifests/a4_label_pilot/a4_summary.json",
            "apnea_is_proxy": True,
        },
        "A5": {
            "role": "immutable_subject_split",
            "summary": "datasets/mmwave/manifests/a5_subject_split/a5_summary.json",
            "split_seed": split["split_seed"],
            "split_sha256": split["split_sha256"],
            "subject_counts": split["subject_counts"],
        },
        "A6": {
            "role": "full_window_conversion",
            "summary": "datasets/mmwave/manifests/a6_full_conversion/a6_summary.json",
            "manifest_sha256": window["a6_manifest_sha256"],
            "total_canonical_windows": window["total_canonical_windows"],
            "structural": window["structural"],
            "pure_supervised_eligible": window["pure_supervised_eligible"],
        },
    }
    b12_path = {
        "role": "phase_b_offline_final_report_and_intermediate_release_readiness",
        "improves_candidate": False,
        "creates_new_model": False,
        "reopens_locked_test": False,
        "reopens_recovery": False,
        "creates_git_tag": False,
        "creates_github_release": False,
        "begins_m_c": False,
        "summary": str(CLOSURE_DIR_REL / "phase_b_closure_summary.json"),
    }

    closure_identity = {
        "schema_version": SCHEMA,
        "phase_id": PHASE_ID,
        "artifact_status": ARTIFACT_STATUS,
        "result_limitation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "candidate_id": SELECTED_CANDIDATE_ID,
        "runtime_model_id": RUNTIME_MODEL_ID,
        "class_map": CLASS_MAP,
        "apnea_is_proxy": True,
        "m_b12_creates_new_model": False,
        "m_b12_improves_candidate": False,
        "selected_candidate_changed": False,
        "artifact_identity_is_sha256_not_git_commit": True,
        "git_commit_sha_supplementary_only": True,
        "m_c_started": False,
    }
    predecessor = {
        "schema_version": SCHEMA,
        "m_b11_lock_dir": str(M_B11_DIR_REL),
        "m_b11_schema_version": identity.get("schema_version"),
        "m_b11_artifact_status": identity.get("artifact_status"),
        "m_b11_checksums_sha256": m11_checksums_sha,
        "m_b11_immutable_registry_sha256": m11_registry_sha,
        "m_b11_sensor_lock": str(SENSOR_LOCK_REL),
        "standalone_m_b11_validator": "scripts/validate_mmwave_m_b11.py",
        "m_b11_validator_required": True,
        "new_locked_test_access": 0,
        "new_recovery_access": 0,
        "new_model_inference": 0,
        "training": 0,
        "conversion": 0,
        "calibration": 0,
        "threshold_tuning": 0,
        "model_reselection": 0,
    }
    population = {
        "schema_version": SCHEMA,
        "raw_archive_repo_relative_path": RAW_ARCHIVE_REL,
        "raw_archive_sha256": source["raw_archive_sha256"],
        "raw_archive_bytes": source["raw_archive_bytes"],
        "doi": source["doi"],
        "version": source["version"],
        "measured_subjects": source["measured_subjects"],
        "measured_recordings": source["measured_recordings"],
        "canonical_npy_repo_relative_path": CANONICAL_NPY_REL,
        "canonical_npy_sha256": canonical["canonical_npy_sha256"],
        "canonical_npy_bytes": canonical["canonical_npy_bytes"],
        "shape": canonical["shape"],
        "dtype": canonical["dtype"],
        "split_artifact_repo_relative_path": A5_SPLIT_REL,
        "split_sha256": split["split_sha256"],
        "split_seed": split["split_seed"],
        "subject_counts": split["subject_counts"],
        "a6_manifest_repo_relative_path": A6_MANIFEST_REL,
        "a6_manifest_sha256": window["a6_manifest_sha256"],
        "total_canonical_windows": window["total_canonical_windows"],
        "structural": window["structural"],
        "pure_supervised_eligible": window["pure_supervised_eligible"],
        "class_totals": window["class_totals"],
        "locked_test_excluded_ambiguous_or_non_eligible": window["locked_test"]["excluded_ambiguous_or_non_eligible"],
        "ambiguous_is_proxy_exclusion_not_clinical_apnea": True,
    }
    path_lineage = {
        "schema_version": SCHEMA,
        "a_series": a_series,
        "selected_path": selected_path,
        "m_b11": lineage.get("m_b11"),
        "m_b12": b12_path,
    }
    candidate = {
        "schema_version": SCHEMA,
        "artifact_status": ARTIFACT_STATUS,
        "candidate_id": SELECTED_CANDIDATE_ID,
        "runtime_model_id": RUNTIME_MODEL_ID,
        "repo_relative_path": SELECTED_TFLITE_REL,
        "sha256": live_model_sha,
        "bytes": live_model_bytes,
        "seed": 42,
        "architecture_id": ARCHITECTURE_ID,
        "training_strategy_id": TRAINING_STRATEGY_ID,
        "preprocessing_profile_id": PREPROCESSING_PROFILE_ID,
        "preprocessing_profile_name": PREPROCESSING_PROFILE_NAME,
        "execution_preprocessing_contract_id": EXECUTION_PREPROCESSING_CONTRACT_ID,
        "calibration_profile": CALIBRATION_ID,
        "class_map": CLASS_MAP,
        "apnea_is_proxy": True,
        "strict_int8": True,
        "flex_ops_present": False,
        "select_tf_ops_present": False,
        "builtin_op_status": model["builtin_op_status"],
        "input_tensor": model["input_tensor"],
        "output_tensor": model["output_tensor"],
        "copied_or_renamed_during_m_b12": False,
        "models_model_manifest_json_modified": False,
    }
    evaluation = {
        "schema_version": SCHEMA,
        "result_designation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "not_final_locked_test_pristine": True,
        "unique_eligible_window_ids": registry["unique_eligible_window_ids"],
        "models": EXPECTED_MODELS,
        "expected_pairs": registry["expected_pairs"],
        "actual_pairs": registry["actual_pairs"],
        "duplicates": registry["duplicates"],
        "missing": registry["missing"],
        "unexpected": registry["unexpected"],
        "cross_model_label_mismatches": registry["cross_model_label_mismatches"],
        "cross_model_subject_mismatches": registry["cross_model_subject_mismatches"],
        "cross_model_recording_mismatches": registry["cross_model_recording_mismatches"],
        "original_m_b10b_accessor_invocations": history["original_m_b10b_accessor_invocations"],
        "original_m_b10b_payload_releases": history["original_m_b10b_payload_releases"],
        "original_m_b10b_model_inference": history["original_m_b10b_model_inference"],
        "m_b10r1b_recovery_accessor_invocations": history["m_b10r1b_recovery_accessor_invocations"],
        "m_b10r1b_recovery_payload_releases": history["m_b10r1b_recovery_payload_releases"],
        "historical_total_payload_releases": history["historical_total_payload_releases"],
        "recovery_model_inference": history["recovery_model_inference"],
        "rerun": False,
        "second_recovery": False,
        "eligible_evaluated": final_eval["eligible_evaluated"],
        "valid": final_eval["valid"],
        "invalid": final_eval["invalid"],
        "tflite_invocations_selected": final_eval["tflite_invocations_selected"],
        "tflite_invocations_all_models": final_eval["tflite_invocations_all_models"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "per_class": metrics["per_class"],
        "apnea_proxy": metrics["apnea_proxy"],
        "confusion_matrix": metrics["confusion_matrix"],
        "prediction_distribution": metrics["prediction_distribution"],
        "class_collapse": metrics["class_collapse"],
        "subject_count": subjects["subject_count"],
        "median_subject_macro_f1": subjects["median_subject_macro_f1"],
        "worst_subject_macro_f1": subjects["worst_subject_macro_f1"],
        "worst_subject_id": subjects["worst_subject_id"],
        "input_saturation_ratio": quant["input_saturation_ratio"],
        "pre_clamp_out_of_range_count": quant["pre_clamp_out_of_range_count"],
        "total_quantized_elements": quant["total_quantized_elements"],
        "samples_with_any_saturation": quant["samples_with_any_saturation"],
        "worst_sample_saturation_ratio": quant["worst_sample_saturation_ratio"],
        "v0_1_macro_f1": baselines["v0_1"]["macro_f1"],
        "v0_2_macro_f1": baselines["v0_2"]["macro_f1"],
        "v0_1": {
            "role": baselines["v0_1"]["role"],
            "model_id": baselines["v0_1"]["model_id"],
            "sha256": baselines["v0_1"]["sha256"],
            "accuracy": baselines["v0_1"]["accuracy"],
            "macro_f1": baselines["v0_1"]["macro_f1"],
            "class_collapse": baselines["v0_1"]["class_collapse"],
            "prediction_distribution": baselines["v0_1"]["prediction_distribution"],
        },
        "v0_2": {
            "role": baselines["v0_2"]["role"],
            "model_id": baselines["v0_2"]["model_id"],
            "sha256": baselines["v0_2"]["sha256"],
            "accuracy": baselines["v0_2"]["accuracy"],
            "macro_f1": baselines["v0_2"]["macro_f1"],
            "class_collapse": baselines["v0_2"]["class_collapse"],
            "prediction_distribution": baselines["v0_2"]["prediction_distribution"],
        },
        "v0_1_collapsed": True,
        "v0_2_collapsed": True,
        "new_model_selection_event": False,
        "inference_rerun_in_m_b12": False,
    }
    scientific = {
        "schema_version": SCHEMA,
        "source_m_b11": "datasets/mmwave/manifests/M-B11_artifact_lock/scientific_limitations.json",
        "locked_limitations_not_immediate_b_series_retuning_defects": True,
        "selected_seed42": limitations["selected_seed42"],
        "preprocessing": {
            "profile_id": prep["selected_profile_id"],
            "execution_contract_id": prep["execution_preprocessing_contract_id"],
            "zscore_train_only": True,
            "zscore_mean": prep["zscore"]["mean"],
            "zscore_std": prep["zscore"]["std"],
        },
        "training": {
            "strategy_id": train["selected_strategy_id"],
            "loss": train["loss"],
            "loss_weighting": train["loss_weighting"],
            "training_seed": train["training_seed"],
        },
        "runtime": {
            "input_contract": runtime["input_contract"],
            "output_contract": runtime["output_contract"],
            "strict_int8": runtime["strict_int8"],
        },
    }
    claim_boundary = {
        "schema_version": SCHEMA,
        "PRISTINE_LOCKED_TEST": False,
        "FIRST_LOCKED_TEST_EVALUATION": False,
        "MR60_device_validation_complete": False,
        "Raspberry_Pi_validation_complete": False,
        "deployment_ready": False,
        "production_ready": False,
        "clinical_apnea_validated": False,
        "locked_test_reopen_allowed": False,
        "recovery_reopen_allowed": False,
        "Phase_B_release_ready": False,
        "phase_b_offline_final_report_complete": True,
        "phase_b_offline_intermediate_release_ready_after_merge": True,
        "git_tag_created": False,
        "github_release_created": False,
        "m_c_started": False,
        "multisensor_integration_complete": False,
        "result_limitation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "artifact_status": ARTIFACT_STATUS,
        "status_label": STATUS_LABEL,
    }
    readiness = {
        "schema_version": SCHEMA,
        "manifest_version": "1.0",
        "track": "MMWAVE",
        "milestone": "PHASE_B_OFFLINE_REAL_DATA_CANDIDATE_CLOSURE",
        "phases": [
            "A0", "A1", "A2", "A3", "A4", "A5", "A6",
            "M-B0", "M-B1", "M-B2", "M-B3", "M-B4", "M-B5", "M-B6", "M-B7",
            "M-B8", "M-B9", "M-B10A", "M-B10B", "M-B10R0", "M-B10R1-A",
            "M-B10R1-B", "M-B11", "M-B12",
        ],
        "status_label": STATUS_LABEL,
        "phase_b_offline_intermediate_release_ready_after_merge": True,
        "Phase_B_release_ready": False,
        "release_ready_after_merge": True,
        "release_target_policy": "M_B12_MERGE_COMMIT_ON_CANONICAL_MAIN",
        "release_commit": "PENDING_POST_MERGE",
        "git_tag_created": False,
        "github_release_created": False,
        "proposed_release_tag": PROPOSED_TAG,
        "do_not_create_tag_or_github_release_in_this_pr": True,
        "artifact_status": ARTIFACT_STATUS,
        "result_limitation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "explicit_exclusions": {
            "pristine_locked_test": False,
            "first_locked_test_evaluation": False,
            "mr60_device_validated": False,
            "raspberry_pi_validated": False,
            "deployment_ready": False,
            "production_ready": False,
            "clinical_apnea_validated": False,
            "multisensor_integration_complete": False,
            "locked_test_reopen_allowed": False,
            "recovery_reopen_allowed": False,
        },
    }
    handoff = {
        "schema_version": SCHEMA,
        "m_c_started": False,
        "future_m_c_must_independently_investigate": [
            "physical MR60BHA2 signal-domain compatibility with this offline candidate",
            "device preprocessing correspondence to M-B10B_SELECTED_REAL_CANDIDATE_BPF_ZSCORE_V1",
            "observed team approximately-20-rpm behavior",
            "domain shift between the offline Zenodo dataset and the physical sensor",
            "runtime input identity on device",
            "Raspberry Pi / device execution behavior",
        ],
        "m_b8_is_mac_m2_latency_only": True,
        "m_b9_is_mock_runtime_not_physical_sensor": True,
        "old_team_ondevice_ai_is_not_validation_evidence": True,
        "apnea_labels_remain_safenest_proxies": True,
    }
    evidence = {
        "schema_version": SCHEMA,
        "artifacts": [
            {
                "artifact_role": "m_b11_checksums",
                "phase": "M-B11",
                "repo_relative_path": str(M_B11_DIR_REL / "checksums.sha256"),
                "sha256": m11_checksums_sha,
                "immutable": True,
            },
            {
                "artifact_role": "m_b11_immutable_registry",
                "phase": "M-B11",
                "repo_relative_path": str(M_B11_DIR_REL / "immutable_artifact_registry.json"),
                "sha256": m11_registry_sha,
                "immutable": True,
            },
            {
                "artifact_role": "selected_tflite",
                "phase": "M-B6",
                "repo_relative_path": SELECTED_TFLITE_REL,
                "sha256": live_model_sha,
                "bytes": live_model_bytes,
                "immutable": True,
            },
            {
                "artifact_role": "sensor_local_candidate_lock",
                "phase": "M-B11",
                "repo_relative_path": str(SENSOR_LOCK_REL),
                "sha256": sha256_file(root / SENSOR_LOCK_REL),
                "immutable": True,
            },
        ],
    }
    summary = {
        "schema_version": SCHEMA,
        "phase_id": PHASE_ID,
        "artifact_status": ARTIFACT_STATUS,
        "result_limitation": RESULT_LIMITATION,
        "result_not_pristine": True,
        "candidate_id": SELECTED_CANDIDATE_ID,
        "runtime_model_id": RUNTIME_MODEL_ID,
        "model_sha256": live_model_sha,
        "model_bytes": live_model_bytes,
        "macro_f1": EXPECTED_MACRO_F1,
        "v0_1_macro_f1": EXPECTED_V01_F1,
        "v0_2_macro_f1": EXPECTED_V02_F1,
        "unique_ids": EXPECTED_ELIGIBLE,
        "models": EXPECTED_MODELS,
        "pairs": EXPECTED_PAIRS,
        "status_label": STATUS_LABEL,
        "phase_b_offline_final_report_complete": True,
        "phase_b_offline_intermediate_release_ready_after_merge": True,
        "Phase_B_release_ready": False,
        "git_tag_created": False,
        "github_release_created": False,
        "m_c_started": False,
        "new_locked_test_access": 0,
        "new_recovery_access": 0,
        "new_model_inference": 0,
    }
    validation_placeholder = {
        "schema_version": SCHEMA,
        "phase_id": PHASE_ID,
        "status": "CLOSURE_ARTIFACTS_GENERATED",
        "standalone_validator": "scripts/validate_mmwave_m_b12.py",
        "validator_required": True,
        "does_not_access_locked_test": True,
        "does_not_invoke_tflite": True,
        "does_not_begin_m_c": True,
    }
    m11_imm = locks["immutable_artifact_registry.json"]
    m11_by_role = {str(item.get("artifact_role")): item for item in (m11_imm.get("artifacts") or [])}
    required_roles: list[dict[str, Any]] = []
    for role, phase, category in REQUIRED_M11_REGISTRY_ROLES:
        item = m11_by_role.get(role)
        if not item:
            _raise(f"M11_REGISTRY_MISSING_REQUIRED_ROLE:{role}")
        required_roles.append(
            {
                "artifact_role": role,
                "phase": phase,
                "category": category,
                "binding": "m_b11_registry",
                "repo_relative_path": item["repo_relative_path"],
                "expected_sha256": item["sha256"],
            }
        )
    for role, phase, category, rel in REQUIRED_M11_LOCK_FILE_ROLES:
        path = root / rel
        if not path.is_file():
            _raise(f"M11_LOCK_FILE_MISSING:{role}")
        required_roles.append(
            {
                "artifact_role": role,
                "phase": phase,
                "category": category,
                "binding": "m_b11_lock_file",
                "repo_relative_path": rel,
                "expected_sha256": sha256_file(path),
            }
        )
    role_registry = {
        "schema_version": SCHEMA,
        "source_m_b11_registry": str(M_B11_DIR_REL / "immutable_artifact_registry.json"),
        "required_role_count": len(required_roles),
        "required_m11_registry_role_count": len(REQUIRED_M11_REGISTRY_ROLES),
        "required_m11_lock_file_role_count": len(REQUIRED_M11_LOCK_FILE_ROLES),
        "roles": required_roles,
    }
    payloads = {
        "phase_b_closure_identity.json": closure_identity,
        "predecessor_gate.json": predecessor,
        "source_and_population_summary.json": population,
        "selected_path_lineage.json": path_lineage,
        "locked_candidate_summary.json": candidate,
        "final_evaluation_summary.json": evaluation,
        "scientific_limitations.json": scientific,
        "claim_boundary.json": claim_boundary,
        "release_readiness_manifest.json": readiness,
        "device_domain_handoff.json": handoff,
        "immutable_evidence_registry.json": evidence,
        "phase_b_required_role_registry.json": role_registry,
        "phase_b_closure_summary.json": summary,
        "validation_result.json": validation_placeholder,
    }
    out = root / CLOSURE_DIR_REL
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        atomic_write_json(out / name, payload)
    report = _render_report(
        identity=closure_identity,
        population=population,
        candidate=candidate,
        evaluation=evaluation,
        limitations=scientific,
        claims=claim_boundary,
        readiness=readiness,
        handoff=handoff,
        summary=summary,
        history=history,
        lineage=selected_path,
        prep=prep,
        train=train,
        subjects=subjects,
        metrics=metrics,
        quant=quant,
        baselines=baselines,
    )
    atomic_write_text(root / REPORT_REL, report)
    report_path = root / REPORT_REL
    atomic_write_json(
        out / "final_report_identity.json",
        {
            "schema_version": SCHEMA,
            "repo_relative_path": str(REPORT_REL),
            "sha256": sha256_file(report_path),
            "bytes": int(report_path.stat().st_size),
            "generated_from_machine_evidence": True,
            "report_schema": SCHEMA,
        },
    )
    write_checksums(out)
    print(f"M-B12 closure written: {CLOSURE_DIR_REL.as_posix()}")
    print(f"Report written: {REPORT_REL.as_posix()}")
    return summary


def _render_report(**kwargs: Any) -> str:
    identity = kwargs["identity"]
    population = kwargs["population"]
    candidate = kwargs["candidate"]
    evaluation = kwargs["evaluation"]
    limitations = kwargs["limitations"]
    claims = kwargs["claims"]
    readiness = kwargs["readiness"]
    handoff = kwargs["handoff"]
    summary = kwargs["summary"]
    history = kwargs["history"]
    lineage = kwargs["lineage"]
    prep = kwargs["prep"]
    train = kwargs["train"]
    subjects = kwargs["subjects"]
    metrics = kwargs["metrics"]
    quant = kwargs["quant"]
    baselines = kwargs["baselines"]
    normal = metrics["per_class"]["NORMAL"]
    rapid = metrics["per_class"]["RAPID_OR_ABNORMAL"]
    apnea = metrics["per_class"]["APNEA"]
    seed42 = limitations["selected_seed42"]
    b4 = lineage["M-B4"]["validation_macro_f1"]
    return f"""# M-B12 mmWave Phase-B Offline Final Report

Generated from stored M-B11 lock evidence and immutable predecessor artifacts.
This report does not create a new model, does not re-evaluate LOCKED_TEST, and
does not begin M-C.

## Prominent closure statements

M-B12 DOES NOT CREATE A NEW MODEL.

THE LOCKED CANDIDATE REMAINS THE M-B11 SEED42 STRICT-INT8 OFFLINE CANDIDATE.

THE FINAL OFFLINE EVALUATION REMAINS A NON-PRISTINE HOLDOUT REUSE EXCEPTION.

- Artifact status: `{identity["artifact_status"]}`
- Result limitation: `{identity["result_limitation"]}`
- `result_not_pristine`: true
- Intermediate-release status: `{summary["status_label"]}`
- Unqualified Phase-B product/deployment release: false
- Git tag created: false
- GitHub Release created: false
- M-C started: false

## What this closure is not

This closure is not deployment ready, MR60 validated, Raspberry Pi validated,
production ready, or clinical apnea validated. It is not a pristine LOCKED_TEST
result. It is not physical sensor integration. It does not authorize reopening
LOCKED_TEST or recovery. It does not create a git tag or GitHub Release.

APNEA labels remain SafeNest proxies derived from voluntary breath-hold
windows. They are not clinical apnea.

## A0–A6 frozen source, representation, and split

- Raw archive: `{population["raw_archive_repo_relative_path"]}`
- Raw SHA-256: `{population["raw_archive_sha256"]}`
- DOI: {population["doi"]} version {population["version"]}
- Population: {population["measured_subjects"]} participants / {population["measured_recordings"]} recordings
- Canonical dataset: `{population["canonical_npy_repo_relative_path"]}`
- Canonical SHA-256: `{population["canonical_npy_sha256"]}`
- Shape/dtype: {population["shape"]} / {population["dtype"]}
- A4 labels: SafeNest APNEA proxies; not clinical apnea
- A5 split: `{population["split_artifact_repo_relative_path"]}`
- A5 SHA-256: `{population["split_sha256"]}`
- A5 split seed: {population["split_seed"]}
- Subjects TRAIN/VALIDATION/LOCKED_TEST: {population["subject_counts"]["TRAIN"]}/{population["subject_counts"]["VALIDATION"]}/{population["subject_counts"]["LOCKED_TEST"]}
- A6 manifest SHA-256: `{population["a6_manifest_sha256"]}`
- Windows structural TRAIN/VALIDATION/LOCKED_TEST: {population["structural"]["TRAIN"]}/{population["structural"]["VALIDATION"]}/{population["structural"]["LOCKED_TEST"]}
- Eligible TRAIN/VALIDATION/LOCKED_TEST: {population["pure_supervised_eligible"]["TRAIN"]}/{population["pure_supervised_eligible"]["VALIDATION"]}/{population["pure_supervised_eligible"]["LOCKED_TEST"]}
- Class totals NORMAL/RAPID_OR_ABNORMAL/APNEA/AMBIGUOUS: {population["class_totals"]["NORMAL"]}/{population["class_totals"]["RAPID_OR_ABNORMAL"]}/{population["class_totals"]["APNEA"]}/{population["class_totals"]["AMBIGUOUS"]}
- LOCKED_TEST excluded ambiguous/non-eligible: {population["locked_test_excluded_ambiguous_or_non_eligible"]}

## B-series selected path

- M-B0: evaluation / leakage / LOCKED_TEST protocol. Cross-split subject/recording/window overlap = 0.
- M-B1 selected preprocessing: `{prep["selected_profile_id"]}` / `{prep["selected_profile_name"]}`
- M-B2 selected training strategy: `{train["selected_strategy_id"]}` (unweighted CE)
- M-B3 selected architecture: `{candidate["architecture_id"]}`
- M-B4 cross-seed VALIDATION Macro F1: seed42={b4["seed42"]}, seed43={b4["seed43"]}, seed44={b4["seed44"]}. Initialization sensitivity is locked, not hidden.
- M-B5 selected calibration: `{candidate["calibration_profile"]}`
- M-B6 frozen strict INT8 SHA-256: `{candidate["sha256"]}`
- M-B7 perturbation robustness: seed42 retained; seed44 moderate-profile collapse recorded
- M-B8 Mac/M2 latency and footprint only. Not Raspberry Pi latency. seed42 median={lineage["M-B8"]["seed42_median_ms"]} ms, p99={lineage["M-B8"]["seed42_p99_ms"]} ms
- M-B9 mock runtime/E2E fail-closed path. Not physical sensor integration.
- M-B10A preregistered `{candidate["candidate_id"]}` before LOCKED_TEST
- M-B10B original accessor=1, payload release=1, inference=0, consumed=true, root cause=`PRETEST_CONTRACT_COUNT_SEMANTICS_CONFLATION`
- M-B10R0 policy=`LIMITED_LOCKED_TEST_REUSE_EXCEPTION_RECOMMENDED`
- M-B10R1-A froze the recovery harness; new access=0
- M-B10R1-B recovery accessor=1, recovery payload release=1, TFLite invokes=225, second recovery=NO, rerun=NO
- Historical total payload releases = {history["historical_total_payload_releases"]}
- M-B11 locked the candidate as `{identity["artifact_status"]}`

## Locked candidate

- Candidate ID: `{candidate["candidate_id"]}`
- Runtime model ID: `{candidate["runtime_model_id"]}`
- Artifact: `{candidate["repo_relative_path"]}`
- SHA-256: `{candidate["sha256"]}`
- Bytes: {candidate["bytes"]}
- Seed: 42
- Preprocessing contract: `{candidate["execution_preprocessing_contract_id"]}`
- BPF: Butterworth {prep["bpf"]["lowcut_hz"]}-{prep["bpf"]["highcut_hz"]} Hz, order {prep["bpf"]["order"]}, zero-phase filtfilt, fs={prep["bpf"]["fs_hz"]} Hz
- Z-score: TRAIN-only mean={prep["zscore"]["mean"]}, std={prep["zscore"]["std"]}
- Training: `{candidate["training_strategy_id"]}`; {train["loss"]} unweighted; Adam lr={train["learning_rate"]}; batch {train["batch_size"]}; max epochs {train["max_epochs"]}; patience {train["early_stopping_patience"]}
- Input: shape {candidate["input_tensor"]["shape"]} dtype {candidate["input_tensor"]["dtype"]} scale={candidate["input_tensor"]["scale"]} zp={candidate["input_tensor"]["zero_point"]}
- Output: shape {candidate["output_tensor"]["shape"]} dtype {candidate["output_tensor"]["dtype"]} scale={candidate["output_tensor"]["scale"]} zp={candidate["output_tensor"]["zero_point"]}
- Strict INT8: true; Flex/Select TF Ops: false
- Class map: 0→NORMAL, 1→RAPID_OR_ABNORMAL, 2→APNEA (APNEA remains a proxy)

## Final offline evaluation (non-pristine)

Reused stored M-B10R1-B / M-B11 evidence. No M-B12 inference.

- Result designation: `{evaluation["result_designation"]}`
- Unique eligible IDs / models / pairs: {evaluation["unique_eligible_window_ids"]} / {evaluation["models"]} / {evaluation["actual_pairs"]}
- Duplicates / missing / unexpected: {evaluation["duplicates"]} / {evaluation["missing"]} / {evaluation["unexpected"]}
- Label / subject / recording mismatches: {evaluation["cross_model_label_mismatches"]} / {evaluation["cross_model_subject_mismatches"]} / {evaluation["cross_model_recording_mismatches"]}
- Eligible evaluated / valid / invalid: {evaluation["eligible_evaluated"]} / {evaluation["valid"]} / {evaluation["invalid"]}
- Accuracy: {evaluation["accuracy"]}
- Macro F1: {evaluation["macro_f1"]}
- Macro precision: {evaluation["macro_precision"]}
- Macro recall: {evaluation["macro_recall"]}
- NORMAL: support={normal["support"]} precision={normal["precision"]} recall={normal["recall"]} F1={normal["f1_score"]} FPR={normal["fpr"]}
- RAPID_OR_ABNORMAL: support={rapid["support"]} precision={rapid["precision"]} recall={rapid["recall"]} F1={rapid["f1_score"]} FPR={rapid["fpr"]}
- APNEA proxy: support={apnea["support"]} precision={apnea["precision"]} recall={apnea["recall"]} F1={apnea["f1_score"]} misses={metrics["apnea_proxy"]["misses"]} FPR={metrics["apnea_proxy"]["fpr"]}
- Confusion: {evaluation["confusion_matrix"]}
- Prediction distribution: {evaluation["prediction_distribution"]}
- Class collapse: {evaluation["class_collapse"]["collapsed"]}
- Subjects: {subjects["subject_count"]}
- Median subject Macro F1: {subjects["median_subject_macro_f1"]}
- Worst subject Macro F1: {subjects["worst_subject_macro_f1"]}
- Worst subject: `{subjects["worst_subject_id"]}`
- Saturation ratio: {quant["input_saturation_ratio"]} (pre-clamp out-of-range {quant["pre_clamp_out_of_range_count"]} / {quant["total_quantized_elements"]})

These limitations are locked scientific facts for future M-C/M-D. They are not
M-B12 blockers and they are not defects requiring immediate B-series retuning.

Locked limitation facts: NORMAL recall={seed42["weak_normal_recall"]}; RAPID recall={seed42["moderate_rapid_recall"]}; APNEA proxy recall={seed42["strong_apnea_proxy_recall"]}; APNEA FPR={seed42["high_apnea_false_positive_rate"]}; worst-subject Macro F1={seed42["weak_worst_subject_generalization"]}; M-B4 seed42 VAL={seed42["seed42_val_macro_f1"]} vs seed44 VAL={seed42["seed44_val_macro_f1"]}.

## Baseline comparison

This is not a new model-selection event.

- seed42 Macro F1: {evaluation["macro_f1"]} (no required-class collapse)
- v0.1 `{baselines["v0_1"]["role"]}` Macro F1: {baselines["v0_1"]["macro_f1"]} (class collapse; all 75 predicted NORMAL)
- v0.2 `{baselines["v0_2"]["role"]}` Macro F1: {baselines["v0_2"]["macro_f1"]} (RAPID_OR_ABNORMAL zero-prediction collapse)

## Intermediate-release readiness

- Status label: `{readiness["status_label"]}`
- Ready after merge onto canonical `main`: {readiness["release_ready_after_merge"]}
- Unqualified `Phase_B_release_ready`: {readiness["Phase_B_release_ready"]}
- Proposed future tag (not created): `{readiness["proposed_release_tag"]}`
- Git tag created: {readiness["git_tag_created"]}
- GitHub Release created: {readiness["github_release_created"]}
- Explicit exclusions remain false: {", ".join(sorted(name for name, value in readiness["explicit_exclusions"].items() if value is False))}

Do not create a GitHub Release or tag in M-B12. Any future tag must target the
exact M-B12 merge commit on canonical `main` after independent review.

## Device-domain handoff for future M-C

M-B12 does not begin M-C. Future M-C must independently investigate:

- {handoff["future_m_c_must_independently_investigate"][0]}
- {handoff["future_m_c_must_independently_investigate"][1]}
- {handoff["future_m_c_must_independently_investigate"][2]}
- {handoff["future_m_c_must_independently_investigate"][3]}
- {handoff["future_m_c_must_independently_investigate"][4]}
- {handoff["future_m_c_must_independently_investigate"][5]}

## Claim boundary

- PRISTINE_LOCKED_TEST: {claims["PRISTINE_LOCKED_TEST"]}
- MR60 validated: {claims["MR60_device_validation_complete"]}
- Raspberry Pi validated: {claims["Raspberry_Pi_validation_complete"]}
- Deployment ready: {claims["deployment_ready"]}
- Production ready: {claims["production_ready"]}
- Clinical apnea validated: {claims["clinical_apnea_validated"]}
- LOCKED_TEST reopen allowed: {claims["locked_test_reopen_allowed"]}
- Recovery reopen allowed: {claims["recovery_reopen_allowed"]}
- M-C started: {claims["m_c_started"]}
- Phase-B offline final report complete: {claims["phase_b_offline_final_report_complete"]}
- Phase-B offline intermediate release ready after merge: {claims["phase_b_offline_intermediate_release_ready_after_merge"]}

## Machine-Verified Final Facts

| Fact | Value |
| --- | --- |
| candidate_status | `{identity["artifact_status"]}` |
| selected_model_sha | `{candidate["sha256"]}` |
| result_designation | `{evaluation["result_designation"]}` |
| result_not_pristine | true |
| final_accuracy | {evaluation["accuracy"]} |
| final_macro_f1 | {evaluation["macro_f1"]} |
| normal_recall | {normal["recall"]} |
| rapid_recall | {rapid["recall"]} |
| apnea_recall | {apnea["recall"]} |
| apnea_fpr | {apnea["fpr"]} |
| v0_1_macro_f1 | {evaluation["v0_1_macro_f1"]} |
| v0_2_macro_f1 | {evaluation["v0_2_macro_f1"]} |
| original_release | {evaluation["original_m_b10b_payload_releases"]} |
| recovery_release | {evaluation["m_b10r1b_recovery_payload_releases"]} |
| historical_total_release | {evaluation["historical_total_payload_releases"]} |
| mr60_validated | false |
| raspberry_pi_validated | false |
| deployment_ready | false |
| clinical_apnea_validated | false |
| intermediate_release_ready | true |
| tag_created | false |
| github_release_created | false |
| m_c_started | false |

New LOCKED_TEST access = 0
New recovery access = 0
New inference = 0

{MACHINE_FACTS_BEGIN}
candidate_status={identity["artifact_status"]}
selected_model_sha={candidate["sha256"]}
result_designation={evaluation["result_designation"]}
result_not_pristine=true
final_accuracy={evaluation["accuracy"]}
final_macro_f1={evaluation["macro_f1"]}
normal_recall={normal["recall"]}
rapid_recall={rapid["recall"]}
apnea_recall={apnea["recall"]}
apnea_fpr={apnea["fpr"]}
v0_1_macro_f1={evaluation["v0_1_macro_f1"]}
v0_2_macro_f1={evaluation["v0_2_macro_f1"]}
original_release={evaluation["original_m_b10b_payload_releases"]}
recovery_release={evaluation["m_b10r1b_recovery_payload_releases"]}
historical_total_release={evaluation["historical_total_payload_releases"]}
mr60_validated=false
raspberry_pi_validated=false
deployment_ready=false
clinical_apnea_validated=false
intermediate_release_ready=true
tag_created=false
github_release_created=false
m_c_started=false
{MACHINE_FACTS_END}
"""


def main() -> int:
    try:
        generate_m_b12_closure()
    except MB12ClosureError as exc:
        print(f"M-B12 GENERATION FAIL: {exc}", file=sys.stderr)
        return 1
    print("M-B12 GENERATION PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
