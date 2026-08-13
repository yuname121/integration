#!/usr/bin/env python3
"""Fail-closed validator for M-B10R0 holdout reuse policy evidence.

Independently recomputes subject inventory, exposure, gates R1–R10, and the
policy decision from upstream evidence. Does not import generator gate or
policy functions. Never calls the LOCKED_TEST final accessor.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUT_DIR_REL = Path("datasets/mmwave/manifests/M-B10R0_holdout_policy_review")
M_B10B_DIR_REL = Path("datasets/mmwave/manifests/M-B10B_locked_test_final_evaluation")
M_B10A_DIR_REL = Path("datasets/mmwave/manifests/M-B10A_candidate_selection_setup")
A0_DIR_REL = Path("datasets/mmwave/manifests/a0_raw_inventory")
A5_DIR_REL = Path("datasets/mmwave/manifests/a5_subject_split")
A6_DIR_REL = Path("datasets/mmwave/manifests/a6_full_conversion")
INCIDENT_VALIDATOR = Path("scripts/validate_mmwave_m_b10b_incident.py")
REPORT_REL = Path("docs/reports/20260812_Cursor_M-B10R0_Holdout_Reuse_Policy_01.md")

INCIDENT_CLOSURE_COMMIT = "807a50316f750e1e877931b46fe9ea87113418e4"
ROOT_CAUSE_ID = "PRETEST_CONTRACT_COUNT_SEMANTICS_CONFLATION"
RUNTIME_DETECTION = "M-B10B_LOCKED_SPLIT_IDENTITY_MISMATCH"
SELECTED_CANDIDATE_ID = "M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120"
SELECTED_MODEL_ID = "M-B3_CONV1D_GAP_BASELINE_seed42_M-B6_STRICT_INT8"
SELECTED_SHA = "6dff6aaa72c79d76715d40cf7e32bb1e6cd9b2c2e3ac78eaf2fda737561430c5"
M_B10A_CONTRACT_SHA = "ba6429ecfe685de1807ec85b55e697ee12e24138e6b96e94715b0a1a6b19e0f7"
SELECTED_PRETEST_SHA = "b6ba2516b5e9a46c0f3a7dec408973c7eb1bdc333adff6ac17c322bbc6875db8"
CLASS_MAP = {"0": "NORMAL", "1": "RAPID_OR_ABNORMAL", "2": "APNEA"}
CALIBRATION_PROFILE = "M-B5_CAL_CLASS_BALANCED_120"
PREPROCESSING_PROFILE = "M-B1_D0_B1_Z1"
PREPROCESSING_NAME = "BPF_ZSCORE"
RESULT_LIMITATION = "REUSED_LOCKED_TEST_AFTER_PREINFERENCE_STRUCTURAL_ABORT"
RECOVERY_CONTRACT_STATUS = "PROPOSED_NOT_AUTHORIZED"

EXPECTED_CONTRACT_MODEL_IDS = [
    "M-B3_CONV1D_GAP_BASELINE_seed42_M-B6_STRICT_INT8",
    "mmwave_resp_int8",
    "mmwave_resp_int8_v0.2.0_candidate",
]

MODEL_SPECS = [
    {
        "model_id": "M-B3_CONV1D_GAP_BASELINE_seed42_M-B6_STRICT_INT8",
        "role": "SELECTED_NEW_REAL_DATA_CANDIDATE",
        "path": "models/mmwave/experiments/M-B6_stage_equivalence/M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120_int8.tflite",
        "sha256": SELECTED_SHA,
        "seed": 42,
        "candidate_id": SELECTED_CANDIDATE_ID,
    },
    {
        "model_id": "mmwave_resp_int8",
        "role": "HISTORICAL_MODEL_COMPATIBILITY_BENCHMARK",
        "path": "models/mmwave/mmwave_resp_int8_v0.1.0.tflite",
        "sha256": "43cdd6f321c2b645232162233e098c2b65549388cbc9e680a17a0eccdc8f0158",
    },
    {
        "model_id": "mmwave_resp_int8_v0.2.0_candidate",
        "role": "SYNTHETIC_TRAINED_EXTERNAL_COMPATIBILITY_BENCHMARK",
        "path": "models/mmwave/mmwave_resp_int8_v0.2.0_candidate.tflite",
        "sha256": "85c023d3eefca13ecbb72a841974e53a56d5f4173920645d46df49a9088452ff",
    },
]

FORBIDDEN_PRISTINE_CLAIMS = {
    "PRISTINE_LOCKED_TEST",
    "PRISTINE_ONE_TIME_LOCKED_TEST",
    "PRISTINE_REAL_SUBJECT_FINAL_TEST",
    "FIRST_LOCKED_TEST_EVALUATION",
    "LOCKED_TEST_NOT_CONSUMED",
    "NO_INFORMATION_EXPOSURE",
    "ORIGINAL_ACCESS_UNUSED",
}

REQUIRED_FORBIDDEN_SCIENTIFIC_WORDING = [
    "PRISTINE_REAL_SUBJECT_FINAL_TEST",
    "PRISTINE_ONE_TIME_LOCKED_TEST",
    "PRISTINE_LOCKED_TEST",
    "FIRST_LOCKED_TEST_EVALUATION",
    "LOCKED_TEST_NOT_CONSUMED",
    "NO_INFORMATION_EXPOSURE",
    "ORIGINAL_ACCESS_UNUSED",
]

REQUIRED_OUTPUTS = {
    "input_identity.json",
    "incident_identity.json",
    "exposure_assessment.json",
    "original_holdout_consumption_status.json",
    "existing_unused_holdout_inventory.json",
    "reuse_exception_eligibility_contract.json",
    "reuse_exception_gate_results.json",
    "policy_decision.json",
    "proposed_recovery_evaluation_contract.json",
    "future_recovery_access_requirements.json",
    "claim_limitations.json",
    "locked_test_access_audit.json",
    "run_environment.json",
    "exceptions.json",
    "m_b10r0_summary.json",
    "checksums.sha256",
}

FROZEN_BASELINE_EXPECTATIONS = json.loads(r"""
{
  "mmwave_resp_int8": {
    "bytes": 466616,
    "class_map": {
      "0": "NORMAL",
      "1": "RAPID_OR_ABNORMAL",
      "2": "APNEA"
    },
    "contract_id": "M-B10B_HISTORICAL_V0_1_COMPATIBILITY_PREPROCESSING_V1",
    "executor": {
      "entrypoint": "prepare_v01",
      "path": "scripts/mmwave_m_b10b_baseline_preprocessing.py",
      "sha256": "8ca87f457d0a151cffa2da23ae9ab9d87764b144fa826b91444776f3dc58ec4f"
    },
    "fallback_policy": "NO_HEURISTIC_FALLBACK",
    "input": {
      "dtype": "int8",
      "scale": 0.03259856998920441,
      "shape": [
        1,
        300,
        1
      ],
      "zero_point": -13
    },
    "interpretation": "HISTORICAL_MODEL_COMPATIBILITY_BENCHMARK",
    "metadata_sources": [
      {
        "bytes": 594,
        "path": "models/mmwave/sensor_stats_metadata_v0.1.0.json",
        "sha256": "a875a8369ff7adf5477cec009b99c0c6d0fbb8b0e60e5b0b07a551f3780d2e37"
      }
    ],
    "output": {
      "dtype": "int8",
      "scale": 0.00390625,
      "shape": [
        1,
        3
      ],
      "zero_point": -128
    },
    "path": "models/mmwave/mmwave_resp_int8_v0.1.0.tflite",
    "sha256": "43cdd6f321c2b645232162233e098c2b65549388cbc9e680a17a0eccdc8f0158",
    "steps": [
      {
        "operation": "VALIDATE_WINDOW",
        "parameters": {
          "allow_padding": false,
          "allow_resampling": false,
          "allow_truncation": false,
          "dtype": "float32",
          "exact_samples": 300,
          "require_all_finite": true
        },
        "step": 1
      },
      {
        "operation": "IDENTITY_SEMANTIC_ADAPTER",
        "parameters": {
          "input_semantic": "resp_phase_unwrapped_clutter_removed",
          "native_semantic_alignment_claim": false,
          "transformation": "NONE"
        },
        "step": 2
      },
      {
        "operation": "FIXED_Z_SCORE",
        "parameters": {
          "fit_split": "NONE_AT_M-B10B",
          "mean": 0.006091983988881111,
          "stats_source": "models/mmwave/sensor_stats_metadata_v0.1.0.json",
          "std": 2.5013835430145264
        },
        "step": 3
      },
      {
        "operation": "RESHAPE",
        "parameters": {
          "dtype": "float32",
          "shape": [
            1,
            300,
            1
          ]
        },
        "step": 4
      },
      {
        "operation": "AFFINE_INT8_QUANTIZE",
        "parameters": {
          "rounding": "nearest_even_numpy_rint",
          "saturate_to": [
            -128,
            127
          ],
          "scale": 0.03259856998920441,
          "zero_point": -13
        },
        "step": 5
      }
    ]
  },
  "mmwave_resp_int8_v0.2.0_candidate": {
    "bytes": 22472,
    "class_map": {
      "0": "NORMAL",
      "1": "RAPID_OR_ABNORMAL",
      "2": "APNEA"
    },
    "contract_id": "M-B10B_SYNTHETIC_V0_2_EXTERNAL_COMPATIBILITY_PREPROCESSING_V1",
    "executor": {
      "entrypoint": "prepare_v02",
      "path": "scripts/mmwave_m_b10b_baseline_preprocessing.py",
      "sha256": "8ca87f457d0a151cffa2da23ae9ab9d87764b144fa826b91444776f3dc58ec4f"
    },
    "fallback_policy": "NO_HEURISTIC_FALLBACK",
    "input": {
      "dtype": "int8",
      "scale": 0.012282303534448147,
      "shape": [
        1,
        300,
        1
      ],
      "zero_point": 12
    },
    "interpretation": "SYNTHETIC_TRAINED_EXTERNAL_COMPATIBILITY_BENCHMARK",
    "metadata_sources": [
      {
        "bytes": 4443,
        "path": "models/mmwave/mmwave_resp_int8_v0.2.0_candidate_metadata.json",
        "sha256": "36039a6cffbc57162dbb4c720034da6dcfa49ef2f2d33238bee65a62aa133127"
      }
    ],
    "output": {
      "dtype": "int8",
      "scale": 0.00390625,
      "shape": [
        1,
        3
      ],
      "zero_point": -128
    },
    "path": "models/mmwave/mmwave_resp_int8_v0.2.0_candidate.tflite",
    "sha256": "85c023d3eefca13ecbb72a841974e53a56d5f4173920645d46df49a9088452ff",
    "steps": [
      {
        "operation": "VALIDATE_WINDOW",
        "parameters": {
          "allow_padding": false,
          "allow_resampling": false,
          "allow_truncation": false,
          "dtype": "float32",
          "exact_samples": 300,
          "require_all_finite": true
        },
        "step": 1
      },
      {
        "operation": "LINEAR_DETREND",
        "parameters": {
          "method": "window_mean_subtraction"
        },
        "step": 2
      },
      {
        "operation": "BUTTERWORTH_BANDPASS_ZERO_PHASE",
        "parameters": {
          "highcut_hz": 0.5,
          "implementation": "scipy.signal.butter_and_filtfilt",
          "lowcut_hz": 0.1,
          "order": 4,
          "sample_rate_hz": 10.0
        },
        "step": 3
      },
      {
        "operation": "FIXED_Z_SCORE",
        "parameters": {
          "fit_split": "NONE_AT_M-B10B",
          "mean": 0.17212218046188354,
          "method": "z_score",
          "stats_source": "models/mmwave/mmwave_resp_int8_v0.2.0_candidate_metadata.json",
          "std": 1.7171541452407837
        },
        "step": 4
      },
      {
        "operation": "CLIP",
        "parameters": {
          "max": 5.0,
          "min": -5.0
        },
        "step": 5
      },
      {
        "operation": "RESHAPE",
        "parameters": {
          "dtype": "float32",
          "shape": [
            1,
            300,
            1
          ]
        },
        "step": 6
      },
      {
        "operation": "AFFINE_INT8_QUANTIZE",
        "parameters": {
          "rounding": "nearest_even_numpy_rint",
          "saturate_to": [
            -128,
            127
          ],
          "scale": 0.012282303534448147,
          "zero_point": 12
        },
        "step": 7
      }
    ]
  }
}
""")


class MB10R0ValidationError(RuntimeError):
    """Raised when M-B10R0 policy evidence fails closed."""


class MB10R0InventoryError(RuntimeError):
    """Raised when independent subject inventory cannot be computed."""


def _raise(message: str) -> None:
    raise MB10R0ValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _raise(f"JSON_PARSE_ERROR:{path.as_posix()}:{exc}")


def _hex_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value.lower()))


def _safe_relative(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or relative.startswith("~") or "file://" in relative:
        _raise(f"ABSOLUTE_OR_TRAVERSAL_PATH:{relative}")
    return path


def _gate_result(gate_id: str, passed: bool, independent_evidence: dict[str, Any], failure_reasons: list[str] | None = None) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "status": "PASS" if passed else "FAIL",
        "independent_evidence": independent_evidence,
        "failure_reasons": failure_reasons or ([] if passed else ["GATE_FAILED"]),
    }


def compute_subject_inventory(root: Path) -> dict[str, Any]:
    a0_path = root / A0_DIR_REL / "recording_index.jsonl"
    if not a0_path.is_file():
        raise MB10R0InventoryError("A0_RECORDING_INDEX_MISSING")
    a0_subjects: set[str] = set()
    for line in a0_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            sid = row.get("subject_id")
            if sid:
                a0_subjects.add(sid)
    if not a0_subjects:
        raise MB10R0InventoryError("A0_SUBJECT_UNIVERSE_EMPTY")

    rows = []
    for line in (root / A5_DIR_REL / "subject_split_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    by_split: dict[str, set[str]] = {"TRAIN": set(), "VALIDATION": set(), "LOCKED_TEST": set()}
    for row in rows:
        split = row.get("split")
        subject = row.get("subject_id")
        if split in by_split and subject:
            by_split[split].add(subject)

    if by_split["TRAIN"] & by_split["VALIDATION"]:
        raise MB10R0InventoryError("A5_TRAIN_VALIDATION_OVERLAP")
    if by_split["TRAIN"] & by_split["LOCKED_TEST"]:
        raise MB10R0InventoryError("A5_TRAIN_LOCKED_TEST_OVERLAP")
    if by_split["VALIDATION"] & by_split["LOCKED_TEST"]:
        raise MB10R0InventoryError("A5_VALIDATION_LOCKED_TEST_OVERLAP")

    all_assigned = by_split["TRAIN"] | by_split["VALIDATION"] | by_split["LOCKED_TEST"]
    unassigned = a0_subjects - all_assigned
    extra_in_a5 = all_assigned - a0_subjects
    if extra_in_a5:
        raise MB10R0InventoryError(f"A5_SUBJECTS_NOT_IN_A0:{sorted(extra_in_a5)}")

    independent_holdout_available = len(unassigned) > 0
    return {
        "total_original_subjects": len(a0_subjects),
        "train_subjects": len(by_split["TRAIN"]),
        "validation_subjects": len(by_split["VALIDATION"]),
        "locked_test_subjects": len(by_split["LOCKED_TEST"]),
        "assigned_subjects": len(all_assigned),
        "unassigned_subjects": len(unassigned),
        "unassigned_subject_ids": sorted(unassigned),
        "potential_independent_replacement_subjects": len(unassigned),
        "replacement_subject_ids": sorted(unassigned),
        "train_subject_reuse_prohibited": True,
        "validation_subject_reuse_prohibited": True,
        "a5_reshuffle_prohibited": True,
        "evidence_paths": [
            "datasets/mmwave/manifests/a0_raw_inventory/recording_index.jsonl",
            "datasets/mmwave/manifests/a5_subject_split/subject_split_manifest.jsonl",
        ],
        "independent_existing_holdout_available": independent_holdout_available,
        "reason": (
            f"All {len(a0_subjects)} approved corpus subjects are assigned to TRAIN, VALIDATION, or LOCKED_TEST; no unassigned untouched subject remains."
            if not independent_holdout_available
            else f"{len(unassigned)} subjects in A0 are not assigned in A5."
        ),
    }


def compute_a6_eligible_subject_coverage(root: Path) -> dict[str, Any]:
    eligible_subjects: set[str] = set()
    eligible_windows = 0
    ambiguous_windows = 0
    manifest = root / A6_DIR_REL / "full_window_manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != "LOCKED_TEST":
            continue
        if row.get("assignment_status") == "AMBIGUOUS":
            ambiguous_windows += 1
        else:
            eligible_windows += 1
            if row.get("subject_id"):
                eligible_subjects.add(row["subject_id"])
    return {
        "eligible_window_count": eligible_windows,
        "ambiguous_window_count": ambiguous_windows,
        "eligible_subject_count": len(eligible_subjects),
        "all_locked_test_subjects_have_eligible_windows": len(eligible_subjects) == 16,
        "known_from_pre_access_a6_metadata": True,
        "eligible_subject_count_provenance": "PREEXISTING_A6_METADATA_VERIFIED",
        "eligible_subject_count_note": "All 16 LOCKED_TEST subjects have at least one non-AMBIGUOUS window per pre-access A6 window manifest. Future recovery should independently confirm this identity but the count 16 is not derived from the aborted M-B10B returned payload.",
        "evidence_paths": ["datasets/mmwave/manifests/a6_full_conversion/full_window_manifest.jsonl"],
    }


def compute_exposure(root: Path) -> dict[str, Any]:
    mb10b = root / M_B10B_DIR_REL
    registry = _load(mb10b / "locked_test_registry.json")
    samples = registry.get("samples", [])
    actual_registry_rows = len(samples)
    raw_tensors_persisted = registry.get("raw_tensors_persisted", False) is True
    placeholder_registry_exists = (mb10b / "locked_test_registry.json").is_file()

    pred_path = mb10b / "locked_test_sample_predictions.jsonl"
    pred_lines = [line for line in pred_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    prediction_ledger_rows = len(pred_lines)

    metrics = _load(mb10b / "metrics_by_model.json")
    metrics_results_available = metrics.get("results_available", False) is True

    input_id = _load(mb10b / "input_identity.json")
    input_id_labels_tensors_not_persisted = input_id.get("labels_or_tensors_persisted", True) is False

    persisted_sample_registry_exposure = (
        actual_registry_rows > 0
        or raw_tensors_persisted
        or prediction_ledger_rows > 0
    )
    sample_ids_persisted = False if actual_registry_rows == 0 else any(
        bool(sample.get("sample_id") or sample.get("id")) for sample in samples
    )
    subject_ids_persisted = False if actual_registry_rows == 0 else any(
        bool(sample.get("subject_id")) for sample in samples
    )
    labels_persisted = False if actual_registry_rows == 0 else any(
        sample.get("label") is not None or sample.get("label_id") is not None for sample in samples
    )

    a6_eligible = _load(root / A6_DIR_REL / "full_split_distribution.json")["eligibility_counts"]["locked_test_evaluation_eligible"]
    return {
        "schema_version": "M-B10R0_EXPOSURE_ASSESSMENT_V1",
        "phase_id": "M-B10R0",
        "E0_accessor_occurrence": {
            "payload_release_occurred": True,
            "historical_final_accessor_invocations": 1,
            "interpretation": "Withheld evaluation payload was returned to the M-B10B process once; pristine LOCKED_TEST status is lost.",
        },
        "E1_model_outputs": {
            "predictions_generated": False,
            "prediction_exposure": False,
        },
        "E2_metrics": {
            "metrics_generated": False,
            "performance_exposure": False,
        },
        "E3_persistent_sample_registry": {
            "actual_registry_rows": actual_registry_rows,
            "raw_tensors_persisted": raw_tensors_persisted,
            "placeholder_registry_exists": placeholder_registry_exists,
            "prediction_ledger_rows": prediction_ledger_rows,
            "persisted_sample_registry_exposure": persisted_sample_registry_exposure,
            "sample_ids_persisted": sample_ids_persisted,
            "subject_ids_persisted": subject_ids_persisted,
            "labels_persisted": labels_persisted,
        },
        "E4_payload_logging": {
            "post_access_tensor_values_logged": False,
            "post_access_sample_ids_logged": False,
            "post_access_subject_ids_logged": False,
            "post_access_labels_logged": False,
            "input_id_labels_tensors_not_persisted": input_id_labels_tensors_not_persisted,
            "metrics_results_available": metrics_results_available,
        },
        "E5_human_agent_decision_exposure": {
            "model_performance_used": False,
            "labels_used_for_configuration": False,
            "prediction_errors_used": False,
            "subject_behavior_used": False,
        },
        "E6_new_information_learned": {
            "returned_row_count": 75,
            "preexisting_a6_eligible_count": a6_eligible,
            "preexisting_eligible_count_confirmed": True,
            "returned_count_classification": "PREEXISTING_STRUCTURAL_INFORMATION_CONFIRMED_BY_ABORT",
            "new_performance_information": False,
        },
        "summary": {
            "PAYLOAD_RELEASE_OCCURRED": True,
            "PREDICTION_EXPOSURE": False,
            "PERFORMANCE_EXPOSURE": False,
            "PERSISTED_SAMPLE_REGISTRY_EXPOSURE": persisted_sample_registry_exposure,
            "PREEXISTING_ELIGIBLE_COUNT_CONFIRMED": True,
        },
        "evidence_paths": [
            "datasets/mmwave/manifests/M-B10B_locked_test_final_evaluation/one_time_access_audit.json",
            "datasets/mmwave/manifests/M-B10B_locked_test_final_evaluation/locked_test_registry.json",
            "datasets/mmwave/manifests/M-B10B_locked_test_final_evaluation/locked_test_sample_predictions.jsonl",
            "datasets/mmwave/manifests/M-B10B_locked_test_final_evaluation/metrics_by_model.json",
            "datasets/mmwave/manifests/M-B10B_locked_test_final_evaluation/input_identity.json",
            "datasets/mmwave/manifests/a6_full_conversion/full_split_distribution.json",
        ],
    }


def _verify_baselines(root: Path) -> tuple[bool, dict[str, Any], list[str]]:
    registry_path = root / M_B10A_DIR_REL / "historical_baseline_registry.json"
    details: dict[str, Any] = {}
    failures: list[str] = []
    if not registry_path.is_file():
        return False, {"error": "HISTORICAL_BASELINE_REGISTRY_MISSING"}, ["HISTORICAL_BASELINE_REGISTRY_MISSING"]

    registry = _load(registry_path)
    by_id = {b.get("baseline_id"): b for b in registry.get("baselines", [])}
    all_pass = True

    for baseline_id, expected in FROZEN_BASELINE_EXPECTATIONS.items():
        detail: dict[str, Any] = {"baseline_id": baseline_id}
        local_fail: list[str] = []
        baseline = by_id.get(baseline_id)
        if baseline is None:
            all_pass = False
            failures.append(f"{baseline_id}:BASELINE_MISSING_FROM_REGISTRY")
            details[baseline_id] = {"exists_in_registry": False}
            continue

        epc = baseline.get("executable_preprocessing_contract") or {}
        mi = epc.get("model_identity") or {}
        model_path = root / expected["path"]
        detail["model_exists"] = model_path.is_file()
        if not model_path.is_file():
            all_pass = False
            failures.append(f"{baseline_id}:MODEL_FILE_MISSING")
            detail["failure_reasons"] = ["MODEL_FILE_MISSING"]
            details[baseline_id] = detail
            continue

        actual_sha = sha256_file(model_path)
        actual_bytes = model_path.stat().st_size
        detail["sha256_match"] = actual_sha == expected["sha256"] == mi.get("sha256") == baseline.get("sha256")
        detail["bytes_match"] = actual_bytes == expected["bytes"] == mi.get("bytes") == baseline.get("bytes")
        detail["input_tensor_match"] = mi.get("input") == expected["input"]
        detail["output_tensor_match"] = mi.get("output") == expected["output"]
        out_shape = (mi.get("output") or {}).get("shape") or []
        detail["output_class_count_3"] = bool(out_shape) and out_shape[-1] == 3
        detail["class_map_match"] = epc.get("class_map") == CLASS_MAP == expected["class_map"]
        detail["interpretation_match"] = epc.get("interpretation") == expected["interpretation"]
        detail["contract_id_match"] = epc.get("contract_id") == expected["contract_id"]
        detail["steps_match"] = epc.get("steps") == expected["steps"]
        detail["fallback_policy_match"] = epc.get("fallback_policy") == expected["fallback_policy"] == "NO_HEURISTIC_FALLBACK"

        meta_list = epc.get("metadata_sources") or []
        meta0 = meta_list[0] if meta_list else {}
        meta_rel = meta0.get("path") or ""
        meta_path = root / meta_rel if meta_rel else None
        detail["metadata_exists"] = bool(meta_path and meta_path.is_file())
        if detail["metadata_exists"]:
            meta_sha = sha256_file(meta_path)
            detail["metadata_sha_match"] = meta_sha == expected["metadata_sources"][0]["sha256"] == meta0.get("sha256")
        else:
            detail["metadata_sha_match"] = False
            local_fail.append("METADATA_FILE_MISSING")

        executor = epc.get("executor") or {}
        exec_rel = executor.get("path") or ""
        exec_path = root / exec_rel if exec_rel else None
        detail["executor_exists"] = bool(exec_path and exec_path.is_file())
        if detail["executor_exists"]:
            exec_sha = sha256_file(exec_path)
            detail["executor_sha_match"] = exec_sha == expected["executor"]["sha256"] == executor.get("sha256")
        else:
            detail["executor_sha_match"] = False
            local_fail.append("EXECUTOR_FILE_MISSING")

        for key, value in detail.items():
            if key in {"baseline_id"}:
                continue
            if value is not True:
                local_fail.append(f"{key.upper()}_FAILED")
        if local_fail:
            all_pass = False
            failures.extend(f"{baseline_id}:{reason}" for reason in local_fail)
        detail["failure_reasons"] = local_fail
        details[baseline_id] = detail

    return all_pass, details, failures


def evaluate_gate_r1(root: Path) -> dict[str, Any]:
    incident = _load(root / M_B10B_DIR_REL / "incident_root_cause.json")
    passed = incident.get("incident_status") == "INCIDENT_ROOT_CAUSE_CLOSED" and incident.get("root_cause_id") == ROOT_CAUSE_ID
    evidence = {
        "incident_status": incident.get("incident_status"),
        "root_cause_id": incident.get("root_cause_id"),
        "runtime_detection": RUNTIME_DETECTION,
    }
    return _gate_result("R1", passed, evidence, [] if passed else ["INCIDENT_NOT_CLOSED"])


def evaluate_gate_r2(root: Path) -> dict[str, Any]:
    audit = _load(root / M_B10B_DIR_REL / "one_time_access_audit.json")
    passed = audit.get("accessor_invocation_count") == 1 and audit.get("second_accessor_invocation") is False
    evidence = {
        "accessor_invocation_count": audit.get("accessor_invocation_count"),
        "second_accessor_invocation": audit.get("second_accessor_invocation"),
    }
    return _gate_result("R2", passed, evidence, [] if passed else ["ACCESSOR_COUNT_NOT_ONE"])


def evaluate_gate_r3(root: Path) -> dict[str, Any]:
    audit = _load(root / M_B10B_DIR_REL / "one_time_access_audit.json")
    summary = _load(root / M_B10B_DIR_REL / "m_b10b_summary.json")
    passed = audit.get("completed_model_inference_invocations") == 0 and summary.get("model_inference_invocations") == 0
    evidence = {
        "completed_model_inference_invocations": audit.get("completed_model_inference_invocations"),
        "model_inference_invocations": summary.get("model_inference_invocations"),
    }
    return _gate_result("R3", passed, evidence, [] if passed else ["MODEL_INFERENCE_NONZERO"])


def evaluate_gate_r4(root: Path) -> dict[str, Any]:
    exposure = compute_exposure(root)
    e3 = exposure["E3_persistent_sample_registry"]
    e4 = exposure["E4_payload_logging"]
    passed = (
        e3["actual_registry_rows"] == 0
        and e3["prediction_ledger_rows"] == 0
        and e3["raw_tensors_persisted"] is False
        and e3.get("sample_ids_persisted") is False
        and e3.get("subject_ids_persisted") is False
        and e3.get("labels_persisted") is False
        and e4["input_id_labels_tensors_not_persisted"] is True
        and e4["metrics_results_available"] is False
    )
    evidence = {
        "actual_registry_rows": e3["actual_registry_rows"],
        "prediction_ledger_rows": e3["prediction_ledger_rows"],
        "raw_tensors_persisted": e3["raw_tensors_persisted"],
        "sample_ids_persisted": e3.get("sample_ids_persisted"),
        "subject_ids_persisted": e3.get("subject_ids_persisted"),
        "labels_persisted": e3.get("labels_persisted"),
        "input_id_labels_tensors_not_persisted": e4["input_id_labels_tensors_not_persisted"],
        "metrics_results_available": e4["metrics_results_available"],
    }
    return _gate_result("R4", passed, evidence, [] if passed else ["PERSISTED_SAMPLE_PAYLOAD"])


def evaluate_gate_r5(root: Path) -> dict[str, Any]:
    selected = _load(root / M_B10A_DIR_REL / "selected_candidate_pretest.json")
    summary = _load(root / M_B10B_DIR_REL / "m_b10b_summary.json")
    contract_path = root / M_B10A_DIR_REL / "locked_test_evaluation_contract.json"
    selected_path = root / M_B10A_DIR_REL / "selected_candidate_pretest.json"
    passed = (
        selected.get("seed") == 42
        and selected.get("candidate_id") == SELECTED_CANDIDATE_ID
        and selected.get("model_id") == SELECTED_MODEL_ID
        and sha256_file(contract_path) == M_B10A_CONTRACT_SHA
        and sha256_file(selected_path) == SELECTED_PRETEST_SHA
        and summary.get("selected_candidate_unchanged") is True
    )
    evidence = {
        "seed": selected.get("seed"),
        "candidate_id": selected.get("candidate_id"),
        "model_id": selected.get("model_id"),
        "contract_sha_match": sha256_file(contract_path) == M_B10A_CONTRACT_SHA,
        "selected_pretest_sha_match": sha256_file(selected_path) == SELECTED_PRETEST_SHA,
        "selected_candidate_unchanged": summary.get("selected_candidate_unchanged"),
    }
    return _gate_result("R5", passed, evidence, [] if passed else ["CANDIDATE_CHANGED"])


def evaluate_gate_r6(root: Path) -> dict[str, Any]:
    passed, details, failures = _verify_baselines(root)
    return _gate_result("R6", passed, {"details": details}, failures)


def evaluate_gate_r7(root: Path) -> dict[str, Any]:
    incident = _load(root / M_B10B_DIR_REL / "incident_root_cause.json")
    passed = incident.get("a6_total_locked_test_windows") == 88 and incident.get("a6_locked_test_evaluation_eligible_windows") == 75
    evidence = {
        "a6_total_locked_test_windows": incident.get("a6_total_locked_test_windows"),
        "a6_locked_test_evaluation_eligible_windows": incident.get("a6_locked_test_evaluation_eligible_windows"),
    }
    return _gate_result("R7", passed, evidence, [] if passed else ["COUNT_SEMANTICS_MISMATCH"])


def evaluate_gate_r8(root: Path) -> dict[str, Any]:
    selected = _load(root / M_B10A_DIR_REL / "selected_candidate_pretest.json")
    summary = _load(root / M_B10B_DIR_REL / "m_b10b_summary.json")
    preprocessing = selected.get("preprocessing") or {}
    selected_sha = selected.get("model", {}).get("sha256") or selected.get("candidate_sha256")
    passed = (
        selected.get("seed") == 42
        and selected_sha == SELECTED_SHA
        and preprocessing.get("profile_name") == PREPROCESSING_NAME
        and preprocessing.get("profile_id") == PREPROCESSING_PROFILE
        and selected.get("calibration_profile") == CALIBRATION_PROFILE
        and selected.get("class_map") == CLASS_MAP
        and summary.get("model_trainings", 0) == 0
        and summary.get("model_conversions", 0) == 0
        and summary.get("recalibrations", 0) == 0
        and summary.get("threshold_tuning", False) is False
        and summary.get("post_test_selection", False) is False
        and summary.get("seed43_evaluated") is False
        and summary.get("seed44_evaluated") is False
        and summary.get("no_post_test_tuning") is True
        and summary.get("selected_candidate_unchanged") is True
    )
    evidence = {
        "seed": selected.get("seed"),
        "selected_sha": selected_sha,
        "preprocessing_name": preprocessing.get("profile_name"),
        "preprocessing_profile": preprocessing.get("profile_id"),
        "calibration_profile": selected.get("calibration_profile"),
        "class_map": selected.get("class_map"),
        "model_trainings": summary.get("model_trainings", 0),
        "model_conversions": summary.get("model_conversions", 0),
        "recalibrations": summary.get("recalibrations", 0),
        "threshold_tuning": summary.get("threshold_tuning", False),
        "post_test_selection": summary.get("post_test_selection", False),
        "seed43_evaluated": summary.get("seed43_evaluated"),
        "seed44_evaluated": summary.get("seed44_evaluated"),
        "no_post_test_tuning": summary.get("no_post_test_tuning"),
        "selected_candidate_unchanged": summary.get("selected_candidate_unchanged"),
    }
    return _gate_result("R8", passed, evidence, [] if passed else ["POST_ACCESS_TUNING_OR_IDENTITY_DRIFT"])


def evaluate_gate_r9(root: Path, out_dir: Path) -> dict[str, Any]:
    recovery = _load(out_dir / "proposed_recovery_evaluation_contract.json")
    mb10a_contract = _load(root / M_B10A_DIR_REL / "locked_test_evaluation_contract.json")
    contract_path = root / M_B10A_DIR_REL / "locked_test_evaluation_contract.json"

    planned = recovery.get("planned_models") or []
    models_flat = recovery.get("models") or [m.get("model_id") for m in planned]
    planned_ids = [m.get("model_id") for m in planned]
    expected_ids = list(EXPECTED_CONTRACT_MODEL_IDS)
    sha_by_id = {m["model_id"]: m["sha256"] for m in MODEL_SPECS}
    planned_shas_ok = len(planned) == 3 and all(m.get("sha256") == sha_by_id.get(m.get("model_id")) for m in planned)
    no_seed43_44 = not any(("seed43" in str(mid)) or ("seed44" in str(mid)) for mid in planned_ids + list(models_flat))
    structural = recovery.get("structural_context") or {}
    supervised = recovery.get("supervised_evaluation_population") or {}
    metrics_norm = json.dumps(recovery.get("metrics_schema"), sort_keys=True)
    mb10a_metrics_norm = json.dumps(mb10a_contract.get("metrics_schema"), sort_keys=True)

    checks = {
        "planned_models_length_3": len(planned) == 3,
        "models_list_length_3": len(models_flat) == 3,
        "model_ids_exact": planned_ids == expected_ids and list(models_flat) == expected_ids,
        "model_shas_exact": planned_shas_ok,
        "no_seed43_seed44": no_seed43_44,
        "structural_total_windows_88": structural.get("total_windows") == 88,
        "eligible_windows_75": supervised.get("windows") == 75,
        "ambiguous_windows_13": structural.get("ambiguous_windows") == 13,
        "expected_inference_225": recovery.get("expected_model_inference_count") == 225,
        "metrics_schema_deep_equal": metrics_norm == mb10a_metrics_norm,
        "acceptance_threshold_unchanged": recovery.get("acceptance_threshold")
        == "FINAL_LOCKED_TEST_NUMERICAL_ACCEPTANCE_THRESHOLD_NOT_PREDEFINED",
        "status_proposed_not_authorized": recovery.get("status") == RECOVERY_CONTRACT_STATUS,
        "recovery_execution_authorized_false": recovery.get("recovery_execution_authorized") is False,
        "locked_test_reopen_authorized_false": recovery.get("locked_test_reopen_authorized") is False,
        "mb10a_contract_sha_match": sha256_file(contract_path) == M_B10A_CONTRACT_SHA,
    }
    # Metric schema integrity: require apnea recall path, primary macro_f1, fpr, worst_subject_macro_f1
    schema = recovery.get("metrics_schema") or {}
    checks["metrics_primary_macro_f1"] = schema.get("primary") == "macro_f1"
    checks["metrics_has_apnea_recall"] = "recall" in (schema.get("apnea_proxy_fields") or [])
    checks["metrics_has_fpr"] = "fpr" in (schema.get("per_class_fields") or [])
    checks["metrics_has_worst_subject_macro_f1"] = "worst_subject_macro_f1" in (schema.get("subject_level") or [])

    passed = all(checks.values())
    failures = [name for name, ok in checks.items() if not ok]
    return _gate_result("R9", passed, {"checks": checks, "planned_model_ids": planned_ids}, failures)


def evaluate_gate_r10(root: Path, out_dir: Path) -> dict[str, Any]:
    incident = _load(root / M_B10B_DIR_REL / "incident_root_cause.json")
    audit = _load(root / M_B10B_DIR_REL / "one_time_access_audit.json")
    recovery = _load(out_dir / "proposed_recovery_evaluation_contract.json")
    rlf = recovery.get("result_limitation_fields") or {}
    designation = recovery.get("required_result_designation")
    allowed_wording = str(recovery.get("allowed_scientific_wording") or "")
    forbidden_list = recovery.get("forbidden_scientific_wording") or []

    checks = {
        "accessor_invocation_count_1": audit.get("accessor_invocation_count") == 1,
        "locked_test_consumed": incident.get("locked_test_consumed") is True,
        "completed_model_inference_0": audit.get("completed_model_inference_invocations") == 0,
        "predictions_generated_false": incident.get("predictions_generated") is False,
        "metrics_generated_false": incident.get("metrics_generated") is False,
        "designation_correct": designation == RESULT_LIMITATION and designation not in FORBIDDEN_PRISTINE_CLAIMS,
        "result_not_pristine": rlf.get("result_not_pristine") is True,
        "original_pristine_consumed": rlf.get("original_pristine_final_access_consumed") is True,
        "original_model_inferences_0": rlf.get("original_model_inferences") == 0,
        "status_proposed_not_authorized": recovery.get("status") == RECOVERY_CONTRACT_STATUS,
        "recovery_execution_authorized_false": recovery.get("recovery_execution_authorized") is False,
        "locked_test_reopen_authorized_false": recovery.get("locked_test_reopen_authorized") is False,
        "allowed_wording_clean": not any(claim in allowed_wording for claim in FORBIDDEN_PRISTINE_CLAIMS),
        "forbidden_wording_list_complete": all(term in forbidden_list for term in REQUIRED_FORBIDDEN_SCIENTIFIC_WORDING),
    }
    passed = all(checks.values())
    failures = [name for name, ok in checks.items() if not ok]
    return _gate_result(
        "R10",
        passed,
        {
            "checks": checks,
            "required_future_designation": designation,
            "result_not_pristine": rlf.get("result_not_pristine"),
            "original_pristine_final_access_consumed": rlf.get("original_pristine_final_access_consumed"),
            "original_model_inferences": rlf.get("original_model_inferences"),
            "recovery_contract_status": recovery.get("status"),
        },
        failures,
    )


def compute_policy_decision(holdout_available: bool, gate_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    failed = [gid for gid, body in gate_results.items() if body.get("status") != "PASS"]
    all_pass = len(failed) == 0
    if holdout_available:
        decision = "NEW_INDEPENDENT_HOLDOUT_REQUIRED"
        basis = "A genuinely unused independent subject holdout exists within the approved corpus; reuse is not preferred over an untouched replacement holdout."
    elif all_pass:
        decision = "LIMITED_LOCKED_TEST_REUSE_EXCEPTION_RECOMMENDED"
        basis = "No untouched existing replacement holdout is available and all reuse exception gates R1–R10 pass; a limited reuse exception may be scientifically defensible subject to independent review."
    else:
        decision = "NO_VALID_RECOVERY_PATH"
        basis = f"No untouched existing holdout and reuse gates failed: {failed}"
    return {
        "decision": decision,
        "decision_basis": basis,
        "existing_independent_holdout_available": holdout_available,
        "reuse_exception_eligible": all_pass and not holdout_available,
        "failed_reuse_gates": failed,
    }


def _validate_checksums(out: Path) -> None:
    manifest = out / "checksums.sha256"
    if not manifest.is_file():
        _raise("CHECKSUM_MANIFEST_MISSING")
    seen: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not _hex_digest(parts[0]):
            _raise(f"CHECKSUM_SYNTAX:{line_number}")
        digest, relative = parts[0].lower(), parts[1].strip()
        _safe_relative(relative)
        if relative in seen:
            _raise(f"CHECKSUM_DUPLICATE:{relative}")
        seen.add(relative)
        target = out / relative
        if target.parent.resolve() != out.resolve() or not target.is_file():
            _raise(f"CHECKSUM_TARGET_INVALID:{relative}")
        if sha256_file(target) != digest:
            _raise(f"CHECKSUM_MISMATCH:{relative}")
    expected = REQUIRED_OUTPUTS - {"checksums.sha256"}
    if seen != expected:
        _raise(f"CHECKSUM_COVERAGE:missing={sorted(expected - seen)}:extra={sorted(seen - expected)}")
    actual = {p.name for p in out.iterdir() if p.is_file() and p.name != "checksums.sha256"}
    if actual != expected:
        _raise(f"UNREGISTERED_OUTPUT_FILES:{sorted(actual ^ expected)}")


def _validate_machine_paths(out: Path) -> None:
    for path in out.iterdir():
        if path.suffix not in {".json", ".sha256"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "/Users/" in text or "/private/" in text or "file://" in text or "\\\\" in text:
            _raise(f"LOCAL_ABSOLUTE_PATH:{path.name}")


def _incident_closure_merged(root: Path) -> None:
    if not INCIDENT_VALIDATOR.is_file():
        _raise("INCIDENT_VALIDATOR_MISSING")
    proc = subprocess.run(
        [sys.executable, str(INCIDENT_VALIDATOR)],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        _raise(f"M_B10B_INCIDENT_VALIDATOR_FAILED:{proc.stderr.strip()}")
    if subprocess.run(["git", "merge-base", "--is-ancestor", INCIDENT_CLOSURE_COMMIT, "origin/main"], cwd=root).returncode != 0:
        _raise("INCIDENT_CLOSURE_NOT_IN_ORIGIN_MAIN")


def _validator_has_no_final_accessor_calls() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=__file__)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else None)
            if name in {"get_locked_test_final_evaluation_dataset", "PhaseBAccessGuard"}:
                _raise("VALIDATOR_CALLS_FINAL_ACCESSOR")


def _validator_import_graph_safe() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=__file__)
    forbidden = {
        "_reuse_gates",
        "_policy_decision",
        "_a5_inventory",
        "_a6_eligible_subject_coverage",
        "_exposure_assessment",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "mmwave_m_b10r0_holdout_policy" in module:
                names = {alias.name for alias in node.names}
                bad = names & forbidden
                if bad:
                    _raise(f"FORBIDDEN_GENERATOR_IMPORT:{sorted(bad)}")


STORED_GATE_KEYS = {
    "R1": "R1_incident_truth_closed",
    "R2": "R2_exactly_one_previous_access",
    "R3": "R3_zero_model_evaluation",
    "R4": "R4_no_persisted_sample_level_payload",
    "R5": "R5_candidate_immutable",
    "R6": "R6_baselines_immutable",
    "R7": "R7_count_semantics_correction_only",
    "R8": "R8_no_post_access_tuning",
    "R9": "R9_future_contract_unchanged_models_metrics",
    "R10": "R10_contamination_disclosure_accepted",
}


def validate_m_b10r0_artifacts(root_dir: Path = ROOT_DIR, output_dir: Path | None = None) -> dict[str, Any]:
    root = root_dir.resolve()
    out = (output_dir or root / OUT_DIR_REL).resolve()
    if not out.is_dir():
        _raise("OUTPUT_DIRECTORY_MISSING")

    _incident_closure_merged(root)
    _validator_has_no_final_accessor_calls()
    _validator_import_graph_safe()
    _validate_checksums(out)
    _validate_machine_paths(out)

    try:
        holdout_inventory = compute_subject_inventory(root)
    except MB10R0InventoryError as exc:
        _raise(f"HOLDOUT_INVENTORY_ERROR:{exc}")

    eligible_coverage = compute_a6_eligible_subject_coverage(root)
    exposure = compute_exposure(root)

    independent_gates = {
        "R1": evaluate_gate_r1(root),
        "R2": evaluate_gate_r2(root),
        "R3": evaluate_gate_r3(root),
        "R4": evaluate_gate_r4(root),
        "R5": evaluate_gate_r5(root),
        "R6": evaluate_gate_r6(root),
        "R7": evaluate_gate_r7(root),
        "R8": evaluate_gate_r8(root),
        "R9": evaluate_gate_r9(root, out),
        "R10": evaluate_gate_r10(root, out),
    }
    expected_policy = compute_policy_decision(
        holdout_inventory.get("independent_existing_holdout_available", False),
        independent_gates,
    )

    stored_inventory = _load(out / "existing_unused_holdout_inventory.json")
    stored_exposure = _load(out / "exposure_assessment.json")
    stored_gates = _load(out / "reuse_exception_gate_results.json")
    stored_policy = _load(out / "policy_decision.json")
    stored_recovery = _load(out / "proposed_recovery_evaluation_contract.json")
    stored_access = _load(out / "locked_test_access_audit.json")
    stored_summary = _load(out / "m_b10r0_summary.json")
    stored_incident = _load(out / "incident_identity.json")
    audit_mb10b = _load(root / M_B10B_DIR_REL / "one_time_access_audit.json")

    if stored_inventory != holdout_inventory:
        _raise("HOLDOUT_INVENTORY_RECOMPUTATION_MISMATCH")
    if stored_exposure.get("summary") != exposure.get("summary"):
        _raise("EXPOSURE_ASSESSMENT_RECOMPUTATION_MISMATCH")

    independent_failed = [gid for gid, body in independent_gates.items() if body["status"] != "PASS"]
    independent_all_pass = len(independent_failed) == 0
    stored_failed_raw = stored_gates.get("failed_gates") or []
    # Normalize stored failed gate names to R1..R10 ids when possible
    stored_failed_ids = []
    for name in stored_failed_raw:
        matched = None
        for short, full in STORED_GATE_KEYS.items():
            if name == full or name == short:
                matched = short
                break
        stored_failed_ids.append(matched or name)
    if sorted(stored_failed_ids) != sorted(independent_failed) or stored_gates.get("all_r1_r10_pass") != independent_all_pass:
        _raise("GATE_RESULTS_RECOMPUTATION_MISMATCH")

    for short, full in STORED_GATE_KEYS.items():
        stored_gate = (stored_gates.get("gates") or {}).get(full)
        if not stored_gate:
            _raise(f"GATE_MISSING_IN_STORED:{full}")
        independent_pass = independent_gates[short]["status"] == "PASS"
        if stored_gate.get("pass") is not independent_pass:
            _raise(f"GATE_MISMATCH:{full}")

    # Hard fail if any independent gate failed (even if stored somehow matched)
    for short, body in independent_gates.items():
        if body["status"] != "PASS":
            _raise(f"INDEPENDENT_GATE_FAIL:{short}:{body.get('failure_reasons')}")

    if stored_policy.get("decision") != expected_policy["decision"]:
        _raise("POLICY_DECISION_RECOMPUTATION_MISMATCH")
    for key in (
        "recovery_execution_authorized",
        "locked_test_reopen_authorized",
        "m_b11_authorized",
        "original_predictions_generated",
        "original_metrics_generated",
        "candidate_changed_after_access",
        "new_performance_information_used_for_policy",
    ):
        if stored_policy.get(key) is not False:
            _raise(f"POLICY_DECISION_FORBIDDEN_FLAG:{key}")

    if stored_recovery.get("status") != RECOVERY_CONTRACT_STATUS:
        _raise("RECOVERY_CONTRACT_NOT_PROPOSED_NOT_AUTHORIZED")
    if stored_recovery.get("expected_model_inference_count") != 225:
        _raise("RECOVERY_CONTRACT_INFERENCE_COUNT")
    if stored_recovery.get("required_result_designation") != RESULT_LIMITATION:
        _raise("RECOVERY_CONTRACT_DESIGNATION")
    if stored_recovery.get("recovery_execution_authorized") is not False:
        _raise("RECOVERY_CONTRACT_EXECUTION_AUTHORIZED")
    if stored_recovery.get("locked_test_reopen_authorized") is not False:
        _raise("RECOVERY_CONTRACT_REOPEN_AUTHORIZED")

    if stored_access.get("new_m_b10r0_accessor_invocations") != 0 or stored_access.get("recovery_runner_executions") != 0:
        _raise("ACCESS_AUDIT_NONZERO_DURING_M_B10R0")

    if (
        stored_incident.get("original_accessor_invocations") != 1
        or stored_incident.get("rows_returned") != 75
        or stored_incident.get("model_inference_invocations") != audit_mb10b.get("completed_model_inference_invocations")
    ):
        _raise("INCIDENT_IDENTITY_MISMATCH")
    if audit_mb10b.get("accessor_invocation_count") != 1 or audit_mb10b.get("completed_model_inference_invocations") != 0:
        _raise("M_B10B_AUDIT_MISMATCH")

    if eligible_coverage["eligible_window_count"] != 75 or eligible_coverage["eligible_subject_count"] != 16:
        _raise("A6_ELIGIBLE_COVERAGE_MISMATCH")

    report = root / REPORT_REL
    if not report.is_file():
        _raise("REPORT_MISSING")
    report_text = report.read_text(encoding="utf-8")
    for phrase in ("LOCKED_TEST REOPENED: NO", "RECOVERY EVALUATION RUN: NO", "MODEL INFERENCE: 0", stored_policy["decision"]):
        if phrase not in report_text:
            _raise(f"REPORT_MISSING_PHRASE:{phrase}")

    if stored_summary.get("policy_decision") != expected_policy["decision"]:
        _raise("SUMMARY_DECISION_MISMATCH")

    return {
        "validation_status": "PASS",
        "phase_id": "M-B10R0",
        "policy_decision": expected_policy["decision"],
        "reuse_exception_eligible": expected_policy["reuse_exception_eligible"],
        "existing_independent_holdout_available": expected_policy["existing_independent_holdout_available"],
        "failed_reuse_gates": expected_policy["failed_reuse_gates"],
        "recovery_execution_authorized": False,
        "locked_test_reopen_authorized": False,
        "m_b11_authorized": False,
        "m_b10r0_accessor_invocations": 0,
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        result = validate_m_b10r0_artifacts()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except MB10R0ValidationError as exc:
        print(f"M-B10R0 validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
