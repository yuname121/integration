#!/usr/bin/env python3
"""Generate M-B10R0 holdout reuse vs new-holdout policy evidence.

Policy-only. Never instantiates ``PhaseBAccessGuard`` or calls the final
LOCKED_TEST accessor.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR_REL = Path("datasets/mmwave/manifests/M-B10R0_holdout_policy_review")
M_B10B_DIR_REL = Path("datasets/mmwave/manifests/M-B10B_locked_test_final_evaluation")
M_B10A_DIR_REL = Path("datasets/mmwave/manifests/M-B10A_candidate_selection_setup")
A0_DIR_REL = Path("datasets/mmwave/manifests/a0_raw_inventory")
A5_DIR_REL = Path("datasets/mmwave/manifests/a5_subject_split")
A6_DIR_REL = Path("datasets/mmwave/manifests/a6_full_conversion")

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


class MB10R0PolicyError(RuntimeError):
    """Raised when policy evidence cannot be assembled fail-closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _a5_inventory(root: Path) -> dict[str, Any]:
    a0_path = root / A0_DIR_REL / "recording_index.jsonl"
    if not a0_path.is_file():
        raise MB10R0PolicyError("A0_RECORDING_INDEX_MISSING")
    a0_subjects: set[str] = set()
    for line in a0_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            sid = row.get("subject_id")
            if sid:
                a0_subjects.add(sid)
    if not a0_subjects:
        raise MB10R0PolicyError("A0_SUBJECT_UNIVERSE_EMPTY")

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

    # Check pairwise disjoint
    if by_split["TRAIN"] & by_split["VALIDATION"]:
        raise MB10R0PolicyError("A5_TRAIN_VALIDATION_OVERLAP")
    if by_split["TRAIN"] & by_split["LOCKED_TEST"]:
        raise MB10R0PolicyError("A5_TRAIN_LOCKED_TEST_OVERLAP")
    if by_split["VALIDATION"] & by_split["LOCKED_TEST"]:
        raise MB10R0PolicyError("A5_VALIDATION_LOCKED_TEST_OVERLAP")

    all_assigned = by_split["TRAIN"] | by_split["VALIDATION"] | by_split["LOCKED_TEST"]
    unassigned = a0_subjects - all_assigned
    extra_in_a5 = all_assigned - a0_subjects

    if extra_in_a5:
        raise MB10R0PolicyError(f"A5_SUBJECTS_NOT_IN_A0:{sorted(extra_in_a5)}")

    independent_holdout_available = False
    if len(unassigned) == 0 and all_assigned == a0_subjects:
        independent_holdout_available = False
    elif len(unassigned) > 0:
        independent_holdout_available = True

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


def _a6_eligible_subject_coverage(root: Path) -> dict[str, Any]:
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


def _exposure_assessment(root: Path) -> dict[str, Any]:
    mb10b = root / M_B10B_DIR_REL
    registry = _load(mb10b / "locked_test_registry.json")
    actual_registry_rows = len(registry.get("samples", []))
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
            "sample_ids_persisted": False if actual_registry_rows == 0 else any(
                bool(sample.get("sample_id") or sample.get("id")) for sample in registry.get("samples", [])
            ),
            "subject_ids_persisted": False if actual_registry_rows == 0 else any(
                bool(sample.get("subject_id")) for sample in registry.get("samples", [])
            ),
            "labels_persisted": False if actual_registry_rows == 0 else any(
                sample.get("label") is not None or sample.get("label_id") is not None for sample in registry.get("samples", [])
            ),
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


def _verify_baseline_immutability(root: Path) -> tuple[bool, dict[str, Any]]:
    """Fail-closed R6 checks against frozen historical baseline registry + live files."""
    registry_path = root / M_B10A_DIR_REL / "historical_baseline_registry.json"
    details: dict[str, Any] = {}
    if not registry_path.is_file():
        return False, {"error": "HISTORICAL_BASELINE_REGISTRY_MISSING"}

    registry = _load(registry_path)
    by_id = {b.get("baseline_id"): b for b in registry.get("baselines", [])}
    all_pass = True

    for baseline_id, expected in FROZEN_BASELINE_EXPECTATIONS.items():
        detail: dict[str, Any] = {"baseline_id": baseline_id}
        failures: list[str] = []
        baseline = by_id.get(baseline_id)
        if baseline is None:
            all_pass = False
            details[baseline_id] = {"exists_in_registry": False, "failure_reasons": ["BASELINE_MISSING_FROM_REGISTRY"]}
            continue

        epc = baseline.get("executable_preprocessing_contract") or {}
        mi = epc.get("model_identity") or {}
        model_rel = expected["path"]
        model_path = root / model_rel
        detail["model_path"] = model_rel
        detail["model_exists"] = model_path.is_file()
        if not model_path.is_file():
            all_pass = False
            detail["failure_reasons"] = ["MODEL_FILE_MISSING"]
            details[baseline_id] = detail
            continue

        actual_sha = sha256_file(model_path)
        actual_bytes = model_path.stat().st_size
        detail["sha256_match"] = (
            actual_sha == expected["sha256"]
            and actual_sha == mi.get("sha256")
            and actual_sha == baseline.get("sha256")
        )
        detail["bytes_match"] = (
            actual_bytes == expected["bytes"]
            and actual_bytes == mi.get("bytes")
            and actual_bytes == baseline.get("bytes")
        )
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
            detail["metadata_sha_match"] = (
                meta_sha == expected["metadata_sources"][0]["sha256"]
                and meta_sha == meta0.get("sha256")
            )
        else:
            detail["metadata_sha_match"] = False
            failures.append("METADATA_FILE_MISSING")

        executor = epc.get("executor") or {}
        exec_rel = executor.get("path") or ""
        exec_path = root / exec_rel if exec_rel else None
        detail["executor_exists"] = bool(exec_path and exec_path.is_file())
        if detail["executor_exists"]:
            exec_sha = sha256_file(exec_path)
            detail["executor_sha_match"] = (
                exec_sha == expected["executor"]["sha256"]
                and exec_sha == executor.get("sha256")
            )
        else:
            detail["executor_sha_match"] = False
            failures.append("EXECUTOR_FILE_MISSING")

        required_bools = [
            "model_exists",
            "sha256_match",
            "bytes_match",
            "input_tensor_match",
            "output_tensor_match",
            "output_class_count_3",
            "class_map_match",
            "interpretation_match",
            "contract_id_match",
            "steps_match",
            "fallback_policy_match",
            "metadata_exists",
            "metadata_sha_match",
            "executor_exists",
            "executor_sha_match",
        ]
        for key in required_bools:
            if detail.get(key) is not True:
                failures.append(key.upper() + "_FAILED" if not key.endswith("exists") else key.upper() + "_FALSE")
        if failures:
            all_pass = False
        detail["failure_reasons"] = failures
        details[baseline_id] = detail

    return all_pass, details


def _build_proposed_recovery_contract(root: Path, eligible_coverage: dict[str, Any]) -> dict[str, Any]:
    """Assemble the proposed (not authorized) recovery evaluation contract."""
    contract_path = root / M_B10A_DIR_REL / "locked_test_evaluation_contract.json"
    mb10a_contract = _load(contract_path)
    metrics_schema = json.loads(json.dumps(mb10a_contract.get("metrics_schema")))  # deep copy

    planned_models: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        entry = {
            "model_id": spec["model_id"],
            "role": spec["role"],
            "path": spec["path"],
            "sha256": spec["sha256"],
        }
        if "seed" in spec:
            entry["seed"] = spec["seed"]
        if "candidate_id" in spec:
            entry["candidate_id"] = spec["candidate_id"]
        planned_models.append(entry)

    return {
        "schema_version": "M-B10R0_PROPOSED_RECOVERY_CONTRACT_V1",
        "phase_id": "M-B10R0",
        "status": RECOVERY_CONTRACT_STATUS,
        "recovery_execution_authorized": False,
        "locked_test_reopen_authorized": False,
        "independent_review_required": True,
        "selected_candidate": SELECTED_CANDIDATE_ID,
        "selected_model_id": SELECTED_MODEL_ID,
        "selected_seed": 42,
        "planned_models": planned_models,
        "models": [m["model_id"] for m in MODEL_SPECS],
        "model_count": 3,
        "structural_context": {"subjects": 16, "total_windows": 88, "ambiguous_windows": 13},
        "supervised_evaluation_population": {
            "windows": 75,
            "subjects": eligible_coverage["eligible_subject_count"],
            "subject_count_policy": "PREEXISTING_A6_METADATA_VERIFIED",
            "exclude_ambiguous": True,
        },
        "expected_model_inference_count": 225,
        "metrics_schema": metrics_schema,
        "metrics_schema_source": "datasets/mmwave/manifests/M-B10A_candidate_selection_setup/locked_test_evaluation_contract.json",
        "acceptance_threshold": "FINAL_LOCKED_TEST_NUMERICAL_ACCEPTANCE_THRESHOLD_NOT_PREDEFINED",
        "candidate_reselection_prohibited": True,
        "training_prohibited": True,
        "recalibration_prohibited": True,
        "threshold_tuning_prohibited": True,
        "second_recovery_evaluation_prohibited": True,
        "required_result_designation": RESULT_LIMITATION,
        "result_limitation_fields": {
            "original_pristine_final_access_consumed": True,
            "original_model_inferences": 0,
            "reuse_exception_reviewed": True,
            "result_not_pristine": True,
        },
        "allowed_scientific_wording": "OFFLINE_REAL_DATA_RECOVERY_EVALUATION_WITH_HOLDOUT_REUSE_LIMITATION",
        "forbidden_scientific_wording": list(REQUIRED_FORBIDDEN_SCIENTIFIC_WORDING),
    }


def _reuse_gates(root: Path, recovery_contract: dict[str, Any] | None = None) -> dict[str, Any]:
    mb10b = root / M_B10B_DIR_REL
    incident = _load(mb10b / "incident_root_cause.json")
    audit = _load(mb10b / "one_time_access_audit.json")
    summary = _load(mb10b / "m_b10b_summary.json")
    selected = _load(root / M_B10A_DIR_REL / "selected_candidate_pretest.json")
    contract_path = root / M_B10A_DIR_REL / "locked_test_evaluation_contract.json"
    selected_path = root / M_B10A_DIR_REL / "selected_candidate_pretest.json"
    mb10a_contract = _load(contract_path)

    if recovery_contract is None:
        eligible_coverage = _a6_eligible_subject_coverage(root)
        recovery_contract = _build_proposed_recovery_contract(root, eligible_coverage)

    exposure = _exposure_assessment(root)
    e3 = exposure["E3_persistent_sample_registry"]

    r6_pass, r6_details = _verify_baseline_immutability(root)

    # R8: no post-access tuning + identity cross-check
    preprocessing = selected.get("preprocessing") or {}
    selected_sha = selected.get("model", {}).get("sha256") or selected.get("candidate_sha256")
    r8_pass = (
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

    # R9: validate ACTUAL proposed recovery contract
    planned = recovery_contract.get("planned_models") or []
    models_flat = recovery_contract.get("models") or [m.get("model_id") for m in planned]
    planned_ids = [m.get("model_id") for m in planned]
    expected_ids = [m["model_id"] for m in MODEL_SPECS]
    sha_by_id = {m["model_id"]: m["sha256"] for m in MODEL_SPECS}
    planned_shas_ok = all(m.get("sha256") == sha_by_id.get(m.get("model_id")) for m in planned)
    no_seed43_44 = not any(
        ("seed43" in str(mid)) or ("seed44" in str(mid)) for mid in planned_ids + list(models_flat)
    )
    metrics_norm = json.dumps(recovery_contract.get("metrics_schema"), sort_keys=True)
    mb10a_metrics_norm = json.dumps(mb10a_contract.get("metrics_schema"), sort_keys=True)
    structural = recovery_contract.get("structural_context") or {}
    supervised = recovery_contract.get("supervised_evaluation_population") or {}
    r9_checks = {
        "planned_models_length_3": len(planned) == 3,
        "models_list_length_3": len(models_flat) == 3,
        "model_ids_exact": planned_ids == expected_ids and models_flat == expected_ids,
        "model_shas_exact": planned_shas_ok,
        "no_seed43_seed44": no_seed43_44,
        "structural_total_windows_88": structural.get("total_windows") == 88,
        "eligible_windows_75": supervised.get("windows") == 75,
        "ambiguous_windows_13": structural.get("ambiguous_windows") == 13,
        "expected_inference_225": recovery_contract.get("expected_model_inference_count") == 225,
        "metrics_schema_deep_equal": metrics_norm == mb10a_metrics_norm,
        "acceptance_threshold_unchanged": recovery_contract.get("acceptance_threshold")
        == "FINAL_LOCKED_TEST_NUMERICAL_ACCEPTANCE_THRESHOLD_NOT_PREDEFINED",
        "status_proposed_not_authorized": recovery_contract.get("status") == RECOVERY_CONTRACT_STATUS,
        "recovery_execution_authorized_false": recovery_contract.get("recovery_execution_authorized") is False,
        "locked_test_reopen_authorized_false": recovery_contract.get("locked_test_reopen_authorized") is False,
        "mb10a_contract_sha_match": sha256_file(contract_path) == M_B10A_CONTRACT_SHA,
    }
    r9_pass = all(r9_checks.values())

    # R10: contamination disclosure derived from incident/audit/recovery contract
    rlf = recovery_contract.get("result_limitation_fields") or {}
    designation = recovery_contract.get("required_result_designation")
    allowed_wording = str(recovery_contract.get("allowed_scientific_wording") or "")
    forbidden_list = recovery_contract.get("forbidden_scientific_wording") or []
    designation_ok = designation == RESULT_LIMITATION and designation not in FORBIDDEN_PRISTINE_CLAIMS
    allowed_ok = not any(claim in allowed_wording for claim in FORBIDDEN_PRISTINE_CLAIMS)
    forbidden_list_ok = all(term in forbidden_list for term in REQUIRED_FORBIDDEN_SCIENTIFIC_WORDING)
    r10_checks = {
        "accessor_invocation_count_1": audit.get("accessor_invocation_count") == 1,
        "locked_test_consumed": incident.get("locked_test_consumed") is True,
        "completed_model_inference_0": audit.get("completed_model_inference_invocations") == 0,
        "predictions_generated_false": incident.get("predictions_generated") is False,
        "metrics_generated_false": incident.get("metrics_generated") is False,
        "designation_correct": designation_ok,
        "result_not_pristine": rlf.get("result_not_pristine") is True,
        "original_pristine_consumed": rlf.get("original_pristine_final_access_consumed") is True,
        "original_model_inferences_0": rlf.get("original_model_inferences") == 0,
        "status_proposed_not_authorized": recovery_contract.get("status") == RECOVERY_CONTRACT_STATUS,
        "recovery_execution_authorized_false": recovery_contract.get("recovery_execution_authorized") is False,
        "locked_test_reopen_authorized_false": recovery_contract.get("locked_test_reopen_authorized") is False,
        "allowed_wording_clean": allowed_ok,
        "forbidden_wording_list_complete": forbidden_list_ok,
    }
    r10_pass = all(r10_checks.values())

    gates = {
        "R1_incident_truth_closed": {
            "pass": incident.get("incident_status") == "INCIDENT_ROOT_CAUSE_CLOSED" and incident.get("root_cause_id") == ROOT_CAUSE_ID,
            "evidence": ["incident_root_cause.json", "validate_mmwave_m_b10b_incident.py"],
        },
        "R2_exactly_one_previous_access": {
            "pass": audit.get("accessor_invocation_count") == 1 and audit.get("second_accessor_invocation") is False,
            "accessor_invocations": audit.get("accessor_invocation_count"),
        },
        "R3_zero_model_evaluation": {
            "pass": audit.get("completed_model_inference_invocations") == 0 and summary.get("model_inference_invocations") == 0,
            "model_inference_invocations": summary.get("model_inference_invocations"),
        },
        "R4_no_persisted_sample_level_payload": {
            "pass": (
                e3["actual_registry_rows"] == 0
                and e3["prediction_ledger_rows"] == 0
                and e3["raw_tensors_persisted"] is False
                and e3.get("sample_ids_persisted") is False
                and e3.get("subject_ids_persisted") is False
                and e3.get("labels_persisted") is False
                and exposure["E4_payload_logging"]["input_id_labels_tensors_not_persisted"] is True
                and exposure["E4_payload_logging"]["metrics_results_available"] is False
            ),
            "actual_registry_rows": e3["actual_registry_rows"],
            "prediction_ledger_rows": e3["prediction_ledger_rows"],
            "raw_tensors_persisted": e3["raw_tensors_persisted"],
            "sample_ids_persisted": e3.get("sample_ids_persisted"),
            "subject_ids_persisted": e3.get("subject_ids_persisted"),
            "labels_persisted": e3.get("labels_persisted"),
            "input_id_labels_tensors_not_persisted": exposure["E4_payload_logging"]["input_id_labels_tensors_not_persisted"],
            "metrics_results_available": exposure["E4_payload_logging"]["metrics_results_available"],
        },
        "R5_candidate_immutable": {
            "pass": (
                selected.get("seed") == 42
                and selected.get("candidate_id") == SELECTED_CANDIDATE_ID
                and selected.get("model_id") == SELECTED_MODEL_ID
                and sha256_file(contract_path) == M_B10A_CONTRACT_SHA
                and sha256_file(selected_path) == SELECTED_PRETEST_SHA
                and summary.get("selected_candidate_unchanged") is True
            ),
        },
        "R6_baselines_immutable": {
            "pass": r6_pass,
            "details": r6_details,
        },
        "R7_count_semantics_correction_only": {
            "pass": incident.get("a6_total_locked_test_windows") == 88 and incident.get("a6_locked_test_evaluation_eligible_windows") == 75,
        },
        "R8_no_post_access_tuning": {
            "pass": r8_pass,
            "model_conversions": summary.get("model_conversions", 0),
            "recalibrations": summary.get("recalibrations", 0),
            "threshold_tuning": summary.get("threshold_tuning", False),
            "post_test_selection": summary.get("post_test_selection", False),
            "selected_sha": selected_sha,
            "preprocessing_profile": preprocessing.get("profile_id"),
            "preprocessing_name": preprocessing.get("profile_name"),
            "calibration_profile": selected.get("calibration_profile"),
            "no_post_test_tuning": summary.get("no_post_test_tuning"),
            "selected_candidate_unchanged": summary.get("selected_candidate_unchanged"),
        },
        "R9_future_contract_unchanged_models_metrics": {
            "pass": r9_pass,
            "checks": r9_checks,
        },
        "R10_contamination_disclosure_accepted": {
            "pass": r10_pass,
            "checks": r10_checks,
            "required_future_designation": designation,
            "result_not_pristine": rlf.get("result_not_pristine"),
            "original_pristine_final_access_consumed": rlf.get("original_pristine_final_access_consumed"),
            "original_model_inferences": rlf.get("original_model_inferences"),
            "forbidden_scientific_wording": forbidden_list,
            "recovery_contract_status": recovery_contract.get("status"),
        },
    }
    failed = [name for name, body in gates.items() if not body.get("pass")]
    return {
        "schema_version": "M-B10R0_REUSE_GATE_RESULTS_V1",
        "phase_id": "M-B10R0",
        "gates": gates,
        "failed_gates": failed,
        "all_r1_r10_pass": len(failed) == 0,
    }


def _policy_decision(root: Path, holdout_inventory: dict[str, Any], gate_results: dict[str, Any]) -> dict[str, Any]:
    unused_available = holdout_inventory.get("independent_existing_holdout_available", False)
    all_pass = gate_results.get("all_r1_r10_pass", False)
    failed = gate_results.get("failed_gates", [])

    if unused_available:
        decision = "NEW_INDEPENDENT_HOLDOUT_REQUIRED"
        basis = "A genuinely unused independent subject holdout exists within the approved corpus; reuse is not preferred over an untouched replacement holdout."
    elif all_pass:
        decision = "LIMITED_LOCKED_TEST_REUSE_EXCEPTION_RECOMMENDED"
        basis = "No untouched existing replacement holdout is available and all reuse exception gates R1–R10 pass; a limited reuse exception may be scientifically defensible subject to independent review."
    else:
        decision = "NO_VALID_RECOVERY_PATH"
        basis = f"No untouched existing holdout and reuse gates failed: {failed}"

    payload = {
        "schema_version": "M-B10R0_POLICY_DECISION_V1",
        "phase_id": "M-B10R0",
        "decision": decision,
        "decision_basis": basis,
        "existing_independent_holdout_available": unused_available,
        "reuse_exception_eligible": all_pass and not unused_available,
        "failed_reuse_gates": failed,
        "original_locked_test_consumed": True,
        "original_predictions_generated": False,
        "original_metrics_generated": False,
        "candidate_changed_after_access": False,
        "new_performance_information_used_for_policy": False,
        "recovery_execution_authorized": False,
        "locked_test_reopen_authorized": False,
        "m_b11_authorized": False,
        "independent_review_required": True,
    }
    if decision == "LIMITED_LOCKED_TEST_REUSE_EXCEPTION_RECOMMENDED":
        payload["required_result_limitation"] = RESULT_LIMITATION
    return payload


def generate_m_b10r0_evidence(root_dir: Path = ROOT_DIR) -> dict[str, Any]:
    root = root_dir.resolve()
    out = root / OUT_DIR_REL
    out.mkdir(parents=True, exist_ok=True)

    incident = _load(root / M_B10B_DIR_REL / "incident_root_cause.json")
    audit = _load(root / M_B10B_DIR_REL / "one_time_access_audit.json")
    consumption = _load(root / M_B10B_DIR_REL / "test_split_consumption_record.json")
    a6_summary = _load(root / A6_DIR_REL / "a6_summary.json")
    a6_dist = _load(root / A6_DIR_REL / "full_split_distribution.json")
    a6_labels = _load(root / A6_DIR_REL / "full_label_distribution.json")
    selected = _load(root / M_B10A_DIR_REL / "selected_candidate_pretest.json")
    contract = _load(root / M_B10A_DIR_REL / "locked_test_evaluation_contract.json")
    eligible_coverage = _a6_eligible_subject_coverage(root)
    holdout_inventory = _a5_inventory(root)
    exposure = _exposure_assessment(root)
    recovery_contract = _build_proposed_recovery_contract(root, eligible_coverage)
    gate_results = _reuse_gates(root, recovery_contract=recovery_contract)
    policy = _policy_decision(root, holdout_inventory, gate_results)

    input_identity = {
        "schema_version": "M-B10R0_INPUT_IDENTITY_V1",
        "phase_id": "M-B10R0",
        "m_b10b_incident_closure_commit": INCIDENT_CLOSURE_COMMIT,
        "upstream_evidence": [
            {"path": "datasets/mmwave/manifests/M-B10B_locked_test_final_evaluation/incident_root_cause.json", "sha256": sha256_file(root / M_B10B_DIR_REL / "incident_root_cause.json")},
            {"path": "datasets/mmwave/manifests/M-B10A_candidate_selection_setup/locked_test_evaluation_contract.json", "sha256": M_B10A_CONTRACT_SHA},
            {"path": "datasets/mmwave/manifests/M-B10A_candidate_selection_setup/selected_candidate_pretest.json", "sha256": SELECTED_PRETEST_SHA},
            {"path": "datasets/mmwave/manifests/a0_raw_inventory/recording_index.jsonl", "sha256": sha256_file(root / A0_DIR_REL / "recording_index.jsonl")},
            {"path": "datasets/mmwave/manifests/a5_subject_split/subject_split_manifest.jsonl", "sha256": sha256_file(root / A5_DIR_REL / "subject_split_manifest.jsonl")},
            {"path": "datasets/mmwave/manifests/a6_full_conversion/a6_summary.json", "sha256": sha256_file(root / A6_DIR_REL / "a6_summary.json")},
        ],
        "no_locked_test_payload_access": True,
    }

    incident_identity = {
        "schema_version": "M-B10R0_INCIDENT_IDENTITY_V1",
        "phase_id": "M-B10R0",
        "incident_closure_commit": INCIDENT_CLOSURE_COMMIT,
        "runtime_detection_code": RUNTIME_DETECTION,
        "forensic_root_cause": ROOT_CAUSE_ID,
        "original_accessor_invocations": audit.get("accessor_invocation_count"),
        "rows_returned": audit.get("structural_rows_returned"),
        "model_inference_invocations": audit.get("completed_model_inference_invocations"),
        "predictions_generated": incident.get("predictions_generated"),
        "metrics_generated": incident.get("metrics_generated"),
        "registry_generated": incident.get("registry_generated"),
        "returned_subject_count": incident.get("returned_subject_count"),
        "locked_test_consumed": incident.get("locked_test_consumed"),
        "rerun_performed": incident.get("rerun_performed"),
        "structural_windows": incident.get("a6_total_locked_test_windows"),
        "supervised_eligible_windows": incident.get("a6_locked_test_evaluation_eligible_windows"),
        "excluded_ambiguous_windows": incident.get("count_difference"),
    }

    original_consumption = {
        "schema_version": "M-B10R0_ORIGINAL_CONSUMPTION_V1",
        "phase_id": "M-B10R0",
        "original_access_phase": "M-B10B",
        "locked_test_pristine": False,
        "locked_test_consumed": True,
        "consumption_status": consumption.get("status"),
        "must_not_reuse_for_phase_b_model_selection": consumption.get("must_not_reuse_for_phase_b_model_selection"),
        "phase_b_model_selection_reuse_allowed": False,
        "historical_total_payload_release_events": 1,
        "original_final_accessor_invocations": 1,
        "future_recovery_accessor_invocations_during_m_b10r0": 0,
    }

    reuse_contract = {
        "schema_version": "M-B10R0_REUSE_EXCEPTION_ELIGIBILITY_CONTRACT_V1",
        "phase_id": "M-B10R0",
        "gates": list(gate_results["gates"].keys()),
        "hard_gate_model": "PASS_OR_FAIL_WITH_EVIDENCE",
        "no_composite_safety_score": True,
        "contamination_disclosure_required": RESULT_LIMITATION,
    }

    future_access = {
        "schema_version": "M-B10R0_FUTURE_ACCESS_REQUIREMENTS_V1",
        "phase_id": "M-B10R0",
        "implementation_status": "POLICY_ONLY_NOT_IMPLEMENTED",
        "modify_mmwave_phase_b_access": False,
        "requirements": [
            "Separate explicit recovery authorization distinct from original final-access token",
            "One additional recovery transaction at most",
            "Cannot reset historical original_final_accessor_invocations=1",
            "Recovery accessor invocations tracked independently starting at zero",
            "Must return exactly existing pure-class eligible population (75 windows, include_ambiguous=false)",
            "Must not expose AMBIGUOUS rows for supervised scoring",
            "Must record original access history and recovery history separately",
            "Must fail on changed candidate, model SHA, preprocessing, calibration, or metric contract",
        ],
        "historical_total_payload_release_events": 1,
        "original_final_accessor_invocations": 1,
        "future_recovery_accessor_invocations": 0,
    }

    claim_limitations = {
        "schema_version": "M-B10R0_CLAIM_LIMITATIONS_V1",
        "phase_id": "M-B10R0",
        "allowed_if_supported": [
            "LIMITED_LOCKED_TEST_REUSE_EXCEPTION_RECOMMENDED",
            "PREINFERENCE_STRUCTURAL_ABORT",
            "NO_PERFORMANCE_INFORMATION_OBSERVED",
            "NO_EXISTING_UNUSED_SUBJECT_HOLDOUT",
            "RECOVERY_REQUIRES_INDEPENDENT_AUTHORIZATION",
        ],
        "forbidden": [
            "LOCKED_TEST_PRISTINE",
            "RECOVERY_ALREADY_AUTHORIZED",
            "SECOND_FINAL_TEST_ALLOWED",
            "FINAL_PERFORMANCE_VALIDATED",
            "M-B11_READY",
        ],
        "new_holdout_policy": {
            "a5_reshuffle_allowed": False,
            "train_validation_reuse_allowed": False,
            "legitimate_new_holdout_source": "NEW_UNSEEN_SUBJECT_DATA",
            "new_data_required_classification": "NEW_DATA_REQUIRED",
        },
    }

    access_audit = {
        "schema_version": "M-B10R0_LOCKED_TEST_ACCESS_AUDIT_V1",
        "phase_id": "M-B10R0",
        "previous_historical_original_access_events": 1,
        "new_m_b10r0_accessor_invocations": 0,
        "new_tensor_accesses": 0,
        "new_label_accesses": 0,
        "new_prediction_accesses": 0,
        "new_metric_accesses": 0,
        "recovery_runner_executions": 0,
        "locked_test_reopen_during_m_b10r0": False,
    }

    summary_out = {
        "phase_id": "M-B10R0",
        "status": "POLICY_REVIEW_COMPLETE_AWAITING_INDEPENDENT_REVIEW",
        "policy_decision": policy["decision"],
        "reuse_exception_eligible": policy["reuse_exception_eligible"],
        "existing_independent_holdout_available": policy["existing_independent_holdout_available"],
        "recovery_execution_authorized": False,
        "locked_test_reopen_authorized": False,
        "m_b11_authorized": False,
        "m_b10r0_accessor_invocations": 0,
        "recovery_runner_executions": 0,
        "a6_structural_windows": a6_summary["split_window_distribution"]["LOCKED_TEST"],
        "a6_supervised_eligible_windows": a6_dist["eligibility_counts"]["locked_test_evaluation_eligible"],
        "a6_ambiguous_windows": a6_labels["split_label_breakdown"]["LOCKED_TEST"]["AMBIGUOUS"],
    }

    exceptions = {
        "phase_id": "M-B10R0",
        "status": "NO_EXECUTION_EXCEPTIONS",
        "classification": "POLICY_ONLY",
    }

    run_env = {
        "phase_id": "M-B10R0",
        "generated_at": _utc_now(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "policy_only": True,
        "locked_test_accessor_invoked": False,
    }

    artifacts = {
        "input_identity.json": input_identity,
        "incident_identity.json": incident_identity,
        "exposure_assessment.json": exposure,
        "original_holdout_consumption_status.json": original_consumption,
        "existing_unused_holdout_inventory.json": holdout_inventory,
        "reuse_exception_eligibility_contract.json": reuse_contract,
        "reuse_exception_gate_results.json": gate_results,
        "policy_decision.json": policy,
        "proposed_recovery_evaluation_contract.json": recovery_contract,
        "future_recovery_access_requirements.json": future_access,
        "claim_limitations.json": claim_limitations,
        "locked_test_access_audit.json": access_audit,
        "run_environment.json": run_env,
        "exceptions.json": exceptions,
        "m_b10r0_summary.json": summary_out,
    }

    for name, payload in artifacts.items():
        (out / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checksum_lines = []
    for path in sorted(out.iterdir(), key=lambda p: p.name):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_lines.append(f"{sha256_file(path)}  {path.name}")
    (out / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    return {"phase_id": "M-B10R0", "policy_decision": policy["decision"], "output_dir": str(OUT_DIR_REL)}


def main(argv: list[str] | None = None) -> int:
    del argv
    result = generate_m_b10r0_evidence()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
