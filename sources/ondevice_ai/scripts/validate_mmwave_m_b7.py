#!/usr/bin/env python3
"""Standalone fail-closed validator for SafeNest mmWave Phase M-B7.

The validator never trains or converts a model.  It rebuilds the authoritative
VALIDATION population, regenerates every perturbation, reruns the frozen strict
INT8 artifacts, and recomputes all metrics before comparing them with the saved
evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from run_mmwave_m_b7 import (  # noqa: E402
    ARCHITECTURE_ID,
    CALIBRATION_PROFILE_ID,
    MANIFEST_RELATIVE,
    REQUIRED_OUTPUT_FILENAMES,
    compute_m_b7_evidence,
)
from validate_mmwave_m_b6 import validate_m_b6_artifacts  # noqa: E402


class MB7ValidationError(Exception):
    """Raised whenever persisted M-B7 evidence is not independently provable."""


REQUIRED_MB7_ARTIFACTS = set(REQUIRED_OUTPUT_FILENAMES) | {"checksums.sha256"}
EXPECTED_SEEDS = [42, 43, 44]
EXPECTED_PROFILE_ORDER = [
    "M-B7_CLEAN",
    "M-B7_GAUSSIAN_SNR20",
    "M-B7_GAUSSIAN_SNR10",
    "M-B7_GAUSSIAN_POST_B1_SNR20",
    "M-B7_GAUSSIAN_POST_B1_SNR10",
    "M-B7_AMP_X0_50",
    "M-B7_AMP_X0_75",
    "M-B7_AMP_X1_25",
    "M-B7_AMP_X1_50",
    "M-B7_DRIFT_MILD",
    "M-B7_DRIFT_SEVERE",
    "M-B7_DROPOUT_SHORT",
    "M-B7_DROPOUT_LONG",
    "M-B7_MISSING_FRAME_1PCT",
    "M-B7_MISSING_FRAME_5PCT",
    "M-B7_MOTION_BURST_MILD",
    "M-B7_MOTION_BURST_SEVERE",
    "M-B7_COMBINED_MODERATE",
]

# This literal is deliberately independent of the implementation's profile
# dictionary.  Profile names alone are not accepted as proof of construction.
EXPECTED_PROFILE_FIELDS: Dict[str, Dict[str, Any]] = {
    "M-B7_CLEAN": {
        "family": "CLEAN",
        "injection_domain": "CLEAN_FROZEN_B1_OUTPUT",
        "stochastic": False,
    },
    "M-B7_GAUSSIAN_SNR20": {
        "family": "GAUSSIAN_NOISE",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "target_snr_db": 20.0,
    },
    "M-B7_GAUSSIAN_SNR10": {
        "family": "GAUSSIAN_NOISE",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "target_snr_db": 10.0,
    },
    "M-B7_GAUSSIAN_POST_B1_SNR20": {
        "family": "GAUSSIAN_NOISE",
        "injection_domain": "POST_B1_MODEL_INPUT",
        "stochastic": True,
        "target_snr_db": 20.0,
    },
    "M-B7_GAUSSIAN_POST_B1_SNR10": {
        "family": "GAUSSIAN_NOISE",
        "injection_domain": "POST_B1_MODEL_INPUT",
        "stochastic": True,
        "target_snr_db": 10.0,
    },
    "M-B7_AMP_X0_50": {
        "family": "AMPLITUDE_SCALING",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": False,
        "scale": 0.5,
    },
    "M-B7_AMP_X0_75": {
        "family": "AMPLITUDE_SCALING",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": False,
        "scale": 0.75,
    },
    "M-B7_AMP_X1_25": {
        "family": "AMPLITUDE_SCALING",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": False,
        "scale": 1.25,
    },
    "M-B7_AMP_X1_50": {
        "family": "AMPLITUDE_SCALING",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": False,
        "scale": 1.5,
    },
    "M-B7_DRIFT_MILD": {
        "family": "BASELINE_DRIFT",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "frequency_hz": 0.05,
        "amplitude_rms_multiplier": 0.25,
    },
    "M-B7_DRIFT_SEVERE": {
        "family": "BASELINE_DRIFT",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "frequency_hz": 0.05,
        "amplitude_rms_multiplier": 0.5,
    },
    "M-B7_DROPOUT_SHORT": {
        "family": "CONTIGUOUS_DROPOUT",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "duration_samples": 5,
        "duration_seconds": 0.5,
        "replacement_policy": "LINEAR_INTERPOLATION_INTERIOR_NEAREST_VALID_HOLD_BOUNDARY",
    },
    "M-B7_DROPOUT_LONG": {
        "family": "CONTIGUOUS_DROPOUT",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "duration_samples": 30,
        "duration_seconds": 3.0,
        "replacement_policy": "LINEAR_INTERPOLATION_INTERIOR_NEAREST_VALID_HOLD_BOUNDARY",
    },
    "M-B7_MISSING_FRAME_1PCT": {
        "family": "MISSING_FRAME",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "missing_fraction": 0.01,
        "missing_count": 3,
        "repair_policy": "A3_MMWAVE_TIMELINE_PROFILE_001_LINEAR_INTERPOLATION",
    },
    "M-B7_MISSING_FRAME_5PCT": {
        "family": "MISSING_FRAME",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "missing_fraction": 0.05,
        "missing_count": 15,
        "repair_policy": "A3_MMWAVE_TIMELINE_PROFILE_001_LINEAR_INTERPOLATION",
    },
    "M-B7_MOTION_BURST_MILD": {
        "family": "MOTION_BURST",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "duration_samples": 5,
        "duration_seconds": 0.5,
        "std_multiplier": 3.0,
        "waveform": "SIGNED_RECTANGULAR_ADDITIVE_BURST",
    },
    "M-B7_MOTION_BURST_SEVERE": {
        "family": "MOTION_BURST",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "duration_samples": 5,
        "duration_seconds": 0.5,
        "std_multiplier": 6.0,
        "waveform": "SIGNED_RECTANGULAR_ADDITIVE_BURST",
    },
    "M-B7_COMBINED_MODERATE": {
        "family": "COMBINED",
        "injection_domain": "CANONICAL_PHASE_PRE_B1",
        "stochastic": True,
        "gaussian_target_snr_db": 20.0,
        "amplitude_scale": 0.75,
        "dropout_duration_samples": 5,
        "dropout_duration_seconds": 0.5,
        "application_order": [
            "GAUSSIAN_SNR20",
            "AMPLITUDE_X0_75",
            "DROPOUT_SHORT_LINEAR_INTERPOLATION",
        ],
    },
}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MB7ValidationError(f"Malformed JSON artifact {path.name}: {exc}") from exc


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    line_number = 0
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("row is not a JSON object")
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise MB7ValidationError(
            f"Malformed JSONL artifact {path.name} near line {line_number}: {exc}"
        ) from exc
    return rows


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise MB7ValidationError(f"{label} mismatch against independent recomputation")


def _validate_checksums(manifest_dir: Path) -> None:
    checksum_path = manifest_dir / "checksums.sha256"
    seen: set[str] = set()
    try:
        raw_lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise MB7ValidationError(f"Unable to read checksums.sha256: {exc}") from exc

    for line_number, raw_line in enumerate(raw_lines, 1):
        if not raw_line.strip():
            continue
        parts = raw_line.split(maxsplit=1)
        if len(parts) != 2:
            raise MB7ValidationError(f"Malformed checksum line {line_number}")
        digest, relative = parts[0].lower(), parts[1].strip()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise MB7ValidationError(f"Malformed checksum digest at line {line_number}")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative.startswith(("~", "\\"))
            or "file://" in relative
            or "\\" in relative
        ):
            raise MB7ValidationError(f"Checksum path traversal/absolute path at line {line_number}")
        if relative in seen:
            raise MB7ValidationError(f"Duplicate checksum target: {relative}")
        seen.add(relative)
        target = manifest_dir / relative
        if target.parent.resolve() != manifest_dir.resolve() or not target.is_file():
            raise MB7ValidationError(f"Checksum target missing or escaping directory: {relative}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != digest:
            raise MB7ValidationError(f"Checksum mismatch for {relative}")

    expected_entries = REQUIRED_MB7_ARTIFACTS - {"checksums.sha256"}
    if seen != expected_entries:
        raise MB7ValidationError(
            f"Checksum coverage mismatch: missing={sorted(expected_entries-seen)}, "
            f"unexpected={sorted(seen-expected_entries)}"
        )


def _validate_machine_paths(manifest_dir: Path, root_dir: Path) -> None:
    forbidden = ("/Users/", "file://", str(root_dir.resolve()))
    for path in manifest_dir.iterdir():
        if path.suffix not in (".json", ".jsonl"):
            continue
        text = path.read_text(encoding="utf-8")
        if any(token and token in text for token in forbidden):
            raise MB7ValidationError(f"Absolute local path found in {path.name}")


def _validate_profile_contract(contract: Dict[str, Any]) -> None:
    _require_equal(contract.get("global_perturbation_seed"), 20260811, "perturbation seed")
    _require_equal(contract.get("sample_rate_hz"), 10.0, "sample rate")
    _require_equal(contract.get("window_samples"), 300, "window length")
    _require_equal(contract.get("profile_order"), EXPECTED_PROFILE_ORDER, "profile order")
    _require_equal(contract.get("perturbation_profile_count"), 17, "profile count")
    _require_equal(contract.get("total_inference_profile_count"), 18, "inference profile count")
    _require_equal(contract.get("profiles"), EXPECTED_PROFILE_FIELDS, "profile construction")
    rng = contract.get("rng_derivation", {})
    _require_equal(
        rng.get("identity_string"),
        "{global_seed}|{canonical_sample_index}|{window_id}|{profile_id}",
        "RNG identity formula",
    )
    _require_equal(rng.get("hash"), "SHA-256", "RNG hash")
    _require_equal(
        rng.get("integer_rule"),
        "unsigned big-endian integer from first 8 digest bytes",
        "RNG integer derivation",
    )


def _validate_input_identity(root_dir: Path, artifact: Dict[str, Any]) -> None:
    inputs = artifact.get("inputs")
    if not isinstance(inputs, list) or artifact.get("total_inputs") != len(inputs) or len(inputs) < 40:
        raise MB7ValidationError("Input identity inventory is incomplete")
    seen: set[str] = set()
    for item in inputs:
        relative = item.get("repository_relative_path")
        digest = item.get("measured_sha256")
        if not isinstance(relative, str) or not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise MB7ValidationError("Malformed input identity row")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in seen:
            raise MB7ValidationError("Unsafe or duplicate input identity path")
        seen.add(relative)
        actual_path = root_dir / relative
        if not actual_path.is_file():
            raise MB7ValidationError(f"Input identity target missing: {relative}")
        actual_digest = hashlib.sha256(actual_path.read_bytes()).hexdigest()
        if actual_digest != digest:
            raise MB7ValidationError(f"Input identity SHA mismatch: {relative}")


def _validate_sample_population(rows: Sequence[Dict[str, Any]], expected: Sequence[Dict[str, Any]]) -> None:
    if len(rows) != 18 * 79:
        raise MB7ValidationError(f"Perturbation sample row count mismatch: {len(rows)}")
    expected_identities = {
        (
            row["canonical_sample_index"],
            row["window_id"],
            row["subject_id"],
            row["recording_id"],
            row["true_class"],
            row["true_label"],
        )
        for row in expected[:79]
    }
    for row in rows:
        if row.get("split") != "VALIDATION" or "LOCKED" in str(row.get("split", "")):
            raise MB7ValidationError("TRAIN/LOCKED_TEST row detected in M-B7 sample population")
        identity = (
            row.get("canonical_sample_index"),
            row.get("window_id"),
            row.get("subject_id"),
            row.get("recording_id"),
            row.get("true_class"),
            row.get("true_label"),
        )
        if identity not in expected_identities:
            raise MB7ValidationError("Validation row identity replaced or invented")
        profile_id = row.get("profile_id")
        profile = EXPECTED_PROFILE_FIELDS.get(str(profile_id))
        if profile is None:
            raise MB7ValidationError("Unknown profile in sample population")
        if profile["stochastic"]:
            identity_text = (
                f"20260811|{int(row['canonical_sample_index'])}|"
                f"{row['window_id']}|{profile_id}"
            )
            derived = int.from_bytes(
                hashlib.sha256(identity_text.encode("utf-8")).digest()[:8],
                byteorder="big",
                signed=False,
            )
            if row.get("derived_rng_seed") != derived or not row.get("rng_used"):
                raise MB7ValidationError("Stochastic RNG seed derivation mismatch")
        elif row.get("derived_rng_seed") is not None or row.get("rng_used"):
            raise MB7ValidationError("Non-stochastic profile unexpectedly consumed RNG")
    _require_equal(list(rows), list(expected), "perturbation construction/fidelity")


def _validate_prediction_npz(path: Path, expected_arrays: Dict[str, np.ndarray]) -> None:
    try:
        with np.load(path, allow_pickle=False) as actual:
            if set(actual.files) != set(expected_arrays):
                raise MB7ValidationError("Prediction-vector array-key mismatch")
            for key, expected in expected_arrays.items():
                value = actual[key]
                expected_value = np.asarray(expected)
                if value.dtype != expected_value.dtype or value.shape != expected_value.shape:
                    raise MB7ValidationError(f"Prediction-vector structure mismatch: {key}")
                if not np.array_equal(value, expected_value, equal_nan=True):
                    raise MB7ValidationError(f"Prediction-vector numeric mismatch: {key}")
    except MB7ValidationError:
        raise
    except Exception as exc:
        raise MB7ValidationError(f"Malformed prediction_vectors.npz: {exc}") from exc


def validate_m_b7_artifacts(
    root_dir: Path = ROOT_DIR,
    manifest_dir: Optional[Path] = None,
    *,
    verify_upstream: bool = True,
    reference: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Independently validate all persisted M-B7 evidence without training/conversion."""
    root_dir = Path(root_dir)
    if manifest_dir is None:
        manifest_dir = root_dir / MANIFEST_RELATIVE
    manifest_dir = Path(manifest_dir)
    if not manifest_dir.is_dir():
        raise MB7ValidationError(f"M-B7 manifest directory missing: {manifest_dir}")
    missing = sorted(name for name in REQUIRED_MB7_ARTIFACTS if not (manifest_dir / name).is_file())
    if missing:
        raise MB7ValidationError(f"Required M-B7 artifacts missing: {missing}")

    _validate_machine_paths(manifest_dir, root_dir)
    if verify_upstream:
        upstream = validate_m_b6_artifacts(root_dir=root_dir)
        if not upstream.get("validation_success"):
            raise MB7ValidationError("Upstream M-B6/A5/A6 validation did not pass")

    # The in-memory reference performs fresh data reconstruction, B1 fitting,
    # perturbation generation/replay, strict-INT8 inference, and all metrics.
    try:
        expected = reference if reference is not None else compute_m_b7_evidence(root_dir)
    except Exception as exc:
        raise MB7ValidationError(f"Independent M-B7 recomputation failed: {exc}") from exc

    json_artifacts = {
        name: _load_json(manifest_dir / name)
        for name in REQUIRED_OUTPUT_FILENAMES
        if name.endswith(".json")
    }
    jsonl_artifacts = {
        name: _load_jsonl(manifest_dir / name)
        for name in REQUIRED_OUTPUT_FILENAMES
        if name.endswith(".jsonl")
    }

    _validate_profile_contract(json_artifacts["perturbation_profile_contract.json"])
    _require_equal(
        json_artifacts["perturbation_profile_contract.json"],
        expected["perturbation_profile_contract.json"],
        "full perturbation profile contract",
    )
    _validate_input_identity(root_dir, json_artifacts["input_identity.json"])
    _require_equal(
        json_artifacts["input_identity.json"], expected["input_identity.json"], "input identity"
    )

    contract = json_artifacts["experiment_contract.json"]
    if (
        contract.get("frozen_training_seeds") != EXPECTED_SEEDS
        or contract.get("architecture") != ARCHITECTURE_ID
        or contract.get("frozen_calibration_profile") != CALIBRATION_PROFILE_ID
        or contract.get("validation_pure_class_windows") != 79
        or contract.get("validation_subjects") != 17
        or contract.get("model_trainings") != 0
        or contract.get("model_conversions") != 0
    ):
        raise MB7ValidationError("Frozen experiment contract mismatch")
    _require_equal(contract, expected["experiment_contract.json"], "experiment contract")

    clean = json_artifacts["clean_baseline_results.json"]
    for seed in EXPECTED_SEEDS:
        checks = clean.get("per_seed", {}).get(str(seed), {}).get("m_b6_identity_checks", {})
        if not checks or not all(checks.values()):
            raise MB7ValidationError(f"M-B6 clean identity failure for seed {seed}")
    _require_equal(clean, expected["clean_baseline_results.json"], "M-B6 clean baseline")

    runs = json_artifacts["perturbation_runs.json"]
    if len(runs.get("runs", {})) != 17 * 3:
        raise MB7ValidationError("Strict-INT8 perturbation run inventory mismatch")
    for seed in EXPECTED_SEEDS:
        structure = runs.get("model_artifacts", {}).get(str(seed), {})
        if (
            structure.get("input_dtype") != "int8"
            or structure.get("output_dtype") != "int8"
            or structure.get("select_tf_ops_count") != 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(structure.get("sha256")))
            or int(structure.get("bytes", 0)) <= 0
            or not structure.get("op_types")
        ):
            raise MB7ValidationError(f"Actual strict-INT8 structure gate failed for seed {seed}")
    _require_equal(runs, expected["perturbation_runs.json"], "actual strict-INT8 artifact/run evidence")

    sample_rows = jsonl_artifacts["perturbation_sample_index.jsonl"]
    _validate_sample_population(sample_rows, expected["perturbation_sample_index.jsonl"])
    _validate_prediction_npz(
        manifest_dir / "prediction_vectors.npz", expected["prediction_vectors.npz"]
    )

    # Compare each scientific evidence family separately so a corruption cannot
    # be hidden by a self-consistent forged summary or PASS flag.
    comparisons = [
        ("perturbation_results.json", "per-profile metrics/degradation/collapse"),
        ("cross_seed_robustness_summary.json", "cross-seed aggregation/worst-seed"),
        ("subject_level_robustness.json", "subject-level confusion metrics"),
        ("prediction_changes.jsonl", "clean-to-perturbed prediction changes"),
        ("perturbation_fidelity_audit.json", "achieved perturbation fidelity"),
        ("preprocessing_attenuation_audit.json", "B1 attenuation evidence"),
        ("quantization_diagnostics.json", "strict-INT8 saturation diagnostics"),
        ("fallback_recommendations.json", "fail-closed fallback recommendations"),
        ("determinism_audit.json", "numeric stochastic regeneration"),
        ("exceptions.json", "finding registry"),
        ("run_environment.json", "pinned run environment"),
        ("m_b7_summary.json", "summary derived from recomputed evidence"),
    ]
    for filename, label in comparisons:
        actual = jsonl_artifacts[filename] if filename.endswith(".jsonl") else json_artifacts[filename]
        _require_equal(actual, expected[filename], label)

    locked = json_artifacts["locked_test_access_audit.json"]
    if (
        locked.get("performance_access_attempts") != 0
        or locked.get("prediction_access_attempts") != 0
        or locked.get("label_access_attempts") != 0
        or not locked.get("lock_preserved")
        or locked.get("evaluated_split") != "VALIDATION"
        or locked.get("locked_test_predictions_generated")
    ):
        raise MB7ValidationError("LOCKED_TEST access violation")
    _require_equal(locked, expected["locked_test_access_audit.json"], "LOCKED_TEST audit")

    _validate_checksums(manifest_dir)
    summary = expected["m_b7_summary.json"]
    return {
        "validation_success": True,
        "m_b7_gate_status": summary["gate_status"],
        "m_b8_entry_status": "READY_AFTER_INDEPENDENT_REVIEW",
        "independently_measured": {
            "upstream_m_b0_through_m_b6_a5_a6_verified": bool(verify_upstream),
            "input_identity_verified": True,
            "clean_m_b6_identity_verified": True,
            "strict_int8_structure_and_identity_verified": True,
            "profile_construction_and_rng_verified": True,
            "stochastic_numeric_regeneration_verified": True,
            "fresh_int8_inference_verified": True,
            "perturbation_fidelity_verified": True,
            "preprocessing_attenuation_verified": True,
            "all_metrics_and_degradations_verified": True,
            "collapse_states_verified": True,
            "quantization_saturation_verified": True,
            "cross_seed_aggregation_verified": True,
            "subject_level_evidence_verified": True,
            "locked_test_access_blocked": True,
            "hardened_checksums_verified": True,
            "validation_windows": 79,
            "validation_subjects": 17,
            "strict_int8_invocations": summary["total_strict_int8_invocations"],
        },
    }


def main() -> None:
    result = validate_m_b7_artifacts()
    measured = result["independently_measured"]
    print("Standalone M-B7 Perturbation-Robustness Validation Result:")
    print(f"Validation Success: {result['validation_success']}")
    print(f"M-B7 Gate Status: {result['m_b7_gate_status']}")
    print(f"M-B8 Entry Status: {result['m_b8_entry_status']}")
    print(f"VALIDATION windows/subjects: {measured['validation_windows']}/{measured['validation_subjects']}")
    print(f"Fresh strict-INT8 invocations verified: {measured['strict_int8_invocations']}")
    print(f"Perturbation fidelity verified: {measured['perturbation_fidelity_verified']}")
    print(f"Metric recomputation verified: {measured['all_metrics_and_degradations_verified']}")
    print(f"Subject-level evidence verified: {measured['subject_level_evidence_verified']}")
    print(f"LOCKED_TEST guard verified: {measured['locked_test_access_blocked']}")
    print(f"Hardened checksums verified: {measured['hardened_checksums_verified']}")


if __name__ == "__main__":
    main()
