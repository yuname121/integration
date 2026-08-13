#!/usr/bin/env python3
"""Execute the preregistered C-B5 offline robustness and final-lock run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.co2.b5_robustness import (  # noqa: E402
    ARTIFACT_DIR_REL,
    CANDIDATE_DIR_REL,
    MODEL_REL,
    SCALER_FINGERPRINT,
    build_final_candidate_metadata,
    build_pre_locked_test_freeze,
    build_protocol,
    evaluate_locked_test_once,
    file_sha256,
    host_latency_sanity,
    load_json,
    load_split_rows,
    reconstruct_features,
    run_robustness,
    stable_sha256,
    validate_pre_locked_test_freeze,
    write_final_lock,
    write_json,
)


def _tensorflow():
    import tensorflow as tf  # type: ignore

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass
    return tf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / ARTIFACT_DIR_REL
    output.mkdir(parents=True, exist_ok=True)
    if (output / "locked_test_evaluation.json").exists():
        print("C_B5_LOCKED_TEST_DOUBLE_EVALUATION", file=sys.stderr)
        return 2

    # The protocol is written before TensorFlow inference or any result file.
    protocol = build_protocol()
    protocol_path = output / "robustness_protocol.json"
    write_json(protocol_path, protocol)
    protocol_sha = file_sha256(root, f"{ARTIFACT_DIR_REL}/robustness_protocol.json")

    tf = _tensorflow()
    robustness = run_robustness(tf, root, protocol)
    robustness_path = output / "robustness_results.json"
    write_json(robustness_path, robustness)
    robustness_sha = file_sha256(root, f"{ARTIFACT_DIR_REL}/robustness_results.json")

    # Latency uses the first unperturbed VALIDATION sample only.  No LOCKED_TEST
    # rows are materialised before the freeze.
    data_rows = load_split_rows(root, "VALIDATION")
    validation_ids = [row["canonical_sample_id"] for row in data_rows if row.get("model_eligible_for_slope_complete_view")]
    baseline = reconstruct_features(data_rows, validation_ids, {"kind": "baseline"})
    scaler = load_json(root / "datasets/co2/manifests/c_b2_imbalance_calibration/preprocessing_scaler_evidence.json")
    first_raw = np.asarray(baseline["records"][validation_ids[0]]["raw"], dtype=np.float64)
    first_scaled = (first_raw - np.asarray(scaler["mean"], dtype=np.float64)) / np.asarray(scaler["scale"], dtype=np.float64)
    latency = host_latency_sanity(tf, root, first_scaled)
    latency_path = output / "host_latency_evidence.json"
    write_json(latency_path, latency)
    latency_sha = file_sha256(root, f"{ARTIFACT_DIR_REL}/host_latency_evidence.json")

    freeze = build_pre_locked_test_freeze(root, protocol, protocol_sha, robustness_sha, latency_sha, robustness, latency)
    freeze_path = output / "pre_locked_test_candidate_freeze.json"
    write_json(freeze_path, freeze)
    validate_pre_locked_test_freeze(root, freeze, protocol, robustness, latency)

    # Exactly one authorized, unperturbed LOCKED_TEST evaluation follows this
    # validation.  No perturbation branch is passed to this function.
    locked = evaluate_locked_test_once(tf, root, freeze)
    write_json(output / "locked_test_evaluation.json", locked)

    metadata = build_final_candidate_metadata(root, robustness, latency, locked, freeze)
    candidate_dir = root / CANDIDATE_DIR_REL
    candidate_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = candidate_dir / "final_candidate_metadata.json"
    write_json(metadata_path, metadata)

    limitations = {
        "manifest_version": "1.0",
        "phase": "C-B5",
        "warnings": [
            {"code": code, "severity": "WARNING"}
            for code in [
                "SOURCE_TIMEZONE_UNVERIFIED", "GROUP_INDEPENDENCE_NOT_VERIFIABLE", "CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED", "DEVICE_UCI_CADENCE_DOMAIN_GAP", "SAFETY_RULE_CONTRACT_OUT_OF_SCOPE", "SENSOR_HEALTH_CONTRACT_OUT_OF_SCOPE", "MULTISENSOR_RISK_CONTRACT_OUT_OF_SCOPE", "DEFERRED_SHARED_INTEGRATION_UPDATE", "INT8_INPUT_SATURATION_OBSERVED", "HOST_MAC_LATENCY_SANITY_ONLY", "SCD40_DEVICE_DOMAIN_VALIDATION_NOT_YET_COMPLETE",
            ]
        ],
        "blockers": [],
        "production_artifacts_modified": False,
        "cross_sensor_changes": False,
        "robustness_classification": "ROBUSTNESS_DIAGNOSTIC_ONLY_UNDER_OFFLINE_TECHNICAL_STRESS",
    }
    write_json(output / "exceptions_and_limitations.json", limitations)
    write_json(output / "run_environment.json", {"manifest_version": "1.0", "phase": "C-B5", "python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "model_sha256": hashlib.sha256((root / MODEL_REL).read_bytes()).hexdigest(), "scaler_fingerprint": SCALER_FINGERPRINT, "locked_test_evaluation_count": 1})
    # Final closure is generated last and excludes its own lock/checksum files.
    final_lock = write_final_lock(root, metadata)

    summary = {
        "status": "PASS_WITH_WARNINGS",
        "protocol_sha256": protocol_sha,
        "robustness_results_sha256": robustness_sha,
        "latency_sha256": latency_sha,
        "pre_locked_test_freeze_sha256": freeze["freeze_sha256"],
        "locked_test_evaluation_count": locked["evaluation_count"],
        "final_lock_sha256": final_lock["final_lock_sha256"],
        "final_candidate_status": metadata["candidate_status"],
        "locked_test_used_during_robustness": robustness["locked_test_used"],
    }
    write_json(output / "c_b5_run_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
