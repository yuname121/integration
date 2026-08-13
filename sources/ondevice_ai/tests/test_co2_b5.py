"""Focused, deterministic C-B5 contract tests."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from datasets.co2.b5_robustness import (
    ARTIFACT_DIR_REL,
    INT8_MODEL_SHA256,
    PROTOCOL_ID,
    THRESHOLD,
    _scenario_grid,
    build_protocol,
    evaluate_locked_test_once,
    load_eligible_ids,
    load_split_rows,
    make_scenario_rows,
    reconstruct_features,
    stable_sha256,
    LockedTestAuthorizationError,
)
from datasets.co2.tflite_equivalence import quantize_int8_input


ROOT = Path(__file__).resolve().parent.parent


def test_protocol_serialization_is_deterministic() -> None:
    first = build_protocol()
    second = build_protocol()
    assert first == second
    assert stable_sha256(first) == stable_sha256(second)
    assert first["protocol_id"] == PROTOCOL_ID
    assert first["locked_test_used"] is False
    assert first["candidate"]["threshold"] == THRESHOLD


def test_protocol_grid_has_baseline_and_registered_levels() -> None:
    grid = _scenario_grid(build_protocol())
    assert len(grid) == 25
    assert grid[0] == {"kind": "baseline"}
    assert sum(row["kind"] == "co2_offset_ppm" for row in grid) == 6
    assert sum(row["kind"] == "co2_linear_drift_ppm_per_min" for row in grid) == 6
    assert sum(row["kind"] == "humidity_noise_sigma_rh" for row in grid) == 3
    assert sum(row["kind"] == "missing_row" for row in grid) == 3
    assert sum(row["kind"] == "stale_history_seconds" for row in grid) == 3
    assert sum(row["kind"] == "timestamp_jitter_seconds" for row in grid) == 3


def test_unperturbed_causal_reconstruction_matches_canonical_validation() -> None:
    rows = load_split_rows(ROOT, "VALIDATION")
    ids = load_eligible_ids(ROOT, "VALIDATION")
    result = reconstruct_features(rows, ids, {"kind": "baseline"})
    assert all(result["records"][sid]["available"] for sid in ids)
    assert np.isclose(result["records"][ids[0]]["raw"][3], 8.516666666666652)
    assert result["records"][ids[0]]["status"] == "FEATURE_AVAILABLE"


def test_missing_rows_are_unavailable_not_imputed() -> None:
    rows = load_split_rows(ROOT, "VALIDATION")
    ids = load_eligible_ids(ROOT, "VALIDATION")
    scenario = {"kind": "missing_row", "pattern": build_protocol()["scenarios"]["missing_row"]["patterns"][1]}
    perturbed, meta = make_scenario_rows(rows, scenario)
    result = reconstruct_features(rows, ids, scenario)
    assert meta["rows_removed"] == 3
    assert any(not record["available"] for record in result["records"].values())
    # The source list is shorter; no synthetic zero/mean slope is inserted.
    assert all(record.get("status") != "IMPUTED" for record in result["records"].values())
    assert len(perturbed) == len(rows) - 3


def test_timestamp_jitter_preserves_strict_order_and_is_seeded() -> None:
    rows = load_split_rows(ROOT, "VALIDATION")
    scenario = {"kind": "timestamp_jitter_seconds", "level": 10, "seed": 20260810}
    first, meta_first = make_scenario_rows(rows, scenario)
    second, meta_second = make_scenario_rows(rows, scenario)
    assert [row["_dt"] for row in first] == [row["_dt"] for row in second]
    assert meta_first == meta_second
    assert all(a["_dt"] < b["_dt"] for a, b in zip(first, first[1:]))


def test_humidity_noise_is_deterministic_and_bounded() -> None:
    rows = load_split_rows(ROOT, "VALIDATION")
    scenario = {"kind": "humidity_noise_sigma_rh", "level": 5.0, "seed": 20260812}
    first, _ = make_scenario_rows(rows, scenario)
    second, _ = make_scenario_rows(rows, scenario)
    assert [row["humidity"] for row in first] == [row["humidity"] for row in second]
    assert all(0.0 <= float(row["humidity"]) <= 100.0 for row in first)


def test_int8_saturation_is_counted_before_clipping() -> None:
    values = np.asarray([[0.0, 0.0, 0.0, 5.0]], dtype=np.float64)
    quantized, flags, overflow = quantize_int8_input(values, 0.03529411926865578, 0)
    assert quantized.dtype == np.int8
    assert flags.tolist() == [[False, False, False, True]]
    assert overflow[0, 3] > 0.0


def test_generated_artifacts_keep_candidate_identity_and_zero_robustness_test_access() -> None:
    results = (ROOT / ARTIFACT_DIR_REL / "robustness_results.json").read_text(encoding="utf-8")
    import json

    payload = json.loads(results)
    assert payload["candidate_model_sha256"] == INT8_MODEL_SHA256
    assert payload["locked_test_used"] is False
    assert payload["locked_test_predictions"] == 0
    assert payload["locked_test_metrics"] == 0


def test_direct_locked_test_guard_rejects_tampered_freeze() -> None:
    import json

    freeze = json.loads((ROOT / ARTIFACT_DIR_REL / "pre_locked_test_candidate_freeze.json").read_text(encoding="utf-8"))
    tampered = copy.deepcopy(freeze)
    tampered["candidate"]["threshold"] = 0.5
    with pytest.raises(LockedTestAuthorizationError, match="CHECKSUM|IDENTITY"):
        evaluate_locked_test_once(object(), ROOT, tampered)


def test_direct_locked_test_guard_rejects_nonzero_prior_access() -> None:
    import json

    freeze = json.loads((ROOT / ARTIFACT_DIR_REL / "pre_locked_test_candidate_freeze.json").read_text(encoding="utf-8"))
    tampered = copy.deepcopy(freeze)
    tampered["locked_test_prior_access"]["predictions"] = 1
    tampered["freeze_sha256"] = stable_sha256({k: v for k, v in tampered.items() if k != "freeze_sha256"})
    with pytest.raises(LockedTestAuthorizationError, match="PRIOR_ACCESS"):
        evaluate_locked_test_once(object(), ROOT, tampered)


def test_completed_locked_test_artifact_blocks_double_evaluation() -> None:
    import json

    freeze = json.loads((ROOT / ARTIFACT_DIR_REL / "pre_locked_test_candidate_freeze.json").read_text(encoding="utf-8"))
    with pytest.raises(LockedTestAuthorizationError, match="DOUBLE_EVALUATION"):
        evaluate_locked_test_once(object(), ROOT, freeze)
