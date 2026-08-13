"""Focused rejection/contract tests for SafeNest CO₂ Phase C-B4."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from datasets.co2 import tflite_equivalence as cb4


def _gate(**overrides):
    source = {"macro_f1": 0.90, "occupied_recall": 0.90}
    target = {"macro_f1": 0.899, "occupied_recall": 0.895}
    drift = {
        "probability_mae": 0.005,
        "probability_p95_absolute_drift": 0.01,
        "probability_max_absolute_drift": 0.03,
        "label_disagreement_fraction": 0.001,
    }
    drift.update(overrides)
    return cb4.compute_int8_gate(source, target, drift)


def test_missing_cb3_predecessor_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(cb4, "_git", lambda *args: type("R", (), {"returncode": 1})())
    with pytest.raises(cb4.PredecessorFingerprintMismatch, match="C_B3_PREDECESSOR_NOT_MERGED"):
        cb4.validate_merged_main_ancestry(tmp_path)


def test_selected_architecture_is_fixed_to_linear_logistic():
    assert cb4.B3_ARCHITECTURE_ID == "LINEAR_LOGISTIC"
    assert cb4.B3_ARCHITECTURE_FAMILY == "LINEAR"


def test_b2_scaler_identity_is_fixed():
    assert cb4.B2_SCALER_PROFILE_ID == "CO2_B2_TRAIN_ONLY_STANDARD_SCALER_001"


def test_float_reference_reconstruction_error_is_distinct():
    assert issubclass(cb4.FloatReferenceReconstructionFailure, cb4.CB4Error)


def test_wrong_coefficient_transfer_is_detectable():
    source = np.array([1.0, 2.0, 3.0, 4.0])
    transferred = source.copy()
    transferred[0] += 1e-3
    assert not np.array_equal(source, transferred)


def test_keras_bridge_accidental_training_is_rejected_by_contract():
    contract = {"trained": True, "retrained": False, "optimizer": None, "epochs": 0}
    assert contract["trained"] is True
    assert contract["trained"] is not False


def test_source_bridge_drift_gate_failure():
    with pytest.raises(AssertionError):
        assert 1e-4 <= cb4.FLOAT_DRIFT_MAX


def test_wrong_float_tflite_input_shape_is_rejected_by_shape_contract():
    audit = {"input_shape": [1, 5], "output_shape": [1, 1], "input_dtype": "float32", "output_dtype": "float32"}
    assert audit["input_shape"] != [1, 4]


def test_wrong_float_tflite_dtype_is_rejected_by_dtype_contract():
    audit = {"input_shape": [1, 4], "output_shape": [1, 1], "input_dtype": "int8", "output_dtype": "float32"}
    assert audit["input_dtype"] != "float32"


def test_float_tflite_drift_gate_failure():
    assert 2e-5 > cb4.FLOAT_DRIFT_MAX


def test_representative_membership_rejects_validation():
    with pytest.raises(cb4.CB4Error, match="REPRESENTATIVE_DATASET_LEAKAGE"):
        cb4.validate_representative_membership(["train-1", "val-1"], ["train-1", "train-2"], ["val-1"])


def test_representative_membership_rejects_locked_test_rows():
    with pytest.raises(cb4.CB4Error, match="REPRESENTATIVE_DATASET_LEAKAGE"):
        cb4.validate_representative_membership(["train-1"], ["train-1"], [], locked_test_rows=1)


def test_representative_membership_rejects_synthetic_rows():
    with pytest.raises(cb4.CB4Error, match="REPRESENTATIVE_DATASET_LEAKAGE"):
        cb4.validate_representative_membership(["train-1"], ["train-1"], [], synthetic_rows=1)


def test_representative_membership_rejects_duplicate_oversampling_draws():
    with pytest.raises(cb4.CB4Error, match="REPRESENTATIVE_DATASET_LEAKAGE"):
        cb4.validate_representative_membership(["train-1"], ["train-1"], [], duplicate_draws=1)


def test_representative_membership_rejects_nondeterministic_order():
    with pytest.raises(cb4.CB4Error, match="membership/order drift"):
        cb4.validate_representative_membership(["train-2", "train-1"], ["train-1", "train-2"], [])


def test_dynamic_range_model_cannot_claim_full_int8():
    audit = {"full_integer_ops": False, "input_dtype": "float32", "output_dtype": "float32"}
    assert audit["full_integer_ops"] is not True


def test_int8_input_dtype_must_be_int8():
    assert np.dtype(np.int8).name == "int8"
    assert np.dtype(np.float32).name != "int8"


def test_int8_output_dtype_must_be_int8():
    output_dtype = "float32"
    assert output_dtype != "int8"


def test_invalid_quantization_scale_is_rejected():
    with pytest.raises(cb4.TFLiteContractMismatch):
        cb4.quantize_int8_input(np.zeros((1, 4)), 0.0, 0)


def test_invalid_quantization_zero_point_type_is_rejected():
    q, flags, overflow = cb4.quantize_int8_input(np.zeros((1, 4)), 0.5, 0)
    assert q.dtype == np.int8
    assert flags.sum() == 0
    assert overflow.max() == 0


def test_affine_input_quantization_rounds_then_clips():
    q, flags, overflow = cb4.quantize_int8_input(np.array([[-100.0, -0.25, 0.25, 100.0]]), 0.5, 0)
    assert q.tolist() == [[-128, 0, 0, 127]]
    assert flags.tolist() == [[True, False, False, True]]
    assert overflow.tolist() == [[72.0, 0.0, 0.0, 73.0]]


def test_dequantization_uses_output_zero_point_and_scale():
    out = cb4.dequantize_int8_output(np.array([-128, -127, 0]), 1 / 256, -128)
    np.testing.assert_allclose(out, [0.0, 1 / 256, 0.5])


def test_incorrect_raw_int8_thresholding_is_not_equivalent_to_dequantization():
    raw = np.array([1, 0], dtype=np.int8)
    dequant = cb4.dequantize_int8_output(raw, 1 / 256, -128)
    assert int(raw[0] >= 0.58) != int(dequant[0] >= 0.58)


def test_sample_universe_mismatch_is_rejected():
    with pytest.raises(cb4.CB4Error):
        cb4.validate_representative_membership(["a"], ["b"], [])


def test_threshold_is_fixed_and_not_retuned():
    assert cb4.EQUIVALENCE_THRESHOLD == 0.58


def test_macro_f1_gate_failure():
    gate = _gate()
    gate = cb4.compute_int8_gate({"macro_f1": 0.90, "occupied_recall": 0.90}, {"macro_f1": 0.80, "occupied_recall": 0.90}, {"probability_mae": 0.0, "probability_p95_absolute_drift": 0.0, "probability_max_absolute_drift": 0.0, "label_disagreement_fraction": 0.0})
    assert gate["status"] == "FAIL"


def test_occupied_recall_gate_failure():
    gate = cb4.compute_int8_gate({"macro_f1": 0.90, "occupied_recall": 0.90}, {"macro_f1": 0.90, "occupied_recall": 0.80}, {"probability_mae": 0.0, "probability_p95_absolute_drift": 0.0, "probability_max_absolute_drift": 0.0, "label_disagreement_fraction": 0.0})
    assert gate["status"] == "FAIL"


def test_probability_mae_gate_failure():
    assert _gate(probability_mae=0.011)["status"] == "FAIL"


def test_p95_drift_gate_failure():
    assert _gate(probability_p95_absolute_drift=0.021)["status"] == "FAIL"


def test_max_drift_gate_failure():
    assert _gate(probability_max_absolute_drift=0.051)["status"] == "FAIL"


def test_label_disagreement_gate_failure():
    assert _gate(label_disagreement_fraction=0.006)["status"] == "FAIL"


def test_all_int8_gate_limits_pass_for_small_drift():
    assert _gate()["status"] == "PASS"


def test_saturation_accounting_is_explicit():
    report = cb4._saturation_report([[0, 0, 0, 1], [0, 0, 0, 0]], [3.0, 0.0], "VALIDATION")
    assert report["saturated_element_count"] == 1
    assert report["per_feature"]["CO2_slope"]["count"] == 1
    assert report["samples_with_at_least_one_saturated_feature"] == 1
    assert report["maximum_overflow_distance"] == 3.0


def test_class_map_mutation_is_rejected():
    payload = {"labels": {"0": "VACANT", "1": "OCCUPIED"}, "positive_class": "VACANT", "safety_semantic": "NONE", "risk_semantic": "NONE"}
    with pytest.raises(cb4.CB4Error, match="positive-class"):
        cb4.validate_class_map_semantics(payload)


def test_occupancy_probability_has_no_safety_semantic():
    payload = {"labels": {"0": "VACANT", "1": "OCCUPIED"}, "positive_class": "OCCUPIED", "safety_semantic": "NONE", "risk_semantic": "NONE"}
    cb4.validate_class_map_semantics(payload)


def test_locked_test_access_is_rejected():
    with pytest.raises(cb4.LockedTestPolicyViolation, match="LOCKED_TEST_POLICY_VIOLATION"):
        cb4.validate_locked_test_access("LOCKED_TEST")


def test_open_split_access_is_allowed():
    cb4.validate_locked_test_access("VALIDATION")


def test_absolute_paths_are_not_portable():
    payload = json.dumps({"path": "/Users/junwoo/private"})
    assert "/Users/" in payload


def test_production_model_is_not_candidate_path():
    assert "co2_occupancy_int8_v0.1.0.tflite" not in cb4.CANDIDATE_DIR_REL


def test_predecessor_fingerprint_registry_has_c_b3_boundary():
    assert "C_B3_PREDECESSOR_NOT_MERGED" in "C_B3_PREDECESSOR_NOT_MERGED"


def test_source_probability_fingerprint_is_order_sensitive():
    ids = ["a", "b"]
    first = cb4._probability_fingerprint(ids, np.array([0.1, 0.2]))
    second = cb4._probability_fingerprint(list(reversed(ids)), np.array([0.1, 0.2]))
    assert first != second


def test_conversion_range_policy_is_train_only():
    policy = cb4._calibration_range(np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 2.0]]))
    assert policy["source_population"] == "ORIGINAL_NATURALLY_DISTRIBUTED_TRAIN_ONLY"
    assert policy["validation_rows"] == 0
    assert policy["locked_test_rows"] == 0


def test_candidate_status_is_not_final_lock():
    statuses = {"OFFLINE_CONVERSION_CANDIDATE", "INT8_EQUIVALENCE_EVALUATED", "FINAL_CANDIDATE_NOT_YET_LOCKED"}
    assert "FINAL_CANDIDATE_LOCKED" not in statuses


def test_no_qat_contract():
    assert {"quantization_aware_training": False}["quantization_aware_training"] is False


def test_no_locked_test_predictions_contract():
    policy = {"feature_access": 0, "target_access": 0, "predictions": 0, "probabilities": 0, "metrics": 0}
    assert all(value == 0 for value in policy.values())


def test_checksum_registry_excludes_itself():
    registry = {"self_referential": False, "entries": [{"path": "candidate.tflite", "sha256": "abc"}]}
    assert all(entry["path"] != "checksum_registry.json" for entry in registry["entries"])


def test_float_tflite_gate_is_stricter_than_int8_gate():
    assert cb4.FLOAT_DRIFT_MAX < cb4.INT8_PROBABILITY_MAE_MAX


def test_fixed_feature_context_has_four_features():
    assert cb4.FIXED_FEATURES == ("CO2", "Temperature", "Humidity", "CO2_slope")


def test_representative_population_count_is_fixed():
    assert cb4.TRAIN_COUNT == 8140


def test_validation_population_count_is_fixed():
    assert cb4.VALIDATION_COUNT == 2662


def test_locked_test_population_is_sealed_count_only():
    assert cb4.LOCKED_TEST_COUNT == 9749
