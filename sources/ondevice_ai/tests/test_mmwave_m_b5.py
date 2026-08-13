# SafeNest mmWave Track — Phase M-B5 Focused Unit Tests

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from mmwave_m_b5_calibration import (
    build_profile_d_distribution_aware,
    build_profile_d_feature_matrix,
    compute_positive_recall_degradation,
    detect_new_quantization_collapse,
    rank_cross_seed_calibration_profiles,
)
from validate_mmwave_m_b5 import MB5ValidationError, validate_m_b5_artifacts


def _base_rank_row(profile_id: str, **overrides):
    row = {
        "profile_id": profile_id,
        "eligible": True,
        "conversion_success_count": 3,
        "strict_int8_eligible": True,
        "new_class_collapse_count": 0,
        "worst_positive_macro_f1_degradation": 0.01,
        "worst_positive_recall_degradation": 0.08,
        "min_top1_agreement": 0.95,
        "max_dequantized_output_mae": 0.01,
        "max_input_saturation_ratio": 0.0,
        "max_output_endpoint_ratio": 0.0,
    }
    row.update(overrides)
    return row


class TestMB5RepresentativeCalibration(unittest.TestCase):
    """Focused negative & integrity unit tests for Phase M-B5 representative calibration artifacts."""

    def setUp(self):
        self.root_dir = ROOT_DIR
        self.manifest_dir = ROOT_DIR / "datasets/mmwave/manifests/M-B5_representative_calibration"
        self.temp_dir = Path(tempfile.mkdtemp())
        self.temp_manifest_dir = self.temp_dir / "M-B5_representative_calibration"
        shutil.copytree(self.manifest_dir, self.temp_manifest_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _rewrite_checksum_for(self, filename: str) -> None:
        target = self.temp_manifest_dir / filename
        digest = __import__("hashlib").sha256(target.read_bytes()).hexdigest()
        lines = []
        for line in (self.temp_manifest_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            _d, name = line.split(maxsplit=1)
            if name.strip() == filename:
                lines.append(f"{digest}  {filename}")
            else:
                lines.append(line)
        (self.temp_manifest_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_validator_passes_on_unmodified_artifacts(self):
        res = validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.manifest_dir)
        self.assertTrue(res["validation_success"])
        self.assertEqual(res["m_b5_gate_status"], "PASS_WITH_WARNINGS")
        self.assertIn(
            res["independently_measured"]["selected_calibration_profile"],
            {
                "M-B5_CAL_TRAIN_ORDER_120",
                "M-B5_CAL_RANDOM_PROPORTIONAL_120",
                "M-B5_CAL_CLASS_BALANCED_120",
                "M-B5_CAL_DISTRIBUTION_AWARE_120",
            },
        )

    def test_validator_fails_on_duplicate_representative_index(self):
        p = self.temp_manifest_dir / "representative_dataset_indices.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data["profile_indices"]["M-B5_CAL_TRAIN_ORDER_120"][1] = data["profile_indices"]["M-B5_CAL_TRAIN_ORDER_120"][0]
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("Duplicate index", str(ctx.exception))

    def test_validator_fails_on_wrong_profile_size(self):
        p = self.temp_manifest_dir / "representative_dataset_indices.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data["profile_indices"]["M-B5_CAL_TRAIN_ORDER_120"].pop()
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("index count mismatch", str(ctx.exception))

    def test_validator_fails_on_profile_nondeterministic_indices(self):
        p = self.temp_manifest_dir / "representative_dataset_indices.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        a, b = (
            data["profile_indices"]["M-B5_CAL_RANDOM_PROPORTIONAL_120"][0],
            data["profile_indices"]["M-B5_CAL_RANDOM_PROPORTIONAL_120"][1],
        )
        data["profile_indices"]["M-B5_CAL_RANDOM_PROPORTIONAL_120"][0] = b
        data["profile_indices"]["M-B5_CAL_RANDOM_PROPORTIONAL_120"][1] = a
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("M-B5_PROFILE_NONDETERMINISTIC", str(ctx.exception))

    def test_validator_fails_on_corrupted_distribution_aware_index(self):
        p = self.temp_manifest_dir / "representative_dataset_indices.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        idx = data["profile_indices"]["M-B5_CAL_DISTRIBUTION_AWARE_120"]
        idx[0], idx[1] = idx[1], idx[0]
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("M-B5_PROFILE_NONDETERMINISTIC", str(ctx.exception))

    def test_validator_fails_on_wrong_lying_sitting_posture_encoding(self):
        p = self.temp_manifest_dir / "representative_profile_contract.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data["distribution_aware_profile"]["posture_vocabulary"] = ["supine", "left", "right"]
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._rewrite_checksum_for("representative_profile_contract.json")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertTrue(
            "posture vocabulary" in str(ctx.exception).lower()
            or "supine/left/right" in str(ctx.exception).lower()
        )

    def test_validator_fails_on_wrong_source_condition_encoding(self):
        p = self.temp_manifest_dir / "representative_profile_contract.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data["distribution_aware_profile"]["source_test_condition_vocabulary"] = ["normal", "rapid", "apnea_proxy"]
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._rewrite_checksum_for("representative_profile_contract.json")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("source_test_condition vocabulary mismatch", str(ctx.exception))

    def test_profile_d_subject_cap_corruption_changes_selection(self):
        windows = [
            {"subject_id": f"s{i % 10}", "safenest_label_id": i % 3, "safenest_label": ["NORMAL", "RAPID_OR_ABNORMAL", "APNEA"][i % 3], "posture": "Lying" if i % 2 == 0 else "Sitting", "source_test_condition": "Rest" if i % 2 == 0 else "Post-exercise"}
            for i in range(40)
        ]
        x = np.random.RandomState(0).randn(40, 250).astype(np.float32)
        idx, meta = build_profile_d_distribution_aware(windows, x, sample_count=20)
        self.assertEqual(len(idx), 20)
        self.assertEqual(meta["subject_cap_initial"], 2)
        # Corrupt by forcing uncapped behavior via mutated subject ids uniqueness
        windows_uncapped = copy.deepcopy(windows)
        for i, w in enumerate(windows_uncapped):
            w["subject_id"] = f"unique_{i}"
        idx2, meta2 = build_profile_d_distribution_aware(windows_uncapped, x, sample_count=20)
        self.assertNotEqual(idx, idx2)
        self.assertEqual(meta2["subject_cap_final_state"], "MAX_2")

    def test_validator_fails_on_locked_test_access_violation(self):
        p = self.temp_manifest_dir / "locked_test_access_audit.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data["performance_access_attempts"] = 1
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("LOCKED_TEST_ACCESS_VIOLATION", str(ctx.exception))

    def test_validator_fails_on_select_tf_ops_detected(self):
        p = self.temp_manifest_dir / "tflite_artifact_manifest.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        k = list(data["tflite_artifacts"].keys())[0]
        data["tflite_artifacts"][k]["select_tf_ops_count"] = 1
        data["tflite_artifacts"][k]["op_types"] = list(data["tflite_artifacts"][k].get("op_types", [])) + ["FlexAdd"]
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        msg = str(ctx.exception)
        self.assertTrue("SELECT_TF_OPS" in msg or "operator inventory" in msg.lower() or "Flex" in msg)

    def test_validator_fails_on_actual_tflite_input_dtype_mismatch(self):
        p = self.temp_manifest_dir / "tflite_artifact_manifest.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        k = list(data["tflite_artifacts"].keys())[0]
        data["tflite_artifacts"][k]["input_dtype"] = "float32"
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("dtype", str(ctx.exception).lower())

    def test_validator_fails_on_actual_tflite_output_dtype_mismatch(self):
        p = self.temp_manifest_dir / "tflite_artifact_manifest.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        k = list(data["tflite_artifacts"].keys())[0]
        data["tflite_artifacts"][k]["output_dtype"] = "float32"
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("dtype", str(ctx.exception).lower())

    def test_validator_fails_on_operator_inventory_mismatch(self):
        p = self.temp_manifest_dir / "tflite_artifact_manifest.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        k = list(data["tflite_artifacts"].keys())[0]
        ops = list(data["tflite_artifacts"][k].get("op_types", ["CONV_2D"]))
        ops.append("FAKE_OP")
        data["tflite_artifacts"][k]["op_types"] = ops
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("operator inventory mismatch", str(ctx.exception).lower())

    def test_validator_fails_on_tflite_file_size_mismatch(self):
        p = self.temp_manifest_dir / "tflite_artifact_manifest.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        k = list(data["tflite_artifacts"].keys())[0]
        data["tflite_artifacts"][k]["bytes"] += 100
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertTrue("bytes" in str(ctx.exception).lower() or "SHA" in str(ctx.exception))

    def test_validator_fails_on_tflite_sha256_mismatch(self):
        p = self.temp_manifest_dir / "tflite_artifact_manifest.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        k = list(data["tflite_artifacts"].keys())[0]
        data["tflite_artifacts"][k]["sha256"] = "0" * 64
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertTrue("SHA" in str(ctx.exception) or "bytes" in str(ctx.exception).lower())

    def test_validator_fails_on_per_class_recall_degradation_corruption(self):
        p = self.temp_manifest_dir / "calibration_results.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        k = list(data["calibration_results"].keys())[0]
        data["calibration_results"][k]["quantization_diagnostics"]["per_class_positive_recall_degradation"]["APNEA"] = 0.999999
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._rewrite_checksum_for("calibration_results.json")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("recall-degradation", str(ctx.exception).lower())

    def test_validator_fails_on_max_recall_degradation_corruption(self):
        p = self.temp_manifest_dir / "calibration_results.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        k = list(data["calibration_results"].keys())[0]
        data["calibration_results"][k]["quantization_diagnostics"]["max_positive_recall_degradation"] = 0.999999
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._rewrite_checksum_for("calibration_results.json")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("Max recall-degradation corruption", str(ctx.exception))

    def test_validator_fails_on_new_collapse_flag_corruption(self):
        p = self.temp_manifest_dir / "calibration_results.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        k = list(data["calibration_results"].keys())[0]
        data["calibration_results"][k]["quantization_diagnostics"]["new_class_collapse"] = True
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._rewrite_checksum_for("calibration_results.json")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("New-collapse flag corruption", str(ctx.exception))

    def test_validator_fails_on_calibration_selection_mismatch(self):
        p = self.temp_manifest_dir / "selected_calibration_profile.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        current = data["selected_calibration_profile"]
        alt = "M-B5_CAL_TRAIN_ORDER_120" if current != "M-B5_CAL_TRAIN_ORDER_120" else "M-B5_CAL_CLASS_BALANCED_120"
        data["selected_calibration_profile"] = alt
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("Calibration profile selection mismatch", str(ctx.exception))

    def test_validator_fails_on_incorrect_selected_profile_ranking_decision(self):
        p = self.temp_manifest_dir / "selected_calibration_profile.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data.setdefault("ranking_decision", {})
        data["ranking_decision"]["deciding_criterion"] = "CRITERION_8_LEXICOGRAPHIC_PROFILE_ID"
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._rewrite_checksum_for("selected_calibration_profile.json")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("ranking decision", str(ctx.exception).lower())

    def test_validator_fails_on_corrupted_deterministic_replay_prediction_vector(self):
        p = self.temp_manifest_dir / "determinism_audit.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        replay = data["selected_profile_three_seed_conversion_replay"]
        seed_key = next(iter(replay["seed_replays"].keys()))
        replay["seed_replays"][seed_key]["checks"]["prediction_vector_equal"] = False
        replay["seed_replays"][seed_key]["functional_equality"] = False
        replay["functional_reproducibility_verified"] = False
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._rewrite_checksum_for("determinism_audit.json")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertTrue("replay" in str(ctx.exception).lower() or "prediction" in str(ctx.exception).lower())

    def test_validator_fails_on_corrupted_replay_quantization_params(self):
        p = self.temp_manifest_dir / "determinism_audit.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        replay = data["selected_profile_three_seed_conversion_replay"]
        seed_key = next(iter(replay["seed_replays"].keys()))
        replay["seed_replays"][seed_key]["checks"]["input_scale_equal"] = False
        replay["seed_replays"][seed_key]["functional_equality"] = False
        replay["functional_reproducibility_verified"] = False
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._rewrite_checksum_for("determinism_audit.json")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertTrue("replay" in str(ctx.exception).lower() or "input_scale" in str(ctx.exception).lower())

    def test_validator_fails_on_path_traversal_checksum(self):
        p = self.temp_manifest_dir / "checksums.sha256"
        content = p.read_text(encoding="utf-8")
        corrupted = content.replace("m_b5_summary.json", "../m_b5_summary.json")
        p.write_text(corrupted, encoding="utf-8")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("Path traversal", str(ctx.exception))

    def test_validator_fails_on_malformed_checksum_digest(self):
        p = self.temp_manifest_dir / "checksums.sha256"
        content = p.read_text(encoding="utf-8")
        corrupted = content.replace(content[:64], "INVALID_SHA_DIGEST")
        p.write_text(corrupted, encoding="utf-8")
        with self.assertRaises(MB5ValidationError) as ctx:
            validate_m_b5_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("Invalid SHA-256 digest format", str(ctx.exception))

    def test_ranking_epsilon_delta_less_than_eps_is_tie(self):
        a = _base_rank_row("M-B5_CAL_CLASS_BALANCED_120", worst_positive_macro_f1_degradation=0.010000)
        b = _base_rank_row("M-B5_CAL_TRAIN_ORDER_120", worst_positive_macro_f1_degradation=0.010000 + 1e-5 - 1e-12)
        ranked = rank_cross_seed_calibration_profiles([a, b], eps=1e-5)
        self.assertEqual(ranked[0]["profile_id"], "M-B5_CAL_TRAIN_ORDER_120")

    def test_ranking_epsilon_delta_equal_eps_is_tie(self):
        a = _base_rank_row("M-B5_CAL_CLASS_BALANCED_120", worst_positive_macro_f1_degradation=0.010000)
        b = _base_rank_row("M-B5_CAL_TRAIN_ORDER_120", worst_positive_macro_f1_degradation=0.010000 + 1e-5)
        ranked = rank_cross_seed_calibration_profiles([a, b], eps=1e-5)
        self.assertEqual(ranked[0]["profile_id"], "M-B5_CAL_TRAIN_ORDER_120")

    def test_ranking_epsilon_delta_greater_than_eps_not_tie(self):
        a = _base_rank_row("M-B5_CAL_CLASS_BALANCED_120", worst_positive_macro_f1_degradation=0.010000)
        b = _base_rank_row("M-B5_CAL_TRAIN_ORDER_120", worst_positive_macro_f1_degradation=0.010000 + 1e-5 + 1e-12)
        ranked = rank_cross_seed_calibration_profiles([a, b], eps=1e-5)
        self.assertEqual(ranked[0]["profile_id"], "M-B5_CAL_CLASS_BALANCED_120")

    def test_positive_recall_degradation_helper(self):
        float_cm = {
            "NORMAL": {"recall": 0.5},
            "RAPID_OR_ABNORMAL": {"recall": 0.4},
            "APNEA": {"recall": 1.0},
        }
        int8_cm = {
            "NORMAL": {"recall": 0.5},
            "RAPID_OR_ABNORMAL": {"recall": 0.3},
            "APNEA": {"recall": 1.0},
        }
        per_class, mx = compute_positive_recall_degradation(float_cm, int8_cm)
        self.assertEqual(per_class["RAPID_OR_ABNORMAL"], 0.1)
        self.assertEqual(mx, 0.1)

    def test_new_collapse_detection_helper(self):
        float_preds = np.array([0, 1, 2, 1, 2])
        int8_preds = np.array([0, 0, 0, 0, 0])
        float_cm = {
            "NORMAL": {"recall": 1.0},
            "RAPID_OR_ABNORMAL": {"recall": 1.0},
            "APNEA": {"recall": 1.0},
        }
        int8_cm = {
            "NORMAL": {"recall": 1.0},
            "RAPID_OR_ABNORMAL": {"recall": 0.0},
            "APNEA": {"recall": 0.0},
        }
        self.assertTrue(detect_new_quantization_collapse(float_preds, float_cm, int8_preds, int8_cm))
        self.assertFalse(detect_new_quantization_collapse(float_preds, float_cm, float_preds, float_cm))

    def test_profile_d_feature_matrix_uses_authoritative_vocab(self):
        windows = [
            {"safenest_label_id": 0, "posture": "Lying", "source_test_condition": "Rest"},
            {"safenest_label_id": 1, "posture": "Sitting", "source_test_condition": "Post-exercise"},
        ]
        x = np.ones((2, 250), dtype=np.float32)
        _matrix, meta = build_profile_d_feature_matrix(windows, x)
        self.assertEqual(meta["posture_vocabulary"], ["Lying", "Sitting"])
        self.assertEqual(meta["source_test_condition_vocabulary"], ["Post-exercise", "Rest"])
        self.assertFalse(meta["snr_available"])


if __name__ == "__main__":
    unittest.main()
