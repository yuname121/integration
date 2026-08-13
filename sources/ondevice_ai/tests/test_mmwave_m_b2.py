# SafeNest mmWave Track — Phase M-B2 Class-Imbalance Strategy Comparison Test Suite

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from mmwave_m_b2_imbalance import (
    STRATEGIES,
    build_multiclass_focal_loss,
    build_oversampling_plan,
    compute_one_vs_rest_false_positives,
    compute_subject_level_diagnostics,
    compute_train_class_weights,
    rank_imbalance_strategies,
)
from mmwave_phase_b_access import LOCKED_TEST_AccessError, PhaseBAccessGuard
from validate_mmwave_m_b2 import MB2ValidationError, validate_m_b2_artifacts


class TestMMWaveMB2(unittest.TestCase):

    def setUp(self) -> None:
        self.guard = PhaseBAccessGuard(root_dir=ROOT_DIR)
        self.manifest_dir = ROOT_DIR / "datasets/mmwave/manifests/M-B2_class_imbalance"

    def test_strategies_count_and_structure(self) -> None:
        self.assertEqual(len(STRATEGIES), 4)
        strategy_ids = [s["strategy_id"] for s in STRATEGIES]
        expected_ids = [
            "M-B2_CE_UNWEIGHTED",
            "M-B2_CE_CLASS_WEIGHT",
            "M-B2_CE_RANDOM_OVERSAMPLE",
            "M-B2_FOCAL_CLASS_ALPHA",
        ]
        self.assertEqual(strategy_ids, expected_ids)

    def test_train_class_weights_calculation(self) -> None:
        labels = [0] * 100 + [1] * 50 + [2] * 150
        weights = compute_train_class_weights(labels)
        self.assertAlmostEqual(weights[0], 300.0 / (3 * 100), places=5)
        self.assertAlmostEqual(weights[1], 300.0 / (3 * 50), places=5)
        self.assertAlmostEqual(weights[2], 300.0 / (3 * 150), places=5)

    def test_oversampling_plan_generation(self) -> None:
        train_windows = [
            {"canonical_sample_index": i, "window_id": f"w_{i}", "subject_id": f"sub_{i%2}", "recording_id": "r1", "safenest_label_id": i % 3}
            for i in range(10)
        ]
        indices, plan_records = build_oversampling_plan(train_windows, seed=42)
        self.assertEqual(len(indices), 12)
        self.assertEqual(len(plan_records), 10)

    def test_multiclass_focal_loss(self) -> None:
        alpha_weights = {0: 1.0, 1: 2.0, 2: 1.0}
        focal_loss_fn = build_multiclass_focal_loss(alpha_weights, gamma=2.0)
        y_true = tf.constant([1], dtype=tf.int32)
        y_pred = tf.constant([[0.1, 0.8, 0.1]], dtype=tf.float32)
        loss_val = focal_loss_fn(y_true, y_pred).numpy()
        self.assertGreaterEqual(loss_val, 0.0)
        self.assertTrue(np.isfinite(loss_val))

    def test_one_vs_rest_false_positives(self) -> None:
        val_true = np.array([0, 0, 1, 1, 2, 2])
        val_pred = np.array([0, 1, 1, 1, 2, 0])
        metrics = compute_one_vs_rest_false_positives(val_true, val_pred)
        self.assertIn("NORMAL", metrics)
        self.assertIn("RAPID_OR_ABNORMAL", metrics)
        self.assertIn("APNEA", metrics)
        self.assertIn("fpr", metrics["NORMAL"])

    def test_subject_level_diagnostics(self) -> None:
        val_windows = [
            {"subject_id": "sub_A", "safenest_label_id": 0},
            {"subject_id": "sub_A", "safenest_label_id": 2},
            {"subject_id": "sub_B", "safenest_label_id": 1},
        ]
        val_preds = np.array([0, 2, 1])
        subj_diag = compute_subject_level_diagnostics(val_windows, val_preds)
        self.assertEqual(subj_diag["summary_across_subjects"]["subject_count"], 2)
        self.assertEqual(subj_diag["summary_across_subjects"]["mean_accuracy"], 1.0)

    def test_locked_test_model_selection_prohibited(self) -> None:
        with self.assertRaises(LOCKED_TEST_AccessError):
            self.guard.get_model_selection_dataset("LOCKED_TEST")

    def test_standalone_m_b2_validator_clean(self) -> None:
        if self.manifest_dir.is_dir():
            res = validate_m_b2_artifacts(root_dir=ROOT_DIR, manifest_dir=self.manifest_dir)
            self.assertTrue(res["validation_success"])
            self.assertEqual(res["m_b2_gate_status"], "PASS_WITH_WARNINGS")
            self.assertEqual(res["m_b3_entry_status"], "READY_WITH_CONDITIONS")
            self.assertTrue(res["independently_measured"]["class_distribution_recomputed"])

    def test_tie_boundary_contract_and_field_consistency(self) -> None:
        base_a = {
            "strategy_id": "M-B2_CE_CLASS_WEIGHT",
            "is_class_collapsed": False,
            "macro_f1": 0.663341,
            "min_per_class_recall": 0.400000,
            "macro_precision": 0.672957,
            "macro_fpr": 0.154284,
        }
        base_b = {
            "strategy_id": "M-B2_CE_UNWEIGHTED",
            "is_class_collapsed": False,
            "macro_f1": 0.663341 + 0.5e-5,
            "min_per_class_recall": 0.400000,
            "macro_precision": 0.672957,
            "macro_fpr": 0.154284,
        }
        ranked = rank_imbalance_strategies([base_a, base_b], eps=1e-5)
        self.assertEqual(ranked[0]["strategy_id"], "M-B2_CE_UNWEIGHTED")

        base_b_high = dict(base_b, macro_f1=0.663341 + 2e-5)
        ranked_high = rank_imbalance_strategies([base_a, base_b_high], eps=1e-5)
        self.assertEqual(ranked_high[0]["strategy_id"], "M-B2_CE_UNWEIGHTED")

        base_b_bound = dict(base_b, macro_f1=0.663341 + 1e-5)
        ranked_bound = rank_imbalance_strategies([base_a, base_b_bound], eps=1e-5)
        self.assertEqual(ranked_bound[0]["strategy_id"], "M-B2_CE_UNWEIGHTED")

        ranked_self = rank_imbalance_strategies([base_a, base_a], eps=1e-5)
        self.assertEqual(len(ranked_self), 2)

    def test_support_zero_confusion_preservation(self) -> None:
        val_w = [
            {"subject_id": "sub_01", "safenest_label_id": 0, "window_id": "w1"},
            {"subject_id": "sub_01", "safenest_label_id": 0, "window_id": "w2"},
        ]
        preds = np.array([0, 1])
        diag = compute_subject_level_diagnostics(val_w, preds)
        sub_cm = diag["per_subject"]["sub_01"]["class_metrics"]
        rapid_cm = sub_cm["RAPID_OR_ABNORMAL"]
        self.assertIsInstance(rapid_cm, dict)
        self.assertEqual(rapid_cm["support"], 0)
        self.assertEqual(rapid_cm["tp"], 0)
        self.assertEqual(rapid_cm["fp"], 1)
        self.assertEqual(rapid_cm["tn"], 1)
        self.assertEqual(rapid_cm["fn"], 0)
        self.assertEqual(rapid_cm["recall"], "NOT_DEFINED_NO_SUPPORT")

    def test_validator_fails_on_corrupted_subject_fp(self) -> None:
        if not self.manifest_dir.is_dir(): return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)
            subj_file = tmp_manifest / "subject_level_metrics.json"
            data = json.loads(subj_file.read_text(encoding="utf-8"))
            first_sid = sorted(data["subject_diagnostics"]["M-B2_CE_UNWEIGHTED"]["per_subject"].keys())[0]
            data["subject_diagnostics"]["M-B2_CE_UNWEIGHTED"]["per_subject"][first_sid]["apnea_fp"] += 99
            subj_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            with self.assertRaises(MB2ValidationError):
                validate_m_b2_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_corrupted_subject_tn(self) -> None:
        if not self.manifest_dir.is_dir(): return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)
            subj_file = tmp_manifest / "subject_level_metrics.json"
            data = json.loads(subj_file.read_text(encoding="utf-8"))
            first_sid = sorted(data["subject_diagnostics"]["M-B2_CE_UNWEIGHTED"]["per_subject"].keys())[0]
            first_c = list(data["subject_diagnostics"]["M-B2_CE_UNWEIGHTED"]["per_subject"][first_sid]["class_metrics"].keys())[0]
            if isinstance(data["subject_diagnostics"]["M-B2_CE_UNWEIGHTED"]["per_subject"][first_sid]["class_metrics"][first_c], dict):
                data["subject_diagnostics"]["M-B2_CE_UNWEIGHTED"]["per_subject"][first_sid]["class_metrics"][first_c]["tn"] += 99
            subj_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            with self.assertRaises(MB2ValidationError):
                validate_m_b2_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_corrupted_subject_prediction_distribution(self) -> None:
        if not self.manifest_dir.is_dir(): return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)
            subj_file = tmp_manifest / "subject_level_metrics.json"
            data = json.loads(subj_file.read_text(encoding="utf-8"))
            first_sid = sorted(data["subject_diagnostics"]["M-B2_CE_UNWEIGHTED"]["per_subject"].keys())[0]
            data["subject_diagnostics"]["M-B2_CE_UNWEIGHTED"]["per_subject"][first_sid]["prediction_distribution"]["NORMAL"] += 99
            subj_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            with self.assertRaises(MB2ValidationError):
                validate_m_b2_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_corrupted_oversampling_canonical_sample_index(self) -> None:
        if not self.manifest_dir.is_dir(): return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)
            ovs_file = tmp_manifest / "oversampling_plan.jsonl"
            lines_data = [json.loads(l) for l in ovs_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            lines_data[0]["canonical_sample_index"] += 9999
            ovs_file.write_text("\n".join(json.dumps(l) for l in lines_data) + "\n", encoding="utf-8")
            with self.assertRaises(MB2ValidationError):
                validate_m_b2_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_corrupted_oversampling_subject_id(self) -> None:
        if not self.manifest_dir.is_dir(): return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)
            ovs_file = tmp_manifest / "oversampling_plan.jsonl"
            lines_data = [json.loads(l) for l in ovs_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            lines_data[0]["subject_id"] = "INVALID_SUBJECT_CORRUPTED"
            ovs_file.write_text("\n".join(json.dumps(l) for l in lines_data) + "\n", encoding="utf-8")
            with self.assertRaises(MB2ValidationError):
                validate_m_b2_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_invalid_effective_multiplicity(self) -> None:
        if not self.manifest_dir.is_dir(): return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)
            ovs_file = tmp_manifest / "oversampling_plan.jsonl"
            lines_data = [json.loads(l) for l in ovs_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            lines_data[0]["effective_multiplicity"] = lines_data[0]["additional_duplicate_count"] + 99
            ovs_file.write_text("\n".join(json.dumps(l) for l in lines_data) + "\n", encoding="utf-8")
            with self.assertRaises(MB2ValidationError):
                validate_m_b2_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_m_b1_initial_weight_mismatch(self) -> None:
        if not self.manifest_dir.is_dir(): return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)
            tr_file = tmp_manifest / "training_runs.json"
            data = json.loads(tr_file.read_text(encoding="utf-8"))
            data["training_runs"]["M-B2_CE_UNWEIGHTED"]["initial_weights_sha256"] = "0" * 64
            tr_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            with self.assertRaises(MB2ValidationError):
                validate_m_b2_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_m_b1_validation_prediction_mismatch(self) -> None:
        if not self.manifest_dir.is_dir(): return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)
            preds_file = tmp_manifest / "validation_predictions.npz"
            preds = dict(np.load(preds_file))
            preds["M-B2_CE_UNWEIGHTED"][0] = (preds["M-B2_CE_UNWEIGHTED"][0] + 1) % 3
            np.savez(preds_file, **preds)
            with self.assertRaises(MB2ValidationError):
                validate_m_b2_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_detects_corrupted_strategy_selection(self) -> None:
        if not self.manifest_dir.is_dir(): return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)
            sel_file = tmp_manifest / "selected_imbalance_strategy.json"
            data = json.loads(sel_file.read_text(encoding="utf-8"))
            data["selected_strategy_id"] = "INVALID_CORRUPTED_STRATEGY"
            sel_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            with self.assertRaises(MB2ValidationError):
                validate_m_b2_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_malformed_checksum_line(self) -> None:
        if not self.manifest_dir.is_dir(): return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)
            chk_file = tmp_manifest / "checksums.sha256"
            chk_file_content = chk_file.read_text(encoding="utf-8")
            chk_file.write_text("malformed_line_without_space\n" + chk_file_content, encoding="utf-8")
            with self.assertRaises(MB2ValidationError):
                validate_m_b2_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_unpinned_environment(self) -> None:
        if not self.manifest_dir.is_dir(): return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)
            env_file = tmp_manifest / "run_environment.json"
            data = json.loads(env_file.read_text(encoding="utf-8"))
            data["numpy_version"] = "2.0.2"
            env_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            with self.assertRaises(MB2ValidationError):
                validate_m_b2_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_stale_upstream_identity(self) -> None:
        if not self.manifest_dir.is_dir(): return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)
            id_file = tmp_manifest / "input_identity.json"
            data = json.loads(id_file.read_text(encoding="utf-8"))
            data["inputs"][0]["measured_sha256"] = "0" * 64
            id_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            with self.assertRaises(MB2ValidationError):
                validate_m_b2_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_oversampling_majority_duplication_rejected(self) -> None:
        if not self.manifest_dir.is_dir(): return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)
            ovs_file = tmp_manifest / "oversampling_plan.jsonl"
            lines_data = [json.loads(l) for l in ovs_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            for l in lines_data:
                if l.get("class_id") == 2:
                    l["additional_duplicate_count"] = 5
                    l["effective_multiplicity"] = 6
                    break
            ovs_file.write_text("\n".join(json.dumps(l) for l in lines_data) + "\n", encoding="utf-8")
            with self.assertRaises(MB2ValidationError):
                validate_m_b2_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_baseline_drift(self) -> None:
        if not self.manifest_dir.is_dir(): return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)
            tr_file = tmp_manifest / "training_runs.json"
            data = json.loads(tr_file.read_text(encoding="utf-8"))
            data["training_runs"]["M-B2_CE_UNWEIGHTED"]["final_weights_sha256"] = "0" * 64
            tr_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            with self.assertRaises(MB2ValidationError):
                validate_m_b2_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)


if __name__ == "__main__":
    unittest.main()
