# SafeNest mmWave Track — Phase M-B4 Multi-Seed Stability Test Suite

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from mmwave_m_b4_multiseed import SEEDS, rank_multiseed_architectures
from validate_mmwave_m_b4 import MB4ValidationError, validate_m_b4_artifacts


class TestMMWaveMB4(unittest.TestCase):

    def setUp(self) -> None:
        self.manifest_dir = ROOT_DIR / "datasets/mmwave/manifests/M-B4_multiseed_stability"

    def test_multiseed_ranking_rules(self) -> None:
        cand_a = {
            "architecture_id": "M-B3_CONV1D_GAP_BASELINE",
            "collapsed_seed_count": 0,
            "macro_f1": {"worst_seed_val": 0.650000, "mean": 0.660000, "std": 0.010000},
            "min_per_class_recall": {"worst_seed_val": 0.400000},
            "total_params": 9315,
            "strict_int8_bytes": 22080,
        }
        cand_b = {
            "architecture_id": "M-B3_SEPARABLECONV1D_GAP",
            "collapsed_seed_count": 0,
            "macro_f1": {"worst_seed_val": 0.450000, "mean": 0.470000, "std": 0.015000},
            "min_per_class_recall": {"worst_seed_val": 0.300000},
            "total_params": 3258,
            "strict_int8_bytes": 19072,
        }

        ranked = rank_multiseed_architectures([cand_a, cand_b], eps=1e-5)
        self.assertEqual(ranked[0]["architecture_id"], "M-B3_CONV1D_GAP_BASELINE")

    def test_collapsed_seed_exclusion(self) -> None:
        cand_collapsed = {
            "architecture_id": "M-B3_CONV1D_GAP_BASELINE",
            "collapsed_seed_count": 1,  # 1 collapsed seed!
            "macro_f1": {"worst_seed_val": 0.800000, "mean": 0.850000, "std": 0.005000},
            "min_per_class_recall": {"worst_seed_val": 0.000000},
            "total_params": 9315,
            "strict_int8_bytes": 22080,
        }
        cand_stable = {
            "architecture_id": "M-B3_SEPARABLECONV1D_GAP",
            "collapsed_seed_count": 0,
            "macro_f1": {"worst_seed_val": 0.450000, "mean": 0.470000, "std": 0.015000},
            "min_per_class_recall": {"worst_seed_val": 0.300000},
            "total_params": 3258,
            "strict_int8_bytes": 19072,
        }

        ranked = rank_multiseed_architectures([cand_collapsed, cand_stable], eps=1e-5)
        # Collapsed model must be excluded from top slot
        self.assertEqual(ranked[0]["architecture_id"], "M-B3_SEPARABLECONV1D_GAP")

    def test_standalone_m_b4_validator_clean(self) -> None:
        if self.manifest_dir.is_dir():
            res = validate_m_b4_artifacts(root_dir=ROOT_DIR, manifest_dir=self.manifest_dir)
            self.assertTrue(res["validation_success"])
            self.assertEqual(res["m_b4_gate_status"], "PASS_WITH_WARNINGS")
            self.assertEqual(res["m_b5_entry_status"], "READY_WITH_CONDITIONS")

    def test_validator_fails_on_missing_seed(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            plan_file = tmp_manifest / "seed_plan.json"
            data = json.loads(plan_file.read_text(encoding="utf-8"))
            data["seeds"] = [42, 43]  # Missing seed 44!
            plan_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(MB4ValidationError):
                validate_m_b4_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_extra_seed(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            plan_file = tmp_manifest / "seed_plan.json"
            data = json.loads(plan_file.read_text(encoding="utf-8"))
            data["seeds"] = [42, 43, 44, 45]  # Extra seed 45!
            plan_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(MB4ValidationError):
                validate_m_b4_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_corrupted_primary_finalist(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            pri_file = tmp_manifest / "primary_float_finalist.json"
            data = json.loads(pri_file.read_text(encoding="utf-8"))
            data["primary_stable_float_finalist"] = "WRONG_ARCHITECTURE_WINNER"
            pri_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(MB4ValidationError):
                validate_m_b4_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_stale_upstream_identity_sha(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            id_file = tmp_manifest / "input_identity.json"
            data = json.loads(id_file.read_text(encoding="utf-8"))
            data["inputs"][0]["measured_sha256"] = "0" * 64
            id_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(MB4ValidationError):
                validate_m_b4_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_corrupted_weight_npz(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            npz_file = tmp_manifest / "seed_weights.npz"
            orig_npz = np.load(npz_file)
            corrupted_dict = {k: orig_npz[k].copy() for k in orig_npz.files}
            corrupted_dict["M-B3_CONV1D_GAP_BASELINE_seed_42_layer_weight_0"] += 1.0
            np.savez_compressed(npz_file, **corrupted_dict)

            with self.assertRaises(MB4ValidationError):
                validate_m_b4_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_malformed_checksum_line(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            chk_file = tmp_manifest / "checksums.sha256"
            content = chk_file.read_text(encoding="utf-8")
            chk_file.write_text("malformed_line_without_space\n" + content, encoding="utf-8")

            with self.assertRaises(MB4ValidationError):
                validate_m_b4_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_corrupted_per_seed_accuracy(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            ps_file = tmp_manifest / "per_seed_results.json"
            data = json.loads(ps_file.read_text(encoding="utf-8"))
            data["per_seed_results"]["M-B3_CONV1D_GAP_BASELINE_seed_42"]["val_accuracy"] = 0.999999
            ps_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(MB4ValidationError):
                validate_m_b4_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_corrupted_multi_seed_mean(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            ms_file = tmp_manifest / "multi_seed_results.json"
            data = json.loads(ms_file.read_text(encoding="utf-8"))
            data["multi_seed_results"][0]["macro_f1"]["mean"] = 0.999999
            ms_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(MB4ValidationError):
                validate_m_b4_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_false_backup_architecture(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            bk_file = tmp_manifest / "backup_architecture.json"
            data = json.loads(bk_file.read_text(encoding="utf-8"))
            data["backup_architecture_id"] = "M-B3_SEPARABLECONV1D_GAP"  # False backup because it collapsed!
            bk_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(MB4ValidationError):
                validate_m_b4_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_seed43_initial_weight_mismatch(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            tr_file = tmp_manifest / "training_runs.json"
            data = json.loads(tr_file.read_text(encoding="utf-8"))
            data["training_runs"]["M-B3_CONV1D_GAP_BASELINE_seed_43"]["initial_weights_sha256"] = "0" * 64
            tr_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(MB4ValidationError):
                validate_m_b4_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_scipy_version_mismatch(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            env_file = tmp_manifest / "run_environment.json"
            data = json.loads(env_file.read_text(encoding="utf-8"))
            data["scipy_version"] = "0.0.0"
            env_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(MB4ValidationError):
                validate_m_b4_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_seed42_prediction_mismatch(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            preds_file = tmp_manifest / "validation_predictions.npz"
            orig_preds = np.load(preds_file)
            corrupted_dict = {k: orig_preds[k].copy() for k in orig_preds.files}
            corrupted_dict["M-B3_CONV1D_GAP_BASELINE_seed_42"][0] = (corrupted_dict["M-B3_CONV1D_GAP_BASELINE_seed_42"][0] + 1) % 3
            np.savez_compressed(preds_file, **corrupted_dict)

            with self.assertRaises(MB4ValidationError):
                validate_m_b4_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_subject_class_fp_or_tn_corruption(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            subj_file = tmp_manifest / "subject_level_seed_metrics.json"
            data = json.loads(subj_file.read_text(encoding="utf-8"))
            run_item = data["subject_diagnostics_by_run"]["M-B3_CONV1D_GAP_BASELINE_seed_42"]
            first_sid = list(run_item["per_subject"].keys())[0]
            run_item["per_subject"][first_sid]["class_metrics"]["NORMAL"]["fp"] += 1
            subj_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(MB4ValidationError):
                validate_m_b4_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_seed42_initial_weight_mismatch(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            tr_file = tmp_manifest / "training_runs.json"
            data = json.loads(tr_file.read_text(encoding="utf-8"))
            data["training_runs"]["M-B3_CONV1D_GAP_BASELINE_seed_42"]["initial_weights_sha256"] = "0" * 64
            tr_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(MB4ValidationError):
                validate_m_b4_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)


if __name__ == "__main__":
    unittest.main()
