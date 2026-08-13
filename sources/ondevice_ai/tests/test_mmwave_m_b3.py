# SafeNest mmWave Track — Phase M-B3 TinyML Architecture Comparison Test Suite

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

from mmwave_m_b3_architecture import (
    ARCHITECTURES,
    build_architecture_a,
    build_architecture_b,
    build_architecture_c,
    convert_to_tflite_strict_int8,
    rank_architectures,
)
from validate_mmwave_m_b3 import MB3ValidationError, validate_m_b3_artifacts


class TestMMWaveMB3(unittest.TestCase):

    def setUp(self) -> None:
        self.manifest_dir = ROOT_DIR / "datasets/mmwave/manifests/M-B3_architecture_comparison"

    def test_architectures_count_and_structure(self) -> None:
        self.assertEqual(len(ARCHITECTURES), 3)
        arch_ids = [a["architecture_id"] for a in ARCHITECTURES]
        expected_ids = [
            "M-B3_CONV1D_GAP_BASELINE",
            "M-B3_SEPARABLECONV1D_GAP",
            "M-B3_CONV1D_BILSTM",
        ]
        self.assertEqual(arch_ids, expected_ids)

    def test_architecture_builders_instantiation(self) -> None:
        m_a = build_architecture_a(input_shape=(300, 1))
        m_b = build_architecture_b(input_shape=(300, 1))
        m_c = build_architecture_c(input_shape=(300, 1))

        self.assertEqual(m_a.count_params(), 9315)
        self.assertEqual(m_b.count_params(), 3258)
        self.assertEqual(m_c.count_params(), 19747)

        # Output shape should be (None, 3) for 3-class classification
        self.assertEqual(m_a.output_shape, (None, 3))
        self.assertEqual(m_b.output_shape, (None, 3))
        self.assertEqual(m_c.output_shape, (None, 3))

    def test_bilstm_strict_int8_unsupported_detection(self) -> None:
        m_c = build_architecture_c(input_shape=(300, 1))

        def dummy_rep_gen():
            for _ in range(5):
                yield [np.random.randn(1, 300, 1).astype(np.float32)]

        success, tflite_bytes, status_code, err_msg = convert_to_tflite_strict_int8(m_c, dummy_rep_gen)
        self.assertFalse(success)
        self.assertEqual(status_code, "STRICT_INT8_UNSUPPORTED")
        self.assertIsNotNone(err_msg)

    def test_rank_architectures_tie_boundary(self) -> None:
        cand_a = {
            "architecture_id": "M-B3_CONV1D_GAP_BASELINE",
            "deployment_eligibility": "DEPLOYMENT_ELIGIBLE_SINGLE_SEED",
            "float_macro_f1": 0.663708,
            "float_min_per_class_recall": 0.400000,
            "float_apnea_recall": 1.000000,
            "total_params": 9315,
            "strict_int8_bytes": 15920,
        }
        cand_b = {
            "architecture_id": "M-B3_SEPARABLECONV1D_GAP",
            "deployment_eligibility": "DEPLOYMENT_ELIGIBLE_SINGLE_SEED",
            "float_macro_f1": 0.663708 - 0.5e-5,  # Within 1e-5 tie tolerance
            "float_min_per_class_recall": 0.400000,
            "float_apnea_recall": 1.000000,
            "total_params": 3307,  # Smaller params -> Should win tie-breaker!
            "strict_int8_bytes": 9328,
        }

        ranked = rank_architectures([cand_a, cand_b], eps=1e-5)
        self.assertEqual(ranked[0]["architecture_id"], "M-B3_SEPARABLECONV1D_GAP")

    def test_standalone_m_b3_validator_clean(self) -> None:
        if self.manifest_dir.is_dir():
            res = validate_m_b3_artifacts(root_dir=ROOT_DIR, manifest_dir=self.manifest_dir)
            self.assertTrue(res["validation_success"])
            self.assertEqual(res["m_b3_gate_status"], "PASS_WITH_WARNINGS")
            self.assertEqual(res["m_b4_entry_status"], "READY_WITH_CONDITIONS")

    def test_validator_detects_baseline_drift(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            tr_file = tmp_manifest / "training_runs.json"
            data = json.loads(tr_file.read_text(encoding="utf-8"))
            data["training_runs"]["M-B3_CONV1D_GAP_BASELINE"]["final_weights_sha256"] = "0" * 64
            tr_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(MB3ValidationError):
                validate_m_b3_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_corrupted_shortlist(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            sel_file = tmp_manifest / "selected_architecture_shortlist.json"
            data = json.loads(sel_file.read_text(encoding="utf-8"))
            data["selected_architecture_shortlist"] = ["INVALID_CORRUPTED_SHORTLIST"]
            sel_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(MB3ValidationError):
                validate_m_b3_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_corrupted_bilstm_classification(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            conv_file = tmp_manifest / "conversion_compatibility.json"
            data = json.loads(conv_file.read_text(encoding="utf-8"))
            data["conversion_compatibility"]["M-B3_CONV1D_BILSTM"]["strict_int8"]["success"] = True
            data["conversion_compatibility"]["M-B3_CONV1D_BILSTM"]["deployment_eligibility"] = "DEPLOYMENT_ELIGIBLE_SINGLE_SEED"
            conv_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(MB3ValidationError):
                validate_m_b3_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_corrupted_checksum_line(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            chk_file = tmp_manifest / "checksums.sha256"
            content = chk_file.read_text(encoding="utf-8")
            chk_file.write_text("malformed_line_without_space\n" + content, encoding="utf-8")

            with self.assertRaises(MB3ValidationError):
                validate_m_b3_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_corrupted_tflite_sha(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            tfl_file = tmp_manifest / "tflite_artifact_manifest.json"
            data = json.loads(tfl_file.read_text(encoding="utf-8"))
            data["tflite_artifacts"][0]["sha256"] = "0" * 64
            tfl_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(MB3ValidationError):
                validate_m_b3_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

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

            with self.assertRaises(MB3ValidationError):
                validate_m_b3_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)

    def test_validator_fails_on_corrupted_weight_npz(self) -> None:
        if not self.manifest_dir.is_dir():
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir)
            shutil.copytree(self.manifest_dir, tmp_manifest, dirs_exist_ok=True)

            npz_file = tmp_manifest / "architecture_weights.npz"
            orig_npz = np.load(npz_file)
            corrupted_dict = {k: orig_npz[k].copy() for k in orig_npz.files}
            corrupted_dict["M-B3_CONV1D_GAP_BASELINE_layer_weight_0"] += 1.0
            np.savez_compressed(npz_file, **corrupted_dict)

            with self.assertRaises(MB3ValidationError):
                validate_m_b3_artifacts(root_dir=ROOT_DIR, manifest_dir=tmp_manifest)


if __name__ == "__main__":
    unittest.main()
