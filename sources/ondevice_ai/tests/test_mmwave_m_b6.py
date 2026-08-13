# SafeNest mmWave Track — Phase M-B6 Focused Unit Tests

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from validate_mmwave_m_b6 import MB6ValidationError, validate_m_b6_artifacts


class TestMB6StageEquivalence(unittest.TestCase):
    """Focused negative & integrity unit tests for Phase M-B6 stage-equivalence artifacts."""

    def setUp(self):
        self.root_dir = ROOT_DIR
        self.manifest_dir = ROOT_DIR / "datasets/mmwave/manifests/M-B6_stage_equivalence"
        self.temp_dir = Path(tempfile.mkdtemp())
        self.temp_manifest_dir = self.temp_dir / "M-B6_stage_equivalence"
        shutil.copytree(self.manifest_dir, self.temp_manifest_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def update_checksums(self):
        """Helper to recompute checksums.sha256 after intentional JSON mutation."""
        checksums_file = self.temp_manifest_dir / "checksums.sha256"
        lines = []
        for f in sorted(self.temp_manifest_dir.glob("*")):
            if f.name != "checksums.sha256" and f.is_file():
                h = hashlib.sha256(f.read_bytes()).hexdigest()
                lines.append(f"{h}  {f.name}")
        checksums_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_validator_passes_on_unmodified_artifacts(self):
        """Clean baseline validator execution must pass."""
        res = validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.manifest_dir)
        self.assertTrue(res["validation_success"])
        self.assertEqual(res["m_b6_gate_status"], "PASS_WITH_WARNINGS")

    def test_validator_fails_on_stale_input_identity_sha(self):
        """Stale or corrupted upstream SHA in input_identity.json must raise validation error."""
        p = self.temp_manifest_dir / "input_identity.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data["inputs"][0]["measured_sha256"] = "0" * 64
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.update_checksums()

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("Upstream identity SHA mismatch", str(ctx.exception))

    def test_validator_fails_on_keras_prediction_corruption(self):
        """Corrupted Keras predictions in keras_predictions.npz must raise validation error."""
        p = self.temp_manifest_dir / "keras_predictions.npz"
        npz = dict(np.load(p))
        first_k = list(npz.keys())[0]
        npz[first_k][0] = (npz[first_k][0] + 1) % 3
        np.savez_compressed(p, **npz)
        self.update_checksums()

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("Keras prediction vector mismatch", str(ctx.exception))

    def test_validator_fails_on_float_tflite_sha_corruption(self):
        """Corrupted Float TFLite file SHA in manifest must raise validation error."""
        p = self.temp_manifest_dir / "stage_artifact_manifest.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        k = [key for key in data["artifacts"] if "stage_b" in key][0]
        data["artifacts"][k]["sha256"] = "0" * 64
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.update_checksums()

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("Float TFLite SHA mismatch", str(ctx.exception))

    def test_validator_fails_on_float_tflite_dtype_mismatch(self):
        """Non-float32 dtype for Float TFLite must raise validation error."""
        p = self.temp_manifest_dir / "stage_artifact_manifest.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        k = [key for key in data["artifacts"] if "stage_b" in key][0]
        data["artifacts"][k]["input_dtype"] = "int8"
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.update_checksums()

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("Stage B manifest vs actual dtype mismatch", str(ctx.exception))

    def test_validator_fails_on_int8_sha_corruption(self):
        """Corrupted Strict INT8 file SHA in manifest must raise validation error."""
        p = self.temp_manifest_dir / "stage_artifact_manifest.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        k = [key for key in data["artifacts"] if "stage_c" in key][0]
        data["artifacts"][k]["sha256"] = "0" * 64
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.update_checksums()

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("Strict INT8 SHA mismatch", str(ctx.exception))

    def test_validator_fails_on_int8_input_dtype_mismatch(self):
        """Non-int8 input dtype for Strict INT8 must raise validation error."""
        p = self.temp_manifest_dir / "stage_artifact_manifest.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        k = [key for key in data["artifacts"] if "stage_c" in key][0]
        data["artifacts"][k]["input_dtype"] = "float32"
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.update_checksums()

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("Stage C manifest vs actual dtype mismatch", str(ctx.exception))

    def test_validator_fails_on_select_tf_ops_detected(self):
        """Presence of Select TF Ops in Strict INT8 manifest must raise validation error."""
        p = self.temp_manifest_dir / "stage_artifact_manifest.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        k = [key for key in data["artifacts"] if "stage_c" in key][0]
        data["artifacts"][k]["select_tf_ops_count"] = 1
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.update_checksums()

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("Select TF Ops detected", str(ctx.exception))

    def test_validator_fails_on_top1_agreement_corruption(self):
        """Corrupted top1_agreement metric in pairwise equivalence JSON must raise error."""
        p = self.temp_manifest_dir / "pairwise_equivalence_metrics.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        first_k = list(data["pairwise_equivalence"].keys())[0]
        data["pairwise_equivalence"][first_k]["a_to_c"]["top1_agreement"] = 0.0
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.update_checksums()

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("Pairwise field 'a_to_c.top1_agreement' mismatch", str(ctx.exception))

    def test_validator_fails_on_subject_level_tp_corruption(self):
        """Subject-level per-class metric corruption must trigger validation failure."""
        p = self.temp_manifest_dir / "subject_level_stage_metrics.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        first_k = list(data["subject_level_stage_metrics"].keys())[0]
        first_subj = list(data["subject_level_stage_metrics"][first_k]["stage_a"]["per_subject"].keys())[0]
        data["subject_level_stage_metrics"][first_k]["stage_a"]["per_subject"][first_subj]["class_metrics"]["NORMAL"]["tp"] += 999
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.update_checksums()

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("class_metrics field 'NORMAL.tp' mismatch", str(ctx.exception))

    def test_validator_fails_on_locked_test_access_violation(self):
        """Non-zero performance access to LOCKED_TEST must raise error."""
        p = self.temp_manifest_dir / "locked_test_access_audit.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data["performance_access_attempts"] = 1
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.update_checksums()

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("LOCKED_TEST_ACCESS_VIOLATION", str(ctx.exception))

    def test_validator_fails_on_path_traversal_checksum(self):
        """Path traversal attempt in checksums.sha256 must raise validation error."""
        p = self.temp_manifest_dir / "checksums.sha256"
        content = p.read_text(encoding="utf-8")
        corrupted = content.replace("m_b6_summary.json", "../m_b6_summary.json")
        p.write_text(corrupted, encoding="utf-8")

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("Path traversal", str(ctx.exception))

    def test_validator_fails_on_malformed_checksum_digest(self):
        """Malformed SHA digest in checksums.sha256 must raise validation error."""
        p = self.temp_manifest_dir / "checksums.sha256"
        content = p.read_text(encoding="utf-8")
        corrupted = content.replace(content[:64], "INVALID_SHA_DIGEST")
        p.write_text(corrupted, encoding="utf-8")

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("Invalid SHA-256 digest format", str(ctx.exception))

    def test_validator_fails_on_corrupted_per_seed_stage_macro_f1(self):
        """Corrupted per_seed_stage_metrics Macro F1 must raise validation error."""
        p = self.temp_manifest_dir / "per_seed_stage_metrics.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        first_k = list(data["per_seed_stage_metrics"].keys())[0]
        data["per_seed_stage_metrics"][first_k]["stage_c_int8_tflite"]["macro_f1"] = 0.999999
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.update_checksums()

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("per_seed_stage_metrics 'stage_c_int8_tflite.macro_f1' mismatch", str(ctx.exception))

    def test_validator_fails_on_corrupted_per_class_tp(self):
        """Corrupted per-class TP in per_seed_stage_metrics must raise validation error."""
        p = self.temp_manifest_dir / "per_seed_stage_metrics.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        first_k = list(data["per_seed_stage_metrics"].keys())[0]
        data["per_seed_stage_metrics"][first_k]["stage_a_float_keras"]["class_metrics"]["NORMAL"]["tp"] += 10
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.update_checksums()

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("class_metrics.NORMAL.tp' mismatch", str(ctx.exception))

    def test_validator_fails_on_corrupted_cross_seed_worst_f1(self):
        """Corrupted worst_macro_f1_degradation in cross_seed_equivalence_summary must raise error."""
        p = self.temp_manifest_dir / "cross_seed_equivalence_summary.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data["cross_seed_a_to_c"]["worst_macro_f1_degradation"] = 0.888888
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.update_checksums()

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("cross_seed_equivalence_summary field 'cross_seed_a_to_c.worst_macro_f1_degradation' mismatch", str(ctx.exception))

    def test_validator_fails_on_corrupted_cross_seed_worst_seed(self):
        """Corrupted worst_seed in cross_seed_equivalence_summary must raise error."""
        p = self.temp_manifest_dir / "cross_seed_equivalence_summary.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data["cross_seed_a_to_c"]["worst_seed"] = 99
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.update_checksums()

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("cross_seed_equivalence_summary field 'cross_seed_a_to_c.worst_seed' mismatch", str(ctx.exception))

    def test_validator_fails_on_false_class_collapse_transition(self):
        """Corrupted boolean flag in class_collapse_transition_audit must raise error."""
        p = self.temp_manifest_dir / "class_collapse_transition_audit.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        first_k = list(data["class_collapse_transitions"].keys())[0]
        data["class_collapse_transitions"][first_k]["new_collapse_a_to_c"] = True
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.update_checksums()

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("class_collapse_transition_audit field 'new_collapse_a_to_c' mismatch", str(ctx.exception))

    def test_validator_fails_on_corrupted_input_saturation(self):
        """Corrupted input_saturation_ratio in quantization_diagnostics must raise error."""
        p = self.temp_manifest_dir / "quantization_diagnostics.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        first_k = list(data["quantization_diagnostics"].keys())[0]
        data["quantization_diagnostics"][first_k]["input_saturation_ratio"] = 0.50
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.update_checksums()

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("quantization_diagnostics field 'input_saturation_ratio' mismatch", str(ctx.exception))

    def test_validator_fails_on_corrupted_quantization_scale(self):
        """Corrupted input_scale in quantization_diagnostics must raise error."""
        p = self.temp_manifest_dir / "quantization_diagnostics.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        first_k = list(data["quantization_diagnostics"].keys())[0]
        data["quantization_diagnostics"][first_k]["input_scale"] = 999.0
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.update_checksums()

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("quantization_diagnostics field 'input_scale' mismatch", str(ctx.exception))

    def test_validator_fails_on_m_b5_selected_int8_sha_mismatch(self):
        """SHA mismatch vs M-B5 selected INT8 artifact must raise validation error."""
        p = self.temp_manifest_dir / "stage_artifact_manifest.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        k = [key for key in data["artifacts"] if "stage_c" in key][0]
        data["artifacts"][k]["sha256"] = "1" * 64
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.update_checksums()

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("Strict INT8 SHA mismatch", str(ctx.exception))

    def test_validator_fails_on_m_b5_reused_flag_false(self):
        """False m_b5_selected_int8_reused flag must raise validation error."""
        p = self.temp_manifest_dir / "stage_artifact_manifest.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        k = [key for key in data["artifacts"] if "stage_c" in key][0]
        data["artifacts"][k]["m_b5_selected_int8_reused"] = False
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.update_checksums()

        with self.assertRaises(MB6ValidationError) as ctx:
            validate_m_b6_artifacts(root_dir=self.root_dir, manifest_dir=self.temp_manifest_dir)
        self.assertIn("m_b5_selected_int8_reused flag must be True", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
