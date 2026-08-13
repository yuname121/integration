#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_check_mmwave_candidate.py
Unit test suite for SafeNest V6 Candidate Quality Check & Defect Detector (Priority 5).
"""

import os
import sys
import json
import hashlib
import tempfile
import unittest
from pathlib import Path

# Add canonical repository root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.validate_metadata import build_mmwave_candidate_metadata
from scripts.check_mmwave_candidate import (
    check_candidate_quality,
    load_acceptance_thresholds,
    DefectItem,
)


class TestCandidateDefectDetector(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="test_candidate_qa_")
        self.base_dir = Path(self.temp_dir.name)

        # Create dummy candidate model file inside models/mmwave under base_dir
        self.models_dir = self.base_dir / "models/mmwave"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cand_path = self.models_dir / "test_candidate.tflite"
        self.cand_bytes = b"DUMMY_TFLITE_MODEL_BYTES_FOR_QA_TESTING"
        with open(self.cand_path, "wb") as f:
            f.write(self.cand_bytes)
        self.actual_sha = hashlib.sha256(self.cand_bytes).hexdigest()

        # Create valid metadata dict using official builder
        self.valid_meta = build_mmwave_candidate_metadata(
            candidate_tflite_path=self.cand_path,
            seed=42,
            epochs=2,
            batch_size=32,
            learning_rate=0.001,
            mean=0.172122,
            std=1.717154,
            float_keras_eval={"accuracy": 1.0, "macro_f1": 1.0},
            float_tflite_eval={"accuracy": 1.0, "macro_f1": 1.0},
            int8_tflite_eval={
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "apnea_window_miss_rate": 0.0,
                "class_collapse": False,
                "input_saturation_ratio": 0.0,
                "evaluated_sample_count": 468,
                "prediction_distribution": {
                    "NORMAL": 187,
                    "RAPID_OR_ABNORMAL": 239,
                    "APNEA": 42,
                },
            },
        )
        self.valid_meta["path"] = f"models/mmwave/{self.cand_path.name}"

        self.meta_path = self.models_dir / "test_candidate_metadata.json"
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.valid_meta, f, indent=2)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_valid_candidate_passes(self):
        """Test that a valid candidate artifact and metadata pass quality check."""
        passed, defects, report = check_candidate_quality(
            candidate_path=self.cand_path,
            metadata_path=self.meta_path,
            model_root=self.base_dir,
        )
        self.assertTrue(passed, f"Valid candidate should pass QA, got defects: {defects}")
        self.assertEqual(len(defects), 0)
        self.assertEqual(report["status"], "PASSED")

    def test_02_candidate_file_missing(self):
        """Test detection of missing candidate model file."""
        non_existent = self.base_dir / "non_existent.tflite"
        passed, defects, report = check_candidate_quality(
            candidate_path=non_existent,
            metadata_path=self.meta_path,
        )
        self.assertFalse(passed)
        self.assertTrue(any(d.code == "CANDIDATE_FILE_MISSING" for d in defects))

    def test_03_metadata_file_missing(self):
        """Test detection of missing candidate metadata JSON file."""
        non_existent = self.base_dir / "non_existent_meta.json"
        passed, defects, report = check_candidate_quality(
            candidate_path=self.cand_path,
            metadata_path=non_existent,
        )
        self.assertFalse(passed)
        self.assertTrue(any(d.code == "METADATA_FILE_MISSING" for d in defects))

    def test_04_malformed_metadata_json(self):
        """Test detection of malformed metadata JSON."""
        bad_meta_path = self.base_dir / "bad_meta.json"
        with open(bad_meta_path, "w", encoding="utf-8") as f:
            f.write("{ INVALID JSON TEXT }")

        passed, defects, report = check_candidate_quality(
            candidate_path=self.cand_path,
            metadata_path=bad_meta_path,
        )
        self.assertFalse(passed)
        self.assertTrue(any(d.code == "METADATA_JSON_PARSE_ERROR" for d in defects))

    def test_05_sha256_mismatch(self):
        """Test detection of candidate SHA-256 mismatch."""
        meta = json.loads(json.dumps(self.valid_meta))
        meta["sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
        bad_sha_meta_path = self.models_dir / "bad_sha_meta.json"
        with open(bad_sha_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        passed, defects, report = check_candidate_quality(
            candidate_path=self.cand_path,
            metadata_path=bad_sha_meta_path,
        )
        self.assertFalse(passed)
        self.assertTrue(any(d.code == "MODEL_METADATA_SHA_MISMATCH" for d in defects))

    def test_06_class_collapse_detection(self):
        """Test detection of class collapse (all predictions assigned to single class)."""
        meta = json.loads(json.dumps(self.valid_meta))
        meta["stage_evaluations"]["int8_tflite"]["prediction_distribution"] = {
            "NORMAL": 468,
            "RAPID_OR_ABNORMAL": 0,
            "APNEA": 0,
        }
        meta["stage_evaluations"]["int8_tflite"]["class_collapse"] = True

        collapse_meta_path = self.models_dir / "collapse_meta.json"
        with open(collapse_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        passed, defects, report = check_candidate_quality(
            candidate_path=self.cand_path,
            metadata_path=collapse_meta_path,
        )
        self.assertFalse(passed)
        codes = [d.code for d in defects]
        self.assertIn("CLASS_COLLAPSE_ALL_SAME_PRED", codes)

    def test_07_zero_apnea_and_rapid_recall(self):
        """Test detection of zero APNEA recall and zero RAPID recall."""
        meta = json.loads(json.dumps(self.valid_meta))
        meta["stage_evaluations"]["int8_tflite"]["prediction_distribution"] = {
            "NORMAL": 468,
            "RAPID_OR_ABNORMAL": 0,
            "APNEA": 0,
        }
        meta["stage_evaluations"]["int8_tflite"]["apnea_window_miss_rate"] = 1.0

        zero_recall_path = self.models_dir / "zero_recall.json"
        with open(zero_recall_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        passed, defects, report = check_candidate_quality(
            candidate_path=self.cand_path,
            metadata_path=zero_recall_path,
        )
        self.assertFalse(passed)
        codes = [d.code for d in defects]
        self.assertIn("ZERO_APNEA_RECALL", codes)
        self.assertIn("ZERO_RAPID_RECALL", codes)

    def test_08_macro_f1_drop_exceeded(self):
        """Test detection of excessive Float-to-INT8 macro F1 degradation (>0.05)."""
        meta = json.loads(json.dumps(self.valid_meta))
        meta["stage_evaluations"]["float_tflite"] = {"accuracy": 0.90, "macro_f1": 0.90}
        meta["stage_evaluations"]["int8_tflite"]["macro_f1"] = 0.80  # Drop = 0.10 > 0.05

        f1_drop_path = self.models_dir / "f1_drop.json"
        with open(f1_drop_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        passed, defects, report = check_candidate_quality(
            candidate_path=self.cand_path,
            metadata_path=f1_drop_path,
        )
        self.assertFalse(passed)
        codes = [d.code for d in defects]
        self.assertIn("INT8_MACRO_F1_DROP_EXCEEDED", codes)

    def test_09_input_saturation_ratio_exceeded(self):
        """Test detection of excessive input saturation ratio (>0.05)."""
        meta = json.loads(json.dumps(self.valid_meta))
        meta["stage_evaluations"]["int8_tflite"]["input_saturation_ratio"] = 0.08  # > 0.05

        sat_path = self.models_dir / "sat_exceeded.json"
        with open(sat_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        passed, defects, report = check_candidate_quality(
            candidate_path=self.cand_path,
            metadata_path=sat_path,
        )
        self.assertFalse(passed)
        codes = [d.code for d in defects]
        self.assertIn("SATURATION_RATIO_EXCEEDED", codes)

    def test_10_scaler_defects(self):
        """Test detection of scaler mean/std defects and invalid source."""
        meta = json.loads(json.dumps(self.valid_meta))
        meta["scaler"] = {
            "method": "z_score",
            "stats_source": "all_data",  # Invalid
            "mean": 0.0,
            "std": 0.0,  # Invalid
        }

        scaler_bad_path = self.models_dir / "scaler_bad.json"
        with open(scaler_bad_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        passed, defects, report = check_candidate_quality(
            candidate_path=self.cand_path,
            metadata_path=scaler_bad_path,
        )
        self.assertFalse(passed)
        codes = [d.code for d in defects]
        self.assertIn("INVALID_SCALER_STD", codes)

    def test_11_multiple_defects_collected(self):
        """Test that multiple independent defects are all collected in a single run."""
        meta = json.loads(json.dumps(self.valid_meta))
        meta["sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
        meta["class_map"] = {"0": "A", "1": "B", "2": "C"}  # Mismatch
        meta["stage_evaluations"]["int8_tflite"]["input_saturation_ratio"] = 0.20  # Exceeded

        multi_path = self.models_dir / "multi_defects.json"
        with open(multi_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        passed, defects, report = check_candidate_quality(
            candidate_path=self.cand_path,
            metadata_path=multi_path,
        )
        self.assertFalse(passed)
        self.assertGreaterEqual(len(defects), 3)


if __name__ == "__main__":
    unittest.main()
