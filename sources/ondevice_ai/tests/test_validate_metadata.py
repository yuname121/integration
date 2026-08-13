#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_validate_metadata.py
Unit tests for SafeNest V6 mmWave Candidate Metadata Schema Builder & Validator
"""

from __future__ import annotations
import os
import sys
import json
import math
import tempfile
import unittest
from pathlib import Path

# Ensure canonical repository root is in python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.validate_metadata import (
    build_mmwave_candidate_metadata,
    validate_mmwave_candidate_metadata,
    save_candidate_metadata_atomically,
)
from scripts.evaluate_mmwave import calculate_sha256


class TestValidateMetadata(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

        # Create models/mmwave subfolder inside temp_path
        self.models_dir = self.tmp_path / "models/mmwave"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.dummy_model_file = self.models_dir / "dummy_candidate.tflite"
        with open(self.dummy_model_file, "wb") as f:
            f.write(b"SafeNest_TFLite_Dummy_Content_12345")
        self.dummy_sha = calculate_sha256(self.dummy_model_file)

        self.valid_eval = {
            "total_samples": 468,
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "apnea_window_miss_rate": 0.0,
            "class_collapse": False,
            "input_saturation_ratio": 0.0,
            "false_alarm_per_hour": None,
            "false_alarm_status": "NOT_COMPUTABLE",
            "false_alarm_reason": "CONTINUOUS_SESSION_TIMELINE_MISSING",
            "prediction_distribution": {
                "NORMAL": 187,
                "RAPID_OR_ABNORMAL": 239,
                "APNEA": 42,
            },
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def _get_valid_metadata(self) -> dict:
        return build_mmwave_candidate_metadata(
            candidate_tflite_path=self.dummy_model_file,
            seed=42,
            epochs=25,
            batch_size=32,
            learning_rate=0.001,
            mean=0.172122,
            std=1.717154,
            float_keras_eval=self.valid_eval,
            float_tflite_eval=self.valid_eval,
            int8_tflite_eval=self.valid_eval,
            created_at="2026-08-06T05:35:00Z",
        )

    def test_01_valid_metadata_passes(self):
        """1. Complete, valid metadata object passes validation"""
        meta = self._get_valid_metadata()
        self.assertTrue(validate_mmwave_candidate_metadata(meta, model_root=self.tmp_path))

    def test_02_missing_required_top_level_fields(self):
        """2. Missing required top-level fields (sha256, stage_evaluations, scaler) raises ValueError"""
        for req_field in ["sha256", "stage_evaluations", "scaler", "project", "seed"]:
            meta = self._get_valid_metadata()
            del meta[req_field]
            with self.assertRaises(ValueError) as ctx:
                validate_mmwave_candidate_metadata(meta)
            self.assertIn(req_field, str(ctx.exception))

    def test_03_missing_nested_fields(self):
        """3. Missing nested fields (prediction_distribution, std) raises ValueError"""
        meta = self._get_valid_metadata()
        del meta["stage_evaluations"]["int8_tflite"]["prediction_distribution"]
        with self.assertRaises(ValueError) as ctx:
            validate_mmwave_candidate_metadata(meta)
        self.assertIn("prediction_distribution", str(ctx.exception))

        meta2 = self._get_valid_metadata()
        del meta2["scaler"]["std"]
        with self.assertRaises(ValueError) as ctx2:
            validate_mmwave_candidate_metadata(meta2)
        self.assertIn("std", str(ctx2.exception))

    def test_04_invalid_types(self):
        """4. Invalid field types raise ValueError"""
        meta = self._get_valid_metadata()
        meta["seed"] = "42"  # String instead of int
        with self.assertRaises(ValueError):
            validate_mmwave_candidate_metadata(meta)

        meta = self._get_valid_metadata()
        meta["stage_evaluations"]["int8_tflite"]["accuracy"] = "1.0"  # String float
        with self.assertRaises(ValueError):
            validate_mmwave_candidate_metadata(meta)

        meta = self._get_valid_metadata()
        meta["stage_evaluations"]["int8_tflite"]["class_collapse"] = "false"  # String bool
        with self.assertRaises(ValueError):
            validate_mmwave_candidate_metadata(meta)

        meta = self._get_valid_metadata()
        meta["stage_evaluations"]["int8_tflite"]["prediction_distribution"]["NORMAL"] = 187.0  # Float int
        with self.assertRaises(ValueError):
            validate_mmwave_candidate_metadata(meta)

    def test_05_invalid_ranges(self):
        """5. Invalid value ranges raise ValueError"""
        meta = self._get_valid_metadata()
        meta["stage_evaluations"]["int8_tflite"]["accuracy"] = 1.1  # > 1.0
        with self.assertRaises(ValueError):
            validate_mmwave_candidate_metadata(meta)

        meta = self._get_valid_metadata()
        meta["stage_evaluations"]["int8_tflite"]["macro_f1"] = -0.1  # < 0.0
        with self.assertRaises(ValueError):
            validate_mmwave_candidate_metadata(meta)

        meta = self._get_valid_metadata()
        meta["scaler"]["std"] = 0.0  # std <= 0
        with self.assertRaises(ValueError):
            validate_mmwave_candidate_metadata(meta)

        meta = self._get_valid_metadata()
        meta["stage_evaluations"]["int8_tflite"]["input_saturation_ratio"] = 2.0  # > 1.0
        with self.assertRaises(ValueError):
            validate_mmwave_candidate_metadata(meta)

    def test_06_invalid_sha(self):
        """6. Invalid SHA or mismatch with model file raises ValueError"""
        meta = self._get_valid_metadata()
        meta["sha256"] = "a" * 64  # Wrong hash
        with self.assertRaises(ValueError) as ctx:
            validate_mmwave_candidate_metadata(meta, model_root=self.tmp_path)
        self.assertIn("mismatch", str(ctx.exception).lower())

    def test_07_prediction_distribution_sum_mismatch(self):
        """7. Prediction distribution sum mismatch with total_samples raises ValueError"""
        meta = self._get_valid_metadata()
        meta["stage_evaluations"]["int8_tflite"]["total_samples"] = 468
        meta["stage_evaluations"]["int8_tflite"]["prediction_distribution"]["NORMAL"] = 100  # Sum = 100+239+42 = 381 != 468
        with self.assertRaises(ValueError) as ctx:
            validate_mmwave_candidate_metadata(meta)
        self.assertIn("prediction_distribution", str(ctx.exception))

    def test_08_nan_and_infinity(self):
        """8. NaN and Infinity values fail validation or serialization"""
        meta = self._get_valid_metadata()
        meta["scaler"]["mean"] = float("nan")
        with self.assertRaises(ValueError):
            validate_mmwave_candidate_metadata(meta)

        meta = self._get_valid_metadata()
        meta["learning_rate"] = float("inf")
        with self.assertRaises(ValueError):
            validate_mmwave_candidate_metadata(meta)

        # Verify atomic saver rejects allow_nan=False
        meta_nan = self._get_valid_metadata()
        meta_nan["scaler"]["mean"] = math.nan
        out_json = self.tmp_path / "test_nan.json"
        with self.assertRaises(ValueError):
            save_candidate_metadata_atomically(meta_nan, out_json)

    def test_09_false_alarm_status_consistency(self):
        """9. False alarm status consistency checks"""
        meta = self._get_valid_metadata()
        meta["stage_evaluations"]["int8_tflite"]["false_alarm_per_hour"] = 0.5  # Non-null prohibited in Phase 1
        with self.assertRaises(ValueError):
            validate_mmwave_candidate_metadata(meta)

        meta = self._get_valid_metadata()
        meta["stage_evaluations"]["int8_tflite"]["false_alarm_status"] = "PASSED"  # Wrong status
        with self.assertRaises(ValueError):
            validate_mmwave_candidate_metadata(meta)

    def test_10_validate_actual_generated_metadata_artifact(self):
        """10. Validate actual generated candidate metadata file in V6 models/mmwave"""
        actual_meta_path = project_root / "models/mmwave/mmwave_resp_int8_v0.2.0_candidate_metadata.json"
        self.assertTrue(actual_meta_path.exists(), f"File missing: {actual_meta_path}")

        with open(actual_meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.assertTrue(validate_mmwave_candidate_metadata(metadata, model_root=project_root))
        self.assertEqual(metadata["schema_version"], "1.0")
        self.assertEqual(metadata["project"], "SafeNest_V6")
        self.assertIn("float_keras", metadata["stage_evaluations"])
        self.assertIn("float_tflite", metadata["stage_evaluations"])
        self.assertIn("int8_tflite", metadata["stage_evaluations"])
        self.assertGreater(metadata["scaler"]["std"], 0)


if __name__ == "__main__":
    unittest.main()
