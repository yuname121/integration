#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_mmwave_v6_pipeline.py
Targeted test suite for SafeNest V6 mmWave On-Device AI Pipeline
"""

from __future__ import annotations
import os
import sys
import json
import unittest
import numpy as np
from pathlib import Path

# Ensure the canonical repository root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preprocessing.mmwave import MMWavePreprocessor
from scripts.evaluate_mmwave import compute_metrics, calculate_sha256
from scripts.check_mmwave_candidate import check_pipeline_smoke_gate, check_release_deployment_gate
from inference.mmwave_interpreter import MMWaveInterpreter


class TestMMWaveV6Pipeline(unittest.TestCase):

    def test_input_contract_file(self):
        contract_path = project_root / "config/mmwave_input_contract.yaml"
        self.assertTrue(contract_path.exists(), "Input contract YAML file must exist")
        content = contract_path.read_text(encoding="utf-8")
        self.assertIn("signal_name: \"resp_phase\"", content)
        self.assertIn("sample_rate_hz: 10", content)
        self.assertIn("window_samples: 300", content)
        self.assertIn("unit: \"UNKNOWN\"", content)
        self.assertIn("unit_verification_status: \"NOT_VERIFIABLE\"", content)
        self.assertIn("preprocessing_status: \"EXPERIMENTAL_PREPROCESSING_V1\"", content)

    def test_common_preprocessor_shape_and_nan_handling(self):
        prep = MMWavePreprocessor(mean=0.0, std=1.0)
        
        # Normal 300-sample signal
        signal = np.sin(np.linspace(0, 10, 300, dtype=np.float32))
        proc, info = prep.preprocess_window(signal)
        
        self.assertEqual(proc.shape, (1, 300, 1))
        self.assertEqual(proc.dtype, np.float32)
        self.assertTrue(info["valid"])
        self.assertEqual(info["nan_count"], 0)

        # Signal with NaN and Inf values
        dirty_signal = signal.copy()
        dirty_signal[10] = np.nan
        dirty_signal[50] = np.inf
        proc_dirty, info_dirty = prep.preprocess_window(dirty_signal)
        
        self.assertEqual(proc_dirty.shape, (1, 300, 1))
        self.assertFalse(info_dirty["valid"])
        self.assertEqual(info_dirty["nan_count"], 1)
        self.assertEqual(info_dirty["inf_count"], 1)
        self.assertFalse(np.isnan(proc_dirty).any())
        self.assertFalse(np.isinf(proc_dirty).any())

    def test_common_preprocessor_train_only_stats(self):
        X_fake_train = np.random.normal(loc=10.0, scale=2.5, size=(100, 300, 1)).astype(np.float32)
        prep = MMWavePreprocessor.from_train_split(X_fake_train)
        
        self.assertTrue(np.isclose(prep.mean, 10.0, atol=0.5))
        self.assertTrue(np.isclose(prep.std, 2.5, atol=0.5))

        batch_proc = prep.preprocess_batch(X_fake_train)
        self.assertEqual(batch_proc.shape, (100, 300, 1))

    def test_group_split_metadata(self):
        split_path = project_root / "datasets/mmwave/splits/mmwave_group_split_v1.json"
        self.assertTrue(split_path.exists(), "Group split JSON must exist")
        with open(split_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        
        self.assertEqual(split_data.get("synthetic_group_isolation"), "CONFIRMED_SYNTHETIC_ONLY")
        self.assertEqual(split_data.get("real_subject_provenance"), "NOT_VERIFIABLE")
        audit = split_data.get("leakage_audit", {})
        self.assertEqual(audit.get("status"), "PASSED")

    def test_compute_metrics_class_collapse(self):
        class_map = {0: "NORMAL", 1: "RAPID_OR_ABNORMAL", 2: "APNEA"}
        
        # All predictions NORMAL -> class collapse
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred_collapsed = np.array([0, 0, 0, 0, 0, 0])
        
        metrics = compute_metrics(y_true, y_pred_collapsed, class_map)
        self.assertTrue(metrics["class_collapse"])
        self.assertEqual(metrics["prediction_distribution"]["NORMAL"], 6)
        self.assertEqual(metrics["prediction_distribution"]["APNEA"], 0)
        self.assertEqual(metrics["apnea_window_miss_rate"], 1.0)
        self.assertEqual(metrics["false_alarm_status"], "NOT_COMPUTABLE")

        # Diverse predictions -> no class collapse
        y_pred_diverse = np.array([0, 1, 2, 0, 1, 2])
        metrics_div = compute_metrics(y_true, y_pred_diverse, class_map)
        self.assertFalse(metrics_div["class_collapse"])
        self.assertEqual(metrics_div["apnea_window_miss_rate"], 0.0)

    def test_candidate_acceptance_gates(self):
        candidate_tflite = project_root / "models/mmwave/mmwave_resp_int8_v0.2.0_candidate.tflite"
        candidate_meta = project_root / "models/mmwave/mmwave_resp_int8_v0.2.0_candidate_metadata.json"
        
        self.assertTrue(candidate_tflite.exists())
        self.assertTrue(candidate_meta.exists())
        
        # Smoke gate must PASS
        smoke_passed, smoke_failures = check_pipeline_smoke_gate(candidate_tflite, candidate_meta)
        self.assertTrue(smoke_passed, f"Smoke gate failed: {smoke_failures}")

        # Release gate must fail/block due to synthetic-only data
        rel_passed, rel_failures = check_release_deployment_gate(candidate_meta)
        self.assertFalse(rel_passed, "Release gate must fail/block for synthetic smoke candidate")
        self.assertIn("Release Gate Blocked: Model validated solely on synthetic NPZ data.", rel_failures)

    def test_mmwave_interpreter_wrapper_integration(self):
        interpreter = MMWaveInterpreter(project_root=project_root)
        dummy_signal = np.sin(np.linspace(0, 10, 300, dtype=np.float32))
        
        prediction = interpreter.predict(dummy_signal)
        self.assertIn(prediction.class_name, ["NORMAL", "RAPID_OR_ABNORMAL", "APNEA"])
        self.assertGreaterEqual(prediction.confidence, 0.0)
        self.assertLessEqual(prediction.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
