#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_thermal_interpreter.py
SafeNest Thermal Interpreter 단위 및 회귀 자동 테스트 수트 (Standard Unittest & Pytest 지원)
"""

import os
import sys
from pathlib import Path
import hashlib
import json
import unittest
import numpy as np

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.thermal_interpreter import ThermalInterpreter


class TestThermalInterpreter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = ThermalInterpreter(project_root=PROJECT_ROOT)

    def test_manifest_hash_matches(self):
        manifest_path = PROJECT_ROOT / "models/model_manifest.json"
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = manifest["models"]["thermal"]
        model_path = PROJECT_ROOT / model["path"]

        actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        self.assertEqual(actual_hash, model["sha256"])
        self.assertEqual(model_path.stat().st_size, model["size_bytes"])

    def test_tensor_contract(self):
        self.assertEqual(self.runner.input_info["shape"].tolist(), [1, 62, 80, 1])
        self.assertEqual(self.runner.output_info["shape"].tolist(), [1, 3])
        self.assertEqual(self.runner.input_info["dtype"], np.int8)
        self.assertEqual(self.runner.output_info["dtype"], np.int8)

        input_scale, _ = self.runner.input_info["quantization"]
        output_scale, _ = self.runner.output_info["quantization"]
        self.assertGreater(input_scale, 0)
        self.assertGreater(output_scale, 0)

    def test_supported_shapes(self):
        for shape in [(62, 80), (62, 80, 1), (1, 62, 80, 1)]:
            frame = np.zeros(shape, dtype=np.float32)
            result = self.runner.predict(frame)

            self.assertIn(result.class_index, (0, 1, 2))
            self.assertEqual(len(result.probabilities), 3)
            self.assertTrue(np.all(np.isfinite(result.probabilities)))
            self.assertAlmostEqual(sum(result.probabilities), 1.0, places=4)

    def test_rejects_wrong_shape(self):
        frame = np.zeros((32, 24), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.runner.predict(frame)

    def test_rejects_invalid_values(self):
        for bad_value in [np.nan, np.inf, -np.inf]:
            frame = np.zeros((62, 80), dtype=np.float32)
            frame[0, 0] = bad_value
            with self.assertRaises(ValueError):
                self.runner.predict(frame)

    def test_current_npz_class_smoke(self):
        dataset_path = PROJECT_ROOT / "thermal/processed_thermal_80x62.npz"
        if not dataset_path.exists():
            self.skipTest("NPZ dataset file not found")

        data = np.load(dataset_path)
        frames = data["X"]
        labels = data["y"]

        seen = set()
        for class_index in (0, 1, 2):
            indices = np.where(labels == class_index)[0]
            self.assertGreater(len(indices), 0, f"class {class_index} is absent")

            result = self.runner.predict(frames[int(indices[0])])
            self.assertIn(result.class_index, (0, 1, 2))
            seen.add(class_index)

        self.assertEqual(seen, {0, 1, 2})

    def test_prediction_does_not_collapse_to_one_class(self):
        dataset_path = PROJECT_ROOT / "thermal/processed_thermal_80x62.npz"
        if not dataset_path.exists():
            self.skipTest("NPZ dataset file not found")

        data = np.load(dataset_path)
        frames = data["X"]
        labels = data["y"]

        selected = []
        for class_index in (0, 1, 2):
            indices = np.where(labels == class_index)[0][:100]
            selected.extend(int(index) for index in indices)

        predictions = {
            self.runner.predict(frames[index]).class_index
            for index in selected
        }

        self.assertGreaterEqual(
            len(predictions), 2, f"model output collapsed to classes: {sorted(predictions)}"
        )


if __name__ == "__main__":
    unittest.main()
