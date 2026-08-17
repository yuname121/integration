#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_mmwave_interpreter.py
P0-5 및 P0-6 mmWave Interpreter & Adapter 정밀 검증 테스트
"""

import sys
from pathlib import Path
import json
import hashlib
import tempfile
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.mmwave_interpreter import MMWaveInterpreter
from inference.model_registry import ModelRegistry
from adapters.mmwave_stream_adapter import MMWaveStreamAdapter

class TestMMWaveInterpreterAndAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.live_manifest_path = PROJECT_ROOT / "models/model_manifest.json"
        live_manifest = json.loads(cls.live_manifest_path.read_text(encoding="utf-8"))
        historical = dict(live_manifest)
        historical["models"] = dict(live_manifest["models"])
        historical["models"]["mmwave"] = live_manifest["models"]["mmwave_v0_1_0"]
        cls._historical_dir = tempfile.TemporaryDirectory()
        cls.historical_manifest_path = Path(cls._historical_dir.name) / "historical_v0_1_0_manifest.json"
        cls.historical_manifest_path.write_text(json.dumps(historical), encoding="utf-8")
        cls.runner = MMWaveInterpreter(
            project_root=PROJECT_ROOT,
            manifest_path=str(cls.historical_manifest_path),
        )
        cls.registry = ModelRegistry(project_root=PROJECT_ROOT)
        cls.stream_adapter = MMWaveStreamAdapter()

    @classmethod
    def tearDownClass(cls):
        cls._historical_dir.cleanup()

    def test_manifest_entry_exists(self):
        self.assertTrue(self.live_manifest_path.is_file())
        manifest = json.loads(self.live_manifest_path.read_text(encoding="utf-8"))
        self.assertIn("mmwave", manifest["models"])
        mmwave_meta = manifest["models"]["mmwave"]
        self.assertEqual(mmwave_meta["model_id"], "MMWAVE_M_N9_FULL_INT8_V1")
        self.assertEqual(mmwave_meta["runtime_role"], "ACTIVE_M_N9")
        self.assertEqual(mmwave_meta["path"], "models/mmwave/m_n9/MMWAVE_M_N9_FULL_INT8_V1.tflite")
        historical = manifest["models"]["mmwave_v0_1_0"]
        self.assertEqual(historical["model_id"], "mmwave_resp_int8")
        self.assertEqual(historical["runtime_role"], "HISTORICAL_V0_1_0")

    def test_real_tflite_is_loaded_and_hash_verified(self):
        health = self.registry.health()["mmwave"]
        self.assertTrue(health["loaded"])
        self.assertTrue(health["model_file_exists"])
        self.assertTrue(health["interpreter_loaded"])
        self.assertTrue(health["sha256_matches"])
        self.assertIsNone(health["load_error_reason"])
        self.assertEqual(health["model_id"], "MMWAVE_M_N9_FULL_INT8_V1")

        historical_sha = hashlib.sha256(self.runner.model_path.read_bytes()).hexdigest()
        self.assertEqual(historical_sha, self.runner.model_meta["sha256"])
        self.assertEqual(self.runner.model_meta["model_id"], "mmwave_resp_int8")

    def test_active_m_n9_tensor_contract(self):
        live = self.registry.mmwave
        self.assertEqual(list(live.input_info["shape"]), [1, 240, 1])
        self.assertEqual(live.input_info["dtype"], np.int8)
        self.assertEqual(live.input_info["quantization"], (0.5623255372047424, 4))
        self.assertEqual(list(live.output_info["shape"]), [1, 3])
        self.assertEqual(live.output_info["quantization"], (0.00390625, -128))
        actual = hashlib.sha256(live.model_path.read_bytes()).hexdigest()
        self.assertEqual(actual, "3b008af4be0facc4037c2afd3fe39292fb794208eb4370dbe6916b2d15aa38a4")

    def test_real_tflite_tensor_contract(self):
        input_info = self.runner.input_info
        output_info = self.runner.output_info
        self.assertEqual(list(input_info["shape"]), [1, 300, 1])
        self.assertEqual(input_info["dtype"], np.int8)
        self.assertEqual(input_info["quantization"], (0.03259856998920441, -13))
        self.assertEqual(list(output_info["shape"]), [1, 3])
        self.assertEqual(output_info["dtype"], np.int8)
        self.assertEqual(output_info["quantization"], (0.00390625, -128))

    def test_real_tflite_prediction_without_fallback(self):
        window = np.sin(2 * np.pi * 0.25 * np.arange(300, dtype=np.float32) / 10.0)
        res = self.runner.predict(window)
        self.assertFalse(res.fallback_used)
        self.assertIsNone(res.fallback_reason)
        self.assertEqual(res.model_id, "mmwave_resp_int8")
        self.assertEqual(len(res.probabilities), 3)
        self.assertTrue(np.all(np.isfinite(res.probabilities)))
        self.assertAlmostEqual(sum(res.probabilities), 1.0, places=5)

    def test_interpreter_window_prep(self):
        window = np.zeros(300, dtype=np.float32)
        prep = self.runner.prepare_window(window)
        self.assertEqual(prep.shape, (1, 300, 1))
        self.assertEqual(prep.dtype, np.int8)

    def test_metadata_matches_training_artifact(self):
        source = json.loads(
            (
                PROJECT_ROOT
                / "models/mmwave/source_sensor_stats_metadata_20260713.json"
            ).read_text(encoding="utf-8")
        )
        self.assertAlmostEqual(self.runner.mean, source["mean"][0], places=8)
        self.assertAlmostEqual(self.runner.std, source["std"][0], places=7)

    def test_invalid_window_is_rejected(self):
        with self.assertRaises(ValueError):
            self.runner.prepare_window(np.zeros(299, dtype=np.float32))
        bad = np.zeros(300, dtype=np.float32)
        bad[10] = np.nan
        with self.assertRaises(ValueError):
            self.runner.prepare_window(bad)

    def test_stream_adapter_ring_buffer(self):
        adapter = MMWaveStreamAdapter(window_samples=300)
        self.assertFalse(adapter.is_ready())
        
        for i in range(300):
            adapter.push_sample(float(i * 0.01))

        self.assertTrue(adapter.is_ready())
        window = adapter.get_window()
        self.assertIsNotNone(window)
        self.assertEqual(window.shape, (300,))


if __name__ == "__main__":
    unittest.main()
