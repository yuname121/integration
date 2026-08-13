#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_v4_config_validation.py
P0-2 TFLite Model & Manifest Ground Truth Validation Regression Tests
"""

import copy
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from inference.validator import GroundTruthValidator, ConfigValidationError, find_repo_root

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestV4ConfigValidation(unittest.TestCase):
    def setUp(self):
        self._repo_wrapper = tempfile.TemporaryDirectory()
        self.repo_root = Path(self._repo_wrapper.name)
        os.symlink(PROJECT_ROOT, self.repo_root / "SafeNest_V4_OnDevice_AI")
        self.validator = GroundTruthValidator(project_root=PROJECT_ROOT)

    def tearDown(self):
        self._repo_wrapper.cleanup()

    def test_01_valid_config_and_manifest_passes(self):
        """1. 정상 config/manifest/model 검증 성공"""
        is_valid, inventory, errors = self.validator.validate_all(generate_inventory=False)
        self.assertTrue(is_valid, f"Validation should succeed for valid repository: {errors}")
        self.assertEqual(len(errors), 0)
        self.assertIn("thermal", inventory["models"])
        self.assertIn("mmwave", inventory["models"])
        self.assertIn("co2", inventory["models"])

    def test_package_local_validator_resolves_repository_root(self):
        """Package-local CLI must target the active V5 project root."""
        script_path = (
            self.repo_root
            / "SafeNest_V4_OnDevice_AI"
            / "scripts"
            / "validate_v4_config.py"
        )
        spec = importlib.util.spec_from_file_location(
            "safenest_package_validator_entrypoint",
            script_path,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.package_root, PROJECT_ROOT)
        self.assertEqual(module.repo_root, PROJECT_ROOT.parent)

    def test_02_missing_model_file_fails(self):
        """2. 모델 파일 누락 실패"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_repo = Path(tmpdir)
            # Copy manifest to tmp_repo
            m_dir = tmp_repo / "SafeNest_V4_OnDevice_AI" / "models"
            m_dir.mkdir(parents=True)
            orig_manifest = (self.repo_root / "SafeNest_V4_OnDevice_AI" / "models" / "model_manifest.json").read_text()
            (m_dir / "model_manifest.json").write_text(orig_manifest)

            val = GroundTruthValidator(repo_root=tmp_repo)
            is_valid, _, errors = val.validate_all(generate_inventory=False)
            self.assertFalse(is_valid)
            self.assertTrue(any("FILE_NOT_FOUND" in e or "missing" in e for e in errors))

    def test_03_sha256_mismatch_fails(self):
        """3. SHA256 불일치 실패"""
        manifest_abs = self.repo_root / "SafeNest_V4_OnDevice_AI" / "models" / "model_manifest.json"
        orig_data = json.loads(manifest_abs.read_text())

        tampered = copy.deepcopy(orig_data)
        tampered["models"]["thermal"]["sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"

        with tempfile.NamedTemporaryFile("w+", suffix=".json") as tmp_file:
            json.dump(tampered, tmp_file)
            tmp_file.flush()

            # Create temp repo structure pointing to fake manifest
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_repo = Path(tmpdir)
                shutil.copytree(self.repo_root / "SafeNest_V4_OnDevice_AI", tmp_repo / "SafeNest_V4_OnDevice_AI")
                (tmp_repo / "SafeNest_V4_OnDevice_AI" / "models" / "model_manifest.json").write_text(json.dumps(tampered))

                val = GroundTruthValidator(repo_root=tmp_repo)
                is_valid, _, errors = val.validate_all(generate_inventory=False)
                self.assertFalse(is_valid)
                self.assertTrue(any("SHA256" in e for e in errors))

    def test_04_dtype_mismatch_fails(self):
        """4. input/output dtype 불일치 실패"""
        manifest_abs = self.repo_root / "SafeNest_V4_OnDevice_AI" / "models" / "model_manifest.json"
        orig_data = json.loads(manifest_abs.read_text())

        tampered = copy.deepcopy(orig_data)
        tampered["models"]["co2"]["input"]["dtype"] = "float32"  # actual is int8

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_repo = Path(tmpdir)
            shutil.copytree(self.repo_root / "SafeNest_V4_OnDevice_AI", tmp_repo / "SafeNest_V4_OnDevice_AI")
            (tmp_repo / "SafeNest_V4_OnDevice_AI" / "models" / "model_manifest.json").write_text(json.dumps(tampered))

            val = GroundTruthValidator(repo_root=tmp_repo)
            is_valid, _, errors = val.validate_all(generate_inventory=False)
            self.assertFalse(is_valid)
            self.assertTrue(any("dtype" in e for e in errors))

    def test_05_tensor_name_mismatch_fails(self):
        """5. tensor name 불일치 실패"""
        manifest_abs = self.repo_root / "SafeNest_V4_OnDevice_AI" / "models" / "model_manifest.json"
        orig_data = json.loads(manifest_abs.read_text())

        tampered = copy.deepcopy(orig_data)
        tampered["models"]["thermal"]["input"]["name"] = "wrong_tensor_name:0"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_repo = Path(tmpdir)
            shutil.copytree(self.repo_root / "SafeNest_V4_OnDevice_AI", tmp_repo / "SafeNest_V4_OnDevice_AI")
            (tmp_repo / "SafeNest_V4_OnDevice_AI" / "models" / "model_manifest.json").write_text(json.dumps(tampered))

            val = GroundTruthValidator(repo_root=tmp_repo)
            is_valid, _, errors = val.validate_all(generate_inventory=False)
            self.assertFalse(is_valid)
            self.assertTrue(any("tensor name" in e for e in errors))

    def test_06_tensor_shape_mismatch_fails(self):
        """6. tensor shape 불일치 실패"""
        manifest_abs = self.repo_root / "SafeNest_V4_OnDevice_AI" / "models" / "model_manifest.json"
        orig_data = json.loads(manifest_abs.read_text())

        tampered = copy.deepcopy(orig_data)
        tampered["models"]["mmwave"]["input"]["shape"] = [1, 500, 1]  # actual is [1, 300, 1]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_repo = Path(tmpdir)
            shutil.copytree(self.repo_root / "SafeNest_V4_OnDevice_AI", tmp_repo / "SafeNest_V4_OnDevice_AI")
            (tmp_repo / "SafeNest_V4_OnDevice_AI" / "models" / "model_manifest.json").write_text(json.dumps(tampered))

            val = GroundTruthValidator(repo_root=tmp_repo)
            is_valid, _, errors = val.validate_all(generate_inventory=False)
            self.assertFalse(is_valid)
            self.assertTrue(any("shape" in e for e in errors))

    def test_07_quantization_scale_or_zero_point_mismatch_fails(self):
        """7. quantization scale 또는 zero point 불일치 실패"""
        manifest_abs = self.repo_root / "SafeNest_V4_OnDevice_AI" / "models" / "model_manifest.json"
        orig_data = json.loads(manifest_abs.read_text())

        tampered = copy.deepcopy(orig_data)
        tampered["models"]["co2"]["input"]["zero_point"] = 999  # actual is 57

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_repo = Path(tmpdir)
            shutil.copytree(self.repo_root / "SafeNest_V4_OnDevice_AI", tmp_repo / "SafeNest_V4_OnDevice_AI")
            (tmp_repo / "SafeNest_V4_OnDevice_AI" / "models" / "model_manifest.json").write_text(json.dumps(tampered))

            val = GroundTruthValidator(repo_root=tmp_repo)
            is_valid, _, errors = val.validate_all(generate_inventory=False)
            self.assertFalse(is_valid)
            self.assertTrue(any("zero point" in e for e in errors))

    def test_08_class_count_mismatch_fails(self):
        """8. 클래스 수 불일치 실패"""
        manifest_abs = self.repo_root / "SafeNest_V4_OnDevice_AI" / "models" / "model_manifest.json"
        orig_data = json.loads(manifest_abs.read_text())

        tampered = copy.deepcopy(orig_data)
        # Thermal model output has 3 classes, let's define 4 classes in manifest class_map
        tampered["models"]["thermal"]["class_map"] = {
            "0": "NOT_HUMAN",
            "1": "HUMAN_NORMAL",
            "2": "HUMAN_FALL",
            "3": "HUMAN_EXTRA"
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_repo = Path(tmpdir)
            shutil.copytree(self.repo_root / "SafeNest_V4_OnDevice_AI", tmp_repo / "SafeNest_V4_OnDevice_AI")
            (tmp_repo / "SafeNest_V4_OnDevice_AI" / "models" / "model_manifest.json").write_text(json.dumps(tampered))

            val = GroundTruthValidator(repo_root=tmp_repo)
            is_valid, _, errors = val.validate_all(generate_inventory=False)
            self.assertFalse(is_valid)
            self.assertTrue(any("class count" in e for e in errors))

    def test_09_class_order_mismatch_fails(self):
        """9. 클래스 순서 불일치 실패"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_repo = Path(tmpdir)
            shutil.copytree(self.repo_root / "SafeNest_V4_OnDevice_AI", tmp_repo / "SafeNest_V4_OnDevice_AI")

            # Tamper config/models.yaml class order to differ from manifest
            cfg_path = tmp_repo / "SafeNest_V4_OnDevice_AI" / "config" / "models.yaml"
            cfg_content = cfg_path.read_text()
            # Swap VACANT and OCCUPIED order in config
            swapped_cfg = cfg_content.replace('0: "VACANT"\n      1: "OCCUPIED"', '0: "OCCUPIED"\n      1: "VACANT"')
            cfg_path.write_text(swapped_cfg)

            val = GroundTruthValidator(repo_root=tmp_repo)
            is_valid, _, errors = val.validate_all(generate_inventory=False)
            self.assertFalse(is_valid)
            self.assertTrue(any("class order" in e or "label mismatch" in e for e in errors))

    def test_10_missing_scaler_or_metadata_fails(self):
        """10. scaler/metadata 누락 실패"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_repo = Path(tmpdir)
            shutil.copytree(self.repo_root / "SafeNest_V4_OnDevice_AI", tmp_repo / "SafeNest_V4_OnDevice_AI")

            # Remove CO2 metadata file
            meta_file = tmp_repo / "SafeNest_V4_OnDevice_AI" / "models" / "co2" / "co2_scaling_metadata_v0.1.0.json"
            if meta_file.is_file():
                meta_file.unlink()

            val = GroundTruthValidator(repo_root=tmp_repo)
            is_valid, _, errors = val.validate_all(generate_inventory=False)
            self.assertFalse(is_valid)
            self.assertTrue(any("metadata file" in e or "FILE_NOT_FOUND" in e for e in errors))

    def test_11_config_runtime_path_mismatch_fails(self):
        """11. config와 runtime 모델 경로 불일치 실패"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_repo = Path(tmpdir)
            shutil.copytree(self.repo_root / "SafeNest_V4_OnDevice_AI", tmp_repo / "SafeNest_V4_OnDevice_AI")

            cfg_path = tmp_repo / "SafeNest_V4_OnDevice_AI" / "config" / "models.yaml"
            cfg_content = cfg_path.read_text()
            mismatched_cfg = cfg_content.replace(
                'path: "models/thermal/thermal_fall_int8_v0.1.0.tflite"',
                'path: "models/thermal/different_thermal.tflite"'
            )
            cfg_path.write_text(mismatched_cfg)

            val = GroundTruthValidator(repo_root=tmp_repo)
            is_valid, _, errors = val.validate_all(generate_inventory=False)
            self.assertFalse(is_valid)
            self.assertTrue(any("path consistency" in e for e in errors))

    def test_12_absolute_path_usage_fails(self):
        """12. 절대경로 사용 실패"""
        manifest_abs = self.repo_root / "SafeNest_V4_OnDevice_AI" / "models" / "model_manifest.json"
        orig_data = json.loads(manifest_abs.read_text())

        tampered = copy.deepcopy(orig_data)
        tampered["models"]["co2"]["path"] = "/Users/junwoo/absolute/path/co2.tflite"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_repo = Path(tmpdir)
            shutil.copytree(self.repo_root / "SafeNest_V4_OnDevice_AI", tmp_repo / "SafeNest_V4_OnDevice_AI")
            (tmp_repo / "SafeNest_V4_OnDevice_AI" / "models" / "model_manifest.json").write_text(json.dumps(tampered))

            val = GroundTruthValidator(repo_root=tmp_repo)
            is_valid, _, errors = val.validate_all(generate_inventory=False)
            self.assertFalse(is_valid)
            self.assertTrue(any("Absolute path is forbidden" in e or "Absolute path detected" in e for e in errors))

    def test_13_different_working_dir_success(self):
        """13. working directory가 달라도 검증 성공"""
        old_cwd = os.cwd() if hasattr(os, "cwd") else os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                val = GroundTruthValidator(repo_root=self.repo_root)
                is_valid, _, errors = val.validate_all(generate_inventory=False)
                self.assertTrue(is_valid, f"Validation failed when run from {tmpdir}: {errors}")
            finally:
                os.chdir(old_cwd)

    def test_14_validation_failure_blocks_inference(self):
        """14. 검증 실패 시 inference/backend 초기화가 시작되지 않음"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_repo = Path(tmpdir)
            shutil.copytree(self.repo_root / "SafeNest_V4_OnDevice_AI", tmp_repo / "SafeNest_V4_OnDevice_AI")

            # Corrupt manifest in tmp_repo
            m_path = tmp_repo / "SafeNest_V4_OnDevice_AI" / "models" / "model_manifest.json"
            m_path.write_text("{}")

            from inference.model_registry import ModelRegistry
            with self.assertRaises(ConfigValidationError):
                ModelRegistry(project_root=tmp_repo / "SafeNest_V4_OnDevice_AI", validate_on_init=True)


if __name__ == "__main__":
    unittest.main()
