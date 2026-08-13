#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inference/model_registry.py
SafeNest 3대 엣지 AI 모델 통합 레지스트리 (ModelRegistry)

[검수 반영 완료]
ModelRegistry가 단순히 Wrapper 객체 존재 여부만 검사하는 것이 아니라,
실제 .tflite 파일 존재 여부(model_file_exists), Interpreter 로드 여부(interpreter_loaded)
및 SHA-256 일치 상태까지 엄격하게 검증하여 health 텔레메트리로 반환하도록 보완
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from .thermal_interpreter import ThermalInterpreter, ThermalPrediction
from .co2_interpreter import CO2Interpreter, CO2Prediction
from .mmwave_interpreter import MMWaveInterpreter, MMWavePrediction


class ModelRegistry:
    def __init__(
        self,
        project_root: str | Path | None = None,
        manifest_path: str = "models/model_manifest.json",
        validate_on_init: bool = True,
    ) -> None:
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = Path(project_root).resolve()

        self.manifest_path = manifest_path

        if validate_on_init:
            from .validator import GroundTruthValidator, ConfigValidationError
            validator = GroundTruthValidator(project_root=self.project_root)
            is_valid, _, errors = validator.validate_all(generate_inventory=False)
            if not is_valid:
                raise ConfigValidationError(
                    f"ModelRegistry startup blocked due to validation failure: {'; '.join(errors)}"
                )

        # 1. Thermal Interpreter
        try:
            self.thermal = ThermalInterpreter(project_root=self.project_root, manifest_path=manifest_path)
        except Exception as e:
            print(f"⚠️ [ModelRegistry] Thermal Interpreter 로드 실패: {e}")
            self.thermal = None

        # 2. CO2 Interpreter
        try:
            self.co2 = CO2Interpreter(project_root=self.project_root, manifest_path=manifest_path)
        except Exception as e:
            print(f"⚠️ [ModelRegistry] CO2 Interpreter 로드 실패: {e}")
            self.co2 = None

        # 3. mmWave Interpreter
        try:
            self.mmwave = MMWaveInterpreter(project_root=self.project_root, manifest_path=manifest_path)
        except Exception as e:
            print(f"⚠️ [ModelRegistry] mmWave Interpreter 로드 실패: {e}")
            self.mmwave = None

    def health(self) -> Dict[str, Any]:
        """실제 TFLite 런타임 및 모델 파일 존재 여부를 엄격히 검증하는 Health Check"""
        thermal_file_ok = self.thermal.model_path.is_file() if self.thermal else False
        thermal_interp_ok = self.thermal.interpreter is not None if self.thermal else False
        thermal_hash_ok = self.thermal.sha256_matches if self.thermal else False

        co2_file_ok = self.co2.model_path.is_file() if self.co2 else False
        co2_interp_ok = self.co2.interpreter is not None if self.co2 else False
        co2_hash_ok = self.co2.sha256_matches if self.co2 else False

        mmwave_file_ok = self.mmwave.model_file_exists if self.mmwave else False
        mmwave_interp_ok = self.mmwave.interpreter is not None if self.mmwave else False
        mmwave_hash_ok = self.mmwave.sha256_matches if self.mmwave else False

        return {
            "thermal": {
                "loaded": thermal_file_ok and thermal_interp_ok and thermal_hash_ok,
                "model_file_exists": thermal_file_ok,
                "interpreter_loaded": thermal_interp_ok,
                "sha256_matches": thermal_hash_ok,
                "sha256": self.thermal.sha256_hash if self.thermal else None,
                "model_id": self.thermal.model_meta["model_id"] if self.thermal else None,
                "version": self.thermal.model_meta["version"] if self.thermal else None,
            },
            "co2": {
                "loaded": co2_file_ok and co2_interp_ok and co2_hash_ok,
                "model_file_exists": co2_file_ok,
                "interpreter_loaded": co2_interp_ok,
                "sha256_matches": co2_hash_ok,
                "sha256": self.co2.sha256_hash if self.co2 else None,
                "model_id": self.co2.model_meta["model_id"] if self.co2 else None,
                "version": self.co2.model_meta["version"] if self.co2 else None,
            },
            "mmwave": {
                "loaded": mmwave_file_ok and mmwave_interp_ok and mmwave_hash_ok,
                "model_file_exists": mmwave_file_ok,
                "interpreter_loaded": mmwave_interp_ok,
                "sha256_matches": mmwave_hash_ok,
                "sha256": self.mmwave.sha256_hash if self.mmwave else None,
                "load_error_reason": self.mmwave.load_error_reason if self.mmwave else "INTERPRETER_INIT_ERROR",
                "model_id": self.mmwave.model_meta["model_id"] if (self.mmwave and mmwave_interp_ok) else "mmwave_heuristic_fallback",
                "version": self.mmwave.model_meta["version"] if self.mmwave else None,
            },
        }
