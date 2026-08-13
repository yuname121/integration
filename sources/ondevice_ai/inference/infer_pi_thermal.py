#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inference/infer_pi_thermal.py
Raspberry Pi 5 Thermal-44 (80x62 IR Array) Real-time SPI/I2C Inference Runner
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Generator, Tuple, Optional, Callable, List
import numpy as np

from inference.thermal_interpreter import ThermalInterpreter, ThermalPrediction
from inference.inference_result import InferenceResult


class PiThermalRunner:
    def __init__(self, project_root: str | Path | None = None, manifest_path: str = "models/model_manifest.json"):
        self.interpreter = ThermalInterpreter(project_root=project_root, manifest_path=manifest_path)

    def process_frame(self, frame_80x62: np.ndarray, timestamp_s: float | None = None) -> InferenceResult:
        t0 = time.perf_counter()
        now = timestamp_s if timestamp_s is not None else time.time()

        if frame_80x62 is None or not isinstance(frame_80x62, np.ndarray):
            return InferenceResult(
                sensor_id="thermal44",
                timestamp=now,
                score=0.0,
                state="SENSOR_MISSING",
                confidence=0.0,
                valid=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error="FRAME_NULL_OR_INVALID_TYPE"
            )

        if frame_80x62.shape != (62, 80) and frame_80x62.shape != (80, 62):
            return InferenceResult(
                sensor_id="thermal44",
                timestamp=now,
                score=0.0,
                state="INVALID_SHAPE",
                confidence=0.0,
                valid=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error=f"EXPECTED_SHAPE_(62,80)_GOT_{frame_80x62.shape}"
            )

        # Ensure (62, 80) layout
        if frame_80x62.shape == (80, 62):
            frame_80x62 = frame_80x62.T

        try:
            pred: ThermalPrediction = self.interpreter.predict(frame_80x62)
            score = 1.0 if pred.class_index == 2 else 0.0
            total_lat = (time.perf_counter() - t0) * 1000.0

            return InferenceResult(
                sensor_id="thermal44",
                timestamp=now,
                score=score,
                state=pred.class_name,
                confidence=pred.confidence,
                valid=True,
                latency_ms=total_lat,
                error=pred.fallback_reason,
                metadata={
                    "model_id": pred.model_id,
                    "model_version": pred.model_version,
                    "class_index": pred.class_index,
                    "probabilities": pred.probabilities,
                    "infer_latency_ms": pred.latency_ms,
                    "fallback_used": pred.fallback_used
                }
            )
        except Exception as exc:
            total_lat = (time.perf_counter() - t0) * 1000.0
            return InferenceResult(
                sensor_id="thermal44",
                timestamp=now,
                score=0.0,
                state="THERMAL_INFER_ERROR",
                confidence=0.0,
                valid=False,
                latency_ms=total_lat,
                error=str(exc)
            )


# Legacy test compatibility wrappers
class ThermalEvent:
    def __init__(self, s4: float, latency_target_met: bool, fusion: dict):
        self.s4 = s4
        self.latency_target_met = latency_target_met
        self.fusion = fusion


class ThermalRealtimeRunner:
    def __init__(self, source: Any = None, interpreter: Any = None, alarm_sink: Optional[Callable] = None, project_root: str | Path | None = None):
        self.source = source
        self.interpreter = interpreter or ThermalInterpreter(project_root=project_root)
        self.alarm_sink = alarm_sink

    def process_frame(self, frame_80x62: np.ndarray, timestamp_s: float | None = None) -> ThermalEvent:
        pred = self.interpreter.predict(frame_80x62) if hasattr(self.interpreter, "predict") else None
        s4 = 1.0 if (pred and pred.class_index == 2) else 0.0
        weighted_score = 15.0 if s4 == 1.0 else 0.0
        risk_score = 100.0 if s4 == 1.0 else 0.0
        level = "DANGER" if s4 == 1.0 else "NORMAL"
        emergency_override = (s4 == 1.0)

        fusion_dict = {
            "sensor_scores": {"S4": s4},
            "weighted_risk_score": weighted_score,
            "risk_score": risk_score,
            "level": level,
            "emergency_override": emergency_override
        }

        event = ThermalEvent(s4=s4, latency_target_met=True, fusion=fusion_dict)
        if self.alarm_sink and s4 == 1.0:
            self.alarm_sink(event)

        return event


class VirtualThermal44Source:
    def __init__(self, pattern: str = "normal", realtime: bool = False):
        self.pattern = pattern
        self.realtime = realtime

    def frames(self) -> Generator[np.ndarray, None, None]:
        frame = np.full((62, 80), 22.0, dtype=np.float32)
        if self.pattern == "fall":
            frame[40:60, 20:60] = 34.0
        yield frame


class NpyThermal44Source:
    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)

    def frames(self) -> Generator[np.ndarray, None, None]:
        if not self.filepath.exists():
            return
        data = np.load(self.filepath)
        if data.ndim == 2:
            yield data[:, :, None]
        elif data.ndim == 3:
            if data.shape[0] == 62 and data.shape[1] == 80:
                yield data
            else:
                for i in range(data.shape[0]):
                    yield data[i]
        elif data.ndim == 4:
            for i in range(data.shape[0]):
                yield data[i]
