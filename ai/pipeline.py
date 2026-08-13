"""Convert current sensor state into isolated model/rule evaluations."""

from __future__ import annotations

from collections import deque
import math
import time
from typing import Mapping

from ai.result import AIResult
from ai.runtime import LazyModel
from gateway.protocol import ThermalFrame
from state.manager import SensorStateManager


class OnDeviceAIPipeline:
    def __init__(
        self,
        manager: SensorStateManager,
        models: Mapping[str, object] | None = None,
        *,
        clock=time.time,
    ) -> None:
        self.manager = manager
        supplied = dict(models or {})
        self.models = {
            sensor_id: supplied.get(sensor_id, LazyModel(sensor_id))
            for sensor_id in ("thermal", "mmwave", "co2")
        }
        self._clock = clock
        self._co2_history: deque[tuple[float, float]] = deque(maxlen=30)
        self._last_co2_sequence: int | None = None

    def evaluate(
        self,
        snapshot: dict[str, object] | None = None,
        thermal_frame: ThermalFrame | None = None,
    ) -> dict[str, object]:
        current = self.manager.snapshot() if snapshot is None else snapshot
        frame = self.manager.latest_thermal_frame() if thermal_frame is None else thermal_frame
        timestamp = float(current.get("timestamp", self._clock()))
        sensors = current["sensors"]

        results = {
            "thermal": self._thermal(sensors["thermal"], frame, timestamp),
            "mmwave": self._mmwave(sensors["mmwave"], timestamp),
            "co2": self._co2(sensors["co2"], timestamp),
            "pir": self._pir(sensors["pir"], timestamp),
        }
        model_results = [results[name] for name in ("thermal", "mmwave", "co2")]
        return {
            "timestamp": timestamp,
            "state_revision": current.get("revision"),
            "ai": {name: result.to_dict() for name, result in results.items()},
            "all_models_available": all(result.available for result in model_results),
            "degraded": any(not result.available for result in model_results),
        }

    def _thermal(self, sensor: dict[str, object], frame: ThermalFrame | None, now: float) -> AIResult:
        unavailable = self._sensor_unavailable("thermal", sensor, now)
        if unavailable:
            return unavailable
        if frame is None:
            return self._unavailable("thermal", now, "THERMAL_FRAME_MISSING")
        try:
            import numpy as np

            pixels = np.frombuffer(frame.pixel_bytes, dtype=">u2").astype(np.float32)
            pixels = pixels.reshape(frame.height, frame.width)
        except Exception as error:
            return self._model_error("thermal", now, error)
        metadata = {
            "raw_minimum": frame.minimum_raw,
            "raw_maximum": frame.maximum_raw,
            "temperature_calibrated": False,
            "preprocessing": "per_frame_minmax",
            "heatmap_preview": _thermal_preview(pixels),
        }
        try:
            prediction = self.models["thermal"].predict(pixels)
            return self._prediction_result(
                "thermal",
                prediction,
                now,
                score=1.0 if prediction.class_name == "HUMAN_FALL" else 0.0,
                metadata={
                    **metadata,
                    "probabilities": list(prediction.probabilities),
                },
            )
        except Exception as error:
            return self._model_error("thermal", now, error, metadata)

    def _mmwave(self, sensor: dict[str, object], now: float) -> AIResult:
        unavailable = self._sensor_unavailable("mmwave", sensor, now)
        if unavailable:
            return unavailable
        values = sensor.get("values", {})
        window = values.get("respiration_phase_window")
        if not isinstance(window, list) or len(window) != 300 or not _all_finite(window):
            return self._unavailable(
                "mmwave",
                now,
                "INPUT_UNAVAILABLE",
                {"missing": ["respiration_phase_window[300]"], "required_sample_rate_hz": 10},
            )
        try:
            prediction = self.models["mmwave"].predict(window)
            if bool(getattr(prediction, "fallback_used", False)):
                return self._unavailable(
                    "mmwave",
                    now,
                    "MODEL_RUNTIME_UNAVAILABLE",
                    {
                        "heuristic_state": prediction.class_name,
                        "fallback_reason": getattr(prediction, "fallback_reason", None),
                    },
                )
            risk = {"NORMAL": 0.0, "RAPID_OR_ABNORMAL": 0.5, "APNEA": 1.0}
            return self._prediction_result(
                "mmwave",
                prediction,
                now,
                score=risk.get(prediction.class_name, 0.5),
                metadata={"probabilities": list(prediction.probabilities), "apnea_verified": False},
            )
        except Exception as error:
            return self._model_error("mmwave", now, error)

    def _co2(self, sensor: dict[str, object], now: float) -> AIResult:
        unavailable = self._sensor_unavailable("co2", sensor, now)
        if unavailable:
            return unavailable
        values = sensor.get("values", {})
        ppm = values.get("ppm")
        humidity = values.get("humidity_percent")
        if not _finite_number(ppm) or not _finite_number(humidity):
            return self._unavailable(
                "co2", now, "INPUT_UNAVAILABLE", {"missing": ["humidity_percent", "co2_slope"]}
            )
        sequence = sensor.get("sequence")
        sample_time = sensor.get("last_update")
        if sequence != self._last_co2_sequence and _finite_number(sample_time):
            self._co2_history.append((float(sample_time), float(ppm)))
            self._last_co2_sequence = sequence
        if len(self._co2_history) < 2:
            return self._unavailable("co2", now, "WINDOW_WARMING_UP")
        elapsed_minutes = (self._co2_history[-1][0] - self._co2_history[0][0]) / 60.0
        if elapsed_minutes <= 0:
            return self._unavailable("co2", now, "WINDOW_WARMING_UP")
        slope = (self._co2_history[-1][1] - self._co2_history[0][1]) / elapsed_minutes
        try:
            prediction = self.models["co2"].predict(slope, float(humidity), float(ppm))
            return self._prediction_result(
                "co2",
                prediction,
                now,
                score=1.0 if prediction.class_name == "OCCUPIED" else 0.0,
                metadata={"probabilities": list(prediction.probabilities), "co2_slope_ppm_per_min": slope},
            )
        except Exception as error:
            return self._model_error("co2", now, error)

    @staticmethod
    def _pir(sensor: dict[str, object], now: float) -> AIResult:
        unavailable = OnDeviceAIPipeline._sensor_unavailable("pir", sensor, now)
        if unavailable:
            return unavailable
        motion = bool(sensor.get("values", {}).get("motion"))
        return AIResult(
            sensor_id="pir",
            timestamp=now,
            available=True,
            source="rule",
            state="MOTION" if motion else "NO_MOTION",
            score=0.0,
            confidence=1.0,
            metadata={"motion": motion, "risk_contribution_deferred": True},
        )

    @staticmethod
    def _sensor_unavailable(sensor_id: str, sensor: dict[str, object], now: float) -> AIResult | None:
        status = str(sensor.get("status", "NO_DATA"))
        if status == "LIVE":
            return None
        return OnDeviceAIPipeline._unavailable(sensor_id, now, f"SENSOR_{status}")

    @staticmethod
    def _prediction_result(
        sensor_id: str, prediction: object, now: float, *, score: float, metadata: dict[str, object]
    ) -> AIResult:
        return AIResult(
            sensor_id=sensor_id,
            timestamp=now,
            available=True,
            source="tflite",
            state=str(prediction.class_name),
            score=float(score),
            confidence=float(prediction.confidence),
            latency_ms=float(prediction.latency_ms),
            model_id=str(prediction.model_id),
            model_version=str(prediction.model_version),
            metadata=metadata,
        )

    @staticmethod
    def _model_error(
        sensor_id: str,
        now: float,
        error: Exception,
        metadata: dict[str, object] | None = None,
    ) -> AIResult:
        details = dict(metadata or {})
        details["detail"] = f"{type(error).__name__}: {error}"
        return OnDeviceAIPipeline._unavailable(
            sensor_id,
            now,
            "MODEL_RUNTIME_UNAVAILABLE",
            details,
        )

    @staticmethod
    def _unavailable(
        sensor_id: str,
        now: float,
        error: str,
        metadata: dict[str, object] | None = None,
    ) -> AIResult:
        return AIResult(
            sensor_id=sensor_id,
            timestamp=now,
            available=False,
            source="unavailable",
            state="INPUT_UNAVAILABLE",
            error=error,
            metadata=metadata or {},
        )


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _all_finite(values: list[object]) -> bool:
    return all(_finite_number(value) for value in values)


def _thermal_preview(pixels: object, width: int = 20, height: int = 16) -> dict[str, object]:
    source_height, source_width = pixels.shape
    minimum = float(pixels.min())
    maximum = float(pixels.max())
    span = maximum - minimum
    x_indices = [round(index * (source_width - 1) / (width - 1)) for index in range(width)]
    y_indices = [round(index * (source_height - 1) / (height - 1)) for index in range(height)]
    values = []
    for y_index in y_indices:
        for x_index in x_indices:
            value = 0.0 if span <= 0 else (float(pixels[y_index, x_index]) - minimum) / span
            values.append(round(min(1.0, max(0.0, value)), 4))
    return {
        "width": width,
        "height": height,
        "source_width": int(source_width),
        "source_height": int(source_height),
        "normalized": True,
        "values": values,
    }
