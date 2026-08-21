"""Convert current sensor state into isolated model/rule evaluations."""

from __future__ import annotations

import math
import time
from typing import Mapping

from ai.result import AIResult
from ai.runtime import LazyModel
from ai.co2_canonical_runtime import CO2SlopeWindowBuilder
from ai.mmwave_canonical_runtime import MR60CanonicalWindowBuilder
from ai.mmwave_spectral_runtime import estimate_respiration
from gateway.protocol import TelemetryPayload, ThermalFrame
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
        self._mmwave_window = MR60CanonicalWindowBuilder()
        self._mmwave_wire_observed = False
        self._co2_window = CO2SlopeWindowBuilder()
        self._co2_wire_observed = False

    def observe_telemetry(self, packet: TelemetryPayload) -> None:
        """Accumulate the MR60 phase stream at wire rate, not at publication rate.

        Both accumulators are fed here. The M-N4 canonical window needs 30
        continuous seconds of ~8 Hz phase events with no gap wider than
        ``max(400 ms, 4x median dt)``, and the C-B6 CO2 slope needs >=150 s of
        measurement-event history. Feeding either from ``evaluate`` would sample
        the stream once per publication interval (15 s by default), which can
        never satisfy those contracts.
        """

        self._mmwave_window.ingest(
            {
                "sequence": packet.header.sequence,
                "boot_id": packet.boot_id,
                "values": {
                    "breath_phase": packet.breath_phase,
                    "ts_monotonic_ms": packet.ts_monotonic_ms,
                    "phase_age_ms": packet.phase_age_ms,
                    "session_id": packet.session_id,
                },
            }
        )
        self._mmwave_wire_observed = True
        self.observe_co2(
            {
                "device_id": packet.device_id,
                "boot_id": packet.boot_id,
                "values": {
                    "measurement_event_valid": packet.co2_measurement_event_valid,
                    "measurement_event_id": packet.co2_measurement_event_id,
                    "measurement_monotonic_ms": packet.co2_measurement_monotonic_ms,
                    "latest_measurement_ppm": packet.co2_ppm,
                },
            }
        )

    def observe_co2(self, sensor: Mapping[str, object]) -> None:
        """Accumulate CO2 measurement events at state-update rate.

        The canonical slope needs >=150 s of source-clock history anchored on
        ``measurement_event_id``; sampling it from the publication loop would
        both starve the history and let the 60 s presentation throttle
        masquerade as the physical measurement cadence.
        """

        self._co2_window.observe(sensor)
        self._co2_wire_observed = True

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
        values = sensor.get("values") if isinstance(sensor.get("values"), dict) else {}
        missing = _mmwave_missing_fields(values)
        if not self._mmwave_wire_observed:
            # Snapshot-driven callers (offline replay, unit tests) have no wire
            # feed; live runtimes use observe_telemetry and must not double-count.
            self._mmwave_window.ingest(sensor)
        window = self._mmwave_window.latest()
        diagnostics = {
            "canonical_window_status": window.status,
            "missing": missing,
            **window.metadata,
        }
        if window.status != "CANONICAL_WINDOW_READY" or window.tensor is None:
            return self._unavailable(
                "mmwave",
                now,
                window.reason or window.status,
                diagnostics,
                state=window.status,
            )
        presence = values.get("presence")
        presence_available = values.get("presence_available") is True
        if presence_available is not True or not isinstance(presence, bool):
            return self._suppressed(
                "mmwave",
                now,
                "PRESENCE_STATE_UNAVAILABLE",
                {
                    **diagnostics,
                    "suppressed": True,
                    "suppression_reason": "PRESENCE_STATE_UNAVAILABLE",
                },
            )
        if presence is False:
            return self._suppressed(
                "mmwave",
                now,
                "NO_VALID_PERSON",
                {
                    **diagnostics,
                    "suppressed": True,
                    "suppression_reason": "NO_VALID_PERSON",
                    "presence_valid": False,
                },
            )
        # Deterministic DSP readout of the same canonical window. It is the
        # trustworthy respiration signal while M-N9 is DEVICE_VALIDATED: NO, and
        # it is what contradicts a physically impossible APNEA-proxy class.
        spectral = estimate_respiration(window.tensor)
        diagnostics = {
            **diagnostics,
            "spectral_status": spectral.status,
            "spectral_rate_rpm": spectral.rate_rpm,
            "spectral_band_power_fraction": spectral.band_power_fraction,
            "spectral_hold_evidence": spectral.hold_evidence,
            "spectral_contradicts_apnea": spectral.contradicts_apnea,
            **spectral.metadata,
        }
        try:
            prediction = self.models["mmwave"].predict(window.tensor)
            if bool(getattr(prediction, "fallback_used", False)):
                return self._unavailable(
                    "mmwave",
                    now,
                    "MODEL_RUNTIME_UNAVAILABLE",
                    {
                        **diagnostics,
                        "heuristic_state": prediction.class_name,
                        "fallback_reason": getattr(prediction, "fallback_reason", None),
                    },
                )
            if (
                str(prediction.class_name) in _APNEA_STATES
                and spectral.contradicts_apnea
            ):
                # The window is periodic in the respiration band and contains no
                # quiet stretch as long as the 6 s hold the APNEA label requires,
                # so a breath-hold did not occur. Refuse rather than publish it.
                return self._unavailable(
                    "mmwave",
                    now,
                    "APNEA_CONTRADICTED_BY_SPECTRUM",
                    {
                        **diagnostics,
                        "refused_class": str(prediction.class_name),
                        "refused_confidence": float(prediction.confidence),
                        "refused_probabilities": list(prediction.probabilities),
                    },
                    state="RESPIRATORY_INFERENCE_REFUSED",
                )
            risk = {"NORMAL": 0.0, "RAPID_OR_ABNORMAL": 0.5, "APNEA-proxy": 1.0}
            return self._prediction_result(
                "mmwave",
                prediction,
                now,
                score=risk.get(prediction.class_name, 0.5),
                metadata={
                    **diagnostics,
                    "probabilities": list(prediction.probabilities),
                    "dequantized_scores": list(prediction.probabilities),
                    "model_sha256": getattr(prediction, "model_sha256", None),
                    "contract_id": "MMWAVE_MR60_COMPAT_INPUT_DATASET_V1",
                    "presence_valid": True,
                    "suppressed": False,
                    "canonical_window_status": window.status,
                    "apnea_verified": False,
                },
            )
        except Exception as error:
            return self._model_error("mmwave", now, error, diagnostics)

    def _co2(self, sensor: dict[str, object], now: float) -> AIResult:
        unavailable = self._sensor_unavailable("co2", sensor, now)
        if unavailable:
            return unavailable
        if not self._co2_wire_observed:
            # Snapshot-driven callers (offline replay, unit tests) have no wire feed.
            self._co2_window.observe(sensor)
        slope = self._co2_window.latest()
        diagnostics = dict(slope.metadata)
        if not slope.ready or slope.ppm is None or slope.slope_ppm_per_min is None:
            return self._unavailable(
                "co2",
                now,
                slope.reason or slope.status,
                diagnostics,
                state=slope.status,
            )
        diagnostics["co2_slope_ppm_per_min"] = slope.slope_ppm_per_min
        try:
            # C-B6 reduced contract: ppm and ppm/min only. Humidity and
            # temperature are in forbidden_additional_inputs.
            prediction = self.models["co2"].predict(slope.ppm, slope.slope_ppm_per_min)
            return self._prediction_result(
                "co2",
                prediction,
                now,
                # class_map declares risk_semantic NONE / safety_semantic NONE, so
                # occupancy must not enter the safety score as a hazard weight.
                score=0.0,
                metadata={
                    **diagnostics,
                    "probabilities": list(prediction.probabilities),
                    "occupancy_probability": getattr(prediction, "occupancy_probability", None),
                    "threshold": getattr(prediction, "threshold", None),
                    "contract_id": getattr(prediction, "contract_id", None),
                    "model_sha256": getattr(prediction, "model_sha256", None),
                    "risk_semantic": getattr(prediction, "risk_semantic", "NONE"),
                    "safety_semantic": getattr(prediction, "safety_semantic", "NONE"),
                    "risk_contribution_deferred": True,
                    "humidity_required": False,
                },
            )
        except Exception as error:
            return self._model_error("co2", now, error, diagnostics)

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
        *,
        state: str = "INPUT_UNAVAILABLE",
    ) -> AIResult:
        return AIResult(
            sensor_id=sensor_id,
            timestamp=now,
            available=False,
            source="unavailable",
            state=state,
            error=error,
            metadata=metadata or {},
        )

    @staticmethod
    def _suppressed(
        sensor_id: str,
        now: float,
        reason: str,
        metadata: dict[str, object],
    ) -> AIResult:
        """Represent an intentional safety gate, not an invalid neural class."""
        return AIResult(
            sensor_id=sensor_id,
            timestamp=now,
            available=False,
            source="unavailable",
            state="RESPIRATORY_INFERENCE_SUPPRESSED",
            error=reason,
            metadata=metadata,
        )


_APNEA_STATES = frozenset({"APNEA", "APNEA-proxy"})


def _mmwave_missing_fields(values: dict[str, object]) -> list[str]:
    missing: list[str] = []
    for field in ("breath_phase", "ts_monotonic_ms", "phase_age_ms"):
        if not _finite_number(values.get(field)):
            missing.append(field)
    if not isinstance(values.get("human_detected_raw"), bool) and not isinstance(
        values.get("presence"), bool
    ):
        missing.append("human_detected_raw")
    return missing


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
