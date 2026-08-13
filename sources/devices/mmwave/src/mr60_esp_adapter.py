#!/usr/bin/env python3
"""Convert ESP MR60 JSON telemetry into a safe Pi-side vital packet."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PhaseEstimate:
    rate_rpm: float | None
    confidence: float
    valid: bool
    reason: str | None
    window_samples: int
    phase_std: float | None
    spectral_peak_ratio: float | None


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.number))
        and bool(np.isfinite(value))
    )


class PhaseRateEstimator:
    """Causal rolling FFT estimator that never interpolates across invalid gaps."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.sample_rate_hz = float(config["sample_rate_hz"])
        self.window_seconds = float(config["window_seconds"])
        self.window_samples = int(round(self.sample_rate_hz * self.window_seconds))
        self.max_gap_seconds = float(config["max_gap_seconds"])
        self.band_min_hz = float(config["respiration_band_min_rpm"]) / 60.0
        self.band_max_hz = float(config["respiration_band_max_rpm"]) / 60.0
        self.minimum_phase_std = float(config["minimum_phase_std"])
        self.timestamps: deque[float] = deque(maxlen=self.window_samples)
        self.values: deque[float] = deque(maxlen=self.window_samples)
        self.last_reason: str | None = "MMWAVE_WINDOW_NOT_READY"

    def reset(self, reason: str) -> None:
        self.timestamps.clear()
        self.values.clear()
        self.last_reason = reason

    def push(self, value: object, timestamp_s: object) -> bool:
        if not _finite_number(value):
            self.reset("MMWAVE_PHASE_INVALID")
            return False
        if not _finite_number(timestamp_s):
            self.reset("MMWAVE_TIMESTAMP_NON_FINITE")
            return False
        timestamp = float(timestamp_s)
        if self.timestamps:
            delta = timestamp - self.timestamps[-1]
            if delta <= 0:
                self.reset("MMWAVE_TIMESTAMP_NON_MONOTONIC")
                return False
            if delta > self.max_gap_seconds:
                self.reset("MMWAVE_STREAM_GAP_TOO_LARGE")
                return False
        self.timestamps.append(timestamp)
        self.values.append(float(value))
        self.last_reason = None
        return True

    def estimate(self) -> PhaseEstimate:
        if len(self.values) < self.window_samples:
            return PhaseEstimate(
                None, 0.0, False, self.last_reason or "MMWAVE_WINDOW_NOT_READY",
                len(self.values), None, None,
            )
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        duration = float(timestamps[-1] - timestamps[0])
        expected_duration = (self.window_samples - 1) / self.sample_rate_hz
        if duration < expected_duration - self.max_gap_seconds:
            return PhaseEstimate(
                None, 0.0, False, "MMWAVE_WINDOW_DURATION_SHORT",
                len(values), None, None,
            )
        target_timestamps = timestamps[0] + np.arange(self.window_samples) / self.sample_rate_hz
        if target_timestamps[-1] > timestamps[-1] + self.max_gap_seconds:
            return PhaseEstimate(
                None, 0.0, False, "MMWAVE_WINDOW_CANNOT_RESAMPLE",
                len(values), None, None,
            )
        uniform = np.interp(target_timestamps, timestamps, values)
        trend = np.polyval(np.polyfit(target_timestamps - target_timestamps[0], uniform, 1),
                           target_timestamps - target_timestamps[0])
        detrended = uniform - trend
        phase_std = float(np.std(detrended))
        if phase_std < self.minimum_phase_std:
            return PhaseEstimate(
                None, 0.0, False, "MMWAVE_PHASE_SIGNAL_TOO_FLAT",
                len(values), phase_std, None,
            )
        nfft = max(4096, 1 << (len(detrended) - 1).bit_length())
        spectrum = np.abs(np.fft.rfft(detrended * np.hanning(len(detrended)), n=nfft))
        frequencies = np.fft.rfftfreq(nfft, d=1.0 / self.sample_rate_hz)
        band = (frequencies >= self.band_min_hz) & (frequencies <= self.band_max_hz)
        band_indices = np.flatnonzero(band)
        if not len(band_indices):
            return PhaseEstimate(
                None, 0.0, False, "MMWAVE_RESPIRATION_BAND_EMPTY",
                len(values), phase_std, None,
            )
        peak_index = int(band_indices[np.argmax(spectrum[band])])
        peak_frequency = float(frequencies[peak_index])
        if 0 < peak_index < len(spectrum) - 1:
            left, center, right = spectrum[peak_index - 1:peak_index + 2]
            denominator = left - 2.0 * center + right
            if abs(denominator) > 1e-12:
                offset = 0.5 * (left - right) / denominator
                peak_frequency += float(offset) * (frequencies[1] - frequencies[0])
        band_floor = float(np.median(spectrum[band]))
        peak_ratio = float(spectrum[peak_index] / max(band_floor, 1e-12))
        confidence = min(1.0, max(0.0, (peak_ratio - 1.0) / 9.0))
        return PhaseEstimate(
            peak_frequency * 60.0, confidence, True, None,
            len(values), phase_std, peak_ratio,
        )


class MR60ESPAdapter:
    """Stateful ESP JSONL adapter for the Raspberry Pi integration layer."""

    def __init__(self, config_path: str | Path | None = None,
                 strict_provenance: bool = True) -> None:
        if config_path is None:
            config_path = Path(__file__).resolve().parents[1] / "config" / "mmwave_processing.json"
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        canonical = json.dumps(self.config, sort_keys=True, separators=(",", ":")).encode()
        self.config_hash = hashlib.sha256(canonical).hexdigest()[:16]
        self.strict_provenance = strict_provenance
        self.estimator = PhaseRateEstimator(self.config)
        self.last_sequence: int | None = None
        self.last_checksum_errors: int | None = None
        self.last_parse_errors: int | None = None
        self.presence_started_s: float | None = None

    def _unknown_packet(self, record: dict, state: str, reason: str,
                        communication_valid: bool, presence: bool | None,
                        timestamp_s: float | None) -> dict:
        return {
            "timestamp_s": timestamp_s,
            "mmwave_mr60": {
                "presence": None if presence is None else int(presence),
                "presence_raw": record.get("human_detected_raw"),
                "distance_cm": record.get("distance_cm_raw"),
                "breath_rpm": None,
                "breath_rpm_raw": record.get("breath_rate_raw"),
                "breath_source": "phase_fft_v1",
                "breath_valid": False,
                "breath_confidence": 0.0,
                "heart_bpm": None,
                "heart_bpm_raw": record.get("heart_rate_raw"),
                "heart_valid": False,
                "heart_verified": False,
                "heart_confidence": 0.0,
                "resp_phase": record.get("breath_phase"),
                "heart_phase": record.get("heart_phase"),
                "apnea": None,
                "apnea_verified": False,
                "state": state,
                "valid": False,
                "communication_valid": communication_valid,
                "stale": reason.endswith("STALE") or "TIMEOUT" in reason,
                "fault_reason": reason,
                "window_samples": len(self.estimator.values),
                "pi_config_hash": self.config_hash,
                "esp_schema_version": record.get("schema_version"),
                "esp_config_hash": record.get("config_hash"),
                "esp_firmware_version": record.get("firmware_version"),
                "sensor_firmware_version": record.get("sensor_firmware_version"),
            },
        }

    def process(self, record: dict[str, Any]) -> dict:
        timestamp_ms = record.get("ts_monotonic_ms")
        timestamp_s = float(timestamp_ms) / 1000.0 if _finite_number(timestamp_ms) else None
        presence_value = record.get("human_detected_stable", record.get("human_detected_raw"))
        presence = presence_value if isinstance(presence_value, bool) else None

        if self.strict_provenance:
            provenance_checks = (
                ("schema_version", "expected_esp_schema_version", "MMWAVE_SCHEMA_MISMATCH"),
                ("firmware_version", "expected_esp_firmware_version", "MMWAVE_FIRMWARE_MISMATCH"),
                ("config_hash", "expected_esp_config_hash", "MMWAVE_CONFIG_HASH_MISMATCH"),
            )
            for record_key, config_key, reason in provenance_checks:
                if record.get(record_key) != self.config[config_key]:
                    self.estimator.reset(reason)
                    return self._unknown_packet(
                        record, "FAULT", reason, False, presence, timestamp_s,
                    )

        sequence = record.get("seq")
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            self.estimator.reset("MMWAVE_SEQUENCE_INVALID")
            return self._unknown_packet(record, "FAULT", "MMWAVE_SEQUENCE_INVALID", False, presence, timestamp_s)
        if self.last_sequence is not None and sequence <= self.last_sequence:
            self.estimator.reset("MMWAVE_SEQUENCE_NON_MONOTONIC")
            return self._unknown_packet(record, "FAULT", "MMWAVE_SEQUENCE_NON_MONOTONIC", False, presence, timestamp_s)
        self.last_sequence = sequence

        checksum_errors = record.get("checksum_errors")
        parse_errors = record.get("parse_errors")
        counter_increased = (
            isinstance(checksum_errors, int) and self.last_checksum_errors is not None
            and checksum_errors > self.last_checksum_errors
        ) or (
            isinstance(parse_errors, int) and self.last_parse_errors is not None
            and parse_errors > self.last_parse_errors
        )
        if isinstance(checksum_errors, int):
            self.last_checksum_errors = checksum_errors
        if isinstance(parse_errors, int):
            self.last_parse_errors = parse_errors
        communication_valid = (
            record.get("uart_frame_ok") is True
            and record.get("checksum_ok") is True
            and not counter_increased
            and record.get("sensor_state") != "FAULT"
        )
        if not communication_valid or timestamp_s is None:
            self.estimator.reset("MMWAVE_UART_OR_PARSE_FAULT")
            return self._unknown_packet(record, "FAULT", "MMWAVE_UART_OR_PARSE_FAULT", False, presence, timestamp_s)

        if presence is not True:
            self.presence_started_s = None
            self.estimator.reset("MMWAVE_PRESENCE_NOT_DETECTED" if presence is False else "MMWAVE_PRESENCE_UNKNOWN")
            return self._unknown_packet(
                record, "UNKNOWN",
                "MMWAVE_PRESENCE_NOT_DETECTED" if presence is False else "MMWAVE_PRESENCE_UNKNOWN",
                True, presence, timestamp_s,
            )

        if self.presence_started_s is None:
            self.presence_started_s = timestamp_s
        distance = record.get("distance_cm_raw")
        distance_valid = (
            _finite_number(distance)
            and float(self.config["distance_min_cm"]) <= float(distance) <= float(self.config["distance_max_cm"])
        )
        phase_age = record.get("phase_age_ms")
        phase_fresh = _finite_number(phase_age) and float(phase_age) <= float(self.config["max_phase_age_ms"])
        if not distance_valid or not phase_fresh:
            reason = "MMWAVE_DISTANCE_INVALID" if not distance_valid else "MMWAVE_PHASE_STALE"
            self.estimator.reset(reason)
            return self._unknown_packet(record, "UNKNOWN", reason, True, presence, timestamp_s)
        if not self.estimator.push(record.get("breath_phase"), timestamp_s):
            return self._unknown_packet(
                record, "UNKNOWN", self.estimator.last_reason or "MMWAVE_PHASE_INVALID",
                True, presence, timestamp_s,
            )

        estimate = self.estimator.estimate()
        elapsed_presence_s = timestamp_s - self.presence_started_s
        warmup = elapsed_presence_s < float(self.config["warmup_seconds"]) or not estimate.valid
        if warmup:
            reason = "MMWAVE_WARMUP" if elapsed_presence_s < float(self.config["warmup_seconds"]) else (estimate.reason or "MMWAVE_WINDOW_NOT_READY")
            return self._unknown_packet(record, "WARMUP", reason, True, presence, timestamp_s)

        heart = record.get("heart_rate_raw")
        heart_age = record.get("heart_age_ms")
        heart_valid = (
            _finite_number(heart) and float(heart) > 0.0
            and _finite_number(heart_age)
            and float(heart_age) <= float(self.config["max_vital_age_ms"])
        )
        mmwave = {
            "presence": 1,
            "presence_raw": record.get("human_detected_raw"),
            "distance_cm": float(distance),
            "breath_rpm": estimate.rate_rpm,
            "breath_rpm_raw": record.get("breath_rate_raw"),
            "breath_source": "phase_fft_v1",
            "breath_valid": True,
            "breath_confidence": estimate.confidence,
            "breath_phase_std": estimate.phase_std,
            "breath_spectral_peak_ratio": estimate.spectral_peak_ratio,
            "heart_bpm": float(heart) if heart_valid else None,
            "heart_bpm_raw": record.get("heart_rate_raw"),
            "heart_valid": heart_valid,
            "heart_verified": bool(self.config["heart_verified"]),
            "heart_confidence": float(self.config["heart_confidence_cap"]) if heart_valid else 0.0,
            "heart_source": self.config["heart_source"],
            "resp_phase": record.get("breath_phase"),
            "heart_phase": record.get("heart_phase"),
            "apnea": None,
            "apnea_verified": False,
            "state": "VALID",
            "valid": True,
            "communication_valid": True,
            "stale": False,
            "fault_reason": None,
            "window_samples": estimate.window_samples,
            "pi_config_hash": self.config_hash,
            "esp_schema_version": record.get("schema_version"),
            "esp_config_hash": record.get("config_hash"),
            "esp_firmware_version": record.get("firmware_version"),
            "sensor_firmware_version": record.get("sensor_firmware_version"),
        }
        return {"timestamp_s": timestamp_s, "mmwave_mr60": mmwave}

    def process_json_line(self, line: str) -> dict:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            self.estimator.reset("MMWAVE_JSON_INVALID")
            return self._unknown_packet({}, "FAULT", "MMWAVE_JSON_INVALID", False, None, None)
        if not isinstance(record, dict):
            self.estimator.reset("MMWAVE_JSON_NOT_OBJECT")
            return self._unknown_packet({}, "FAULT", "MMWAVE_JSON_NOT_OBJECT", False, None, None)
        return self.process(record)

    def timeout_packet(self) -> dict:
        """Emit an explicit stale packet when the serial source times out."""
        self.estimator.reset("MMWAVE_SERIAL_TIMEOUT")
        self.presence_started_s = None
        return self._unknown_packet(
            {}, "UNKNOWN", "MMWAVE_SERIAL_TIMEOUT", False, None, None,
        )

    @staticmethod
    def to_json(packet: dict) -> str:
        return json.dumps(packet, ensure_ascii=False, separators=(",", ":"))


def estimate_to_dict(estimate: PhaseEstimate) -> dict:
    return asdict(estimate)
