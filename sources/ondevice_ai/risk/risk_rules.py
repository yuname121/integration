#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
risk/risk_rules.py
SafeNest 융합 위험도 연산 및 Fault Injection 방어 순수 Python 모듈

[검수 3차 정밀 수정]
1. validate_timestamp 공통 검증 함수 수립 (non-numeric, non-finite, non-monotonic, NaN/Inf 차단)
2. evaluate_motion에 presence_confirmed 조건 연결 (presence 미확인 시 LONG_NO_MOTION 누적 차단 및 타이머 리셋)
3. mmWave TFLite 부재(TFLITE_MODEL_FILE_MISSING) 및 AI DEGRADED 시 system_status="DEGRADED" 강제 전파
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np


# Version 4.0 public fusion contract.  Keep this separate from the legacy
# five-feature evaluator below so older callers remain source compatible.
V4_SENSOR_WEIGHTS = {
    "S1": 0.35,  # mmWave apnea
    "S2": 0.35,  # CO2 enclosure risk
    "S3": 0.15,  # PIR motion risk
    "S4": 0.15,  # Thermal-44 posture risk
}


def normalize_risk_value(value: object) -> float:
    """Return a finite normalized risk value, rejecting ambiguous inputs."""
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if not isinstance(value, (int, float, np.number)):
        raise ValueError("risk value must be numeric")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("risk value must be finite")
    return min(1.0, max(0.0, number))


def classify_v4_risk(risk_score: float) -> str:
    """Apply the Version 4.0 boundaries exactly: 30 and 60 are inclusive."""
    score = float(risk_score)
    if not np.isfinite(score):
        raise ValueError("risk score must be finite")
    if score >= 60.0:
        return "DANGER"
    if score >= 30.0:
        return "CAUTION"
    return "NORMAL"


def calculate_v4_risk(sensor_scores: dict[str, object]) -> float:
    """Calculate R = 100 * (0.35S1 + 0.35S2 + 0.15S3 + 0.15S4)."""
    missing = set(V4_SENSOR_WEIGHTS).difference(sensor_scores)
    if missing:
        raise ValueError(f"missing v4 sensor scores: {sorted(missing)}")
    weighted = sum(
        weight * normalize_risk_value(sensor_scores[name])
        for name, weight in V4_SENSOR_WEIGHTS.items()
    )
    return min(100.0, max(0.0, 100.0 * weighted))


@dataclass
class RuleResult:
    score: float
    reasons: list[str] = field(default_factory=list)
    status: str = "NORMAL"
    emergency_override: bool = False


@dataclass
class SystemEvaluation:
    risk_score: float
    level: str
    is_emergency: bool
    reasons: list[str]
    sensor_status: dict[str, str]
    system_status: str


def validate_timestamp(value: float | int | None, previous: float | None = None) -> tuple[bool, str | None]:
    if value is None:
        return False, "SENSOR_TIMESTAMP_MISSING"
    if not isinstance(value, (int, float, np.number)):
        return False, "SENSOR_TIMESTAMP_INVALID_TYPE"
    if not np.isfinite(value):
        return False, "SENSOR_TIMESTAMP_NON_FINITE"
    if previous is not None and value < previous:
        return False, "SENSOR_TIMESTAMP_NON_MONOTONIC"
    return True, None


MMWAVE_DEGRADED_REASONS = {
    "MMWAVE_MODEL_INVOKE_ERROR",
    "MMWAVE_WINDOW_NOT_READY",
    "MMWAVE_WINDOW_STALE",
    "MMWAVE_PRESENCE_NOT_DETECTED",
    "MMWAVE_STREAM_GAP_TOO_LARGE",
    "MMWAVE_TIMESTAMP_NON_MONOTONIC",
    "MMWAVE_VALUE_NAN_OR_INF",
    "TFLITE_MODEL_FILE_MISSING",
    "MODEL_INPUT_CONTRACT_ERROR",
}


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.number))
        and bool(np.isfinite(value))
    )


class RiskRulesEvaluator:
    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            base_dir = Path(__file__).resolve().parent
            config_path = base_dir / "risk_config.json"
        else:
            config_path = Path(config_path)

        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.weights = self.config["risk"]["weights"]

        total_w = sum(self.weights.values())
        if abs(total_w - 1.0) > 1e-5:
            raise ValueError(f"Risk weights must sum to 1.0, got {total_w:.4f}")

        self.apnea_started_at: float | None = None
        self.apnea_timer_s: float = 0.0
        self.no_motion_started_at: float | None = None
        self.no_motion_timer_s: float = 0.0

    def evaluate_respiration(
        self,
        breath_rpm: float | None,
        apnea: int | None,
        ai_class: int | None = None,
        valid: bool = True,
        dt_s: float | None = None,
        sample_timestamp: float | None = None
    ) -> RuleResult:
        if not valid or not _is_finite_number(breath_rpm) or apnea not in (0, 1, None):
            self.apnea_started_at = None
            self.apnea_timer_s = 0.0
            return RuleResult(score=0.5, reasons=["RESP_SENSOR_FAULT"], status="FAULT")

        if sample_timestamp is not None:
            ts_valid, ts_reason = validate_timestamp(sample_timestamp, previous=self.apnea_started_at)
            if not ts_valid:
                self.apnea_started_at = None
                self.apnea_timer_s = 0.0
                return RuleResult(score=0.5, reasons=[ts_reason or "SENSOR_TIMESTAMP_INVALID"], status="FAULT")

        if apnea == 1:
            return RuleResult(
                score=1.0,
                reasons=["EMERGENCY_APNEA"],
                status="CRITICAL",
                emergency_override=True
            )

        is_apnea_candidate = (breath_rpm <= 0.5 or ai_class == 2)
        if is_apnea_candidate:
            if sample_timestamp is not None:
                if self.apnea_started_at is None:
                    self.apnea_started_at = sample_timestamp
                elapsed = sample_timestamp - self.apnea_started_at
            elif dt_s is not None:
                self.apnea_timer_s += dt_s
                elapsed = self.apnea_timer_s
            else:
                elapsed = 2.0

            apnea_confirm_threshold = self.config["respiration"].get("apnea_confirm_seconds", 2.0)
            if elapsed + 1e-9 >= apnea_confirm_threshold:
                return RuleResult(
                    score=1.0,
                    reasons=["EMERGENCY_APNEA"],
                    status="CRITICAL",
                    emergency_override=True
                )
        else:
            self.apnea_started_at = None
            self.apnea_timer_s = 0.0

        min_rpm = self.config["respiration"]["normal_min_rpm"]
        max_rpm = self.config["respiration"]["normal_max_rpm"]

        if breath_rpm < min_rpm or breath_rpm > max_rpm or ai_class == 1:
            return RuleResult(score=0.75, reasons=["ABNORMAL_RESPIRATION_RPM"], status="CAUTION")

        return RuleResult(score=0.0, reasons=[], status="NORMAL")

    def evaluate_environment(
        self,
        co2_ppm: float | None,
        slope_ppm_per_min: float | None = 0.0,
        valid: bool = True
    ) -> RuleResult:
        if not valid or not _is_finite_number(co2_ppm):
            return RuleResult(score=0.2, reasons=["CO2_SENSOR_FAULT"], status="DEGRADED")

        reasons = []
        warn_ppm = self.config["co2"]["warning_ppm"]
        danger_ppm = self.config["co2"]["danger_ppm"]
        slope_warn = self.config["co2"]["slope_warning_ppm_per_min"]

        if co2_ppm >= danger_ppm:
            reasons.append("HIGH_CO2_DANGER")
        elif co2_ppm >= warn_ppm:
            reasons.append("HIGH_CO2_WARNING")

        if _is_finite_number(slope_ppm_per_min) and slope_ppm_per_min >= slope_warn:
            reasons.append("FAST_CO2_RISE")

        co2_norm = min(1.0, max(0.0, (co2_ppm - 500.0) / 2000.0))
        score = min(1.0, co2_norm)

        status = "DANGER" if "HIGH_CO2_DANGER" in reasons else ("CAUTION" if reasons else "NORMAL")
        return RuleResult(score=score, reasons=reasons, status=status)

    def evaluate_vital_hr(
        self,
        heart_bpm: float | None,
        valid: bool = True
    ) -> RuleResult:
        if not valid or not _is_finite_number(heart_bpm):
            return RuleResult(score=0.0, reasons=["HR_SENSOR_MISSING"], status="DEGRADED")

        if heart_bpm > 110.0:
            score = min(1.0, (heart_bpm - 105.0) / 25.0)
            return RuleResult(score=score, reasons=["TACHYCARDIA_HIGH_HR"], status="CAUTION")
        elif heart_bpm < 55.0:
            score = min(1.0, (60.0 - heart_bpm) / 25.0)
            return RuleResult(score=score, reasons=["BRADYCARDIA_LOW_HR"], status="CAUTION")

        return RuleResult(score=0.0, reasons=[], status="NORMAL")

    def evaluate_posture(
        self,
        thermal_fall_class: int | None,
        confidence: float = 1.0,
        valid: bool = True
    ) -> RuleResult:
        if not valid or thermal_fall_class is None:
            return RuleResult(score=0.0, reasons=["THERMAL_SENSOR_MISSING"], status="DEGRADED")

        if thermal_fall_class == 2 and confidence >= 0.8:
            return RuleResult(
                score=1.0,
                reasons=["EMERGENCY_FALL"],
                status="CRITICAL",
                emergency_override=True
            )

        return RuleResult(score=0.0, reasons=[], status="NORMAL")

    def evaluate_motion(
        self,
        pir_motion: int | None,
        presence_confirmed: bool = True,
        valid: bool = True,
        dt_s: float | None = None,
        sample_timestamp: float | None = None
    ) -> RuleResult:
        if not valid or pir_motion not in (0, 1):
            self.no_motion_started_at = None
            self.no_motion_timer_s = 0.0
            return RuleResult(score=0.0, reasons=["PIR_SENSOR_MISSING"], status="DEGRADED")

        # presence 미확인 시 장시간 무움직임 타이머 누적 금지 및 타이머 초기화
        if not presence_confirmed:
            self.no_motion_started_at = None
            self.no_motion_timer_s = 0.0
            return RuleResult(score=0.0, reasons=["PRESENCE_NOT_CONFIRMED"], status="NORMAL")

        if sample_timestamp is not None:
            ts_valid, ts_reason = validate_timestamp(sample_timestamp, previous=self.no_motion_started_at)
            if not ts_valid:
                self.no_motion_started_at = None
                self.no_motion_timer_s = 0.0
                return RuleResult(score=0.0, reasons=[ts_reason or "PIR_TIMESTAMP_INVALID"], status="DEGRADED")

        if pir_motion == 0:
            if sample_timestamp is not None:
                if self.no_motion_started_at is None:
                    self.no_motion_started_at = sample_timestamp
                elapsed = sample_timestamp - self.no_motion_started_at
            elif dt_s is not None:
                self.no_motion_timer_s += dt_s
                elapsed = self.no_motion_timer_s
            else:
                elapsed = 15.0

            threshold = self.config["motion"].get("pir_no_motion_seconds", 15.0)
            if elapsed + 1e-9 >= threshold:
                return RuleResult(score=1.0, reasons=["LONG_NO_MOTION"], status="CAUTION")
            return RuleResult(score=0.5, reasons=["NO_MOTION_DETECTED"], status="NORMAL")
        else:
            self.no_motion_started_at = None
            self.no_motion_timer_s = 0.0
            return RuleResult(score=0.0, reasons=[], status="NORMAL")

    def evaluate_system(
        self,
        respiration_eval: RuleResult,
        environment_eval: RuleResult,
        vital_hr_eval: RuleResult,
        posture_eval: RuleResult,
        motion_eval: RuleResult,
        all_sensors_missing: bool = False
    ) -> SystemEvaluation:
        if all_sensors_missing:
            return SystemEvaluation(
                risk_score=0.0,
                level="FAULT",
                is_emergency=False,
                reasons=["ALL_SENSORS_MISSING", "RESP_SENSOR_FAULT", "THERMAL_SENSOR_MISSING", "CO2_SENSOR_FAULT"],
                sensor_status={"mmwave": "MISSING", "co2": "MISSING", "thermal": "MISSING", "pir": "MISSING", "vital_hr": "MISSING"},
                system_status="FAULT"
            )

        reasons = []
        for eval_res in (respiration_eval, environment_eval, vital_hr_eval, posture_eval, motion_eval):
            for r in eval_res.reasons:
                if r not in reasons:
                    reasons.append(r)

        is_emergency = respiration_eval.emergency_override or posture_eval.emergency_override

        # mmWave DEGRADED 판정: MMWAVE_DEGRADED_REASONS 집합 매칭
        mmwave_degraded = (
            "RESP_SENSOR_FAULT" in respiration_eval.reasons
            or any(r in MMWAVE_DEGRADED_REASONS for r in respiration_eval.reasons)
        )

        sensor_status = {
            "mmwave": "FAULT" if "RESP_SENSOR_FAULT" in respiration_eval.reasons else ("DEGRADED" if mmwave_degraded else "OK"),
            "co2": "DEGRADED" if ("CO2_SENSOR_FAULT" in environment_eval.reasons or "CO2_MODEL_INVOKE_ERROR" in environment_eval.reasons) else "OK",
            "thermal": "DEGRADED" if ("THERMAL_SENSOR_MISSING" in posture_eval.reasons or "THERMAL_MODEL_INVOKE_ERROR" in posture_eval.reasons) else "OK",
            "pir": "DEGRADED" if ("PIR_SENSOR_MISSING" in motion_eval.reasons or "PIR_TIMESTAMP_INVALID" in motion_eval.reasons) else "OK",
            "vital_hr": "DEGRADED" if "HR_SENSOR_MISSING" in vital_hr_eval.reasons else "OK",
        }

        has_fault = any(st in ("FAULT", "DEGRADED", "MISSING") for st in sensor_status.values())
        system_status = "DEGRADED" if has_fault else "OK"

        if is_emergency:
            return SystemEvaluation(
                risk_score=100.0,
                level="DANGER",
                is_emergency=True,
                reasons=reasons,
                sensor_status=sensor_status,
                system_status=system_status
            )

        raw_r = 100.0 * (
            self.weights["respiration"] * respiration_eval.score +
            self.weights["environment"] * environment_eval.score +
            self.weights["vital_hr"] * vital_hr_eval.score +
            self.weights["posture_fall"] * posture_eval.score +
            self.weights["motion"] * motion_eval.score
        )
        raw_r = min(100.0, max(0.0, raw_r))

        if raw_r >= self.config["risk"]["caution_max"]:
            level = "DANGER"
        elif raw_r >= self.config["risk"]["normal_max"]:
            level = "CAUTION"
        else:
            level = "NORMAL"

        return SystemEvaluation(
            risk_score=raw_r,
            level=level,
            is_emergency=False,
            reasons=reasons,
            sensor_status=sensor_status,
            system_status=system_status
        )
