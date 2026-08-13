#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
risk/risk_engine.py
SafeNest V5 Multi-Sensor Risk Fusion Engine

Formula:
  R = 100 * (0.35 * S1 + 0.35 * S2 + 0.15 * S3 + 0.15 * S4)

  - S1: mmWave Apnea Risk Score [0.0, 1.0]
  - S2: CO2 Occupancy / Elevation Risk Score [0.0, 1.0]
  - S3: PIR Motion / Long No Motion Risk Score [0.0, 1.0]
  - S4: Thermal-44 Fall Risk Score [0.0, 1.0]

Thresholds:
  - NORMAL: R < 30.0
  - CAUTION: 30.0 <= R < 60.0
  - DANGER: R >= 60.0
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import time
from typing import Dict, Any, List, Optional
import numpy as np

from inference.inference_result import InferenceResult, SafeNestRiskOutput
from risk.fallback import FallbackEngine, evaluate_sensor_health_and_risk


@dataclass
class V4RiskEvaluation:
    risk_score: float
    weighted_risk_score: float
    level: str
    emergency_override: bool
    sensor_scores: Dict[str, float]
    sensor_status: Dict[str, str]
    fallback_used: bool
    fallback_budget_met: bool
    fallback_reasons: List[str]
    system_status: str

    def to_dict(self) -> dict:
        return asdict(self)


class RiskEngineV4:
    """Legacy V4 Packet & Channel Evaluator for backward compatibility with pipeline tests."""
    def __init__(self, last_good_ttl_s: Optional[Dict[str, float]] = None):
        self.weights = {"S1": 0.35, "S2": 0.35, "S3": 0.15, "S4": 0.15}
        self.last_good_ttl_s = last_good_ttl_s or {"S1": 3.0, "S2": 5.0, "S3": 5.0, "S4": 3.0}
        self.last_good_scores: Dict[str, float] = {}
        self.last_good_timestamps: Dict[str, float] = {}

    def _normalize(self, val: Any) -> Optional[float]:
        if val is None or isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.number)):
            return None
        num = float(val)
        if np.isfinite(num):
            return min(1.0, max(0.0, num))
        return None

    def evaluate(
        self,
        sensor_scores: Dict[str, Any],
        raw_inputs: Optional[Dict[str, Any]] = None,
        timestamp_s: Optional[float] = None
    ) -> V4RiskEvaluation:
        now = timestamp_s if timestamp_s is not None else time.time()
        scores: Dict[str, float] = {}
        statuses: Dict[str, str] = {}
        fallback_used = False
        reasons: List[str] = []

        for channel in ["S1", "S2", "S3", "S4"]:
            raw_val = sensor_scores.get(channel)
            norm_val = self._normalize(raw_val)

            if norm_val is not None:
                scores[channel] = norm_val
                statuses[channel] = "OK"
                self.last_good_scores[channel] = norm_val
                self.last_good_timestamps[channel] = now
            else:
                last_ts = self.last_good_timestamps.get(channel)
                ttl = self.last_good_ttl_s.get(channel, 3.0)
                if last_ts is not None and (now - last_ts) <= ttl and channel in self.last_good_scores:
                    scores[channel] = self.last_good_scores[channel]
                    statuses[channel] = "FALLBACK_LAST_GOOD"
                    fallback_used = True
                    reasons.append(f"{channel}_LAST_GOOD")
                else:
                    fallback_used = True
                    statuses[channel] = "FALLBACK_RULE"
                    reasons.append(f"{channel}_RULE_FALLBACK")
                    if raw_inputs and channel == "S1" and self._normalize(raw_inputs.get("apnea")) == 1.0:
                        scores[channel] = 1.0
                    elif raw_inputs and channel == "S2" and self._normalize(raw_inputs.get("co2_ppm")) is not None and float(raw_inputs.get("co2_ppm", 0)) >= 2000:
                        scores[channel] = 1.0
                    elif raw_inputs and channel == "S3" and self._normalize(raw_inputs.get("pir_motion")) == 0.0:
                        scores[channel] = 1.0
                    elif raw_inputs and channel == "S4" and raw_inputs.get("thermal_class") == 2:
                        scores[channel] = 1.0
                    else:
                        scores[channel] = 0.0

        weighted = sum(self.weights[k] * scores[k] for k in ["S1", "S2", "S3", "S4"]) * 100.0
        emergency = False
        if raw_inputs and raw_inputs.get("thermal_class") == 2 and self._normalize(raw_inputs.get("thermal_confidence", 0)) is not None and float(raw_inputs.get("thermal_confidence", 0)) >= 0.8:
            emergency = True

        risk_score = 100.0 if emergency else weighted

        if risk_score >= 60.0:
            level = "DANGER"
        elif risk_score >= 30.0:
            level = "CAUTION"
        else:
            level = "NORMAL"

        sys_status = "DEGRADED" if fallback_used else "OK"

        return V4RiskEvaluation(
            risk_score=float(risk_score),
            weighted_risk_score=float(weighted),
            level=level,
            emergency_override=emergency,
            sensor_scores=scores,
            sensor_status=statuses,
            fallback_used=fallback_used,
            fallback_budget_met=True,
            fallback_reasons=reasons,
            system_status=sys_status
        )

    def evaluate_packet(
        self,
        packet: dict,
        s4: Optional[float] = None,
        s1: Optional[float] = None,
        s2: Optional[float] = None,
        s3: Optional[float] = None,
        emergency_override: bool = False
    ) -> V4RiskEvaluation:
        ts_val = packet.get("timestamp_s", time.time())
        ts = ts_val if self._normalize(ts_val) is not None else time.time()

        channel_scores = {}

        # S1
        if s1 is not None and self._normalize(s1) is not None:
            channel_scores["S1"] = s1
        else:
            mm = packet.get("mmwave_mr60", {})
            if isinstance(mm, dict) and mm.get("valid", True) and self._normalize(mm.get("breath_rpm")) is not None and self._normalize(mm.get("apnea")) is not None:
                channel_scores["S1"] = 1.0 if mm.get("apnea") == 1 else 0.0
            else:
                channel_scores["S1"] = None

        # S2
        if s2 is not None and self._normalize(s2) is not None:
            channel_scores["S2"] = s2
        else:
            c2 = packet.get("co2_scd40", {})
            if isinstance(c2, dict) and c2.get("valid", True) and self._normalize(c2.get("co2_ppm")) is not None:
                channel_scores["S2"] = 1.0 if float(c2.get("co2_ppm", 0)) > 1500 else 0.0
            else:
                channel_scores["S2"] = None

        # S3
        if s3 is not None and self._normalize(s3) is not None:
            channel_scores["S3"] = s3
        else:
            pr = packet.get("pir", {})
            if isinstance(pr, dict) and pr.get("valid", True) and self._normalize(pr.get("motion")) is not None:
                channel_scores["S3"] = 1.0 if pr.get("motion") == 0 else 0.0
            else:
                channel_scores["S3"] = None

        # S4
        channel_scores["S4"] = s4 if self._normalize(s4) is not None else None

        res = self.evaluate(channel_scores, raw_inputs=packet, timestamp_s=ts)
        if emergency_override or s4 == 1.0:
            res.emergency_override = True
            res.risk_score = 100.0
            res.level = "DANGER"

        return res


class SafeNestRiskEngine:
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        normal_max: float = 30.0,
        caution_max: float = 60.0,
        stale_sec: float | Dict[str, float] = 3.0,
    ):
        self.weights = weights if weights else {
            "mmwave": 0.35,  # S1
            "co2": 0.35,     # S2
            "pir": 0.15,     # S3
            "thermal": 0.15  # S4
        }
        total_w = sum(self.weights.values())
        if not np.isclose(total_w, 1.0, atol=1e-5):
            raise ValueError(f"Risk engine weights must sum to 1.0, got sum={total_w}")

        self.normal_max = normal_max
        self.caution_max = caution_max
        self.stale_sec = stale_sec
        self.fallback_engine = FallbackEngine(
            weights=self.weights,
            stale_sec=stale_sec,
        )

    def evaluate(
        self,
        sensor_results: Dict[str, Any],
        now: Optional[float] = None
    ) -> SafeNestRiskOutput:
        return evaluate_sensor_health_and_risk(
            sensor_results=sensor_results,
            weights=self.weights,
            stale_sec=self.stale_sec,
            normal_max=self.normal_max,
            caution_max=self.caution_max,
            now=now,
        )
