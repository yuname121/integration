#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
risk/fallback.py
SafeNest V5 Safe Fallback Handler for Sensor Faults & Missing Models
"""

from __future__ import annotations
import logging
import time
from typing import Dict, Any, List
import numpy as np

from inference.inference_result import InferenceResult, SafeNestRiskOutput

logger = logging.getLogger(__name__)

CANONICAL_SENSOR_ORDER = ["mmwave", "co2", "pir", "thermal"]


def evaluate_sensor_health_and_risk(
    sensor_results: Dict[str, Any],
    weights: Dict[str, float] | None = None,
    stale_sec: float | Dict[str, float] = 3.0,
    normal_max: float = 30.0,
    caution_max: float = 60.0,
    now: float | None = None,
) -> SafeNestRiskOutput:
    t0 = time.perf_counter()
    current_time = now if now is not None else time.time()

    if weights is None:
        raw_weights = {
            "mmwave": 0.35,
            "co2": 0.35,
            "pir": 0.15,
            "thermal": 0.15,
        }
    else:
        raw_weights = {}
        for k, v in weights.items():
            canonical_k = "thermal" if k == "thermal44" else k
            raw_weights[canonical_k] = v

    valid_sensors: List[str] = []
    invalid_sensors: List[str] = []
    stale_sensors: List[str] = []
    component_scores: Dict[str, float | None] = {}
    reasons: List[str] = []
    # Copy original sensor_results into sensor_dicts to preserve exact caller keys (e.g. thermal44)
    sensor_dicts: Dict[str, dict] = {
        k: (v.to_dict() if isinstance(v, InferenceResult) else (v if isinstance(v, dict) else {"valid": False}))
        for k, v in sensor_results.items()
    }

    for sensor_key in CANONICAL_SENSOR_ORDER:
        raw_key = sensor_key
        result = sensor_results.get(sensor_key)
        if result is None and sensor_key == "thermal":
            if "thermal44" in sensor_results:
                raw_key = "thermal44"
                result = sensor_results.get("thermal44")
            elif weights and "thermal44" in weights:
                raw_key = "thermal44"

        if result is None:
            invalid_sensors.append(sensor_key)
            component_scores[sensor_key] = None
            reasons.append(f"{sensor_key.upper()}_MISSING")
            if sensor_key == "thermal":
                reasons.append("THERMAL44_MISSING")
            if raw_key not in sensor_dicts:
                sensor_dicts[raw_key] = {"valid": False, "error": "MISSING"}
            continue

        if isinstance(result, InferenceResult):
            sensor_dicts[raw_key] = result.to_dict()
            is_valid = result.valid and result.error is None
            score = result.score
            state = result.state
            ts = result.timestamp
        elif isinstance(result, dict):
            sensor_dicts[raw_key] = result
            is_valid = bool(result.get("valid", True)) and not result.get("error")
            score = result.get("score")
            state = result.get("state", "UNKNOWN")
            ts = result.get("timestamp", current_time)
        else:
            invalid_sensors.append(sensor_key)
            component_scores[sensor_key] = None
            reasons.append(f"{sensor_key.upper()}_INVALID_FORMAT")
            if sensor_key == "thermal":
                reasons.append("THERMAL44_INVALID_FORMAT")
            sensor_dicts[raw_key] = {"valid": False, "error": "INVALID_FORMAT"}
            continue

        # Check score validity
        valid_score = (
            is_valid
            and score is not None
            and not isinstance(score, (bool, np.bool_))
            and isinstance(score, (int, float, np.number))
            and np.isfinite(score)
            and (0.0 <= float(score) <= 1.0)
        )

        if not valid_score:
            invalid_sensors.append(sensor_key)
            component_scores[sensor_key] = None
            reasons.append(f"{sensor_key.upper()}_{state}")
            if sensor_key == "thermal":
                reasons.append(f"THERMAL44_{state}")
            continue

        # Check timestamp staleness
        valid_timestamp = (
            not isinstance(ts, (bool, np.bool_))
            and isinstance(ts, (int, float, np.number))
            and np.isfinite(ts)
            and float(ts) >= 0.0
        )
        if not valid_timestamp:
            invalid_sensors.append(sensor_key)
            component_scores[sensor_key] = None
            reasons.append(f"{sensor_key.upper()}_INVALID_TIMESTAMP")
            if sensor_key == "thermal":
                reasons.append("THERMAL44_INVALID_TIMESTAMP")
            continue

        ttl = stale_sec.get(sensor_key, stale_sec.get(raw_key, 3.0)) if isinstance(stale_sec, dict) else stale_sec
        if (current_time - ts) > ttl:
            stale_sensors.append(sensor_key)
            component_scores[sensor_key] = None
            reasons.append(f"{sensor_key.upper()}_STALE_TIMESTAMP")
            if sensor_key == "thermal":
                reasons.append("THERMAL44_STALE_TIMESTAMP")
            continue

        valid_sensors.append(sensor_key)
        component_scores[sensor_key] = float(score)

    invalid_sensors.sort(key=lambda s: CANONICAL_SENSOR_ORDER.index(s))
    stale_sensors.sort(key=lambda s: CANONICAL_SENSOR_ORDER.index(s))

    if not invalid_sensors and not stale_sensors:
        system_health = "HEALTHY"
        degraded_mode = False
    elif valid_sensors:
        system_health = "DEGRADED"
        degraded_mode = True
    else:
        system_health = "FAILED"
        degraded_mode = True

    if system_health == "FAILED":
        risk_score = None
        risk_level = None
        is_emergency = False
        reasons.insert(0, "ALL_SENSORS_FAULT_OR_MISSING")
    else:
        is_emergency = False
        if "thermal" in valid_sensors and component_scores["thermal"] == 1.0:
            is_emergency = True
            reasons.insert(0, "EMERGENCY_HUMAN_FALL")
        elif "mmwave" in valid_sensors and component_scores["mmwave"] == 1.0:
            is_emergency = True
            reasons.insert(0, "EMERGENCY_HARDWARE_APNEA")

        if is_emergency:
            risk_score = 100.0
            risk_level = "DANGER"
        else:
            total_valid_weight = sum(raw_weights[k] for k in valid_sensors)
            if total_valid_weight > 0:
                weighted_sum = sum(
                    component_scores[k] * (raw_weights[k] / total_valid_weight)
                    for k in valid_sensors
                )
                r_score = float(weighted_sum * 100.0)
                r_score = min(max(r_score, 0.0), 100.0)
                risk_score = r_score

                if risk_score >= caution_max:
                    risk_level = "DANGER"
                elif risk_score >= normal_max:
                    risk_level = "CAUTION"
                else:
                    risk_level = "NORMAL"
            else:
                system_health = "FAILED"
                risk_score = None
                risk_level = None
                degraded_mode = True

    if system_health == "HEALTHY" and not reasons:
        reasons.append("SYSTEM_HEALTHY")

    logger.info(
        "Risk Level: %s | System Health: %s | Degraded Mode: %s | Invalid Sensors: %s | Stale Sensors: %s",
        risk_level, system_health, degraded_mode, invalid_sensors, stale_sensors
    )

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return SafeNestRiskOutput(
        timestamp=current_time,
        risk_score=risk_score,
        risk_level=risk_level,
        system_health=system_health,
        degraded_mode=degraded_mode,
        invalid_sensors=invalid_sensors,
        stale_sensors=stale_sensors,
        component_scores=component_scores,
        is_emergency=is_emergency,
        reasons=reasons,
        sensors=sensor_dicts,
        metadata={
            "schema_version": "5.0",
            "calc_latency_ms": latency_ms,
            "valid_sensors": valid_sensors,
            "invalid_sensors": invalid_sensors,
            "stale_sensors": stale_sensors,
            "stale_sec": (
                dict(stale_sec)
                if isinstance(stale_sec, dict)
                else float(stale_sec)
            ),
        }
    )


class FallbackEngine:
    def __init__(
        self,
        weights: Dict[str, float] | None = None,
        stale_sec: float | Dict[str, float] = 3.0
    ):
        if weights is None:
            self.weights = {
                "mmwave": 0.35,  # S1
                "co2": 0.35,     # S2
                "pir": 0.15,     # S3
                "thermal": 0.15  # S4
            }
        else:
            self.weights = weights
        self.stale_sec = stale_sec

    def evaluate_fallback(
        self,
        sensor_results: Dict[str, Any],
        now: float | None = None
    ) -> SafeNestRiskOutput:
        return evaluate_sensor_health_and_risk(
            sensor_results=sensor_results,
            weights=self.weights,
            stale_sec=self.stale_sec,
            now=now,
        )
