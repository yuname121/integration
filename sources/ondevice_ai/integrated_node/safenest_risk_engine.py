#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
safenest_risk_engine.py
Legacy compatibility engine retained for V4 tests and demos.

V5 production must use ``integrated_node/run_node.py`` with
``risk/risk_engine.py``. New sensor providers must not integrate here.

SafeNest 생체연동 융합 위험도 연산 가상노드 엔진 (Risk Engine Gateway v6.2)

[검수 3차 정밀 연산]
1. PIR motion 연산 시 presence_confirmed (mmWave/Thermal 융합 presence) 전달
2. mmWave TFLite 부재(TFLITE_MODEL_FILE_MISSING) 및 AI DEGRADED 시 system_status="DEGRADED" 강제 전파
3. validate_timestamp 지원 및 time.time() sentinel 버그 완전 수정
"""

import os
import sys
import json
import collections
import time
import numpy as np

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from inference.model_registry import ModelRegistry
from adapters.mmwave_stream_adapter import MMWaveStreamAdapter
from risk.risk_rules import RiskRulesEvaluator
from risk.risk_engine import RiskEngineV4

class MMWaveClutterCalibrator:
    """FMCW 60GHz 복소 rFFT Clutter Subtraction 엔진"""
    def __init__(self, num_bins=64):
        self.num_bins = num_bins
        self.clutter_map = np.zeros(num_bins, dtype=np.complex128)
        self.is_calibrated = False

    def calibrate_background(self, rfft_frames):
        self.clutter_map = np.mean(rfft_frames, axis=0)
        self.is_calibrated = True
        return self.clutter_map

    def filter_frame(self, raw_rfft_frame):
        if not self.is_calibrated:
            return raw_rfft_frame
        return raw_rfft_frame - self.clutter_map

def find_adaptive_chest_bin(rfft_buffer, distances, fs=10.0):
    """0.1~0.5Hz 호흡 대역 PSD 에너지 기반 적응형 흉부 거리 빈 (r*) 추적 수식"""
    if len(rfft_buffer) < 10:
        return 5
    valid_idx = np.where((distances >= 0.5) & (distances <= 3.0))[0]
    if len(valid_idx) == 0:
        valid_idx = np.arange(2, 12)
        
    best_bin = valid_idx[0]
    max_resp_energy = -1.0
    
    for r in valid_idx:
        phase = np.unwrap(np.angle(rfft_buffer[:, r]))
        detrended = phase - np.mean(phase)
        fft_vals = np.fft.rfft(detrended)
        freqs = np.fft.rfftfreq(len(phase), d=1.0/fs)
        
        resp_mask = (freqs >= 0.1) & (freqs <= 0.5)
        resp_energy = np.sum(np.abs(fft_vals[resp_mask]) ** 2)
        
        if resp_energy > max_resp_energy:
            max_resp_energy = resp_energy
            best_bin = int(r)
            
    return best_bin

class SafeNestRiskEngine:
    def __init__(self, manifest_path="models/model_manifest.json"):
        # 1. ModelRegistry 중앙 로드
        self.registry = ModelRegistry(project_root=project_root, manifest_path=manifest_path)
        self.thermal_runner = self.registry.thermal
        self.co2_runner = self.registry.co2
        self.mmwave_runner = self.registry.mmwave

        # 2. RiskRulesEvaluator 로드
        self.rules_evaluator = RiskRulesEvaluator()
        self.v4_engine = RiskEngineV4()

        self.mmwave_stream_adapter = MMWaveStreamAdapter(window_samples=300, sample_rate_hz=10.0)

        # mmWave Clutter 및 파이프라인 버퍼
        self.mmwave_calibrator = MMWaveClutterCalibrator(num_bins=64)
        self.mmwave_rfft_history = collections.deque(maxlen=30)
        self.distances = np.linspace(0.0, 5.0, 64)
        self.active_chest_bin = 5

        # CO2 타임스탬프 히스토리 (timestamp, ppm)
        self.co2_history = collections.deque(maxlen=30)
        self.risk_history = collections.deque(maxlen=6)
        self.curr_smoothed_r = 0.0
        self.prev_status = "NORMAL"

    @staticmethod
    def _finite_number(value):
        return (
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.number))
            and bool(np.isfinite(value))
        )

    def calculate_quality_gate(self, packet: dict):
        """Quality Gate: 센서 결측 및 범위 검사를 통한 신뢰도 품질 지수 q_i (0.0 ~ 1.0) 연산"""
        thermal = packet.get("thermal_80x62")
        q_thermal = 1.0 if (
            isinstance(thermal, np.ndarray)
            and thermal.shape == (62, 80)
            and np.all(np.isfinite(thermal))
            and packet.get("thermal_valid", True) is not False
        ) else 0.0
        
        co2_data = packet.get("co2_scd40", {})
        co2_ppm = co2_data.get("co2_ppm") if isinstance(co2_data, dict) else None
        co2_valid = (
            isinstance(co2_data, dict)
            and co2_data.get("valid", True) is not False
            and not co2_data.get("fault_reason")
            and co2_data.get("stale", False) is not True
        )
        q_co2 = 1.0 if (
            co2_valid and self._finite_number(co2_ppm) and 300 <= co2_ppm <= 10000
        ) else 0.2
        
        mmwave_data = packet.get("mmwave_mr60", {})
        q_mmwave = 1.0 if (
            isinstance(mmwave_data, dict)
            and mmwave_data.get("valid", True) is not False
            and not mmwave_data.get("fault_reason")
            and mmwave_data.get("stale", False) is not True
            and self._finite_number(mmwave_data.get("breath_rpm"))
            and mmwave_data.get("apnea") in (0, 1)
        ) else 0.0
        
        pir_data = packet.get("pir", {})
        q_pir = 1.0 if (
            isinstance(pir_data, dict)
            and pir_data.get("valid", True) is not False
            and not pir_data.get("fault_reason")
            and pir_data.get("stale", False) is not True
            and pir_data.get("motion") in (0, 1)
        ) else 0.5
        
        return {
            "thermal": q_thermal,
            "co2": q_co2,
            "mmwave": q_mmwave,
            "pir": q_pir
        }

    def evaluate_risk(self, packet: dict):
        sample_ts = packet.get("timestamp_s", packet.get("timestamp", time.time()))
        if not self._finite_number(sample_ts):
            sample_ts = time.time()
        q_gate = self.calculate_quality_gate(packet)

        # ⭐ 빈 패킷 / 센서 미수신 특수 검사
        all_missing = (q_gate["thermal"] == 0.0 and q_gate["mmwave"] == 0.0 and q_gate["co2"] <= 0.2 and q_gate["pir"] <= 0.5)
        if all_missing or len(packet) == 0:
            eval_res = self.rules_evaluator.evaluate_system(
                respiration_eval=self.rules_evaluator.evaluate_respiration(None, None, valid=False, sample_timestamp=sample_ts),
                environment_eval=self.rules_evaluator.evaluate_environment(None, valid=False),
                vital_hr_eval=self.rules_evaluator.evaluate_vital_hr(None, valid=False),
                posture_eval=self.rules_evaluator.evaluate_posture(None, valid=False),
                motion_eval=self.rules_evaluator.evaluate_motion(None, presence_confirmed=False, valid=False, sample_timestamp=sample_ts),
                all_sensors_missing=True
            )
            v4_fusion = self.v4_engine.evaluate_packet(packet, None)
            return {
                "risk_score": 0.0,
                "status_str": "FAULT",
                "status_code": -1,
                "is_emergency": False,
                "reasons": eval_res.reasons,
                "sensor_quality": q_gate,
                "system_status": "FAULT",
                "v4_fusion": v4_fusion.to_dict(),
                "legacy_fusion": {
                    "risk_score": 0.0, "level": "FAULT", "is_emergency": False,
                },
                "active_chest_bin": self.active_chest_bin,
                "active_chest_dist_m": float(self.distances[self.active_chest_bin]),
                "derived_metrics": {
                    "co2_slope_ppm_per_min": None,
                    "presence_confirmed": False,
                    "mmwave_window_samples": len(self.mmwave_stream_adapter.buffer),
                    "mmwave_window_ready": False,
                },
                "model_meta": {
                    "thermal": {
                        "source": "none", "version": "0.0.0", "latency_ms": 0.0,
                        "ai_status": "MISSING", "class_index": None,
                        "class_name": None, "confidence": None, "probabilities": None
                    },
                    "co2": {
                        "source": "none", "version": "0.0.0", "latency_ms": 0.0,
                        "ai_status": "MISSING", "class_index": None,
                        "class_name": None, "confidence": None, "probabilities": None
                    },
                    "mmwave": {
                        "source": "none", "version": "0.0.0", "latency_ms": 0.0,
                        "ai_status": "MISSING", "class_index": None,
                        "class_name": None, "confidence": None, "probabilities": None,
                        "fallback_used": False, "fallback_reason": "ALL_SENSORS_MISSING"
                    }
                }
            }

        # 1. Thermal Posture (S4)
        thermal_grid = packet.get("thermal_80x62")
        thermal_class, thermal_conf, thermal_lat = None, 0.0, 0.0
        thermal_pred_obj = None
        thermal_ai_status = "NOT_RUN"
        thermal_fallback_reason = None

        if self.thermal_runner is not None and q_gate["thermal"] == 1.0:
            try:
                thermal_pred_obj = self.thermal_runner.predict(thermal_grid)
                thermal_class = thermal_pred_obj.class_index
                thermal_conf = thermal_pred_obj.confidence
                thermal_lat = thermal_pred_obj.latency_ms
                thermal_ai_status = "OK"
            except Exception as e:
                print(f"⚠️ [RiskEngine] Thermal invoke exception: {e}")
                q_gate["thermal"] = 0.0
                thermal_ai_status = "DEGRADED"
                thermal_fallback_reason = "THERMAL_MODEL_INVOKE_ERROR"

        posture_eval = self.rules_evaluator.evaluate_posture(thermal_class, thermal_conf, valid=(q_gate["thermal"] == 1.0))
        if thermal_fallback_reason and thermal_fallback_reason not in posture_eval.reasons:
            posture_eval.reasons.append(thermal_fallback_reason)

        # 2. CO2 Environment (S2)
        co2_data = packet.get("co2_scd40", {})
        co2_ppm = co2_data.get("co2_ppm") if isinstance(co2_data, dict) else None
        co2_lat = 0.0
        co2_pred_obj = None
        co2_ai_status = "NOT_RUN"
        co2_fallback_reason = None

        if co2_ppm is not None and q_gate["co2"] == 1.0:
            humidity = co2_data.get("humidity", 45.0)
            self.co2_history.append((sample_ts, co2_ppm))
            
            if len(self.co2_history) > 1:
                elapsed_min = (self.co2_history[-1][0] - self.co2_history[0][0]) / 60.0
                if elapsed_min > 0:
                    co2_slope_per_min = (self.co2_history[-1][1] - self.co2_history[0][1]) / elapsed_min
                else:
                    co2_slope_per_min = 0.0
            else:
                co2_slope_per_min = 0.0
            
            if self.co2_runner is not None:
                try:
                    co2_pred_obj = self.co2_runner.predict(co2_slope_per_min, humidity, co2_ppm)
                    co2_lat = co2_pred_obj.latency_ms
                    co2_ai_status = "OK"
                except Exception as e:
                    print(f"⚠️ [RiskEngine] CO2 invoke exception: {e}")
                    co2_ai_status = "DEGRADED"
                    co2_fallback_reason = "CO2_MODEL_INVOKE_ERROR"
            
            env_eval = self.rules_evaluator.evaluate_environment(co2_ppm, co2_slope_per_min, valid=True)
        else:
            co2_ppm, co2_slope_per_min = None, 0.0
            env_eval = self.rules_evaluator.evaluate_environment(None, valid=False)

        if co2_fallback_reason and co2_fallback_reason not in env_eval.reasons:
            env_eval.reasons.append(co2_fallback_reason)

        # 3. mmWave Respiration (S1) & Vital HR (S0)
        mmwave_data = packet.get("mmwave_mr60", {})
        mm_class, mm_lat = None, 0.0
        mm_pred_obj = None
        mm_ai_status = "NOT_RUN"
        mm_fallback_reason = None
        presence_confirmed = False

        if isinstance(mmwave_data, dict) and "breath_rpm" in mmwave_data and q_gate["mmwave"] == 1.0:
            apnea = mmwave_data.get("apnea", 0)
            breath_rpm = mmwave_data.get("breath_rpm", 16.0)
            heart_bpm = mmwave_data.get("heart_bpm")
            resp_phase_val = mmwave_data.get("resp_phase")
            presence_val = mmwave_data.get("presence", 1)
            presence_confirmed = (presence_val == 1)

            if resp_phase_val is not None:
                push_res = self.mmwave_stream_adapter.push_sample(resp_phase_val, timestamp_s=sample_ts, presence=presence_val)
                if not push_res.accepted:
                    mm_fallback_reason = push_res.reason
                    mm_ai_status = "DEGRADED"

            if self.mmwave_runner is not None and mm_fallback_reason is None:
                if self.mmwave_stream_adapter.is_stale(current_time_s=sample_ts):
                    mm_fallback_reason = "MMWAVE_WINDOW_STALE"
                    mm_ai_status = "DEGRADED"
                elif not self.mmwave_stream_adapter.is_ready():
                    mm_fallback_reason = "MMWAVE_WINDOW_NOT_READY"
                    mm_ai_status = "DEGRADED"
                else:
                    window = self.mmwave_stream_adapter.get_window(current_time_s=sample_ts)
                    if window is not None:
                        try:
                            mm_pred_obj = self.mmwave_runner.predict(window)
                            mm_class, mm_lat = mm_pred_obj.class_index, mm_pred_obj.latency_ms
                            mm_ai_status = "OK" if not mm_pred_obj.fallback_used else "DEGRADED"
                            if mm_pred_obj.fallback_reason:
                                mm_fallback_reason = mm_pred_obj.fallback_reason
                        except Exception as e:
                            print(f"⚠️ [RiskEngine] mmWave invoke exception: {e}")
                            mm_ai_status = "DEGRADED"
                            mm_fallback_reason = "MMWAVE_MODEL_INVOKE_ERROR"

            # TFLite 미존재 및 fallback 일 때도 MMWAVE_DEGRADED 이유 명시
            if self.mmwave_runner is not None and not self.mmwave_runner.model_file_exists:
                mm_ai_status = "DEGRADED"
                if not mm_fallback_reason:
                    mm_fallback_reason = "TFLITE_MODEL_FILE_MISSING"

            raw_rfft = mmwave_data.get("rfft_frame")
            if raw_rfft is not None:
                if not self.mmwave_calibrator.is_calibrated and len(self.mmwave_rfft_history) >= 15:
                    self.mmwave_calibrator.calibrate_background(np.array(self.mmwave_rfft_history))
                filtered_rfft = self.mmwave_calibrator.filter_frame(raw_rfft)
                self.mmwave_rfft_history.append(filtered_rfft)
                if len(self.mmwave_rfft_history) >= 10:
                    self.active_chest_bin = find_adaptive_chest_bin(
                        np.array(self.mmwave_rfft_history), self.distances, fs=10.0
                    )

            resp_eval = self.rules_evaluator.evaluate_respiration(breath_rpm, apnea, mm_class, valid=True, sample_timestamp=sample_ts)
            hr_valid = (heart_bpm is not None and np.isfinite(heart_bpm) and 20.0 <= heart_bpm <= 240.0)
            hr_eval = self.rules_evaluator.evaluate_vital_hr(heart_bpm, valid=hr_valid)
        else:
            resp_eval = self.rules_evaluator.evaluate_respiration(None, None, valid=False, sample_timestamp=sample_ts)
            hr_eval = self.rules_evaluator.evaluate_vital_hr(None, valid=False)
            mm_ai_status = "DEGRADED"
            mm_fallback_reason = "RESP_SENSOR_FAULT"

        if mm_fallback_reason and mm_fallback_reason not in resp_eval.reasons:
            resp_eval.reasons.append(mm_fallback_reason)

        # 4. PIR Motion (S3) - presence_confirmed 전달
        pir_data = packet.get("pir", {})
        pir_motion = pir_data.get("motion") if isinstance(pir_data, dict) else None
        pir_valid = (pir_motion is not None and pir_motion in (0, 1))
        motion_eval = self.rules_evaluator.evaluate_motion(
            pir_motion,
            presence_confirmed=presence_confirmed,
            valid=pir_valid,
            sample_timestamp=sample_ts
        )

        # 5. RiskRulesEvaluator 통합 시스템 평가
        sys_eval = self.rules_evaluator.evaluate_system(
            respiration_eval=resp_eval,
            environment_eval=env_eval,
            vital_hr_eval=hr_eval,
            posture_eval=posture_eval,
            motion_eval=motion_eval
        )

        # LPF & 히스테리시스 적용
        if sys_eval.is_emergency:
            R = 100.0
            self.curr_smoothed_r = 100.0
            status_str = "DANGER"
            status_code = 2
        else:
            self.risk_history.append(sys_eval.risk_score)
            smoothed_R = float(np.mean(self.risk_history))
            self.curr_smoothed_r += 0.25 * (smoothed_R - self.curr_smoothed_r)
            R = min(100.0, max(0.0, self.curr_smoothed_r))

            if self.prev_status == "DANGER":
                status_str = "DANGER" if R > 65.0 else ("CAUTION" if R > 35.0 else "NORMAL")
            else:
                status_str = "DANGER" if R >= 75.0 else ("CAUTION" if R >= 40.0 else "NORMAL")
            status_code = 2 if status_str == "DANGER" else (1 if status_str == "CAUTION" else 0)

        self.prev_status = status_str

        # mmWave provenance 정보
        if mm_pred_obj is not None:
            mm_source = mm_pred_obj.model_id
            mm_fallback = mm_pred_obj.fallback_used
            mm_reason = mm_pred_obj.fallback_reason
        elif self.mmwave_runner is not None and not self.mmwave_runner.model_file_exists:
            mm_source = "mmwave_heuristic_fallback"
            mm_fallback = True
            mm_reason = mm_fallback_reason or "TFLITE_MODEL_FILE_MISSING"
        else:
            mm_source = "none"
            mm_fallback = True
            mm_reason = mm_fallback_reason or "NO_INPUT_WINDOW"

        v4_fusion = self.v4_engine.evaluate_packet(
            packet,
            1.0 if thermal_class == 2 and thermal_conf >= 0.8 else (
                0.0 if q_gate["thermal"] == 1.0 else None
            ),
            s1=resp_eval.score if q_gate["mmwave"] == 1.0 else None,
            s2=env_eval.score if q_gate["co2"] == 1.0 else None,
            s3=motion_eval.score if q_gate["pir"] == 1.0 else None,
            emergency_override=sys_eval.is_emergency,
        )

        official_status = v4_fusion.level
        official_status_code = 2 if official_status == "DANGER" else (
            1 if official_status == "CAUTION" else 0
        )
        combined_reasons = list(sys_eval.reasons)
        for reason in v4_fusion.fallback_reasons:
            if reason not in combined_reasons:
                combined_reasons.append(reason)
        official_system_status = (
            "DEGRADED"
            if sys_eval.system_status != "OK" or v4_fusion.system_status != "OK"
            else "OK"
        )

        invalid_sensors = []
        stale_sensors = []
        comp_scores = {
            "mmwave": resp_eval.score if q_gate["mmwave"] == 1.0 else None,
            "co2": env_eval.score if q_gate["co2"] == 1.0 else None,
            "pir": motion_eval.score if q_gate["pir"] == 1.0 else None,
            "thermal": 1.0 if thermal_class == 2 and thermal_conf >= 0.8 else (0.0 if q_gate["thermal"] == 1.0 else None),
        }
        for k in ["mmwave", "co2", "pir", "thermal"]:
            if comp_scores[k] is None:
                invalid_sensors.append(k)

        degraded_mode = bool(invalid_sensors or stale_sensors or official_system_status == "DEGRADED")
        if len(invalid_sensors) == 4 or official_system_status in ("FAULT", "FAILED"):
            system_health = "FAILED"
        elif degraded_mode:
            system_health = "DEGRADED"
        else:
            system_health = "HEALTHY"

        return {
            "risk_score": float(v4_fusion.risk_score) if system_health != "FAILED" else None,
            "risk_level": official_status if system_health != "FAILED" else None,
            "system_health": system_health,
            "degraded_mode": degraded_mode,
            "invalid_sensors": invalid_sensors,
            "stale_sensors": stale_sensors,
            "component_scores": comp_scores,
            "status_str": official_status,
            "status_code": official_status_code,
            "is_emergency": v4_fusion.emergency_override,
            "reasons": combined_reasons,
            "sensor_quality": q_gate,
            "system_status": official_system_status,
            "v4_fusion": v4_fusion.to_dict(),
            "legacy_fusion": {
                "risk_score": float(R),
                "level": status_str,
                "status_code": status_code,
                "is_emergency": sys_eval.is_emergency,
            },
            "active_chest_bin": self.active_chest_bin,
            "active_chest_dist_m": float(self.distances[self.active_chest_bin]),
            "derived_metrics": {
                "co2_slope_ppm_per_min": (
                    float(co2_slope_per_min) if co2_ppm is not None else None
                ),
                "presence_confirmed": bool(presence_confirmed),
                "mmwave_window_samples": len(self.mmwave_stream_adapter.buffer),
                "mmwave_window_ready": self.mmwave_stream_adapter.is_ready(),
            },
            "model_meta": {
                "thermal": {
                    "source": self.thermal_runner.model_meta["model_id"] if self.thermal_runner else "none",
                    "version": self.thermal_runner.model_meta["version"] if self.thermal_runner else "0.0.0",
                    "ai_status": thermal_ai_status,
                    "latency_ms": thermal_lat,
                    "fallback_used": thermal_fallback_reason is not None,
                    "fallback_reason": thermal_fallback_reason,
                    "class_index": thermal_class,
                    "class_name": thermal_pred_obj.class_name if thermal_pred_obj else None,
                    "confidence": float(thermal_conf) if thermal_pred_obj else None,
                    "probabilities": thermal_pred_obj.probabilities if thermal_pred_obj else None,
                },
                "co2": {
                    "source": self.co2_runner.model_meta["model_id"] if self.co2_runner else "none",
                    "version": self.co2_runner.model_meta["version"] if self.co2_runner else "0.0.0",
                    "ai_status": co2_ai_status,
                    "latency_ms": co2_lat,
                    "fallback_used": co2_fallback_reason is not None,
                    "fallback_reason": co2_fallback_reason,
                    "class_index": co2_pred_obj.class_index if co2_pred_obj else None,
                    "class_name": co2_pred_obj.class_name if co2_pred_obj else None,
                    "confidence": float(co2_pred_obj.confidence) if co2_pred_obj else None,
                    "probabilities": co2_pred_obj.probabilities if co2_pred_obj else None,
                },
                "mmwave": {
                    "source": mm_source,
                    "version": self.mmwave_runner.model_meta["version"] if self.mmwave_runner else "0.0.0",
                    "ai_status": mm_ai_status,
                    "latency_ms": mm_lat,
                    "fallback_used": mm_fallback,
                    "fallback_reason": mm_reason,
                    "class_index": mm_pred_obj.class_index if mm_pred_obj else None,
                    "class_name": mm_pred_obj.class_name if mm_pred_obj else None,
                    "confidence": float(mm_pred_obj.confidence) if mm_pred_obj else None,
                    "probabilities": mm_pred_obj.probabilities if mm_pred_obj else None,
                }
            }
        }
