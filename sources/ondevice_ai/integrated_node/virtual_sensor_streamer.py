#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
virtual_sensor_streamer.py
SafeNest 생태생리학 동적 믹스 시뮬레이터 (Dynamic Mixed Scenario Streamer)

정상 ➔ CO2 상승/빈맥 ➔ 쓰러짐 ➔ 무호흡/서맥 ➔ 회복으로 이어지는 연속 자동 믹스 루프를 제공하며,
바이탈 레이더 파형에 호흡파(RR)와 심박 맥박파(HR Pulse)를 생리학적으로 합성 렌더링합니다.
"""

import os
import time
import math
import numpy as np

class VirtualSensorStreamer:
    def __init__(self, npz_path="../thermal/processed_thermal_80x62.npz"):
        self.scenario_mode = 0  # 0: 정상, 1: CO2상승/빈맥, 2: 쓰러짐, 3: 무호흡기절, 4: 자동 믹스 루프
        self.auto_loop = True   # 기본적으로 자동 다이내믹 믹스 루프 활성화 (디버깅 용이)
        self.start_time = time.time()
        self.step_count = 0
        self.sample_rate_hz = 10.0
        
        # 생체 상태 변수 (지수 평활 연속 보간)
        self.curr_co2 = 500.0
        self.curr_humidity = 45.0
        self.curr_breath_rpm = 16.0
        self.curr_heart_bpm = 72.0  # 심박수 (BPM)
        self.curr_apnea = 0
        self.curr_pir_motion = 1
        
        # 80x62 열화상 데이터셋 로드
        self.thermal_samples = {0: [], 1: [], 2: []}
        possible_paths = [
            npz_path,
            os.path.join(os.path.dirname(__file__), "..", "thermal", "processed_thermal_80x62.npz"),
            os.path.join("thermal", "processed_thermal_80x62.npz")
        ]
        
        found_path = None
        for p in possible_paths:
            if os.path.exists(p):
                found_path = p
                break
                
        if found_path:
            data = np.load(found_path)
            X, y = data['X'], data['y']
            for cls_idx in [0, 1, 2]:
                indices = np.where(y == cls_idx)[0]
                if len(indices) > 0:
                    self.thermal_samples[cls_idx] = X[indices]

    def set_scenario(self, mode: int):
        """0: 정상, 1: CO2상승/빈맥, 2: 쓰러짐, 3: 무호흡기절, 4: 자동 다이내믹 믹스 루프"""
        self.scenario_mode = mode
        if mode == 4:
            self.auto_loop = True
            print("\n🎬 [Streamer] 🔄 자동 다이내믹 믹스 루프 모드 (정상➔CO2상승➔쓰러짐➔무호흡➔회복) 활성화")
        else:
            self.auto_loop = False
            labels = [
                "0: 정상 작업 (HR 72 BPM | RR 16 RPM)",
                "1: CO2 상승 & 빈맥 (HR 128 BPM Tachycardia | RR 28 과호흡)",
                "2: 바닥 쓰러짐 (HR 95 BPM | 움직임 정지)",
                "3: 무호흡 기절 (HR 35 BPM Bradycardia | RR 0 무호흡)"
            ]
            print(f"\n🎬 [Streamer] 고정 시나리오 선택 ➔ {labels[mode]}")

    def generate_packet(self):
        """1개 타임스텝의 생체 연동 데이터 생성"""
        self.step_count += 1
        t = self.step_count / self.sample_rate_hz  # mmWave 모델 계약과 동일한 10Hz
        
        # 1. 자동 다이내믹 믹스 루프 모드 처리 (약 60초 주기 전체 생체 사고/회복 시나리오)
        if self.auto_loop:
            cycle_samples = int(60.0 * self.sample_rate_hz)
            cycle_t = (self.step_count % cycle_samples) / cycle_samples  # 0.0 ~ 1.0 (60초 주기)
            if cycle_t < 0.25:
                active_mode = 0  # 정상
            elif cycle_t < 0.50:
                active_mode = 1  # CO2 상승 & 빈맥 (과호흡)
            elif cycle_t < 0.75:
                active_mode = 2  # 쓰러짐
            else:
                active_mode = 3  # 무호흡 기절 (긴급)
        else:
            active_mode = self.scenario_mode

        # 2. 모드별 생체 파라미터 목표값 설정
        if active_mode == 0:  # 정상
            target_co2 = 500.0 + 10.0 * math.sin(t * 0.02)
            target_humidity = 45.0 + 2.0 * math.sin(t * 0.05)
            target_breath_rpm = 16.0 + 1.5 * math.sin(t * 0.4)
            target_heart_bpm = 72.0 + 3.0 * math.sin(t * 0.3)  # 정상 72 BPM
            target_apnea = 0
            target_pir_motion = 1 if (self.step_count % 30 < 20) else 0
            thermal_class = 1  # Normal

        elif active_mode == 1:  # CO2 상승 & 빈맥
            target_co2 = 2200.0 + 400.0 * math.sin(t * 0.2)
            target_humidity = 62.0
            target_breath_rpm = 28.0 + 3.0 * math.sin(t * 0.8)  # 과호흡 28 RPM
            target_heart_bpm = 128.0 + 6.0 * math.sin(t * 0.5)  # ⭐ 빈맥(Tachycardia) 128 BPM
            target_apnea = 0
            target_pir_motion = 1 if (self.step_count % 12 < 8) else 0
            thermal_class = 1  # Normal (앉음)

        elif active_mode == 2:  # 바닥 쓰러짐
            target_co2 = 3000.0
            target_humidity = 68.0
            target_breath_rpm = 8.0 + 1.0 * math.sin(t * 0.2)   # 미세호흡 8 RPM
            target_heart_bpm = 92.0 + 4.0 * math.sin(t * 0.2)   # 잔여 심박
            target_apnea = 0
            target_pir_motion = 0  # 움직임 정지
            thermal_class = 2  # Fall

        else:  # active_mode == 3: 무호흡 기절
            target_co2 = 3800.0
            target_humidity = 75.0
            target_breath_rpm = 0.0  # ⭐ 무호흡 0 RPM
            target_heart_bpm = 35.0 + 2.0 * math.sin(t * 0.1)   # ⭐ 서맥(Bradycardia) 35 BPM
            target_apnea = 1
            target_pir_motion = 0  # 움직임 정지
            thermal_class = 2  # Fall

        # 3. 부드러운 생체 반응 지수 평활 보간
        # 기존 5Hz의 α=0.08과 비슷한 실제 시간 응답을 유지하기 위해 10Hz에서는 0.04 사용
        alpha = 0.04
        # CO2는 호흡·심박보다 훨씬 느린 환경 신호이므로 별도 완만한 계수를 사용한다.
        self.curr_co2 += 0.001 * (target_co2 - self.curr_co2)
        self.curr_humidity += alpha * (target_humidity - self.curr_humidity)
        self.curr_breath_rpm += alpha * (target_breath_rpm - self.curr_breath_rpm)
        self.curr_heart_bpm += alpha * (target_heart_bpm - self.curr_heart_bpm)
        self.curr_apnea = target_apnea
        self.curr_pir_motion = target_pir_motion

        # 4. Thermal 80x62 프레임 획득
        if len(self.thermal_samples[thermal_class]) > 0:
            idx = self.step_count % len(self.thermal_samples[thermal_class])
            thermal_grid = self.thermal_samples[thermal_class][idx].copy()
            thermal_grid += np.random.normal(0, 0.005, size=(62, 80))
            thermal_grid = np.clip(thermal_grid, 0.0, 1.0)
        else:
            thermal_grid = np.zeros((62, 80), dtype=np.float32)
            if thermal_class == 1:
                thermal_grid[10:45, 35:45] = 0.85
            else:
                thermal_grid[45:55, 15:65] = 0.90

        # 5. ⭐ 바이탈 레이더 파형: [호흡파(RR) + 실시간 심박동 맥박파(HR ECG/PPG)] 생리학적 합성!
        rr_freq = (max(0.1, self.curr_breath_rpm) / 60.0) * 2 * math.pi
        hr_freq = (self.curr_heart_bpm / 60.0) * 2 * math.pi
        
        # 호흡 신호 (Sine wave)
        if self.curr_apnea == 1 or self.curr_breath_rpm < 1.0:
            resp_component = 0.0  # 무호흡 시 호흡파 소멸
        else:
            resp_component = 1.0 * math.sin(t * rr_freq)

        # 심박동 맥박 신호 (Heartbeat Pulse Spike)
        pulse_phase = (t * (self.curr_heart_bpm / 60.0)) % 1.0
        if pulse_phase < 0.15:
            # QRS 심박 스파이크 파형
            heart_component = 1.2 * math.sin(pulse_phase / 0.15 * math.pi)
        else:
            heart_component = 0.0

        # 바이탈파 = 호흡성분 + 심박성분 + 노이즈
        wave_val = resp_component + heart_component + np.random.normal(0, 0.03)
        # AI 입력은 심박 spike를 섞은 화면용 wave_val과 분리한다.
        # 이 값은 파이프라인 동작 확인용 synthetic resp_phase이며 실센서 정확도 근거가 아니다.
        resp_phase = resp_component + np.random.normal(0, 0.01)
        sample_timestamp_s = self.start_time + t

        packet = {
            "timestamp": sample_timestamp_s,
            "timestamp_s": sample_timestamp_s,
            "step": self.step_count,
            "scenario_mode": active_mode,
            "auto_loop": self.auto_loop,
            "thermal_80x62": thermal_grid,
            "co2_scd40": {
                "co2_ppm": float(self.curr_co2),
                "humidity": float(self.curr_humidity)
            },
            "mmwave_mr60": {
                "presence": 1,
                "breath_rpm": float(self.curr_breath_rpm),
                "heart_bpm": float(self.curr_heart_bpm),  # ⭐ 실시간 동적 심박수
                "apnea": int(self.curr_apnea),
                "wave_val": float(wave_val),
                "resp_phase": float(resp_phase)
            },
            "pir": {
                "motion": int(self.curr_pir_motion)
            }
        }
        return packet
