#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
safenest_integrated_plotter.py
SafeNest 4분할 생체연동 실시간 관제 GUI 시뮬레이터

mmWave 생체파형, CO2 시계열, PIR 동체 반응, Thermal-44 80x62 히트맵을
한 화면에 연동하고 키보드(0, 1, 2, 3)로 모의 시연합니다.
"""

import sys
import platform
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

try:
    from .virtual_sensor_streamer import VirtualSensorStreamer
    from .safenest_risk_engine import SafeNestRiskEngine
except ImportError:
    from virtual_sensor_streamer import VirtualSensorStreamer
    from safenest_risk_engine import SafeNestRiskEngine

# Matplotlib 시스템 폰트 설정
system_name = platform.system()
if system_name == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif system_name == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False

class SafeNestIntegratedPlotter:
    def __init__(self):
        self.streamer = VirtualSensorStreamer()
        self.risk_engine = SafeNestRiskEngine()

        self.max_pts = 100  # 10Hz 기준 최근 10초
        self.co2_pts = collections.deque(maxlen=self.max_pts)
        self.wave_pts = collections.deque(maxlen=self.max_pts)
        self.pir_pts = collections.deque(maxlen=self.max_pts)
        self.time_pts = collections.deque(maxlen=self.max_pts)
        
        self.step = 0

        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(14, 9), facecolor='#0f172a')
        self.fig.canvas.manager.set_window_title('SafeNest Physiological Edge AI Multi-Sensor Control Node')
        
        gs = GridSpec(2, 2, figure=self.fig, hspace=0.35, wspace=0.25)
        
        # 1. Top-Left: Thermal 80x62 히트맵
        self.ax_thermal = self.fig.add_subplot(gs[0, 0])
        self.ax_thermal.set_title("[Thermal-44] 80x62 IR Array / Posture AI", fontsize=12, fontweight='bold', color='#38bdf8')
        init_grid = np.zeros((62, 80))
        self.im_thermal = self.ax_thermal.imshow(init_grid, cmap='inferno', vmin=0.0, vmax=1.0, aspect='auto')
        self.cbar = self.fig.colorbar(self.im_thermal, ax=self.ax_thermal, fraction=0.046, pad=0.04)
        self.txt_thermal = self.ax_thermal.text(
            0.03, 0.90, "", transform=self.ax_thermal.transAxes,
            fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.75)
        )

        # 2. Top-Right: CO2 농도 & Slope 그래프
        self.ax_co2 = self.fig.add_subplot(gs[0, 1])
        self.ax_co2.set_title("[SCD40] CO2 Concentration & Derived Slope", fontsize=12, fontweight='bold', color='#38bdf8')
        self.line_co2, = self.ax_co2.plot([], [], color='#a855f7', lw=2.5, label='CO2 (ppm)')
        self.ax_co2.set_ylim(400, 4200)
        self.ax_co2.set_ylabel("CO2 (ppm)", color='#a855f7')
        self.ax_co2.grid(True, linestyle='--', alpha=0.3)
        self.txt_co2 = self.ax_co2.text(
            0.03, 0.85, "", transform=self.ax_co2.transAxes,
            fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.75)
        )

        # 3. Bottom-Left: mmWave 호흡 & 심박수 (Heart Rate) 라이브 파형
        self.ax_wave = self.fig.add_subplot(gs[1, 0])
        self.ax_wave.set_title("[MR60BHA2] Vital Radar / Respiration AI", fontsize=12, fontweight='bold', color='#38bdf8')
        self.line_wave, = self.ax_wave.plot([], [], color='#22c55e', lw=2.0, label='Vital Wave')
        self.ax_wave.set_ylim(-2.5, 2.5)
        self.ax_wave.set_ylabel("Amplitude", color='#22c55e')
        self.ax_wave.grid(True, linestyle='--', alpha=0.3)
        self.txt_wave = self.ax_wave.text(
            0.03, 0.85, "", transform=self.ax_wave.transAxes,
            fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.75)
        )

        # 4. Bottom-Right: PIR 동체 반응. 융합 R은 배너로 함께 표시한다.
        self.ax_pir = self.fig.add_subplot(gs[1, 1])
        self.ax_pir.set_title("[PIR] Motion Response / Fusion Risk", fontsize=12, fontweight='bold', color='#38bdf8')
        self.line_pir, = self.ax_pir.step([], [], where='post', color='#fb923c', lw=3.0, label='Motion')
        self.ax_pir.set_ylim(-0.1, 1.1)
        self.ax_pir.set_yticks([0, 1], labels=['STILL', 'MOTION'])
        self.ax_pir.set_ylabel("PIR Digital State", color='#fb923c')
        self.ax_pir.grid(True, linestyle='--', alpha=0.3)
        
        self.txt_risk_banner = self.ax_pir.text(
            0.5, 0.52, "INITIALIZING...", transform=self.ax_pir.transAxes,
            fontsize=15, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1e293b', edgecolor='#94a3b8', lw=2)
        )

        # 키보드 이벤트 연결
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        self.fig.text(
            0.5, 0.01,
            "[Controls] 0 Normal | 1 CO2+Tachycardia | 2 Fall | 3 Apnea | 4 Auto Loop",
            ha='center', fontsize=10.5, fontweight='bold', color='#f1f5f9',
            bbox=dict(boxstyle='round', facecolor='#1e1b4b', alpha=0.9)
        )

    def on_key_press(self, event):
        if event.key in ['0', '1', '2', '3', '4']:
            mode = int(event.key)
            self.streamer.set_scenario(mode)

    def update(self, frame):
        self.step += 1
        
        packet = self.streamer.generate_packet()
        risk_res = self.risk_engine.evaluate_risk(packet)
        model_meta = risk_res.get("model_meta", {})
        derived = risk_res.get("derived_metrics", {})
        
        self.time_pts.append(self.step)
        self.co2_pts.append(packet["co2_scd40"]["co2_ppm"])
        self.wave_pts.append(packet["mmwave_mr60"]["wave_val"])
        self.pir_pts.append(packet["pir"]["motion"])
        
        # UI [1] Thermal 80x62
        thermal_grid = packet["thermal_80x62"]
        self.im_thermal.set_array(thermal_grid)
        
        thermal_meta = model_meta.get("thermal", {})
        thermal_class = thermal_meta.get("class_index")
        thermal_name = thermal_meta.get("class_name") or "NO_PREDICTION"
        thermal_conf = thermal_meta.get("confidence")
        conf_text = f"{thermal_conf * 100:.1f}%" if thermal_conf is not None else "N/A"

        if thermal_class == 2:
            self.txt_thermal.set_text(f"[FALL DETECTED]\nAI: {thermal_name} | Conf: {conf_text}")
            self.txt_thermal.set_color('#f87171')
        elif thermal_class == 1:
            self.txt_thermal.set_text(f"[NORMAL POSTURE]\nAI: {thermal_name} | Conf: {conf_text}")
            self.txt_thermal.set_color('#4ade80')
        elif thermal_class == 0:
            self.txt_thermal.set_text(f"[NOT HUMAN]\nAI: {thermal_name} | Conf: {conf_text}")
            self.txt_thermal.set_color('#9ca3af')
        else:
            self.txt_thermal.set_text(
                f"[THERMAL AI {thermal_meta.get('ai_status', 'NOT_RUN')}]\n"
                f"Reason: {thermal_meta.get('fallback_reason') or 'NO_OUTPUT'}"
            )
            self.txt_thermal.set_color('#facc15')

        # UI [2] CO2
        self.line_co2.set_data(list(self.time_pts), list(self.co2_pts))
        self.ax_co2.set_xlim(max(0, self.step - self.max_pts), self.step + 2)
        co2_val = packet["co2_scd40"]["co2_ppm"]
        slope_val = derived.get("co2_slope_ppm_per_min")
        slope_text = f"{slope_val:+.1f} ppm/min" if slope_val is not None else "N/A"
        co2_meta = model_meta.get("co2", {})
        co2_class = co2_meta.get("class_name") or "NO_PREDICTION"
        co2_conf = co2_meta.get("confidence")
        co2_conf_text = f"{co2_conf * 100:.1f}%" if co2_conf is not None else "N/A"
        self.txt_co2.set_text(
            f"CO2: {co2_val:.0f} ppm | Slope: {slope_text}\n"
            f"AI: {co2_class} ({co2_conf_text}) | {co2_meta.get('ai_status', 'NOT_RUN')}"
        )

        # UI [3] mmWave 호흡 & 심박수 (Heart Rate)
        self.line_wave.set_data(list(self.time_pts), list(self.wave_pts))
        self.ax_wave.set_xlim(max(0, self.step - self.max_pts), self.step + 2)
        mmwave_packet = packet["mmwave_mr60"]
        rpm_val = mmwave_packet["breath_rpm"]
        bpm_val = mmwave_packet["heart_bpm"]
        apnea = mmwave_packet["apnea"]
        mmwave_meta = model_meta.get("mmwave", {})
        mmwave_class = mmwave_meta.get("class_name") or mmwave_meta.get("ai_status", "NOT_RUN")
        window_count = derived.get("mmwave_window_samples", 0)
        ai_line = f"AI: {mmwave_class} | Window: {window_count}/300"
        
        if apnea == 1:
            self.txt_wave.set_text(
                f"[APNEA DETECTED]\nHR: {bpm_val:.0f} BPM | RR: 0 RPM\n{ai_line}"
            )
            self.txt_wave.set_color('#f87171')
        elif bpm_val > 110:
            self.txt_wave.set_text(
                f"[TACHYCARDIA]\nHR: {bpm_val:.0f} BPM | RR: {rpm_val:.1f} RPM\n{ai_line}"
            )
            self.txt_wave.set_color('#facc15')
        else:
            self.txt_wave.set_text(
                f"[VITAL MONITOR]\nHR: {bpm_val:.0f} BPM | RR: {rpm_val:.1f} RPM\n{ai_line}"
            )
            self.txt_wave.set_color('#4ade80')

        # UI [4] PIR + Risk Score 배너
        self.line_pir.set_data(list(self.time_pts), list(self.pir_pts))
        self.ax_pir.set_xlim(max(0, self.step - self.max_pts), self.step + 2)
        
        v4_fusion = risk_res.get("v4_fusion", {})
        r_score = v4_fusion.get("risk_score", risk_res["risk_score"])
        status_str = v4_fusion.get("level", risk_res["status_str"])
        status_code = 2 if status_str == "DANGER" else (1 if status_str == "CAUTION" else 0)
        system_status = risk_res["system_status"]
        
        scenario_labels = ["0: Normal", "1: CO2 & HR Risen", "2: Fall", "3: Apnea Danger"]
        curr_scenario = scenario_labels[packet["scenario_mode"]]
        if packet.get("auto_loop"):
            curr_scenario = f"Auto / {curr_scenario}"
        reason_list = risk_res.get("reasons", [])
        reason_text = ", ".join(reason_list[:2]) if reason_list else "NONE"
        
        if status_str == "FAULT":
            banner_text = (
                f"FAULT / SYSTEM {system_status}\n"
                f"Risk unavailable: {r_score:.1f}\n"
                f"Mode: {curr_scenario}\nReason: {reason_text}"
            )
            face_col = '#3f3f46'
            edge_col = '#f87171'
            txt_col = '#f4f4f5'
        elif status_code == 2:
            banner_text = (
                f"DANGER / SYSTEM {system_status}\n"
                f"Risk Index R: {r_score:.1f} / 100\n"
                f"Mode: {curr_scenario}\nReason: {reason_text}"
            )
            face_col = '#881337'
            edge_col = '#f43f5e'
            txt_col = '#ffe4e6'
        elif status_code == 1:
            banner_text = (
                f"CAUTION / SYSTEM {system_status}\n"
                f"Risk Index R: {r_score:.1f} / 100\n"
                f"Mode: {curr_scenario}\nReason: {reason_text}"
            )
            face_col = '#713f12'
            edge_col = '#eab308'
            txt_col = '#fef9c3'
        else:
            banner_text = (
                f"{status_str} / SYSTEM {system_status}\n"
                f"Risk Index R: {r_score:.1f} / 100\n"
                f"Mode: {curr_scenario}\nReason: {reason_text}"
            )
            face_col = '#14532d'
            edge_col = '#22c55e'
            txt_col = '#dcfce7'

        self.txt_risk_banner.set_text(banner_text)
        self.txt_risk_banner.set_color(txt_col)
        self.txt_risk_banner.set_bbox(dict(boxstyle='round,pad=0.5', facecolor=face_col, edgecolor=edge_col, lw=2.5))

        return self.im_thermal, self.line_co2, self.line_wave, self.line_pir

    def run(self):
        print("🚀 [Integrated Control Node] GUI Simulation Starting...")
        print("💡 Controls: 0 Normal | 1 CO2 & HR Risen | 2 Fall | 3 Apnea | 4 Auto Loop")
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=100,
            blit=False,
            cache_frame_data=False,
        )
        self.fig.subplots_adjust(
            left=0.07, right=0.96, bottom=0.09, top=0.93,
            hspace=0.35, wspace=0.25,
        )
        plt.show()

if __name__ == "__main__":
    app = SafeNestIntegratedPlotter()
    app.run()
