# co2_gui_plotter.py
import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Mac / Raspberry Pi 호환 TFLite 인터프리터 동적 임포트
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None

class CO2TFLiteGUIPlotter:
    def __init__(self, model_path=None, metadata_path=None, poll_interval_sec=10, window_size_min=5):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pkg_dir = os.path.dirname(os.path.dirname(base_dir))
        baseline_dir = os.path.join(pkg_dir, "models", "co2", "baseline")

        if metadata_path is None:
            metadata_path = os.path.join(baseline_dir, "co2_scaling_metadata.json")
            if not os.path.exists(metadata_path):
                metadata_path = "co2_scaling_metadata.json"

        if model_path is None:
            model_path = os.path.join(baseline_dir, "co2_occupancy_quant.tflite")
            if not os.path.exists(model_path):
                model_path = "co2_occupancy_quant.tflite"

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"메타데이터 파일 {metadata_path}을 찾을 수 없습니다.")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.mean = np.array(self.metadata['mean'], dtype=np.float32)
        self.scale = np.array(self.metadata['scale'], dtype=np.float32)
        self.features = self.metadata['features']

        self.model_loaded = False
        if tflite is not None and os.path.exists(model_path):
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
            self.output_scale, self.output_zero_point = self.output_details[0]['quantization']
            self.model_loaded = True
            print(f"[{model_path}] TFLite INT8 실제 모델 GUI 플로터 로드 성공!")
        else:
            print(f"[경고] TFLite 모델 파일/인터프리터 미발급으로 기본 룰 모드로 플롯을 가동합니다.")

        self.poll_interval = poll_interval_sec
        self.window_size_min = window_size_min
        self.max_buffer_size = int((window_size_min * 60) / poll_interval_sec)
        self.co2_buffer = []

        # 그래프용 실시간 데이터 적재 (최근 60개 샘플 = 10분 관제)
        self.plot_len = 60
        self.plot_x = np.arange(self.plot_len) * poll_interval_sec
        self.co2_history = np.zeros(self.plot_len) + 450.0
        self.hum_history = np.zeros(self.plot_len) + 35.0

        self.pred_class = 0
        self.conf = 0.0
        self.slope = 0.0
        self.inference_time_ms = 0.0

        self.state_labels = {0: "VACANT", 1: "OCCUPIED"}
        self.state_colors = {0: "#00FF66", 1: "#FF0033"}

    def add_sensor_data(self, co2_value):
        self.co2_buffer.append(co2_value)
        if len(self.co2_buffer) > self.max_buffer_size:
            self.co2_buffer.pop(0)

    def is_buffer_ready(self):
        return len(self.co2_buffer) >= self.max_buffer_size

    def predict(self, current_humidity):
        if not self.is_buffer_ready():
            return 0, 0.0, 0.0

        co2_now = self.co2_buffer[-1]
        co2_past = self.co2_buffer[0]
        co2_slope = (co2_now - co2_past) / self.window_size_min

        if not self.model_loaded:
            pred = 1 if (co2_now > 800 or co2_slope > 10.0) else 0
            return pred, 0.95, co2_slope

        raw = np.array([[co2_slope, current_humidity, co2_now]], dtype=np.float32)
        norm = (raw - self.mean) / (self.scale + 1e-8)

        quantized = (norm / self.input_scale) + self.input_zero_point
        quantized = np.clip(quantized, -128, 127).astype(np.int8)

        start_time = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]['index'], quantized)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        self.inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        probs = (output_data[0].astype(np.float32) - self.output_zero_point) * self.output_scale
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        # Hybrid Rule Fallback Engine for CO2 Plateau
        if co2_now >= 800.0 or co2_slope >= 3.0:
            pred_class = 1
            confidence = max(confidence, 0.92)

        return pred_class, confidence, co2_slope

def main():
    try:
        plotter = CO2TFLiteGUIPlotter()
    except Exception as e:
        print(f"[ERROR] Failed to load metadata: {e}")
        return

    plt.style.use('dark_background')
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax1 = plt.subplots(figsize=(11, 6))
    fig.canvas.manager.set_window_title('SafeNest TFLite INT8 CO2 & Humidity Live AI Plotter')

    color_co2 = '#00E5FF'
    ax1.set_xlabel('Time History (seconds ago)', fontsize=10, color='#AAAAAA')
    ax1.set_ylabel('CO2 Concentration (ppm)', fontsize=10, color=color_co2)
    line_co2, = ax1.plot(plotter.plot_x, plotter.co2_history, color=color_co2, linewidth=2.2, label='CO2 (ppm)')
    ax1.tick_params(axis='y', labelcolor=color_co2)
    ax1.set_ylim(350, 1300)
    ax1.grid(True, color='#333333', linestyle='--')

    color_hum = '#FF00E5'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Relative Humidity (%)', fontsize=10, color=color_hum)
    line_hum, = ax2.plot(plotter.plot_x, plotter.hum_history, color=color_hum, linewidth=1.8, linestyle=':', label='Humidity (%)')
    ax2.tick_params(axis='y', labelcolor=color_hum)
    ax2.set_ylim(30, 45)

    lines = [line_co2, line_hum]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    status_text = ax1.text(0.02, 0.88, "Buffer Filling...", transform=ax1.transAxes,
                           fontsize=11, fontweight='bold', color='#FFFFFF',
                           bbox=dict(facecolor='black', alpha=0.8, boxstyle='round,pad=0.5'))

    state_tracker = {"t_sec": 0, "base_co2": 450.0, "current_humidity": 35.0}

    def update(frame):
        t = state_tracker["t_sec"]
        co2_val = state_tracker["base_co2"]
        hum_val = state_tracker["current_humidity"]

        # Scenario virtual data stream (Vacant -> Entering -> Plateau -> Ventilation)
        if t < 300:
            co2_val += np.random.normal(loc=0.3, scale=0.3)
            hum_val += np.random.normal(loc=0.0, scale=0.01)
        elif t < 600:
            co2_val += np.random.normal(loc=12.0, scale=1.0)
            hum_val += np.random.normal(loc=0.08, scale=0.02)
        elif t < 900:
            co2_val += np.random.normal(loc=0.0, scale=0.5)
            hum_val += np.random.normal(loc=0.0, scale=0.01)
        else:
            co2_val += np.random.normal(loc=-15.0, scale=1.5)
            hum_val += np.random.normal(loc=-0.1, scale=0.02)
            if co2_val < 450.0: co2_val = 450.0
            if hum_val < 35.0: hum_val = 35.0

        state_tracker["base_co2"] = co2_val
        state_tracker["current_humidity"] = hum_val

        plotter.add_sensor_data(co2_val)

        plotter.co2_history = np.roll(plotter.co2_history, -1)
        plotter.co2_history[-1] = co2_val
        line_co2.set_ydata(plotter.co2_history)

        plotter.hum_history = np.roll(plotter.hum_history, -1)
        plotter.hum_history[-1] = hum_val
        line_hum.set_ydata(plotter.hum_history)

        if plotter.is_buffer_ready():
            plotter.pred_class, plotter.conf, plotter.slope = plotter.predict(hum_val)
            current_color = plotter.state_colors.get(plotter.pred_class, "#FFFFFF")
            state_lbl = plotter.state_labels.get(plotter.pred_class, "UNKNOWN")

            status_text.set_text(
                f" TFLite AI STATE: {state_lbl}\n"
                f" Confidence: {plotter.conf*100:.1f}%\n"
                f" CO2 Slope: {plotter.slope:+.2f} ppm/m\n"
                f" Latency: {plotter.inference_time_ms:.2f}ms (TFLite INT8)"
            )
            status_text.set_bbox(dict(facecolor='black', edgecolor=current_color, alpha=0.8, boxstyle='round,pad=0.5'))
            ax1.set_title(f"SafeNest TFLite INT8 CO2 Occupancy Monitoring [RUNNING]", color=current_color, fontsize=12, fontweight='bold')
        else:
            needed = plotter.max_buffer_size - len(plotter.co2_buffer)
            status_text.set_text(f"BUFFER LOADING... ({len(plotter.co2_buffer)}/{plotter.max_buffer_size})\n(Need {needed} more data points)")
            ax1.set_title("SafeNest TFLite INT8 CO2 Occupancy Monitoring [INITIALIZING]", color='#888888', fontsize=12)

        state_tracker["t_sec"] += 10
        return line_co2, line_hum, status_text

    # 100ms 대기시간 간격으로 시뮬레이션 애니메이션 렌더링
    ani = FuncAnimation(fig, update, interval=100, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
