import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        from tensorflow.keras import models as tflite

class SafeNestGUIPlotter:
    def __init__(self, model_path=None, metadata_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "mmwave", "safenest_lstm_quant.tflite")
        if metadata_path is None:
            metadata_path = os.path.join(base_dir, "models", "mmwave", "sensor_stats_metadata.json")

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"모델 파일({model_path}) 또는 메타데이터 파일({metadata_path})이 존재하지 않습니다.")

        # 1. Z-Score 정규화 상수 로드
        with open(metadata_path, 'r') as f:
            self.stats = json.load(f)
        self.mean = np.array(self.stats['mean'], dtype=np.float32)
        self.std = np.array(self.stats['std'], dtype=np.float32)

        # 2. TFLite 인터프리터 로드
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # 3. 양자화 파라미터 로드
        self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
        self.output_scale, self.output_zero_point = self.output_details[0]['quantization']

        # 4. 실시간 데이터 버퍼 및 변수 셋업
        self.window_size = 300  # 30초 학습 모델 규격
        self.buffer = []
        
        # 그래프에 그릴 실시간 뷰 범위 (최근 10초 = 100 샘플)
        self.plot_len = 100
        self.plot_x = np.arange(self.plot_len) / 10.0  # 시간축 (0~10초)
        self.plot_y = np.zeros(self.plot_len)

        # AI 결과 상태 변수
        self.pred_class = 0
        self.conf = 0.0
        self.inference_time_ms = 0.0

        # 상태 매핑
        self.state_labels = {0: "NORMAL", 1: "CAUTION", 2: "DANGER (APNEA)"}
        self.state_colors = {0: "#00FF66", 1: "#FFCC00", 2: "#FF0033"} # 네온 그린, 노랑, 네온 레드

    def preprocess(self, window_buffer):
        normalized = (np.array(window_buffer, dtype=np.float32) - self.mean) / (self.std + 1e-8)
        quantized = (normalized / self.input_scale) + self.input_zero_point
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
        return np.expand_dims(quantized, axis=0)

    def run_inference(self, quantized_data):
        start_time = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]['index'], quantized_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        self.inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        probabilities = (output_data[0].astype(np.float32) - self.output_zero_point) * self.output_scale
        class_idx = np.argmax(probabilities)
        return class_idx, float(probabilities[class_idx])

def generate_virtual_respiration(t, mode):
    """
    현실적인 가상 호흡 신호 생성 (0.1초 간격)
    """
    noise = np.random.normal(0, 0.03)
    if mode == "NORMAL":
        signal = 1.2 * np.sin(2 * np.pi * 0.25 * t) + noise # 정상 호흡
    elif mode == "RAPID":
        signal = 0.5 * np.sin(2 * np.pi * 0.45 * t) + noise # 가쁘고 얕은 호흡
    elif mode == "APNEA":
        signal = 0.03 * np.sin(2 * np.pi * 0.02 * t) + noise # 무호흡 (Flatline)
    else:
        signal = noise
    return float(signal)

def main():
    plotter = SafeNestGUIPlotter()
    
    # -------------------------------------------------------------
    # 1. ROS2 rqt_plot 스타일의 고화질 다크 테마 GUI 창 구성
    # -------------------------------------------------------------
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.canvas.manager.set_window_title('SafeNest Edge AI Real-Time Plotter')
    
    # 그래프 선 초기 설정 (네온 그린으로 시작)
    line, = ax.plot(plotter.plot_x, plotter.plot_y, color='#00FF66', linewidth=2.5, label='Respiration Wave')
    
    # 축 셋업
    ax.set_xlim(0, plotter.plot_len / 10.0)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel("Time (seconds)", fontsize=10, color='#AAAAAA')
    ax.set_ylabel("Chest Phase / Displacement", fontsize=10, color='#AAAAAA')
    ax.grid(True, color='#333333', linestyle='--')
    
    # 텍스트 오버레이 (AI 예측 결과 출력 영역)
    status_text = ax.text(0.02, 0.90, "Buffer Filling...", transform=ax.transAxes,
                          fontsize=12, fontweight='bold', color='#FFFFFF',
                          bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.5'))
    
    # 시간 변수 및 프레임 카운트
    state_tracker = {"t": 0.0}

    # -------------------------------------------------------------
    # 2. 10Hz 애니메이션 업데이트 루프 함수
    # -------------------------------------------------------------
    def update(frame):
        t = state_tracker["t"]
        
        # 1) 시나리오 주기 결정 (45초 루프)
        cycle_time = int(t) % 45
        if cycle_time < 15:
            mode = "NORMAL"
        elif cycle_time < 30:
            mode = "RAPID"
        else:
            mode = "APNEA"
            
        # 2) 10Hz 데이터 포인트 생성 및 버퍼 추가
        val = generate_virtual_respiration(t, mode)
        plotter.buffer.append([val])
        
        # 그래프 플롯용 데이터 시프트 (최근 10초 치)
        plotter.plot_y = np.roll(plotter.plot_y, -1)
        plotter.plot_y[-1] = val
        line.set_ydata(plotter.plot_y)
        
        # 3) 30초 버퍼 관리 및 1초 주기 AI 연산 실행
        if len(plotter.buffer) > plotter.window_size:
            plotter.buffer.pop(0)
            
            # 1초에 한 번 (10프레임 주기) AI 추론 호출
            if frame % 10 == 0:
                quantized_input = plotter.preprocess(plotter.buffer)
                plotter.pred_class, plotter.conf = plotter.run_inference(quantized_input)
                
            # 4) AI 판정에 따른 그래프 선 색상 및 타이틀 텍스트 동적 반영
            current_color = plotter.state_colors.get(plotter.pred_class, "#FFFFFF")
            line.set_color(current_color)
            
            state_label = plotter.state_labels.get(plotter.pred_class, "UNKNOWN")
            status_text.set_text(
                f" AI STATE: {state_label}\n"
                f" Confidence: {plotter.conf*100:.1f}%\n"
                f" Latency: {plotter.inference_time_ms:.1f}ms\n"
                f" Mode: {mode}"
            )
            status_text.set_bbox(dict(facecolor='black', edgecolor=current_color, alpha=0.8, boxstyle='round,pad=0.5'))
            ax.set_title(f"SafeNest Live Monitoring [{mode}]", color=current_color, fontsize=12, fontweight='bold')
        else:
            # 버퍼 적재 중일 때 표시
            status_text.set_text(f"BUFFER LOADING... ({len(plotter.buffer)}/{plotter.window_size})")
            ax.set_title("SafeNest Live Monitoring [INITIALIZING]", color='#888888', fontsize=12)

        state_tracker["t"] += 0.1
        return line, status_text

    # 3. FuncAnimation 구동 (0.1초 주기 = 100ms 갱신)
    ani = FuncAnimation(fig, update, interval=100, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
