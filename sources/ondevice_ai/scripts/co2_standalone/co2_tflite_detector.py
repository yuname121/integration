# co2_tflite_detector.py
import os
import sys
import json
import time
import numpy as np

# Mac / Raspberry Pi 호환 TFLite 인터프리터 동적 임포트
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None

# ANSI 색상 코드 정의 (터미널 시각 효과)
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"

class CO2TFLiteDetector:
    def __init__(self, model_path=None, metadata_path=None, poll_interval_sec=10, window_size_min=5):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pkg_dir = os.path.dirname(os.path.dirname(base_dir))
        baseline_dir = os.path.join(pkg_dir, "models", "co2", "baseline")

        if metadata_path is None:
            metadata_path = os.path.join(baseline_dir, "co2_scaling_metadata.json")
            if not os.path.exists(metadata_path):
                metadata_path = os.path.join(base_dir, "co2_scaling_metadata.json")

        if model_path is None:
            model_path = os.path.join(baseline_dir, "co2_occupancy_quant.tflite")
            if not os.path.exists(model_path):
                model_path = os.path.join(base_dir, "co2_occupancy_quant.tflite")

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
            print(f"[{model_path}] TFLite INT8 실제 모델 100% 로드 성공!")
        else:
            if tflite is None:
                print(f"[알림] 파이썬 환경에 'tflite-runtime' 또는 'tensorflow'가 미설치되어 가상 시뮬레이션 모드로 동작합니다.")
                print(f"        👉 실제 .tflite 모델 연동을 위해: pip install tflite-runtime (또는 tensorflow)")
            elif not os.path.exists(model_path):
                print(f"[알림] {model_path} 파일이 존재하지 않아 시뮬레이션 모드로 동작합니다.")

        self.poll_interval = poll_interval_sec
        self.window_size_min = window_size_min
        self.max_buffer_size = int((window_size_min * 60) / poll_interval_sec)
        self.co2_buffer = []

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
            # TFLite 미발급 시 시뮬레이션 규칙 추론
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
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        probs = (output_data[0].astype(np.float32) - self.output_zero_point) * self.output_scale
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        # 융합 폴백 하이브리드 엔진 (CO2 정체 구간 plateau 오진 방지 보정)
        if co2_now >= 800.0 or co2_slope >= 3.0:
            pred_class = 1
            confidence = max(confidence, 0.92)

        return pred_class, confidence, co2_slope

if __name__ == "__main__":
    print("\n" + "="*80)
    print(f"{COLOR_CYAN}{COLOR_BOLD}  Mac / Pi 호환 TFLite 실시간 CO2 온디바이스 AI 추론 데모{COLOR_RESET}")
    print("="*80)

    try:
        detector = CO2TFLiteDetector()
    except Exception as e:
        print(f"[오류] 메타데이터 로드 실패: {e}")
        sys.exit(1)

    base_co2 = 450.0
    current_humidity = 35.5

    print("\n--- 맥북 단독 실시간 추론 시뮬레이션 루프 기동 ---")
    for sec in range(0, 360, 10):
        base_co2 += np.random.normal(loc=3.5, scale=0.8)
        current_humidity += np.random.normal(loc=0.05, scale=0.02)
        detector.add_sensor_data(base_co2)

        if detector.is_buffer_ready():
            pred, conf, slope = detector.predict(current_humidity)
            state_str = f"{COLOR_RED}[OCCUPIED]{COLOR_RESET}" if pred == 1 else f"{COLOR_GREEN}[VACANT]{COLOR_RESET}"
            print(f"[{sec}초] CO2: {base_co2:.1f}ppm | 기울기: {slope:+.2f} ppm/m | 습도: {current_humidity:.2f}% | 재실: {state_str} (신뢰도: {conf*100:.1f}%)")
        else:
            needed = detector.max_buffer_size - len(detector.co2_buffer)
            print(f"[{sec}초] CO2: {base_co2:.1f}ppm | 버퍼 가동 중... ({needed}개 남음)")

        time.sleep(0.1)
