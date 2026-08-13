import os
import sys
import time
import json
import numpy as np
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        from tensorflow.keras import models as tflite

# ANSI 색상 코드 정의 (터미널 시각 효과용)
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"

class SafeNestRealTimeDemo:
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

        # 3. 양자화 스케일 및 제로포인트 로드
        self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
        self.output_scale, self.output_zero_point = self.output_details[0]['quantization']

        # 4. 실시간 버퍼 셋업 (30초 윈도우 = 10Hz * 30초 = 300 샘플)
        self.window_size = 300
        self.buffer = []

        # 5. 상태 레이블 정의
        self.labels = {
            0: f"{COLOR_GREEN}{COLOR_BOLD}[정상 호흡 (NORMAL)]{COLOR_RESET}",
            1: f"{COLOR_YELLOW}{COLOR_BOLD}[호흡 이상 (CAUTION)]{COLOR_RESET}",
            2: f"{COLOR_RED}{COLOR_BOLD}[무호흡 기절 (DANGER)]{COLOR_RESET}"
        }

    def preprocess(self, window_buffer):
        """
        Z-Score 정규화 및 INT8 완전 양자화 수행
        """
        # Z-Score
        normalized = (np.array(window_buffer, dtype=np.float32) - self.mean) / (self.std + 1e-8)
        # Quantize: Int8 = (Float / Scale) + Zero_Point
        quantized = (normalized / self.input_scale) + self.input_zero_point
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
        return np.expand_dims(quantized, axis=0)

    def run_inference(self, quantized_data):
        """
        양자화 추론 후 결과 디양자화
        """
        self.interpreter.set_tensor(self.input_details[0]['index'], quantized_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Dequantize: Float = (Int8 - Zero_Point) * Scale
        probabilities = (output_data[0].astype(np.float32) - self.output_zero_point) * self.output_scale
        class_idx = np.argmax(probabilities)
        return class_idx, float(probabilities[class_idx])

    def print_ascii_wave(self, val, state_str, conf):
        """
        터미널에 실시간 스크롤 호흡 그래프 및 AI 상태 출력
        """
        # 값을 0~60 칸 사이의 아스키 그래프로 맵핑
        width = 50
        # 호흡 신호 진폭을 0 ~ width 사이로 스케일링 (가상 신호 범위가 약 -2.0 ~ 2.0라고 가정)
        val_clamped = max(-2.0, min(2.0, val))
        pos = int((val_clamped + 2.0) / 4.0 * width)
        pos = max(0, min(width - 1, pos))
        
        line = [" "] * width
        line[pos] = "●"
        line_str = "".join(line)
        
        # 출력 스트림 전송
        sys.stdout.write(f"|{line_str}| {state_str} (확신도: {conf*100:5.1f}%)\r")
        sys.stdout.flush()

def generate_virtual_respiration(t, mode):
    """
    모드에 맞는 현실적인 실시간 호흡 위상 데이터 생성
    """
    noise = np.random.normal(0, 0.05)
    
    if mode == "NORMAL":
        # 1분에 15회 호흡 주기 (0.25 Hz 주파수)
        signal = 1.5 * np.sin(2 * np.pi * 0.25 * t) + noise
    elif mode == "RAPID":
        # 분당 27회 가쁜 과호흡 (0.45 Hz 주파수), 얕은 깊이
        signal = 0.6 * np.sin(2 * np.pi * 0.45 * t) + noise
    elif mode == "APNEA":
        # 호흡 없음 (미세한 노이즈 및 서서히 편향되는 drift만 존재)
        signal = 0.05 * np.sin(2 * np.pi * 0.02 * t) + noise
    else:
        signal = noise
        
    return float(signal)

def main():
    demo = SafeNestRealTimeDemo()
    
    print("\n" + "="*80)
    print(f"{COLOR_CYAN}{COLOR_BOLD}  SafeNest 온디바이스 Edge AI 실시간 모니터링 데모 (10Hz){COLOR_RESET}")
    print("="*80)
    print("  * 1~15초 : 정상 호흡 구간 (Sin Wave)")
    print("  * 16~30초 : 과호흡/얕은호흡 구간 (High Freq, Low Amp)")
    print("  * 31~45초 : 무호흡 질식 구간 (Flatline)")
    print("  * 터미널 창에 실시간 파형 그래프와 AI의 판단 결과가 10Hz(0.1초) 주기로 갱신됩니다.")
    print("  * 종료하려면 Ctrl + C를 누르세요.\n")
    time.sleep(3.0)

    t = 0.0
    mode = "NORMAL"
    pred_class = 0
    conf = 1.0

    try:
        while True:
            # 1. 시간에 따른 시나리오 모드 결정
            cycle_time = int(t) % 45
            if cycle_time < 15:
                mode = "NORMAL"
            elif cycle_time < 30:
                mode = "RAPID"
            else:
                mode = "APNEA"
                
            # 2. 10Hz 데이터 생성 및 버퍼 추가
            val = generate_virtual_respiration(t, mode)
            demo.buffer.append([val])
            
            # 3. 30초 크기 버퍼 유지 (300 타임스탬프)
            if len(demo.buffer) > demo.window_size:
                demo.buffer.pop(0)
                
                # 1초에 한 번씩 AI 연산 실행 (10 step마다)
                if int(t * 10) % 10 == 0:
                    input_data = demo.preprocess(demo.buffer)
                    pred_class, conf = demo.run_inference(input_data)
            
            # 4. 실시간 상태 텍스트
            if len(demo.buffer) < demo.window_size:
                state_str = f"버퍼 채우는 중... ({len(demo.buffer)}/{demo.window_size})"
                conf_val = 0.0
            else:
                state_str = demo.labels.get(pred_class, "알 수 없음")
                conf_val = conf
                
            # 5. 아스키 플롯 출력
            demo.print_ascii_wave(val, state_str, conf_val)
            
            # 10Hz (0.1초 대기)
            time.sleep(0.1)
            t += 0.1
            
    except KeyboardInterrupt:
        print("\n\n모니터링 데모를 종료합니다.")

if __name__ == "__main__":
    main()
