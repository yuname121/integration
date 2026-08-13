# co2_visual_demo.py
import os
import sys
import time
import json
import numpy as np
import joblib

# ANSI 색상 코드 정의 (터미널 시각 효과용)
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"

class CO2VisualDemo:
    def __init__(self, model_path=None, metadata_path=None, poll_interval_sec=10, window_size_min=5):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pkg_dir = os.path.dirname(os.path.dirname(base_dir))
        baseline_dir = os.path.join(pkg_dir, "models", "co2", "baseline")

        if metadata_path is None:
            metadata_path = os.path.join(baseline_dir, "co2_scaling_metadata.json")
            if not os.path.exists(metadata_path):
                metadata_path = "co2_scaling_metadata.json"

        if model_path is None:
            model_path = os.path.join(baseline_dir, "lgb_occupancy_model.joblib")
            if not os.path.exists(model_path):
                model_path = "lgb_occupancy_model.joblib"

        # 1. 정규화 파라미터 로드
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"메타데이터 파일 {metadata_path}을 찾을 수 없습니다.")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.mean = np.array(self.metadata['mean'], dtype=np.float32)
        self.scale = np.array(self.metadata['scale'], dtype=np.float32)
        self.features = self.metadata.get('features', ['CO2_slope', 'Humidity'])
        
        # 2. 학습된 모델 로드
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일 {model_path}을 찾을 수 없습니다.")
        self.model = joblib.load(model_path)
        
        # 3. 실시간 CO2 계산용 슬라이딩 윈도우 설정
        self.poll_interval = poll_interval_sec
        self.window_size_min = window_size_min
        self.max_buffer_size = int((window_size_min * 60) / poll_interval_sec)
        self.co2_buffer = []
        
        # 4. 상태 레이블 정의
        self.labels = {
            0: f"{COLOR_GREEN}{COLOR_BOLD}[비어 있음 (VACANT)]{COLOR_RESET}",
            1: f"{COLOR_RED}{COLOR_BOLD}[사람 있음 (OCCUPIED)]{COLOR_RESET}"
        }

    def add_sensor_data(self, co2_value):
        self.co2_buffer.append(co2_value)
        if len(self.co2_buffer) > self.max_buffer_size:
            self.co2_buffer.pop(0)

    def is_buffer_ready(self):
        return len(self.co2_buffer) >= self.max_buffer_size

    def predict(self, current_humidity):
        if not self.is_buffer_ready():
            return 0, 0.0, 0.0
            
        # 1. 실시간 CO2 기울기 계산 (ppm/minute)
        co2_now = self.co2_buffer[-1]
        co2_past = self.co2_buffer[0]
        co2_slope = (co2_now - co2_past) / self.window_size_min
        
        # 2. 피처 벡터 구성 및 정규화
        raw_features_list = []
        for feature in self.features:
            if feature == 'CO2_slope':
                raw_features_list.append(co2_slope)
            elif feature == 'Humidity':
                raw_features_list.append(current_humidity)
            elif feature == 'CO2':
                raw_features_list.append(co2_now)
                
        raw_features = np.array([raw_features_list], dtype=np.float32)
        scaled_features = (raw_features - self.mean) / (self.scale + 1e-8)
        
        # 3. 추론
        occupancy_pred = int(self.model.predict(scaled_features)[0])
        probability = float(self.model.predict_proba(scaled_features)[0][occupancy_pred])
        
        return occupancy_pred, probability, co2_slope

    def print_ascii_wave(self, co2_val, state_str, conf, slope):
        """
        터미널에 실시간 CO2 농도를 아스키 차트 그래프로 시각화 출력
        """
        width = 40
        # 400 ~ 1200 ppm 범위를 0 ~ width 칸으로 맵핑
        co2_clamped = max(400.0, min(1200.0, co2_val))
        pos = int((co2_clamped - 400.0) / 800.0 * width)
        pos = max(0, min(width - 1, pos))
        
        line = [" "] * width
        for i in range(pos):
            line[i] = "░"
        line[pos] = "█"
        line_str = "".join(line)
        
        # 출력 양식: 그래프 | 현재 수치 | 기울기 | AI 판정 결과 (신뢰도)
        sys.stdout.write(
            f"|{line_str}| {co2_val:5.1f}ppm (기울기: {slope:+.2f} ppm/m) | {state_str} (신뢰도: {conf*100:5.1f}%)\r"
        )
        sys.stdout.flush()

def main():
    model_file = "lgb_occupancy_model.joblib"
    metadata_file = "co2_scaling_metadata.json"
    
    if not os.path.exists(model_file) or not os.path.exists(metadata_file):
        print(f"[오류] {model_file} 또는 {metadata_file} 파일이 존재하지 않습니다.")
        print("파일들이 정상적으로 전송되었는지 확인해 주세요.")
        return
        
    demo = CO2VisualDemo(model_path=model_file, metadata_path=metadata_file, poll_interval_sec=10, window_size_min=5)
    
    print("\n" + "="*95)
    print(f"{COLOR_CYAN}{COLOR_BOLD}  SafeNest CO2 기반 AI 재실 감지 실시간 모니터링 데모 (시뮬레이션){COLOR_RESET}")
    print("="*95)
    print("  * 0 ~ 4분 (초기 버퍼 수집): CO2 450ppm 유지 (비재실 상태)")
    print("  * 5 ~ 9분 (진입 및 호흡): CO2가 450ppm에서 950ppm까지 급상승 (재실 발생)")
    print("  * 10 ~ 14분 (장기 재실 정체): CO2가 950ppm에 정체 (기울기는 0에 수렴하지만 절대 농도 유지)")
    print("  * 15 ~ 18분 (환기 및 퇴실): 환기 팬 작동으로 CO2가 다시 450ppm으로 급하강")
    print("  * 터미널 창에 실시간 CO2 아스키 그래프와 AI 판정 결과가 갱신됩니다. (1초마다 업데이트)")
    print("  * 종료하려면 Ctrl + C를 누르세요.\n")
    time.sleep(3.0)

    t_sec = 0
    base_co2 = 450.0
    current_humidity = 35.0
    pred_class = 0
    conf = 1.0
    slope = 0.0

    try:
        while True:
            # 시뮬레이션 타임라인 정의 (각 구간은 수집 속도가 빠르므로 압축하여 진행)
            # 버퍼(30개)를 빨리 채우고 변화를 쉽게 보실 수 있게 타임라인을 압축 시뮬레이션 합니다.
            # 1 step = 10초 센서 업데이트 주기
            
            # 0~30 step (0~300초): 초기 버퍼 수집 및 비재실
            if t_sec < 300:
                base_co2 += np.random.normal(loc=0.0, scale=0.5)
                current_humidity += np.random.normal(loc=0.0, scale=0.01)
                
            # 30~60 step (300~600초): 진입으로 인한 CO2 급상승
            elif t_sec < 600:
                base_co2 += np.random.normal(loc=17.0, scale=1.5)  # 분당 약 100ppm 상승 효과
                current_humidity += np.random.normal(loc=0.1, scale=0.02)
                
            # 60~90 step (600~900초): 정체 (호흡으로 농도는 유지되나 상승세 멈춤)
            elif t_sec < 900:
                base_co2 += np.random.normal(loc=0.0, scale=0.8)
                current_humidity += np.random.normal(loc=0.0, scale=0.01)
                
            # 90 step 이상: 퇴실 및 환기로 급하강
            else:
                base_co2 += np.random.normal(loc=-20.0, scale=2.0)
                current_humidity += np.random.normal(loc=-0.15, scale=0.02)
                if base_co2 < 450.0:
                    base_co2 = 450.0
                if current_humidity < 35.0:
                    current_humidity = 35.0

            # 버퍼에 값 추가
            demo.add_sensor_data(base_co2)
            
            # 추론
            if demo.is_buffer_ready():
                pred_class, conf, slope = demo.predict(current_humidity)
                state_str = demo.labels.get(pred_class, "알 수 없음")
            else:
                state_str = f"버퍼 수집 중 ({len(demo.co2_buffer)}/{demo.max_buffer_size})"
                conf = 0.0
                slope = 0.0

            # 아스키 파형 출력
            demo.print_ascii_wave(base_co2, state_str, conf, slope)
            
            # 시뮬레이션용 시간 축적 및 대기 시간 단축 (실감나게 보기 위해 0.1초 단위 대기)
            time.sleep(0.1)
            t_sec += 10 # 1 step = 10초 흐름
            
    except KeyboardInterrupt:
        print("\n\nCO2 재실 모니터링 데모를 종료합니다.")

if __name__ == "__main__":
    main()
