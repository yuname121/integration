# co2_occupancy_detector.py
import os
import json
import time
import numpy as np
import joblib

class CO2OccupancyDetector:
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

        # 1. 정규화 파라미터 및 학습 피처 로드
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"메타데이터 파일 {metadata_path}을 찾을 수 없습니다.")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.mean = np.array(self.metadata['mean'], dtype=np.float32)
        self.scale = np.array(self.metadata['scale'], dtype=np.float32)
        # 피처 이름 목록이 없으면 기존 하위 호환용으로 ['CO2_slope', 'Humidity'] 적용
        self.features = self.metadata.get('features', ['CO2_slope', 'Humidity'])
        
        # 2. 학습된 모델 로드 (LightGBM 또는 Logistic Regression)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일 {model_path}을 찾을 수 없습니다.")
        self.model = joblib.load(model_path)
        print(f"[{model_path}] 모델이 성공적으로 로드되었습니다. 피처 목록: {self.features}")
        
        # 3. 실시간 CO2 계산용 슬라이딩 윈도우 설정
        self.poll_interval = poll_interval_sec  # 센서 데이터 수집 주기 (초)
        self.window_size_min = window_size_min  # 슬라이딩 윈도우 기간 (분)
        
        # 5분을 채우기 위해 큐에 적재되어야 하는 데이터 수 계산
        self.max_buffer_size = int((window_size_min * 60) / poll_interval_sec)
        self.co2_buffer = []
        print(f"버퍼 설정 완료: 수집 주기 {poll_interval_sec}초 | 윈도우 {window_size_min}분 (버퍼 크기: {self.max_buffer_size}개)")
        
    def add_sensor_data(self, co2_value):
        """
        SCD40 센서로부터 새로 입수된 CO2 데이터를 버퍼에 적재합니다.
        """
        self.co2_buffer.append(co2_value)
        if len(self.co2_buffer) > self.max_buffer_size:
            self.co2_buffer.pop(0)

    def is_buffer_ready(self):
        """
        기울기를 계산하기 위해 충분한 시계열 데이터가 확보되었는지 확인합니다.
        """
        return len(self.co2_buffer) >= self.max_buffer_size

    def predict(self, current_humidity):
        """
        최신 습도 값을 받아 실시간 재실 여부를 예측합니다.
        """
        if not self.is_buffer_ready():
            # 버퍼가 아직 가득 차지 않은 경우, 초기 추론은 0으로 처리하거나 
            # 가능한 시점까지의 간이 기울기를 계산할 수 있으나, 안전을 위해 보수적으로 판단
            return 0, 0.0
            
        # 1. 실시간 CO2 기울기(Slope) 계산: (현재 CO2 - N분 전 CO2) / N분
        co2_now = self.co2_buffer[-1]
        co2_past = self.co2_buffer[0]
        
        # ppm/minute 단위 통일
        co2_slope = (co2_now - co2_past) / self.window_size_min
        
        # 2. 피처 목록에 맞게 피처 벡터 동적 구성
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
        
        # 3. 모델 추론 실행
        occupancy_pred = int(self.model.predict(scaled_features)[0])
        probability = float(self.model.predict_proba(scaled_features)[0][occupancy_pred])
        
        return occupancy_pred, probability

# ---------------------------------------------------------------------
# [라즈베리파이 실행 테스트 데몬]
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 라이브러리가 없는 환경에 대응하기 위해 lightgbm, joblib 등을 pip install 하셔야 합니다.
    # 예: pip install lightgbm joblib scikit-learn numpy
    
    # 10초 주기로 센서를 쿼리하여 재실을 감지하는 상황 시뮬레이션
    detector = CO2OccupancyDetector(
        model_path="lgb_occupancy_model.joblib",
        metadata_path="co2_scaling_metadata.json",
        poll_interval_sec=10,  # 10초 주기
        window_size_min=5      # 5분 윈도우
    )
    
    # 가상의 CO2 및 습도 상승 데이터 스트림 생성 (사람이 들어와 호흡하기 시작한 시나리오)
    # CO2: 450ppm에서 10초마다 3ppm씩 증가
    # Humidity: 35% 대역에서 소폭 상승
    base_co2 = 450.0
    current_humidity = 35.5
    
    print("\n--- 실시간 센서 루프 기동 (시뮬레이션) ---")
    for sec in range(0, 360, 10):  # 총 6분간 시뮬레이션 (36개 샘플)
        base_co2 += np.random.normal(loc=3.5, scale=0.8)  # 호흡으로 인한 지속적인 상승
        current_humidity += np.random.normal(loc=0.05, scale=0.02)
        
        detector.add_sensor_data(base_co2)
        
        if detector.is_buffer_ready():
            pred, conf = detector.predict(current_humidity)
            # 버퍼 내의 실제 기울기 출력
            real_slope = (detector.co2_buffer[-1] - detector.co2_buffer[0]) / detector.window_size_min
            print(f"[{sec}초] CO2: {base_co2:.1f}ppm (기울기: {real_slope:+.2f} ppm/m) | 습도: {current_humidity:.2f}% | "
                  f"재실 판단: {'[사람 있음 (1)]' if pred == 1 else '[비어 있음 (0)]'} (신뢰도: {conf*100:.1f}%)")
        else:
            needed = detector.max_buffer_size - len(detector.co2_buffer)
            print(f"[{sec}초] CO2: {base_co2:.1f}ppm | 데이터 수집 중... (기울기 측정을 위해 {needed}개 더 필요)")
            
        time.sleep(0.1)  # 빠른 시뮬레이션을 위해 대기 시간을 0.1초로 단축
