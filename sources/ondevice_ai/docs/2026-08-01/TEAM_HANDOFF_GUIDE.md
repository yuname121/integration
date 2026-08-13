# SafeNest V4 온디바이스 AI 팀원 통합 인수인계 가이드 (프롬프트 모음)

본 문서는 **SafeNest V4 온디바이스 AI 패키지를 팀원들이 본인 담당 분야(센서 하드웨어, 라즈베리 파이 5, 웹 UI)에 즉시 적용할 수 있도록 하나로 통합한 단일 프롬프트 가이드 문서**입니다.

---

## 1. 온디바이스 AI 시스템 및 연동 규격 개요

### (1) 센서별 위험도 수식
온디바이스 위험도 점수 $R$은 4개 센서 채널의 가중 합산으로 산출됩니다:

$$R = 100 \times (0.35 S_1 + 0.35 S_2 + 0.15 S_3 + 0.15 S_4)$$

- **$S_1$ (mmWave)**: 호흡 이상/무호흡 점수 $[0.0, 1.0]$ (무호흡 1.0, 이상 0.5, 정상 0.0)
- **$S_2$ (CO2)**: SCD40 재실/고농도 점수 $[0.0, 1.0]$ (재실/1500ppm 초과 1.0, 정상 0.0)
- **$S_3$ (PIR)**: 인체 움직임 점수 $[0.0, 1.0]$ (15초 이상 미움직임 1.0, 움직임 0.0)
- **$S_4$ (Thermal-44)**: 80x62 열화상 낙상 점수 $[0.0, 1.0]$ (낙상 1.0, 정상 0.0)

### (2) 비상 오버라이드 (Emergency Override)
- Thermal 낙상 감지 ($S_4=1.0$) 또는 mmWave 무호흡 감지 ($S_1=1.0$) 발생 시, 가중 합산을 우회하여 즉시 **$R=100.0$ (`DANGER`)** 처리됩니다.

---

## 2. 담당 분야별 연동 프롬프트 모음 (복사하여 즉시 활용)

아래 프롬프트 중 본인이 담당한 분야의 프롬프트를 복사하여 AI 또는 개발 환경에 입력하면 즉시 작업을 수행할 수 있습니다.

---

### 📌 프롬프트 1: Thermal-44 열화상 하드웨어 담당자용
```text
[역할] SafeNest Thermal-44 열화상 하드웨어 엔지니어

[목표] `SafeNest_V4_OnDevice_AI/sensors/thermal44/thermal44_driver.py` 파일 내에 실기기 I2C/SPI 드라이버를 연결하십시오.

[연동 수칙]
1. 라즈베리 파이 5의 I2C 주소 0x33 및 SPI 디바이스 0을 초기화합니다.
2. 10Hz 주기로 원시 4960픽셀 (80x62) IR 온도를 읽어옵니다.
3. `sensors.thermal44.frame_parser.ThermalFrameParser`를 사용해 (62, 80) 행렬로 파싱합니다.
4. `read()` 메서드 호출 시 규격화된 `InferenceResult` 객체를 반환합니다.
   - 낙상(HUMAN_FALL) 감지 시 score S4 = 1.0, state = "HUMAN_FALL"
   - 정상(HUMAN_NORMAL) 시 score S4 = 0.0, state = "HUMAN_NORMAL"
5. `risk/risk_engine.py` 수식이나 `models/model_manifest.json` 설정 코드는 절대 수정하지 마십시오.
```

---

### 📌 프롬프트 2: mmWave 레이더 담당자용
```text
[역할] SafeNest mmWave 레이더 하드웨어 엔지니어

[목표] Seeed Studio MR60BHA2 (60GHz) 센서를 `SafeNest_V4_OnDevice_AI/sensors/mmwave/mmwave_adapter.py`에 연결하십시오.

[연동 수칙]
1. 라즈베리 파이 5의 시리얼 포트 `/dev/ttyAMA0` (115200 baud)를 엽니다.
2. 10Hz 주기로 30초(300 샘플) 롤링 링버퍼(Ring Buffer)를 유지합니다.
3. `models/mmwave/sensor_stats_metadata_v0.1.0.json`에 정의된 mean(0.00609)과 std(2.50138)로 Z-score 정규화를 적용합니다.
4. TFLite 모델 추론 결과를 위험도 점수 S1으로 매핑합니다:
   - APNEA (무호흡) -> S1 = 1.0
   - RAPID_OR_ABNORMAL (이상 호흡) -> S1 = 0.5
   - NORMAL (정상 호흡) -> S1 = 0.0
5. 센서 단선이나 NaN 발생 시 `valid=False`로 반환하고, 위험도 엔진 코드는 수정하지 마십시오.
```

---

### 📌 프롬프트 3: CO₂ 센서 담당자용
```text
[역할] SafeNest CO₂ 센서 엔지니어

[목표] SCD40 I2C 센서를 `SafeNest_V4_OnDevice_AI/sensors/co2/co2_adapter.py`에 연결하십시오.

[연동 수칙]
1. I2C 주소 0x62에서 CO2 (ppm) 및 습도 (%)를 수신하고, 최근 30초 히스토리 윈도우 기반으로 CO2_slope (ppm/min)를 산출합니다.
2. 모델 입력 피처 순서를 정확히 유지하십시오: `[CO2_slope, Humidity, CO2]`
3. `CO2Interpreter.predict(co2_slope, humidity, co2_ppm)`를 호출합니다.
4. 재실/고농도(OCCUPIED_ELEVATED)이거나 CO2 > 1500 ppm인 경우 score S2 = 1.0, 그렇지 않은 경우 0.0을 반환합니다.
```

---

### 📌 프롬프트 4: PIR 인체 감지 센서 담당자용
```text
[역할] SafeNest PIR 센서 엔지니어

[목표] 라즈베리 파이 5 GPIO 17번 핀을 `SafeNest_V4_OnDevice_AI/sensors/pir/pir_adapter.py`에 연결하십시오.

[연동 수칙]
1. GPIO 17번 핀의 High/Low 움직임 인터럽트를 모니터링합니다.
2. 마지막 움직임이 감지된 시각 이후 경과 시간을 추적합니다.
3. 미움직임 지연 시간이 15.0초를 초과(LONG_NO_MOTION)하면 score S3 = 1.0, 움직임 감지 시 0.0을 반환합니다.
4. GPIO 핀 고장 시 `valid=False`를 보고하십시오.
```

---

### 📌 프롬프트 5: 라즈베리 파이 5 하드웨어 & 시스템 설치 담당자용
```text
[역할] SafeNest 라즈베리 파이 5 시스템 통합 엔지니어

[목표] 라즈베리 파이 5 환경에서 온디바이스 AI 파이프라인을 구축하고 24시간 자동 실행 서비스로 등록하십시오.

[설치 및 실행 절차]
1. OS 시스템 패키지 설치:
   sudo apt-get update && sudo apt-get install -y python3-pip python3-venv i2c-tools
2. raspi-config에서 I2C, SPI, Serial UART(/dev/ttyAMA0) 인터페이스 활성화.
3. 가상환경 구축 및 패키지 설치:
   cd SafeNest_V4_OnDevice_AI
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements-pi.txt
4. 테스트 검증:
   python3 -m unittest discover -s tests -p "test_*.py"
5. 프로덕션 메인 노드 실행:
   python3 integrated_node/run_node.py --mode real
6. systemd 자동 실행 서비스 등록 (`/etc/systemd/system/safenest.service` 작성 후 enable)
```

---

### 📌 프롬프트 6: 웹 UI 개발 팀원 연동용
```text
[역할] SafeNest 웹 UI 개발자

[목표] `integrated_node/run_node.py`가 stdout으로 출력하는 JSON Lines 실시간 스트림 데이터를 수신하여 웹 시각화 대시보드를 구성하십시오.

[연동 규격 및 데이터 포맷]
1. 파이프(Pipe) 또는 서브프로세스로 `python3 integrated_node/run_node.py --mode mock` (또는 `--mode real`)을 실행합니다.
2. 표준 출력(stdout)으로 연신 들어오는 한 줄 단위의 JSON 데이터(JSON Lines)를 파싱합니다.
3. 데이터 예시:
   {
     "timestamp": 1722150000.0,
     "risk_score": 100.0,
     "risk_level": "DANGER",
     "system_health": "HEALTHY",
     "degraded_mode": false,
     "invalid_sensors": [],
     "stale_sensors": [],
     "component_scores": {
       "mmwave": 0.0,
       "co2": 0.0,
       "pir": 0.0,
       "thermal": 1.0
     },
     "is_emergency": true,
     "reasons": ["EMERGENCY_HUMAN_FALL"],
     "sensors": {
       "thermal44": {"score": 1.0, "state": "HUMAN_FALL", "valid": true},
       "mmwave": {"score": 0.0, "state": "NORMAL", "valid": true},
       "co2": {"score": 0.0, "state": "UNOCCUPIED_NORMAL", "valid": true},
       "pir": {"score": 0.0, "state": "MOTION", "valid": true}
     },
     "level": "DANGER",
     "system_status": "OK",
     "fallback_used": false
   }
4. 신규 연동은 `risk_level`, `system_health`, `degraded_mode`, `invalid_sensors`, `stale_sensors`, `component_scores`를 기준으로 합니다. `level`, `system_status`, `fallback_used`는 기존 수신기 호환용 별칭입니다.
5. `risk_score`/`risk_level`은 사람 위험, `system_health`는 센서·모델 파이프라인 건강 상태입니다. `FAILED`일 때 위험값은 `null`이며 이를 `NORMAL` 또는 `0`으로 표시하면 안 됩니다.
6. 웹 서버나 HTML 화면 구축은 웹 UI 팀의 영역이며 온디바이스 AI 노드는 표준 JSON Lines 스트림만 제공합니다.
```

---

### 📌 프롬프트 7: QA 및 모델 재학습 담당자용
```text
[역할] SafeNest QA 및 AI 모델 재학습 엔지니어

[목표] 전처리 NPZ 데이터셋으로 모델을 재학습하고 유닛 테스트를 검증하십시오.

[절차]
1. 전체 테스트 스위트를 자동 발견 방식으로 실행하고 `FAILED=0`, `ERRORS=0`을 확인합니다. 오프라인 NPZ 미포함으로 인한 SKIP은 사유를 별도 기록합니다:
   python3 -m unittest discover -s tests -p "test_*.py"
2. CO2 / mmWave 전처리 NPZ 로드:
   `datasets/co2/processed/co2_occupancy_v1.npz`
   `datasets/mmwave/processed/mmwave_respiration_v1.npz`
3. INT8 TFLite 모델 재학습 후 내보내기 및 `models/model_manifest.json` 내 SHA-256 해시 갱신.
```
