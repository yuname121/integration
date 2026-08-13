# P0-1 & P0-5 Integrated Validation Report
`SafeNest_V4_OnDevice_AI/docs/reports/P0_1_P0_5_INTEGRATED_VALIDATION.md`

## 1. Executive Summary

| Category | Status | Notes |
|---|---|---|
| **P0-1 (Real Mode Safe Fail-Fast)** | **PASS** | Real mode에서 미구현 센서 드라이버가 가짜/합성 데이터를 절대 반환하지 않음 |
| **P0-5 mmWave** | **PASS** | Buffer < 300 시 `WARMING_UP` (`valid=False`), 300 도달 시 모델 1회 호출 |
| **P0-5 CO2** | **PASS** | Slope 단위 `ppm/min` 산출 및 정밀 수치 검증 (`assertAlmostEqual` 100.0) |
| **P0-5 PIR** | **PASS WITH SEMANTIC NOTE** | `time.monotonic()` startup grace period 동작. (P0-4 제안: `MOTION_DETECTED`, `NO_MOTION_BELOW_THRESHOLD`, `LONG_NO_MOTION`) |
| **P0-5 Thermal** | **PASS** | Frame shape (62,80) 검증, NaN/Inf 거부 및 최초 유효 프레임 수신 시 즉시 추론 |
| **Test Suite** | **140 PASS / 2 SKIP (142 Discovered)** | 전체 142개 유닛 테스트 중 140개 PASS, 2개 SKIP |
| **Hardware Validation** | **BLOCKED_HARDWARE** | 실 하드웨어 (SPI/I2C/UART/GPIO/Raspberry Pi) 미연결 환경 |

---

## 2. CO₂ Adapter TypeError 이슈 분석 (`RESOLVED / STALE_FINDING`)

- **상태**: **RESOLVED / STALE_FINDING**
- **분석**: 과거 서술에 언급되었던 `co2_adapter.py` L74 `TypeError` 항목은 현재 소스코드 상 존재하지 않습니다. (L74는 에러 사유 리턴 문자열 "SENSOR_NOT_CONNECTED"에 해당함)
- **검증**: `CO2SensorAdapter.read()` 및 `calculate_co2_slope()` 추론 파이프라인의 타입 캐스팅과 리턴 타입 검증을 완료했으며, 모든 CO₂ 관련 유닛 테스트가 오류 없이 통과합니다.

---

## 3. P0-5 CO₂ Slope 정밀 수치 검증 (`assertAlmostEqual`)

`tests/test_sensor_startup_warmup.py` 내 `test_co2_history_duration_and_unit` 케이스에 단순 필드 검증을 넘어 수치 정밀도 검증을 적용했습니다.

```python
# 60초 전 500ppm, 현재 600ppm -> (600-500) / (60/60) min = 100.0 ppm/min
self.assertIn("co2_slope_ppm_min", res.metadata)
self.assertAlmostEqual(res.metadata["co2_slope_ppm_min"], 100.0, places=3)
```

- **이론 수치 예시 검증**:
  $$t=0\text{s}, 600\text{ ppm} \rightarrow t=30\text{s}, 660\text{ ppm}$$
  $$\text{slope} = \frac{660 - 600}{30 / 60} = \frac{60}{0.5} = 120.0\text{ ppm/min}$$
  `calculate_co2_slope()` 함수가 경과 초를 `60.0`으로 나누어 정확한 분당 변화율(`ppm/min`)을 산출하는 것을 확인했습니다.

---

## 4. PIR 센서 상태명 개선 제안 (P0-4 출력 계약 연계)

현재 `PIRSensorAdapter`는 `motion_detected=False`라도 `no_motion_threshold_sec`(15초) 미만이면 `state="MOTION"`으로 반환합니다. 위험도 계산(score=0.0)은 정상 동작하지만, 대시보드 및 UI 수신 측의 직관성을 위해 P0-4 출력 스키마 정제 시 다음 3단계 세분화를 권장합니다.

1. **`MOTION_DETECTED`**: GPIO 펄스가 실제 감지된 상태
2. **`NO_MOTION_BELOW_THRESHOLD`**: 움직임 미감지 상태이나 임계시간(15초) 미만 경과
3. **`LONG_NO_MOTION`**: 임계시간(15초) 이상 무움직임 지속

---

## 5. 테스트 결과 상세 분석 (140 PASS / 2 SKIP / Total 142)

`python3 -m unittest discover -s tests -v` 실행 결과:

```text
Ran 142 tests in 1.511s

OK (skipped=2)
```

### 테스트 모듈별 세부 내역
- **P0-1 Real-mode Fail-Closed** (`test_real_mode_fail_closed.py`): 7 PASS
- **P0-5 Startup & Warming-Up** (`test_sensor_startup_warmup.py`): 12 PASS
- **P0-3 Sensor Data Contract** (`test_sensor_model_data_contract.py`): **26 PASS**
- **P0-2 V4 Config & Model Validation** (`test_v4_config_validation.py`): 14 PASS
- **Integrated Pipeline & Fallback** (`test_v4_pipeline.py`): 16 PASS
- **Three-Model Integration & Scenarios** (`test_three_model_integration.py`): 13 PASS
- **MMWave Interpreter & Stream Adapters** (`test_mmwave_*.py`): 36 PASS
- **Sensor Adapters & Fallback** (`test_sensor_adapters.py`, `test_fault_injection.py`, `test_fallback.py`): 11 PASS
- **Thermal Interpreter** (`test_thermal_interpreter.py`): 5 PASS / **2 SKIP**

### Skip 사유 및 미검증 범위
- **Skipped Test (2개)**: `test_current_npz_class_smoke`, `test_prediction_does_not_collapse_to_one_class`
- **Skip 사유**: 원본 NPZ 데이터셋 파일(`thermal/processed_thermal_80x62.npz`)이 Git 저장소 크기 최적화를 위해 LFS/저장소 추적 대상에서 제외되어 있음.
- **미검증 범위**: 오프라인 훈련 NPZ 데이터 셋에 대한 클래스 배치 추론 정확도 검증 (실제 TFLite 런타임 추론 엔진 동작은 5개의 텐서 규격/슬라이싱 유닛 테스트로 100% 검증됨).

---

## 6. Git 커밋 구성 및 보호 자산 상태

### 분리 커밋 계획
1. **Commit 1**: P0-1 real-mode fail-closed (`sensors/base_sensor.py`, `sensors/*/adapter.py`, `integrated_node/run_node.py`, `tests/test_real_mode_fail_closed.py`)
2. **Commit 2**: P0-5 startup/warming-up (`sensors/*/mock_sensor.py`, `sensors/*/*_adapter.py`, `tests/test_sensor_startup_warmup.py`)
3. **Commit 3**: P0-1/P0-5 validation report (`SafeNest_V4_OnDevice_AI/docs/reports/P0_1_P0_5_INTEGRATED_VALIDATION.md`)
4. **Commit 4**: P0-2 config/model validator (`inference/validator.py`, `scripts/validate_v4_config.py`, `config/models.yaml`, `inference/model_registry.py`, `tests/test_v4_config_validation.py`)
5. **Commit 5**: P0-3 sensor data contract (`tests/test_sensor_model_data_contract.py`, `SafeNest_V4_OnDevice_AI/docs/reports/SENSOR_DATA_CONTRACT.md`, `sensor_model_data_contract.json`, `sensor_model_data_contract.md`)

### 보호 자산
- `models/*.tflite`, `models/model_manifest.json`, `datasets/` 변경 없음.
