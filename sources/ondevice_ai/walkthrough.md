# SafeNest V5 On-Device AI 워크스루

## 1. 시작점과 버전

V5 작업본은 `version_archives/2026-08-03/SafeNest_V4_P0_FINAL/`의 검증된 파일을 복제해 생성했다. V4 스냅샷과 원본의 포함 파일 SHA-256은 일치한다. 프로젝트는 V5지만 배포 모델 세 개는 재학습·변환하지 않았으며 모두 `v0.1.0`이다.

## 2. 한 번의 step 흐름

1. `integrated_node/run_node.py`가 `config/sensors.yaml`을 검증한다.
2. mock adapter 또는 팀원이 주입한 provider에서 `InferenceResult`를 읽는다.
3. provider 결과의 고정 `sensor_id`, 유한 timestamp/score/confidence/latency, 오류 계약을 검증한다.
4. `risk/risk_engine.py`가 센서별 stale TTL을 `risk/fallback.py`에 전달한다.
5. 위험도와 시스템 건강 상태를 분리해 `metadata.schema_version=5.0` JSON Lines로 출력한다.

공식 융합 엔진은 `risk/risk_engine.py`다. `integrated_node/safenest_risk_engine.py`는 legacy compatibility에만 남아 있다.

## 3. 센서 AI 계약

- Thermal-44: AI 입력 `(62, 80)` float32, finite. AI 계층이 frame min-max normalization, INT8 quantization, TFLite 추론, `HUMAN_FALL → S4=1.0`을 담당한다.
- mmWave: finite phase와 strictly increasing timestamp. AI 계층이 300-sample window, metadata normalization, TFLite class mapping을 담당한다.
- CO₂: `co2_ppm`, `humidity_percent`, `temperature_celsius`, Unix timestamp. AI 계층이 history, slope, `[CO2_slope, Humidity, CO2]`, TFLite 및 threshold rule을 담당한다.
- PIR: boolean motion과 Unix timestamp. AI 계층이 startup grace, 마지막 움직임 이후 시간, 15초 long-no-motion rule을 담당한다.

SPI/I2C/UART/GPIO 통신과 프레임 규약은 이 배포판에서 구현하지 않는다. 해당 코드는 팀원 provider 경계 밖에서 준비한다.

## 4. Fail-closed 예

외부 provider 없이 real mode를 시작하면 각 센서 결과는 `valid=false`, `error=EXTERNAL_SENSOR_PROVIDER_REQUIRED`다. 네 센서가 모두 관측 불가능하므로 `system_health=FAILED`, `risk_score=null`, `risk_level=null`이다. 관측 불가능 상태를 정상 점수 0으로 바꾸지 않는다.

## 5. 검증 범위

- P0 온디바이스 AI 코어: 검증 완료
- 모델 metadata/interpreter load/SHA-256: 검증 대상
- Mock end-to-end: 검증 대상
- provider injection 및 장애 분리: 검증 대상
- 실센서 드라이버와 실센서 통합: 진행 예정
- 웹 UI: 범위 밖
- Raspberry Pi 5 성능: 실측 전까지 미검증
