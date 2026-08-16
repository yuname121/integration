# PHASE 5 — On-device AI 연결

## 결과

기존 모델 3개와 추론 코드를 `sources/ondevice_ai/`에 동결했고, `ai/`에서 지연 로드하는 통합 경계를 추가했다. 모델 파일은 재학습하거나 변환하지 않았으며 manifest의 SHA-256과 파일 크기를 자동 테스트로 검증한다.

| 센서 | 모델 | 입력 계약 | 현재 통신 데이터 | 동작 |
|---|---|---|---|---|
| Thermal | `thermal_fall_int8` 0.1.0 | `(1,62,80,1)` int8, 프레임별 min-max | 80×62 U16 BE 프레임 | 실행 가능 |
| mmWave | `mmwave_resp_int8` 0.1.0 | 10 Hz `resp_phase` 300개 | 호흡수·심박수 scalar만 수신 | `INPUT_UNAVAILABLE` |
| CO₂ legacy runtime primary | `co2_occupancy_int8` 0.1.0 | `[CO2_slope, Humidity, CO2]` | ppm만 수신 | `INPUT_UNAVAILABLE`, rule fallback 유지 |
| CO₂ B-complete offline candidate | 별도 검증/승격 대기 | `[CO2, CO2_slope]` | measurement event provenance 수신 | humidity 불필요, history/Capture는 후속 범위 |
| PIR | 모델 없음 | boolean motion | motion 수신 | rule 결과 |

mmWave와 CO₂ 입력을 임의 생성하지 않는다. 이후 ESP32 packet schema가 필요한 필드를 실제로 보내면 현재 파이프라인이 window와 feature를 검증한 뒤 모델을 호출할 수 있다.

## Thermal 처리 경로

1. strict TCP decoder가 길이, endian, 메타데이터 min/max를 검증한다.
2. `SensorStateManager`가 최신 immutable frame을 보관한다.
3. AI 파이프라인이 big-endian U16을 `(62,80)` float array로 변환한다.
4. 동결 interpreter가 finite 검사와 학습 코드에서 확인한 프레임별 min-max 정규화를 수행한다.
5. TFLite 결과를 `NOT_HUMAN`, `HUMAN_NORMAL`, `HUMAN_FALL`로 반환한다.

현재 TCP thermal payload에는 센서 die temperature나 보정 계수가 없다. 그러므로 온도(°C)를 추정하지 않으며 모든 결과에 `temperature_calibrated: false`를 표시한다. 모델은 절대온도가 아닌 프레임 내 공간 패턴으로만 실행된다.

## 실패 격리

- 모델은 완전한 최신 입력이 들어올 때만 로드한다.
- LiteRT 패키지 누락, hash/shape 불일치, NaN/Inf, invoke 예외는 해당 센서 결과만 `MODEL_RUNTIME_UNAVAILABLE`로 만든다.
- mmWave adapter의 heuristic fallback은 AI 성공으로 위장하지 않고 unavailable 결과의 참고 메타데이터로만 남긴다.
- PIR rule은 다른 모델 실패와 무관하게 계속 실행된다.
- 결과 객체는 finite number만 허용하며 `json.dumps(..., allow_nan=False)`로 출력한다.

## Raspberry Pi 실행

```bash
cd ~/integration
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r ./sources/ondevice_ai/requirements-pi.txt
python ./gateway/run_ai_gateway.py --host 0.0.0.0 --port 9000
```

출력은 매초 Sensor State revision과 `thermal/mmwave/co2/pir` 결과를 포함한 JSON 한 줄이다. 종료는 `Ctrl+C`이다.

## 검증 범위

자동 테스트는 다음을 확인한다.

- 모델 artifact hash/size
- Thermal U16 frame shape와 결과 매핑
- stale/no-data 입력에서 모델 미호출
- mmWave 300-sample 계약과 fallback provenance
- legacy primary의 습도 계약과 B-complete candidate의 실제 measurement-event history 계약을 구분
- 모델 예외 격리와 PIR rule 지속
- NaN 차단과 strict JSON 직렬화

Windows 번들 환경에는 LiteRT가 없어 실제 `.tflite` invoke는 수행하지 않았다. Raspberry Pi에서 위 의존성을 설치한 뒤 실제 센서 프레임 smoke test와 latency 측정이 남아 있다.

## 변경 전/후

- 변경 전: 모델과 interpreter는 존재하지만 중앙 Sensor State 및 운영 receiver에 연결되지 않았다.
- 변경 후: 수신 → freshness 상태 → 입력 계약 검사 → 지연 추론/rule → JSON-safe 결과가 하나의 실행 진입점으로 연결되었다.

PHASE 5 On-device AI 연결 완료.
