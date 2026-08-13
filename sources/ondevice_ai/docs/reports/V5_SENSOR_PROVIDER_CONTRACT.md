# V5 Sensor Provider Contract

## 공통 인터페이스

V5 node-level provider는 `connect`, `read`, `close` 세 메서드를 제공한다. `read`의 유일한 공통 출력은 `InferenceResult`다. transport 구현과 vendor protocol은 provider 팀 소유이며 이 문서가 레지스터·packet 규약을 추정하지 않는다.

Provider 미주입 real mode 상태 코드는 `EXTERNAL_SENSOR_PROVIDER_REQUIRED`다. 이는 소프트웨어 코어 결함 상태가 아니라 외부 구현 경계가 아직 연결되지 않았다는 의미다. node는 이 상태에서 센서값을 합성하지 않는다.

## InferenceResult invariant

| 필드 | 계약 |
|---|---|
| `sensor_id` | provider key와 동일한 고정 문자열 |
| `timestamp` | finite wall-clock Unix seconds, stale data는 원 timestamp 유지 |
| `score` | finite `0.0..1.0` |
| `state` | 비어 있지 않은 상태 문자열 |
| `confidence` | finite `0.0..1.0` |
| `valid` | 관측과 AI/rule 결과가 신뢰 가능할 때만 true |
| `latency_ms` | finite, 0 이상 |
| `error` | `valid=false`일 때 비어 있지 않은 원인 코드 |
| `metadata` | dict, NaN/Inf 금지 |

Provider 결과 type, sensor ID, timestamp, latency, metadata가 계약을 어기면 node가 원 결과를 채택하지 않고 `PROVIDER_*` 오류의 invalid 결과로 바꾼다.

## Thermal-44 AI 입력 경계

팀 provider/transport가 AI adapter에 전달할 frame:

```text
shape: (62, 80)
dtype: float32
values: finite only
orientation: model training orientation과 동일
```

AI 소유: shape/finite 검증, per-frame min-max, INT8 quantization, TFLite, `HUMAN_FALL → score 1.0`.

팀 소유: I2C 초기화, SPI frame 수신, byte order, header/checksum, raw conversion, 실제 orientation 확인.

## mmWave AI 입력 경계

```text
phase: finite float
timestamp: strictly increasing
presence/quality: metadata 또는 합의 필드
sample cadence: 실제 측정값
```

AI 소유: 300-sample rolling window, timestamp 역전/NaN/Inf 검사, model metadata normalization, TFLite, `NORMAL/RAPID_OR_ABNORMAL/APNEA` mapping.

팀 소유: MR60BHA2 UART command/frame parsing과 phase 의미 검증. 재연결 시 buffer를 초기화한다.

## CO₂ AI 입력 경계

```text
co2_ppm: float
humidity_percent: float
temperature_celsius: float
timestamp: Unix seconds
```

AI 소유: recent history, `CO2_slope`, feature order `[CO2_slope, Humidity, CO2]`, TFLite, high concentration rule, S2.

팀 소유: SCD40 I2C command/CRC와 실제 cadence. CO₂ 단일-array interpreter 호출 문제는 세 scalar 호출로 `RESOLVED` 상태다.

## PIR AI 입력 경계

```text
motion_detected: bool
timestamp: Unix seconds
```

AI 소유: startup grace, elapsed since last motion, 15초 long-no-motion, S3. 팀 소유는 GPIO 초기화와 event acquisition이다.

## Runtime health mapping

- provider 4개 정상: `HEALTHY`
- invalid/stale 일부: `DEGRADED`
- 유효·fresh 센서 0개: `FAILED`, `risk_score=null`, `risk_level=null`
- Thermal `HUMAN_FALL/score=1`: `DANGER`, `R=100`
- mmWave `APNEA/score=1`: `DANGER`, `R=100`

호환 필드 `level`, `system_status`, `fallback_used`를 유지하며 기본 schema는 `metadata.schema_version="5.0"`이다.
