# PHASE 4 — Raspberry Pi Sensor State Manager

## 목적

통신 계층에서 검증된 packet을 센서별 최신 상태로 변환한다. 마지막으로 받은 숫자를 현재 정상값처럼
표시하지 않도록 `last_received`, `last_valid`, `connected`, `stale`, `valid`을 분리한다.

## 기본 freshness

| 센서 | TTL |
|---|---:|
| mmWave | 3초 |
| Thermal | 3초 |
| CO₂ | 10초 |
| PIR | 10초 |

CO₂는 약 5초 cadence이므로 3초 공통 TTL을 사용하지 않는다.

## 센서 상태

| 상태 | 의미 |
|---|---|
| `NO_DATA` | 한 번도 packet을 받지 않음 |
| `LIVE` | 연결·형식·freshness가 모두 정상 |
| `INVALID` | 최신 packet은 도착했지만 센서 값이 무효 |
| `STALE` | 연결 표시는 남아 있으나 TTL 초과 |
| `DISCONNECTED` | 해당 ESP32 peer 연결 종료 |

각 센서 상태는 다음 필드를 제공한다.

```json
{
  "status": "LIVE",
  "connected": true,
  "stale": false,
  "valid": true,
  "current": true,
  "last_update": 0.0,
  "last_valid_at": 0.0,
  "age_seconds": 0.0,
  "ttl_seconds": 3.0,
  "values": {}
}
```

freshness 계산에는 monotonic clock을 사용하고 외부 schema에는 Unix wall-clock을 제공한다.

## 현재 telemetry mapping

- mmWave: respiration, heart, 각 valid flag
- CO₂: ppm, valid flag
- PIR: motion boolean
- Thermal: dimensions, sequence, raw min/max, frame availability

현재 ESP32 TCP telemetry에는 mmWave presence가 없다. 따라서 state manager는 presence를 추정하지 않고
`presence=null`, `presence_available=false`로 명시한다. PHASE 5/Risk 연결 전에 firmware schema를 확장하거나
검증된 별도 presence source를 연결해야 한다.

Thermal `pixel_bytes`는 상태 JSON에 넣지 않는다. `latest_thermal_frame()`을 통해 AI/heatmap consumer에만
전달하여 API와 DB에 매 frame 9,920 bytes가 복제되는 것을 막는다.

## Device health

ESP32 scalar telemetry의 queue, transport, CO₂ 취득, Thermal status query 누적값은 센서별 값이 아니라
장치 런타임 health다. 따라서 canonical 위치는 state snapshot의 top-level `device_health`이며, legacy
소비자 호환을 위해 `sensors.mmwave.values.health`에는 같은 snapshot의 복사본을 제공한다. 이 alias는
별도의 mutable state가 아니며, legacy sender가 health를 보내지 않으면 값은 `null` 또는 부재로 남을 수 있다.

`/api/status`와 `/api/sensors`도 top-level `device_health`를 노출한다. Health counter는 AI/risk 입력이나
센서 presence 판정이 아니다.

## Raspberry Pi 실행

```bash
cd ~/integration
source .venv/bin/activate
python3 -m gateway.run_state_gateway \
  --host 0.0.0.0 \
  --port 9000 \
  --packet-deadline 5 \
  --snapshot-interval 1
```

stdout에는 1초마다 JSON snapshot이 출력되고 통신 오류는 stderr로 분리된다.

## PHASE 5 입력 경계

AI provider는 `status == LIVE`인 값만 현재 입력으로 사용한다. `INVALID`, `STALE`, `DISCONNECTED`,
`NO_DATA` 값은 모델에 넣지 않고 명시적인 unavailable `InferenceResult`로 변환해야 한다.
