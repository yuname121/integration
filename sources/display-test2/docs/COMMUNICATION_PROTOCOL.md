# SafeNest 통신 규격 요약

## 포트

| 용도 | 프로토콜 | 기본 포트 | 방향 |
|---|---|---:|---|
| 센서 데이터 | TCP | 9000 | ESP32 → Raspberry Pi |
| LCD/API | HTTP | 8080 | LCD·노트북 → Raspberry Pi |

## TCP 패킷 헤더

모든 정수는 네트워크 바이트 순서(big-endian)입니다. 헤더 크기는 16바이트입니다.

| 오프셋 | 크기 | 필드 | 값/설명 |
|---:|---:|---|---|
| 0 | 4 | magic | ASCII `SNST` |
| 4 | 1 | version | `1` |
| 5 | 1 | type | `1`: telemetry JSON, `2`: thermal frame |
| 6 | 2 | flags | v1에서는 `0` |
| 8 | 4 | sequence | 패킷 순번 |
| 12 | 4 | payload_length | 뒤따르는 payload의 바이트 수 |

TCP는 패킷 경계를 보존하지 않으므로 Raspberry Pi는 `recv_exact()`로 헤더와 payload 길이만큼 끝까지 읽습니다.

## Type 1: 센서 telemetry JSON

스키마 이름은 `safenest.telemetry.v1`입니다.

```json
{
  "schema": "safenest.telemetry.v1",
  "device_id": "esp32-01",
  "seq": 42,
  "uptime_ms": 12345,
  "resp_rate_bpm": 16.25,
  "heart_rate_bpm": 72.5,
  "co2_ppm": 820,
  "pir_motion": true,
  "valid": {
    "respiration": true,
    "heart": true,
    "co2": true
  }
}
```

유효하지 않거나 아직 측정되지 않은 수치 값은 `null`로 보냅니다. `valid` 플래그와 값을 함께 확인해야 합니다.

## Type 2: 열화상 frame

payload는 16바이트 metadata 다음에 80 × 62개의 `uint16` 픽셀이 이어집니다.

| 오프셋 | 크기 | 필드 |
|---:|---:|---|
| 0 | 2 | width (`80`) |
| 2 | 2 | height (`62`) |
| 4 | 4 | frame_sequence |
| 8 | 4 | uptime_ms |
| 12 | 2 | minimum_raw |
| 14 | 2 | maximum_raw |
| 16 | 9,920 | 픽셀 4,960개, 각 `uint16` big-endian |

현재 Raspberry Pi 서버는 전체 payload를 안전하게 읽은 뒤 프레임 수만 증가시킵니다. 열화상 보정·시각화는 이 공유본의 범위가 아닙니다.

## HTTP API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/display` | Raspberry Pi LCD 화면 |
| GET | `/control` | 노트북 원격 제어 화면 |
| GET | `/api/state` | 화면 상태와 최신 센서 데이터 |
| POST | `/api/state` | 화면 상태 또는 공간 이름 변경 |
| GET | `/health` | 서버, 부저, 센서 연결 상태 |

상태 변경 예시:

```bash
curl -s -X POST http://127.0.0.1:8080/api/state \
  -H 'Content-Type: application/json' \
  -d '{"state":"emergency"}'
```

허용 상태:

- `normal-empty`
- `normal-occupied`
- `warning`
- `danger`
- `emergency`
- `offline`

## 장애 확인 순서

1. Raspberry Pi에서 `ss -ltn | grep ':9000 '`으로 TCP 수신 포트를 확인합니다.
2. ESP32 시리얼 로그에서 Wi-Fi 연결과 `Raspberry Pi connected`를 확인합니다.
3. `/health`의 `sensors.status`, `peer`, `age_seconds`, `listener_error`를 확인합니다.
4. HTTP 서버는 정상이지만 LCD가 보이지 않으면 `logs/chromium.log`를 확인합니다.
5. 종료할 때는 `bash stop_lcd.sh`를 사용합니다.
