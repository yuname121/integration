# ESP32/Arduino 코드 업데이트 변경 로그

## 범위

이 문서는 `sources/display-test2/esp32_sensor_node/`의 canonical ESP32 flash source에 적용한 변경 사항을 기록합니다. 기존 backend, Raspberry Pi, AI, 웹 대시보드, LCD 및 기타 저장소 파일은 변경하지 않습니다.

## 파일

### 수정

- `sources/display-test2/esp32_sensor_node/esp32_sensor_node.ino`
  - 기존 저장소 버전을 기준으로 최신 로컬 수정본의 MR60 phase, CO₂ physical-event identity 보강, Thermal 무결성 검증, 전송 주기 변경을 병합했습니다.
  - 저장소 기존 계약인 PIR transition identity, TCP/UDP health counter, chunked Thermal UDP 전송은 유지했습니다.

### 추가

- `sources/display-test2/esp32_sensor_node/ESP32_UPDATE_CHANGELOG_KO.md`

### 의도적으로 수정하지 않은 파일

- `sources/display-test2/esp32_sensor_node/secrets.example.h`
- 저장소의 모든 ESP32 외부 파일
- 실제 Wi-Fi/Raspberry Pi 주소와 비밀번호가 들어 있는 로컬 `secrets.h`는 보안상 커밋하지 않습니다.

## 이전 동작과 업데이트 동작

| 영역 | 이전 저장소 동작 | 업데이트 동작 |
|---|---|---|
| scalar telemetry | TCP JSON을 약 1Hz로 발행 | TCP JSON snapshot을 100ms 주기(약 10Hz)로 발행 |
| MR60 phase | respiration/heart rate scalar만 전송 | MR60BHA2 라이브러리의 `getHeartBreathPhases()`로 실제 0x0A13 phase를 읽고 `mmwave` 객체로 전송 |
| phase freshness | phase 필드 없음 | 500ms freshness, phase sample timestamp/sequence/age를 함께 전송하고 stale phase 값은 `null` |
| CO₂ identity | boot/event 필드는 있었지만 snapshot이 stale event tuple을 계속 재전송할 수 있음 | 성공한 SCD4x 측정에서만 event ID를 증가시키고, stale/invalid이면 ID·timestamp를 0, valid를 false로 fail-closed |
| Thermal 입력 | READY 후 수신 frame을 바로 queue | CRC-16/CCITT-FALSE와 header min/max를 모두 검증한 frame만 queue |
| Thermal 전송 | 기존 chunked UDP `RPI_HOST:5005` | 동일한 UDP 계약 유지; 유효 frame만 송신하며 CRC/range 오류 counter를 health log에 추가 |
| PIR/health 계약 | PIR transition identity와 네트워크/센서 counter 제공 | 기존 필드와 counter를 그대로 보존 |
| phase 표현 | 해당 없음 | phase 숫자는 JSON에서 소수점 6자리로 표현 |

## 통신 관련 변경

- scalar telemetry는 기존 SafeNest TCP v1 framing과 `safenest.telemetry.v1` schema를 유지합니다.
- 기존 Thermal UDP v1 chunking(`SNTU`, port 5005, 1200-byte datagram)을 유지합니다.
- TCP scalar JSON에는 기존 `health` 및 PIR event 필드를 유지하면서 다음 `mmwave` metadata를 추가합니다.
  - `breath_phase`, `total_phase`, `heart_phase`
  - `breath_rate_raw`, `phase_age_ms`, `ts_monotonic_ms`, `seq`
  - `firmware_version`, `schema_version`
- JSON buffer를 768바이트에서 1536바이트로 확장하고 `snprintf` truncation을 전송 실패로 처리합니다.
- TCP와 Thermal UDP는 계속 별도 FreeRTOS task에서 실행됩니다. Thermal UDP를 TCP로 전환하거나 끄지 않았습니다.

## 센서 처리 변경

### MR60BHA2

- vendor Seeed library API `getHeartBreathPhases()`를 사용합니다.
- raw UART parser, 임의 phase 생성, B-model gate는 추가하지 않았습니다.
- phase sample 관측 시점과 TCP snapshot 전송 시점을 분리해 timestamp/age를 계산합니다.

### SCD4x CO₂

- `readMeasurement()`가 성공하고 CO₂ 값이 0이 아닐 때만 새 physical event로 인정합니다.
- 측정 성공 시 `co2_measurement_event_id`를 1씩 증가시키고 `millis()`를 measurement monotonic timestamp로 기록합니다.
- uint32 wrap 시 0을 건너뛰어 event identity가 0으로 재사용되지 않게 합니다.
- 마지막 측정이 `CO2_STALE_MS`를 넘으면 snapshot identity tuple을 `0/0/false`로 초기화합니다.

### Thermal camera

- SPI 수신 후 pixel payload에 CRC-16/CCITT-FALSE를 계산합니다.
- header의 CRC(`thermalCapture[7]`)와 계산값이 다르면 frame을 폐기합니다.
- header min/max(`[6]..[5]`)가 계산한 pixel min/max와 다르면 frame을 폐기합니다.
- 오류는 각각 `crc_errors`, `range_errors`로 누적하고 주기 health log에 표시합니다.
- 기존 8MHz SPI, 400kHz I²C 및 독립 Thermal UDP 경로는 유지합니다.

## 타이밍 및 예상 동작 차이

- telemetry publication interval: 1000ms → 100ms.
- SCD4x polling period, PIR polling period, MR60 scalar stale timeout, Thermal UDP packet format은 기존 값을 유지합니다.
- 10Hz TCP publication으로 Raspberry Pi 수신/로거 부하가 증가할 수 있습니다. Thermal UDP는 여전히 별도 경로입니다.
- phase가 아직 수신되지 않았거나 500ms보다 오래되면 JSON의 phase 값이 `null`이 됩니다.
- CRC/range 검증에 실패한 Thermal frame은 전송되지 않으므로 오류 상황에서는 `thermal_frames`와 `udp_sent` 증가 속도가 달라질 수 있습니다.
- 기존 PIR transition event 및 health counter는 회귀 없이 계속 전송됩니다.

## 검증 및 제한

- main 브랜치의 기존 `.ino`와 로컬 수정본을 정적 비교하고, 병합 후 required marker/필드 및 중복 frame sequence를 확인했습니다.
- 실제 ESP32 컴파일, flash, Pi live capture 및 hardware acceptance는 이 변경 작업에서 수행하지 않았습니다.
- 로컬 `secrets.h`는 커밋하지 않았으며, flash 전에 대상 환경의 secrets를 별도로 준비해야 합니다.
