# PHASE 1 — Repository Audit

## 기준

- Single Source of Truth: GitHub `main`
- 감사 기준 commit: `6baf38d8df936b694a1ff2e9b5e5fb2af2bfe50f`
- 선택한 원본은 `sources/`에 내용 변경 없이 동결
- 문서보다 실제 entry point, import, sender/receiver 구현과 config를 우선

## Repository 역할 분류

| 원본 영역 | 역할 | 판정 |
|---|---|---|
| `display-test2/esp32_sensor_node/` | 4개 센서 수집 및 SafeNest TCP v1 통합 sender | ✅ 통합 firmware 기준 |
| `display-test2/raspberry_pi_lcd/server.py` | TCP receiver와 LCD server가 결합된 기존 구현 | ⚠️ parser 근거만 사용, 중복 runtime 미채택 |
| `devices/mmwave/` | MR60 UART/JSONL firmware와 Pi adapter | ✅ 단독 센서 검증 근거 |
| `devices/thermal/` | Thermal TCP 구형 receiver와 UDP 시험 코드 | ⚠️ 현재 TCP v1과 직접 호환되지 않는 legacy/test 구분 |
| `ondevice_ai/inference/` | Thermal/mmWave/CO₂ TFLite adapter | ✅ 기존 모델 호출 계약 채택 |
| `ondevice_ai/models/` | 모델 3개, metadata, manifest | ✅ 재학습 없이 동결 채택 |
| `ondevice_ai/risk/` | V4/V5 risk 구현과 설정 | ✅ locked V4 계약 기준 |
| `display-test/` | 기존 LCD 수동 상태 API/UI | ⚠️ 호환 schema만 참고, 수동 demo 값 미채택 |
| `docs/dashboard/` | dashboard 문서 위치 | ❓ 실제 production dashboard asset 부재 |

## 센서별 감사 결과

| Sensor | ESP32 | Pi 입력 | AI/Rule | 초기 충돌 |
|---|---|---|---|---|
| mmWave MR60 | UART로 호흡·심박 수집 | TCP telemetry JSON | 300 sample AI 계약, 호흡 rule fallback | TCP v1에 phase window와 presence 없음 |
| Thermal-44 | 80×62 U16 frame | TCP v1 type 2 | 3-class TFLite | 구형 9-byte receiver 및 UDP 시험 코드와 충돌 |
| SCD40 | I²C periodic measurement | TCP telemetry JSON | CO₂ rule 및 slope/AI 계약 | legacy primary는 humidity 부재로 미충족; B-complete candidate는 `CO2 + CO2_slope` history가 필요 |
| PIR HC-SR501 | GPIO boolean | TCP telemetry JSON | presence 확인 후 no-motion rule | 단독 presence 센서로 사용하면 안 됨 |

## 통신 감사

✅ 채택한 실제 통신은 SafeNest TCP protocol v1이다.

| 항목 | 값 |
|---|---|
| Transport | TCP |
| 기본 Pi port | 9000 |
| Header | 16 bytes, `!4sBBHII` |
| Endian | Big Endian |
| Telemetry | UTF-8 JSON, 최대 4096 bytes |
| Thermal | metadata 16 bytes + 80×62×2 bytes = 9936 bytes |
| Sequence | packet type별 uint32 |
| CRC | 없음 |
| Partial recv | `recv_exact()`로 처리 |
| Timeout | packet 전체 deadline |
| Reconnect | Pi accept loop 유지, ESP32 network task 재연결 |

## 발견된 주요 통합 충돌

- Thermal 구형 9-byte little-endian receiver와 현재 16-byte big-endian sender 불일치
- TCP stream에서 `recv(n)` 한 번으로 전체 packet을 받는 legacy 가정
- receiver와 Flask/LCD가 결합된 중복 실행 구조
- ESP32 `writeAll()` 3초 timeout에 따른 Thermal 연결 종료 가능성
- mmWave presence와 300 sample phase window가 통합 telemetry에 없음
- 운영 legacy CO₂ primary는 humidity가 없어 입력 계약 미충족이다. B-complete candidate에는 humidity가 필요하지 않으며 실제 측정 이벤트 기반 `CO2_slope` history가 아직 필요하다.
- 기존 frontend는 실제 관제가 아니라 수동 scenario POST와 고정 예시 값 중심
- SQLite schema/repository가 main에 없어 신규 통합 계층 필요

## 감사 이후 원칙

- 동결 원본을 직접 수정하지 않고 `gateway/state/ai/risk/backend/database/web` 계층으로 분리
- 데이터가 없는 presence, humidity, 온도 calibration 값을 추측하지 않음
- AI 실패가 전체 runtime을 중단하지 않도록 센서별로 격리
- raw Thermal frame을 API JSON이나 SQLite snapshot에 저장하지 않음

**PHASE 1 Repository Audit 완료**
