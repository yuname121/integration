# Source Manifest

## 2026-08-14 Thermal UDP transport update

- 운영 sender `display-test2/esp32_sensor_node/esp32_sensor_node.ino`는 mmWave, CO2, PIR telemetry를 기존 SafeNest TCP v1 port 9000으로 유지한다.
- Thermal-44만 SafeNest Thermal UDP v1 port 5005의 1,200-byte 이하 chunk로 송신한다.
- `gateway/thermal_udp.py`가 CRC32, timeout, 순서 무관 reassembly와 bounded pending-frame 정책을 적용한 후 기존 `gateway/protocol.py::decode_thermal()`로 전달한다.
- `devices/thermal/thermal_sensor_test/`의 과거 단일 oversized UDP datagram은 참고 자료일 뿐 운영 packet contract로 사용하지 않는다.

이 파일은 standalone 저장소 루트 기준의 원본·통합 경계를 기록한다. `sources/ondevice_ai/`는 동결된 upstream snapshot이며, 실제 운영 모델 승격은 별도 검증과 승인 없이는 수행하지 않는다. `sources/` 동결 원칙의 명시적 예외는 `sources/display-test2/esp32_sensor_node/` 하나이며, 배포 문서와 테스트가 함께 관리하는 canonical flash source다.

기계 판독 가능한 출처와 배포 결정은 루트의 `LATEST_SOURCE_PROVENANCE.json`에도 기록했다.

`sources/ondevice_ai/`는 최신 GitHub `origin/main` 커밋 `fa8cf13`(component source `77b1695ac66fd595bd037e4574d1626b8917654c`)에서 전체 1,069개 tracked 파일을 무수정 추출했다. 그 밖의 `sources/`는 통신 계약을 확정했던 `6baf38d8df936b694a1ff2e9b5e5fb2af2bfe50f`에서 무수정 추출했다.

| 원본 경로 | 통합 용도 | 판단 |
|---|---|---|
| `sources/display-test2/esp32_sensor_node/esp32_sensor_node.ino` | 통합 센서 sender | `ACTIVE_FLASH_SOURCE`, 유지보수 예외 |
| `display-test2/docs/COMMUNICATION_PROTOCOL.md` | SafeNest TCP v1 명세 | 채택 |
| `display-test2/raspberry_pi_lcd/server.py` | 기존 TCP receiver | parser 분리의 기준 |
| `devices/mmwave/firmware/` | MR60 UART/JSONL firmware | 검증 참고 원본 |
| `devices/mmwave/src/` | MR60 Python adapter | 독립 검증용 |
| `devices/thermal/thermal_integration/tcp_thermal_receiver_rpi.py` | Thermal TCP parser 참고 | 구형 receiver 직접 사용 안 함 |
| `devices/thermal/thermal_sensor_test/` | XIAO C6/UDP 시험 참고 | 운영 UDP 사용 안 함 |
| `ondevice_ai/inference/thermal_interpreter.py` | Thermal 3-class 추론 | PHASE 5 지연 로드 |
| `ondevice_ai/inference/mmwave_interpreter.py` | 300-sample 호흡 이상 추론 | 입력 확보 시에만 실행 |
| `ondevice_ai/inference/co2_interpreter.py` | CO₂ 점유 추론 | 현재 candidate 입력은 `CO2 + CO2_slope`; device-domain history 확보 후 별도 승격 |
| `ondevice_ai/models/model_manifest.json` | 모델 shape/dtype/hash 계약 | 채택 |
| `ondevice_ai/models/**` | primary 3개, CO₂/mmWave offline candidate와 experiment evidence | 재학습 없이 포함, manifest selection만 runtime 사용 |
| `ondevice_ai/models/**/*metadata*.json` | 정규화 통계 | 채택 |
| `ondevice_ai/thermal_prep.py` | Thermal 학습 전처리 근거 | 프레임별 min-max 확인 |
| `ondevice_ai/requirements-pi.txt` | Pi LiteRT 의존성 | 채택 |
| `ondevice_ai/risk/risk_engine.py` | 공식 V4/V5 위험도 융합 구현 | PHASE 6 계약 기준 |
| `ondevice_ai/risk/fallback.py` | 결측·stale·health 분리 | 통합 adapter 설계 기준 |
| `ondevice_ai/risk/risk_rules.py` | 호흡·CO₂·PIR·Thermal 규칙 | 필요한 검증 규칙 채택 |
| `ondevice_ai/risk/risk_config.json` | 가중치·경계·rule threshold | `v4_implementation_locked_hil_pending` |
| `ondevice_ai/config/risk_rules.json` | 보수적 호흡/융합 정책 | 참고·교차검증 |
| `display-test/server.py` | 기존 `/api/state`, `/health`, LCD state 계약 | PHASE 7 호환 기준 |
| `display-test2/raspberry_pi_lcd/server.py` | TCP receiver가 결합된 LCD server | 중복 receiver 미채택 |
| `docs/dashboard/README.md` | 기존 dashboard 자산 부재 확인 | PHASE 9 판단 근거 |
| `display-test/static/*` | 수동 LCD 시나리오 UI 원본 | PHASE 9 비교용 동결, 측정값 미채택 |
| `display-test2/raspberry_pi_lcd/static/*` | 중복된 수동 LCD 시나리오 UI 원본 | PHASE 9 비교용 동결, 측정값 미채택 |

`main` 전체에서 SQLite import, SQL schema, database repository는 발견되지 않았다. 따라서 PHASE 8의 `database/`는 기존 파일 복사가 아니라 위에서 동결한 sensor/AI/risk/API 계약을 소비하는 새 통합 코드다.

이름만 존재하고 실제 하드웨어 값을 읽지 않는 `co2_adapter.py`, `pir_adapter.py`, `thermal44_driver.py`, `mmwave_adapter.py` 계열은 동결 번들에서 제외했다.

PHASE 9에서는 기존 정적 UI가 `/api/state`에 테스트 상태를 POST하고 고정 예시 수치를 표시하는 시나리오 제어 화면임을 확인했다. 새 `web/dashboard/`는 이 값을 재사용하지 않고 통합 백엔드의 실제 read-only API만 소비한다.

## 운영 분류

| 구성요소 | 경로 | 분류 |
|---|---|---|
| ESP32 flash source | `sources/display-test2/esp32_sensor_node/` | `ACTIVE_FLASH_SOURCE` |
| AI 통합 ESP32 사본 | `sources/ondevice_ai/integrated_node/esp32_sensor_node.ino` | `FROZEN_UPSTREAM_SNAPSHOT` |
| Pi scalar receiver | `gateway/receiver.py`, `gateway/protocol.py` | `ACTIVE_RUNTIME` |
| Pi Thermal receiver | `gateway/thermal_udp.py` | `ACTIVE_RUNTIME` |
| Pi state | `state/manager.py` | `ACTIVE_RUNTIME` |
| Pi entry point | `deployment/run_pi.sh` → `backend/run_backend.py` | `ACTIVE_RUNTIME` |
