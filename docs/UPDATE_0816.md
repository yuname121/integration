# SafeNest ESP↔Pi HIL Observability 후속 수정 보고서

## 1. 작업 메타데이터

| 항목 | 값 |
|---|---|
| 작업일 | 2026-08-16 (Asia/Seoul) |
| 실행 에이전트 | Codex (GPT-5) |
| 작업 단계 | `CURRENT_RUNTIME_HIL_OBSERVABILITY_CORRECTIVE` |
| 작업 성격 | PR #5 후속 corrective, 현재 runtime 관측성 보완 |
| 저장소 | `yuname121/integration` |
| 기준 branch | `main` |
| 기준 SHA | `83867c557b5d97a5bb71cd3e1784522109eb2dda` |
| 작업 branch | `feature/esp32-pi-hil-observability-followup` |
| Pull Request | [#7](https://github.com/yuname121/integration/pull/7) |
| merge 상태 | 미병합, PR review 대기 |

이 문서는 merged PR #5의 현재 runtime 계약을 다시 읽은 뒤, 확인된 ESP↔Raspberry Pi
HIL observability gap만 수정한 결과를 기록한다. Capture v1, AI 활성화, Risk 정책 변경,
mmWave phase/waveform 확장은 이 작업의 범위가 아니다.

## 2. 작업 단계와 판단 기준

### Stage A — 기준선 확인

- canonical ESP sender는 `sources/display-test2/esp32_sensor_node/esp32_sensor_node.ino`로 고정했다.
- scalar telemetry TCP `:9000`, Thermal UDP `:5005`, `safenest.telemetry.v1`, `boot_id`, CO₂/PIR provenance를 유지했다.
- 작업 트리의 미추적 HIL 파일 `hil/rp_x0_concurrent_soak.py`와 `hil/rp_x0_real_collection_watch.sh`는 사용자 작업으로 보고 수정·stage·commit하지 않았다.

### Stage B — gap 확인

- CO₂ `getDataReadyStatus()` 오류와 정상적인 `not-ready` 결과가 기존 코드에서 함께 반환되고 있었다.
- CO₂ `readMeasurement()` 오류가 기존 코드에서 누적 관측되지 않았다.
- Thermal fallback I2C status query 실패가 기존 코드에서 조용히 false로 처리되고 있었다.
- ESP device health가 `mmwave.values.health` 내부에만 있어 장치 단위 health임을 API 계약에서 명확히 드러내지 못했다.

### Stage C — 최소 corrective 구현

- 기존 v1 schema를 유지하고 health object에 optional counter 3개를 추가했다.
- 기존 consumer 호환을 위해 `mmwave.values.health` alias를 남겼지만, Pi 내부 canonical source는
  top-level `state.device_health` 하나로 두었다.
- 새 값은 실제로 관찰 가능한 오류에만 사용했으며, 측정할 수 없는 오류를 0으로 위조하지 않았다.

### Stage D — 검증 및 인계

- Protocol, state manager, backend/API, logger, Thermal 관련 자동 테스트를 실행했다.
- ESP firmware compile과 물리 HIL은 장비·toolchain이 없는 환경에서 완료로 표시하지 않았다.
- 두 개의 reviewable commit으로 분리해 PR #7에 push했다.

## 3. 구현된 변경

### 3.1 ESP health counter

새 health field는 다음 세 가지다.

| Field | 증가 조건 | 증가하지 않는 조건 |
|---|---|---|
| `co2_data_ready_query_failures` | SCD4x `getDataReadyStatus()`가 오류 반환 | 정상 `not-ready` 응답 |
| `co2_read_failures` | SCD4x `readMeasurement()`가 오류 반환 | data-ready가 아니거나, 성공했지만 `co2 == 0`인 경우 |
| `thermal_status_query_failures` | Thermal READY 핀 fallback에서 I2C status query 실패 | READY 핀이 HIGH인 경우, 정상 `not-ready` status |

CO₂ `co2_measurement_event_id`는 기존 의미를 유지한다. 성공적인 물리 측정과 유효한 CO₂ 값이
확인될 때만 증가하며, not-ready·read failure·재전송은 새 event가 아니다.

Thermal SPI read에 대해 현재 코드가 독립적인 acquisition result 또는 camera-native CRC를
제공하지 않으므로 별도 실패 counter를 만들지 않았다. 이 제한은
`THERMAL_ACQUISITION_ERROR_DETAIL_UNAVAILABLE`로 문서화했다.

### 3.2 Pi protocol/state/API

- `gateway/protocol.py`가 새 health counter를 uint32 optional field로 파싱한다.
- `state/manager.py`가 장치 health를 `state.device_health`에 canonical하게 보관한다.
- 호환 consumer를 위해 snapshot 생성 시 `state.sensors.mmwave.values.health`를 같은 canonical
  값에서 복사해 제공한다.
- `/api/status`와 `/api/sensors`의 top-level `device_health`에 동일한 장치 health를 노출한다.
- legacy sender가 health를 보내지 않는 경우 Pi는 필드를 임의의 0으로 채우지 않고 unavailable 상태를 유지한다.

### 3.3 Logger와 범위 경계

기존 logger 구현은 수정하지 않았다. CO₂ event dedup은 프로세스 수명 동안만 유지되며,
프로세스 재시작을 넘는 exact-once 보장은 이번 작업의 범위가 아니다
(`LEGACY_LOGGER_PROCESS_LIFETIME_DEDUP_ONLY`).

다음 항목은 의도적으로 변경하지 않았다.

- PIR persistent Capture 및 Capture v1 storage contract
- AI/model activation, input contract, threshold 또는 Risk 계산
- mmWave phase/waveform/300-sample B-model 입력
- PIR를 Capture 상태로 승격하는 동작
- telemetry schema version 변경

## 4. 변경 파일 및 commit

| Commit | 내용 |
|---|---|
| `72a3d96853b75792c3e8210bc60850b32d1cf396` | ESP health counter, Pi decoder, protocol test, 통신 계약 문서 |
| `17f6b5ae2ecf25be2137fa030dc78a82958edd9c` | canonical device health, API view, state/API test, 단계 문서 |

주요 변경 경로:

- `sources/display-test2/esp32_sensor_node/esp32_sensor_node.ino`
- `gateway/protocol.py`
- `state/manager.py`
- `backend/views.py`
- `tests/test_gateway_protocol.py`
- `tests/test_sensor_state_manager.py`
- `tests/test_backend.py`
- `sources/display-test2/docs/COMMUNICATION_PROTOCOL.md`
- `docs/PHASE4_SENSOR_STATE.md`
- `docs/PHASE7_FASTAPI.md`
- `README.md`

## 5. 검증 결과

| 검증 항목 | 결과 | 비고 |
|---|---|---|
| Python 전체 테스트 | PASS | 150 tests passed, workspace Python 3.12.13 |
| Protocol/state/backend/logger/Thermal 회귀 | PASS | 전체 suite에 포함해 실행 |
| Python `compileall` | PASS | `backend gateway state storage tests` |
| `git diff --check` | PASS | origin/main 대비 clean |
| ESP firmware compile | `NOT_RUN_TOOLCHAIN_UNAVAILABLE` | Arduino CLI/PlatformIO 미설치 |
| 실제 Raspberry Pi HIL | `HIL_REQUIRED` | 물리 ESP32·센서·Pi 장비 미연결 |
| GitHub PR checks | 미보고 | PR branch에 자동 check 없음 |

기본 macOS `python3`는 3.9.6으로 저장소의 기존 `type1 | type2` 문법을 로드하지 못했다.
저장소 코드를 수정해 우회하지 않고 workspace Python 3.12.13으로 검증했다.

## 6. HIL 인계 시 확인할 것

1. canonical firmware를 ESP-WROOM-32/ESP32 Dev Module에 compile·flash한다. 현재 GPIO/UART/SPI 설정으로 XIAO ESP32-C6에 업로드하지 않는다.
2. SCD4x 정상 not-ready 구간에서 `co2_data_ready_query_failures`와 `co2_read_failures`가 증가하지 않는지 확인한다.
3. SCD4x status query 오류와 measurement read 오류를 각각 유도해 두 counter가 서로 분리되는지 확인한다.
4. 정상 물리 측정에서만 `co2_measurement_event_id`가 증가하고, TCP 재전송에서는 증가하지 않는지 확인한다.
5. Thermal READY fallback의 I2C status query 오류에서만 `thermal_status_query_failures`가 증가하는지 확인한다.
6. `/api/status.device_health`와 `state.sensors.mmwave.values.health`가 같은 snapshot health를 보여주는지 확인한다.
7. ESP reboot 후 `boot_id`, CO₂ event identity, counter reset 및 Pi 재연결 상태를 JSON·Serial log와 함께 보존한다.
8. HIL collector JSON, Pi backend log, ESP Serial Monitor log를 PR review evidence로 첨부한다.

## 7. 최종 판정

`PASS_WITH_LIMITATIONS`

현재 runtime corrective 구현과 자동 검증은 완료됐다. 다만 ESP firmware compile, 실제 센서 오류
주입, ESP reboot, Thermal transport, Raspberry Pi API evidence는 하드웨어 HIL에서 확인해야 하며,
그 전에는 운영 hardware pass 또는 merge 완료로 판정하지 않는다.
