# PHASE 2 Sensor Standalone Audit

기준: GitHub `main` `6baf38d8df936b694a1ff2e9b5e5fb2af2bfe50f`

표기:

- **CONFIRMED**: 코드 또는 자동 테스트에서 확인
- **INCONSISTENCY**: 구현 사이 불일치
- **UNKNOWN**: 실물 장비가 없어 확인할 수 없음

## 채택 원칙

1. ESP32 통합 송신의 기준은 `display-test2/esp32_sensor_node/esp32_sensor_node.ino`이다.
2. SafeNest TCP v1의 기준은 `display-test2/docs/COMMUNICATION_PROTOCOL.md`와 Pi `server.py`이다.
3. 검증된 mmWave 단독 firmware와 adapter는 회귀 비교용으로 보존한다.
4. 고정값을 반환하는 CO₂/PIR/Thermal `devices/*/src` adapter는 실제 hardware provider로 채택하지 않는다.
5. Thermal UDP 코드는 XIAO ESP32-C6 단독 검증 참고용이며 운영 transport로 채택하지 않는다.

## mmWave — MR60BHA2

- **CONFIRMED**: WROOM-32 firmware는 UART2 115200bps를 사용하고 JSONL telemetry를 출력한다.
- **CONFIRMED**: stream adapter의 NaN/Inf, 역순 timestamp, 큰 gap, presence=0, stale 테스트 9개 통과.
- **INCONSISTENCY**: `test_mr60_esp_adapter.py`는 존재하지 않는 `ondevice_ai.src`를 import한다.
- **INCONSISTENCY**: `mr60_20260728_manifest.json`이 없어 manifest 테스트 2개가 실패한다.
- **INCONSISTENCY**: 통합 TCP telemetry는 respiration phase 300-sample AI window를 전송하지 않는다.
- **UNKNOWN**: 현재 장비에서 UART frame 수신과 호흡수 정확도.

채택: 단독 firmware와 `mr60_esp_adapter.py`는 회귀 기준으로 보존한다. PHASE 3에서 TCP telemetry provider를 별도로 작성한다.

## Thermal-44

- **CONFIRMED**: 통합 sender와 Pi receiver 모두 TCP 9000, 16-byte `SNST` header, big-endian을 사용한다.
- **CONFIRMED**: Thermal payload 계산은 16-byte metadata + 9,920-byte pixels = 9,936 bytes이다.
- **CONFIRMED**: Pi receiver에 partial receive를 처리하는 `recv_exact()`가 있다.
- **INCONSISTENCY**: LCD receiver는 thermal payload의 width/height/정확한 길이를 검증하지 않고 frame count만 증가시킨다.
- **INCONSISTENCY**: calibration, raw-to-temperature, AI inference 경로가 TCP receiver에 연결되지 않았다.
- **UNKNOWN**: 실물 Thermal-44에서 지속 FPS, 온도 33°C 재현, ESP32 `writeAll()` timeout 재현.

채택: TCP sender를 운영 기준으로, UDP sender/receiver는 XIAO C6 참고 자료로 보존한다.

## CO₂ — SCD40/SCD4x

- **CONFIRMED**: 통합 ESP32 firmware가 I2C `0x62`에서 periodic measurement를 읽는다.
- **CONFIRMED**: 약 5초 센서 cadence와 15초 firmware freshness가 정의되어 있다.
- **CONFIRMED**: telemetry는 `co2_ppm`과 `valid.co2`를 전송한다.
- **INCONSISTENCY**: `devices/co2/src/co2_adapter.py`는 실제 I2C가 아니라 650ppm 고정값을 반환한다.
- **INCONSISTENCY**: AI가 요구하는 30-sample CO₂ history/provider가 TCP 수신기와 연결되지 않았다.
- **UNKNOWN**: 실물 SCD40의 측정값과 보정 상태.

채택: ESP32 telemetry만 실제 센서 경로로 인정한다. 고정값 Python adapter는 제외한다.

## PIR — HC-SR501

- **CONFIRMED**: 통합 firmware가 GPIO13을 20ms 주기로 읽고 boolean `pir_motion`을 전송한다.
- **CONFIRMED**: Pi receiver가 boolean 형식을 검증한다.
- **INCONSISTENCY**: `devices/pir/src/pir_adapter.py`는 `read_gpio()`에서 항상 `True`를 반환한다.
- **INCONSISTENCY**: UI freshness는 PIR 독립 timestamp가 아니라 전체 telemetry 도착 시각을 사용한다.
- **UNKNOWN**: 실물 센서의 active-high, hold time, false trigger 특성.

채택: ESP32 telemetry만 실제 센서 경로로 인정한다. 고정값 Python adapter는 제외한다.

## 보드 결정

- **CONFIRMED**: 검증된 mmWave와 통합 TCP firmware는 ESP-WROOM-32/`esp32dev` 핀맵이다.
- **CONFIRMED**: Thermal UDP 단독 코드는 XIAO ESP32-C6 핀맵이다.
- **INCONSISTENCY**: 최종 목표 보드는 XIAO ESP32-C6이지만 현재 통합 firmware는 그대로 호환되지 않는다.

PHASE 3 전에 다음 중 하나를 확정해야 한다.

1. 우선 WROOM-32으로 E2E 통신을 검증한 뒤 C6으로 포팅한다.
2. 먼저 C6 핀/UART/SPI/FreeRTOS 호환성을 포팅하고 그 firmware만 사용한다.

기본 권고는 1번이다. 이미 확인된 TCP protocol과 센서 코드를 보존하면서 보드 포팅 문제를 분리할 수 있다.

## 자동 검증 결과

### PASS

- `./tests`: 8 tests OK
- 동결한 `display-test2` TCP receiver: 4 tests OK
- 선택한 Python source 6개: 문법 compile OK
- mmWave stream/input 안전성 테스트: 9 tests OK

### FAIL — 기존 main 결함

- mmWave 전체 테스트 중 1개 module import error:
  `ondevice_ai.src.integrated_node.run_mr60_usb_node`가 존재하지 않음
- MR60 manifest 테스트 2개 error:
  `ondevice_ai/datasets/mmwave/mr60_20260728_manifest.json`이 존재하지 않음

### UNKNOWN — 실행 환경 또는 실물 필요

- PlatformIO/Arduino CLI가 현재 Windows 실행 환경에 없어 ESP32 compile 미실행
- ESP32, MR60, Thermal-44, SCD40, PIR 실물 연결 시험 미실행
- Raspberry Pi GPIO/I2C/SPI/UART 시험 미실행
