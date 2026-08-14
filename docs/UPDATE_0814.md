# SafeNest 통합 패키지 0814 업데이트

## 개요

이 문서는 `safenest_integration_package_0814`의 실제 저장소 내용을 기존 GitHub 저장소 루트에 반영한 결과를 기록한다. 패키지의 바깥 디렉터리는 복사하지 않았으며 `README.md`, `ai/`, `backend/`, `gateway/`, `risk/`, `sources/`, `tests/`, `web/` 구조를 유지했다.

## 주요 변경

- Thermal-44 전송을 기존 TCP 스트림에서 독립적인 청크 UDP 채널로 분리했다.
- DANGER 전용 긴급 HMI, 경고 래치, 119 모의 신고, 담당자 SMS, GPIO/mock buz저와 이벤트 기록을 추가했다.
- mmWave, CO2, Thermal 센서 데이터를 비동기 rolling dataset으로 저장하고 용량·여유 공간 기준 정리 정책을 추가했다.
- CO2 유효값 채택 주기를 60초로 제한하면서 수신 상태는 계속 갱신하도록 상태 관리 로직을 보완했다.
- 위험도 계산을 기본 15초 주기로 게시하고, 대시보드·LCD 호환 뷰가 같은 게시 상태를 사용하도록 연결했다.

## 통신 변경

- mmWave, CO2, PIR telemetry는 SafeNest TCP v1 포트 9000을 유지한다.
- Thermal-44는 UDP 포트 5005에서 1,200바이트 이하 청크 9개로 전송한다.
- Raspberry Pi 수신기는 CRC32, shape, min/max, timeout, 중복·순서 변경과 pending-frame 상한을 검증한다.
- `/health`에 Thermal UDP 패킷, 완료·폐기 프레임, 실효 FPS와 재조립 시간 진단을 노출한다.

## Raspberry Pi 변경

- FastAPI/runtime 시작 시 Thermal UDP 수신기, 센서 데이터 로거와 긴급 서비스가 함께 시작·종료된다.
- `.env.example`에 UDP, 데이터 보존, SMS, GPIO buzzer 설정 예시를 추가했다.
- SQLite schema와 repository에 긴급 동작 및 상태 기록 지원을 추가했다.
- `deployment/run_pi.sh`는 GitHub 저장소 루트에서 직접 실행되도록 경로를 정규화했다.

## ESP32 변경

- `esp32_sensor_node.ino`에서 scalar telemetry용 TCP 작업과 Thermal UDP 작업을 분리했다.
- Thermal 프레임 헤더, 청크 메타데이터, CRC32 생성과 UDP 송신 진단 카운터를 추가했다.
- TCP 재연결 지연이 Thermal UDP 송신을 막지 않도록 별도 FreeRTOS 작업을 사용한다.

## 센서 및 AI/runtime 변경

- CO2 값·통신 상태의 갱신 시점을 분리하고 60초 채택 주기를 적용했다.
- mmWave/CO2 JSONL 및 Thermal NPZ 저장을 bounded queue 기반 writer에서 처리한다.
- 기존 AI 모델과 후보 승격 정책은 변경하지 않았으며 위험도 fallback 동작도 유지했다.
- 위험도 스케줄링, 센서 offline 상태와 DANGER latch가 backend·WebSocket·대시보드에서 일관되게 표시되도록 보완했다.

## 테스트 결과

커밋 전 다음 검증을 수행했다.

- Python 전체 단위 테스트: 139개 통과 (`python -m unittest discover -s tests -p "test_*.py" -v`)
- 대시보드 JavaScript 구문 검사: 통과 (`node --check web/dashboard/app.js`)
- Python compileall 및 루트 import 검사: 통과
- 필수 디렉터리·firmware·manifest 존재 및 중첩 패키지 디렉터리 부재 확인: 통과
- 비밀정보, 캐시, 생성 DB와 GitHub 대용량 파일 검사: 통과

## 알려진 제한과 남은 하드웨어 검증

- 실제 Raspberry Pi GPIO buzzer 배선과 BCM 핀은 현장에서 확인해야 한다.
- 실제 Naver Cloud SENS 계정·발신 번호를 사용한 SMS 전송은 별도 자격증명으로 검증해야 한다.
- ESP32에서 UDP 손실·순서 변경·Wi-Fi 재연결 상황의 장시간 HIL 검증이 남아 있다.
- Raspberry Pi 저장장치에서 Thermal NPZ의 실제 증가율과 quota 정리 동작을 장시간 측정해야 한다.
- 119 기능은 모의 시연이며 실제 119 또는 공공 긴급망과 통신하지 않는다.
