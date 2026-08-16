# SafeNest 전체 시스템 통합 단계 요약

이 문서는 standalone 저장소 루트에 모인 PHASE 1~10 및 HIL 준비 산출물의 통합 목차다. 기존 GitHub `main` 원본은 `sources/`에 동결했다. 유일한 명시적 유지보수 예외인 `sources/display-test2/esp32_sensor_node/`는 canonical flash source이며, `sources/ondevice_ai/`는 계속 동결한다.

## 단계별 결과

| 단계 | 수행 내용 | 핵심 결과 | 상세 문서 |
|---|---|---|---|
| PHASE 1 | GitHub main 전체 구조와 실제 사용 코드 감사 | ESP32, Pi receiver, AI, Risk, Backend, DB, Frontend의 사용·중복·결측 구분 | `docs/PHASE1_REPOSITORY_AUDIT.md` |
| PHASE 2 | 센서별 단독 코드 선정 | mmWave, Thermal-44, SCD40, PIR 및 통합 ESP32 sender 원본 동결 | `docs/PHASE2_SENSOR_AUDIT.md` |
| PHASE 3 | ESP32 ↔ Pi 통신 통합 | SafeNest TCP v1, 16-byte big-endian header, partial recv, strict parser, reconnect | `docs/PHASE3_COMMUNICATION.md` |
| PHASE 4 | 중앙 Sensor State Manager | 센서별 최신값, 연결, 유효성, TTL, LIVE/STALE/DISCONNECTED/NO_DATA 분리 | `docs/PHASE4_SENSOR_STATE.md` |
| PHASE 5 | On-device AI 연결 | 기존 TFLite adapter 지연 로드, 입력 검증, NaN/Inf 차단, 장애 격리, rule 지속 | `docs/PHASE5_AI.md` |
| PHASE 6 | Risk Engine 연결 | 기존 locked V4 가중치, NORMAL/WARNING/DANGER, emergency override, presence 교차검증 | `docs/PHASE6_RISK.md` |
| PHASE 7 | FastAPI Backend | status/sensors/events/state/health API와 WebSocket, 기존 LCD 호환 view | `docs/PHASE7_FASTAPI.md` |
| PHASE 8 | SQLite 로그 | versioned schema, WAL, snapshot/event transaction, 재시작 복원, history API | `docs/PHASE8_SQLITE.md` |
| PHASE 9 | Web Dashboard | 반응형 관제 UI, WebSocket/polling 전환, 센서·위험·이벤트·history·Thermal heatmap | `docs/PHASE9_DASHBOARD.md` |
| PHASE 10 | 전체 E2E 테스트 | 실제 loopback TCP부터 State·AI·Risk·SQLite·API까지 10개 시나리오 검증 | `docs/PHASE10_E2E.md` |
| HIL 준비 | Raspberry Pi·ESP32 실제 장비 검증 패키지 | Pi preflight, 모델 해시, 10개 시나리오 증거 수집·판정, one-command 실행 | `docs/HIL_ACCEPTANCE.md` |

## 통합 폴더 구조

```text
integration/
├── sources/       GitHub main 무수정 원본
├── gateway/       SafeNest TCP v1 수신·파싱·실행 진입점
├── state/         Sensor State Manager와 freshness
├── ai/            기존 모델 loader와 장애 격리 pipeline
├── risk/          V4 Risk Engine 통합 adapter
├── backend/       전체 runtime, FastAPI, API view
├── database/      SQLite schema와 repository
├── web/dashboard/ 실시간 관제 화면
├── e2e/           소프트웨어 loopback E2E 하네스
├── hil/           실제 장비 증거 수집·판정·preflight
├── deployment/    Raspberry Pi 실행 스크립트
├── tests/         전체 자동 회귀 테스트
└── docs/          PHASE별 상세 판단 근거와 실행 방법
```

## 최종 실행 흐름

```text
mmWave / SCD40 / PIR → ESP-WROOM-32 → SafeNest TCP v1 :9000 ─┐
Thermal-44           → ESP-WROOM-32 → Thermal UDP v1 :5005 ──┤
→ Raspberry Pi receiver
→ Sensor State Manager
→ On-device AI + Rule
→ Risk Engine
→ FastAPI + SQLite
→ Web Dashboard
```

Raspberry Pi에서는 다음 한 명령으로 설치·점검·실행한다.

```bash
cd ~/integration
bash deployment/run_pi.sh --install
```

## 현재 검증 상태

- 전체 자동 회귀 테스트: 110개 통과
- PHASE 10 software E2E: 10개 통과
- HIL 판정 기준 테스트: 14개 통과
- 최신 manifest 등록 모델 4개 SHA-256 일치 예정(검증 명령으로 확인)
- 실제 Raspberry Pi·ESP32·센서 HIL: 장비에서 수행 필요

## 남은 통합 결정

1. 비정상 호흡 5 rpm과 CO₂ 700 ppm 조건의 locked V4 결과는 29.75점 `NORMAL`이다. 요구사항의 WARNING/DANGER와 충돌하므로 공식 정책 결정이 필요하다.
2. 현재 SafeNest TCP v1에는 검증된 mmWave presence가 없다. MR60 presence source가 확정되면 ESP32 sender와 Pi decoder를 같은 protocol revision으로 함께 확장해야 한다.
3. Thermal 온도 calibration 계약이 없으므로 현재 UI와 DB는 raw 범위만 표시하고 °C를 생성하지 않는다.
4. mmWave 연속 respiration phase/waveform 및 300-sample B-model 입력은 `PENDING_MMWAVE_DEVICE_CONTRACT_VALIDATION`이며 현재 scalar rate telemetry가 이를 충족한다고 간주하지 않는다.
