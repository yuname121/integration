# SafeNest System Integration

이 저장소는 `safenest-embedded-competition`의 단계별 시스템 통합 산출물을 한곳에 모은 독립 실행 번들이다. 기존 코드는 원칙적으로 `sources/`에 동결 복사하고, 새 통합 코드는 별도 모듈로 작성한다. 단, `sources/display-test2/esp32_sensor_node/`는 배포 문서가 지목하는 유일한 통합 소유·유지보수 ESP32 플래시 원본이다. `sources/ondevice_ai/`를 포함한 나머지 source snapshot은 운영 firmware가 아니다.

전체 단계의 작업 내용과 결과는 [`INTEGRATION_PHASE_SUMMARY.md`](INTEGRATION_PHASE_SUMMARY.md)에서 한 번에 확인할 수 있다.
압축 전달, 최초 설치, 평상시 실행·종료 순서는 [`PACKAGE_AND_OPERATION_GUIDE.md`](PACKAGE_AND_OPERATION_GUIDE.md)를 따른다.
최신 AI 변경 감사와 반영 결정은 [`docs/ON_DEVICE_UPDATE_AUDIT.md`](docs/ON_DEVICE_UPDATE_AUDIT.md)에 정리했다.
현재 ESP↔Pi HIL observability corrective와 인계 사항은 [`docs/UPDATE_0816.md`](docs/UPDATE_0816.md)에 정리했다.
긴급 대응 HMI, DANGER 래치, 119 모의 신고, 서버 측 SMS, GPIO/mock buzzer와 오프라인 시연 순서는 [`docs/EMERGENCY_HMI_AND_OPERATIONS_KO.md`](docs/EMERGENCY_HMI_AND_OPERATIONS_KO.md)를 따른다.

현재 완료 단계:

- PHASE 1: GitHub main repository 구조, 사용 코드, 통신·모델·backend·frontend 감사
- PHASE 2: 센서별 동작 코드 선정 및 동결
- PHASE 3: SafeNest TCP v1 수신·검증·재연결
- PHASE 4: 중앙 Sensor State Manager와 freshness 관리
- PHASE 5: 기존 TFLite 모델의 지연 로드, 입력 검증, 장애 격리, PIR rule 연결
- PHASE 6: 공식 V4 가중치 기반 Risk Engine, rule fallback, emergency override 연결
- PHASE 7: FastAPI status/sensors/events API, health, WebSocket, LCD 호환 view 연결
- PHASE 8: versioned SQLite snapshot/event 로그, WAL, 재시작 복원, history API 연결
- PHASE 9: 반응형 실시간 관제 대시보드, WebSocket/polling 전환, Thermal 정규화 미리보기 연결
- 긴급 대응: DANGER 전용 HMI 오버레이, 119 모의 신고, 담당자 SMS 쿨다운, GPIO/mock buzzer, SQLite 이벤트 기록
- PHASE 10: loopback TCP 기반 전체 E2E 시나리오, 재연결·재부팅·AI 장애 검증
- HIL 실행 패키지: Pi preflight, 물리 시나리오 증거 수집·판정, one-command 구동

## 주요 경로

- `sources/ondevice_ai/`: 최신 GitHub `origin/main` `fa8cf13` / component source `77b1695`의 AI component 1,069개 파일 무수정 snapshot
- `sources/display-test2/esp32_sensor_node/`: ESP-WROOM-32용 유일한 canonical flash source. scalar TCP `:9000`과 Thermal UDP `:5005` sender
- 그 밖의 `sources/`: 기존 통합 기준 `6baf38d8df936b694a1ff2e9b5e5fb2af2bfe50f`의 통신·센서·UI 원본
- `gateway/`: TCP 수신기와 단계별 실행 진입점
- `state/manager.py`: 센서별 최신값, 연결, 유효성, freshness 상태
- `ai/`: 모델 지연 로더, 공통 결과 계약, AI/rule 파이프라인
- `risk/`: 센서별 risk component 변환과 `NORMAL/WARNING/DANGER` 융합
- `backend/`: 최신 publication store, API view, FastAPI app, 전체 runtime
- `database/`: SQLite schema, transactional repository, 장애 격리 persistent store
- `web/dashboard/`: Raspberry Pi 백엔드와 같은 origin에서 제공하는 실시간 관제 UI
- `e2e/`: 실제 TCP framing부터 SQLite/API view까지 연결하는 결정적 테스트 하네스
- `hil/`: 실제 장비 API 표본 수집, 10개 acceptance 판정, JSON 증거 보고서
- `deployment/run_pi.sh`: Pi 의존성 설치·preflight·통합 backend 실행 진입점
- `docs/`: 단계별 판단 근거와 실행 방법
- `tests/`: 프로토콜, 상태, AI 경계 자동 테스트

## 검증

저장소 루트에서 실행한다.

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Raspberry Pi의 실제 TFLite 실행과 ESP32 하드웨어 검증은 정적·단위 테스트와 구분해 `docs/PHASE5_AI.md`부터 `docs/PHASE10_E2E.md`까지 기록했다.

## 현재 센서 계약 경계

- scalar telemetry는 backward-compatible `safenest.telemetry.v1`이며 `boot_id`, CO₂ 측정 이벤트, PIR 전환 이벤트와 health counter를 선택 필드로 보존한다. 새 canonical firmware는 이 확장 필드를 항상 송신하고 legacy sender의 필드 부재도 Pi가 의도적으로 허용한다.
- ESP health의 canonical Pi 경로는 `state.device_health`와 `/api/status`·`/api/sensors`의 top-level `device_health`다. 기존 소비자를 위해 `state.sensors.mmwave.values.health`에는 같은 snapshot의 compatibility alias를 제공하며, 두 mutable state를 따로 유지하지 않는다. CO₂ 취득 실패와 Thermal status query 실패도 이 health counter로 구분한다.
- CO₂ 실제 측정은 SCD4x periodic mode의 약 5초 cadence이고 telemetry publication은 약 1초 cadence다. 동일 `(device_id, boot_id, co2_measurement_event_id)` 재전송은 새 물리 측정이 아니다.
- 기존 logger의 이벤트 dedup은 프로세스 수명 동안만 유지한다(`LEGACY_LOGGER_PROCESS_LIFETIME_DEDUP_ONLY`). 프로세스 재시작을 넘는 exact-once 보장은 이번 corrective 범위가 아니다.
- 현재 CO₂ offline candidate의 AI 입력 요구는 `CO2 + CO2_slope`다. 온도·습도는 요구 입력이 아니다.
- mmWave runtime은 vendor-derived `resp_rate_bpm`/`heart_rate_bpm` scalar만 유지한다. 연속 phase/waveform 및 300-sample B-model 입력 작업 상태는 `PENDING_MMWAVE_DEVICE_CONTRACT_VALIDATION`이다.
