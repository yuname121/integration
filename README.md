# SafeNest Standalone Integration

이 저장소는 SafeNest 임베디드 시스템의 standalone 통합 릴리스다. GitHub 저장소 루트에 있는 파일만으로 테스트와 Raspberry Pi 실행을 수행할 수 있다. 원본 온디바이스 코드는 `sources/`에 동결 복사하고, 통합 코드는 별도 모듈로 유지한다.

전체 단계의 작업 내용과 결과는 [`INTEGRATION_PHASE_SUMMARY.md`](INTEGRATION_PHASE_SUMMARY.md)에서 한 번에 확인할 수 있다.
압축 전달, 최초 설치, 평상시 실행·종료 순서는 [`PACKAGE_AND_OPERATION_GUIDE.md`](PACKAGE_AND_OPERATION_GUIDE.md)를 따른다.
최신 AI 변경 감사와 반영 결정은 [`docs/ON_DEVICE_UPDATE_AUDIT.md`](docs/ON_DEVICE_UPDATE_AUDIT.md)에 정리했다.
standalone 저장소 공개 전 감사는 [`docs/STANDALONE_REPOSITORY_AUDIT.md`](docs/STANDALONE_REPOSITORY_AUDIT.md)에 정리했다.

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
- PHASE 10: loopback TCP 기반 전체 E2E 시나리오, 재연결·재부팅·AI 장애 검증
- HIL 실행 패키지: Pi preflight, 물리 시나리오 증거 수집·판정, one-command 구동

## 주요 경로

- `sources/ondevice_ai/`: 최신 GitHub `origin/main` `fa8cf13` / component source `77b1695`의 AI component 1,069개 파일 무수정 snapshot
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

## 최초 실행

```bash
git clone https://github.com/yuname121/integration.git
cd integration
bash deployment/run_pi.sh --install
```

일반 실행은 `bash deployment/run_pi.sh`이며, 실행 전 ESP32 설정과 실제 하드웨어 연결이 필요하다. 비밀번호·Wi-Fi 값은 저장소에 포함하지 않는다.

## 검증

저장소 루트에서 실행한다.

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Raspberry Pi의 실제 TFLite 실행과 ESP32 하드웨어 검증은 정적·단위 테스트와 구분해 `docs/PHASE5_AI.md`부터 `docs/PHASE10_E2E.md`까지 기록했다.
