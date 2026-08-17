# PHASE 7 — FastAPI 연결

## Repository 감사 결과

GitHub `main`에는 FastAPI, APIRouter, WebSocket backend가 없다. 운영에 가장 가까운 API는 `display-test/server.py`와 `display-test2/raspberry_pi_lcd/server.py`의 표준 라이브러리 HTTP 서버다.

기존 endpoint:

- `GET /api/state`
- `POST /api/state` — 수동 LCD 상태 변경
- `GET /health`

두 번째 서버는 SafeNest TCP receiver까지 내부에 중복 구현한다. PHASE 3 strict decoder와 PHASE 4 SensorStateManager를 우회하므로 새 backend에서는 이 receiver를 재사용하지 않았다. 기존 파일은 `sources/display-test/`와 `sources/display-test2/`에 동결되어 있다.

## 통합 구조

```text
ESP32 TCP :9000
  → strict decoder
  → SensorStateManager
  → On-device AI / rules
  → Risk Engine
  → RuntimeStore (latest + bounded events)
  → FastAPI :8000
```

평가 loop는 하나의 Sensor State snapshot을 AI와 Risk에 함께 전달한다. 따라서 API 한 publication 안의 `state`, `ai`, `risk`는 같은 state revision을 기준으로 한다.

## API 계약

### `GET /display`

통합 backend가 LCD용 `display.html`과 `common.css`를 같은 origin으로 제공한다. LCD 브라우저는
별도 센서 receiver를 실행하지 않고 `GET /api/state`를 사용하므로, backend의 TCP 9000 listener와
센서 상태를 공유한다. Raspberry Pi LCD 주소는 `http://<raspberry-pi-ip>:8000/display`다.

### `GET /api/status`

최소 필드:

- `timestamp`
- `revision`, `publication_revision`
- `system`, `system_health`
- `device_health` — ESP32 장치·transport·센서 취득 관측성 누적값
- `risk`
- `mmwave`, `thermal`, `co2`, `pir`

각 센서는 `state`, `ai`, `risk_component`를 함께 제공한다. raw Thermal frame은 API JSON에 포함하지 않는다.

### `GET /api/sensors`

top-level `device_health`와 네 센서의 state/AI/risk overlay를 반환한다. stale 상태와 마지막 값은 구분되어 유지된다.

### `GET /api/events?limit=100`

최신순으로 최대 200개를 반환한다. PHASE 7에서는 메모리 ring buffer이며 다음 전이를 기록한다.

- risk level
- system health
- emergency 시작/해제
- 센서 status
- runtime error

PHASE 7 구현 시에는 프로세스를 재시작하면 사라지는 메모리 ring buffer였다. PHASE 8 완료 후에는 같은 endpoint가 SQLite 영속 이벤트를 우선 반환한다.

### `GET /api/state`

기존 LCD 서버의 필수 필드 `state`, `room`, `revision`, `updated_at`을 유지하는 읽기 전용 호환 view다.

| Risk | LCD state |
|---|---|
| emergency | `emergency` |
| DANGER | `danger` |
| WARNING | `warning` |
| NORMAL + Thermal human | `normal-occupied` |
| NORMAL + no confirmed human | `normal-empty` |
| risk unavailable | `offline` |

기존 `POST /api/state`는 센서 기반 risk를 수동으로 덮어쓰므로 통합 backend에서는 제공하지 않는다.

### `GET /health`

HTTP process liveness(`ok`)와 첫 publication 준비 여부(`ready`)를 분리한다. receiver 통계와 마지막 runtime error도 포함한다.

### `WS /ws`

`publication_revision`이 바뀔 때 `/api/status`와 같은 schema를 전송한다. 클라이언트가 끊겨도 sensor runtime은 계속 동작한다.

FastAPI 기본 문서는 `/docs`, OpenAPI JSON은 `/openapi.json`에서 확인할 수 있다.

## Raspberry Pi 설치 및 실행

```bash
cd ~/integration
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r ./sources/ondevice_ai/requirements-pi.txt
python -m pip install -r ./requirements-backend.txt
python ./backend/run_backend.py \
  --api-host 0.0.0.0 --api-port 8000 \
  --sensor-host 0.0.0.0 --sensor-port 9000
```

확인:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/status
curl http://127.0.0.1:8000/api/sensors
curl 'http://127.0.0.1:8000/api/events?limit=20'
```

한 명령이 TCP gateway, state, AI, risk, FastAPI lifecycle을 모두 시작하고 종료한다. `Ctrl+C` 시 FastAPI lifespan이 receiver와 evaluation thread를 정리한다.

## 검증 범위

- thread-safe publication과 동시 read/write
- strict JSON 및 NaN 차단
- event transition, 순서, capacity, query limit
- 필수 API response schema
- 기존 LCD state 매핑
- 초기 no-data 상태가 `FAILED/null`로 전달되는지 확인
- FastAPI optional dependency 경계
- backend CLI 인자/도움말

현재 Windows 검증 runtime에는 `fastapi`, `uvicorn`, `httpx`가 설치되어 있지 않다. 그래서 framework route의 실제 HTTP/WebSocket 요청 시험은 수행하지 않았고, 순수 response/store/runtime 계약 67개 테스트를 통과했다. Raspberry Pi에서 requirements 설치 후 `/docs`, curl, WebSocket smoke test가 필요하다.

API는 읽기 전용이지만 기본 실행은 LAN 전체에 bind한다. 외부 네트워크에 직접 공개하지 말고 방화벽 또는 reverse proxy 인증 경계 안에서 사용해야 한다.

## 변경 전/후

- 변경 전: LCD용 수동 상태 API와 별도 중복 receiver만 존재했다.
- 변경 후: strict sensor gateway의 동일 publication을 REST와 WebSocket으로 제공하며 기존 LCD GET schema도 유지한다.

PHASE 7 FastAPI 연결 완료.
