# PHASE 8 — SQLite 로그 연결

## Repository 감사 결과

GitHub `main`에는 `sqlite3`, SQL schema, database repository, event persistence 구현이 없다. 기존 센서 CSV/JSONL 로그는 학습·검증 자료이며 운영 backend DB가 아니다.

따라서 표준 라이브러리 `sqlite3`만 사용하는 schema version 1을 새로 추가했다. 별도 Python 패키지는 필요하지 않는다.

## 저장 구조

### `sensor_snapshots`

평가 publication마다 다음 값을 저장한다.

- timestamp, state/publication revision
- system, system health
- 네 센서 status
- mmWave presence, respiration rate, heart rate
- Thermal maximum raw, human probability, AI state
- CO₂ ppm
- PIR motion
- risk score, risk level, emergency, reason
- event type=`SNAPSHOT`

현재 Thermal TCP payload에는 온도 보정 계수가 없다. 그러므로:

- `thermal_max_raw`: 센서 raw maximum 저장
- `thermal_max_temp_c`: `NULL`

33°C 등의 값을 역산하거나 임의 변환하지 않는다.

### `risk_events`

PHASE 7 transition event를 영속화한다.

- event ID와 전역 sequence
- timestamp와 event type
- 연결된 publication revision
- 당시 risk score/level/system health
- event별 details JSON

프로세스 재시작 시 DB의 마지막 publication revision과 event sequence를 읽어 이어서 기록한다. 마지막 risk·sensor status도 baseline으로 복원하여 재시작 직후 변화가 `SNAPSHOT_INITIALIZED`로 잘못 사라지지 않게 한다.

## 저장하지 않는 데이터

Thermal 80×62 raw frame 전체는 SQLite에 저장하지 않는다. schema에 BLOB/pixel column이 없으며 metadata와 AI 결과만 기록한다. 이로써 초당 snapshot 로그가 프레임 크기 때문에 급격히 증가하는 문제를 막는다.

## 안정성

- 한 publication snapshot과 그 transition events는 하나의 transaction으로 기록한다.
- `publication_revision`, `event_id` 중복만 idempotent하게 무시한다.
- CHECK constraint 오류는 숨기지 않고 transaction 전체를 rollback한다.
- WAL, `synchronous=NORMAL`, 5초 busy timeout을 사용한다.
- SQLite write가 실패해도 TCP/state/AI/risk/API loop는 메모리 store로 계속 동작한다.
- `/health`의 `database.available/error/counts/schema_version`에서 장애를 확인할 수 있다.
- NaN/Inf는 SQL 실행 전에 strict JSON 검증으로 차단한다.
- 알 수 없는 schema version은 자동으로 덮어쓰지 않고 시작을 거부한 뒤 메모리 fallback 상태로 전환한다.

## API 변화

- `GET /api/events`: SQLite의 영속 이벤트를 최신순으로 반환
- `GET /api/history?limit=100`: snapshot history를 최신순으로 반환
- `GET /health`: DB path, availability, 오류, row count 포함

limit는 1–200만 허용한다. DB 장애 중에는 events/history가 메모리 fallback을 사용하며 response의 `persistence`가 SQLite가 아님을 표시한다.

## Raspberry Pi 실행

기본 DB 위치는 `./data/safenest.db`다.

```bash
cd ~/integration
source .venv/bin/activate
python ./backend/run_backend.py \
  --api-host 0.0.0.0 --api-port 8000 \
  --sensor-host 0.0.0.0 --sensor-port 9000 \
  --db-path ./data/safenest.db
```

확인:

```bash
curl 'http://127.0.0.1:8000/api/events?limit=20'
curl 'http://127.0.0.1:8000/api/history?limit=20'
curl http://127.0.0.1:8000/health
```

SQLite CLI가 설치되어 있다면:

```bash
sqlite3 ./data/safenest.db '.tables'
sqlite3 ./data/safenest.db \
  'SELECT timestamp,risk_level,risk_score FROM sensor_snapshots ORDER BY id DESC LIMIT 10;'
```

## 운영상 남은 사항

- 자동 보존 기간/용량 삭제 정책은 아직 없다. 장기 운영 전 디스크 예산과 retention 정책을 확정해야 한다.
- 실행 중 파일 하나만 단순 복사하면 WAL의 최신 transaction이 빠질 수 있다. backup 시 프로세스를 정상 종료하거나 SQLite `.backup` 기능을 사용한다.
- DB에는 사람 존재·생체·환경 상태가 포함된다. 외부 공유와 파일 권한을 제한해야 한다.
- 실제 Pi 저장장치의 write latency와 장기 내구성 benchmark는 남아 있다.

## 검증 범위

- schema 생성과 version 확인
- 모든 요구 필드 round trip
- Thermal raw frame 미저장
- transaction rollback과 중복 idempotency
- NaN/Inf 차단과 query limit
- DB close와 unknown schema fail-closed
- 재시작 후 revision/event sequence와 transition 연결
- 20-thread 동시 publication
- DB 장애 시 메모리/API 지속
- 전체 이전 단계 회귀 테스트

## 변경 전/후

- 변경 전: events와 최신 publication이 프로세스 메모리에만 존재했다.
- 변경 후: snapshot/event가 transaction으로 SQLite에 남고 API history로 조회되며, DB 장애는 sensor runtime과 분리된다.

PHASE 8 SQLite 로그 연결 완료.
