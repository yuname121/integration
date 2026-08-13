# SafeNest V5 Release Readiness

검증 기준일: 2026-08-03

## 범위 판정

- P0 온디바이스 AI 코어: 검증 완료
- 실센서 드라이버: 센서 담당 팀원 영역
- 실센서 통합: 진행 예정
- 웹 UI: 범위 밖
- Raspberry Pi 5 성능: 실측 전까지 미검증

V5 production 경로는 `integrated_node/run_node.py → sensors/provider → inference → risk/risk_engine.py → risk/fallback.py → inference/inference_result.py`다. `integrated_node/safenest_risk_engine.py`는 legacy compatibility다.

## Release gate

| 항목 | 상태 |
|---|---|
| V4 원본/스냅샷 포함 파일 SHA-256 | PASS |
| V4 tar.gz checksum 및 압축 해제 비교 | PASS |
| V5 source가 검증된 스냅샷에서 생성 | PASS |
| TFLite v0.1.0 세 모델 불변성 | PASS |
| V5 경로 독립 validator | PASS |
| sensor provider injection/fail-closed | PASS |
| sensor별 stale TTL 및 CO₂ 5초 cadence | PASS |
| Risk/System Health 분리 및 schema 5.0 | PASS |
| Mock JSON Lines | PASS — HEALTHY, schema 5.0, 4개 component/sensor, NaN/Inf 없음 |
| Real provider 미주입 fail-closed | PASS — FAILED, risk_score/risk_level null, 명시적 오류 코드 |
| 전체 unittest FAIL/ERROR 0 | PASS — 175 실행, 173 PASS, 2 Thermal NPZ SKIP |
| ZIP required files/member SHA-256 | PASS — 생성기 자체 검증 및 sidecar checksum |
| ZIP 압축 해제 후 validator/전체 unittest | PASS — 독립 임시 디렉터리 재검증 |

## 모델

| 모델 | 버전 | SHA-256 |
|---|---|---|
| Thermal fall INT8 | 0.1.0 | `5b56da8d127ccef85f30b6459cc0cfe2d86490e41f3caa5bd2a7b70bbc46ae84` |
| mmWave respiration INT8 | 0.1.0 | `43cdd6f321c2b645232162233e098c2b65549388cbc9e680a17a0eccdc8f0158` |
| CO₂ occupancy INT8 | 0.1.0 | `3a8c86c4c132df0f1edaac668d9a136c3f6234789df48f02bdda8e92f29d0462` |

## 남은 외부 검증

- Thermal-44 packet/orientation/physical-unit 확인
- MR60BHA2 phase semantics, presence/quality, 실제 cadence 확인
- SCD40 CRC, warm-up, 0.2 Hz cadence 확인
- PIR electrical configuration과 event timing 확인
- Raspberry Pi 5에서 latency, CPU/memory, thermal throttling 측정
- 장시간 운용 시험 계획 수립 및 실행

실센서 provider가 아직 합쳐지지 않은 것은 이 AI 배포판의 소프트웨어 차단 사유로 보지 않는다. 코어 회귀와 artifact 검증이 모두 통과하면 상태는 `V5_READY_FOR_TEAM_SENSOR_INTEGRATION`이다.
