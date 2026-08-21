# 센서 연결 전 런타임 검수 및 위험도 산식 v1 (2026-08-21)

대상 저장소: `integration` (main @ `3d2f651`, PR #27 머지 직후)
검증 환경: Linux x86_64 / Python 3.12.3 / `ai_edge_litert` (실제 TFLite invoke 수행)
하드웨어: 미사용 (Raspberry Pi, ESP32, MR60BHA2, MLX90640 모두 불필요)

검수 도구: `hil/preconnect_runtime_audit.py`
- `data/mmwave/*.jsonl`, `data/co2/*.jsonl` 실측 필드 캡처를 실제 `safenest.telemetry.v1`
  TCP 프레임으로 재생하고, 실제 루프백 소켓으로 살아있는 `SafeNestRuntime`에 주입한다.
- 스텁 모델을 쓰지 않는다. `LazyModel` → `sources/ondevice_ai/inference/*` → 실제 `.tflite`.
- 실측 thermal 캡처가 저장소에 없으므로 thermal 프레임만 합성이며, 보고서에
  `SYNTHETIC_*`로 명시된다.

실행:

```bash
python hil/preconnect_runtime_audit.py                          # 캡처 그대로
python hil/preconnect_runtime_audit.py --inject-presence        # presence 결손만 보정
python hil/preconnect_runtime_audit.py --inject-presence --thermal-shape lying
python hil/preconnect_runtime_audit.py --risk legacy            # 구 V4 산식과 비교
```

---

## 1. 결론 요약

| 질문 | 결과 | 근거 |
|---|---|---|
| AI가 읽을 수 있는 데이터가 오는가 | **통과** (수정 후) | 4개 센서 전부 `LIVE`, `system=ONLINE`, M-N4 캐노니컬 윈도우 `CANONICAL_WINDOW_READY` (연속 구간 31,160 ms / 채택 업데이트 266개) |
| AI가 그것을 연산하는가 | **부분 통과** | thermal·mmWave는 실제 TFLite invoke 성공. **CO₂는 구조적으로 불가** (`humidity_percent`가 와이어 스키마에 없음) |
| 위험도 산식이 통과하는가 | **통과** | 신규 `SAFENEST_RISK_V1`이 점수·등급·비상 판정을 산출하고 `publish → SQLite`까지 저장 확인 |

오늘 작업 중 **수정 1건, 신규 2건**을 반영했고, 남은 블로커 3건은 코드가 아니라
펌웨어/와이어 스키마와 모델 품질 문제다.

---

## 2. 발견 P0 — MR60 위상 윈도우가 잘못된 계층에서 누적되고 있었음 (수정 완료)

PR #27은 ESP의 중첩 `mmwave` 필드를 프로토콜·상태 계층까지 배선했지만,
`MR60CanonicalWindowBuilder.ingest()`를 **발행(publication) 경로**에만 연결했다.

수정 전 호출 지점은 `ai/pipeline.py::_mmwave` 단 한 곳이었고, 이 함수는
`evaluate()` 당 1회만 실행된다. 운영 발행 주기는 기본 15초(`backend/run_backend.py`)다.

M-N4 계약(`sources/ondevice_ai/scripts/mmwave_m_n4_canonical.py`)이 요구하는 것:

- `WINDOW_SECONDS = 30.0`, `SAMPLE_COUNT = 240` (8 Hz)
- 갭 허용치 `max(GAP_FLOOR_S=0.40, 4 × median_dt)`

즉 15초 간격 샘플링으로는 30초 연속 구간을 채울 수도 없고, 채워도 갭 검사에서
`WINDOW_CONTAINS_LARGE_GAP`으로 영구 탈락한다. 실측 재생에서 이 상태가 그대로 재현됐다:

```
! mmwave: INSUFFICIENT_CONTINUOUS_DURATION
    canonical_window_status: RESPIRATORY_WINDOW_WARMING_UP
    accepted_update_count: 1          <-- 1017개의 위상 업데이트를 보냈는데 1개만 채택
```

기존 테스트가 이 결함을 못 잡은 이유: `tests/test_mmwave_mn9_runtime.py`가
`pipeline.evaluate(snapshot_for(i))`를 125 ms 간격으로 241회 호출해
**발행 주기가 와이어 주기와 같다고 가정**하기 때문이다.

### 수정 내용

- `ai/pipeline.py`: `OnDeviceAIPipeline.observe_telemetry(packet)` 추가.
  수신 스레드가 패킷마다 호출한다. 스냅샷 기반 호출자(오프라인 재생, 단위 테스트)를
  위해 `_mmwave_wire_observed` 플래그로 이중 누적을 방지한다.
- `backend/runtime.py`: `_on_packet`에서 `TelemetryPayload`일 때 `observe_telemetry` 호출.
- `gateway/run_ai_gateway.py`, `gateway/run_risk_gateway.py`: 동일 배선.
- `ai/mmwave_canonical_runtime.py`: 수신 스레드와 발행 스레드가 동시 접근하므로
  누적 상태를 `threading.RLock`으로 보호.
- `tests/test_mmwave_mn9_runtime.py`: 회귀 테스트 2건 추가
  (`WireRatePhaseAccumulationTests`) — 발행 1회로는 윈도우가 만들어지지 않는다는
  네거티브 테스트를 포함.

수정 후 실측 재생 결과:

```
canonical_window_status: CANONICAL_WINDOW_READY
continuous_span_ms: 31160.0
accepted_update_count: 266
input_shape: [1, 240, 1]
MAD: 0.10109088032157235   mad_collapsed: False
```

---

## 3. 실측 필드 캡처 프로파일 (`data/mmwave/20260817_09_mmwave.jsonl`)

| 항목 | 값 | 판정 |
|---|---|---|
| 재생 레코드 | 1200 | - |
| `breath_phase` 존재 | 1200 / 1200, 범위 [-0.633, 0.373] | 정상 |
| 구별되는 위상 업데이트 | 1017 (8 ms 전진 허용치 적용 후) | 정상 |
| 업데이트 간격 | median 140 ms, p95 179 ms, **max 221,502 ms** | 장시간 단절 존재 |
| M-N4 갭 임계 | 560 ms, **15회 초과** | 캡처 중 스트림 단절 15회 |
| `respiration_rate_bpm` | min 0.0 / mean **9.63** / max 27.0 | 아래 참고 |
| `human_detected_raw` | **0 / 1200** | 블로커 (§4) |
| `humidity_percent` | **0 / 1200** | 블로커 (§5) |

median 140 ms는 약 7.1 Hz로 8 Hz 계약과 정합한다. 갭 15회는 정상 동작이며,
윈도우 빌더가 그 구간을 폐기하고 최신 30초만 사용하도록 설계되어 있다.

`respiration_rate_bpm` 평균이 9.63이고 0.0까지 내려가는 점이 중요하다. 구 V4 산식의
정상 대역 12–20 rpm을 쓰면 **실측 데이터 대부분이 상시 `ABNORMAL_RESPIRATION_RPM`으로
집계**된다. v1에서 대역을 10–24 rpm으로 넓히고 지속성 카운터를 도입한 근거다.

---

## 4. 블로커 B1 — `human_detected_raw`가 운영 펌웨어에 없음

`ai/pipeline.py::_mmwave`는 캐노니컬 윈도우가 준비되어도 presence가 확인되지 않으면
추론을 차단한다(안전 게이트, 의도된 동작):

```
! mmwave: PRESENCE_STATE_UNAVAILABLE
    canonical_window_status: CANONICAL_WINDOW_READY
    missing: ['human_detected_raw']
```

그런데 `LATEST_SOURCE_PROVENANCE.json`이 `canonical_flash_source`로 지정한
`sources/display-test2/esp32_sensor_node/esp32_sensor_node.ino`의 텔레메트리 JSON
빌더(약 671–706행)에는 `human_detected_raw` 필드가 **없다**. 해당 필드는 별개
펌웨어인 `sources/devices/mmwave/firmware/src/main.cpp:342`에만 존재한다.

즉 **센서를 연결해도 mmWave AI는 영구히 억제된다.** 런타임 측은 이미 준비됐고
(`gateway/protocol.py:248`가 중첩·최상위 양쪽에서 읽음, `state/manager.py:199-201`가
그대로 전달), 남은 작업은 펌웨어 1줄 추가다.

필요 조치: 운영 `.ino`의 `mmwave` 중첩 객체에 `"human_detected_raw":%s` 추가
(MR60 라이브러리의 presence 불리언 사용). 프로토콜은 선택 필드로 이미 수용하므로
후방 호환 확장이다.

검증: `--inject-presence`로 이 필드만 보정하면 M-N9 TFLite invoke가 성공한다.

---

## 5. 블로커 B2 — `humidity_percent` 경로가 어디에도 없음 → CO₂ AI 도달 불가

CO₂ 모델(`co2_occupancy_int8_v0.1.0.tflite`)의 입력 계약은
`[CO2_slope, Humidity, CO2]` 3-피처 `[1,3]` int8이다
(`sources/ondevice_ai/inference/co2_interpreter.py:125-126`).

- ESP 펌웨어는 SCD40에서 습도를 **실제로 읽는다** (`esp32_sensor_node.ino:479-480`).
- 그러나 텔레메트리 JSON에 담지 않는다.
- `gateway/protocol.py`의 `safenest.telemetry.v1` 스키마에 `humidity_percent`가 없다.
- `state/manager.py::_ingest_co2`도 습도를 쓰지 않는다.

검증: `--inject-humidity 45`로 최상위에 필드를 넣어도 디코더가 무시하므로
결과가 변하지 않는다 (`INPUT_UNAVAILABLE`, `missing: ['humidity_percent','co2_slope']` 유지).

필요 조치 (3파일):
1. `.ino` 텔레메트리 JSON에 `"humidity_percent"` 추가 (값은 이미 읽고 있음)
2. `gateway/protocol.py` — 선택 유한 실수 필드로 디코딩
3. `state/manager.py::_ingest_co2` — `values["humidity_percent"]`에 반영

참고: 위험도 자체는 CO₂ AI 없이도 산출된다. v1의 CO₂ 성분은 규칙 기반 ppm 곡선이며
모델은 재실(occupancy) 판정 보조용이다. 따라서 B2는 위험도 산식의 블로커가 아니라
"3-모델 전부 가동" 목표의 블로커다.

---

## 6. 블로커 B3 — M-N9 모델 출력이 판별력이 없음

presence를 보정해 실제 필드 위상 데이터로 M-N9 invoke를 성공시킨 결과:

```
mmwave  True  tflite  APNEA-proxy  conf=0.418
  probabilities: [0.22265625, 0.359375, 0.41796875]
  model_sha256: 3b008af4be0facc4037c2afd3fe39292fb794208eb4370dbe6916b2d15aa38a4
```

3클래스 헤드에서 1위 0.418, 2위 0.359 — 마진 0.059다. 균등분포(0.333)와 거의 구분되지
않으며, 그 상태로 최고 위험 클래스 `APNEA-proxy`를 선택했다. `LATEST_SOURCE_PROVENANCE.json`의
`DEVICE_VALIDATED: "NO"`, `PI_SMOKE: "NOT_PERFORMED"`와 일치하는 상태다.

thermal 모델은 대비적으로 정상 판별력을 보인다 (동일 런타임에서 직접 확인):

| 합성 입력 | 결과 | 신뢰도 |
|---|---|---|
| 균일 프레임 | `NOT_HUMAN` | 1.000 |
| 세로 블롭(직립) | `HUMAN_NORMAL` | 1.000 |
| 가로 블롭(누움) | `HUMAN_FALL` | 1.000 |

이 때문에 v1 산식은 (a) 판별력 게이트로 무의미한 출력을 채점하지 않고,
(b) mmWave 가중치를 낮추고, (c) 미검증 `APNEA-proxy`가 단독으로 DANGER를 만들지
못하게 한다(§7). B3 해소는 재학습/실기 검증 과제이며 런타임 배선 과제가 아니다.

---

## 7. 신규 위험도 산식 v1 (`SAFENEST_RISK_V1`)

구 V4(`sources/ondevice_ai/risk/risk_config.json`, 가중치 0.35/0.35/0.15/0.15,
임계 30/60)는 사용하지 않는다. 신규 구현:

- 산식: `risk/formula_v1.py` (`SafeNestRiskFormulaV1`)
- 설정: `risk/risk_formula_v1.json` (이 저장소 소유, 프리즈된 `sources/` 트리 밖)
- 기본 적용: `backend/runtime.py`, `gateway/run_risk_gateway.py`
- 테스트: `tests/test_risk_formula_v1.py` (17건 통과)
- 출력 문서는 구 문서의 **상위집합**이므로 `backend/store.py`, `backend/views.py`,
  `database/repository.py`는 무수정으로 동작한다.

### 7.1 기본 식

```
R = 100 × Σ(wᵢ·sᵢ, 가용 성분) / Σ(wᵢ, 가용 성분)
level = max(level(R), 에스컬레이션 플로어)
```

가중치 (근거: 현재 증거 성숙도)

| 성분 | v1 | 구 V4 | 근거 |
|---|---|---|---|
| co2 | **0.30** | 0.35 | 유일하게 연속 신뢰 가능한 라이브 신호이고 밀폐공간 1차 위험원 |
| thermal | **0.30** | 0.15 | 실측 필드 FLOAT/INT8 등가성 감사 완료, 판별력 확인 |
| mmwave | **0.25** | 0.35 | `DEVICE_VALIDATED: NO`, 출력이 `APNEA-proxy` (§6) |
| pir | **0.15** | 0.15 | 보조 신호 |

임계: `WARNING ≥ 30`, `DANGER ≥ 65`

### 7.2 구 V4에 없던 3가지 성질

**(1) 에스컬레이션 플로어** — 가중합만 쓰면 심각한 단일 신호가 평온한 나머지에 희석된다.
구 V4에서는 CO₂ 3000 ppm(위험 수준)이 다른 성분이 정상일 때 NORMAL로 집계될 수 있었다.
v1은 개별 신호가 등급 하한을 강제한다.

| 조건 | 하한 |
|---|---|
| thermal `HUMAN_FALL` & conf ≥ 0.80 | DANGER + 비상 |
| CO₂ ≥ 5000 ppm | DANGER + 비상 |
| CO₂ ≥ 2500 ppm | WARNING |
| CO₂ 상승률 ≥ 50 ppm/min | WARNING |
| presence 확인 + 무움직임 ≥ 180 s | WARNING |
| `APNEA-proxy` 2회 연속 (미검증) | **WARNING까지만** |
| `APNEA` + `apnea_verified=true` (하드웨어 확인) | DANGER + 비상 |

**(2) 증거 충분성 게이트** — 가용 가중치 합이 0.50 미만이면 NORMAL을 발행하지 않고
`INDETERMINATE`를 낸다. PIR 하나(0.15)만 살아있을 때 "정상"이라고 말하는 것은
위험하다. 전 성분 불가용이면 등급 `None` + `system_health=FAILED`로 fail-closed.

**(3) 판별력 게이트** — AI 결과를 채점하기 전에 신뢰도 ≥ 0.40 **그리고** 상위 두 확률의
마진 ≥ 0.15를 요구한다. 미달이면 성분을 불가용 처리하고
`{SENSOR}_AI_OUTPUT_INDECISIVE` 사유를 남긴다. §6의 M-N9 출력(마진 0.059)이 여기서
정확히 차단되고 규칙 폴백으로 내려간다.

### 7.3 성분 점수

- **thermal**: `NOT_HUMAN`/`HUMAN_NORMAL` → 0.0. `HUMAN_FALL` → conf ≥0.80 시 1.0,
  ≥0.60 시 0.70, 그 미만 0.40 (사유 기록).
- **mmwave** (AI): `NORMAL` 0.0 / `RAPID_OR_ABNORMAL` 0.5 / `APNEA-proxy`·`APNEA` **0.9**
  (프록시이므로 1.0을 주지 않음).
  (규칙 폴백): 정상 대역 **10–24 rpm**(구 12–20에서 확장, §3 근거). 이탈 시 0.5,
  3회 연속 지속 시 0.75. rpm이 0 또는 무효면 성분 불가용.
- **co2**: ppm 구간 선형 곡선 `600→0.0, 1000→0.15, 2000→0.50, 5000→0.90, 10000→1.00`
  (5000 ppm은 OSHA 8시간 TWA 기준선). 상승률 ≥15 ppm/min 시 +0.10, ≥50 시 +0.25, 1.0 클립.
  실측 1184 ppm → 0.202 (구 산식은 0.346으로 과대평가).
- **pir**: 움직임 0.0. **presence 미확인 시 0.0이 아니라 불가용** — 0.0을 주면 총점을
  조용히 끌어내린다. presence 확인 + 무움직임은 유예 30 s부터 위험 180 s까지 선형 상승.
  (구 V4는 15 s에 곧바로 1.0으로, 가만히 앉아있는 사람에게도 최대 점수를 줬다.)

### 7.4 부수 사항

- 구 `risk/engine.py`의 비상 오버라이드는 `mmwave.state == "APNEA"`를 검사하는데
  파이프라인은 `"APNEA-proxy"`를 발행한다. **구 산식의 무호흡 오버라이드는 사문화 상태**였다.
  v1은 두 문자열을 모두 처리한다.
- `sources/ondevice_ai/config/risk_rules.json`과 `risk_rules.yaml`은 어떤 코드도
  읽지 않는다(문서·해시감사 전용). v1은 이들을 대체하지 않고 무시한다.
- `INDETERMINATE`는 `backend/views.py:114-127`의 등급 분기에 없어 표시 상태가
  `offline`로 떨어진다. fail-closed이므로 안전하지만, 대시보드 문구 보강이 필요하다.

---

## 8. 실측 재생 최종 결과

`python hil/preconnect_runtime_audit.py --inject-presence` (presence 결손만 보정)

```
[Q1 INGEST]  mmwave/thermal/co2/pir 전부 LIVE,  system = ONLINE
[Q2 COMPUTE] thermal  tflite  HUMAN_NORMAL   conf 1.000   0.146 ms
             mmwave   tflite  APNEA-proxy    conf 0.418   0.145 ms
             co2      unavailable  INPUT_UNAVAILABLE (missing humidity_percent)
             pir      rule    NO_MOTION
[Q3 RISK]    SAFENEST_RISK_V1  score 18.5645  level NORMAL
             effective_weight 1.0  evidence_sufficient True
             component_status {mmwave: RULE_FALLBACK, co2: RULE, pir: RULE, thermal: AI}
             reasons [PRESENCE_FROM_MMWAVE, ABNORMAL_RESPIRATION_RPM, HIGH_CO2_WARNING]
[Q3 PERSIST] publish -> SQLite 저장 확인 (score 18.5645 / level NORMAL)
```

`--thermal-shape lying` (낙상 형상) 추가 시 비상 경로까지 관통:

```
thermal  tflite  HUMAN_FALL  conf 1.000  probabilities [0.0, 0.0, 1.0]
SAFENEST_RISK_V1  score 100.0  level DANGER  is_emergency True
  score_level WARNING (level_source=EMERGENCY)
  escalation_floors ['thermal_fall_confident']
  reasons [EMERGENCY_HUMAN_FALL, ..., FLOOR_THERMAL_FALL_CONFIDENT]
SQLite 저장: score 100.0 / level DANGER
```

`degraded_mode`는 계속 `True`다. mmWave가 규칙 폴백이고 CO₂ AI가 불가용인 한 정상적인
표시이며, B2·B3가 해소되어야 `HEALTHY`가 된다.

---

## 9. 남은 작업 (센서 연결 전)

| # | 항목 | 담당 영역 | 상태 |
|---|---|---|---|
| 1 | 위상 윈도우 와이어 레이트 누적 | Pi 런타임 | **완료** (§2) |
| 2 | 위험도 산식 v1 | Pi 런타임 | **완료** (§7) |
| 3 | `human_detected_raw` 펌웨어 추가 | ESP32 `.ino` | **미착수 — mmWave AI 최대 블로커** |
| 4 | `humidity_percent` 3파일 배선 | ESP32 + protocol + state | **미착수 — CO₂ AI 블로커** |
| 5 | M-N9 판별력 확보 | 모델 재학습/실기 검증 | **미착수** (§6) |
| 6 | 실측 thermal 캡처 커밋 | 데이터 | 미착수 (`data/thermal/` 비어 있음, 합성 프레임으로 대체 중) |
| 7 | 대시보드 O4 요소 복구 | 웹 | 미착수 — `runtimeBadge`, `thermalSensor`, `thermalAiStatus`, `co2Ai`, `pirAi`가 새 `web/dashboard/`에 없어 기계판독 계약이 깨짐 |
| 8 | 비상 시 SMS/119 자동 연동 | 서비스 | 설계상 수동. `is_emergency`는 부저 래치와 이벤트 로그까지만 자동 |

### 기존 테스트 실패 22건에 대하여

본 작업 전후 실패 집합은 **완전히 동일**하다(회귀 없음). 22건 전부 PR #26의
웹/포털 백엔드 도입으로 라우트와 대시보드 DOM이 바뀐 뒤 갱신되지 않은 테스트다
(`test_backend`, `test_dashboard`, `test_o4_partial_availability_ui`,
`test_stage7_offline_preflight`, `test_stage9_smoke`). 런타임 데이터·AI·위험도
경로와는 무관하되 위 7번 항목과 같은 원인이다.

참고: `python -m hil.preflight`를 저장소 루트에서 직접 실행하면 `ok: true`이며,
필수 실패는 없다(라즈베리파이 항목만 optional 실패).


---

## 10. 변경 파일 전체 목록

### 수정 (6)

| 파일 | 변경 내용 |
|---|---|
| `ai/pipeline.py` | `OnDeviceAIPipeline.observe_telemetry(packet)` 신설 — 수신 스레드가 패킷마다 MR60 위상을 누적기에 넣는다. `_mmwave_wire_observed` 플래그로 스냅샷 기반 호출자(오프라인 재생·단위 테스트)와의 이중 누적을 차단. `TelemetryPayload` import 추가. |
| `ai/mmwave_canonical_runtime.py` | `MR60CanonicalWindowBuilder`의 누적 상태를 `threading.RLock`으로 보호 (수신 스레드가 쓰고 발행 스레드가 읽음). `ingest`/`latest`를 락 래퍼 + `_ingest_locked`/`_latest_locked`로 분리. 계약 로직 자체는 무변경. |
| `backend/runtime.py` | `_on_packet`에서 `TelemetryPayload`일 때 `ai_pipeline.observe_telemetry(packet)` 호출, 예외는 `mmwave_phase_window` 런타임 에러로 기록. 기본 위험도 엔진을 `SafeNestRiskEngine` → `SafeNestRiskFormulaV1`로 교체. `risk_engine` 파라미터 타입을 주입 가능한 `object | None`으로 완화. |
| `gateway/run_risk_gateway.py` | 위상 와이어 레이트 누적 배선 + 위험도 엔진 v1 적용. |
| `gateway/run_ai_gateway.py` | 위상 와이어 레이트 누적 배선. |
| `tests/test_mmwave_mn9_runtime.py` | `WireRatePhaseAccumulationTests` 추가 (2건). ① 런타임 수신 경로만으로 260 패킷을 넣으면 추가 발행 없이 `CANONICAL_WINDOW_READY`가 되고 텐서가 `(1,240,1)`이며 presence 부재로 추론은 게이트된다. ② 네거티브 — 상태 매니저에 260 패킷을 넣고 발행을 1회만 하면 `accepted_update_count == 1`이다(회귀 방지). |

### 신규 (5)

| 파일 | 역할 |
|---|---|
| `risk/formula_v1.py` | `SafeNestRiskFormulaV1`. 가중 융합 + 에스컬레이션 플로어 + 증거 충분성 게이트 + AI 판별력 게이트. 출력 `RiskEvaluationV1`은 구 문서의 상위집합(`formula_id`, `score_level`, `level_source`, `effective_weight`, `evidence_sufficient`, `escalation_floors` 추가)이라 `backend/store.py`·`backend/views.py`·`database/repository.py`가 무수정 동작. `RiskComponent`는 `risk/engine.py`에서 재사용. |
| `risk/risk_formula_v1.json` | v1 설정. 가중치·임계·CO₂ 곡선·판별력 기준·플로어·비상 오버라이드. 프리즈된 `sources/ondevice_ai/` 트리 밖, 이 저장소 소유. 각 값의 채택 근거를 `rationale`에 기록. |
| `tests/test_risk_formula_v1.py` | v1 행위 계약 17건. 설정 계약, 판별력 게이트(균등분포 거부·저신뢰 거부·TTL 초과 거부), 플로어(낙상 비상·CO₂ 희석 방지·즉시위험·미검증 APNEA는 WARNING 상한·하드웨어 확인 APNEA는 비상), 증거 충분성(가중치 소수 → `INDETERMINATE`·전부 불가용 → fail-closed·과반 → NORMAL 허용), PIR 의미론, CO₂ 곡선 단조성. |
| `hil/preconnect_runtime_audit.py` | 센서 연결 전 감사 도구. 실측 캡처를 실제 `safenest.telemetry.v1` TCP 프레임으로 재생해 루프백 소켓으로 살아있는 `SafeNestRuntime`에 주입하고, `state → AI → risk → store → SQLite` 전 구간을 관통시켜 Q1/Q2/Q3 판정표를 낸다. 스텁 모델 없음. thermal만 합성이며 `SYNTHETIC_*`로 명시. |
| `docs/20260821_Preconnect_Runtime_Audit_And_Risk_Formula_V1_KO.md` | 본 문서. |

무변경: `gateway/protocol.py`, `state/manager.py`, `risk/engine.py`, `backend/store.py`,
`backend/views.py`, `database/*`, `sources/**` (프리즈 스냅샷), ESP32 펌웨어.

### 회귀 확인

`git stash` 대조로 변경 전후 실패 집합이 **문자열 단위로 동일**함을 확인했다
(22 failed / 1 skipped). 신규 통과 19건(v1 17 + 위상 회귀 2)만 증가.

```
before: 22 failed, 234 passed
after : 22 failed, 253 passed
diff(FAILED 목록) = 빈 집합
```

---

## 11. 재현 절차

이 저장소는 `requirements-backend.txt`에 FastAPI/uvicorn만 선언하고 있어, AI 경로를
돌리려면 numpy/scipy와 TFLite 런타임이 추가로 필요하다.

```bash
cd integration

# pip/venv 모듈이 없는 배포판에서의 부트스트랩 (시스템 파이썬은 건드리지 않는다)
python3 -m venv --without-pip .venv
curl -sSLO https://bootstrap.pypa.io/get-pip.py && .venv/bin/python get-pip.py

.venv/bin/pip install numpy scipy "fastapi>=0.110,<1" "uvicorn[standard]>=0.29,<1" pytest
.venv/bin/pip install ai-edge-litert     # 실제 TFLite invoke에 필요

# 전체 테스트
.venv/bin/python -m pytest tests -q

# 센서 연결 전 감사
.venv/bin/python hil/preconnect_runtime_audit.py
.venv/bin/python hil/preconnect_runtime_audit.py --inject-presence
.venv/bin/python hil/preconnect_runtime_audit.py --inject-presence --thermal-shape lying
.venv/bin/python hil/preconnect_runtime_audit.py --risk legacy --inject-presence   # 구 V4 비교
.venv/bin/python hil/preconnect_runtime_audit.py --json /tmp/audit.json            # 증거 저장
```

감사 도구 옵션:

| 옵션 | 의미 |
|---|---|
| `--mmwave` / `--co2` | 재생할 캡처 파일 지정 (기본: `data/` 내 최신) |
| `--limit N` | 재생 레코드 수 (기본 1200 ≈ 8 Hz로 2.5분) |
| `--inject-presence` | 결손된 `mmwave.human_detected_raw=true`만 합성 (B1 격리 검증용) |
| `--inject-humidity P` | `humidity_percent` 합성 시도 — **와이어 스키마에 없어서 무효임을 보이는 용도** (B2 증명) |
| `--thermal-shape` | `upright` / `lying` / `flat` 합성 thermal 형상 |
| `--risk` | `v1`(기본, 런타임 기본값과 동일) / `legacy`(구 V4 비교) |
| `--json PATH` | 전체 증거 문서 저장 |

`.venv/`는 이미 `.gitignore`에 포함되어 있다.
