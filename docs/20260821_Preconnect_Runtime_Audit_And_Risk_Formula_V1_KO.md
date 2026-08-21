# 센서 연결 전 런타임 검수 및 위험도 산식 v1 (2026-08-21)

대상 저장소: `integration` (main @ `3d2f651`, PR #27 머지 직후)
검증 환경: Linux x86_64 / Python 3.12.3 / `ai_edge_litert` (실제 TFLite invoke 수행)
하드웨어: 미사용 (Raspberry Pi, ESP32, MR60BHA2, MLX90640 모두 불필요)

검수 도구: `hil/preconnect_runtime_audit.py` (신규)
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
| AI가 읽을 수 있는 데이터가 오는가 | **통과** (수정 후) | 4개 센서 전부 `LIVE`, `system=ONLINE`, M-N4 캐노니컬 윈도우 `CANONICAL_WINDOW_READY` (연속 구간 31,180 ms / 채택 업데이트 266개), C-B6 CO₂ 슬로프 `CO2_SLOPE_READY` |
| AI가 그것을 연산하는가 | **통과** (수정 후) | thermal·mmWave·CO₂ 3개 모델 전부 실제 TFLite invoke 성공, `all_models_available=True` |
| 위험도 산식이 통과하는가 | **통과** | 신규 `SAFENEST_RISK_V1`이 점수·등급·비상 판정을 산출하고 `publish → SQLite`까지 저장 확인, `system_health=HEALTHY` |

`python hil/preconnect_runtime_audit.py --inject-presence --limit 4000` 기준 7개 게이트 전부 PASS.

수정 P0 2건(위상 윈도우 계층, CO₂ 선택자·어댑터), 신규 위험도 산식 v1을 반영했다.
남은 블로커는 펌웨어 필드 1건(B1)과 모델 품질 1건(B3)이다.

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

## 4-1. B1 수정 (완료) — 펌웨어 1.3.0이 `human_detected_raw`를 발행한다

`sources/display-test2/esp32_sensor_node/esp32_sensor_node.ino`를 `1.3.0` /
mmWave 스키마 `1.3`으로 올리고, 중첩 `mmwave` 객체에 `human_detected_raw`를 추가했다.
`safenest.telemetry.v1`의 선택 필드이므로 후방 호환 확장이다.

### 함정 1 — 라이브러리의 `isHumanDetected()`는 그대로 쓸 수 없다

설치 라이브러리(`Love4yzp/Seeed-mmWave-library`, `name=Seeed Arduino mmWave`)의
`SEEED_MR60BHA2::isHumanDetected()`는 **"빈 방"과 "아직 보고 없음"을 둘 다 `false`로
반환**하고, 읽는 순간 유효 플래그를 스스로 지운다.

```cpp
bool SEEED_MR60BHA2::isHumanDetected() {
  if (!_isHumanDetectionValid) return false;   // <-- 미보고도 false
  _isHumanDetectionValid = false;
  return _isHumanDetected;                     // <-- 실제 부재도 false
}
```

이걸 그대로 JSON에 넣으면 계약이 요구하는 3-상태(`true` / `false` / `null`)를 표현할 수
없고, "모름"이 "사람 없음"으로 새어 나간다. 다른 게터(`getBreathRate` 등)는 값을
out-파라미터로 넘기고 반환값을 "새 값이 있었는가"로만 쓰기 때문에 이 문제가 없다.

### 해결 — `handleType()` 오버라이드로 0x0F09를 직접 받는다

`SEEED_MR60BHA2::handleType()`은 `public virtual`이고, 기반 클래스
`SeeedmmWave::processFrame()`이 **헤더·데이터 체크섬을 모두 검증한 뒤에** 호출한다.
따라서 서브클래스에서 0x0F09만 가로채면 UART 프레이밍을 재구현하지 않고도
모호성 없는 3-상태를 얻는다. 기반 구현도 그대로 호출해 라이브러리 상태를 보존한다.

```cpp
class SafeNestMR60BHA2 : public SEEED_MR60BHA2 {
  bool handleType(uint16_t type, const uint8_t *data, size_t len) override {
    if (type == (uint16_t)TypeHeartBreath::ReportHumanDetection) {
      if (len < 1) return false;        // 벤더 핸들러는 data[0]을 무검사로 읽는다
      presenceRaw_ = data[0] != 0;
      presencePending_ = true;
    }
    return SEEED_MR60BHA2::handleType(type, data, len);
  }
  bool takePresence(bool &value);      // getBreathRate와 같은 out-파라미터 관용구
};
```

`PRESENCE_MAX_AGE_MS = 5000`을 넘겨 보고가 끊기면 `null`로 떨어진다(부재 주장 아님).
`isFresh()`가 `timestamp != 0`도 검사하므로 **한 번도 보고가 없던 노드는 `false`가
아니라 `null`을 발행한다.** 참조 펌웨어의 `updateStablePresence()` 다수결 안정화는
**도입하지 않았다** — 와이어 계약은 raw 불리언만 요구한다.

### 함정 2 — 이 파일은 2026-08-17부터 컴파일이 안 되고 있었다

커밋 `177db97`이 `formatNullableFloat` → `formatNullablePhase`로 이름을 바꾸면서
(정밀도 `%.2f` → `%.6f`) **호출 지점 3곳을 남겨 두었다.** 즉 `canonical_flash_source`가
미정의 함수를 호출하는 상태였고 플래시 자체가 불가능했다. `%.2f` 헬퍼를 복원했다
(호출부를 `formatNullablePhase`로 돌리면 호흡·심박 정밀도가 조용히 바뀌므로).

### 검증 (하드웨어 없이)

`.ino`에서 헬퍼 4개와 `sendTelemetry()`를 **텍스트로 추출해** 호스트에서 컴파일했다.
포맷 문자열이 실물과 어긋날 수 없다.

- `g++ -Wformat=2 -Werror` 통과 → 포맷/인자 목록 일치
- 최악 길이 **1109 B** < `char json[1536]` → 절단 없음 (`length >= sizeof(json)` 방어 유지)
- 컴파일된 실물 빌더가 낸 바이트를 `decode_telemetry` → `SensorStateManager`에 통과시켜
  `true` / `false` / `null` 3-상태가 `presence_available`로 정확히 사상되는 것을 확인

### 남은 제약 — 감사 도구는 여전히 `--inject-presence`가 필요하다

커밋된 캡처는 전부 `firmware_version: safenest-esp32-sensor-node/1.2.0`,
`schema_version: 1.2` 스탬프이고 `human_detected_raw`가 **0/1200**건이다. 펌웨어를
고쳐도 2026-08-17에 뜬 파일이 소급해서 필드를 갖지는 않는다. 따라서

> `--inject-presence` 없이 전 게이트 PASS는 **코드 변경으로 달성 불가**이며,
> `>=1.3.0` 펌웨어로 mmWave를 재캡처해야 한다 (§9 항목 6b).

스키마 버전을 올린 이유가 이것이다. 이제 캡처만 보고 그 캡처가 재실 게이트를 만족시킬
수 있는지 기계적으로 판정할 수 있다 — B1이 오래 눈에 띄지 않은 원인이 정확히 이
구분이 불가능했다는 점이었다.

---

## 5. 정정 및 수정 P0-2 — CO₂ 입력 계약은 이미 `[CO2, CO2_slope]`였고, 런타임만 안 붙어 있었다

초기 검수에서 이 항목을 "습도가 와이어 스키마에 없어서 CO₂ AI 도달 불가"로 보고했는데,
**틀렸다.** 잠긴 계약은 습도를 요구하지 않는다. 오히려 금지한다.

`sources/ondevice_ai/models/rp_x0_b_complete/co2/input_contract.json`:

```json
{
  "candidate_id": "C_B6_REDUCED_CO2_SLOPE_CANDIDATE_001",
  "feature_count": 2,
  "feature_order": ["CO2", "CO2_slope"],
  "forbidden_additional_inputs": ["Temperature", "Humidity", "Light", "time_of_day", ...],
  "humidity_included": false,
  "temperature_included": false,
  "history_seconds": 150.0,
  "max_internal_gap_seconds": 90.0,
  "slope_method": "ENDPOINT_DIFFERENCE",
  "causality": "PAST_ONLY"
}
```

즉 습도는 미사용이 아니라 **금지 입력**이다. 특성 순서도 구 3-특성
(`[CO2_slope, humidity, CO2]`)과 달리 `[CO2, CO2_slope]`로 역순이다.

### 실제 결손 지점

`hil/rp_x0_b_complete_provisioning_manifest.json`의 CO₂ 항목이 정확히 지목하고 있었다.

```json
"deployment_status": "AVAILABLE_NOT_PRODUCTION_SELECTED",
"runtime_adapter_compatible": false,
"known_limitations": [
  "production adapter still uses historical [CO2_slope, humidity, CO2] v0.1.0",
  "physical-event SCD4x fields still required before live C-B6"
]
```

- `runtime_adapter_compatible: false` — **C-B6 어댑터가 런타임에 없었다.**
- `models.co2` 선택자가 여전히 구 3-특성 v0.1.0을 가리켰다 → 그래서 런타임이 습도를 요구했다.
- 캐노니컬 슬로프 빌더도 없었다. `ai/pipeline.py::_co2`는 30-샘플 deque에
  임의 구간 endpoint difference를 썼고, 150 s 계약·90 s 갭 정책·SOURCE_ACQUISITION_CLOCK
  기준을 전혀 따르지 않았다.
- 두 번째 제약 "physical-event SCD4x fields"는 **이미 충족되어 있었다.** ESP가
  `co2_measurement_event_id` / `co2_measurement_monotonic_ms` /
  `co2_measurement_event_valid`를 보내고 있고 필드 캡처에도 전부 들어 있다.

따라서 이건 펌웨어 과제가 아니라 온디바이스 AI + 파이 런타임 배선 과제였다.

### 수정 내용

1. **신규 어댑터** `sources/ondevice_ai/inference/co2_c_b6_interpreter.py` (`CB6Interpreter`).
   M-N9 어댑터와 같은 엄격도로 아티팩트 SHA-256, 텐서 `[1,2]` int8, 양자화
   `(0.03921568766236305, 0)`, 출력 `[1,1]` int8 `(0.00390625, -128)`,
   스케일러 지문·특성 순서, 임계 0.43, 클래스 맵을 전부 검증한다.
   `humidity_included`가 `false`가 아니면 로드를 거부한다.
   `predict(co2_ppm, co2_slope_ppm_per_min)` — 인자 2개다. 구 3-인자 호출은 `TypeError`.

2. **신규 슬로프 빌더** `ai/co2_canonical_runtime.py` (`CO2SlopeWindowBuilder`).
   `CO2_SLOPE_FEATURE_PROFILE_001`을 그대로 구현한다.
   - 단위 ppm/min, `(co2_now - co2_history_start) / (elapsed_s / 60.0)`
   - endpoint = 소스 시계 기준 age ≥ 150 s인 **가장 오래된 과거** 샘플
   - 시계 기준은 ESP의 `co2_measurement_monotonic_ms` (파이 벽시계 아님)
   - `measurement_event_id`가 바뀔 때만 이력 전진 — 펌웨어가 매 패킷 재발행하고
     런타임이 표시값을 60초로 스로틀하므로, 다른 키로 세면 평평한 가짜 슬로프가 만들어진다
   - 갭 > 90 s, boot 경계, 비단조 시계 → 이력 재시작
   - 보간·미래 샘플·중심 윈도우 금지, float64
   - 상태 코드를 프로파일 어휘로 노출: `CO2_SLOPE_READY` /
     `FEATURE_UNAVAILABLE_WARMUP` / `FEATURE_UNAVAILABLE_GAP_RESTART` /
     `NO_CANONICAL_SLOPE` / `CO2_MEASUREMENT_CLOCK_UNAVAILABLE`.
     **슬로프 미확보를 0.0 ppm/min으로 보고하지 않는다.**

3. **선택자 승격**: 매니페스트에 `co2_occupancy_c_b6` 항목 추가
   (`runtime_role: ACTIVE_C_B6`). 구 `co2` 항목은 `HISTORICAL_CO2_V0_1_0` /
   `HISTORICAL_NOT_ACTIVE: true` / `superseded_by: co2_occupancy_c_b6`로 표기만 하고
   `deployment_allowed`는 건드리지 않아 프리즈 스냅샷 자체 도구가 계속 동작한다.

4. **`ai/runtime.py`**: `_ADAPTERS`를 3-튜플
   `(파일, 클래스, 매니페스트 선택자 키)`로 확장. CO₂만 선택자 키가 `co2`와 다르기 때문이다.
   `_assert_deployment_allowed`도 선택자 키로 조회한다.

5. **와이어 레이트 누적**: `observe_telemetry`가 mmWave 위상과 함께 CO₂ 측정 이벤트도
   먹인다. 150 s 이력을 15초 발행 루프로 채우려 하면 §2와 같은 실패가 재현된다.

6. **점유(occupancy)는 위험 점수가 아니다.** `class_map.json`이
   `risk_semantic: NONE` / `safety_semantic: NONE`이라고 선언하므로, AI 결과의
   `score`는 0.0 + `risk_contribution_deferred: True`로 두고 v1의 CO₂ 성분은
   계속 ppm 규칙으로 계산한다. 점유 확률은 메타데이터로만 노출한다.
   어댑터도 `risk_semantic`이 `NONE`이 아니면 로드를 거부한다.

### 잠금 장부 변경 (확인 필요)

C-B6 아티팩트는 물리적으로 `rp_x0_b_complete` 오버레이 안에 있고, 그 오버레이는
`runtime_role: HISTORICAL_B_STAGE`로 표기되어 있었다. 선택자를 그쪽으로 돌리면
`production_selection_changed`가 `false → true`로 바뀐다. 다음 파일을 일관되게 갱신했다.

| 파일 | 변경 |
|---|---|
| `hil/rp_x0_b_complete_provisioning_manifest.json` | `production_selection_changed: true`, `production_selection_change_scope: "CO2_ONLY_C_B6_REDUCED_FEATURE"`, CO₂ 항목 `PRODUCTION_SELECTED_C_B6` / `runtime_adapter_compatible: true` |
| `.../rp_x0_b_complete/artifact_inventory.json` | C-B6 CO₂ → `LOCKED_B_STAGE, PRODUCTION_SELECTED, ACTIVE_C_B6, OCCUPANCY_ONLY_RISK_SEMANTIC_NONE`. 구 v0.1.0 → `SUPERSEDED_BY_C_B6_REDUCED_FEATURE` |
| `LATEST_SOURCE_PROVENANCE.json` | `tracked_file_count` 1075→1076, `co2_c_b6_promotion` 블록 추가, 오버레이 `runtime_role: HISTORICAL_B_STAGE_EXCEPT_CO2_C_B6` |

**mmWave와 thermal의 B-stage 잠금은 손대지 않았다.**
`mmwave_live_b_gate: "CLOSED"` 유지, mmWave 매니페스트 `HISTORICAL_B_NOT_ACTIVE: true` 유지,
`thermal44_deployment_validated: false` 유지, M-B3 아티팩트는 계속 `LIVE_B_GATE_CLOSED`.
승격 범위는 CO₂ 단독이다.

C-B6 자체의 미검증 항목은 그대로 남는다: `DEVICE_VALIDATED: NO`,
`PI_SMOKE: NOT_PERFORMED`, 임계 0.43은 `TRAIN_INTERNAL_ONLY`, 스케일러·캘리브레이션은
UCI 도메인이며 SCD40 정렬은 C-C 단계 미착수.

### 실측 검증

```
co2  True  tflite  OCCUPIED  conf 0.996  0.046 ms
  probabilities: [0.00390625, 0.99609375]
  model_sha256: c5969b367f5b5e28c4d27f1bdd6220f7d02da92e99604e3f85c9f1291e98dd3b
```

필드 캡처의 실제 SCD40 측정 이벤트 간격은 median 4,750 ms (min 4,739 / p95 5,000)로
90 s 갭 한도 안에 충분히 들어온다. 세션 중 220 s 단절이 1회 있고, 빌더가 그 지점에서
`FEATURE_UNAVAILABLE_GAP_RESTART`를 내고 이후 150 s가 다시 모이면 복구한다.

어댑터 단독 응답 곡선 (임계 0.43):

| ppm | slope | P(OCCUPIED) | 판정 |
|---|---|---|---|
| 300 | 0.0 | 0.012 | VACANT |
| 420 | 0.0 | 0.059 | VACANT |
| 600 | 0.0 | 0.305 | VACANT |
| 800 | 2.0 | 0.914 | OCCUPIED |
| 1184 | 1.0 | 0.996 | OCCUPIED |
| 600 | -5.0 | 0.078 | VACANT |

빈 방 수준(300–420 ppm)에서 VACANT, 재실 수준에서 OCCUPIED로 단조 증가한다.

## 6. 블로커 B3(정정) — M-N9는 판별력이 없는 게 아니라, 호흡 중인 구간에 고신뢰 APNEA-proxy를 낸다

초기 검수에서 단일 윈도우(레코드 1200)만 보고 "판별력 없음"으로 보고했는데, 표본이
부족했다. 같은 캡처의 독립 윈도우 7개를 훑으면 실제 양상은 더 나쁘다.

```
python hil/preconnect_runtime_audit.py --inject-presence --sweep 1200,2400,4000,4800,5600,6400,7000
```

| 레코드 | 클래스 | 신뢰도 | 1·2위 마진 | 판별력 | 위험도 | health |
|---|---|---|---|---|---|---|
| 1200 | APNEA-proxy | 0.418 | 0.059 | **아니오** | 18.56 NORMAL | DEGRADED |
| 2400 | APNEA-proxy | 0.836 | 0.707 | 예 | 28.64 NORMAL | HEALTHY |
| 4000 | APNEA-proxy | 0.996 | 0.996 | 예 | 28.64 NORMAL | HEALTHY |
| 4800 | APNEA-proxy | 0.824 | 0.695 | 예 | 28.55 NORMAL | HEALTHY |
| 5600 | NORMAL | 0.492 | 0.070 | **아니오** | 8.11 NORMAL | DEGRADED |
| 6400 | APNEA-proxy | 0.957 | 0.922 | 예 | 28.58 NORMAL | HEALTHY |
| 7000 | APNEA-proxy | 0.973 | 0.953 | 예 | 28.60 NORMAL | HEALTHY |

판별 가능한 윈도우 5개 중 **5개 전부 APNEA-proxy**, 최고 신뢰도 0.996.

그런데 이 캡처는 전 구간 `respiration_valid=true`이고 호흡수가 4–27 rpm으로 관측된다.
즉 호흡이 관측되는 구간에 대해 모델이 **고신뢰 무호흡 오탐**을 내고 있다.
"판별력 없음"보다 안전상 더 위험한 양상이다. 무비판적으로 융합하면
`APNEA-proxy → score 0.9 → 가중치 0.25`가 상시 켜진 채로 운영된다.

`LATEST_SOURCE_PROVENANCE.json`의 `DEVICE_VALIDATED: "NO"`,
`PI_SMOKE: "NOT_PERFORMED"`와 정합하는 상태다.

thermal 모델은 대비적으로 정상이다 (동일 런타임 직접 호출).

| 합성 입력 | 결과 | 신뢰도 |
|---|---|---|
| 균일 프레임 | `NOT_HUMAN` | 1.000 |
| 세로 블롭(직립) | `HUMAN_NORMAL` | 1.000 |
| 가로 블롭(누움) | `HUMAN_FALL` | 1.000 |

v1 산식이 이 오탐에 대해 걸어 둔 방어 3중:

1. 판별력 게이트 — 마진 < 0.15인 윈도우(1200, 5600)는 AI 채점 자체를 거부하고
   규칙 폴백으로 내린다.
2. 지속성 — `APNEA-proxy`가 2회 연속이어야 에스컬레이션 후보가 된다.
   1회는 `APNEA_PROXY_AWAITING_PERSISTENCE`만 기록한다.
3. 등급 상한 — 미검증 `APNEA-proxy`는 **WARNING까지만**. 위 표에서
   판별 성공 + score 0.9인데도 최종 등급이 NORMAL/WARNING을 넘지 않는 이유다.
   하드웨어로 확인된 `apnea_verified=true`만 DANGER + 비상으로 간다.

해소는 재학습 / 실기 스모크 과제이며 런타임 배선 과제가 아니다. 재실 게이트는
절대 끄지 말아야 한다 — 게이트가 없으면 빈 방의 zero 입력이 이 경로로 APNEA를 낸다.

## 6-1. 즉시 사용 가능한 대안 — 캐노니컬 윈도우 스펙트럼 판독 (오늘 적용)

재학습은 라벨이 없어 오늘 불가능하다(§6 하단). 그래서 **재학습 없이 오늘 쓸 수 있는**
결정론적 DSP 판독을 붙였다. `ai/mmwave_spectral_runtime.py`.

모델이 아니다. **M-N9가 먹는 것과 정확히 같은 동결된 `[1,240,1]` 캐노니컬 윈도우**를
읽는다. 새 센서 필드도, 새 전처리 계약도, 새 잠금 아티팩트도 생기지 않는다.

### 산출물 2개

**`rate_rpm`** — 호흡대역 최대 피크에서 구한 호흡수. 240샘플/8 Hz의 원 빈 간격은 2.0 rpm이고,
로그 파워 포물선 보간으로 사실상 제거한다.

| 검증 | 결과 |
|---|---|
| 합성 정현파 8–30 rpm | 오차 **0.00 rpm** |
| 노이즈 sd 0.15 추가 | 오차 ±0.04 rpm |
| 비대칭 파형(빠른 날숨) | 오차 0.00 rpm |
| 2차 고조파 60·100% | 오차 ≤0.5 rpm (하위고조파 보정 적용) |
| 36개 합성 케이스 중 오차 >2.5 rpm | **0개** |

**`hold_evidence`** — APNEA 라벨 정의가 요구하는 6 s 숨참기만큼 조용한 구간이 윈도우 안에
있는지. 대역 파워만으로는 판별 불가능하다(30 s 중 22 s 호흡 + 8 s 정지도 여전히 강하게 주기적).
6 s 슬라이딩 RMS의 최솟값이 윈도우 중앙값 RMS의 45% 이하면 정지 구간이 있다고 본다.

### 고조파 함정 (실제로 밟았고 고쳤음)

캐노니컬 채널은 R2 **미분**이고 미분은 n차 고조파를 n배 증폭한다. 순수 정현파만으로
검증했을 때는 안 보였는데, 위상에 2차 고조파가 60% 섞이면 피크가 **2배 주파수로 점프**한다
(12 rpm → 24.00 rpm 보고). 피치 검출의 표준 하위고조파 검사를 넣었다: 피크의 절반
주파수가 대역 안에 있고 피크 파워의 15% 이상을 가지면 그쪽을 기본파로 택한다.

**실측 6346개 윈도우 중 54.5%가 이 보정을 받는다.** 즉 절반 이상이 보정 전에는 2배 값을
보고하고 있었다. 보정 후 평균이 20.56 → **15.37 rpm**으로 내려갔고, 안정 성인으로 훨씬
타당하다. 보정이 대역을 벗어나지 못하도록 in-band 빈만 후보로 둔다.

### MR60 자체 스칼라보다 신뢰할 수 있다

동일한 6346개 윈도우에서:

| | 평균 | 표준편차 | 최소 | 최대 |
|---|---|---|---|---|
| 스펙트럼 판독 | **15.37** | 4.98 | 5.28 | 30.19 |
| MR60 `breath_rate_raw` | 10.21 | 9.17 | **0.00** | 29.00 |

호흡대역 파워 비율 평균 0.869. MR60 스칼라는 같은 윈도우에서 0.00 rpm까지 떨어진다.
그래서 v1의 호흡 규칙 성분이 **스펙트럼을 1순위, MR60 스칼라를 최후 수단**으로 쓴다
(`respiration_rate_source`에 출처 기록).

### APNEA 반증 게이트

호흡대역에 주기성이 있고 **6 s 정지 구간이 전혀 없으면** 숨참기는 일어나지 않았다.
이때 M-N9의 APNEA-proxy는 물리적으로 성립할 수 없으므로 발행을 거부하고
`APNEA_CONTRADICTED_BY_SPECTRUM` / 상태 `RESPIRATORY_INFERENCE_REFUSED`를 남긴다.

극성이 안전 방향인지 확인했다.

| 시나리오 | hold_evidence | APNEA 반증 |
|---|---|---|
| 연속 20 rpm 호흡 | False | **반증함** |
| 20 rpm + 6·8·10·15 s 숨참기 | True | 반증 안 함 (실제 무호흡 보존) |
| 30 s 전체 정지 | — (not ready) | 반증 안 함 |
| 백색소음 | — (not ready) | 반증 안 함 |

즉 **반증은 "정지 없음"이라는 적극적 증거가 있을 때만** 발동한다. 실측 7개 윈도우 중
6개가 APNEA-proxy였고 4개가 반증으로 거부됐다. 나머지 2개는 정지 구간이 있어 거부하지 않는다.

### `neural_trust: OBSERVE_ONLY` (오늘 기본값)

`risk_formula_v1.json`의 `mmwave.neural_trust`를 추가했다. 기본 `OBSERVE_ONLY`.
M-N9의 클래스는 `observed_neural_state`로 **기록되지만 점수에 들어가지 않고**,
호흡 성분은 스펙트럼 규칙이 낸다. `DEVICE_VALIDATED`가 true가 된 뒤 `TRUSTED`로 바꾸면 된다.

한 가지 예외를 명시적으로 뒀다 — **하드웨어로 확인된 `apnea_verified: true`는 이 스위치가
억제하지 않는다.** 그건 모델 의견이 아니라 장비 출처이기 때문이다. 초기 구현에서 이것까지
같이 막혀 있었고 자체 테스트 2건이 그 안전 퇴행을 잡아냈다.

### 적용 후 실측 (7개 윈도우 전부)

```
python hil/preconnect_runtime_audit.py --inject-presence --sweep 1200,2400,4000,4800,5600,6400,7000

 records M-N9 class           conf  margin spectral rpm   band  hold  risk src      risk level
    1200 RESPIRATORY_INFERE      -   0.059       12.304  0.758 False  SPECTRAL_C   6.054 NORMAL
    2400 RESPIRATORY_INFERE      -   0.707       12.914  0.909 False  SPECTRAL_C   6.149 NORMAL
    4000 RESPIRATORY_INFERE      -   0.996       23.715  0.832 False  SPECTRAL_C   6.138 NORMAL
    4800 APNEA-proxy         0.824   0.695       15.000  0.917  True  SPECTRAL_C   6.054 NORMAL
    5600 NORMAL              0.492   0.070       20.074  0.805 False  SPECTRAL_C   6.085 NORMAL
    6400 APNEA-proxy         0.957   0.922       10.159  0.834  True  SPECTRAL_C   6.075 NORMAL
    7000 RESPIRATORY_INFERE      -   0.953       14.976  0.818 False  SPECTRAL_C   6.096 NORMAL

  spectral estimate available : 7 / 7      APNEA-proxy refused by spectrum : 4
```

7개 전 윈도우가 `SPECTRAL_CANONICAL_WINDOW`를 쓰고, 호흡수 10.2–23.7 rpm 전부 v1 정상
대역(10–24) 안이며, 위험도가 6.05–6.15로 안정된다. 적용 전에는 APNEA-proxy가 통과한
윈도우에서 28.5까지 튀었다(WARNING 경계 30 바로 아래).

낙상 비상 경로는 그대로다: `--thermal-shape lying` → score 100.0 / DANGER /
`is_emergency True` / `FLOOR_THERMAL_FALL_CONFIDENT` → SQLite 저장.

### 한계 (명시)

- 이건 호흡수 추정과 무호흡 **반증**이다. 무호흡 **검출**이 아니다. 그건 여전히 M-N9의
  일이고 M-N9는 아직 신뢰할 수 없다.
- 30 s 창에서 6 rpm은 3주기뿐이라 그 이하는 분해되지 않는다. 대역은 6–36 rpm이다.
- `hold_evidence`가 True인 윈도우에서는 반증하지 않으므로 M-N9의 오탐이 통과할 수 있다.
  현재 v1의 `OBSERVE_ONLY`가 그걸 점수에서 막고 있다.
- `system_health`는 계속 `DEGRADED`다. 검증된 호흡 모델 없이 운영 중이라는 뜻이고
  정직한 표시다.

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
  상승률은 **C-B6 캐노니컬 슬로프를 우선 사용**한다
  (`ai.metadata.co2_slope_ppm_per_min`, `slope_source`에 출처 기록). 런타임에 슬로프
  정의가 둘 존재하지 않도록 하기 위함이고, 캐노니컬 슬로프가 warm-up이면
  `RISK_LOCAL_ENDPOINT` 폴백으로만 내려간다.
  C-B6 점유 출력은 `risk_semantic: NONE`이므로 위험 가중치가 되지 않고
  `occupancy_state` / `occupancy_probability` 메타데이터로만 노출된다.
- **pir**: 움직임 0.0. **presence 미확인 시 0.0이 아니라 불가용** — 0.0을 주면 총점을
  조용히 끌어내린다. presence 확인 + 무움직임은 유예 30 s부터 위험 180 s까지 선형 상승.
  (구 V4는 15 s에 곧바로 1.0으로, 가만히 앉아있는 사람에게도 최대 점수를 줬다.)

### 7.4 부수 사항

- 구 `risk/engine.py`의 비상 오버라이드는 `mmwave.state == "APNEA"`를 검사하는데
  파이프라인은 `"APNEA-proxy"`를 발행한다. **구 산식의 무호흡 오버라이드는 사문화 상태**였다.
  v1은 두 문자열을 모두 처리한다.
- `sources/ondevice_ai/config/risk_rules.json`과 `risk_rules.yaml`은 어떤 코드도
  읽지 않는다(문서·해시감사 전용). v1은 이들을 대체하지 않고 무시한다.
- `sources/ondevice_ai/config/models.yaml`도 런타임이 로드하지 않는다. CO₂ 항목은
  아직 3-특성 v0.1.0을 기술하고 있으나 실제 선택은 `model_manifest.json`이 한다.
  혼동을 피하려면 별도로 정리가 필요하다.
- `INDETERMINATE`는 `backend/views.py:114-127`의 등급 분기에 없어 표시 상태가
  `offline`로 떨어진다. fail-closed이므로 안전하지만, 대시보드 문구 보강이 필요하다.

---

## 8. 실측 재생 최종 결과

`python hil/preconnect_runtime_audit.py --inject-presence --limit 4000`

```
[SOURCE]     data/mmwave/20260817_09_mmwave.jsonl + data/co2/20260817_09_co2.jsonl
             thermal은 SYNTHETIC_UPRIGHT (실측 캡처 미커밋)

[Q1 INGEST]  mmwave/thermal/co2/pir 전부 LIVE,  system = ONLINE
             mmwave CANONICAL_WINDOW_READY (연속 31180 ms / 채택 266 / MAD 0.0307)
             co2    CO2_SLOPE_READY

[Q2 COMPUTE] thermal  tflite  HUMAN_NORMAL   conf 1.000   0.099 ms  [0.0, 1.0, 0.0]
             mmwave   tflite  APNEA-proxy    conf 0.996   0.060 ms  [0.0, 0.0, 0.996]
             co2      tflite  OCCUPIED       conf 0.996   0.046 ms  [0.004, 0.996]
             pir      rule    NO_MOTION
             all_models_available = True     degraded = False

[Q3 RISK]    SAFENEST_RISK_V1  score 28.638  level NORMAL
             system_health HEALTHY   effective_weight 1.0
             component_status {mmwave: AI, co2: RULE, pir: RULE, thermal: AI}
             reasons [PRESENCE_FROM_MMWAVE, APNEA_PROXY_AWAITING_PERSISTENCE,
                      HIGH_CO2_WARNING]

[Q3 PERSIST] publish -> SQLite  score 28.638 / level NORMAL / health HEALTHY

[VERDICT]    7 / 7 PASS
  Q1 wire decode + state LIVE      PASS   LIVE=[co2, mmwave, pir, thermal] of 4
  Q1 mmwave canonical window       PASS   CANONICAL_WINDOW_READY, span 31180 ms
  Q2 real TFLite inference         PASS   tflite=[co2, mmwave, thermal] of 3
  Q3 risk score published          PASS   score 28.638, level NORMAL
  Q3 all components contribute     PASS   no UNAVAILABLE component
  Q3 no degraded mode              PASS   health HEALTHY
  Q3 risk persisted to SQLite      PASS   rows 1, level NORMAL
```

`component_status.co2`가 `RULE`인 것은 정상이다. C-B6 점유는
`risk_semantic: NONE`이라 위험 성분이 되지 않고, CO₂ 위험 점수는 ppm 규칙이 낸다.

`--thermal-shape lying` (낙상 형상) 추가 시 비상 경로까지 관통한다.

```
thermal  tflite  HUMAN_FALL  conf 1.000  probabilities [0.0, 0.0, 1.0]
SAFENEST_RISK_V1  score 100.0  level DANGER  is_emergency True
  score_level WARNING (level_source=EMERGENCY)
  escalation_floors ['thermal_fall_confident']
  reasons [EMERGENCY_HUMAN_FALL, ..., FLOOR_THERMAL_FALL_CONFIDENT]
SQLite 저장: score 100.0 / level DANGER
```

기본 `--limit`은 2400이다. 커밋된 20260817 캡처에서 M-N4의 30 s 윈도우와
C-B6의 150 s CO₂ 이력을 동시에 채우는 최소 지점이기 때문이다. 그보다 작으면
CO₂가 `FEATURE_UNAVAILABLE_GAP_RESTART`로 남고, 3200 부근은 캡처의 221.5 s 단절
(레코드 인덱스 807)과 겹쳐 mmWave가 `WINDOW_CONTAINS_LARGE_GAP`이 된다. 둘 다
계약대로 거부한 결과이고 결함이 아니다.

## 9. 남은 작업 (센서 연결 전)

| # | 항목 | 담당 영역 | 상태 |
|---|---|---|---|
| 1 | 위상 윈도우 와이어 레이트 누적 | Pi 런타임 | **완료** (§2) |
| 2 | CO₂ C-B6 어댑터 + 캐노니컬 슬로프 + 선택자 승격 | 온디바이스 AI / Pi 런타임 | **완료** (§5) |
| 3 | 위험도 산식 v1 | Pi 런타임 | **완료** (§7) |
| 4 | `human_detected_raw` 펌웨어 추가 | ESP32 `.ino` | **완료** (§4-1). 단 감사 도구의 `--inject-presence` 제거는 실기 재캡처가 선행 조건 |
| 5 | mmWave 호흡 신호 (스펙트럼 판독) | 온디바이스 AI / Pi 런타임 | **완료 — 오늘 적용** (§6-1) |
| 5b | M-N9 오탐 해소 | 모델 재학습 / 실기 스모크 | **미착수** (§6). 선행 조건: MR60 + 독립 기준(MOVESENSE_CHEST_ACC) 동시 수집 라벨 |
| 6 | 실측 thermal 캡처 커밋 | 데이터 | 미착수 (`data/thermal/` 비어 있음, 합성 프레임으로 대체 중) |
| 6b | **`human_detected_raw` 포함 mmWave 재캡처** | 데이터 / 실기 | 미착수. 커밋된 캡처는 전부 firmware `1.2.0` 스탬프로 이 필드가 없다. 감사 도구를 `--inject-presence` 없이 통과시키려면 `>=1.3.0` 펌웨어로 다시 떠야 한다 (§4-1) |
| 7 | C-B6 SCD40 도메인 정렬 (C-C) + 임계 재선정 | 온디바이스 AI | 미착수. 현 임계 0.43은 `TRAIN_INTERNAL_ONLY`, 캘리브레이션은 UCI 도메인 |
| 8 | 대시보드 O4 요소 복구 | 웹 | 미착수 — `runtimeBadge`, `thermalSensor`, `thermalAiStatus`, `co2Ai`, `pirAi`가 새 `web/dashboard/`에 없어 기계판독 계약이 깨짐 |
| 9 | 비상 시 SMS/119 자동 연동 | 서비스 | 설계상 수동. `is_emergency`는 부저 래치와 이벤트 로그까지만 자동 |
| 10 | Stage 9 파이 스모크 / 30분 soak → `DEVICE_VALIDATED=true` | 실기 | 미착수. mmWave·CO₂ 모두 `DEVICE_VALIDATED: NO` 유지 |
| 11 | 팀 저장소 `RaspberryPi/Runtime` 이식 + 파이 재배포 | 배포 | 미착수. `/api/status`에서 `canonical_window_status=CANONICAL_WINDOW_READY`, `component_status.mmwave=AI` 확인 필요 |

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

### 수정 (15)

| 파일 | 변경 내용 |
|---|---|
| `ai/pipeline.py` | `OnDeviceAIPipeline.observe_telemetry(packet)` 신설 — 수신 스레드가 패킷마다 MR60 위상을 누적기에 넣는다. `_mmwave_wire_observed` 플래그로 스냅샷 기반 호출자(오프라인 재생·단위 테스트)와의 이중 누적을 차단. `TelemetryPayload` import 추가. |
| `ai/mmwave_canonical_runtime.py` | `MR60CanonicalWindowBuilder`의 누적 상태를 `threading.RLock`으로 보호 (수신 스레드가 쓰고 발행 스레드가 읽음). `ingest`/`latest`를 락 래퍼 + `_ingest_locked`/`_latest_locked`로 분리. 계약 로직 자체는 무변경. |
| `backend/runtime.py` | `_on_packet`에서 `TelemetryPayload`일 때 `ai_pipeline.observe_telemetry(packet)` 호출, 예외는 `mmwave_phase_window` 런타임 에러로 기록. 기본 위험도 엔진을 `SafeNestRiskEngine` → `SafeNestRiskFormulaV1`로 교체. `risk_engine` 파라미터 타입을 주입 가능한 `object | None`으로 완화. |
| `gateway/run_risk_gateway.py` | 위상 와이어 레이트 누적 배선 + 위험도 엔진 v1 적용. |
| `gateway/run_ai_gateway.py` | 위상 와이어 레이트 누적 배선. |
| `ai/runtime.py` | `_ADAPTERS`를 3-튜플 `(파일, 클래스, 매니페스트 선택자 키)`로 확장하고 `_assert_deployment_allowed`를 선택자 키로 조회. CO₂ 어댑터를 `co2_c_b6_interpreter.py::CB6Interpreter` / 선택자 `co2_occupancy_c_b6`으로 교체. |
| `sources/ondevice_ai/models/model_manifest.json` | `co2_occupancy_c_b6` 항목 추가(`ACTIVE_C_B6`, `[1,2]` int8, `feature_order ["CO2","CO2_slope"]`, `risk_semantic NONE`). 구 `co2` 항목에 `HISTORICAL_CO2_V0_1_0` / `HISTORICAL_NOT_ACTIVE` / `superseded_by` 표기 추가(플래그 미변경). |
| `hil/rp_x0_b_complete_provisioning_manifest.json` | CO₂ 단독 승격 기록. `production_selection_changed: true` + `production_selection_change_scope: CO2_ONLY_C_B6_REDUCED_FEATURE`, CO₂ 항목 `runtime_adapter_compatible: true`. mmWave 게이트·thermal 플래그 미변경. |
| `sources/.../rp_x0_b_complete/artifact_inventory.json` | C-B6 CO₂ 분류에 `PRODUCTION_SELECTED, ACTIVE_C_B6, OCCUPANCY_ONLY_RISK_SEMANTIC_NONE` 추가, 구 v0.1.0에 `SUPERSEDED_BY_C_B6_REDUCED_FEATURE`. |
| `LATEST_SOURCE_PROVENANCE.json` | `tracked_file_count` 1075→1076, `snapshot_additions`, `co2_c_b6_promotion` 블록, 오버레이 `runtime_role: HISTORICAL_B_STAGE_EXCEPT_CO2_C_B6`. |
| `risk/formula_v1.py` | `_co2_component`가 C-B6 캐노니컬 슬로프를 우선 사용(`slope_source` 기록)하고, 점유는 `risk_semantic NONE`을 지켜 메타데이터로만 노출. |
| `tests/test_ai_pipeline.py` | `test_co2_requires_humidity_and_history`를 C-B6 계약 테스트 2건으로 교체(150 s 이력·90 s 갭 재시작·2-인자 호출·점유 비가중). `_ADAPTERS` 3-튜플, 프리즈 파일 수 1076으로 갱신. |
| `tests/test_locked_b_stage_artifacts.py` | 매니페스트 SHA 핀, CO₂ 선택자 승격 단정, 프리즈 파일 수, CO₂ 단독 승격 범위 단정 갱신. |
| `tests/test_hil_criteria.py` | 프리플라이트 모델 해시 검사 수 5→6. |
| `tests/test_mmwave_mn9_runtime.py` | `WireRatePhaseAccumulationTests` 추가 (2건). ① 런타임 수신 경로만으로 260 패킷을 넣으면 추가 발행 없이 `CANONICAL_WINDOW_READY`가 되고 텐서가 `(1,240,1)`이며 presence 부재로 추론은 게이트된다. ② 네거티브 — 상태 매니저에 260 패킷을 넣고 발행을 1회만 하면 `accepted_update_count == 1`이다(회귀 방지). |

### 신규 (10)

| 파일 | 역할 |
|---|---|
| `risk/formula_v1.py` | `SafeNestRiskFormulaV1`. 가중 융합 + 에스컬레이션 플로어 + 증거 충분성 게이트 + AI 판별력 게이트. 출력 `RiskEvaluationV1`은 구 문서의 상위집합(`formula_id`, `score_level`, `level_source`, `effective_weight`, `evidence_sufficient`, `escalation_floors` 추가)이라 `backend/store.py`·`backend/views.py`·`database/repository.py`가 무수정 동작. `RiskComponent`는 `risk/engine.py`에서 재사용. |
| `risk/risk_formula_v1.json` | v1 설정. 가중치·임계·CO₂ 곡선·판별력 기준·플로어·비상 오버라이드. 프리즈된 `sources/ondevice_ai/` 트리 밖, 이 저장소 소유. 각 값의 채택 근거를 `rationale`에 기록. |
| `tests/test_risk_formula_v1.py` | v1 행위 계약 17건. 설정 계약, 판별력 게이트(균등분포 거부·저신뢰 거부·TTL 초과 거부), 플로어(낙상 비상·CO₂ 희석 방지·즉시위험·미검증 APNEA는 WARNING 상한·하드웨어 확인 APNEA는 비상), 증거 충분성(가중치 소수 → `INDETERMINATE`·전부 불가용 → fail-closed·과반 → NORMAL 허용), PIR 의미론, CO₂ 곡선 단조성. |
| `hil/preconnect_runtime_audit.py` | 센서 연결 전 감사 도구. 실측 캡처를 실제 `safenest.telemetry.v1` TCP 프레임으로 재생해 루프백 소켓으로 살아있는 `SafeNestRuntime`에 주입하고, `state → AI → risk → store → SQLite` 전 구간을 관통시켜 Q1/Q2/Q3 판정표를 낸다. 스텁 모델 없음. thermal만 합성이며 `SYNTHETIC_*`로 명시. |
| `sources/ondevice_ai/inference/co2_c_b6_interpreter.py` | `CB6Interpreter`. C-B6 축소 특성 CO₂ 점유 어댑터. SHA-256·텐서 `[1,2]` int8·양자화·스케일러 지문·특성 순서·임계·클래스 맵을 전부 검증하고, `humidity_included`가 false가 아니거나 `risk_semantic`이 NONE이 아니면 로드를 거부한다. `predict(ppm, slope)` 2-인자. |
| `ai/mmwave_spectral_runtime.py` | `estimate_respiration`. 동결된 캐노니컬 윈도우의 결정론적 스펙트럼 판독 — 호흡대역 피크 + 로그파워 포물선 보간(합성 오차 0.00 rpm), 미분이 만드는 2차 고조파 함정을 막는 하위고조파 보정(실측 54.5%에 적용), 6 s 숨참기 탐색(`hold_evidence`). 모델이 아니고 새 잠금 아티팩트가 없다. |
| `tests/test_mmwave_spectral_runtime.py` | 15건 + 서브테스트 31건. 정확도(정현파·노이즈·비대칭·2차 고조파 60/100%), 대역 이탈 금지, 소음·평탄·길이오류 거부, **APNEA 반증 극성**(연속 호흡만 반증, 6/8/10/15 s 숨참기는 전부 보존), 실측 캡처 통계 및 MR60 스칼라 대비 안정성. |
| `ai/co2_canonical_runtime.py` | `CO2SlopeWindowBuilder`. `CO2_SLOPE_FEATURE_PROFILE_001` 구현 — ppm/min, ENDPOINT_DIFFERENCE, age ≥ 150 s 과거 endpoint, SOURCE_ACQUISITION_CLOCK(`measurement_monotonic_ms`), 측정 이벤트 기준 전진, 90 s 갭·boot 경계·비단조 시계 재시작, 보간 금지, float64. 미확보를 0.0으로 보고하지 않고 프로파일 상태 코드로 노출. |
| `tests/test_co2_c_b6_runtime.py` | 슬로프 계약 9건 + 어댑터 계약 5건. 단위·부호·warm-up·갭 재시작과 복구·재발행 무시·무효 이벤트·boot 경계·비단조 시계, 어댑터 식별·2-인자 강제(3-인자는 `TypeError`)·ppm 단조성·빈 방 VACANT·비유한 fail-closed. |
| `docs/20260821_Preconnect_Runtime_Audit_And_Risk_Formula_V1_KO.md` | 본 문서. |

무변경: `gateway/protocol.py`, `state/manager.py`, `risk/engine.py`, `backend/store.py`,
`backend/views.py`, `database/*`, ESP32 펌웨어, 그리고 mmWave·thermal의 B-stage 잠금
(`mmwave_live_b_gate: CLOSED`, mmWave `HISTORICAL_B_NOT_ACTIVE: true`,
`thermal44_deployment_validated: false`).

### 회귀 확인

`git stash` 대조로 변경 전후 실패 집합이 **문자열 단위로 동일**함을 확인했다
(22 failed / 1 skipped). 

```
before: 22 failed, 234 passed,  1 skipped
after : 22 failed, 268 passed,  1 skipped
diff(FAILED 목록) = 빈 집합
```

신규 통과 34건: 위험도 v1 17 + 위상 회귀 2 + CO₂ C-B6 14 + CO₂ 파이프라인 계약 1.

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
.venv/bin/python hil/preconnect_runtime_audit.py --inject-presence --limit 4000    # 7/7 PASS
.venv/bin/python hil/preconnect_runtime_audit.py                                   # 캡처 그대로
.venv/bin/python hil/preconnect_runtime_audit.py --inject-presence --thermal-shape lying
.venv/bin/python hil/preconnect_runtime_audit.py --risk legacy --inject-presence   # 구 V4 비교
.venv/bin/python hil/preconnect_runtime_audit.py --json /tmp/audit.json            # 증거 저장

# 모델 거동 스윕 (M-N9 오탐 재현)
.venv/bin/python hil/preconnect_runtime_audit.py --inject-presence \
    --sweep 1200,2400,4000,4800,5600,6400,7000
```

감사 도구 옵션:

| 옵션 | 의미 |
|---|---|
| `--mmwave` / `--co2` | 재생할 캡처 파일 지정 (기본: `data/` 내 최신) |
| `--limit N` | 재생 레코드 수 (기본 2400 — M-N4 30 s 윈도우와 C-B6 150 s 이력을 동시에 채우는 최소 지점) |
| `--sweep a,b,c` | 같은 캡처의 독립 윈도우별 모델 거동 표 (§6) |
| `--inject-presence` | 결손된 `mmwave.human_detected_raw=true`만 합성 (B1 격리 검증용) |
| `--inject-humidity P` | `humidity_percent` 합성 시도 — C-B6 계약이 금지 입력으로 지정했으므로 **무시됨을 보이는 용도** |
| `--thermal-shape` | `upright` / `lying` / `flat` 합성 thermal 형상 |
| `--risk` | `v1`(기본, 런타임 기본값과 동일) / `legacy`(구 V4 비교) |
| `--json PATH` | 전체 증거 문서 저장 |

`.venv/`는 이미 `.gitignore`에 포함되어 있다.
