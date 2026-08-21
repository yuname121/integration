# B1 작업 보고 — 운영 ESP32 펌웨어 `mmwave.human_detected_raw` 발행 (2026-08-21)

| 항목 | 값 |
|---|---|
| 대상 저장소 | `yuname121/integration` |
| 작업 브랜치 | `fix/mmwave-b1-firmware-presence-field` → `main` 머지 완료 |
| PR | [#29](https://github.com/yuname121/integration/pull/29) — **머지 완료** (`main` @ `c759205`) |
| 팀 저장소 PR | [jinsu1011/safenest-embedded-competition#37](https://github.com/jinsu1011/safenest-embedded-competition/pull/37) — 미머지 |
| 선행 HEAD | `1f210b2` |
| 검증 환경 | Linux x86_64 / Python 3.12.3 / `ai_edge_litert` / g++ 13 |
| 하드웨어 | **미사용** (ESP32·MR60BHA2·Raspberry Pi 모두 없음) |

관련 문서: `docs/20260821_Preconnect_Runtime_Audit_And_Risk_Formula_V1_KO.md` §4-1(본 작업), §9(남은 작업)

---

## 1. 결론

| 판정 기준 | 결과 |
|---|---|
| 1. `--inject-presence` 없이 감사 도구 전 게이트 PASS | **달성 불가 — 코드 문제 아님** (§5) |
| 2. 테스트 실패 목록이 작업 전과 동일 (22건) | **충족** — 문자열 단위 동일 |
| 3. 파이 `/api/status` 합격 기준 7개 항목 | **미확인 — 파이 도달 불가** (§6) |
| 4. 펌웨어 JSON 파싱 테스트 3건 통과 | **충족** — 6건 추가, 전부 통과 |
| 5. `DEVICE_VALIDATED` 여전히 false | **유지** — 3곳 모두 `NO` |

펌웨어 필드 추가 자체는 완료했다. 판정 기준 1과 3은 **하드웨어 없이는 원리적으로 만족할 수
없는 항목**이며, 그 이유를 §5·§6에 분리해 기록한다.

---

## 2. 변경 내용

### 2.1 대상 파일

`sources/display-test2/esp32_sensor_node/esp32_sensor_node.ino`
(`LATEST_SOURCE_PROVENANCE.json`의 `canonical_flash_source`,
`integration_policy`에서 `maintained_active_flash_source_exception`으로 수정 허용)

`NODE_FIRMWARE_VERSION` `1.2.0` → `1.3.0`, `MMWAVE_SCHEMA_VERSION` `1.2` → `1.3`.

### 2.2 와이어 계약

중첩 `mmwave` 객체에 `human_detected_raw`를 추가했다. `safenest.telemetry.v1`의 선택
필드이므로 후방 호환 확장이며, 1.3 미만 노드도 계속 수용된다.

```jsonc
"mmwave": {
  "breath_phase": -0.136825,
  "total_phase": 1.0,
  "heart_phase": 0.2,
  "breath_rate_raw": 7.0,
  "human_detected_raw": true,      // true | false | null  <-- 추가
  "phase_age_ms": 12,
  "ts_monotonic_ms": 3718,
  "seq": 42,
  "firmware_version": "safenest-esp32-sensor-node/1.3.0",
  "schema_version": "1.3"
}
```

3-상태이며 `null`은 **"최근 0x0F09 보고 없음"**이고 부재 주장이 아니다.

---

## 3. 재실 값 획득 경로 — 경로 A를 그대로 쓸 수 없었다

### 3.1 라이브러리에 게터는 있다

설치 라이브러리는 `Love4yzp/Seeed-mmWave-library` (`library.properties`의
`name=Seeed Arduino mmWave`, `includes=Seeed_Arduino_mmWave.h`)이고
`SEEED_MR60BHA2.h`에 `bool isHumanDetected();`가 존재한다. 이 머신에는 라이브러리가
설치돼 있지 않아 상위 저장소를 클론해 헤더와 `.cpp`를 직접 읽었다.

### 3.2 그런데 3-상태를 표현할 수 없다

```cpp
bool SEEED_MR60BHA2::isHumanDetected() {
  if (!_isHumanDetectionValid) return false;   // 아직 보고 없음  -> false
  _isHumanDetectionValid = false;
  return _isHumanDetected;                     // 실제 부재      -> false
}
```

"빈 방"과 "모름"이 **둘 다 `false`**이고, 읽는 순간 유효 플래그를 스스로 지운다.
반환값 하나에 두 의미가 겹쳐 있어 out-파라미터도 없다.

이걸 그대로 JSON에 넣으면 "모름"이 "사람 없음"으로 새어 나간다. 그렇다고 모든 `false`를
`null`로 취급하면 한 번 `true`가 나온 뒤 영구히 재실 상태로 고착되어, 빈 방에서 추론이
돌아가는 `NO_PERSON_INFERENCE_GATING_HAZARD` 자체를 재현한다. 어느 쪽으로도 안전하지 않다.

참고로 같은 라이브러리의 `getBreathRate(float&)` / `getHeartRate(float&)`는 값을
out-파라미터로 넘기고 반환값을 "새 값이 있었는가"로만 쓰기 때문에 이 문제가 없다.
`isHumanDetected()`만 관용구에서 벗어나 있다.

### 3.3 해결 — `handleType()` 오버라이드

`SEEED_MR60BHA2::handleType()`은 `public virtual`이고, 기반 클래스
`SeeedmmWave::processFrame()`이 **헤더 체크섬과 데이터 체크섬을 모두 검증한 뒤에**
호출한다(`SeeedmmWave.cpp:249-272`). 따라서 서브클래스에서 0x0F09만 가로채면
UART 프레이밍·체크섬을 재구현하지 않고도(= 경로 B의 비용 없이) 모호성 없는 3-상태를 얻는다.

```cpp
class SafeNestMR60BHA2 : public SEEED_MR60BHA2 {
 public:
  bool handleType(uint16_t type, const uint8_t *data, size_t dataLength) override {
    if (type == static_cast<uint16_t>(TypeHeartBreath::ReportHumanDetection)) {
      if (dataLength < 1) return false;   // 벤더 핸들러는 data[0]을 무검사로 읽는다
      presenceRaw_ = data[0] != 0;
      presencePending_ = true;
    }
    return SEEED_MR60BHA2::handleType(type, data, dataLength);  // 라이브러리 상태 보존
  }

  // getBreathRate와 같은 out-파라미터 관용구.
  // 반환값은 "새 보고가 파싱됐는가"이며 "사람이 없다"가 아니다.
  bool takePresence(bool &value);
};
```

부수 효과로 벤더 핸들러의 무검사 `data[0]` 접근(길이 0 프레임에서 OOB read)도 막힌다.

### 3.4 신선도 판정

`PRESENCE_MAX_AGE_MS = 5000`. 판정은 **발행 시점**에 한다. `pollMmWave()`는
`if (!mmWave.update(0)) return;`로 조기 반환하므로, 그 안에서 staleness를 보면
프레임이 아예 끊겼을 때 검사가 영구히 실행되지 않는다.

```cpp
snapshot.humanDetectedKnown = isFresh(lastPresenceMs, now, PRESENCE_MAX_AGE_MS);
```

`isFresh()`가 `timestamp != 0`도 검사하므로 **한 번도 보고가 없던 노드는 `false`가 아니라
`null`을 발행한다.** 레이더가 죽으면 5초 뒤 `null`로 떨어져 추론이 억제된다(부재 주장 아님).

### 3.5 하지 않은 것

- 재실 게이트 우회·비활성화 없음. `PRESENCE_GATE_REQUIRED: YES` 유지.
- 호흡수 등 다른 신호로 재실을 추론하지 않음. occupancy 임계값 도입 없음.
  MR60이 주는 불리언만 그대로 쓴다.
- 참조 펌웨어(`sources/devices/mmwave/firmware/src/main.cpp:190`)의
  `updateStablePresence()` 다수결 안정화 **미도입** — 와이어 계약은 raw 불리언만 요구한다.
- `sources/ondevice_ai/` 미변경.
- B-stage 잠금 미변경.

---

## 4. 함께 발견·수정 — 이 파일은 2026-08-17부터 컴파일이 안 되고 있었다

`formatNullableFloat`가 **3곳에서 호출되는데 정의가 없다.**

```
712:  formatNullableFloat(respiration, ...)
714:  formatNullableFloat(heart, ...)
738:  formatNullableFloat(breathRateRawText, ...)
```

원인은 커밋 `177db97`이다. 위상 정밀도를 `%.2f` → `%.6f`로 바꾸면서
`formatNullableFloat` → `formatNullablePhase`로 **이름만 바꾸고 호출 지점 3곳을 남겨
두었다.** 즉 `canonical_flash_source`가 미정의 함수를 호출하는 상태였고, B1의 산출물인
"플래시 가능한 펌웨어" 자체가 성립하지 않았다.

정의는 프리즈 스냅샷(`sources/ondevice_ai/integrated_node/esp32_sensor_node.ino:518`)에만
남아 있었다.

**조치:** `%.2f` 헬퍼를 복원했다. 호출부를 `formatNullablePhase`로 돌리는 쪽이 아니라
헬퍼를 되살린 이유는, 전자는 호흡·심박 정밀도를 `%.2f` → `%.6f`로 조용히 바꾸고
JSON 길이를 불리기 때문이다.

---

## 5. 검증

### 5.1 펌웨어 JSON 빌더 — 호스트 컴파일

`.ino`에서 헬퍼 4개와 `sendTelemetry()`를 **텍스트로 추출해** 호스트에서 컴파일했다.
손으로 옮겨 적지 않았으므로 포맷 문자열이 실물과 어긋날 수 없다.

| 확인 | 결과 |
|---|---|
| `g++ -std=c++17 -Wall -Wextra -Wformat=2 -Werror` | 통과 → 포맷 문자열과 인자 목록 일치 |
| 최악 payload 길이 | **1109 B** < `char json[1536]` (여유 427 B) |
| 절단 방어 | `length >= sizeof(json)` → `return false` 유지 |

모든 수치 필드를 포화시키고(`UINT32_MAX`, 음수 6자리 소수) presence 3-상태 × 위상 신선도
2가지를 조합해 측정했다.

### 5.2 컴파일된 실물 빌더의 출력을 런타임에 통과

호스트에서 컴파일한 **실제 빌더가 낸 바이트**를 `decode_telemetry` →
`SensorStateManager`에 넣었다.

| 펌웨어 출력 | `packet.human_detected_raw` | `presence_available` | `presence` |
|---|---|---|---|
| `true` | `True` | `True` | `True` |
| `false` | `False` | `True` | `False` |
| `null` | `None` | `False` | `None` |

`false`는 "레이더가 보고했고, 아무도 없다"이므로 `presence_available`이 `True`인 것이 맞다.

### 5.3 추가 테스트 6건

`tests/test_mmwave_mn9_runtime.py::Firmware13PresenceFieldTests` — 전부 통과.

1. 중첩 `true` → 승격되고 `presence_available` `True`
2. 중첩 `false` → 확정된 빈 방(`unknown`이 아님)
3. 중첩 `null` → `unknown` 유지, `false`로 변질되지 않음
4. 필드 자체 없음(레거시 1.2.0 패킷) → 여전히 디코딩 성공
5. `true`면 준비된 캐노니컬 윈도우가 모델에 도달 (`model.calls == 1`)
6. `null`이면 **같은 윈도우**가 억제 (`PRESENCE_STATE_UNAVAILABLE`, `model.calls == []`)

5·6은 presence만 다른 대조쌍이다. 게이트가 실제로 presence 때문에 열리고 닫히는지를
확인하기 위한 것이다.

### 5.4 회귀

```
작업 전: 22 failed, 287 passed, 1 skipped, 86 subtests passed
작업 후: 22 failed, 293 passed, 1 skipped, 86 subtests passed   (287 + 신규 6)
diff /tmp/before.txt /tmp/after.txt  ->  차이 없음
```

실패 22건은 문자열 단위로 동일하다. 전부 PR #26의 웹/포털 백엔드 도입 이후 갱신되지 않은
기존 실패이며 본 작업과 무관하다.

### 5.5 잠금 상태

| 플래그 | 값 |
|---|---|
| `mmwave_live_b_gate` | `CLOSED` |
| `thermal44_deployment_validated` | `false` |
| `HISTORICAL_B_NOT_ACTIVE` | `true` |
| `PRESENCE_GATE_REQUIRED` | `YES` |
| mmWave / CO₂ / firmware `DEVICE_VALIDATED` | 전부 `NO` |
| `PI_SMOKE` | `NOT_PERFORMED` |

`neural_trust`는 `OBSERVE_ONLY` 그대로 두었다.

---

## 6. 판정 기준 1이 달성 불가인 이유 (코드 문제가 아니다)

감사 도구 실행 결과:

```
--inject-presence 있음 -> 9개 게이트 전부 PASS
--inject-presence 없음 -> Q2 mmwave FAIL (PRESENCE_STATE_UNAVAILABLE)
```

원인은 펌웨어가 아니라 **캡처**다.

```
data/mmwave/20260817_09_mmwave.jsonl -> mmwave.firmware_version = "safenest-esp32-sensor-node/1.2.0"
                                        mmwave.schema_version   = "1.2"
grep -c human_detected_raw data/mmwave/*.jsonl -> 전 파일 0
```

커밋된 mmWave 캡처 7개는 전부 `1.2.0` 스탬프이고 이 필드가 0건이다. 펌웨어를 고쳐도
2026-08-17에 뜬 파일이 소급해서 필드를 갖지는 않는다. 따라서

> `--inject-presence` 없이 전 게이트 PASS는 **코드 변경으로 달성 불가**이며,
> `>=1.3.0` 펌웨어로 mmWave를 **재캡처**해야 한다.

이 항목을 `docs/...KO.md` §9에 **6b**로 신설했다. 감사 도구의 `--inject-presence`
도움말도 "캡처를 보정하는 것이며 펌웨어를 보정하는 것이 아니다"로 고쳤다.

스키마를 `1.3`으로 올린 실질적 이유가 이것이다. 이제 캡처 파일만 보고 그 캡처가 재실
게이트를 만족시킬 수 있는지 기계적으로 판정할 수 있다 — B1이 오래 눈에 띄지 않은 원인이
정확히 이 구분이 불가능했다는 점이었다.

---

## 7. 미확인 항목

### 7.1 파이 배포 및 `/api/status` 확인 (판정 기준 3)

**수행하지 못했다.** 파이에 도달할 수 없다.

- `sandi` / `raspberrypi` 이름 해석 실패
- `~/.ssh/config`에 해당 항목 없음
- `known_hosts`는 해시 처리되어 호스트명 확인 불가
- 인터넷 자체는 정상(`8.8.8.8` 응답), 로컬 인터페이스는 `172.21.20.184/22`

`/22`는 1024 호스트 규모이고 대학 서브넷으로 보여 **무허가 포트 스캔은 하지 않았다.**
파이 주소와 접속 방법이 확보되면 배포·확인까지 진행 가능하다.

따라서 합격 기준 표의 7개 항목(`canonical_window_status`, `presence_available`,
`spectral_status`, `respiration_rate_source`, `component_status.mmwave`,
`risk_score`/`risk_level`, `system_health`) 전부 미확인이다.

### 7.2 `RaspberryPi/Runtime` 포팅

**하지 않았다.** `integration` main과 본 브랜치 차이가 24파일 / 약 4,400줄이고,
합격 기준 자체가 라이브 `/api/status` 확인이다. 검증 없이 팀 저장소에 올리는 것은
적절하지 않다고 판단했다.

ESP32 스케치만 포팅했다(`ESP32/Arduino/esp32_sensor_node/esp32_sensor_node.ino`,
`paths.py::ESP32_SKETCH`가 가리키는 경로). 단일 파일이고, 변경 전 `integration` 사본과
**바이트 단위로 동일**했으며, 동일 방식으로 검증 가능했기 때문이다. 팀 저장소 사본도
§4의 컴파일 불가 상태를 그대로 갖고 있었다.

### 7.3 ESP32 툴체인 빌드

**하지 않았다.** 이 머신에 ESP32 코어와 Seeed 라이브러리가 설치돼 있지 않다
(`~/.arduino15/libraries`에 없음). 검증한 것은 JSON 빌더의 호스트 컴파일까지이며,
`.ino` **전체**가 Arduino 툴체인에서 컴파일되는지는 확인하지 못했다.

특히 `SafeNestMR60BHA2` 서브클래스가 **설치된** 라이브러리 버전과 맞는지는 컴파일해야
확정된다. 상위 저장소 헤더에서 `handleType`이 `public virtual`인 것은 확인했으나,
현장 설치 버전이 다르면 깨질 수 있다.

**플래시 전에 `arduino-cli compile` 1회 권장.**

### 7.4 `PRESENCE_MAX_AGE_MS` 근거

MR60의 0x0F09 실제 보고 주기를 **측정하지 못했다.** 5000 ms는 같은 파일의 기존 mmWave
staleness 정책(`MMWAVE_STALE_MS = 5000`)과 맞춘 값이다. 참조 펌웨어는 presence staleness에
1000 ms를 쓴다. 너무 짧으면 `null` 플래핑으로 게이트가 간헐 차단되고, 너무 길면 레이더
사망 후 오래된 `true`가 유지된다. **실기 스모크 때 확인이 필요한 유일한 튜닝 값이다.**

---

## 8. 변경 파일

### `yuname121/integration` (PR #29, 커밋 `29c8012`)

| 파일 | 내용 |
|---|---|
| `sources/display-test2/esp32_sensor_node/esp32_sensor_node.ino` | `human_detected_raw` 발행, `SafeNestMR60BHA2`, `formatNullableBool`, `formatNullableFloat` 복원, 1.3.0 |
| `tests/test_mmwave_mn9_runtime.py` | `Firmware13PresenceFieldTests` 6건 추가 |
| `hil/preconnect_runtime_audit.py` | `--inject-presence` 도움말 정정 (캡처 보정임을 명시) |
| `docs/20260821_Preconnect_Runtime_Audit_And_Risk_Formula_V1_KO.md` | §4-1 신설, §9 항목 4 완료 처리 + 6b 신설 |
| `LATEST_SOURCE_PROVENANCE.json` | `esp32_firmware_presence_field` 항목 추가 (잠금 플래그 미변경) |

### `jinsu1011/safenest-embedded-competition` (PR #37, 커밋 `7d0fc76`)

| 파일 | 내용 |
|---|---|
| `ESP32/Arduino/esp32_sensor_node/esp32_sensor_node.ino` | 위와 동일 패치 |
| `ESP32/Arduino/esp32_sensor_node/ESP32_UPDATE_CHANGELOG_KO.md` | 1.3.0 항목 추가 |

---

## 9. 머지 결과

`integration`은 스택 구조였다.

```
main <- #28 (fix/mmwave-phase-wire-rate-and-risk-v1) <- #29 (fix/mmwave-b1-firmware-presence-field)
```

번호순이 아니라 안쪽부터 머지해야 했고, 실제로 그 순서로 처리됐다.

| PR | 머지 시각 (UTC) | 머지 대상 |
|---|---|---|
| #29 | 2026-08-21 12:49:15 | `fix/mmwave-phase-wire-rate-and-risk-v1` (`a0a27ca`) |
| #28 | 2026-08-21 12:49:32 | `main` (`c759205`) |

`main`에 B1 커밋 `29c8012`가 포함된 것과, `esp32_sensor_node.ino`에
`human_detected_raw`가 존재하는 것을 확인했다.

#29 머지 시 GitHub이 head 브랜치를 자동 삭제했으므로, 본 보고서는 `main`에서 분기한
별도 브랜치(`docs/b1-mmwave-presence-report`)로 올린다.

팀 저장소 #37은 `main` 직행이며 위 두 개와 의존 관계가 없다. **2026-08-21 기준 미머지.**

---

## 10. 다음 담당자에게

우선순위 순.

1. **`arduino-cli compile`** — 플래시 전 필수. §7.3.
2. **`>=1.3.0` 펌웨어로 mmWave 재캡처** — 이게 있어야 감사 도구가
   `--inject-presence` 없이 통과한다. §6. 겸사겸사 `data/thermal/`도 비어 있다.
3. **파이 배포 + `/api/status` 7개 항목 확인** — §7.1. 기동은
   `bash ./start_runtime_lcd.sh`. `RaspberryPi/LCD/start_lcd.sh`나 `server.py`를
   다시 띄우면 포트 9000을 뺏어 런타임 수신이 죽는다.
4. **`RaspberryPi/Runtime` 포팅** — §7.2.
5. **`PRESENCE_MAX_AGE_MS` 실측 재검토** — §7.4.

`DEVICE_VALIDATED`는 별도 Stage 9 스모크 / 30분 soak 전까지 `false` 유지.

### 고치려 들지 말아야 하는 정상 동작

- `component_status.mmwave = RULE_FALLBACK` — M-N9가 `neural_trust: OBSERVE_ONLY`
- `component_status.co2 = RULE` — C-B6 점유는 `risk_semantic: NONE`
- `system_health = DEGRADED` — 검증된 호흡 모델 없이 운영 중이라는 정직한 표시
- `mmwave.ai.error = APNEA_CONTRADICTED_BY_SPECTRUM` — 물리적으로 불가능한 무호흡 주장 거부
- 기동 직후 `FEATURE_UNAVAILABLE_WARMUP`(CO₂ 약 2.5–3분),
  `RESPIRATORY_WINDOW_WARMING_UP`(mmWave 30초)
