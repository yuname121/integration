# SafeNest mmWave 라이브 현장 상태 보고서

- 작성일: 2026-08-22
- 작성 저장소: `yuname121/integration` (`origin/main` @ `c759205`)
- 대상: on-device AI / runtime 배선, 현장 운영
- 성격: **관측·증거 정리**. 재학습, 펌웨어 재작성, `DEVICE_VALIDATED=true` 선언이 아님
- 관련 선행 문서:
  - `docs/mmwave/20260818_SafeNest_mmWave_M-N9_Runtime_Wiring_Handoff_KO_01.md`
  - `docs/20260821_Preconnect_Runtime_Audit_And_Risk_Formula_V1_KO.md`
  - `snapshots/20260820_pi_live/README.md`

이 문서는 2026-08-16부터 2026-08-22까지 쌓인 캡처·스냅샷·머지된 런타임 배선과, 2026-08-22 01:32 KST 전후 파이 라이브 관측을 한 곳에 묶는다. 현장에서 반복되는 `WINDOW_UNAVAILABLE`과 `APNEA-proxy`가 **같은 원인이 아님**을 증거로 고정하는 것이 목적이다.

---

## 0. 한 줄 결론

파이에서 mmWave 위상·재실은 이미 들어오고 M-N9 invoke도 된다. 그런데 **30초 창이 자주 large-gap으로 깨지고**, 창이 열리는 순간 모델은 **정상 호흡 구간에도 고신뢰 `APNEA-proxy`를 낸다.** 위험엔진은 그 클래스를 무호흡 경보로 쓰지 않는다. 거리 재조정으로 사라질 문제가 아니다.

```text
LIVE + human_detected_raw=true + phase 삼총사
        │
        ├─ 창 미완성 / 중간 끊김  →  WINDOW_UNAVAILABLE
        │                           reason = WINDOW_CONTAINS_LARGE_GAP
        │
        └─ 창 완성                 →  M-N9 = APNEA-proxy (conf ≈ 0.996)
                                    apnea_verified = false
                                    neural_trust = OBSERVE_ONLY
                                    component_status = RULE_FALLBACK
```

---

## 1. 범위와 비범위

### 이 보고서가 다루는 것

- 커밋된 mmWave JSONL (`data/mmwave/20260816_*`, `20260817_*`)
- 2026-08-20 파이 API 스냅샷 (`snapshots/20260820_pi_live/`)
- 2026-08-21 센서 연결 전 런타임 검수 (PR #28 문서)
- 2026-08-22 라이브 파이 읽기 전용 점검과 정지 1인 테스트 해석
- integration에 이미 머지된 mmWave 배선 (PR #22, #25, #27, #28, #29)

### 이 보고서가 하지 않는 것

- ESP 펌웨어 전체 재작성
- 재실 게이트 해제 (`PRESENCE_GATE_REQUIRED`는 유지)
- 호흡수로 occupancy를 만들어 내는 일
- M-N4 계약 변경 (`[1, 240, 1]`, 30 s, 8 Hz)
- `DEVICE_VALIDATED=true` 또는 Stage 9 soak 선언
- 임상 무호흡 진단 주장
- CO₂ 습도 경로, LCD `server.py` / `:9000` 부활
- 팀 저장소 `jinsu1011/safenest-embedded-competition` 직접 수정

---

## 2. 현재 배치 지도

현장 파이와 integration git은 **같은 트리가 아니다.** 숫자를 읽을 때 이 구분을 먼저 적용해야 한다.

| 구분 | 위치 | 관측 시점 | SHA / 상태 |
|---|---|---|---|
| integration 권위 `main` | `yuname121/integration` | 2026-08-22 | `c759205` (PR #28 머지, 가계에 PR #29 포함) |
| 2026-08-20 파이 스냅샷 | `sandi@192.168.137.249:/home/sandi/safenest-runtime` | 00:29 KST | 팀 `c3f95b8`, 센서 단절, `MODEL_PENDING` |
| 2026-08-22 라이브 파이 | 같은 호스트 `/home/sandi/safenest-team-main` | ~01:32 KST | 팀 `f718430` (LCD PR #40 계열). HTTP 8000 / TCP 9000 / UDP 5005 |
| 테스트 종료 후 | 동일 | ~01:35 KST 이후 | 운영자가 ESP 전원 차단. 신규 위상 없음 |

라이브 점검 당시 파이는 integration `c759205`를 `git pull`한 상태가 아니었다. 다만 `/api/status`에 `neural_trust=OBSERVE_ONLY`와 `observed_neural_state=APNEA-proxy`가 보여, 팀 트리에도 위험도 v1의 관찰 전용 경로 일부가 들어가 있었다. **스펙트럼 반증으로 `APNEA-proxy` 발행 자체를 거절했는지**는 당시 요약에 `spectral_contradicts_apnea`가 남지 않아 단정하지 않는다.

배포 권위는 팀 파이 클론이다. 이 문서의 코드 인용은 integration `main` 기준이다.

---

## 3. 타임라인

| 시각 (KST) | 사건 | 의미 |
|---|---|---|
| 2026-08-16 | `data/mmwave/20260816_{13,14,15}_mmwave.jsonl` | 호흡 스칼라만 기록. nested phase / presence 없음 |
| 2026-08-17 오전 | `20260817_06`, `_07` | `_07`부터 nested `mmwave.breath_phase` 삼총사 등장. presence 없음 |
| 2026-08-17 이후 | `20260817_08`, `_09` | 가장 큰 실측 위상 캡처. 전 구간 `human_detected_raw` 부재. `_08` 마지막 줄 1건 널 바이트 손상 |
| 2026-08-18 | PR #22 | 잠긴 M-N9 FULL_INT8을 integration 활성 선택자로 수입. Mac smoke only |
| 2026-08-19 | PR #23 닫힘 | 런타임 배선 초안. Stage 7 preflight를 되감으면 안 되어 충돌만 고친 채 머지하지 않음 |
| 2026-08-20 00:22 | 라이브 증상 보고 | 통신/LCD/저장은 동작, `risk.component_status.mmwave=RULE_FALLBACK`, `ai_error=INPUT_UNAVAILABLE` |
| 2026-08-20 00:29 | `snapshots/20260820_pi_live/` | 센서 단절 스냅샷. mmWave `MODEL_PENDING` / `MR60_NATIVE_MODEL_PENDING`, `presence_available=false` |
| 2026-08-20 | PR #27 작성 | nested ESP phase 승격 + presence 게이트 유지. 2026-08-21 10:27 UTC 머지 |
| 2026-08-21 | PR #28 | **P0:** 위상 창을 발행 주기가 아니라 수신(wire) 주기로 누적. 스펙트럼 호흡 판독 + 위험도 v1 |
| 2026-08-21 | PR #29 | 펌웨어 1.3.0이 `mmwave.human_detected_raw`를 true/false/**null** 삼상태로 발행. B1 닫힘 |
| 2026-08-22 ~01:32 | 파이 읽기 전용 점검 | 정지 1인·0.6–1.0 m 테스트 중. large-gap과 APNEA-proxy가 **동시에** 관측 |
| 2026-08-22 ~01:35 이후 | ESP 전원 차단 | 이후 `/api/status`는 단절로 떨어짐. 끄기 직전 jsonl이 그 테스트의 증거 |

---

## 4. 계약 (바뀌지 않은 것)

### 4.1 M-N4 입력

동결 모듈 `sources/ondevice_ai/scripts/mmwave_m_n4_canonical.py`.

| 항목 | 값 |
|---|---|
| 창 길이 | 30.0 s |
| 샘플 | 240 @ 8 Hz → 텐서 `[1, 240, 1]` |
| genuine update | `ts_monotonic_ms - phase_age_ms`가 직전 대비 **8 ms 초과** 전진할 때만 |
| large gap | `max(0.40 s, 4 × median_dt)`를 넘는 간격이면 `WINDOW_CONTAINS_LARGE_GAP` |
| 정규화 | 창 로컬 MAD divide-only |
| 300샘플 경로 | M-N9에 전달하지 않음 |

30초는 **“사람이 인지된 시간”이 아니다.** 파형 길이 계약이다. 창은 presence와 따로 쌓인다. `predict`는 `human_detected_raw`가 불리언 **true**일 때만 호출한다.

### 4.2 M-N9 출력

- 아티팩트: `MMWAVE_M_N9_FULL_INT8_V1`
- SHA-256: `3b008af4be0facc4037c2afd3fe39292fb794208eb4370dbe6916b2d15aa38a4`
- 클래스: `0=NORMAL`, `1=RAPID_OR_ABNORMAL`, `2=APNEA-proxy`
- `DEVICE_VALIDATED: NO`
- `PI_SMOKE` 잠금 필드: `NOT_PERFORMED` (아래 §9의 현장 점검은 이 잠금을 풀지 않음)
- `PRESENCE_GATE_REQUIRED: true`

### 4.3 APNEA-proxy 라벨 정의

`MMWAVE_LABEL_MAPPING_PROFILE_001`:

- 임상 무호흡을 주장하지 않음
- 자발적 숨참기를 대리 라벨로 사용
- 30초 창과 **6초 이상 겹치고**, 이벤트가 **8초 이상**이면 클래스 2

런타임은 항상 `apnea_verified=false`를 붙인다. 이 클래스만으로 119/emergency에 올라가지 않는다.

### 4.4 재실 필드 삼상태 (PR #29)

운영 펌웨어 `safenest-esp32-sensor-node/1.3.0`:

| 와이어 값 | 의미 | 파이프라인 |
|---|---|---|
| `true` | MR60 0x0F09가 사람 있음 | `predict` 허용 |
| `false` | 사람 없음 (unknown이 아님) | `NO_VALID_PERSON` |
| `null` | 최근 0x0F09 없음. 부재가 아님 | `PRESENCE_STATE_UNAVAILABLE` |
| 필드 자체 없음 | 구 캡처 / 구 펌웨어 | 동일하게 억제 |

라이브러리 `isHumanDetected()`는 unknown과 absent를 둘 다 `false`로 만들기 때문에 쓰지 않는다. 호흡수로 occupancy를 만들지 않는다.

---

## 5. 쌓인 데이터

### 5.1 저장소에 커밋된 JSONL

경로: `data/mmwave/`. 모두 구 펌웨어 캡처라 **`human_detected_raw` 발생 횟수 = 0**. 연결 전 검수는 `--inject-presence`가 필요하다.

| 파일 | 행 | nested phase | presence | genuine dt p50 | ≤8 ms (재발행) | >500 ms | >1.5 s | dt max | 비고 |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `20260816_13` | 2,737 | 없음 | 없음 | — | — | — | — | — | 호흡 스칼라만 |
| `20260816_14` | 1,518 | 없음 | 없음 | — | — | — | — | — | 동일 |
| `20260816_15` | 478 | 없음 | 없음 | — | — | — | — | — | 동일 |
| `20260817_06` | 276 | 없음 | 없음 | — | — | — | — | — | 스칼라 구간 |
| `20260817_07` | 10,466 | 9,736행에 삼총사 | 없음 | 126 ms | 2,914 | 290 | 10 | 7.0 s | phase null 943, 음수 dt 2,818 (리셋/시계) |
| `20260817_08` | 21,064 (+손상 1행) | 전 행 nested | 없음 | 140 ms | 3,337 | 181 | 31 | 436 s | 검수 주 캡처. 마지막 줄 `\x00` 손상 1건 |
| `20260817_09` | 7,436 | 전 행 nested | 없음 | 139 ms | 1,195 | 21 | 10 | 221 s | phase null 0 |

`20260817_08` 호흡 스칼라: `respiration_valid=true` 21,056 / `false` 8, 평균 11.14 bpm, 중앙 12, 범위 0–30, 0 bpm 3,851건. nested 키는 `breath_phase`, `ts_monotonic_ms`, `phase_age_ms`, `breath_rate_raw`, `seq`, `firmware_version`, `schema_version`. **top-level phase는 없고 nested `mmwave.*`만 있다.** 이것이 PR #27 이전 파서가 창을 못 만들던 직접 원인이다.

공통 패턴 (라이브 2026-08-21 jsonl과 같음):

- 중앙 갱신 간격 ≈ 126–140 ms → 약 **7 Hz**, 계약 8 Hz에 가깝다
- 8 ms 이하 전진이 수천 건 → 같은 위상의 재발행. 창 빌더는 의도적으로 버린다
- 500 ms가 넘는 간격이 수십~수백 건 → M-N4 large-gap 탈락의 원재료
- 음수 dt → 부팅/세션/시계 리셋. 창은 경계에서 끊는다

### 5.2 2026-08-20 파이 API 스냅샷

`snapshots/20260820_pi_live/` (`.venv`·thermal NPZ·시크릿 제외).

당시 `/api/status`:

- `system=DEGRADED`, 전 성분 `UNAVAILABLE`
- mmWave `DISCONNECTED`, `presence_available=false`, published values에 `breath_phase` 없음
- `runtime_status.mmwave.ai_status=MODEL_PENDING`, `blocked_reason=MR60_NATIVE_MODEL_PENDING`
- 위험 가중치는 구 V4 (`mmwave 0.35 / co2 0.35 / thermal 0.15`)
- 경로가 `/home/sandi/safenest-runtime` — 이틀 뒤 라이브 경로와 다름

이 스냅샷은 **“모델이 없다”가 아니라 “당시 런타임이 M-N9 경로를 아직 투영하지 못하고, 센서도 단절됐다”**는 사진이다. PR #27이 `runtime_status` 하드코딩을 걷어 낸 이유다.

### 5.3 2026-08-21 연결 전 검수 (커밋 캡처 재생)

도구: `hil/preconnect_runtime_audit.py`. 하드웨어 없이 실측 jsonl을 TCP 프레임으로 주입하고 실제 `.tflite`를 invoke.

PR #27만으로는 창이 안 만들어졌다. `MR60CanonicalWindowBuilder.ingest()`가 **발행 경로(`evaluate` 15 s)에만** 붙어 있어, 1017개 위상 중 1개만 채택되고 `INSUFFICIENT_CONTINUOUS_DURATION`이 났다. PR #28이 수신 스레드에서 패킷마다 `observe_telemetry()`를 호출하도록 고쳤다.

같은 `20260817_08` 캡처, presence만 주입한 뒤:

| 항목 | 값 |
|---|---|
| `canonical_window_status` | `CANONICAL_WINDOW_READY` |
| 연속 span | 31,160 ms |
| 채택 업데이트 | 266 |
| 입력 shape | `[1, 240, 1]` |
| MAD | ≈ 0.101 |
| mmWave 클래스 | `APNEA-proxy` (첫 창 conf 0.418, 이후 창은 0.82–0.996) |

독립 창 7개 스윕:

| 레코드 | 클래스 | 신뢰도 | 1·2위 마진 | 판별력 게이트 |
|---|---|---:|---:|---|
| 1200 | APNEA-proxy | 0.418 | 0.059 | 거부 (마진 부족) |
| 2400 | APNEA-proxy | 0.836 | 0.707 | 통과 |
| 4000 | APNEA-proxy | 0.996 | 0.996 | 통과 |
| 4800 | APNEA-proxy | 0.824 | 0.695 | 통과 |
| 5600 | NORMAL | 0.492 | 0.070 | 거부 |
| 6400 | APNEA-proxy | 0.957 | 0.922 | 통과 |
| 7000 | APNEA-proxy | 0.973 | 0.953 | 통과 |

판별 가능한 5개 **전부 APNEA-proxy**. 그런데 이 캡처는 전 구간 호흡이 관측된다 (4–27 rpm, `respiration_valid=true`). 즉 모델은 호흡 중인 파형에 고신뢰 무호흡 오탐을 낸다.

스펙트럼 반증 (`ai/mmwave_spectral_runtime.py`): 호흡대역 주기성이 있고 6초 정지 구간이 없으면 `APNEA_CONTRADICTED_BY_SPECTRUM`으로 발행 거절. 위 7개 중 APNEA 6개에서 **4개가 반증으로 거절**됐다. 나머지 2개는 창 안에 정지 구간이 있어 거절하지 않는다. 반증은 “정지 없음”의 적극적 증거가 있을 때만 발동한다.

### 5.4 2026-08-22 라이브 jsonl (파이 디스크, 커밋하지 않음)

파일: `/home/sandi/safenest-team-main/RaspberryPi/Runtime/data/mmwave/20260821_16_mmwave.jsonl`  
행 수: **15,962** (점검 시점). git에 복사하지 않음.

| 항목 | 값 |
|---|---|
| `human_detected_raw=true` | 14,600 |
| `false` | 1,360 |
| `breath_phase` null | 911 |
| genuine dt 중앙값 | 133–143 ms (약 7 Hz) |
| 재발행성 ≤8 ms | 2,344 |
| gap >500 ms | 34 |
| gap >1.5 s | 10 |
| gap max | ≈ 66 s |
| `phase_age_ms` 중앙 / p95 / max | 34 / 78 / 473 |
| 음수 dt | 있음 (부팅/시계) |

구 캡처와 다른 점: **ESP가 presence를 실제로 보낸다.** PR #29 펌웨어가 현장에 올라가 있거나, 동등한 필드가 팀 펌웨어에 있다. 구 캡처의 B1(필드 없음)은 이 세션에서 재현되지 않았다.

같은 세션에서 30초 span·241개 채택이 나와도 large-gap으로 거절된 적이 있다. 샘플 개수 ≠ 계약 충족.

---

## 6. 2026-08-22 라이브 현장

### 6.1 점검 방법

- 호스트 `192.168.137.249`, 사용자 `sandi`
- **읽기 전용.** 파이 파일·프로세스·git을 수정하지 않음
- `/api/status`와 최신 jsonl만 조회
- 이 보고서에 SSH 자격 증명을 기록하지 않음

### 6.2 운영자가 수행한 테스트

센서 고정, 사람 1명, 거리 대략 **0.6–1.0 m**, 정지 자세 약 1분. MR60BHA2 호흡 권장 범위(대략 0.4–1.5 m, 정지 1인) 안이다. 테스트 후 ESP 전원을 껐다.

거리/자세 미스가 1순위가 아니다. 그 조건에서도 아래 두 현상이 나왔다.

### 6.3 라이브 `/api/status` (ESP ON, ~01:32 KST)

한 순간:

- 센서 `LIVE`
- `human_detected_raw=true`
- `breath_phase` / `ts_monotonic_ms` / `phase_age_ms` 존재
- `canonical_window_status=WINDOW_UNAVAILABLE`
- **reason = `WINDOW_CONTAINS_LARGE_GAP`**
- 채택 ≈ 241, span ≈ 30,949 ms (30초는 채움)

수 초 뒤, 같은 세션:

- `ai.state=APNEA-proxy`
- `source=tflite`
- confidence ≈ **0.996**
- `MAD` ≈ **0.02**
- `apnea_verified=false`
- `risk.component_status.mmwave=RULE_FALLBACK`
- `neural_trust=OBSERVE_ONLY`
- `observed_neural_state=APNEA-proxy`

해석:

1. 위상 삼총사와 재실은 통과했다. PR #27·#29가 노리던 배선 결함은 이 세션에서 더 이상 1순위가 아니다.
2. 창 거절 reason은 모호한 `INPUT_UNAVAILABLE`이 아니라 **large-gap**이다.
3. 창이 열리면 모델은 즉시 고신뢰 APNEA-proxy를 낸다. 2026-08-17 재생과 같다.
4. 위험엔진은 그 클래스를 점수에 넣지 않았다. UI/`ai.state`에 APNEA-proxy가 보여도 무호흡 경보가 아니다.
5. 라이브 `MAD≈0.02`는 재생 창 `MAD≈0.10`보다 훨씬 작다. 창이 평탄했고, 스펙트럼이 6초 정지를 인정해 **반증 게이트가 안 걸렸을 가능성**이 있다. 당시 `spectral_hold_evidence` / `spectral_contradicts_apnea`는 요약에 남지 않았다.

### 6.4 ESP OFF 이후

전원을 끈 뒤에는 새 위상이 없다. 끄기 직전 `20260821_16_mmwave.jsonl`이 정지 테스트의 증거이다. ESP off 상태의 `/api/status`로 모델 품질을 재평가하지 않는다.

---

## 7. 두 문제는 같은 버그가 아니다

### 문제 A — 창이 자주 깨진다

증상: `WINDOW_UNAVAILABLE` / `WINDOW_CONTAINS_LARGE_GAP`.

원인 후보 (증거 있는 것만):

| 후보 | 증거 | 조치 성격 |
|---|---|---|
| 수신 계층이 15 s마다만 ingest | PR #28 이전 재생에서 채택 1개. **코드 수정으로 닫힘** | 완료 (integration main) |
| nested phase를 파서가 무시 | 2026-08-17 jsonl은 nested만 있음. **PR #27로 닫힘** | 완료 |
| genuine 7 Hz + 재발행 | 커밋 캡처·라이브 모두 ≤8 ms 수천 건 | 정상 동작. 재발행은 버려야 함 |
| 실제 500 ms–수 초 gap | 라이브 34 / 10회, 커밋 `_08` 181 / 31회 | 센서/펌웨어/움직임/레이더 정지. 계약을 느슨히 하지 않음 |
| boot/session 리셋 | 음수 dt | 창 리셋이 맞음 |
| 사람 이동으로 phase 갱신 중단 | MR60 문서: 비정지 시 호흡 갱신 중단 가능 | 현장 규율. 이번 정지도 갭이 남음 |

문제 A를 “30초 동안 서 있으라”로 바꾸면 안 된다. span이 30초여도 창 안 한 곳이 비면 거절한다.

### 문제 B — 창이 열리면 APNEA-proxy

증상: `source=tflite`, 클래스 `APNEA-proxy`, 신뢰도 0.8–0.996.

원인:

- 학습 분포가 실기 MR60 위상과 같다고 검증되지 않음 (`DEVICE_VALIDATED: NO`)
- 오프라인 한계 (M-B12 seed42): APNEA-proxy recall ≈ 0.94, NORMAL recall ≈ 0.20, APNEA FP rate ≈ **0.52**
- 2026-08-17 실측 재생: 판별 가능 창 5/5가 APNEA-proxy, 최고 conf 0.996, 동시에 호흡 4–27 rpm
- 2026-08-22 라이브 정지 테스트: 동일 클래스, conf ≈ 0.996, `MAD≈0.02`

이것은 센서가 사람을 무호흡으로 “맞춘” 것이 아니다. **호흡 파형에 대한 고신뢰 오탐**이다. 재학습·실기 검증 과제이며, 거리만 바꿔서 닫히지 않는다.

빈 방 zero 입력이 이 클래스를 내지 못하게 재실 게이트를 유지한다.

---

## 8. 위험엔진이 하는 일 / 안 하는 일

integration `risk/risk_formula_v1.json` (`SAFENEST_RISK_V1`):

| 방어 | 동작 |
|---|---|
| `neural_trust=OBSERVE_ONLY` | M-N9 클래스는 `observed_neural_state`로만 기록. 점수에 안 들어감 |
| 호흡 성분 | 스펙트럼 `rate_rpm` 1순위, MR60 `breath_rate_raw`는 최후 수단 |
| 스펙트럼 반증 | 주기적 호흡 + 6 s 정지 없음 → `APNEA_CONTRADICTED_BY_SPECTRUM` / `RESPIRATORY_INFERENCE_REFUSED` |
| 판별력 게이트 | conf ≥ 0.40 **그리고** top-2 마진 ≥ 0.15. 미달이면 `{SENSOR}_AI_OUTPUT_INDECISIVE` |
| 지속성 | APNEA-proxy 1회는 에스컬레이션 후보가 아님 |
| 등급 상한 | 미검증 APNEA-proxy는 **WARNING까지**. 단독 DANGER 금지 |
| `apnea_verified=true` | 하드웨어 확인만 DANGER+비상. 모델 의견이 아님. 현재 경로는 항상 false |

라이브에서 `component_status=RULE_FALLBACK`인 것은 고장 표시가 아니라, **미검증 신경망을 점수에 안 넣는 의도된 상태**다. 볼 값은 `ai.state`보다 `apnea_verified`, `spectral_contradicts_apnea`, `neural_trust`, `component_status`다.

구 V4 `risk/engine.py`의 비상 오버라이드는 문자열 `"APNEA"`만 봐서, 파이프라인이 내는 `"APNEA-proxy"`에는 사문화되어 있었다. v1은 둘 다 처리하되 미검증은 WARNING 상한이다.

---

## 9. 이미 머지된 배선 (integration)

| PR | 제목 | 이 보고서에 대한 효과 |
|---|---|---|
| [#22](https://github.com/yuname121/integration/pull/22) | M-N9 INT8 수입 | 활성 선택자가 B-stage가 아님 |
| [#25](https://github.com/yuname121/integration/pull/25) | Stage 7 preflight ↔ M-N9 | Mac 오프라인 선택자 계약 |
| [#27](https://github.com/yuname121/integration/pull/27) | nested phase 승격 | ESP `mmwave.breath_phase` 삼총사를 파서가 읽음. presence 없으면 구체적 에러 |
| [#28](https://github.com/yuname121/integration/pull/28) | wire-rate 누적 + 스펙트럼 + 위험도 v1 | 15 s 발행 주기 버그 수정. OBSERVE_ONLY. APNEA 반증 |
| [#29](https://github.com/yuname121/integration/pull/29) | 펌웨어 presence 삼상태 | B1 닫힘. 라이브 jsonl에 true/false가 보임 |

닫힌 PR #23은 남기되 머지하지 않았다. Stage 7 계약을 되감으면 안 되기 때문이다. 런타임 내용은 #27로 다시 올렸다.

`LATEST_SOURCE_PROVENANCE.json`의 `PI_SMOKE: NOT_PERFORMED`는 **잠긴 아티팩트 필드**다. 2026-08-22 읽기 전용 점검은  Informal 현장 관측이며, 이 필드를 `PERFORMED`로 바꾸지 않는다. 공식 Pi smoke / DEVICE_VALIDATED는 별도 과제다.

---

## 10. 아직 열린 것

1. **문제 A (large-gap)**  
   라이브에서도 30초 span 안에 갭이 들어간다. 계약을 완화하지 않는다. 다음 관측은 `phase_age_ms`, genuine dt, boot_id/session_id, `canonical_window_status`의 **reason**이다.

2. **문제 B (APNEA-proxy 오탐)**  
   재학습 또는 실기 분포 검증 없이 모델 클래스를 신뢰하지 않는다. 오늘은 스펙트럼 규칙 + OBSERVE_ONLY가 호흡 성분이다.

3. **파이 ↔ integration 트리 불일치**  
   현장은 `safenest-team-main` `f718430`. integration 최신은 `c759205`. 배포는 팀 워크플로대로 PR 머지 후 파이 `git pull --ff-only` ( `data/`·`.env` 보존 ). 이 문서만으로 파이를 핫패치하지 않는다.

4. **커밋 캡처에 presence가 없음**  
   2026-08-16/17 파일은 펌웨어 1.3.0 이전이다. 게이트 없는 재생은 `--inject-presence`가 계속 필요하다. presence가 찍힌 재캡처는 아직 integration에 없다 (`20260821_16`은 파이 디스크에만 있음).

5. **CO₂ 습도 (B2)**  
   mmWave 범위 밖. C-B6는 습도를 금지한다.

6. **공식 `DEVICE_VALIDATED` / Stage 9 soak**  
   수행하지 않음. 이 보고서가 그 자리를 대체하지 않는다.

---

## 11. 현장에서 바로 읽을 값

ESP가 켜져 있고 사람이 레이더 앞에 있을 때:

```bash
curl -s http://127.0.0.1:8000/api/status | python3 -c "
import json,sys
d=json.load(sys.stdin)
m=d['mmwave']; ai=m.get('ai') or {}; meta=ai.get('metadata') or {}
r=d.get('risk') or {}
print('ai.state', ai.get('state'), 'source', ai.get('source'), 'conf', ai.get('confidence'))
print('error', ai.get('error'))
print('window', meta.get('canonical_window_status'), 'reason', meta.get('reason') or meta.get('canonical_window_reason'))
print('MAD', meta.get('MAD'), 'apnea_verified', meta.get('apnea_verified'))
print('spectral', meta.get('spectral_status'), 'rpm', meta.get('spectral_rate_rpm'),
      'hold', meta.get('spectral_hold_evidence'), 'contradicts', meta.get('spectral_contradicts_apnea'))
print('presence', (m.get('state') or {}).get('values', {}).get('human_detected_raw'))
print('component', (r.get('component_status') or {}).get('mmwave'),
      'trust', ((r.get('components') or {}).get('mmwave') or {}).get('metadata', {}).get('neural_trust'))
"
```

판정 가이드:

| 보이는 것 | 의미 |
|---|---|
| `WINDOW_CONTAINS_LARGE_GAP` | 문제 A. 30초 연속 타이밍이 깨짐 |
| `APNEA-proxy` + `apnea_verified=false` + `OBSERVE_ONLY` | 문제 B. 표시일 뿐 무호흡 경보 아님 |
| `APNEA_CONTRADICTED_BY_SPECTRUM` | 모델 오탐을 DSP가 거절. 정상 방어 |
| `PRESENCE_STATE_UNAVAILABLE` / `null` | 재실 미확인. 추론 안 함 |
| `NO_VALID_PERSON` / `false` | 빈 방으로 본 것. 창이 있어도 predict 안 함 |

---

## 12. 명시적 금지

- 재실 게이트를 끄지 않는다. 끄면 빈 방 zero 입력이 APNEA-proxy로 나간다.
- `DEVICE_VALIDATED`를 이 관측만으로 `true`로 쓰지 않는다.
- `APNEA-proxy`를 UI/보고서에 “무호흡 감지”로 번역하지 않는다.
- large-gap을 없애려고 gap 임계를 느슨히 하거나 8 Hz 계약을 깨지 않는다.
- 파이 `main` 핫패치로 이 문서의 결론을 심지 않는다. 팀 워크트리 + PR만 사용한다.

---

## 13. 다음 액션 (우선순위)

1. 팀 파이를 integration `c759205` 이후와 맞출지 팀 배포 규칙으로 결정한다. 맞추면 스펙트럼 반증·wire-rate 누적이 현장에 있는지 필드 단위로 확인한다.
2. ESP를 다시 켤 때 §11 한 줄을 로그로 남긴다. 특히 `canonical_window_status`의 **reason**, `MAD`, `spectral_*`.
3. presence가 있는 `20260821_16` 급 jsonl을 비밀 없이 잘라 integration에 올리는 일은 별도 PR이다. 이 문서는 요약을 인용할 뿐 원본 15,962행을 포함하지 않는다.
4. 문제 B의 해소는 재학습/실기 검증이다. 런타임 배선을 더 꼬아서 클래스를 `NORMAL`로 위장하지 않는다.

---

## 14. 증거 인덱스

| 증거 | 위치 |
|---|---|
| 커밋 JSONL | `data/mmwave/20260816_*.jsonl`, `20260817_*.jsonl` |
| 2026-08-20 파이 API | `snapshots/20260820_pi_live/api/status.json` |
| 연결 전 검수 | `docs/20260821_Preconnect_Runtime_Audit_And_Risk_Formula_V1_KO.md` |
| 라벨 정의 | `sources/ondevice_ai/datasets/mmwave/manifests/a4_label_pilot/label_mapping_profile.json` |
| 오프라인 한계 | `sources/ondevice_ai/datasets/mmwave/manifests/M-B12_phase_b_offline_final/scientific_limitations.json` |
| 스펙트럼 반증 | `ai/mmwave_spectral_runtime.py` |
| 위험도 v1 | `risk/risk_formula_v1.json` |
| 라이브 jsonl | 파이 `.../data/mmwave/20260821_16_mmwave.jsonl` (미커밋) |
| 라이브 파이 git | `safenest-team-main` @ `f718430` (integration 아님) |
