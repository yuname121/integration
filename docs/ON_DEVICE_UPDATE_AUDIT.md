# ON-DEVICE UPDATE AUDIT

## 감사 기준

- 기존 통합 기준: GitHub `main` `6baf38d8df936b694a1ff2e9b5e5fb2af2bfe50f`
- 최신 확인 기준: GitHub `origin/main` `fa8cf13` (`fa8cf13` merge, component source `77b1695`)
- 최신 AI component source snapshot: `77b1695ac66fd595bd037e4574d1626b8917654c`
- 확인일: 2026-08-13
- 모델 재학습·재변환: 수행하지 않음

## 1. 최신 온디바이스 관련 파일 목록

✅ **CONFIRMED** 최신 `ondevice_ai/` 전체 1,069개 tracked 파일을 `sources/ondevice_ai/`에 무수정 동결했다. 이번 intermediate release에서 489개 파일이 추가·수정되었고, 코드, config, 데이터 provenance, benchmark, model, validator, report가 포함되며 archive tree를 runtime fallback으로 사용하지 않는다.

운영 경계에 직접 관련된 파일은 다음과 같다.

| 영역 | 최신 파일 |
|---|---|
| Inference | `inference/thermal_interpreter.py`, `co2_interpreter.py`, `mmwave_interpreter.py`, `validator.py` |
| Model | `models/model_manifest.json`, Thermal/CO₂/mmWave primary와 CO₂/mmWave offline candidate artifacts |
| Preprocessing | `thermal_prep.py`, 신규 `preprocessing/mmwave.py` |
| Integrated runtime | `integrated_node/run_node.py` |
| Risk | `risk/risk_engine.py`, `fallback.py`, `risk_rules.py`, `risk_config.json` |
| CO₂ 검증 | `datasets/co2/`, `scripts/*co2*`, `tests/test_co2_*` |
| Thermal 검증 | `datasets/thermal/`, `scripts/*thermal*`, `tests/test_thermal_*` |
| mmWave 검증 | `datasets/mmwave/`, `models/mmwave/experiments/`, `scripts/*mmwave*`, 관련 tests |

## 2. 기존 통합 당시 파일과 비교

| 항목 | 기존 | 최신 | 판정 |
|---|---|---|---|
| Thermal interpreter/model | v0.1, `(1,62,80,1)` INT8 | 동일 blob/model 계약 | ✅ 그대로 유지 |
| CO₂ interpreter/model | v0.1, `(1,3)` INT8 | 동일 blob/model 계약 | ✅ 그대로 유지 |
| mmWave interpreter | manifest `models.mmwave` 사용 | 동일 blob/API | ✅ API 동일 |
| mmWave primary model | v0.1, 사용 가능 여부 미표기 | v0.1, `deployment_allowed=false` | ⚠️ release gate 필요 |
| mmWave candidate | 없음 | M-B6~M-B12 offline lock candidate 추가 | ⚠️ adapter 없이 자동 교체 금지 |
| mmWave preprocessing | interpreter 내부 scalar z-score | 별도 experimental 7-stage module 추가 | ⚠️ 서로 아직 연결되지 않음 |
| Risk Engine | locked V4 | 같은 blob | ✅ 그대로 유지 |

## 3. 새로 추가된 파일

| File | 역할 | 기존 대응 파일 | 영향 |
|---|---|---|---|
| `preprocessing/mmwave.py` | detrend, optional bandpass, train-only z-score, clipping | interpreter 내부 `prepare_window()` | 실험용이며 현재 runtime에 미연결 |
| `models/co2/candidates/c_b4/**`, `c_b5/**` | C-B4/C-B5 오프라인 후보와 계약 | 기존 co2 v0.1 primary | device-domain 검증 전 자동 선택 금지 |
| `models/mmwave/experiments/M-B6_stage_equivalence/**` | M-B6 strict INT8 후보 artifact | 기존 v0.1 primary | 오프라인 후보, runtime manifest opt-in 필요 |
| `models/mmwave/mmwave_offline_candidate_lock_v1.json` | M-B11 후보 lock/제약 | 기존 v0.1 manifest | `deployment_ready=false`, 자동 선택 금지 |
| `datasets/co2/**` | C-A0~A6, C-B0~C-B5 integrity/provenance | 제한된 기존 dataset | runtime input API는 변경 없음 |
| `datasets/thermal/**` | T-A0~A6 grouping/conversion evidence | 제한된 기존 dataset | 새 Thermal model 후보 없음 |
| `datasets/mmwave/**`, experiment models | M-A/M-B6~M-B12 평가 evidence | 기존 v0.1 자료 | 모델 release 판단 근거 추가 |
| `AGENTS.md` | active component와 fail-closed 규칙 | 없음 | 새 snapshot 운영 원칙 |

## 4. 수정된 파일

| File | 주요 변경점 | 기존 시스템 영향 |
|---|---|---|
| `inference/mmwave_interpreter.py` | opt-in runtime manifest, BPF+Z-score preprocessing trace, artifact identity 검증 추가 | 기본 legacy manifest/API는 유지되며, phase manifest는 별도 의존성·계약 필요 |
| `models/model_manifest.json` | 기존 primary manifest는 유지 | 새 후보는 명시적 runtime manifest 없이는 선택되지 않음 |
| `integrated_node/run_node.py` | production deployability gate와 sensor disable 추가 | 새 통합 runtime은 이 node를 직접 import하지 않음 |
| `inference/validator.py` | active component root 탐색, legacy wrapper 경로 제거 | validation 도구에만 영향 |
| `datasets/MANIFEST.json`, docs/reports | multisensor provenance 동기화 | runtime 없음 |

## 5. 대체/삭제된 파일

✅ **CONFIRMED** 기존 Thermal, CO₂ interpreter와 Risk Engine은 삭제·대체되지 않았다. mmWave interpreter는 수정됐지만 기본 manifest 경로와 `predict(window)` 호출 호환성을 유지한다. 최신 sync collision plan은 5개 수정 파일과 484개 추가 파일이며, 운영 통신 파일은 건드리지 않았다.

❓ **UNKNOWN** mmWave offline candidate를 언제 primary key `mmwave`로 승격할지는 최신 main에 확정된 release pointer가 없다.

## 6. 최신 모델 파일

| Manifest key | 파일 | Size | Input | Output | 상태 |
|---|---|---:|---|---|---|
| `thermal` | `thermal_fall_int8_v0.1.0.tflite` | 318,184 | `(1,62,80,1)` INT8, scale `0.003921568859...`, zp `-128` | `(1,3)` INT8 | deployment allowed, synthetic only |
| `co2` | `co2_occupancy_int8_v0.1.0.tflite` | 4,464 | `(1,3)` INT8, scale `0.005828449968...`, zp `57` | `(1,2)` INT8 | deployment allowed, synthetic only |
| `mmwave` | `mmwave_resp_int8_v0.1.0.tflite` | 466,616 | `(1,300,1)` INT8, scale `0.032598569989...`, zp `-13` | `(1,3)` INT8 | **BLOCKED**, class collapse |
| `mmwave offline lock` | `M-B6...seed42...int8.tflite` | 22,080 | `(1,300,1)` INT8, scale `0.041720833629...`, zp `-3` | `(1,3)` INT8 | offline candidate, `deployment_ready=false` |

참고: 최신 manifest의 후보 항목에는 `size_bytes`가 없고 실제 artifact 크기는 22,472 bytes다. 또한 `deployment_allowed: true`와 `hardware_validation: BLOCKED_HARDWARE`가 함께 있어 운영 승격 기준으로는 모순될 수 있으므로, 이 패키지는 더 엄격한 후자를 적용해 후보를 자동 선택하지 않는다.

모든 manifest entry SHA-256은 새 번들의 실제 파일과 자동 대조한다. experiment model은 evidence로 포함하지만 runtime에서 자동 탐색하지 않는다.

## 7. Input / Output 변경사항

### Thermal

✅ shape, dtype, quantization, class map 모두 동일하다.

- Input: `[1,62,80,1]`, INT8
- Output: `[1,3]`, `NOT_HUMAN/HUMAN_NORMAL/HUMAN_FALL`

### CO₂

✅ shape, dtype, quantization, class map 모두 동일하다.

- 기존 운영 primary input semantic: `co2_slope, humidity, co2_ppm`. 이번 snapshot의 B-complete offline candidate 계약은 `CO2 + CO2_slope`이며 별도 device-domain 검증 전 자동 승격하지 않는다.
- Output: `VACANT/OCCUPIED`
- ⚠️ 현재 runtime adapter는 기존 3-feature primary를 계속 선택하므로 humidity 부재 시 그 legacy primary inference는 unavailable이고 CO₂ rule fallback을 사용한다. B-complete candidate 자체에는 humidity가 필요하지 않지만 아직 runtime 승격 대상이 아니다.

### mmWave

✅ class map과 300 samples/10 Hz/30 sec 계약은 동일하다.

⚠️ candidate의 quantization과 scaler는 v0.1과 다르다. 기존 interpreter는 `models["mmwave"]`와 v0.1 metadata를 읽으므로 candidate 파일만 교체하면 잘못된 전처리·양자화가 된다.

## 8. Preprocessing 변경사항

### Thermal

✅ 기존 per-frame min-max 학습 근거와 integration pipeline을 유지한다. 최신 검증 자료는 raw unit, invalid-pixel, geometry, temporal/semantic 제한을 추가했지만 production 전처리 교체를 승인하지 않았다.

### CO₂

✅ runtime interpreter는 변경되지 않았다. 최신 C-A 자료는 slope lineage와 canonical sample provenance를 강화했다.

### mmWave

⚠️ 최신 `MMWaveInterpreter`는 opt-in phase runtime manifest에서 BPF+Z-score trace를 구현하지만, 통합 기본 경로는 legacy manifest를 사용한다.

1. 300 samples / finite input 확인
2. 10 Hz, 0.1~0.5 Hz Butterworth bandpass (phase manifest only)
3. TRAIN fit mean/std 기반 z-score
4. INT8 quantization 및 saturation trace
5. `[1,300,1]` model tensor

새 통합은 fail-closed 원칙 때문에 wrong length와 NaN/Inf를 보정해 모델에 넣지 않고 기존 입력 거부를 유지한다. phase runtime manifest와 scipy 의존성은 운영 primary로 승격하지 않는다.

## 9. Inference API 변경사항

✅ `ThermalInterpreter.predict(frame)`, `CO2Interpreter.predict(slope, humidity, ppm)`, `MMWaveInterpreter.predict(window)` 호출은 호환된다. mmWave prediction에 preprocessing profile/hash 선택 필드가 추가됐지만 기존 필드는 유지된다.

⚠️ 최신 `run_node.py`에 production deployability gate가 추가됐지만 새 통합은 별도 `SafeNestRuntime`을 사용한다. 따라서 `ai/runtime.py` 경계에 동일한 manifest gate를 adapter로 반영했다.

## 10. 현재 통합 코드와 충돌하는 부분

1. **mmWave v0.1 release block**: 기존 loader는 manifest release 상태를 확인하지 않았다.
2. **candidate selection 불명확**: offline lock candidate는 별도 runtime manifest 없이는 최신 interpreter가 참조하지 않는다.
3. **phase preprocessing opt-in**: BPF+Z-score runtime manifest와 legacy z-score 경로가 병존한다.
4. **missing live inputs**: TCP v1에는 mmWave phase window/presence가 없다. CO₂에는 B-complete candidate의 slope history를 만들 실제 measurement-event provenance가 필요하며 humidity는 해당 candidate 입력이 아니다.
5. **validator rename**: `validate_v4_config`가 `validate_active_config`로 바뀌었지만 integration runtime은 이를 호출하지 않는다.

## 11. 영향받는 파일

| 변경 | 영향 파일 |
|---|---|
| 최신 전체 AI snapshot | `sources/ondevice_ai/**` |
| model release gate | `ai/runtime.py` |
| blocked mmWave regression | `tests/test_ai_pipeline.py` |
| manifest entry 증가 | `hil/preflight.py`, `tests/test_hil_criteria.py`, package verifier |
| provenance와 실행 안내 | `README.md`, `SOURCE_MANIFEST.md`, package docs |

## 12. 수정 필요 없는 파일

✅ SafeNest TCP v1 parser/receiver, auto-reconnect, Sensor State Manager, Risk Engine 계산, FastAPI/SQLite/WebSocket, dashboard contract는 최신 AI interface 변경의 영향을 받지 않아 유지했다.

✅ Thermal·CO₂ model adapter와 integration pipeline 호출 schema도 동일하여 유지했다.

## 13. 권장 반영 방법과 실제 결정

- ✅ `SAFE TO REPLACE`: `sources/ondevice_ai/`를 최신 `origin/main` 전체 snapshot으로 교체
- ⚠️ `ADAPTER REQUIRED`: integration `LazyModel`에서 primary manifest entry의 `deployment_allowed=false`를 차단
- ❌ `BREAKING CHANGE`: offline candidate 자동 승격, phase preprocessor 강제 연결, 통신 schema 임의 확장은 수행하지 않음
- ✅ mmWave blocked/입력 미제공 시 기존 respiration rule fallback 지속
- ✅ AI failure isolation과 strict invalid-input 차단 유지

## 이전 버전과 달라진 점

- 최신 multisensor intermediate release(`fa8cf13`/`77b1695`)가 추가됐다.
- CO₂ C-B0~C-B5, Thermal T-A5~T-A6, mmWave M-B6~M-B12 오프라인 증거가 추가됐다.
- mmWave interpreter에 opt-in BPF+Z-score runtime manifest와 artifact identity 검증이 추가됐다.
- mmWave 오프라인 후보는 `deployment_ready=false`; CO₂ 후보도 SCD40 device-domain 검증 전이다.

## 기존 통합 시스템에서 수정해야 하는 점

- loader가 최신 manifest의 release block과 오프라인 후보의 `deployment_ready=false`를 준수해야 한다.
- phase runtime manifest를 선택할 때만 scipy/BPF 계약과 후보 artifact hash를 별도 검증해야 한다.
- model hash/preflight는 새 candidate entry까지 검사해야 한다.
- 문서가 최신 source commit과 candidate/blocked 상태를 밝혀야 한다.

## 그대로 유지해도 되는 점

- ESP32/TCP v1, receiver, State Manager
- Thermal·CO₂ inference 호출과 전처리 경계
- Risk/Backend/SQLite/Dashboard schema
- rule fallback과 AI 장애 격리
