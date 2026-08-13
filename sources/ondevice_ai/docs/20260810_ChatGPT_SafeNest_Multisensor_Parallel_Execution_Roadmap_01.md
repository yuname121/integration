# SafeNest 멀티센서 병렬 A–E 실행 로드맵

- 문서 버전: `02`
- 기준일: `2026-08-13`
- canonical component root: 이 문서의 상위 저장소에서 `AGENTS.md`가 위치한 디렉터리
- 적용 대상: mmWave, CO₂, Thermal 온디바이스 AI 데이터·모델·runtime 검증
- 후속 통합 대상: PIR 보조 신호 및 멀티센서 risk fusion
- 상태: `ACTIVE_MASTER_ROADMAP`

이 문서는 기존 mmWave A–E 실행 문서를 원문 그대로 포함하면서 CO₂와 Thermal을 독립 트랙으로 병렬 실행하기 위한 상위 제어 문서다. 기존 mmWave 문서는 역사·세부 실행 근거로 계속 유효하지만, 신규 작업의 센서 간 순서와 공통 gate는 이 문서를 우선한다.

## 0. 문서 우선순위와 해석 규칙

1. 최상위 `AGENTS.md`가 경로·안전·provenance·팀 이관 규칙의 최우선 기준이다.
2. 이 문서는 세 센서의 병렬 순서, 합류 gate, 공용 파일 변경 규칙을 정한다.
3. 아래에 원문 그대로 이관된 mmWave 문서는 mmWave 세부 실행 기준이다.
4. 각 센서 phase 보고서와 machine-readable manifest가 완료 수치의 근거다.
5. 문서와 artifact가 충돌하면 validator가 통과한 machine-readable evidence를 우선하고 문서 불일치를 별도 수정한다.

`A/B/C/D/E`라는 문자만 단독으로 쓰면 센서가 불명확하므로 신규 branch, report, manifest, issue에서는 다음 접두어를 사용한다.

| 접두어 | 트랙 | 의미 |
|---|---|---|
| `M-` | mmWave | radar raw/canonical phase, 호흡 proxy 모델, MR60 domain |
| `C-` | CO₂ | 실제 occupancy 원본, feature model, SCD40 domain |
| `T-` | Thermal | 실제 thermal frame/sequence, fall model, Thermal-44 domain |
| `I-` | Integration | 공용 provider 계약, replay simulation, risk fusion, Pi 통합 |

기존 mmWave 문서의 `A0`, `B0` 등은 각각 `M-A0`, `M-B0`와 같은 의미다. 기존 본문은 evidence 보존을 위해 이름을 일괄 치환하지 않는다.

## 0.1 팀 저장소 PR·브랜치 증거 오버레이 (2026-08-13 검토)

이 절은 팀 저장소의 실제 센서 구현·실측 자료와 현재 로드맵의 연결점을 기록하기 위한 보강 규칙이다. 검토 기준은 팀 저장소 `main`의 `f3bd342eabcad27dc2c3ecdc16f035b8b13cb153`과 그 시점에 존재한 원격 branch·PR이다. 팀 저장소의 구버전 `ondevice_ai/` 트리는 이번 방향성 판정과 phase evidence의 근거에서 **제외**한다. 이 절은 파일을 자동 이관하라는 지시가 아니며, 미병합 자료를 standalone 학습 데이터로 자동 편입하지 않는다.

### 0.1.1 증거 층위와 병합 상태 해석

팀 저장소 자료는 다음 세 층위로 분리한다.

| 증거 층위 | 대표 경로/자료 | 로드맵에서의 용도 | 금지 해석 |
|---|---|---|---|
| standalone offline evidence | 이 저장소의 `datasets/`, `models/`, A/B manifest·validator | 재현 가능한 전처리·split·모델 비교·locked test | 팀 실센서 로그를 근거 없이 A/B에 섞기 |
| device-domain evidence | 팀 `devices/<sensor>/`의 firmware, raw log, calibration, 분석 보고서 | M-C/C-C/T-C의 실제 입력·장치 결함·domain gap 검증 | public dataset 성능이나 모델 일반화 성능으로 승격 |
| integration evidence | 팀 `devices/esp32_node/`, `integration/`, 통신 규격·LCD/Pi 자료 | I-0/I-1/I-2의 packet·timestamp·provider·fail-closed 계약 검증 | 통합 노드 동작을 개별 AI 정확도로 주장 |

PR이 `open`이거나 branch에만 존재하면 **후보 evidence**일 뿐 팀 `main`의 승인된 기준이 아니다. 에이전트는 PR/branch 자료를 사용할 때 source branch, head SHA, base SHA, 병합 상태, 원본 checksum, 제한사항을 report에 남긴다. 병합되지 않은 자료는 원본을 보존한 채 C/I 검증 입력으로만 참조하고, standalone A/B artifact를 덮어쓰지 않는다.

### 0.1.2 2026-08-13 원격 PR·branch 확인 결과

다음 목록은 팀 저장소의 `ondevice_ai/` 변경을 제외하고 센서·실행·통합 방향에 영향을 주는 원격 자료만 요약한 것이다.

| 상태 | 대상 | 확인된 사실 | 이 로드맵에서의 인지·처리 |
|---|---|---|---|
| `OPEN` | PR #14 `feature/co2-scd40-verification` (`ea925e4`) | SCD40 preflight 30/30 valid, 재시도 baseline 300/300 valid, 호기 상승·복귀 329/360 valid, 결측률 8.61%. 센서 분리 60초 원시 시험은 `NOT VERIFIED`; 보고서 판정은 `PARTIAL` | C-C의 실제 입력·결측·재연결 evidence로 참조한다. 분리·stale·reconnect 계약이 끝나기 전 C-C 완료나 최종 CO₂ 증거로 승격하지 않는다 |
| `OPEN` | PR #15 `feature/thermal-v5-real-validation` (`e4cb7d8`) | 실제 열화상 raw frame 수신·62×80 parse·TFLite·fail-closed·UDP 경로 검증 자료가 있다. TCP 단계의 brownout/655.3°C 오류와 UDP 전환이 기록되어 있다. 문서에는 Thermal-90/MI48 계열과 Thermal-44 명칭이 혼재한다 | T-C의 runtime/domain-gap 입력으로 보존한다. `ALL PASS` 문구만으로 T-B 학습 성능·낙상 일반화·T-C 완료를 인정하지 않는다. 센서 모델명, 원시 frame 단위·calibration·orientation·프로토콜을 먼저 reconcile한다 |
| `OPEN` | PR #12 `feature/esp32-lcd-integration` (`c9f4583`) | ESP32 4센서 수집과 Pi/LCD 상태 전달을 실제 장치에서 확인했다. 다만 Thermal은 약 70% 고정/무효 pixel 때문에 full-frame stream을 끄고 `thermal_max_c`만 전송하며, 호흡수 noise와 통신 조건도 기록되어 있다 | I-0/I-1 packet·validity·timestamp·runtime evidence로 참조한다. 이 경로의 scalar thermal telemetry를 full-frame T-A/T-B 입력과 동일시하지 않는다 |
| `OPEN` | PR #11 `agent/add-competition-package` (`4ac9878`) | ESP32/Pi 실행 패키지와 통신·설치 문서를 추가한다. 새로운 학습 데이터나 센서 정확도 evidence는 추가하지 않는다 | I-6 handoff/운용 참고로만 사용한다. A/B/C 성능 근거로 사용하지 않는다 |
| `NO PR` | `codex/mmwave-20rpm-root-cause` (`0e8538c`), `codex/mmwave-phase-integration` (`b0d3c95`) | MR60 실제 로그·phase·presence·window 자료와 20 rpm 저-SNR 원인 분석이 있다. 20 rpm 오차의 직접 원인은 TFLite보다 입력 품질과 estimator validity gate로 분석되었다 | M-C domain/gap 분석 및 M-D 수집 설계의 근거로 보존한다. production 변경이나 A/B 학습 편입은 별도 review·회귀검증 후 결정한다 |

PR #13의 `ondevice_ai` 동기화 변경은 이 오버레이의 범위에서 제외한다. PR 설명에 언급된 `feature/pir-verification`은 이 검토 시점 원격 branch 목록에서 확인되지 않았으므로 승인된 PIR evidence로 간주하지 않는다.

### 0.1.3 센서별 필수 인지 사항

- **mmWave**: 팀 MR60 자료는 presence·distance·raw phase·timestamp가 있는 장치 domain 자료다. 단일 피험자·제한된 조건, 저진폭/phase stale·presence loss·퇴실 hysteresis가 존재한다. `breath_phase`의 단위·sampling·window·gap·quality gate를 standalone canonical signal과 먼저 매핑하고, 실측 로그는 무작위로 학습에 섞지 않는다. A4의 voluntary breath-hold label은 SafeNest proxy이며 clinical apnea가 아니다.
- **CO₂**: PR #14의 SCD40 ppm 시계열은 실제 장치 증거지만 occupancy/환경 변화 자료이지 질식·유해가스 ground truth가 아니다. UCI occupancy target과 SCD40 safety rule을 분리한다. 첫 baseline 실패와 호기 시험의 결측을 숨기지 않고, 센서 분리·stale·reconnect를 완료한 뒤 C-C를 잠근다.
- **Thermal**: PR #15의 full-frame path와 PR #12의 `thermal_max_c` scalar path는 서로 다른 runtime contract다. Thermal-90/MI48/Thermal-44 명칭, 62×80 geometry, raw uint16→°C 변환, calibration, orientation, invalid pixel 정책을 하나의 provenance로 reconcile하기 전에는 서로의 결과를 합치지 않는다. 자세/`LYING` proxy는 실제 낙상 사건이나 임상 성능을 의미하지 않는다.
- **PIR**: 독립 AI 재학습 트랙이 아니라 mmWave 재실·퇴실과 위험 rule을 보조하는 binary evidence다. 별도 branch/실측이 승인되기 전에는 I-0 계약·fault/replay 대상으로만 다룬다.
- **통합**: ESP32 `safenest.telemetry.v1`, TCP 9000, thermal frame type 2, `thermal_max_c` scalar, `valid`/`SensorState`/stale semantics를 I-0 inventory에 함께 기록한다. 센서별 offline candidate가 고정되기 전에 learned fusion을 최적화하지 않는다.

### 0.1.4 PR·branch 자료를 phase에 연결하는 추가 gate

1. `M-C`, `C-C`, `T-C`는 팀 PR의 “실제 동작 확인”을 그대로 완료로 복사하지 않고, raw source·checksum·unit·timestamp·calibration·failure registry를 standalone canonical contract와 대조한다.
2. 실제 하드웨어 log는 immutable raw evidence로 보존하고, 재생(replay)용 파생 파일을 별도 namespace로 만든다. 원본을 relabel하거나 overwrite하지 않는다.
3. C phase에서 발견한 통신 결측, dead pixel, stale phase, low-SNR, sensor identity mismatch가 모델 문제인지 입력/장치 문제인지 분리하여 기록한다. 입력 문제를 곧바로 재학습 사유로 쓰지 않는다.
4. 팀 PR이 병합되었더라도 `devices/`의 device-domain evidence와 standalone `datasets/`의 A/B evidence를 자동으로 합치지 않는다. 병합은 source/base SHA와 충돌·ownership 검토가 끝난 뒤 별도 handoff에서 수행한다.
5. 최종 흐름은 변경하지 않는다: **standalone A/B 재현성 확보 → 실제 장치 C 검증 → 측정된 gap만 D에서 보완 → I에서 replay·rule fusion·Pi 검증**.

## 1. 현재 출발 상태

| 트랙 | 현재 검증 상태 | 즉시 시작 가능한 작업 | 금지되는 선행 작업 |
|---|---|---|---|
| mmWave | `M-A0~M-A6 PASS_WITH_WARNINGS` | `M-B0` 평가 protocol·near-duplicate | LOCKED_TEST를 이용한 선택·튜닝 |
| CO₂ | 기존 NPZ는 synthetic smoke 성격이며 실제 UCI provenance 미복원 | `C-A0` 원본 identity·inventory | 기존 synthetic 지표를 실제 성능으로 승격 |
| Thermal | 실제 평가 dataset 부재로 일부 테스트 skip | `T-A0` dataset 선정·identity | 실제 낙상 성능·일반화 성능 주장 |
| Integration | mock·fail-closed wiring은 존재, 실센서 통합 미검증 | `I-0` 계약 차이 inventory만 가능 | 센서별 candidate lock 전 fusion 최적화 |

Phase A 완료는 실센서 성능이나 배포 준비 완료를 뜻하지 않는다. 각 센서는 offline evidence, device-domain evidence, Pi evidence를 별도 상태로 보존한다.

## 2. 병렬 실행 구조

```text
Track M  M-A 완료 ──> M-B offline model ──> M-C MR60 domain ──> M-D gap data ──┐
                                                                                │
Track C  C-A real-data reconstruction ──> C-B model ──> C-C SCD40 domain ──────┤
                                                                                ├─> I-0~I-6 integration
Track T  T-A dataset reconstruction ──> T-B model ──> T-C Thermal-44 domain ───┤
                                                                                │
Track I  contract inventory only ───────────────────────────────────────────────┘
```

센서 트랙 사이는 병렬 실행한다. 한 센서 트랙 내부에서는 이전 gate가 통과하기 전에 다음 phase를 성능 탐색 목적으로 시작하지 않는다.

### 2.1 지금 권장하는 동시 작업

- mmWave 작업자: `M-B0` evaluation protocol, exact/near-duplicate audit, LOCKED_TEST 접근 통제
- CO₂ 작업자: `C-A0` 실제 UCI source identity, license, raw file inventory, checksum
- Thermal 작업자: `T-A0` dataset 후보 비교, license, subject/session/sequence provenance 적합성
- 통합 작업자: 코드 수정 없이 local/team provider 계약과 signal semantics 차이 목록 작성

### 2.2 병렬 작업의 hard dependency

- `C-B`는 `C-A6` 통과 후에만 시작한다.
- `T-B`는 `T-A6` 통과 후에만 시작한다.
- scaler, normalizer, feature selector, calibration threshold는 각 센서의 TRAIN에서만 fit한다.
- LOCKED_TEST는 architecture, preprocessing, imbalance, threshold, representative dataset 선택에 사용하지 않는다.
- `I-2` replay integration은 최소 한 센서의 locked offline candidate와 나머지 센서의 명시적 unavailable/mock 상태가 있어야 시작할 수 있다.
- `I-3` fusion 최적화는 M/C/T 세 트랙의 validation contract가 고정된 후 시작한다.
- 실센서 성능 주장은 해당 센서의 C phase와 Pi 측정이 완료된 뒤에만 가능하다.

## 3. 공용 파일과 branch 충돌 방지

센서별 구현은 독립 branch와 report namespace를 사용한다.

```text
feature/mmwave-b0-evaluation-protocol
feature/co2-a0-real-data-inventory
feature/thermal-a0-dataset-selection
refactor/integration-provider-contract
```

다음 공용 파일은 여러 센서 branch가 동시에 직접 갱신하지 않는다.

- `AGENTS.md`
- `README.md`
- `docs/README.md`
- `datasets/MANIFEST.json`
- `models/model_manifest.json`
- 공통 evaluation schema와 risk configuration
- 팀 저장소의 `shared/contracts/`, `.github/`, root 문서

센서 phase branch에서는 센서 전용 manifest·report를 먼저 생성한다. 공용 inventory 반영은 phase 승인 후 별도 integration commit에서 evidence를 읽어 갱신한다. `git add .`는 사용하지 않는다.

## 4. 공통 sensor-local A–E 의미

| Phase | 공통 의미 | 센서별 변형 |
|---|---|---|
| A | 실제 원본에서 canonical dataset과 immutable split을 만드는 단계 | reader, label, grouping unit가 modality별로 다름 |
| B | 실제 데이터 offline 학습·비교·locked candidate 고정 | metric과 architecture가 modality별로 다름 |
| C | 실제 기기 출력과 학습 입력 사이의 domain·runtime 검증 | MR60, SCD40, Thermal-44 각각 별도 |
| D | C에서 측정된 gap을 메우는 추가 dataset 확장 | gap 없는 무목적 수집 금지 |
| E | sensor-local artifact·contract·report lock | 기존 mmWave 원문의 E는 상위 `I` fusion 단계로 재해석 |

각 센서의 A 종료물에는 최소한 source identity, license, checksum, inventory, label contract, group split, canonical data, sample provenance, quality audit, duplicate/leakage audit, standalone validator가 있어야 한다.

## 5. 병렬 synchronization gate

### Gate P0 — 현재

- M-A 완료
- C-A0, T-A0 시작 가능
- M-B0 시작 가능
- 통합은 contract inventory만 허용

### Gate P1 — 세 트랙의 데이터 기반 확보

- M-B protocol과 near-duplicate 판정 완료
- C-A6 통과
- T-A6 통과
- 공용 manifest가 세 실제 dataset을 synthetic fixture와 구분

### Gate P2 — offline candidate lock

- M-B, C-B, T-B가 각자 Float/TFLite/INT8 equivalence와 multi-seed 결과를 보유
- sensor별 LOCKED_TEST 사용 이력이 기록됨
- 일반 성능과 배포 성능 상태가 분리됨

### Gate P3 — device-domain 검증

- MR60, SCD40, Thermal-44 실제 출력 contract와 offline 입력 contract 비교
- domain gap, missingness, latency, warming-up, stale 정책 측정
- 필요할 때만 M/C/T-D 진입

### Gate P4 — integration readiness

- 세 sensor provider가 동일한 fail-closed output contract를 구현
- replay simulation과 fault injection 통과
- Pi 5 latency·memory·thermal·장시간 안정성 측정
- 임의의 mock 성공을 실센서 성공으로 보고하지 않음

## 6. 신규 작업 에이전트 필수 시작 절차

모든 센서 작업 에이전트는 작업 전에 다음을 보고해야 한다.

1. 선택한 sensor track과 phase ID
2. authoritative input artifact와 checksum
3. 수정할 전용 파일과 건드리지 않을 공용 파일
4. split/grouping/LOCKED_TEST 정책
5. 완료 validator와 회귀 테스트
6. 생성될 report·manifest 경로

phase ID 없이 “모델 개선”, “데이터 전처리”, “성능 향상”처럼 범위가 불명확한 작업은 시작하지 않는다.

---

# Part II — 기존 mmWave 실행 로드맵 원문 이관

아래 내용은 `docs/20260806_ChatGPT_SafeNest_mmWave_Execution_Sequence_01.md`의 전체 내용을 원문 그대로 이관한 것이다. mmWave의 세부 수치·Priority 7–18·A–E 판단 근거는 이 부분을 따른다. 상위 병렬 운용에서 기존 `A~E`는 `M-A~M-E`로 식별하며, 기존 Phase E의 fusion 작업은 세 센서가 준비된 뒤 `I` 트랙에서 실행한다.

---

# 0. 외부 에이전트 참조용 로컬 작업 공간 현황 및 디렉터리 맵 (Local Workspace & File Structure Overview)

> **[CAUTION] 작업 에이전트를 위한 안내**:
> 본 문서는 SafeNest 활성 작업 공간의 디렉터리 구조, 실측 아티팩트 해시, 모델 계보 및 실행 순서를 정의합니다. 모든 작업은 먼저 최상위 `AGENTS.md`를 읽고 이 문서의 canonical-root 규칙을 따라야 합니다.

---

### 0.1 최상위 디렉터리 및 경로 규칙
- **유일한 활성 프로젝트 루트**: 이 문서의 상위 디렉터리인 `embed2/`
- **활성 코드 위치**: `config/`, `datasets/`, `models/`, `preprocessing/`, `inference/`, `sensors/`, `integrated_node/`, `risk/`, `scripts/`, `tests/` 등 최상위 직속 경로
- **과거 버전 보존 위치 (READ-ONLY)**: `archive/version_snapshots/`
- **금지 사항**: `SafeNest_V4_*`, `SafeNest_V5_*`, `SafeNest_V6/`, `ondevice_ai/`를 별도 활성 루트로 생성하거나 archive의 코드·manifest·모델을 runtime에서 자동 선택하지 않는다.
- **경로 기록 원칙**: 활성 JSON/YAML/manifest/metadata에는 저장소 상대경로만 기록하고 사용자별 절대경로와 `file://` URI를 저장하지 않는다.
- **버전 관리 원칙**: 현재 버전은 폴더명이 아니라 model/dataset manifest, 보고서, Git tag 및 release artifact로 표현한다.

---

### 0.2 로컬 디렉터리 & 주요 파일 트리 구조 (Actual Local Tree Snapshot)

```text
embed2/
├── AGENTS.md                              # canonical-root, archive, path, phase 규약
├── config/                                # 활성 입력·센서·risk 계약
├── datasets/                              # 활성 dataset, raw archive, A0–A6 manifest
├── models/                                # 활성 모델과 명시적 historical baseline
├── preprocessing/                         # canonical/experimental 전처리
├── inference/                             # 모델 loader·interpreter
├── sensors/                               # mock·real provider contract/adapter
├── integrated_node/                       # 최상위 노드 실행·위험도 연결
├── risk/                                  # 위험도·fallback
├── scripts/                               # 현재 phase·학습·검증 실행기
├── tests/                                 # 현재 작업본 회귀 테스트
├── benchmarks/                            # 활성 기준·결과
├── docs/
│   ├── 20260806_ChatGPT_SafeNest_mmWave_Execution_Sequence_01.md
│   └── reports/
├── releases/                              # 배포 산출물; 활성 source root 아님
└── archive/
    └── version_snapshots/                 # V4/V5/구 V6 전체 스냅샷, 읽기 전용
```

---

### 0.3 주요 로컬 아티팩트 실측 해시 & 파이프라인 검증 상태

| 자산 구분 | 파일 경로 (Relative to canonical root) | 실측 SHA-256 Hash / MD5 | 보존 및 계보 상태 (Lineage Status) |
|---|---|---|---|
| **Zenodo 60GHz Raw Archive** | `datasets/raw_archives/external_datasets/db_records.zip` | SHA256: `f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0`<br>MD5: `370de95033f1a98b78e57dbbea92a8bc` | `LOCAL_REPACKAGED_ARCHIVE_CONFIRMED`<br>(110 participants, 4 posture/test conditions) |
| **V6 Processed NPZ** | `datasets/mmwave/processed/mmwave_respiration_v1.npz` | SHA256: `a08072f3d9b55cd95b530c7b5b90f17ef80f6015ee76119f217b9d834c1107fb` | `SYNTHETIC_SMOKE_AND_RETRAINING_ASSET`<br>(3,433 windows, 10Hz/30s) |
| **mmWave v0.1.0 INT8 (기존)** | `models/mmwave/mmwave_resp_int8_v0.1.0.tflite` | SHA256: `43cdd6f321c2b645232162233e098c2b65549388cbc9e680a17a0eccdc8f0158` | `HISTORICAL_SOURCE_MAPPING_INCOMPLETE`<br>(기존 외부 실데이터 개발 이력) |
| **V6 Candidate INT8** | `models/mmwave/mmwave_resp_int8_v0.2.0_candidate.tflite` | SHA256: `85c023d3eefca13ecbb72a841974e53a56d5f4173920645d46df49a9088452ff` | `SYNTHETIC_SMOKE_ONLY`<br>(Z-score: mean=0.006092, std=2.501384) |

---

### 0.4 로컬 개발 환경 검증 실행 CLI 명령어
```bash
# canonical project root(embed2) 진입
cd "<path-to-embed2>"

# 1. candidate 기술 결함 및 품질 정밀 검사 구동 (Exit Code 0 성공 검증)
python3 scripts/check_mmwave_candidate.py

# 2. V6 mmWave 파이프라인 pyTest 구동
python3 -m pytest tests/test_mmwave_v6_pipeline.py -v

# 3. candidate 재학습 및 양자화 구동 (결정성 보장 seed=42)
python3 scripts/train_mmwave.py --seed 42 --epochs 25

# 4. Mock 파이프라인 bounded 1-step smoke 테스트
python3 -c 'from integrated_node.run_node import SafeNestIntegratedNode as N; n=N(mode="mock"); n.start(); print(n.step().to_json()); n.shutdown()'
```

---

# SafeNest mmWave Priority 7–18 및 A–E 상세 실행 순서

- 작성일: 2026-08-06
- 문서 목적: Priority 6 이후 mmWave 데이터·학습·양자화·장치 도메인·멀티모달 융합 작업의 선행관계와 실행 순서를 구체화
- 대상: Zenodo 60 GHz radar 원본 재가공, SafeNest mmWave 실데이터 모델, MR60BHA2 장치 도메인, 후속 데이터 확장, 멀티모달 융합
- 내용: 최상단 Section 0에 로컬 디렉터리 맵, 실측 해시, 스크립트 실행 명령 포함. 이하 본문은 실행 순서 및 방법론 기술.

---

## 1. 핵심 결론

Priority 7부터 바로 시작하지 않는다. 가장 먼저 수행할 작업은 **A. Zenodo 실제 raw-to-NPZ pipeline 복원**이다.

현재 합성 NPZ는 학습·양자화·평가 코드의 smoke test에는 유용하지만, class 패턴이 쉽게 분리되어 성능이 포화될 수 있다. 이 상태에서 preprocessing, class imbalance, model architecture의 우열을 결정하면 실제 인체·radar domain 성능과 관계없는 결론을 얻을 수 있다.

최종 전체 순서는 다음과 같다.

```text
Priority 6 자산·gap 분석
→ A. Zenodo raw-to-NPZ 복원·무결성 감사
→ B. 실데이터 모델 실험·학습·비교
→ Priority 7–18을 실데이터 기준으로 재구성해 수행
→ C. MR60BHA2 실측 domain 검증·적응
→ D. 남은 gap을 보완할 추가 dataset 확장
→ E. 멀티모달 model·risk fusion 개선
```

---

## 2. 전 단계 공통 원칙

### 2.1 계보 분리

다음 모델은 서로 다른 lineage로 관리한다.

| 모델 | 역할 | 해석 원칙 |
|---|---|---|
| Historical v0.1.0 | 기존 외부 실데이터 개발 이력의 역사적 모델 | 사용자 확정 이력은 인정하되 exact raw-file-to-model mapping 부족은 별도 표시 |
| V6 v0.2.0 candidate | 합성 NPZ 기반 smoke·재현성 모델 | 실세계 성능 근거로 사용 금지 |
| 신규 real-data offline candidate | Zenodo 110명 계보 복원 이후 학습할 신규 모델 | real-subject offline 성능 대상 |
| MR60-adapted candidate | MR60BHA2 실측 domain을 반영한 모델 | 실제 deployment candidate. offline candidate와 분리 |

### 2.2 불가역 산출물 분리

원본에서 만들어진 canonical signal과 실험적 전처리 결과를 분리한다.

- raw rFFT에서 복원한 canonical respiration phase를 우선 보존한다.
- detrending, band-pass filtering, Z-score를 유일한 NPZ에 불가역적으로 박아 넣지 않는다.
- preprocessing ablation을 수행할 수 있도록 canonical signal과 `preprocessing_profile`을 분리한다.
- Z-score 통계는 subject split 이후 train data로만 계산한다.

### 2.3 locked test 원칙

- 전처리, imbalance, architecture, seed, calibration 선택은 train·validation으로만 수행한다.
- subject-wise test는 최종 candidate가 선정된 후 원칙적으로 한 번 사용한다.
- 여러 실험의 test 점수를 보고 configuration을 선택하지 않는다.
- v0.1.0, v0.2.0, 신규 real-data candidate의 최종 비교는 동일 locked test에서 수행한다.

### 2.4 일반 성능과 배포 성능 분리

- Zenodo offline 성능은 `OFFLINE_REAL_DATA` 또는 `REAL_SUBJECT_GENERALIZATION`으로 표시한다.
- MR60 실측 전에 `REAL_SENSOR_VALIDATION`을 주장하지 않는다.
- Mac latency를 Raspberry Pi latency 또는 sensor-to-alarm latency로 해석하지 않는다.
- 임상 apnea와 voluntary breath hold를 동일한 것으로 표현하지 않는다.

---

## 3. Phase A — Zenodo 실제 raw-to-NPZ pipeline 복원

### A0. 원본 identity·schema·inventory 고정

#### 목적

전체 변환 전에 원본 archive의 identity와 내부 recording 구조를 machine-readable inventory로 고정한다.

#### 세부 작업

1. 원본 archive의 공식 dataset identity, version, DOI, license, 바이트 크기, checksum을 기록한다.
2. 로컬 archive가 공식 archive와 byte-identical하지 않으면 공식 hash와 로컬 repackaged hash를 모두 보존한다.
3. participant, posture, activity/test, recording, radar data, timestamp, chirp config, reference signal, annotation 목록을 inventory로 만든다.
4. 누락 파일, zero/damaged frame, timestamp 역전·중복·gap, 손상 recording을 식별한다.
5. 각 recording에 고유한 `dataset_id`, `subject_id`, `session_id`, `recording_id`, `source_file_id`를 부여한다.

#### 완료 판단

- 전체 participant·recording 수와 조건별 구성을 설명할 수 있다.
- 각 rFFT가 timestamp·chirp config·annotation·reference 파일과 연결된다.
- 제외·주의 recording이 이유와 함께 별도 표시된다.

---

### A1. 안전한 rFFT reader와 소규모 pilot

#### 목적

전체 110명을 처리하기 전에 소수 participant/recording으로 schema와 signal 해석을 확정한다.

#### 세부 작업

1. rFFT container의 serialization, frame 수, array shape, dtype, complex value 여부, virtual antenna·range-bin 순서를 확인한다.
2. 외부 serialization은 출처·hash를 확인한 입력만 읽고, 임의 object execution을 허용하지 않는 방식을 선택한다.
3. chirp config에서 frame periodicity, antenna 수, range-bin 간격, 파장·주파수 정보를 읽어 recording metadata에 연결한다.
4. radar timestamp 수와 rFFT frame 수를 대조한다.
5. sitting/lying, rest/post-exercise, breath-hold 포함/미포함을 고르게 포함한 pilot subset을 선정한다.

#### 완료 판단

- pilot 모든 recording이 같은 규칙으로 decoding된다.
- frame·timestamp alignment 오류가 숫자로 기록된다.
- 전체 변환을 시작하기 전에 예외 schema가 식별된다.

---

### A2. target range-bin·phase extraction 규칙 결정

#### 목적

rFFT에서 SafeNest canonical respiration phase를 일관되고 재현 가능하게 추출한다.

#### 세부 작업

1. 탐색 가능한 거리 구간과 제외할 near-field·background bin을 정한다.
2. target bin 선택 후보를 비교한다.
   - magnitude 최대 bin
   - static clutter 제거 후 energy 최대 bin
   - respiration band 에너지 최대 bin
   - phase coherence/SNR 기반 bin
   - 인접 bin·virtual antenna 통합
3. label이나 test 결과를 보고 bin을 선택하지 않고 deterministic signal-quality rule을 사용한다.
4. complex phase 추출, unwrap, discontinuity 처리, zero/damaged frame 정책을 정한다.
5. multi-antenna 중 단일 antenna를 선택할지 coherence-weighted aggregation을 사용할지 비교한다.
6. 추출된 phase의 시간 파형, spectrum, respiration-band energy, SNR, motion indicator를 pilot에서 확인한다.

#### 중요 제약

- 0.1–0.5 Hz BPF를 canonical signal의 유일 보존본에 박아 넣지 않는다.
- filter 전 phase와 filter 후 derived profile을 구분해야 Priority 7 ablation을 수행할 수 있다.
- range-bin selection rule과 선택 결과를 sample provenance에 남긴다.

#### 완료 판단

- pilot 전반에서 respiration-related phase가 시각·스펙트럼·reference 근거로 해석 가능하다.
- 선택 규칙이 participant·posture·label에 따라 수작업으로 바뀌지 않는다.
- 실패·low-quality recording의 판정 조건이 명시된다.

---

### A3. timestamp·resampling·window 정책

#### 목적

연속 radar timeline을 SafeNest 10 Hz, 30초 canonical window로 변환하되 시간 provenance와 연속성을 잃지 않는다.

#### 세부 작업

1. config의 nominal frame period과 실제 timestamp 간격을 대조한다.
2. duplicate, backward timestamp, gap, dropped frame의 허용·제외 기준을 정한다.
3. 원본이 이미 10 Hz이면 불필요한 resampling을 하지 않는다.
4. irregular timestamp인 경우 small gap interpolation과 large gap rejection을 분리한다.
5. 30초 window와 stride를 정하고, overlap된 window가 동일 연속 recording에서 파생된 사실을 보존한다.
6. train에서 overlap augmentation을 사용하더라도 validation·test에서 과도한 상관 window가 지표를 부풀리지 않도록 non-overlap 또는 event-centered 평가를 별도 설계한다.
7. 연속 timeline을 보존해 향후 false alarms/hour, event detection delay, event miss rate를 계산할 수 있게 한다.

#### 완료 판단

- 모든 window가 source recording과 start/end timestamp로 연결된다.
- window 생성으로 인한 중복·상관 수치가 기록된다.
- gap을 조용히 보간하여 없던 호흡 신호를 만들지 않는다.

---

### A4. annotation·label mapping 정책

#### 목적

원본 test 조건과 non-breathing annotation을 SafeNest label에 의미적으로 연결한다.

#### 세부 작업

1. 원본 label·test condition, annotation timestamp, SafeNest target label, mapping 규칙을 분리한다.
2. `NORMAL`, `RAPID_OR_ABNORMAL`, `APNEA` 각각에 대해 direct/derived/ambiguous 매핑을 지정한다.
3. voluntary breath hold는 clinical apnea와 별도 원본 label로 보존하고 SafeNest APNEA로의 매핑은 `DERIVED`로 표시한다.
4. post-exercise recording 전체를 자동으로 RAPID으로 지정하지 않고, 실제 호흡률·불규칙성·reference 가용성에 기반한 파생 조건을 정한다.
5. event overlap, event-centered window, transition window, mixed window, ambiguous window 정책을 비교한다.
6. annotation 해상도보다 정밀한 label을 임의로 만들지 않는다.

#### 현재 반드시 재검토할 정책

기존 안의 “30초 window 중 non-breathing overlap이 50%, 즉 15초 이상이면 APNEA” 규칙은 그대로 고정하지 않는다. 현재 inventory 기준 breath-hold event는 대부분 약 10–11초이므로 15초 기준은 APNEA sample을 거의 제거할 수 있다.

다음을 비교한 후 정책을 선정한다.

- 10초 이상 event overlap
- window 내 event 비율+최소 event 길이 결합
- event midpoint 기준 30초 window
- transition window 학습 제외, 평가 별도 보고
- event detection 평가와 window classification 평가 병행

#### 완료 판단

- 모든 sample에 original label, SafeNest label, mapping type, overlap/duration 근거가 있다.
- label 분포와 제외·ambiguous 수치가 기록된다.
- posture·activity·recording condition artifact가 class label을 대신하지 않는다.

---

### A5. subject-wise split·sample provenance

#### 현재 상태 (2026-08-08)

- `MMWAVE_SUBJECT_SPLIT_PROFILE_001` 생성 및 검증 완료
- 110명 subject를 seed 42로 TRAIN 77 / VALIDATION 17 / LOCKED_TEST 16에 단일 배정
- 440개 recording을 subject split에 고정: TRAIN 308 / VALIDATION 68 / LOCKED_TEST 64
- 각 split의 lying/sitting × rest/post-exercise 조건 균형 확인
- subject overlap 0건, recording overlap 0건
- A4 pilot 15 windows의 `mapping_type`, `assignment_status`, label provenance 보존
- `AMBIGUOUS` window는 provenance에 남기고 pure-class training에서 제외
- 인구통계 companion metadata 미보유로 age/sex/height/weight 균형은 `NOT_VERIFIABLE`
- A5 gate: `PASS_WITH_WARNINGS`, A6 entry: `READY_WITH_CONDITIONS`

#### 목적

중첩 window를 만들기 전에 subject 단위 분할을 고정하고 모든 sample을 source에 연결한다.

#### 세부 작업

1. participant를 train, validation, test에 중복 없이 배정한다.
2. 가능하면 posture, activity, sex/age group, label event 분포를 그룹 단위로 균형화한다.
3. 동일 subject의 모든 recording·window를 하나의 split에만 배정한다.
4. split seed, grouping key, subject 목록, 배정 이유를 machine-readable manifest로 보존한다.
5. 각 window에 다음 계보를 보존한다.
   - sample/dataset/source file ID
   - subject/session/recording ID
   - posture/activity/device/environment
   - start/end timestamp
   - selected range bin·antenna·phase extraction profile
   - original/SafeNest label·mapping type
   - split·synthetic flag·quality flag

#### 완료 판단

- subject overlap 0건
- recording overlap 0건
- duplicate window hash의 cross-split overlap 0건
- 모든 NPZ index가 provenance record에 1:1로 연결

---

### A6. 전체 변환·품질 감사·A 종료 gate

#### 현재 상태 (2026-08-08)

- 전체 110명·440 recording 변환 완료: `SUCCESS` 90, `SUCCESS_WITH_WARNINGS` 350, 실패 0
- canonical real-data window 530개 생성, 각 window는 300 sample `float64`
- window/provenance/NPY 530행의 1:1 의미·신호 SHA-256 정렬 확인
- NaN/Inf/constant·near-constant window 0건
- cross-split subject·recording·window·exact-signal overlap 0건
- acquisition timestamp는 공통 수집 컴퓨터 clock 기준이며 timezone은 `UNVERIFIED`; UTC 변환을 주장하지 않음
- A6 gate: `PASS_WITH_WARNINGS`, Phase B entry: `READY_WITH_CONDITIONS`
- standalone A6 validator는 모든 A0 recording의 성공 상태·window 수를 확인하고, 530개 window/provenance/NPY 행의 식별자·label·split·eligibility·signal hash를 전수 대조한다.
- annotation read/parse 실패는 정상 label로 대체하지 않고 해당 recording을 차단하며 exception registry에 기록한다.
- checksum gate는 필수 산출물 목록의 누락·중복·형식 오류·project root 이탈을 거부한다.

#### 목적

pilot에서 확정한 규칙으로 전체 110명을 변환하고 B 단계에 사용해도 되는지 판정한다.

#### 세부 작업

1. 전체 recording에 동일한 extraction·window·label 규칙을 적용한다.
2. 처리 성공/실패/제외 수, 제외 이유, condition·subject별 신호 품질을 요약한다.
3. NaN/Inf, constant signal, extreme amplitude, zero frame, timestamp gap, low SNR을 감사한다.
4. duplicate·near-duplicate·cross-split leakage를 감사한다.
5. class·subject·posture·activity·recording 분포를 요약한다.
6. canonical processed dataset, provenance, split manifest, preprocessing/extraction config의 checksum을 고정한다.
7. 임의 수의 원본 recording에서 processed window까지 역추적하는 spot check를 수행한다.

#### A 종료 기준

- raw → canonical phase → window → label → split chain이 재실행 가능하다.
- subject/sample provenance가 machine-readable하게 보존된다.
- split·duplicate·window leakage 감사가 통과한다.
- 제외·low-quality sample이 조용히 삭제되지 않고 이유와 함께 기록된다.
- 이 기준을 충족하기 전에 B의 모델 탐색을 시작하지 않는다.

---

## 4. Phase B — 실데이터 모델 학습·비교

### B0. 평가 protocol·baseline·test lock

#### 목적

실험을 반복하며 test에 맞추는 것을 방지하고 v0.1.0, v0.2.0, 신규 모델을 비교할 공통 규칙을 먼저 정한다.

#### 세부 작업

1. train/validation/test subject 목록과 checksum을 고정한다.
   - A5의 TRAIN/VALIDATION/LOCKED_TEST 배정을 재계산하거나 변경하지 않는다.
   - scaler·normalizer·feature-selection 통계는 TRAIN에서만 fit한다.
   - architecture 비교 전 exact duplicate 감사에 더해 near-duplicate 진단을 수행한다.
2. model selection metric과 final test metric을 분리한다.
3. 필수 metric을 정한다.
   - macro F1
   - class별 precision/recall/F1
   - APNEA/breath-hold recall·miss rate
   - confusion matrix
   - class prediction distribution·collapse
   - continuous timeline이 있을 경우 false alarms/hour·event miss·detection delay
4. v0.1.0의 exact historical preprocessor가 불완전하면 현재 canonical contract에서의 결과를 “historical-model compatibility benchmark”로 표시한다.
5. v0.2.0의 real test 결과는 실데이터로 학습했다는 근거가 아니라 합성 학습 모델의 external compatibility 결과로 표시한다.
6. 신규 model이 확정되기 전에 locked test 점수를 실험 선택에 사용하지 않는다.

---

### B1. Priority 7 — preprocessing ablation

#### 실행 시점

A6 통과 후, architecture·imbalance 탐색 전에 수행한다.

#### 실험 설계

기존 4개 누적 mode만으로는 세 기법의 “독립 기여도”를 완전히 알 수 없다. 다음 두 수준 중 하나를 사전 선택한다.

#### 권장 설계 A — full factorial

Detrend, BPF, Z-score의 on/off 8개 조합을 동일 split·seed·architecture·loss에서 비교한다. main effect와 interaction을 구분할 수 있다.

#### 권장 설계 B — 최소 충분 ablation

자원을 줄여야 하면 full pipeline, no detrend, no BPF, no Z-score, raw/minimal 조건을 비교한다. 각 조건은 full pipeline에서 한 요소만 제거해 marginal effect를 본다.

#### 추가 분석

- 0.1–0.5 Hz BPF가 >30 bpm 신호를 감쇠시키는지 확인한다.
- BPF 유무 ablation과 0.1–0.5/0.1–0.8 Hz band tuning을 하나의 결론으로 섞지 않는다.
- APNEA/breath-hold처럼 거의 constant인 구간에 high-pass·detrending이 미치는 영향을 별도 본다.
- 성능 외에 saturation, signal amplitude distribution, 제외·warning 비율을 보고한다.

#### 완료 판단

- validation metric으로 preprocessing profile을 선정한다.
- test result를 보고 profile을 변경하지 않는다.
- 선정된 profile과 대안 profile의 신호·성능 trade-off가 기록된다.

---

### B2. Priority 8 — class imbalance 전략

#### 실행 시점

Priority 7에서 preprocessing profile을 고정한 후 수행한다.

#### 세부 작업

1. 실제 train split에서 class count와 subject당 event/window 수를 재계산한다.
2. 합성 NPZ에서 유도된 고정 class weight를 재사용하지 않는다.
3. 동일 split·preprocessor·architecture·seed에서 다음을 비교한다.
   - standard cross-entropy, no weighting
   - real train split에서 계산한 class weighting
   - train-only random oversampling
   - multi-class focal loss
4. oversampling은 validation/test에 적용하지 않고, subject diversity를 늘리지 않는다는 한계를 표시한다.
5. macro F1뿐 아니라 APNEA recall, precision, false positive, subject별 편차를 비교한다.
6. 임계값 선택이 필요하면 validation에서만 선정한다.

#### 완료 판단

- 소수 class recall을 높이면서 precision·false alarm이 과도하게 악화되지 않는 전략을 선정한다.
- 고정 수치가 아니라 실제 split 기반 설정과 선택 근거를 남긴다.

---

### B3. Priority 9 — TinyML architecture 비교

#### 실행 시점

preprocessing과 imbalance 전략을 일단 고정한 후 수행한다.

#### 비교 대상

- Conv1D + Global Average Pooling baseline
- SeparableConv1D 계열
- Conv1D + BiLSTM 계열: full INT8 변환 가능성을 먼저 확인하고 미지원 operator·Select TF Ops가 필요하면 TinyML 배포 후보에서 분리

#### 공정 비교 조건

- 동일 subject split
- 동일 preprocessing profile
- 동일 loss/imbalance strategy
- 동일 epoch budget·early stopping 원칙
- 동일 evaluation code·metric
- parameter count, Float/INT8 크기, validation macro F1·class recall, 변환 성공 여부 비교

#### 완료 판단

- Float 성능만 높은 모델이 아니라 full INT8 변환, footprint, recall, stability를 포함한 상위 1–2개 구조를 선별한다.
- 타겟 제약을 넘는 구조는 성능이 높아도 deployment finalist에서 분리한다.

---

### B4. Priority 10 — multi-seed 재현성

#### 실행 시점

모든 실험 조합에 수행하지 않고 Priority 9의 상위 1–2개 configuration에 수행한다.

#### 세부 작업

1. 최소 3개 training initialization seed에서 반복한다.
2. 각 seed의 training history, best epoch, validation macro F1, class recall, model checksum을 보존한다.
3. mean, standard deviation, minimum/worst-seed 성능을 보고한다.
4. initialization seed 안정성과 subject split 변화 안정성을 구분한다.
5. 실제 generalization 안정성이 중요하면 별도 subject-group split seed 또는 group cross-validation을 후속 실험으로 정의한다.

#### 완료 판단

- 평균만이 아니라 worst-seed 성능이 수용 가능한 구조를 선정한다.
- `std ≤ 0.05`, `mean F1 ≥ 0.80`같은 기준은 실제 baseline 분포를 보기 전에 불변 진리로 놓지 않고, 선정 규칙으로 사전 합의한다.

---

### B5. Priority 13 — representative dataset 구성 비교

#### 실행 시점

Float finalist가 선별된 후 INT8 candidate 생성 전에 수행한다.

#### 실험 설계

class-balanced calibration을 즉시 “개선된 정답”으로 고정하지 않고 다음을 비교한다.

- deterministic train-order baseline
- train distribution 비율을 반영한 random sample
- class-balanced sample
- amplitude·SNR·subject·condition·extreme range를 반영한 distribution-aware sample

#### 필수 기록

- train split에서만 선정
- calibration sample index·sample ID
- class·subject·condition 분포
- preprocessed tensor min/max/percentile
- input/output saturation
- Float→INT8 metric drop·output MAE·Top-1 agreement

#### 완료 판단

- class balance자체가 아니라 activation range 표현, INT8 성능, saturation 결과로 calibration profile을 선정한다.

---

### B6. Priority 12 — Float Keras → Float TFLite → INT8 equivalence

#### 실행 시점

각 finalist 및 calibration 후보에 수행한다. 최종 candidate 선정 전 필수 검사이다.

#### 세부 작업

1. 동일 validation input을 세 stage에 입력한다.
2. Keras→Float TFLite, Float TFLite→INT8의 다음을 계산한다.
   - Top-1 agreement
   - dequantized output MAE·max error
   - class별 prediction change
   - macro F1·recall drop
   - input/output saturation
3. 출력이 softmax probability이면 `logit MAE`라고 부르지 않고 probability/output MAE로 표시한다.
4. mismatch sample을 sample ID와 함께 보존해 특정 class·subject·signal range에서 변환 오차가 집중되는지 분석한다.

#### 완료 판단

- 변환 단계별 성능 하락과 오차가 기록된다.
- 사전 정한 agreement, output error, F1/recall drop, saturation 기준을 충족한다.

---

### B7. Priority 11 — input perturbation robustness

#### 실행 시점

INT8 finalist에 수행하되, MR60 실측 후 C 단계에서 device-realistic perturbation으로 반복한다.

#### 세부 작업

1. 교란 주입 지점을 canonical phase 전·후 중 명시한다.
2. 다음 교란을 독립 및 필요 시 결합 조건에서 평가한다.
   - Gaussian noise: SNR 20 dB, 10 dB 등
   - amplitude scaling
   - baseline drift
   - short/long dropout
   - timestamp jitter·missing frame
   - motion burst·outlier
3. 각 교란의 정의, random seed, SNR 계산 방식, dropout mask를 보존한다.
4. clean 대비 macro F1·class recall 하락, collapse, saturation, confidence 변화를 보고한다.
5. BPF·detrending이 당연히 제거하는 교란만으로 robustness를 과대평가하지 않는다.

#### 완료 판단

- clean 성능과 교란별 성능 하락이 비교된다.
- 변형 불가능/위험 조건은 모델 추론 대신 invalid/fallback으로 처리할지 결정한다.
- 이 결과를 실센서 robustness로 표현하지 않는다.

---

### B8. Priority 14 — Mac offline latency·footprint

#### 실행 시점

구조 후보 상대 비교와 finalist 확인 단계에 수행한다.

#### 측정 조건

- warm-up 후 반복 측정
- 단일 interpreter 재사용
- thread 수, delegate, runtime/version, CPU 환경 기록
- model invoke-only latency와 preprocessing+quantization+invoke latency 분리
- mean, median, P95, P99, min/max
- TFLite 파일 크기, parameter 수, 가능하면 peak memory

#### 해석 원칙

- 100회는 최소 smoke 측정으로 보고 안정적 percentile에 필요한 반복 수를 늘릴 수 있다.
- `<5 ms`, `P99 <15 ms`는 Mac 개발 기준일 뿐 Pi 5·end-to-end 성능을 보장하지 않는다.
- 30초 window startup latency와 model invoke latency를 분리한다.

#### 완료 판단

- 모델별 동일 환경 상대 지연·크기 비교가 가능하다.
- 실측 환경과 측정 범위가 결과에 포함된다.

---

### B9. Priority 15 — Mock end-to-end integration

#### 실행 시점

선정 전 finalist가 runtime에서 실제로 로드될 수 있는지 검증한다.

#### 필수 조건

1. 테스트가 명시적으로 해당 finalist model·metadata·checksum을 선택해야 한다.
2. 기존 runtime default model을 로드한 것을 finalist 통합 성공으로 판정하지 않는다.
3. 현재 지원하지 않는 `--steps` 같은 명령을 완료 조건으로 쓰지 않고 bounded test harness 또는 명시적 종료 조건을 준비한다.
4. NORMAL, RAPID_OR_ABNORMAL, APNEA, invalid/fault, missing/stale 조건을 포함한다.
5. 다음을 검증한다.
   - actual loaded model ID/version/checksum
   - fallback 사용 여부·이유
   - input window contract
   - `InferenceResult` class/score/confidence/latency/valid/error
   - risk input·JSON output
   - timeout·stale·sensor fault 처리

#### 완료 판단

- finalist의 checksum이 runtime metadata와 일치한다.
- 모든 시나리오가 예외 중단 없이 올바른 valid/fallback/fault 계약으로 종료된다.
- scenario name으로 정답 score를 강제한 결과와 모델이 실제로 만든 prediction을 구분한다.

---

### B10. Priority 16 — real-data offline candidate 선정

#### 선정 전 필수 산출물

- preprocessing ablation table
- imbalance comparison
- architecture comparison
- multi-seed stability
- representative calibration comparison
- Float/Float TFLite/INT8 equivalence
- perturbation robustness
- latency·footprint
- Mock E2E 결과

#### 선정 규칙

candidate 선정 방법을 최종 test를 보기 전에 고정한다. 다음을 함께 본다.

- validation macro F1
- APNEA/breath-hold recall·precision
- subject별 worst-case 성능
- seed 분산
- Float→INT8 drop·agreement·saturation
- robustness 하락
- model size·latency
- runtime 호환성

단순히 F1이 가장 높은 모델을 선정하지 않는다. APNEA recall이 0이거나 class collapse, 과도한 saturation, runtime 미지원, lineage 불일치가 있는 후보는 제외한다.

#### 최종 test

확정된 후보 하나를 locked subject-wise test에 평가한다. 동일 test에서 v0.1.0, v0.2.0 candidate, 신규 real-data candidate를 비교한다. 이 결과를 보고 다시 7–13을 tuning하지 않는다. 필요하면 새 실험 cycle과 새 holdout 정책을 명시한다.

#### 완료 판단

- 하나의 **Real-Data Offline Candidate**가 근거와 함께 선정된다.
- 이 후보를 MR60 deployment 최종 모델으로 즉시 선언하지 않는다.

---

### B11. Priority 17 — offline candidate artifact lock

#### 고정 항목

- raw archive/dataset identity·checksum
- processed dataset·provenance·split manifest checksum
- extraction·preprocessing profile/version
- label mapping/version
- scaler mean/std·clip·filter
- training config·seed·environment
- Keras·Float TFLite·INT8 checksum
- representative dataset identity·indices
- input/output tensor contract·class map
- validation/test metric·scope
- runtime role·fallback·known limitations

#### 완료 판단

- manifest·metadata·artifact의 path, checksum, scaler, class map, contract가 일치한다.
- 이전 v0.1.0·v0.2.0 lineage를 덮어쓰지 않는다.
- 상태를 `REAL_DATA_OFFLINE_CANDIDATE`에 상응하게 분리하고 MR60 실센서 검증 완료로 표현하지 않는다.

---

### B12. Priority 18 — 실데이터 offline 검증 보고

#### 필수 내용

1. raw-to-NPZ lineage 요약
2. participant·recording·window·class·split 통계
3. 제외·low-quality·ambiguous sample 통계
4. Priority 7–15 실험 비교표
5. v0.1.0 vs v0.2.0 vs real-data candidate 최종 test
6. Float/TFLite/INT8 lineage·equivalence
7. robustness·latency·Mock E2E
8. 선정·제외 candidate 이유
9. `REAL_SUBJECT_GENERALIZATION`, `REAL_SENSOR_VALIDATION`, `BLOCKED_HARDWARE`, `NOT_VERIFIABLE` 범위 분리
10. C 단계 MR60 인수인계 조건

#### 완료 판단

- 합성 smoke 성과와 실데이터 성과가 분리된다.
- 실제 실행·실측한 수치만 포함한다.
- 외부 검토자가 최종 candidate의 source-to-runtime chain을 확인할 수 있다.
- C 단계에서 재검증할 항목이 명시된다.

---

## 5. Phase C — MR60BHA2 실측 device-domain 검증

### C0. 하드웨어 가용성 gate

- MR60BHA2, data capture 경로, 전원, timestamp 기준, 안정적 recording 환경이 없으면 `BLOCKED_HARDWARE`로 표시한다.
- hardware가 없다고 Mac 가능 작업을 중단하지 않고 D의 gap-driven dataset 조사·adapter 설계를 병행할 수 있다.

### C1. MR60 canonical signal contract 확정

1. 장치가 제공하는 total phase, breath phase, heart phase, breath rate, distance, presence 중 실제 사용 가능한 필드를 확인한다.
2. API update timeout과 실제 sample interval을 동일하게 가정하지 않고 timestamp로 실측한다.
3. breath phase의 단위, scale, firmware filter, clipping, smoothing, missing value, reset behavior을 확인한다.
4. Zenodo의 range-bin-derived phase와 MR60 firmware-derived breath phase가 같은 semantics인지 평가한다.
5. shape·10 Hz·30초가 같아도 semantics가 다르면 source-specific preprocessing/adaptation을 적용한다.

### C2. 제어된 MR60 수집

최소한 다음 변수를 분리할 수 있는 수집 계획을 세운다.

- subject/session
- distance
- sensor angle·height
- posture: lying/sitting 등
- normal breathing·rapid breathing·breath hold
- body movement·position change·background movement
- blanket/clothing·environment clutter
- firmware/device version

사람 대상 수집은 동의·개인정보·보관 정책을 프로젝트 운영 기준에 맞게 적용한다. voluntary breath hold를 임상 apnea 데이터로 표현하지 않는다.

### C3. domain gap 계량

Zenodo train/validation/test와 MR60 수집 데이터에서 다음을 비교한다.

- sample interval·gap·jitter
- amplitude·phase range·percentile
- respiration-band spectrum
- SNR·motion artifact·dropout
- distance·angle·posture별 분포
- preprocessing 후 scaler range·clipping·INT8 saturation
- existing candidate의 confidence·class distribution·recall·false alarm

### C4. adaptation 전략 선택

가장 작은 변경으로 시작한다.

1. external test only
2. MR60-specific input adapter
3. device-specific scaler
4. source-specific preprocessing profile
5. fine-tuning
6. joint retraining
7. domain adaptation·multi-stage training

MR60 sample이 적거나 subject diversity가 부족하면 최종 test를 학습에 사용하지 않고 calibration/fine-tuning/evaluation 세트를 분리한다.

### C5. MR60 candidate 재검증·고정

adaptation으로 model·scaler·preprocessor·contract이 바뀌었다면 최소한 다음을 반복한다.

- multi-seed 안정성
- Float/TFLite/INT8 equivalence
- representative calibration
- MR60-realistic robustness
- target runtime latency
- Mock/real adapter E2E
- candidate quality check
- artifact lock·report

C 종료 전에만 `MR60_REAL_SENSOR_VALIDATED` 또는 해당하는 deployment candidate 상태를 사용한다.

---

## 6. Phase D — gap-driven 추가 dataset 확장

### 시작 조건

A/B의 real-subject 결과와 가능하면 C의 MR60 domain 결과를 먼저 본다. “좋아 보이는 공개 호흡 dataset”이 아니라 확인된 실패 조건을 채우는 dataset만 선정한다.

### gap 예시

- MR60 device domain
- distance·angle·posture
- motion·cough·position change·background movement
- low SNR·dropout·multipath
- rapid·irregular·shallow breathing
- apnea/breath-hold event 수·길이
- subject age·body type·health diversity
- continuous session·event timeline

### dataset별 용도 선정

각 dataset을 다음 중 하나 이상으로 지정한다.

- source-only benchmark
- external test only
- joint retraining
- fine-tuning
- domain adaptation
- reference-domain only

비레이더 생리 신호는 별도 전이 전략이 없는 한 radar phase dataset에 직접 병합하지 않는다.

### 진입 절차

1. gap→candidate→intended role 정의
2. source·license·waveform·provenance 검증
3. 사용자 승인
4. 원본 archive 보존·checksum
5. source-specific adapter·canonical contract 변환
6. 기존 dataset과 분리된 무결성 감사
7. source-only/external test
8. 필요한 경우에만 retraining/fine-tuning
9. 기존 candidate와 동일 protocol 비교

---

## 7. Phase E — 멀티모달 model·risk fusion 개선

### 시작 조건

- mmWave 개별 모델의 real-data 입출력 계약과 failure condition이 안정됨
- Thermal, CO₂, PIR의 timestamp·valid·stale·error·confidence 계약이 일관됨
- sensor 간 시간 정렬 방법이 정의됨
- fusion 평가에 사용할 실제 scenario·event label이 있음

### 단계별 접근

1. **Late-fusion baseline**
   - 기존 sensor별 score·valid·confidence·stale 입력을 사용
   - 우선 rule-based fusion의 오탐·미탐·fault isolation을 측정
2. **Calibration**
   - sensor별 score/confidence calibration
   - missing sensor·stale·fallback 조건 처리
3. **Scenario evaluation**
   - normal, fall, apnea/breath anomaly, elevated CO₂, no motion, sensor fault, 복합 상황
4. **Weight/logic tuning**
   - 실제 validation scenario에서만 조정
   - synthetic scenario 성공만으로 실제 fusion 개선을 주장하지 않음
5. **Learned fusion 검토**
   - rule-based baseline의 한계가 확인되고 충분한 synchronized data가 있을 때만 후보로 추가

### 필수 평가

- hazard·scenario별 recall·precision
- false alarms/hour
- event detection delay
- sensor dropout·fault 주입
- risk output과 system health 분리
- calibration·confidence reliability
- end-to-end latency

### 완료 판단

- 개별 sensor 오류가 정상 risk 0으로 바뀌지 않는다.
- 복합 상황의 개선이 개별 modality 성능 저하를 숨기지 않는다.
- learned fusion이 rule baseline보다 실제 holdout에서 일관된 이득을 보일 때만 채택한다.

---

## 8. 중간 gate와 사용자 결정 지점

| Gate | 확인 대상 | 통과 후 다음 작업 | 실패 시 |
|---|---|---|---|
| G0 | Priority 6 asset·gap 분석 | A0 | 불일치 정정 후 재시작 |
| G1 | pilot rFFT decoding·phase 타당성 | A3–A6 | reader·bin·phase rule 수정 |
| G2 | full NPZ provenance·split·integrity | B0 | model 탐색 중단, dataset 문제 수정 |
| G3 | validation 기반 finalist | Priority 12·11·14·15 | preprocessing/loss/architecture 후보 재검토 |
| G4 | Real-Data Offline Candidate | C | offline 한계를 보고하고 실험 cycle 재정의 |
| G5 | MR60 domain 실측 | adaptation 또는 D | device contract·capture 수정 |
| G6 | MR60-adapted candidate | D/E | 실센서 한계 유지 |

다음은 별도 승인·결정 지점으로 본다.

- label mapping 정책 확정
- subject split 고정
- locked test 최초 평가
- offline candidate 선정·manifest 등록
- MR60 사람 대상 수집
- 추가 외부 dataset 다운로드
- offline candidate를 deployment candidate로 승격
- learned multimodal fusion 도입

---

## 9. agent 작업 단위 권장

하나의 실행 프롬프트에 너무 많은 판단·변경을 섞지 않는다. 다음처럼 독립 작업으로 나누는 것을 권장한다.

1. A0 inventory·source identity
2. A1 safe reader·schema pilot
3. A2 range-bin·phase extraction pilot
4. A3 timestamp·window policy
5. A4 label policy 분석·결정
6. A5 subject split·provenance schema
7. A6 full conversion·integrity audit
8. B0 evaluation protocol·test lock
9. Priority 7 preprocessing ablation
10. Priority 8 imbalance
11. Priority 9 architecture
12. Priority 10 multi-seed
13. Priority 13 representative calibration
14. Priority 12 stage equivalence
15. Priority 11 robustness
16. Priority 14 latency·footprint
17. Priority 15 Mock E2E
18. Priority 16 selection
19. Priority 17 artifact lock
20. Priority 18 report
21. C1 MR60 contract·capture design
22. C2 MR60 capture
23. C3 domain analysis
24. C4 adaptation
25. C5 deployment candidate 재검증
26. D dataset gap 확장
27. E fusion baseline·개선

각 작업 프롬프트는 최소한 다음을 포함하도록 구체화한다.

- 정확한 목적·비목적
- 선행 산출물·입력
- 수정 허용 범위·금지 범위
- 실행 방법·실험 변수·고정 변수
- machine-readable output schema
- metric·판정 기준
- 실패·부족 evidence 표기
- lineage·checksum·provenance 요구사항
- 수행하지 않을 검증
- 완료 보고 형식

---

## 10. 최종 순서 checklist

### Phase A

- [x] A0 archive identity·inventory
- [x] A1 safe rFFT reader·pilot
- [x] A2 range-bin·phase extraction
- [x] A3 timestamp·resampling·window
- [x] A4 annotation·label mapping pilot
- [x] A5 subject split·pilot sample provenance
- [x] A6 full conversion·integrity audit

### Phase B / Priority 7–18

- [ ] B0 evaluation protocol·locked test
- [ ] Priority 7 preprocessing ablation
- [ ] Priority 8 imbalance strategy
- [ ] Priority 9 architecture comparison
- [ ] Priority 10 multi-seed stability
- [ ] Priority 13 representative calibration
- [ ] Priority 12 Float/TFLite/INT8 equivalence
- [ ] Priority 11 perturbation robustness
- [ ] Priority 14 Mac latency·footprint
- [ ] Priority 15 explicit candidate Mock E2E
- [ ] Priority 16 Real-Data Offline Candidate selection
- [ ] Priority 17 artifact·metadata·manifest lock
- [ ] Priority 18 offline validation report

### Phase C

- [ ] MR60 hardware gate
- [ ] MR60 canonical signal contract
- [ ] controlled MR60 capture
- [ ] Zenodo↔MR60 domain-gap analysis
- [ ] adaptation strategy
- [ ] MR60 candidate rerun·lock·report

### Phase D

- [ ] residual gap ranking
- [ ] gap-driven external dataset selection
- [ ] approval·acquisition·source audit
- [ ] external test/adaptation/retraining

### Phase E

- [ ] synchronized multimodal evaluation data
- [ ] rule-based late-fusion baseline
- [ ] calibration·fault robustness
- [ ] scenario holdout evaluation
- [ ] learned fusion conditional comparison

---

## 11. 최종 종료 조건

전체 로드맵은 다음을 모두 충족할 때 완료로 본다.

1. Zenodo raw→canonical phase→window→label→split→model chain이 checksum·provenance와 함께 재현된다.
2. real-subject locked test에서 v0.1.0, v0.2.0, 신규 model이 동일 계약으로 비교된다.
3. preprocessing, imbalance, architecture, seed, calibration, conversion 선택의 근거가 validation 실측으로 남아 있다.
4. 최종 INT8 model의 quantization equivalence, robustness, latency, runtime 연결이 검증된다.
5. offline candidate와 MR60 deployment candidate가 분리되어 있다.
6. MR60 실측 결과나 `BLOCKED_HARDWARE`가 정직하게 보고된다.
7. 추가 dataset이 실제 gap을 보완하는 용도로만 통합된다.
8. multimodal fusion이 개별 sensor 오류를 숨기지 않고 실제 holdout에서 개선을 보인다.

---

# Part III — CO₂ 실제 데이터·모델 A–E 트랙

## C-A. 실제 CO₂ raw-to-canonical reconstruction

### C-A0. source identity·license·inventory

#### 목적

현재 synthetic smoke NPZ와 실제 UCI Occupancy 원본을 분리하고, 이후 모든 CO₂ sample이 실제 source row로 역추적되게 한다.

#### 세부 작업

1. 공식 source URL, DOI, license, 내려받은 파일명과 byte size를 기록한다.
2. raw 파일별 SHA-256을 streaming 방식으로 계산한다.
3. CSV header, delimiter, encoding, timestamp format, row count, feature 목록, label 목록을 inventory로 만든다.
4. duplicate row, missing value, nonfinite value, timestamp 역행, 비정상 범위를 원본별로 계수한다.
5. 기존 `datasets/build_processed_npz.py`의 synthetic fixture 생성 결과를 실제 UCI 처리 결과로 설명하지 않는다.
6. 원본 파일은 수정하지 않고 `datasets/raw_archives/` 또는 승인된 외부 storage에 read-only로 둔다.

#### 완료 판단

- source identity와 license가 명시됨
- raw 파일 전부 checksum 보유
- inventory가 실제 row에서 derive됨
- synthetic/real 계보 혼동 0건

### C-A1. safe CSV reader·schema contract

#### 목적

원본별 column 차이와 timestamp 해석을 명시적으로 처리하는 deterministic reader를 만든다.

#### 세부 작업

1. 필수 column과 optional column을 분리한다.
2. timestamp는 원본 문자열, parse 결과, timezone 검증 상태를 함께 보존한다.
3. label·feature column을 position이 아니라 이름으로 선택한다.
4. NaN/Inf, 잘못된 숫자, 결측 timestamp, 중복 header를 fail-closed 처리한다.
5. reader가 원본 row index와 source file ID를 잃지 않게 한다.
6. 소규모 pilot 파일에서 row 수·범위·label 분포를 독립 validator로 재계산한다.

### C-A2. timeline·session·group contract

#### 목적

인접한 시간 row가 train과 test에 섞이는 temporal leakage를 막는다.

#### 세부 작업

1. 연속 timestamp 구간을 session 또는 contiguous block으로 식별한다.
2. 큰 gap, 중복 timestamp, 역행을 기록하고 block 경계를 고정한다.
3. room/building/day/session 정보가 있으면 grouping evidence로 보존한다.
4. 정확한 독립 group을 원본만으로 확인할 수 없으면 `NOT_VERIFIABLE`로 표시하고 row-random split을 금지한다.
5. rolling history를 만들기 전에 group split 단위를 먼저 결정한다.

### C-A3. feature reconstruction contract

#### 목적

`[CO2_slope, Humidity, CO2]` 입력이 실제 raw measurement에서 어떻게 생성됐는지 재현한다.

#### 세부 작업

1. CO₂ ppm, humidity와 원본 sample interval을 검증한다.
2. `CO2_slope`의 window 길이, 단위 `ppm/min`, 회귀 또는 차분 계산식을 고정한다.
3. feature history가 session 경계를 넘지 않게 한다.
4. warm-up 부족 row, gap 인접 row, 센서 범위 초과 row의 포함·제외 규칙을 기록한다.
5. canonical feature는 unscaled 값으로 보존하고 scaler는 B 단계에서 TRAIN으로만 fit한다.
6. 임의 pilot row를 원본 CSV에서 canonical feature까지 수작업 역산한다.

### C-A4. label semantics·safety separation

#### 목적

공개 dataset의 occupancy label, 실제 CO₂ 농도, SafeNest 위험 상태를 서로 다른 의미로 보존한다.

#### 세부 작업

1. original occupancy label을 그대로 보존한다.
2. occupancy를 고농도 위험 또는 사람 안전 상태와 동일시하지 않는다.
3. `CO2 > 1500 ppm` 같은 rule-based safety threshold는 모델 label과 분리된 risk rule로 관리한다.
4. label source, mapping type, mapping rule ID와 assignment status를 sample provenance에 둔다.
5. 불확실하거나 label이 없는 row는 억지로 NORMAL로 만들지 않는다.

### C-A5. group-wise split·sample provenance

#### 목적

시간적으로 이어진 row와 history window가 여러 split에 섞이지 않게 한다.

#### 세부 작업

1. A2에서 승인된 room/session/day/block 중 가장 강한 독립 단위를 split group으로 사용한다.
2. 같은 group의 모든 row와 window를 TRAIN, VALIDATION, LOCKED_TEST 중 하나에만 둔다.
3. seed, hash assignment rule, group 목록과 split 비율을 manifest로 고정한다.
4. label·농도 범위·session 길이의 split 분포를 보고하되 LOCKED_TEST를 최적화에 사용하지 않는다.
5. sample마다 source file, row, timestamp, group, feature profile, label, split, quality를 보존한다.

### C-A6. full conversion·integrity audit

#### 목적

전체 실제 CO₂ 원본을 canonical feature dataset으로 변환하고 C-B 진입 여부를 결정한다.

#### 세부 작업

1. 전체 row/session 처리 성공·경고·실패·제외를 기록한다.
2. NaN/Inf, constant feature, 범위 이상, timestamp gap, duplicate와 cross-split leakage를 감사한다.
3. canonical numeric artifact와 provenance를 1:1 검증한다.
4. raw archive와 output manifest checksum을 고정한다.
5. 여러 source file과 split에서 deterministic lineage spot check를 수행한다.
6. standalone C-A6 validator가 모든 필수 artifact와 split inheritance를 검사한다.

#### C-A 종료 기준

- 실제 raw→feature→label→split chain 재현 가능
- temporal/group leakage 0건
- synthetic fixture와 실제 canonical artifact 분리
- 모든 제외·실패에 이유 존재
- LOCKED_TEST training eligibility 0건

## C-B. CO₂ offline model comparison

### C-B0. evaluation protocol·baseline lock

- primary metric: macro F1과 occupied recall
- secondary metric: precision, specificity, calibration, confusion matrix
- 농도 구간·session별 error slice를 사전 정의
- rule baseline, logistic regression, tree/boosting baseline, 현재 TFLite를 같은 split로 비교
- LOCKED_TEST 실행 횟수와 접근 주체 기록

### C-B1. preprocessing·history ablation

- raw CO₂/humidity와 slope 조합 비교
- history 길이와 slope 계산법 비교
- missing/gap 처리 비교
- 모든 scaler·imputer는 TRAIN에서만 fit
- validation protocol 고정 후 한 요소씩 비교

### C-B2. imbalance·calibration

- class weight, balanced sampling, threshold calibration을 분리 비교
- threshold는 VALIDATION에서만 선택
- false-negative와 false-positive 비용을 모두 보고
- occupancy probability와 CO₂ safety rule을 별도 출력으로 유지

### C-B3. architecture·multi-seed

- linear/MLP/tree 계열과 TinyML 후보를 공정 비교
- 입력 feature와 split, seed set, stopping rule을 고정
- 최소 3개 seed의 평균·표준편차와 worst seed 기록
- 단일 최고 seed만으로 후보를 선택하지 않음

### C-B4. Float→TFLite→INT8 equivalence

- 대표 dataset은 TRAIN에서만 구성
- float model, float TFLite, INT8의 동일 sample prediction을 비교
- macro F1, occupied recall, probability drift, input saturation을 기록
- 변환 artifact·scaler·class map·metadata checksum 고정

### C-B5. robustness·latency·candidate lock

- ppm offset/drift, humidity noise, missing row, stale history, timestamp jitter 시험
- Mac latency는 기술 sanity check로만 기록
- 최종 locked-test 평가는 model·threshold·artifact 확정 뒤 한 번 수행
- offline candidate와 SCD40 deployment candidate를 분리

## C-C. SCD40 device-domain validation

1. 실제 출력 column, unit, sampling interval, warm-up, self-calibration 상태를 기록한다.
2. 공개 dataset feature와 SCD40 feature의 범위·분포·drift 차이를 측정한다.
3. controlled empty/occupied/ventilation scenario를 안전하게 수집한다.
4. missing, stale, disconnect, out-of-range를 fail-closed로 검증한다.
5. Pi 5에서 history build·inference latency와 장시간 안정성을 측정한다.
6. domain gap이 크면 C-D에 구체적 gap hypothesis를 전달한다.

## C-D. gap-driven CO₂ dataset expansion

- 공개 dataset에 없는 SafeNest 환경·환기·warm-up·drift 조건만 추가한다.
- 개인 식별 정보 없이 room/session/condition provenance를 보존한다.
- adaptation set과 external test set을 분리한다.
- 기존 LOCKED_TEST를 새 수집 데이터와 섞거나 재분할하지 않는다.

## C-E. CO₂ artifact lock·integration readiness

- model, scaler, feature profile, class map, threshold, checksum 고정
- offline/SCD40/Pi 검증 상태 분리
- provider output state와 metadata contract 고정
- 실제 검증이 없으면 `BLOCKED_HARDWARE` 또는 `NOT_VERIFIABLE` 유지

---

# Part IV — Thermal 실제 데이터·모델 A–E 트랙

## T-A. 실제 Thermal raw-to-canonical reconstruction

### T-A0. dataset selection·source identity

#### 목적

실제 frame/sequence와 subject/session provenance를 제공하는 thermal dataset을 선정한다.

#### 후보 평가 기준

1. 공식 배포처, license, checksum 제공 가능성
2. thermal 원본인지 RGB 변환 이미지뿐인지
3. radiometric value·unit·sensor 정보 보유 여부
4. subject, scene, session, sequence, camera ID 보유 여부
5. fall/normal label 정의와 event boundary 품질
6. SafeNest Thermal-44 해상도·시야와 비교 가능한지
7. 연구·재배포·모델 학습 사용 조건

선정 전에는 dataset 다운로드 수나 논문 metric만으로 적합 판정을 하지 않는다.

### T-A1. safe reader·raw unit contract

1. 지원 파일 형식, frame dtype, shape, channel, endianness를 명시한다.
2. radiometric temperature, sensor count, normalized image를 구분한다.
3. corrupt/truncated file, shape mismatch, NaN/Inf, saturation을 fail-closed 처리한다.
4. source sequence·frame index·timestamp를 보존한다.
5. 원본을 8-bit visualization로 바꾼 값을 canonical physical input이라고 부르지 않는다.

### T-A2. geometry·calibration·canonical frame

1. source orientation, rotation, crop, resize, invalid pixel 처리 규칙을 고정한다.
2. 현재 model input `80×62`와 실제 Thermal-44 frame orientation을 명시한다.
3. resize가 필요하면 interpolation과 aspect ratio 정책을 기록한다.
4. canonical raw/physical frame과 모델용 normalized tensor를 분리한다.
5. ambient/reference compensation을 사용할 경우 식과 parameter source를 기록한다.
6. visual spot check와 numeric inverse trace를 함께 수행한다.

### T-A3. sequence·window·event policy

1. frame rate와 timestamp reliability를 측정한다.
2. fall event 전·중·후 범위와 window 길이를 고정한다.
3. 한 event의 인접 frame이 여러 split으로 나뉘지 않게 한다.
4. frame 단위 sample과 sequence/event 단위 sample을 구분한다.
5. dropped frame, 큰 gap, duplicate frame의 처리와 exception을 기록한다.

### T-A4. label semantics·ambiguity

1. original activity/fall annotation을 보존한다.
2. staged fall, lying, sitting, bending, entering/exiting을 구분한다.
3. transition·경계 frame은 `AMBIGUOUS` 또는 별도 transition 상태로 보존한다.
4. posture를 곧바로 fall label로 치환하지 않는다.
5. 위험한 실제 낙상 시험을 요구하지 않으며 공개 데이터와 안전한 연출만 사용한다.

### T-A5. subject/session/event-wise split

1. subject를 우선 group으로 사용하고 불가능하면 scene/session/event hierarchy를 사용한다.
2. 동일 subject·event·연속 sequence의 cross-split overlap을 금지한다.
3. augmentation 파생본은 원본과 같은 TRAIN group에만 둔다.
4. split별 subject, event, activity, scene, camera 분포를 보고한다.
5. group provenance가 없으면 일반화 성능은 `NOT_VERIFIABLE`로 제한한다.

### T-A6. full conversion·integrity audit

1. 전체 file/sequence/frame 성공·경고·실패·제외를 기록한다.
2. corrupt, nonfinite, constant, saturated, duplicate/near-duplicate frame을 감사한다.
3. canonical tensor, label manifest, provenance를 sample index로 1:1 검증한다.
4. subject/session/event cross-split leakage를 독립 재계산한다.
5. 원본과 output checksum, preprocessing profile을 고정한다.
6. 실제 dataset이 준비되기 전 존재하던 synthetic fixture와 경로·claim을 분리한다.

#### T-A 종료 기준

- 실제 source→frame/sequence→label→split chain 재현 가능
- subject/session/event leakage 0건 또는 제한이 명시됨
- 실제 evaluation artifact 존재로 thermal skip 원인이 해소됨
- 모든 quality/exclusion evidence 보존

## T-B. Thermal offline model comparison

### T-B0. evaluation protocol·baseline

- frame metric과 event metric을 분리
- primary: fall event recall, macro F1
- secondary: false alarms per hour/sequence, detection delay, normal activity specificity
- subject·scene·activity별 error slice 사전 정의
- rule/image baseline과 현재 TFLite, 신규 후보를 동일 split로 비교

### T-B1. preprocessing·augmentation ablation

- raw temperature/relative temperature/normalized tensor 비교
- crop, resize, background compensation 비교
- augmentation은 TRAIN에만 적용하고 원본 group을 상속
- 실제 validation improvement 없는 복잡한 pipeline을 채택하지 않음

### T-B2. imbalance·hard-negative strategy

- class weight, event-balanced sampling, focal loss를 분리 비교
- lying/bending/sitting/partial body를 hard negative로 분석
- false alarm 감소가 fall recall 붕괴를 숨기지 않게 함

### T-B3. frame vs temporal architecture

- small CNN, temporal pooling, lightweight sequence model을 공정 비교
- parameter, MACs, input history, latency를 함께 기록
- 동일 split·seed·early stopping으로 비교
- 최소 3개 seed 평균·worst case 기록

### T-B4. Float→TFLite→INT8 equivalence

- representative frames/sequences는 TRAIN에서만 선택
- float/TFLite/INT8 prediction·event decision parity 측정
- quantization saturation과 온도 범위별 error 기록
- 모델·preprocessing·class map·metadata checksum 고정

### T-B5. robustness·latency·candidate lock

- ambient offset, dead pixel, partial occlusion, hot object, missing frame, orientation 오류 시험
- Mac와 Pi latency를 분리
- locked-test는 최종 후보 확정 후 한 번 실행
- offline candidate와 Thermal-44 deployment candidate 분리

## T-C. Thermal-44 device-domain validation

1. 실제 frame shape, orientation, dtype, temperature conversion, invalid pixel semantics를 고정한다.
2. public dataset과 Thermal-44의 range·noise·FOV·ambient distribution 차이를 측정한다.
3. 정상 활동과 안전한 연출 scenario를 수집해 false alarm과 sequence behavior를 확인한다.
4. disconnect, partial frame, stale frame, NaN/Inf, sensor warm-up을 fail-closed로 검증한다.
5. Pi 5에서 acquisition+preprocess+inference latency, memory, 장시간 안정성을 측정한다.

## T-D. gap-driven Thermal dataset expansion

- C에서 확인된 FOV, ambient, occlusion, activity hard negative만 보강한다.
- 사람·session·event provenance와 촬영 동의를 보존한다.
- adaptation subject와 external holdout subject를 분리한다.
- 실제 위험 낙상이나 부상 가능 시험을 금지한다.

## T-E. Thermal artifact lock·integration readiness

- model, input geometry, normalization, event policy, threshold, checksum 고정
- offline/Thermal-44/Pi evidence 상태 분리
- provider가 raw frame과 inference result의 책임 경계를 명확히 구현
- 실제 hardware evidence가 없으면 deployment 승격 금지

---

# Part V — Multisensor Integration I 트랙

## I-0. shared contract inventory·freeze proposal

### 시작 시점

M-B0, C-A0, T-A0와 병렬로 읽기·설계 작업만 수행할 수 있다.

### 세부 작업

1. standalone provider contract, 팀 `shared/contracts/`, `devices/*/src`, integration node의 차이를 inventory한다.
2. `connect()`, `read()`, `close()`, sensor ID, timestamp, validity, error, latency, metadata schema를 비교한다.
3. `WARMING_UP`, `STALE`, `NOT_CONNECTED`, `INFER_FAILED`, hardware unavailable 상태를 fail-closed로 정의한다.
4. 실제 sensor driver ownership은 `devices/`, 공용 interface는 `shared/contracts/`, inference·risk orchestration은 `ondevice_ai/`로 유지한다.
5. contract proposal만으로 기존 device code나 firmware를 일괄 변경하지 않는다.

## I-1. sensor input/output contract conformance

### 시작 조건

- 각 센서 canonical input contract가 최소 pilot 수준으로 고정됨
- 팀 담당자와 provider ownership 합의

### 검증 항목

- mmWave: phase unit·sampling·window·gap가 M-A/M-C와 일치
- CO₂: ppm/humidity/slope unit·history·warm-up가 C-A/C-C와 일치
- Thermal: frame shape·orientation·unit·invalid pixel가 T-A/T-C와 일치
- 모든 provider: invalid/stale/missing을 정상값으로 합성하지 않음

## I-2. deterministic replay simulation

1. 실제 source-derived canonical replay와 synthetic fault fixture를 구분한다.
2. 각 node가 동일 timestamp contract로 replay되게 한다.
3. 정상, 단일 이상, 동시 이상, missing, stale, disconnect, out-of-order scenario를 실행한다.
4. replay output을 checksum 가능한 JSONL로 기록한다.
5. mock wiring 성공을 실제 sensor performance로 설명하지 않는다.

## I-3. rule-based fusion baseline

1. 현재 risk rule과 emergency override를 frozen baseline으로 평가한다.
2. 센서별 score calibration 차이를 확인한다.
3. missing sensor reweighting과 system health를 사람 위험도와 분리한다.
4. 시나리오 holdout에서 false alarm, missed critical event, detection delay를 기록한다.
5. validation 근거 없이 weight와 threshold를 조정하지 않는다.

## I-4. learned fusion conditional comparison

### 시작 조건

- synchronized 실제 multi-sensor scenario data 존재
- 개별 sensor candidate와 calibration 고정
- subject/scenario holdout 설계 완료

조건을 만족하지 않으면 learned fusion은 `DEFERRED_NO_SYNCHRONIZED_EVIDENCE`로 둔다. 합성 score 조합만으로 성능 우위를 주장하지 않는다.

## I-5. Raspberry Pi 5 system validation

- acquisition+inference+fusion end-to-end latency
- CPU, memory, temperature, power, storage growth
- 1시간 이상 안정성과 restart behavior
- sensor reconnect·partial failure·clock drift
- model/manifest/config loading과 checksum
- 경보·dashboard 출력의 schema와 설명 가능성

## I-6. team handoff·release gate

1. `AGENTS.md`의 team repository handoff contract를 따른다.
2. 기존 팀 `ondevice_ai/`와 충돌을 replace/merge/preserve/relocate/retire로 분류한다.
3. device implementation을 `ondevice_ai/sensors/`에 중복 복사하지 않는다.
4. source commit과 team base commit SHA, model/data checksum, test evidence를 PR에 기록한다.
5. direct main push, force push, `git add .`를 금지한다.
6. MR60/SCD40/Thermal-44/Pi 미검증 항목을 숨기지 않는다.

---

# Part VI — 통합 실행 checklist

## Parallel Wave 1

- [ ] M-B0 evaluation protocol·near-duplicate
- [ ] C-A0 source identity·inventory
- [ ] T-A0 dataset selection·identity
- [ ] I-0 contract inventory

## Parallel Wave 2

- [ ] M-B1~M-B4 preprocessing·imbalance·architecture·multi-seed
- [ ] C-A1~C-A6 full reconstruction
- [ ] T-A1~T-A4 reader·geometry·sequence·label pilot
- [ ] 공용 manifest integration commit

## Parallel Wave 3

- [ ] M-B5~M-B12 conversion·robustness·candidate lock·report
- [ ] C-B0~C-B5 offline candidate
- [ ] T-A5~T-A6 split·full conversion
- [ ] I-1 contract conformance

## Parallel Wave 4

- [ ] M-C MR60 domain
- [ ] C-C SCD40 domain
- [ ] T-B offline candidate
- [ ] I-2 deterministic replay

## Parallel Wave 5

- [ ] gap가 있는 sensor만 M/C/T-D 진입
- [ ] T-C Thermal-44 domain
- [ ] I-3 rule fusion baseline
- [ ] I-4 learned fusion 조건 판정

## Final Wave

- [ ] M/C/T-E artifact·contract lock
- [ ] I-5 Pi 5 validation
- [ ] I-6 team handoff·release report

## 전체 종료 조건

1. M/C/T 각각 실제 source부터 locked offline candidate까지 독립 재현 가능하다.
2. 세 센서의 split·provenance·checksum·validator가 machine-readable하다.
3. synthetic smoke, offline real-data, device-domain, Pi evidence가 분리된다.
4. 실제 기기 입력과 학습 입력의 signal semantics가 측정으로 연결된다.
5. 공용 provider 계약이 invalid·stale·missing을 fail-closed 처리한다.
6. replay와 실제 integration 결과가 구분된다.
7. 팀 저장소 이관이 책임 경계와 CODEOWNERS를 보존한다.
8. 실측되지 않은 임상·하드웨어·배포 성능을 주장하지 않는다.
