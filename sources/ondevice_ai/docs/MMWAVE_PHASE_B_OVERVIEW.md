# SafeNest mmWave Phase B 개요

## 1. 문서 개요 및 핵심 한 줄 요약

> **Phase A가 Zenodo 110 피험자의 실제 mmWave 레이더 신호를 신뢰할 수 있는 30초 단위의 비필터·비정규화 canonical phase 데이터셋과 자발적 숨참기 기반 SafeNest APNEA proxy label로 재구성하는 단계였다면, Phase B는 그 데이터를 바탕으로 최적의 신호 처리(전처리)와 온디바이스 AI 모델을 비교·선정하여 최종 INT8 양자화 모델을 완성하는 단계입니다.**

본 문서는 SafeNest 레이더 신호 처리 및 온디바이스 AI 파이프라인을 직접 구현하지 않은 팀원도 **mmWave Phase B의 목적, 필요성, 수행 흐름, 검증 원칙 및 최종 산출물**을 명확히 이해할 수 있도록 작성된 종합 개요 문서입니다.

---

## 2. Phase A와 Phase B의 관계 및 불변 경계

Phase B는 Phase A에서 구축하고 검증을 완료한 **공식 정제 데이터셋 및 계보(Lineage)**를 출발점으로 사용합니다.

### 2.1 Phase A에서 완결된 작업
Phase A(A0~A6)에서는 raw 레이더 아카이브로부터 데이터의 신뢰성을 보장하기 위해 다음 작업을 완료했습니다:
- **A0 (Raw 수집 감사)**: Zenodo 60 GHz 레이더 아카이브(`db_records.zip`, 110명 피험자, 440개 레코딩) 전수 인벤토리 및 무결성 고정
- **A1 (Safe Decode)**: 안전한 rFFT 복원 및 텐서 해독
- **A2 (위상 추출)**: 라벨 독립적 거치 거리-빈(Range-bin) 및 가상 채널(Virtual channel) 선정 기반 위상(Phase) 추출
- **A3 (타임라인 정규화)**: 10 Hz 정규 타임라인 보정 및 30초/300샘플 단위 윈도우 생성
- **A4 (라벨 매핑)**: Movesense 가슴 가속도계 기준 호흡률 및 자발적 숨참기 아티팩트 기반 라벨 매핑 (`NORMAL`, `RAPID_OR_ABNORMAL`, `APNEA`, `AMBIGUOUS`)
- **A5 (피험자 수직 분할)**: 동일 피험자의 데이터가 섞이지 않도록 피험자 단위(Subject-level) 분할 고정 (TRAIN 77명 / VALIDATION 17명 / LOCKED_TEST 16명)
- **A6 (전체 변환 및 무결성 감사)**: 530개 30초 윈도우 수치 데이터셋([`datasets/mmwave/processed/mmwave_canonical_real_v1.npy`](../datasets/mmwave/processed/mmwave_canonical_real_v1.npy), $530 \times 300$ float64) 생성 및 교차 분할 누수(Cross-split leakage) 0건 검증

### 2.2 Phase-A 산출물의 불변 경계 (Immutable Boundary)
- Phase B는 Phase A에서 승인된 계보, 라벨 매핑, 피험자 분할(Split) 및 canonical 수치 데이터셋(`mmwave_canonical_real_v1.npy`)을 **절대 묵인 하에 변경하거나 재계산하지 않습니다.**
- Phase B의 모든 전처리 및 모델 비교 실험은 정제된 530개 위상 윈도우 데이터셋을 입력으로 사용하며, Phase A의 생성 이력은 불변 보존됩니다.

---

## 3. Phase A vs Phase B 핵심 비교

| 구분 | Phase A (데이터 재구성 & 무결성 감사) | Phase B (모델 비교 & 온디바이스 경량화) |
|---|---|---|
| **핵심 질문** | *"이 데이터를 믿고 학습에 사용할 수 있는가?"* | *"이 데이터에서 어떤 전처리와 AI 모델이 가장 잘 작동하는가?"* |
| **주요 작업** | Raw 레이더 해석, 위상 추출, 윈도우화, 라벨 매핑, 피험자 분할, 무결성 검증 | 전처리 소거 실험(Ablation), 불균형 대응, 모델 구조 탐색, 시드 재현성, TFLite/INT8 양자화 |
| **주요 위험 요소** | 잘못된 신호 해석, 라벨 오류, 피험자 간 데이터 누수(Leakage) | 과적합(Overfitting), 테스트 데이터 오염(Test Contamination), 클래스 붕괴(Class Collapse), 양자화 열화 |
| **최종 산출물** | 검증 완료된 Canonical 수치 데이터셋 및 매니페스트 | 검증 완료된 온디바이스 mmWave AI 모델 후보 및 `.tflite` 양자화 자산 |

---

## 4. Phase B 전체 실행 흐름

Phase B는 다음과 같은 엄격한 단계를 거쳐 진행됩니다:

```text
Phase-A 정제 30초 위상 윈도우 (mmwave_canonical_real_v1.npy)
                     ↓
[B0] 평가 프로토콜 정립 및 LOCKED_TEST 잠금 (Evaluation Protocol & Test Lock)
                     ↓
[B1] 전처리 소거 실험 (Preprocessing Ablation: Raw, Detrend, BPF, Z-score)
                     ↓
[B2] 클래스 불균형 대응 (Class Imbalance Strategy)
                     ↓
[B3] TinyML 모델 구조 탐색 (Model Architecture Exploration)
                     ↓
[B4] 다중 시드 재현성 검증 (Multi-seed Reproducibility)
                     ↓
[B5] 대표 캘리브레이션 샘플 구성 (Representative Dataset for INT8)
                     ↓
[B6] Float Keras → Float TFLite → INT8 TFLite 동등성 검증 (Equivalence Audit)
                     ↓
[최종] LOCKED_TEST 독립 평가 (Final LOCKED_TEST Independent Evaluation)
```

> 💡 **참고**: 위 흐름은 Phase B의 계획된 실행 순서이며, Phase A가 통과됨에 따라 순차적으로 진행됩니다.

---

## 5. B0 — 평가 프로토콜 및 LOCKED_TEST 잠금

### 5.1 데이터 분할의 역할 정의
Phase B에서는 Phase A5에서 확정된 피험자 단위 분할(Subject-level split)을 그대로 승계하며, 각 분할은 엄격히 지정된 역할로만 사용됩니다:

- **TRAIN (77명 피험자, 358개 윈도우)**:
  - AI 모델 파라미터 학습
  - Z-score 스케일링 등 전처리 통계량 계산 (TRAIN 데이터로만 fit)
- **VALIDATION (17명 피험자, 84개 윈도우)**:
  - 전처리 기법(Ablation) 비교 및 선택
  - 모델 구조 및 하이퍼파라미터 탐색
  - 클래스 불균형 전략 및 양자화 캘리브레이션 비교
- **LOCKED_TEST (16명 피험자, 88개 윈도우)**:
  - **최종 선정된 1개 모델에 대한 단 1회 독립 성능 평가용**
  - **모델 선택, 전처리 선택, 하이퍼파라미터 튜닝 과정에 절대 사용 금지**

### 5.2 LOCKED_TEST 오염(Contamination) 방지 이유
실험 과정에서 LOCKED_TEST의 점수를 보며 모델이나 전처리를 반복 수정하면, 모델이 테스트 피험자의 특성에 편향되어 실제 환경에서의 일반화 성능을 신뢰할 수 없게 됩니다. 따라서 LOCKED_TEST는 최종 단계 전까지 완벽히 격리됩니다.

### 5.3 피험자 단위 분할(Subject-level Split)의 필요성
동일한 사람의 호흡 신호는 시간상 연속되어 강한 상관관계를 가집니다. 만약 한 피험자의 일부 윈도우가 TRAIN에 들어가고 일부가 VALIDATION/TEST에 들어가면 모델이 사람 고유의 패턴을 암기하여 성능이 과대평가되는 **데이터 누수(Data Leakage)**가 발생합니다. SafeNest는 피험자 단위로 분할하여 이를 완벽히 차단합니다.

---

## 6. B1 — 전처리 소거 실험 (Preprocessing Ablation)

Phase A의 위상 데이터셋(`mmwave_canonical_real_v1.npy`)은 필터링이 적용되지 않은 최소 가공 신호(Unfiltered phase)입니다. Phase B1에서는 다양한 전처리 조합을 비교하여 **어떤 전처리 조합이 실세계 레이더 데이터에서 실제 성능을 향상시키는지** 검증합니다.

### 6.1 검증 대상 전처리 요소
- **Raw Phase**: 가공되지 않은 위상 신호
- **Detrending**: 몸의 미세 이동이나 기계적 드리프트(Drift) 제거
- **Band-pass Filtering (BPF)**: 호흡 주파수 대역(예: 0.1 ~ 0.5 Hz / 6 ~ 30 bpm) 외 잡음 제거
- **Z-score Normalization**: 진폭 정규화 (TRAIN 통계량 기준)

### 6.2 소거 실험(Ablation)의 목적
"과거 구현에서 특정 필터를 사용했다"는 이유만으로 이를 무비판적으로 정답으로 채택하지 않습니다. 전처리 단계 각각을 온/오프(On/Off)하며 거치는 소거 실험을 통해, **자발적 숨참기 기반 APNEA proxy 및 이상 호흡 분류 성능을 높이는 최적의 전처리 조합**을 수치적 근거(VALIDATION 성능)로 선별합니다.

---

## 7. 모델 평가 지표 및 클래스 라벨

### 7.1 Authoritative 클래스 라벨
SafeNest mmWave 파이프라인의 라벨 체계는 다음과 같습니다:
- **`NORMAL`**: 정상 호흡 상태 ($10.0 \le \text{RR} < 25.0$ bpm)
- **`RAPID_OR_ABNORMAL`**: 빈호흡($\ge 25.0$ bpm) 또는 서호흡($< 10.0$ bpm) 및 불규칙 호흡
- **`APNEA`**: 자발적 숨참기 아티팩트 기반 무호흡 대리 라벨 ($\ge 6.0$초 비호흡 중첩)
- **`AMBIGUOUS`**: 전이 구간 및 불확실 구간 (순수 클래스 학습 및 평가 시 제외, 프로버넌스 보존)

### 7.2 단순 Accuracy(정확도) 지표의 한계
데이터셋 내에서 정상 호흡의 비중이 높을 경우, 모델이 모든 샘플을 `NORMAL`로만 예측해도 높은 Accuracy가 나올 수 있습니다. 이를 **클래스 붕괴(Class Collapse)**라 부르며, SafeNest `APNEA` proxy class를 전혀 감지하지 못하므로 후보 모델로 사용할 수 없습니다.

### 7.3 Phase B 핵심 평가 지표
따라서 Phase B에서는 단순 Accuracy 대신 다음 지표를 종합적으로 검증합니다:
- **Macro F1-score**: 각 클래스별 F1-score의 단순 평균 (소수 클래스 비중 반영)
- **클래스별 Recall (재현율) / Precision (정밀도)**: 특히 `APNEA` 및 `RAPID_OR_ABNORMAL` 감지율
- **Confusion Matrix (혼동 행렬)**: 클래스 간 오분류 경향성 파악
- **예측 분포 모니터링**: 특정 클래스로 편향되는 붕괴 현상 감지

---

## 8. TFLite 변환 및 INT8 경량화·양자화

### 8.1 온디바이스(On-device) 배포 목적
SafeNest 시스템은 딥러닝 서버가 아닌 라즈베리 파이 등 엣지 장비(Edge Device)에서 실시간으로 동작하는 것을 목표로 합니다. 이를 위해 모델의 연산 속도를 높이고 메모리 사용량을 줄이는 **TFLite 변환 및 INT8 양자화(Quantization)**가 필수적입니다.

### 8.2 양자화 동등성 검증 (Equivalence Audit)
- **Float32 Keras Model $\to$ Float32 TFLite $\to$ INT8 TFLite**
- 단순 변환 성공(파일 생성)만으로는 충분하지 않습니다. 32비트 부동소수점 모델을 8비트 정수(INT8) 모델로 양자화할 때 연산 정밀도 손실로 인해 성능이 급격히 떨어질 수 있습니다.
- Phase B에서는 변환 단계별로 **Macro F1 하락 폭, `APNEA` Recall 유지 여부, Top-1 예측 일치도**를 전수 검증하여 성능이 붕괴되지 않는 경량화 모델을 완성합니다.

---

## 9. Phase B 실행 환경 (macOS / Standalone Scope)

Phase B의 전체 실험 및 검증 과정은 **맥북(macOS) 단독 환경**에서 완결되도록 설계되어 있습니다.

- **독립적 실행 가능**: 물리적 MR60BHA2 레이더 하드웨어나 라즈베리 파이 5 장비가 **필요하지 않습니다.**
- **실세계 수집 데이터 기반**: 이미 다운로드 및 검증이 완료된 Zenodo 110명 실세계 레이더 데이터셋(`mmwave_canonical_real_v1.npy`)을 사용합니다 (합성 전용 데이터가 아님).
- **오프라인 수행**: 전처리 소거 실험, 모델 학습, VALIDATION 평가, TFLite 변환, INT8 양자화, 결정성(Seed) 검증 및 매니페스트 감사가 모두 macOS 상에서 오프라인으로 실행됩니다.
- **실장치 배포 평가와의 분리**: 실제 센서 하드웨어 연동 및 타겟 보드 배포 성능 검증은 Phase B 이후의 별도 융합/하드웨어 단계로 명확히 분리됩니다.

---

## 10. Phase B 최종 기대 산출물

Phase B의 최종 목표는 단순히 `.tflite` 파일 하나를 출력하는 것이 아닙니다. **누구나 검증 및 재현이 가능한 투명한 근거 체인(Evidence Chain)**을 완성하는 것입니다.

최종 산출물 체인 포함 사항:
1. 어떤 Phase-A canonical 데이터셋이 사용되었는가 (`mmwave_canonical_real_v1.npy`)
2. 어떤 전처리 조합이 비교·선정되었는가 (Ablation 결과 및 근거)
3. 어떤 모델 구조가 선택되었으며 TRAIN/VALIDATION 성능은 어떠한가
4. LOCKED_TEST가 모델 선정 과정에서 완벽히 격리되었는가
5. 선택된 최종 모델의 LOCKED_TEST 평가 결과는 어떠한가
6. TFLite 변환 및 INT8 양자화 과정에서 성능 동등성이 유지되었는가
7. 각 자산의 해시(SHA-256) 및 메타데이터 계보가 명확한가

---

## 11. 문서 탐색 및 관련 참조 링크

본 문서는 SafeNest 활성 개발 작업 공간의 관련 문서와 연결되어 있습니다:

- **개발 루트 문서 안내**: [`docs/README.md`](README.md)
- **전체 실행 순서 규격**: [`docs/20260806_ChatGPT_SafeNest_mmWave_Execution_Sequence_01.md`](20260806_ChatGPT_SafeNest_mmWave_Execution_Sequence_01.md)
- **Phase A5 결과 보고서**: [`docs/reports/20260808_Antigravity_A5_Subject_Split_Provenance_01.md`](reports/20260808_Antigravity_A5_Subject_Split_Provenance_01.md)
- **Phase A6 결과 보고서**: [`docs/reports/20260808_Antigravity_A6_Full_Conversion_Integrity_Audit_01.md`](reports/20260808_Antigravity_A6_Full_Conversion_Integrity_Audit_01.md)
- **작업 공간 실행 규약**: [`AGENTS.md`](../AGENTS.md)
