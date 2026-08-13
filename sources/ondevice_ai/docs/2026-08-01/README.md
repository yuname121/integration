# SafeNest V4 On-Device AI 최종 배포 패키지

본 디렉터리는 **SafeNest V4 온디바이스 AI 파이프라인**의 전체 소스 코드, 양자화 TFLite 모델, 전처리 데이터셋, 위험도 융합 엔진, 센서 어댑터 및 유닛 테스트 스위트를 포함하는 최상위 배포 폴더입니다.

---

## 1. 팀원 인수인계 설명서 (통합 프롬프트 모음)

팀원들이 각자 담당하는 센서 및 파트(Thermal, mmWave, CO2, PIR, 라즈베리 파이 5, 웹 UI)에 즉시 복사하여 적용할 수 있는 통합 인수인계 설명서는 아래 단일 문서에 작성되어 있습니다:

👉 **[통합 팀원 인수인계 가이드 (docs/TEAM_HANDOFF_GUIDE.md)](docs/TEAM_HANDOFF_GUIDE.md)**

---

## 2. 디렉터리 구조

```text
SafeNest_V4_OnDevice_AI/
├── config/              # 센서 버스, 모델 양자화 스케일, 위험도 수식 설정 (YAML)
├── inference/           # TFLite 모델 추론기 (Thermal-44, mmWave, CO2) & Registry
├── sensors/             # 센서 드라이버 및 Mock/Real 어댑터 (Thermal, mmWave, CO2, PIR)
├── risk/                # 멀티센서 위험도 융합 엔진 & 세이프 폴백 핸들러
├── integrated_node/     # 실시간 JSON Lines 스트리밍 메인 노드 (run_node.py, run_demo.py)
├── models/              # INT8 양자화 TFLite 모델 및 model_manifest.json
├── datasets/            # 전처리 NPZ 데이터셋 & build_processed_npz.py
├── docs/                # [통합 팀원 인수인계 가이드 (TEAM_HANDOFF_GUIDE.md)]
├── tests/               # unittest 자동 발견 방식의 유닛·통합 테스트 스위트
├── adapters/            # Ring-buffer 스트림 및 CSV 어댑터
├── benchmarks/          # 추론 지연시간 벤치마크
├── scripts/             # 데모 시각화 및 검증 유틸리티
├── README.md            # 본 안내 문서 (한국어)
├── walkthrough.md       # 온디바이스 AI 구성 요약 문서 (한국어)
├── requirements.txt     # 전체 파이썬 의존성 패키지
├── requirements-pi.txt  # 라즈베리 파이 5 전용 경량 의존성
└── requirements-mac.txt # macOS 테스트 전용 의존성
```

---

## 3. 위험도 융합 수식 & 산출 기준

온디바이스 위험도 점수 $R$은 4개 센서 채널의 가중 합산으로 산출됩니다:

$$R = 100 \times (0.35 S_1 + 0.35 S_2 + 0.15 S_3 + 0.15 S_4)$$

- **$S_1$ (mmWave)**: 호흡 이상 및 무호흡 위험도 $[0.0, 1.0]$
- **$S_2$ (CO2)**: 재실 및 농도 상승 위험도 $[0.0, 1.0]$
- **$S_3$ (PIR)**: 움직임 및 장기 미움직임 위험도 $[0.0, 1.0]$
- **$S_4$ (Thermal-44)**: 열화상 기반 사람 낙상 위험도 $[0.0, 1.0]$

### 비상 오버라이드 (Emergency Overrides)
- **Thermal-44 낙상 감지 ($S_4 = 1.0$)** 또는 **mmWave 무호흡 감지 ($S_1 = 1.0$)** 시, 가중 합산을 우회하여 즉시 **$R = 100.0$ (`DANGER`)** 비상 경보를 발령합니다.

---

## 4. 실행 방법

### (1) 실시간 스트리밍 노드 실행 (Mock 모드)
```bash
python3 integrated_node/run_node.py --mode mock
```

### (2) 실기기 라즈베리 파이 5 센서 연동 실행
```bash
python3 integrated_node/run_node.py --mode real
```

### (3) 전체 유닛 & 통합 테스트 실행
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

2026-08-01 macOS 검증 기준: **163개 발견, 161 PASS, 2 SKIP, 0 FAIL/ERROR**. 테스트 수는 추가·삭제될 수 있으므로 고정 개수보다 실행 결과의 `0 FAILED / 0 ERRORS`를 합격 기준으로 사용합니다.
