# SafeNest 온디바이스 AI 팀 인수인계 가이드

본 문서는 팀 저장소 `safenest-embedded-competition`의 `ondevice_ai/` 컴포넌트를  
하드웨어·라즈베리파이·웹 UI 담당자가 안전하게 연동하기 위한 **최신 인수인계 기준**입니다.

- 컴포넌트 루트: `ondevice_ai/`
- 동기화 소스: `https://github.com/sheepmeat/test` @ `9a66a3b`
- 상태: **개발 동기화 완료 / 배포 미승인 (NOT_READY)**

## 0. 모든 팀원·에이전트 공통 준수사항

1. 작업 시작 전 `ondevice_ai/AGENTS.md`를 읽습니다.
2. AI 작업의 활성 루트는 `ondevice_ai/`입니다. `embed2/`, `SafeNest_V6/`, `ondevice_ai/ondevice_ai/`를 만들지 않습니다.
3. 실하드웨어 드라이버는 `devices/<device>/src/`, 공개 계약은 `shared/contracts/`를 따릅니다.
4. `ondevice_ai/sensors/*`는 AI-side mock/adapter/계약용이며, 팀 실드라이버를 대체하지 않습니다.
5. 새 JSON/YAML/metadata에는 저장소 상대경로만 기록하고 `/Users/...`나 `file://` 경로를 남기지 않습니다.
6. `real` mode는 실센서 연동·fail-closed 개발 검사용이며, 실배포 승격은 별도 production gate와 MR60/Pi/Thermal-44/SCD40 실측 후에만 합니다.
7. Mock 성공만으로 “실센서 통합 완료/배포 준비 완료”를 주장하지 않습니다.

---

## 1. 현재 센서 트랙 상태 (2026-08-10 동기화)

### mmWave
- 완료: M-A0..M-A6, M-B0..M-B5
- M-B5 선택 calibration profile: `M-B5_CAL_CLASS_BALANCED_120` (TRAIN-only)
- 미완: M-B6+ formal Float/TFLite/INT8 stage equivalence, locked evaluation
- LOCKED_TEST 모델선택 접근: 0
- MR60/Pi 실측: 미완
- APNEA = voluntary breath-hold proxy (임상 apnea 아님)

### CO₂
- 완료: C-A0..C-A6 (raw→canonical integrity lock)
- 미완: C-B 모델 재학습/비교, SCD40 device-domain validation
- 기존 CO₂ 모델의 실데이터 재학습 완료를 의미하지 않음

### Thermal
- 완료: T-A0..T-A4 (제한사항 포함)
- 미완: T-A5 split, T-A6 full conversion, T-B training
- `LYING`은 frame-level post-fall posture proxy이며 verified fall-onset label이 아님

---

## 2. 온디바이스 위험도 규격

위험도 점수 $R$:

$$R = 100 \times (0.35 S_1 + 0.35 S_2 + 0.15 S_3 + 0.15 S_4)$$

- **$S_1$ (mmWave)**: 호흡 이상/무호흡 점수 `[0,1]` (APNEA proxy 1.0, RAPID_OR_ABNORMAL 0.5, NORMAL 0.0)
- **$S_2$ (CO2)**: 재실/고농도 점수 `[0,1]`
- **$S_3$ (PIR)**: 장시간 미움직임 점수 `[0,1]`
- **$S_4$ (Thermal-44)**: 낙상 점수 `[0,1]`

Emergency override: Thermal 낙상($S_4=1.0$) 또는 mmWave APNEA proxy($S_1=1.0$) 시 즉시 `R=100` / `DANGER`.

---

## 3. 담당 분야별 연동 프롬프트

### 📌 Thermal-44 하드웨어
```text
[역할] SafeNest Thermal-44 하드웨어 엔지니어
[목표] 팀 devices/thermal 실드라이버를 ondevice_ai provider 계약에 연결
[주의]
- ondevice_ai/sensors/thermal44 는 AI mock/adapter 영역
- LYING은 post-fall posture proxy이며 fall-onset 확정이 아님
- risk_engine.py / model_manifest.json 임의 수정 금지
[계약] connect()/read()/close(), InferenceResult, fail-closed
```

### 📌 mmWave (MR60)
```text
[역할] SafeNest mmWave 하드웨어 엔지니어
[목표] devices/mmwave 실드라이버를 ondevice_ai mmWave provider에 연결
[주의]
- LOCKED_TEST를 모델 선택/캘리브레이션에 사용 금지
- M-B5 calibration 선택 ≠ 최종 INT8 배포 승인
- APNEA는 clinical apnea가 아닌 voluntary breath-hold proxy
[계약] connect()/read()/close(), InferenceResult, fail-closed
```

### 📌 CO₂ (SCD40)
```text
[역할] SafeNest CO₂ 하드웨어 엔지니어
[목표] devices 쪽 SCD40 실측을 ondevice_ai CO₂ provider에 연결
[주의]
- C-A6 완료는 데이터 체인 락이며 C-B 재학습 완료가 아님
- 결측/NaN/stale을 정상값으로 합성 금지
[계약] connect()/read()/close(), InferenceResult, fail-closed
```

### 📌 PIR
```text
[역할] SafeNest PIR 하드웨어 엔지니어
[목표] GPIO 기반 움직임 provider를 ondevice_ai PIR adapter 계약에 연결
[주의] 고장 시 valid=False, 정상 합성 금지
```

### 📌 Raspberry Pi 5 시스템
```text
[역할] SafeNest Pi5 시스템 통합 엔지니어
[설치]
1) sudo apt-get update && sudo apt-get install -y python3-pip python3-venv i2c-tools
2) raspi-config에서 I2C/SPI/Serial 활성화
3) cd <team-repo>/ondevice_ai
4) python3 -m venv .venv && source .venv/bin/activate
5) pip install -r requirements-pi.txt
6) python3 -m unittest discover -s tests -p "test_*.py"
7) python3 integrated_node/run_node.py --mode real
[주의] Pi 실측 전 production 승격 금지
```

### 📌 웹 UI
```text
[역할] SafeNest 웹 UI 개발자
[목표] integrated_node/run_node.py JSON Lines stdout 스트림 시각화
[주의]
- risk_score/risk_level = 사람 위험
- system_health = 파이프라인 건강
- FAILED일 때 risk 값을 NORMAL/0으로 표시 금지
```

### 📌 QA / 모델
```text
[역할] QA 및 오프라인 모델 검증 엔지니어
[절차]
cd ondevice_ai
python3 -m unittest discover -s tests -p "test_*.py" -v
# raw payload 미전송 phase는 NOT_RUN_RAW_PAYLOAD_NOT_TRANSFERRED 로 기록
[주의] checksum/manifest와 실제 아티팩트 불일치 시 fail-closed
```

---

## 4. 팀 전용으로 보존된 파일 (삭제 금지)

다음 파일은 스탠드얼론 소스에 없어도 팀 컴포넌트에 남겨 둡니다.

- `integrated_node/esp32_sensor_node.ino`
- `models/mmwave/safenest_lstm_quant.tflite`
- `models/mmwave/sensor_stats_metadata.json`
- `models/thermal/thermal_fall_model.h5`
- `scripts/build_v4_archive.py`
- `scripts/build_v5_archive.py`
- `scripts/validate_v4_config.py`
- `tests/test_v4_config_validation.py`
- `tests/test_v5_release.py`

충돌 전체 표: `docs/integration/collision_matrix.json`  
요약: `docs/integration/collision_summary.md`

---

## 5. 의도적으로 전송하지 않은 것

- `.git/`, `.github/`, `archive/`, `hardware/`, `releases/`
- `datasets/raw_archives/` 및 ignored raw/thermal payload
- 로컬 캐시, venv, 절대경로 메타데이터
- 팀 root CI (필요 시 별도 owner 리뷰 커밋)

---

## 6. 남은 개발 과제

1. mmWave M-B6+ stage equivalence / candidate lock
2. CO₂ C-B+ real-data model comparison
3. Thermal T-A5+ / T-B
4. devices + shared/contracts 실연동 및 fail-closed 실측
5. Raspberry Pi 성능·안정성 측정
6. final multisensor fusion (별도 승인 후)

Rollback: 이번 통합 feature branch/PR 커밋을 revert 합니다.
