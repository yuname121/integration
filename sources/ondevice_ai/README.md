# SafeNest On-Device AI (`ondevice_ai/`)

> ⚠️ **배포 금지 / NOT_READY**
>
> 이 디렉터리는 팀 저장소의 **온디바이스 AI 컴포넌트**입니다.  
> 소프트웨어 mock·오프라인 데이터 계약·phase validator 증거는 동기화되어 있지만, **실센서/라즈베리파이/임상 성능 검증은 완료되지 않았습니다.**

## 비담당자가 먼저 알면 되는 것

| 질문 | 답 |
|---|---|
| 여기가 뭐하는 폴더인가요? | 전처리·추론·모델 자산·데이터 계약·위험도 로직·validator/보고서가 모인 AI 컴포넌트입니다. |
| 실기기 드라이버도 여기 있나요? | **아니요.** 실하드웨어 드라이버는 팀 저장소의 `devices/<device>/src/` 쪽입니다. |
| 지금 배포해도 되나요? | **안 됩니다.** Mock 통과 ≠ 실배포 승인입니다. |
| 최신 동기화 기준은? | 스탠드얼론 소스 `https://github.com/sheepmeat/test` 커밋 `9a66a3b` |

## 이번 동기화에서 바뀐 점 (요약)

1. **mmWave**
   - `M-A0`~`M-A6` 실데이터 변환/무결성 락 유지
   - `M-B0`~`M-B5` 추가 동기화
   - `M-B5`에서 TRAIN-only representative calibration profile 선택 (`M-B5_CAL_CLASS_BALANCED_120`)
   - 이는 **최종 배포 INT8 모델 승인 완료가 아님** (M-B6 stage equivalence 및 locked evaluation 남음)
2. **CO₂**
   - `C-A0`~`C-A6` raw→canonical 데이터 체인 락 동기화
   - **C-B 모델 재학습/비교는 아직 시작 전**
3. **Thermal**
   - `T-A0`~`T-A4` 동기화 (`LYING`은 frame-level post-fall posture proxy)
   - **T-A5 split / T-A6 full conversion / T-B 학습은 아직 없음**
4. **통합 규칙**
   - 팀 전용 파일(예: `esp32_sensor_node.ino`, 구버전 빌드/검증 스크립트, 팀 보유 모델 바이너리)은 **삭제하지 않고 보존**
   - 실드라이버 중복 복사 금지, fail-closed 유지
5. **문서**
   - 본 README와 `docs/TEAM_HANDOFF_GUIDE.md`를 팀 `ondevice_ai/` 기준으로 최신화
   - 충돌 결정 기록: `docs/integration/collision_summary.md`

## 현재 개발 상태 (정직하게)

### mmWave
- 완료: M-A0..M-A6, M-B0..M-B5 (경고/조건 포함)
- 미완: M-B6 이후 formal stage equivalence / locked evaluation
- LOCKED_TEST 모델 선택 접근: **0**
- MR60 실기기·Pi 배포 검증: **미완**

### CO₂
- 완료: C-A0..C-A6
- 미완: C-B+ 모델 비교/재학습, SCD40 device-domain validation
- UCI 소스만으로 cross-room/cross-building 일반화 주장 불가

### Thermal
- 완료: T-A0..T-A4 (제한사항 포함)
- 미완: T-A5+, T-B, Thermal-44 device validation
- `LYING` ≠ verified fall-event onset label

### 통합/배포
다음을 **주장하지 마세요.**
- 실센서 통합 완료
- Raspberry Pi 검증 완료
- 임상 검증 완료
- final fusion 최적화 완료
- deployment ready

## 책임 경계

```text
devices/<device>/src/     실하드웨어 드라이버
shared/contracts/         공개 센서 인터페이스
ondevice_ai/              AI 전처리·추론·모델·데이터계약·risk·mock·validator
```

실센서 provider가 없는 `real` mode는 정상값을 합성하지 않습니다.  
센서는 `valid=false` / `EXTERNAL_SENSOR_PROVIDER_REQUIRED`로 실패하고 시스템은 `FAILED`로 판정합니다.

## 실행 (팀 저장소 루트 기준)

```bash
cd ondevice_ai

# Mock end-to-end
python3 integrated_node/run_node.py --mode mock

# provider 없는 fail-closed 확인
python3 integrated_node/run_node.py --mode real
```

Provider 주입 예:

```python
from integrated_node.run_node import SafeNestIntegratedNode

node = SafeNestIntegratedNode(
    mode="real",
    sensors={
        "thermal44": thermal_provider,
        "mmwave": mmwave_provider,
        "co2": co2_provider,
        "pir": pir_provider,
    },
)
node.start()
print(node.step().to_json())
node.shutdown()
```

각 provider는 `connect() -> bool`, `read() -> InferenceResult`, `close() -> None`을 구현해야 합니다.

## 위험도 수식

```text
R = 100 * (
    0.35 * S_mmwave
  + 0.35 * S_co2
  + 0.15 * S_pir
  + 0.15 * S_thermal
)
```

Thermal 낙상 또는 mmWave 무호흡(APNEA proxy)은 emergency override로 `R=100` / `DANGER`입니다.  
APNEA는 **자발적 호흡정지 프록시**이며 임상 apnea가 아닙니다.

## 검증

```bash
cd ondevice_ai
python3 scripts/validate_models.py
python3 -m unittest discover -s tests -p "test_*.py" -v
python3 -m compileall -q inference risk sensors integrated_node scripts tests
```

원본 raw archive가 팀 저장소에 전송되지 않은 phase validator는  
`NOT_RUN_RAW_PAYLOAD_NOT_TRANSFERRED`로 보고해야 하며, fixture를 만들어 통과시켜서는 안 됩니다.

## 문서

- [팀 인수인계 가이드](docs/TEAM_HANDOFF_GUIDE.md)
- [통합 충돌 요약](docs/integration/collision_summary.md)
- [멀티센서 병렬 roadmap](docs/20260810_ChatGPT_SafeNest_Multisensor_Parallel_Execution_Roadmap_01.md)
- [mmWave 실행 순서](docs/20260806_ChatGPT_SafeNest_mmWave_Execution_Sequence_01.md)
- [mmWave Phase B 개요](docs/MMWAVE_PHASE_B_OVERVIEW.md)
- [Sensor provider 계약](docs/reports/V5_SENSOR_PROVIDER_CONTRACT.md)
