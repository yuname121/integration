# PHASE 6 — Risk Engine 연결

## 채택한 기존 계약

GitHub `main`의 `ondevice_ai/risk/` 구현과 `risk_config.json`을 `sources/ondevice_ai/risk/`에 수정 없이 동결했다. 통합 엔진은 공식 V4 가중치와 30/60 경계를 사용한다.

| Component | 센서 | 가중치 |
|---|---|---:|
| S1 | mmWave 호흡/무호흡 | 0.35 |
| S2 | CO₂ 환경 위험 | 0.35 |
| S3 | PIR 장시간 무움직임 | 0.15 |
| S4 | Thermal 자세/낙상 | 0.15 |

공식 구현의 `CAUTION` 명칭만 최종 요구사항에 맞춰 `WARNING`으로 표시한다.

- `NORMAL`: `R < 30`
- `WARNING`: `30 <= R < 60`
- `DANGER`: `R >= 60`

일부 component가 unavailable이면 기존 fallback 계약처럼 사용 가능한 가중치만 재정규화한다. unavailable은 0점으로 대체하지 않는다. 모든 component가 unavailable이면 `risk_score`와 `risk_level`은 `null`, `system_health`는 `FAILED`이다.

## 센서별 위험도 입력

### mmWave

1. 유효한 mmWave AI 결과가 있으면 해당 score를 사용한다.
2. AI 입력 window가 없거나 모델이 실패해도 최신 호흡수가 있으면 공식 정상 범위 12–20 rpm 규칙을 사용한다.
3. 범위 밖 호흡은 score 0.75와 `ABNORMAL_RESPIRATION_RPM`을 만든다.
4. PHASE 5 metadata가 `apnea_verified: false`인 APNEA 추정은 emergency로 승격하지 않는다.

현재 TCP schema에는 hardware apnea와 presence가 없으므로 이를 추정해 만들지 않는다.

### CO₂

CO₂ 모델은 점유 분류 모델이므로 환경 위험 score를 직접 대신하지 않는다. 최신 ppm으로 기존 rule을 적용한다.

- warning: 1000 ppm 이상
- danger reason: 2500 ppm 이상
- score: `clip((ppm - 500) / 2000, 0, 1)`
- 연속 sample로 계산한 상승률이 15 ppm/min 이상이면 `FAST_CO2_RISE`

현재 선택된 legacy primary는 습도가 없으면 unavailable이지만 ppm rule은 계속 동작하며 시스템은 `DEGRADED`로 표시된다. B-complete candidate 입력은 `CO2 + CO2_slope`이고 humidity를 요구하지 않지만, 이 corrective pass에서 모델을 승격하거나 Risk 정책을 바꾸지 않는다.

### PIR

PIR 무움직임은 사람 존재가 Thermal 또는 실제 mmWave presence로 확인된 경우에만 누적한다.

- motion: score 0
- 확인된 사람 + 15초 미만 무움직임: score 0.5
- 확인된 사람 + 15초 이상 무움직임: score 1.0, `LONG_NO_MOTION`
- presence 미확인: score 0, `PRESENCE_NOT_CONFIRMED`

### Thermal

최신 Thermal AI 결과만 사용한다. `HUMAN_FALL`, score 1.0, confidence 0.8 이상이면 weighted score와 관계없이 `DANGER/100` emergency override를 적용한다.

## 교차검증과 health

- 실제 mmWave presence와 Thermal human 판정이 반대이면 `MMWAVE_THERMAL_MISMATCH`를 기록한다.
- AI 대신 rule fallback을 사용하거나 센서가 unavailable이면 `system_health=DEGRADED`이다.
- 위험도(`risk_level`)와 시스템 건강도(`system_health`)는 별개다. 예를 들어 `NORMAL + DEGRADED`가 가능하다.
- stale, invalid, disconnected 센서는 component score가 `null`이며 계산에서 제외된다.
- NaN/Inf 및 범위 밖 score는 사용할 수 없는 입력으로 처리한다.

## Raspberry Pi 실행

```bash
cd ~/integration
source .venv/bin/activate
python ./gateway/run_risk_gateway.py --host 0.0.0.0 --port 9000
```

한 줄 JSON에 동일 revision의 `state`, `ai`, `risk`가 함께 출력된다.

## 검증 범위

- 공식 가중치와 정확한 30/60 경계
- `CAUTION` → `WARNING` 출력 계약
- 전체 결측이 NORMAL로 바뀌지 않는지 확인
- stale 제외 및 available weight 재정규화
- Thermal confidence 기반 emergency override
- 확인되지 않은 APNEA의 emergency 차단
- AI 실패 후 호흡·CO₂ rule 지속
- presence 확인 전 PIR 무움직임 누적 차단
- mmWave/Thermal cross validation
- packet 객체부터 state, AI, risk까지 통합 흐름
- strict JSON과 NaN 차단

현재 동결 config 상태는 `v4_implementation_locked_hil_pending`이다. 따라서 가중치와 임계값은 기존 구현을 그대로 사용했지만 실제 차량/밀폐공간 HIL 검증이 끝난 임상·안전 인증값으로 표현하지 않는다.

## 변경 전/후

- 변경 전: 기존 Risk Engine은 별도 inference contract에 묶여 현재 TCP Sensor State와 직접 연결되지 않았다.
- 변경 후: 하나의 state revision에서 AI 결과와 검증된 rule fallback을 만들고, health를 분리한 `NORMAL/WARNING/DANGER` 결과를 출력한다.

PHASE 6 Risk Engine 연결 완료.
