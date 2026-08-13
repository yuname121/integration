# SafeNest V5 팀 인수인계 가이드

## 상태와 소유권

- P0 온디바이스 AI 코어: 검증 완료
- 실센서 드라이버: 센서 담당 팀원 영역
- 실센서 통합: 진행 예정
- 웹 UI: 범위 밖
- Raspberry Pi 5 성능: 실측 전까지 미검증

V5의 목표는 팀원이 구현한 센서 계층을 안전하게 주입해 `InferenceResult → 위험도 융합 → JSON Lines`로 연결하는 것이다. 이 저장본에는 Thermal-44 SPI/I2C, MR60BHA2 UART, SCD40 I2C, PIR GPIO 통신 구현이 없다.

## 반드시 사용할 경로

```text
integrated_node/run_node.py
→ sensors/* adapter 또는 외부 provider
→ inference/* interpreter
→ risk/risk_engine.py
→ risk/fallback.py
→ inference/inference_result.py
```

`integrated_node/safenest_risk_engine.py`는 legacy compatibility다. 신규 작업은 이 모듈에 연결하지 않는다. production 코드는 `archive/`, `version_archives/`, `releases/`의 코드를 import하지 않는다.

## Provider 주입

```python
node = SafeNestIntegratedNode(
    mode="real",
    sensors={
        "thermal44": thermal_provider,
        "mmwave": mmwave_provider,
        "co2": co2_provider,
        "pir": pir_provider,
    },
)
```

Provider 공통 메서드:

```python
connect() -> bool
read() -> InferenceResult
close() -> None
```

키와 `InferenceResult.sensor_id`는 각각 `thermal44`, `mmwave`, `co2`, `pir`로 고정한다. `connect()`가 `True`가 아니거나 예외가 발생하면 해당 센서는 fail-closed 처리된다. 일부 연결 실패는 `DEGRADED`, 전체 연결 실패는 `FAILED`다.

## 결과 계약

```python
InferenceResult(
    sensor_id="co2",
    timestamp=wall_clock_unix_seconds,
    score=0.0,
    state="VACANT",
    confidence=0.9,
    valid=True,
    latency_ms=1.2,
    error=None,
    metadata={},
)
```

- `score`, `confidence`: finite `0.0..1.0`
- `timestamp`: finite wall-clock Unix seconds. 오래된 관측은 원래 timestamp 유지
- `latency_ms`: finite, 0 이상
- `valid=false`: 명시적 오류 코드 필수
- metadata를 포함한 출력 전체에 NaN/Inf 금지
- 관측 불가능 상태를 `valid=true`, `score=0`으로 바꾸지 않음
- 재연결의 `connect()`에서 rolling buffer와 temporal state 초기화

## 설정 합의

`config/sensors.yaml`은 `enabled`, `stale_sec`, `timeout_sec`, `sample_rate_hz`, `window_samples`, `window_seconds`, `loop_interval_sec`의 runtime source다. provider가 같은 이름의 attribute 또는 `runtime_settings` mapping을 선언하면 startup에서 설정과 대조하며 불일치는 예외로 중단한다.

현재 stale 기본값은 Thermal 3초, mmWave 3초, CO₂ 10초, PIR 10초다. CO₂ 0.2 Hz 관측은 5초 간격이므로 3초 공통 TTL로 판정하지 않는다. 팀 센서 측정 주기가 확정되면 설정과 provider 선언을 함께 변경한다.

## 팀별 체크

Thermal 담당:

- AI에 `(62, 80)` float32 finite frame 제공
- byte order, checksum, raw conversion, training orientation 확인은 팀 영역
- normalization/INT8 quantization은 `ThermalInterpreter`에 중복 구현하지 않음

mmWave 담당:

- finite phase, strictly increasing wall-clock timestamp, presence/quality 제공
- UART frame parsing은 팀 영역
- 세션 재연결 시 buffer 초기화

CO₂ 담당:

- ppm, humidity %, temperature °C, Unix timestamp 제공
- I2C command/CRC는 팀 영역
- 약 5초 cadence와 10초 stale 설정 합의

PIR 담당:

- boolean motion과 Unix timestamp 제공
- GPIO interrupt/pull 설정은 팀 영역
- startup grace와 15초 temporal rule을 중복 구현하지 않음

## 통합 전 검증

```bash
python3 scripts/validate_v4_config.py
python3 -m unittest discover -s tests -p "test_*.py" -v
python3 integrated_node/run_node.py --mode mock
```

실제 provider integration test는 정상 네 센서 `HEALTHY`, 한 센서 실패 `DEGRADED`, 전체 실패 `FAILED`, Thermal 낙상과 mmWave 무호흡 각각 `DANGER/R=100`을 확인한다.
