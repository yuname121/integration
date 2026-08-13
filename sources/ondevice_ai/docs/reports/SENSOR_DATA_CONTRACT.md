# SafeNest V4 Sensor Data Contract

## 1. Scope

This document formally defines and fixes the end-to-end sensor-to-model data contract for SafeNest V4 On-Device AI across all four sensor subsystems:
1. Thermal-44 IR Array Camera
2. mmWave 60GHz Respiration Radar (Seeed Studio MR60BHA2)
3. CO₂ Environmental Sensor (Sensirion SCD40)
4. PIR Motion Detector (GPIO)

**Workspace Root**: `SafeNest_V4_OnDevice_AI/`  
**Execution Phase**: Contract Verification & Fixation (Non-Modifying Inspection Phase). Existing drivers, parsers, interpreters, manifests, configs, and TFLite model binaries are preserved as strictly read-only.

---

## 2. Evidence and Verification Status

Every assertion in this data contract is qualified by one of fourteen mandatory verification status codes:

- `VERIFIED_CODE`: Confirmed by direct line-level inspection of executable source code in the repository.
- `VERIFIED_MODEL`: Confirmed by TFLite input/output tensor inspection or manifest validation.
- `VERIFIED_ARTIFACT`: Confirmed by repository metadata, real captured JSONL logs, or exported CSV datasets.
- `VERIFIED_DATASHEET`: Confirmed by official manufacturer datasheet specifications.
- `CONTRACT_MISMATCH`: Divergence between code, manifest, config, or documentation.
- `IMPLEMENTATION_DEFECT`: Runtime code execution flaw (e.g., parameter type mismatch or incorrect buffer interpretation).
- `RESOLVED`: Previously reported mismatch is corrected in code and guarded by regression tests.
- `CONTRACT_GAP`: Feature specified in target architecture but missing or unhandled in current implementation.
- `DESIGN_LIMITATION`: Architectural constraint of the chosen algorithm/preprocessing method.
- `DORMANT_CODE_CONTRACT_RISK`: Code contains a specification mismatch, but is uncalled in active production runtime.
- `BLOCKED_HARDWARE`: Physical hardware or real binary packet capture missing; cannot be verified in software alone.
- `UNVERIFIED_DATASHEET`: Official manufacturer datasheet or technical specification unavailable in repository.
- `NOT_APPLICABLE`: Property not applicable to the subsystem (e.g., TFLite model tensor contract for PIR).
- `TRAIN_INFERENCE_PREPROCESSING_MATCH`: Preprocessing method used in model training exactly matches inference implementation.
- `STALE_CONTRACT_ARTIFACT`: Values in legacy documentation or artifact diverged from actual metadata/code ground truth.

---

## 3. End-to-End Contract Summary

### Overall Verification Status

- `contract_validation_status`: **`PASS_WITH_BLOCKERS`** (Documentation and JSON schema 100% align with current codebase and ground truth artifacts; all known defects, gaps, and blockers are fully documented.)
- `system_deployment_status`: **`NOT_READY`** (Physical hardware drivers remain unwritten, and mmWave model semantic compatibility remains unverified.)

### Sensor Pipeline Summary

| 센서 | 원시 입력 | parser 출력 | 전처리 출력 | 모델/규칙 입력 | shape | dtype | 단위 | 주기/window | timestamp 의미 | valid 의미 | warm-up | fault 조건 | 품질 필드 | 검증 상태 | 코드 근거 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Thermal-44** | SPI Byte Packet ($9,920\text{B}$ candidate payload + $160\text{B}$ unverified) | `np.ndarray` (62, 80) float32 (raw unit unverified) | Per-frame Min-Max to [0.0, 1.0], INT8 Quantization | `serving_default_keras_tensor:0` | `[1, 62, 80, 1]` | `int8` | Vendor raw physical (`UNVERIFIED_DATASHEET`) / Model Normalized [0, 1] | 10 Hz (100ms) | Read start wall-clock time (`time.time()`) | Non-zero finite float32 frame | None (SPI read immediate) | Size != 4960, NaN/Inf, SPI timeout | `connected`, `age_sec`, `valid` | `DORMANT_CODE_CONTRACT_RISK` / `BLOCKED_HARDWARE` | `sensors/thermal44/frame_parser.py:L15-L35`, `inference/thermal_interpreter.py:L141-L163` |
| **mmWave** | UART Frame `0x0A13` (`totalPhase`, `breathPhase`, `heartPhase`) | `deque[float]` (300 samples) | Z-Score (`(x - mean)/std`, `mean=0.006092`, `std=2.501384`), INT8 Quantization | `serving_default_keras_tensor_8:0` | `[1, 300, 1]` | `int8` | Vendor phase float (`UNVERIFIED_DATASHEET`) | 10 Hz, 30s window (300 samples) | Monotonic sample timestamp ($t_s$) | Monotonic timestamp, gap $<0.5\text{s}$, presence == 1 | 300 samples warming window | Non-monotonic ts, gap $>0.5\text{s}$, presence == 0 | `presence`, `quality`, `buffer_len`, `stale` | `VERIFIED_CODE` / `VERIFIED_ARTIFACT` | `firmware/esp_wroom32_mr60_monitor/src/main.cpp:L124-L130`, `adapters/mmwave_stream_adapter.py:L48-L77`, `inference/mmwave_interpreter.py:L162-L190` |
| **CO₂** | I2C Periodic Measurement Packet | `tuple[float, float, float]` | Z-Score scaling (`(x - mean)/scale`), INT8 Quantization | `co2_slope_humidity_co2_ppm` | `[1, 3]` | `int8` | ppm/min, %, ppm | 0.2 Hz (5s interval), 30-sample history | Read start wall-clock time (`time.time()`) | Valid I2C read, 300-10000 ppm range, no fault | Initial sensor stabilization | ppm $<300$ or $>10000$, I2C CRC failure, NaN/Inf | `connected`, `age_sec`, `co2_valid` | `RESOLVED` / `VERIFIED_ARTIFACT` | `models/co2/co2_scaling_metadata_v0.1.0.json`, `inference/co2_interpreter.py`, `sensors/co2/co2_adapter.py` |
| **PIR** | GPIO Logic Level (HIGH / LOW) | `bool` (motion detected) | Elapsed timer calculation (`now_mono - last_motion_mono`) | Primary Engine / Rule Evaluator | `NOT_APPLICABLE` | `NOT_APPLICABLE` | Boolean / Seconds | Integrated node polling (target: GPIO edge `CONTRACT_GAP`) | Read start wall-clock time (`time.time()`) | Connected GPIO & valid level read | 5.0s startup grace period (`WARMING_UP`) | GPIO stuck high/low (`CONTRACT_GAP`), presence not confirmed (`CONTRACT_GAP`), stale $>10\text{s}$ | Primary metadata: `is_no_motion`, `elapsed_since_motion_sec` (Health: `age_sec`) | `VERIFIED_CODE` | `sensors/pir/pir_adapter.py:L43-L75`, `integrated_node/run_node.py:L84`, `risk/risk_engine.py:L238` |

---

## 4. Common Timestamp and Validity Contract

### Timestamp Taxonomy
To avoid ambiguity, timestamps across SafeNest V4 are categorized into four distinct time domains:

1. **Unix Epoch Wall-Clock** (`time.time()`):
   - **Producer**: Host Python OS runtime
   - **Time Domain**: Real-world UTC wall-clock time
   - **Unit**: Seconds (`float`)
   - **Ordering Guarantee**: Non-monotonic! Subject to NTP clock sync steps or manual system time adjustments.
   - **Cross-Device Comparable**: Yes (when NTP synchronized).
   - **Clock Reset/Wrap**: Subject to manual/NTP clock jumps.

2. **Process Monotonic Clock** (`time.monotonic()`, `time.perf_counter()`):
   - **Producer**: Python OS process runtime
   - **Time Domain**: Monotonic timer from arbitrary system start point
   - **Unit**: Seconds (`float`) / Milliseconds (`float`)
   - **Ordering Guarantee**: Strictly monotonic ($t_i \ge t_{i-1}$), never steps backward.
   - **Cross-Device Comparable**: No (process local).
   - **Clock Reset/Wrap**: None during process execution.

3. **ESP Boot-Relative Monotonic Milliseconds** (`ts_monotonic_ms`):
   - **Producer**: ESP32 Firmware (`millis()`)
   - **Time Domain**: Milliseconds elapsed since ESP32 MCU boot
   - **Unit**: Milliseconds (`uint32`)
   - **Ordering Guarantee**: Monotonic until overflow.
   - **Cross-Device Comparable**: No (board boot relative).
   - **Clock Reset/Wrap**: Wraps around every $2^{32} \text{ ms} \approx 49.7 \text{ days}$.

4. **CSV Session-Relative Timestamp** (`timestamp_s`):
   - **Producer**: `export_mmwave_csv.py` (`(ts_monotonic_ms - origin_ms) / 1000.0`)
   - **Time Domain**: Seconds elapsed since session start ($t_0 = 0.0$)
   - **Unit**: Seconds (`float`, 4 decimal places)
   - **Ordering Guarantee**: Session-relative time requirement (assumed monotonic within session, not guaranteed by OS).
   - **Cross-Device Comparable**: No (session relative).
   - **Clock Reset/Wrap**: Resets to $0.0$ at start of each recording session.

### Validity (`valid`) Semantics
- **`valid = true`**: High-level contract condition met when:
  1. Sensor hardware interface is connected (`connected == True`).
  2. Raw transport packet parsed cleanly without size or CRC errors.
  3. Physical values are finite ($\text{NaN} / \pm\infty$ free) and within physical bounds.
  4. Preprocessing and model/rule inference executed successfully.
- **`valid = false`**: Set whenever communication fails, packets drop, values exceed limits, or model inference throws an exception.
- **Risk & Fallback Propagation**:
  When `valid = false`:
  - Sensor quality score $q_i$ drops to $0.0$ (or $0.2$ for CO₂ fault).
  - Risk engine downgrades sensor status to `DEGRADED` or `FAULT`.
  - System status transitions to `DEGRADED` (`risk/risk_rules.py:L353-L355`).
  - Fallback logic is invoked to prevent system crash.

---

## 5. Common Warm-up, Stale and Fault Contract

### Sensor State Transition Matrix

| State | CURRENT_IMPLEMENTATION | TARGET_CONTRACT | score 사용 가능 | fusion 포함 | system 영향 | 종료 조건 |
|---|---|---|---|---|---|---|
| `NOT_CONNECTED` | `BaseSensor.current_state` | `BaseSensor.current_state` | `false` (0.0) | `false` | `DEGRADED` / `FAULT` | Hardware connection established |
| `HARDWARE_BACKEND_NOT_IMPLEMENTED` | `BaseSensor.current_state` | `BaseSensor.current_state` | `false` (0.0) | `false` | `DEGRADED` / `FAULT` | Real hardware driver installed & initialized |
| `WARMING_UP` | `SensorState.WARMING_UP` + `PIRSensorAdapter` (L71) + `MMWaveStreamAdapter.is_ready()` | `SensorState.WARMING_UP` | `false` (0.0) | `false` | `DEGRADED` / `FAULT` | Ring buffer / Startup grace period (5.0s) completed |
| `NORMAL` | `BaseSensor.current_state` | `BaseSensor.current_state` | `true` | `true` | `OK` (`NORMAL` / `CAUTION` / `DANGER`) | Disconnection, timeout, or invalid format |
| `READ_TIMEOUT` | `BaseSensor.current_state` | `BaseSensor.current_state` | `false` (0.0) | `false` | `DEGRADED` | Packet received before `timeout_sec` |
| `INVALID_FORMAT` | `BaseSensor.current_state` | `BaseSensor.current_state` | `false` (0.0) | `false` | `DEGRADED` | Valid packet format & frame length restored |
| `NAN_OR_INF` | `BaseSensor.current_state` | `BaseSensor.current_state` | `false` (0.0) | `false` | `DEGRADED` | Finite numeric telemetry restored |
| `OUT_OF_BOUNDS` | `BaseSensor.current_state` | `BaseSensor.current_state` | `false` (0.0) | `false` | `DEGRADED` | Values return to plausible physical range |
| `STALE` | `BaseSensor.health()` age check | `BaseSensor.current_state` | `false` (0.0) | `false` | `DEGRADED` | Fresh sample received within `stale_sec` |
| `INFER_FAILED` | `BaseSensor.current_state` | `BaseSensor.current_state` | `false` (0.0) | `false` | `DEGRADED` | Interpreter exception resolved |
| `FAULT` | Handled via `InferenceResult.state="FAULT"` | `SensorState.FAULT` | `false` (0.0) | `false` | `DEGRADED` / `FAULT` | Sensor reboot or hardware fault cleared |
| `SHUTDOWN` | `BaseSensor.close()` | `BaseSensor.current_state` | `false` (0.0) | `false` | `SHUTDOWN` | Re-initialization / system restart |

---

## 6. Thermal-44 Contract

### Data Pipeline Stage Trace
- **Current Mock Provenance**: `MockThermalSensor.read() -> ThermalInterpreter -> InferenceResult -> SafeNestRiskEngine.evaluate()` (`mock_sensor.py:L35`, `run_node.py:L48`)
- **Current Real Provenance**: injected external provider or `EXTERNAL_SENSOR_PROVIDER_REQUIRED -> InferenceResult -> risk/risk_engine.py`
- **Target Real Provenance**: `Physical SPI Sensor -> ThermalFrameParser -> ThermalInterpreter -> SafeNestRiskEngine` (`thermal44_driver.py:L55`)

### Stage-by-Stage Verification & Gap Analysis

| Property | CURRENT_IMPLEMENTATION | TARGET_CONTRACT | Verification Status | Code Rationale |
|---|---|---|---|---|
| Grid Resolution | $80$ cols $\times 62$ rows ($4,960$ pixels) | $80 \times 62$ ($4,960$ pixels) | `VERIFIED_CODE` | `sensors/thermal44/frame_parser.py:L21` |
| Raw Physical Unit | `UNVERIFIED_DATASHEET` | Vendor raw physical unit (`UNVERIFIED_DATASHEET`) | `UNVERIFIED_DATASHEET` | Missing SPI raw packet datasheet |
| Model Input Tensor | `[1, 62, 80, 1]`, `int8`, scale=0.0039215688, zero_point=-128 | `[1, 62, 80, 1]`, `int8`, scale=0.0039215688, zero_point=-128 | `VERIFIED_MODEL` | `models/model_manifest.json:L18-L27` |
| Model Output Tensor | `[1, 3]`, `int8`, scale=0.00390625, zero_point=-128 | `[1, 3]`, `int8`, scale=0.00390625, zero_point=-128 | `VERIFIED_MODEL` | `models/model_manifest.json:L29-L38` |
| Model Class Map | `0: NOT_HUMAN`, `1: HUMAN_NORMAL`, `2: HUMAN_FALL` | `0: NOT_HUMAN`, `1: HUMAN_NORMAL`, `2: HUMAN_FALL` | `VERIFIED_MODEL` | `models/model_manifest.json:L39-L43` |
| Training Preprocessing | Per-frame min-max `(arr - min)/(max - min)` | Per-frame min-max `(arr - min)/(max - min)` | `TRAIN_INFERENCE_PREPROCESSING_MATCH` | `thermal_prep.py:L142-L146`, `thermal_interpreter.py:L155-L162` |
| Quantization Ownership | `ThermalInterpreter` exclusively owns normalization & INT8 quantizer | Single quantization owner (`ThermalInterpreter`) | `VERIFIED_CODE` | `inference/thermal_interpreter.py:L155-L188` |
| Absolute Temp (°C) Loss | Temperature context eliminated by per-frame scaling | Documented algorithm constraint | `DESIGN_LIMITATION` | `thermal_interpreter.py:L155-L162` |
| Parser Runtime Path | `ThermalFrameParser` imported in driver but uncalled on production path | Hardware SPI parser invocation (`CONTRACT_GAP`) | `DORMANT_CODE_CONTRACT_RISK` | `thermal44_driver.py:L15`, `safenest_risk_engine.py:L215` |
| Parser Quantization Scale | `frame_parser.py:L31` uses `0.003814697265625` (1/262.144) vs manifest `0.003921568859368563` | Deprecate parser quantizer to avoid double quantization | `DORMANT_CODE_CONTRACT_RISK` | `sensors/thermal44/frame_parser.py:L31` |
| Parser Buffer Format | `np.frombuffer(raw_buffer, dtype=np.float32)` (requires 19,840B) | 16-bit raw packet parser (9,920B) | `IMPLEMENTATION_DEFECT` | `sensors/thermal44/frame_parser.py:L17` |
| Byte Order & Endianness | Unhandled in parser | Explicit Little-Endian / Big-Endian specification | `UNVERIFIED_DATASHEET` / `BLOCKED_HARDWARE` | Missing raw SPI datasheet/capture |
| Raw Count to °C Formula | Unhandled in parser | Verified conversion equation | `UNVERIFIED_DATASHEET` / `BLOCKED_HARDWARE` | Missing raw SPI datasheet/capture |
| Invalid Pixel Masking | Unhandled (`np.isfinite()` check only) | Bad pixel mask & interpolation (`CONTRACT_GAP`) | `CONTRACT_GAP` | `sensors/thermal44/frame_parser.py:L25` |
| All-Zero Frame Handling | Returns zero array when `max == min == 0` | Mark `valid = false` (`CONTRACT_GAP`) | `CONTRACT_GAP` | `thermal_prep.py:L147`, `thermal_interpreter.py:L161` |

---

## 7. Thermal 9,920B vs 10,080B Investigation

### Numerical Breakdown & Rationale
- Width: $80$
- Height: $62$
- Total Pixel Count: $80 \times 62 = 4,960 \text{ pixels}$
- Candidate Pixel Payload: $4,960 \text{ pixels} \times \text{assumed } 2 \text{ bytes/pixel} = 9,920 \text{ bytes}$
- Legacy Claimed Frame Size: $10,080 \text{ bytes}$
- Unverified Difference: $10,080 - 9,920 = 160 \text{ bytes}$

### Formal Status Specification

```yaml
candidate_pixel_payload_bytes: 9920
assumption: 4960 pixels × assumed 2 bytes/pixel
hardware_verified: false
claimed_total_frame_bytes: 10080
unverified_overhead_bytes: 160
parser_implementation_allowed: false
status: BLOCKED_HARDWARE
datasheet_status: UNVERIFIED_DATASHEET
reason: 공식 Thermal-44 packet specification 또는 실제 raw packet dump가 존재하지 않음
```

---

## 8. mmWave Contract and Operating Mode

### Mode Classification & Gate Analysis
- **Software Mode**: `MODEL_SEQUENCE_MODE` (`VERIFIED_CODE` / `VERIFIED_ARTIFACT`)
- **Software Path Status**: `VERIFIED_CODE`
- **Capture Status**: `VERIFIED_ARTIFACT`
- **Model Semantic Compatibility**: `BLOCKED_HARDWARE`
- **Deployment Ready**: `false`

### Empirical Log Capture Metrics Table

#### Recalculation Methodology
- **`null_ratio`**: Calculated on total records before filtering missing or non-finite values:
  $$\text{null\_ratio} = \frac{\text{count}(p \text{ is Null or non-finite})}{\text{total\_records}}$$
- **`consecutive_dup_ratio`**: Calculated strictly over valid adjacent pairs from the raw record sequence without skipping/bridging nulls:
  $$\text{valid\_pairs} = \{ (p_i, p_{i+1}) \mid \text{is\_finite}(p_i) \land \text{is\_finite}(p_{i+1}) \}$$
  $$\text{pair\_count} = \text{len}(\text{valid\_pairs})$$
  $$\text{dup\_ratio} = \frac{\sum_{(a,b) \in \text{valid\_pairs}} (a == b)}{\text{pair\_count}}$$

| Metric / Parameter | Value from Captured JSONL Artifacts | Recalculation Formula & Methodology | Source File Evidence |
|---|---|---|---|
| Captured Log Files Audited | 23 JSONL log files ($>20,000$ total records) | Count of `.jsonl` files in `firmware/.../logs` | `firmware/esp_wroom32_mr60_monitor/logs/` |
| Telemetry Output Interval | Mean $100.05\text{ ms}$ ($10\text{ Hz}$) | $\text{mean}(\Delta t_{\text{monotonic\_ms}})$ | `logs/breath/2026-07-25_breath_paced_12rpm.jsonl` |
| Sensor Phase Age (`phase_age_ms`) | Mean $29.59\text{ ms}$ (Max $56\text{ ms}$) | Freshness observation only; frame rate is `BLOCKED_HARDWARE` | `logs/breath/2026-07-25_breath_paced_12rpm.jsonl` |
| Actual `0x0A13` Hardware Frame Rate | `BLOCKED_HARDWARE` | Unverified without raw hardware frame timestamps | `main.cpp:L129` |
| `breath_phase` Null Ratio (Presence) | $0.0000$ ($0 / 2,391$ records) | $0 \text{ nulls} / 2,391 \text{ total records}$ | `logs/breath/2026-07-25_breath_paced_12rpm.jsonl` |
| Unique `breath_phase` Values (Presence) | 188 distinct floating point values | $\text{len}(\text{set}(\text{finite\_phases}))$ | `logs/breath/2026-07-25_breath_paced_12rpm.jsonl` |
| Consecutive Duplicate Ratio (Presence) | $0.1188$ ($11.88\%$ consecutive duplicates) | $284 \text{ duplicate pairs} / 2,390 \text{ valid pairs}$ | `logs/breath/2026-07-25_breath_paced_12rpm.jsonl` |
| Consecutive Duplicate Ratio (Empty Room) | $1.0000$ ($100.00\%$ static $0.00$ values) | $3,597 \text{ duplicate pairs} / 3,597 \text{ valid pairs}$ | `logs/baseline/2026-07-25_empty_gate_v1_360s.jsonl` |
| 30-Second Windows Generated (CSV Adapter)| 90 windows generated from 2,998-row, 299.816s exported CSV (`window=30s`, `stride=3s`) | `MMWaveCSVAdapter.iter_windows()` execution | `firmware/.../2026-07-25_occupied_d09_v1_360s__S001_NORMAL_5MIN_01.csv` |

### Pipeline Stage & Semantic Trace

- **Current Mock Provenance**: `MockMMWaveSensor.read() -> MMWaveInterpreter -> InferenceResult -> SafeNestRiskEngine.evaluate()` (`mock_sensor.py:L40`, `run_node.py:L49`)
- **Current Real Provenance**: injected external provider or `EXTERNAL_SENSOR_PROVIDER_REQUIRED -> InferenceResult -> risk/risk_engine.py`
- **Captured Artifact Pipeline**: `MR60BHA2 -> ESP main.cpp -> JSONL -> export_mmwave_csv.py -> CSV`
- **Verified Adapter Stage**: `CSV -> MMWaveCSVAdapter -> MMWaveWindow`
- **Unconnected Candidate Stage**: `MMWaveWindow -> MMWaveInterpreter` (status: `CONTRACT_GAP`)
- **Target Online Provenance**: `MR60BHA2 UART -> Python UART parser -> MMWaveStreamAdapter -> MMWaveInterpreter -> SafeNestRiskEngine`

| Stage | CURRENT_IMPLEMENTATION | TARGET_CONTRACT | Verification Status | Code Rationale |
|---|---|---|---|---|
| UART Parsing | ESP32 parses `0x0A13` into `breathPhase` float | ESP32 parses `0x0A13` into `breathPhase` float | `VERIFIED_CODE` | `firmware/.../main.cpp:L127` |
| JSON / CSV Export | Exports `breath_phase` raw float to CSV `resp_phase` | Exports `breath_phase` raw float to CSV `resp_phase` | `VERIFIED_ARTIFACT` | `export_mmwave_csv.py:L116` |
| Resampling & Window | 10Hz linear interpolation into 300-sample window | 10Hz linear interpolation into 300-sample window | `VERIFIED_CODE` | `adapters/mmwave_csv_adapter.py:L104` |
| Python Serial Driver | Raises `HardwareBackendUnavailable` | Hardware Python UART serial reader (`CONTRACT_GAP`)| `CONTRACT_GAP` / `BLOCKED_HARDWARE` | `sensors/mmwave/mmwave_adapter.py:L30` |
| Model Preprocessing | Z-Score normalization `(x - mean)/std` (`mean=0.006091983988881111`, `std=2.5013835430145264`) | Z-Score normalization `(x - mean)/std` | `VERIFIED_MODEL` / `VERIFIED_ARTIFACT` | `inference/mmwave_interpreter.py:L177`, `models/mmwave/sensor_stats_metadata_v0.1.0.json:L8-L9` |
| Model Input Tensor | `[1, 300, 1]`, `int8`, scale=0.03259857, zero_point=-13 | `[1, 300, 1]`, `int8`, scale=0.03259857, zero_point=-13 | `VERIFIED_MODEL` | `models/model_manifest.json:L92-L105` |
| Model Semantic Alignment | Firmware `breathPhase` vs Metadata `resp_phase_unwrapped_clutter_removed` | Vendor phase float / `UNVERIFIED_DATASHEET` | `BLOCKED_HARDWARE` / `UNVERIFIED_DATASHEET` | Missing unwrapping proof for ESP `breathPhase` |

---

## 9. CO₂ Contract

### Pipeline Stage Trace
- **Current Mock Provenance**: `MockCO2Sensor.read() -> CO2Interpreter -> InferenceResult -> SafeNestRiskEngine.evaluate()` (`mock_sensor.py:L45`, `run_node.py:L50`)
- **Current Real Provenance**: injected external provider or `EXTERNAL_SENSOR_PROVIDER_REQUIRED -> InferenceResult -> risk/risk_engine.py`
- **Target Real Provenance**: `I2C_SCD40 -> CO2SensorAdapter -> CO2Interpreter -> SafeNestRiskEngine`

### Stage-by-Stage Verification & Gap Analysis

| Property | CURRENT_IMPLEMENTATION | TARGET_CONTRACT | Verification Status | Code Rationale |
|---|---|---|---|---|
| Feature Order | `[co2_slope, humidity, co2_ppm]` | `[co2_slope, humidity, co2_ppm]` | `VERIFIED_ARTIFACT` / `VERIFIED_MODEL` | `co2_scaling_metadata_v0.1.0.json:L2-L6`, `co2_interpreter.py:L125` |
| Slope Calculation | `(current_ppm - first_ppm) / elapsed_min` (30-sample history) | `(current_ppm - first_ppm) / elapsed_min` | `VERIFIED_CODE` | `sensors/co2/co2_adapter.py:L39-L43` |
| Metadata Scaling | `mean=[0.011184, 25.73058, 606.48127]`, `scale=[4.37341, 5.53244, 314.38715]` | `mean=[0.011184, 25.73058, 606.48127]`, `scale=[4.37341, 5.53244, 314.38715]` | `VERIFIED_ARTIFACT` | `models/co2/co2_scaling_metadata_v0.1.0.json:L7-L16` |
| Model Input Tensor | `[1, 3]`, `int8`, scale=0.00582845, zero_point=57 | `[1, 3]`, `int8`, scale=0.00582845, zero_point=57 | `VERIFIED_MODEL` | `models/model_manifest.json:L56-L65` |
| Model Output Tensor | `[1, 2]`, `int8`, `0: VACANT`, `1: OCCUPIED` | `[1, 2]`, `int8`, `0: VACANT`, `1: OCCUPIED` | `VERIFIED_MODEL` | `models/model_manifest.json:L66-L79` |
| Adapter Predict Invocation | `predict(co2_slope, humidity, co2_ppm)` 3 float scalars | Same 3-scalar call | `RESOLVED` | `sensors/co2/co2_adapter.py` and `tests/test_sensor_startup_warmup.py` |
| Python I2C Driver | Raises `HardwareBackendUnavailable` | Real I2C SCD40 reader (`CONTRACT_GAP`) | `CONTRACT_GAP` / `BLOCKED_HARDWARE` | `sensors/co2/co2_adapter.py:L30` |
| Valid PPM Bounds | $300 \le \text{co2\_ppm} \le 10000$ checked in risk engine | $300 \le \text{co2\_ppm} \le 10000$ | `VERIFIED_CODE` | `integrated_node/safenest_risk_engine.py:L127` |
| SCD40 Warm-up & CRC-8 | Unhandled in adapter | I2C CRC-8 & Data-Ready polling (`CONTRACT_GAP`) | `UNVERIFIED_DATASHEET` / `BLOCKED_HARDWARE` | Missing SCD40 I2C hardware driver |

---

## 10. PIR Temporal Rule Contract

### Pipeline Stage Trace
- **Current Mock Provenance**: `MockPIRSensor.read() -> InferenceResult -> SafeNestRiskEngine.evaluate()` (`mock_sensor.py:L51`, `run_node.py:L51`)
- **Current Real Provenance**: injected external provider or `EXTERNAL_SENSOR_PROVIDER_REQUIRED -> InferenceResult -> risk/risk_engine.py`
- **Alternate Pipeline Provenance**: `packet["pir"]["motion"] -> integrated_node/safenest_risk_engine.py:L353 -> RiskRulesEvaluator.evaluate_motion()`
- **Target Real Provenance**: `GPIO_PIN -> PIRSensorAdapter -> SafeNestRiskEngine`

### Quality Fields Categorization

| Quality Field | Field Category | Provided In Current Output? | Code Location |
|---|---|---|---|
| `is_no_motion`, `elapsed_since_motion_sec` | Current Primary Telemetry | Yes (`InferenceResult.metadata`) | `sensors/pir/pir_adapter.py:L64` |
| `startup_grace_period_sec`, `elapsed_since_start` | Current WARMING_UP Telemetry | Yes (`InferenceResult.metadata`) | `sensors/pir/pir_adapter.py:L84` |
| `age_sec`, `read_count`, `error_count` | Sensor Health Only | Yes (`BaseSensor.health()`) | `sensors/base_sensor.py:L77` |
| `presence_confirmed` | Alternate / Target Pipeline Only | No (`CONTRACT_GAP` in primary) | `safenest_risk_engine.py:L353` |

### Stage-by-Stage Verification & Gap Analysis

| Property | CURRENT_IMPLEMENTATION | TARGET_CONTRACT | Verification Status | Code Rationale |
|---|---|---|---|---|
| Model Specification | `Model: NOT_APPLICABLE`, `Decision type: TEMPORAL_RULE` | `Model: NOT_APPLICABLE`, `Decision type: TEMPORAL_RULE` | `NOT_APPLICABLE` | `sensors/pir/pir_adapter.py:L52-L74` |
| Telemetry Timestamp | `time.time()` wall-clock seconds (`pir_adapter.py:L43`) | `time.time()` wall-clock seconds | `VERIFIED_CODE` | `sensors/pir/pir_adapter.py:L43`, `L52` |
| Temporal Elapsed Clock | `time.monotonic()` process-monotonic (`pir_adapter.py:L44`) | `time.monotonic()` process-monotonic | `VERIFIED_CODE` | `sensors/pir/pir_adapter.py:L44`, `L47` |
| Current Sampling | Integrated node periodic polling (`run_node.py:L84`) | GPIO event/edge-driven interrupt (`CONTRACT_GAP`) | `VERIFIED_CODE` / `CONTRACT_GAP` | `integrated_node/run_node.py:L84` |
| Startup Grace Period | `startup_grace_period_sec = 5.0` (`pir_adapter.py:L21`) | 5.0s grace period (`WARMING_UP` state) | `VERIFIED_CODE` | `sensors/pir/pir_adapter.py:L71` |
| Last Motion Timestamp | Updated on `motion_detected == True` | Updated on `motion_detected == True` | `VERIFIED_CODE` | `sensors/pir/pir_adapter.py:L48` |
| No-Motion Threshold | `no_motion_threshold_sec = 15.0` seconds | `no_motion_threshold_sec = 15.0` seconds | `VERIFIED_CODE` | `config/sensors.yaml:L51`, `risk_rules.py:L303` |
| Presence Gate | Reset timer if `presence_confirmed == False` | Reset timer if `presence_confirmed == False` (`CONTRACT_GAP`) | `CONTRACT_GAP` | `risk/risk_rules.py:L280-L284` (Alternate Engine only) |
| Active High Polarity | Configured in `sensors.yaml`, unapplied in Python code | Active-high logic in GPIO driver (`CONTRACT_GAP`) | `CONTRACT_GAP` | `config/sensors.yaml:L50`, `sensors/pir/pir_adapter.py:L52` |
| Hardware Debounce | Unhandled in adapter | Hardware / software GPIO debounce (`CONTRACT_GAP`) | `CONTRACT_GAP` | `sensors/pir/pir_adapter.py:L52` |
| Stuck High/Low Fault | Simulated in `mock_sensor.py`, unhandled in adapter | Stuck GPIO state fault detector (`CONTRACT_GAP`)| `CONTRACT_GAP` | `sensors/pir/mock_sensor.py:L53`, `pir_adapter.py:L52` |
| Python GPIO Driver | Raises `HardwareBackendUnavailable` | Real GPIO pin interrupt reader (`CONTRACT_GAP`) | `CONTRACT_GAP` / `BLOCKED_HARDWARE` | `sensors/pir/pir_adapter.py:L24` |

---

## 11. Sensor JSON Schema

All individual sensor adapters output an `InferenceResult` telemetry object:

```json
{
  "sensor_id": "thermal44",
  "timestamp": 1722500000.0,
  "score": 1.0,
  "state": "HUMAN_FALL",
  "confidence": 0.95,
  "valid": true,
  "latency_ms": 12.4,
  "error": null,
  "metadata": {}
}
```

### Sensor Field Specification

| Field | Type | Nullable | Unit | Producer | `valid=false` Fallback | Consumer | Verification Status |
|---|---|---|---|---|---|---|---|
| `sensor_id` | `str` | No | Identifier | Sensor Adapter | Sensor ID string | Risk Engine | `VERIFIED_CODE` |
| `timestamp` | `float` | No | Epoch Seconds ($s$) | Sensor Adapter | Current wall-clock (`time.time()`) | Risk Engine | `VERIFIED_CODE` |
| `score` | `float` | No | Normalized $[0.0, 1.0]$ | Sensor Adapter / Interpreter | `0.0` | Risk Engine | `VERIFIED_CODE` |
| `state` | `str` | No | State String | Sensor Adapter / Interpreter | `"NOT_CONNECTED"` / `"INFER_ERROR"` | Risk Engine | `VERIFIED_CODE` |
| `confidence` | `float` | No | Probability $[0.0, 1.0]$ | Interpreter / Rule | `0.0` | Risk Engine | `VERIFIED_CODE` |
| `valid` | `bool` | No | Boolean Flag | Sensor Adapter | `false` | Risk Engine Quality Gate | `VERIFIED_CODE` |
| `latency_ms` | `float` | No | Milliseconds ($ms$) | Sensor Adapter | `0.0` or measured elapsed | System Health Telemetry | `VERIFIED_CODE` |
| `error` | `str` | Yes | Error Description | Sensor Adapter / Interpreter | Exception message / Error code | System Diagnostics | `VERIFIED_CODE` |
| `metadata` | `dict` | No | Key-Value Metadata | Sensor Adapter / Interpreter | `{}` or error context | Debug / Logging | `VERIFIED_CODE` |

---

## 12. System JSON Schema

### PRIMARY_RUNTIME_SCHEMA
The primary runtime schema emitted by `inference/inference_result.py:SafeNestRiskOutput` and `risk/risk_engine.py`:

```json
{
  "timestamp": 1722500000.0,
  "risk_score": 42.5,
  "risk_level": "CAUTION",
  "system_health": "DEGRADED",
  "degraded_mode": true,
  "invalid_sensors": ["co2"],
  "stale_sensors": [],
  "component_scores": {
    "mmwave": 0.5,
    "co2": null,
    "pir": 0.0,
    "thermal": 0.0
  },
  "is_emergency": false,
  "reasons": ["CO2_SENSOR_DISCONNECTED"],
  "sensors": {},
  "metadata": {},
  "level": "CAUTION",
  "system_status": "DEGRADED",
  "fallback_used": true
}
```

### ALTERNATE_PIPELINE_SCHEMA
The alternate schema emitted by integrated node virtual engine (`integrated_node/safenest_risk_engine.py:L428-L491`):

```json
{
  "risk_score": 100.0,
  "status_str": "DANGER",
  "status_code": 2,
  "is_emergency": true,
  "reasons": ["EMERGENCY_FALL"],
  "sensor_quality": {"thermal": 1.0, "co2": 1.0, "mmwave": 1.0, "pir": 1.0},
  "system_status": "OK",
  "v4_fusion": {},
  "legacy_fusion": {},
  "derived_metrics": {},
  "model_meta": {}
}
```

---

## 13. Contract Mismatches

| ID | Location | Target Component | Description | Category | Impact |
|---|---|---|---|---|---|
| `MISMATCH-01` | `sensors/thermal44/frame_parser.py:L17` | Thermal Parser | `np.frombuffer(raw_buffer, dtype=np.float32)` expects 4 bytes per pixel ($19,840\text{B}$ for $4,960$ pixels), breaking 16-bit raw ($9,920\text{B}$) packets. | `IMPLEMENTATION_DEFECT` | Prevents ingestion of 16-bit raw SPI hardware frames (dormant in active runtime). |
| `MISMATCH-02` | `sensors/thermal44/frame_parser.py:L31` | Thermal Parser | Uses quantization scale `0.003814697265625` (1/262.144) instead of model manifest scale `0.003921568859368563` (1/255.0). | `DORMANT_CODE_CONTRACT_RISK` | Quantization discrepancy in unused parser function. |
| `MISMATCH-03` | `inference/thermal_interpreter.py:L155-L162` | Thermal Interpreter | Applies per-frame min-max normalization `(T - min)/(max - min)`, matching training prep (`thermal_prep.py:L145`). Absolute °C temperature context is lost. | `TRAIN_INFERENCE_PREPROCESSING_MATCH` / `DESIGN_LIMITATION` | Training/Inference match, but absolute temperature information eliminated by design. |
| `MISMATCH-04` | `sensors/co2/co2_adapter.py` | CO₂ Adapter | `predict(co2_slope, humidity, co2_ppm)` now passes the required three scalar floats. | `RESOLVED` | Regression test enforces call arity and feature order. |

---

## 14. Blocked Hardware Verification

1. `BLOCKER-01` (**Thermal 160B Overhead**): Candidate $9,920\text{B}$ payload vs claimed $10,080\text{B}$ frame size. Missing datasheet or raw packet spec (`BLOCKED_HARDWARE` / `UNVERIFIED_DATASHEET`).
2. `BLOCKER-02` (**Thermal Real SPI Driver**): `Thermal44Sensor.connect()` raises `HardwareBackendUnavailable`. Physical SPI read unverified (`BLOCKED_HARDWARE`).
3. `BLOCKER-03` (**mmWave Real UART Driver**): `MMWaveSensorAdapter.connect()` raises `HardwareBackendUnavailable`. Physical serial port reader unverified (`BLOCKED_HARDWARE`).
4. `BLOCKER-04` (**SCD40 Real I2C Driver**): `CO2SensorAdapter.read_raw_values()` raises `HardwareBackendUnavailable`. Real I2C CRC and warm-up unverified (`BLOCKED_HARDWARE`).
5. `BLOCKER-05` (**PIR Real GPIO Driver**): `PIRSensorAdapter.read_gpio()` raises `HardwareBackendUnavailable`. Physical GPIO edge interrupt unverified (`BLOCKED_HARDWARE`).

---

## 15. Required Captures and Datasheets

To clear `BLOCKED_HARDWARE` and `UNVERIFIED_DATASHEET` statuses:
1. **Thermal-44 SPI Capture & Datasheet**: Capture 10 raw SPI binary frame buffers; obtain official manufacturer pinout and frame structure datasheet.
2. **MR60BHA2 UART Capture**: Capture 60 seconds of 115200 baud UART stream to verify `0x0A13` frame timing stability on Raspberry Pi 5.
3. **SCD40 I2C Capture & Datasheet**: Capture raw 9-byte I2C response packets to verify CRC-8 and data-ready bit timing.
4. **PIR Logic Analyzer Trace**: Capture GPIO 17 pulse trace during motion to confirm active-high polarity and bounce duration.

---

## 16. Required Follow-up Implementation

Recommended changes for follow-up PRs (without modifying existing files in P0-3):
1. **Single Quantization Ownership**: Confirm `ThermalInterpreter` as the sole owner of per-frame normalization and TFLite INT8 quantization. `ThermalFrameParser` should be limited strictly to SPI packet validation and float frame generation (`(62, 80) float32`). Deprecate or remove `normalize_to_int8()` in `ThermalFrameParser` to prevent double quantization risks.
2. **CO₂ provider integration**: Preserve the resolved three-scalar AI call while the sensor team supplies SCD40 transport values and measured cadence.

---

## 17. Acceptance Checklist

- [x] P0-2 ground truth validation completed (`GroundTruthValidator.validate_all()` returned `is_valid: True`).
- [x] Thermal preprocessing reclassified as `TRAIN_INFERENCE_PREPROCESSING_MATCH` (matches `thermal_prep.py:L145`) and Celsius temperature loss classified as `DESIGN_LIMITATION`.
- [x] Thermal 9,920 bytes expressed as candidate payload with `hardware_verified: false`. Baseless Recommendations (uint16/int16 change, WARMING_UP enum addition) deleted from Section 16.
- [x] ThermalFrameParser classified as `DORMANT_CODE_CONTRACT_RISK` (uncalled in active runtime path). Single quantization ownership confirmed for `ThermalInterpreter`.
- [x] mmWave captured JSONL metrics table populated with empirical log calculations (recalculated null ratio before filtering and duplicate pair ratio = 1.0 for empty log).
- [x] mmWave semantic compatibility marked `BLOCKED_HARDWARE` (`deployment_ready: false`). Phase units marked `Vendor phase float / UNVERIFIED_DATASHEET`.
- [x] All timestamps classified into wall-clock, process-monotonic, ESP-monotonic-ms, or CSV-session-relative. Wall-clock `time.time()` is explicitly documented as non-monotonic. PIR wall-clock (`time.time()`) and elapsed clock (`time.monotonic()`) clearly separated.
- [x] PIR quality fields separated into current primary metadata, health-only fields, and alternate/target fields (`presence_confirmed`).
- [x] mmWave offline pipeline split into captured_artifact, verified_adapter, and unconnected_candidate (`CONTRACT_GAP`).
- [x] Alternate PIR provenance corrected to `packet["pir"]["motion"] -> integrated_node/safenest_risk_engine.py:L353 -> RiskRulesEvaluator.evaluate_motion()`.
- [x] Stale line references updated with class and method symbols (e.g. `sensors/co2/co2_adapter.py:L117`, `integrated_node/safenest_risk_engine.py:L353`).
- [x] Automated contract test suite (`tests/test_sensor_model_data_contract.py`) strengthened with AST call/assignment inspection, PIR dual-clock check, startup grace period check, synthetic missing-value regression tests, MMWaveCSVAdapter regression (90 windows), and exact capture metric recalculations.
