# P0-1~P0-5 Integrated Validation Report

Validation date: 2026-08-01 (macOS host)

## 1. Executive Summary

| Scope | Status | Verified result |
|---|---|---|
| P0-1 Real-mode fail-closed | PASS (software) | Unimplemented hardware backends return explicit invalid results and never synthetic normal data |
| P0-2 Model/config ground truth | PASS | Three real TFLite files, manifest, config, SHA-256, tensors, quantization, class order and runtime mappings agree |
| P0-3 Sensor-model data contract | PASS WITH BLOCKERS | Contract and code/artifacts agree; unresolved wire formats and physical units remain explicitly blocked by hardware/datasheets |
| P0-4 Risk/system-health separation | PASS | Human risk and pipeline health are separate; invalid/stale channels are explicit and valid channels are reweighted |
| P0-5 Startup/warm-up safety | PASS (software) | Historical-window sensors suppress inference until ready; reconnect and non-monotonic timestamps are covered |
| Full unittest suite | PASS | 163 discovered, 161 passed, 2 skipped, 0 failures, 0 errors |
| Raspberry Pi 5 and physical sensors | BLOCKED_HARDWARE | SPI/I2C/UART/GPIO electrical communication and target performance were not measured on this Mac |

The P0 software baseline is ready for team integration. This does **not** mean the physical-sensor product is production-certified. Real Raspberry Pi 5 drivers, raw packet contracts and target-device benchmarks remain separate hardware tasks.

## 2. Corrections Applied During Final Review

1. `CO2SensorAdapter` now calls the real interpreter contract as `predict(co2_slope, humidity, co2_ppm)`. Autospec and real-TFLite integration tests prevent regression to a single-array call.
2. P0-4 output preserves the new primary fields while keeping existing receiver aliases:
   - Primary: `risk_level`, `system_health`, `degraded_mode`, `invalid_sensors`, `stale_sensors`, `component_scores`
   - Compatibility: `level`, `system_status`, `fallback_used`
3. Runtime input key `thermal44` remains accepted and preserved in `sensors`; the normalized component key is `thermal`.
4. All-sensor failure produces `system_health="FAILED"`, `risk_score=null`, and `risk_level=null`. Compatibility aliases remain `system_status="FAULT"` and `level="FAULT"`; failure cannot appear as `NORMAL`.
5. The package-local `scripts/validate_v4_config.py` entrypoint no longer forwards with `exec`. The previous forwarding retained the wrong `__file__`, misidentified the repository root and falsely reported a missing manifest when launched inside the package.
6. README and team handoff examples no longer claim the obsolete 74-test count and now document the P0-4 output contract.

## 3. Risk and System-Health Contract

The healthy-channel formula is:

```text
R = 100 * (0.35*S_mmwave + 0.35*S_co2 + 0.15*S_pir + 0.15*S_thermal)
```

Thresholds are `NORMAL < 30`, `CAUTION >= 30 and < 60`, and `DANGER >= 60`. A valid Thermal fall or valid mmWave apnea score of `1.0` applies the documented emergency override (`R=100`, `DANGER`).

When one or more channels are invalid or stale, the engine:

- lists the channels in `invalid_sensors` or `stale_sensors`;
- sets their `component_scores` value to `null`;
- sets `system_health="DEGRADED"` and `degraded_mode=true`;
- renormalizes only the remaining valid channel weights;
- keeps the human-risk result separate from the degraded health result.

When no valid channel remains, the engine does not invent a risk score.

## 4. Startup and Recovery Results

- mmWave: fewer than 300 samples returns `WARMING_UP` and performs no inference; the 300-sample window invokes the model.
- CO2: insufficient history returns `WARMING_UP`; slope is calculated in `ppm/min`; a 500-to-600 ppm change over 60 seconds is verified as `100.0 ppm/min`.
- PIR: startup grace uses `time.monotonic()`; reconnect resets temporal state. Current runtime state names are `WARMING_UP`, `MOTION`, and `LONG_NO_MOTION`.
- Thermal: `(62, 80)` frame shape and finite values are checked before inference; reconnect/startup does not reuse a synthetic frame.
- All adapters reject reversed timestamps where a temporal stream is required, and reconnect clears their rolling state.

## 5. Reproduction Commands and Results

From `SafeNest_V4_OnDevice_AI/`:

```bash
python3 scripts/validate_v4_config.py
python3 -m unittest \
  tests.test_fallback \
  tests.test_real_mode_fail_closed \
  tests.test_sensor_startup_warmup \
  tests.test_risk_health_separation \
  tests.test_risk_engine -v
python3 -m unittest discover -s tests -p "test_*.py" -v
python3 -m compileall -q inference risk sensors integrated_node scripts
```

Observed results:

```text
Model/config ground-truth validation: PASSED
Focused P0 regression suite: 36 passed, 0 failed, 0 errors
Full suite: 163 discovered, 161 passed, 2 skipped, 0 failed, 0 errors
Python bytecode compilation: PASSED
```

The two skipped Thermal tests require the offline `thermal/processed_thermal_80x62.npz` dataset. Model loading, SHA-256, tensor contract, invalid input rejection and integration paths still run without that NPZ.

## 6. Remaining Hardware-Dependent Work

The following items are not software-test failures and must not be marked complete until verified on the Raspberry Pi 5 with actual devices:

1. Thermal-44 SPI framing, byte order, 9,920-byte pixel payload versus the unverified extra 160 bytes, raw-count temperature conversion and orientation.
2. mmWave UART framing, actual sample cadence and proof that the delivered phase value matches the model's training semantic.
3. SCD40 I2C data-ready timing, CRC-8, warm-up behavior and sustained 30-second history capture.
4. PIR GPIO edge handling, electrical disconnect/stuck-state detection and debounce behavior.
5. Raspberry Pi 5 latency, throughput, memory and long-duration stability measurements.

Therefore the correct final label is:

```text
P0 SOFTWARE BASELINE: PASS
PHYSICAL SENSOR / RASPBERRY PI 5 VALIDATION: BLOCKED_HARDWARE
FULL PRODUCT PRODUCTION READINESS: NOT YET CLAIMED
```
