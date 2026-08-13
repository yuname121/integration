# SafeNest mmWave M-B8 — macOS Offline Latency & Footprint

- Scope: `MAC_OFFLINE_LATENCY_AND_FOOTPRINT`
- Machine: `Mac14,2` / `Apple M2` / macOS `26.5.2`
- Runtime: TensorFlow `2.20.0`, `num_threads=1`, `DEFAULT_TFLITE_CPU_RUNTIME_WITH_AUTOMATIC_XNNPACK_CPU_DELEGATE_OBSERVED`
- Inputs: 79 deterministic VALIDATION windows; LOCKED_TEST access `0`
- Formal policy: 3 rotated series × 1000 samples/seed/metric after 100 warm-ups

## Timing definitions

- `TFLITE_INVOKE_ONLY`: `set_tensor` is outside the timed interval; only `interpreter.invoke()` is timed.
- `PREPROCESSING_ONLY`: frozen M-B1 BPF + TRAIN-fitted Z-score and the M-B6 model-ready `float32` cast on an in-memory canonical 300-sample window.
- `QUANTIZATION_ONLY`: frozen strict-INT8 input quantization only.
- `PREPROCESSING_QUANTIZATION_INVOKE`: preprocessing, quantization, `set_tensor`, and invoke; output dequantization/argmax excluded.

## Raw-sample provenance

- `latency_raw_samples.npz` contains 39 positive integer-nanosecond arrays; warm-up samples are excluded.
- `benchmark_run_index.json` binds every array to its seed, strict-INT8 model path/SHA/bytes, metric, thread count, delegate/runtime mode, warm-up count, series, and deterministic 79-window cycle.
- All primary summaries use every valid measured sample with NumPy percentile method `linear`; no latency outliers were removed.

## Per-seed strict-INT8 latency

| Seed | Invoke median ms | Invoke P95 ms | Invoke P99 ms | Pipeline median ms | Pipeline P95 ms | Pipeline P99 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.006583000 | 0.007375000 | 0.009333010 | 0.137500000 | 0.150166050 | 0.164461750 |
| 43 | 0.006625000 | 0.008209000 | 0.008917000 | 0.139292000 | 0.152750000 | 0.164212330 |
| 44 | 0.006667000 | 0.007084000 | 0.008959410 | 0.138042000 | 0.150208050 | 0.166380000 |

## Per-seed preprocessing and quantization latency

| Seed | Preprocessing median ms | Preprocessing P95 ms | Preprocessing P99 ms | Quantization median ms | Quantization P95 ms | Quantization P99 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.122042000 | 0.133458000 | 0.142669920 | 0.005083000 | 0.005375000 | 0.005959410 |
| 43 | 0.128208000 | 0.140416050 | 0.163178250 | 0.005167000 | 0.005916050 | 0.012877910 |
| 44 | 0.120083000 | 0.130335100 | 0.143382500 | 0.005042000 | 0.005458000 | 0.006000830 |

## Cross-seed runtime summary

- Mean of invoke medians: `0.006625000 ms`
- Invoke median relative spread: `0.012760140`
- Maximum invoke P99: `0.009333010 ms`
- Maximum pipeline P99: `0.166380000 ms`

## Mac-development reference comparison

- All invoke medians below 5 ms: `True`
- All invoke P99 values below 15 ms: `True`
- Interpretation: `MAC_DEVELOPMENT_REFERENCE_ONLY`; these are not deployment or hardware acceptance criteria.

## Static footprint and memory observation

- Parameter count: `9315`
- Strict-INT8 file bytes: seed42=22080, seed43=22136, seed44=22136
- Memory method: `PROCESS_RSS_PROXY` / `PS_RSS_KIB_PROCESS_PROXY`
- Process RSS is an observational proxy, not a TFLite arena or model-RAM claim.
- Peak memory status: `PEAK_MODEL_MEMORY_NOT_RELIABLY_MEASURABLE_ON_CURRENT_MAC_RUNTIME`

## Confirmation stability

- seed42: median difference `0.076105`; P95 ratio `0.988746`; warning `False`
- seed43: median difference `0.006340`; P95 ratio `0.822268`; warning `False`
- seed44: median difference `0.043648`; P95 ratio `1.005788`; warning `False`

## Interpretation and limitations

The model consumes a 30-second observation window, but that window-acquisition duration is not CPU model-inference time.
These values describe steady-state offline compute on this specific Mac only. `<5 ms` and `P99 <15 ms` are `MAC_DEVELOPMENT_REFERENCE_ONLY`, not Raspberry Pi, real-sensor, sensor-to-alarm, MR60, or production-real-time claims.
M-B8 benchmarks this specific Mac environment only; it performs no model training, model conversion, seed selection, or LOCKED_TEST access.
