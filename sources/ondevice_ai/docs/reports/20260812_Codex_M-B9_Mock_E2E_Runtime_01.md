# SafeNest mmWave M-B9 — Explicit-Finalist Mock E2E Runtime

- Scope: `EXPLICIT_FINALIST_MOCK_E2E_RUNTIME_COMPATIBILITY`
- Frozen finalists: seeds `42`, `43`, `44`; no seed selection was performed.
- Input scope: deterministic pure-class VALIDATION windows only; LOCKED_TEST access `0`.
- Model scope: M-B6 Stage-C strict INT8 artifacts through phase-local runtime manifests; no binaries duplicated.
- Execution scope: bounded `SafeNestIntegratedNode(..., sensors=...)` calls with `start()`, one `step()`, and `shutdown()` in `finally`.
- M-B8 formal latency benchmark was completed in the predecessor phase; M-B9 did not rerun it.

## Shared default warning

The shared `models/model_manifest.json` still identifies the historical blocked mmWave model. M-B9 does not modify it and does not use it for finalist inference; the integrated node reads it only for its existing non-production deployability gate.

## Preprocessing identity

Authoritative M-B1 `BPF_ZSCORE` (0.1–0.5 Hz, fourth-order zero-phase Butterworth, TRAIN-fitted global z-score) was compared independently with the repaired explicit runtime path. BPF, z-score, model-ready float32, int8 input, saturation count, top-1, and output int8/probability vectors are required to match exactly.
The pre-run audit classified the legacy z-score-only path as `M-B9_RUNTIME_PREPROCESSING_MISMATCH`; the required refinement is recorded as resolved locally in the explicit interpreter path.

| Seed | Runtime model ID | SHA-256 | Bytes | Strict INT8 | Flex/Select absent |
|---:|---|---|---:|---|---|
| 42 | `M-B3_CONV1D_GAP_BASELINE_seed42_M-B6_STRICT_INT8` | `6dff6aaa72c79d76715d40cf7e32bb1e6cd9b2c2e3ac78eaf2fda737561430c5` | 22080 | `True` | `True` |
| 43 | `M-B3_CONV1D_GAP_BASELINE_seed43_M-B6_STRICT_INT8` | `cf39c5ce28b4e495d2d721ec5456713618a8f19c3dbe55c600ca222d0228d8f6` | 22136 | `True` | `True` |
| 44 | `M-B3_CONV1D_GAP_BASELINE_seed44_M-B6_STRICT_INT8` | `30a487f73239078e9e22ce09b530750ac16f4850e33cce5af11e6feced98d08d` | 22136 | `True` | `True` |

| Identity check | Exact result |
|---|---|
| BPF output | `True` |
| TRAIN z-score output | `True` |
| model-ready float32 | `True` |
| int8 input and saturation | `True` / `True` |
| direct/runtime top-1, probability, output-int8 | `True` / `True` / `True` |

## Deterministic VALIDATION selection

The lowest canonical index eligible for each pure class was selected before predictions. These are VALIDATION windows, not LOCKED_TEST and not a seed-selection mechanism.

| Truth class | Canonical index | Window ID |
|---|---:|---|
| `NORMAL` | 59 | `dataset-10_5281_zenodo_18599983-p012-lying-post_exercise__W0000` |
| `RAPID_OR_ABNORMAL` | 43 | `dataset-10_5281_zenodo_18599983-p009-lying-post_exercise__W0000` |
| `APNEA` | 44 | `dataset-10_5281_zenodo_18599983-p009-lying-rest__W0000` |

## Scenario and audit boundary

A/B/C use metadata-only scenario truth and actual model prediction; mismatches remain visible. D/O cover history and provider connectivity, E–K cover invalid/fault/exception/timeout paths, L/M cover missing or identity-mismatched finalists, and N is an explicit valid finalist smoke. CO₂/PIR/Thermal providers are neutral wiring support only.

| Scenario | Seed | Truth | Prediction | Match | Score source | Valid/error |
|---|---:|---|---|---|---|---|
| `A_NORMAL` | 42 | NORMAL | RAPID_OR_ABNORMAL | False | `MODEL_PREDICTION` | True / `` |
| `B_RAPID_OR_ABNORMAL` | 42 | RAPID_OR_ABNORMAL | NORMAL | False | `MODEL_PREDICTION` | True / `` |
| `C_APNEA` | 42 | APNEA | APNEA | True | `MODEL_PREDICTION` | True / `` |
| `A_NORMAL` | 43 | NORMAL | RAPID_OR_ABNORMAL | False | `MODEL_PREDICTION` | True / `` |
| `B_RAPID_OR_ABNORMAL` | 43 | RAPID_OR_ABNORMAL | RAPID_OR_ABNORMAL | True | `MODEL_PREDICTION` | True / `` |
| `C_APNEA` | 43 | APNEA | APNEA | True | `MODEL_PREDICTION` | True / `` |
| `A_NORMAL` | 44 | NORMAL | APNEA | False | `MODEL_PREDICTION` | True / `` |
| `B_RAPID_OR_ABNORMAL` | 44 | RAPID_OR_ABNORMAL | NORMAL | False | `MODEL_PREDICTION` | True / `` |
| `C_APNEA` | 44 | APNEA | APNEA | True | `MODEL_PREDICTION` | True / `` |
| `D_INSUFFICIENT_HISTORY` | 42 | - | - | - | `NO_VALID_PREDICTION` | False / `INSUFFICIENT_HISTORY` |
| `E_INVALID_SHAPE` | 42 | - | - | - | `NO_VALID_PREDICTION` | False / `INVALID_SHAPE` |
| `F_NAN` | 42 | - | - | - | `NO_VALID_PREDICTION` | False / `NAN_OR_INF` |
| `G_INF` | 42 | - | - | - | `NO_VALID_PREDICTION` | False / `NAN_OR_INF` |
| `H_STALE` | 42 | - | RAPID_OR_ABNORMAL | - | `MODEL_PREDICTION` | True / `` |
| `I_PROVIDER_SENSOR_FAULT` | 42 | - | - | - | `NO_VALID_PREDICTION` | False / `SIMULATED_MMWAVE_SENSOR_FAULT` |
| `J_READ_EXCEPTION` | 42 | - | - | - | `NO_VALID_PREDICTION` | False / `PROVIDER_READ_EXCEPTION` |
| `K_TIMEOUT` | 42 | - | - | - | `NO_VALID_PREDICTION` | False / `PROVIDER_READ_TIMEOUT` |
| `L_MISSING_MODEL` | - | - | - | - | `NO_VALID_PREDICTION` | False / `M-B9_FINALIST_MODEL_MANIFEST_MISSING` |
| `M_SHA_MISMATCH` | 42 | - | - | - | `NO_VALID_PREDICTION` | False / `M-B9_FINALIST_ARTIFACT_IDENTITY_MISMATCH` |
| `O_NOT_CONNECTED_PROVIDER` | 42 | - | - | - | `NO_VALID_PREDICTION` | False / `PROVIDER_CONNECT_FAILED` |
| `N_VALID_EXPLICIT_FINALIST` | 42 | NORMAL | RAPID_OR_ABNORMAL | False | `MODEL_PREDICTION` | True / `` |

The injected disagreement scenario used APNEA as metadata-only truth on a NORMAL-selected window; the node still used the actual model class, score mapping, and confidence.

## InferenceResult, risk, JSON, fallback, and LOCKED_TEST audits

- InferenceResult fields and finalist metadata were captured for `21` bounded node results; valid finalist rows use `score_source=MODEL_PREDICTION`, explicit model ID/SHA, class index, probabilities, and `fallback_used=false`.
- Fresh risk-engine recomputation against the exact sensor dictionaries entering risk matched node core fields: `True`.
- `SafeNestRiskOutput.to_json()` parsed with finite values and current schema fields for every row: `True`.
- The standalone validator independently reconstructs and compares InferenceResult, fallback, fault/stale/timeout, risk-input, risk-engine, and JSON audits against fresh bounded execution; timestamps and latency are the only excluded nondeterministic fields.
- Missing/wrong-identity finalist scenarios record the legacy fallback identity as invalid and never as finalist success; valid finalist rows have no fallback: `True`.
- LOCKED_TEST access attempts, labels, predictions, and performance reads: `0`; the immutable lock remains in force.

## Results

- M-B9 gate: `PASS_WITH_WARNINGS`
- Runtime identity exact: `True`
- Scenario records: `21`
- Risk recomputation exact: `True`
- JSON/schema finite audit: `True`
- LOCKED_TEST accesses: `0`

## Limitations

This is mock-provider/runtime compatibility evidence only. It does not claim production readiness, Raspberry Pi performance, MR60 real-sensor validation, sensor-to-alarm latency, or clinical apnea performance. `APNEA` remains a voluntary breath-hold proxy. No M-B10 candidate selection or LOCKED_TEST gate was started.

## M-B9 RESULT

- Shared default model: historical blocked manifest left unchanged; explicit phase manifests used.
- Explicit finalist strategy: all seeds 42/43/44, deterministic VALIDATION selection, no seed selection.
- Preprocessing before/after: legacy z-score-only path repaired locally to authoritative BPF_ZSCORE for explicit manifests.
- Runtime files: strict interpreter manifest loading, finalist mock provider, bounded integrated node.
- M-B8 wording: predecessor formal benchmark completed; `formal_m_b8_latency_measurement_rerun_during_m_b9=false`.
- Validator-truth closure: stored-vs-fresh scenarios and all six fresh audit gates are independently checked; real isolated corruption workspaces must fail closed.
- Real validator-failure corruption tests: 33 isolated temporary-workspace cases, including SHA/bytes/seed/quantization/preprocessing, prediction/truth/fallback, fault/stale/timeout, risk/JSON/LOCKED_TEST, checksum, absolute/traversal paths, and duplicate-binary rejection.
- Findings: `REQUIRED REFINEMENT` M-B9_RUNTIME_PREPROCESSING_MISMATCH is `RESOLVED_LOCALLY`; `NON-BLOCKING IMPROVEMENT` M-B9_MOCK_SCOPE_ONLY records mock-only scope; no `BLOCKER` remains.

YES — M-B10 candidate-selection setup may begin after independent review; LOCKED_TEST remains locked until the separately authorized M-B10 final-test gate
