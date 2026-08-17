# RP-X0 O2.6 — Thermal real-field FLOAT↔INT8 equivalence

**Document date:** 2026-08-18
**Document ID:** `RP-X0-O2-6-THERMAL-FIELD-FLOAT-INT8-EQUIV-01`
**Technical status:** `COMPLETE`
**Remote delivery:** `DEFERRED_GITHUB_UNAVAILABLE` (local commit only)
**Gate:** `PASS_WITH_LIMITATIONS`
**Classification:** `INT8_QUANTIZATION_REVIEW_REQUIRED`
**O3 adapter implementation:** `NO`
**Production Thermal activation:** `NO`

Quantization-equivalence audit only. No accuracy, T-C, fall-detection, or
production claim. No independent ground truth.

Stacked on O2.5 `1929bc33cea1f7004e3d9d19900bf4023e0efb97`.

---

## 1. Sampling

Read-only snapshot `data/thermal`. 1979 NPZ listed. **15** unreadable
`EOFError` files classified `FIELD_CAPTURE_ARTIFACT` (not a T-B5 failure).
**23788** readable frames.

Algorithm (frozen before inference): chronological sort → 24 equal time bins
over `[t_min, t_max]` → 10 evenly spaced frames per occupied bin. Empty bins
are not backfilled.

| Target | Actual |
|---|---:|
| 24 × 10 = 240 | **154** |

Eight time bins were empty (capture gaps, including the corrupted-NPZ window).
Bin 2 had only 4 valid frames, all used. This is the specified algorithm, not
output-dependent compensation.

---

## 2. Artifacts and preprocessing

| Item | Value |
|---|---|
| FLOAT | `SMALL_CNN_BASELINE_V1_P1_float32.tflite` |
| FLOAT SHA | `fbe891520f07e0534b1a7074dc819d8ed44bca58b27e35c78916c3ddae73a779` |
| INT8 | `SMALL_CNN_BASELINE_V1_P1_full_int8.tflite` |
| INT8 SHA | `fa9730c29535477a3994c11e664474a0ca0116afaaa172889f47446ab2ac46be` |
| Same lineage | YES |
| Celsius | `raw_uint16 / 10.0 - 273.15` |
| P1 | mean `22.769290618485442`, std `2.8684523405441222` |
| INT8 q | scale `0.31791284680366516`, zp `-125` (read back from artifact) |
| Orientation | native `(62, 80)` |

FLOAT binary was SHA-verified from the T-B4 `TFLITE_FP32` counterpart. It is
not committed to integration Git.

---

## 3. Canonical AI policy

Local `sheepmeat/test` T-B4:

```text
CANONICAL_AI_EQUIVALENCE_POLICY = FOUND
THERMAL_T_B4_FLOAT_TFLITE_FP32_FULL_INT8_EQUIVALENCE_001
EQUIVALENCE_MEASURED_NO_PREEXISTING_ABSOLUTE_ACCEPTANCE_THRESHOLD
```

The RP-X0 95% fallback is **not** an AI acceptance threshold. It is cited
only as a narrow integration overlay after the field measurements. T-B4
VALIDATION reference: FLOAT↔INT8 argmax `0.997125`. T-B4 SDT REAL_EVAL
reference: `0.8565`.

---

## 4. FLOAT↔INT8 metrics (154 frames)

| Metric | Value |
|---|---|
| top-1 agreement | **139 / 154 = 90.26%** |
| disagreement | 15 |
| ranking agreement | 90.26% |
| output MAE median | 0.00130 |
| output MAE p90 | 0.0770 |
| output MAE p95 | 0.6628 |
| output MAE max | 0.6654 |

Median MAE matches INT8 softmax grain (`1/256`). The tail is class-flip
scale, not grain.

FLOAT→INT8 transitions (rows = FLOAT):

|  | NOT_HUMAN | HUMAN_NORMAL | HUMAN_FALL |
|---|---:|---:|---:|
| NOT_HUMAN | 0 | 1 | 0 |
| HUMAN_NORMAL | 1 | 17 | 3 |
| HUMAN_FALL | 2 | 8 | 122 |

Dominant disagreement: FLOAT `HUMAN_FALL` → INT8 `HUMAN_NORMAL` (8).

---

## 5. Saturation

|  | Value |
|---|---|
| low-side (`unclipped < -128`) median | 0.434 |
| low-side p95 | 0.831 |
| high-side | 0.000 |

| q=−128 bin | n | agree | disagree | rate |
|---|---:|---:|---:|---:|
| 0–10% | 2 | 2 | 0 | 1.000 |
| 10–25% | 34 | 33 | 1 | 0.971 |
| 25–50% | 48 | 46 | 2 | 0.958 |
| 50–75% | 51 | 43 | 8 | 0.843 |
| 75–100% | 19 | 15 | 4 | 0.789 |

Bins with n≥8 show monotone falling agreement as low-side saturation rises
(97.1% → 95.8% → 84.3% → 78.9%).

**Saturation relationship:** `STRONG_ASSOCIATION_OBSERVED`

This is association, not proven causation. Five-frame O2.5 could not see it;
154 frames can.

---

## 6. Predicted-class distribution

Descriptive only. No labels.

| Class | FLOAT | INT8 |
|---|---:|---:|
| NOT_HUMAN | 1 | 3 |
| HUMAN_NORMAL | 21 | 26 |
| HUMAN_FALL | 132 | 125 |

HUMAN_FALL is 85.7% FLOAT / 81.2% INT8. Below the 90% collapse flag, so
`REAL_DEVICE_OUTPUT_COLLAPSE_WARNING` is **not** raised. Concentration remains
suspicious for unlabeled MI48 field data and is a device-domain question,
secondary to the quantization association.

`HUMAN_FALL` remains a LYING posture proxy, not a temporal fall event.

---

## 7. Decision

```text
INT8_QUANTIZATION_REVIEW_REQUIRED
O3_ADAPTER_IMPLEMENTATION = NO
THERMAL_PRODUCTION_ACTIVATION = NO
```

Field top-1 agreement **90.26%** is below T-B4 VALIDATION (~99.7%) and below
the unused 95% integration overlay. Disagreements concentrate at high
low-side saturation. Next owner: `sheepmeat/test` Thermal quantization /
calibration review. Do not modify T-B5 in integration. Do not start O3.

Device-domain performance remains unvalidated (no ground truth). That does
not override the quantization-association finding.

---

## 8. Hygiene

- snapshot modified: NO
- Pi `.venv` executed: NO
- models modified: NO
- production manifest: NO
- raw frames committed: NO
- RP-A2: NO

---

## 9. Future PR (deferred)

**Title:** RP-X0 O2.6: audit Thermal real-field FLOAT vs INT8 equivalence
**Base:** O2.5 `audit/rp-x0-o2-5-thermal-quant-compat` @ `1929bc3`
**Do not invent a PR number while GitHub delivery is deferred.**
