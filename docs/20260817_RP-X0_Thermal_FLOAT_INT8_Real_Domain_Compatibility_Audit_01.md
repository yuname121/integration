# RP-X0 O2.5 — Thermal FLOAT vs INT8 real-domain quantization compatibility

**Document date:** 2026-08-17
**Document ID:** `RP-X0-O2-5-THERMAL-FLOAT-INT8-COMPAT-01`
**Status:** `PASS_WITH_LIMITATIONS`
**Classification:** `DEVICE_DOMAIN_GAP_LIKELY`
**O3 adapter implementation:** `NO`
**Production Thermal activation:** `NO`

This audit asks whether O2's large INT8 low-side saturation is a
deployment-blocking T-B5 quantizer failure, or whether FLOAT and INT8 remain
consistent on the same real MI48 frames. It is not accuracy evaluation, T-C,
production activation, adapter implementation, or retraining.

Stacked on O2 commit `925330c1c54eb0f5762ae56b6ff6f6a81897aad5`
(PR https://github.com/yuname121/integration/pull/14). O2 was not merged at
the time of this work.

---

## 1. Artifact lineage

AI authority: `sheepmeat/test` T-B4 / T-B5.

| Artifact | Identity | SHA-256 | Size |
|---|---|---|---:|
| FLOAT counterpart | `TFLITE_FP32` / `SMALL_CNN_BASELINE_V1_P1_float32.tflite` | `fbe891520f07e0534b1a7074dc819d8ed44bca58b27e35c78916c3ddae73a779` | 1252048 |
| Locked INT8 | `FULL_INT8` / `SMALL_CNN_BASELINE_V1_P1_full_int8.tflite` | `fa9730c29535477a3994c11e664474a0ca0116afaaa172889f47446ab2ac46be` | 318280 |

Proven from T-B4 conversion chain (no retraining):

```text
FLOAT_KERAS  P1_TRAIN_FITTED_GLOBAL_ZSCORE.weights.h5
  SHA 7aba32fe8d0e241546429bdc3e8cd059b10d4d8f548e9e12c4085abeba308a75
        ↓  TFLite, optimizations=[], quantization_mode=NONE
TFLITE_FP32  true unquantized float32 I/O, 0 quantized tensors
        ↓  full-integer INT8, TRAIN-only 512-sample calibration
FULL_INT8    T-B5 selected candidate
```

| Question | Answer |
|---|---|
| same architecture | YES — `SMALL_CNN_BASELINE_V1`, `[1,62,80,1]` → `[1,3]` |
| same weights before quantization | YES — T-B4 `retraining=NO`, conversion only |
| same P1 contract | YES — mean `22.769290618485442`, std `2.8684523405441222` |

The FLOAT file used here was read from
`origin/policy/git-track-locked-b-binaries` in `sheepmeat/test` and SHA-verified.
It is **not** committed to integration Git.

Former `TFLITE_DYNAMIC_RANGE` (`297de231…`, 317344 B) is **not** the FLOAT
counterpart.

---

## 2. Equivalence policy

T-B4 protocol:

```text
EQUIVALENCE_MEASURED_NO_PREEXISTING_ABSOLUTE_ACCEPTANCE_THRESHOLD
```

No pass threshold was invented after seeing O2.5 numbers. Diagnostics are
descriptive. T-B4 VALIDATION already measured FLOAT↔INT8 argmax agreement
`0.997125` (23/8000 disagreements) with rare input clipping. T-B4
`REAL_EVAL_DEVELOPMENT` (SDT, not MI48) already showed a separate domain gap
(FLOAT Macro F1 `0.594` vs VALIDATION `0.995`; FLOAT↔INT8 argmax `0.8565`).

---

## 3. Inputs

Same five O2 frames, native `(62, 80)`, hashes verified. No new frame picking.
No clip, min-max, histogram, orientation search, or invalid-pixel policy.

```text
C = raw_uint16 / 10.0 - 273.15
z = (C - 22.769290618485442) / 2.8684523405441222
FLOAT  ← z as float32
INT8   ← clip(rint(z / 0.31791284680366516 - 125), -128, 127)
```

Snapshot used read-only. 15 O2-noted unreadable NPZ files were not needed.

---

## 4. INT8 physical mapping (precise)

| Quantity | Celsius | Meaning |
|---|---:|---|
| q = −125 | 22.7693 | P1 mean; value *represented* by the zero-point |
| q = −128 | 20.0335 | value *represented by* dequantizing q=−128 |
| rint clip onto −128 | 19.5776 | largest C for which `rint(z/scale+zp) ≤ −129` then clips |

**q=−128 represented temperature is not the clipping boundary.** Pixels colder
than about 19.58 °C saturate; q=−128 also covers a band around 20.03 °C that
does not clip.

---

## 5. Results (five frames)

| Role | sat. low | C median | FLOAT class | INT8 class | top-1 | rank | MAE |
|---|---:|---:|---|---|---|---|---:|
| early | 0.130 | 23.85 | HUMAN_FALL | HUMAN_FALL | YES | YES | 0.0013 |
| middle | 0.697 | 16.15 | HUMAN_FALL | HUMAN_NORMAL | NO | NO | 0.6589 |
| late | 0.784 | 15.05 | HUMAN_NORMAL | HUMAN_NORMAL | YES | YES | 0.0013 |
| low-looking | 0.240 | 23.15 | HUMAN_FALL | HUMAN_FALL | YES | YES | 0.0013 |
| high-looking | 0.292 | 22.25 | HUMAN_FALL | HUMAN_FALL | YES | YES | 0.0013 |

High-side saturation: 0 on all frames.

Agreeing frames differ only by INT8 softmax grain (`output scale=1/256`).
FLOAT outputs are already near-one-hot (`margin ≈ 1.0`). INT8 is also
near-one-hot (`margin ≈ 0.98–1.00`).

The single disagreement is the middle frame. The late frame has **higher**
low-side saturation (0.784 vs 0.697) and still agrees. Five samples cannot
support “saturation causes the flip.”

**Saturation relationship:** `INSUFFICIENT_EVIDENCE`

Cold outliers remain `SENSOR_OUTLIER_UNRESOLVED`. They were left unchanged.
No clamp policy.

---

## 6. Decision

```text
DEVICE_DOMAIN_GAP_LIKELY
```

FLOAT and INT8 are mostly consistent (4/5 top-1 and ranking). The large INT8
saturation is therefore **not** shown to be a standalone T-B5 quantizer
blocker. Both models already emit extreme, high-margin posture-proxy outputs
on unlabeled MI48 field frames. That points to SDT-trained T-B5 vs real MI48
field domain, which T-B4 already flagged on SDT REAL_EVAL_DEVELOPMENT.

Not `INT8_COMPATIBLE_FOR_ADAPTER_IMPLEMENTATION`: one material class flip
remains, and both paths look domain-extreme rather than adapter-ready.

Not `INT8_QUANTIZATION_REVIEW_REQUIRED` as the primary: FLOAT is already
extreme on the same tensors; the late high-saturation frame agrees.

Next action belongs to Thermal AI / device-domain work in `sheepmeat/test`,
not to integration O3 adapter implementation.

```text
O3_ADAPTER_IMPLEMENTATION = NO
PRODUCTION_THERMAL_ACTIVATION = NO
```

---

## 7. What was not validated

- independent ground truth / accuracy / F1
- T-C / Thermal-44 device-domain performance
- temporal fall detection
- production activation
- live Pi
- RP-A2
- CO2 / mmWave / soak / conversion re-investigation

---

## 8. Gate

```text
RP_X0_O2_5_THERMAL_QUANT_COMPAT = PASS_WITH_LIMITATIONS
```

STOP after this PR. Do not start O3.
