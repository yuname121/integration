# RP-X0 O2 — Thermal real-snapshot replay through P1 and T-B5

**Document date:** 2026-08-17
**Document ID:** `RP-X0-O2-THERMAL-REAL-SNAPSHOT-TB5-01`
**Status:** `PASS_WITH_LIMITATIONS`
**Runtime recommendation:** `THERMAL_RUNTIME_ADAPTER_ELIGIBLE_WITH_LIMITATIONS`

This is an offline pipeline-compatibility test. It is not Thermal accuracy
validation, T-C device-domain validation, temporal fall detection, production
activation, or a live Pi test.

---

## 1. Scope

Real saved `uint16` → verified MI48 `0.1 K` conversion → Celsius → frozen
`P1_TRAIN_FITTED_GLOBAL_ZSCORE` → strict INT8 quantization → locked T-B5.

Native stored orientation only. No rotate/flip search. No P1 refit. No
invalid-pixel invention. T-B5 was invoked offline only.

---

## 2. Evidence source

| Item | Value |
|---|---|
| Snapshot | `/Users/junwoo/Library/Mobile Documents/com~apple~CloudDocs/대학/2026/safenest-pi-integration-snapshot` |
| Role | `READ_ONLY_FIELD_EVIDENCE` |
| Listed NPZ | 1979 |
| Readable NPZ | 1964 |
| Unreadable NPZ | 15 `EOFError` files around `20260817_084856`–`084924` UTC |
| Snapshot modified | **NO** |
| Snapshot `.venv` executed | **NO** |
| Real NPZ committed | **NO** |

The 15 unreadable files were skipped for metadata scan and were not rewritten.
They are a snapshot-read limitation, not a conversion-formula failure.

---

## 3. Selected replay set

Deterministic selection from readable `data/thermal/20260817_*.npz`:

| Role | File | SHA-256 | Frame |
|---|---|---|---|
| early | `20260817_065524_939278_0000007852-0000007864.npz` | `09dc0e3b70beb357951dc7ea1fc4057babaada9c46e9ae673bdc59769af5b2d6` | 0 |
| middle | `20260817_082122_819638_0000006423-0000006434.npz` | `dde0240f086f2a59da24b73237fde4fcfd5369935cd1b1453aa48f8e8fb2e009` | 0 |
| late | `20260817_093125_194406_0000030438-0000030438.npz` | `c378a964a315c55ee1b8d3093f8fa52f4387ac5428e6bb2311cb4cd86f8b02cb` | 0 |
| low-looking | `20260817_081844_396547_0000005431-0000005442.npz` | `77ef9787155b434c7c8836ce6fba80de83843d75869803b1722e8ca6ff71c46e` | 2 |
| high-looking | `20260817_084720_900175_0000015484-0000015496.npz` | `0dd2e560172b4234e6bccafc5aa079c0ac906068e09faa2d664c843057169300` | 5 |

NPZ contract on every selected frame: `uint16`, shape `(62, 80)`, timestamps
present, original array unchanged. No extra reshape/transpose.

---

## 4. Physical conversion

```text
physical_C = raw_uint16 / 10.0 - 273.15
```

No clamp. No host `/100`. No SDT `(raw - 27315) / 100`.

Selected-frame Celsius:

| Role | min | p5 | median | p95 | max | finite |
|---|---:|---:|---:|---:|---:|---:|
| early | 3.75 | 17.05 | 23.85 | 31.05 | 39.45 | 4960 |
| middle | −61.85 | 2.35 | 16.15 | 29.15 | 41.75 | 4960 |
| late | −55.85 | 6.55 | 15.05 | 26.55 | 43.65 | 4960 |
| low-looking | −64.65 | 11.75 | 23.15 | 36.05 | 47.55 | 4960 |
| high-looking | −64.55 | 6.64 | 22.25 | 40.05 | 70.35 | 4960 |

NaN/Inf: none.

---

## 5. Extreme pixels

Vendor MI48xx replaces identified bad pixels in-chip. SPI temperature words
are documented as `0.1 K`, not as a sentinel map. Selected frames contain
**no** `raw=0` and **no** `raw=65535`.

Pixels below −40 °C are rare (0–11 of 4960). They appear in small adjacent
clusters that **do not** stay fixed across other frames in the same NPZ.

Metadata scan of readable files: 7446 / 23788 frames have `minimum_raw < 2300`
(about −43 °C). That counts frames that contain at least one cold sample, not
the pixel fraction.

**Classification:** `SENSOR_OUTLIER_UNRESOLVED`

They were left unchanged. P1 remains finite. They saturate at INT8 `-128`
like other sub-~20 °C pixels. They are a recorded limitation, not an O2 stop.

---

## 6. Frozen P1

```text
z = (C - 22.769290618485442) / 2.8684523405441222
```

`p1_lock.json` epsilon `1e-06` is unused because `std ≫ epsilon`. No
per-frame min-max, no clip, no histogram match. Float64 z-scores. All finite.

---

## 7. INT8 quantization

Read from the locked artifact and matched to the known contract:

| Item | Value |
|---|---|
| scale | `0.31791284680366516` |
| zero_point | `-125` |
| rule | `q = clip(rint(z / scale + zp), -128, 127)` |

Unclipped INT8 dynamic range mapped back to Celsius:

| INT8 | Celsius |
|---|---|
| −128 | 20.03 °C |
| −125 (zp) | 22.77 °C |
| +127 | 252.57 °C |

Field indoor/cool pixels therefore pile up on `-128`. High-side saturation
was **0** on all selected frames.

| Role | int8 min/max | unique | low sat. | high sat. |
|---|---|---:|---:|---:|
| early | −128 / −107 | 22 | 0.130 | 0 |
| middle | −128 / −104 | 25 | 0.697 | 0 |
| late | −128 / −102 | 25 | 0.784 | 0 |
| low-looking | −128 / −98 | 31 | 0.240 | 0 |
| high-looking | −128 / −73 | 44 | 0.292 | 0 |

P1 was **not** modified to reduce saturation.

---

## 8. Locked T-B5 invocation

| Item | Value |
|---|---|
| Artifact | `SMALL_CNN_BASELINE_V1_P1_full_int8.tflite` |
| SHA-256 | `fa9730c29535477a3994c11e664474a0ca0116afaaa172889f47446ab2ac46be` |
| Load | PASS |
| Input | `[1, 62, 80, 1] int8` |
| Output | `[1, 3] int8` |
| Class map | `NOT_HUMAN`, `HUMAN_NORMAL`, `HUMAN_FALL` |

Representative outputs (interpreted class is **not** a performance claim):

| Role | raw int8 | dequantized | class | latency |
|---|---|---|---|---|
| early | `[-128, -128, 127]` | `[0, 0, 0.996]` | `HUMAN_FALL` | ~0.8 ms |
| middle | `[-128, 125, -125]` | `[0, 0.988, 0.012]` | `HUMAN_NORMAL` | ~0.1 ms |
| late | `[-128, 127, -128]` | `[0, 0.996, 0]` | `HUMAN_NORMAL` | ~0.1 ms |
| low-looking | `[-128, -128, 127]` | `[0, 0, 0.996]` | `HUMAN_FALL` | ~0.1 ms |
| high-looking | `[-128, -128, 127]` | `[0, 0, 0.996]` | `HUMAN_FALL` | ~0.1 ms |

`HUMAN_FALL` is a LYING-derived posture proxy. It is **not** verified
temporal fall detection. There is no independent ground truth for these
frames. No accuracy / F1 / detection-success number is reported.

Zero-point synthetic invoke still matches the locked contract
`[-29, -70, -29]`.

---

## 9. Orientation

| Item | Status |
|---|---|
| Native `62×80` raster | VERIFIED |
| Field-camera vs SDT training orientation | UNVERIFIED |
| Transform applied | none |

---

## 10. What was not validated

- independent ground truth
- accuracy / precision / recall / F1
- T-C / Thermal-44 device-domain performance
- temporal fall events
- production activation
- live Pi integration
- RP-A2

The production thermal interpreter still applies per-frame min-max onto
historical `v0.1.0`. That path must **not** be reused for T-B5. O3 may
implement the verified O1+O2 path; this task does not.

---

## 11. Runtime eligibility

```text
THERMAL_RUNTIME_ADAPTER_ELIGIBLE_WITH_LIMITATIONS
```

The verified real-input preprocessing path can now be implemented in
`safenest-integration`. Limitations that remain in that adapter:

1. INT8 floor at ~20.03 °C saturates a large fraction of field pixels
2. unresolved rare cold clusters
3. unverified field-vs-training orientation
4. 15 unreadable snapshot NPZ files
5. no production selection / no T-C claim

Not `BLOCKED_NUMERIC_SATURATION`: invoke succeeded, high-side saturation is
zero, and selected frames still retain 22–44 distinct INT8 values rather than
a constant tensor.

---

## 12. Gate

```text
RP_X0_O2_THERMAL_REAL_REPLAY = PASS_WITH_LIMITATIONS
```

STOP after O2. Do not start O3 in this task.
