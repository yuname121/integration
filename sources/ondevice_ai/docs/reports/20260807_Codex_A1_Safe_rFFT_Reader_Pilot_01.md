# Phase A1 Safe rFFT Reader Pilot

## 1. Executive Summary

Phase A1 gate: **`PASS_WITH_WARNINGS`**. A2 entry: **`READY_WITH_CONDITIONS`**. The deterministic 12-recording, 6-subject pilot safely decoded 12 recordings and failed 0. The decoded payload is a zlib-compressed protocol-5 pickle containing `[rFFTs, rBins]`; no arbitrary object execution occurred. A strict `pickletools` opcode/global allowlist decoded only primitive NumPy buffer structures.

The shared in-memory A1 validator completed before gate derivation: **`True`** (0 errors).

This proves structural radar decoding only. It does not prove respiration extraction.

## 2. A0 Input Baseline

- A0 gate: `PASS_WITH_WARNINGS`; A1 entry: `READY_WITH_CONDITIONS`.
- A0 authoritative inventory: 110 participants, 440 recordings, two annotation/file-role schema profiles.
- Pre-A1 measured archive SHA-256: `f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0`.
- A0 serialization claim is preserved but contradicted by A1 direct evidence; see `A1-EXC-0001`.

## 3. Pilot Selection

The selection is derived deterministically from the A0 recording index plus a bounded read-only scan of all 440 linked `radar_timestamps.csv` ZIP members. No rFFT member is opened during this scan. Measured strata: 400 rows: 2 recordings, 500 rows: 348 recordings, 600 rows: 90 recordings. Complete low/high subject anchors, A0 exceptions/candidates, and one representative for every measured timestamp-count stratum are selected before deterministic fill.

| Recording | Posture | Condition | A0 profile | Annotation | Selection reason |
|---|---|---|---|---:|---|
| `dataset-10_5281_zenodo_18599983-p001-lying-post_exercise` | `Lying` | `Post-exercise` | `SCHEMA_PROFILE_002` | `False` | `LOW_ID_SUBJECT_FULL_FACTORIAL_ANCHOR` |
| `dataset-10_5281_zenodo_18599983-p001-lying-rest` | `Lying` | `Rest` | `SCHEMA_PROFILE_001` | `True` | `LOW_ID_SUBJECT_FULL_FACTORIAL_ANCHOR` |
| `dataset-10_5281_zenodo_18599983-p001-sitting-post_exercise` | `Sitting` | `Post-exercise` | `SCHEMA_PROFILE_002` | `False` | `LOW_ID_SUBJECT_FULL_FACTORIAL_ANCHOR` |
| `dataset-10_5281_zenodo_18599983-p001-sitting-rest` | `Sitting` | `Rest` | `SCHEMA_PROFILE_001` | `True` | `LOW_ID_SUBJECT_FULL_FACTORIAL_ANCHOR` |
| `dataset-10_5281_zenodo_18599983-p002-lying-post_exercise` | `Lying` | `Post-exercise` | `SCHEMA_PROFILE_002` | `False` | `A0_REPORT_RECOMMENDED_PILOT` |
| `dataset-10_5281_zenodo_18599983-p004-lying-post_exercise` | `Lying` | `Post-exercise` | `SCHEMA_PROFILE_002` | `False` | `ZIP_TIMESTAMP_COUNT_STRATUM_REPRESENTATIVE` |
| `dataset-10_5281_zenodo_18599983-p007-sitting-post_exercise` | `Sitting` | `Post-exercise` | `SCHEMA_PROFILE_002` | `False` | `A0_RECORDED_TIMESTAMP_LENGTH_EXCEPTION` |
| `dataset-10_5281_zenodo_18599983-p075-sitting-rest` | `Sitting` | `Rest` | `SCHEMA_PROFILE_001` | `True` | `A0_RECORDED_TIMESTAMP_LENGTH_EXCEPTION` |
| `dataset-10_5281_zenodo_18599983-p110-lying-post_exercise` | `Lying` | `Post-exercise` | `SCHEMA_PROFILE_002` | `False` | `HIGH_ID_SUBJECT_FULL_FACTORIAL_ANCHOR` |
| `dataset-10_5281_zenodo_18599983-p110-lying-rest` | `Lying` | `Rest` | `SCHEMA_PROFILE_001` | `True` | `HIGH_ID_SUBJECT_FULL_FACTORIAL_ANCHOR` |
| `dataset-10_5281_zenodo_18599983-p110-sitting-post_exercise` | `Sitting` | `Post-exercise` | `SCHEMA_PROFILE_002` | `False` | `HIGH_ID_SUBJECT_FULL_FACTORIAL_ANCHOR` |
| `dataset-10_5281_zenodo_18599983-p110-sitting-rest` | `Sitting` | `Rest` | `SCHEMA_PROFILE_001` | `True` | `HIGH_ID_SUBJECT_FULL_FACTORIAL_ANCHOR` |

## 4. Safe Serialization Investigation

- Pipeline: bounded ZIP member stream → validated zlib stream → bounded decompressed bytes → `pickletools` opcode stream → symbolic allowlisted NumPy dtype/from-buffer VM.
- Detected root: `PYTHON_PICKLE_PROTOCOL_5_NUMPY_ARRAY_PAIR` representing `[rFFTs, rBins]`.
- Allowed symbolic globals: `numpy.dtype`, `numpy.core.numeric._frombuffer` (and NumPy 2 spelling `numpy._core.numeric._frombuffer`). They are never imported or called by the VM.
- Any unsupported opcode, global, dtype, shape, order, root structure, or trailing pickle byte fails closed.
- Unsafe normal object deserialization required: **NO**.
- Object-execution-capable source container detected: **YES**.

[Official dataset `helper_fns.py`](https://zenodo.org/api/records/18599983/files/helper_fns.py/content) confirms the producer expected `rFFTs, rBins` after zlib/pickle loading.

## 5. rFFT Container/Compression Structure

- ZIP member name: `radar_rFFTs.zlib`.
- Inner compression: zlib, header(s): `78da`.
- Decompression checks: valid header, bounded compressed and decompressed sizes, EOF required, output cap, no trailing/unused bytes, no concatenated stream.
- Observed decompressed sizes and compression ratios are retained per recording in `pilot_decode_results.jsonl`.

## 6. Decoded Tensor Structure

- Shape(s): `[400, 8, 64]`, `[500, 8, 64]`, `[600, 8, 64]`.
- Dtype(s): `complex128`; little-endian complex values with interleaved float64 real/imag storage.
- Frame axis: `0` (official example code and exact timestamp dimension consistency).
- Virtual-antenna axis: `1` (official notebook documentation plus 2 TX × 4 RX config).
- Range-bin axis: `2` (official notebook documentation, 64-element stored `rBins`, and 64 ADC samples).
- Range vector: `float64[64]`, from `0.0` m to `19.986163866666665` m.
- Virtual-channel ordering: **NOT_VERIFIABLE** from current config/documentation.

[The official example notebook](https://zenodo.org/api/records/18599983/files/ExampleCode.ipynb/content) explicitly documents `(frames, virtual antennas, range bins)` and names axes 0/1/2.

## 7. Chirp Configuration Interpretation

- Unique configuration hashes: `1` (A1 SHA-256); A0-compatible hash(es): `bd48b829076ed279`.
- Start frequency: `60250000000.0` Hz.
- Sampled bandwidth/end/center: `480000000.0` / `60730000000.0` / `60490000000.0` Hz.
- ADC samples: `64`; loop count: `32`; explicit chirps per frame: `NOT_VERIFIABLE`.
- TX/RX/derived virtual count: `2` / `4` / `8`.
- Period/frame rate: `0.1` s / `10.0` Hz.
- Configured `R_BIN`: `0.31228381041666664` m; stored `rBins` median spacing: `0.31724069629629614` m. These differ and are not silently reconciled.
- Each original config key/value, interpretation, and evidence is preserved per pilot result.

## 8. Timestamp and Frame Alignment

- Exact alignments: `12`; mismatches: `0`.
- Decoded frame-count strata: `400`, `500`, `600`; each measured timestamp stratum has at least one safely decoded tensor in the pilot.
- Timestamp median Δt value(s): `0.1` seconds; empirical frame rate value(s): `10.0` Hz.
- Duplicate/backward/large-gap totals: `0` / `0` / `0`.
- The two 400-frame A0 exceptions decode as 400-frame tensors with exactly 400 timestamps. No truncation occurs.

## 9. Decoder Profiles

| Profile | Recordings | Shape pattern | Supported A0 profiles |
|---|---:|---|---|
| `RFFT_DECODER_PROFILE_001` | `12` | `[None, 8, 64]` | SCHEMA_PROFILE_001, SCHEMA_PROFILE_002 |

Both A0 file-role/annotation profiles map to the same radar decoder profile. Frame count is variable; tensor representation and non-frame axes are uniform in the pilot.

## 10. Exceptions and Failures

| ID | Severity | Category | Direct observation |
|---|---|---|---|
| `A1-EXC-0001` | `WARNING` | `A0_CONTRADICTION` | Direct decompressed-payload inspection found protocol-5 pickle; A0 recorded radar_serialization=['ZLIB_RAW_COMPRESSION'] and stated no pickle was needed. |
| `A1-EXC-0002` | `WARNING` | `UNSAFE_SERIALIZATION` | The source container is object-execution-capable, but all pilot records were decoded by a pickletools-tokenized symbolic VM allowing only NumPy dtype and _frombuffer structures. |
| `A1-EXC-0003` | `WARNING` | `CHIRP_CONFIG` | Stored rBins median spacing is 0.31724069629629614 m while chirp R_BIN is 0.31228381041666664 m. |
| `A1-EXC-0004` | `INFO` | `SCHEMA_VARIANT` | The pilot directly decoded 400-frame tensors for the A0-recorded short-duration cases; each has exactly 400 timestamps. |
| `A1-EXC-0005` | `WARNING` | `A0_CONTRADICTION` | The committed A0 human report describes P002/Lying/Post-exercise as 600 frames, but A1 directly decoded 500 frames and parsed 500 timestamps. |
| `A1-EXC-0006` | `INFO` | `AXIS_SEMANTICS` | The antenna axis and count are verified, but TX/RX-to-virtual-channel ordering is absent from config and official example documentation. |

Decode failures: `0`. Blockers: `0`. Errors: `0`. Warnings: `4`.

## 11. A0 Contradictions

A0 labeled the inner representation as raw zlib-compressed numeric data and stated a pickle reader was unnecessary. Direct A1 decompression found a protocol-5 pickle beginning with `80 05` and NumPy `_frombuffer` opcodes. The committed A0 human report also describes `P002/Lying/Post-exercise` as a 600-frame candidate, while A1 measures 500 frames and 500 timestamps. A0 outputs are unchanged; both discrepancies are preserved in the A1 exception registry.

## 12. A1 Gate Decision

**`PASS_WITH_WARNINGS`**. This state is derived only after the shared in-memory validator returns `validation_success=True`. The format, tensor contract, axes, frame counts, timing, and chirp linkage are measured; all pilot records decode and align. Warnings remain because the source serialization is object-execution-capable and because configured/stored range spacing differs.

## 13. A2 Entry Decision

**`READY_WITH_CONDITIONS`**. The range and antenna axes are verified, so A2 can begin. Conditions: keep the restricted reader fail-closed, use stored `rBins` for decoded physical coordinates, preserve the config-spacing discrepancy, and do not claim physical virtual-channel ordering without new evidence.

## 14. Remaining Unknowns for A2

- TX/RX-to-virtual-antenna channel ordering.
- Why the config `R_BIN` equals a different spacing convention than the stored inclusive 64-element range vector.
- These do not make the range/antenna dimensions unresolved; they constrain physical interpretation.

## 15. Files Created/Modified

- `scripts/mmwave_rfft_reader.py`: bounded zlib and non-executing restricted NumPy-pickle reader.
- `scripts/run_mmwave_rfft_pilot.py`: deterministic selection, real pilot, profiles, exceptions, summary, and report generation.
- `scripts/validate_mmwave_rfft_pilot.py`: cross-manifest A1 validator.
- `tests/test_mmwave_rfft_reader.py`: archive-independent synthetic tests.
- `datasets/mmwave/manifests/a1_rfft_pilot/`: A1 machine-readable artifacts.
- `docs/reports/20260807_Codex_A1_Safe_rFFT_Reader_Pilot_01.md`: this report.

## 16. Commands and Tests

```text
python3 -m unittest tests/test_mmwave_rfft_reader.py -v
python3 scripts/run_mmwave_rfft_pilot.py
python3 scripts/validate_mmwave_rfft_pilot.py
git diff --check
shasum -a 256 datasets/raw_archives/external_datasets/db_records.zip
```

Final measured command outcomes are recorded in the Phase A1 handoff after execution.

## 17. Explicit Non-Scope Confirmation

Target range-bin selection, respiration scoring, respiration phase extraction, unwrap, antenna performance selection/aggregation, respiration spectrum/SNR work, detrending, filtering, normalization, resampling, windowing, labels, subject splits, full conversion, training, evaluation, TFLite, and quantization were **NOT PERFORMED**.
