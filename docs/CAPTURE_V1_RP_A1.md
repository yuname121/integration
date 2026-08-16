# SafeNest Capture v1 — RP-A1 Contract

**Status:** Implemented on `feature/rp-a1-capture-foundation`.
**Roadmap authorization:** `APPROVED_FOR_RP-A1_ONLY`.
**RP-A2:** Unauthorized until this contract is reviewed and merged into `junwoo/rpi-ai-runtime`.

RP-A1 defines the storage language for future Raspberry Pi Capture. It does not write real sensor sessions.

## Placement

| Existing path | Classification | Reason |
|---|---|---|
| `storage/` | `EXTEND` | Persistence package; Capture schema lives at `storage/capture_v1/` |
| `storage/sensor_logger.py` | `KEEP_SEPARATE` / `DEFER_TO_RP_A2` | Current hourly JSONL/NPZ logger is not Capture v1 |
| `hil/capture.py` | `KEEP_SEPARATE` | HIL API evidence reports, different schema |
| `tests/` | `EXTEND` | Synthetic Capture fixtures and unittest coverage |
| `.gitignore` | `EXTEND` | Ignore runtime `captures/` only |
| top-level `capture/` | not created | Roadmap mentioned a future `capture/` module; RP-A1 extends `storage/` instead of a parallel tree |

Capture schema version `safenest.capture.v1` (`SAFENEST_CAPTURE_V1`) is independent of `safenest.telemetry.v1`, SQLite, and AI input contracts.

## Session identity

`session_id` format:

```text
sncap-<YYYYMMDDTHHMMSSZ>-<12 lowercase hex>
```

Example: `sncap-20260816T093012Z-a1b2c3d4e5f6`

The UTC second is operator-readable. The entropy suffix is required so restarts, reboots, and concurrent experiments that share a UTC second cannot collide. A plain timestamp is not a session ID.

One future runtime session is one integrated Pi application run unless an operator starts an explicit experiment session. RP-A1 does not implement session lifecycle.

## Event identity

`capture_event_id` is a canonical lowercase UUID4 assigned by Pi Capture. It is the stable reference for replay, inference provenance, SQLite cross-links, and risk decisions.

It is not:

- CO₂ `source_measurement_event_id`
- Thermal `frame_sequence` or `payload.frame_id`
- a future mmWave phase-sample sequence
- a PIR transition identity

Those source IDs are stored separately when the device contract supplies them.

## Time semantics

These clocks are never collapsed into a generic `timestamp`:

| Field | Meaning |
|---|---|
| `source_measurement_event_id` / `source_measurement_monotonic_ms` | Device/source measurement identity and time |
| `device_uptime_ms` | ESP uptime |
| `pi_receive_wall_time` | Pi `time.time()` |
| `pi_receive_monotonic_time` | Pi `time.monotonic()` |

If ESP telemetry does not yet provide source measurement identity/time, both fields are `null` and `source_timing_unavailable_reason` is `SOURCE_TIMING_UNAVAILABLE`. Unknown values must not be stored as `0`. Numeric `0` is a real value (for example packet sequence 0 or uptime 0).

`boot_id` is currently `null` with `DEVICE_BOOT_ID_NOT_IN_TELEMETRY_V1`. Thermal UDP `device_id` may be `null` with `DEVICE_ID_UNAVAILABLE_THERMAL_UDP`.

## Validity and errors

| Field | Meaning |
|---|---|
| `parse_valid` | Packet/frame decoded and structurally parsed |
| `sensor_valid` | Source observation is a valid sensor reading |
| `stale` | Too old for current live inference; still valid Capture evidence |

An event may be `parse_valid=true`, `sensor_valid=true`, and `stale=true` at the same time. Stale observations are not discarded by this contract.

`error_code` is a bounded machine-readable token. `error_reason` is optional short human context. Stack traces are rejected.

## Sensor contracts

- **CO₂:** `co2_ppm` plus optional source identity/time. Humidity, temperature, and `CO2_slope` are forbidden. Rolling history is not part of RP-A1.
- **Thermal:** metadata and payload reference only. Reserved container name `npz_uint16_lossless` for a future lossless, deterministic, bounded, checksum-able, close-recoverable NPZ writer. RP-A1 does not implement that writer. Pixel arrays must not appear in event JSON. `payload_reference` is session-relative (`thermal/frames_NNNN.npz` or `#index`), never an absolute path.
- **PIR:** `pir_motion` with optional `is_transition`. No high-rate PIR model and no PIR AI.
- **mmWave:** common metadata plus optional current scalar respiration/heart fields. Phase/window/BPF payload remains `PENDING_MMWAVE_DEVICE_CONTRACT_VALIDATION`.

## Unknown fields

Unknown fields are **rejected**. Capture v1 fails closed. A later schema version must bump `capture_schema_version` before adding fields.

## Runtime path

Future writer output is reserved at repository-relative `captures/`, which is Git-ignored. Synthetic fixtures remain tracked under `tests/fixtures/capture_v1/`.

Planned layout (writer is RP-A2+):

```text
captures/<session_id>/manifest.json
captures/<session_id>/events_0001.jsonl
captures/<session_id>/thermal/frames_0001.npz
```

## Validator

```bash
python -m storage.capture_v1 validate tests/fixtures/capture_v1/session_valid.json
python -m unittest tests.test_capture_v1 -v
```

The validator checks structure only. It does not run AI, require hardware, read real payloads, modify data, or repair records.

## RP-A1 limitations

- No persistent Capture writer, disk queue, or rotation
- No CO₂ slope or runtime buffer
- No Thermal production payload writer
- No SQLite, ESP, replay, risk, dashboard, or B-model changes
- mmWave phase not defined
