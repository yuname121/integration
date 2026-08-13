# ondevice_ai Intermediate Release Collision Summary

## 2026-08-13 incremental synchronization

- Standalone source SHA: `77b1695ac66fd595bd037e4574d1626b8917654c`
- Previous synchronized source SHA: `9a66a3b21baef9a6a51cb1a66942284c63d0b8a4`
- Team base SHA: `f3bd342eabcad27dc2c3ecdc16f035b8b13cb153`
- Destination: `ondevice_ai/`
- Source changes reviewed: 489 files, comprising 484 additions and 5 modifications.
- Track distribution: mmWave 268, CO₂ 141, Thermal 61, and shared documentation 19.
- Collision result: all five modified paths were byte-identical to the previously synchronized source before replacement; no team-diverged collision was overwritten.
- Device and shared-contract impact: none. No file under team `devices/`, `shared/contracts/`, root `.github/`, root `docs/`, or root `archive/` is changed.
- `sensors/mmwave/finalist_mock_provider.py` is included only as a bounded offline B9 test provider. It is not a real MR60 driver and does not replace `devices/mmwave/`.

The exact five collision decisions and the inclusion/exclusion plan are recorded in `20260813_intermediate_release_collision_matrix.json` and `20260813_intermediate_release_apply_plan.json`. The material below is retained as the historical summary of the initial synchronization.

## Initial synchronization retained for history

- Standalone source SHA: `9a66a3b21baef9a6a51cb1a66942284c63d0b8a4`
- Team base SHA: `6baf38d8df936b694a1ff2e9b5e5fb2af2bfe50f`
- Destination: `ondevice_ai/`
- Team ondevice_ai files: 123
- Standalone transferable files: 569
- Source-only: 455
- Modified collisions: 11
- Identical collisions: 103
- Team-only preserved: 9
- Excluded: 16

## Decision counts
- `ADD_SOURCE`: 455
- `PRESERVE_TEAM`: 112
- `REPLACE`: 11

## Modified collisions
- `README.md` → **REPLACE** — standalone reviewed active AI evidence/docs/code replaces older team component copy
- `datasets/MANIFEST.json` → **REPLACE** — standalone reviewed active AI evidence/docs/code replaces older team component copy
- `datasets/README.md` → **REPLACE** — standalone reviewed active AI evidence/docs/code replaces older team component copy
- `datasets/build_processed_npz.py` → **REPLACE** — standalone reviewed active AI evidence/docs/code replaces older team component copy
- `docs/README.md` → **REPLACE** — standalone reviewed active AI evidence/docs/code replaces older team component copy
- `docs/TEAM_HANDOFF_GUIDE.md` → **REPLACE** — standalone reviewed active AI evidence/docs/code replaces older team component copy
- `docs/reports/model_inventory.json` → **REPLACE** — standalone reviewed active AI evidence/docs/code replaces older team component copy
- `inference/validator.py` → **REPLACE** — standalone active AI runtime path
- `integrated_node/run_node.py` → **REPLACE** — standalone active AI runtime path
- `models/model_manifest.json` → **REPLACE** — standalone reviewed active AI evidence/docs/code replaces older team component copy
- `requirements-mac.txt` → **REPLACE** — standalone reviewed active AI evidence/docs/code replaces older team component copy

## Team-only preserved
- `integrated_node/esp32_sensor_node.ino`
- `models/mmwave/safenest_lstm_quant.tflite`
- `models/mmwave/sensor_stats_metadata.json`
- `models/thermal/thermal_fall_model.h5`
- `scripts/build_v4_archive.py`
- `scripts/build_v5_archive.py`
- `scripts/validate_v4_config.py`
- `tests/test_v4_config_validation.py`
- `tests/test_v5_release.py`


## Validation note (team checkout without raw payloads)

- Active `unittest` suite (excluding preserved team-only `test_v4_config_validation.py` / `test_v5_release.py`):
  - 444 run, 0 fail, 63 error, 2 skip
  - All 63 errors are `NOT_RUN_RAW_PAYLOAD_NOT_TRANSFERRED` (missing `datasets/raw_archives/...`)
- Preserved team-only legacy V4/V5 tests remain in `tests/` and currently fail against the synced multisensor manifests by design; they were not deleted.
