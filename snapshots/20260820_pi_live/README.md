# 2026-08-20 Pi live snapshot

Captured 2026-08-20 00:29 KST from `sandi@192.168.137.249:/home/sandi/safenest-runtime`.

- Pi process: `backend/run_backend.py` on TCP 9000, UDP 5005, HTTP 8000
- Pi git: `c3f95b8` on `main` (team layout), **not** `yuname121/integration` `66b5231`
- Live `/api/status` at capture: sensors disconnected/stale, `mmwave.runtime_status.ai_status=MODEL_PENDING`, `blocked_reason=MR60_NATIVE_MODEL_PENDING`
- `presence_available=false`; no `breath_phase` in the published mmWave values
- Excluded from git: `.venv`, thermal NPZ, full `safenest-runtime/` tree, `.env` secrets

`meta/env.redacted` is a redacted env example only.
