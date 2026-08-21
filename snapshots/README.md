# Integration snapshots

This folder stores **small, git-safe field/local evidence**. It is not a second repository.

| Path | Role | Up to date vs `yuname121/integration` `main`? |
|---|---|---|
| Active repo (this tree) | Development source of truth | Local branch was ahead of `origin/main` when captured |
| `20260820_pi_live/` | Pi `/home/sandi/safenest-runtime` API/meta capture | No. Pi git was `c3f95b8` on team `main`, not this integration SHA |
| `20260821_integration_local/` | Local git inventory at capture time | Describes this machine, not the Pi |

Do **not** copy into git:

- `/Users/junwoo/Library/Mobile Documents/com~apple~CloudDocs/대학/2026/safenest-pi-integration-snapshot` (forensic 2026-08-17 copy, ~682 MB including Pi `.venv`)
- `safenest-pi-snapshot-20260820/safenest-runtime` (~694 MB tree)
- `.env` secrets, Thermal NPZ, SQLite DB

The 2026-08-17 sibling snapshot stays read-only at `diagnostic/rp-x0-b-runtime-wiring` `@ 1ffbc7d`.
