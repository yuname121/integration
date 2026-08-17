# RP-X0 Pi field capture inventory

- Date copied: 2026-08-17 (KST)
- Pi host at copy: `sandi@172.20.10.3:/home/sandi/integration`
- Pi git: `diagnostic/rp-x0-b-runtime-wiring` @ `1ffbc7d`
- Classification: `RP_X0_PI_FIELD_CAPTURE_SNAPSHOT`

These files are a **read-only snapshot** of Pi logger output so Mac/offline review does not require the Pi. They are not a live-B gate opening and not a claim of clinical performance.

## Tracked in Git (JSONL)

| Path | Bytes | SHA-256 |
|---|---:|---|
| `data/mmwave/20260816_13_mmwave.jsonl` | 676647 | `8a24fb0a4f09f3d7d9df2212969235332c90d0f978adccc79cd148b719713e6e` |
| `data/mmwave/20260816_14_mmwave.jsonl` | 375404 | `63b9231d216ca2f1c3f82b34a8350f4a7d05d96b70e2f6fe3f2714c552baef5a` |
| `data/mmwave/20260816_15_mmwave.jsonl` | 118359 | `4559f2b7a7d3735d9f276d38bd0a8b4021213063adfcbe61ef039e9735828b61` |
| `data/mmwave/20260817_06_mmwave.jsonl` | 68114 | `b9dd06d501d37f0480c0e5a8404d20d0e112f6763e7717f8dce37a5a5ee84c09` |
| `data/mmwave/20260817_07_mmwave.jsonl` | 4858233 | `e496b1c4c8a249248f3e453f25ac54cb31cc52c2829d30e805b345909ce9a3dc` |
| `data/mmwave/20260817_08_mmwave.jsonl` | 10153795 | `0d31bfa7a7e86e3fa03a0421534c96a338e52499c90003314ee71266c0a40b75` |
| `data/mmwave/20260817_09_mmwave.jsonl` | 3593145 | `5f6774bbd5f0b9442fa52921148ccc263376a40b0544132456e13c7671727567` |
| `data/co2/20260816_13_co2.jsonl` | 9380 | `7441bd71f52ada892834a18483539a9683808b1fc14a0b782e7921871c79157f` |
| `data/co2/20260817_06_co2.jsonl` | 1343 | `c7229f101c322656ed89f78aac316904fdc974bfe31981dfedbee184d1cbb25a` |
| `data/co2/20260817_07_co2.jsonl` | 75747 | `5327b634ada75e4543cda6bfd976afe9c6e42d88792e3d10ebb9e9909081aa2b` |
| `data/co2/20260817_08_co2.jsonl` | 147456 | `9ed706632682cfd1260ada6a2b871bb8f0a8998c4cc576aa934b5ec28dbabc04` |
| `data/co2/20260817_09_co2.jsonl` | 49865 | `479c35dca9283d4e2efc0a07edd695dd0d5ac1ab5663f7d2bdc3292a2c9f3970` |

Stage 8 used `data/mmwave/20260817_08_mmwave.jsonl` (SHA matches the Stage 8 copy).

## Not tracked (Pi-local)

| Item | On Pi | Why not in Git |
|---|---|---|
| `data/thermal/*.npz` | 1979 files, **140 MB** | raw MI48 frames; gitignore remains; live uint16→P1 still UNVERIFIED |
| `data/safenest.db` | ignored `*.db` | runtime sqlite |
| `logs/` | ignored | runtime logs |

Thermal frames stay on the Pi. Ask if a separate evidence zip is needed; do not `git add data/thermal`.

## Gitignore policy after this snapshot

- `data/mmwave/*.jsonl` and `data/co2/*.jsonl` are **exceptions** (tracked snapshots).
- Future live logger output in those folders will show as local modifications. Prefer copying a new session to a new filename rather than rewriting a tracked snapshot.
- `data/thermal/*` stays ignored except `.gitkeep`.
