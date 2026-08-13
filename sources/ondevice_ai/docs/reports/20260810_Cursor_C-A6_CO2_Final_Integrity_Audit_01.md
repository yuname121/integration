# SafeNest CO₂ Phase C-A6 — Final Raw-to-Canonical Integrity Audit, Artifact Lock, and A-Series Release Readiness

- Document Version: `01`
- Author: `Cursor` (CO₂ Track Implementation Agent)
- Execution Date: `2026-08-10`
- Phase: `C-A6 — CO₂ Final Raw-to-Canonical Integrity Audit, Artifact Lock, and A-Series Release Readiness`
- Status: `PASS_WITH_WARNINGS`
- C-A6 merge readiness: `YES`
- A-series release: `READY_AFTER_MERGE` (Git tag / GitHub Release **not** created on feature branch)
- Isolated worktree: `/private/tmp/safenest-co2-ca6` (shared-checkout contamination avoided)

---

## 1. Executive Summary

Phase **C-A6** independently re-verified the complete C-A0→C-A5 real-data lineage for the UCI Occupancy Detection source, confirmed 1:1 source→canonical conservation for **20,560** samples (model-eligible **20,551**, warm-up **9** preserved), locked A-series machine-readable artifact identities, and recorded conservative release-readiness evidence. No scaler fitting, model training, split/target/slope redesign, tag, or GitHub Release was performed.

---

## 2. Predecessor Status

| Phase | Validator |
|---|---|
| C-A0 | `PASS_WITH_WARNINGS` |
| C-A1 | `PASS_WITH_WARNINGS` |
| C-A2 | `PASS_WITH_WARNINGS` |
| C-A3 | `PASS_WITH_WARNINGS` |
| C-A4 | `PASS_WITH_WARNINGS` |
| C-A5 | `PASS_WITH_WARNINGS` |
| C-A6 | `PASS_WITH_WARNINGS` |

C-A5 remains present on canonical `main` (PR #26 lineage). C-A6 does not redesign predecessor semantics.

---

## 3. Official UCI Identity / License / Raw Archive

| Field | Value |
|---|---|
| Dataset | UCI Occupancy Detection |
| UCI Dataset ID | 357 |
| UCI dataset DOI | `10.24432/C5X01N` |
| Journal publication DOI | `10.1016/j.enbuild.2015.11.071` |
| License | CC-BY-4.0 (`VERIFIED`) |
| Raw path | `datasets/raw_archives/external_datasets/occupancy+detection.zip` |
| Size | 335713 |
| SHA-256 | `4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a` |
| Git visibility | `GIT_IGNORED_RAW_ARCHIVE` |
| Release inclusion | **not** included in Git release payload |

### Raw member exact-byte identities (independently hashed)

| Member | Size | Rows | SHA-256 |
|---|---:|---:|---|
| datatest.txt | 200766 | 2665 | `1b92c7c1…584c7f` |
| datatraining.txt | 596674 | 8143 | `b2c4d0ce…21ab56` |
| datatest2.txt | 699664 | 9752 | `d026d1bd…985095` |

---

## 4. Schema / Timeline / Split / Target / Slope

- Schema: 7 named header fields / 8 physical data fields (`C-A1_UCI_OCCUPANCY_SCHEMA_PROFILE_001`)
- Timestamp: `SOURCE_ACQUISITION_CLOCK`, timezone `UNVERIFIED`, UTC conversion claimed `NO`
- Blocks: `BLOCK_01_DATATEST` / `BLOCK_02_DATATRAINING` / `BLOCK_03_DATATEST2`
- Split: VALIDATION / TRAIN / LOCKED_TEST respectively; random row-wise split `PROHIBITED`
- Group independence: `GROUP_INDEPENDENCE_NOT_VERIFIABLE` (not upgraded)
- Target: `CO2_OCCUPANCY_TARGET_PROFILE_001` — Occupancy `0→VACANT` (15810), `1→OCCUPIED` (4750); derivation `NONE`; labels modified `0`
- Slope: `CO2_SLOPE_FEATURE_PROFILE_001` — `ENDPOINT_DIFFERENCE`, `ppm/min`, history classification `CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED`, runtime equivalence claimed `NO`

---

## 5. Canonical Conservation / Eligibility / Ordering

| Metric | Count |
|---|---:|
| Source observations | 20560 |
| Canonical source samples | 20560 |
| Model-eligible | 20551 |
| Warm-up preserved | 9 |
| Missing source mappings | 0 |
| Duplicate source mappings | 0 |
| Duplicate canonical IDs | 0 |

Ordering remains C-A2 chronological member order: `datatest.txt` → `datatraining.txt` → `datatest2.txt`.

---

## 6. Round-Trip / Population / Fingerprint Closure

- Full population invariants independently re-derived via live raw reader + materialization.
- Representative round-trip cases cover first/last canonical rows, block edges, warm-ups, first eligible-per-block, TRAIN/VALIDATION interiors, LOCKED_TEST integrity-only sample, and VACANT/OCCUPIED examples.
- Method compares live reconstruction to raw observations and stored C-A5 JSONL (not file-self comparison).
- C-A5 predecessor fingerprint registry re-verified: `LOCKED`.

---

## 7. Synthetic / Model / LOCKED_TEST Protections

- `datasets/co2/processed/co2_occupancy_v1.npz` remains `SYNTHETIC_SMOKE_FIXTURE` and unused as real source.
- Existing model/scaler lineage: `MODEL_TRAINING_LINEAGE_UNVERIFIED` / `SCALER_FIT_LINEAGE_UNVERIFIED` / manifest `CONFIRMED_SYNTHETIC_ONLY`.
- Scaler fitted / model trained / model selected / quantization in C-A0..C-A6: **NO**.
- LOCKED_TEST not used for fitting, tuning, selection, or threshold calibration; integrity inspection only.

---

## 8. Artifact Lock

- Lock profile: `CO2_A_SERIES_ARTIFACT_LOCK_PROFILE_001`
- Path: `datasets/co2/manifests/c_a6_final_integrity_lock/artifact_lock_manifest.json`
- Locked artifact count: **33** (C-A0..C-A5 machine-readable release artifacts)
- Self-reference policy: lock does **not** hash itself; `checksums.sha256` hashes lock + other C-A6 evidence but **not** itself
- Raw ZIP identity recorded in lock but not committed

---

## 9. Release Readiness

- Scope: CO₂ real raw-to-canonical reconstruction milestone
- Status label: `CO2_A_SERIES_RELEASE_READY_AFTER_MERGE`
- `release_target_policy`: `C_A6_MERGE_COMMIT_ON_CANONICAL_MAIN`
- `release_commit`: `PENDING_POST_MERGE`
- Proposed tag: `co2-a-series-raw-to-canonical`
- Git tag created: **NO**
- GitHub Release created: **NO**

### Explicit exclusions

Not claimed by this release: real-data model validation, deployment readiness, SCD40 validation, safety-threshold calibration, multisensor integration validation, Raspberry Pi performance.

### Post-merge release procedure

1. `git fetch origin --prune`
2. Identify exact canonical `main` commit produced by the C-A6 merge (merge commit or squash result)
3. Verify that commit contains C-A6 locked artifacts
4. Re-run `python3 scripts/validate_co2_final_integrity.py`
5. Confirm proposed tag does not already exist
6. Create tag `co2-a-series-raw-to-canonical` pointing **exactly** to that merge result
7. Push tag
8. Create GitHub Release from that tag using `release_notes_draft.md`
9. Do not claim model/hardware/safety/integration validation
10. Report tag, target SHA, release URL

If later track merges land after C-A6: **do not** tag arbitrary latest `main`.

---

## 10. Validation Evidence

| Check | Result |
|---|---|
| C-A0..C-A6 validators | all `PASS_WITH_WARNINGS` |
| Focused CO₂ tests (`tests/test_co2_*.py`) | 79 passed |
| Focused C-A6 tests | 10 passed |
| Full repository `pytest tests/` | **533 passed, 5 skipped, 0 failed** |
| Import/compile | PASS |
| `git diff --check` | PASS |

## 11. Parallel Isolation

C-A6 implementation executed in dedicated worktree `/private/tmp/safenest-co2-ca6` while the shared checkout remained on unrelated `feature/M-B4-multiseed-stability` with dirty mmWave files. Branch-history and PR-diff isolation are re-checked at closeout.

---

## 12. Generated Artifacts

Under `datasets/co2/manifests/c_a6_final_integrity_lock/`:

- `full_chain_integrity_summary.json`
- `full_chain_audit_manifest.json`
- `artifact_lock_manifest.json`
- `predecessor_fingerprint_closure.json`
- `release_readiness_manifest.json`
- `release_notes_draft.md`
- `exceptions_and_limitations.json`
- `generation_metadata.json`
- `checksums.sha256`

Code/tests/report:

- `datasets/co2/integrity_audit.py`
- `scripts/audit_co2_final_integrity.py`
- `scripts/validate_co2_final_integrity.py`
- `tests/test_co2_final_integrity.py`
- `docs/reports/20260810_Cursor_C-A6_CO2_Final_Integrity_Audit_01.md`

---

## 13. C-B Boundary

C-A6 completion authorizes A-series release **after merge**, not model promotion. Next technical phase after release: **C-B0** offline real-data model comparison consuming locked C-A artifacts.
