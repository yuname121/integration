# SafeNest CO₂ Phase C-A5 — Canonical Sample Provenance and Group-Wise Split Materialization

- Document Version: `01`
- Author: `Cursor` (CO₂ Track Implementation Agent)
- Execution Date: `2026-08-10`
- Phase: `C-A5 — CO₂ Canonical Sample Provenance and Group-Wise Split Materialization`
- Status: `PASS_WITH_WARNINGS`
- C-A6 Authorization (post-merge gate readiness): `YES` (do not start C-A6 on this branch)
- A-series release: `DEFERRED_UNTIL_C-A6`

---

## 1. Executive Summary

Phase **C-A5** materializes a deterministic canonical sample contract joining C-A1 source-row provenance, C-A2 temporal-block/split membership, C-A3 `CO2_slope` lineage/status, and C-A4 occupancy targets for all **20,560** real UCI source rows. Warm-up rows remain represented. Model-eligible samples (**20,551**) are an explicit derived view. No scaler fitting, model training, split redesign, or synthetic-fixture mixing was performed.

---

## 2. Predecessor Gate

| Gate | Result |
|---|---|
| Canonical base | `origin/main` @ `812061e` |
| C-A4 merge | Present on `main` (PR #24 lineage) |
| C-A0..C-A4 validators | all `PASS_WITH_WARNINGS` |
| Fresh branch | `feature/C-A5-co2-canonical-samples` from updated `main` |

---

## 3. Canonical Sample Profile

- **Profile ID:** `CO2_CANONICAL_SAMPLE_PROFILE_001`
- **Grain:** one C-A1 source row → one canonical source sample
- **Canonical sample ID:** `co2cs_` + `sha256(archive_sha256\|member\|source_row_identifier\|physical_line)[:32]`
- **Ordering:** `CHRONOLOGICAL_C_A2_MEMBER_ORDER` — `datatest.txt` → `datatraining.txt` → `datatest2.txt`
- **Total canonical source samples:** `20560`

---

## 4. Provenance Chain

```text
raw UCI archive
→ C-A1 source observation
→ C-A2 temporal block + future_split_role
→ C-A3 CO2_slope value/status + history lineage
→ C-A4 occupancy source value + canonical class
→ C-A5 canonical sample record
```

Each JSONL record retains archive/member/row identifiers, timestamps, block ID, split role, measured fields, slope status/value, and occupancy target fields.

---

## 5. Immutable Inherited Contracts

| Contract | Inheritance |
|---|---|
| Split | `BLOCK_02→TRAIN`, `BLOCK_01→VALIDATION`, `BLOCK_03→LOCKED_TEST` |
| Target | `Occupancy` `0→VACANT`, `1→OCCUPIED`; derivation `NONE` |
| Slope | `CO2_SLOPE_FEATURE_PROFILE_001` / `ENDPOINT_DIFFERENCE` / warm-up preserved |

---

## 6. Counts

| Metric | Count |
|---|---:|
| Canonical source samples | 20560 |
| CO2_slope eligible | 20551 |
| Warm-up / unavailable | 9 |
| Missing source mappings | 0 |
| Duplicate source mappings | 0 |
| Duplicate canonical IDs | 0 |

### Per split

| Role | Canonical | Slope-eligible | Warm-up | VACANT | OCCUPIED |
|---|---:|---:|---:|---:|---:|
| TRAIN | 8143 | 8140 | 3 | 6414 | 1729 |
| VALIDATION | 2665 | 2662 | 3 | 1693 | 972 |
| LOCKED_TEST | 9752 | 9749 | 3 | 7703 | 2049 |

---

## 7. Canonical vs Model-Eligible

- **CANONICAL_SOURCE_SAMPLE:** all 20,560 rows (including warm-up).
- **MODEL_ELIGIBLE_SAMPLE:** 20,551 rows with `co2_slope_status == FEATURE_AVAILABLE`.
- Exclusions use reason `FEATURE_UNAVAILABLE_WARMUP` (not malformed data).

---

## 8. LOCKED_TEST / Scaler Boundary

- LOCKED_TEST membership and provenance are materialized.
- LOCKED_TEST is **not** authorized for fit, tuning, feature-contract tuning, or threshold calibration.
- Future scaler-fit population is **TRAIN only**.
- C-A5 does **not** compute scaler statistics.

---

## 9. Predecessor Fingerprint Lock

`predecessor_fingerprint_registry.json` locks SHA-256 identities of consumed C-A1..C-A4 machine-readable artifacts plus the raw archive path/hash. Validator fails if upstream evidence changes without regeneration.

---

## 10. Synthetic Isolation

`datasets/co2/processed/co2_occupancy_v1.npz` remains `SYNTHETIC_SMOKE_FIXTURE` and is not part of real canonical lineage.

---

## 11. Determinism

Audit generation was run repeatedly; `checksums.sha256` was identical across successive runs (`DETERMINISM_CHECKSUMS:IDENTICAL`).

---

## 12. Artifacts

Directory: `datasets/co2/manifests/c_a5_canonical_samples/`

- `canonical_sample_profile.json`
- `predecessor_fingerprint_registry.json`
- `split_membership_manifest.json`
- `feature_availability_manifest.json`
- `materialization_integrity_summary.json`
- `exceptions_and_limitations.json`
- `generation_metadata.json`
- `canonical_source_samples.jsonl`
- `model_eligible_sample_ids.jsonl`
- `artifact_identity.json`
- `checksums.sha256`

Code / validation:

- `datasets/co2/canonical_samples.py`
- `scripts/audit_co2_canonical_samples.py`
- `scripts/validate_co2_canonical_samples.py`
- `tests/test_co2_canonical_samples.py`

---

## 13. Validation Evidence (final closeout)

| Check | Result |
|---|---|
| C-A0 validator | `PASS_WITH_WARNINGS` |
| C-A1 validator | `PASS_WITH_WARNINGS` |
| C-A2 validator | `PASS_WITH_WARNINGS` |
| C-A3 validator | `PASS_WITH_WARNINGS` |
| C-A4 validator | `PASS_WITH_WARNINGS` |
| C-A5 standalone validator | `PASS_WITH_WARNINGS` (0 errors, 8 warnings) |
| Focused C-A5 tests | 8 passed |
| All CO₂ tests (`tests/test_co2_*`) | **69 passed**, 0 failed, 0 errors, 0 skipped |
| Full repository regression (`pytest tests/`) | **487 passed**, 0 failed, 0 errors, **4 skipped** |
| Import/compile | PASS (`compileall` + `datasets.co2.*` imports) |
| `git diff --check` / branch diff `--check` | PASS |
| Artifact SHA-256 vs committed bytes | PASS (all listed `checksums.sha256` entries) |
| Determinism | identical `checksums.sha256` across regenerations; committed bytes match |

Inherited non-blocking warnings retained (timezone, single-room group independence, model/scaler lineage unverified, slope history lineage, SCD40 cadence gap, deferred shared update, A-series release deferred).

---

## 14. Parallel Isolation (verified remote PR)

Execution used isolated worktree `/tmp/safenest-ca5-worktree` on `feature/C-A5-co2-canonical-samples`. Primary workspace remained on unrelated `feature/M-B3-architecture-comparison` and was not staged into C-A5.

### Verified Git evidence

| Gate | Result |
|---|---|
| Working-tree isolation (C-A5 worktree) | PASS (clean after commit) |
| Branch-history isolation | PASS — unique commits vs `origin/main` are C-A5-only |
| PR-diff isolation | PASS — remote PR #26 file list is CO₂ C-A5 only |
| Unrelated mmWave commits in C-A5 ancestry | 0 |
| Unrelated Thermal commits in C-A5 ancestry | 0 |

### Remote PR inspection (not local-only)

| Field | Value |
|---|---|
| PR | https://github.com/sheepmeat/test/pull/26 |
| PR base | `main` |
| PR head | `feature/C-A5-co2-canonical-samples` |
| Implementation commit | `a1fe383` — `feat(co2): complete Phase C-A5 canonical sample provenance and split materialization` |
| Closeout docs commit | recorded in PR history after this report revision |
| Changed-file count | 16 (implementation paths; report-only closeout may add no new paths) |
| mmWave files in PR | 0 |
| Thermal files in PR | 0 |
| Unauthorized shared files in PR | 0 |
| Raw payload in PR | 0 |
| Synthetic NPZ modifications in PR | 0 |

Note: after C-A5 branch creation, `origin/main` advanced with merged Thermal T-A2 (`bde7a0f`). Those Thermal commits are on `main`, not in C-A5 unique ancestry. Local merge-probe of C-A5 onto current `origin/main` completed without conflict.

---

## 15. Deferred Work

| Item | Status |
|---|---|
| C-A6 final conversion integrity audit / artifact lock | DEFERRED |
| C-B model selection / scaler fit | DEFERRED |
| C-C SCD40 domain | DEFERRED |
| CO₂ A-series release/tag | `DEFERRED_UNTIL_C-A6` |
| Shared inventory/contract refresh | `DEFERRED_SHARED_INTEGRATION_UPDATE` |

---

## 16. C-A6 Authorization Gate (readiness only)

C-A6 may proceed only after this C-A5 contract is **merged** and isolation gates remain clean. This branch must not begin C-A6 implementation.

C-A6 integrity target chain:

```text
RAW 20,560
  ↓ 1:1
CANONICAL 20,560
  ↓ availability filter only
MODEL-ELIGIBLE 20,551
```

A-series release/tag remains `DEFERRED_UNTIL_C-A6`.
