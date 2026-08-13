# SafeNest CO₂ A-Series Comprehensive Final Audit

- Document Version: `01`
- Author: `Cursor` (Independent Verification / Audit Agent)
- Execution Date: `2026-08-10`
- Audit scope: `C-A0 → C-A6` end-to-end raw-to-canonical verification
- Audit mode: **read-only** (no artifact regeneration for pass-hiding; no automatic repair)
- Overall verdict: `PASS_WITH_WARNINGS`
- Release gate: `READY`
- Audited commit (exact C-A6 merge on canonical main): `bfd860cad2bb8dafe35ef7600cfa931d7d2d554d`
- Audit worktree: `/private/tmp/safenest-co2-aseries-audit` (detached HEAD at C-A6 merge)

---

## A. Executive Verdict

```text
CO₂ A-Series Overall Verdict: PASS_WITH_WARNINGS
Release Gate: READY
```

Independent recomputation of raw ZIP bytes → source rows → temporal split → slope → target → canonical samples → artifact lock → release readiness passed all blocking integrity gates. Known non-blocking provenance/domain limitations remain intentionally retained.

One non-blocking documentation residue was found (stale incorrect DOI string in a validator module docstring). Authoritative machine-readable evidence and validator assertions correctly use `10.24432/C5X01N`.

---

## B. Raw Source Integrity

| Field | Independent result |
|---|---|
| Dataset | UCI Occupancy Detection |
| UCI Dataset ID | 357 |
| UCI dataset DOI | `10.24432/C5X01N` |
| Journal publication DOI | `10.1016/j.enbuild.2015.11.071` |
| DOI conflation | NO |
| License | CC-BY-4.0 (`VERIFIED`) |
| Raw path | `datasets/raw_archives/external_datasets/occupancy+detection.zip` |
| Size | 335713 |
| SHA-256 | `4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a` |
| Git tracked | NO (`GIT_IGNORED_RAW_ARCHIVE`) |

### Raw member exact-byte identities

| Member | Size | Rows | SHA-256 | Match |
|---|---:|---:|---|---|
| `datatest.txt` | 200766 | 2665 | `1b92c7c1b2838963464fa891a610cf3c5db4becb7189189b29b330107a584c7f` | PASS |
| `datatraining.txt` | 596674 | 8143 | `b2c4d0ce2b9e4e453c476f7125ef31aeec2d1f5c7f5572d0e80de3df6521ab56` | PASS |
| `datatest2.txt` | 699664 | 9752 | `d026d1bd5aeccd4aff4f3b3710d48e40613bd5fc370db7e61bbdcaa50d985095` | PASS |

Total source rows: **20560**

Schema: **7** named header fields / **8** physical data fields; unexpected physical field counts: **0**

---

## C. Source-Row Integrity (C-A1)

| Metric | Result |
|---|---:|
| Raw rows | 20560 |
| Safe-reader rows | 20560 |
| Missing rows | 0 |
| Duplicate source mappings | 0 |
| Silent filtering | 0 |

C-A1 does not perform slope reconstruction, model scaling, normalization, label rewriting, or random splitting.

---

## D. Temporal / Split Integrity (C-A2)

| Block | Member | Start | End | Rows | Role |
|---|---|---|---|---:|---|
| BLOCK_01_DATATEST | datatest.txt | 2015-02-02 14:19:00 | 2015-02-04 10:43:00 | 2665 | VALIDATION |
| BLOCK_02_DATATRAINING | datatraining.txt | 2015-02-04 17:51:00 | 2015-02-10 09:33:00 | 8143 | TRAIN |
| BLOCK_03_DATATEST2 | datatest2.txt | 2015-02-11 14:48:00 | 2015-02-18 09:19:00 | 9752 | LOCKED_TEST |

| Check | Result |
|---|---|
| Timestamp reversals | 0 |
| Duplicate timestamps | 0 |
| Timestamp reference | `SOURCE_ACQUISITION_CLOCK` |
| Source timezone | `UNVERIFIED` |
| UTC conversion claimed | NO |
| Inter-block gap BLOCK_01→02 | 25680 s |
| Inter-block gap BLOCK_02→03 | 105300 s |
| Cross-block slope history | 0 |
| Random row-wise split | PROHIBITED |
| Group independence | `GROUP_INDEPENDENCE_NOT_VERIFIABLE` |

---

## E. Feature Integrity (C-A3)

| Field | Result |
|---|---|
| Profile | `CO2_SLOPE_FEATURE_PROFILE_001` |
| Method | `ENDPOINT_DIFFERENCE` |
| Unit | `ppm/min` |
| Causality | PAST_ONLY |
| History classification | `CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED` |
| Runtime equivalence claimed | NO |
| Verified historical training contract | NO |
| Slope-eligible samples | 20551 |
| Warm-up samples | 9 (preserved; not deleted/malformed) |
| Non-finite eligible slopes | 0 |

Independent endpoint-difference formula recomputation matched reconstructed values.

Per-split availability:

| Role | Eligible | Warm-up |
|---|---:|---:|
| TRAIN | 8140 | 3 |
| VALIDATION | 2662 | 3 |
| LOCKED_TEST | 9749 | 3 |

---

## F. Target Integrity (C-A4)

| Field | Result |
|---|---|
| Profile | `CO2_OCCUPANCY_TARGET_PROFILE_001` |
| Target | Occupancy |
| Mapping | 0→VACANT, 1→OCCUPIED |
| Semantic | ROOM_OCCUPANCY |
| Label derivation | NONE |
| Threshold-based occupancy relabeling | PROHIBITED / 0 |
| VACANT (0) | 15810 |
| OCCUPIED (1) | 4750 |
| Target modifications | 0 |

Split occupancy counts match C-A2/C-A4 contract (TRAIN 6414/1729, VALIDATION 1693/972, LOCKED_TEST 7703/2049).

Occupancy is not conflated with CO₂ danger, sensor health, or multisensor risk.

---

## G. Canonical Materialization (C-A5)

| Field | Result |
|---|---|
| Profile | `CO2_CANONICAL_SAMPLE_PROFILE_001` |
| Canonical source samples | 20560 |
| Model-eligible samples | 20551 |
| Warm-up preserved in canonical universe | YES |
| Source→canonical mapping | 1:1 |
| Missing source mappings | 0 |
| Duplicate source mappings | 0 |
| Duplicate canonical IDs | 0 |
| Canonical ID recompute mismatches | 0 |
| Ordering | datatest.txt → datatraining.txt → datatest2.txt |

Full-population live vs stored C-A5 JSONL mismatch categories (id/member/row/line/timestamp/block/split/target/slope/measured fields): all **0**.

Representative round-trip provenance (including raw ZIP physical-line traces): **PASS**.

---

## H. Artifact / Fingerprint Lock (C-A5/C-A6)

| Field | Result |
|---|---|
| Predecessor fingerprint closure | PASS |
| Artifact lock profile | `CO2_A_SERIES_ARTIFACT_LOCK_PROFILE_001` |
| Lock path | `datasets/co2/manifests/c_a6_final_integrity_lock/artifact_lock_manifest.json` |
| Locked artifact count | 33 |
| Independently recomputed lock SHA-256 | `b63f5e2da988f8e685cf1a01ec8e79c2c37f5bc77359be647f1147ecfb04e3da` |
| Matches previously reported lock hash | YES |
| Per-entry hash/size verification failures | 0 |
| Self-referential checksum cycle | NO |
| Unverifiable checksum entries | 0 |

---

## I. Synthetic / Model / Scaler Isolation

| Check | Result |
|---|---|
| `datasets/co2/processed/co2_occupancy_v1.npz` used as real source | NO |
| Existing model lineage | `MODEL_TRAINING_LINEAGE_UNVERIFIED` |
| Model manifest CO₂ status | `CONFIRMED_SYNTHETIC_ONLY` |
| Existing scaler lineage | `SCALER_FIT_LINEAGE_UNVERIFIED` |
| Scaler fitted during A-series | NO |
| Model trained during A-series | NO |
| Quantization / model promotion during A-series | NO |

---

## J. LOCKED_TEST Protection

| Usage | Result |
|---|---|
| Scaler fitting | NO |
| Feature-contract tuning | NO |
| Model selection | NO |
| Hyperparameter tuning | NO |
| Threshold calibration | NO |
| Integrity-only inspection | YES |

---

## K. Determinism

Method: independent double live materialization of canonical sample IDs; rebuild of artifact-lock and release-readiness builder outputs without overwriting tracked artifacts.

Result: **PASS** (identical IDs/ordering/builder outputs).

---

## L. Validators / Tests

| Check | Exit / Result |
|---|---|
| C-A0 `validate_co2_raw_inventory.py` | 0 / `PASS_WITH_WARNINGS` (errors 0, warnings 5) |
| C-A1 `validate_co2_safe_reader.py` | 0 / `PASS_WITH_WARNINGS` (errors 0, warnings 5) |
| C-A2 `validate_co2_temporal_blocks.py` | 0 / `PASS_WITH_WARNINGS` (errors 0, warnings 5) |
| C-A3 `validate_co2_slope_feature.py` | 0 / `PASS_WITH_WARNINGS` (errors 0, warnings 8) |
| C-A4 `validate_co2_target_semantics.py` | 0 / `PASS_WITH_WARNINGS` (errors 0, warnings 10) |
| C-A5 `validate_co2_canonical_samples.py` | 0 / `PASS_WITH_WARNINGS` (errors 0, warnings 8) |
| C-A6 `validate_co2_final_integrity.py` | 0 / `PASS_WITH_WARNINGS` (errors 0, warnings 11) |
| CO₂ tests `tests/test_co2_*.py` | **79 passed** |
| Full repository `pytest tests/` | **533 passed, 5 skipped, 0 failed** |
| `compileall` (CO₂ datasets/scripts) | PASS |
| `git diff --check` | PASS |

---

## M. Git / Parallel Isolation

| Gate | Result |
|---|---|
| Working-tree isolation | PASS (clean detached audit worktree) |
| Branch-history isolation | PASS (audited exact C-A6 merge commit) |
| Diff isolation | PASS |
| CO₂ path drift `bfd860c` → current `origin/main` | 0 files |
| Raw payload tracked | NO |
| Unrelated mmWave/Thermal contamination of audit evidence | NONE |

Note: at audit time, `origin/main` had advanced with later M-B4/Thermal merges beyond the C-A6 merge commit. Those later commits are legitimate canonical-main history and do not alter CO₂ A-series artifacts.

---

## N. Remaining Warnings (evidence-supported)

- `SOURCE_TIMEZONE_UNVERIFIED`
- `GROUP_INDEPENDENCE_NOT_VERIFIABLE`
- `MODEL_TRAINING_LINEAGE_UNVERIFIED`
- `SCALER_FIT_LINEAGE_UNVERIFIED`
- `CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED`
- `DEVICE_UCI_CADENCE_DOMAIN_GAP`
- `SAFETY_RULE_CONTRACT_OUT_OF_SCOPE`
- `SENSOR_HEALTH_CONTRACT_OUT_OF_SCOPE`
- `MULTISENSOR_RISK_CONTRACT_OUT_OF_SCOPE`
- `DEFERRED_SHARED_INTEGRATION_UPDATE`

Plus C-A6 release-tag deferred warning semantics where recorded in the C-A6 exception registry.

---

## O. Release Scope

The CO₂ A-series release certifies:

```text
real raw source identity established
→ safe ingestion established
→ temporal grouping established
→ canonical slope semantics established
→ occupancy target semantics established
→ canonical sample universe established
→ artifacts/fingerprints locked
```

It does **not** certify:

- real-data model validation
- SCD40 device validation
- model production readiness
- safety-threshold validation
- multisensor integration validation
- Raspberry Pi deployment/performance validation

Release overclaim detected: **NO**

---

## P. Defects

```text
BLOCKING DEFECTS:
NONE
```

### Non-blocking documentation residue

| Field | Value |
|---|---|
| File | `scripts/validate_co2_raw_inventory.py` |
| Issue | Module docstring still mentions incorrect DOI `10.24432/C5CW2B` |
| Expected | Documentation should say `10.24432/C5X01N` |
| Observed | Docstring stale; **validator assertions correctly require `10.24432/C5X01N`** |
| Severity | Non-blocking documentation inconsistency |
| Phase ownership | C-A0 validator docstring hygiene |
| Blocks A-series release? | NO |

This audit did **not** auto-repair the docstring.

---

## Historical Correction Closure

| Correction | Status |
|---|---|
| UCI DOI `10.24432/C5X01N` (not `C5CW2B`) in authoritative manifests / validator logic | CLOSED |
| Residual stale docstring `C5CW2B` | OPEN (non-blocking) |
| C-A3 offline 150s baseline not claimed as active-runtime equivalent | CLOSED |

---

## Recommended Next Action

1. Tag/release against the exact C-A6 merge commit on canonical main: `bfd860c` (`co2-a-series-raw-to-canonical` per C-A6 proposal).
2. Do **not** tag an arbitrary later `main` HEAD solely because other tracks merged afterward.
3. Optionally fix the stale C-A0 validator docstring in a separate hygiene commit that does not move the release tag target.
4. Proceed to **C-B0** after A-series release.

---

## Final Machine-Readable-Style Block

```text
CO2 A-SERIES COMPREHENSIVE AUDIT RESULT:
- Audit scope: C-A0 through C-A6 end-to-end independent recomputation
- Repository root: /private/tmp/safenest-co2-aseries-audit
- Audited branch/commit: detached HEAD bfd860cad2bb8dafe35ef7600cfa931d7d2d554d (Merge PR #29 C-A6)
- Canonical main reference: origin/main at audit time (later non-CO2 merges present; CO2 paths unchanged)
- Working-tree isolation: PASS
- Branch-history isolation: PASS
- Diff isolation: PASS
- UCI dataset: UCI Occupancy Detection
- UCI dataset ID: 357
- UCI dataset DOI: 10.24432/C5X01N
- Journal DOI: 10.1016/j.enbuild.2015.11.071
- License: CC-BY-4.0 VERIFIED
- Raw archive path: datasets/raw_archives/external_datasets/occupancy+detection.zip
- Raw archive size: 335713
- Raw archive SHA-256: 4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a
- Raw member hashes: PASS
- Raw source rows: 20560
- Safe-reader rows: 20560
- Schema contract: 7 header / 8 physical; unexpected=0
- Timestamp reference: SOURCE_ACQUISITION_CLOCK
- Source timezone: UNVERIFIED
- UTC conversion claimed: NO
- Temporal block count: 3
- BLOCK_01 role: VALIDATION
- BLOCK_02 role: TRAIN
- BLOCK_03 role: LOCKED_TEST
- Random row-wise split: PROHIBITED
- Group independence: GROUP_INDEPENDENCE_NOT_VERIFIABLE
- Slope profile: CO2_SLOPE_FEATURE_PROFILE_001
- Slope method: ENDPOINT_DIFFERENCE
- Slope unit: ppm/min
- Slope history classification: CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED
- Runtime equivalence: NO
- Slope-eligible samples: 20551
- Warm-up samples: 9
- Warm-up preservation: YES
- Target profile: CO2_OCCUPANCY_TARGET_PROFILE_001
- Target semantic: ROOM_OCCUPANCY (0=VACANT, 1=OCCUPIED)
- VACANT count: 15810
- OCCUPIED count: 4750
- Target modifications: 0
- Canonical sample profile: CO2_CANONICAL_SAMPLE_PROFILE_001
- Canonical source samples: 20560
- Model-eligible samples: 20551
- Source-to-canonical mapping: 1:1
- Missing source mappings: 0
- Duplicate source mappings: 0
- Duplicate canonical IDs: 0
- Canonical ordering: datatest.txt → datatraining.txt → datatest2.txt
- Synthetic fixture used in real lineage: NO
- Existing model lineage: MODEL_TRAINING_LINEAGE_UNVERIFIED / CONFIRMED_SYNTHETIC_ONLY
- Existing scaler lineage: SCALER_FIT_LINEAGE_UNVERIFIED
- Scaler fit during A-series: NO
- Model training during A-series: NO
- LOCKED_TEST protection: PASS
- Predecessor fingerprint closure: PASS
- Artifact lock profile: CO2_A_SERIES_ARTIFACT_LOCK_PROFILE_001
- Locked artifact count: 33
- Artifact lock SHA-256: b63f5e2da988f8e685cf1a01ec8e79c2c37f5bc77359be647f1147ecfb04e3da
- Artifact lock verification: PASS
- Self-referential checksum defect: NO
- Full population audit: PASS
- Round-trip provenance audit: PASS
- Determinism: PASS
- Portable path audit: PASS
- Raw payload tracked: NO
- C-A0 validator: PASS_WITH_WARNINGS
- C-A1 validator: PASS_WITH_WARNINGS
- C-A2 validator: PASS_WITH_WARNINGS
- C-A3 validator: PASS_WITH_WARNINGS
- C-A4 validator: PASS_WITH_WARNINGS
- C-A5 validator: PASS_WITH_WARNINGS
- C-A6 validator: PASS_WITH_WARNINGS
- CO2 tests: 79 passed
- Full repository tests: 533 passed, 5 skipped, 0 failed
- Import/compile: PASS
- git diff --check: PASS
- Known DOI correction closed: YES for authoritative manifests/assertions; residual stale docstring C5CW2B in validate_co2_raw_inventory.py
- C-A3 runtime-equivalence correction closed: YES
- Release scope: CO2_REAL_RAW_TO_CANONICAL_RECONSTRUCTION_MILESTONE
- Release overclaim detected: NO
- Non-blocking warnings: retained A-series limitation set + stale docstring DOI residue
- Blocking defects: NONE
- Overall verdict: PASS_WITH_WARNINGS
- CO2 A-series release gate: READY
- Recommended next action: Tag exact C-A6 merge commit bfd860c; then proceed to C-B0
```
