# Phase A5 — Subject-Wise Split and End-to-End Sample Provenance

## 1. Executive Summary

- A5 gate: **PASS_WITH_WARNINGS**
- A6 entry: **READY_WITH_CONDITIONS**
- 110 subjects and 440 recordings received one deterministic subject-level split.
- Subject, recording, pilot-window, and pilot exact-signal cross-split overlap: **0**.

## 2. Git Baseline

- Repository: `https://github.com/sheepmeat/test.git`
- Baseline main commit: `9c1fae1ab693425a81229d151cad285d4a485c52`
- Branch: `feature/phase-a5-subject-split-provenance`
- A4 profile: `MMWAVE_LABEL_MAPPING_PROFILE_001`

## 3. A0–A4 Input Contracts

A0 supplied the complete subject/recording roster; A2 supplied selected phase coordinates; A3 supplied `MMWAVE_TIMELINE_PROFILE_001`; A4 supplied the unchanged pilot labels under `MMWAVE_LABEL_MAPPING_PROFILE_001`.

## 4. Full Subject Inventory

- Subjects: 110
- Recordings: 440 (440 unique IDs)
- Recordings per subject: minimum 4, maximum 4 (derived from A0)

## 5. Available Stratification Metadata

Posture, source condition, annotation presence, and recording count are available. `ParticipantsInfo.xlsx` is absent, so age/sex/height/weight balance is not verifiable and none was inferred.

## 6. Split Ratio Decision

No approved real-data ratio existed in main. The prompt baseline 70/15/15 was applied with largest-remainder integer allocation: TRAIN 77, VALIDATION 17, LOCKED_TEST 16.

## 7. Deterministic Allocation Method

Profile `MMWAVE_SUBJECT_SPLIT_PROFILE_001` uses seed `20260808` and orders subjects by `SHA256("20260808:<subject_id>")`; filesystem order and Python random state are irrelevant.

## 8. Train Subject Assignment

P001, P002, P003, P004, P005, P006, P007, P008, P010, P011, P014, P015, P018, P020, P021, P022, P026, P028, P030, P032, P034, P035, P037, P038, P040, P041, P042, P043, P045, P048, P049, P051, P052, P053, P054, P056, P057, P058, P061, P062, P064, P065, P066, P068, P069, P070, P071, P072, P076, P077, P078, P080, P081, P082, P083, P084, P085, P088, P089, P090, P092, P093, P094, P095, P096, P097, P098, P099, P100, P102, P103, P105, P106, P107, P108, P109, P110

## 9. Validation Subject Assignment

P009, P012, P013, P016, P024, P025, P027, P031, P036, P047, P050, P060, P073, P074, P075, P087, P104

## 10. Locked-Test Subject Assignment

P017, P019, P023, P029, P033, P039, P044, P046, P055, P059, P063, P067, P079, P086, P091, P101

## 11. Recording Inheritance

All 440 A0 recordings inherit `subject_split_map[subject_id]`; cross-split recording overlap is 0.

## 12. A4 Pilot Window Inheritance

All 15 A4 pilot windows inherit their subject split without label recalculation. Fourteen remain ASSIGNED; one AMBIGUOUS window is retained.

## 13. Training / Validation / Locked-Test Eligibility

- Training eligible: 13
- Validation eligible: 1
- Locked-test evaluation eligible: 0
- AMBIGUOUS windows are ineligible for all pure-class roles.

## 14. Provenance Schema

`provenance_schema.json` defines archive→member→subject→recording→A1→A2→A3→A4→A5→future NPZ index linkage. Current records are `synthetic=false`. Timestamp reference is `COMMON_ACQUISITION_COMPUTER_CLOCK`, source timezone is `UNVERIFIED`, and UTC conversion is not claimed. A preserved legacy trailing `Z` is not treated as UTC evidence.

## 15. Split Balance Audit

The measured A0 roster has identical two-posture/two-condition coverage and two annotation-bearing recordings per subject. Pilot label statistics are explicitly marked `A4_PILOT_ONLY`, not full-dataset class balance.

## 16. Subject Leakage Audit

TRAIN∩VALIDATION=0, TRAIN∩LOCKED_TEST=0, VALIDATION∩LOCKED_TEST=0; union coverage=110/110.

## 17. Recording Leakage Audit

All 440 recording IDs appear once and inherit their subject split; overlap=0.

## 18. Pilot Window / Duplicate Hash Audit

All 15 window IDs appear once. Exact hashes use SHA-256 over contiguous little-endian float64 canonical phase samples. Cross-split window overlap=0 and exact-signal duplicate overlap=0. This is a pilot-only audit.

## 19. Reproducibility

Generation was repeated; manifests and checksums were byte-identical. Input-order invariance is covered by unit tests.

## 20. Exceptions / Warnings

No blocker or error. Warnings: verified participant demographics unavailable; full class distribution is deferred to A6.

## 21. A5 Gate

**PASS_WITH_WARNINGS** after the standalone/in-memory validator passed.

## 22. A6 Entry Decision

**READY_WITH_CONDITIONS**: A6 must inherit this immutable split and audit full label/quality balance.

## 23. Remaining Limitations

Demographic balance and full class balance are unknown. Voluntary breath hold remains a derived SafeNest APNEA proxy, not clinical ground truth.

## 24. Explicit Non-Scope

```text
Full 440-recording conversion: NOT PERFORMED
Full A4 label application: NOT PERFORMED
Training NPZ generation: NOT PERFORMED
Preprocessing ablation: NOT PERFORMED
Class balancing: NOT PERFORMED
Model training: NOT PERFORMED
Validation-set model selection: NOT PERFORMED
Locked-test model evaluation: NOT PERFORMED
TFLite conversion: NOT PERFORMED
INT8 quantization: NOT PERFORMED
A6: NOT PERFORMED
```

## 25. Files Changed

A5 adds three modular scripts, one unit-test module, nine mandatory manifest/checksum artifacts, one split lookup contract, and this report. A0–A4 artifacts were not modified.

## 26. Commands / Tests

The A5 unit suite, A0–A4 regressions, real generator, standalone validator, deterministic regeneration, checksum audit, archive pre/post hash, and `git diff --check` were run. Archive SHA-256 remained `f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0`.
