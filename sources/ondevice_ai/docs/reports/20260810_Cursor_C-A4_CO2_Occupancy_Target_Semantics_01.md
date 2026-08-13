# SafeNest CO₂ Phase C-A4 — Occupancy Label Semantics, Safety Separation, and Canonical Target Contract

- Document Version: `01`
- Author: `Cursor` (CO₂ Track Implementation Agent)
- Execution Date: `2026-08-10`
- Phase: `C-A4 — CO₂ Occupancy Label Semantics, Safety Separation, and Canonical Target Contract`
- Status: `PASS_WITH_WARNINGS`
- C-A5 Authorization: `YES`

---

## 1. Executive Summary

Phase **C-A4** locks a deterministic occupancy target contract for all **20,560** real UCI source rows. Original labels are preserved exactly (`0→VACANT`, `1→OCCUPIED` per active model class-map naming). Occupancy is explicitly separated from measured CO₂, derived `CO2_slope`, future model predictions, rule-based CO₂ safety state, sensor health, and multisensor risk. No scaler fitting, model training, threshold relabeling, or class balancing was performed.

---

## 2. Predecessor Gate

| Gate | Result |
|---|---|
| Canonical base | `origin/main` @ `2a4e250` |
| C-A3 PR #21 | MERGED (`fd9d21f`; tip `071177f` in main) |
| C-A0..C-A3 validators | all `PASS_WITH_WARNINGS` |
| Fresh branch | `feature/C-A4-co2-target-semantics` from updated `main` |

---

## 3. Source Occupancy Semantics

- **Source field:** `Occupancy`
- **Source values:** `{0, 1}` only
- **Meaning:** room occupancy state as defined by the original UCI Occupancy Detection dataset
- **Canonical mapping:** `0 → VACANT`, `1 → OCCUPIED` (aligned with `models/model_manifest.json` CO₂ `class_map`)
- **Label derivation:** `NONE` (identity source preservation)
- Source integer values remain authoritative alongside semantic names

---

## 4. Target Integrity

| Metric | Value |
|---|---:|
| Total rows | 20560 |
| Occupancy 0 | 15810 |
| Occupancy 1 | 4750 |
| Unexpected labels | 0 |
| Missing labels | 0 |
| Modified labels | 0 |
| Derived/reconstructed labels | 0 |

Per inherited C-A2 role:

| Role | Rows | Occ 0 | Occ 1 |
|---|---:|---:|---:|
| TRAIN | 8143 | 6414 | 1729 |
| VALIDATION | 2665 | 1693 | 972 |
| LOCKED_TEST | 9752 | 7703 | 2049 |

---

## 5. Feature / Target Role Classification

| Field | Role |
|---|---|
| Temperature | MEASURED_FEATURE |
| Humidity | MEASURED_FEATURE |
| Light | MEASURED_FEATURE |
| CO2 | MEASURED_FEATURE |
| HumidityRatio | MEASURED_FEATURE |
| CO2_slope | DERIVED_FEATURE |
| Occupancy | SOURCE_TARGET_LABEL |

Feature selection for model input remains deferred; C-A4 does not choose `[CO2_slope, Humidity, CO2]` as a feature set decision.

---

## 6. Semantic Separation

Locked invariants:

1. Occupancy==1 does **not** mean dangerous CO₂ exposure.
2. Occupancy==0 does **not** mean a safe CO₂ environment.
3. CO₂ above any safety threshold does **not** automatically mean Occupancy==1.
4. CO₂ at/below any safety threshold does **not** automatically mean Occupancy==0.
5. `CO2_slope` is a feature, not a target.
6. Occupancy model prediction is not SafeNest emergency risk.

Documented out-of-scope safety threshold (`CO2 > 1500 ppm` in active adapter/risk code) is inspected for separation only and is **not** embedded into target mapping (`DEFERRED_SAFETY_RULE_CONTRACT`).

---

## 7. Prohibitions Enforced

- Threshold-based relabeling: PROHIBITED
- Feature-driven relabeling: PROHIBITED
- Class balancing at label-definition stage: PROHIBITED
- Scaler fitting / model training: NO
- Synthetic NPZ as real-label evidence: NO
- LOCKED_TEST-driven contract design: NO
- Label smoothing/debounce/hysteresis: NO

---

## 8. Label Transition Audit

Descriptive provenance only (no label mutation). Per-block 0→1 / 1→0 transition and run-length summaries are recorded in `label_transition_audit.json`.

---

## 9. Validation

| Check | Result |
|---|---|
| C-A0..C-A3 validators | PASS_WITH_WARNINGS |
| C-A4 validator | PASS_WITH_WARNINGS |
| C-A4 focused tests | 11 passed |
| CO₂ focused suite (C-A0..C-A4) | 61 passed |
| Full `tests/` regression | 481 passed, 2 skipped, 0 failed |
| `git diff --check` | clean |

---

## 10. Parallel Git Isolation

C-A4 branch contains only CO₂ C-A4 files relative to `origin/main`. mmWave / Thermal / unauthorized shared / raw payload / adapter / scaling metadata changes: **0**.

---

## 11. Deferred Work / C-A5 Authorization

Deferred:

- C-A5 group-wise split / sample provenance materialization
- C-A6 full conversion integrity audit
- C-B model comparison / TRAIN-only scaler fit
- C-C SCD40 domain alignment
- Shared inventory updates (`DEFERRED_SHARED_INTEGRATION_UPDATE`)
- Safety/sensor-health/multisensor-risk contract ownership

C-A5 authorization: **YES**.

---

## 12. Generated Artifacts

- `datasets/co2/target_semantics.py`
- `scripts/audit_co2_target_semantics.py`
- `scripts/validate_co2_target_semantics.py`
- `tests/test_co2_target_semantics.py`
- `datasets/co2/manifests/c_a4_target_semantics/*`
- `docs/reports/20260810_Cursor_C-A4_CO2_Occupancy_Target_Semantics_01.md`
