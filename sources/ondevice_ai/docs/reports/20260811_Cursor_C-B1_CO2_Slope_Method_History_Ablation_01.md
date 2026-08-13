# C-B1 CO₂ Slope Method / History Ablation Report

Phase: **C-B1** — Controlled CO2_slope method/history ablation with fixed leakage-safe reference probe.

## Objective

Select the strongest pre-registered causal slope reconstruction candidate under one fixed B0 comparison universe, SCD40-native feature context, TRAIN-only scaler, nearest-centroid reference probe, and B0 metric contract.

## Predecessor

- C-B0 contract: `CO2_B0_OFFLINE_EXPERIMENT_CONTRACT_001`
- A-series tag: `co2-a-series-raw-to-canonical` @ `bfd860cad2bb8dafe35ef7600cfa931d7d2d554d`
- Universe: TRAIN 8140 / VALIDATION 2662 / LOCKED_TEST 9749 SEALED

## Candidate Grid (pre-registered)

| Candidate | Method | Min history |
|---|---|---|
| `ENDPOINT_H60` | ENDPOINT_DIFFERENCE | 60s |
| `ENDPOINT_H120` | ENDPOINT_DIFFERENCE | 120s |
| `ENDPOINT_H150` | ENDPOINT_DIFFERENCE | 150s |
| `LINEAR_REGRESSION_H60` | CAUSAL_LINEAR_REGRESSION | 60s |
| `LINEAR_REGRESSION_H120` | CAUSAL_LINEAR_REGRESSION | 120s |
| `LINEAR_REGRESSION_H150` | CAUSAL_LINEAR_REGRESSION | 150s |

## ENDPOINT_H150 ↔ C-A3 Parity

- Status: **PASS**
- Status mismatches: 0
- Value mismatches: 0
- Max abs difference: 0.0

## Ranking (VALIDATION macro F1 primary)

| Rank | Candidate | macro F1 | balanced acc | OCCUPIED recall |
|---|---|---|---|---|
| 1 | `ENDPOINT_H150` | 0.852366 | 0.858731 | 0.850361 |
| 2 | `LINEAR_REGRESSION_H150` | 0.851255 | 0.857845 | 0.850361 |
| 3 | `ENDPOINT_H120` | 0.848435 | 0.854895 | 0.846233 |
| 4 | `LINEAR_REGRESSION_H120` | 0.847637 | 0.854083 | 0.845201 |
| 5 | `ENDPOINT_H60` | 0.843887 | 0.850909 | 0.844169 |
| 6 | `LINEAR_REGRESSION_H60` | 0.843887 | 0.850909 | 0.844169 |

## Selected Experimental Slope Profile

- Profile: `CO2_B1_SELECTED_SLOPE_PROFILE_001`
- Winner: `ENDPOINT_H150` / ENDPOINT_DIFFERENCE / 150s
- VALIDATION macro F1: 0.852366
- VALIDATION balanced accuracy: 0.858731
- VALIDATION OCCUPIED recall: 0.850361
- Deployment status: `NOT_VALIDATED`
- Final feature selection: `NOT_PERFORMED`
- A-series profile modified: **NO** (`CO2_SLOPE_FEATURE_PROFILE_001` retained)

## Incremental Slope Evidence vs No-Slope Control

- No-slope VALIDATION macro F1: 0.844005
- Δ macro F1: 0.008361
- Δ balanced accuracy: 0.007380
- Δ OCCUPIED recall: 0.004128
- Status: `ESTABLISHED`

## Hard Boundaries Observed

- LOCKED_TEST predictions/metrics: 0
- Production scaler/model: unchanged
- Complex architecture / threshold / imbalance interventions: not performed
- Device-domain equivalence: not claimed (`DEVICE_UCI_CADENCE_DOMAIN_GAP`)

## Artifacts

Repository-relative directory: `datasets/co2/manifests/c_b1_slope_method_history_ablation/`

