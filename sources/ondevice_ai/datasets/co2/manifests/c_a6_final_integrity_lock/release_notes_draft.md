# CO₂ A-Series Release Notes Draft (Raw-to-Canonical)

> Draft only. Do **not** publish a GitHub Release from the C-A6 feature branch.
> Tag/release target policy: exact C-A6 merge commit on canonical `main`.

## Milestone

CO₂ real raw-to-canonical reconstruction milestone complete (C-A0 through C-A6).

Proposed tag: `co2-a-series-raw-to-canonical`

## Source identity

- Dataset: UCI Occupancy Detection (UCI Dataset ID 357)
- UCI dataset DOI: `10.24432/C5X01N`
- Journal publication DOI (separate): `10.1016/j.enbuild.2015.11.071`
- License: CC-BY-4.0 (verified)
- Raw archive path: `datasets/raw_archives/external_datasets/occupancy+detection.zip`
- Raw archive size: 335713 bytes
- Raw archive SHA-256: `4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a`
- Raw payload: **not** included in Git release; materialize separately under the approved provenance contract.

## Reconstruction counts

- Source rows / canonical source samples: 20560
- Model-eligible (CO2_slope available): 20551
- Warm-up preserved: 9
- Temporal blocks: BLOCK_01_DATATEST, BLOCK_02_DATATRAINING, BLOCK_03_DATATEST2
- Split roles: TRAIN / VALIDATION / LOCKED_TEST (immutable C-A2 block assignment)
- Target: Occupancy 0=VACANT (15810), 1=OCCUPIED (4750); derivation NONE
- Slope profile: `CO2_SLOPE_FEATURE_PROFILE_001` / ENDPOINT_DIFFERENCE / ppm/min

## Artifact lock

C-A0..C-A6 machine-readable artifacts are checksum-locked under
`datasets/co2/manifests/c_a6_final_integrity_lock/`.

## Explicit non-claims

This release does **not** mean:

- CO₂ model real-data validated
- CO₂ model deployment-ready
- SCD40 device validated
- CO₂ safety thresholds calibrated
- Multisensor integration validated
- Raspberry Pi performance validated

Existing model/scaler lineage remains `MODEL_TRAINING_LINEAGE_UNVERIFIED` /
`SCALER_FIT_LINEAGE_UNVERIFIED` / `CONFIRMED_SYNTHETIC_ONLY` where applicable.

## Major limitations retained

- SOURCE_TIMEZONE_UNVERIFIED
- GROUP_INDEPENDENCE_NOT_VERIFIABLE
- CO2_SLOPE_HISTORY_LINEAGE_UNVERIFIED
- DEVICE_UCI_CADENCE_DOMAIN_GAP
- SAFETY_RULE_CONTRACT_OUT_OF_SCOPE
- SENSOR_HEALTH_CONTRACT_OUT_OF_SCOPE
- MULTISENSOR_RISK_CONTRACT_OUT_OF_SCOPE

## Next phase

After tag/release on the exact C-A6 merge commit: begin C-B0 offline real-data model comparison against this locked baseline.
