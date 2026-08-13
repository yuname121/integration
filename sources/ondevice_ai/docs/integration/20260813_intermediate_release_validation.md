# 2026-08-13 Intermediate Release Integration Validation

This record distinguishes portable team-repository checks from checks that require intentionally excluded standalone data or Git history.

## Identity and scope

- Standalone source commit: `77b1695ac66fd595bd037e4574d1626b8917654c`
- Standalone prerelease: `multisensor-intermediate-2026-08-13`
- Team base commit: `f3bd342eabcad27dc2c3ecdc16f035b8b13cb153`
- Destination: `ondevice_ai/`
- Team device, shared-contract, root workflow, and firmware files changed: 0

## Focused results

- Python compile check over `ondevice_ai/`: PASS.
- Git whitespace/error check: 20 inherited source warnings (18 Markdown hard-break/EOF warnings and 1 test-line warning, as reported by Git). The transferred release files remain byte-identical to the source commit, so these were reported rather than silently rewritten during integration; the six new integration-record files pass the check.
- CO₂ C-B5 standalone validator in the team component: `PASS_WITH_WARNINGS`. Preserved warnings are INT8 input saturation, Mac-host-only latency evidence, and pending SCD40 validation.
- Thermal T-A6 full-dataset compact-evidence validator in the team component: `PASS_WITH_LIMITATIONS`; `t_b_authorized=false` remains enforced.
- Thermal T-A6 Stage-2 bundle validator: PASS with zero errors and zero warnings.
- mmWave M-B12 validator against the exact standalone release commit with the ignored local raw archive available: PASS. It reports `Phase_B_release_ready=false`, zero new locked-test/recovery access, and the documented non-pristine holdout limitation.
- mmWave M-B12 validator in the team checkout without raw archives: expected fail-closed result `RAW_ARCHIVE_MISSING`. No raw archive was copied to bypass this boundary.

## Complete component test run

The complete `ondevice_ai/tests` pytest collection was run from the team repository root:

- Passed: 1,081
- Failed: 211
- Skipped: 6
- Warnings: 99

The 211 failures are environment/boundary-accounted rather than hidden:

- 196 tests require the intentionally excluded raw mmWave or CO₂ archives, or the standalone-only `co2-a-series-raw-to-canonical` Git tag. The affected files comprise 28 CO₂ tests and 168 mmWave tests; many negative mutation tests fail early at the same fail-closed prerequisite gate.
- 15 tests are preserved team-only V4/V5 legacy checks (`test_v4_config_validation.py` and `test_v5_release.py`) that target historical wrapper/archive behavior and already conflicted with the prior multisensor synchronization.

No affected `devices/<device>/tests` were run because this branch changes no `devices/`, `shared/contracts/`, firmware, calibration, or team sensor-threshold file. Physical MR60, SCD40, Thermal-device, Raspberry Pi, and multisensor integration validation remain unverified and are not claimed by this synchronization.
