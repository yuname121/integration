# ondevice_ai Integration Source Snapshot

- Standalone source repository: `https://github.com/sheepmeat/test`
- Standalone source SHA: `77b1695ac66fd595bd037e4574d1626b8917654c`
- Standalone prerelease: `multisensor-intermediate-2026-08-13`
- Team repository: `https://github.com/jinsu1011/safenest-embedded-competition`
- Team base SHA: `f3bd342eabcad27dc2c3ecdc16f035b8b13cb153`
- Destination: `ondevice_ai/`
- Branch: `feature/ondevice-ai-multisensor-intermediate-release`

## Included phase state

- mmWave: M-A0..M-A6, M-B0..M-B12
- CO₂: C-A0..C-A6, C-B0..C-B5
- Thermal: T-A0..T-A6; T-B remains unauthorized

## Decision artifacts

- `20260813_intermediate_release_collision_matrix.json`
- `20260813_intermediate_release_apply_plan.json`
- `20260813_intermediate_release_validation.md`
- `collision_summary.md`

## Intentionally excluded

- `.git/`, `.github/`, `archive/`, `hardware/`, `releases/`, `repro_test_dir/`
- `datasets/raw_archives/` and ignored raw/thermal payloads
- local caches, credentials, absolute-path metadata

## Incremental basis

- Previous synchronized standalone SHA: `9a66a3b21baef9a6a51cb1a66942284c63d0b8a4`
- The transfer contains only the 489 Git-tracked files added or modified after that SHA.
- The earlier `collision_matrix.json` and `apply_plan.json` remain as evidence of the initial component import.
