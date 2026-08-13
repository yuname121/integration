# SafeNest On-Device AI Component Instructions

## Canonical component root

- This file lives at `ondevice_ai/AGENTS.md` inside the team repository `safenest-embedded-competition`.
- The directory containing this `AGENTS.md` (`ondevice_ai/`) is the only active AI component root.
- Active AI code, configuration, datasets, models, tests, manifests, and reports live directly under `ondevice_ai/`.
- Do not create `embed2/`, `SafeNest_V4_*`, `SafeNest_V5_*`, `SafeNest_V6/`, or `ondevice_ai/ondevice_ai/` as alternate active roots.
- Version names belong in model metadata, release tags, reports, and archived snapshot names, not around the active source tree.

## Team responsibility boundaries

- `devices/<device>/src/`: team-owned real hardware drivers and acquisition.
- `shared/contracts/`: public cross-domain sensor interfaces.
- `ondevice_ai/`: AI preprocessing, inference, model assets, dataset contracts, risk logic, mocks/replay, validators, and reports.
- Do not modify `devices/`, `shared/contracts/`, root `.github/`, or team hardware/firmware thresholds from AI-only work unless separately authorized.

## Archive boundary

- Historical snapshots are not a runtime fallback.
- Never import code, auto-discover manifests, or resolve runtime models from archived snapshot trees.
- Do not edit archived reports or snapshots to make them look current.
- A historical model needed for an active comparison may remain under `models/` only when its lineage and role are explicit in `models/model_manifest.json`.

## Path and provenance rules

- Store repository-relative POSIX paths in JSON, YAML, manifests, metadata, and generated reports.
- Paths inside this component are relative to `ondevice_ai/` unless a team-wide contract explicitly requires repository-root-relative paths.
- Do not persist `/Users/...`, `file://...`, home-relative, drive-specific, or version-wrapper paths in active machine-readable artifacts.
- Runtime path resolution starts from `ondevice_ai/` and must not fall back to a versioned sibling or archived snapshot.
- Every generated dataset sample must preserve source dataset, subject, session, recording, time/window, extraction profile, label mapping, split, and quality provenance when applicable.

## Multisensor phase workflow

- Follow `docs/20260810_ChatGPT_SafeNest_Multisensor_Parallel_Execution_Roadmap_01.md` as the active master roadmap.
- Use `docs/20260806_ChatGPT_SafeNest_mmWave_Execution_Sequence_01.md` and `docs/MMWAVE_PHASE_B_OVERVIEW.md` for inherited mmWave details.
- Run mmWave M-B, CO₂ C-A, Thermal T-A, and integration contract inventory I-0 in parallel when their files and evidence are independent; preserve the required phase order inside each sensor track.
- For each sensor track, complete and validate its A0 through A6 before starting that sensor's Phase B model selection.
- A4 voluntary breath-hold labels are derived SafeNest APNEA proxies and must never be described as clinical apnea.
- A5 uses subject-level grouping. All recordings and windows from one subject must remain in exactly one split.
- `AMBIGUOUS` A4 windows are excluded from pure-class training but retained for provenance and transition analysis.
- A6 must extend the approved pilot contracts to all 440 recordings and must report every failure, exclusion, and low-quality result.
- A6 annotation evidence must never fail open. An annotation read/parse failure blocks that recording and must appear in the exception registry; optional reference-sensor failures must be recorded as warnings.
- The A6 exit gate must validate semantic 1:1 correspondence among every window row, provenance row, and canonical numeric row, successful accounting for every A0 recording, and complete checksum coverage.
- Phase B must inherit the immutable A5 subject split, fit preprocessing statistics on TRAIN only, keep LOCKED_TEST unavailable to model selection, and run a near-duplicate diagnostic before comparative evaluation.

## Change and verification discipline

- Preserve user changes and existing A0-A4 artifacts.
- Generated artifacts must be deterministic for the same inputs and configuration; record checksums with repository-relative paths.
- Run the focused phase validator and upstream regression tests after each phase.
- Do not describe an A phase as complete solely because generation finished; the standalone phase validator must pass against the generated evidence.
- Mock success proves software wiring only unless the result is derived exclusively from the actual selected model prediction.
- Do not claim MR60 real-sensor validation, Raspberry Pi performance, or clinical performance without corresponding measurements.

## Fail-closed semantics

- Missing, invalid, stale, NaN, or unavailable device/sensor data must not become a synthetic normal value.
- Real mode without an injected team provider must report `valid=false` and fail closed.

## Current synced development status (source SHA `77b1695`)

- mmWave: M-A0..M-A6 and M-B0..M-B12 completed with recorded warnings and a non-pristine final-evaluation limitation. The locked artifact is an offline candidate only. MR60 and Raspberry Pi validation remain pending.
- CO₂: C-A0..C-A6 and C-B0..C-B5 completed. The locked UCI occupancy artifact is an offline candidate only. SCD40 device-domain validation remains pending.
- Thermal: T-A0..T-A6 completed with recorded data and evaluation limitations. T-B is not authorized (`t_b_authorized=false`), so there is no new Thermal model candidate. Thermal device-domain validation remains pending.
- The synchronized checkpoint is the standalone prerelease `multisensor-intermediate-2026-08-13`; it is not production, physical-sensor, Raspberry Pi, clinical, safety, or multisensor-integration approval.
