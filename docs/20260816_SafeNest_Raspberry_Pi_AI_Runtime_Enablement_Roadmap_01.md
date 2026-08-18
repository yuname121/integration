# SafeNest Raspberry Pi AI Runtime Enablement Roadmap

**Document date:** 2026-08-18
**Document ID:** `RP-AI-ENABLEMENT-ROADMAP-01`
**한국어 요약본:** [`20260816_SafeNest_Raspberry_Pi_AI_Runtime_Enablement_Roadmap_01_KO.md`](20260816_SafeNest_Raspberry_Pi_AI_Runtime_Enablement_Roadmap_01_KO.md)
**Roadmap status:** `RP-A1_IMPLEMENTED_UNDER_INDEPENDENT_REVIEW / RP-X0_FIELD_STATE_CORRECTIVE`
**Status meaning:** RP-A0 audit/design is the documentation baseline. RP-A1 is implemented and under independent review, but RP-X0 does **not** authorize its merge. Later RP-A2+/RP-B/RP-C/RP-D phases and B-complete production activation remain unauthorized by this document.

This roadmap describes how the Raspberry Pi integration runtime must be improved so that the team B-complete offline AI candidates can eventually be used correctly with real sensor evidence. It does not implement Capture, change ESP32 firmware, retrain models, change frozen preprocessing, change class maps, change risk thresholds, or change dashboard risk-decision, alarm, or operational behavior.

Evidence tags used below:

| Tag | Meaning |
|---|---|
| `CODE_VERIFIED` | Confirmed from current integration source |
| `TEST_VERIFIED` | Confirmed from current integration tests |
| `DOCUMENTED_ONLY` | Present in documents/manifests, not re-executed here |
| `OWNER_REPORTED` | Stated by team PR/handoff, not independently re-run |
| `OBSERVED` | Directly recorded in a named field report with its runtime and topology identified |
| `INFERRED` | Reasonable from adjacent code, labeled as such |
| `UNVERIFIED_HYPOTHESIS` | A possible explanation that has not been established by code or field evidence |
| `PLANNED` | Proposed future architecture, not current code |
| `BLOCKED_HARDWARE` | Requires physical device measurement |
| `BLOCKED_DEPENDENCY` | Requires an external contract, artifact, or owner decision |

Proposed architecture is marked `PLANNED`. Do not treat it as existing code.

---

## 0. 2026-08-17 RP-X0 Field-State Corrective

This dated corrective is authoritative for **field runtime status and near-term
ordering**. Sections that describe the 2026-08-16 source audit remain useful
historical code evidence. Development authority is fixed below; do not silently
combine evidence from distinct repositories, Git SHAs, processes, ESP boots, or
network topologies.

### 0.1 Governance and authorization remain unchanged

```text
RP-X0 = OUT_OF_BAND_DIAGNOSTIC_TASK

RP-A0 = COMPLETE
RP-A1 = IMPLEMENTED / UNDER INDEPENDENT REVIEW
RP-A1 merge = NOT AUTHORIZED by RP-X0
RP-A2 = NOT AUTHORIZED
RP-A3+ = NOT AUTHORIZED

MMWAVE_B_LIVE_GATE = CLOSED
```

RP-X0 persistence and B-runtime diagnostics are not normal Capture delivery.
They do not authorize RP-A2, do not merge RP-A1, do not select a production
model, and do not turn a smoke/soak result into Phase C validation.

```text
RP-X0 Step 3.x / Stage 7–12
DO NOT map one-to-one to RP-A/RP-B normal phases.

RP-X0 persistence work != RP-A2
RP-X0 B runtime != normal RP-B complete
RP-X0 real hardware smoke != Phase C validation
```

### 0.2 Development authority is fixed; current runtime state is operational

Two repository/runtime contexts were observed on 2026-08-17. Their difference
requires explicit evidence attribution, but it is **not** a repository-authority
election.

| Context | Repository / identity | Meaning | Evidence |
|---|---|---|---|
| Active RP-X0 development and field-work authority | `yuname121/integration`; frozen reference `/home/sandi/integration`, `diagnostic/rp-x0-b-runtime-wiring` @ `1ffbc7d39792e68edc552fbe08359732b0dcbefd` | Pi integration development, technical validation, and RP-X0 field work continue here | `OBSERVED` / `OWNER_REPORTED` |
| Currently running team backend, 18:00 KST | `/home/sandi/safenest-embedded-competition`, backend PID `1722`, started about 17:52 KST | Temporary operational owner of the active Pi backend/ports while another teammate is using them; later forward-port target, not a change of development authority | `OBSERVED` |

```text
CURRENTLY_RUNNING_TEAM_BACKEND
!=
CHANGE_OF_DEVELOPMENT_AUTHORITY
```

The fixed responsibility split is:

```text
RP_X0_ACTIVE_DEVELOPMENT_AUTHORITY
= yuname121/integration

AI_ARTIFACT_AUTHORITY
= sheepmeat/test (embed2)

TEAM_FORWARD_PORT_TARGET
= jinsu1011/safenest-embedded-competition
```

Do not interrupt a teammate's active Pi backend merely to restore integration.
Pi absence or Pi-port ownership does **not** pause active integration
development. The frozen snapshot stays evidence only; current implementation
continues in `yuname121/integration` on Mac whenever the next validation
boundary is not hardware-specific. The team tree is compared and forward-ported
only after verified integration behavior is ready for handoff.

```text
MAC_FIRST_DEVELOPMENT_POLICY

MAC_OFFLINE_VALIDATED
!=
PI_RUNTIME_VALIDATED
!=
REAL_SENSOR_VALIDATED
```

Use these dependency tags as reusable status vocabulary, not as phases or
gates:

| Tag | Meaning |
|---|---|
| `MAC_OFFLINE_READY` | Implementation or meaningful validation can proceed now in active integration using Mac-native tooling, deterministic fixtures, or read-only snapshot replay. |
| `SENSOR_REQUIRED` | The next evidence specifically needs a physical sensor. |
| `PI_REQUIRED` | The next evidence specifically needs Pi hardware, OS, ARM runtime, or Pi I/O. |
| `SENSOR_AND_PI_REQUIRED` | The next evidence needs a real sensor in the actual Pi deployment topology. |
| `EXTERNAL_AI_DEPENDENCY` | Integration awaits an approved model/data contract from `sheepmeat/test`. |
| `MAC_OFFLINE_FIX_REQUIRED` | A small Mac-only tooling/contract corrective is still required before the hardware boundary. This is not Pi execution and not a new RP/O stage. |

`PI_AVAILABLE` and `SENSOR_AVAILABLE` are independent conditions. A Mac plus
the snapshot can continue replay and integration work; a Pi without MI48 still
cannot produce new MI48 device-domain data; a legitimate sensor-acquisition
topology may satisfy a sensor boundary without being final Pi deployment.

```text
STAGE_7_OFFLINE_PREPARATION = MAC_OFFLINE_READY
STAGE_7_PI_EXECUTION = PI_REQUIRED
STAGE_8_PRIORITY = IMMEDIATE_NON_DISRUPTIVE
```

### 0.3 Mandatory evidence scope rule

```text
REAL_HARDWARE_EVIDENCE_IS_RUNTIME_AND_TOPOLOGY_SCOPED
```

Every future real-device result must identify at least:

```text
Pi repository
Pi Git SHA
backend PID / startup
B-mode or production mode
ESP device_id
ESP boot_id
ESP firmware version
ESP peer IP where relevant
collection time window
```

Evidence from one topology/runtime is historical evidence for another only; it
does not automatically validate a different checkout or deployment tree.

### 0.4 RP-X0 completed preparation and artifact state

| Area | Dated status | Evidence |
|---|---|---|
| Pi AI environment | LiteRT and required Pi AI environment verified: `COMPLETE` | `OBSERVED` |
| Operational persistence | CO2/mmWave JSONL and Thermal NPZ paths verified: `COMPLETE_FOR_RP_X0_DIAGNOSTICS`; this is **not** Capture v1/RP-A2 completion | `OBSERVED` |
| CO2 C-B6 | `C_B6_REDUCED_CO2_SLOPE_CANDIDATE_001`, SHA `c5969b367f5b5e28c4d27f1bdd6220f7d02da92e99604e3f85c9f1291e98dd3b` available/provisioned | `OBSERVED` |
| mmWave B artifact | `M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120`, SHA `6dff6aaa72c79d76715d40cf7e32bb1e6cd9b2c2e3ac78eaf2fda737561430c5` available/provisioned | `OBSERVED` |
| Thermal T-B5 | `SMALL_CNN_BASELINE_V1_P1_full_int8.tflite`, SHA `fa9730c29535477a3994c11e664474a0ca0116afaaa172889f47446ab2ac46be`; isolated LiteRT load/invoke passed | `OBSERVED` |

The selected artifacts are available, while historical v0.1.0 artifacts remain
preserved and production `model_manifest.json` has not been switched.

```text
ARTIFACT_AVAILABLE
!=
PRODUCTION_SELECTED
!=
LIVE_DEVICE_VALIDATED
```

Integration synchronization is a separate repository action:

```text
branch: chore/sync-latest-b-stage-models
SHA: ddd8150
PR: integration PR #8
```

It synchronizes artifact availability; it does not select a production model or
validate a live device.

### 0.5 Scoped real-hardware observations

For approximately 2026-08-17 17:04–17:32 KST, the integration RP-X0 reference
runtime recorded concurrent real sensor operation. The relevant ESP boot was
`89c5b9bd...`; observed behavior included roughly 10 Hz scalar/mmWave telemetry,
about 6 FPS Thermal UDP, active CO2 physical-event provenance, persisted
mmWave `breath_phase`, and logger drops of zero. The earlier Thermal UDP
disappearance around 16:36 was not reproduced in this later window.

```text
RP_X0_REAL_CONCURRENT_SENSOR_SOAK = PASS_WITH_LIMITATIONS
```

This is not formal device-domain validation. It is scoped to the integration
B-runtime, its `1ffbc7d...` lineage, the named ESP boot, and that network
topology. It does not validate the temporarily running team backend plus ESP
peer `192.168.137.69`; that distinction is evidence scoping, not an authority
decision.

CO2 source fields `boot_id`, `measurement_event_id`, and
`measurement_monotonic_ms` were observed. C-B6 was observed through physical
provenance, warm-up, valid 150-second history, `CO2_slope`, and LiteRT output.

```text
MODEL_OUTPUT_OBSERVED
!=
GROUND_TRUTH_PERFORMANCE_VALIDATED
!=
SAFETY_DECISION_CERTIFIED
```

CO2 is the only B-stage model with real RP-X0 sensor input observed at LiteRT.

Real mmWave fields included `breath_phase`, `phase_age_ms`,
`ts_monotonic_ms`, and nested mmWave sequence. They varied across samples and
must not be reinterpreted as a BPM-derived fake waveform. Stage 8/8.5 later
reviewed cadence/window evidence and source semantics; the resulting
`MMWAVE_SIGNAL_CONTRACT = MISMATCH` supersedes the prior pending phase-review
state. The historical B live gate remains
closed because vendor `breathPhase` is not the frozen complex-range-bin phase
signal, not because more resampling is pending.

For Thermal, the following is a historical observation from this dated field
runtime, not the current physical-conversion status:

```text
THERMAL_B_ARTIFACT_READY = YES
THERMAL_B_COMPUTE_READY = YES
THERMAL_LIVE_B_INPUT_COMPATIBILITY = NO
THERMAL44_DEPLOYMENT_VALIDATED = NO
```

At that time the live MI48/Thermal-44 frame was known only as `uint16` raw
while P1 expects Celsius. This conversion uncertainty was superseded for the
named snapshot by O1/O2: `physical_C = raw_uint16 / 10.0 - 273.15`, followed
by verified P1 replay. It does not authorize live `HUMAN_FALL`; that remains a
`LYING`-derived posture proxy, not a verified temporal fall event.

### 0.6 Corrected near-term RP-X0 diagnostic order

| Stage | Status | Required outcome |
|---|---|---|
| 7 — Integration RP-X0 Runtime Continuation / Restore | Offline preparation: `IMPLEMENTED / MERGED` (PR #20); preflight mmWave selector drift: `RESOLVED_IN_CODE`; actual Pi execution: `PI_REQUIRED / NOT_RUN` | Mac-offline runtime wiring, configuration, parser/backend/API, status/failure isolation, replay/static checks, and deployment preflight are merged. Stage 7 preflight asserts the PR #22 M-N9 FULL_INT8 selector (`deployment_allowed=true`), keeps historical B inactive, and does not treat that as Pi/device validation. Deploy a verified commit and verify actual Pi processes/ports only when Pi access is available; do not develop from the frozen snapshot. |
| 8 — mmWave Real-Phase Offline Cadence / Window Audit | `PASS_WITH_LIMITATIONS` | Completed on `20260817_08_mmwave.jsonl`; Stage 8.5 found `MMWAVE_SIGNAL_CONTRACT = MISMATCH`. Keep the historical B gate closed and move MR60-native model work to the AI authority track. |
| 9 — Minimal Post-Deployment Live Smoke | Tooling preparation: `IMPLEMENTED / MERGED` (PR #21); live smoke: `SENSOR_AND_PI_REQUIRED / NOT_RUN` | Runner, probes, evaluator, structured report, and Mac-offline fixture tests are merged. Execute backend/health, TCP `:9000`, UDP `:5005`, ESP connection, increasing CO2/Thermal/mmWave/PIR records, expected AI statuses, and no new unexpected logger-drop condition only in the live topology. Do not repeat a 30-minute soak without a regression reason. Do not mark Stage 9 globally complete from tooling. |
| 10 — Thermal Physical-Domain Contract | `COMPLETE_FOR_NAMED_SNAPSHOT / EXTERNAL_AI_DEPENDENCY` | O1/O2 established MI48 `uint16` → Celsius → `P1_TRAIN_FITTED_GLOBAL_ZSCORE` → INT8 T-B5 replay for the named snapshot. The current next path is `TRAIN_DOMAIN_RANGE_GAP` in Thermal AI work; do not activate T-B5. |
| 11 — Further Live-B Gate Reviews | `EVIDENCE_ONLY` | Review any future mmWave path only after MR60-native model handoff and sufficient targeted evidence. The historical `MMWAVE_B_LIVE_GATE` remains closed; no authorization can make its mismatched signal contract valid. |
| 12 — Repository / Team Handoff | `LATER / AFTER_VERIFIED_INTEGRATION_EVIDENCE` | Keep integration PR #8, embed2 locked-binary policy work, RP-X0 diagnostic docs/tools, and a forward-port of verified integration changes into the team tree as distinct artifacts. |

### 0.7 Required current-status summary

```text
2026-08-18 RP-X0 STATUS

Active Pi integration development authority:
yuname121/integration

AI/model authority:
sheepmeat/test

Future team forward-port target:
jinsu1011/safenest-embedded-competition

Pi compute/storage preparation:
READY

Locked B artifacts:
CO2 READY
mmWave READY
Thermal READY

Real B inference:
CO2 OBSERVED
mmWave status projection: MODEL_PENDING / MR60_NATIVE_MODEL_PENDING
mmWave primary selector on main: M-N9 FULL_INT8 (PR #22); not historical B
mmWave live-sensor AI validation: NOT_RUN
Thermal BLOCKED BY INT8_QUANTIZATION_REVIEW_REQUIRED
Thermal next AI validation path: TRAIN_DOMAIN_RANGE_GAP

Real sensor persistence:
CO2 OBSERVED
PIR OBSERVED
mmWave breath_phase OBSERVED
Thermal OBSERVED

30-minute concurrent sensor soak:
PASS_WITH_LIMITATIONS

Immediate non-disruptive next work:
O3 runtime-status cleanup IMPLEMENTED / MERGED (integration PR #17)
O4 LCD/Web availability-status alignment IMPLEMENTED / MERGED (integration PR #19)
Stage 7 offline preparation IMPLEMENTED / MERGED (PR #20)
Stage 9 tooling preparation IMPLEMENTED / MERGED (PR #21)
STAGE7_PREFLIGHT_MMWAVE_SELECTOR_DRIFT = RESOLVED_IN_CODE

mmWave offline phase audit:
PASS_WITH_LIMITATIONS — SIGNAL_CONTRACT_MISMATCH

Stage 7 offline preparation:
IMPLEMENTED / MERGED (PR #20)

Stage 7 preflight mmWave selector drift:
RESOLVED_IN_CODE

Stage 7 actual Pi deployment/execution:
PI_REQUIRED / NOT_RUN

Stage 9 tooling preparation:
IMPLEMENTED / MERGED (PR #21)

Stage 9 live smoke:
SENSOR_AND_PI_REQUIRED / NOT_RUN

Repository consolidation:
LATER
```

The current actionable posture is:

| Work now | Dependency boundary | Explicitly not claimed |
|---|---|---|
| Remaining Mac-offline RP-X0 Stage 7/9 tooling, including M-N9 preflight selector contract | `IMPLEMENTED / MERGED` (PRs #20/#21) / `RESOLVED_IN_CODE` | Pi-runtime or new live-sensor validation |
| Thermal new labeled/device-domain evidence | `SENSOR_REQUIRED` / `EXTERNAL_AI_DEPENDENCY` | T-B5 activation |
| Stage 7 Pi deployment and process/port/ARM verification | `PI_REQUIRED` | Completion from Mac-only work |
| Stage 9 real TCP/UDP sensor smoke | `SENSOR_AND_PI_REQUIRED` where applicable | Completion from tooling preparation |
| MR60-native replacement model | `EXTERNAL_AI_DEPENDENCY` | Reopening the old mmWave B gate |

```text
Further Mac RP-X0 implementation required:
NO

FUTURE_OPERATOR_CAN_EXECUTE_WITHOUT_CHAT_HISTORY:
YES for Mac-offline RP-X0 integration work

CURRENT EFFECT:
PR #22 active M-N9 selector = authoritative
historical B = inactive
O3 status projection = MODEL_PENDING
Stage 7 preflight asserts M-N9 selector identity
deployment_allowed=true is not device validation
DEVICE_VALIDATED = false
PI_SMOKE = NOT_PERFORMED
PRESENCE_GATE_REQUIRED = true

Remaining boundary:
Stage 7 actual Pi execution = PI_REQUIRED / NOT_RUN
Stage 9 live smoke = SENSOR_AND_PI_REQUIRED / NOT_RUN
```

---

### 0.8 2026-08-17 RP-X0 Offline Development / Snapshot Evidence Revision

This corrective preserves the completed RP-X0 field result while moving all
remaining work into a safe Mac-offline workflow. It does not authorize runtime,
firmware, model, Capture, or deployment changes.

#### 0.8.1 Three distinct roles

```text
Pi field snapshot
= frozen evidence / replay source
= READ_ONLY

yuname121/integration
= active Raspberry Pi integration development authority

Raspberry Pi
= later deployment / minimal live-smoke target
```

The canonical development flow is:

```text
Pi field evidence
  → frozen Mac snapshot
  → read-only offline replay / analysis
  → active development in safenest-integration
  → Git-reviewed integration commit
  → later Pi deployment
  → short real-sensor smoke only
```

`yuname121/integration` is authoritative for Pi receivers/runtime, backend, AI
runtime adapters, offline replay utilities, runtime status/model selection,
LCD/Web integration, sensor-persistence integration, and later Pi deployment.
`sheepmeat/test` remains authoritative for datasets, preprocessing, training,
model selection/artifacts, quantization, AI lineage, and device-native mmWave
retraining. `jinsu1011/safenest-embedded-competition` remains a later
forward-port target for verified integration work, not a current authority.

#### 0.8.2 Pi snapshot preservation policy

```text
RP_X0_FIELD_EVIDENCE_SNAPSHOT
= safenest-pi-integration-snapshot

SNAPSHOT_ROLE
= READ_ONLY_FIELD_EVIDENCE
```

Canonical snapshot path:

```text
/Users/junwoo/Library/Mobile Documents/com~apple~CloudDocs/대학/2026/safenest-pi-integration-snapshot
```

The approximately 682 MB `FIELD_FREEZE_SNAPSHOT` contains about 395 MB of
Pi `.venv`, 160 MB of data, and 108 MB of `sources`. It preserves the exact
field state, real measurements, offline replay/forensic inputs, and reference
for dirty Pi-only changes. It is **not** a new canonical repository, a branch
to continue from, a replacement for `safenest-integration`, or a source to
push directly to GitHub.

Snapshot Git identity is `diagnostic/rp-x0-b-runtime-wiring` at `1ffbc7d`,
dated 2026-08-16 22:21 KST, with remote
`https://github.com/yuname121/integration.git`. Its field-local dirty state
includes LCD/display changes, staged T-B5 files, removal of
`MISSING_FULL_INT8.txt`, a real-collection watch script, and stale runtime
selection metadata. Do not normalize, reset, clean, commit, or otherwise
modify the snapshot during roadmap work.

```text
SNAPSHOT_CODE = FORENSIC_REFERENCE
ACTIVE_INTEGRATION_CODE = DEVELOPMENT_SOURCE_OF_TRUTH
```

The Pi snapshot `.venv` is an aarch64 Raspberry Pi environment and must not be
executed on the Mac. Mac offline analysis uses a separate Mac-native Python
environment and reads snapshot data only.

#### 0.8.3 Firmware and evidence provenance

```text
RUNTIME_EVIDENCE_IS_SCOPED_TO_ACTUAL_RUNTIME_AND_FIRMWARE_PROVENANCE
```

The ESP `.ino` frozen in the snapshot shows approximately
`TELEMETRY_PERIOD_MS = 1000` and does not expose nested `breath_phase`, while
newer field JSONL identifies firmware `safenest-esp32-sensor-node/1.2.0`,
schema `1.2`, and nested mmWave `breath_phase`, `breath_rate_raw`,
`phase_age_ms`, `ts_monotonic_ms`, and `seq`. Do not infer the live ESP
firmware implementation solely from source frozen in the Pi snapshot.

This extends, rather than replaces,
`REAL_HARDWARE_EVIDENCE_IS_RUNTIME_AND_TOPOLOGY_SCOPED`: every observation
must retain its actual runtime, firmware, ESP boot, and collection-window
provenance. JSONL/NPZ is real field evidence; SQLite is operational summary
state (`sensor_snapshots`, `risk_events`) and is not Capture v1.

#### 0.8.4 Frozen evidence inventory

| Stream | Frozen evidence | Roadmap use / limitation |
|---|---|---|
| mmWave | 7 JSONL files, about 19 MB / 44,975 records. Phase-domain evidence is only `20260817_08_mmwave.jsonl` and `20260817_09_mmwave.jsonl`. The Stage 8 source `20260817_08_mmwave.jsonl` SHA-256 is `0d31bfa7a7e86e3fa03a0421534c96a338e52499c90003314ee71266c0a40b75`. | `MMWAVE_OFFLINE_PHASE_EVIDENCE = 20260817_08 + 20260817_09 only`; earlier files are BPM/PIR telemetry, not phase-domain input. |
| CO2 | About 5 JSONL files / 949 records with `co2_measurement_event_id`, `co2_measurement_monotonic_ms`, `co2_measurement_event_valid`, `device_id`, and `boot_id`. | Real C-B6 LiteRT invocation is already established; retain the frozen `[CO2, CO2_slope]` contract and regression protection rather than repeat broad validation. |
| Thermal | 1,979 NPZ files / about 139 MB from approximately 15:55–18:31 KST; `uint16` `(N, 62, 80)` frames with timing, sequence, source uptime, min/max raw, and analysis metadata. | The historical conversion gap was superseded for offline O2 replay by `physical_C = raw_uint16 / 10.0 - 273.15` (PR #14). O2.5/O2.6 leave `INT8_QUANTIZATION_REVIEW_REQUIRED`; T-B5 stays inactive. |
| SQLite | `data/safenest.db`, including operational snapshots/events. | Operational state/summary only; do not redefine it as Capture v1. |

#### 0.8.5 Completed mmWave evidence stages and strategic transition

```text
RP_X0_STAGE_8
= PASS_WITH_LIMITATIONS

RP_X0_STAGE_8_5
= PASS_WITH_LIMITATIONS
```

Stage 8 observed real numeric `breath_phase` variation, no material stale
republication, Pi JSON receive median about 100 ms, phase-update timestamp
median about 120 ms, and 300 real samples in a median about 32.6 s. The
resampling hypothesis is superseded by Stage 8.5 source semantics:

```text
Frozen B training signal
= Zenodo range FFT → complex range-bin → angle → unwrap → radians → BPF_ZSCORE

Real MR60 runtime signal
= MR60 UART 0x0A13 → vendor breathPhase float32 → ESP → Pi

MMWAVE_SIGNAL_CONTRACT
= MISMATCH
```

No SafeNest-side hidden scaling or documented deterministic conversion from
vendor `breathPhase` to unwrapped complex range-bin radians was found. Normal
UART exposes processed vendor phase; practical documented raw ADC/IQ/range-FFT
access is unavailable. Possible RF-ADC SPI access lacks an enable protocol,
framing, ADC format, channel order, and open vendor algorithm, so it is not a
current Pi-development dependency. A vendor inquiry may be a separate future
option.

The historical locked artifact
`M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120` remains valid
offline evidence, but is:

```text
VALID_OFFLINE_B_ARTIFACT
NOT_LIVE_MR60_COMPATIBLE

MMWAVE_B_LIVE_GATE
= CLOSED

MODEL_PENDING_MR60_NATIVE_RETRAINING
```

Do not resample vendor phase to fit the old model, apply empirical amplitude
factors, per-window normalization, or open the old live gate. MR60-native
dataset, preprocessing, retraining/model selection, quantization, and lineage
belong to `sheepmeat/test`; integration receives a newly approved artifact and
complete contract only after that work is handed off.

#### 0.8.6 Sensor availability and runtime-status semantics

mmWave remains an available sensor and telemetry source for presence, distance,
movement, respiration/heart rate, `breath_phase`, `phase_age_ms`, and timing
provenance. Only its AI inference is disabled:

```text
mmWave sensor_status = AVAILABLE
mmWave telemetry_status = AVAILABLE
mmWave ai_status = MODEL_PENDING
mmWave blocked_reason = MR60_NATIVE_MODEL_PENDING

Thermal sensor_status = AVAILABLE
Thermal artifact_status = PRESENT
Thermal ai_status = BLOCKED
Thermal blocked_reason = INT8_QUANTIZATION_REVIEW_REQUIRED

CO2 sensor_status = AVAILABLE
CO2 artifact_status = PRESENT
CO2 ai_status = ACTIVE

PIR sensor_status = AVAILABLE
PIR ai_status = NOT_APPLICABLE
```

Integration PR #17 implemented the existing backend/API status metadata. It
independently represents sensor
connectivity, data freshness, artifact availability, input-contract validity,
AI availability/output, and blocked reason. A runtime with intentionally
unavailable AI paths can be `READY_WITH_LIMITATIONS`; one unavailable model
must not take down the other sensors. The stale snapshot runtime-selection
metadata is forensic evidence only. The implementation is non-persistent and
does not select models, invoke new inference paths, or alter risk behavior.
O4 LCD/Web availability-status alignment presents this contract without
changing risk-decision, alarm, or operational policy. Physical Pi LCD
verification remains `PI_REQUIRED`.

The PR #17 Thermal API blocked reason remains
`INT8_QUANTIZATION_REVIEW_REQUIRED`, matching O2.6. The physical conversion
and P1 replay are no longer the current blocker. The subsequent Thermal AI
handoff records `TRAIN_DOMAIN_RANGE_GAP` (sheepmeat/test PR #99) as the next
device-domain data/validation path; this does not authorize a model switch or
production activation.

Production Thermal and CO2 paths remain historical v0.1.0. Integration PR #22
points the active mmWave selector at locked M-N9 FULL_INT8; this is not
historical B live-gate reopening and does not by itself change the O3
`MODEL_PENDING` status projection. No T-B5 production replacement is
authorized by this document.

#### 0.8.7 Offline priority and later Pi deployment

The immediate Pi-side AI gap is Thermal. Its known path is MI48/Thermal
source → UDP → Pi → `uint16` 62×80 → NPZ persistence; T-B5 is available with
SHA `fa9730c29535477a3994c11e664474a0ca0116afaaa172889f47446ab2ac46be`.
O2 replay established the 0.1 K conversion
`physical_C = raw_uint16 / 10.0 - 273.15` for the named snapshot evidence.
O2.5/O2.6 then left `INT8_QUANTIZATION_REVIEW_REQUIRED`; the subsequent
Thermal AI handoff records `TRAIN_DOMAIN_RANGE_GAP` as the next data/validation
path. O3 adapter implementation and production activation remain `NO`.

While sensors are unavailable, work only in active `safenest-integration` and
keep evidence external to integration Git:

```text
snapshot real JSONL/NPZ/SQLite (read-only)
  → Mac-native replay / analysis
  → integration parser, adapter, backend, tests
  → reviewed integration commit / PR
  → later Pi deployment
```

Offline replay tooling must conceptually accept an external evidence path;
code lives in integration while real evidence remains outside integration Git.
Only small deterministic fixtures may later be added when justified. The `O*`
labels below are offline ordering tags, **not** normal RP phases:

| Order | Work | Status / boundary |
|---|---|---|
| O1 | Thermal raw `uint16` physical-domain contract investigation | Complete for the named O2 snapshot replay; historical evidence remains scoped |
| O2 | Thermal snapshot NPZ replay through verified conversion, P1, and T-B5 | Complete with limitations; O2.5/O2.6 require INT8 quantization review, not T-C validation |
| O3 | Integration runtime adapter / model-status cleanup | `IMPLEMENTED / MERGED` in integration PR #17; backend/API status projection only, with T-B5 adapter activation out of scope |
| O4 | Partial-availability runtime semantics and LCD/Web status alignment | `IMPLEMENTED / MERGED` in integration PR #19; Mac-offline LCD/Web presentation consumes PR #17. Physical Pi LCD / live-sensor UI remains `PI_REQUIRED` |
| O5 | MR60-native mmWave redevelopment in `sheepmeat/test` | Parallel AI-authority track; M-N9 FULL_INT8 selector imported by integration PR #22. O3 still reports `MODEL_PENDING`. Live mmWave AI smoke remains `NOT_RUN` |
| O6 | Deploy a verified integration commit and run only a minimal Pi smoke | Stage 9 tooling preparation is `IMPLEMENTED / MERGED` (PR #21); deployment and live smoke remain `PI_REQUIRED` / `SENSOR_AND_PI_REQUIRED / NOT_RUN` as applicable |
| O7 | Integrate a newly approved MR60-native model and perform a targeted mmWave smoke | Only after full AI handoff |
| O8 | Forward-port verified integration results to the team repository | Later |

O4 does not change UI risk-decision, alarm, or operational policy. It is limited
to making sensor/AI availability and blocked-status presentation consistent.

Stage 7 offline preparation is `IMPLEMENTED / MERGED` (PR #20).
`STAGE7_PREFLIGHT_MMWAVE_SELECTOR_DRIFT = RESOLVED_IN_CODE`: Stage 7 preflight
asserts the current M-N9 selector (`deployment_allowed=true`), keeps historical B
inactive, and does not treat that as Pi/device validation. Only actual Pi
deployment/execution remains `PI_REQUIRED`. Stage 9 tooling preparation is
`IMPLEMENTED / MERGED` (PR #21). Stage 9 remains
`MINIMAL_POST_DEPLOYMENT_LIVE_SMOKE`: the runner, expected status contract,
probes, evaluator, and structured report exist on Mac, but actual
backend/health, TCP `:9000`, UDP `:5005`, ESP connection, increasing
CO2/Thermal/mmWave/PIR records, and live-topology checks remain
`SENSOR_AND_PI_REQUIRED / NOT_RUN`. Do not repeat a 30-minute soak by default;
escalate only for a concrete regression/anomaly. Do not mark Stage 9 complete.

```text
ALREADY_PROVEN
!=
RETEST_BY_DEFAULT
```

No-retest-by-default includes Pi LiteRT installation, C-B6 synthetic/real
invocation, T-B5 basic synthetic load/invoke, old mmWave synthetic invocation,
synthetic soak, the concurrent field soak, basic persistence, and Pi resource
benchmarking. This does not waive a targeted regression test when a change
provides a reason.

#### 0.8.8 Explicit boundaries and handoff

Until separately authorized, defer new live ESP tests, long field soaks, live
Thermal T-B5 invocation, Pi-only timing tests, wiring/firmware changes, and
mmWave live AI invocation. Do not start Capture v1/RP-A2.

Before a future MR60-native integration path is designed, the AI handoff must
provide artifact identity/SHA, input signal and shape, sample/timestamp/window
contract, preprocessing/normalization, class mapping, quantization, and
threshold/output interpretation. Do not guess the future model shape.

Keep the snapshot until replay work is complete, relevant evidence has
checksums/identities, and the new integration runtime is successfully deployed.
The team forward-port order is integration development → offline replay
validation → later Pi minimal smoke → verified integration result → selected
forward-port → targeted team-tree regression.

| Work item | Current status |
|---|---|
| RP-X0 Stage 8 / 8.5 | `PASS_WITH_LIMITATIONS` |
| Existing mmWave B live | `BLOCKED — SIGNAL_CONTRACT_MISMATCH` |
| MR60-native retraining | `MOVED_TO_AI_AUTHORITY_TRACK` |
| CO2 C-B6 real Pi path | `PROVEN / MAINTAIN` |
| Thermal T-B5 artifact | `PRESENT` |
| Thermal physical conversion / P1 | `O2_REPLAY_VERIFIED_WITH_LIMITATIONS`; no longer the current blocker |
| Thermal current activation path | `INT8_QUANTIZATION_REVIEW_REQUIRED` in PR #17 status; `TRAIN_DOMAIN_RANGE_GAP` in subsequent AI handoff; T-B5 remains inactive |
| Mac offline replay | `AUTHORIZED` |
| Snapshot | `READ_ONLY_FIELD_EVIDENCE` |
| Integration repository | `ACTIVE_DEVELOPMENT` |
| Stage 7 offline preparation | `IMPLEMENTED / MERGED` (PR #20) |
| Stage 7 preflight mmWave selector drift | `RESOLVED_IN_CODE` |
| Stage 7 Pi execution | `PI_REQUIRED / NOT_RUN` |
| Stage 9 tooling preparation | `IMPLEMENTED / MERGED` (PR #21) |
| Stage 9 live smoke | `SENSOR_AND_PI_REQUIRED / NOT_RUN` |
| Broad repeated soak | `NOT_REQUIRED_BY_DEFAULT` |
| Team forward-port | `LATER` |

---

## 1. Executive Summary

The current Raspberry Pi runtime is a working **latest-state operator system**. It receives ESP32 scalar TCP and Thermal UDP, keeps freshness-aware latest values, evaluates historical v0.1.0 TFLite adapters when inputs happen to match, fuses V4 risk, and publishes to FastAPI, SQLite, LCD-compatible views, and the dashboard. `CODE_VERIFIED`

That is not yet an AI-evidence runtime. The B-complete candidates need unique physical observations, temporally continuous windows, frozen preprocessing, exact INT8 artifacts, and replayable lineage. The current Pi path instead:

- retains latest values and downsamples CO₂ to a 60-second usable tick; `CODE_VERIFIED`
- stores a useful but incomplete operational recorder, not a Capture/evidence contract; `CODE_VERIFIED`
- still loads historical `v0.1.0` models from `model_manifest.json` / `models.yaml`; `CODE_VERIFIED` / `OWNER_REPORTED`
- cannot reconstruct the frozen CO₂ slope contract or the Thermal T-B5 physical-frame contract; `CODE_VERIFIED`
- has RP-X0-observed and persisted real `breath_phase`, but Stage 8/8.5 establishes that vendor MR60 `breathPhase` mismatches the frozen B model's complex-range-bin phase signal. Telemetry remains available while AI is `MODEL_PENDING_MR60_NATIVE_RETRAINING`; the historical B live gate remains closed. `OBSERVED` / `BLOCKED_DEPENDENCY`

The governing transformation is:

```text
physical observation
  → ESP/device acquisition
  → Pi receive
  → persistent Capture
  → canonical sensor observation
  → bounded runtime buffer
  → frozen preprocessing
  → exact B-complete TFLite/INT8 artifact
  → sensor-local AI context
  → deterministic risk logic
  → alarm / LCD / dashboard
  → operational SQLite with Capture linkage
```

The Raspberry Pi must never guess or fabricate missing model inputs. Capture must start before B-model activation so real sensor evidence is not lost while AI integration is debugged.

**Historical implementation authorization in the 2026-08-16 baseline:**
`RP-A1_ONLY`. The current 2026-08-17 status is that RP-A1 is implemented and
under independent review; this corrective does not authorize its merge, any
new Capture work, model switching, or a real-device validation claim.

---

## 2. Scope and Non-Goals

### In scope

- Architecture audit of the current Pi runtime
- B-complete contract verification from team PR #20
- Capture / runtime-state / AI-buffer / SQLite separation
- Phase-gated Raspberry Pi enablement roadmap
- Replay, provenance, failure, retention, and Phase C placement
- Current-vs-target matrices, dependency matrix, risk register, and definition of done

### Out of scope / not authorized

- Capture or Pi runtime implementation
- ESP32 firmware changes
- Model retraining, quantization, class-map changes, or threshold changes
- Frozen preprocessing changes
- Risk-engine weight/threshold edits
- Dashboard/HMI behavior changes
- Hardware testing or claims of hardware validation
- Treating Capture sessions as training data
- Committing real Capture payloads to Git
- Operating in the sibling standalone AI repository `../embed2` (`https://github.com/sheepmeat/test`)

---

## 3. Repository and AI Baselines

### 3.1 Integration repository

| Item | Value | Evidence |
|---|---|---|
| Local root | `/Users/junwoo/Library/Mobile Documents/com~apple~CloudDocs/대학/2026/safenest-integration` | `CODE_VERIFIED` |
| Git toplevel | same path | `CODE_VERIFIED` |
| Remote | `https://github.com/yuname121/integration.git` | `CODE_VERIFIED` |
| Branch | `main` tracking `origin/main` | `CODE_VERIFIED` |
| Integration HEAD reviewed | `df75640c5a196dea869423770c3938bb90839b83` | `CODE_VERIFIED` 2026-08-16 |
| Prompt-recorded integration SHA | `0cab3afb330b0480a78ed9d74ea50dcf321ea023` | **stale**; that SHA is the team PR branch tip, not this repo |

The prompt’s integration SHA `0cab3af…` is the team `feature/ondevice-ai-b-complete-intermediate-sync` tip. This repository’s `origin/main` is `df75640…` (`Merge pull request #2 from yuname121/agent/update-safenest-integration-0814`).

### 3.2 Frozen AI snapshot currently inside integration

| Item | Value | Evidence |
|---|---|---|
| Snapshot location | `sources/ondevice_ai/` | `CODE_VERIFIED` |
| Recorded origin/main | `fa8cf13` | `DOCUMENTED_ONLY` `LATEST_SOURCE_PROVENANCE.json` |
| Recorded component source | `77b1695ac66fd595bd037e4574d1626b8917654c` | `DOCUMENTED_ONLY` |
| Policy | copied without source edits; not an active second runtime root | `DOCUMENTED_ONLY` |
| Contains C-B6 / T-B5 lock files? | **No** | `CODE_VERIFIED` |
| Runtime default models | historical `v0.1.0` | `CODE_VERIFIED` |

`sources/ondevice_ai/` is therefore **older than the B-complete baseline**. Pi model activation must not treat this snapshot as PR #20.

### 3.3 B-complete AI baseline reviewed

Team PR: [jinsu1011/safenest-embedded-competition#20](https://github.com/jinsu1011/safenest-embedded-competition/pull/20)
Title: `feat(ondevice-ai): sync B-complete offline candidate baseline`
State: **MERGED** `OWNER_REPORTED`

| Record | SHA / pointer | Evidence |
|---|---|---|
| Standalone source | `https://github.com/sheepmeat/test` `efc7e2eb61a49e221ce0ebf6057b0c1617525ad1` | `OWNER_REPORTED` |
| Team base | `3d86bf2a7a4e527d7aba2dfabcb087201ffeb46e` | `OWNER_REPORTED` |
| PR branch | `feature/ondevice-ai-b-complete-intermediate-sync` | `OWNER_REPORTED` |
| PR head | `0cab3afb330b0480a78ed9d74ea50dcf321ea023` | `OWNER_REPORTED` |
| Merge commit reviewed | `6c3faea3126cff0d17565e534d019d344edc6d1a` | `OWNER_REPORTED` |
| Machine-readable pointer | `ondevice_ai/docs/integration/20260816_b_complete_active_offline_candidates.json` | `DOCUMENTED_ONLY` |

**Meaning of B-complete:** frozen offline candidate + reproducibility/deployment contract, sufficient for team integration planning.
**Not meaning:** real-device validation, Pi production deployment, final multisensor validation, or safety certification. `OWNER_REPORTED`

### 3.4 Responsibility boundary

| Owner | Owns | Must not own |
|---|---|---|
| Integration / Pi | ESP→Pi transport consumption, receiver, state, Capture, runtime buffers, AI adapters, inference orchestration, risk/output integration, replay, Pi deployment validation | dataset provenance, training, model comparison, quantization, offline evaluation, AI validators except as consumed contracts |
| Standalone / team `ondevice_ai` | preprocessing selection, training, candidate lock, artifacts, validators, Phase C measurement guides | Pi Capture writer, SQLite schema, dashboard, ESP firmware |

---

## 4. Current Raspberry Pi Runtime

### 4.1 Deployed entry point

```text
deployment/run_pi.sh
  → hil/preflight.py
  → backend/run_backend.py
      → PersistentRuntimeStore (SQLite)
      → SafeNestRuntime
          → TCP :9000 + Thermal UDP :5005
          → SensorStateManager
          → OnDeviceAIPipeline
          → SafeNestRiskEngine
          → RuntimeStore / FastAPI :8000
          → SensorDataLogger
```

`CODE_VERIFIED` `deployment/run_pi.sh`, `backend/run_backend.py`, `backend/runtime.py`

Standalone runners `gateway/run_*_gateway.py` are diagnostic. The combined production path is `run_pi.sh`.

There is **no active `state.json` writer**. `GET /api/state` is generated from the in-memory publication. Legacy `sources/display-test*/**/state.json` files are frozen source material only. `CODE_VERIFIED`

### 4.2 Current architecture

```mermaid
flowchart TD
  ESP["ESP-WROOM-32 firmware<br/>sources/display-test2/esp32_sensor_node"]
  TCP["SafeNest TCP v1 :9000<br/>mmWave scalar + CO2 + PIR"]
  UDP["Thermal UDP v1 :5005<br/>chunked 80x62 uint16 BE"]
  RX["gateway/receiver.py"]
  TH["gateway/thermal_udp.py"]
  PARSE["gateway/protocol.py"]
  ST["state/manager.py<br/>latest values + freshness"]
  LOG["storage/sensor_logger.py<br/>partial JSONL/NPZ"]
  AI["ai/pipeline.py + ai/runtime.py<br/>historical v0.1.0 LazyModel"]
  RISK["risk/engine.py<br/>V4 weights + rule fallback"]
  STORE["backend/store.py"]
  DB["database/ schema v2<br/>data/safenest.db"]
  API["backend/app.py FastAPI :8000"]
  UI["web/dashboard + LCD view"]
  ALARM["emergency latch / GPIO or mock buzzer"]

  ESP --> TCP
  ESP --> UDP
  TCP --> RX --> PARSE --> ST
  UDP --> TH --> PARSE --> ST
  ST --> LOG
  ST --> AI --> RISK --> STORE
  STORE --> DB
  STORE --> API --> UI
  STORE --> ALARM
```

### 4.3 Component inventory

| Component | Path | Current role | Persistence |
|---|---|---|---|
| TCP receiver | `gateway/receiver.py` | Strict framing, per-connection sequence, reconnect | In-memory stats |
| Protocol | `gateway/protocol.py` | `safenest.telemetry.v1` + Thermal 80×62 BE | None |
| Thermal UDP | `gateway/thermal_udp.py` | 9-chunk reassembly, CRC32, timeout, pending bound | In-memory metrics |
| State | `state/manager.py` | Latest per-sensor record, TTL, latest Thermal frame | RAM only |
| AI | `ai/pipeline.py`, `ai/runtime.py` | Lazy v0.1.0 adapters; PIR rule | Latest result in RAM; 30-sample CO₂ deque |
| Risk | `risk/engine.py` | V4 fusion, ppm rules, emergency override | Latest + 30-sample CO₂ deque + PIR no-motion timer |
| Store | `backend/store.py` | Publication, events, emergency latch | RAM; persistent subclass mirrors SQLite |
| SQLite | `database/` | Snapshots and risk/system events | `data/safenest.db` WAL |
| Logger | `storage/sensor_logger.py` | Async mmWave JSONL, 60 s CO₂ JSONL, Thermal NPZ | `data/{mmwave,co2,thermal}` |
| API/UI | `backend/app.py`, `web/dashboard/` | Status, sensors, events, history, WS, dashboard | No separate state file |
| Alarm | emergency HMI + GPIO/mock buzzer | DANGER latch, 119 mock, SMS cooldown | SQLite events + RAM latch |

`CODE_VERIFIED`

### 4.4 Timing currently in force

| Clock / cadence | Current value | Evidence |
|---|---|---|
| ESP scalar telemetry | 1000 ms | `CODE_VERIFIED` `TELEMETRY_PERIOD_MS` |
| SCD40 poll on ESP | readiness poll ~250 ms; cached latest published at 1 Hz | `DOCUMENTED_ONLY` firmware / prior Capture audit
| Thermal requested rate | 25 FPS / divider 4 ≈ 6.25 FPS requested | `CODE_VERIFIED` firmware comment; actual FPS operational |
| CO₂ usable-state promotion | 60 s Pi monotonic | `CODE_VERIFIED` `DEFAULT_CO2_UPDATE_INTERVAL_SECONDS` |
| Risk/AI evaluation | start once, then every 15 s | `CODE_VERIFIED` `evaluation_interval_seconds` |
| Freshness TTL | mmWave/Thermal 3 s; CO₂/PIR 10 s | `CODE_VERIFIED` |
| Thermal UDP timeout | 0.5 s, max 8 pending frames | `CODE_VERIFIED` |

Freshness uses Pi monotonic time. Wall-clock is for operator correlation. `CODE_VERIFIED`

---

## 5. Current Sensor Transport Contracts

### 5.1 Scalar TCP v1

The 2026-08-16 code audit observed `safenest.telemetry.v1` carrying
`device_id`, `seq`, `uptime_ms`, `resp_rate_bpm`, `heart_rate_bpm`, `co2_ppm`,
`pir_motion`, and `valid.{respiration,heart,co2}`. `CODE_VERIFIED`

The later RP-X0 integration reference runtime observed additional scalar
provenance `boot_id`, `measurement_event_id`, and
`measurement_monotonic_ms`, plus nested mmWave `breath_phase`, `phase_age_ms`,
`ts_monotonic_ms`, and sequence. `OBSERVED`

These observations supersede the earlier “event ID / boot ID / phase absent”
statement **only for the named RP-X0 runtime and topology**. Keep later team
backend observations attributed to their actual contract rather than assuming
identical fields; this is evidence scoping, not a development-authority gate.
Humidity and temperature remain unavailable for this roadmap; Thermal pixels
remain UDP-only.

### 5.2 Thermal UDP v1

ESP sends 9 bounded datagrams (`SNTU`, CRC32, 1200-byte datagrams). Pi reassembles a logical payload of 16-byte metadata + 4960 big-endian `uint16` pixels (80×62). Incomplete, CRC, shape, or min/max failures discard the frame. `CODE_VERIFIED`

ESP uses a one-slot Thermal queue and overwrites an unsent older frame when the network is slow. Those device-side drops are not currently persisted on Pi. `CODE_VERIFIED` firmware comment

`thermal_max_c` is **not** an AI frame. SQLite `thermal_max_temp_c` is currently `NULL` because no Thermal-44 °C contract exists. `CODE_VERIFIED` `docs/PHASE8_SQLITE.md`

### 5.3 mmWave device contract

RP-X0 Stage 8/8.5 completed the MR60 `breath_phase` cadence/window and source-
semantics investigation. The frozen 10 Hz / 30 s / 300-sample BPF+Z-score model
expects unwrapped complex range-bin phase; live MR60 provides vendor
`breathPhase` over UART. No documented deterministic conversion exists.

```text
MMWAVE_SIGNAL_CONTRACT = MISMATCH
MMWAVE_B_LIVE_GATE = CLOSED
mmWave AI = MODEL_PENDING_MR60_NATIVE_RETRAINING
```

Pi must not synthesize phase from respiration/heart scalars, resample the
vendor signal to force the old model, apply empirical amplitude factors, or
per-window-normalize it for compatibility. MR60-native retraining belongs to
the AI authority track; Pi integration waits for a newly approved model
contract. `OBSERVED` / `BLOCKED_DEPENDENCY`

---

## 6. Current Persistence / Data-Loss Audit

Required sensor table:

| Sensor | Pi receives | Memory only | Persisted | Irreversibly lost | AI input reconstructable? |
|---|---|---|---|---|---|
| mmWave | 2026-08-16 audit: scalar respiration/heart only. RP-X0 reference later observed nested vendor `breath_phase`, `phase_age_ms`, source monotonic time, and sequence. `CODE_VERIFIED` / `OBSERVED` | Latest scalar state; telemetry remains operational | Stage 8/8.5 diagnostic JSONL review completed; other-runtime observations retain their own attribution | Historical B input and vendor phase have a confirmed signal-contract mismatch | Preserve telemetry/status and wait for MR60-native model handoff; historical live B remains blocked. `BLOCKED_DEPENDENCY` |
| CO₂ | 2026-08-16 audit: cached `co2_ppm` + validity. RP-X0 reference later observed `boot_id`, measurement event ID, and source monotonic time. `CODE_VERIFIED` / `OBSERVED` | Receive timing and runtime buffers | RP-X0 diagnostic path verified JSONL plus physical-provenance history; this is not Capture v1 | Other-runtime identity/timing must be recorded separately; Capture semantics remain incomplete | C-B6 observed only in RP-X0 reference runtime; no safety or ground-truth validation. |
| Thermal | Chunked UDP → validated 80×62 `uint16` BE full frame | Pending chunks, loss metrics, latest frame, Pi monotonic receive time | Complete queued frames in NPZ if written and retained; SQLite max raw / AI summary | Incomplete/timeout/CRC frames, frame gaps, CRC/monotonic metadata, logger-queue drops, ESP overwrite drops | Saved pixels are lossless. Session lineage and dropped-frame chronology are **not**. O1/O2 established the named-snapshot 0.1 K conversion and P1 replay; new device-domain data/validation remains blocked. |
| PIR | Boolean in every scalar packet `CODE_VERIFIED` | Latest boolean; Risk no-motion start | Periodic SQLite snapshot only. Logger has **no PIR file**. | Transition timing, packet identity, repeated samples | Periodic summary only. Exact transition replay **no**. |

### 6.1 Other persistence

| Data | Current fate |
|---|---|
| AI predictions | Latest in RAM; Thermal class/probability summarized in SQLite; no model SHA, tensor hash, or source-event linkage |
| Model identity | Loader checks v0.1.0 SHA at load; not recorded per inference in SQLite |
| Risk decisions | Latest in RAM; snapshots + `risk_events` in SQLite |
| Alarm / buzzer / emergency | RAM latch + SQLite emergency fields/events |
| Sensor health | RAM freshness + SQLite status columns |
| Display state | Generated from publication; no `state.json` |
| Logger diagnostics | `/health` counters only; drops are not durable evidence records |

### 6.2 Central irreversible losses

1. Receiver parses to typed objects before any Capture envelope exists.
2. The 2026-08-16 audit lacked CO₂ physical identity; RP-X0 later observed it, but the result is runtime/topology scoped and does not complete Capture v1.
3. PIR is not written by `SensorDataLogger`.
4. Thermal failures and gaps are memory metrics only.
5. The 2026-08-16 audit traffic could not reconstruct mmWave phase; RP-X0 diagnostic JSONL now contains observed phase samples, but cadence/window validity remains unverified and any other-runtime observation must retain its own attribution.
6. Queue overflow, process crash, and power loss can lose queued logger items.
7. Quota cleanup deletes per-sensor files independently, not as a replayable session.

`CODE_VERIFIED` / Capture companion contract

---

## 7. B-Complete AI Contract Summary

Reviewed from PR #20 merge `6c3faea…` and `20260816_b_complete_active_offline_candidates.json`. `DOCUMENTED_ONLY` / `OWNER_REPORTED`

### 7.1 mmWave — frozen offline candidate

| Field | Frozen value |
|---|---|
| Candidate | `M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120` |
| Artifact | `models/mmwave/experiments/M-B6_stage_equivalence/M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120_int8.tflite` |
| SHA-256 | `6dff6aaa72c79d76715d40cf7e32bb1e6cd9b2c2e3ac78eaf2fda737561430c5` |
| Input | INT8 `[1,300,1]`, 10 Hz, 30 s, 300 samples |
| Preprocessing | `BPF_ZSCORE` (`M-B1_D0_B1_Z1` / `M-B10B_SELECTED_REAL_CANDIDATE_BPF_ZSCORE_V1`) |
| INT8 scale/zp | `0.041720833629369736` / `-3` |
| Output | INT8 `[1,3]`, scale `0.00390625`, zp `-128` |
| Class map | `0=NORMAL`, `1=RAPID_OR_ABNORMAL`, `2=APNEA` |
| APNEA meaning | Voluntary breath-hold **proxy**, not clinical apnea |
| Runtime default | Historical `mmwave_resp_int8_v0.1.0.tflite` still default; v0.1.0 is `deployment_allowed=false` (class collapse) |
| Device domain | `SIGNAL_CONTRACT_MISMATCH`: valid offline evidence, but **not live MR60 compatible**. Pi waits for an AI-authority MR60-native replacement contract. |

Historical v0.1.0 uses z-score-only (`mean=0.00609198`, `std=2.50138`) and a different quantization. Pointing `models.yaml` at the candidate file without the BPF contract is incorrect. `CODE_VERIFIED` / `DOCUMENTED_ONLY`

### 7.2 CO₂ — frozen C-B6 occupancy candidate

| Field | Frozen value |
|---|---|
| Candidate | `C_B6_REDUCED_CO2_SLOPE_CANDIDATE_001` |
| INT8 artifact | `models/co2/candidates/c_b6/full_integer_int8.tflite` |
| INT8 SHA-256 | `c5969b367f5b5e28c4d27f1bdd6220f7d02da92e99604e3f85c9f1291e98dd3b` |
| Feature order | `CO2`, `CO2_slope` |
| Forbidden inputs | Temperature, Humidity, Light, time-of-day, previous predictions, sensor metadata, new derived features |
| Slope method | `ENDPOINT_DIFFERENCE` over source-clock history |
| History | 150 s; minimum 2 samples; minimum elapsed 150 s |
| Gap policy | Restart history after gap `> 90 s` |
| Timestamp basis | `SOURCE_ACQUISITION_CLOCK` |
| Scaler | TRAIN-only `StandardScaler`; fingerprint `a92123ad37e9b284929ba0fe53179126345d54d487ec4b3a73c910d00490a462` |
| Scaler mean | `[606.5058118345612, 0.011527303414630624]` |
| Scaler scale | `[314.3524240597083, 5.661675596121919]` |
| INT8 input | `[1,2]`, scale `0.03921568766236305`, zp `0` |
| INT8 output | `[1,1]` logistic occupancy score, scale `0.00390625`, zp `-128` |
| Threshold | `0.43`, `TRAIN_INTERNAL_ONLY` |
| Class map | `0=VACANT`, `1=OCCUPIED`; semantic `ROOM_OCCUPANCY` |
| Risk/safety semantic | **NONE**. Occupancy is not a CO₂ safety threshold, sensor health, or multisensor risk. |
| Device domain | SCD40 Phase C **NOT COMPLETE** |

This is a **breaking change** versus the current Pi adapter:

| Item | Current Pi / v0.1.0 | C-B6 |
|---|---|---|
| Features | `CO2_slope, Humidity, CO2` | `CO2, CO2_slope` |
| Input shape | `[1,3]` | `[1,2]` |
| Output | `[1,2]` softmax | `[1,1]` logistic |
| Humidity | required; currently missing → `INPUT_UNAVAILABLE` | forbidden |
| Slope window | 30 promoted 60 s samples, elapsed-minutes over deque | 150 s source-clock endpoint difference, 90 s gap reset |
| Artifact in local snapshot | v0.1.0 + C-B5 four-feature candidate | C-B6 **not present** in `sources/ondevice_ai/` |

`CODE_VERIFIED` vs `DOCUMENTED_ONLY`

### 7.3 Thermal — frozen T-B5 FULL_INT8

| Field | Frozen value |
|---|---|
| Candidate | `FULL_INT8` / `SMALL_CNN_BASELINE_V1_P1_full_int8.tflite` |
| Logical path | `T-B4/artifacts/SMALL_CNN_BASELINE_V1_P1_full_int8.tflite` |
| SHA-256 | `fa9730c29535477a3994c11e664474a0ca0116afaaa172889f47446ab2ac46be` |
| Size | 318280 bytes |
| Artifact availability | T-B5 FULL_INT8 binary available/provisioned; SHA verified and isolated LiteRT load/invoke observed. Integration synchronization is `chore/sync-latest-b-stage-models` @ `ddd8150` / PR #8. This is not production selection or Thermal-44 validation. `OBSERVED` / `OWNER_REPORTED` |
| Input | INT8 `[1,62,80,1]`, scale `0.31791284680366516`, zp `-125` |
| Output | INT8 `[1,3]`, scale `0.00390625`, zp `-128` |
| Float preprocessing lock | `P1_TRAIN_FITTED_GLOBAL_ZSCORE` on **Celsius** frames; TRAIN mean `22.769290618485442`, std `2.8684523405441222` |
| Offline geometry source | SDT canonical `(62,80)` from distributed `(480,640)` via `G1_FIXED_ASPECT_CROP_BILINEAR`; unit `°C = (encoded_uint16 - 27315) / 100` for **SDT**, not Thermal-44 |
| Class compatibility layer | `NOT_HUMAN` / `HUMAN_NORMAL` / `HUMAN_FALL` |
| `HUMAN_FALL` meaning | `DERIVED_POSTURE_PROXY` from source `LYING`. **Not** a verified `FALL_EVENT`. |
| Device domain | Thermal-44 real-device **NOT COMPLETE**; orientation/unit `UNVERIFIED` |

Current Pi Thermal path uses **per-frame min-max to [0,1]** then v0.1.0 quantization `scale=0.003921568859368563`, `zp=-128`. That is **not** the T-B5 contract. Feeding raw or min-max frames into T-B5, or claiming `HUMAN_FALL` is a fall event, is forbidden. `CODE_VERIFIED` / `DOCUMENTED_ONLY`

Exact Pi inference graph (Celsius conversion → P1 z-score → INT8 vs converter-absorbed quantization) must be frozen by the AI owner in RP-B0. Do not invent a second Thermal-44 Kelvin formula. `OWNER_DECISION_REQUIRED` / `BLOCKED_HARDWARE`

### 7.4 PIR

No PIR AI model exists in current source or B-complete. PIR remains supporting evidence / risk context. `CODE_VERIFIED`

---

## 8. Current-vs-Target Gap Analysis

| Component | Current Pi behavior | B-complete requirement | Gap | Required Pi change | External dependency |
|---|---|---|---|---|---|
| CO₂ | 60 s promoted ppm; humidity-gated v0.1.0 `[1,3]`; slope from 30-sample deque | RP-X0 observed physical identity, 150 s endpoint slope, `[CO2, CO2_slope]`, and C-B6 LiteRT output | Historical runtime features/shape/output differ; Capture v1 remains separate; old integration AI snapshot lacks C-B6 | Preserve the frozen C-B6 contract and add regression/status protection only | AI snapshot/artifact sync; Capture/Phase C only when separately authorized |
| Thermal | UDP reassembly; latest frame; per-frame min-max v0.1.0 `[1,62,80,1]` zp `-128` | Full raw frame Capture; verified named-snapshot `uint16`→Celsius→P1/T-B5 INT8; posture proxy only | Production adapter is absent; O2.6/AI follow-up identifies quantization and `TRAIN_DOMAIN_RANGE_GAP`, while sessionized evidence remains absent | Retain blocked status; future activation only after approved AI/device-domain handoff; no `FALL_EVENT` rewrite | Thermal AI data/validation path; approved production handoff |
| mmWave | RP-X0 integration reference observed persisted vendor phase and Stage 8/8.5 completed | Historical B expects 10 Hz complex-range-bin phase, 300 samples, `BPF_ZSCORE`, INT8 `[1,300,1]` zp `-3` | Live vendor `breathPhase` signal contract mismatches the frozen B input; gate closed | Keep telemetry/status available; await MR60-native model handoff from AI authority | MR60-native dataset/model/preprocessing/quantization contract |
| PIR | Latest bool; no raw file; risk no-motion rule | Transition/event evidence; supporting risk context | Transitions lost | Capture first-state + transitions | None for Capture; presence source still incomplete |
| Model loading | `LazyModel` → `model_manifest.json` v0.1.0; mmWave blocked | Resolve B-complete from active-candidate pointer; SHA-256; interpreter compatibility | Defaults are historical; RP-X0 artifact availability is not production selection or deployment activation | Dedicated activation phase; checksum; missing-artifact fail closed | Approved locked-binary policy; integration artifact discovery; later team forward-port |
| Preprocessing | Interpreter-internal v0.1.0; experimental mmWave BPF unused by default | One canonical frozen implementation for runtime and replay | Duplication risk A/B/C | Shared preprocessing module/contract | AI owner freeze of Pi-callable functions |
| Capture | Hourly JSONL/NPZ logger, quota cleanup, no sessions | Sessionized append-only evidence + Thermal binary | Not an evidence contract | Capture v1 | CO₂ identity for exact slope replay |
| SQLite | Operational snapshots/events, schema v2 | Keep as summary; add Capture/inference linkage | No evidence IDs | Additive linkage columns/JSON | None |
| Replay | NPZ/JSONL can be read ad hoc; no validator | Capture → canonical → frozen prep → exact model → compare | Missing | Replay layer after Capture | Models/artifacts |
| Risk integration | AI Thermal used; CO₂ occupancy unused for score; mmWave AI unused due to missing input; ppm/rpm/PIR rules | AI is context; do not replace ppm/rpm rules with occupancy/APNEA proxy | Semantic confusion risk if C-B6 OCCUPIED or HUMAN_FALL is treated as emergency without policy | Explicit context fields; keep V4 rules until a reviewed policy change | Owner decision: whether occupancy context changes fusion; do not change thresholds in this roadmap |

---

## 9. Target Raspberry Pi Architecture

```mermaid
flowchart LR
  ESP["ESP/device<br/>CURRENT"]
  PI["Pi receive<br/>TCP+UDP CURRENT"]
  subgraph CAP["Capture / Replay path"]
    W["Capture writer NEW"]
    SESS["captures/session NEW"]
    VAL["Capture validator NEW"]
    REP["Replay NEW"]
  end
  subgraph RUN["Runtime / AI / Risk path"]
    ST["State manager EXTEND"]
    BUF["Sensor rolling buffers NEW"]
    PRE["Canonical frozen preprocessing NEW"]
    MOD["B-complete TFLite INT8 NEW / BLOCKED mmWave"]
    CTX["Sensor-local AI context NEW"]
    RISK["Deterministic risk EXTEND"]
    OUT["Alarm / LCD / dashboard CURRENT"]
    DB["SQLite operational EXTEND"]
  end
  ESP --> PI
  PI --> W --> SESS
  PI --> ST --> BUF --> PRE --> MOD --> CTX --> RISK --> OUT
  RISK --> DB
  CTX --> DB
  SESS --> VAL --> REP
  REP --> PRE
  DB -. session_id / capture_event_id .-> SESS
```

Status legend:

| Mark | Meaning |
|---|---|
| CURRENT | Keep and extend |
| NEW | Required Pi work |
| FUTURE | After a named gate |
| BLOCKED | External dependency |

Four storage responsibilities remain separate:

| Layer | Purpose | Mechanism |
|---|---|---|
| A. Capture / evidence | Real-device evidence, replay, fault diagnosis, model-input reconstruction, Phase C | `captures/<session>/` JSONL + Thermal NPZ |
| B. Runtime sensor state | Latest valid state, freshness, health, availability | `SensorStateManager` RAM |
| C. AI rolling buffers | CO₂ 150 s history, future mmWave 300-sample window, latest Thermal frame | Bounded RAM, reset on continuity breaks |
| D. Operational summary | Risk history, alarms, AI outputs, operator views | Existing SQLite |

Do not force one mechanism to perform all four jobs.

---

## 10. Capture v1

Adopt the companion contract rather than inventing a second layout.

Recommended session layout `PLANNED`:

```text
captures/
└── <session_id>/
    ├── manifest.json
    ├── events_0001.jsonl
    ├── events_0002.jsonl
    ├── thermal/
    │   ├── frames_0001.npz
    │   └── ...
    ├── inference/
    │   └── records_0001.jsonl
    └── session_close.json
```

This matches existing NumPy use, keeps 4960 pixels out of generic JSONL, and stays out of `data/safenest.db`.

### 10.1 Why not reuse `storage/sensor_logger.py` as-is

Preserve its useful ideas (async queue, no disk I/O in receive callbacks, Thermal batching, `/health` counters). Replace the contract:

- sessionized instead of hourly independent sensor directories
- all sensors including PIR and explicit loss events
- no 60 s CO₂ downsample in persistent Capture
- monotonic receive time, CRC, and payload checksums
- capture health distinct from SQLite health
- retention by closed session, not per-sensor file deletion

Classification: `EXTEND` patterns, `REPLACE_LATER` the current file-as-contract.

### 10.2 Thermal payload format

Recommendation: **chunked lossless NPZ**, `allow_pickle=False`.

| Option | Write | Compression | Power-loss | Replay | Checksum | Random access | Complexity |
|---|---|---|---|---|---|---|---|
| JSONL pixels | Poor | Weak | Line-append OK | Awkward | Per line | Poor | Low, rejected |
| NPY per frame | Many files | None | Partial files | Easy | Sidecar | Easy | Medium |
| **NPZ batch** | Good | ZIP deflate, lossless uint16 | Needs temp+rename | Easy | Archive + per-frame SHA | Index in event log | Medium; already used |
| Custom archive | Best potential | Tunable | Custom | Custom | Custom | Custom | High, unnecessary now |

NPZ is selected because the current logger already writes `(batch, 62, 80) uint16` with `allow_pickle=False`, and Capture only needs to add session metadata, CRC, monotonic time, and atomic close. `CODE_VERIFIED` current logger / `PLANNED` Capture

---

## 11. Session / Timestamp / Provenance Contract

### 11.1 Session lifecycle

Recommended for competition use: **one Capture session per integrated Pi application run**, with optional operator-started experiment sessions. ESP reconnects stay inside the session as transport events. A new process start opens a new session. `PLANNED`

| Step | Behavior |
|---|---|
| Create | New `session_id`, directory, `manifest.json` |
| Active | Append-only JSONL + Thermal archives |
| Rotation | Close segment, open next; not a session close |
| Close | Drain writers, checksum, `session_close.json` |
| Crash | No close marker; next start reports unclean previous session |

### 11.2 Timestamp meanings — never collapse to one `timestamp`

| Name | Source | Use |
|---|---|---|
| Device/source measurement time | SCD40/MR60/Thermal acquisition clock when the contract supplies it | Slope, phase cadence, uniqueness |
| Device uptime | ESP `uptime_ms` | Reboot/discontinuity detection |
| Pi receive wall-clock | `time.time()` | Operator correlation |
| Pi receive monotonic | `time.monotonic()` | Freshness, receive order |
| AI inference time | evaluation clock | Provenance |
| Risk-decision time | 15 s evaluator clock today | Provenance |
| Display/alarm publication time | store publish clock | HMI/SQLite |

`CODE_VERIFIED` that the current runtime already distinguishes wall vs monotonic for receive; Capture must preserve that distinction and add source measurement time when the device contract supplies it.

### 11.3 Common Capture metadata

Populate only fields the runtime can actually know. Missing values are explicit unavailability, not invented IDs.

| Field | Can populate now? | Notes |
|---|---|---|
| `schema_version` | Yes | Capture schema |
| `session_id` | Yes after RP-A1 | Pi generated |
| `sensor_type` | Yes | `co2` / `thermal` / `pir` / `mmwave` / `runtime` |
| `device_id` | Yes for TCP | Thermal UDP currently has no device_id in the frame object; record unavailability or attach from last scalar peer by explicit policy, do not silently invent |
| `boot_id` | **No** | Device contract absent; `null` + reason |
| `packet_sequence` | Yes | Transport seq, not physical measurement ID |
| `device_uptime_ms` | Yes | ESP uptime |
| `source_measurement_event_id` | **No** on active v1 | Required for exact CO₂ uniqueness; until then `measurement_identity_unavailable=true` |
| `source_measurement_monotonic_ms` | **No** on active v1 | Same |
| `pi_receive_wall_time` | Yes | Already available |
| `pi_receive_monotonic_time` | Yes | Currently not persisted |
| `parse_valid` | Yes | Decoder/CRC result |
| `sensor_valid` | Yes | Source valid flags |
| `stale` | Runtime-derived only | Do not rewrite source evidence as stale |
| `error_code` / `error_reason` | Yes | Failures must be events |
| `payload_reference` | Yes for Thermal | Repo-relative path, never `/Users/...` |

---

## 12. CO₂ Runtime Plan

Target path `PLANNED`:

```text
SCD40 measurement
  → ESP cache + telemetry
  → measurement event identity + source time   [ESP_CONTRACT_DEPENDENCY]
  → Pi Capture (unique physical events)
  → runtime history buffer (150 s + gap margin)
  → frozen ENDPOINT_DIFFERENCE CO2_slope
  → TRAIN-only scaler, feature order [CO2, CO2_slope]
  → C-B6 INT8 [1,2]
  → occupancy probability vs threshold 0.43
  → VACANT / OCCUPIED context
  → risk still uses ppm/slope safety rules unless a later reviewed policy says otherwise
```

### 12.1 Persistent Capture vs runtime buffer vs reset

| Layer | Store | Dedup | Reset |
|---|---|---|---|
| Capture | Every unique SCD40 measurement; until identity exists, every transport observation marked `measurement_identity_unavailable` | Same physical ID must not become another measurement. Do **not** use value equality as identity. Do **not** apply the 60 s logger gate to Capture. | Never periodic. Session close / retention only. |
| Runtime buffer | `(capture_event_id, source_time or receive_monotonic_fallback, ppm, valid)` for ~150 s plus margin for the 90 s gap check | Retransmissions of the same event do not append | New session, device reboot/uptime rollback, gap `> 90 s`, long invalid period, timestamp discontinuity |
| Derived slope | RAM + inference provenance; not a replacement for ppm | Recompute from buffer | Invalidated with the buffer |

Wall-clock hourly/minute timers must not reset the slope history.

### 12.2 Fail-closed inference statuses

| Condition | Status | Inference |
|---|---|---|
| < 2 valid samples or elapsed < 150 s | `INPUT_WARMUP` / `FEATURE_UNAVAILABLE_WARMUP` | `NO_INFERENCE` |
| Internal gap > 90 s | `INPUT_INVALID` / `FEATURE_UNAVAILABLE_GAP_RESTART` | `NO_INFERENCE`; reset buffer |
| Non-finite ppm, parse invalid | `INPUT_INVALID` | `NO_INFERENCE` |
| State STALE vs evaluation | `INPUT_STALE` | `NO_INFERENCE` |
| C-B6 artifact missing/hash mismatch | `MODEL_UNAVAILABLE` | `NO_INFERENCE`; ppm risk rules continue |
| Valid features | occupancy context | Map logistic ≥ 0.43 → `OCCUPIED` else `VACANT`. Do not treat scores as calibrated probabilities. |

Do not convert invalid input into `VACANT`.

### 12.3 Cadence warning

Offline C-B6 was trained on UCI occupancy with nominal ~60 s source cadence. SCD40 data-ready is typically ~5 s and ESP currently publishes the latest cached ppm at 1 Hz. Device-domain gap `DEVICE_UCI_CADENCE_DOMAIN_GAP` remains. Pi must compute the frozen formula on real event times, not resample to fake UCI cadence. `DOCUMENTED_ONLY` / `BLOCKED_HARDWARE`

---

## 13. Thermal Runtime Plan

Target path `PLANNED`:

```text
Thermal-44
  → ESP chunked UDP
  → Pi reassembly + CRC/shape/minmax     [CURRENT]
  → full raw frame Capture               [NEW]
  → physical-unit conversion             [BLOCKED_HARDWARE until Thermal-44 unit]
  → canonical orientation/geometry       [BLOCKED_HARDWARE]
  → frozen T-B5 preprocessing            [NEW after AI freeze]
  → FULL_INT8 inference                  [MODEL_ARTIFACT_DEPENDENCY]
  → NOT_HUMAN / HUMAN_NORMAL / HUMAN_FALL context
```

### 13.1 Already implemented vs new

| Piece | Status |
|---|---|
| Chunked UDP, CRC, 80×62, min/max, timeout | CURRENT `CODE_VERIFIED` |
| Latest full frame in RAM | CURRENT |
| NPZ batch of raw uint16 | CURRENT partial logger |
| Sessionized Capture + checksum + monotonic + loss events | NEW |
| Thermal-44 raw → °C | BLOCKED; SDT formula must not be silently reused |
| Orientation/transpose/flip contract | BLOCKED |
| P1 global z-score / T-B5 INT8 | NEW after RP-B0 freeze |
| T-B5 binary on Pi | MODEL_ARTIFACT_DEPENDENCY |
| `thermal_max_c` as model input | FORBIDDEN |

### 13.2 Runtime memory

T-B5 is frame-only. Runtime RAM needs the current validated frame and at most a tiny queue for evaluator/Capture handoff. Unlimited Thermal frames in RAM are forbidden. Persistent Capture is separate.

### 13.3 Semantics

Preserve original posture semantics. `LYING` / `HUMAN_FALL` is a derived posture proxy. Risk may currently emergency-override on `HUMAN_FALL` with confidence ≥ 0.8 (`CODE_VERIFIED`). That existing override must be reviewed in RP-C0 so a lying-proxy class is not silently treated as a verified fall event. This roadmap does not change that threshold.

---

## 14. mmWave Dependency Plan

Do **not** implement or enable the current model input on Pi now. RP-X0 has
observed real persisted phase, but that is evidence for the read-only offline
cadence/window audit, not authorization for a live model path.

Gate:

```text
MR60 breath_phase real-device contract
  → cadence verification
  → semantic compatibility with 10 Hz / 300 samples / BPF_ZSCORE
  → only then Pi phase Capture, rolling window, gap handling, INT8 inference
```

Until the gate:

- Continue recording scalar respiration/heart as operational/Capture observations.
- Preserve observed phase samples when the named runtime supplies them; otherwise write `phase_unavailable` rather than a fake window.
- Keep mmWave AI `INPUT_UNAVAILABLE`; risk continues to use rpm rule fallback.
- Do not resample heart/respiration into phase.
- Do not enable v0.1.0 (class collapse) or the B-complete candidate as a live default.

This is a future RP-B3 path, blocked by `MR60_NATIVE_MODEL_HANDOFF_DEPENDENCY`.

After the gate, the rolling buffer is 300 samples at the verified cadence, with gap handling from the frozen contract (historical V4 text mentioned 0.5 s; the B-complete contract must be re-read at unblocking time and not assumed here). `DOCUMENTED_ONLY` / `BLOCKED_DEPENDENCY`

---

## 15. PIR Integration Plan

Keep PIR simple. No PIR model.

| Preserve | How |
|---|---|
| Latest state | `SensorStateManager` CURRENT |
| Transition event | Capture on first observation and boolean change |
| Source timing | Packet seq + ESP uptime + Pi wall/monotonic |
| Validity | Parse/sensor valid; stale via TTL |
| Risk | Existing no-motion rule with presence confirmation |

Do not persist every 1 Hz duplicate `false/false` as a new observation. Sequence gaps still need transport events or counters. Replay forward-fills last valid state until the next transition or continuity break.

---

## 16. AI Artifact / Model Resolution

Historical defaults still in force:

| Sensor | Default artifact | B-complete artifact |
|---|---|---|
| Thermal | `thermal_fall_int8_v0.1.0.tflite` SHA `5b56da8d…` | T-B5 SHA `fa9730c2…` is available/provisioned and SHA-verified for RP-X0 diagnostics; production selection and Thermal-44 compatibility remain blocked |
| CO₂ | `co2_occupancy_int8_v0.1.0.tflite` SHA `3a8c86c4…` | C-B6 SHA `c5969b36…` provisioned for RP-X0; observed live output is not a production selection or safety validation |
| mmWave | `mmwave_resp_int8_v0.1.0.tflite` blocked | M-B3 INT8 SHA `6dff6aaa…` remains historical/offline evidence; `NOT_LIVE_MR60_COMPATIBLE`, live gate closed, MR60-native model pending |

`CODE_VERIFIED` / `DOCUMENTED_ONLY`

### 16.1 Activation is a dedicated phase (RP-B0), after Capture

Do not merely retarget `models.yaml`. Before switching any runtime default verify:

- team-approved artifact availability on the Pi
- SHA-256
- manifest / candidate pointer
- input contract, preprocessing profile, class map, dtype, quantization
- LiteRT interpreter compatibility
- tests for fail-closed missing/mismatch
- explicit operator/config opt-in, not silent replacement of v0.1.0 tests

### 16.2 Thermal binary deployment strategy

The prior `EXTERNAL_SSD_ONLY` statement is obsolete for the dated RP-X0 artifact
state. T-B5 is available/provisioned with the required SHA, and the locked-binary
policy/synchronization work is tracked separately. Distribution and production
selection remain separate questions:

| Option | Recommendation |
|---|---|
| Commit the binary to integration git? | Follow the approved locked-binary policy and exact SHA; do not infer production selection from Git availability |
| Release attachment? | Acceptable if SHA-256 is in git and download is pinned |
| Fetch at deploy time? | Allowed only with checksum, fail closed if missing, no unpinned URL |
| Copy during Pi provisioning? | **Preferred for competition**: provisioned path + SHA-256 in candidate pointer; preflight fails if absent |
| Missing on Pi? | `MODEL_UNAVAILABLE`; Capture and ppm/rpm rules continue; never skip checksum |

Do not assume the file exists in a **different** runtime or that a running
process scanned it at startup. Stage 7/9 must verify discovery in the restored
integration runtime; missing discovery remains `MODEL_UNAVAILABLE`.

### 16.3 Integration snapshot sync

RP-B0 requires the B-complete contracts inside the integration tree or an equivalent pinned vendor path. That is an AI baseline copy, not retraining. Runtime defaults must remain v0.1.0 until the activation checklist passes. `AI_BASELINE_DEPENDENCY`

---

## 17. Preprocessing / Input Contract Validation

### 17.1 One canonical implementation

Avoid three drifting copies (offline / Pi / replay). Prefer:

```text
frozen B preprocessing functions
  reused by Pi runtime
  reused by replay
  covered by the same fixtures
```

Where the standalone code is importable without pulling training stacks, wrap it through `ai/` adapters. If not, copy the frozen numeric contract into one Pi module with checksummed fixtures — still one implementation, not two formulas.

### 17.2 Per-inference validation

Before invoke, validate shape, dtype, feature order, physical unit, validity, temporal continuity, and preprocessing-state identity.

| Failure | Result |
|---|---|
| Missing required field / wrong shape / NaN | `INPUT_INVALID` / `NO_INFERENCE` |
| History warming | `INPUT_WARMUP` |
| Freshness TTL exceeded | `INPUT_STALE` |
| Artifact missing/hash/interpreter | `MODEL_UNAVAILABLE` |
| mmWave historical B signal-contract mismatch | `MODEL_PENDING` / `MR60_NATIVE_MODEL_PENDING` (telemetry remains available) |

Invalid input must not become a normal-class prediction.

---

## 18. AI Provenance

Per inference, record at least `PLANNED`:

```text
inference_id
session_id
source event/frame references
model_id
model_sha256
model_format
preprocessing_profile
input_contract_version
input_validation_result
input_tensor_hash (optional but useful)
output scores / selected class
threshold used (CO₂ 0.43) without claiming calibration
risk result / reasons references
inference_time / risk_time
```

Model outputs are not assumed to be calibrated probabilities. C-B6 threshold is `TRAIN_INTERNAL_ONLY`. Thermal/mmWave INT8 dequantized vectors are scores, not clinical probabilities. `DOCUMENTED_ONLY`

Store inference JSONL under the Capture session. Put only summaries + IDs in SQLite.

---

## 19. SQLite / Operational Logging

Keep SQLite as the operational summary store. Schema v2 already has snapshots, risk events, emergency/buzzer fields. `CODE_VERIFIED` `database/schema.sql`

Do not put raw Thermal frames or 300-sample phase windows into SQLite.

### 19.1 Future linkage

Additive, after inspecting current columns. Prefer nullable JSON/text fields rather than a breaking rewrite:

| SQLite location | Link |
|---|---|
| `sensor_snapshots` | `session_id`, latest `capture_event_id` per sensor, `inference_id` |
| `risk_events.details_json` | `session_id`, `capture_event_id`s, `inference_id`, model SHA |

Exact column names are an RP-B4 design choice; do not bikeshed them before schema migration tests.

Restart restoration currently rebuilds status/risk/emergency, not sensor values or Thermal bytes. That should remain true. `CODE_VERIFIED`

---

## 20. Replay

Dedicated phase RP-A5 (Capture replay) then RP-B4 (inference equivalence).

```text
stored Capture
  → validator (manifest, checksums, schema, NPZ dtype/shape)
  → canonical observation stream
  → same frozen preprocessing
  → same exact model artifact
  → prediction
  → compare with original runtime result if provenance exists
```

Replay must be able to answer:

| Question | How |
|---|---|
| Was source data wrong? | Capture `sensor_valid` / `error_code` |
| Was transport incomplete? | `transport_gap`, incomplete Thermal events, logger/capture drops |
| Was the observation stale? | Runtime stale events vs source evidence |
| Was preprocessing different? | Profile/hash in provenance vs replay |
| Was the wrong model used? | `model_sha256` |
| Did quantization change output? | Float reference vs INT8 comparison fixtures, not live guessing |
| Did risk rather than AI trigger the alarm? | Risk reasons vs AI class; emergency override flags |

If CO₂ identity is unavailable, replay must say exact slope reconstruction is unavailable rather than silently deduplicating by ppm value.

---

## 21. Risk Engine Integration

Current consumption `CODE_VERIFIED` `risk/engine.py`:

| Component | Uses real sensor? | Uses AI? | Missing/stale | Emergency |
|---|---|---|---|---|
| Thermal | LIVE required | **Yes**, AI class/score; no rule fallback | unavailable | `HUMAN_FALL` + confidence ≥ 0.8 → DANGER 100 |
| mmWave | LIVE + rpm | AI if 300-window present (currently never) | rpm rule fallback 12–20 | `APNEA` only if `apnea_verified is True` (currently always false) |
| CO₂ | LIVE + ppm | Occupancy **not** used for score | unavailable | ppm 1000/2500 and slope 15 ppm/min rules |
| PIR | LIVE + bool | rule only | unavailable; no-motion timer reset | none; long no-motion needs presence |

AI is context, not autonomous safety authority, except the existing Thermal confidence override. Introducing B-complete outputs:

- C-B6 `OCCUPIED`/`VACANT` → metadata/context only unless a later reviewed policy changes fusion. Do not replace 1000/2500 ppm rules with occupancy.
- T-B5 `HUMAN_FALL` → still a posture proxy; RP-C0 must not silently strengthen the emergency override.
- mmWave classes → still blocked; when enabled, APNEA remains unverified without an explicit verification contract.

Do not change V4 weights `0.35/0.35/0.15/0.15` or 30/60 thresholds in this roadmap.

---

## 22. Buffer Reset / Rotation / Retention

| Data | Reset / rotate | Delete |
|---|---|---|
| Persistent Capture | Session close; file rotation by size/time | Oldest **closed** sessions after budget/free-space policy. Never periodic wipe of the active session. |
| AI rolling buffers | New session, reboot, reconnect+continuity break, gap, timestamp discontinuity, long invalid | RAM only; deleting RAM must not delete Capture |
| Current state | Replaced by newer valid state per TTL | Process lifetime |
| SQLite | Separate retention/archival | Do not erase because an AI buffer reset |

Rotation ≠ reset ≠ retention deletion. See §10 and §22 of this roadmap.

Git ignore: add `captures/` before any real session exists. Current `.gitignore` already ignores `data/mmwave|co2|thermal/*` and `*.db`. `CODE_VERIFIED`

---

## 23. Failure Handling

Capture failure must be observable. Do not report Capture enabled when nothing is persisted.

| Failure | Behavior `PLANNED` |
|---|---|
| Disk full | `capture_failed` or `degraded` before drop; retention only under policy |
| Writer failure | Bounded retry, then failed; identity preserved |
| Queue overflow | `capture_drop` event/counters; `degraded`; receive path stays live |
| Corrupt payload | Do not rewrite history; quarantine; error event |
| Rotation failure | Do not overwrite old segment; failed |
| Process crash / power loss | No false `session_close.json`; next start unclean |
| SQLite failure | Separate from Capture health; memory fallback already exists `CODE_VERIFIED` |

Health states: `capture_healthy` / `capture_degraded` / `capture_failed`. Expose on `/health`.

---

## 24. Raspberry Pi Performance Validation

Later hardware phase RP-C2. **Not passed now.** Measure:

CPU, RAM, disk throughput, storage growth, Thermal write bandwidth, inference latency, model load time, queue depth, network loss, Pi temperature, multi-hour runtime, restart recovery.

Do not report these as complete from this documentation pass.

---

## 25. Real-Device Phase C

Place formal device-domain validation **after** Capture, input reconstruction, model activation, and replay work.

Separate tracks:

| Track | Depends on |
|---|---|
| MR60 / M-C | Device phase contract + Pi phase Capture/window |
| SCD40 / C-C | Measurement identity + C-B6 slope reproduction |
| Thermal / T-C | Unit/orientation + T-B5 artifact + full-frame Capture |
| Multisensor synchronized | All three + clock/session alignment |

Capture is not automatically training data:

```text
Capture → quality review → scenario/label review
  → privacy/consent review where applicable
  → dataset admission → canonicalization → training
```

That training path lives in the AI repository, not in Pi runtime phases.

---

## 26. Phase-by-Phase Roadmap

Existing integration phases are `PHASE 1`–`PHASE 10` plus HIL. No `RP-*` IDs exist. This series uses prefix **`RP-`** and does not collide. `CODE_VERIFIED`

RP-A0 is satisfied as a documentation gate. The historical next implementation
was RP-A1; as of this corrective RP-A1 is implemented and under independent
review, with merge authorization outside RP-X0. All `Next` labels in the normal
roadmap below describe **sequence only**, not current authorization. RP-A2 and
later normal-roadmap implementation remain unauthorized.

---

### RP-X0 Diagnostic Track — Stages 7–12

RP-X0 is an out-of-band diagnostic track, not a substitute numbering system for
normal RP-A/RP-B phases. Its current order is fixed in Section 0.6:

1. Preserve snapshot evidence and retain the completed O1/O2 Thermal contract/replay results without repeating them by default.
2. Keep PR #17 backend/API runtime status cleanup and PR #19 LCD/Web O4 merged; do not change risk policy.
3. In parallel, keep historical mmWave B closed. PR #22 imported the M-N9 FULL_INT8 selector; O3 still reports `MODEL_PENDING` until a later authorized status-contract update.
4. Stage 7 Mac-offline preparation is merged, including the M-N9 preflight selector contract (`RESOLVED_IN_CODE`). Defer only Pi deployment/execution.
5. Stage 9 tooling is merged; run its minimal live smoke only in the required hardware topology.
6. Later MR60-native model handoff and repository/team forward-port.

No RP-X0 stage authorizes RP-A2, model production selection, a mmWave live-gate
opening, or a Thermal live fall claim. Normal RP-A1 remains independently under
review; its merge authorization is outside this diagnostic track.

---

### RP-A0 — Current Pi Runtime / Persistence Audit

**Objective:** Freeze what the Pi actually does and what it currently loses, so later phases do not design against directory names.

**Why this phase exists:** Latest-state architecture cannot be extended blindly into B-complete windows.

**Preconditions:** Integration checkout on a known SHA.

**In scope:** Source audit, persistence table, AI default identification, Capture/state/SQLite separation.
**Out of scope:** Code changes, ESP changes, model activation.

**Current code affected:** None (read-only).
**Expected implementation:** This roadmap Markdown.

**Inputs:** Integration tree; PR #20 contracts.
**Artifacts:** This Markdown.

**Tests / validators:** Static documentation review only.

**Acceptance criteria:** End-to-end flow, storage table, v0.1.0 defaults, C-B6/T-B5/mmWave contracts, and four-layer storage split are recorded with evidence tags.
**Blocking conditions:** None remaining for documentation; field-encoding freeze waits for RP-A1 review.

**Owner role:** Pi runtime owner.
**Required reviewer:** Team integration reviewer.

**Evidence produced:** `CODE_VERIFIED` audit.
**Normal-roadmap next step (not an authorization):** RP-A1; current state is
implemented / under independent review, and its merge is not authorized by
RP-X0.
**Dependency class:** `PI_IMPLEMENTABLE_NOW` (docs already produced).

---

### RP-A1 — Capture v1 Schema and Session Contract

**Objective:** Freeze machine-readable Capture schema, session IDs, event types, timestamp fields, and Thermal payload index without writing production sessions.

**Why:** Writers without a frozen schema will drift.

**Preconditions:** RP-A0 accepted.

**In scope:** JSON schemas, fixtures, `.gitignore` for `captures/`, checksum rules.
**Out of scope:** Live writer on the receive path; ESP firmware; models.

**Current code affected:** `docs/`, future `capture/` schema modules, `.gitignore`.
**Expected implementation:** Schema + synthetic fixtures only.

**Inputs:** This roadmap’s Capture v1 section and current `storage/sensor_logger.py` behavior.
**Artifacts:** Schema files, example `manifest.json`, synthetic event lines, NPZ fixture.

**Tests:** Schema validation; path audit (no absolute paths); Git-ignore test.
**Validators:** Capture schema validator.

**Acceptance criteria:** A synthetic session validates; real payloads are not in Git.
**Blocking conditions:** Unresolved Thermal metadata encoding or CO₂ identity policy (identity may remain explicitly unavailable).

**Owner:** Pi runtime owner.
**Reviewer:** Team integration reviewer; AI owner for later replay fields.

**Evidence:** `TEST_VERIFIED` synthetic.
**Normal-roadmap next step (not an authorization):** RP-A2 — currently `NOT AUTHORIZED`.
**Dependency:** `PI_IMPLEMENTABLE_NOW`; CO₂ exact uniqueness remains `ESP_CONTRACT_DEPENDENCY` but must not block schema with an explicit unavailable field.

---

### RP-A2 — Generic Capture Writer and Storage Health

**Objective:** Session lifecycle, append-only JSONL, atomic rotation, health states, crash/unclean detection, queue isolation from receivers.

**Why:** Evidence is worthless if writes can fail silently.

**Preconditions:** RP-A1 schemas.

**In scope:** Writer, health on `/health`, rotation, fsync/rename for archives, unclean-session detection.
**Out of scope:** Sensor-specific semantics beyond generic events; model activation.

**Current code affected:** `backend/runtime.py` hook points; `storage/sensor_logger.py` (refactor toward Capture); `backend/app.py` health.

**Expected implementation:** New Capture writer; keep current logger until cutover tests pass, then retire hourly independent files.

**Tests:** append-only, rotation, disk-full, queue overflow, crash without close marker, dual health vs SQLite.
**Acceptance:** `capture_failed` cannot coexist with a healthy Capture claim; receive path never blocks on disk.

**Owner:** Pi runtime owner.
**Reviewer:** Team integration reviewer.
**Normal-roadmap next step (not an authorization):** RP-A3 — currently `NOT AUTHORIZED`.
**Dependency:** `PI_IMPLEMENTABLE_NOW`.

---

### RP-A3 — CO₂ / PIR Event Capture

**Objective:** Persist CO₂ observations and PIR transitions with transport and Pi timing.

**Why:** Slope and PIR supporting evidence cannot be reconstructed from 60 s files and SQLite snapshots.

**Preconditions:** RP-A2 writer healthy.

**In scope:** CO₂ events without 60 s downsample; PIR first-state + transitions; explicit invalid events.
**Out of scope:** C-B6 inference; inventing measurement IDs; ESP changes.

**Current code affected:** `backend/runtime.py` submit path; logger CO₂ gate must not apply to Capture; PIR currently omitted.

**Tests:** unique-vs-duplicate policy with `measurement_identity_unavailable`; PIR transition-only; invalid CO₂ persisted as error/invalid, not dropped.

**Acceptance:** A 1 Hz telemetry minute produces Capture evidence richer than one 60 s logger row; PIR edges are replayable.

**Owner:** Pi runtime owner / CO₂ owner for identity review.
**Reviewer:** CO₂ owner; team integration reviewer.
**Normal-roadmap next step (not an authorization):** RP-A4 — currently `NOT AUTHORIZED`.
**Dependency:** `PI_IMPLEMENTABLE_NOW` for transport observations; exact physical uniqueness `ESP_CONTRACT_DEPENDENCY`.

---

### RP-A4 — Thermal Full-Frame Capture

**Objective:** Persist every validated 80×62 raw frame losslessly with metadata; persist incomplete/CRC/timeout as error events, never as fake frames.

**Why:** T-B5 needs the full physical frame, not `thermal_max_c`.

**Preconditions:** RP-A2.

**In scope:** NPZ archives, payload references, CRC/SHA, monotonic time.
**Out of scope:** °C conversion, T-B5 inference, putting pixels in SQLite.

**Current code affected:** `gateway/thermal_udp.py` metrics→events; `storage/sensor_logger.py` NPZ path.

**Tests:** round-trip `(N,62,80) uint16`; incomplete frames absent from valid archives; atomic close.

**Acceptance:** Replay reads identical pixels; `/health` loss counters have matching Capture events.

**Owner:** Pi runtime owner / Thermal owner.
**Reviewer:** Thermal owner.
**Normal-roadmap next step (not an authorization):** RP-A5 — currently `NOT AUTHORIZED`.
**Dependency:** `PI_IMPLEMENTABLE_NOW`.

---

### RP-A5 — Canonical Replay Layer

**Objective:** Read Capture into a canonical observation stream with validators, without requiring B-models yet.

**Why:** Later AI activation must compare against evidence, not live luck.

**Preconditions:** RP-A3 and RP-A4 producing synthetic and (when hardware exists) real sessions.

**In scope:** Reader, ordering, checksum, PIR forward-fill, CO₂ identity-unavailable honesty.
**Out of scope:** Training admission; mmWave phase synthesis.

**Tests:** fixture replay; corrupt archive quarantine; unclean session visible.

**Acceptance:** Replay can answer source/transport/stale questions for CO₂, PIR, Thermal pixels.

**Owner:** Pi runtime owner.
**Reviewer:** Team integration reviewer.
**Normal-roadmap next step (not an authorization):** RP-B0 — currently `NOT AUTHORIZED`.
**Dependency:** `PI_IMPLEMENTABLE_NOW`.

---

### RP-B0 — AI Artifact Resolution and Contract Validator

**Objective:** Make B-complete artifacts resolvable, checksummed, and fail-closed **without** switching live defaults until the checklist passes.

**Why:** Historical v0.1.0 remains the runtime default. RP-X0 has artifact availability, but restored-integration-runtime discovery and production activation remain separate fail-closed checks.

**Preconditions:** RP-A5; approved locked-binary policy; B-complete files available to integration.

**In scope:** Candidate pointer, SHA-256 preflight, missing-artifact behavior, preprocessing-module identity, INT8 tensor contracts.
**Out of scope:** Live default switch before tests; retraining; ESP.

**Current code affected:** `ai/runtime.py`, `hil/preflight.py`, `deployment/verify_bundle.py`, `sources/ondevice_ai/` snapshot sync (copy, not edit of frozen contracts).

**Acceptance:** Preflight reports v0.1.0 vs B-complete identities separately; missing T-B5 is `MODEL_UNAVAILABLE`, not a crash into a random file.

**Owner:** AI owner + Pi runtime owner.
**Reviewer:** Team integration reviewer; AI owner.
**Normal-roadmap next step (not an authorization):** RP-B1 — currently `NOT AUTHORIZED`; C-B6 availability does not authorize activation.
**Dependency:** `AI_BASELINE_DEPENDENCY`, `MODEL_ARTIFACT_DEPENDENCY`, `OWNER_DECISION_REQUIRED`.

---

### RP-B1 — CO₂ B-Complete Runtime Integration

**Objective:** Compute frozen C-B6 features from the runtime buffer and run the INT8 occupancy model as **context**.

**Why:** Current humidity gate makes CO₂ AI permanently unavailable, and the 3-feature v0.1.0 contract is obsolete.

**Preconditions:** RP-A3, RP-B0 C-B6 artifact present; Capture running.

**In scope:** 150 s buffer, 90 s gap reset, scaler, `[1,2]` INT8, threshold 0.43, provenance.
**Out of scope:** Changing 1000/2500 ppm rules; humidity as input; fabricating measurement IDs.

**Current code affected:** `ai/pipeline.py` `_co2`; `ai/runtime.py`; do not reuse `CO2Interpreter.predict(slope, humidity, ppm)` unchanged.

**Tests:** warmup, gap restart, runtime/replay equivalence on fixtures, no VACANT from invalid input.

**Acceptance:** `CO2_RUNTIME_READY` criteria in §31; occupancy never silently drives alarm.

**Owner:** CO₂ owner + Pi runtime owner.
**Reviewer:** AI owner; risk owner.
**Normal-roadmap next step (not an authorization):** RP-B2 — currently `NOT AUTHORIZED`.
**Dependency:** `PI_IMPLEMENTABLE_NOW` given artifact; exact device-domain `HARDWARE_VALIDATION_DEPENDENCY` later.

---

### RP-B2 — Thermal B-Complete Runtime Integration

**Objective:** Run T-B5 on canonical full frames once unit/preprocessing/artifact gates pass.

**Why:** Current min-max v0.1.0 is a different model.

**Preconditions:** RP-A4, RP-B0 T-B5 binary on Pi with SHA `fa9730c2…`; AI-frozen preprocessing function; Thermal-44 unit/orientation either verified or explicitly deferred with `MODEL_UNAVAILABLE` rather than a guessed °C formula.

**In scope:** Canonical frame → frozen prep → INT8 `[1,62,80,1]`; class map preserved; posture-proxy labeling.
**Out of scope:** Rewriting `HUMAN_FALL` to `FALL_EVENT`; using `thermal_max_c`; unlimited RAM frames.

**Tests:** fixture INT8 invoke; missing binary; wrong SHA; preprocessing equivalence vs offline fixtures if available.

**Acceptance:** `THERMAL_RUNTIME_READY` if and only if unit conversion is approved **or** the runtime honestly stays `MODEL_UNAVAILABLE` while Capture continues.

**Owner:** Thermal owner + Pi runtime owner.
**Reviewer:** AI owner.
**Normal-roadmap sequence (not an authorization):** RP-B3 remains blocked; RP-B4 is also currently `NOT AUTHORIZED`.
**Dependency:** `MODEL_ARTIFACT_DEPENDENCY`, `HARDWARE_VALIDATION_DEPENDENCY` for Thermal-44 units, `OWNER_DECISION_REQUIRED` for preprocessing graph.

---

### RP-B3 — mmWave Integration after MR60-Native Model Handoff

**Objective:** After the AI authority supplies an approved MR60-native model,
add only that exact signal/timestamp/window/preprocessing/INT8 contract as
context.

**Why:** RP-X0 Stage 8/8.5 establishes that the historical B artifact uses a
different signal domain from live vendor `breathPhase`. Synthesis, resampling,
empirical amplitude scaling, and per-window normalization to force the old
model are forbidden.

**Preconditions:** An AI-authority handoff containing artifact identity/SHA,
input signal and shape, sample/timestamp/window contract, preprocessing and
normalization, class mapping, quantization, and threshold/output semantics;
plus separately authorized normal-roadmap prerequisites where applicable.

**In scope:** Future MR60-native samples/timing and the approved replacement
adapter only.
**Out of scope:** Activating M-B3 live, enabling v0.1.0, calling APNEA clinical,
or emergency decisions from unverified outputs.

**Acceptance:** `MMWAVE_RUNTIME_READY` only with the approved MR60-native
contract and targeted evidence; current status remains `MODEL_PENDING`.

**Owner:** mmWave AI owner + Pi runtime owner.
**Reviewer:** AI owner; team integration reviewer.
**Dependency:** `MR60_NATIVE_MODEL_HANDOFF_DEPENDENCY` — **blocked now**.

---

### RP-B4 — AI Provenance + SQLite/Capture Linkage

**Objective:** Every inference and risk row can name the evidence, model SHA, and preprocessing profile.

**Preconditions:** RP-A5; at least CO₂ or Thermal inference path.

**In scope:** `inference/` JSONL; SQLite linkage fields; replay comparison of stored vs recomputed outputs.
**Out of scope:** Dashboard redesign.

**Acceptance:** `AI_PROVENANCE_READY` and `REPLAY_READY` for activated sensors.

**Owner:** Pi runtime owner.
**Reviewer:** Team integration reviewer.
**Dependency:** `PI_IMPLEMENTABLE_NOW`.

---

### RP-C0 — Risk Engine Context Integration

**Objective:** Consume B-model outputs as named context without confusing AI class, physical threshold, health, and risk state.

**Why:** Occupancy and lying-proxy can be misread as enclosure danger or fall events.

**Preconditions:** RP-B1 and/or RP-B2 producing context; V4 weights/thresholds unchanged unless a separate policy PR exists (not this roadmap).

**In scope:** Explicit metadata, reasons, fail-closed when AI unavailable, review of Thermal emergency override semantics.
**Out of scope:** Silent threshold edits; dashboard UX redesign.

**Tests:** missing sensor, stale, AI unavailable, invalid input, emergency override still requires documented conditions.

**Owner:** Risk owner.
**Reviewer:** Team integration reviewer; Thermal/CO₂ owners for semantics.
**Dependency:** `OWNER_DECISION_REQUIRED` for whether occupancy context affects fusion; `PI_IMPLEMENTABLE_NOW` for plumbing.

---

### RP-C1 — Fault Injection / Fail-Closed Validation

**Objective:** Prove invalid/missing/stale/model-missing paths never become normal-class inferences or healthy Capture.

**Preconditions:** RP-A2, RP-C0 plumbing.

**In scope:** Tests extending `tests/test_ai_pipeline.py`, logger/Capture failures, risk unavailable paths.
**Out of scope:** Hardware soak.

**Owner:** Pi runtime owner.
**Reviewer:** Team integration reviewer.
**Dependency:** `PI_IMPLEMENTABLE_NOW`.

---

### RP-C2 — Pi Performance / Long-Run Validation

**Objective:** Measure CPU, RAM, disk, Thermal write, inference latency, queue depth, temperature, multi-hour run, restart recovery.

**Preconditions:** Capture + at least one activated model or an explicit model-unavailable long-run.

**In scope:** Instrumentation and a written evidence report.
**Out of scope:** Claiming pass from this document.

**Owner:** Pi runtime owner.
**Reviewer:** Team integration reviewer.
**Dependency:** `HARDWARE_VALIDATION_DEPENDENCY`.

---

### RP-D0 — Real-Device Phase C Orchestration

**Objective:** Run separate MR60, SCD40, and Thermal device-domain protocols against Capture+replay+activated contracts.

**Preconditions:** `PI_CAPTURE_READY`; relevant `*_RUNTIME_READY`; replay of the same sessions.

**In scope:** Orchestration, evidence packing, fail-closed reporting.
**Out of scope:** Declaring production safety; dataset admission by default.

**Owner:** Sensor owners + Pi runtime owner.
**Reviewer:** AI owner; team integration reviewer.
**Dependency:** `HARDWARE_VALIDATION_DEPENDENCY`, `MR60_NATIVE_MODEL_HANDOFF_DEPENDENCY` for future mmWave AI.

---

### RP-D1 — Multisensor Deployment Reproduction Gate

**Objective:** A second machine/operator can provision Pi, verify artifact SHAs, start Capture, run models that are activated, and replay a session to matching hashes.

**Preconditions:** RP-C2 evidence; RP-D0 per-sensor reports as applicable.

**Acceptance:** `FINAL_RUNTIME_REPRODUCIBLE` — not merely “models run without crashing.”

**Owner:** Team integration reviewer.
**Dependency:** `OWNER_DECISION_REQUIRED` for what “final” means in the competition setting; remaining hardware gates stay explicit.

---

## 27. Dependency Matrix

| Requirement | Pi can implement now | External dependency | Blocking phase | Owner/reviewer |
|---|---:|---|---|---|
| Capture schema/writer/health | Yes | — | RP-A1/A2 | Pi runtime / integration reviewer |
| CO₂ transport observation Capture | Yes | Exact uniqueness needs ESP identity | RP-A3 | Pi / CO₂ / ESP |
| PIR transition Capture | Yes | — | RP-A3 | Pi runtime |
| Thermal full-frame Capture | Yes | — | RP-A4 | Pi / Thermal |
| Replay of raw evidence | Yes after A3/A4 | — | RP-A5 | Pi runtime |
| B-complete snapshot in integration | Artifacts synchronized/provisioned for RP-X0 diagnostics; production selection remains no | Exact restored-integration-tree verification | RP-X0 Stage 7/9, then RP-B0 | AI owner / Pi runtime |
| C-B6 INT8 on Pi | After snapshot | Approved artifact policy; later team forward-port is separate | RP-B1 | AI / CO₂ / Pi |
| T-B5 INT8 on Pi | Artifact available/provisioned and isolated load/invoke observed | Restored integration runtime must discover it at startup; Thermal input remains incompatible | RP-X0 Stage 9 / RP-B2 | AI / Thermal / Pi runtime |
| Thermal-44 °C / orientation | No | Device-domain contract | RP-B2 / D0 | Thermal owner |
| C-B6 slope exact replay | Partial | ESP measurement ID/time | RP-B1 / D0 | ESP / CO₂ |
| mmWave phase Capture/inference | Stage 8/8.5 complete; telemetry/status may remain available, live inference no | MR60-native AI artifact and complete replacement contract | Future AI handoff / RP-B3 | mmWave AI owner |
| Occupancy→risk fusion change | Plumbing yes | Policy decision | RP-C0 | Risk owner |
| V4 threshold/weight change | Not in this roadmap | Policy PR | deferred | Risk owner |
| Pi long-run metrics | No | Hardware | RP-C2 | Pi runtime |
| Phase C device-domain | No | Hardware + Capture/replay | RP-D0 | Sensor owners |
| Dashboard changes | No | Out of scope | — | Dashboard owner |

The detailed normal-roadmap labels in older phase descriptions remain historical
planning context. For every current execution decision, use the single §0
dependency vocabulary instead: `MAC_OFFLINE_READY`, `SENSOR_REQUIRED`,
`PI_REQUIRED`, `SENSOR_AND_PI_REQUIRED`, and `EXTERNAL_AI_DEPENDENCY`.

---

## 28. Ownership Matrix

Person names are not assigned. Roles only.

| Area | Owner role | Reviewer |
|---|---|---|
| Pi receiver/state/runtime | Pi runtime owner | Team integration reviewer |
| Capture/replay/storage health | Pi runtime owner | Team integration reviewer |
| ESP/device telemetry identity | ESP/device owner | Pi runtime owner |
| B-complete contracts/artifacts | AI owner | Team integration reviewer |
| mmWave phase contract | mmWave owner | AI owner |
| CO₂ slope/SCD40 domain | CO₂ owner | AI owner |
| Thermal-44 unit/orientation | Thermal owner | AI owner |
| Risk fusion semantics | Risk owner | Team integration reviewer |
| Dashboard/LCD/HMI | Dashboard owner | Out of scope here except health fields |
| Competition reproduction gate | Team integration reviewer | Sensor owners |

---

## 29. Validation Matrix

| Domain | Future tests |
|---|---|
| Capture | Append-only; rotation; session lifecycle; checksums; power-loss/unclean; disk-full; queue overflow; dual health |
| CO₂ | Unique event history when IDs exist; identity-unavailable honesty; 150 s slope reproduction; warmup; 90 s gap; runtime/replay equivalence |
| Thermal | UDP reassembly; lossless frame persist; no partial-as-valid; preserve the verified named-snapshot conversion/P1 contract; device-domain validation, prep equivalence, and INT8 review |
| mmWave | Blocked until device-contract gate; then cadence, window continuity, gap, BPF/Z-score equivalence |
| AI | Artifact checksum; input contract; model resolution; Float/INT8 expected behavior; runtime/replay equivalence |
| Risk | Missing/stale/AI unavailable/invalid input; emergency override; occupancy not silently alarming |
| Pi | Latency, CPU, RAM, disk, temperature, long-run, restart recovery — RP-C2 only |

---

## 30. Risk Register

Severity assigned from source evidence, not from the prompt’s guess list alone.

| ID | Finding | Sev | Evidence |
|---|---|---|---|
| R1 | Historical `v0.1.0` remains runtime default; B-complete is not loaded | P0 | `CODE_VERIFIED` `models.yaml` / `model_manifest.json`; `OWNER_REPORTED` PR #20 |
| R2 | T-B5 is artifact-available, but production adapter activation and device-domain validation remain blocked by the Thermal AI path | P0 | O2.6 `INT8_QUANTIZATION_REVIEW_REQUIRED`; sheepmeat/test PR #99 `TRAIN_DOMAIN_RANGE_GAP` |
| R3 | Integration `sources/ondevice_ai/` is 2026-08-13 snapshot, missing C-B6/T-B5 lock | P0 | `CODE_VERIFIED` vs PR #20 |
| R4 | No sessionized raw Capture; current logger downsamples CO₂ and omits PIR | P0 | `CODE_VERIFIED` |
| R5 | CO₂ physical identity is observed in the RP-X0 reference runtime, but not yet reconciled as a Capture v1 contract | P0 | `OBSERVED` / `BLOCKED_DEPENDENCY` |
| R6 | Current CO₂ AI requires humidity and `[1,3]` softmax; C-B6 is `[1,2]` logistic without humidity | P0 | `CODE_VERIFIED` / `DOCUMENTED_ONLY` |
| R7 | Current Thermal AI is per-frame min-max v0.1.0; T-B5 is Celsius + P1 z-score + different quant | P0 | `CODE_VERIFIED` / `DOCUMENTED_ONLY` |
| R8 | Historical mmWave B input is blocked by a confirmed live-vendor-to-training-signal contract mismatch; MR60-native model is pending | P0 | `OBSERVED` / `BLOCKED_DEPENDENCY` |
| R9 | Historical raw-unit uncertainty is resolved for the named MI48 snapshot; live device-domain performance/orientation claims remain unvalidated | P1 | O1/O2 replay; `SENSOR_REQUIRED` / `EXTERNAL_AI_DEPENDENCY` |
| R10 | Preprocessing duplication risk across offline, Pi `ai/pipeline.py`, and future replay | P1 | `INFERRED` from three existing Thermal/CO₂ paths |
| R11 | Stale-vs-current: CO₂ usable value can lag 60 s while packets still refresh communication | P1 | `CODE_VERIFIED` — correct for health, dangerous if treated as 1 Hz samples |
| R12 | Model checksum at load for v0.1.0 exists, but not per-inference provenance in SQLite | P1 | `CODE_VERIFIED` |
| R13 | Capture/logger failure is counter-only; silent loss possible while runtime looks live | P1 | `CODE_VERIFIED` |
| R14 | SQLite not linked to evidence | P2 | `CODE_VERIFIED` schema |
| R15 | `HUMAN_FALL` emergency override can treat a lying proxy as DANGER | P1 | `CODE_VERIFIED` risk engine + T-A4/T-B5 limitations |
| R16 | C-B6 `OCCUPIED` could be mistaken for enclosure danger | P1 | `DOCUMENTED_ONLY` class_map `risk_semantic: NONE` |
| R17 | mmWave v0.1.0 class collapse if someone bypasses `deployment_allowed` | P1 | `CODE_VERIFIED` manifest |
| R18 | Experimental `preprocessing/mmwave.py` is not the locked runtime path unless RP-B3 binds it | P2 | `CODE_VERIFIED` |
| R19 | `captures/` not yet gitignored | P2 | `CODE_VERIFIED` `.gitignore` |
| R20 | Documentation snapshot SHAs in README/`LATEST_SOURCE_PROVENANCE.json` predate PR #20 | P3 | `DOCUMENTED_ONLY` |
| R21 | Pi long-run / Phase C not done | P2 | `OWNER_REPORTED`; not claimed here |
| R22 | A teammate may temporarily own the Pi team backend/ports, blocking only Stage 7 Pi execution | P1 | `OBSERVED`; do not interrupt; Stage 7 offline preparation remains `MAC_OFFLINE_READY` |

---

## 31. Definition of Done

`FINAL_DEPLOYMENT_READY` is **not** “models run without crashing.”

| Gate | Objective criteria |
|---|---|
| `PI_CAPTURE_READY` | Session manifest, append-only events, Thermal NPZ, close/unclean detection, visible Capture health, Git has no real payloads, PIR transitions + CO₂ observations stored, mmWave phase not fabricated |
| `CO2_RUNTIME_READY` | Unique-or-honestly-unidentified events; 150 s endpoint slope; 90 s gap reset; C-B6 SHA `c5969b36…`; `[1,2]` INT8; threshold 0.43; no humidity; no inference on invalid/warmup; occupancy is context only |
| `THERMAL_RUNTIME_READY` | Full-frame Capture round-trip; approved unit/orientation **or** explicit `MODEL_UNAVAILABLE`; T-B5 SHA `fa9730c2…` verified; frozen prep shared with replay; `HUMAN_FALL` labeled as posture proxy |
| `MMWAVE_RUNTIME_READY` | Approved MR60-native artifact/SHA; explicit live input signal, timestamp/window, preprocessing/normalization, class/quantization/output contract; targeted evidence; no historical-B signal coercion |
| `AI_PROVENANCE_READY` | Per-inference IDs, model SHA, prep profile, source references, stored separately from SQLite blobs |
| `REPLAY_READY` | Capture validator + same prep + same artifact reproduces stored outputs or explains differences |
| `RISK_CONTEXT_READY` | AI class, ppm/rpm thresholds, health, and risk state are distinct in API/SQLite; fail-closed paths tested |
| `PI_LONG_RUN_READY` | RP-C2 measurements recorded, not inferred |
| `REAL_DEVICE_VALIDATION_READY` | Per-sensor Phase C reports exist; Capture of those runs is replayable |
| `FINAL_RUNTIME_REPRODUCIBLE` | Second provisioned Pi matches artifact SHAs, Capture schema, and replay hashes for a declared session set |

---

## 32. Deferred Work

- Stage 7 Pi deployment/execution when the Pi becomes available; team forward-port follows verified integration work
- CO₂ identity/Capture contract beyond RP-X0 reference observation
- MR60-native dataset/model/preprocessing/quantization contract from AI authority; historical B gate remains closed
- Thermal device-domain data/validation and the approved AI response to `TRAIN_DOMAIN_RANGE_GAP`
- Thermal T-B5 production adapter/process discovery only after approved AI handoff and Pi deployment boundary
- Switching runtime defaults away from v0.1.0
- Changing V4 risk weights/thresholds or Thermal emergency override
- Dashboard behavior
- Treating Capture as training data
- Clinical apnea or verified fall-event claims
- XIAO ESP32-C6 firmware port
- Any implementation listed in RP-A1 through RP-D1 until separately authorized

---

## Appendix A. Sensor Contract Matrix

| Sensor | Device→Pi data | Required model input | Pi-derived data | Persistent storage | Rolling state | Current readiness |
|---|---|---|---|---|---|---|
| CO₂ | 2026 audit: scalar ppm/valid/seq/uptime; RP-X0 later observed boot and measurement provenance | `[CO2, CO2_slope]` INT8 `[1,2]` | Endpoint slope from 150 s source history | Unique events (or honest transport observations) | ~150 s + gap margin | RP-X0 C-B6 observed; Capture identity contract remains blocked |
| Thermal | UDP 80×62 `uint16` BE full frame | Canonical `(62,80)` physical/prep → INT8 `[1,62,80,1]` | Verified named-snapshot 0.1 K conversion + P1/T-B5 replay | JSONL metadata + NPZ frames | Latest frame only | Transport: observed. T-B5 artifact: ready. Production remains blocked by AI/device-domain validation. |
| mmWave | RP-X0 observed vendor `breath_phase`; any other runtime observation must be attributed separately | Future MR60-native input contract; historical B used 300-sample `BPF_ZSCORE` complex phase | Historical artifact and live vendor signal mismatch | Diagnostic JSONL preserved externally; Capture remains separately gated | Telemetry latest state; no historical-B AI window | `MODEL_PENDING_MR60_NATIVE_RETRAINING`; live B blocked |
| PIR | 1 Hz boolean | None | No-motion elapsed for risk | Transitions | Latest + timer | State ready; Capture missing |

## Appendix B. Storage Responsibility Matrix

| Data | RAM | Capture | SQLite | Derived/replay | Reset policy |
|---|---:|---:|---:|---:|---|
| Raw CO₂ observation | Latest + history refs | Yes | Summary ppm | Slope replay | State: newer valid. Buffer: continuity. Capture: retention |
| CO₂ slope | Yes, derived | No (recompute) | Optional summary | Replay from events | With buffer |
| Thermal frame | Latest only | Yes, NPZ | max raw / AI class only | Prep+infer | Latest replaced; Capture retained |
| PIR transition | Latest + timer | Yes, edges | Snapshot bool | Forward-fill | Timer resets on motion/unavailable |
| Future mmWave phase sample | 300-window later | Yes after gate | No waveforms | BPF window | Continuity |
| AI tensor | Transient | Optional hash | No | Replay | Per inference |
| AI output | Latest | Inference JSONL | Summary + IDs | Compare | Latest replaced |
| Risk result | Latest | Link IDs | Snapshots + events | Replay reasons | Independent of AI buffer |
| Alarm transition | Latch | Optional event | `risk_events` + emergency fields | — | Latch policy unchanged here |

## Appendix C. Current Code Reuse

| Area | Disposition | Reason |
|---|---|---|
| `gateway/receiver.py`, `protocol.py` | PRESERVE | Strict TCP v1 is sound |
| `gateway/thermal_udp.py` | EXTEND | Add Capture loss events from metrics |
| `state/manager.py` | EXTEND | Keep freshness split; do not make it a Capture store; later expose event refs |
| `storage/sensor_logger.py` | REFACTOR → REPLACE_LATER | Useful async writer; wrong contract |
| `database/` SQLite | PRESERVE + EXTEND | Operational summary; additive linkage |
| `ai/runtime.py` | EXTEND | Candidate resolution + checksum + missing artifact |
| `ai/pipeline.py` | REFACTOR | C-B6/T-B5 contracts; stop humidity gate for B path; no phase synthesis |
| `sources/ondevice_ai/inference/*` | PRESERVE v0.1.0 until activation; wrap or replace call sites for B adapters | Do not silently retarget |
| `risk/engine.py` | EXTEND | Context plumbing; do not change weights/thresholds here |
| `backend/` API | PRESERVE + EXTEND health | No dashboard redesign |
| `deployment/run_pi.sh`, preflight | EXTEND | Artifact provisioning/checksum |
| `web/dashboard/` | PRESERVE | Out of scope |
| Full rewrite of Pi | Rejected | Architecture is usable; gaps are persistence, contracts, and activation |

## Appendix D. Storage sizing formulas

Use only known cadences. Unknown rates stay symbolic.

The 2026-08-16 source-audit scalar cadence was **1.0 Hz** `CODE_VERIFIED`.
RP-X0 later observed an approximately 10 Hz diagnostic stream; Stage 8/8.5
reviewed that evidence, but it remains runtime/firmware/topology scoped and
does not make vendor phase a generic historical-B model input.
Thermal requested ≈ **6.25 FPS** `CODE_VERIFIED` as firmware request, not as measured Pi FPS.

| Stream | Formula | 1-hour order of magnitude |
|---|---|---|
| CO₂ Capture if 1 Hz transport observations | `bytes/event × 1 × 3600` | ~0.4–0.7 MB at 120–200 B/event |
| CO₂ if unique SCD40 ~0.2 Hz later | `bytes/event × 0.2 × 3600` | ~0.09–0.14 MB |
| PIR transitions | `bytes/event × transitions/hour` | typically ≪ 0.1 MB if edges only |
| PIR if naively 1 Hz | same as 1 Hz JSONL | ~0.4–0.7 MB; not recommended |
| Thermal raw uncompressed | `9920 B/frame × FPS × 3600` | at 6.25 FPS ≈ **223 MB/h** before NPZ compression |
| mmWave scalars 1 Hz | ~180–250 B × 3600 | ~0.65–0.90 MB/h `DOCUMENTED_ONLY` current logger estimate |
| Future mmWave phase | `bytes/sample × samples/sec × duration` | unknown until cadence verified; if 10 Hz float32: `4 × 10 × 3600 ≈ 144 kB/h` samples only |
| Inference metadata | `bytes/inference × inferences/hour` | at 15 s evaluation: 240/h, typically ≪ 1 MB |

Thermal dominates. Retain by closed session against a disk budget; do not promise a fixed day count without measuring NPZ compression on device. Current logger quotas (10 GB total / 8.5 GB Thermal / 2 GB free reserve) are **not** the Capture v1 retention commitment. `CODE_VERIFIED` current defaults

---

## Appendix E. Authorization stamp

```text
Roadmap status:         RP-A1_IMPLEMENTED_UNDER_INDEPENDENT_REVIEW / RP-X0_FIELD_STATE_CORRECTIVE
Pi Capture code:        NO
Pi runtime modification: NO
ESP firmware:           NO
Model activation:       NO
models.yaml changes:    NO
Preprocessing changes:  NO
Risk Engine changes:    NO
Dashboard changes:      NO
Phase C execution:      NO
Hardware testing by this document: NO
Historical RP-X0 scoped field evidence: YES (recorded; not re-executed here)
RP-A1 current state:    IMPLEMENTED / UNDER INDEPENDENT REVIEW
```

Recommended next action: preserve the snapshot read-only.
`STAGE7_PREFLIGHT_MMWAVE_SELECTOR_DRIFT = RESOLVED_IN_CODE`. Further Mac RP-X0
implementation is not required. Stage 9 tooling preparation is
`IMPLEMENTED / MERGED` (PR #21); do not treat that as live smoke. Remaining
boundary: Stage 7 actual Pi execution = `PI_REQUIRED / NOT_RUN`; Stage 9 live
smoke = `SENSOR_AND_PI_REQUIRED / NOT_RUN`. Merge authorization for RP-A1 and
every normal-roadmap implementation authorization remain separate decisions.
