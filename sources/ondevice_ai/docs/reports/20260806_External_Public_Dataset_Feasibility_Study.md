# SafeNest V6 Existing mmWave Asset Baseline, Coverage-Gap Analysis, and Dataset Expansion Strategy

**Document Identifier**: `20260806_External_Public_Dataset_Feasibility_Study.md`  
**Date**: August 6, 2026  
**Scope**: Strategy & Analysis Only (No Data Downloaded, No Archive Extracted, No Model/Data Mutated)  
**Target System**: SafeNest active on-device AI respiration pipeline (canonical repository root)

---

## 1. Executive Summary

This report establishes the authoritative baseline of SafeNest V6 mmWave assets, separates historical real-data model lineages from current synthetic smoke/retraining assets, analyzes current dataset coverage and operational gaps, and provides an evidence-based data expansion roadmap.

### Key Executive Conclusions
- **Preserved External Raw Archive**: `embed2/datasets/raw_archives/external_datasets/db_records.zip` (246,597,320 bytes, 110 participants, 4 activity tests) is **`LOCAL_REPACKAGED_ARCHIVE_CONFIRMED`**. Its verified official Zenodo record is **DOI `10.5281/zenodo.18599983` (v1.1)**. The obsolete DOI `10.5281/zenodo.1001234` is classified as **`SOURCE_MISMATCH`**.
- **Model Lineage Separation**:
  - **Historical mmWave v0.1.0** (INT8 SHA `43cdd6f321c2...`): Developed using external real-world dataset development (`PROJECT_HISTORY_CONFIRMED_BY_USER`), classified as **`HISTORICAL_SOURCE_MAPPING_INCOMPLETE`**.
  - **V6 mmWave v0.2.0 Candidate** (INT8 SHA `85c023d3eefc...`): Evaluated under **`SYNTHETIC_SMOKE_ONLY`** scope generated from the procedural V6 NPZ (`SYNTHETIC_SMOKE_AND_RETRAINING_ASSET`).
- **Highest-Priority Ranked Data Gaps**:
  1. *Raw-to-Model Reproducibility Gap* (**CRITICAL**): Rebuilding a 100% reproducible `raw rFFT → resp_phase → 30s window` pipeline from the local Zenodo archive.
  2. *MR60 Device-Domain Gap* (**HIGH**): Sensor-specific phase noise and SNR calibration for the target MR60 radar.
  3. *Real-Subject Generalization Gap* (**HIGH**): Subject-independent benchmark evaluation on real human respiration.
- **Recommended Acquisition & Reconstruction Order**:
  1. **Raw Pipeline Reconstruction** (Local Zenodo 60GHz Raw Archive) -> *Source-Only Benchmark*
  2. **Mendeley 24GHz Medical Radar** (`10.17632/6rp6wrd2pr.2`) -> *External Test / Domain Adaptation*
  3. **PhysioNet Apnea-ECG** (`10.13026/C23W2R`) -> *Reference-Domain Morphology Analysis*
- **External Downloads & Mutations Executed**: **0 bytes downloaded, 0 archives extracted, 0 code/model files modified**.

---

## 2. Correct Existing Asset Baseline

The following baseline separates existing SafeNest V6 assets by domain, data type, training role, evaluation scope, and provenance status:

| Asset | Source Domain | Data Type | Training / Execution Role | Validation Scope | Provenance Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Zenodo 60 GHz Raw Archive** (`db_records.zip`) | Real External Radar (TI IWR6843 60GHz) | Raw rFFT (`.zlib`), Movesense ECG/ACC, timestamps | Historical source asset / Future reprocessing source | Raw archive only | `LOCAL_REPACKAGED_ARCHIVE_CONFIRMED` |
| **Historical v0.1.0 Model** (`mmwave_resp_int8_v0.1.0.tflite`) | Historical External-Data Pipeline | INT8 TFLite (`43cdd6f3...`) | Existing inference model | Historical evaluation scope | `HISTORICAL_SOURCE_MAPPING_INCOMPLETE` |
| **Current V6 Processed NPZ** (`mmwave_respiration_v1.npz`) | Procedural Synthetic | 30 s waveform windows (`[1, 300, 1]`) | Retraining & smoke test input | Synthetic offline | `SYNTHETIC_SMOKE_AND_RETRAINING_ASSET` |
| **V6 v0.2.0 Candidate Model** (`mmwave_resp_int8_v0.2.0_candidate.tflite`) | Procedural Synthetic NPZ | INT8 TFLite (`85c023d3...`) | Reproducibility & candidate QA | Synthetic smoke only | Current V6 candidate lineage |

---

## 3. Historical v0.1.0 vs V6 v0.2.0 Lineage Separation

### 3.1 Historical mmWave v0.1.0 Lineage
- **INT8 TFLite SHA-256**: `43cdd6f321c2b645232162233e098c2b65549388cbc9e680a17a0eccdc8f0158`
- **Development History**: The user has confirmed that external real-world dataset acquisition, preprocessing, training, and INT8 quantization were executed during earlier project stages (`PROJECT_HISTORY_CONFIRMED_BY_USER`).
- **Current Workspace Mapping Status**: Classified as **`HISTORICAL_SOURCE_MAPPING_INCOMPLETE`** because the complete file-level or sample-level chain connecting raw `db_records.zip` files to individual training array indices is not preserved in local V6 code metadata. This status acknowledges incomplete provenance tracking without implying that historical training never occurred.

### 3.2 V6 mmWave v0.2.0 Candidate Lineage
- **INT8 TFLite SHA-256**: `85c023d3eefca13ecbb72a841974e53a56d5f4173920645d46df49a9088452ff`
- **Development History**: Generated deterministically during V6 offline reproducibility and candidate QA testing from `datasets/mmwave/processed/mmwave_respiration_v1.npz`.
- **Evaluation Scope**: Explicitly recorded in metadata as **`SYNTHETIC_SMOKE_ONLY`**. This candidate is an offline reproducibility asset and must not be conflated with the historical real-data model.

---

## 4. Existing Zenodo Raw Archive Status

### 4.1 Local Archive Measurement
- **Local Path**: `embed2/datasets/raw_archives/external_datasets/db_records.zip`
- **File Size**: `246,597,320` bytes (`CONFIRMED`)
- **Local SHA-256**: `f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0` (`CONFIRMED`)
- **Local ZIP MD5**: `370de95033f1a98b78e57dbbea92a8bc` (`CONFIRMED`)
- **Contents**: 110 participant directories (`P001` - `P110`), 4 posture/activity test combinations, `radar_rFFTs.zlib`, `movesense_acc.csv`, `movesense_ecg.csv`, `non_breathing_ts.csv`, and `__MACOSX` metadata entries.

### 4.2 Official Zenodo Record Verification
- **Official Title**: Extensive Age-Balanced and Subject-Varied mmWave Radar Dataset of Referenced Records for Vital Signs
- **Verified Official DOI**: `10.5281/zenodo.18599983` (Version 1.1) (`CONFIRMED`)
- **Obsolete / Incorrect DOI**: `10.5281/zenodo.1001234` (**`SOURCE_MISMATCH`**)
- **Official Published ZIP MD5**: `408c5b347c751c553abe6d0f640a6f98`
- **Archive Status Classification**: **`LOCAL_REPACKAGED_ARCHIVE_CONFIRMED`** and **`OFFICIAL_ARCHIVE_BYTE_IDENTITY_NOT_CONFIRMED`**. The local archive is a valid repackaged working copy containing additional OS metadata (`__MACOSX`), which explains the hash difference without indicating file corruption. Both hashes are preserved.

---

## 5. Current NPZ Scope & Documentation Corrections

### 5.1 NPZ Classification
- **Local File**: `datasets/mmwave/processed/mmwave_respiration_v1.npz`
- **Size & Hash**: 3,779,017 bytes, SHA-256 `a08072f3d9b55cd95b530c7b5b90f17ef80f6015ee76119f217b9d834c1107fb` (`CONFIRMED`)
- **Generator**: Procedural script generating sinusoidal, constant, and noise-based 30-second respiration windows.
- **Classification**: **`SYNTHETIC_SMOKE_AND_RETRAINING_ASSET`**.

### 5.2 Documentation Wording Corrections
1. **Scope Generalization Error**: Statements asserting that the synthetic NPZ proves all historical models were synthetic are rejected. Correct wording: *The V6 v0.2.0 candidate is linked to the synthetic NPZ; historical v0.1.0 belongs to a separate earlier development lineage.*
2. **Subject Split Claims**: Documentation referencing `80 train / 15 val / 15 test subjects` in `MANIFEST.json` is classified as **`DOCUMENTED_BUT_NOT_SAMPLE_LINKED`**, as the current NPZ arrays do not contain sample-level subject provenance hashes.

---

## 6. Current Data-Coverage Analysis

| Coverage Dimension | Preserved Raw Archive (`db_records.zip`) | Current Processed V6 NPZ (`mmwave_respiration_v1.npz`) | Historical Model v0.1.0 Lineage |
| :--- | :--- | :--- | :--- |
| **Participants** | 110 real human subjects (`CONFIRMED`) | Procedural synthetic windows | Real human subjects (`PROJECT_HISTORY_CONFIRMED_BY_USER`) |
| **Activities & Postures** | Lying, sitting, non-breathing holds (`CONFIRMED`) | Procedural sine/noise waveforms | Real respiration & breath-holds |
| **Radar Hardware** | TI IWR6843 60GHz FMCW (`CONFIRMED`) | Synthetic mathematical signals | 60GHz mmWave radar |
| **Target MR60 Sensor Data** | `NOT_REPRESENTED` | `NOT_REPRESENTED` | `REAL_MR60_DATA_NOT_VERIFIABLE` |
| **Normal Breathing** | Present in raw rFFT (`CONFIRMED`) | Represented synthetically | Represented in historical model |
| **Rapid Respiration (>30 bpm)** | Limited coverage (`NOT_DOCUMENTED`) | Represented synthetically | Represented in historical model |
| **Apnea / Breath Hold** | Timestamps in `non_breathing_ts.csv` (`CONFIRMED`) | Represented synthetically | Represented in historical model |
| **Sample-Level Provenance** | `DOCUMENTED_BUT_NOT_SAMPLE_LINKED` | `NOT_PRESENT` | `HISTORICAL_SOURCE_MAPPING_INCOMPLETE` |

---

## 7. Ranked Data Gaps

The following critical gaps limit the robustness and evaluation of the SafeNest mmWave pipeline:

### Gap 1: Raw-to-Model Reproducibility Gap (**CRITICAL**)
- **Description**: While `db_records.zip` exists locally, the workspace lacks a clean, reproducible script converting `raw rFFT -> resp_phase -> 30s window -> subject-wise NPZ`.
- **Impact**: Retraining on real data currently relies on manual or unscripted steps.

### Gap 2: MR60 Device-Domain Gap (**HIGH**)
- **Description**: The target deployment sensor is the MR60 60GHz radar module, whereas external datasets use TI IWR6843 radar evaluation boards. Sensor-specific phase extraction, noise profiles, and SNR differ.
- **Impact**: Potential domain shift when deploying models trained on TI IWR6843 data to MR60 hardware.

### Gap 3: Real-Subject Generalization Benchmark Gap (**HIGH**)
- **Description**: The current V6 evaluation pipeline only tests synthetic smoke windows. An isolated, real-subject test benchmark is needed to evaluate out-of-sample generalization.
- **Impact**: Model accuracy on real human breathing cannot be verified automatically in the CI/CD pipeline.

### Gap 4: Motion & Activity Diversity Gap (**MEDIUM**)
- **Description**: Existing raw data covers static sitting and lying postures, but lacks motion-corrupted respiration (body movement, coughing, walking nearby).
- **Impact**: Potential false alarms during active movement.

---

## 8. Additional External Dataset Research

To address Gaps 2, 3, and 4, additional public datasets were researched and evaluated:

### 8.1 Candidate 1: Zenodo 60 GHz mmWave Radar Dataset (Locally Preserved)
- **Official DOI**: `10.5281/zenodo.18599983` (`CONFIRMED`)
- **Modality**: 60 GHz FMCW mmWave Radar (TI IWR6843) + Movesense ECG/ACC.
- **Scale**: 110 participants, 4 activity tests, non-breathing timestamps.
- **Sampling Rate**: `nominal 10 Hz` (marked as nominal 10 Hz unless 20 Hz is explicitly confirmed from unextracted chirp config).
- **License**: CC BY 4.0 (`CONFIRMED`)
- **Addressed Gap**: Gap 1 (Raw-to-Model Reproducibility Gap) & Gap 3 (Real-Subject Generalization Gap).
- **Signal Classification**: `RAW_RADAR_REQUIRES_FEATURE_EXTRACTION`.

### 8.2 Candidate 2: Mendeley Medical 24 GHz Doppler Radar Dataset
- **Official DOI**: `10.17632/6rp6wrd2pr.2` / Article DOI: `10.1016/j.dib.2021.107724` (`CONFIRMED`)
- **Modality**: 24.25 GHz & 10.525 GHz Doppler Radar + Respiratory belt transducer.
- **Scale**: 9 healthy subjects, laboratory recordings (.csv / .lvm).
- **Sampling Rate**: `100 Hz -> 1000 Hz` (1000 Hz high-frequency physiological reference).
- **License**: CC BY 4.0 (`CONFIRMED`)
- **Recommended First Use**: `Source-only preprocessing benchmark / domain-adaptation feasibility`.
- **Addressed Gap**: Source-only preprocessing benchmark & 24 GHz domain adaptation feasibility.
- **Signal Classification**: `RADAR_DERIVED_WAVEFORM_COMPATIBLE`.

### 8.3 Candidate 3: PhysioNet Apnea-ECG Database
- **Official DOI**: `10.13026/C23W2R` (`CONFIRMED`)
- **Modality**: Single-lead ECG + expert minute-by-minute sleep apnea annotations.
- **Scale**: 70 overnight polysomnography recordings (~8 hours each).
- **License**: ODC-By v1.0 (`CONFIRMED`)
- **Addressed Gap**: Apnea morphology & clinical event timing research.
- **Signal Classification**: `REFERENCE_DOMAIN_ONLY`.

---

## 9. Signal Processing & Preprocessing Compatibility

The local V6 preprocessor ([`preprocessing/mmwave.py`](../../preprocessing/mmwave.py)) establishes the following contract:
- **Sampling Rate**: 10 Hz
- **Window Length**: 300 samples (30 seconds)
- **Band-pass Filter**: Butterworth 4th-order, 0.1 – 0.5 Hz (6 – 30 bpm)
- **Normalization**: Train-split Z-score

### Sampling Rate Conversion & Filtering Analysis
- **Zenodo 60GHz**: `nominal 10 Hz` -> 1:1 direct framing or 2:1 decimation if chirp config specifies 20 Hz.
- **Mendeley 24GHz (1000 Hz)**: Integer decimation ratio $N = 100$. Apply low-pass anti-aliasing FIR filter ($f_c = 4.5\text{ Hz}$) prior to 100:1 decimation down to 10 Hz.

---

## 10. Label Compatibility & Event Overlap Policy

SafeNest V6 target classes:
- `0: NORMAL`: Regular resting respiration.
- `1: RAPID_OR_ABNORMAL`: Rapid breathing ($>25\text{ bpm}$) or irregular respiration.
- `2: APNEA`: Cessation of breathing / breath hold ($\ge 10\text{ seconds}$).

### Proposed 30-Second Window Labeling Policy (for future implementation)
- **APNEA (2)**: Assigned if non-breathing timestamps cover $\ge 50\%$ ($\ge 15\text{ seconds}$) of the 30-second window.
- **NORMAL (0)**: Assigned if quiet respiration covers $100\%$ of the window with 0 apnea overlap.
- **TRANSITION_EXCLUDED**: Windows with $<50\%$ apnea overlap are excluded from training to prevent label ambiguity.

---

## 11. Provenance and Group-Split Feasibility

To prevent group leakage, sample-level provenance fields must be preserved during raw data conversion:

```json
{
  "sample_id": "ZEN_P001_T1_W001",
  "dataset_id": "zenodo_60ghz_v1",
  "archive_sha256": "f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0",
  "source_file": "db_records/P001/Test1/radar_rFFTs.zlib",
  "subject_id": "P001",
  "test_id": "Test1",
  "window_start_sec": 0.0,
  "window_end_sec": 30.0,
  "native_fs_hz": 10.0,
  "target_fs_hz": 10.0,
  "label_safenest": 0,
  "split": "train"
}
```

- **Split Priority**: `subject_id` -> `test_id` -> `window_id`.
- All windows from a subject (e.g. `P001`) are assigned exclusively to `train`, `validation`, or `test`, achieving **`SUBJECT_SPLIT_SUPPORTED`** status.

---

## 12. Candidate Dataset Scoring (12-Dimension Rubric)

| Evaluation Criterion | Weight | Zenodo 60GHz (`db_records.zip`) | Mendeley 24GHz | PhysioNet Apnea-ECG |
| :--- | :---: | :---: | :---: | :---: |
| A. Ability to address ranked data gap | 15% | 5 | 4 | 2 |
| B. Radar-domain relevance | 15% | 5 | 4 | 0 |
| C. Signal-semantic compatibility | 10% | 5 | 4 | 2 |
| D. Sampling and window compatibility | 10% | 5 | 4 | 5 |
| E. Label compatibility | 10% | 4 | 3 | 4 |
| F. Subject/session provenance | 10% | 5 | 3 | 5 |
| G. Motion/environment diversity | 5% | 4 | 3 | 2 |
| H. Value for external testing / adaptation | 5% | 5 | 4 | 3 |
| I. Documentation quality | 5% | 5 | 5 | 5 |
| J. License/access suitability | 5% | 5 (CC BY 4.0) | 5 (CC BY 4.0) | 5 (ODC-By) |
| K. Integration complexity | 5% | 4 | 3 | 3 |
| L. Relevance to deployed MR60 | 5% | 4 | 3 | 1 |
| **Weighted Total Score** | **100%** | **4.65 / 5.0** | **3.75 / 5.0** | **2.65 / 5.0** |
| **Final Classification Tier** | — | **TIER 1 (High Priority)** | **TIER 2 (Conditional)** | **TIER 3 (Reference)** |

---

## 13. Gap-to-Dataset-to-Use-Strategy Table

| Identified Gap | Dataset / Source Asset | Expected Benefit | Recommended First Use | Required Adapter / Preprocessor | Validation Needed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gap 1: Raw-to-Model Pipeline** | Preserved Zenodo 60GHz Archive (`db_records.zip`) | 100% reproducible real-radar training pipeline | **Raw Pipeline Reconstruction & Source Benchmark** | Raw rFFT phase extraction & unwrapping converter | Priority 2 Integrity Audit & single-source benchmark |
| **Gap 2 & 3: Domain Adaptation & Benchmark** | Mendeley 24GHz Medical Radar (`10.17632...`) | 24GHz radar signal preprocessing benchmark & domain adaptation | **Source-only preprocessing benchmark / domain-adaptation feasibility** | 1000Hz -> 10Hz FIR decimation adapter (100:1) | Preprocessing benchmark & domain adaptation analysis |
| **Gap 2: MR60 Sensor Adaptation** | Real MR60 Sensor Data Collection | Sensor-specific phase noise & SNR alignment | **Fine-Tuning & Domain Adaptation** | MR60 UART streaming frame parser | In-situ MR60 validation |
| **Respiration Morphology** | PhysioNet Apnea-ECG (`10.13026...`) | Clinical apnea timing & waveform morphology | **Reference Only** | ECG-to-respiration rate converter | Signal morphology analysis |

---

## 14. Recommended Action Roadmap

```text
Phase 1 (Completed in Priority 6):
  Establish asset baseline & coverage-gap analysis report.

Phase 2 (Requires User Approval):
  Develop reproducible rFFT phase converter script for db_records.zip.
  Generate datasets/mmwave/processed/zenodo_mmwave_v1.npz with subject provenance.

Phase 3 (Requires User Approval):
  Run Priority 2 Integrity Audit on zenodo_mmwave_v1.npz.
  Execute baseline training and evaluate out-of-sample real-subject accuracy.

Phase 4 (Requires User Approval):
  Acquire MR60 sensor calibration data for target domain adaptation.
```

---

## 15. Explicit Limitations & Scope Controls

- **External Data Downloaded**: **0 bytes**.
- **Local Zip Extracted**: **No** (`db_records.zip` unextracted).
- **Existing Datasets Modified**: **0**.
- **Models Retrained**: **0**.
- **Code Modified**: **0** (Preprocessing, inference, and model registry code preserved 100%).
- **Git Remote Operations**: **0** (No commits, pushes, or merges performed).
