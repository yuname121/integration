# Phase A2: Deterministic Range-Bin and Phase Extraction Pilot

## 1. Executive Summary

Phase A2 decoded 13 pilot recordings and established a deterministic, label-independent extraction profile. The gate is `PASS_WITH_WARNINGS` and A3 entry is `READY_WITH_CONDITIONS`. The canonical output is the unfiltered, unnormalised `np.unwrap(np.angle(z))` phase; diagnostic detrending and periodograms are not canonical outputs.

## 2. Git / Input Baseline

The work used merged A1 commit `be92a00e58f76b48bb85ec38e022f4fd3a313cbe`. The measured archive SHA-256 before and after execution was `f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0`.

## 3. A1 Decoder Contract Used

All inputs used `RFFT_DECODER_PROFILE_001`: `complex128[frames, 8, 64]`, axes frame/virtual-channel/range-bin, plus authoritative stored `float64[64]` rBins. Restricted symbolic pickle decoding remained mandatory; no arbitrary object execution occurred.

## 4. Pilot Composition

The approved 12-recording A1 pilot was preserved and `P004/Lying/Rest` was added deterministically because the [official Zenodo record](https://zenodo.org/records/18599983) identifies it as the largest initial-zero-frame case (11 frames). Coverage includes both postures, both activity conditions, both A0 schema profiles, annotations present/absent, and 400/500/600-frame recordings.

## 5. Stored rBins / Search Region

Stored rBins span `0.0` to `19.986163866666665` m. `PILOT_SEARCH_REGION_001` admits indices `[1, 2, 3, 4, 5, 6]` (`0.3`–`2.0` m threshold, whose actual admitted coordinates are `[0.31724069629629625, 0.6344813925925925, 0.9517220888888888, 1.268962785185185, 1.5862034814814812, 1.9034441777777775]`). Bin 0 is excluded as the zero-range/near-field coordinate. Bins above 2 m are excluded from this pilot search because the documented acquisition placed the radar about 0.5 m from the thorax; the limit is conservative pilot methodology, not universal hardware truth. See [Scientific Data](https://www.nature.com/articles/s41597-026-07172-9).

## 6. Candidate Range-Bin Strategies

The same eligible candidates were compared with A mean magnitude, B static-component-reduced dynamic energy, C 0.1–0.5 Hz diagnostic energy, D rank-based phase quality, and E adjacent-bin agreement. The selected range rule is B using the median across anonymous virtual channels. In all 13 pilot recordings it selected stored rBins index 2.

## 7. Virtual-Channel Strategies

V1 single-channel phase-quality selection, V2 quality-weighted aligned phase aggregation, and V3 median aligned phase consensus were compared. V1 was retained because it preserves a direct raw complex lineage and avoids opaque fusion while physical TX/RX ordering remains unknown. Channels are reported only as `virtual_channel_N`.

## 8. Canonical Phase Extraction

The selected complex timeline is preserved by checksum together with real/imaginary and magnitude statistics. Wrapped phase uses `np.angle`; canonical phase uses `np.unwrap` with default discontinuity π and period 2π. Near-zero samples are flagged and preserved without interpolation. No detrending, filtering, smoothing, resampling, or normalisation is applied to canonical phase.

## 9. Strategy Comparison

There are 13 per-recording comparison records. The pilot-selected profile favours perfect range stability ({'2': 13}) plus deterministic channel-quality selection ({'6': 4, '1': 2, '5': 5, '0': 1, '4': 1}), reproducible byte checksums, direct provenance, and implementation simplicity. Strategy C remains diagnostic because 0.1–0.5 Hz is not asserted to cover all post-exercise respiration.

## 10. Selected Extraction Profile

`MMWAVE_PHASE_EXTRACTION_PROFILE_001` uses B median-channel dynamic energy for range, V1 rank-composite phase quality for channel, a single stored bin, and fixed ties by lowest bin then lowest channel.

## 11. Time-Domain Diagnostics

All canonical phase lengths equal their frame and timestamp counts. Near-zero samples were retained and flagged rather than repaired. Phase step percentiles, unwrap corrections, large steps, and magnitude outliers are recorded per recording.

## 12. Frequency-Domain Diagnostics

Temporary linearly detrended Hann periodograms use a 0.05–2.0 Hz total band and 0.1–0.5 Hz respiration diagnostic band. Dominant frequencies across valid pilots span `0.05`–`0.26666666666666666` Hz. These diagnostics do not alter the canonical phase.

## 13. Annotation-Based Post-Selection Validation

Annotations were loaded only after bin/channel selection. Outcomes were `['SUPPORT', 'SUPPORT', 'SUPPORT', 'SUPPORT', 'SUPPORT', 'SUPPORT']` using the predeclared inside/outside phase-step-energy ratio thresholds. These are annotated voluntary non-breathing/breath-hold intervals, not clinical apnea, and never selection evidence.

## 14. Failure / Low-Quality Cases

Extraction failures: 0. Warning-bearing selected results: 1. The added P004/Lying/Rest case preserves its initial zero-magnitude frames and emits a quality warning.

## 15. Exceptions

The registry contains 4 items. Physical virtual-channel ordering remains unknown, configured R_BIN differs from stored rBins spacing, and restricted pickle decoding remains required.

## 16. Validation

The shared in-memory validator ran before the gate was derived and returned `True`. It checked pilot coverage, A1 decode contract, coordinates, channels, label independence, profile linkage, phase lengths, nonfinite values, counts, and gate consistency.

## 17. A2 Gate

`PASS_WITH_WARNINGS`: deterministic extraction succeeded, with non-blocking preserved limitations.

## 18. A3 Entry Decision

`READY_WITH_CONDITIONS`: A3 may consume the unfiltered canonical phase and quality/provenance metadata, while respecting the warnings above.

## 19. Remaining Limitations

This is a 13-recording pilot, not a full 440-recording validation. Stored selected range is a radar coordinate, not a claim of true chest distance. Virtual-channel physical mapping and the config/stored range-spacing discrepancy remain unresolved.

## 20. Explicit Non-Scope Confirmation

No permanent detrending/BPF/Z-score, resampling, 30-second windows, SafeNest label mapping, subject split, full conversion, NPZ generation, model training, or A3 work was performed.

## 21. Files Changed

Implementation, validator, synthetic tests, this report, and the eight required manifest/checksum files were added for Phase A2.

## 22. Commands / Tests

`python3 -m unittest tests/test_mmwave_phase_extractor.py -v`; `python3 scripts/run_mmwave_phase_pilot.py`; `python3 scripts/validate_mmwave_phase_pilot.py`; isolated regeneration and SHA-256 comparison; `git diff --check`; archive SHA-256 before/after.
