# SafeNest active dataset guide

The dataset manifest separates real-source reconstruction assets from synthetic
pipeline smoke fixtures. They must never share a performance claim or lineage.

## Zenodo real-radar reconstruction

- Source: Zenodo 60 GHz dataset, DOI `10.5281/zenodo.18599983`
- Local raw archive: `datasets/raw_archives/external_datasets/db_records.zip`
- Scope: 110 real subjects, 440 recordings
- A5 split: `datasets/mmwave/splits/mmwave_real_subject_split_v1.json`
- Current status: `A6_PASS_WITH_WARNINGS_PHASE_B_READY_WITH_CONDITIONS`
- Canonical A6 matrix: `datasets/mmwave/processed/mmwave_canonical_real_v1.npy`
  with shape `(530, 300)` and `float64` dtype
- Full lineage: `datasets/mmwave/manifests/a6_full_conversion/`

The A5 split fixes 77 TRAIN, 17 VALIDATION, and 16 LOCKED_TEST subjects. The
canonical real-data matrix inherits that split through its window and provenance
manifests. The existing `mmwave_respiration_v1.npz` is not derived from those people.

## Synthetic smoke fixtures

`datasets/mmwave/processed/mmwave_respiration_v1.npz` is a procedural synthetic
fixture with 3,433 windows. It verifies training, conversion, quantization,
evaluation, and runtime code paths. It is not Zenodo-processed data and cannot
support real-subject, real-sensor, or clinical performance claims.

`datasets/mmwave/splits/mmwave_group_split_v1.json` describes synthetic group
isolation only. It is separate from the real A5 subject split.

`datasets/build_processed_npz.py` generates deterministic synthetic smoke
fixtures only:

```bash
python3 datasets/build_processed_npz.py --dataset mmwave
python3 datasets/build_processed_npz.py --dataset co2
```

If a raw `--source-root`, `--mmwave-root`, or `--co2-root` is supplied, the
script fails explicitly instead of silently replacing real conversion with
random data.

## A6 contract and Phase B entry rules

- consume `MMWAVE_SUBJECT_SPLIT_PROFILE_001` without reassigning subjects;
- preserve canonical unfiltered phase separately from experimental B-stage
  preprocessing;
- keep A4 `mapping_type` and `assignment_status`;
- retain ambiguous windows for provenance and transition analysis, excluding
  them from pure-class training;
- calculate normalization statistics from train subjects only;
- keep LOCKED_TEST unavailable to preprocessing, threshold, and model selection;
- run a near-duplicate diagnostic before comparative Phase B evaluation;
- store only repository-relative paths in active artifacts.
