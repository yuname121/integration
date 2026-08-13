#!/usr/bin/env python3
"""SafeNest Phase A6 — Full mmWave Real-Data Conversion Validator.

Provides independent, deep evidence-derived validation for Phase A6 full conversion,
verifying A0-A5 contract compliance, timestamp provenance, zero cross-split leakage,
path provenance, canonical .npy 1:1 alignment, checksum integrity, and Phase-A exit gate criteria.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))


class A6ValidationError(Exception):
    """Raised when Phase A6 full conversion validation fails."""


CORE_CHECKSUM_TARGETS = {
    "processing_profile.json",
    "full_recording_results.jsonl",
    "full_window_manifest.jsonl",
    "full_provenance_manifest.jsonl",
    "full_label_distribution.json",
    "full_split_distribution.json",
    "full_quality_audit.json",
    "full_duplicate_audit.json",
    "spot_check_results.json",
    "exceptions.json",
    "../../processed/mmwave_canonical_real_v1.npy",
}

WINDOW_PROVENANCE_FIELDS = (
    "window_id",
    "recording_id",
    "subject_id",
    "split",
    "safenest_label",
    "safenest_label_id",
    "mapping_type",
    "mapping_rule_id",
    "assignment_status",
    "canonical_signal_hash",
    "training_eligible",
    "validation_eligible",
    "locked_test_evaluation_eligible",
)


def _validate_checksums(root_dir: Path, manifest_dir: Path) -> int:
    """Validate checksum syntax, coverage, uniqueness, containment, and content."""
    checksums_file = manifest_dir / "checksums.sha256"
    if not checksums_file.is_file():
        raise A6ValidationError(f"Checksums manifest missing: {checksums_file}")

    root_resolved = root_dir.resolve()
    listed_names: set[str] = set()
    for line_number, line in enumerate(checksums_file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", parts[0]) is None:
            raise A6ValidationError(f"Malformed checksum entry at line {line_number}: {line!r}")

        exp_hash, rel_name = parts
        if rel_name in listed_names:
            raise A6ValidationError(f"Duplicate checksum target: {rel_name}")
        listed_names.add(rel_name)

        target_f = (manifest_dir / rel_name).resolve()
        if target_f != root_resolved and root_resolved not in target_f.parents:
            raise A6ValidationError(f"Checksum target escapes canonical project root: {rel_name}")
        if not target_f.is_file():
            raise A6ValidationError(f"Manifest file listed in checksums missing: {rel_name}")
        actual_hash = hashlib.sha256(target_f.read_bytes()).hexdigest()
        if actual_hash != exp_hash:
            raise A6ValidationError(
                f"Checksum mismatch for file {rel_name}: expected {exp_hash}, got {actual_hash}"
            )

    missing_targets = CORE_CHECKSUM_TARGETS - listed_names
    if missing_targets:
        raise A6ValidationError(f"Required checksum targets missing: {sorted(missing_targets)}")
    return len(listed_names)


def _validate_alignment(
    windows: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
    canonical_matrix: np.ndarray,
) -> None:
    """Validate semantic and byte-level 1:1 alignment across all A6 sample artifacts."""
    if len(windows) != len(provenance) or len(windows) != canonical_matrix.shape[0]:
        raise A6ValidationError(
            f"1:1 alignment mismatch! Windows: {len(windows)}, Provenance: {len(provenance)}, "
            f"NPY rows: {canonical_matrix.shape[0]}"
        )

    window_ids: set[str] = set()
    for idx, (window, prov, npy_row) in enumerate(zip(windows, provenance, canonical_matrix)):
        if window.get("canonical_sample_index") != idx or prov.get("canonical_sample_index") != idx:
            raise A6ValidationError(f"Canonical sample index non-contiguous at row {idx}!")

        window_id = window.get("window_id")
        if not window_id or window_id in window_ids:
            raise A6ValidationError(f"Missing or duplicate window_id at row {idx}: {window_id!r}")
        window_ids.add(window_id)

        for field in WINDOW_PROVENANCE_FIELDS:
            if field not in window or field not in prov:
                raise A6ValidationError(f"Required alignment field '{field}' missing at row {idx}")
            if window[field] != prov[field]:
                raise A6ValidationError(
                    f"Window/provenance mismatch at row {idx} for '{field}': "
                    f"window={window[field]!r}, provenance={prov[field]!r}"
                )

        row_bytes = np.ascontiguousarray(npy_row, dtype=np.float64).tobytes()
        row_hash = hashlib.sha256(row_bytes).hexdigest()
        if window["canonical_signal_hash"] != row_hash:
            raise A6ValidationError(
                f"Canonical signal hash mismatch at index {idx} between window manifest and .npy row!"
            )
        if prov["canonical_signal_hash"] != row_hash:
            raise A6ValidationError(
                f"Canonical signal hash mismatch at index {idx} between provenance and .npy row!"
            )


def _validate_recording_accounting(
    a0_recordings: list[dict[str, Any]],
    rec_results: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> None:
    """Require every authoritative recording to complete and own its declared samples."""
    a0_by_id = {record["recording_id"]: record for record in a0_recordings}
    if len(a0_by_id) != len(a0_recordings):
        raise A6ValidationError("Duplicate recording IDs in authoritative A0 inventory")

    result_by_id = {record.get("recording_id"): record for record in rec_results}
    if None in result_by_id or len(result_by_id) != len(rec_results):
        raise A6ValidationError("Missing or duplicate recording entries in full_recording_results.jsonl")

    missing_recs = set(a0_by_id) - set(result_by_id)
    unknown_recs = set(result_by_id) - set(a0_by_id)
    if missing_recs:
        raise A6ValidationError(f"A0 recordings missing from A6 full conversion results: {missing_recs}")
    if unknown_recs:
        raise A6ValidationError(f"Unknown recordings in A6 conversion results: {unknown_recs}")

    allowed_statuses = {"SUCCESS", "SUCCESS_WITH_WARNINGS"}
    window_counts = Counter(window.get("recording_id") for window in windows)
    provenance_counts = Counter(row.get("recording_id") for row in provenance)

    for rec_id, result in result_by_id.items():
        status = result.get("status")
        if status not in allowed_statuses:
            raise A6ValidationError(f"Recording {rec_id} has non-success A6 status: {status}")

        expected_subject = a0_by_id[rec_id]["subject_id"]
        if result.get("subject_id") != expected_subject:
            raise A6ValidationError(
                f"Recording {rec_id} subject mismatch: {result.get('subject_id')} != {expected_subject}"
            )

        declared_count = result.get("window_count")
        if not isinstance(declared_count, int) or declared_count <= 0:
            raise A6ValidationError(f"Recording {rec_id} has invalid window_count: {declared_count}")
        if window_counts[rec_id] != declared_count or provenance_counts[rec_id] != declared_count:
            raise A6ValidationError(
                f"Recording {rec_id} sample accounting mismatch: declared={declared_count}, "
                f"windows={window_counts[rec_id]}, provenance={provenance_counts[rec_id]}"
            )

    for artifact_name, rows in (("window", windows), ("provenance", provenance)):
        for row in rows:
            rec_id = row.get("recording_id")
            if rec_id not in a0_by_id:
                raise A6ValidationError(f"Unknown recording ID in {artifact_name} artifact: {rec_id}")
            if row.get("subject_id") != a0_by_id[rec_id]["subject_id"]:
                raise A6ValidationError(
                    f"{artifact_name.title()} subject mismatch for recording {rec_id}: "
                    f"{row.get('subject_id')} != {a0_by_id[rec_id]['subject_id']}"
                )


def validate_full_conversion_artifacts(
    root_dir: Path = ROOT_DIR,
    manifest_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate all Phase A6 full conversion manifest artifacts against contracts."""
    if manifest_dir is None:
        manifest_dir = root_dir / "datasets/mmwave/manifests/a6_full_conversion"

    if not manifest_dir.is_dir():
        raise A6ValidationError(f"Phase A6 manifest directory not found: {manifest_dir}")

    # 1. Check raw archive SHA-256 immutability
    archive_path = root_dir / "datasets/raw_archives/external_datasets/db_records.zip"
    if not archive_path.is_file():
        raise A6ValidationError(f"Raw dataset archive zip not found: {archive_path}")

    hasher = hashlib.sha256()
    with open(archive_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    current_archive_sha256 = hasher.hexdigest()

    expected_archive_sha256 = "f0bcfdac94f88b43bb34d3da8e8f071a787291f86c97798059b8dbf4d4be08b0"
    if current_archive_sha256 != expected_archive_sha256:
        raise A6ValidationError(
            f"Raw archive SHA-256 changed! Expected {expected_archive_sha256}, got {current_archive_sha256}"
        )

    # 2. Check checksums.sha256 syntax, coverage, containment, and content
    checksum_entry_count = _validate_checksums(root_dir, manifest_dir)

    # 3. Validate Canonical Numeric Dataset (.npy array)
    canonical_npy_path = root_dir / "datasets/mmwave/processed/mmwave_canonical_real_v1.npy"
    if not canonical_npy_path.is_file():
        raise A6ValidationError(f"Canonical numeric dataset artifact missing: {canonical_npy_path}")

    canonical_matrix = np.load(canonical_npy_path)
    if canonical_matrix.ndim != 2 or canonical_matrix.shape[1] != 300 or canonical_matrix.dtype != np.float64:
        raise A6ValidationError(
            f"Canonical numeric matrix invalid shape or dtype: expected (N, 300) float64, got {canonical_matrix.shape} {canonical_matrix.dtype}"
        )

    # 4. Load A0 inventory & A5 splits
    a0_manifest = root_dir / "datasets/mmwave/manifests/a0_raw_inventory/recording_index.jsonl"
    if not a0_manifest.is_file():
        raise A6ValidationError(f"Authoritative A0 inventory missing: {a0_manifest}")

    a0_recordings = []
    with open(a0_manifest, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                a0_recordings.append(json.loads(line))

    a5_split_json = root_dir / "datasets/mmwave/splits/mmwave_real_subject_split_v1.json"
    if not a5_split_json.is_file():
        raise A6ValidationError(f"Authoritative A5 split JSON missing: {a5_split_json}")

    a5_split_data = json.loads(a5_split_json.read_text(encoding="utf-8"))
    a5_subject_split_map = a5_split_data.get("subject_split_map", {})

    # 5. Load A6 manifests
    rec_results = []
    with open(manifest_dir / "full_recording_results.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec_results.append(json.loads(line))

    windows = []
    with open(manifest_dir / "full_window_manifest.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                windows.append(json.loads(line))

    provenance = []
    with open(manifest_dir / "full_provenance_manifest.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                provenance.append(json.loads(line))

    quality_audit = json.loads((manifest_dir / "full_quality_audit.json").read_text(encoding="utf-8"))

    # 6. Verify semantic and byte-level 1:1 alignment
    _validate_alignment(windows, provenance, canonical_matrix)

    # 7. Verify Future NPZ Sample Index is None/null
    for p in provenance:
        if p.get("future_npz_sample_index") is not None:
            raise A6ValidationError(f"future_npz_sample_index must be None/null until Phase B training NPZ creation, got {p.get('future_npz_sample_index')}")

    # 8. Verify every A0 recording completed and owns its declared samples
    _validate_recording_accounting(a0_recordings, rec_results, windows, provenance)

    # 9. Verify Immutable Split Inheritance
    valid_splits = {"TRAIN", "VALIDATION", "LOCKED_TEST"}
    for r in rec_results:
        rec_id = r["recording_id"]
        subj_id = r["subject_id"]
        rec_split = r["split"]
        expected_split = a5_subject_split_map.get(subj_id)

        if expected_split not in valid_splits:
            raise A6ValidationError(f"Invalid split value for subject {subj_id}: {expected_split}")
        if rec_split != expected_split:
            raise A6ValidationError(f"Recording {rec_id} split '{rec_split}' != inherited subject split '{expected_split}'")

    for w in windows:
        win_id = w["window_id"]
        subj_id = w["subject_id"]
        win_split = w["split"]
        expected_split = a5_subject_split_map.get(subj_id)

        if win_split != expected_split:
            raise A6ValidationError(f"Window {win_id} split '{win_split}' != inherited subject split '{expected_split}'")

    # 10. Verify Eligibility Restrictions
    for w in windows:
        win_id = w["window_id"]
        split = w["split"]
        status = w["assignment_status"]

        if split == "LOCKED_TEST" and w.get("training_eligible", False):
            raise A6ValidationError(f"LOCKED_TEST window {win_id} has training_eligible=True!")

        if status == "AMBIGUOUS":
            if w.get("training_eligible", False) or w.get("validation_eligible", False) or w.get("locked_test_evaluation_eligible", False):
                raise A6ValidationError(f"AMBIGUOUS window {win_id} has pure-class eligibility set to True!")

    # 11. Verify Timestamp Contract (No trailing Z on newly generated window timestamps)
    for w in windows:
        for key in ("start_timestamp", "last_sample_timestamp", "end_timestamp_exclusive"):
            ts_str = str(w.get(key, ""))
            if ts_str.endswith("Z"):
                raise A6ValidationError(f"Window manifest timestamp '{key}' has unverified trailing Z suffix: {ts_str}")

    for p in provenance:
        if p.get("timestamp_reference") != "COMMON_ACQUISITION_COMPUTER_CLOCK":
            raise A6ValidationError(f"Invalid timestamp reference: {p.get('timestamp_reference')}")
        if p.get("source_timezone") != "UNVERIFIED":
            raise A6ValidationError(f"Invalid source timezone: {p.get('source_timezone')}")
        if p.get("utc_conversion_claimed") is not False:
            raise A6ValidationError("utc_conversion_claimed must be False!")

    # 12. Verify Path Provenance (No absolute local paths in canonical fields)
    for p in provenance:
        for key in ("archive_identifier", "source_radar_member", "source_timestamp_member", "a1_decoder_profile"):
            val = str(p.get(key, ""))
            if val.startswith("/Users/") or val.startswith("file://") or val.startswith("C:\\"):
                raise A6ValidationError(f"Absolute local path found in canonical provenance field '{key}': {val}")

    # 13. INDEPENDENT RE-COMPUTATION of Cross-Split Leakage from Evidence
    split_subjects = defaultdict(set)
    split_recordings = defaultdict(set)
    split_window_ids = defaultdict(set)
    hash_groups = defaultdict(list)

    for w in windows:
        sp = w["split"]
        split_subjects[sp].add(w["subject_id"])
        split_recordings[sp].add(w["recording_id"])
        split_window_ids[sp].add(w["window_id"])
        hash_groups[w["canonical_signal_hash"]].append(sp)

    train_subjs = split_subjects["TRAIN"]
    val_subjs = split_subjects["VALIDATION"]
    test_subjs = split_subjects["LOCKED_TEST"]
    subj_leakage = len(train_subjs & val_subjs) + len(train_subjs & test_subjs) + len(val_subjs & test_subjs)
    if subj_leakage > 0:
        raise A6ValidationError(f"CRITICAL LEAKAGE: cross-split subject overlap = {subj_leakage}")

    train_recs = split_recordings["TRAIN"]
    val_recs = split_recordings["VALIDATION"]
    test_recs = split_recordings["LOCKED_TEST"]
    rec_leakage = len(train_recs & val_recs) + len(train_recs & test_recs) + len(val_recs & test_recs)
    if rec_leakage > 0:
        raise A6ValidationError(f"CRITICAL LEAKAGE: cross-split recording overlap = {rec_leakage}")

    train_wins = split_window_ids["TRAIN"]
    val_wins = split_window_ids["VALIDATION"]
    test_wins = split_window_ids["LOCKED_TEST"]
    win_leakage = len(train_wins & val_wins) + len(train_wins & test_wins) + len(val_wins & test_wins)
    if win_leakage > 0:
        raise A6ValidationError(f"CRITICAL LEAKAGE: cross-split window ID overlap = {win_leakage}")

    cross_split_signal_hash_leakage = sum(1 for grp in hash_groups.values() if len(set(grp)) > 1)
    if cross_split_signal_hash_leakage > 0:
        raise A6ValidationError(f"CRITICAL LEAKAGE: cross-split exact signal hash overlap = {cross_split_signal_hash_leakage}")

    # 14. INDEPENDENT RE-COMPUTATION of Signal Quality Metrics directly from .npy matrix
    recalc_nan = int(np.isnan(canonical_matrix).sum())
    recalc_inf = int(np.isinf(canonical_matrix).sum())
    recalc_exact_const = int(sum(np.all(row == row[0]) for row in canonical_matrix))
    recalc_near_const = int(sum(np.std(row) < 1e-6 for row in canonical_matrix))
    recalc_mean_std = float(np.mean([np.std(row) for row in canonical_matrix]))

    if recalc_nan != 0 or recalc_inf != 0:
        raise A6ValidationError(f"Quality audit detected nonfinite signal samples: NaN={recalc_nan}, Inf={recalc_inf}")

    if recalc_exact_const != 0 or recalc_near_const != 0:
        raise A6ValidationError(f"Quality audit detected degenerate constant windows: exact={recalc_exact_const}, near={recalc_near_const}")

    # Compare recalculated metrics against full_quality_audit.json manifest values
    if quality_audit.get("nan_sample_count") != recalc_nan:
        raise A6ValidationError(f"Quality audit NaN mismatch: manifest={quality_audit.get('nan_sample_count')}, recalculated={recalc_nan}")
    if quality_audit.get("inf_sample_count") != recalc_inf:
        raise A6ValidationError(f"Quality audit Inf mismatch: manifest={quality_audit.get('inf_sample_count')}, recalculated={recalc_inf}")
    if quality_audit.get("exact_constant_window_count") != recalc_exact_const:
        raise A6ValidationError(f"Quality audit exact constant mismatch: manifest={quality_audit.get('exact_constant_window_count')}, recalculated={recalc_exact_const}")
    if quality_audit.get("near_constant_window_count") != recalc_near_const:
        raise A6ValidationError(f"Quality audit near constant mismatch: manifest={quality_audit.get('near_constant_window_count')}, recalculated={recalc_near_const}")

    manifest_mean_std = float(quality_audit.get("mean_window_phase_std_dev", 0.0))
    if abs(manifest_mean_std - recalc_mean_std) > 1e-5:
        raise A6ValidationError(f"Quality audit mean std mismatch: manifest={manifest_mean_std}, recalculated={recalc_mean_std}")

    return {
        "validation_success": True,
        "a6_gate_status": "PASS_WITH_WARNINGS",
        "phase_b_entry_status": "READY_WITH_CONDITIONS",
        "total_recordings_validated": len(rec_results),
        "total_windows_validated": len(windows),
        "total_provenance_validated": len(provenance),
        "canonical_npy_rows_validated": canonical_matrix.shape[0],
        "checksum_entries_validated": checksum_entry_count,
        "raw_archive_sha256": current_archive_sha256,
        "leakage_recalculated": {
            "cross_split_subject_overlap": subj_leakage,
            "cross_split_recording_overlap": rec_leakage,
            "cross_split_window_id_overlap": win_leakage,
            "cross_split_exact_signal_overlap": cross_split_signal_hash_leakage,
        },
    }


def main() -> None:
    res = validate_full_conversion_artifacts()
    print("Standalone A6 Full Conversion Validation Result:")
    print(f"Validation Success: {res['validation_success']}")
    print(f"A6 Gate Status: {res['a6_gate_status']}")
    print(f"Phase-B Entry Status: {res['phase_b_entry_status']}")
    print(f"Validated Recordings: {res['total_recordings_validated']}")
    print(f"Validated Windows: {res['total_windows_validated']}")
    print(f"Canonical NPY Matrix Rows: {res['canonical_npy_rows_validated']}")


if __name__ == "__main__":
    main()
