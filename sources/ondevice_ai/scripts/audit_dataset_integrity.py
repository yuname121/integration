#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/audit_dataset_integrity.py
SafeNest V6 Automated Audit of mmWave Data Integrity, Duplicate Signals, and Group Leakage

Performs canonical SHA-256 waveform hashing, label consistency checks,
index split leakage audits, cross-split signal duplicate audits, and group isolation checks.
"""

from __future__ import annotations
import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Set, Optional
import numpy as np

# Ensure canonical repository root is in python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


CLASS_MAP = {0: "NORMAL", 1: "RAPID_OR_ABNORMAL", 2: "APNEA"}


def artifact_path(path: Path) -> str:
    """Serialize an input path relative to the canonical repository root."""
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return f"EXTERNAL_INPUT/{path.name}"


def calculate_file_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def canonical_window_hash(window: np.ndarray) -> Tuple[str, bytes]:
    """
    Canonicalize a 300-sample respiration window into a deterministic SHA-256 hash.
    1. Squeeze or flatten to 1D shape (300,)
    2. Cast to float32
    3. Normalize -0.0 to 0.0
    4. Ensure little-endian (<f4) and C-contiguous memory layout
    5. Hash canonical shape string b'(300,)' + raw bytes
    """
    arr = np.array(window, copy=True)
    if arr.ndim == 2 and arr.shape == (300, 1):
        arr = arr.squeeze(axis=-1)
    elif arr.ndim == 2 and arr.shape == (1, 300):
        arr = arr.squeeze(axis=0)
    elif arr.ndim > 2:
        arr = arr.reshape(-1)

    if arr.shape != (300,):
        raise ValueError(f"Invalid canonical window shape: {arr.shape}, expected (300,)")

    arr = arr.astype(np.float32)
    arr[arr == 0.0] = 0.0

    arr_contiguous = np.ascontiguousarray(arr.astype("<f4"))
    raw_bytes = arr_contiguous.tobytes()

    canonical_shape_bytes = b"(300,)"
    hasher = hashlib.sha256()
    hasher.update(canonical_shape_bytes)
    hasher.update(raw_bytes)
    return hasher.hexdigest(), raw_bytes


def load_dataset_and_splits(
    npz_path: Path, split_path: Path
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, List[int]], Dict[str, Any]]:
    """
    Loads NPZ dataset and split definitions.
    Supports:
    - Separate split arrays (X_train, X_val, X_test, y_train, y_val, y_test)
    - Unified arrays (X, y, group_ids) paired with split index lists in split JSON
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ dataset missing: {npz_path}")

    if not split_path.exists():
        raise FileNotFoundError(f"Split JSON missing: {split_path}")

    npz_data = np.load(npz_path, allow_pickle=True)
    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    npz_keys = list(npz_data.files)

    # 1. Check for split arrays in NPZ
    if "X_train" in npz_keys and "X_val" in npz_keys and "X_test" in npz_keys:
        X_tr = npz_data["X_train"]
        y_tr = npz_data["y_train"]
        X_va = npz_data["X_val"]
        y_va = npz_data["y_val"]
        X_te = npz_data["X_test"]
        y_te = npz_data["y_test"]

        n_tr, n_va, n_te = len(X_tr), len(X_va), len(X_te)
        X = np.concatenate([X_tr, X_va, X_te], axis=0)
        y = np.concatenate([y_tr, y_va, y_te], axis=0)

        split_indices = {
            "train": list(range(0, n_tr)),
            "val": list(range(n_tr, n_tr + n_va)),
            "test": list(range(n_tr + n_va, n_tr + n_va + n_te)),
        }

        group_ids = None
        for g_key in ["group_ids", "groups", "synthetic_subject_group_ids"]:
            if g_key in npz_keys:
                g_arr = npz_data[g_key]
                if len(g_arr) == len(X):
                    group_ids = g_arr
                    break

        metadata = {
            "npz_keys": npz_keys,
            "x_key": "X_train,X_val,X_test",
            "y_key": "y_train,y_val,y_test",
            "group_key": "synthetic_subject_group_ids" if group_ids is not None else None,
            "split_mode": "PRE_SPLIT_NPZ_CONCATENATED",
        }
        return X, y, group_ids, split_indices, metadata

    # 2. Check for unified arrays in NPZ
    x_key = None
    for cand in ["X", "x", "respiration_windows", "signals"]:
        if cand in npz_keys:
            x_key = cand
            break

    y_key = None
    for cand in ["y", "labels", "targets"]:
        if cand in npz_keys:
            y_key = cand
            break

    group_key = None
    for cand in ["group_ids", "groups", "synthetic_group_ids", "subject_ids"]:
        if cand in npz_keys:
            group_key = cand
            break

    if x_key is None or y_key is None:
        raise KeyError(f"Unable to locate X and y keys in NPZ keys: {npz_keys}")

    X = npz_data[x_key]
    y = npz_data[y_key]
    group_ids = npz_data[group_key] if group_key else None

    # Load split index lists from split JSON if available
    split_indices = {}
    for s_name in ["train", "val", "test"]:
        idx_key = f"{s_name}_indices"
        if idx_key in split_data and isinstance(split_data[idx_key], list):
            split_indices[s_name] = split_data[idx_key]
        elif "indices" in split_data and s_name in split_data["indices"]:
            split_indices[s_name] = split_data["indices"][s_name]

    metadata = {
        "npz_keys": npz_keys,
        "x_key": x_key,
        "y_key": y_key,
        "group_key": group_key,
        "split_mode": "UNIFIED_NPZ",
    }
    return X, y, group_ids, split_indices, metadata


def run_integrity_audit(
    npz_path: Path,
    split_path: Path,
    output_path: Path,
    expected_windows: int = 3118,
) -> Dict[str, Any]:
    npz_sha256 = calculate_file_sha256(npz_path)
    split_sha256 = calculate_file_sha256(split_path)

    X, y, group_ids, split_indices, meta = load_dataset_and_splits(npz_path, split_path)

    total_windows = len(X)
    count_matches = total_windows == expected_windows
    count_mismatch_reason = (
        None
        if count_matches
        else f"Observed total window count ({total_windows}) differs from CLI expected count ({expected_windows})."
    )

    # 1. Preliminary Schema Validation
    schema_failures = []
    if len(X) != len(y):
        schema_failures.append(f"Length mismatch: X length ({len(X)}) != y length ({len(y)})")

    if group_ids is not None and len(group_ids) != len(X):
        schema_failures.append(f"Length mismatch: group_ids length ({len(group_ids)}) != X length ({len(X)})")

    if np.isnan(X).any() or np.isinf(X).any():
        schema_failures.append("X array contains NaN or Inf values")

    if X.ndim not in [2, 3]:
        schema_failures.append(f"Invalid X dim: {X.ndim}")
    else:
        sample_shape = X[0].shape
        if sample_shape not in [(300,), (300, 1), (1, 300)]:
            schema_failures.append(f"Invalid sample shape: {sample_shape}, expected (300,) or (300, 1)")

    unique_y = set(np.unique(y))
    if not unique_y.issubset({0, 1, 2}):
        schema_failures.append(f"y contains invalid label classes: {unique_y - {0, 1, 2}}")

    dataset_integrity_status = "PASSED" if not schema_failures else "FAILED"

    # 2. Canonical Hashing & Duplicate Waveform Audit
    hash_to_samples: Dict[str, List[Dict[str, Any]]] = {}
    
    # Build reverse map for split assignment
    index_to_split: Dict[int, str] = {}
    for s_name, idx_list in split_indices.items():
        for i in idx_list:
            if isinstance(i, int):
                index_to_split[i] = s_name

    for i in range(total_windows):
        w_hash, _ = canonical_window_hash(X[i])
        lbl = int(y[i])
        grp = str(group_ids[i]) if group_ids is not None else "UNLINKED"
        spl = index_to_split.get(i, "UNASSIGNED")

        if w_hash not in hash_to_samples:
            hash_to_samples[w_hash] = []
        hash_to_samples[w_hash].append({
            "index": i,
            "label": lbl,
            "label_name": CLASS_MAP.get(lbl, "UNKNOWN"),
            "split": spl,
            "group_id": grp,
        })

    unique_window_hashes = len(hash_to_samples)
    duplicate_groups = {h: insts for h, insts in hash_to_samples.items() if len(insts) > 1}
    duplicate_hash_group_count = len(duplicate_groups)
    duplicate_instance_count = sum(len(insts) - 1 for insts in duplicate_groups.values())
    windows_in_duplicate_groups = sum(len(insts) for insts in duplicate_groups.values())

    duplicate_examples = []
    for h, insts in list(duplicate_groups.items())[:20]:
        duplicate_examples.append({
            "hash": h,
            "count": len(insts),
            "indices": [x["index"] for x in insts],
            "labels": [x["label"] for x in insts],
            "splits": [x["split"] for x in insts],
            "group_ids": [x["group_id"] for x in insts],
        })

    # 3. Label Consistency Audit
    inconsistent_groups = {}
    for h, insts in hash_to_samples.items():
        distinct_labels = set(x["label"] for x in insts)
        if len(distinct_labels) > 1:
            inconsistent_groups[h] = insts

    has_label_inconsistency = len(inconsistent_groups) > 0
    inconsistent_hash_count = len(inconsistent_groups)
    inconsistent_window_count = sum(len(insts) for insts in inconsistent_groups.values())
    inconsistent_examples = []
    for h, insts in list(inconsistent_groups.items())[:20]:
        inconsistent_examples.append({
            "hash": h,
            "indices": [x["index"] for x in insts],
            "labels": [x["label"] for x in insts],
            "label_names": [x["label_name"] for x in insts],
            "splits": [x["split"] for x in insts],
            "group_ids": [x["group_id"] for x in insts],
        })

    # 4. Split Index Integrity Audit
    split_failures = []
    split_index_checks: Dict[str, Any] = {}
    split_sets: Dict[str, Set[int]] = {}

    for s_name in ["train", "val", "test"]:
        raw_list = split_indices.get(s_name, [])
        duplicates_in_split = len(raw_list) - len(set(raw_list))
        non_int_count = sum(1 for x in raw_list if not isinstance(x, (int, np.integer)))
        neg_count = sum(1 for x in raw_list if isinstance(x, (int, np.integer)) and x < 0)
        oob_count = sum(1 for x in raw_list if isinstance(x, (int, np.integer)) and x >= total_windows)

        if duplicates_in_split > 0:
            split_failures.append(f"Split {s_name} contains {duplicates_in_split} duplicate indices")
        if non_int_count > 0:
            split_failures.append(f"Split {s_name} contains {non_int_count} non-integer indices")
        if neg_count > 0:
            split_failures.append(f"Split {s_name} contains {neg_count} negative indices")
        if oob_count > 0:
            split_failures.append(f"Split {s_name} contains {oob_count} out-of-bounds indices")
        if len(raw_list) == 0:
            split_failures.append(f"Split {s_name} is empty")

        split_sets[s_name] = set(raw_list)
        split_index_checks[s_name] = {
            "count": len(raw_list),
            "unique_count": len(split_sets[s_name]),
            "duplicate_index_count": duplicates_in_split,
            "non_int_count": non_int_count,
            "negative_index_count": neg_count,
            "out_of_bounds_count": oob_count,
        }

    tr_set = split_sets.get("train", set())
    va_set = split_sets.get("val", set())
    te_set = split_sets.get("test", set())

    tr_va_overlap = sorted(list(tr_set & va_set))
    tr_te_overlap = sorted(list(tr_set & te_set))
    va_te_overlap = sorted(list(va_set & te_set))

    if tr_va_overlap:
        split_failures.append(f"train-val index overlap: {len(tr_va_overlap)} indices")
    if tr_te_overlap:
        split_failures.append(f"train-test index overlap: {len(tr_te_overlap)} indices")
    if va_te_overlap:
        split_failures.append(f"val-test index overlap: {len(va_te_overlap)} indices")

    all_indices = set(range(total_windows))
    assigned_indices = tr_set | va_set | te_set
    unassigned_indices = sorted(list(all_indices - assigned_indices))
    unexpected_indices = sorted(list(assigned_indices - all_indices))
    coverage_ratio = len(assigned_indices) / total_windows if total_windows > 0 else 0.0

    if len(unassigned_indices) > 0:
        split_failures.append(f"Incomplete split coverage: {len(unassigned_indices)} unassigned samples")
    if len(unexpected_indices) > 0:
        split_failures.append(f"Unexpected split indices: {len(unexpected_indices)} indices out of range")

    index_split_status = "PASSED" if not split_failures else "FAILED"

    # 5. Cross-Split Signal Leakage Audit
    split_hashes: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    for h, insts in hash_to_samples.items():
        for item in insts:
            s_name = item["split"]
            if s_name in split_hashes:
                split_hashes[s_name].add(h)

    tr_va_dup_hashes = split_hashes["train"] & split_hashes["val"]
    tr_te_dup_hashes = split_hashes["train"] & split_hashes["test"]
    va_te_dup_hashes = split_hashes["val"] & split_hashes["test"]

    has_cross_split_signal_leakage = (
        len(tr_va_dup_hashes) > 0 or len(tr_te_dup_hashes) > 0 or len(va_te_dup_hashes) > 0
    )

    cross_split_leakage_examples = []
    all_cross_leakage_hashes = tr_va_dup_hashes | tr_te_dup_hashes | va_te_dup_hashes
    for h in list(all_cross_leakage_hashes)[:20]:
        insts = hash_to_samples[h]
        cross_split_leakage_examples.append({
            "hash": h,
            "indices": [x["index"] for x in insts],
            "labels": [x["label"] for x in insts],
            "splits": [x["split"] for x in insts],
            "group_ids": [x["group_id"] for x in insts],
        })

    signal_leakage_failures = []
    if has_cross_split_signal_leakage:
        signal_leakage_failures.append("Identical respiration waveforms cross split boundaries")
    if has_label_inconsistency:
        signal_leakage_failures.append("Identical respiration waveforms have conflicting labels")

    signal_leakage_status = "PASSED" if not signal_leakage_failures else "FAILED"

    # 6. Group Isolation Audit
    group_isolation_section: Dict[str, Any] = {}
    if group_ids is None:
        group_isolation_status = "NOT_VERIFIABLE"
        group_isolation_section = {
            "status": "NOT_VERIFIABLE",
            "reason": "SAMPLE_LEVEL_GROUP_IDS_MISSING_OR_UNLINKED",
            "synthetic_group_isolation": "NOT_VERIFIABLE",
            "real_subject_provenance": "NOT_VERIFIABLE",
            "has_group_leakage": None,
            "train_val_group_overlap_count": None,
            "train_test_group_overlap_count": None,
            "val_test_group_overlap_count": None,
            "overlap_group_examples": [],
        }
    else:
        group_sets: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
        for i, grp in enumerate(group_ids):
            s_name = index_to_split.get(i, "UNASSIGNED")
            if s_name in group_sets:
                group_sets[s_name].add(str(grp))

        tr_va_grp_overlap = sorted(list(group_sets["train"] & group_sets["val"]))
        tr_te_grp_overlap = sorted(list(group_sets["train"] & group_sets["test"]))
        va_te_grp_overlap = sorted(list(group_sets["val"] & group_sets["test"]))

        has_group_leakage = (
            len(tr_va_grp_overlap) > 0 or len(tr_te_grp_overlap) > 0 or len(va_te_grp_overlap) > 0
        )
        group_isolation_status = "FAILED" if has_group_leakage else "PASSED"

        group_isolation_section = {
            "status": group_isolation_status,
            "reason": "Synthetic Group IDs isolated across splits" if not has_group_leakage else "Group leakage detected",
            "synthetic_group_isolation": "PASSED" if not has_group_leakage else "FAILED",
            "real_subject_provenance": "NOT_VERIFIABLE",
            "has_group_leakage": has_group_leakage,
            "train_val_group_overlap_count": len(tr_va_grp_overlap),
            "train_test_group_overlap_count": len(tr_te_grp_overlap),
            "val_test_group_overlap_count": len(va_te_grp_overlap),
            "overlap_group_examples": sorted(list(set(tr_va_grp_overlap + tr_te_grp_overlap + va_te_grp_overlap)))[:20],
        }

    # 7. Overall Status Determination
    if (
        dataset_integrity_status == "FAILED"
        or index_split_status == "FAILED"
        or signal_leakage_status == "FAILED"
        or group_isolation_status == "FAILED"
    ):
        overall_status = "FAILED"
    elif group_isolation_status == "NOT_VERIFIABLE":
        overall_status = "NOT_VERIFIABLE"
    else:
        overall_status = "PASSED"

    report_data = {
        "audit_name": "mmwave_dataset_integrity_audit",
        "audit_scope": "SYNTHETIC_NPZ_INTEGRITY_ONLY",
        "overall_status": overall_status,
        "dataset_integrity_status": dataset_integrity_status,
        "index_split_status": index_split_status,
        "signal_leakage_status": signal_leakage_status,
        "group_isolation_status": group_isolation_status,
        "execution_metadata": {
            "script_version": "1.0.0",
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "numpy_version": np.__version__,
        },
        "inputs": {
            "dataset_path": artifact_path(npz_path),
            "dataset_sha256": npz_sha256,
            "split_path": artifact_path(split_path),
            "split_sha256": split_sha256,
        },
        "schema": {
            "npz_keys": meta["npz_keys"],
            "x_key": meta["x_key"],
            "y_key": meta["y_key"],
            "group_key": meta["group_key"],
            "x_shape": list(X.shape),
            "x_dtype": str(X.dtype),
            "y_shape": list(y.shape),
            "y_dtype": str(y.dtype),
            "observed_windows": total_windows,
            "expected_windows": expected_windows,
            "count_matches_expectation": count_matches,
            "count_mismatch_reason": count_mismatch_reason,
            "schema_failures": schema_failures,
        },
        "duplicates": {
            "total_windows": total_windows,
            "unique_window_hashes": unique_window_hashes,
            "duplicate_hash_group_count": duplicate_hash_group_count,
            "duplicate_instance_count": duplicate_instance_count,
            "windows_in_duplicate_groups": windows_in_duplicate_groups,
            "duplicate_examples": duplicate_examples,
        },
        "label_consistency": {
            "has_label_inconsistency": has_label_inconsistency,
            "inconsistent_hash_count": inconsistent_hash_count,
            "inconsistent_window_count": inconsistent_window_count,
            "inconsistent_examples": inconsistent_examples,
            "class_map_used": CLASS_MAP,
        },
        "split_integrity": {
            "status": index_split_status,
            "split_index_checks": split_index_checks,
            "train_val_index_overlap_count": len(tr_va_overlap),
            "train_test_index_overlap_count": len(tr_te_overlap),
            "val_test_index_overlap_count": len(va_te_overlap),
            "train_val_index_overlap": tr_va_overlap,
            "train_test_index_overlap": tr_te_overlap,
            "val_test_index_overlap": va_te_overlap,
            "unassigned_index_count": len(unassigned_indices),
            "unassigned_indices": unassigned_indices,
            "unexpected_index_count": len(unexpected_indices),
            "unexpected_indices": unexpected_indices,
            "coverage_ratio": coverage_ratio,
            "split_failures": split_failures,
        },
        "cross_split_signal_leakage": {
            "status": signal_leakage_status,
            "has_cross_split_signal_leakage": has_cross_split_signal_leakage,
            "train_val_duplicate_hash_count": len(tr_va_dup_hashes),
            "train_test_duplicate_hash_count": len(tr_te_dup_hashes),
            "val_test_duplicate_hash_count": len(va_te_dup_hashes),
            "cross_split_leakage_examples": cross_split_leakage_examples,
            "signal_leakage_failures": signal_leakage_failures,
        },
        "group_isolation": group_isolation_section,
        "limitations": {
            "real_subject_provenance": "NOT_VERIFIABLE",
            "real_sensor_performance": "NOT_VERIFIABLE",
            "note": "Source NPZ dataset lacks sample-level subject/session provenance. Group isolation audit reports NOT_VERIFIABLE."
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return report_data


def main():
    parser = argparse.ArgumentParser(
        description="SafeNest V6 Automated Audit of mmWave Data Integrity, Duplicate Signals, and Group Leakage"
    )
    parser.add_argument(
        "--npz",
        type=str,
        default="datasets/mmwave/processed/mmwave_respiration_v1.npz",
        help="Path to NPZ dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="datasets/mmwave/splits/mmwave_group_split_v1.json",
        help="Path to split JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/mmwave_dataset_integrity_audit.json",
        help="Path to output audit JSON",
    )
    parser.add_argument(
        "--expected-windows",
        type=int,
        default=3118,
        help="Expected total window count for schema comparison",
    )

    args = parser.parse_args()

    npz_path = (project_root / args.npz).resolve() if not Path(args.npz).is_absolute() else Path(args.npz)
    split_path = (project_root / args.split).resolve() if not Path(args.split).is_absolute() else Path(args.split)
    output_path = (project_root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    try:
        report = run_integrity_audit(
            npz_path=npz_path,
            split_path=split_path,
            output_path=output_path,
            expected_windows=args.expected_windows,
        )
    except Exception as e:
        print(f"❌ Input or Execution Error: {e}", file=sys.stderr)
        sys.exit(3)

    overall_status = report.get("overall_status", "FAILED")
    print(f"📋 Audit Complete. Overall Status: [{overall_status}]")
    print(f"  - Report written to: {output_path}")
    print(f"  - Total Windows Audited: {report['schema']['observed_windows']}")
    print(f"  - Unique Waveform Hashes: {report['duplicates']['unique_window_hashes']}")
    print(f"  - Duplicate Hash Groups: {report['duplicates']['duplicate_hash_group_count']}")
    print(f"  - Duplicate Instance Count: {report['duplicates']['duplicate_instance_count']}")
    print(f"  - Conflicting Label Hashes: {report['label_consistency']['inconsistent_hash_count']}")
    print(f"  - Cross-Split Duplicate Signals: {report['cross_split_signal_leakage']['has_cross_split_signal_leakage']}")
    print(f"  - Group Isolation Status: {report['group_isolation']['status']}")

    if overall_status == "PASSED":
        sys.exit(0)
    elif overall_status == "FAILED":
        sys.exit(1)
    elif overall_status == "NOT_VERIFIABLE":
        sys.exit(2)
    else:
        sys.exit(3)


if __name__ == "__main__":
    main()
