#!/usr/bin/env python3
"""SafeNest mmWave Phase B Data Access Guard & LOCKED_TEST Access Controller.

Provides controlled data access for mmWave Phase-B experiments, enforcing
TRAIN-only fitting, VALIDATION-only selection, and strict LOCKED_TEST isolation.
Attempts to access LOCKED_TEST during model selection fail closed with an exception.
Structural leakage audits on LOCKED_TEST return sanitized identity/signal structures
without exposing class labels or label derivation metadata.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]

ALLOWED_STRUCTURAL_FIELDS = {
    "canonical_sample_index",
    "canonical_signal_hash",
    "window_id",
    "subject_id",
    "recording_id",
    "split",
    "start_timestamp",
    "last_sample_timestamp",
    "end_timestamp_exclusive",
}

FORBIDDEN_LABEL_FIELDS = {
    "safenest_label",
    "safenest_label_id",
    "mapping_evidence",
    "mapping_rule_id",
    "original_annotation_type",
    "movesense_reference_rr",
    "assignment_status",
    "training_eligible",
    "validation_eligible",
    "locked_test_evaluation_eligible",
}


class LOCKED_TEST_AccessError(Exception):
    """Raised when model selection or hyperparameter search attempts to access LOCKED_TEST data."""


class PhaseBAccessGuard:
    """Guards dataset access for Phase B experiments and enforces split isolation."""

    def __init__(self, root_dir: Path = ROOT_DIR) -> None:
        self.root_dir = root_dir
        self.split_json_path = root_dir / "datasets/mmwave/splits/mmwave_real_subject_split_v1.json"
        self.window_manifest_path = root_dir / "datasets/mmwave/manifests/a6_full_conversion/full_window_manifest.jsonl"
        self.provenance_manifest_path = root_dir / "datasets/mmwave/manifests/a6_full_conversion/full_provenance_manifest.jsonl"
        self.canonical_npy_path = root_dir / "datasets/mmwave/processed/mmwave_canonical_real_v1.npy"

        self._load_datasets()

    def _load_datasets(self) -> None:
        if not self.split_json_path.is_file():
            raise FileNotFoundError(f"Real subject split JSON missing: {self.split_json_path}")
        if not self.window_manifest_path.is_file():
            raise FileNotFoundError(f"Window manifest missing: {self.window_manifest_path}")
        if not self.provenance_manifest_path.is_file():
            raise FileNotFoundError(f"Provenance manifest missing: {self.provenance_manifest_path}")
        if not self.canonical_npy_path.is_file():
            raise FileNotFoundError(f"Canonical numeric NPY missing: {self.canonical_npy_path}")

        self.split_data = json.loads(self.split_json_path.read_text(encoding="utf-8"))
        self.subject_split_map = self.split_data.get("subject_split_map", {})

        self.windows = []
        with open(self.window_manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.windows.append(json.loads(line))

        self.provenance = []
        with open(self.provenance_manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.provenance.append(json.loads(line))

        self.canonical_matrix = np.load(self.canonical_npy_path)

        if len(self.windows) != len(self.provenance) or len(self.windows) != self.canonical_matrix.shape[0]:
            raise ValueError(
                f"1:1 alignment mismatch! Windows: {len(self.windows)}, Provenance: {len(self.provenance)}, NPY: {self.canonical_matrix.shape[0]}"
            )

    def get_train_data(self, include_ambiguous: bool = False) -> dict[str, Any]:
        """Return TRAIN split dataset (windows, provenance, and NPY matrix slices)."""
        return self._get_split_dataset("TRAIN", include_ambiguous=include_ambiguous)

    def get_validation_data(self, include_ambiguous: bool = False) -> dict[str, Any]:
        """Return VALIDATION split dataset (windows, provenance, and NPY matrix slices)."""
        return self._get_split_dataset("VALIDATION", include_ambiguous=include_ambiguous)

    def get_model_selection_dataset(self, split_name: str, include_ambiguous: bool = False) -> dict[str, Any]:
        """Return split dataset for model selection. Refuses LOCKED_TEST access."""
        split_upper = split_name.upper()
        if split_upper == "LOCKED_TEST":
            raise LOCKED_TEST_AccessError(
                "LOCKED_TEST data is strictly prohibited during model selection, preprocessing ablation, or hyperparameter tuning!"
            )
        if split_upper not in ("TRAIN", "VALIDATION"):
            raise ValueError(f"Invalid split name for model selection: {split_name}")

        return self._get_split_dataset(split_upper, include_ambiguous=include_ambiguous)

    def get_structural_audit_dataset(self, split_name: str) -> dict[str, Any]:
        """Return sanitized dataset specifically for structural audits (split isolation, duplicate checks).

        Strips all class labels and label derivation metadata for LOCKED_TEST to prevent label leakage.
        """
        split_upper = split_name.upper()
        if split_upper not in ("TRAIN", "VALIDATION", "LOCKED_TEST"):
            raise ValueError(f"Invalid split name: {split_name}")

        indices = []
        sanitized_windows = []

        for idx, w in enumerate(self.windows):
            if w["split"] == split_upper:
                indices.append(idx)
                # Create sanitized window record containing ONLY non-label structural fields
                clean_w = {k: copy.deepcopy(v) for k, v in w.items() if k in ALLOWED_STRUCTURAL_FIELDS}
                sanitized_windows.append(clean_w)

        signals_copy = np.copy(self.canonical_matrix[indices]) if indices else np.empty((0, 300), dtype=np.float64)

        return {
            "split": split_upper,
            "sample_indices": indices,
            "windows": sanitized_windows,
            "signals": signals_copy,
            "total_count": len(sanitized_windows),
            "sanitized_for_structural_audit": True,
        }

    def get_locked_test_final_evaluation_dataset(self, authorization_token: str | None = None) -> dict[str, Any]:
        """Return LOCKED_TEST dataset for final independent evaluation. Requires explicit authorization token."""
        if not authorization_token or authorization_token != "AUTHORIZED_FINAL_LOCKED_TEST_EVALUATION_TOKEN_V1":
            raise LOCKED_TEST_AccessError(
                "LOCKED_TEST final evaluation dataset requires an explicit authorization token and pre-registered candidate model!"
            )
        return self._get_split_dataset("LOCKED_TEST", include_ambiguous=False)

    def _get_split_dataset(self, split_name: str, include_ambiguous: bool = False) -> dict[str, Any]:
        indices = []
        sub_windows = []
        sub_provenance = []

        for idx, (w, p) in enumerate(zip(self.windows, self.provenance)):
            if w["split"] == split_name:
                if not include_ambiguous and w["assignment_status"] == "AMBIGUOUS":
                    continue
                indices.append(idx)
                sub_windows.append(copy.deepcopy(w))
                sub_provenance.append(copy.deepcopy(p))

        npy_slices = np.copy(self.canonical_matrix[indices]) if indices else np.empty((0, 300), dtype=np.float64)

        return {
            "split": split_name,
            "sample_indices": indices,
            "windows": sub_windows,
            "provenance": sub_provenance,
            "signals": npy_slices,
            "total_count": len(sub_windows),
        }
