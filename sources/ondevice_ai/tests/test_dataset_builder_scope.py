#!/usr/bin/env python3
"""Regression tests preventing synthetic fixtures from impersonating real data."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from datasets.build_processed_npz import build_co2_npz, build_mmwave_npz


ROOT = Path(__file__).resolve().parent.parent


class TestDatasetBuilderScope(unittest.TestCase):
    def test_mmwave_real_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Phase A6"):
                build_mmwave_npz(root / "out", source_root=root / "raw")
            self.assertFalse((root / "out" / "mmwave_respiration_v1.npz").exists())

    def test_co2_real_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "not implemented"):
                build_co2_npz(root / "out", source_root=root / "raw")
            self.assertFalse((root / "out" / "co2_occupancy_v1.npz").exists())

    def test_manifest_separates_real_and_synthetic_mmwave(self) -> None:
        manifest = json.loads((ROOT / "datasets/MANIFEST.json").read_text(encoding="utf-8"))
        real = manifest["datasets"]["mmwave"]
        synthetic = manifest["datasets"]["mmwave_synthetic_smoke"]
        self.assertEqual(
            real["processed_file_path"],
            "datasets/mmwave/processed/mmwave_canonical_real_v1.npy",
        )
        self.assertTrue((ROOT / real["processed_file_path"]).is_file())
        self.assertTrue((ROOT / real["window_manifest_path"]).is_file())
        self.assertTrue((ROOT / real["provenance_manifest_path"]).is_file())
        self.assertEqual(real["processing_status"], "A6_PASS_WITH_WARNINGS_PHASE_B_READY_WITH_CONDITIONS")
        self.assertNotEqual(real["processed_file_path"], synthetic["file_path"])
        self.assertEqual(synthetic["data_scope"], "SYNTHETIC_SMOKE_AND_RETRAINING_ASSET")
        self.assertEqual(synthetic["real_subject_provenance"], "NOT_VERIFIABLE")


if __name__ == "__main__":
    unittest.main()
