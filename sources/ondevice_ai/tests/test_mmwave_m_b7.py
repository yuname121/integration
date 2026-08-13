"""Focused fail-closed corruption tests for SafeNest mmWave M-B7."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from mmwave_phase_b_access import PhaseBAccessGuard  # noqa: E402
from run_mmwave_m_b7 import (  # noqa: E402
    MANIFEST_RELATIVE,
    REQUIRED_OUTPUT_FILENAMES,
    compute_m_b7_evidence,
)
from validate_mmwave_m_b7 import (  # noqa: E402
    MB7ValidationError,
    validate_m_b7_artifacts,
)


class TestMmwaveMB7(unittest.TestCase):
    """Prove that self-consistent artifact corruption cannot pass validation."""

    @classmethod
    def setUpClass(cls) -> None:
        # Recompute once with fresh strict-INT8 inference; individual corruption
        # tests reuse this immutable independent reference to keep the suite fast.
        cls.reference = compute_m_b7_evidence(ROOT_DIR)
        cls.train_window = PhaseBAccessGuard(root_dir=ROOT_DIR).get_model_selection_dataset(
            "TRAIN"
        )["windows"][0]
        cls.source_manifest = ROOT_DIR / MANIFEST_RELATIVE

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="safenest_m_b7_test_")
        self.manifest = Path(self.temp_dir.name) / "M-B7_perturbation_robustness"
        shutil.copytree(self.source_manifest, self.manifest)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _load_json(self, filename: str) -> Dict[str, Any]:
        return json.loads((self.manifest / filename).read_text(encoding="utf-8"))

    def _write_json(self, filename: str, value: Dict[str, Any]) -> None:
        (self.manifest / filename).write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def _load_jsonl(self, filename: str) -> List[Dict[str, Any]]:
        return [
            json.loads(line)
            for line in (self.manifest / filename).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _write_jsonl(self, filename: str, rows: List[Dict[str, Any]]) -> None:
        (self.manifest / filename).write_text(
            "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _refresh_checksums(self) -> None:
        lines = []
        for filename in REQUIRED_OUTPUT_FILENAMES:
            digest = hashlib.sha256((self.manifest / filename).read_bytes()).hexdigest()
            lines.append(f"{digest}  {filename}")
        (self.manifest / "checksums.sha256").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def _corrupt_json(self, filename: str, mutation: Callable[[Dict[str, Any]], None]) -> None:
        value = self._load_json(filename)
        mutation(value)
        self._write_json(filename, value)
        self._refresh_checksums()

    def _corrupt_jsonl(
        self, filename: str, mutation: Callable[[List[Dict[str, Any]]], None]
    ) -> None:
        rows = self._load_jsonl(filename)
        mutation(rows)
        self._write_jsonl(filename, rows)
        self._refresh_checksums()

    def _assert_rejected(self) -> None:
        with self.assertRaises(MB7ValidationError):
            validate_m_b7_artifacts(
                root_dir=ROOT_DIR,
                manifest_dir=self.manifest,
                verify_upstream=False,
                reference=self.reference,
            )

    def _first_profile_row(self, rows: List[Dict[str, Any]], profile_id: str) -> Dict[str, Any]:
        return next(row for row in rows if row["profile_id"] == profile_id)

    def test_00_clean_artifacts_pass_independent_recomputation(self) -> None:
        result = validate_m_b7_artifacts(
            root_dir=ROOT_DIR,
            manifest_dir=self.manifest,
            verify_upstream=False,
            reference=self.reference,
        )
        self.assertTrue(result["validation_success"])
        self.assertTrue(result["independently_measured"]["fresh_int8_inference_verified"])

    def test_01_rejects_clean_m_b6_baseline_identity_corruption(self) -> None:
        self._corrupt_json(
            "clean_baseline_results.json",
            lambda value: value["per_seed"]["42"]["top1_predictions"].__setitem__(0, 2),
        )
        self._assert_rejected()

    def test_02_rejects_wrong_model_sha(self) -> None:
        self._corrupt_json(
            "perturbation_runs.json",
            lambda value: value["model_artifacts"]["42"].__setitem__("sha256", "0" * 64),
        )
        self._assert_rejected()

    def test_03_rejects_validation_row_replaced_by_train_row(self) -> None:
        def mutate(rows: List[Dict[str, Any]]) -> None:
            row = rows[0]
            for key in (
                "canonical_sample_index",
                "window_id",
                "subject_id",
                "recording_id",
                "safenest_label_id",
            ):
                if key in self.train_window:
                    row[key if key != "safenest_label_id" else "true_class"] = self.train_window[key]
            row["true_label"] = self.train_window["safenest_label"]
            row["split"] = "TRAIN"

        self._corrupt_jsonl("perturbation_sample_index.jsonl", mutate)
        self._assert_rejected()

    def test_04_rejects_locked_test_row_inserted(self) -> None:
        def mutate(rows: List[Dict[str, Any]]) -> None:
            inserted = copy.deepcopy(rows[0])
            inserted["split"] = "LOCKED_TEST"
            rows.append(inserted)

        self._corrupt_jsonl("perturbation_sample_index.jsonl", mutate)
        self._assert_rejected()

    def test_05_rejects_gaussian_wrong_target_snr(self) -> None:
        def mutate(rows: List[Dict[str, Any]]) -> None:
            self._first_profile_row(rows, "M-B7_GAUSSIAN_SNR20")["parameters"][
                "target_snr_db"
            ] = 19.0

        self._corrupt_jsonl("perturbation_sample_index.jsonl", mutate)
        self._assert_rejected()

    def test_06_rejects_gaussian_nondeterministic_seed(self) -> None:
        def mutate(rows: List[Dict[str, Any]]) -> None:
            row = self._first_profile_row(rows, "M-B7_GAUSSIAN_SNR10")
            row["derived_rng_seed"] += 1

        self._corrupt_jsonl("perturbation_sample_index.jsonl", mutate)
        self._assert_rejected()

    def test_07_rejects_amplitude_wrong_scale(self) -> None:
        def mutate(rows: List[Dict[str, Any]]) -> None:
            self._first_profile_row(rows, "M-B7_AMP_X0_50")["parameters"]["scale"] = 0.51

        self._corrupt_jsonl("perturbation_sample_index.jsonl", mutate)
        self._assert_rejected()

    def test_08_rejects_drift_wrong_amplitude(self) -> None:
        def mutate(rows: List[Dict[str, Any]]) -> None:
            row = self._first_profile_row(rows, "M-B7_DRIFT_MILD")
            row["parameters"]["amplitude"] += 0.25

        self._corrupt_jsonl("perturbation_sample_index.jsonl", mutate)
        self._assert_rejected()

    def test_09_rejects_drift_wrong_frequency_metadata(self) -> None:
        def mutate(rows: List[Dict[str, Any]]) -> None:
            self._first_profile_row(rows, "M-B7_DRIFT_SEVERE")["parameters"][
                "frequency_hz"
            ] = 0.06

        self._corrupt_jsonl("perturbation_sample_index.jsonl", mutate)
        self._assert_rejected()

    def test_10_rejects_dropout_wrong_duration(self) -> None:
        def mutate(rows: List[Dict[str, Any]]) -> None:
            self._first_profile_row(rows, "M-B7_DROPOUT_SHORT")["parameters"][
                "duration_samples"
            ] = 6

        self._corrupt_jsonl("perturbation_sample_index.jsonl", mutate)
        self._assert_rejected()

    def test_11_rejects_dropout_mask_corruption(self) -> None:
        def mutate(rows: List[Dict[str, Any]]) -> None:
            mask = self._first_profile_row(rows, "M-B7_DROPOUT_LONG")["parameters"][
                "dropout_mask"
            ]
            mask[0] = 1 - mask[0]

        self._corrupt_jsonl("perturbation_sample_index.jsonl", mutate)
        self._assert_rejected()

    def test_12_rejects_missing_frame_index_corruption(self) -> None:
        def mutate(rows: List[Dict[str, Any]]) -> None:
            indices = self._first_profile_row(rows, "M-B7_MISSING_FRAME_1PCT")["parameters"][
                "removed_indices"
            ]
            indices[0] += 1

        self._corrupt_jsonl("perturbation_sample_index.jsonl", mutate)
        self._assert_rejected()

    def test_13_rejects_motion_burst_wrong_magnitude(self) -> None:
        def mutate(rows: List[Dict[str, Any]]) -> None:
            row = self._first_profile_row(rows, "M-B7_MOTION_BURST_SEVERE")
            row["parameters"]["signed_amplitude"] *= 0.9

        self._corrupt_jsonl("perturbation_sample_index.jsonl", mutate)
        self._assert_rejected()

    def test_14_rejects_combined_wrong_application_order(self) -> None:
        def mutate(rows: List[Dict[str, Any]]) -> None:
            order = self._first_profile_row(rows, "M-B7_COMBINED_MODERATE")["parameters"][
                "application_order"
            ]
            order[0], order[1] = order[1], order[0]

        self._corrupt_jsonl("perturbation_sample_index.jsonl", mutate)
        self._assert_rejected()

    def test_15_rejects_preprocessing_attenuation_corruption(self) -> None:
        def mutate(value: Dict[str, Any]) -> None:
            value["profiles"]["M-B7_GAUSSIAN_SNR20"]["post_to_pre_rms_ratio"][
                "mean"
            ] += 0.1

        self._corrupt_json("preprocessing_attenuation_audit.json", mutate)
        self._assert_rejected()

    def test_16_rejects_prediction_vector_corruption(self) -> None:
        path = self.manifest / "prediction_vectors.npz"
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in source.files}
        key = "seed_42__M-B7_GAUSSIAN_SNR20__predictions"
        arrays[key][0] = (int(arrays[key][0]) + 1) % 3
        np.savez_compressed(path, **arrays)
        self._refresh_checksums()
        self._assert_rejected()

    def test_17_rejects_macro_f1_degradation_corruption(self) -> None:
        def mutate(value: Dict[str, Any]) -> None:
            metrics = value["profiles"]["M-B7_GAUSSIAN_SNR20"]["per_seed"]["42"]
            metrics["relative_to_clean"]["positive_macro_f1_degradation"] += 0.01

        self._corrupt_json("perturbation_results.json", mutate)
        self._assert_rejected()

    def test_18_rejects_per_class_recall_degradation_corruption(self) -> None:
        def mutate(value: Dict[str, Any]) -> None:
            metrics = value["profiles"]["M-B7_GAUSSIAN_SNR10"]["per_seed"]["43"]
            metrics["relative_to_clean"]["per_class_positive_recall_degradation"][
                "APNEA"
            ] += 0.01

        self._corrupt_json("perturbation_results.json", mutate)
        self._assert_rejected()

    def test_19_rejects_false_collapse_state(self) -> None:
        def mutate(value: Dict[str, Any]) -> None:
            state = value["profiles"]["M-B7_GAUSSIAN_SNR20"]["per_seed"]["42"][
                "class_collapse_state"
            ]
            state["collapsed"] = not state["collapsed"]

        self._corrupt_json("perturbation_results.json", mutate)
        self._assert_rejected()

    def test_20_rejects_saturation_corruption(self) -> None:
        def mutate(value: Dict[str, Any]) -> None:
            run = value["runs"]["seed_42__M-B7_MOTION_BURST_SEVERE"]
            run["saturated_element_count"] += 1

        self._corrupt_json("quantization_diagnostics.json", mutate)
        self._assert_rejected()

    def test_21_rejects_confidence_summary_corruption(self) -> None:
        def mutate(value: Dict[str, Any]) -> None:
            confidence = value["profiles"]["M-B7_GAUSSIAN_POST_B1_SNR20"]["per_seed"][
                "44"
            ]["confidence"]["all_predictions"]
            confidence["mean"] += 0.01

        self._corrupt_json("perturbation_results.json", mutate)
        self._assert_rejected()

    def test_22_rejects_subject_level_confusion_corruption(self) -> None:
        def mutate(value: Dict[str, Any]) -> None:
            per_subject = value["profiles"]["M-B7_GAUSSIAN_SNR20"]["per_seed"]["42"][
                "per_subject"
            ]
            first_subject = sorted(per_subject)[0]
            per_subject[first_subject]["per_class"]["NORMAL"]["fp"] += 1

        self._corrupt_json("subject_level_robustness.json", mutate)
        self._assert_rejected()

    def test_23_rejects_cross_seed_worst_seed_corruption(self) -> None:
        def mutate(value: Dict[str, Any]) -> None:
            metric = value["profiles"]["M-B7_GAUSSIAN_SNR20"]["macro_f1_degradation"]
            metric["worst_seed"] = 44 if metric["worst_seed"] != 44 else 42

        self._corrupt_json("cross_seed_robustness_summary.json", mutate)
        self._assert_rejected()

    def test_24_rejects_locked_test_audit_corruption(self) -> None:
        self._corrupt_json(
            "locked_test_access_audit.json",
            lambda value: value.__setitem__("performance_access_attempts", 1),
        )
        self._assert_rejected()

    def test_25_rejects_profile_contract_name_only_forgery(self) -> None:
        def mutate(value: Dict[str, Any]) -> None:
            value["profiles"]["M-B7_DRIFT_MILD"]["frequency_hz"] = 0.04

        self._corrupt_json("perturbation_profile_contract.json", mutate)
        self._assert_rejected()

    def test_26_rejects_determinism_fingerprint_corruption(self) -> None:
        def mutate(value: Dict[str, Any]) -> None:
            example = value["profiles"]["M-B7_GAUSSIAN_SNR20"][
                "numeric_fingerprint_example"
            ]
            example["model_input_sha256_float32"] = "f" * 64

        self._corrupt_json("determinism_audit.json", mutate)
        self._assert_rejected()

    def test_27_rejects_malformed_checksum(self) -> None:
        path = self.manifest / "checksums.sha256"
        path.write_text(path.read_text(encoding="utf-8") + "not-a-digest\n", encoding="utf-8")
        self._assert_rejected()

    def test_28_rejects_checksum_path_traversal(self) -> None:
        path = self.manifest / "checksums.sha256"
        lines = path.read_text(encoding="utf-8").splitlines()
        digest = lines[0].split(maxsplit=1)[0]
        lines[0] = f"{digest}  ../escaped.json"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._assert_rejected()


if __name__ == "__main__":
    unittest.main()
