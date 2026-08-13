"""Focused fail-closed tests for M-B11 artifact lock.

Mutations use temporary copies. No LOCKED_TEST or recovery access.
No TFLite invoke is required for a passing validation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import mmwave_m_b11_artifact_lock as generator
from scripts import validate_mmwave_m_b11 as validator
from scripts.mmwave_m_b10r1_result_writer import SELECTED_MODEL_ID, V01_MODEL_ID, V02_MODEL_ID
from scripts.mmwave_m_b11_artifact_lock import B_DIR_REL, LOCK_DIR_REL, SENSOR_LOCK_REL, write_checksums

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / LOCK_DIR_REL
B_EVIDENCE = ROOT / B_DIR_REL
SENSOR = ROOT / SENSOR_LOCK_REL


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_checksums(path: Path) -> None:
    write_checksums(path)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class MB11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (LOCK / "artifact_lock_identity.json").is_file():
            generator.generate_m_b11_artifact_lock(ROOT)

    def _copy_lock(self) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="m_b11_lock_"))
        shutil.copytree(LOCK, tmp / "lock")
        return tmp / "lock"

    def _validate_copy(self, lock_dir: Path) -> None:
        validator.validate_m_b11(ROOT, lock_dir=lock_dir)

    def _mutate(self, filename: str, mutator, rewrite: bool = True) -> Path:
        lock_dir = self._copy_lock()
        path = lock_dir / filename
        payload = _load(path)
        mutator(payload)
        _dump(path, payload)
        if rewrite:
            _rewrite_checksums(lock_dir)
        return lock_dir

    def _copy_b(self) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="m_b11_b_"))
        shutil.copytree(B_EVIDENCE, tmp / "b")
        return tmp / "b"

    def _mutate_b_ledger(self, mutator) -> Path:
        b_dir = self._copy_b()
        path = b_dir / "recovery_sample_predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        mutator(rows)
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        return b_dir

    def _expect_fail(self, lock_dir: Path, fragment: str) -> None:
        with self.assertRaises(validator.MB11ValidationError) as ctx:
            self._validate_copy(lock_dir)
        self.assertIn(fragment, str(ctx.exception))

    def _expect_fail_b(self, b_dir: Path, fragment: str) -> None:
        with self.assertRaises(validator.MB11ValidationError) as ctx:
            validator.validate_m_b11(ROOT, b_dir=b_dir)
        self.assertIn(fragment, str(ctx.exception))

    def test_valid_lock_passes(self) -> None:
        result = validator.validate_m_b11(ROOT)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["generator_ledger_analyzer_reused"])
        self.assertEqual(result["source_ledger"]["unique_ids"], 75)
        self.assertEqual(result["source_ledger"]["models"], 3)
        self.assertEqual(result["source_ledger"]["pairs"], 225)
        self.assertEqual(result["source_ledger"]["duplicates"], 0)
        self.assertEqual(result["source_ledger"]["missing"], 0)
        self.assertEqual(result["source_ledger"]["unexpected"], 0)
        self.assertEqual(result["source_ledger"]["label_mismatches"], 0)
        self.assertEqual(result["source_ledger"]["subject_mismatches"], 0)
        self.assertEqual(result["source_ledger"]["recording_mismatches"], 0)

    def test_no_access_monkeypatch_still_passes(self) -> None:
        def boom(*_args, **_kwargs):
            raise AssertionError("ACCESSOR_TOUCHED")

        with mock.patch(
            "scripts.mmwave_phase_b_access.PhaseBAccessGuard.get_locked_test_final_evaluation_dataset",
            side_effect=boom,
        ), mock.patch(
            "scripts.mmwave_m_b10r1_recovery_access.LimitedReuseRecoveryAccessController.get_locked_test_recovery_evaluation_dataset",
            side_effect=boom,
        ):
            result = validator.validate_m_b11(ROOT)
        self.assertEqual(result["status"], "PASS")

    def test_no_inference_monkeypatch_still_passes(self) -> None:
        import tensorflow as tf  # type: ignore

        def boom(*_args, **_kwargs):
            raise AssertionError("TFLITE_INVOKE_TOUCHED")

        with mock.patch.object(tf.lite.Interpreter, "invoke", side_effect=boom):
            result = validator.validate_m_b11(ROOT)
        self.assertEqual(result["status"], "PASS")

    def test_selected_model_sha_changed(self) -> None:
        lock_dir = self._mutate(
            "model_artifact_lock.json",
            lambda payload: payload.__setitem__("sha256", "0" * 64),
        )
        self._expect_fail(lock_dir, "MODEL_SHA")

    def test_a5_split_sha_changed(self) -> None:
        lock_dir = self._mutate(
            "subject_split_lock.json",
            lambda payload: payload.__setitem__("split_sha256", "0" * 64),
        )
        self._expect_fail(lock_dir, "A5_SPLIT_SHA")

    def test_a6_identity_changed(self) -> None:
        lock_dir = self._mutate(
            "window_population_lock.json",
            lambda payload: payload.__setitem__("a6_manifest_sha256", "0" * 64),
        )
        self._expect_fail(lock_dir, "A6_MANIFEST_SHA")

    def test_preprocessing_id_changed(self) -> None:
        lock_dir = self._mutate(
            "preprocessing_lock.json",
            lambda payload: payload.__setitem__("selected_profile_id", "M-B1_D1_B1_Z1"),
        )
        self._expect_fail(lock_dir, "PREPROCESSING_ID")

    def test_calibration_id_changed(self) -> None:
        lock_dir = self._mutate(
            "model_artifact_lock.json",
            lambda payload: payload.__setitem__("calibration_profile", "M-B5_CAL_TRAIN_ORDER_120"),
        )
        self._expect_fail(lock_dir, "CALIBRATION_ID")

    def test_one_sample_removed(self) -> None:
        def mutate(payload: dict) -> None:
            payload["samples"].pop()
            payload["ordered_window_ids"].pop()
            payload["unique_eligible_window_ids"] = 74
            payload["actual_pairs"] = 222

        lock_dir = self._mutate("final_sample_registry_lock.json", mutate)
        self._expect_fail(lock_dir, "LOCK_UNIQUE_IDS")

    def test_one_sample_duplicated(self) -> None:
        def mutate(payload: dict) -> None:
            payload["samples"].append(payload["samples"][0])
            payload["unique_eligible_window_ids"] = 76

        lock_dir = self._mutate("final_sample_registry_lock.json", mutate)
        self._expect_fail(lock_dir, "LOCK_UNIQUE_IDS")

    def test_one_model_sample_pair_removed(self) -> None:
        def mutate(payload: dict) -> None:
            del payload["samples"][0]["models"][V01_MODEL_ID]
            payload["actual_pairs"] = 224

        lock_dir = self._mutate("final_sample_registry_lock.json", mutate)
        self._expect_fail(lock_dir, "LOCK_PAIRS")

    def test_duplicate_pair_inserted(self) -> None:
        def mutate(payload: dict) -> None:
            extra = json.loads(json.dumps(payload["samples"][0]))
            extra["window_id"] = extra["window_id"] + "__DUP"
            payload["samples"].append(extra)

        lock_dir = self._mutate("final_sample_registry_lock.json", mutate)
        self._expect_fail(lock_dir, "LOCK_SAMPLE_COUNT")

    def test_label_changed_for_only_one_model_row(self) -> None:
        def mutate(payload: dict) -> None:
            payload["samples"][0]["models"][V01_MODEL_ID]["true_class"] = "APNEA"

        lock_dir = self._mutate("final_sample_registry_lock.json", mutate)
        self._expect_fail(lock_dir, "LOCK_CROSS_MODEL_LABEL")

    def test_subject_changed_for_only_one_model_row(self) -> None:
        def mutate(payload: dict) -> None:
            payload["samples"][0]["models"][V02_MODEL_ID]["subject_id"] = "mutated-subject"

        lock_dir = self._mutate("final_sample_registry_lock.json", mutate)
        self._expect_fail(lock_dir, "LOCK_CROSS_MODEL_SUBJECT")

    def test_fourth_model_inserted(self) -> None:
        def mutate(payload: dict) -> None:
            payload["model_ids"].append("fourth_model")
            clone = json.loads(json.dumps(payload["samples"][0]["models"][SELECTED_MODEL_ID]))
            clone["model_id"] = "fourth_model"
            payload["samples"][0]["models"]["fourth_model"] = clone

        lock_dir = self._mutate("final_sample_registry_lock.json", mutate)
        self._expect_fail(lock_dir, "LOCK_SAMPLE_MODEL_SET")

    def test_seed43_inserted(self) -> None:
        def mutate(payload: dict) -> None:
            payload["candidate_id"] = "M-B3_CONV1D_GAP_BASELINE_seed43_M-B5_CAL_CLASS_BALANCED_120"
            payload["seed"] = 43

        lock_dir = self._mutate("model_artifact_lock.json", mutate)
        self._expect_fail(lock_dir, "SELECTED_MODEL_RESELECTION")

    def test_macro_f1_altered(self) -> None:
        lock_dir = self._mutate(
            "final_metric_lock.json",
            lambda payload: payload.__setitem__("macro_f1", 0.999999),
        )
        self._expect_fail(lock_dir, "LOCK_METRIC_macro_f1")

    def test_confusion_matrix_altered(self) -> None:
        def mutate(payload: dict) -> None:
            payload["confusion_matrix"][0][0] = 99

        lock_dir = self._mutate("final_metric_lock.json", mutate)
        self._expect_fail(lock_dir, "LOCK_METRIC_confusion_matrix")

    def test_apnea_misses_altered(self) -> None:
        def mutate(payload: dict) -> None:
            payload["apnea_proxy"]["misses"] = 99

        lock_dir = self._mutate("final_metric_lock.json", mutate)
        self._expect_fail(lock_dir, "LOCK_APNEA_MISSES")

    def test_fpr_altered(self) -> None:
        def mutate(payload: dict) -> None:
            payload["apnea_proxy"]["fpr"] = 0.0

        lock_dir = self._mutate("final_metric_lock.json", mutate)
        self._expect_fail(lock_dir, "LOCK_APNEA_FPR")

    def test_subject_median_altered(self) -> None:
        lock_dir = self._mutate(
            "final_subject_metric_lock.json",
            lambda payload: payload.__setitem__("median_subject_macro_f1", 0.999999),
        )
        self._expect_fail(lock_dir, "LOCK_MEDIAN")

    def test_worst_subject_altered(self) -> None:
        lock_dir = self._mutate(
            "final_subject_metric_lock.json",
            lambda payload: payload.__setitem__("worst_subject_id", "mutated-worst"),
        )
        self._expect_fail(lock_dir, "LOCK_WORST_ID")

    def test_original_release_set_to_0(self) -> None:
        lock_dir = self._mutate(
            "recovery_access_history_lock.json",
            lambda payload: payload.__setitem__("original_m_b10b_payload_releases", 0),
        )
        self._expect_fail(lock_dir, "ORIG_RELEASE")

    def test_recovery_release_set_to_0(self) -> None:
        lock_dir = self._mutate(
            "recovery_access_history_lock.json",
            lambda payload: payload.__setitem__("m_b10r1b_recovery_payload_releases", 0),
        )
        self._expect_fail(lock_dir, "REC_RELEASE")

    def test_historical_total_set_to_1(self) -> None:
        lock_dir = self._mutate(
            "recovery_access_history_lock.json",
            lambda payload: payload.__setitem__("historical_total_payload_releases", 1),
        )
        self._expect_fail(lock_dir, "HIST_TOTAL")

    def test_historical_total_set_to_3(self) -> None:
        lock_dir = self._mutate(
            "recovery_access_history_lock.json",
            lambda payload: payload.__setitem__("historical_total_payload_releases", 3),
        )
        self._expect_fail(lock_dir, "HIST_TOTAL")

    def test_rerun_true(self) -> None:
        lock_dir = self._mutate(
            "recovery_access_history_lock.json",
            lambda payload: payload.__setitem__("rerun", True),
        )
        self._expect_fail(lock_dir, "RERUN_TRUE")

    def test_second_recovery_true(self) -> None:
        lock_dir = self._mutate(
            "recovery_access_history_lock.json",
            lambda payload: payload.__setitem__("second_recovery", True),
        )
        self._expect_fail(lock_dir, "SECOND_RECOVERY_TRUE")

    def test_result_not_pristine_false(self) -> None:
        lock_dir = self._mutate(
            "claim_boundary_lock.json",
            lambda payload: payload.__setitem__("result_not_pristine", False),
        )
        self._expect_fail(lock_dir, "RESULT_NOT_PRISTINE_FALSE")

    def test_pristine_claim_inserted(self) -> None:
        lock_dir = self._mutate(
            "claim_boundary_lock.json",
            lambda payload: payload.__setitem__("PRISTINE_LOCKED_TEST", True),
        )
        self._expect_fail(lock_dir, "FORBIDDEN_POSITIVE_CLAIM")

    def test_mr60_validated_true(self) -> None:
        lock_dir = self._mutate(
            "claim_boundary_lock.json",
            lambda payload: payload.__setitem__("MR60_device_validation_complete", True),
        )
        self._expect_fail(lock_dir, "FORBIDDEN_POSITIVE_CLAIM")

    def test_deployment_ready_true(self) -> None:
        lock_dir = self._mutate(
            "claim_boundary_lock.json",
            lambda payload: payload.__setitem__("deployment_ready", True),
        )
        self._expect_fail(lock_dir, "FORBIDDEN_POSITIVE_CLAIM")

    def test_clinical_apnea_validated_true(self) -> None:
        lock_dir = self._mutate(
            "claim_boundary_lock.json",
            lambda payload: payload.__setitem__("clinical_apnea_validated", True),
        )
        self._expect_fail(lock_dir, "FORBIDDEN_POSITIVE_CLAIM")

    def test_identity_result_limitation_pristine(self) -> None:
        lock_dir = self._mutate(
            "artifact_lock_identity.json",
            lambda payload: payload.__setitem__("result_limitation", "PRISTINE_LOCKED_TEST"),
        )
        self._expect_fail(lock_dir, "FORBIDDEN_POSITIVE_CLAIM")

    def test_sensor_local_deployment_ready_true(self) -> None:
        sensor_tmp = Path(tempfile.mkdtemp(prefix="m_b11_sensor_")) / SENSOR.name
        shutil.copy2(SENSOR, sensor_tmp)
        payload = _load(sensor_tmp)
        payload["deployment_ready"] = True
        _dump(sensor_tmp, payload)
        with self.assertRaises(validator.MB11ValidationError) as ctx:
            validator.validate_m_b11(ROOT, sensor_lock_path=sensor_tmp)
        self.assertIn("FORBIDDEN_POSITIVE_CLAIM", str(ctx.exception))

    def test_summary_phase_b_release_ready_true(self) -> None:
        lock_dir = self._mutate(
            "artifact_lock_summary.json",
            lambda payload: payload.__setitem__("Phase_B_release_ready", True),
        )
        self._expect_fail(lock_dir, "FORBIDDEN_POSITIVE_CLAIM")

    def test_recording_changed_for_only_one_model_row(self) -> None:
        def mutate(payload: dict) -> None:
            payload["samples"][0]["models"][V01_MODEL_ID]["recording_id"] = "mutated-recording"

        lock_dir = self._mutate("final_sample_registry_lock.json", mutate)
        self._expect_fail(lock_dir, "LOCK_CROSS_MODEL_RECORDING")

    def test_sample_level_recording_changed(self) -> None:
        def mutate(payload: dict) -> None:
            payload["samples"][0]["recording_id"] = "mutated-sample-recording"

        lock_dir = self._mutate("final_sample_registry_lock.json", mutate)
        self._expect_fail(lock_dir, "LOCK_SAMPLE_RECORDING")

    def test_source_duplicate_ledger_pair(self) -> None:
        def mutate(rows: list) -> None:
            rows.append(json.loads(json.dumps(rows[0])))

        self._expect_fail_b(self._mutate_b_ledger(mutate), "SOURCE_DUPLICATE_PAIR")

    def test_source_missing_ledger_pair(self) -> None:
        def mutate(rows: list) -> None:
            rows.pop(0)

        self._expect_fail_b(self._mutate_b_ledger(mutate), "SOURCE_MISSING_PAIR")

    def test_source_unexpected_fourth_model_pair(self) -> None:
        def mutate(rows: list) -> None:
            extra = json.loads(json.dumps(rows[0]))
            extra["model_id"] = "fourth_model"
            rows.append(extra)

        self._expect_fail_b(self._mutate_b_ledger(mutate), "SOURCE_UNEXPECTED_PAIR")

    def test_source_recording_mismatch(self) -> None:
        def mutate(rows: list) -> None:
            window_id = rows[0]["window_id"]
            for row in rows:
                if row["window_id"] == window_id and row["model_id"] == V01_MODEL_ID:
                    row["recording_id"] = "mutated-source-recording"
                    return

        self._expect_fail_b(self._mutate_b_ledger(mutate), "SOURCE_RECORDING_MISMATCH")

    def test_validator_does_not_reuse_generator_ledger_analyzer(self) -> None:
        source = Path(validator.__file__).read_text(encoding="utf-8")
        self.assertNotIn("analyze_recovery_ledger", source)

    def test_absolute_path(self) -> None:
        lock_dir = self._mutate(
            "source_lineage_lock.json",
            lambda payload: payload.__setitem__(
                "raw_archive_repo_relative_path",
                "/tmp/db_records.zip",
            ),
        )
        self._expect_fail(lock_dir, "UNSAFE_PATH")

    def test_traversal_path(self) -> None:
        lock_dir = self._mutate(
            "source_lineage_lock.json",
            lambda payload: payload.__setitem__(
                "raw_archive_repo_relative_path",
                "datasets/mmwave/../raw_archives/external_datasets/db_records.zip",
            ),
        )
        self._expect_fail(lock_dir, "UNSAFE_PATH")

    def test_missing_file(self) -> None:
        lock_dir = self._copy_lock()
        (lock_dir / "final_metric_lock.json").unlink()
        self._expect_fail(lock_dir, "CHECKSUM_TARGET_MISSING")

    def test_checksum_corruption(self) -> None:
        lock_dir = self._copy_lock()
        checksum = lock_dir / "checksums.sha256"
        lines = checksum.read_text(encoding="utf-8").splitlines()
        digest, name = lines[0].split()
        lines[0] = ("0" * 64) + "  " + name
        checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._expect_fail(lock_dir, "CHECKSUM_MISMATCH")

    def test_malformed_digest(self) -> None:
        lock_dir = self._copy_lock()
        checksum = lock_dir / "checksums.sha256"
        lines = checksum.read_text(encoding="utf-8").splitlines()
        _digest, name = lines[0].split()
        lines[0] = "not-a-digest  " + name
        checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._expect_fail(lock_dir, "CHECKSUM_LINE_INVALID")

    def test_duplicate_inconsistent_checksum_path(self) -> None:
        lock_dir = self._copy_lock()
        checksum = lock_dir / "checksums.sha256"
        text = checksum.read_text(encoding="utf-8")
        first = text.splitlines()[0]
        name = first.split()[1]
        checksum.write_text(text + ("1" * 64) + "  " + name + "\n", encoding="utf-8")
        self._expect_fail(lock_dir, "CHECKSUM_DUPLICATE_INCONSISTENT")

    def test_checksum_absolute_path(self) -> None:
        lock_dir = self._copy_lock()
        checksum = lock_dir / "checksums.sha256"
        checksum.write_text(("a" * 64) + "  /tmp/evil.json\n", encoding="utf-8")
        self._expect_fail(lock_dir, "CHECKSUM_UNSAFE_PATH")

    def test_checksum_traversal(self) -> None:
        lock_dir = self._copy_lock()
        checksum = lock_dir / "checksums.sha256"
        checksum.write_text(("a" * 64) + "  ../evil.json\n", encoding="utf-8")
        self._expect_fail(lock_dir, "CHECKSUM_UNSAFE_PATH")

    def test_immutable_registry_sha_changed(self) -> None:
        def mutate(payload: dict) -> None:
            payload["artifacts"][0]["sha256"] = "0" * 64

        lock_dir = self._mutate("immutable_artifact_registry.json", mutate)
        self._expect_fail(lock_dir, "REGISTRY_SHA_MISMATCH")


if __name__ == "__main__":
    unittest.main()
