"""Focused fail-closed tests for M-B12 Phase-B offline final closure.

Mutations use temporary copies. No LOCKED_TEST or recovery access.
No TFLite invoke is required for mutation tests.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import mmwave_m_b12_phase_b_closure as generator
from scripts import validate_mmwave_m_b12 as validator
from scripts.mmwave_m_b12_phase_b_closure import CLOSURE_DIR_REL, M_B11_DIR_REL, REPORT_REL, write_checksums

ROOT = Path(__file__).resolve().parents[1]
CLOSURE = ROOT / CLOSURE_DIR_REL
M11_REGISTRY = ROOT / M_B11_DIR_REL / "immutable_artifact_registry.json"
REPORT = ROOT / REPORT_REL


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MB12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (CLOSURE / "phase_b_closure_identity.json").is_file():
            generator.generate_m_b12_closure(ROOT)

    def _copy_closure(self) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="m_b12_closure_"))
        shutil.copytree(CLOSURE, tmp / "closure")
        return tmp / "closure"

    def _mutate(self, filename: str, mutator, rewrite: bool = True) -> Path:
        closure_dir = self._copy_closure()
        path = closure_dir / filename
        payload = _load(path)
        mutator(payload)
        _dump(path, payload)
        if rewrite:
            write_checksums(closure_dir)
        return closure_dir

    def _copy_m11_registry(self) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="m_b12_m11reg_")) / "immutable_artifact_registry.json"
        shutil.copy2(M11_REGISTRY, tmp)
        return tmp

    def _expect_fail(self, closure_dir: Path, fragment: str, **kwargs) -> None:
        with self.assertRaises(validator.MB12ValidationError) as ctx:
            validator.validate_m_b12(ROOT, closure_dir=closure_dir, skip_m_b11=True, **kwargs)
        self.assertIn(fragment, str(ctx.exception))

    def _expect_fail_registry(self, registry_path: Path, fragment: str) -> None:
        with self.assertRaises(validator.MB12ValidationError) as ctx:
            validator.validate_m_b12(ROOT, skip_m_b11=True, m11_registry_path=registry_path)
        self.assertIn(fragment, str(ctx.exception))

    def test_valid_closure_passes(self) -> None:
        result = validator.validate_m_b12(ROOT)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["macro_f1"], 0.494836)
        self.assertEqual(result["model_sha256"], generator.EXPECTED_MODEL_SHA)
        self.assertTrue(result["phase_b_offline_intermediate_release_ready_after_merge"])
        self.assertFalse(result["Phase_B_release_ready"])
        self.assertFalse(result["git_tag_created"])
        self.assertFalse(result["github_release_created"])
        self.assertFalse(result["m_c_started"])
        self.assertEqual(result["new_locked_test_access"], 0)
        self.assertEqual(result["new_recovery_access"], 0)
        self.assertEqual(result["new_model_inference"], 0)
        self.assertEqual(result["source_ledger"]["unique_ids"], 75)
        self.assertEqual(result["source_ledger"]["pairs"], 225)
        self.assertEqual(result["source_ledger"]["recording_mismatches"], 0)
        self.assertEqual(result["role_completeness"]["missing"], 0)
        self.assertTrue(result["gates"]["required_registry_roles_complete"])
        self.assertTrue(result["gates"]["candidate_contract_exact"])
        self.assertTrue(result["gates"]["final_result_contract_exact"])
        self.assertTrue(result["gates"]["report_sha_valid"])
        self.assertTrue(result["gates"]["report_machine_consistent"])

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
            result = validator.validate_m_b12(ROOT)
        self.assertEqual(result["status"], "PASS")

    def test_no_inference_monkeypatch_still_passes(self) -> None:
        import tensorflow as tf  # type: ignore

        def boom(*_args, **_kwargs):
            raise AssertionError("TFLITE_INVOKE_TOUCHED")

        with mock.patch.object(tf.lite.Interpreter, "invoke", side_effect=boom):
            result = validator.validate_m_b12(ROOT)
        self.assertEqual(result["status"], "PASS")

    def test_identity_result_limitation_pristine(self) -> None:
        closure_dir = self._mutate(
            "phase_b_closure_identity.json",
            lambda payload: payload.__setitem__("result_limitation", "PRISTINE_LOCKED_TEST"),
        )
        self._expect_fail(closure_dir, "FORBIDDEN_POSITIVE_CLAIM")

    def test_summary_phase_b_release_ready_true(self) -> None:
        closure_dir = self._mutate(
            "phase_b_closure_summary.json",
            lambda payload: payload.__setitem__("Phase_B_release_ready", True),
        )
        self._expect_fail(closure_dir, "FORBIDDEN_POSITIVE_CLAIM")

    def test_readiness_deployment_ready_true(self) -> None:
        closure_dir = self._mutate(
            "release_readiness_manifest.json",
            lambda payload: payload.__setitem__("deployment_ready", True),
        )
        self._expect_fail(closure_dir, "FORBIDDEN_POSITIVE_CLAIM")

    def test_git_tag_created_true(self) -> None:
        closure_dir = self._mutate(
            "release_readiness_manifest.json",
            lambda payload: payload.__setitem__("git_tag_created", True),
        )
        self._expect_fail(closure_dir, "FORBIDDEN_POSITIVE_CLAIM")

    def test_github_release_created_true(self) -> None:
        closure_dir = self._mutate(
            "claim_boundary.json",
            lambda payload: payload.__setitem__("github_release_created", True),
        )
        self._expect_fail(closure_dir, "FORBIDDEN_POSITIVE_CLAIM")

    def test_m_c_started_true(self) -> None:
        closure_dir = self._mutate(
            "device_domain_handoff.json",
            lambda payload: payload.__setitem__("m_c_started", True),
        )
        self._expect_fail(closure_dir, "FORBIDDEN_POSITIVE_CLAIM")

    def test_clinical_apnea_validated_true(self) -> None:
        closure_dir = self._mutate(
            "claim_boundary.json",
            lambda payload: payload.__setitem__("clinical_apnea_validated", True),
        )
        self._expect_fail(closure_dir, "FORBIDDEN_POSITIVE_CLAIM")

    def test_result_not_pristine_false(self) -> None:
        closure_dir = self._mutate(
            "final_evaluation_summary.json",
            lambda payload: payload.__setitem__("result_not_pristine", False),
        )
        self._expect_fail(closure_dir, "RESULT_NOT_PRISTINE_FALSE")

    def test_macro_f1_altered(self) -> None:
        closure_dir = self._mutate(
            "final_evaluation_summary.json",
            lambda payload: payload.__setitem__("macro_f1", 0.999999),
        )
        self._expect_fail(closure_dir, "MACRO_F1")

    def test_model_sha_altered(self) -> None:
        closure_dir = self._mutate(
            "locked_candidate_summary.json",
            lambda payload: payload.__setitem__("sha256", "0" * 64),
        )
        self._expect_fail(closure_dir, "CANDIDATE_SHA")

    def test_historical_total_altered(self) -> None:
        closure_dir = self._mutate(
            "final_evaluation_summary.json",
            lambda payload: payload.__setitem__("historical_total_payload_releases", 1),
        )
        self._expect_fail(closure_dir, "HIST_TOTAL")

    def test_recording_mismatch_field_altered(self) -> None:
        closure_dir = self._mutate(
            "final_evaluation_summary.json",
            lambda payload: payload.__setitem__("cross_model_recording_mismatches", 1),
        )
        self._expect_fail(closure_dir, "RECORDING")

    def test_seed43_reselection(self) -> None:
        def mutate(payload: dict) -> None:
            payload["candidate_id"] = "M-B3_CONV1D_GAP_BASELINE_seed43_M-B5_CAL_CLASS_BALANCED_120"
            payload["seed"] = 43

        closure_dir = self._mutate("locked_candidate_summary.json", mutate)
        self._expect_fail(closure_dir, "CANDIDATE_ID")

    def test_absolute_path(self) -> None:
        closure_dir = self._mutate(
            "source_and_population_summary.json",
            lambda payload: payload.__setitem__("raw_archive_repo_relative_path", "/tmp/db_records.zip"),
        )
        self._expect_fail(closure_dir, "UNSAFE_PATH")

    def test_missing_file(self) -> None:
        closure_dir = self._copy_closure()
        (closure_dir / "final_evaluation_summary.json").unlink()
        self._expect_fail(closure_dir, "CHECKSUM_TARGET_MISSING")

    def test_checksum_corruption(self) -> None:
        closure_dir = self._copy_closure()
        checksum = closure_dir / "checksums.sha256"
        lines = checksum.read_text(encoding="utf-8").splitlines()
        digest, name = lines[0].split()
        lines[0] = ("0" * 64) + "  " + name
        checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._expect_fail(closure_dir, "CHECKSUM_MISMATCH")

    def test_validator_does_not_generate_or_invoke(self) -> None:
        source = Path(validator.__file__).read_text(encoding="utf-8")
        self.assertNotIn("generate_m_b12_closure", source)
        self.assertNotIn("analyze_recovery_ledger", source)

    def _remove_registry_role(self, role: str) -> Path:
        path = self._copy_m11_registry()
        payload = _load(path)
        payload["artifacts"] = [item for item in payload["artifacts"] if item.get("artifact_role") != role]
        _dump(path, payload)
        return path

    def test_remove_b4_role(self) -> None:
        self._expect_fail_registry(self._remove_registry_role("b4_summary"), "MISSING_REQUIRED_ROLE")

    def test_remove_b7_role(self) -> None:
        self._expect_fail_registry(self._remove_registry_role("b7_summary"), "MISSING_REQUIRED_ROLE")

    def test_remove_b10b_incident_role(self) -> None:
        self._expect_fail_registry(self._remove_registry_role("b10b_incident_root_cause"), "MISSING_REQUIRED_ROLE")

    def test_remove_b10r1a_role(self) -> None:
        self._expect_fail_registry(self._remove_registry_role("b10r1a_summary"), "MISSING_REQUIRED_ROLE")

    def test_remove_b10r1b_role(self) -> None:
        self._expect_fail_registry(self._remove_registry_role("b10r1b_summary"), "MISSING_REQUIRED_ROLE")

    def test_remove_selected_model_role(self) -> None:
        self._expect_fail_registry(self._remove_registry_role("selected_tflite"), "MISSING_REQUIRED_ROLE")

    def test_remove_final_ledger_role(self) -> None:
        self._expect_fail_registry(self._remove_registry_role("b10r1b_ledger"), "MISSING_REQUIRED_ROLE")

    def test_remove_m_b11_lock_identity_role(self) -> None:
        def mutate(payload: dict) -> None:
            payload["roles"] = [item for item in payload["roles"] if item.get("artifact_role") != "m_b11_lock_identity"]

        closure_dir = self._mutate("phase_b_required_role_registry.json", mutate)
        self._expect_fail(closure_dir, "REQUIRED_ROLE_SET_DRIFT")

    def test_duplicate_required_role_different_path(self) -> None:
        path = self._copy_m11_registry()
        payload = _load(path)
        extra = json.loads(json.dumps(next(item for item in payload["artifacts"] if item["artifact_role"] == "b4_summary")))
        extra["repo_relative_path"] = "datasets/mmwave/manifests/M-B7_perturbation_robustness/m_b7_summary.json"
        payload["artifacts"].append(extra)
        _dump(path, payload)
        self._expect_fail_registry(path, "DUPLICATE_REQUIRED_ROLE")

    def test_required_role_path_changed(self) -> None:
        path = self._copy_m11_registry()
        payload = _load(path)
        for item in payload["artifacts"]:
            if item.get("artifact_role") == "selected_tflite":
                item["repo_relative_path"] = "models/mmwave/mmwave_resp_int8_v0.1.0.tflite"
        _dump(path, payload)
        self._expect_fail_registry(path, "ROLE_PATH_MISMATCH")

    def test_required_role_live_sha_mismatch(self) -> None:
        path = self._copy_m11_registry()
        payload = _load(path)
        for item in payload["artifacts"]:
            if item.get("artifact_role") == "b7_summary":
                item["sha256"] = "0" * 64
        _dump(path, payload)
        self._expect_fail_registry(path, "LIVE_SHA_MISMATCH")

    def test_training_strategy_changed(self) -> None:
        closure_dir = self._mutate(
            "locked_candidate_summary.json",
            lambda payload: payload.__setitem__("training_strategy_id", "M-B2_WEIGHTED"),
        )
        self._expect_fail(closure_dir, "TRAINING_STRATEGY")

    def test_preprocessing_profile_changed(self) -> None:
        closure_dir = self._mutate(
            "locked_candidate_summary.json",
            lambda payload: payload.__setitem__("preprocessing_profile_id", "M-B1_D1_B1_Z1"),
        )
        self._expect_fail(closure_dir, "PREPROCESSING_ID")

    def test_execution_preprocessing_contract_changed(self) -> None:
        closure_dir = self._mutate(
            "locked_candidate_summary.json",
            lambda payload: payload.__setitem__("execution_preprocessing_contract_id", "CHANGED_CONTRACT"),
        )
        self._expect_fail(closure_dir, "PREPROCESSING_CONTRACT")

    def test_calibration_profile_changed(self) -> None:
        closure_dir = self._mutate(
            "locked_candidate_summary.json",
            lambda payload: payload.__setitem__("calibration_profile", "M-B5_CAL_TRAIN_ORDER_120"),
        )
        self._expect_fail(closure_dir, "CALIBRATION_ID")

    def test_model_path_changed(self) -> None:
        closure_dir = self._mutate(
            "locked_candidate_summary.json",
            lambda payload: payload.__setitem__("repo_relative_path", "models/mmwave/mmwave_resp_int8_v0.1.0.tflite"),
        )
        self._expect_fail(closure_dir, "MODEL_PATH")

    def test_input_shape_changed(self) -> None:
        def mutate(payload: dict) -> None:
            payload["input_tensor"]["shape"] = [1, 200, 1]

        closure_dir = self._mutate("locked_candidate_summary.json", mutate)
        self._expect_fail(closure_dir, "INPUT_TENSOR")

    def test_input_quantization_changed(self) -> None:
        def mutate(payload: dict) -> None:
            payload["input_tensor"]["scale"] = 0.99

        closure_dir = self._mutate("locked_candidate_summary.json", mutate)
        self._expect_fail(closure_dir, "INPUT_TENSOR")

    def test_output_shape_changed(self) -> None:
        def mutate(payload: dict) -> None:
            payload["output_tensor"]["shape"] = [1, 2]

        closure_dir = self._mutate("locked_candidate_summary.json", mutate)
        self._expect_fail(closure_dir, "OUTPUT_TENSOR")

    def test_strict_int8_false(self) -> None:
        closure_dir = self._mutate(
            "locked_candidate_summary.json",
            lambda payload: payload.__setitem__("strict_int8", False),
        )
        self._expect_fail(closure_dir, "STRICT_INT8")

    def test_class_map_changed(self) -> None:
        def mutate(payload: dict) -> None:
            payload["class_map"] = {"0": "APNEA", "1": "NORMAL", "2": "RAPID_OR_ABNORMAL"}

        closure_dir = self._mutate("locked_candidate_summary.json", mutate)
        self._expect_fail(closure_dir, "CLASS_MAP")

    def test_normal_recall_changed(self) -> None:
        def mutate(payload: dict) -> None:
            payload["per_class"]["NORMAL"]["recall"] = 0.99

        closure_dir = self._mutate("final_evaluation_summary.json", mutate)
        self._expect_fail(closure_dir, "PER_CLASS_NORMAL")

    def test_rapid_recall_changed(self) -> None:
        def mutate(payload: dict) -> None:
            payload["per_class"]["RAPID_OR_ABNORMAL"]["recall"] = 0.99

        closure_dir = self._mutate("final_evaluation_summary.json", mutate)
        self._expect_fail(closure_dir, "PER_CLASS_RAPID_OR_ABNORMAL")

    def test_apnea_fpr_changed(self) -> None:
        def mutate(payload: dict) -> None:
            payload["per_class"]["APNEA"]["fpr"] = 0.0
            payload["apnea_proxy"]["fpr"] = 0.0

        closure_dir = self._mutate("final_evaluation_summary.json", mutate)
        self._expect_fail(closure_dir, "PER_CLASS_APNEA")

    def test_apnea_misses_changed(self) -> None:
        def mutate(payload: dict) -> None:
            payload["apnea_proxy"]["misses"] = 99

        closure_dir = self._mutate("final_evaluation_summary.json", mutate)
        self._expect_fail(closure_dir, "APNEA_PROXY_METRICS")

    def test_prediction_distribution_changed(self) -> None:
        def mutate(payload: dict) -> None:
            payload["prediction_distribution"]["APNEA"] = 0

        closure_dir = self._mutate("final_evaluation_summary.json", mutate)
        self._expect_fail(closure_dir, "PRED_DIST")

    def test_class_collapse_changed(self) -> None:
        def mutate(payload: dict) -> None:
            payload["class_collapse"]["collapsed"] = True

        closure_dir = self._mutate("final_evaluation_summary.json", mutate)
        self._expect_fail(closure_dir, "COLLAPSE")

    def test_subject_median_changed(self) -> None:
        closure_dir = self._mutate(
            "final_evaluation_summary.json",
            lambda payload: payload.__setitem__("median_subject_macro_f1", 0.999),
        )
        self._expect_fail(closure_dir, "SUBJECT_MEDIAN")

    def test_worst_subject_macro_f1_changed(self) -> None:
        closure_dir = self._mutate(
            "final_evaluation_summary.json",
            lambda payload: payload.__setitem__("worst_subject_macro_f1", 0.999),
        )
        self._expect_fail(closure_dir, "SUBJECT_WORST")

    def test_worst_subject_id_changed(self) -> None:
        closure_dir = self._mutate(
            "final_evaluation_summary.json",
            lambda payload: payload.__setitem__("worst_subject_id", "mutated-worst"),
        )
        self._expect_fail(closure_dir, "WORST_ID")

    def test_saturation_ratio_changed(self) -> None:
        closure_dir = self._mutate(
            "final_evaluation_summary.json",
            lambda payload: payload.__setitem__("input_saturation_ratio", 0.5),
        )
        self._expect_fail(closure_dir, "SAT_RATIO")

    def test_valid_count_changed(self) -> None:
        closure_dir = self._mutate(
            "final_evaluation_summary.json",
            lambda payload: payload.__setitem__("valid", 74),
        )
        self._expect_fail(closure_dir, "VALID")

    def _mutate_report_only(self, old: str, new: str) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="m_b12_report_")) / "report.md"
        text = REPORT.read_text(encoding="utf-8")
        if old not in text:
            raise AssertionError(f"report marker missing: {old}")
        tmp.write_text(text.replace(old, new, 1), encoding="utf-8")
        return tmp

    def test_markdown_only_change_fails(self) -> None:
        report_tmp = self._mutate_report_only("final_macro_f1=0.494836", "final_macro_f1=0.999999")
        with self.assertRaises(validator.MB12ValidationError) as ctx:
            validator.validate_m_b12(ROOT, skip_m_b11=True, report_path=report_tmp)
        self.assertIn("REPORT_SHA", str(ctx.exception))

    def _mutate_report_and_identity(self, old: str, new: str) -> Path:
        closure_dir = self._copy_closure()
        report_tmp = Path(tempfile.mkdtemp(prefix="m_b12_report_")) / "report.md"
        text = REPORT.read_text(encoding="utf-8").replace(old, new, 1)
        report_tmp.write_text(text, encoding="utf-8")
        identity = _load(closure_dir / "final_report_identity.json")
        identity["sha256"] = _sha(report_tmp)
        identity["bytes"] = int(report_tmp.stat().st_size)
        _dump(closure_dir / "final_report_identity.json", identity)
        write_checksums(closure_dir)
        with self.assertRaises(validator.MB12ValidationError) as ctx:
            validator.validate_m_b12(ROOT, closure_dir=closure_dir, skip_m_b11=True, report_path=report_tmp)
        return ctx.exception

    def test_report_macro_f1_altered(self) -> None:
        exc = self._mutate_report_and_identity("final_macro_f1=0.494836", "final_macro_f1=0.999999")
        self.assertIn("REPORT_FACT_MISMATCH", str(exc))

    def test_report_normal_recall_altered(self) -> None:
        exc = self._mutate_report_and_identity("normal_recall=0.2", "normal_recall=0.99")
        self.assertIn("REPORT_FACT_MISMATCH", str(exc))

    def test_report_apnea_fpr_altered(self) -> None:
        exc = self._mutate_report_and_identity("apnea_fpr=0.522727", "apnea_fpr=0.0")
        self.assertIn("REPORT_FACT_MISMATCH", str(exc))

    def test_report_selected_sha_altered(self) -> None:
        exc = self._mutate_report_and_identity(
            f"selected_model_sha={generator.EXPECTED_MODEL_SHA}",
            f"selected_model_sha={'0' * 64}",
        )
        self.assertIn("REPORT_FACT_MISMATCH", str(exc))

    def test_report_historical_total_altered(self) -> None:
        exc = self._mutate_report_and_identity("historical_total_release=2", "historical_total_release=1")
        self.assertIn("REPORT_FACT_MISMATCH", str(exc))

    def test_report_result_designation_pristine(self) -> None:
        exc = self._mutate_report_and_identity(
            "result_designation=REUSED_LOCKED_TEST_AFTER_PREINFERENCE_STRUCTURAL_ABORT",
            "result_designation=PRISTINE_LOCKED_TEST",
        )
        self.assertIn("REPORT_FACT_MISMATCH", str(exc))

    def test_report_mr60_validated_true(self) -> None:
        exc = self._mutate_report_and_identity("mr60_validated=false", "mr60_validated=true")
        self.assertIn("REPORT_FACT_MISMATCH", str(exc))

    def test_report_deployment_ready_true(self) -> None:
        exc = self._mutate_report_and_identity("deployment_ready=false", "deployment_ready=true")
        self.assertIn("REPORT_FACT_MISMATCH", str(exc))


if __name__ == "__main__":
    unittest.main()
