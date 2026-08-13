"""Focused fail-closed tests for M-B10R1-A recovery harness pre-freeze.

No real recovery LOCKED_TEST access. Mutations use temporary copies / mocks.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import validate_mmwave_m_b10r1a as validator
from scripts.mmwave_m_b10r1_metrics import (
    MB10R1MetricsError,
    metric_bundle,
    saturation_audit_from_rows,
    subject_metrics,
    quantize_with_saturation,
)
from scripts.mmwave_m_b10r1_recovery_access import (
    EXPECTED_ELIGIBLE,
    EXPECTED_INFERENCES,
    ORIGINAL_FINAL_TOKEN,
    RECOVERY_AUTHORIZATION_TOKEN,
    LimitedReuseRecoveryAccessController,
    RecoveryAccessError,
    RecoveryReadiness,
)
from scripts.mmwave_m_b10r1_recovery_eval import (
    SELECTED_MODEL_ID,
    SELECTED_PREPROCESSING_CONTRACT_ID,
    V01_PREPROCESSING_CONTRACT_ID,
    V02_PREPROCESSING_CONTRACT_ID,
    MB10R1EvalError,
    authorize_pre_access_freeze_binding,
    build_bound_contract_identity,
    evaluate_recovery_payload,
    execute_authorized_recovery,
    frozen_model_specs,
    load_frozen_execution_identity,
    readiness_summary,
    require_frozen_bound_contract,
    validate_frozen_recovery_models,
    verify_live_against_frozen,
)
from scripts.mmwave_m_b10r1_result_writer import (
    B_AUTHORIZATION_STATUS_GRANTED,
    persist_recovery_results,
    persist_terminal_failure,
    not_authorized_overlay_template,
)
from scripts import validate_mmwave_m_b10r1b as b_validator
from scripts.mmwave_m_b10r1a_prefreeze import generate_m_b10r1a_prefreeze
from scripts import run_mmwave_m_b10r1 as runner_cli

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / validator.OUT_DIR_REL


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_checksums(path: Path) -> None:
    lines = []
    for item in sorted(path.iterdir(), key=lambda value: value.name):
        if not item.is_file() or item.name == "checksums.sha256":
            continue
        lines.append(f"{_sha256_file(item)}  {item.name}")
    (path / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_output() -> tuple[tempfile.TemporaryDirectory[str], Path]:
    holder = tempfile.TemporaryDirectory()
    destination = Path(holder.name) / "evidence"
    shutil.copytree(OUT, destination)
    return holder, destination


def _mutate_json(path: Path, filename: str, **updates: object) -> None:
    target = path / filename
    data = json.loads(target.read_text(encoding="utf-8"))
    data.update(updates)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_checksums(path)


def _nested_set(data: dict, dotted: str, value: object) -> None:
    parts = dotted.split(".")
    cur = data
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


def _mutate_nested(path: Path, filename: str, dotted: str, value: object) -> None:
    target = path / filename
    data = json.loads(target.read_text(encoding="utf-8"))
    _nested_set(data, dotted, value)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_checksums(path)


def _bound_for_tests(root: Path = ROOT) -> dict:
    return build_bound_contract_identity(root)


def _authorized_readiness() -> RecoveryReadiness:
    return RecoveryReadiness(
        recovery_execution_authorized=True,
        recovery_payload_release_authorized=True,
        independent_review_required=True,
        mechanism_implemented=True,
        runner_implemented=True,
        pre_access_validator_pass=True,
        M_B10R1B_started=True,
    )


class MetricEngineTests(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        labels = [0, 1, 2, 0, 1, 2]
        preds = [0, 1, 2, 0, 1, 2]
        bundle = metric_bundle(labels, preds)
        self.assertEqual(bundle["accuracy"], 1.0)
        self.assertEqual(bundle["macro_f1"], 1.0)
        self.assertEqual(bundle["apnea_proxy"]["misses"], 0)
        self.assertFalse(bundle["class_collapse"]["collapsed"])

    def test_support_zero_semantics(self) -> None:
        labels = [0, 0, 1, 1]
        preds = [0, 0, 1, 1]
        bundle = metric_bundle(labels, preds)
        self.assertEqual(bundle["per_class"]["APNEA"]["support"], 0)
        self.assertEqual(bundle["per_class"]["APNEA"]["precision"], 0.0)
        self.assertEqual(bundle["per_class"]["APNEA"]["recall"], 0.0)
        self.assertEqual(bundle["per_class"]["APNEA"]["f1_score"], 0.0)
        self.assertIn("APNEA", bundle["class_collapse"]["zero_prediction_classes"])

    def test_confusion_and_apnea_misses(self) -> None:
        labels = [2, 2, 2, 0]
        preds = [2, 0, 1, 0]
        bundle = metric_bundle(labels, preds)
        self.assertEqual(bundle["apnea_proxy"]["misses"], 2)
        self.assertEqual(bundle["confusion_matrix"][2][2], 1)

    def test_subject_metrics_worst_and_median(self) -> None:
        records = [
            {"subject_id": "A", "true_class_index": 0, "predicted_class_index": 0},
            {"subject_id": "A", "true_class_index": 1, "predicted_class_index": 1},
            {"subject_id": "B", "true_class_index": 0, "predicted_class_index": 2},
            {"subject_id": "B", "true_class_index": 1, "predicted_class_index": 2},
        ]
        result = subject_metrics(records)
        self.assertEqual(result["subject_count"], 2)
        self.assertEqual(result["worst_subject_id"], "B")
        self.assertGreater(result["median_subject_macro_f1"], result["worst_subject_macro_f1"] - 1e-9)

    def test_saturation_audit_and_quantize(self) -> None:
        ready = np.full((1, 300, 1), 100.0, dtype=np.float32)
        q = quantize_with_saturation(ready, scale=0.01, zero_point=0, contract_id="TEST")
        self.assertGreater(q["input_saturation_count"], 0)
        audit = saturation_audit_from_rows(
            [
                {
                    "window_id": "w1",
                    "input_saturation_count": q["input_saturation_count"],
                    "input_saturation_ratio": q["input_saturation_ratio"],
                }
            ]
        )
        self.assertEqual(audit["total_quantized_elements"], 300)
        self.assertEqual(audit["samples_with_any_saturation"], 1)

    def test_metric_bundle_refuses_empty_with_positive_count(self) -> None:
        with self.assertRaises(MB10R1MetricsError) as ctx:
            metric_bundle([], [], evaluated_sample_count=75)
        self.assertIn("METRIC_EMPTY_LABELS_WITH_POSITIVE_EVALUATED_COUNT", str(ctx.exception))

    def test_metric_bundle_refuses_count_mismatch(self) -> None:
        with self.assertRaises(MB10R1MetricsError) as ctx:
            metric_bundle([0, 1], [0, 1], evaluated_sample_count=75)
        self.assertIn("METRIC_EVALUATED_SAMPLE_COUNT_MISMATCH", str(ctx.exception))


class RecoveryAccessNegativeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.state_path = Path(self.tmpdir.name) / "state.json"
        self.controller = LimitedReuseRecoveryAccessController(ROOT, audit_state_path=self.state_path)
        self.bound = _bound_for_tests()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_no_auth_refused(self) -> None:
        with self.assertRaises(RecoveryAccessError):
            self.controller.get_locked_test_recovery_evaluation_dataset(
                None, self.bound, _authorized_readiness()
            )

    def test_wrong_original_final_token_refused(self) -> None:
        with self.assertRaises(RecoveryAccessError) as ctx:
            self.controller.get_locked_test_recovery_evaluation_dataset(
                ORIGINAL_FINAL_TOKEN, self.bound, _authorized_readiness()
            )
        self.assertIn("ORIGINAL_FINAL_TOKEN_REJECTED", str(ctx.exception))

    def test_malformed_auth_refused(self) -> None:
        with self.assertRaises(RecoveryAccessError):
            self.controller.get_locked_test_recovery_evaluation_dataset(
                "NOT_A_VALID_TOKEN", self.bound, _authorized_readiness()
            )

    def test_readiness_false_refused(self) -> None:
        readiness = RecoveryReadiness(
            recovery_execution_authorized=False,
            recovery_payload_release_authorized=False,
        )
        with self.assertRaises(RecoveryAccessError):
            self.controller.get_locked_test_recovery_evaluation_dataset(
                RECOVERY_AUTHORIZATION_TOKEN, self.bound, readiness
            )

    def test_contract_sha_mismatch_refused(self) -> None:
        bad = copy.deepcopy(self.bound)
        bad["selected_model_sha256"] = "0" * 64
        with self.assertRaises(RecoveryAccessError) as ctx:
            self.controller.get_locked_test_recovery_evaluation_dataset(
                RECOVERY_AUTHORIZATION_TOKEN, bad, _authorized_readiness()
            )
        self.assertIn("BOUND_CONTRACT_SHA_MISMATCH", str(ctx.exception))

    def test_include_ambiguous_true_refused(self) -> None:
        bad = copy.deepcopy(self.bound)
        bad["include_ambiguous"] = True
        with self.assertRaises(RecoveryAccessError):
            self.controller.get_locked_test_recovery_evaluation_dataset(
                RECOVERY_AUTHORIZATION_TOKEN, bad, _authorized_readiness()
            )

    def test_second_recovery_after_consumed(self) -> None:
        # Simulate consumed without real load.
        self.controller._state["payload_consumed"] = True
        self.controller._state["recovery_payload_release_events"] = 1
        self.controller._persist()
        with self.assertRaises(RecoveryAccessError):
            self.controller.get_locked_test_recovery_evaluation_dataset(
                RECOVERY_AUTHORIZATION_TOKEN, self.bound, _authorized_readiness()
            )

    def test_retry_after_release_keeps_consumed(self) -> None:
        fake_payload = {
            "total_count": EXPECTED_ELIGIBLE,
            "windows": [
                {"assignment_status": "PURE", "split": "LOCKED_TEST", "subject_id": f"s{i % 16}"}
                for i in range(EXPECTED_ELIGIBLE)
            ],
            "signals": [None] * EXPECTED_ELIGIBLE,
        }

        def _fake_load(**_kwargs):
            return fake_payload

        with mock.patch.object(self.controller, "_load_eligible_locked_test", side_effect=_fake_load):
            payload = self.controller.get_locked_test_recovery_evaluation_dataset(
                RECOVERY_AUTHORIZATION_TOKEN, self.bound, _authorized_readiness()
            )
            self.assertEqual(payload["total_count"], EXPECTED_ELIGIBLE)
        snap = self.controller.snapshot()
        self.assertTrue(snap["payload_consumed"])
        self.assertEqual(snap["recovery_payload_release_events"], 1)
        self.assertEqual(snap["historical_total_payload_release_events"], 2)
        # Second access refused
        with self.assertRaises(RecoveryAccessError):
            self.controller.get_locked_test_recovery_evaluation_dataset(
                RECOVERY_AUTHORIZATION_TOKEN, self.bound, _authorized_readiness()
            )
        # Historical original never reset
        self.assertEqual(self.controller.snapshot()["original_final_accessor_invocations"], 1)

    def test_loader_raises_before_return_no_release(self) -> None:
        """Audit policy B: load throws before return → release=0, historical_total=1, consumed=true."""

        def _boom(**_kwargs):
            raise RuntimeError("simulated load failure before return")

        with mock.patch.object(self.controller, "_load_eligible_locked_test", side_effect=_boom):
            with self.assertRaises(RuntimeError):
                self.controller.get_locked_test_recovery_evaluation_dataset(
                    RECOVERY_AUTHORIZATION_TOKEN, self.bound, _authorized_readiness()
                )
        snap = self.controller.snapshot()
        self.assertTrue(snap["payload_consumed"])
        self.assertEqual(snap["recovery_accessor_invocations"], 1)
        self.assertEqual(snap["recovery_payload_release_events"], 0)
        self.assertEqual(snap["historical_total_payload_release_events"], 1)
        with self.assertRaises(RecoveryAccessError):
            self.controller.get_locked_test_recovery_evaluation_dataset(
                RECOVERY_AUTHORIZATION_TOKEN, self.bound, _authorized_readiness()
            )

    def test_verify_fails_after_payload_return_keeps_release(self) -> None:
        """Audit policy A: loader returns, verify fails → release=1, historical_total=2, consumed."""
        bad_payload = {
            "total_count": 10,  # wrong — verify will fail
            "windows": [{"assignment_status": "PURE", "split": "LOCKED_TEST", "subject_id": "s0"}]
            * 10,
            "signals": [None] * 10,
        }

        def _fake_load(**_kwargs):
            return bad_payload

        with mock.patch.object(self.controller, "_load_eligible_locked_test", side_effect=_fake_load):
            with self.assertRaises(RecoveryAccessError):
                self.controller.get_locked_test_recovery_evaluation_dataset(
                    RECOVERY_AUTHORIZATION_TOKEN, self.bound, _authorized_readiness()
                )
        snap = self.controller.snapshot()
        self.assertTrue(snap["payload_consumed"])
        self.assertEqual(snap["recovery_payload_release_events"], 1)
        self.assertEqual(snap["historical_total_payload_release_events"], 2)
        self.assertEqual(snap["recovery_accessor_invocations"], 1)
        with self.assertRaises(RecoveryAccessError) as ctx:
            self.controller.get_locked_test_recovery_evaluation_dataset(
                RECOVERY_AUTHORIZATION_TOKEN, self.bound, _authorized_readiness()
            )
        self.assertTrue(
            "SECOND_RECOVERY" in str(ctx.exception) or "ALREADY_CONSUMED" in str(ctx.exception)
        )

    def test_post_release_failure_still_consumed_no_retry(self) -> None:
        def _boom(**_kwargs):
            raise RuntimeError("simulated post-mark failure")

        with mock.patch.object(self.controller, "_load_eligible_locked_test", side_effect=_boom):
            with self.assertRaises(RuntimeError):
                self.controller.get_locked_test_recovery_evaluation_dataset(
                    RECOVERY_AUTHORIZATION_TOKEN, self.bound, _authorized_readiness()
                )
        self.assertTrue(self.controller.snapshot()["payload_consumed"])
        self.assertEqual(self.controller.snapshot()["recovery_payload_release_events"], 0)
        with self.assertRaises(RecoveryAccessError):
            self.controller.get_locked_test_recovery_evaluation_dataset(
                RECOVERY_AUTHORIZATION_TOKEN, self.bound, _authorized_readiness()
            )


class PrefreezeValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Ensure evidence exists for validator tests.
        if not OUT.is_dir():
            generate_m_b10r1a_prefreeze(ROOT)

    def test_validator_passes(self) -> None:
        # Never stamp/mutate the live prefreeze tree from unit tests.
        result = validator.validate_m_b10r1a_artifacts(
            ROOT, skip_upstream=True, mark_validator_pass=False
        )
        self.assertEqual(result["validation_status"], "PASS")
        self.assertFalse(result["recovery_execution_authorized"])

    def test_historical_counter_reset_fails(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_json(
                destination,
                "recovery_access_audit.json",
                **{"historical_original_final_accessor_invocations": 0},
            )
            with self.assertRaises(validator.MB10R1AValidationError):
                validator.validate_m_b10r1a_artifacts(
                    ROOT, output_dir=destination, skip_upstream=True, mark_validator_pass=False
                )
        finally:
            holder.cleanup()

    def test_original_consumed_false_fails(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_json(destination, "recovery_access_audit.json", original_locked_test_consumed=False)
            with self.assertRaises(validator.MB10R1AValidationError):
                validator.validate_m_b10r1a_artifacts(
                    ROOT, output_dir=destination, skip_upstream=True, mark_validator_pass=False
                )
        finally:
            holder.cleanup()

    def test_wrong_population_counts_fail(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_json(destination, "recovery_population_contract.json", supervised_eligible_windows=88)
            with self.assertRaises(validator.MB10R1AValidationError):
                validator.validate_m_b10r1a_artifacts(
                    ROOT, output_dir=destination, skip_upstream=True, mark_validator_pass=False
                )
        finally:
            holder.cleanup()

    def test_include_ambiguous_true_fails(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_json(destination, "recovery_population_contract.json", include_ambiguous=True)
            with self.assertRaises(validator.MB10R1AValidationError):
                validator.validate_m_b10r1a_artifacts(
                    ROOT, output_dir=destination, skip_upstream=True, mark_validator_pass=False
                )
        finally:
            holder.cleanup()

    def test_seed43_model_fails(self) -> None:
        holder, destination = _copy_output()
        try:
            data = json.loads((destination / "model_identity_registry.json").read_text(encoding="utf-8"))
            data["models"][0]["model_id"] = "M-B3_CONV1D_GAP_BASELINE_seed43_M-B6_STRICT_INT8"
            (destination / "model_identity_registry.json").write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _rewrite_checksums(destination)
            with self.assertRaises(validator.MB10R1AValidationError):
                validator.validate_m_b10r1a_artifacts(
                    ROOT, output_dir=destination, skip_upstream=True, mark_validator_pass=False
                )
        finally:
            holder.cleanup()

    def test_seed44_and_fourth_model_fail(self) -> None:
        holder, destination = _copy_output()
        try:
            data = json.loads((destination / "model_identity_registry.json").read_text(encoding="utf-8"))
            data["models"].append({"model_id": "M-B3_CONV1D_GAP_BASELINE_seed44_M-B6_STRICT_INT8"})
            (destination / "model_identity_registry.json").write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _rewrite_checksums(destination)
            with self.assertRaises(validator.MB10R1AValidationError):
                validator.validate_m_b10r1a_artifacts(
                    ROOT, output_dir=destination, skip_upstream=True, mark_validator_pass=False
                )
        finally:
            holder.cleanup()

    def test_baseline_executor_sha_change_fails(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_nested(
                destination,
                "baseline_identity_registry.json",
                "executor_sha256",
                "0" * 64,
            )
            with self.assertRaises(validator.MB10R1AValidationError):
                validator.validate_m_b10r1a_artifacts(
                    ROOT, output_dir=destination, skip_upstream=True, mark_validator_pass=False
                )
        finally:
            holder.cleanup()

    def test_baseline_metadata_sha_change_fails(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_nested(
                destination,
                "baseline_identity_registry.json",
                "v0_1.metadata_sha256",
                "0" * 64,
            )
            with self.assertRaises(validator.MB10R1AValidationError):
                validator.validate_m_b10r1a_artifacts(
                    ROOT, output_dir=destination, skip_upstream=True, mark_validator_pass=False
                )
        finally:
            holder.cleanup()

    def test_metric_schema_corruption_fails(self) -> None:
        holder, destination = _copy_output()
        try:
            data = json.loads((destination / "metric_contract.json").read_text(encoding="utf-8"))
            data["metrics_schema"]["primary"] = "accuracy"
            (destination / "metric_contract.json").write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _rewrite_checksums(destination)
            with self.assertRaises(validator.MB10R1AValidationError):
                validator.validate_m_b10r1a_artifacts(
                    ROOT, output_dir=destination, skip_upstream=True, mark_validator_pass=False
                )
        finally:
            holder.cleanup()

    def test_authorization_true_fails(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_json(destination, "recovery_access_readiness.json", recovery_execution_authorized=True)
            with self.assertRaises(validator.MB10R1AValidationError):
                validator.validate_m_b10r1a_artifacts(
                    ROOT, output_dir=destination, skip_upstream=True, mark_validator_pass=False
                )
        finally:
            holder.cleanup()


class CliAndMonkeypatchTests(unittest.TestCase):
    def test_default_cli_no_access(self) -> None:
        called = {"recovery": False}

        def _boom(*_a, **_k):
            called["recovery"] = True
            raise AssertionError("recovery must not be called")

        with mock.patch(
            "scripts.mmwave_m_b10r1_recovery_access.LimitedReuseRecoveryAccessController.get_locked_test_recovery_evaluation_dataset",
            side_effect=_boom,
        ):
            rc = runner_cli.main([])
        self.assertEqual(rc, 0)
        self.assertFalse(called["recovery"])
        summary = readiness_summary(ROOT)
        self.assertFalse(summary["recovery_accessor_invoked"])

    def test_execute_flag_without_token_refuses(self) -> None:
        rc = runner_cli.main(["--execute-authorized-limited-reuse-recovery"])
        self.assertEqual(rc, 2)

    def test_execute_with_token_refused_by_b_overlay(self) -> None:
        called = {"recovery": False}

        def _boom(*_a, **_k):
            called["recovery"] = True
            raise AssertionError("recovery must not be called")

        a_runtime = json.loads(
            (OUT / "recovery_access_runtime_state.json").read_text(encoding="utf-8")
        )
        with mock.patch(
            "scripts.mmwave_m_b10r1_recovery_access.LimitedReuseRecoveryAccessController.get_locked_test_recovery_evaluation_dataset",
            side_effect=_boom,
        ):
            rc = runner_cli.main(
                [
                    "--execute-authorized-limited-reuse-recovery",
                    "--authorization-token",
                    RECOVERY_AUTHORIZATION_TOKEN,
                ]
            )
        self.assertEqual(rc, 2)
        self.assertFalse(called["recovery"])
        after = json.loads((OUT / "recovery_access_runtime_state.json").read_text(encoding="utf-8"))
        self.assertEqual(after, a_runtime)
        self.assertEqual(after["recovery_accessor_invocations"], 0)
        self.assertEqual(after["recovery_payload_release_events"], 0)
        with self.assertRaises(MB10R1EvalError) as ctx:
            execute_authorized_recovery(ROOT, RECOVERY_AUTHORIZATION_TOKEN)
        self.assertIn("B_AUTHORIZATION", str(ctx.exception))

    def test_monkeypatch_forbids_real_recovery_during_generator_validator(self) -> None:
        def _forbidden(*_a, **_k):
            raise RuntimeError("FORBIDDEN_M_B10R1A_REAL_RECOVERY_ACCESS")

        holder = tempfile.TemporaryDirectory()
        try:
            with mock.patch(
                "scripts.mmwave_m_b10r1_recovery_access.LimitedReuseRecoveryAccessController.get_locked_test_recovery_evaluation_dataset",
                side_effect=_forbidden,
            ):
                # Pre-access CLI must not call recovery get_*
                self.assertEqual(runner_cli.main(["--pre-access"]), 0)
                # Validator against live committed evidence must not call recovery get_*
                outcome = validator.validate_m_b10r1a_artifacts(
                    ROOT, skip_upstream=True, mark_validator_pass=False
                )
                self.assertEqual(outcome["validation_status"], "PASS")
                # Validate an isolated evidence copy (never mutate committed tree)
                dest = Path(holder.name) / "evidence"
                shutil.copytree(OUT, dest)
                outcome2 = validator.validate_m_b10r1a_artifacts(
                    ROOT, output_dir=dest, skip_upstream=True, mark_validator_pass=False
                )
                self.assertEqual(outcome2["validation_status"], "PASS")
        finally:
            holder.cleanup()

    def test_recovery_module_never_calls_final_accessor(self) -> None:
        source = (ROOT / "scripts/mmwave_m_b10r1_recovery_access.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                self.assertNotEqual(name, "get_locked_test_final_evaluation_dataset")

    def test_validator_source_never_calls_recovery_get(self) -> None:
        source = (ROOT / "scripts/validate_mmwave_m_b10r1a.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                self.assertNotEqual(name, "get_locked_test_recovery_evaluation_dataset")
                self.assertNotEqual(name, "get_locked_test_final_evaluation_dataset")


class FakeRunner:
    """Counts invoke calls; returns a finite 3-class dequantized vector."""

    def __init__(self) -> None:
        self.invocations = 0
        self.invoke_attempts = 0

    def invoke(self, input_int8: np.ndarray) -> dict:
        self.invoke_attempts += 1
        array = np.asarray(input_int8, dtype=np.int8).reshape(1, 300, 1)
        del array  # shape validated; FakeRunner does not run TFLite
        self.invocations += 1
        probs = [0.7, 0.2, 0.1]
        return {
            "raw_output_int8": [10, 0, -10],
            "dequantized_output": probs,
            "predicted_class_index": 0,
            "predicted_class": "NORMAL",
            "confidence": 0.7,
        }


def _mock_recovery_payload() -> dict:
    windows = []
    for i in range(EXPECTED_ELIGIBLE):
        label_id = i % 3
        windows.append(
            {
                "window_id": f"w{i:03d}",
                "subject_id": f"s{i % 16:02d}",
                "recording_id": f"r{i:03d}",
                "safenest_label": ["NORMAL", "RAPID_OR_ABNORMAL", "APNEA"][label_id],
                "safenest_label_id": label_id,
                "assignment_status": "PURE",
                "split": "LOCKED_TEST",
            }
        )
    signals = np.random.RandomState(0).randn(EXPECTED_ELIGIBLE, 300).astype(np.float64)
    return {
        "total_count": EXPECTED_ELIGIBLE,
        "windows": windows,
        "signals": signals,
    }


class EvaluateRecoveryPayloadTests(unittest.TestCase):
    def test_full_mock_75x3_orchestration(self) -> None:
        specs = validate_frozen_recovery_models(ROOT)
        payload = _mock_recovery_payload()
        runners = {spec["model_id"]: FakeRunner() for spec in specs}

        # Must NOT call real recovery accessor.
        with mock.patch(
            "scripts.mmwave_m_b10r1_recovery_access.LimitedReuseRecoveryAccessController.get_locked_test_recovery_evaluation_dataset",
            side_effect=AssertionError("MUST_NOT_CALL_REAL_RECOVERY_ACCESSOR"),
        ):
            result = evaluate_recovery_payload(ROOT, payload, specs, runners=runners)

        self.assertEqual(result["status"], "RECOVERY_EXECUTED")
        self.assertEqual(result["ledger_row_count"], EXPECTED_INFERENCES)
        self.assertEqual(result["actual_total_tflite_invocations"], EXPECTED_INFERENCES)
        self.assertEqual(len(result["ledger"]), 225)
        for spec in specs:
            mid = spec["model_id"]
            cov = result["coverage_by_model"][mid]
            self.assertEqual(cov["evaluation_rows_attempted"], 75)
            self.assertEqual(cov["tflite_invoke_count"], 75)
            self.assertEqual(cov["valid_count"], 75)
            self.assertEqual(runners[mid].invocations, 75)
            self.assertEqual(
                result["metrics_by_model"][mid]["evaluated_sample_count"], 75
            )
            self.assertEqual(spec["preprocessing_contract_id"], {
                SELECTED_MODEL_ID: SELECTED_PREPROCESSING_CONTRACT_ID,
                "mmwave_resp_int8": V01_PREPROCESSING_CONTRACT_ID,
                "mmwave_resp_int8_v0.2.0_candidate": V02_PREPROCESSING_CONTRACT_ID,
            }[mid])
        for row in result["ledger"]:
            self.assertIn(
                row["preprocessing_contract_id"],
                {
                    SELECTED_PREPROCESSING_CONTRACT_ID,
                    V01_PREPROCESSING_CONTRACT_ID,
                    V02_PREPROCESSING_CONTRACT_ID,
                },
            )
            self.assertFalse(row["invalid"])
            self.assertEqual(len(row["dequantized_output"]), 3)
            self.assertTrue(all(np.isfinite(row["dequantized_output"])))
        serialized = json.dumps(result).lower()
        self.assertNotIn("seed43", serialized)
        self.assertNotIn("seed44", serialized)

    def test_window_vs_signal_regression(self) -> None:
        """Behavioral regression: preprocess first arg must be ndarray, not window dict.

        Would fail head 008808d which called preprocess_for_spec(window, ...).
        """
        specs = validate_frozen_recovery_models(ROOT)
        payload = _mock_recovery_payload()
        runners = {spec["model_id"]: FakeRunner() for spec in specs}
        seen: list[type] = []

        from scripts import mmwave_m_b10b_final_eval as mb10b

        real_preprocess = mb10b.preprocess_for_spec

        def _asserting_preprocess(first, spec):
            seen.append(type(first))
            if isinstance(first, dict):
                raise AssertionError("PREPROCESS_RECEIVED_WINDOW_DICT")
            if not isinstance(first, np.ndarray):
                raise AssertionError(f"PREPROCESS_EXPECTED_NDARRAY:{type(first)}")
            return real_preprocess(first, spec)

        with mock.patch.object(mb10b, "preprocess_for_spec", side_effect=_asserting_preprocess):
            result = evaluate_recovery_payload(ROOT, payload, specs, runners=runners)

        self.assertEqual(result["status"], "RECOVERY_EXECUTED")
        self.assertEqual(len(seen), EXPECTED_INFERENCES)
        self.assertTrue(all(issubclass(t, np.ndarray) for t in seen))

        source = (ROOT / "scripts/mmwave_m_b10r1_recovery_eval.py").read_text(encoding="utf-8")
        self.assertIn("preprocess_for_spec(signal", source)
        self.assertNotIn("preprocess_for_spec(window", source)


class FrozenBindingCorruptionTests(unittest.TestCase):
    def test_authorize_rejects_mutated_policy_before_payload(self) -> None:
        holder = tempfile.TemporaryDirectory()
        try:
            dest = Path(holder.name) / "tree"
            # Minimal tree: copy freeze evidence + mutate a bound policy file via overlay.
            # Use live root for models but temp copy of freeze identity + policy.
            # Simpler: mutate live-relative check via temp root symlink is hard on Darwin.
            # Instead call verify_live_against_frozen with a mutated frozen dict expected SHA.
            frozen = load_frozen_execution_identity(ROOT)
            bad = copy.deepcopy(frozen)
            bad["artifact_sha256"] = dict(bad.get("artifact_sha256") or {})
            bad["artifact_sha256"]["policy_decision_sha256"] = "0" * 64
            bad["harness_module_sha256"] = dict(bad.get("harness_module_sha256") or {})
            with self.assertRaises(MB10R1EvalError) as ctx:
                verify_live_against_frozen(ROOT, bad)
            self.assertIn("FROZEN_LIVE_MISMATCH", str(ctx.exception))
        finally:
            holder.cleanup()

    def test_harness_module_corruption_rejected(self) -> None:
        frozen = load_frozen_execution_identity(ROOT)
        for rel in (
            "scripts/mmwave_m_b10r1_recovery_access.py",
            "scripts/mmwave_m_b10r1_recovery_eval.py",
            "scripts/mmwave_m_b10r1_metrics.py",
            "scripts/run_mmwave_m_b10r1.py",
            "scripts/mmwave_m_b10r1_result_writer.py",
            "scripts/validate_mmwave_m_b10r1b.py",
        ):
            bad = copy.deepcopy(frozen)
            bad["harness_module_sha256"] = dict(bad["harness_module_sha256"])
            bad["harness_module_sha256"][rel] = "0" * 64
            with self.assertRaises(MB10R1EvalError) as ctx:
                verify_live_against_frozen(ROOT, bad)
            self.assertIn("FROZEN_HARNESS_LIVE_MISMATCH", str(ctx.exception))

    def test_proposed_and_metric_contract_corruption_rejected(self) -> None:
        frozen = load_frozen_execution_identity(ROOT)
        for key in (
            "proposed_recovery_evaluation_contract_sha256",
            "m_b10a_metric_contract_sha256",
        ):
            bad = copy.deepcopy(frozen)
            bad["artifact_sha256"] = dict(bad["artifact_sha256"])
            bad["artifact_sha256"][key] = "0" * 64
            # Also poison top-level mirrors used by secondary checks
            if key == "m_b10a_metric_contract_sha256":
                bad["m_b10a_metric_contract_sha256"] = "0" * 64
            if key == "proposed_recovery_evaluation_contract_sha256":
                bad["m_b10r0_proposed_contract_sha256"] = "0" * 64
            with self.assertRaises(MB10R1EvalError):
                verify_live_against_frozen(ROOT, bad)

    def test_preaccess_auth_rejects_before_mock_payload(self) -> None:
        """Mutated freeze identity file in temp evidence → authorize fails before load."""
        holder, destination = _copy_output()
        try:
            freeze_path = destination / "execution_freeze_identity.json"
            if not freeze_path.is_file():
                self.skipTest("execution_freeze_identity.json not yet generated")
            data = json.loads(freeze_path.read_text(encoding="utf-8"))
            data["harness_module_sha256"]["scripts/mmwave_m_b10r1_recovery_eval.py"] = "0" * 64
            freeze_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _rewrite_checksums(destination)

            # Point OUT_DIR via temp root is awkward; call verify directly on mutated frozen.
            with self.assertRaises(MB10R1EvalError):
                verify_live_against_frozen(ROOT, data)

            # authorize_pre_access_freeze_binding on live tree still passes (live evidence intact)
            authorize_pre_access_freeze_binding(ROOT)
        finally:
            holder.cleanup()

    def test_exact_preprocessing_contract_ids_frozen(self) -> None:
        specs = frozen_model_specs()
        by_id = {s["model_id"]: s["preprocessing_contract_id"] for s in specs}
        self.assertEqual(by_id[SELECTED_MODEL_ID], SELECTED_PREPROCESSING_CONTRACT_ID)
        self.assertEqual(by_id["mmwave_resp_int8"], V01_PREPROCESSING_CONTRACT_ID)
        self.assertEqual(
            by_id["mmwave_resp_int8_v0.2.0_candidate"], V02_PREPROCESSING_CONTRACT_ID
        )
        # Must not use profile / model_id as contract id
        self.assertNotEqual(SELECTED_PREPROCESSING_CONTRACT_ID, "M-B1_D0_B1_Z1")
        self.assertNotEqual(V01_PREPROCESSING_CONTRACT_ID, "mmwave_resp_int8")

    def test_missing_bound_contract_fail_closed(self) -> None:
        frozen = load_frozen_execution_identity(ROOT)
        bad = copy.deepcopy(frozen)
        del bad["bound_contract_identity"]
        with self.assertRaises(MB10R1EvalError) as ctx:
            require_frozen_bound_contract(bad)
        self.assertIn("FROZEN_BOUND_CONTRACT_IDENTITY_MISSING_STOP_BEFORE_PAYLOAD", str(ctx.exception))
        source = (ROOT / "scripts/mmwave_m_b10r1_recovery_eval.py").read_text(encoding="utf-8")
        self.assertNotIn("or build_bound_contract_identity", source)


def _granted_mock_authorization() -> dict:
    overlay = not_authorized_overlay_template(freeze_sha="a" * 64, a_head="b" * 40)
    overlay["approval"] = True
    overlay["status"] = B_AUTHORIZATION_STATUS_GRANTED
    overlay["independent_reviewer_authorization"] = True
    overlay["recovery_execution_authorized"] = True
    overlay["recovery_payload_release_authorized"] = True
    overlay["reviewed_m_b10r1a_head_sha"] = "b" * 40
    overlay["reviewed_m_b10r1a_head_sha_status"] = "BOUND"
    return overlay


def _mock_runtime_after_release() -> dict:
    return {
        "schema_version": "M-B10R1B_RECOVERY_ACCESS_RUNTIME_STATE_V1",
        "original_final_accessor_invocations": 1,
        "original_locked_test_consumed": True,
        "original_final_payload_release_events": 1,
        "recovery_accessor_invocations": 1,
        "recovery_payload_release_events": 1,
        "historical_total_payload_release_events": 2,
        "payload_consumed": True,
        "rerun_performed": False,
        "automatic_retry": False,
    }


def _write_mock_b_tree(destination: Path) -> dict:
    specs = validate_frozen_recovery_models(ROOT)
    payload = _mock_recovery_payload()
    runners = {spec["model_id"]: FakeRunner() for spec in specs}
    evaluation = evaluate_recovery_payload(ROOT, payload, specs, runners=runners)
    persist_recovery_results(
        destination,
        evaluation,
        runtime_state=_mock_runtime_after_release(),
        authorization=_granted_mock_authorization(),
        frozen=load_frozen_execution_identity(ROOT),
        specs=specs,
    )
    return evaluation


class DurableResultAndBValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.holder = tempfile.TemporaryDirectory()
        cls.golden = Path(cls.holder.name) / "golden"
        cls.evaluation = _write_mock_b_tree(cls.golden)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.holder.cleanup()

    def _copy_golden(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        holder = tempfile.TemporaryDirectory()
        dest = Path(holder.name) / "b"
        shutil.copytree(self.golden, dest)
        return holder, dest

    def test_mock_end_to_end_persist_and_b_validator(self) -> None:
        self.assertEqual(self.evaluation["ledger_row_count"], EXPECTED_INFERENCES)
        self.assertEqual(self.evaluation["actual_total_tflite_invocations"], EXPECTED_INFERENCES)
        ledger_path = self.golden / "recovery_sample_predictions.jsonl"
        rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 225)
        result = b_validator.validate_m_b10r1b_artifacts(output_dir=self.golden)
        self.assertEqual(result["validation_status"], "PASS")
        self.assertEqual(result["ledger_row_count"], 225)
        self.assertFalse(result["locked_test_accessed"])

    def test_no_reaccess_monkeypatch_still_validates_mock_tree(self) -> None:
        def _forbidden(*_a, **_k):
            raise RuntimeError("ACCESSOR_MUST_NOT_BE_CALLED")

        with mock.patch(
            "scripts.mmwave_m_b10r1_recovery_access.LimitedReuseRecoveryAccessController.get_locked_test_recovery_evaluation_dataset",
            side_effect=_forbidden,
        ), mock.patch(
            "scripts.mmwave_phase_b_access.PhaseBAccessGuard.get_locked_test_final_evaluation_dataset",
            side_effect=_forbidden,
        ), mock.patch(
            "scripts.mmwave_phase_b_access.PhaseBAccessGuard._get_split_dataset",
            side_effect=_forbidden,
        ):
            result = b_validator.validate_m_b10r1b_artifacts(output_dir=self.golden)
        self.assertEqual(result["validation_status"], "PASS")

    def test_b_validator_source_never_calls_accessors(self) -> None:
        source = (ROOT / "scripts/validate_mmwave_m_b10r1b.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else (
                    func.attr if isinstance(func, ast.Attribute) else None
                )
                self.assertNotEqual(name, "get_locked_test_recovery_evaluation_dataset")
                self.assertNotEqual(name, "get_locked_test_final_evaluation_dataset")
                self.assertNotEqual(name, "execute_authorized_recovery")

    def test_terminal_failure_persistence(self) -> None:
        holder = tempfile.TemporaryDirectory()
        try:
            dest = Path(holder.name) / "fail"
            persist_terminal_failure(
                dest,
                runtime_state=_mock_runtime_after_release(),
                authorization=_granted_mock_authorization(),
                exception=RuntimeError("simulated post-payload failure"),
                failure_stage="POST_PAYLOAD_EVALUATION",
                ledger=[{"window_id": "w000", "model_id": SELECTED_MODEL_ID, "invalid": True}],
                completed_inference_count=3,
            )
            summary = json.loads((dest / "m_b10r1b_summary.json").read_text(encoding="utf-8"))
            audit = json.loads((dest / "one_time_recovery_access_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "PARTIAL_INCOMPLETE")
            self.assertEqual(audit["recovery_payload_release_events"], 1)
            self.assertEqual(audit["historical_total_payload_release_events"], 2)
            self.assertTrue(audit["payload_consumed"])
            self.assertFalse(audit["rerun_performed"])
            self.assertFalse(summary["metrics_populated"])
            with self.assertRaises(b_validator.MB10R1BValidationError):
                b_validator.validate_m_b10r1b_artifacts(output_dir=dest)
        finally:
            holder.cleanup()

    def _assert_b_fail(self, dest: Path) -> None:
        with self.assertRaises(b_validator.MB10R1BValidationError):
            b_validator.validate_m_b10r1b_artifacts(output_dir=dest)

    def test_prediction_changed_fails(self) -> None:
        holder, dest = self._copy_golden()
        try:
            rows = [json.loads(line) for line in (dest / "recovery_sample_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            rows[0]["predicted_class_index"] = (int(rows[0]["predicted_class_index"]) + 1) % 3
            rows[0]["predicted_class"] = ["NORMAL", "RAPID_OR_ABNORMAL", "APNEA"][rows[0]["predicted_class_index"]]
            (dest / "recovery_sample_predictions.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
            )
            _rewrite_checksums(dest)
            self._assert_b_fail(dest)
        finally:
            holder.cleanup()

    def test_true_label_changed_fails(self) -> None:
        holder, dest = self._copy_golden()
        try:
            rows = [json.loads(line) for line in (dest / "recovery_sample_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            rows[0]["true_class_index"] = (int(rows[0]["true_class_index"]) + 1) % 3
            (dest / "recovery_sample_predictions.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
            )
            _rewrite_checksums(dest)
            self._assert_b_fail(dest)
        finally:
            holder.cleanup()

    def test_row_deleted_fails(self) -> None:
        holder, dest = self._copy_golden()
        try:
            rows = [json.loads(line) for line in (dest / "recovery_sample_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            (dest / "recovery_sample_predictions.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows[:-1]), encoding="utf-8"
            )
            _rewrite_checksums(dest)
            self._assert_b_fail(dest)
        finally:
            holder.cleanup()

    def test_row_duplicated_fails(self) -> None:
        holder, dest = self._copy_golden()
        try:
            rows = [json.loads(line) for line in (dest / "recovery_sample_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            rows.append(copy.deepcopy(rows[0]))
            (dest / "recovery_sample_predictions.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
            )
            _rewrite_checksums(dest)
            self._assert_b_fail(dest)
        finally:
            holder.cleanup()

    def test_model_sha_changed_fails(self) -> None:
        holder, dest = self._copy_golden()
        try:
            rows = [json.loads(line) for line in (dest / "recovery_sample_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            rows[0]["model_sha256"] = "0" * 64
            (dest / "recovery_sample_predictions.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
            )
            _rewrite_checksums(dest)
            self._assert_b_fail(dest)
        finally:
            holder.cleanup()

    def test_preprocessing_contract_changed_fails(self) -> None:
        holder, dest = self._copy_golden()
        try:
            rows = [json.loads(line) for line in (dest / "recovery_sample_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            rows[0]["preprocessing_contract_id"] = "MUTATED_CONTRACT"
            (dest / "recovery_sample_predictions.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
            )
            _rewrite_checksums(dest)
            self._assert_b_fail(dest)
        finally:
            holder.cleanup()

    def test_seed43_inserted_fails(self) -> None:
        holder, dest = self._copy_golden()
        try:
            rows = [json.loads(line) for line in (dest / "recovery_sample_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            rows[0]["model_id"] = "M-B3_CONV1D_GAP_BASELINE_seed43_M-B6_STRICT_INT8"
            (dest / "recovery_sample_predictions.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
            )
            _rewrite_checksums(dest)
            self._assert_b_fail(dest)
        finally:
            holder.cleanup()

    def test_fourth_model_inserted_fails(self) -> None:
        holder, dest = self._copy_golden()
        try:
            rows = [json.loads(line) for line in (dest / "recovery_sample_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            extra = copy.deepcopy(rows[0])
            extra["model_id"] = "unexpected_fourth_model"
            extra["window_id"] = "w_extra"
            rows.append(extra)
            (dest / "recovery_sample_predictions.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
            )
            _rewrite_checksums(dest)
            self._assert_b_fail(dest)
        finally:
            holder.cleanup()

    def test_stored_macro_f1_changed_fails(self) -> None:
        holder, dest = self._copy_golden()
        try:
            data = json.loads((dest / "metrics_by_model.json").read_text(encoding="utf-8"))
            data[SELECTED_MODEL_ID]["macro_f1"] = 0.123456
            (dest / "metrics_by_model.json").write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _rewrite_checksums(dest)
            self._assert_b_fail(dest)
        finally:
            holder.cleanup()

    def test_confusion_matrix_changed_fails(self) -> None:
        holder, dest = self._copy_golden()
        try:
            data = json.loads((dest / "metrics_by_model.json").read_text(encoding="utf-8"))
            data[SELECTED_MODEL_ID]["confusion_matrix"][0][0] += 1
            (dest / "metrics_by_model.json").write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _rewrite_checksums(dest)
            self._assert_b_fail(dest)
        finally:
            holder.cleanup()

    def test_apnea_misses_changed_fails(self) -> None:
        holder, dest = self._copy_golden()
        try:
            data = json.loads((dest / "metrics_by_model.json").read_text(encoding="utf-8"))
            data[SELECTED_MODEL_ID]["apnea_proxy"]["misses"] = 99
            (dest / "metrics_by_model.json").write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _rewrite_checksums(dest)
            self._assert_b_fail(dest)
        finally:
            holder.cleanup()

    def test_subject_metric_changed_fails(self) -> None:
        holder, dest = self._copy_golden()
        try:
            data = json.loads((dest / "subject_level_metrics.json").read_text(encoding="utf-8"))
            data[SELECTED_MODEL_ID]["worst_subject_macro_f1"] = 0.0
            (dest / "subject_level_metrics.json").write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _rewrite_checksums(dest)
            self._assert_b_fail(dest)
        finally:
            holder.cleanup()

    def test_saturation_changed_fails(self) -> None:
        holder, dest = self._copy_golden()
        try:
            data = json.loads((dest / "selected_candidate_quantization_audit.json").read_text(encoding="utf-8"))
            data["input_saturation_ratio"] = 0.999
            data["samples_with_any_saturation"] = 75
            (dest / "selected_candidate_quantization_audit.json").write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _rewrite_checksums(dest)
            self._assert_b_fail(dest)
        finally:
            holder.cleanup()

    def test_inference_count_changed_fails(self) -> None:
        holder, dest = self._copy_golden()
        try:
            data = json.loads((dest / "m_b10r1b_summary.json").read_text(encoding="utf-8"))
            data["actual_total_tflite_invocations"] = 224
            (dest / "m_b10r1b_summary.json").write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _rewrite_checksums(dest)
            self._assert_b_fail(dest)
        finally:
            holder.cleanup()

    def test_access_release_not_1_fails(self) -> None:
        holder, dest = self._copy_golden()
        try:
            data = json.loads((dest / "one_time_recovery_access_audit.json").read_text(encoding="utf-8"))
            data["recovery_payload_release_events"] = 0
            (dest / "one_time_recovery_access_audit.json").write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _rewrite_checksums(dest)
            self._assert_b_fail(dest)
        finally:
            holder.cleanup()

    def test_historical_total_not_2_fails(self) -> None:
        holder, dest = self._copy_golden()
        try:
            data = json.loads((dest / "one_time_recovery_access_audit.json").read_text(encoding="utf-8"))
            data["historical_total_payload_release_events"] = 1
            (dest / "one_time_recovery_access_audit.json").write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _rewrite_checksums(dest)
            self._assert_b_fail(dest)
        finally:
            holder.cleanup()

    def test_rerun_true_fails(self) -> None:
        holder, dest = self._copy_golden()
        try:
            data = json.loads((dest / "m_b10r1b_summary.json").read_text(encoding="utf-8"))
            data["rerun_performed"] = True
            (dest / "m_b10r1b_summary.json").write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _rewrite_checksums(dest)
            self._assert_b_fail(dest)
        finally:
            holder.cleanup()

    def test_result_not_pristine_false_fails(self) -> None:
        holder, dest = self._copy_golden()
        try:
            data = json.loads((dest / "m_b10r1b_summary.json").read_text(encoding="utf-8"))
            data["result_not_pristine"] = False
            (dest / "m_b10r1b_summary.json").write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _rewrite_checksums(dest)
            self._assert_b_fail(dest)
        finally:
            holder.cleanup()

    def test_checksum_corruption_fails(self) -> None:
        holder, dest = self._copy_golden()
        try:
            data = json.loads((dest / "m_b10r1b_summary.json").read_text(encoding="utf-8"))
            data["note"] = "tampered"
            (dest / "m_b10r1b_summary.json").write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            self._assert_b_fail(dest)
        finally:
            holder.cleanup()


if __name__ == "__main__":
    unittest.main()
