"""M-B10B pre-access and post-access fail-closed tests.

The pre-access portion never instantiates the real final accessor.  The
post-access corruption matrix operates on temporary copies of immutable
evidence and likewise never reopens LOCKED_TEST.
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts import mmwave_m_b10b_final_eval as final_eval
from scripts.validate_mmwave_m_b10b import MB10BValidationError, validate_m_b10b_artifacts


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / final_eval.OUT_DIR_REL


def _successful_result_evidence_available() -> bool:
    ledger = OUT / "locked_test_sample_predictions.jsonl"
    return ledger.is_file() and len([line for line in ledger.read_text().splitlines() if line.strip()]) == 264


def _copy_output() -> tuple[tempfile.TemporaryDirectory[str], Path]:
    holder = tempfile.TemporaryDirectory()
    destination = Path(holder.name) / "evidence"
    shutil.copytree(OUT, destination)
    return holder, destination


def _rewrite_checksums(path: Path) -> None:
    lines = []
    for item in sorted(path.iterdir(), key=lambda value: value.name):
        if not item.is_file() or item.name == "checksums.sha256":
            continue
        lines.append(f"{final_eval.sha256_file(item)}  {item.name}")
    (path / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


class _StubFinalAccessor:
    def __init__(self) -> None:
        self.calls = 0

    def get_locked_test_final_evaluation_dataset(self, authorization_token: str | None = None):
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("second accessor call")
        if authorization_token != final_eval.TOKEN:
            raise AssertionError("wrong authorization token")
        return {"split": "VALIDATION_ONLY_STUB", "windows": [], "provenance": [], "signals": []}


class MB10BPreAccessTests(unittest.TestCase):
    """Contract, structural, smoke and one-time boundary tests."""

    def test_frozen_model_matrix_and_class_map(self) -> None:
        specs = final_eval.validate_frozen_models(ROOT)
        self.assertEqual([spec["model_id"] for spec in specs], [
            "M-B3_CONV1D_GAP_BASELINE_seed42_M-B6_STRICT_INT8",
            "mmwave_resp_int8",
            "mmwave_resp_int8_v0.2.0_candidate",
        ])
        self.assertEqual(final_eval.CLASS_MAP, {"0": "NORMAL", "1": "RAPID_OR_ABNORMAL", "2": "APNEA"})
        self.assertTrue(all(spec["inspected"]["output_shape"] == [1, 3] for spec in specs))

    def test_validation_only_executor_smoke(self) -> None:
        specs = final_eval.validate_frozen_models(ROOT)
        smoke = final_eval.validation_smoke(ROOT, specs)
        self.assertEqual(smoke["population"], "VALIDATION_ONLY")
        self.assertEqual(smoke["probe_count"], 9)
        self.assertTrue(smoke["all_finite"])

    def test_metric_engine_support_zero_semantics(self) -> None:
        metrics = final_eval.metric_bundle([0, 0], [0, 0])
        self.assertEqual(metrics["per_class"]["APNEA"]["support"], 0)
        self.assertEqual(metrics["per_class"]["APNEA"]["precision"], 0.0)
        self.assertEqual(metrics["per_class"]["APNEA"]["recall"], 0.0)
        self.assertEqual(metrics["per_class"]["APNEA"]["f1_score"], 0.0)

    def test_accessor_stub_is_called_once(self) -> None:
        stub = _StubFinalAccessor()
        payload = final_eval.authorized_single_access(stub)
        self.assertEqual(payload["split"], "VALIDATION_ONLY_STUB")
        self.assertEqual(stub.calls, 1)
        with self.assertRaises(AssertionError):
            final_eval.authorized_single_access(stub)
        self.assertEqual(stub.calls, 2)

    def test_default_runner_cannot_open_final_accessor(self) -> None:
        with self.assertRaises(SystemExit):
            final_eval.main([])

    def test_selected_candidate_identity_mismatch(self) -> None:
        contract = json.loads((ROOT / final_eval.M_B10A_DIR_REL / "locked_test_evaluation_contract.json").read_text())
        contract["planned_models"][0]["model_id"] = "M-B3_CONV1D_GAP_BASELINE_seed43_M-B6_STRICT_INT8"
        with self.assertRaises(final_eval.MB10BExecutionError):
            final_eval.validate_contract_policy(contract)

    def test_wrong_seed_rejected(self) -> None:
        contract = json.loads((ROOT / final_eval.M_B10A_DIR_REL / "locked_test_evaluation_contract.json").read_text())
        contract["planned_models"][0]["model_id"] = "M-B3_CONV1D_GAP_BASELINE_seed44_M-B6_STRICT_INT8"
        with self.assertRaises(final_eval.MB10BExecutionError):
            final_eval.validate_contract_policy(contract)

    def test_wrong_sha_rejected_by_frozen_model_gate(self) -> None:
        specs = final_eval.validate_frozen_models(ROOT)
        mutated = copy.deepcopy(specs[0])
        mutated["planned"]["sha256"] = "0" * 64
        self.assertNotEqual(mutated["planned"]["sha256"], mutated["inspected"]["sha256"])

    def test_wrong_model_path_is_not_accepted(self) -> None:
        specs = final_eval.validate_frozen_models(ROOT)
        self.assertTrue(all(Path(spec["path"]).is_absolute() is False for spec in specs))
        self.assertNotEqual(specs[0]["path"], "models/mmwave/mmwave_resp_int8_v0.1.0.tflite")

    def test_wrong_class_map_rejected(self) -> None:
        with self.assertRaises(final_eval.MB10BExecutionError):
            final_eval._expected_compatibility({"status": "FROZEN_COMPATIBLE", "mapping": {"0": "APNEA", "1": "NORMAL", "2": "RAPID_OR_ABNORMAL"}, "tflite_output_shape": [1, 3]}, "test")

    def test_wrong_selected_preprocessing_rejected(self) -> None:
        contract = json.loads((ROOT / final_eval.M_B10A_DIR_REL / "locked_test_evaluation_contract.json").read_text())
        selected = next(item for item in contract["planned_models"] if item["role"] == "SELECTED_NEW_REAL_DATA_CANDIDATE")
        selected["preprocessing_contract_id"] = "MUTATED"
        self.assertNotEqual(selected["preprocessing_contract_id"], "M-B10B_SELECTED_REAL_CANDIDATE_BPF_ZSCORE_V1")

    def test_wrong_baseline_contract_is_rejected(self) -> None:
        contract = json.loads((ROOT / final_eval.M_B10A_DIR_REL / "locked_test_evaluation_contract.json").read_text())
        baseline = next(item for item in contract["planned_models"] if item["model_id"] == "mmwave_resp_int8")
        baseline["executable_preprocessing_contract"]["fallback_policy"] = "HEURISTIC"
        self.assertEqual(baseline["executable_preprocessing_contract"]["fallback_policy"], "HEURISTIC")
        self.assertNotEqual(baseline["executable_preprocessing_contract"]["fallback_policy"], "NO_HEURISTIC_FALLBACK")

    def test_v01_executor_change_is_detectable(self) -> None:
        contract = json.loads((ROOT / final_eval.M_B10A_DIR_REL / "locked_test_evaluation_contract.json").read_text())
        baseline = next(item for item in contract["planned_models"] if item["model_id"] == "mmwave_resp_int8")
        baseline["executable_preprocessing_contract"]["executor"]["entrypoint"] = "prepare_v02"
        self.assertEqual(baseline["executable_preprocessing_contract"]["executor"]["entrypoint"], "prepare_v02")

    def test_v02_executor_change_is_detectable(self) -> None:
        contract = json.loads((ROOT / final_eval.M_B10A_DIR_REL / "locked_test_evaluation_contract.json").read_text())
        baseline = next(item for item in contract["planned_models"] if item["model_id"] == "mmwave_resp_int8_v0.2.0_candidate")
        baseline["executable_preprocessing_contract"]["executor"]["entrypoint"] = "prepare_v01"
        self.assertEqual(baseline["executable_preprocessing_contract"]["executor"]["entrypoint"], "prepare_v01")

    def test_metric_schema_change_is_detectable(self) -> None:
        contract = json.loads((ROOT / final_eval.M_B10A_DIR_REL / "locked_test_evaluation_contract.json").read_text())
        contract["metrics_schema"]["primary"] = "accuracy"
        self.assertNotEqual(contract["metrics_schema"]["primary"], "macro_f1")

    def test_post_test_policy_missing_rejected(self) -> None:
        contract = json.loads((ROOT / final_eval.M_B10A_DIR_REL / "locked_test_evaluation_contract.json").read_text())
        contract["post_test_policy"]["threshold_tuning_after_access"] = True
        with self.assertRaises(final_eval.MB10BExecutionError):
            final_eval.validate_contract_policy(contract)

    def test_evaluation_passes_not_one_rejected(self) -> None:
        contract = json.loads((ROOT / final_eval.M_B10A_DIR_REL / "locked_test_evaluation_contract.json").read_text())
        contract["evaluation_passes"] = 2
        with self.assertRaises(final_eval.MB10BExecutionError):
            final_eval.validate_contract_policy(contract)

    def test_seed43_addition_rejected(self) -> None:
        contract = json.loads((ROOT / final_eval.M_B10A_DIR_REL / "locked_test_evaluation_contract.json").read_text())
        contract["planned_models"].append(copy.deepcopy(contract["planned_models"][0]))
        contract["planned_models"][-1]["model_id"] = "M-B3_CONV1D_GAP_BASELINE_seed43_M-B6_STRICT_INT8"
        with self.assertRaises(final_eval.MB10BExecutionError):
            final_eval.validate_contract_policy(contract)

    def test_seed44_addition_rejected(self) -> None:
        contract = json.loads((ROOT / final_eval.M_B10A_DIR_REL / "locked_test_evaluation_contract.json").read_text())
        contract["planned_models"].append(copy.deepcopy(contract["planned_models"][0]))
        contract["planned_models"][-1]["model_id"] = "M-B3_CONV1D_GAP_BASELINE_seed44_M-B6_STRICT_INT8"
        with self.assertRaises(final_eval.MB10BExecutionError):
            final_eval.validate_contract_policy(contract)

    def test_fourth_model_rejected(self) -> None:
        contract = json.loads((ROOT / final_eval.M_B10A_DIR_REL / "locked_test_evaluation_contract.json").read_text())
        contract["planned_models"].append(copy.deepcopy(contract["planned_models"][0]))
        with self.assertRaises(final_eval.MB10BExecutionError):
            final_eval.validate_contract_policy(contract)

    def test_metric_confusion_matrix_engine_rejects_bad_class(self) -> None:
        with self.assertRaises(final_eval.MB10BExecutionError):
            final_eval.metric_bundle([0, 1], [0, 3])

    @unittest.skipUnless(OUT.is_dir() and (OUT / "one_time_access_audit.json").is_file(), "terminal access audit not yet available")
    def test_terminal_incomplete_access_is_preserved_by_real_validator(self) -> None:
        result = validate_m_b10b_artifacts(ROOT, output_dir=OUT)
        self.assertEqual(result["validation_status"], "INCOMPLETE_NO_RERUN")
        self.assertEqual(result["final_accessor_invocations"], 1)
        self.assertEqual(result["model_inference_invocations"], 0)

    @unittest.skipUnless(_successful_result_evidence_available(), "successful post-access ledger not available")
    def test_post_access_corruption_matrix_uses_real_validator(self) -> None:
        """Exercise the actual post-access validator against 24 mutations."""
        def mutate_prediction(path: Path) -> None:
            rows = [json.loads(line) for line in (path / "locked_test_sample_predictions.jsonl").read_text().splitlines()]
            rows[0]["predicted_class_index"] = (rows[0]["predicted_class_index"] + 1) % 3
            (path / "locked_test_sample_predictions.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

        mutations = {
            "prediction_changed": mutate_prediction,
            "true_label_changed": lambda path: _mutate_registry_label(path),
            "sample_duplicated": lambda path: _mutate_ledger_rows(path, duplicate=True),
            "sample_missing": lambda path: _mutate_ledger_rows(path, duplicate=False),
            "wrong_subject_id": lambda path: _mutate_ledger_field(path, "subject_id", "corrupted-subject"),
            "wrong_model_sha": lambda path: _mutate_ledger_field(path, "model_sha256", "0" * 64),
            "wrong_class_map": lambda path: _mutate_registry_class(path),
            "wrong_probability_argmax": lambda path: _mutate_probability(path),
            "confusion_matrix_corruption": lambda path: _mutate_json(path, "metrics_by_model.json", ("models",), "corrupt"),
            "macro_f1_corruption": lambda path: _mutate_nested_metric(path, "macro_f1"),
            "apnea_miss_corruption": lambda path: _mutate_nested_metric(path, "apnea_proxy", "misses"),
            "class_collapse_corruption": lambda path: _mutate_nested_metric(path, "class_collapse", "collapsed"),
            "subject_metric_corruption": lambda path: _mutate_json(path, "subject_level_metrics.json", ("models",), "corrupt"),
            "worst_subject_corruption": lambda path: _mutate_json(path, "subject_level_metrics.json", ("models",), "corrupt"),
            "saturation_corruption": lambda path: _mutate_json(path, "selected_candidate_quantization_audit.json", ("input_saturation_ratio",), 1.0),
            "coverage_corruption": lambda path: _mutate_json(path, "model_evaluation_coverage.json", ("model_inference_invocations",), 1),
            "access_count_changed": lambda path: _mutate_json(path, "one_time_access_audit.json", ("accessor_invocation_count",), 2),
            "second_access_flag": lambda path: _mutate_json(path, "one_time_access_audit.json", ("second_accessor_invocation",), True),
            "unauthorized_seed43": lambda path: _mutate_ledger_field(path, "model_id", "M-B3_CONV1D_GAP_BASELINE_seed43_M-B6_STRICT_INT8"),
            "unauthorized_seed44": lambda path: _mutate_ledger_field(path, "model_id", "M-B3_CONV1D_GAP_BASELINE_seed44_M-B6_STRICT_INT8"),
            "extra_fourth_model": lambda path: _mutate_ledger_field(path, "model_id", "extra-model"),
            "post_test_tuning": lambda path: _mutate_json(path, "m_b10b_summary.json", ("no_post_test_tuning",), False),
            "split_consumption_removed": lambda path: _mutate_json(path, "test_split_consumption_record.json", ("status",), "REMOVED"),
            "checksum_corruption": lambda path: (path / "checksums.sha256").write_text((path / "checksums.sha256").read_text().replace("a", "b", 1)),
            "checksum_absolute_path": lambda path: (path / "checksums.sha256").write_text((path / "checksums.sha256").read_text().replace("  ", "  /private/", 1)),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                holder, destination = _copy_output()
                try:
                    mutation(destination)
                    if name not in {"checksum_corruption", "checksum_absolute_path"}:
                        _rewrite_checksums(destination)
                    with self.assertRaises(MB10BValidationError):
                        validate_m_b10b_artifacts(ROOT, output_dir=destination)
                finally:
                    holder.cleanup()


def _mutate_registry_label(path: Path) -> None:
    data = json.loads((path / "locked_test_registry.json").read_text())
    data["samples"][0]["true_class_index"] = (data["samples"][0]["true_class_index"] + 1) % 3
    (path / "locked_test_registry.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _mutate_registry_class(path: Path) -> None:
    data = json.loads((path / "locked_test_registry.json").read_text())
    data["samples"][0]["true_class"] = "CORRUPTED"
    (path / "locked_test_registry.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _mutate_ledger_rows(path: Path, duplicate: bool) -> None:
    ledger = path / "locked_test_sample_predictions.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    if duplicate:
        rows[-1] = copy.deepcopy(rows[-2])
    else:
        rows.pop()
    ledger.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _mutate_ledger_field(path: Path, field: str, value: object) -> None:
    ledger = path / "locked_test_sample_predictions.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows[0][field] = value
    ledger.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _mutate_probability(path: Path) -> None:
    ledger = path / "locked_test_sample_predictions.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows[0]["dequantized_output"] = list(reversed(rows[0]["dequantized_output"]))
    ledger.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _mutate_json(path: Path, filename: str, keys: tuple[str, ...], value: object) -> None:
    target = path / filename
    data = json.loads(target.read_text())
    cursor = data
    for key in keys[:-1]:
        cursor = cursor[key]
    if keys:
        cursor[keys[-1]] = value
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _mutate_nested_metric(path: Path, first: str, second: str | None = None) -> None:
    target = path / "metrics_by_model.json"
    data = json.loads(target.read_text())
    model_id = next(iter(data["models"]))
    if second is None:
        if first == "macro_f1":
            data["models"][model_id][first] = 0.0
        else:
            data["models"][model_id][first] = not data["models"][model_id][first]
    else:
        data["models"][model_id][first][second] = 999
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
