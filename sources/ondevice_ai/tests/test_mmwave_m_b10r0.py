"""Focused fail-closed tests for M-B10R0 holdout policy evidence."""

from __future__ import annotations

import json
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import validate_mmwave_m_b10r0 as validator
from scripts.mmwave_m_b10r0_holdout_policy import (
    OUT_DIR_REL,
    MODEL_SPECS,
    generate_m_b10r0_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / OUT_DIR_REL
M_B10A_DIR_REL = Path("datasets/mmwave/manifests/M-B10A_candidate_selection_setup")
M_B10B_DIR_REL = Path("datasets/mmwave/manifests/M-B10B_locked_test_final_evaluation")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        lines.append(f"{_sha256_file(item)}  {item.name}")
    (path / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mutate_json(path: Path, filename: str, **updates: object) -> None:
    target = path / filename
    data = json.loads(target.read_text(encoding="utf-8"))
    data.update(updates)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_checksums(path)


def _mutate_nested(path: Path, filename: str, mutator) -> None:
    target = path / filename
    data = json.loads(target.read_text(encoding="utf-8"))
    mutator(data)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_checksums(path)


def _copy_minimal_root_for_r6(*, include_v01: bool = True, include_v02: bool = True) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    holder = tempfile.TemporaryDirectory()
    fake_root = Path(holder.name)
    shutil.copytree(ROOT / M_B10B_DIR_REL, fake_root / M_B10B_DIR_REL)
    shutil.copytree(ROOT / M_B10A_DIR_REL, fake_root / M_B10A_DIR_REL)
    shutil.copytree(ROOT / "datasets/mmwave/manifests/a5_subject_split", fake_root / "datasets/mmwave/manifests/a5_subject_split")
    shutil.copytree(ROOT / "datasets/mmwave/manifests/a6_full_conversion", fake_root / "datasets/mmwave/manifests/a6_full_conversion")
    shutil.copytree(ROOT / "datasets/mmwave/manifests/a0_raw_inventory", fake_root / "datasets/mmwave/manifests/a0_raw_inventory")

    # Executor + metadata always needed for full R6 pass paths.
    for rel in (
        "scripts/mmwave_m_b10b_baseline_preprocessing.py",
        "models/mmwave/sensor_stats_metadata_v0.1.0.json",
        "models/mmwave/mmwave_resp_int8_v0.2.0_candidate_metadata.json",
    ):
        src = ROOT / rel
        dst = fake_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for m in MODEL_SPECS[1:]:
        if m["model_id"] == "mmwave_resp_int8" and not include_v01:
            continue
        if m["model_id"] == "mmwave_resp_int8_v0.2.0_candidate" and not include_v02:
            continue
        src = ROOT / m["path"]
        dst = fake_root / m["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return holder, fake_root


class MB10R0PolicyTests(unittest.TestCase):
    def test_validator_passes_on_current_evidence(self) -> None:
        result = validator.validate_m_b10r0_artifacts(ROOT, output_dir=OUT)
        self.assertEqual(result["validation_status"], "PASS")
        self.assertEqual(result["policy_decision"], "LIMITED_LOCKED_TEST_REUSE_EXCEPTION_RECOMMENDED")
        self.assertFalse(result["recovery_execution_authorized"])
        self.assertFalse(result["locked_test_reopen_authorized"])
        self.assertFalse(result["m_b11_authorized"])
        self.assertEqual(result["m_b10r0_accessor_invocations"], 0)

    def test_validator_never_calls_final_accessor(self) -> None:
        def _boom(*_args, **_kwargs):
            raise AssertionError("FORBIDDEN_M_B10R0_LOCKED_TEST_ACCESS")

        with mock.patch(
            "scripts.mmwave_phase_b_access.PhaseBAccessGuard.get_locked_test_final_evaluation_dataset",
            side_effect=_boom,
        ):
            result = validator.validate_m_b10r0_artifacts(ROOT, output_dir=OUT)
        self.assertEqual(result["validation_status"], "PASS")

    def test_validator_does_not_import_forbidden_generator_symbols(self) -> None:
        source = Path(validator.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from scripts.mmwave_m_b10r0_holdout_policy import", source)
        validator._validator_import_graph_safe()
        import ast

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "mmwave_m_b10r0_holdout_policy" in node.module:
                names = {a.name for a in node.names}
                self.assertFalse(
                    names
                    & {
                        "_reuse_gates",
                        "_policy_decision",
                        "_a5_inventory",
                        "_a6_eligible_subject_coverage",
                        "_exposure_assessment",
                    }
                )

    def test_rejects_previous_accessor_zero(self) -> None:
        holder, dest = _copy_output()
        try:
            _mutate_json(dest, "incident_identity.json", original_accessor_invocations=0)
            with self.assertRaises(validator.MB10R0ValidationError):
                validator.validate_m_b10r0_artifacts(ROOT, output_dir=dest)
        finally:
            holder.cleanup()

    def test_rejects_previous_accessor_two(self) -> None:
        holder, dest = _copy_output()
        try:
            _mutate_json(dest, "incident_identity.json", original_accessor_invocations=2)
            with self.assertRaises(validator.MB10R0ValidationError):
                validator.validate_m_b10r0_artifacts(ROOT, output_dir=dest)
        finally:
            holder.cleanup()

    def test_rejects_inference_count_positive(self) -> None:
        holder, dest = _copy_output()
        try:
            _mutate_json(dest, "incident_identity.json", model_inference_invocations=1)
            with self.assertRaises(validator.MB10R0ValidationError):
                validator.validate_m_b10r0_artifacts(ROOT, output_dir=dest)
        finally:
            holder.cleanup()

    def test_rejects_predictions_generated(self) -> None:
        holder, dest = _copy_output()
        try:
            _mutate_json(dest, "policy_decision.json", original_predictions_generated=True)
            with self.assertRaises(validator.MB10R0ValidationError):
                validator.validate_m_b10r0_artifacts(ROOT, output_dir=dest)
        finally:
            holder.cleanup()

    def test_rejects_metrics_generated(self) -> None:
        holder, dest = _copy_output()
        try:
            _mutate_json(dest, "policy_decision.json", original_metrics_generated=True)
            with self.assertRaises(validator.MB10R0ValidationError):
                validator.validate_m_b10r0_artifacts(ROOT, output_dir=dest)
        finally:
            holder.cleanup()

    def test_rejects_recovery_authorized(self) -> None:
        holder, dest = _copy_output()
        try:
            _mutate_json(dest, "policy_decision.json", recovery_execution_authorized=True)
            with self.assertRaises(validator.MB10R0ValidationError):
                validator.validate_m_b10r0_artifacts(ROOT, output_dir=dest)
        finally:
            holder.cleanup()

    def test_rejects_locked_test_reopen_authorized(self) -> None:
        holder, dest = _copy_output()
        try:
            _mutate_json(dest, "policy_decision.json", locked_test_reopen_authorized=True)
            with self.assertRaises(validator.MB10R0ValidationError):
                validator.validate_m_b10r0_artifacts(ROOT, output_dir=dest)
        finally:
            holder.cleanup()

    def test_rejects_m_b11_authorized(self) -> None:
        holder, dest = _copy_output()
        try:
            _mutate_json(dest, "policy_decision.json", m_b11_authorized=True)
            with self.assertRaises(validator.MB10R0ValidationError):
                validator.validate_m_b10r0_artifacts(ROOT, output_dir=dest)
        finally:
            holder.cleanup()

    def test_rejects_reuse_with_unused_holdout(self) -> None:
        holder, dest = _copy_output()
        try:
            inv = json.loads((dest / "existing_unused_holdout_inventory.json").read_text())
            inv["independent_existing_holdout_available"] = True
            inv["potential_independent_replacement_subjects"] = 5
            (dest / "existing_unused_holdout_inventory.json").write_text(json.dumps(inv, indent=2, sort_keys=True) + "\n")
            _mutate_json(dest, "policy_decision.json", decision="LIMITED_LOCKED_TEST_REUSE_EXCEPTION_RECOMMENDED")
            with self.assertRaises(validator.MB10R0ValidationError):
                validator.validate_m_b10r0_artifacts(ROOT, output_dir=dest)
        finally:
            holder.cleanup()

    def test_rejects_reuse_with_failed_gate(self) -> None:
        holder, dest = _copy_output()
        try:
            gates = json.loads((dest / "reuse_exception_gate_results.json").read_text())
            gates["gates"]["R3_zero_model_evaluation"]["pass"] = False
            gates["failed_gates"] = ["R3_zero_model_evaluation"]
            gates["all_r1_r10_pass"] = False
            (dest / "reuse_exception_gate_results.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")
            _mutate_json(dest, "policy_decision.json", decision="LIMITED_LOCKED_TEST_REUSE_EXCEPTION_RECOMMENDED", failed_reuse_gates=[])
            with self.assertRaises(validator.MB10R0ValidationError):
                validator.validate_m_b10r0_artifacts(ROOT, output_dir=dest)
        finally:
            holder.cleanup()

    def test_rejects_train_subject_as_replacement(self) -> None:
        holder, dest = _copy_output()
        try:
            inv = json.loads((dest / "existing_unused_holdout_inventory.json").read_text())
            inv["replacement_subject_ids"] = ["train-subject-1"]
            inv["train_subject_reuse_prohibited"] = False
            (dest / "existing_unused_holdout_inventory.json").write_text(json.dumps(inv, indent=2, sort_keys=True) + "\n")
            with self.assertRaises(validator.MB10R0ValidationError):
                validator.validate_m_b10r0_artifacts(ROOT, output_dir=dest)
        finally:
            holder.cleanup()

    def test_rejects_recovery_contract_authorized(self) -> None:
        holder, dest = _copy_output()
        try:
            _mutate_json(dest, "proposed_recovery_evaluation_contract.json", status="AUTHORIZED")
            with self.assertRaises(validator.MB10R0ValidationError):
                validator.validate_m_b10r0_artifacts(ROOT, output_dir=dest)
        finally:
            holder.cleanup()

    def test_rejects_nonzero_m_b10r0_accessor(self) -> None:
        holder, dest = _copy_output()
        try:
            _mutate_json(dest, "locked_test_access_audit.json", new_m_b10r0_accessor_invocations=1)
            with self.assertRaises(validator.MB10R0ValidationError):
                validator.validate_m_b10r0_artifacts(ROOT, output_dir=dest)
        finally:
            holder.cleanup()

    def test_rejects_checksum_corruption(self) -> None:
        holder, dest = _copy_output()
        try:
            checksum = dest / "checksums.sha256"
            checksum.write_text(checksum.read_text().replace("a", "b", 1), encoding="utf-8")
            with self.assertRaises(validator.MB10R0ValidationError):
                validator.validate_m_b10r0_artifacts(ROOT, output_dir=dest)
        finally:
            holder.cleanup()

    def test_rejects_checksum_traversal(self) -> None:
        holder, dest = _copy_output()
        try:
            lines = []
            for item in sorted(dest.iterdir(), key=lambda value: value.name):
                if not item.is_file() or item.name == "checksums.sha256":
                    continue
                rel = f"../{item.name}" if item.name == "policy_decision.json" else item.name
                lines.append(f"{_sha256_file(item)}  {rel}")
            (dest / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(validator.MB10R0ValidationError):
                validator.validate_m_b10r0_artifacts(ROOT, output_dir=dest)
        finally:
            holder.cleanup()

    def test_generator_produces_expected_decision(self) -> None:
        generate_m_b10r0_evidence(ROOT)
        result = validator.validate_m_b10r0_artifacts(ROOT, output_dir=OUT)
        self.assertEqual(result["policy_decision"], "LIMITED_LOCKED_TEST_REUSE_EXCEPTION_RECOMMENDED")

    # --- Inventory ---

    def test_rejects_missing_a0_subject_in_a5(self) -> None:
        fake_inv = {
            "total_original_subjects": 110,
            "train_subjects": 78,
            "validation_subjects": 15,
            "locked_test_subjects": 16,
            "assigned_subjects": 109,
            "unassigned_subjects": 1,
            "unassigned_subject_ids": ["fake-subject-999"],
            "potential_independent_replacement_subjects": 1,
            "replacement_subject_ids": ["fake-subject-999"],
            "train_subject_reuse_prohibited": True,
            "validation_subject_reuse_prohibited": True,
            "a5_reshuffle_prohibited": True,
            "evidence_paths": [
                "datasets/mmwave/manifests/a0_raw_inventory/recording_index.jsonl",
                "datasets/mmwave/manifests/a5_subject_split/subject_split_manifest.jsonl",
            ],
            "independent_existing_holdout_available": True,
            "reason": "1 subjects in A0 are not assigned in A5.",
        }
        with mock.patch.object(validator, "compute_subject_inventory", return_value=fake_inv):
            holder, dest = _copy_output()
            try:
                with self.assertRaises(validator.MB10R0ValidationError):
                    validator.validate_m_b10r0_artifacts(ROOT, output_dir=dest)
            finally:
                holder.cleanup()

    def test_rejects_duplicate_subject_across_splits(self) -> None:
        holder = tempfile.TemporaryDirectory()
        try:
            fake_root = Path(holder.name)
            a0_dir = fake_root / validator.A0_DIR_REL
            a0_dir.mkdir(parents=True)
            a5_dir = fake_root / validator.A5_DIR_REL
            a5_dir.mkdir(parents=True)
            (a0_dir / "recording_index.jsonl").write_text(
                '{"subject_id": "s1"}\n{"subject_id": "s2"}\n{"subject_id": "s3"}\n'
            )
            (a5_dir / "subject_split_manifest.jsonl").write_text(
                '{"subject_id": "s1", "split": "TRAIN"}\n'
                '{"subject_id": "s2", "split": "VALIDATION"}\n'
                '{"subject_id": "s2", "split": "LOCKED_TEST"}\n'
                '{"subject_id": "s3", "split": "LOCKED_TEST"}\n'
            )
            with self.assertRaises(validator.MB10R0InventoryError) as ctx:
                validator.compute_subject_inventory(fake_root)
            self.assertIn("OVERLAP", str(ctx.exception))
        finally:
            holder.cleanup()

    def test_rejects_a0_a5_subject_set_mismatch(self) -> None:
        holder = tempfile.TemporaryDirectory()
        try:
            fake_root = Path(holder.name)
            a0_dir = fake_root / validator.A0_DIR_REL
            a0_dir.mkdir(parents=True)
            a5_dir = fake_root / validator.A5_DIR_REL
            a5_dir.mkdir(parents=True)
            (a0_dir / "recording_index.jsonl").write_text('{"subject_id": "s1"}\n')
            (a5_dir / "subject_split_manifest.jsonl").write_text(
                '{"subject_id": "s1", "split": "TRAIN"}\n'
                '{"subject_id": "s_extra", "split": "VALIDATION"}\n'
            )
            with self.assertRaises(validator.MB10R0InventoryError) as ctx:
                validator.compute_subject_inventory(fake_root)
            self.assertIn("NOT_IN_A0", str(ctx.exception))
        finally:
            holder.cleanup()

    # --- R4 ---

    def test_rejects_registry_with_sample_rows(self) -> None:
        holder = tempfile.TemporaryDirectory()
        try:
            fake_root = Path(holder.name)
            mb10b = fake_root / M_B10B_DIR_REL
            mb10b.mkdir(parents=True)
            real_mb10b = ROOT / M_B10B_DIR_REL
            for f in real_mb10b.iterdir():
                shutil.copy2(f, mb10b / f.name)
            registry = json.loads((mb10b / "locked_test_registry.json").read_text())
            registry["samples"] = [{"id": "fake", "subject_id": "s", "label": 1}]
            (mb10b / "locked_test_registry.json").write_text(json.dumps(registry))
            shutil.copytree(ROOT / "datasets/mmwave/manifests/a6_full_conversion", fake_root / "datasets/mmwave/manifests/a6_full_conversion")
            gate = validator.evaluate_gate_r4(fake_root)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    def test_rejects_persisted_sample_id(self) -> None:
        holder = tempfile.TemporaryDirectory()
        try:
            fake_root = Path(holder.name)
            mb10b = fake_root / M_B10B_DIR_REL
            mb10b.mkdir(parents=True)
            real_mb10b = ROOT / M_B10B_DIR_REL
            for f in real_mb10b.iterdir():
                shutil.copy2(f, mb10b / f.name)
            (mb10b / "locked_test_sample_predictions.jsonl").write_text('{"sample_id": "x"}\n')
            shutil.copytree(ROOT / "datasets/mmwave/manifests/a6_full_conversion", fake_root / "datasets/mmwave/manifests/a6_full_conversion")
            exposure = validator.compute_exposure(fake_root)
            self.assertTrue(exposure["E3_persistent_sample_registry"]["persisted_sample_registry_exposure"])
            self.assertTrue(exposure["summary"]["PERSISTED_SAMPLE_REGISTRY_EXPOSURE"])
        finally:
            holder.cleanup()

    def test_rejects_persisted_label_or_tensor(self) -> None:
        holder = tempfile.TemporaryDirectory()
        try:
            fake_root = Path(holder.name)
            mb10b = fake_root / M_B10B_DIR_REL
            mb10b.mkdir(parents=True)
            real_mb10b = ROOT / M_B10B_DIR_REL
            for f in real_mb10b.iterdir():
                shutil.copy2(f, mb10b / f.name)
            registry = json.loads((mb10b / "locked_test_registry.json").read_text())
            registry["raw_tensors_persisted"] = True
            (mb10b / "locked_test_registry.json").write_text(json.dumps(registry))
            shutil.copytree(ROOT / "datasets/mmwave/manifests/a6_full_conversion", fake_root / "datasets/mmwave/manifests/a6_full_conversion")
            exposure = validator.compute_exposure(fake_root)
            self.assertTrue(exposure["E3_persistent_sample_registry"]["persisted_sample_registry_exposure"])
        finally:
            holder.cleanup()

    # --- R6 ---

    def test_r6_rejects_missing_v01_model(self) -> None:
        holder, fake_root = _copy_minimal_root_for_r6(include_v01=False, include_v02=True)
        try:
            gate = validator.evaluate_gate_r6(fake_root)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    def test_r6_rejects_missing_v02_model(self) -> None:
        holder, fake_root = _copy_minimal_root_for_r6(include_v01=True, include_v02=False)
        try:
            gate = validator.evaluate_gate_r6(fake_root)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    def test_r6_rejects_baseline_sha_mismatch(self) -> None:
        holder, fake_root = _copy_minimal_root_for_r6()
        try:
            model = fake_root / MODEL_SPECS[1]["path"]
            model.write_bytes(b"corrupted-model-bytes")
            gate = validator.evaluate_gate_r6(fake_root)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    def test_r6_rejects_executor_sha_mismatch(self) -> None:
        holder, fake_root = _copy_minimal_root_for_r6()
        try:
            executor = fake_root / "scripts/mmwave_m_b10b_baseline_preprocessing.py"
            executor.write_text(executor.read_text(encoding="utf-8") + "\n# mutated\n", encoding="utf-8")
            gate = validator.evaluate_gate_r6(fake_root)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    def test_r6_rejects_metadata_sha_mismatch(self) -> None:
        holder, fake_root = _copy_minimal_root_for_r6()
        try:
            meta = fake_root / "models/mmwave/sensor_stats_metadata_v0.1.0.json"
            meta.write_text(meta.read_text(encoding="utf-8").replace("0", "1", 1), encoding="utf-8")
            gate = validator.evaluate_gate_r6(fake_root)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    def test_r6_rejects_preprocessing_step_changed(self) -> None:
        holder, fake_root = _copy_minimal_root_for_r6()
        try:
            reg_path = fake_root / M_B10A_DIR_REL / "historical_baseline_registry.json"
            reg = json.loads(reg_path.read_text(encoding="utf-8"))
            reg["baselines"][0]["executable_preprocessing_contract"]["steps"][0]["parameters"]["exact_samples"] = 299
            reg_path.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            gate = validator.evaluate_gate_r6(fake_root)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    def test_r6_rejects_class_map_changed(self) -> None:
        holder, fake_root = _copy_minimal_root_for_r6()
        try:
            reg_path = fake_root / M_B10A_DIR_REL / "historical_baseline_registry.json"
            reg = json.loads(reg_path.read_text(encoding="utf-8"))
            reg["baselines"][0]["executable_preprocessing_contract"]["class_map"]["0"] = "OTHER"
            reg_path.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            gate = validator.evaluate_gate_r6(fake_root)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    def test_r6_rejects_interpretation_changed(self) -> None:
        holder, fake_root = _copy_minimal_root_for_r6()
        try:
            reg_path = fake_root / M_B10A_DIR_REL / "historical_baseline_registry.json"
            reg = json.loads(reg_path.read_text(encoding="utf-8"))
            reg["baselines"][0]["executable_preprocessing_contract"]["interpretation"] = "CHANGED"
            reg_path.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            gate = validator.evaluate_gate_r6(fake_root)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    # --- R9 ---

    def test_r9_rejects_seed43_model(self) -> None:
        holder, dest = _copy_output()
        try:
            def mut(data):
                data["planned_models"][0]["model_id"] = "M-B3_CONV1D_GAP_BASELINE_seed43_M-B6_STRICT_INT8"
                data["models"][0] = data["planned_models"][0]["model_id"]

            _mutate_nested(dest, "proposed_recovery_evaluation_contract.json", mut)
            gate = validator.evaluate_gate_r9(ROOT, dest)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    def test_r9_rejects_seed44_model(self) -> None:
        holder, dest = _copy_output()
        try:
            def mut(data):
                data["planned_models"][0]["model_id"] = "M-B3_CONV1D_GAP_BASELINE_seed44_M-B6_STRICT_INT8"
                data["models"][0] = data["planned_models"][0]["model_id"]

            _mutate_nested(dest, "proposed_recovery_evaluation_contract.json", mut)
            gate = validator.evaluate_gate_r9(ROOT, dest)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    def test_r9_rejects_fourth_model(self) -> None:
        holder, dest = _copy_output()
        try:
            def mut(data):
                data["planned_models"].append({"model_id": "fourth", "role": "EXTRA", "path": "x", "sha256": "0" * 64})
                data["models"].append("fourth")
                data["model_count"] = 4

            _mutate_nested(dest, "proposed_recovery_evaluation_contract.json", mut)
            gate = validator.evaluate_gate_r9(ROOT, dest)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    def test_r9_rejects_baseline_removed(self) -> None:
        holder, dest = _copy_output()
        try:
            def mut(data):
                data["planned_models"] = [m for m in data["planned_models"] if m["model_id"] != "mmwave_resp_int8"]
                data["models"] = [m["model_id"] for m in data["planned_models"]]
                data["model_count"] = len(data["models"])

            _mutate_nested(dest, "proposed_recovery_evaluation_contract.json", mut)
            gate = validator.evaluate_gate_r9(ROOT, dest)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    def test_r9_rejects_duplicate_model(self) -> None:
        holder, dest = _copy_output()
        try:
            def mut(data):
                data["planned_models"][2] = dict(data["planned_models"][1])
                data["models"] = [m["model_id"] for m in data["planned_models"]]

            _mutate_nested(dest, "proposed_recovery_evaluation_contract.json", mut)
            gate = validator.evaluate_gate_r9(ROOT, dest)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    def test_r9_rejects_model_sha_changed(self) -> None:
        holder, dest = _copy_output()
        try:
            def mut(data):
                data["planned_models"][0]["sha256"] = "0" * 64

            _mutate_nested(dest, "proposed_recovery_evaluation_contract.json", mut)
            gate = validator.evaluate_gate_r9(ROOT, dest)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    def test_r9_rejects_inference_count(self) -> None:
        holder, dest = _copy_output()
        try:
            _mutate_json(dest, "proposed_recovery_evaluation_contract.json", expected_model_inference_count=224)
            gate = validator.evaluate_gate_r9(ROOT, dest)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    def test_r9_rejects_eligible_count(self) -> None:
        holder, dest = _copy_output()
        try:
            def mut(data):
                data["supervised_evaluation_population"]["windows"] = 74

            _mutate_nested(dest, "proposed_recovery_evaluation_contract.json", mut)
            gate = validator.evaluate_gate_r9(ROOT, dest)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    def test_r9_rejects_structural_count(self) -> None:
        holder, dest = _copy_output()
        try:
            def mut(data):
                data["structural_context"]["total_windows"] = 87

            _mutate_nested(dest, "proposed_recovery_evaluation_contract.json", mut)
            gate = validator.evaluate_gate_r9(ROOT, dest)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    def test_r9_rejects_metrics_schema_corruption(self) -> None:
        holder, dest = _copy_output()
        try:
            def mut(data):
                schema = data["metrics_schema"]
                schema["apnea_proxy_fields"] = [x for x in schema["apnea_proxy_fields"] if x != "recall"]
                schema["primary"] = "accuracy"
                schema["per_class_fields"] = [x for x in schema["per_class_fields"] if x != "fpr"]
                schema["subject_level"] = [x for x in schema["subject_level"] if x != "worst_subject_macro_f1"]

            _mutate_nested(dest, "proposed_recovery_evaluation_contract.json", mut)
            gate = validator.evaluate_gate_r9(ROOT, dest)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    # --- R10 ---

    def test_r10_rejects_pristine_designation(self) -> None:
        holder, dest = _copy_output()
        try:
            _mutate_json(dest, "proposed_recovery_evaluation_contract.json", required_result_designation="PRISTINE_ONE_TIME_LOCKED_TEST")
            gate = validator.evaluate_gate_r10(ROOT, dest)
            self.assertEqual(gate["status"], "FAIL")
            with self.assertRaises(validator.MB10R0ValidationError):
                validator.validate_m_b10r0_artifacts(ROOT, output_dir=dest)
        finally:
            holder.cleanup()

    def test_r10_rejects_result_not_pristine_false(self) -> None:
        holder, dest = _copy_output()
        try:
            def mut(data):
                data["result_limitation_fields"]["result_not_pristine"] = False

            _mutate_nested(dest, "proposed_recovery_evaluation_contract.json", mut)
            gate = validator.evaluate_gate_r10(ROOT, dest)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    def test_r10_rejects_original_pristine_consumed_false(self) -> None:
        holder, dest = _copy_output()
        try:
            def mut(data):
                data["result_limitation_fields"]["original_pristine_final_access_consumed"] = False

            _mutate_nested(dest, "proposed_recovery_evaluation_contract.json", mut)
            gate = validator.evaluate_gate_r10(ROOT, dest)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    def test_r10_rejects_recovery_status_authorized(self) -> None:
        holder, dest = _copy_output()
        try:
            _mutate_json(dest, "proposed_recovery_evaluation_contract.json", status="AUTHORIZED")
            gate = validator.evaluate_gate_r10(ROOT, dest)
            self.assertEqual(gate["status"], "FAIL")
        finally:
            holder.cleanup()

    # --- Independence ---

    def test_independence_gate_results_mismatch(self) -> None:
        holder, dest = _copy_output()
        try:
            gates = json.loads((dest / "reuse_exception_gate_results.json").read_text())
            gates["gates"]["R6_baselines_immutable"]["pass"] = False
            gates["failed_gates"] = ["R6_baselines_immutable"]
            gates["all_r1_r10_pass"] = False
            (dest / "reuse_exception_gate_results.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")
            _rewrite_checksums(dest)
            with self.assertRaises(validator.MB10R0ValidationError) as ctx:
                validator.validate_m_b10r0_artifacts(ROOT, output_dir=dest)
            self.assertIn("GATE", str(ctx.exception))
        finally:
            holder.cleanup()

    def test_independence_policy_decision_mismatch(self) -> None:
        holder, dest = _copy_output()
        try:
            _mutate_json(dest, "policy_decision.json", decision="NO_VALID_RECOVERY_PATH")
            with self.assertRaises(validator.MB10R0ValidationError) as ctx:
                validator.validate_m_b10r0_artifacts(ROOT, output_dir=dest)
            self.assertIn("POLICY_DECISION_RECOMPUTATION_MISMATCH", str(ctx.exception))
        finally:
            holder.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
