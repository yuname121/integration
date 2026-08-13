"""Focused fail-closed tests for M-B10B incident-truth closure.

These tests never call the formal LOCKED_TEST final accessor. Mutations operate
on temporary evidence copies only.
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

from scripts import validate_mmwave_m_b10b_incident as incident


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / incident.OUT_DIR_REL


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


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


def _mutate_incident(path: Path, **updates: object) -> None:
    target = path / "incident_root_cause.json"
    data = json.loads(target.read_text(encoding="utf-8"))
    data.update(updates)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_checksums(path)


def _mutate_json(path: Path, filename: str, **updates: object) -> None:
    target = path / filename
    data = json.loads(target.read_text(encoding="utf-8"))
    data.update(updates)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_checksums(path)


class MB10BIncidentTruthTests(unittest.TestCase):
    def test_incident_validator_passes_on_current_evidence(self) -> None:
        result = incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=OUT)
        self.assertEqual(result["validation_status"], "PASS")
        self.assertEqual(result["forensic_root_cause"], incident.ROOT_CAUSE_ID)
        self.assertEqual(result["runtime_detection_code"], incident.RUNTIME_DETECTION_CODE)
        self.assertFalse(result["scientific_final_performance_available"])
        self.assertFalse(result["recovery_evaluation_authorized"])
        self.assertFalse(result["locked_test_reopen_authorized"])
        self.assertFalse(result["m_b11_authorized"])
        self.assertEqual(result["a6"]["windows"], 88)
        self.assertEqual(result["a6"]["eligible"], 75)
        self.assertEqual(result["a6"]["difference"], 13)

    def test_incident_validator_never_calls_final_accessor(self) -> None:
        def _boom(*_args, **_kwargs):
            raise AssertionError("final accessor must not be called by incident validator")

        with mock.patch(
            "scripts.mmwave_phase_b_access.PhaseBAccessGuard.get_locked_test_final_evaluation_dataset",
            side_effect=_boom,
        ):
            result = incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=OUT)
        self.assertEqual(result["validation_status"], "PASS")

    def test_incident_validator_source_has_no_final_accessor_call(self) -> None:
        source = (ROOT / "scripts/validate_mmwave_m_b10b_incident.py").read_text(encoding="utf-8")
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
                self.assertNotEqual(name, "PhaseBAccessGuard")

    def test_rejects_total_windows_as_75(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_incident(destination, a6_total_locked_test_windows=75)
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_eligible_windows_as_88(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_incident(destination, a6_locked_test_evaluation_eligible_windows=88)
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_difference_not_13(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_incident(destination, count_difference=12)
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_accessor_malfunction_claim(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_incident(
                destination,
                accessor_behavior_classification="ACCESSOR_MALFUNCTION",
                accessor_malfunction_evidence=True,
            )
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_dataset_corruption_claim(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_incident(destination, dataset_corruption_evidence=True)
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_split_mutation_claim(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_incident(destination, split_mutation_evidence=True)
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_runtime_detection_rewrite(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_incident(destination, runtime_detection_code="REWRITTEN_CODE")
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_zero_accessor_invocations(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_incident(destination, formal_accessor_invocations=0)
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_multiple_accessor_invocations(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_incident(destination, formal_accessor_invocations=2)
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_nonzero_model_inference(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_incident(destination, model_inference_invocations=1)
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_final_performance_available(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_incident(destination, scientific_final_performance_available=True)
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_pristine_locked_test_claim(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_incident(destination, locked_test_consumed=False)
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_rerun_authorization(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_incident(destination, rerun_performed=True)
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_recovery_evaluation_authorization(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_incident(destination, recovery_evaluation_authorized=True)
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_m_b11_authorization(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_incident(destination, m_b11_authorized=True)
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_consumption_record_made_reusable(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_json(
                destination,
                "test_split_consumption_record.json",
                must_not_reuse_for_phase_b_model_selection=False,
                status="LOCKED_TEST_NOT_USED",
            )
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_completed_performance_artifact(self) -> None:
        holder, destination = _copy_output()
        try:
            _mutate_json(
                destination,
                "metrics_by_model.json",
                results_available=True,
                status="COMPLETE",
                accuracy=0.0,
            )
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_checksum_corruption(self) -> None:
        holder, destination = _copy_output()
        try:
            checksum = destination / "checksums.sha256"
            checksum.write_text(checksum.read_text(encoding="utf-8").replace("a", "b", 1), encoding="utf-8")
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_absolute_checksum_path(self) -> None:
        holder, destination = _copy_output()
        try:
            lines = []
            for item in sorted(destination.iterdir(), key=lambda value: value.name):
                if not item.is_file() or item.name == "checksums.sha256":
                    continue
                lines.append(f"{_sha256_file(item)}  /tmp/{item.name}")
            (destination / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_checksum_path_traversal(self) -> None:
        holder, destination = _copy_output()
        try:
            lines = []
            for item in sorted(destination.iterdir(), key=lambda value: value.name):
                if not item.is_file() or item.name == "checksums.sha256":
                    continue
                relative = f"../{item.name}" if item.name == "incident_root_cause.json" else item.name
                lines.append(f"{_sha256_file(item)}  {relative}")
            (destination / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=destination)
        finally:
            holder.cleanup()

    def test_rejects_accessor_source_include_ambiguous_true(self) -> None:
        original = incident._static_verify_accessor_semantics

        def _mutated(root: Path):
            result = original(root)
            # Simulate a source inspection that no longer finds include_ambiguous=False.
            raise incident.MB10BIncidentValidationError("FINAL_ACCESSOR_DOES_NOT_EXCLUDE_AMBIGUOUS")

        with mock.patch.object(incident, "_static_verify_accessor_semantics", side_effect=_mutated):
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=OUT)

    def test_rejects_a6_count_corruption(self) -> None:
        original = incident._validate_a6_counts

        def _mutated(root: Path):
            data = original(root)
            corrupted = copy.deepcopy(data)
            corrupted["windows"] = 75
            corrupted["eligible"] = 88
            corrupted["difference"] = -13
            return corrupted

        with mock.patch.object(incident, "_validate_a6_counts", side_effect=_mutated):
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=OUT)

    def test_rejects_m_b10a_contract_hash_change(self) -> None:
        original = incident.sha256_file

        def _mutated(path: Path) -> str:
            if path.name == "locked_test_evaluation_contract.json":
                return "0" * 64
            return original(path)

        with mock.patch.object(incident, "sha256_file", side_effect=_mutated):
            with self.assertRaises(incident.MB10BIncidentValidationError):
                incident.validate_m_b10b_incident_artifacts(ROOT, output_dir=OUT)

    def test_static_accessor_exclusion_still_present(self) -> None:
        result = incident._static_verify_accessor_semantics(ROOT)
        self.assertFalse(result["accessor_include_ambiguous"])
        self.assertTrue(result["ambiguous_exclusion_verified_from_source"])
        self.assertEqual(result["accessor_behavior_classification"], "EXPECTED_EXISTING_ACCESSOR_BEHAVIOR")


if __name__ == "__main__":
    unittest.main(verbosity=2)
