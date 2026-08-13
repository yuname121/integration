#!/usr/bin/env python3
"""V5 provider, runtime, artifact, and standalone-release regressions."""

from __future__ import annotations

from pathlib import Path
import ast
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
import zipfile

from inference.inference_result import InferenceResult
from inference.validator import GroundTruthValidator
from integrated_node.run_node import SafeNestIntegratedNode
from integrated_node.runtime_config import RuntimeConfigError
from risk.risk_engine import SafeNestRiskEngine
from scripts.build_v5_archive import (
    ARCHIVE_ROOT,
    REQUIRED_FILES,
    build_archive,
    sha256_file,
    verify_archive,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = Path(
    os.environ.get("SAFENEST_REPOSITORY_ROOT", str(PROJECT_ROOT.parent))
).resolve()
NESTED = os.environ.get("SAFENEST_ARCHIVE_NESTED") == "1"


def result(sensor_id, score=0.0, state="NORMAL", timestamp=None):
    return InferenceResult(
        sensor_id=sensor_id,
        timestamp=time.time() if timestamp is None else timestamp,
        score=score,
        state=state,
        confidence=1.0,
        valid=True,
        latency_ms=0.1,
        metadata={},
    )


class Provider:
    def __init__(self, sensor_id, value=None, connects=True):
        self.sensor_id = sensor_id
        self.value = value or result(sensor_id)
        self.connects = connects
        self.closed = False

    def connect(self):
        return self.connects

    def read(self):
        return self.value

    def close(self):
        self.closed = True


def providers(now=None):
    ts = time.time() if now is None else now
    return {
        "thermal44": Provider("thermal44", result("thermal44", timestamp=ts)),
        "mmwave": Provider("mmwave", result("mmwave", timestamp=ts)),
        "co2": Provider("co2", result("co2", timestamp=ts)),
        "pir": Provider("pir", result("pir", state="MOTION", timestamp=ts)),
    }


class TestV5Runtime(unittest.TestCase):
    def test_validator_targets_v5_from_v5_directory(self):
        valid, inventory, errors = GroundTruthValidator(
            project_root=PROJECT_ROOT
        ).validate_all(generate_inventory=False)
        self.assertTrue(valid, errors)
        for model in inventory["models"].values():
            self.assertTrue(
                model["repository_relative_model_path"].startswith(
                    "SafeNest_V5_OnDevice_AI/"
                )
            )
        with tempfile.TemporaryDirectory() as temp:
            env = os.environ.copy()
            env["PYTHONPYCACHEPREFIX"] = str(Path(temp) / "pycache")
            env["MPLCONFIGDIR"] = str(Path(temp) / "mpl")
            proc = subprocess.run(
                [sys.executable, "scripts/validate_v4_config.py"],
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("SafeNest_V5_OnDevice_AI/docs/reports/model_inventory.json", proc.stdout)

    def test_external_provider_injection_and_connect_failures(self):
        node = SafeNestIntegratedNode(mode="real", sensors=providers())
        node.start()
        healthy = node.step()
        node.shutdown()
        self.assertEqual(healthy.system_health, "HEALTHY")
        self.assertTrue(all(item.closed for item in node.sensors.values()))

        one_failed = providers()
        one_failed["co2"] = Provider("co2", connects=False)
        node = SafeNestIntegratedNode(mode="real", sensors=one_failed)
        node.start()
        degraded = node.step()
        node.shutdown()
        self.assertEqual(degraded.system_health, "DEGRADED")
        self.assertIn("co2", degraded.invalid_sensors)

        all_failed = {key: Provider(key, connects=False) for key in providers()}
        node = SafeNestIntegratedNode(mode="real", sensors=all_failed)
        node.start()
        failed = node.step()
        node.shutdown()
        self.assertEqual(failed.system_health, "FAILED")
        self.assertIsNone(failed.risk_score)

    def test_missing_real_providers_fail_closed(self):
        node = SafeNestIntegratedNode(mode="real")
        node.start()
        payload = node.step().to_dict()
        node.shutdown()
        self.assertEqual(payload["system_health"], "FAILED")
        self.assertIsNone(payload["risk_score"])
        self.assertIsNone(payload["risk_level"])
        for sensor in payload["sensors"].values():
            self.assertFalse(sensor["valid"])
            self.assertEqual(sensor["error"], "EXTERNAL_SENSOR_PROVIDER_REQUIRED")

    def test_mock_schema_and_compatibility_fields(self):
        node = SafeNestIntegratedNode(mode="mock")
        node.start()
        payload = node.step().to_dict()
        node.shutdown()
        self.assertEqual(payload["system_health"], "HEALTHY")
        self.assertEqual(payload["metadata"]["schema_version"], "5.0")
        self.assertEqual(set(payload["component_scores"]), {"thermal", "mmwave", "co2", "pir"})
        for name in ("level", "system_status", "fallback_used"):
            self.assertIn(name, payload)

    def test_sensor_ttl_co2_five_seconds_and_thermal_alias(self):
        now = time.time()
        engine = SafeNestRiskEngine(
            stale_sec={"thermal": 3.0, "mmwave": 3.0, "co2": 10.0, "pir": 10.0}
        )
        output = engine.evaluate(
            {
                "thermal44": result("thermal44", timestamp=now - 4.0),
                "mmwave": result("mmwave", timestamp=now),
                "co2": result("co2", timestamp=now - 5.0),
                "pir": result("pir", timestamp=now),
            },
            now=now,
        )
        self.assertIn("thermal", output.stale_sensors)
        self.assertNotIn("co2", output.stale_sensors)
        self.assertIn("thermal", output.component_scores)
        self.assertNotIn("thermal44", output.component_scores)
        self.assertIn("thermal44", output.sensors)

    def test_config_is_effective_and_mismatch_fails_startup(self):
        node = SafeNestIntegratedNode(mode="real", sensors=providers())
        self.assertEqual(node.risk_engine.stale_sec["co2"], 10.0)
        self.assertEqual(node.runtime_settings.loop_interval_sec, 0.2)
        mismatched = providers()
        mismatched["co2"].stale_sec = 3.0
        with self.assertRaises(RuntimeConfigError):
            SafeNestIntegratedNode(mode="real", sensors=mismatched)

    def test_thermal_and_mmwave_emergency_overrides(self):
        for key, state in (("thermal44", "HUMAN_FALL"), ("mmwave", "APNEA")):
            injected = providers()
            injected[key].value = result(key, score=1.0, state=state)
            node = SafeNestIntegratedNode(mode="real", sensors=injected)
            node.start()
            output = node.step()
            node.shutdown()
            self.assertEqual((output.risk_level, output.risk_score), ("DANGER", 100.0))


class TestV5Integrity(unittest.TestCase):
    def test_models_match_v5_and_v4_manifests(self):
        manifest = json.loads((PROJECT_ROOT / "models/model_manifest.json").read_text())
        if NESTED:
            self.assertTrue(all(x["version"] == "0.1.0" for x in manifest["models"].values()))
            return
        frozen = json.loads(
            (
                REPOSITORY_ROOT
                / "version_archives/2026-08-03/SNAPSHOT_MANIFEST.json"
            ).read_text()
        )
        for key in ("thermal", "mmwave", "co2"):
            entry = manifest["models"][key]
            digest = hashlib.sha256((PROJECT_ROOT / entry["path"]).read_bytes()).hexdigest()
            self.assertEqual(entry["version"], "0.1.0")
            self.assertEqual(digest, entry["sha256"])
            self.assertEqual(digest, frozen["models"][key]["sha256"])

    def test_production_does_not_import_archives_or_releases(self):
        forbidden = {"archive", "version_archives", "releases"}
        violations = []
        for root in ("inference", "sensors", "risk", "integrated_node"):
            for path in (PROJECT_ROOT / root).rglob("*.py"):
                tree = ast.parse(path.read_text(), filename=str(path))
                for node in ast.walk(tree):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        names = [node.module]
                    if any(name.split(".")[0] in forbidden for name in names):
                        violations.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(violations, [])

    def test_frozen_snapshot_tree_and_checksums_after_v5(self):
        if NESTED:
            self.assertEqual(PROJECT_ROOT.name, ARCHIVE_ROOT)
            return
        base = REPOSITORY_ROOT / "version_archives/2026-08-03"
        snapshot = base / "SafeNest_V4_P0_FINAL"
        tar_path = base / "SafeNest_V4_P0_FINAL.tar.gz"
        manifest = json.loads((base / "SNAPSHOT_MANIFEST.json").read_text())
        expected = dict(
            line.split("  ", 1)[::-1]
            for line in (base / "SHA256SUMS.txt").read_text().splitlines()
        )
        self.assertEqual(sha256_file(tar_path), manifest["archive_sha256"])
        self.assertEqual(sha256_file(tar_path), expected[tar_path.name])
        files = {
            path.relative_to(snapshot).as_posix(): path
            for path in snapshot.rglob("*")
            if path.is_file()
        }
        checksums = {
            name.removeprefix("SafeNest_V4_P0_FINAL/"): digest
            for name, digest in expected.items()
            if name.startswith("SafeNest_V4_P0_FINAL/")
        }
        self.assertEqual(set(files), set(checksums))
        dataless = 0x40000000
        with tarfile.open(tar_path, "r:gz") as archive:
            members = {
                item.name.removeprefix("SafeNest_V4_P0_FINAL/"): item
                for item in archive.getmembers()
                if item.isfile()
            }
            self.assertEqual(set(members), set(checksums))
            for relative, digest in checksums.items():
                member = members[relative]
                path = files[relative]
                self.assertEqual(path.stat().st_size, member.size)
                stream = archive.extractfile(member)
                self.assertIsNotNone(stream)
                self.assertEqual(hashlib.sha256(stream.read()).hexdigest(), digest)
                if not (getattr(path.stat(), "st_flags", 0) & dataless):
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


class TestV5Archive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = None
        if not NESTED:
            cls.temp = tempfile.TemporaryDirectory()
            cls.archive, cls.sidecar = build_archive(
                PROJECT_ROOT, Path(cls.temp.name) / "releases"
            )

    @classmethod
    def tearDownClass(cls):
        if cls.temp:
            cls.temp.cleanup()

    def test_required_files_and_archive_sha256(self):
        if NESTED:
            self.assertEqual(PROJECT_ROOT.name, ARCHIVE_ROOT)
            return
        verify_archive(self.archive)
        digest, filename = self.sidecar.read_text().strip().split("  ", 1)
        self.assertEqual((digest, filename), (sha256_file(self.archive), self.archive.name))
        with zipfile.ZipFile(self.archive) as archive:
            names = set(archive.namelist())
        self.assertTrue({f"{ARCHIVE_ROOT}/{x}" for x in REQUIRED_FILES}.issubset(names))

    def test_extract_runs_validator_and_full_suite(self):
        if NESTED:
            self.assertEqual(PROJECT_ROOT.name, ARCHIVE_ROOT)
            return
        extracted = Path(self.temp.name) / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(self.archive) as archive:
            archive.extractall(extracted)
        project = extracted / ARCHIVE_ROOT
        env = os.environ.copy()
        env["SAFENEST_ARCHIVE_NESTED"] = "1"
        env["PYTHONPYCACHEPREFIX"] = str(Path(self.temp.name) / "pycache")
        env["MPLCONFIGDIR"] = str(Path(self.temp.name) / "mpl")
        validation = subprocess.run(
            [sys.executable, "scripts/validate_v4_config.py"], cwd=project,
            env=env, text=True, capture_output=True, timeout=180, check=False,
        )
        self.assertEqual(validation.returncode, 0, validation.stdout + validation.stderr)
        suite = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
            cwd=project, env=env, text=True, capture_output=True, timeout=300, check=False,
        )
        output = suite.stdout + suite.stderr
        self.assertEqual(suite.returncode, 0, output)
        self.assertRegex(output, r"Ran \d+ tests")


if __name__ == "__main__":
    unittest.main()
