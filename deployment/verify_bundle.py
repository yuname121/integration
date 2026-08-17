#!/usr/bin/env python3
"""Verify that a SafeNest folder is complete and safe to distribute."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = (
    "README.md",
    "PACKAGE_AND_OPERATION_GUIDE.md",
    "INTEGRATION_PHASE_SUMMARY.md",
    "LATEST_SOURCE_PROVENANCE.json",
    "SOURCE_MANIFEST.md",
    "requirements-backend.txt",
    "backend/run_backend.py",
    "backend/app.py",
    "gateway/protocol.py",
    "gateway/receiver.py",
    "state/manager.py",
    "ai/pipeline.py",
    "risk/engine.py",
    "database/schema.sql",
    "database/repository.py",
    "web/dashboard/index.html",
    "web/dashboard/styles.css",
    "web/dashboard/app.js",
    "sources/display-test2/raspberry_pi_lcd/static/display.html",
    "sources/display-test2/raspberry_pi_lcd/static/common.css",
    "deployment/run_pi.sh",
    "hil/capture.py",
    "docs/PHASE1_REPOSITORY_AUDIT.md",
    "docs/PHASE10_E2E.md",
    "docs/HIL_ACCEPTANCE.md",
    "docs/ON_DEVICE_UPDATE_AUDIT.md",
    "sources/display-test2/esp32_sensor_node/esp32_sensor_node.ino",
    "sources/display-test2/esp32_sensor_node/secrets.example.h",
    "sources/ondevice_ai/requirements-pi.txt",
    "sources/ondevice_ai/models/model_manifest.json",
    "sources/ondevice_ai/AGENTS.md",
)
FORBIDDEN_NAMES = {"secrets.h"}
FORBIDDEN_SUFFIXES = {".db", ".db-wal", ".db-shm", ".pyc"}


def verify(root: Path = ROOT) -> dict[str, object]:
    root = root.resolve()
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    forbidden = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in FORBIDDEN_NAMES or any(path.name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            forbidden.append(str(path.relative_to(root)))
    caches = [str(path.relative_to(root)) for path in root.rglob("__pycache__") if path.is_dir()]
    model_checks = _model_checks(root)
    checks = {
        "required_files_present": not missing,
        "model_hashes_match": bool(model_checks) and all(item["match"] for item in model_checks),
        "no_secrets_or_databases": not forbidden,
        "no_python_caches": not caches,
    }
    return {
        "schema": "safenest.bundle.verification.v1",
        "ok": all(checks.values()),
        "root": str(root),
        "checks": checks,
        "missing": missing,
        "forbidden": forbidden,
        "caches": caches,
        "models": model_checks,
        "file_count": sum(1 for path in root.rglob("*") if path.is_file()),
    }


def _model_checks(root: Path) -> list[dict[str, object]]:
    model_root = root / "sources" / "ondevice_ai"
    manifest_path = model_root / "models" / "model_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    results = []
    for sensor_id, entry in manifest.get("models", {}).items():
        path = model_root / str(entry.get("path", ""))
        expected = str(entry.get("sha256", ""))
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            actual = None
        results.append({
            "sensor_id": sensor_id,
            "path": str(path.relative_to(root)),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": bool(expected) and actual == expected,
        })
    return results


def main() -> int:
    result = verify()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
