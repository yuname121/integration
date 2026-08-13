#!/usr/bin/env python3
"""Read-only Raspberry Pi deployment readiness checks."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys


ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    checks = [
        _check("python_3_10", sys.version_info >= (3, 10), platform.python_version()),
        _check("linux", sys.platform.startswith("linux"), sys.platform, required=False),
        _check("raspberry_pi", "raspberry pi" in _device_model().lower(), _device_model(), required=False),
    ]
    for module in ("fastapi", "uvicorn", "numpy"):
        checks.append(_check(f"module_{module}", importlib.util.find_spec(module) is not None, module))
    runtime_modules = ("ai_edge_litert", "tflite_runtime", "tensorflow")
    available_runtimes = [name for name in runtime_modules if importlib.util.find_spec(name) is not None]
    checks.append(_check("tflite_runtime", bool(available_runtimes), available_runtimes))
    for path in (
        ROOT / "backend" / "run_backend.py",
        ROOT / "database" / "schema.sql",
        ROOT / "web" / "dashboard" / "index.html",
        ROOT / "sources" / "ondevice_ai" / "models" / "model_manifest.json",
    ):
        checks.append(_check(f"file_{path.name}", path.is_file(), str(path)))
    for port in (8000, 9000):
        available, detail = _port_available(port)
        checks.append(_check(f"port_{port}_available", available, detail))
    checks.extend(_model_hash_checks())

    failed = [check for check in checks if check["required"] and not check["passed"]]
    document = {"ok": not failed, "checks": checks}
    print(json.dumps(document, ensure_ascii=False, indent=2))
    return 0 if not failed else 1


def _check(name: str, passed: bool, observed: object, *, required: bool = True) -> dict[str, object]:
    return {"name": name, "passed": bool(passed), "required": required, "observed": observed}


def _device_model() -> str:
    path = Path("/proc/device-tree/model")
    try:
        return path.read_text(encoding="utf-8").rstrip("\x00")
    except OSError:
        return "unavailable"


def _port_available(port: int) -> tuple[bool, str]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("0.0.0.0", port))
        return True, "available"
    except OSError as error:
        return False, f"{type(error).__name__}: {error}"


def _model_hash_checks() -> list[dict[str, object]]:
    model_root = ROOT / "sources" / "ondevice_ai"
    manifest_path = model_root / "models" / "model_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [_check("model_manifest_readable", False, f"{type(error).__name__}: {error}")]
    checks = []
    for sensor_id, model in manifest.get("models", {}).items():
        path = model_root / str(model.get("path", ""))
        expected = str(model.get("sha256", ""))
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            checks.append(_check(f"model_{sensor_id}_sha256", False, f"{type(error).__name__}: {error}"))
            continue
        checks.append(_check(
            f"model_{sensor_id}_sha256",
            bool(expected) and actual == expected,
            {"expected": expected, "actual": actual},
        ))
    return checks


if __name__ == "__main__":
    raise SystemExit(main())
