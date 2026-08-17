#!/usr/bin/env python3
"""Deployment readiness checks.

Default mode is the existing Pi-start preflight used by deployment/run_pi.sh.
`--offline-preflight` is the Stage 7 Mac-offline structural validator. It does
not bind ports, does not require Raspberry Pi, and does not treat a Mac PASS as
a Pi PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import socket
import sys


ROOT = Path(__file__).resolve().parent.parent

REQUIRED_RUNTIME_FILES = (
    "backend/run_backend.py",
    "backend/app.py",
    "backend/runtime.py",
    "backend/runtime_status.py",
    "backend/views.py",
    "deployment/run_pi.sh",
    "database/schema.sql",
    "gateway/receiver.py",
    "gateway/thermal_udp.py",
    "state/manager.py",
    "ai/pipeline.py",
    "ai/runtime.py",
    "risk/engine.py",
    "web/dashboard/index.html",
    "web/dashboard/app.js",
    "web/dashboard/styles.css",
    "sources/display-test2/raspberry_pi_lcd/static/display.html",
    "sources/display-test2/raspberry_pi_lcd/static/common.css",
    "sources/ondevice_ai/models/model_manifest.json",
    ".env.example",
)

EXPECTED_ENV_KEYS = (
    "SMS_ACCESS_KEY",
    "SMS_SECRET_KEY",
    "SMS_SERVICE_ID",
    "SMS_FROM_NUMBER",
    "SMS_API_BASE_URL",
    "SMS_TIMEOUT_SECONDS",
    "MANAGER_PHONE_NUMBER",
    "MANAGER_NAME",
    "SAFENEST_SMS_COOLDOWN_SECONDS",
    "SAFENEST_GPIO_MODE",
    "SAFENEST_BUZZER_GPIO_PIN",
    "SAFENEST_BUZZER_FREQUENCY_HZ",
    "SAFENEST_CO2_UPDATE_INTERVAL_SECONDS",
    "SAFENEST_SENSOR_DATA_ENABLED",
    "SAFENEST_SENSOR_DATA_ROOT",
    "SAFENEST_SENSOR_DATA_MAX_GB",
    "SAFENEST_MIN_FREE_DISK_GB",
    "SAFENEST_MMWAVE_DATA_MAX_GB",
    "SAFENEST_CO2_DATA_MAX_GB",
    "SAFENEST_THERMAL_DATA_MAX_GB",
    "SAFENEST_SENSOR_DATA_QUEUE_CAPACITY",
    "SAFENEST_THERMAL_BATCH_FRAMES",
    "SAFENEST_THERMAL_FLUSH_SECONDS",
    "SAFENEST_SENSOR_CLEANUP_INTERVAL_SECONDS",
    "SAFENEST_THERMAL_UDP_HOST",
    "SAFENEST_THERMAL_UDP_PORT",
    "SAFENEST_THERMAL_UDP_FRAME_TIMEOUT_SECONDS",
    "SAFENEST_THERMAL_UDP_MAX_PENDING_FRAMES",
    "SAFENEST_VENV_PATH",
)

FORBIDDEN_THERMAL_SELECTORS = (
    "SMALL_CNN_BASELINE",
    "T-B5",
    "T_B5",
    "full_int8.tflite",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SafeNest deployment readiness checks")
    parser.add_argument(
        "--offline-preflight",
        action="store_true",
        help="Stage 7 Mac-offline structural checks. Does not validate Pi or sensors.",
    )
    args = parser.parse_args(argv)
    document = offline_preflight_document() if args.offline_preflight else pi_start_document()
    print(json.dumps(document, ensure_ascii=False, indent=2))
    return 0 if document["ok"] else 1


def pi_start_document(root: Path = ROOT) -> dict[str, object]:
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
        root / "backend" / "run_backend.py",
        root / "database" / "schema.sql",
        root / "web" / "dashboard" / "index.html",
        root / "sources" / "ondevice_ai" / "models" / "model_manifest.json",
    ):
        checks.append(_check(f"file_{path.name}", path.is_file(), str(path)))
    for port in (8000, 9000):
        available, detail = _port_available(port)
        checks.append(_check(f"port_{port}_available", available, detail))
    checks.extend(_model_hash_checks(root))
    failed = [check for check in checks if check["required"] and not check["passed"]]
    return {"ok": not failed, "mode": "PI_START_PREFLIGHT", "checks": checks}


def offline_preflight_document(root: Path = ROOT) -> dict[str, object]:
    checks = [
        _check("python_version_reported", True, platform.python_version()),
        _check(
            "python_3_10_for_runtime_import",
            sys.version_info >= (3, 10),
            platform.python_version(),
            required=False,
        ),
    ]
    checks.extend(_structure_checks(root))
    checks.extend(_config_checks(root))
    checks.extend(_listener_contract_checks(root))
    checks.extend(_path_safety_checks(root))
    checks.extend(_model_hash_checks(root))
    checks.extend(_artifact_selection_checks(root))
    checks.extend(_status_contract_checks(root))
    checks.extend(_o4_asset_checks(root))
    checks.extend(_writable_directory_checks(root))
    checks.append(_runtime_import_check())
    checks.append(_check("pi_checks", True, "NOT_RUN", required=False))
    checks.append(_check("sensor_checks", True, "NOT_RUN", required=False))
    failed = [check for check in checks if check["required"] and not check["passed"]]
    return {
        "schema": "safenest.stage7.offline_preflight.v1",
        "ok": not failed,
        "mode": "MAC_OFFLINE_PREFLIGHT",
        "python": platform.python_version(),
        "pi_checks": "NOT_RUN",
        "sensor_checks": "NOT_RUN",
        "mac_pass_does_not_imply_pi_pass": True,
        "checks": checks,
    }


def _structure_checks(root: Path) -> list[dict[str, object]]:
    checks = []
    for relative in REQUIRED_RUNTIME_FILES:
        path = root / relative
        checks.append(_check(f"file_{relative.replace('/', '_')}", path.is_file(), str(path)))
    return checks


def _config_checks(root: Path) -> list[dict[str, object]]:
    example = root / ".env.example"
    text = example.read_text(encoding="utf-8") if example.is_file() else ""
    present = set(re.findall(r"^([A-Z][A-Z0-9_]+)=", text, flags=re.MULTILINE))
    missing = [key for key in EXPECTED_ENV_KEYS if key not in present]
    return [
        _check(
            "env_example_documents_runtime_keys",
            not missing,
            {"missing": missing, "documented": sorted(present)},
        )
    ]


def _listener_contract_checks(root: Path) -> list[dict[str, object]]:
    backend = (root / "backend" / "run_backend.py").read_text(encoding="utf-8")
    runtime = (root / "backend" / "runtime.py").read_text(encoding="utf-8")
    return [
        _check("http_default_port_8000", 'default=8000' in backend or "api-port" in backend and "8000" in backend, "8000"),
        _check("tcp_default_port_9000", "sensor_port: int = 9000" in runtime and "default=9000" in backend, "9000"),
        _check(
            "udp_default_port_5005",
            'SAFENEST_THERMAL_UDP_PORT", "5005"' in backend and "thermal_udp_port: int = 5005" in runtime,
            "5005",
        ),
        _check("tcp_default_bind_all", 'default="0.0.0.0"' in backend and 'sensor_host: str = "0.0.0.0"' in runtime, "0.0.0.0"),
        _check("udp_default_bind_all", 'SAFENEST_THERMAL_UDP_HOST", "0.0.0.0"' in backend, "0.0.0.0"),
    ]


def _path_safety_checks(root: Path) -> list[dict[str, object]]:
    scanned = (
        "backend/run_backend.py",
        "backend/runtime.py",
        "backend/app.py",
        "deployment/run_pi.sh",
    )
    hits = []
    for relative in scanned:
        text = (root / relative).read_text(encoding="utf-8")
        for match in re.finditer(r"/Users/[^\s\"']+", text):
            hits.append(f"{relative}:{match.group(0)}")
    db_default = 'parent.parent / "data" / "safenest.db"' in (root / "backend" / "run_backend.py").read_text(encoding="utf-8")
    venv_default = "${REPOSITORY_ROOT}/.venv" in (root / "deployment" / "run_pi.sh").read_text(encoding="utf-8")
    return [
        _check("no_developer_absolute_runtime_paths", not hits, hits),
        _check("db_path_repository_relative", db_default, "data/safenest.db"),
        _check("venv_path_repository_relative", venv_default, "<repo>/.venv"),
    ]


def _artifact_selection_checks(root: Path) -> list[dict[str, object]]:
    manifest_path = root / "sources" / "ondevice_ai" / "models" / "model_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [_check("artifact_manifest_readable", False, f"{type(error).__name__}: {error}")]
    models = manifest.get("models") if isinstance(manifest.get("models"), dict) else {}
    thermal = models.get("thermal") if isinstance(models.get("thermal"), dict) else {}
    mmwave = models.get("mmwave") if isinstance(models.get("mmwave"), dict) else {}
    thermal_path = str(thermal.get("path") or "")
    forbidden = [token for token in FORBIDDEN_THERMAL_SELECTORS if token.lower() in thermal_path.lower()]
    return [
        _check(
            "thermal_production_path_is_historical_v0_1_0",
            thermal_path.endswith("thermal_fall_int8_v0.1.0.tflite") and not forbidden,
            {"path": thermal_path, "forbidden_hits": forbidden},
        ),
        _check(
            "mmwave_primary_deployment_blocked",
            mmwave.get("deployment_allowed") is False,
            {
                "deployment_allowed": mmwave.get("deployment_allowed"),
                "block_reason": mmwave.get("block_reason"),
            },
        ),
        _check(
            "runtime_loads_manifest_sensor_keys_only",
            'self._assert_deployment_allowed()' in (root / "ai" / "runtime.py").read_text(encoding="utf-8"),
            "ai/runtime.py LazyModel",
        ),
    ]


def _status_contract_checks(root: Path) -> list[dict[str, object]]:
    status = (root / "backend" / "runtime_status.py").read_text(encoding="utf-8")
    views = (root / "backend" / "views.py").read_text(encoding="utf-8")
    return [
        _check("runtime_status_module_present", "def runtime_status_document" in status, "runtime_status_document"),
        _check("ready_with_limitations_preserved", "READY_WITH_LIMITATIONS" in status, "READY_WITH_LIMITATIONS"),
        _check("thermal_ai_blocked_reason_preserved", "INT8_QUANTIZATION_REVIEW_REQUIRED" in status, "INT8_QUANTIZATION_REVIEW_REQUIRED"),
        _check("pir_not_applicable_preserved", 'ai_status": "NOT_APPLICABLE"' in status, "NOT_APPLICABLE"),
        _check("status_api_exposes_runtime_status", '"runtime_status": copy.deepcopy(runtime_status)' in views, "/api/status"),
        _check("lcd_state_reuses_runtime_status", '"runtime_status": copy.deepcopy(status["runtime_status"])' in views, "/api/state"),
    ]


def _o4_asset_checks(root: Path) -> list[dict[str, object]]:
    html = (root / "web" / "dashboard" / "index.html").read_text(encoding="utf-8")
    js = (root / "web" / "dashboard" / "app.js").read_text(encoding="utf-8")
    lcd = (root / "sources" / "display-test2" / "raspberry_pi_lcd" / "static" / "display.html").read_text(encoding="utf-8")
    return [
        _check("dashboard_runtime_badge", 'id="runtimeBadge"' in html, "runtimeBadge"),
        _check("dashboard_thermal_sensor_ai_split", 'id="thermalSensor"' in html and 'id="thermalAiStatus"' in html, "thermal split"),
        _check("dashboard_consumes_backend_runtime_status", "payload.runtime_status?.status" in js, "backend authority"),
        _check("lcd_consumes_backend_runtime_status", "runtimeDocument(payload).status" in lcd, "LCD /api/state"),
    ]


def _writable_directory_checks(root: Path) -> list[dict[str, object]]:
    checks = []
    for relative in ("data", "data/co2", "data/mmwave", "data/thermal"):
        path = root / relative
        checks.append(_check(f"runtime_dir_{relative.replace('/', '_')}", path.is_dir(), str(path)))
    return checks


def _runtime_import_check() -> dict[str, object]:
    if sys.version_info < (3, 10):
        return _check(
            "runtime_import_construct",
            True,
            f"SKIPPED_PYTHON_LT_3_10:{platform.python_version()}",
            required=False,
        )
    try:
        observed, passed = _construct_offline_runtime()
        return _check("runtime_import_construct", passed, observed)
    except Exception as error:
        return _check(
            "runtime_import_construct",
            False,
            f"{type(error).__name__}: {error}",
        )


def _construct_offline_runtime() -> tuple[dict[str, object], bool]:
    from backend.runtime import SafeNestRuntime
    from backend.store import RuntimeStore
    from backend.views import status_document

    store = RuntimeStore()
    runtime = SafeNestRuntime(store=store, sensor_port=0, thermal_udp_port=0)
    publication = runtime.evaluate_once()
    document = status_document(publication)
    observed = {
        "runtime_status": document.get("runtime_status", {}).get("status"),
        "thermal_ai": document.get("thermal", {}).get("runtime_status", {}).get("ai_status"),
    }
    passed = observed["thermal_ai"] == "BLOCKED" and observed["runtime_status"] in {
        "NOT_READY",
        "DEGRADED",
        "READY_WITH_LIMITATIONS",
    }
    return observed, passed


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


def _model_hash_checks(root: Path = ROOT) -> list[dict[str, object]]:
    model_root = root / "sources" / "ondevice_ai"
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
