#!/usr/bin/env python3
"""SafeNest active integrated node with external-provider injection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import argparse
import math
import signal
import sys
import time

# Support the documented direct entry point:
# ``python3 integrated_node/run_node.py --mode ...``.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from sensors.base_sensor import HardwareBackendUnavailable
from sensors.thermal44.mock_sensor import MockThermalSensor
from sensors.mmwave.mock_sensor import MockMMWaveSensor
from sensors.co2.mock_sensor import MockCO2Sensor
from sensors.pir.mock_sensor import MockPIRSensor
from sensors.provider_contract import (
    EXPECTED_PROVIDER_IDS,
    MissingExternalSensorProvider,
    invalid_provider_result,
    validate_provider_interface,
    validate_provider_result,
)
from integrated_node.runtime_config import (
    NodeRuntimeSettings,
    RuntimeConfigError,
    load_runtime_settings,
    validate_provider_settings,
)
import json
from inference.inference_result import InferenceResult, SafeNestRiskOutput
from risk.risk_engine import SafeNestRiskEngine


def assert_model_deployable(
    model_name: str,
    manifest: dict,
    mode: str,
) -> None:
    models = manifest.get("models", {})
    if model_name not in models:
        return
    model = models[model_name]
    # `real` is the hardware-integration/fail-closed diagnostic mode. It must be
    # able to report missing or broken providers even while a model is blocked.
    # Only `production` asserts the release gate before startup.
    if mode == "production" and not model.get("deployment_allowed", False):
        reason = model.get("block_reason", "UNSPECIFIED")
        raise RuntimeError(
            f"MODEL_RELEASE_BLOCKED: model={model_name}, reason={reason}"
        )


SENSOR_TO_MODEL_KEY = {
    "thermal44": "thermal",
    "mmwave": "mmwave",
    "co2": "co2",
}


class SafeNestIntegratedNode:
    def __init__(
        self,
        mode: str = "mock",
        project_root: str | Path | None = None,
        sensors: Mapping[str, object] | None = None,
        config_path: str | Path = "config/sensors.yaml",
        disabled_sensors: list[str] | set[str] | None = None,
    ):
        if mode not in {"mock", "real", "production"}:
            raise ValueError("mode must be 'mock', 'real', or 'production'")
        self.mode = mode
        self.project_root = (
            Path(project_root).resolve()
            if project_root
            else Path(__file__).resolve().parent.parent
        )
        self.disabled_sensors = set(disabled_sensors or [])
        self.runtime_settings: NodeRuntimeSettings = load_runtime_settings(
            self.project_root, config_path=config_path
        )
        self.manifest = self._load_manifest()
        self._validate_model_deployability()
        self.running = False
        self.backend_errors: dict[str, str] = {}
        self.backend_error_details: dict[str, str] = {}
        self.sensors = self._build_sensors(sensors)
        self._validate_startup_contract()
        self.risk_engine = SafeNestRiskEngine(
            stale_sec=self.runtime_settings.stale_by_sensor
        )

    def _load_manifest(self) -> dict:
        manifest_path = self.project_root / "models" / "model_manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _validate_model_deployability(self) -> None:
        if not self.manifest:
            return
        for sensor_key, model_key in SENSOR_TO_MODEL_KEY.items():
            if sensor_key in self.disabled_sensors:
                continue
            assert_model_deployable(model_key, self.manifest, self.mode)

    def _mock_sensors(self) -> dict[str, object]:
        thermal = self.runtime_settings.sensors["thermal44"]
        mmwave = self.runtime_settings.sensors["mmwave"]
        co2 = self.runtime_settings.sensors["co2"]
        pir = self.runtime_settings.sensors["pir"]
        return {
            "thermal44": MockThermalSensor(
                project_root=self.project_root,
                timeout_sec=thermal.timeout_sec,
                stale_sec=thermal.stale_sec,
                sample_rate_hz=thermal.sample_rate_hz or 10.0,
            ),
            "mmwave": MockMMWaveSensor(
                project_root=self.project_root,
                timeout_sec=mmwave.timeout_sec,
                stale_sec=mmwave.stale_sec,
                sample_rate_hz=mmwave.sample_rate_hz or 10.0,
                window_samples=mmwave.window_samples or 300,
                window_seconds=mmwave.window_seconds or 30.0,
            ),
            "co2": MockCO2Sensor(
                project_root=self.project_root,
                timeout_sec=co2.timeout_sec,
                stale_sec=co2.stale_sec,
                sample_rate_hz=co2.sample_rate_hz or 0.2,
                window_samples=co2.window_samples or 30,
                window_seconds=co2.window_seconds or 150.0,
            ),
            "pir": MockPIRSensor(
                timeout_sec=pir.timeout_sec,
                stale_sec=pir.stale_sec,
            ),
        }

    def _missing_provider(self, key: str) -> MissingExternalSensorProvider:
        settings = self.runtime_settings.sensors[key]
        return MissingExternalSensorProvider(
            EXPECTED_PROVIDER_IDS[key], settings.to_dict()
        )

    def _build_sensors(self, injected: Mapping[str, object] | None):
        supplied = dict(injected or {})
        unknown = sorted(set(supplied) - set(EXPECTED_PROVIDER_IDS))
        if unknown:
            raise ValueError(f"Unknown injected sensor provider keys: {unknown}")
        if self.mode == "mock":
            configured = self._mock_sensors()
            configured.update(supplied)
            return configured
        return {
            key: supplied.get(key, self._missing_provider(key))
            for key in EXPECTED_PROVIDER_IDS
        }

    def _validate_startup_contract(self) -> None:
        if set(self.sensors) != set(EXPECTED_PROVIDER_IDS):
            raise RuntimeConfigError("Runtime must expose exactly four sensor providers")
        for key, provider in self.sensors.items():
            validate_provider_interface(provider, key)
            validate_provider_settings(provider, self.runtime_settings.sensors[key])

    def start(self) -> None:
        self.running = True
        self.backend_errors.clear()
        self.backend_error_details.clear()
        for name, sensor in self.sensors.items():
            settings = self.runtime_settings.sensors[name]
            if name in self.disabled_sensors:
                self.backend_errors[name] = "SENSOR_DISABLED_BY_CLI"
                self.backend_error_details[name] = "Disabled via CLI flag --disable-sensor"
                connected = False
            elif not settings.enabled:
                self.backend_errors[name] = "SENSOR_DISABLED_BY_CONFIG"
                self.backend_error_details[name] = "enabled=false in config/sensors.yaml"
                connected = False
            elif isinstance(sensor, MissingExternalSensorProvider):
                self.backend_errors[name] = "EXTERNAL_SENSOR_PROVIDER_REQUIRED"
                self.backend_error_details[name] = "No external provider was injected"
                connected = False
            else:
                try:
                    connected = sensor.connect()
                    if connected is not True:
                        self.backend_errors[name] = "PROVIDER_CONNECT_FAILED"
                        self.backend_error_details[name] = (
                            f"connect() returned {connected!r}; expected True"
                        )
                        connected = False
                except HardwareBackendUnavailable as exc:
                    self.backend_errors[name] = "PROVIDER_CONNECT_FAILED"
                    self.backend_error_details[name] = str(exc)
                    connected = False
                except Exception as exc:
                    self.backend_errors[name] = "PROVIDER_CONNECT_EXCEPTION"
                    self.backend_error_details[name] = f"{type(exc).__name__}: {exc}"
                    connected = False
            print(f"🔌 [{name.upper()}] Sensor connected: {connected}", file=sys.stderr)

    def _read_provider(self, name: str, sensor: object) -> InferenceResult:
        if name in self.backend_errors:
            code = self.backend_errors[name]
            return invalid_provider_result(
                EXPECTED_PROVIDER_IDS[name],
                code,
                metadata={
                    "mode": self.mode,
                    "detail": self.backend_error_details.get(name, code),
                },
            )
        started = time.perf_counter()
        try:
            value = sensor.read()
        except Exception as exc:
            return invalid_provider_result(
                EXPECTED_PROVIDER_IDS[name],
                "PROVIDER_READ_EXCEPTION",
                metadata={"detail": f"{type(exc).__name__}: {exc}"},
            )
        elapsed_sec = time.perf_counter() - started
        timeout_sec = self.runtime_settings.sensors[name].timeout_sec
        if elapsed_sec > timeout_sec:
            return invalid_provider_result(
                EXPECTED_PROVIDER_IDS[name],
                "PROVIDER_READ_TIMEOUT",
                metadata={"elapsed_sec": elapsed_sec, "timeout_sec": timeout_sec},
            )
        valid, error = validate_provider_result(value, name)
        if not valid:
            return invalid_provider_result(
                EXPECTED_PROVIDER_IDS[name], error or "PROVIDER_RESULT_INVALID"
            )
        return value

    def step(self) -> SafeNestRiskOutput:
        results: dict[str, InferenceResult] = {}
        for name, sensor in self.sensors.items():
            results[name] = self._read_provider(name, sensor)
        output = self.risk_engine.evaluate(results, now=time.time())
        output.metadata["runtime_config"] = self.runtime_settings.to_metadata()
        output.metadata["mode"] = self.mode
        if self.disabled_sensors:
            output.system_health = "DEGRADED"
            output.metadata["disabled_sensors"] = sorted(self.disabled_sensors)
            output.metadata["deployment_status"] = "DEVELOPMENT_ONLY"
        return output

    def run_loop(self, interval_sec: float | None = None) -> None:
        configured = self.runtime_settings.loop_interval_sec
        interval = configured if interval_sec is None else float(interval_sec)
        if not math.isclose(interval, configured, abs_tol=1e-9):
            raise RuntimeConfigError(
                "Runtime loop interval differs from config/sensors.yaml: "
                f"config={configured}, runtime={interval}"
            )
        self.start()
        print(
            f"🚀 SafeNest V5 On-Device AI Node Running [Mode: {self.mode.upper()}]",
            file=sys.stderr,
        )
        try:
            while self.running:
                sys.stdout.write(self.step().to_json() + "\n")
                sys.stdout.flush()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user. Shutting down...", file=sys.stderr)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.running = False
        for name, sensor in self.sensors.items():
            try:
                sensor.close()
                print(f"🔒 [{name.upper()}] Sensor closed safely.", file=sys.stderr)
            except Exception as exc:
                print(f"⚠️ Error closing {name}: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="SafeNest V5 On-Device AI Node")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--interval", type=float, default=None)
    parser.add_argument(
        "--disable-sensor",
        action="append",
        default=[],
        help="Disable specific sensor and bypass deployment block check for development",
    )
    args = parser.parse_args()
    node = SafeNestIntegratedNode(
        mode=args.mode, disabled_sensors=args.disable_sensor
    )

    def signal_handler(sig, frame):
        print(f"\nReceived signal {sig}. Stopping node...", file=sys.stderr)
        node.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    node.run_loop(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
