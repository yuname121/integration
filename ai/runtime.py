"""Lazy, failure-isolated loading of the frozen TFLite adapters."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import threading
from typing import Callable


VENDOR_ROOT = Path(__file__).resolve().parent.parent / "sources" / "ondevice_ai"


class ModelRuntimeUnavailable(RuntimeError):
    """The model adapter or its TFLite runtime could not be loaded."""


class LazyModel:
    """Load a frozen interpreter only when a complete input first arrives."""

    _ADAPTERS = {
        "thermal": ("thermal_interpreter.py", "ThermalInterpreter"),
        "mmwave": ("mmwave_interpreter.py", "MMWaveInterpreter"),
        "co2": ("co2_interpreter.py", "CO2Interpreter"),
    }

    def __init__(self, sensor_id: str, factory: Callable[[], object] | None = None) -> None:
        if sensor_id not in self._ADAPTERS:
            raise ValueError(f"unknown model sensor: {sensor_id}")
        self.sensor_id = sensor_id
        self._factory = factory
        self._instance: object | None = None
        self._load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def predict(self, *args: object) -> object:
        instance = self._load()
        try:
            return instance.predict(*args)
        except Exception as error:
            raise ModelRuntimeUnavailable(
                f"{self.sensor_id} inference failed: {type(error).__name__}: {error}"
            ) from error

    def _load(self) -> object:
        if self._instance is not None:
            return self._instance
        if self._load_error is not None:
            raise ModelRuntimeUnavailable(self._load_error)
        with self._lock:
            if self._instance is not None:
                return self._instance
            if self._load_error is not None:
                raise ModelRuntimeUnavailable(self._load_error)
            try:
                self._instance = self._factory() if self._factory else self._load_frozen_adapter()
            except Exception as error:
                self._load_error = (
                    f"{self.sensor_id} model unavailable: {type(error).__name__}: {error}"
                )
                raise ModelRuntimeUnavailable(self._load_error) from error
            return self._instance

    def _load_frozen_adapter(self) -> object:
        self._assert_deployment_allowed()
        filename, class_name = self._ADAPTERS[self.sensor_id]
        adapter_path = VENDOR_ROOT / "inference" / filename
        module_name = f"_safenest_frozen_{self.sensor_id}_interpreter"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, adapter_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load adapter {adapter_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
        adapter_class = getattr(module, class_name)
        return adapter_class(project_root=VENDOR_ROOT)

    def _assert_deployment_allowed(self) -> None:
        manifest_path = VENDOR_ROOT / "models" / "model_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = manifest.get("models", {}).get(self.sensor_id)
        if not isinstance(metadata, dict):
            raise ModelRuntimeUnavailable(
                f"MODEL_MANIFEST_ENTRY_MISSING: sensor={self.sensor_id}"
            )
        if metadata.get("deployment_allowed") is False:
            reason = metadata.get("block_reason", "UNSPECIFIED")
            raise ModelRuntimeUnavailable(
                f"MODEL_RELEASE_BLOCKED: sensor={self.sensor_id}, reason={reason}"
            )
