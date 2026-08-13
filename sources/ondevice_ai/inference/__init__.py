# -*- coding: utf-8 -*-
"""Lazy public imports for the SafeNest inference package."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ThermalInterpreter",
    "ThermalPrediction",
    "CO2Interpreter",
    "CO2Prediction",
    "MMWaveInterpreter",
    "MMWavePrediction",
    "ModelRegistry",
]

_EXPORT_MODULES = {
    "ThermalInterpreter": ".thermal_interpreter",
    "ThermalPrediction": ".thermal_interpreter",
    "CO2Interpreter": ".co2_interpreter",
    "CO2Prediction": ".co2_interpreter",
    "MMWaveInterpreter": ".mmwave_interpreter",
    "MMWavePrediction": ".mmwave_interpreter",
    "ModelRegistry": ".model_registry",
}


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_MODULES[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
