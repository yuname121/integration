"""Derived roadmap-0.8.6 runtime capability status for existing API responses.

This module deliberately stores nothing. It projects the existing latest sensor
state and AI result documents into stable, machine-readable capability fields.
It does not select models, invoke inference, or change risk behavior.
"""

from __future__ import annotations

from typing import Any, Mapping


SENSOR_IDS = ("mmwave", "thermal", "co2", "pir")


def runtime_status_document(
    state: Mapping[str, Any],
    ai_results: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a non-persistent capability projection for the current publication."""

    sensors = _mapping(state.get("sensors"))
    projected = {
        "co2": _co2_status(_mapping(sensors.get("co2")), _mapping(ai_results.get("co2"))),
        "thermal": _thermal_status(
            _mapping(sensors.get("thermal")), _mapping(ai_results.get("thermal"))
        ),
        "mmwave": _mmwave_status(
            _mapping(sensors.get("mmwave")), _mapping(ai_results.get("mmwave"))
        ),
        "pir": _pir_status(_mapping(sensors.get("pir"))),
    }
    global_status = _global_status(projected)
    return {
        "status": global_status,
        "sensors": projected,
        "limitations": [
            sensor_id
            for sensor_id, entry in projected.items()
            if entry["ai_status"] in {"BLOCKED", "MODEL_PENDING", "UNAVAILABLE"}
        ],
    }


def _co2_status(sensor: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    status = _sensor_base(sensor)
    status["artifact_status"] = "PRESENT"
    if status["sensor_status"] != "AVAILABLE":
        return _blocked_for_sensor(status)

    if bool(result.get("available")):
        status.update(
            {
                "input_contract_status": "SATISFIED",
                "ai_status": "ACTIVE",
                "blocked_reason": None,
                "output_status": "AVAILABLE",
            }
        )
        return status

    error = str(result.get("error") or "")
    if error == "INPUT_UNAVAILABLE":
        status.update(
            {
                "input_contract_status": "UNSATISFIED",
                "ai_status": "BLOCKED",
                "blocked_reason": "INPUT_CONTRACT_UNSATISFIED",
                "output_status": "NOT_AVAILABLE",
            }
        )
    elif error == "WINDOW_WARMING_UP":
        status.update(
            {
                "input_contract_status": "WARMING_UP",
                "ai_status": "BLOCKED",
                "blocked_reason": "INPUT_WARMUP",
                "output_status": "NOT_AVAILABLE",
            }
        )
    else:
        status.update(
            {
                "input_contract_status": "UNKNOWN",
                "ai_status": "UNAVAILABLE",
                "blocked_reason": error or "MODEL_RUNTIME_UNAVAILABLE",
                "output_status": "NOT_AVAILABLE",
            }
        )
    return status


def _thermal_status(sensor: Mapping[str, Any], _result: Mapping[str, Any]) -> dict[str, Any]:
    status = _sensor_base(sensor)
    # O2/O2.5/O2.6 establish that the selected T-B5 artifact exists and that
    # raw-frame conversion can be replayed. O3/production activation remains
    # forbidden while the INT8/domain review is unresolved.
    status["artifact_status"] = "PRESENT"
    if status["sensor_status"] != "AVAILABLE":
        return _blocked_for_sensor(status)
    status.update(
        {
            "input_contract_status": "VALIDATED_WITH_LIMITATIONS",
            "ai_status": "BLOCKED",
            "blocked_reason": "INT8_QUANTIZATION_REVIEW_REQUIRED",
            "output_status": "NOT_REQUESTED",
        }
    )
    return status


def _mmwave_status(sensor: Mapping[str, Any], _result: Mapping[str, Any]) -> dict[str, Any]:
    status = _sensor_base(sensor)
    status.update(
        {
            "artifact_status": "PRESENT_HISTORICAL_ONLY",
            "input_contract_status": "SIGNAL_CONTRACT_MISMATCH",
            "ai_status": "MODEL_PENDING",
            "blocked_reason": "MR60_NATIVE_MODEL_PENDING",
            "output_status": "NOT_AVAILABLE",
        }
    )
    return status


def _pir_status(sensor: Mapping[str, Any]) -> dict[str, Any]:
    status = _sensor_base(sensor)
    motion = bool(_mapping(sensor.get("values")).get("motion"))
    status.update(
        {
            "artifact_status": "NOT_APPLICABLE",
            "input_contract_status": "NOT_APPLICABLE",
            "ai_status": "NOT_APPLICABLE",
            "blocked_reason": None,
            "output_status": "NOT_APPLICABLE",
            "sensor_value_status": "MOTION" if motion else "NO_MOTION",
        }
    )
    return status


def _sensor_base(sensor: Mapping[str, Any]) -> dict[str, Any]:
    raw_status = str(sensor.get("status") or "NO_DATA")
    sensor_status = {
        "LIVE": "AVAILABLE",
        "STALE": "STALE",
        "INVALID": "INVALID",
    }.get(raw_status, "UNAVAILABLE")
    connectivity = (
        "DISCONNECTED"
        if raw_status == "DISCONNECTED"
        else "CONNECTED"
        if raw_status in {"LIVE", "STALE", "INVALID"} or bool(sensor.get("connected"))
        else "UNKNOWN"
    )
    freshness = {
        "LIVE": "CURRENT",
        "STALE": "STALE",
        "INVALID": "INVALID",
        "DISCONNECTED": "STALE" if bool(sensor.get("stale")) else "UNKNOWN",
    }.get(raw_status, "UNKNOWN")
    return {
        "sensor_status": sensor_status,
        "sensor_connectivity": connectivity,
        "data_freshness": freshness,
        "artifact_status": "UNKNOWN",
        "input_contract_status": "NOT_EVALUATED",
        "ai_status": "NOT_EVALUATED",
        "blocked_reason": None,
        "output_status": "NOT_AVAILABLE",
    }


def _blocked_for_sensor(status: dict[str, Any]) -> dict[str, Any]:
    reason = {
        "STALE": "SENSOR_STALE",
        "INVALID": "SENSOR_INVALID",
        "UNAVAILABLE": "SENSOR_UNAVAILABLE",
    }.get(status["sensor_status"], "SENSOR_UNAVAILABLE")
    status.update(
        {
            "input_contract_status": "NOT_EVALUATED",
            "ai_status": "BLOCKED",
            "blocked_reason": reason,
            "output_status": "NOT_AVAILABLE",
        }
    )
    return status


def _global_status(projected: Mapping[str, Mapping[str, Any]]) -> str:
    sensor_states = [entry["sensor_status"] for entry in projected.values()]
    available_count = sensor_states.count("AVAILABLE")
    if available_count == 0:
        return "NOT_READY"
    if any(value != "AVAILABLE" for value in sensor_states):
        return "DEGRADED"
    if any(
        entry["ai_status"] in {"BLOCKED", "MODEL_PENDING", "UNAVAILABLE"}
        for entry in projected.values()
    ):
        return "READY_WITH_LIMITATIONS"
    return "READY"


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
