"""Pure Stage 9 smoke evaluation. No network, sockets, or sleeps."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


PROBE_STATUSES = ("PASS", "FAIL", "NOT_RUN", "NOT_APPLICABLE", "NOT_OBSERVABLE")
OVERALL_RESULTS = ("PASS", "PASS_WITH_LIMITATIONS", "FAIL", "NOT_RUN")
KNOWN_GLOBAL_STATUSES = {"READY", "READY_WITH_LIMITATIONS", "DEGRADED", "NOT_READY"}
KNOWN_AI_STATUSES = {"ACTIVE", "BLOCKED", "MODEL_PENDING", "NOT_APPLICABLE", "UNAVAILABLE"}
EXPECTED_AI = {
    "thermal": "BLOCKED",
    "pir": "NOT_APPLICABLE",
    "mmwave": "MODEL_PENDING",
}
CO2_ALLOWED_AI = {"ACTIVE", "BLOCKED", "UNAVAILABLE"}
REQUIRED_PROBES = (
    "backend_health",
    "tcp_9000",
    "udp_5005",
    "co2_progress",
    "thermal_progress",
    "mmwave_progress",
    "pir_progress",
    "runtime_status",
)
LIMITATION_PROBES = ("esp_connection", "logger_drops")
PROGRESS_KEYS = {
    "co2": ("values.measurement_event_count", "values.measurement_event_id", "last_received_at"),
    "thermal": ("values.frame_sequence", "sequence", "last_received_at"),
    "mmwave": ("sequence", "last_received_at"),
    "pir": ("sequence", "last_received_at", "values.event_id"),
}


def evaluate_observation(
    observation: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    """Evaluate one before/after observation envelope."""

    probes = {
        "backend_health": evaluate_backend_health(observation),
        "tcp_9000": evaluate_tcp_9000(observation),
        "udp_5005": evaluate_udp_5005(observation),
        "esp_connection": evaluate_esp_connection(observation),
        "co2_progress": evaluate_sensor_progress(observation, "co2"),
        "thermal_progress": evaluate_sensor_progress(observation, "thermal"),
        "mmwave_progress": evaluate_sensor_progress(observation, "mmwave"),
        "pir_progress": evaluate_sensor_progress(observation, "pir"),
        "runtime_status": evaluate_runtime_status(observation),
        "logger_drops": evaluate_logger_drops(observation),
    }
    result = overall_result(probes, mode=mode)
    return {"probes": probes, "result": result}


def evaluate_backend_health(observation: Mapping[str, Any]) -> dict[str, Any]:
    error = observation.get("health_error_after")
    health_after = observation.get("health_after")
    if error and not isinstance(health_after, Mapping):
        reason = str(error)
        if "invalid" in reason.lower():
            label = "HTTP reachable but invalid response"
        else:
            label = "HTTP unreachable"
        return _probe("backend_health", "FAIL", reason, "health.ok is true", label)
    health = _mapping(health_after) or (
        {} if error else _mapping(observation.get("health_before"))
    )
    if not health:
        return _probe("backend_health", "FAIL", None, "health.ok is true", "HTTP unreachable")
    if health.get("ok") is not True:
        return _probe(
            "backend_health",
            "FAIL",
            {"ok": health.get("ok"), "ready": health.get("ready")},
            "health.ok is true",
            "backend unhealthy",
        )
    status = _mapping(observation.get("status_after")) or _mapping(observation.get("status_before"))
    runtime = str(_mapping(status.get("runtime_status")).get("status") or "UNKNOWN")
    return _probe(
        "backend_health",
        "PASS",
        {
            "ok": True,
            "ready": health.get("ready"),
            "runtime_status": runtime,
        },
        "health.ok is true",
        f"backend healthy; runtime {runtime}",
    )


def evaluate_tcp_9000(observation: Mapping[str, Any]) -> dict[str, Any]:
    return _evaluate_listener(observation, "tcp", 9000, "tcp_9000")


def evaluate_udp_5005(observation: Mapping[str, Any]) -> dict[str, Any]:
    return _evaluate_listener(observation, "udp", 5005, "udp_5005")


def evaluate_esp_connection(observation: Mapping[str, Any]) -> dict[str, Any]:
    health_after = observation.get("health_after")
    if observation.get("health_error_after") and not isinstance(health_after, Mapping):
        return _probe(
            "esp_connection",
            "NOT_OBSERVABLE",
            observation.get("health_error_after"),
            "CONNECTED",
            "ESP connection was not observed because /health was unreachable",
        )
    health = _mapping(health_after) or _mapping(observation.get("health_before"))
    receiver = _mapping(health.get("receiver"))
    if "connections" not in receiver or "disconnects" not in receiver:
        return _probe(
            "esp_connection",
            "NOT_OBSERVABLE",
            {"receiver_keys": sorted(receiver)},
            "connections > disconnects",
            "ESP connection counters are not exposed",
        )
    connections = receiver.get("connections")
    disconnects = receiver.get("disconnects")
    if not _number(connections) or not _number(disconnects):
        return _probe(
            "esp_connection",
            "NOT_OBSERVABLE",
            {"connections": connections, "disconnects": disconnects},
            "connections > disconnects",
            "ESP connection counters are not numeric",
        )
    connected = int(connections) > int(disconnects)
    status_doc = _mapping(observation.get("status_after")) or _mapping(observation.get("status_before"))
    sensor_connectivity = {
        sensor_id: _sensor_runtime(status_doc, sensor_id).get("sensor_connectivity")
        for sensor_id in ("co2", "mmwave", "pir")
    }
    observed = {
        "state": "CONNECTED" if connected else "DISCONNECTED",
        "connections": int(connections),
        "disconnects": int(disconnects),
        "sensor_connectivity": sensor_connectivity,
    }
    if connected:
        return _probe("esp_connection", "PASS", observed, "CONNECTED", "ESP TCP session is open")
    return _probe("esp_connection", "FAIL", observed, "CONNECTED", "ESP disconnected")


def evaluate_sensor_progress(observation: Mapping[str, Any], sensor_id: str) -> dict[str, Any]:
    before = _sensor_state(observation.get("status_before"), sensor_id)
    after = _sensor_state(observation.get("status_after"), sensor_id)
    name = f"{sensor_id}_progress"
    if not before or not after:
        error = observation.get("status_error_after") or observation.get("status_error_before")
        return _probe(
            name,
            "FAIL",
            error or {"before": bool(before), "after": bool(after)},
            f"{sensor_id} identity advanced",
            f"{sensor_id} status was not observed",
        )
    advanced, observed = _identity_advanced(before, after, PROGRESS_KEYS[sensor_id])
    expected = f"{sensor_id} identity/timestamp/sequence advanced; value change is not required"
    if sensor_id == "pir":
        value_status = _sensor_runtime(observation.get("status_after"), "pir").get("sensor_value_status")
        observed["sensor_value_status"] = value_status
        observed["no_motion_accepted"] = value_status in {"MOTION", "NO_MOTION"}
    if advanced:
        reason = f"{sensor_id} progress observed"
        if sensor_id == "co2":
            reason = "CO2 progress observed without requiring ppm change"
        return _probe(name, "PASS", observed, expected, reason)
    return _probe(name, "FAIL", observed, expected, f"{sensor_id} stream stalled")


def evaluate_runtime_status(observation: Mapping[str, Any]) -> dict[str, Any]:
    status_doc = _mapping(observation.get("status_after")) or _mapping(observation.get("status_before"))
    runtime = _mapping(status_doc.get("runtime_status"))
    if not runtime:
        return _probe(
            "runtime_status",
            "FAIL",
            observation.get("status_error_after") or None,
            EXPECTED_AI,
            "runtime_status was not observed",
        )
    sensors = _mapping(runtime.get("sensors"))
    observed = {
        "status": runtime.get("status"),
        "sensors": {
            sensor_id: {
                "sensor_status": _mapping(sensors.get(sensor_id)).get("sensor_status"),
                "ai_status": _mapping(sensors.get(sensor_id)).get("ai_status"),
                "blocked_reason": _mapping(sensors.get(sensor_id)).get("blocked_reason"),
                "sensor_value_status": _mapping(sensors.get(sensor_id)).get("sensor_value_status"),
            }
            for sensor_id in ("co2", "thermal", "mmwave", "pir")
        },
    }
    failures = []
    global_status = runtime.get("status")
    if global_status not in KNOWN_GLOBAL_STATUSES:
        failures.append(f"unknown global runtime status {global_status!r}")
    for sensor_id, expected in EXPECTED_AI.items():
        ai_status = observed["sensors"][sensor_id]["ai_status"]
        if ai_status not in KNOWN_AI_STATUSES:
            failures.append(f"unknown {sensor_id} AI status {ai_status!r}")
        elif ai_status != expected:
            failures.append(f"{sensor_id} AI {ai_status!r} != {expected}")
    co2_ai = observed["sensors"]["co2"]["ai_status"]
    if co2_ai not in KNOWN_AI_STATUSES:
        failures.append(f"unknown co2 AI status {co2_ai!r}")
    elif co2_ai not in CO2_ALLOWED_AI:
        failures.append(f"co2 AI {co2_ai!r} is not an allowed smoke status")
    pir_value = observed["sensors"]["pir"]["sensor_value_status"]
    if pir_value not in {"MOTION", "NO_MOTION", None}:
        failures.append(f"unknown PIR value status {pir_value!r}")
    thermal_sensor = observed["sensors"]["thermal"]["sensor_status"]
    thermal_ai = observed["sensors"]["thermal"]["ai_status"]
    if thermal_sensor == "AVAILABLE" and thermal_ai == "BLOCKED":
        observed["thermal_sensor_available_ai_blocked"] = True
    if failures:
        return _probe("runtime_status", "FAIL", observed, EXPECTED_AI, "; ".join(failures))
    return _probe(
        "runtime_status",
        "PASS",
        observed,
        EXPECTED_AI,
        "partial-availability contract holds; READY_WITH_LIMITATIONS is accepted",
    )


def evaluate_logger_drops(observation: Mapping[str, Any]) -> dict[str, Any]:
    before = _dropped_total(observation.get("health_before"))
    after = _dropped_total(observation.get("health_after"))
    if before is None or after is None:
        return _probe(
            "logger_drops",
            "NOT_OBSERVABLE",
            {"before": before, "after": after},
            "after - before == 0",
            "logger drop counters are not exposed on /health",
        )
    delta = after - before
    observed = {"before": before, "after": after, "new_drops": delta}
    if delta > 0:
        return _probe("logger_drops", "FAIL", observed, "new_drops == 0", "new unexpected logger-drop condition")
    return _probe(
        "logger_drops",
        "PASS",
        observed,
        "new_drops == 0",
        "no new logger drops; historical lifetime count is not a failure",
    )


def overall_result(probes: Mapping[str, Mapping[str, Any]], *, mode: str) -> str:
    if mode == "PLAN":
        return "NOT_RUN"
    required_statuses = [str(probes[name]["status"]) for name in REQUIRED_PROBES if name in probes]
    if mode == "LIVE" and any(status == "NOT_RUN" for status in required_statuses):
        return "FAIL"
    if any(str(probes[name]["status"]) == "FAIL" for name in REQUIRED_PROBES if name in probes):
        return "FAIL"
    if any(str(probes[name]["status"]) == "FAIL" for name in LIMITATION_PROBES if name in probes):
        return "FAIL"
    limitation = any(
        str(probes[name]["status"]) == "NOT_OBSERVABLE"
        for name in (*REQUIRED_PROBES, *LIMITATION_PROBES)
        if name in probes
    )
    if limitation:
        return "PASS_WITH_LIMITATIONS"
    return "PASS"


def _evaluate_listener(observation: Mapping[str, Any], protocol: str, port: int, name: str) -> dict[str, Any]:
    from hil.stage9_sockets import listener_present, parse_listen_ports

    error = observation.get("socket_error")
    text = observation.get("socket_table")
    if error and not text:
        return _probe(name, "FAIL", str(error), f"{protocol.upper()} :{port} listening", "listener table unavailable")
    if not isinstance(text, str):
        return _probe(name, "FAIL", None, f"{protocol.upper()} :{port} listening", "listener table unavailable")
    parsed = parse_listen_ports(text)
    present = listener_present(parsed, protocol, port)
    observed = {"protocol": protocol, "port": port, "present": present, "parsed": {key: sorted(value) for key, value in parsed.items()}}
    if present:
        return _probe(name, "PASS", observed, f"{protocol.upper()} :{port} listening", f"{protocol.upper()} :{port} observed")
    return _probe(
        name,
        "FAIL",
        observed,
        f"{protocol.upper()} :{port} listening",
        f"{protocol.upper()} listener absent",
    )


def _identity_advanced(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    keys: Sequence[str],
) -> tuple[bool, dict[str, Any]]:
    observed: dict[str, Any] = {"identities": {}}
    advanced = False
    for key in keys:
        previous = _nested(before, key)
        current = _nested(after, key)
        observed["identities"][key] = {"before": previous, "after": current}
        if _number(previous) and _number(current) and float(current) > float(previous):
            advanced = True
    observed["ppm_before"] = _nested(before, "values.ppm")
    observed["ppm_after"] = _nested(after, "values.ppm")
    return advanced, observed


def _dropped_total(health: object) -> int | None:
    dropped = _nested(_mapping(health), "receiver.sensor_logging.dropped")
    if not isinstance(dropped, Mapping):
        return None
    total = 0
    for value in dropped.values():
        if not _number(value):
            return None
        total += int(value)
    return total


def _sensor_state(status_doc: object, sensor_id: str) -> dict[str, Any]:
    document = _mapping(status_doc)
    sensor = _mapping(document.get(sensor_id))
    state = _mapping(sensor.get("state"))
    return dict(state) if state else {}


def _sensor_runtime(status_doc: object, sensor_id: str) -> dict[str, Any]:
    document = _mapping(status_doc)
    direct = _mapping(_mapping(document.get(sensor_id)).get("runtime_status"))
    if direct:
        return dict(direct)
    return dict(_mapping(_mapping(document.get("runtime_status")).get("sensors")).get(sensor_id))


def _nested(value: object, path: str) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _probe(name: str, status: str, observed: object, expected: object, reason: str) -> dict[str, Any]:
    if status not in PROBE_STATUSES:
        raise ValueError(f"unknown probe status: {status}")
    return {
        "name": name,
        "status": status,
        "observed": observed,
        "expected": expected,
        "reason": reason,
    }
