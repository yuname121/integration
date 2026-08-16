from __future__ import annotations

import importlib.util
import json
import threading
import unittest

from backend.app import BackendDependencyError, create_app
from backend.runtime import SafeNestRuntime
from backend.store import RuntimeStore
from backend.views import (
    ROUTE_CONTRACTS,
    events_document,
    health_document,
    legacy_state_document,
    sensors_document,
    status_document,
)


SENSOR_IDS = ("mmwave", "thermal", "co2", "pir")


def documents(timestamp=100.0, risk_level="NORMAL", health="HEALTHY", emergency=False,
              sensor_status="LIVE", device_health=None):
    state = {
        "timestamp": timestamp,
        "revision": int(timestamp),
        "system": "ONLINE" if sensor_status == "LIVE" else "DEGRADED",
        "device_health": device_health,
        "sensors": {
            name: {
                "sensor_id": name,
                "status": sensor_status,
                "values": {"motion": False} if name == "pir" else {},
            }
            for name in SENSOR_IDS
        },
    }
    ai = {
        "timestamp": timestamp,
        "state_revision": int(timestamp),
        "ai": {
            name: {
                "sensor_id": name,
                "timestamp": timestamp,
                "available": True,
                "state": "HUMAN_NORMAL" if name == "thermal" else "NORMAL",
                "score": 0.0,
            }
            for name in SENSOR_IDS
        },
    }
    risk = {
        "timestamp": timestamp,
        "risk_score": 0.0 if risk_level == "NORMAL" else 100.0,
        "risk_level": risk_level,
        "system_health": health,
        "degraded_mode": health != "HEALTHY",
        "is_emergency": emergency,
        "components": {
            name: {"sensor_id": name, "available": True, "score": 0.0}
            for name in SENSOR_IDS
        },
    }
    return state, ai, risk


class RuntimeStoreTests(unittest.TestCase):
    def test_publish_and_views_expose_required_contract(self):
        store = RuntimeStore()
        publication = store.publish(*documents())
        status = status_document(publication)
        self.assertEqual(status["system"], "ONLINE")
        self.assertEqual(status["risk"]["risk_level"], "NORMAL")
        for sensor_id in SENSOR_IDS:
            self.assertIn(sensor_id, status)
            self.assertIn("state", status[sensor_id])
            self.assertIn("ai", status[sensor_id])
            self.assertIn("risk_component", status[sensor_id])
        sensors = sensors_document(publication)
        self.assertEqual(set(sensors["sensors"]), set(SENSOR_IDS))
        json.dumps(status, allow_nan=False)

    def test_status_and_sensors_expose_device_health(self):
        publication = RuntimeStore().publish(
            *documents(device_health={"co2_read_failures": 2})
        )
        status = status_document(publication)
        sensors = sensors_document(publication)
        self.assertEqual(status["device_health"], {"co2_read_failures": 2})
        self.assertEqual(sensors["device_health"], {"co2_read_failures": 2})

    def test_transition_events_are_deterministic_and_newest_first(self):
        store = RuntimeStore()
        store.publish(*documents(timestamp=100.0))
        store.publish(*documents(
            timestamp=101.0,
            risk_level="DANGER",
            health="DEGRADED",
            emergency=True,
            sensor_status="STALE",
        ))
        events = store.events(20)
        event_types = [event["event_type"] for event in events]
        self.assertIn("SNAPSHOT_INITIALIZED", event_types)
        self.assertIn("RISK_LEVEL_CHANGED", event_types)
        self.assertIn("SYSTEM_HEALTH_CHANGED", event_types)
        self.assertIn("EMERGENCY_STARTED", event_types)
        self.assertEqual(event_types.count("SENSOR_STATUS_CHANGED"), 4)
        sequences = [event["sequence"] for event in events]
        self.assertEqual(sequences, sorted(sequences, reverse=True))

    def test_event_store_is_bounded_and_limit_validated(self):
        store = RuntimeStore(event_capacity=2)
        store.publish(*documents(timestamp=100.0))
        store.record_runtime_error("test", "one")
        store.record_runtime_error("test", "two")
        self.assertEqual(len(store.events(100)), 2)
        for invalid in (0, 201, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                store.events(invalid)

    def test_strict_json_rejects_nan(self):
        state, ai, risk = documents()
        risk["risk_score"] = float("nan")
        with self.assertRaises(ValueError):
            RuntimeStore().publish(state, ai, risk)

    def test_legacy_lcd_state_mapping(self):
        store = RuntimeStore()
        normal = store.publish(*documents())
        self.assertEqual(
            legacy_state_document(normal, room="A-01")["state"],
            "normal-occupied",
        )
        danger = store.publish(*documents(timestamp=101.0, risk_level="DANGER", emergency=True))
        legacy = legacy_state_document(danger, room="A-01")
        self.assertEqual(legacy["state"], "emergency")
        self.assertEqual(legacy["room"], "A-01")
        self.assertIn("updated_at", legacy)

    def test_danger_latch_is_stable_and_ack_does_not_clear_risk(self):
        store = RuntimeStore()
        store.publish(*documents(timestamp=100.0))
        danger = store.publish(*documents(timestamp=101.0, risk_level="DANGER", emergency=True))
        repeated = store.publish(*documents(timestamp=102.0, risk_level="DANGER", emergency=True))

        self.assertTrue(danger["emergency"]["active"])
        self.assertEqual(danger["emergency"]["transition_id"], repeated["emergency"]["transition_id"])
        self.assertEqual(
            [item["event_type"] for item in store.events(100)].count("DANGER_ENTERED"),
            1,
        )

        acknowledged = store.acknowledge_alarm()
        self.assertTrue(acknowledged["acknowledged"])
        self.assertFalse(acknowledged["buzzer_active"])
        self.assertTrue(store.latest()["emergency"]["active"])
        self.assertEqual(store.latest()["risk"]["risk_level"], "DANGER")

        cleared = store.publish(*documents(timestamp=103.0, risk_level="WARNING"))
        self.assertFalse(cleared["emergency"]["active"])

    def test_concurrent_publish_and_read_are_safe(self):
        store = RuntimeStore()
        errors = []

        def writer():
            try:
                for index in range(1, 101):
                    store.publish(*documents(timestamp=float(index)))
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=writer)
        thread.start()
        while thread.is_alive():
            latest = store.latest()
            if latest is not None:
                json.dumps(status_document(latest), allow_nan=False)
        thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(store.diagnostics()["publication_revision"], 100)

    def test_runtime_can_publish_initial_failed_state_without_network(self):
        runtime = SafeNestRuntime(sensor_port=0)
        publication = runtime.evaluate_once()
        self.assertEqual(publication["risk"]["system_health"], "FAILED")
        self.assertIsNone(publication["risk"]["risk_level"])
        self.assertTrue(runtime.store.diagnostics()["ready"])

    def test_health_and_event_documents(self):
        health = health_document({"ready": False, "event_count": 0}, {"connections": 0})
        self.assertTrue(health["ok"])
        self.assertFalse(health["ready"])
        self.assertEqual(events_document([])["persistence"], "memory_only_phase7")


class FastAPIContractTests(unittest.TestCase):
    def test_route_contracts_are_complete(self):
        self.assertEqual(set(ROUTE_CONTRACTS), {
            "GET /dashboard",
            "GET /api/status",
            "GET /api/sensors",
            "GET /api/events",
            "GET /api/history",
            "GET /api/state",
            "GET /api/emergency/state",
            "POST /api/emergency/119/simulation/start",
            "POST /api/emergency/119/simulation/complete",
            "POST /api/emergency/contact",
            "POST /api/emergency/acknowledge",
            "POST /api/emergency/voice",
            "POST /api/client-connection",
            "GET /health",
            "WS /ws",
        })

    def test_app_factory_has_clear_dependency_boundary_or_routes(self):
        if importlib.util.find_spec("fastapi") is None:
            with self.assertRaisesRegex(BackendDependencyError, "requirements-backend"):
                create_app(start_runtime=False)
            return
        app = create_app(start_runtime=False)
        paths = {route.path for route in app.routes}
        for path in (
            "/dashboard", "/dashboard/", "/api/status", "/api/sensors", "/api/events",
            "/api/history", "/api/state", "/api/emergency/state",
            "/api/emergency/119/simulation/start", "/api/emergency/119/simulation/complete",
            "/api/emergency/contact", "/api/emergency/acknowledge", "/api/emergency/voice",
            "/api/client-connection", "/health", "/ws",
        ):
            self.assertIn(path, paths)


if __name__ == "__main__":
    unittest.main()
