#!/usr/bin/env python3
"""Regression tests for ESP32 telemetry ingestion and LCD API state."""

from __future__ import annotations

import json
import socket
import threading
import unittest

import server


def sample_telemetry() -> dict[str, object]:
    return {
        "schema": "safenest.telemetry.v1",
        "device_id": "safenest-esp32-01",
        "seq": 42,
        "uptime_ms": 12_345,
        "resp_rate_bpm": 16.25,
        "heart_rate_bpm": 72.5,
        "co2_ppm": 820,
        "pir_motion": True,
        "valid": {"respiration": True, "heart": True, "co2": True},
    }


class SensorStoreTests(unittest.TestCase):
    def test_live_snapshot_contains_latest_sensor_values(self) -> None:
        store = server.SensorStore(stale_seconds=5.0)
        store.set_connected(True, ("192.168.1.50", 45678))
        store.record_telemetry(sample_telemetry())

        snapshot = store.snapshot()
        self.assertEqual(snapshot["status"], "live")
        self.assertTrue(snapshot["fresh"])
        self.assertEqual(snapshot["device_id"], "safenest-esp32-01")
        self.assertEqual(snapshot["resp_rate_bpm"], 16.25)
        self.assertEqual(snapshot["heart_rate_bpm"], 72.5)
        self.assertEqual(snapshot["co2_ppm"], 820)
        self.assertTrue(snapshot["pir_motion"])

    def test_disconnect_marks_previous_values_stale(self) -> None:
        store = server.SensorStore(stale_seconds=5.0)
        store.set_connected(True, ("192.168.1.50", 45678))
        store.record_telemetry(sample_telemetry())
        store.set_connected(False)

        snapshot = store.snapshot()
        self.assertEqual(snapshot["status"], "stale")
        self.assertFalse(snapshot["fresh"])
        self.assertEqual(snapshot["co2_ppm"], 820)

    def test_invalid_schema_is_rejected(self) -> None:
        payload = sample_telemetry()
        payload["schema"] = "unknown"
        with self.assertRaises(ValueError):
            server.SensorStore().record_telemetry(payload)


class SensorProtocolTests(unittest.TestCase):
    def test_receiver_consumes_telemetry_and_thermal_packets(self) -> None:
        store = server.SensorStore(stale_seconds=5.0)
        receiver = server.SensorReceiver("127.0.0.1", 0, store)
        server_socket, client_socket = socket.socketpair()

        def receive_until_close() -> None:
            try:
                receiver._handle_connection(server_socket, ("127.0.0.1", 40000))
            except ConnectionError:
                pass

        thread = threading.Thread(target=receive_until_close)
        thread.start()
        telemetry = json.dumps(sample_telemetry()).encode("utf-8")
        client_socket.sendall(
            server.PACKET_HEADER.pack(
                server.SENSOR_MAGIC,
                server.SENSOR_PROTOCOL_VERSION,
                server.PACKET_TELEMETRY_JSON,
                0,
                42,
                len(telemetry),
            )
            + telemetry
        )
        thermal = b"thermal-test"
        client_socket.sendall(
            server.PACKET_HEADER.pack(
                server.SENSOR_MAGIC,
                server.SENSOR_PROTOCOL_VERSION,
                server.PACKET_THERMAL_U16_BE,
                0,
                7,
                len(thermal),
            )
            + thermal
        )
        client_socket.close()
        thread.join(timeout=2.0)
        server_socket.close()

        self.assertFalse(thread.is_alive())
        snapshot = store.snapshot()
        self.assertEqual(snapshot["seq"], 42)
        self.assertEqual(snapshot["thermal_frames_received"], 1)


if __name__ == "__main__":
    unittest.main()
