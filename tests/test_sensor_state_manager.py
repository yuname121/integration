from __future__ import annotations

import json
import struct
import threading
import unittest

from gateway.protocol import (
    PACKET_TELEMETRY_JSON,
    PACKET_THERMAL_U16_BE,
    PacketHeader,
    TelemetryPayload,
    ThermalFrame,
)
from state.manager import SensorStateManager


PEER = ("192.168.1.20", 40_000)


def telemetry(
    sequence: int = 1,
    *,
    respiration: float | None = 16.0,
    heart: float | None = 72.0,
    co2: float | None = 800.0,
    motion: bool = False,
) -> TelemetryPayload:
    valid = {
        "respiration": respiration is not None,
        "heart": heart is not None,
        "co2": co2 is not None,
    }
    return TelemetryPayload(
        header=PacketHeader(PACKET_TELEMETRY_JSON, sequence, 100),
        device_id="esp32-01",
        uptime_ms=sequence * 1_000,
        respiration_rate_bpm=respiration,
        heart_rate_bpm=heart,
        co2_ppm=co2,
        pir_motion=motion,
        valid=valid,
    )


def thermal(sequence: int = 1) -> ThermalFrame:
    pixel_count = 80 * 62
    pixels = struct.pack(f"!{pixel_count}H", *([1_000] * pixel_count))
    return ThermalFrame(
        header=PacketHeader(PACKET_THERMAL_U16_BE, sequence, 9_936),
        width=80,
        height=62,
        frame_sequence=sequence,
        uptime_ms=sequence * 1_000,
        minimum_raw=1_000,
        maximum_raw=1_000,
        pixel_bytes=pixels,
    )


class SensorStateManagerTests(unittest.TestCase):
    def test_initial_state_is_explicit_no_data(self) -> None:
        state = SensorStateManager().snapshot(now=100.0, monotonic_now=50.0)
        self.assertEqual(state["system"], "OFFLINE")
        for sensor in state["sensors"].values():
            self.assertEqual(sensor["status"], "NO_DATA")
            self.assertFalse(sensor["connected"])
            self.assertFalse(sensor["current"])
            self.assertIsNone(sensor["last_update"])

    def test_telemetry_updates_three_sensors_not_thermal(self) -> None:
        manager = SensorStateManager()
        manager.ingest(telemetry(4, motion=True), PEER, received_at=100.0, monotonic_at=10.0)
        state = manager.snapshot(now=100.1, monotonic_now=10.1)

        self.assertEqual(state["system"], "DEGRADED")
        self.assertEqual(state["sensors"]["mmwave"]["status"], "LIVE")
        self.assertEqual(state["sensors"]["co2"]["values"]["ppm"], 800.0)
        self.assertTrue(state["sensors"]["pir"]["values"]["motion"])
        self.assertEqual(state["sensors"]["thermal"]["status"], "NO_DATA")

    def test_missing_presence_is_not_invented(self) -> None:
        manager = SensorStateManager()
        manager.ingest(telemetry(), PEER, received_at=100.0, monotonic_at=10.0)
        values = manager.snapshot(now=100.0, monotonic_now=10.0)["sensors"]["mmwave"]["values"]
        self.assertIsNone(values["presence"])
        self.assertFalse(values["presence_available"])

    def test_invalid_latest_value_does_not_replace_last_valid_timestamp(self) -> None:
        manager = SensorStateManager()
        manager.ingest(telemetry(1), PEER, received_at=100.0, monotonic_at=10.0)
        manager.ingest(
            telemetry(2, respiration=None, heart=None, co2=None),
            PEER,
            received_at=102.0,
            monotonic_at=12.0,
        )
        state = manager.snapshot(now=102.1, monotonic_now=12.1)["sensors"]

        self.assertEqual(state["mmwave"]["status"], "INVALID")
        self.assertEqual(state["mmwave"]["last_update"], 102.0)
        self.assertEqual(state["mmwave"]["last_valid_at"], 100.0)
        self.assertEqual(state["co2"]["status"], "INVALID")
        self.assertEqual(state["pir"]["status"], "LIVE")

    def test_connected_but_invalid_is_degraded_not_offline(self) -> None:
        manager = SensorStateManager()
        manager.ingest(
            telemetry(respiration=None, heart=None, co2=None),
            PEER,
            received_at=100.0,
            monotonic_at=10.0,
        )
        state = manager.snapshot(now=100.0, monotonic_now=10.0)
        self.assertEqual(state["system"], "DEGRADED")

    def test_sensor_specific_ttl_marks_mmwave_stale_before_co2_and_pir(self) -> None:
        manager = SensorStateManager()
        manager.ingest(telemetry(), PEER, received_at=100.0, monotonic_at=10.0)
        sensors = manager.snapshot(now=104.0, monotonic_now=14.0)["sensors"]
        self.assertEqual(sensors["mmwave"]["status"], "STALE")
        self.assertTrue(sensors["mmwave"]["stale"])
        self.assertEqual(sensors["co2"]["status"], "LIVE")
        self.assertEqual(sensors["pir"]["status"], "LIVE")

    def test_disconnect_and_stale_are_separate_facts(self) -> None:
        manager = SensorStateManager()
        manager.ingest(telemetry(), PEER, received_at=100.0, monotonic_at=10.0)
        manager.mark_peer_disconnected(PEER, disconnected_at=101.0)

        immediate = manager.snapshot(now=101.0, monotonic_now=11.0)["sensors"]["mmwave"]
        self.assertEqual(immediate["status"], "DISCONNECTED")
        self.assertFalse(immediate["connected"])
        self.assertFalse(immediate["stale"])

        later = manager.snapshot(now=104.0, monotonic_now=14.0)["sensors"]["mmwave"]
        self.assertEqual(later["status"], "DISCONNECTED")
        self.assertTrue(later["stale"])

    def test_new_packet_recovers_disconnected_state(self) -> None:
        manager = SensorStateManager()
        manager.ingest(telemetry(1), PEER, received_at=100.0, monotonic_at=10.0)
        manager.mark_peer_disconnected(PEER, disconnected_at=101.0)
        new_peer = ("192.168.1.20", 40_001)
        manager.ingest(telemetry(0), new_peer, received_at=102.0, monotonic_at=12.0)
        sensor = manager.snapshot(now=102.0, monotonic_now=12.0)["sensors"]["mmwave"]
        self.assertEqual(sensor["status"], "LIVE")
        self.assertTrue(sensor["connected"])
        self.assertEqual(sensor["peer"], "192.168.1.20:40001")

    def test_thermal_bytes_are_kept_out_of_json_snapshot(self) -> None:
        manager = SensorStateManager()
        frame = thermal(3)
        manager.ingest(frame, PEER, received_at=100.0, monotonic_at=10.0)
        snapshot = manager.snapshot(now=100.0, monotonic_now=10.0)
        encoded = json.dumps(snapshot, allow_nan=False)

        self.assertEqual(snapshot["sensors"]["thermal"]["status"], "LIVE")
        self.assertNotIn("pixel_bytes", encoded)
        self.assertIs(manager.latest_thermal_frame(), frame)

    def test_all_four_live_means_online(self) -> None:
        manager = SensorStateManager()
        manager.ingest(telemetry(), PEER, received_at=100.0, monotonic_at=10.0)
        manager.ingest(thermal(), PEER, received_at=100.0, monotonic_at=10.0)
        state = manager.snapshot(now=100.1, monotonic_now=10.1)
        self.assertEqual(state["system"], "ONLINE")

    def test_snapshot_is_safe_during_concurrent_ingest(self) -> None:
        manager = SensorStateManager()
        errors = []

        def writer() -> None:
            try:
                for sequence in range(100):
                    manager.ingest(
                        telemetry(sequence),
                        PEER,
                        received_at=100.0 + sequence,
                        monotonic_at=10.0 + sequence,
                    )
            except Exception as exc:  # pragma: no cover - assertion capture
                errors.append(exc)

        thread = threading.Thread(target=writer)
        thread.start()
        for _ in range(100):
            manager.snapshot(now=300.0, monotonic_now=300.0)
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
