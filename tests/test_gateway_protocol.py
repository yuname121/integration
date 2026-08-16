from __future__ import annotations

import json
import socket
import struct
import threading
import time
import unittest

from gateway.protocol import (
    HEADER,
    MAGIC,
    PACKET_TELEMETRY_JSON,
    PACKET_THERMAL_U16_BE,
    PROTOCOL_VERSION,
    THERMAL_HEIGHT,
    THERMAL_META,
    THERMAL_PAYLOAD_BYTES,
    THERMAL_WIDTH,
    PacketHeader,
    ProtocolError,
    ReceiveDeadlineExceeded,
    SequenceError,
    SequenceTracker,
    TelemetryPayload,
    ThermalFrame,
    read_packet,
    recv_exact,
)
from gateway.receiver import (
    ConnectionProcessor,
    SafeNestTCPServer,
)


def telemetry_payload(sequence: int = 1, **updates) -> bytes:
    data = {
        "schema": "safenest.telemetry.v1",
        "device_id": "esp32-01",
        "seq": sequence,
        "uptime_ms": 12_345,
        "resp_rate_bpm": 16.2,
        "heart_rate_bpm": 72.5,
        "co2_ppm": 820,
        "pir_motion": True,
        "valid": {"respiration": True, "heart": True, "co2": True},
    }
    data.update(updates)
    return json.dumps(data, separators=(",", ":")).encode("utf-8")


def thermal_payload(sequence: int = 1) -> bytes:
    pixel_count = THERMAL_WIDTH * THERMAL_HEIGHT
    pixels = [1_000] * pixel_count
    pixels[-1] = 2_000
    pixel_bytes = struct.pack(f"!{pixel_count}H", *pixels)
    metadata = THERMAL_META.pack(
        THERMAL_WIDTH,
        THERMAL_HEIGHT,
        sequence,
        45_000,
        1_000,
        2_000,
    )
    return metadata + pixel_bytes


def wire_packet(packet_type: int, sequence: int, payload: bytes) -> bytes:
    return HEADER.pack(
        MAGIC,
        PROTOCOL_VERSION,
        packet_type,
        0,
        sequence,
        len(payload),
    ) + payload


def decode_from_socket(data: bytes, *, fragment_size: int | None = None):
    reader, writer = socket.socketpair()
    reader.settimeout(0.01)

    def send() -> None:
        try:
            step = fragment_size or len(data)
            for offset in range(0, len(data), step):
                writer.sendall(data[offset : offset + step])
        finally:
            writer.close()

    thread = threading.Thread(target=send)
    thread.start()
    try:
        return read_packet(reader, deadline_seconds=1.0)
    finally:
        reader.close()
        thread.join(timeout=1.0)


class ProtocolDecodeTests(unittest.TestCase):
    def test_fragmented_telemetry_one_byte_at_a_time(self) -> None:
        payload = telemetry_payload(7)
        packet = decode_from_socket(
            wire_packet(PACKET_TELEMETRY_JSON, 7, payload), fragment_size=1
        )
        self.assertIsInstance(packet, TelemetryPayload)
        self.assertEqual(packet.header.sequence, 7)
        self.assertEqual(packet.co2_ppm, 820.0)

    def test_fragmented_thermal_frame(self) -> None:
        payload = thermal_payload(9)
        self.assertEqual(len(payload), THERMAL_PAYLOAD_BYTES)
        packet = decode_from_socket(
            wire_packet(PACKET_THERMAL_U16_BE, 9, payload), fragment_size=37
        )
        self.assertIsInstance(packet, ThermalFrame)
        self.assertEqual((packet.width, packet.height), (80, 62))
        self.assertEqual((packet.minimum_raw, packet.maximum_raw), (1_000, 2_000))

    def test_bad_magic_is_rejected(self) -> None:
        payload = telemetry_payload(1)
        data = HEADER.pack(b"BAD!", 1, 1, 0, 1, len(payload)) + payload
        with self.assertRaisesRegex(ProtocolError, "invalid magic"):
            decode_from_socket(data)

    def test_wrong_thermal_length_is_rejected_before_payload_read(self) -> None:
        data = HEADER.pack(MAGIC, 1, PACKET_THERMAL_U16_BE, 0, 1, 12)
        with self.assertRaisesRegex(ProtocolError, "invalid thermal payload length"):
            decode_from_socket(data)

    def test_header_json_sequence_mismatch_is_rejected(self) -> None:
        payload = telemetry_payload(4)
        with self.assertRaisesRegex(ProtocolError, "sequence mismatch"):
            decode_from_socket(wire_packet(PACKET_TELEMETRY_JSON, 3, payload))

    def test_nan_telemetry_is_rejected(self) -> None:
        payload = telemetry_payload(2, resp_rate_bpm=float("nan"))
        with self.assertRaisesRegex(ProtocolError, "must be finite"):
            decode_from_socket(wire_packet(PACKET_TELEMETRY_JSON, 2, payload))

    def test_valid_flag_must_match_nullable_value(self) -> None:
        payload = telemetry_payload(2, resp_rate_bpm=None)
        with self.assertRaisesRegex(ProtocolError, "valid.respiration"):
            decode_from_socket(wire_packet(PACKET_TELEMETRY_JSON, 2, payload))

    def test_extended_telemetry_preserves_boot_events_and_health(self) -> None:
        payload = telemetry_payload(
            8,
            boot_id="0123456789abcdef0123456789abcdef",
            co2_measurement_event_id=42,
            co2_measurement_monotonic_ms=12_000,
            co2_measurement_event_valid=True,
            pir_event_id=3,
            pir_last_transition_monotonic_ms=11_500,
            health={
                "thermal_queue_overwrites": 2,
                "tcp_send_failures": 1,
                "co2_data_ready_query_failures": 3,
                "co2_read_failures": 4,
                "thermal_status_query_failures": 5,
            },
        )
        packet = decode_from_socket(wire_packet(PACKET_TELEMETRY_JSON, 8, payload))
        self.assertEqual(packet.boot_id, "0123456789abcdef0123456789abcdef")
        self.assertEqual(packet.co2_measurement_event_id, 42)
        self.assertEqual(packet.co2_measurement_monotonic_ms, 12_000)
        self.assertTrue(packet.co2_measurement_event_valid)
        self.assertEqual(packet.pir_event_id, 3)
        self.assertEqual(packet.health["thermal_queue_overwrites"], 2)
        self.assertEqual(packet.health["co2_data_ready_query_failures"], 3)
        self.assertEqual(packet.health["co2_read_failures"], 4)
        self.assertEqual(packet.health["thermal_status_query_failures"], 5)

    def test_legacy_and_unknown_extra_fields_remain_compatible(self) -> None:
        payload = telemetry_payload(9, future_optional_field={"ignored": True})
        packet = decode_from_socket(wire_packet(PACKET_TELEMETRY_JSON, 9, payload))
        self.assertIsNone(packet.boot_id)
        self.assertIsNone(packet.co2_measurement_event_id)

    def test_event_provenance_rejects_bad_types_ranges_and_mismatch(self) -> None:
        cases = (
            ({"co2_measurement_event_id": -1, "co2_measurement_monotonic_ms": 1, "co2_measurement_event_valid": True, "boot_id": "boot-a"}, "uint32"),
            ({"co2_measurement_event_id": 1.5, "co2_measurement_monotonic_ms": 1, "co2_measurement_event_valid": True, "boot_id": "boot-a"}, "integer"),
            ({"co2_measurement_event_id": 1, "co2_measurement_monotonic_ms": 0x1_0000_0000, "co2_measurement_event_valid": True, "boot_id": "boot-a"}, "uint32"),
            ({"co2_measurement_event_id": 1, "co2_measurement_monotonic_ms": 1, "co2_measurement_event_valid": "true", "boot_id": "boot-a"}, "boolean"),
            ({"co2_measurement_event_id": 1, "co2_measurement_monotonic_ms": 1, "co2_measurement_event_valid": True}, "boot_id"),
            ({"co2_measurement_event_id": 0, "co2_measurement_monotonic_ms": 0, "co2_measurement_event_valid": True, "boot_id": "boot-a"}, "non-zero"),
            ({"co2_measurement_event_id": 0, "co2_measurement_event_valid": False}, "appear together"),
        )
        for index, (updates, message) in enumerate(cases, start=20):
            with self.subTest(updates=updates), self.assertRaisesRegex(ProtocolError, message):
                payload = telemetry_payload(index, **updates)
                decode_from_socket(wire_packet(PACKET_TELEMETRY_JSON, index, payload))

    def test_wrong_schema_is_rejected_intentionally(self) -> None:
        payload = telemetry_payload(10, schema="safenest.telemetry.v2")
        with self.assertRaisesRegex(ProtocolError, "unsupported telemetry schema"):
            decode_from_socket(wire_packet(PACKET_TELEMETRY_JSON, 10, payload))

    def test_thermal_metadata_min_max_must_match_pixels(self) -> None:
        payload = bytearray(thermal_payload(5))
        payload[12:16] = struct.pack("!HH", 999, 2_000)
        with self.assertRaisesRegex(ProtocolError, "min/max metadata mismatch"):
            decode_from_socket(
                wire_packet(PACKET_THERMAL_U16_BE, 5, bytes(payload))
            )

    def test_mid_header_deadline_does_not_silently_restart_framing(self) -> None:
        reader, writer = socket.socketpair()
        reader.settimeout(0.005)
        writer.sendall(b"SNS")
        try:
            with self.assertRaisesRegex(ReceiveDeadlineExceeded, "3 of 16"):
                recv_exact(reader, HEADER.size, deadline_seconds=0.03)
        finally:
            reader.close()
            writer.close()


class SequenceTests(unittest.TestCase):
    def test_sequences_are_independent_per_packet_type(self) -> None:
        tracker = SequenceTracker()
        self.assertEqual(tracker.accept(PacketHeader(1, 10, 1)), 0)
        self.assertEqual(tracker.accept(PacketHeader(2, 1, THERMAL_PAYLOAD_BYTES)), 0)
        self.assertEqual(tracker.accept(PacketHeader(1, 12, 1)), 1)
        self.assertEqual(tracker.accept(PacketHeader(2, 2, THERMAL_PAYLOAD_BYTES)), 0)

    def test_duplicate_and_backward_sequences_are_rejected(self) -> None:
        duplicate = SequenceTracker()
        duplicate.accept(PacketHeader(1, 5, 1))
        with self.assertRaisesRegex(SequenceError, "duplicate"):
            duplicate.accept(PacketHeader(1, 5, 1))

        backward = SequenceTracker()
        backward.accept(PacketHeader(1, 5, 1))
        with self.assertRaisesRegex(SequenceError, "backward"):
            backward.accept(PacketHeader(1, 4, 1))

    def test_uint32_wrap_is_allowed(self) -> None:
        tracker = SequenceTracker()
        tracker.accept(PacketHeader(1, 0xFFFFFFFF, 1))
        self.assertEqual(tracker.accept(PacketHeader(1, 0, 1)), 0)


class ConnectionProcessorTests(unittest.TestCase):
    def test_new_connection_allows_esp_reboot_sequence_reset(self) -> None:
        received = []
        errors = []
        processor = ConnectionProcessor(
            lambda packet, _peer: received.append(packet.header.sequence),
            on_error=lambda error, _peer: errors.append(type(error).__name__),
            packet_deadline_seconds=0.2,
        )

        for sequence in (100, 0):
            reader, writer = socket.socketpair()
            writer.sendall(
                wire_packet(
                    PACKET_TELEMETRY_JSON,
                    sequence,
                    telemetry_payload(sequence),
                )
            )
            writer.close()
            processor.process(reader, ("127.0.0.1", 40_000 + sequence))
            reader.close()

        self.assertEqual(received, [100, 0])
        self.assertEqual(processor.stats.connections, 2)
        self.assertEqual(processor.stats.disconnects, 2)
        self.assertEqual(processor.stats.protocol_errors, 0)

    def test_malformed_connection_does_not_poison_next_connection(self) -> None:
        received = []
        processor = ConnectionProcessor(
            lambda packet, _peer: received.append(packet.header.sequence),
            packet_deadline_seconds=0.2,
        )

        bad_reader, bad_writer = socket.socketpair()
        bad_writer.sendall(HEADER.pack(b"BAD!", 1, 1, 0, 1, 1))
        bad_writer.close()
        processor.process(bad_reader, ("127.0.0.1", 1))
        bad_reader.close()

        good_reader, good_writer = socket.socketpair()
        good_writer.sendall(
            wire_packet(PACKET_TELEMETRY_JSON, 1, telemetry_payload(1))
        )
        good_writer.close()
        processor.process(good_reader, ("127.0.0.1", 2))
        good_reader.close()

        self.assertEqual(received, [1])
        self.assertEqual(processor.stats.protocol_errors, 1)


class TCPServerLoopbackTests(unittest.TestCase):
    def test_listener_accepts_disconnect_and_reconnect(self) -> None:
        received = []
        server = SafeNestTCPServer(
            lambda packet, _peer: received.append(packet),
            host="127.0.0.1",
            port=0,
            packet_deadline_seconds=0.2,
        )
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        deadline = time.monotonic() + 2.0
        while (server._listener is None or server.port == 0) and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertIsNotNone(server._listener)
        self.assertNotEqual(server.port, 0)

        try:
            with socket.create_connection(("127.0.0.1", server.port), timeout=1.0) as client:
                client.sendall(
                    wire_packet(PACKET_TELEMETRY_JSON, 80, telemetry_payload(80))
                )
            with socket.create_connection(("127.0.0.1", server.port), timeout=1.0) as client:
                client.sendall(
                    wire_packet(PACKET_THERMAL_U16_BE, 0, thermal_payload(0))
                )

            deadline = time.monotonic() + 2.0
            while len(received) < 2 and time.monotonic() < deadline:
                time.sleep(0.005)
        finally:
            server.stop()
            thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(received), 2)
        self.assertIsInstance(received[0], TelemetryPayload)
        self.assertIsInstance(received[1], ThermalFrame)
        self.assertEqual(server.stats.connections, 2)
        self.assertEqual(server.stats.protocol_errors, 0)


if __name__ == "__main__":
    unittest.main()
