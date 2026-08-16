from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from gateway.protocol import (
    PACKET_TELEMETRY_JSON,
    PACKET_THERMAL_U16_BE,
    PacketHeader,
    TelemetryPayload,
    ThermalFrame,
    decode_telemetry,
)
from storage.sensor_logger import SensorDataLogger, SensorStorageConfig


def telemetry(
    sequence: int,
    co2: float = 800.0,
    *,
    boot_id: str | None = None,
    event_id: int | None = None,
    pir_motion: bool = False,
    pir_event_id: int | None = None,
    pir_transition_ms: int | None = None,
    mmwave: dict | None = None,
) -> TelemetryPayload:
    return TelemetryPayload(
        header=PacketHeader(PACKET_TELEMETRY_JSON, sequence, 100),
        device_id="esp32-test",
        uptime_ms=sequence * 1_000,
        respiration_rate_bpm=16.0 + sequence,
        heart_rate_bpm=70.0 + sequence,
        co2_ppm=co2,
        pir_motion=pir_motion,
        valid={"respiration": True, "heart": True, "co2": True},
        boot_id=boot_id,
        co2_measurement_event_id=event_id,
        co2_measurement_monotonic_ms=event_id * 5_000 if event_id is not None else None,
        co2_measurement_event_valid=event_id is not None or None,
        pir_event_id=pir_event_id,
        pir_last_transition_monotonic_ms=pir_transition_ms,
        mmwave=mmwave,
    )


def thermal(sequence: int = 1) -> ThermalFrame:
    values = list(range(80 * 62))
    return ThermalFrame(
        header=PacketHeader(PACKET_THERMAL_U16_BE, sequence, 9_936),
        width=80,
        height=62,
        frame_sequence=sequence,
        uptime_ms=sequence * 100,
        minimum_raw=0,
        maximum_raw=len(values) - 1,
        pixel_bytes=struct.pack(f"!{len(values)}H", *values),
    )


def config(root: Path, **changes) -> SensorStorageConfig:
    values = {
        "root": root,
        "min_free_bytes": 0,
        "max_total_bytes": 1_000_000_000,
        "max_sensor_bytes": {
            "mmwave": 1_000_000_000,
            "co2": 1_000_000_000,
            "thermal": 1_000_000_000,
        },
        "thermal_batch_frames": 2,
        "thermal_flush_seconds": 0.05,
        "cleanup_interval_seconds": 3600.0,
    }
    values.update(changes)
    return SensorStorageConfig(**values)


class SensorDataLoggerTests(unittest.TestCase):
    def test_mmwave_and_co2_have_separate_files_and_co2_is_sixty_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logger = SensorDataLogger(config(root))
            logger.start()
            logger.submit(telemetry(1, 700.0), received_at=100.0, monotonic_at=10.0)
            logger.submit(telemetry(2, 900.0), received_at=159.9, monotonic_at=69.9)
            logger.submit(telemetry(3, 950.0), received_at=160.0, monotonic_at=70.0)
            logger.stop()

            mmwave = [json.loads(line) for path in (root / "mmwave").glob("*.jsonl") for line in path.read_text().splitlines()]
            co2 = [json.loads(line) for path in (root / "co2").glob("*.jsonl") for line in path.read_text().splitlines()]
            self.assertEqual(len(mmwave), 3)
            self.assertEqual([item["co2_ppm"] for item in co2], [700.0, 950.0])
            self.assertEqual([item["receive_monotonic"] for item in co2], [10.0, 70.0])
            self.assertNotIn("co2_ppm", mmwave[0])

    def test_thermal_npz_preserves_raw_frames_metadata_and_ai_context(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logger = SensorDataLogger(config(root))
            logger.start()
            logger.set_analysis_context(
                {"ai": {"thermal": {"state": "HUMAN_NORMAL", "available": True}}},
                {"risk_level": "NORMAL", "risk_score": 5.0, "timestamp": 100.0},
            )
            logger.submit(thermal(1), received_at=100.0, monotonic_at=10.0)
            logger.submit(thermal(2), received_at=100.1, monotonic_at=10.1)
            logger.stop()

            paths = list((root / "thermal").glob("*.npz"))
            self.assertEqual(len(paths), 1)
            with np.load(paths[0], allow_pickle=False) as saved:
                self.assertEqual(saved["frames"].shape, (2, 62, 80))
                self.assertEqual(saved["frames"].dtype, np.dtype("uint16"))
                self.assertEqual(int(saved["frames"][0, 61, 79]), 4_959)
                self.assertEqual(saved["frame_sequences"].tolist(), [1, 2])
                self.assertEqual(saved["receive_monotonic"].tolist(), [10.0, 10.1])
                context = json.loads(str(saved["analysis_json"][0]))
                self.assertEqual(context["ai"]["state"], "HUMAN_NORMAL")

    def test_co2_logger_deduplicates_by_boot_and_measurement_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logger = SensorDataLogger(config(root))
            logger.start()
            logger.submit(telemetry(1, 700.0, boot_id="boot-a", event_id=42), received_at=100.0, monotonic_at=10.0)
            logger.submit(telemetry(2, 700.0, boot_id="boot-a", event_id=42), received_at=101.0, monotonic_at=11.0)
            logger.submit(telemetry(3, 705.0, boot_id="boot-a", event_id=43), received_at=102.0, monotonic_at=12.0)
            logger.submit(telemetry(1, 710.0, boot_id="boot-b", event_id=42), received_at=103.0, monotonic_at=13.0)
            logger.stop()

            rows = [json.loads(line) for path in (root / "co2").glob("*.jsonl") for line in path.read_text().splitlines()]
            self.assertEqual([(row["boot_id"], row["co2_measurement_event_id"]) for row in rows], [("boot-a", 42), ("boot-a", 43), ("boot-b", 42)])
            self.assertEqual(rows[0]["co2_measurement_monotonic_ms"], 210_000)

    def test_restart_keeps_existing_sensor_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = SensorDataLogger(config(root))
            first.start()
            first.submit(telemetry(1), received_at=100.0, monotonic_at=10.0)
            first.stop()
            existing = {path: path.read_bytes() for path in root.rglob("*.jsonl")}

            second = SensorDataLogger(config(root))
            second.start()
            second.stop()
            self.assertEqual({path: path.read_bytes() for path in root.rglob("*.jsonl")}, existing)

    def test_per_sensor_fifo_deletes_oldest_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logger = SensorDataLogger(config(
                root,
                max_sensor_bytes={"mmwave": 8, "co2": 100, "thermal": 100},
            ))
            logger.start()
            paths = []
            for index in range(3):
                path = root / "mmwave" / f"20260814_00000{index}.jsonl"
                path.write_bytes(b"1234")
                os.utime(path, (100 + index, 100 + index))
                paths.append(path)
            logger.cleanup_now()
            logger.stop()

            self.assertFalse(paths[0].exists())
            self.assertTrue(paths[1].exists())
            self.assertTrue(paths[2].exists())

    def test_minimum_free_space_policy_cleans_before_critical_full(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logger = SensorDataLogger(config(root, min_free_bytes=100))
            logger.start()
            oldest = root / "thermal" / "20260814_000000_000000.npz"
            newest = root / "thermal" / "20260814_000001_000000.npz"
            oldest.write_bytes(b"old")
            newest.write_bytes(b"new")
            os.utime(oldest, (100, 100))
            os.utime(newest, (101, 101))
            free_values = [0, 0, 1_000]
            with mock.patch(
                "storage.sensor_logger.shutil.disk_usage",
                side_effect=lambda _path: mock.Mock(free=free_values.pop(0)),
            ):
                logger.cleanup_now()
            logger.stop()

            self.assertFalse(oldest.exists())
            self.assertFalse(newest.exists())
            self.assertGreaterEqual(logger.diagnostics()["deleted"]["thermal"], 2)

    def test_mmwave_logger_keeps_raw_phase_timing_and_pir_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logger = SensorDataLogger(config(root))
            logger.start()
            phase = -0.375
            logger.submit(
                telemetry(
                    4,
                    boot_id="boot-a",
                    pir_motion=True,
                    pir_event_id=3,
                    pir_transition_ms=11_500,
                    mmwave={
                        "breath_phase": phase,
                        "breath_rate_raw": 14.25,
                        "phase_age_ms": 80,
                        "ts_monotonic_ms": 12_345,
                        "seq": 9001,
                        "firmware_version": "mr60-owner-1",
                        "config_hash": "abc123",
                        "schema_version": "mmwave.telemetry.owner-1",
                        "error_state": None,
                    },
                ),
                received_at=1_700_000_000.5,
                monotonic_at=88.25,
            )
            logger.stop()

            records = [
                json.loads(line)
                for path in (root / "mmwave").glob("*.jsonl")
                for line in path.read_text().splitlines()
            ]
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["timestamp"], 1_700_000_000.5)
            self.assertEqual(record["receive_monotonic"], 88.25)
            self.assertEqual(record["sequence"], 4)
            self.assertEqual(record["source_uptime_ms"], 4_000)
            self.assertEqual(record["boot_id"], "boot-a")
            self.assertTrue(record["pir_motion"])
            self.assertEqual(record["pir_event_id"], 3)
            self.assertEqual(record["pir_last_transition_monotonic_ms"], 11_500)
            self.assertEqual(record["mmwave"]["breath_phase"], phase)
            self.assertEqual(record["mmwave"]["seq"], 9001)
            self.assertEqual(record["mmwave"]["phase_age_ms"], 80)
            self.assertEqual(record["mmwave"]["ts_monotonic_ms"], 12_345)
            self.assertEqual(record["mmwave"]["firmware_version"], "mr60-owner-1")
            self.assertEqual(record["mmwave"]["config_hash"], "abc123")
            self.assertEqual(record["mmwave"]["schema_version"], "mmwave.telemetry.owner-1")
            self.assertIsNone(record["mmwave"]["error_state"])
            self.assertNotIn("co2_ppm", record)

    def test_legacy_scalar_packet_does_not_invent_mmwave_or_pir_event_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logger = SensorDataLogger(config(root))
            logger.start()
            logger.submit(telemetry(1), received_at=100.0, monotonic_at=10.0)
            logger.stop()
            record = [
                json.loads(line)
                for path in (root / "mmwave").glob("*.jsonl")
                for line in path.read_text().splitlines()
            ][0]
            self.assertNotIn("mmwave", record)
            self.assertNotIn("pir_event_id", record)
            self.assertNotIn("boot_id", record)
            self.assertFalse(record["pir_motion"])

    def test_wire_json_parser_logger_disk_preserves_owner_mmwave_fields(self) -> None:
        payload = json.dumps(
            {
                "schema": "safenest.telemetry.v1",
                "device_id": "esp32-test",
                "seq": 21,
                "uptime_ms": 21000,
                "resp_rate_bpm": 16.2,
                "heart_rate_bpm": 72.5,
                "co2_ppm": 820,
                "pir_motion": False,
                "valid": {"respiration": True, "heart": True, "co2": True},
                "breath_phase": -0.375,
                "phase_age_ms": 80,
                "ts_monotonic_ms": 12345,
                "firmware_version": "mr60-owner-1",
                "config_hash": "abc123",
                "schema_version": "mmwave.telemetry.owner-1",
            },
            separators=(",", ":"),
        ).encode("utf-8")
        packet = decode_telemetry(
            PacketHeader(PACKET_TELEMETRY_JSON, 21, len(payload)),
            payload,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logger = SensorDataLogger(config(root))
            logger.start()
            logger.submit(packet, received_at=200.0, monotonic_at=5.5)
            logger.stop()
            record = [
                json.loads(line)
                for path in (root / "mmwave").glob("*.jsonl")
                for line in path.read_text().splitlines()
            ][0]
        self.assertEqual(record["sequence"], 21)
        self.assertEqual(record["timestamp"], 200.0)
        self.assertEqual(record["receive_monotonic"], 5.5)
        self.assertEqual(record["mmwave"]["breath_phase"], -0.375)
        self.assertEqual(record["mmwave"]["phase_age_ms"], 80)
        self.assertEqual(record["mmwave"]["ts_monotonic_ms"], 12345)
        self.assertEqual(record["mmwave"]["firmware_version"], "mr60-owner-1")
        self.assertEqual(record["mmwave"]["config_hash"], "abc123")
        self.assertEqual(record["mmwave"]["schema_version"], "mmwave.telemetry.owner-1")


if __name__ == "__main__":
    unittest.main()
