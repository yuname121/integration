"""Synthetic runtime mechanics only; this is not MR60/model-quality validation."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
import unittest

from ai.mmwave_canonical_runtime import MR60CanonicalWindowBuilder
from ai.pipeline import OnDeviceAIPipeline
from gateway.protocol import PACKET_TELEMETRY_JSON, PacketHeader, decode_telemetry
from risk.engine import SafeNestRiskEngine
from state.manager import SensorStateManager


class FakeModel:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, tensor):
        self.calls.append(tensor)
        return SimpleNamespace(
            class_name="NORMAL", probabilities=[0.9, 0.08, 0.02], confidence=0.9,
            latency_ms=1.0, model_id="MMWAVE_M_N9_FULL_INT8_V1",
            model_version="m_n9_full_int8_v1", model_sha256="3b008af4be0facc4037c2afd3fe39292fb794208eb4370dbe6916b2d15aa38a4",
            fallback_used=False,
        )


def sensor(sequence: int, time_ms: float, phase: float, *, presence=True, age=3.0, boot="boot-a"):
    return {
        "status": "LIVE", "sequence": sequence, "boot_id": boot,
        "values": {
            "breath_phase": phase,
            "ts_monotonic_ms": time_ms,
            "phase_age_ms": age,
            "presence": presence,
            "presence_available": isinstance(presence, bool),
            "human_detected_raw": presence if isinstance(presence, bool) else None,
            "respiration_rate_bpm": 19.0,
            "respiration_valid": True,
        },
    }


def snapshot_for(index: int, *, presence=True):
    ms = index * 125.0
    return {
        "timestamp": ms / 1000.0,
        "revision": index,
        "sensors": {
            "thermal": {"status": "NO_DATA", "values": {}},
            "co2": {"status": "NO_DATA", "values": {}},
            "pir": {"status": "NO_DATA", "values": {}},
            "mmwave": sensor(
                index,
                ms,
                math.sin(2 * math.pi * 0.25 * ms / 1000.0),
                presence=presence,
            ),
        },
    }


class CanonicalBuilderTests(unittest.TestCase):
    def test_republication_equal_values_gap_and_boot_boundaries(self):
        builder = MR60CanonicalWindowBuilder()
        builder.ingest(sensor(1, 0, 1.0))
        builder.ingest(sensor(2, 125, 1.0))  # equal value but new event
        builder.ingest(sensor(3, 126, 9.0))  # republished timestamp
        result = builder.latest()
        self.assertEqual(result.metadata["accepted_update_count"], 2)
        self.assertEqual(result.metadata["republication_count"], 1)
        builder.ingest(sensor(4, 20_000, 1.0))
        builder.ingest(sensor(5, 30_000, 1.0))
        self.assertEqual(builder.latest().status, "WINDOW_UNAVAILABLE")
        builder.ingest(sensor(6, 30_125, 1.0, boot="boot-b"))
        self.assertLessEqual(builder.latest().metadata["accepted_update_count"], 1)

    def test_missing_freshness_never_row_counts(self):
        builder = MR60CanonicalWindowBuilder()
        builder.ingest({"sequence": 1, "values": {"breath_phase": 1.0}})
        result = builder.latest()
        self.assertEqual(result.status, "WINDOW_UNAVAILABLE")
        self.assertEqual(result.reason, "CANONICAL_FRESHNESS_METADATA_MISSING")


class MN9PipelineRuntimeTests(unittest.TestCase):
    def test_warmup_repeated_inference_and_no_person_suppression(self):
        model = FakeModel()
        pipeline = OnDeviceAIPipeline(SensorStateManager(), {"mmwave": model})
        warming = pipeline.evaluate(snapshot_for(0))["ai"]["mmwave"]
        self.assertFalse(warming["available"])
        self.assertEqual(warming["state"], "RESPIRATORY_WINDOW_WARMING_UP")
        self.assertEqual(warming["error"], "INSUFFICIENT_CONTINUOUS_DURATION")
        self.assertEqual(warming["metadata"]["canonical_window_status"], "RESPIRATORY_WINDOW_WARMING_UP")
        result = warming
        for i in range(241):
            result = pipeline.evaluate(snapshot_for(i))["ai"]["mmwave"]
        self.assertTrue(result["available"])
        self.assertEqual(result["source"], "tflite")
        self.assertEqual(result["metadata"]["canonical_window_status"], "CANONICAL_WINDOW_READY")
        self.assertEqual(model.calls[-1].shape, (1, 240, 1))
        self.assertGreaterEqual(len(model.calls), 1)
        for i in range(240, 245):
            pipeline.evaluate(snapshot_for(i))
        self.assertGreaterEqual(len(model.calls), 2)

        suppressed = pipeline.evaluate(snapshot_for(246, presence=False))["ai"]["mmwave"]
        self.assertFalse(suppressed["available"])
        self.assertEqual(suppressed["state"], "RESPIRATORY_INFERENCE_SUPPRESSED")
        self.assertEqual(suppressed["error"], "NO_VALID_PERSON")
        self.assertTrue(suppressed["metadata"]["suppressed"])

    def test_missing_presence_suppresses_ready_window(self):
        model = FakeModel()
        pipeline = OnDeviceAIPipeline(SensorStateManager(), {"mmwave": model})
        for i in range(241):
            pipeline.evaluate(snapshot_for(i))
        calls_before = len(model.calls)
        snap = snapshot_for(241)
        snap["sensors"]["mmwave"]["values"]["presence"] = None
        snap["sensors"]["mmwave"]["values"]["presence_available"] = False
        snap["sensors"]["mmwave"]["values"]["human_detected_raw"] = None
        blocked = pipeline.evaluate(snap)["ai"]["mmwave"]
        self.assertEqual(len(model.calls), calls_before)
        self.assertFalse(blocked["available"])
        self.assertEqual(blocked["error"], "PRESENCE_STATE_UNAVAILABLE")
        self.assertEqual(blocked["metadata"]["canonical_window_status"], "CANONICAL_WINDOW_READY")
        self.assertEqual(blocked["metadata"]["suppression_reason"], "PRESENCE_STATE_UNAVAILABLE")
        self.assertIn("human_detected_raw", blocked["metadata"]["missing"])

    def test_presence_and_ready_window_make_risk_use_ai(self):
        model = FakeModel()
        pipeline = OnDeviceAIPipeline(SensorStateManager(), {"mmwave": model})
        snap = snapshot_for(0)
        for i in range(241):
            snap = snapshot_for(i)
            ai = pipeline.evaluate(snap)
        mmwave = ai["ai"]["mmwave"]
        self.assertTrue(mmwave["available"])
        self.assertEqual(mmwave["source"], "tflite")
        self.assertIsNone(mmwave["error"])
        self.assertEqual(mmwave["metadata"]["canonical_window_status"], "CANONICAL_WINDOW_READY")
        risk = SafeNestRiskEngine().evaluate(snap, ai)
        self.assertEqual(risk.component_status["mmwave"], "AI")
        self.assertEqual(risk.components["mmwave"]["source"], "ai")

    def test_nested_esp_json_fills_canonical_fields_without_inventing_presence(self):
        body = json.dumps(
            {
                "schema": "safenest.telemetry.v1",
                "device_id": "esp32-01",
                "seq": 17,
                "uptime_ms": 3730,
                "resp_rate_bpm": 19.0,
                "heart_rate_bpm": 62.0,
                "co2_ppm": 800.0,
                "pir_motion": False,
                "valid": {"respiration": True, "heart": True, "co2": True},
                "mmwave": {
                    "breath_phase": -0.136825,
                    "breath_rate_raw": 7.0,
                    "phase_age_ms": 12,
                    "ts_monotonic_ms": 3718,
                    "seq": 42,
                    "firmware_version": "safenest-esp32-sensor-node/1.2.0",
                    "schema_version": "1.2",
                },
            }
        ).encode("utf-8")
        packet = decode_telemetry(PacketHeader(PACKET_TELEMETRY_JSON, 17, len(body)), body)
        self.assertAlmostEqual(packet.breath_phase, -0.136825)
        self.assertEqual(packet.ts_monotonic_ms, 3718.0)
        self.assertEqual(packet.phase_age_ms, 12.0)
        self.assertIsNone(packet.human_detected_raw)
        manager = SensorStateManager()
        manager.ingest(packet, ("127.0.0.1", 5000), received_at=100.0, monotonic_at=10.0)
        state = manager.snapshot(now=100.0, monotonic_now=10.0)
        values = state["sensors"]["mmwave"]["values"]
        self.assertAlmostEqual(values["breath_phase"], -0.136825)
        self.assertEqual(values["ts_monotonic_ms"], 3718.0)
        self.assertEqual(values["phase_age_ms"], 12.0)
        self.assertFalse(values["presence_available"])
        model = FakeModel()
        result = OnDeviceAIPipeline(manager, {"mmwave": model}).evaluate(state)["ai"]["mmwave"]
        self.assertFalse(result["available"])
        self.assertNotEqual(result["error"], "INPUT_UNAVAILABLE")
        self.assertIn(result["error"], {
            "INSUFFICIENT_CONTINUOUS_DURATION",
            "CANONICAL_FRESHNESS_METADATA_MISSING",
            "PRESENCE_STATE_UNAVAILABLE",
        })
        self.assertEqual(model.calls, [])



class WireRatePhaseAccumulationTests(unittest.TestCase):
    """The canonical window must accumulate per received packet, not per publication.

    The M-N4 contract needs 30 continuous seconds of ~8 Hz phase events with no
    gap wider than max(400 ms, 4x median dt). Sampling the phase stream from the
    publication loop (15 s by default) can never satisfy that, so the runtime
    feeds the accumulator from the receive path.
    """

    @staticmethod
    def _telemetry(index: int) -> object:
        ts_ms = 1_000 + index * 125
        body = json.dumps(
            {
                "schema": "safenest.telemetry.v1",
                "device_id": "esp32-01",
                "boot_id": "boot-a",
                "seq": index + 1,
                "uptime_ms": ts_ms + 10,
                "resp_rate_bpm": 16.0,
                "heart_rate_bpm": 62.0,
                "co2_ppm": 800.0,
                "pir_motion": False,
                "valid": {"respiration": True, "heart": True, "co2": True},
                "mmwave": {
                    "breath_phase": math.sin(2 * math.pi * 0.25 * ts_ms / 1000.0),
                    "phase_age_ms": 5,
                    "ts_monotonic_ms": ts_ms,
                    "seq": index + 1,
                },
            }
        ).encode("utf-8")
        return decode_telemetry(
            PacketHeader(PACKET_TELEMETRY_JSON, index + 1, len(body)), body
        )

    def test_runtime_receive_path_builds_the_window_without_extra_evaluations(self):
        from backend.runtime import SafeNestRuntime
        from storage.sensor_logger import SensorStorageConfig

        model = FakeModel()
        manager = SensorStateManager()
        pipeline = OnDeviceAIPipeline(manager, {"mmwave": model})
        runtime = SafeNestRuntime(
            sensor_host="127.0.0.1",
            sensor_port=0,
            thermal_udp_host="127.0.0.1",
            thermal_udp_port=0,
            evaluation_interval_seconds=3600.0,
            manager=manager,
            ai_pipeline=pipeline,
            storage_config=SensorStorageConfig(root=".", enabled=False),
        )

        for index in range(260):
            runtime._on_packet(self._telemetry(index), ("127.0.0.1", 5000))

        window = pipeline._mmwave_window.latest()
        self.assertEqual(window.status, "CANONICAL_WINDOW_READY")
        self.assertGreaterEqual(window.metadata["continuous_span_ms"], 30_000)
        self.assertEqual(window.tensor.shape, (1, 240, 1))
        # Presence is absent from this firmware contract, so inference stays gated.
        result = pipeline.evaluate(manager.snapshot())["ai"]["mmwave"]
        self.assertEqual(result["metadata"]["canonical_window_status"], "CANONICAL_WINDOW_READY")
        self.assertEqual(result["error"], "PRESENCE_STATE_UNAVAILABLE")
        self.assertEqual(model.calls, [])

    def test_single_publication_cannot_build_a_window_on_its_own(self):
        pipeline = OnDeviceAIPipeline(SensorStateManager(), {"mmwave": FakeModel()})
        manager = SensorStateManager()
        for index in range(260):
            manager.ingest(
                self._telemetry(index), ("127.0.0.1", 5000),
                received_at=100.0 + index, monotonic_at=10.0 + index,
            )
        # One publication sees exactly one phase event.
        pipeline.evaluate(manager.snapshot(now=360.0, monotonic_now=270.0))
        self.assertEqual(pipeline._mmwave_window.latest().metadata["accepted_update_count"], 1)



class Firmware13PresenceFieldTests(unittest.TestCase):
    """The `human_detected_raw` field emitted by the canonical ESP32 flash source.

    `sources/display-test2/esp32_sensor_node/esp32_sensor_node.ino` v1.3.0 adds
    the field to its nested `mmwave` object. The MR60 occupancy boolean is
    tri-state on the wire: true, false, or null when no 0x0F09 report has been
    parsed recently. null must not collapse to false, because the presence gate
    reads false as a confirmed empty room and would suppress inference for the
    wrong reason -- and because a firmware that cannot see occupancy must not be
    able to assert absence.

    Field order below mirrors the firmware's snprintf template so the fixture
    stays recognizably the real packet.
    """

    @staticmethod
    def _packet(sequence: int, nested: dict) -> object:
        body = json.dumps(
            {
                "schema": "safenest.telemetry.v1",
                "device_id": "esp32-01",
                "boot_id": "a" * 32,
                "seq": sequence,
                "uptime_ms": 3730,
                "resp_rate_bpm": 19.0,
                "heart_rate_bpm": 62.0,
                "co2_ppm": 800.0,
                "co2_measurement_event_id": 4,
                "co2_measurement_monotonic_ms": 3600,
                "co2_measurement_event_valid": True,
                "pir_motion": False,
                "pir_event_id": 2,
                "pir_last_transition_monotonic_ms": 3000,
                "valid": {"respiration": True, "heart": True, "co2": True},
                "mmwave": nested,
            }
        ).encode("utf-8")
        return decode_telemetry(
            PacketHeader(PACKET_TELEMETRY_JSON, sequence, len(body)), body
        )

    @staticmethod
    def _nested(**overrides) -> dict:
        nested = {
            "breath_phase": -0.136825,
            "total_phase": 1.0,
            "heart_phase": 0.2,
            "breath_rate_raw": 7.0,
            "human_detected_raw": True,
            "phase_age_ms": 12,
            "ts_monotonic_ms": 3718,
            "seq": 42,
            "firmware_version": "safenest-esp32-sensor-node/1.3.0",
            "schema_version": "1.3",
        }
        nested.update(overrides)
        return nested

    def _values(self, packet) -> dict:
        manager = SensorStateManager()
        manager.ingest(
            packet, ("127.0.0.1", 5000), received_at=100.0, monotonic_at=10.0
        )
        state = manager.snapshot(now=100.0, monotonic_now=10.0)
        return state["sensors"]["mmwave"]["values"]

    def test_nested_true_is_promoted_and_marks_presence_available(self):
        packet = self._packet(17, self._nested(human_detected_raw=True))
        self.assertIs(packet.human_detected_raw, True)
        values = self._values(packet)
        self.assertIs(values["human_detected_raw"], True)
        self.assertIs(values["presence"], True)
        self.assertIs(values["presence_available"], True)

    def test_nested_false_is_a_confirmed_empty_room_not_unknown(self):
        packet = self._packet(18, self._nested(human_detected_raw=False))
        self.assertIs(packet.human_detected_raw, False)
        values = self._values(packet)
        self.assertIs(values["human_detected_raw"], False)
        self.assertIs(values["presence"], False)
        # Available, because the radar did report -- it reported nobody.
        self.assertIs(values["presence_available"], True)

    def test_nested_null_stays_unknown_and_never_becomes_false(self):
        packet = self._packet(19, self._nested(human_detected_raw=None))
        self.assertIsNone(packet.human_detected_raw)
        values = self._values(packet)
        self.assertIsNone(values["human_detected_raw"])
        self.assertIsNone(values["presence"])
        self.assertIs(values["presence_available"], False)
        self.assertIsNot(values["presence"], False)

    def test_legacy_packet_without_the_field_still_decodes(self):
        nested = self._nested()
        del nested["human_detected_raw"]
        nested["firmware_version"] = "safenest-esp32-sensor-node/1.2.0"
        nested["schema_version"] = "1.2"
        packet = self._packet(20, nested)
        # Backward-compatible extension: pre-1.3 nodes keep being accepted.
        self.assertIsNone(packet.human_detected_raw)
        self.assertAlmostEqual(packet.breath_phase, -0.136825)
        values = self._values(packet)
        self.assertIsNone(values["human_detected_raw"])
        self.assertIs(values["presence_available"], False)

    def test_true_lets_a_ready_window_reach_the_model(self):
        """The whole point of B1: presence true is what unblocks inference."""
        model = FakeModel()
        pipeline = OnDeviceAIPipeline(SensorStateManager(), {"mmwave": model})
        manager = SensorStateManager()
        for index in range(260):
            nested = self._nested(
                breath_phase=math.sin(index * 0.35) * 0.5,
                ts_monotonic_ms=1_000 + index * 125,
                phase_age_ms=3,
                seq=index + 1,
                human_detected_raw=True,
            )
            packet = self._packet(index + 1, nested)
            pipeline.observe_telemetry(packet)
            manager.ingest(
                packet,
                ("127.0.0.1", 5000),
                received_at=100.0 + index * 0.125,
                monotonic_at=10.0 + index * 0.125,
            )
        state = manager.snapshot(now=132.5, monotonic_now=42.5)
        result = pipeline.evaluate(state)["ai"]["mmwave"]
        self.assertEqual(
            result["metadata"]["canonical_window_status"], "CANONICAL_WINDOW_READY"
        )
        self.assertNotEqual(result["error"], "PRESENCE_STATE_UNAVAILABLE")
        self.assertEqual(len(model.calls), 1)

    def test_null_suppresses_the_same_ready_window(self):
        """Negative control for the test above: only presence differs."""
        model = FakeModel()
        pipeline = OnDeviceAIPipeline(SensorStateManager(), {"mmwave": model})
        manager = SensorStateManager()
        for index in range(260):
            nested = self._nested(
                breath_phase=math.sin(index * 0.35) * 0.5,
                ts_monotonic_ms=1_000 + index * 125,
                phase_age_ms=3,
                seq=index + 1,
                human_detected_raw=None,
            )
            packet = self._packet(index + 1, nested)
            pipeline.observe_telemetry(packet)
            manager.ingest(
                packet,
                ("127.0.0.1", 5000),
                received_at=100.0 + index * 0.125,
                monotonic_at=10.0 + index * 0.125,
            )
        state = manager.snapshot(now=132.5, monotonic_now=42.5)
        result = pipeline.evaluate(state)["ai"]["mmwave"]
        self.assertEqual(
            result["metadata"]["canonical_window_status"], "CANONICAL_WINDOW_READY"
        )
        self.assertEqual(result["error"], "PRESENCE_STATE_UNAVAILABLE")
        self.assertEqual(model.calls, [])
