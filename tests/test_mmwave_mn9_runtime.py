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
