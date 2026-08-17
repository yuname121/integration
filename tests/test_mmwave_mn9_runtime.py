"""Synthetic runtime mechanics only; this is not MR60/model-quality validation."""

from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

from ai.mmwave_canonical_runtime import MR60CanonicalWindowBuilder
from ai.pipeline import OnDeviceAIPipeline
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
        "values": {"breath_phase": phase, "ts_monotonic_ms": time_ms, "phase_age_ms": age,
                   "presence": presence, "presence_available": isinstance(presence, bool)},
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
        for i in range(241):
            ms = i * 125.0
            snap = {"timestamp": ms / 1000.0, "revision": i, "sensors": {
                "thermal": {"status": "NO_DATA", "values": {}},
                "co2": {"status": "NO_DATA", "values": {}},
                "pir": {"status": "NO_DATA", "values": {}},
                "mmwave": sensor(i, ms, math.sin(2 * math.pi * .25 * ms / 1000.0)),
            }}
            result = pipeline.evaluate(snap)["ai"]["mmwave"]
        self.assertTrue(result["available"])
        self.assertEqual(model.calls[-1].shape, (1, 240, 1))
        self.assertGreaterEqual(len(model.calls), 1)
        for i in range(240, 245):
            ms = i * 125.0
            snap["timestamp"] = ms / 1000.0
            snap["sensors"]["mmwave"] = sensor(i, ms, math.sin(2 * math.pi * .25 * ms / 1000.0))
            pipeline.evaluate(snap)
        self.assertGreaterEqual(len(model.calls), 2)

        snap["sensors"]["mmwave"] = sensor(246, 30_750, 0.0, presence=False)
        suppressed = pipeline.evaluate(snap)["ai"]["mmwave"]
        self.assertFalse(suppressed["available"])
        self.assertEqual(suppressed["state"], "RESPIRATORY_INFERENCE_SUPPRESSED")
        self.assertEqual(suppressed["error"], "NO_VALID_PERSON")
        self.assertTrue(suppressed["metadata"]["suppressed"])
