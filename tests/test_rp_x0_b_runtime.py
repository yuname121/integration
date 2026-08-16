from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from ai.pipeline import OnDeviceAIPipeline
from ai.rp_x0_b_runtime import (
    CB6Runtime,
    MMWaveBRuntime,
    PhysicalCO2Event,
    TB5Runtime,
)
from ai.runtime import LazyModel
from state.manager import SensorStateManager
from tests.test_ai_pipeline import snapshot, sensor


ROOT = Path(__file__).resolve().parent.parent
CO2_ARTIFACT = (
    ROOT
    / "sources/ondevice_ai/models/rp_x0_b_complete/co2"
    / "C_B6_REDUCED_CO2_SLOPE_CANDIDATE_001_full_integer_int8.tflite"
)
MMWAVE_ARTIFACT = (
    ROOT
    / "sources/ondevice_ai/models/rp_x0_b_complete/mmwave"
    / "M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120_int8.tflite"
)


def live_co2(event_id: int, monotonic_ms: int, ppm: float, *, sequence: int = 1) -> dict:
    return sensor(
        status="LIVE",
        sequence=sequence,
        last_update=monotonic_ms / 1000.0,
        values={
            "ppm": ppm,
            "latest_measurement_ppm": ppm,
            "measurement_event_id": event_id,
            "measurement_monotonic_ms": monotonic_ms,
            "measurement_event_valid": True,
            "humidity_percent": 99.0,
        },
    )


def event(event_id: int, monotonic_ms: int, ppm: float) -> PhysicalCO2Event:
    return PhysicalCO2Event(
        device_id="esp32-test",
        boot_id="boot-a",
        event_id=event_id,
        monotonic_ms=monotonic_ms,
        ppm=ppm,
    )


class RPX0BRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = SensorStateManager()

    def test_historical_default_is_off(self) -> None:
        pipeline = OnDeviceAIPipeline(self.manager, b_runtime=False)
        self.assertFalse(pipeline.b_runtime)
        self.assertIsNone(pipeline._b_co2)
        self.assertNotIn("rp_x0_b_runtime", pipeline.evaluate(snapshot()))

    def test_co2_feature_order_and_humidity_stripped(self) -> None:
        runtime = CB6Runtime()
        self.assertEqual(runtime.provenance()["feature_contract"], ["CO2", "CO2_slope"])
        self.assertTrue(runtime.provenance()["humidity_removed"])
        warmup = runtime.observe(event(1, 0, 600.0))
        self.assertEqual(warmup.status, "WARMUP")
        runtime.observe(event(2, 60_000, 610.0))
        runtime.observe(event(3, 120_000, 620.0))
        ready = runtime.observe(event(4, 180_000, 630.0))
        self.assertEqual(ready.status, "AVAILABLE")
        self.assertEqual(ready.feature_vector[0], 630.0)
        self.assertAlmostEqual(ready.feature_vector[1], 10.0)
        self.assertFalse(ready.humidity_passed)

    def test_duplicate_transport_does_not_advance_history(self) -> None:
        runtime = CB6Runtime()
        for index in range(1, 5):
            runtime.observe(event(index, (index - 1) * 60_000, 600.0 + index))
        before = len(runtime.history)
        duplicate = runtime.observe(event(4, 180_000, 999.0))
        self.assertEqual(duplicate.status, "DUPLICATE_TRANSPORT")
        self.assertEqual(len(runtime.history), before)
        self.assertEqual(runtime.history[-1].ppm, 604.0)

    def test_gap_reset(self) -> None:
        runtime = CB6Runtime()
        runtime.observe(event(1, 0, 600.0))
        runtime.observe(event(2, 60_000, 610.0))
        reset = runtime.observe(event(3, 60_000 + 91_000, 800.0))
        self.assertEqual(reset.status, "GAP_RESTART")
        self.assertEqual(len(runtime.history), 1)
        self.assertEqual(runtime.history[0].ppm, 800.0)

    def test_pipeline_b_co2_ignores_humidity_and_invokes(self) -> None:
        if not CO2_ARTIFACT.is_file():
            self.skipTest("C-B6 artifact not provisioned")
        pipeline = OnDeviceAIPipeline(self.manager, b_runtime=True)
        missing = snapshot(co2=sensor(status="LIVE", values={"ppm": 800.0, "humidity_percent": 50.0}))
        result = pipeline.evaluate(missing)["ai"]["co2"]
        self.assertEqual(result["error"], "PHYSICAL_EVENT_IDENTITY_MISSING")
        self.assertTrue(result["metadata"]["humidity_removed"])

        def feed(event_id: int, ms: int, ppm: float) -> dict:
            body = snapshot(
                co2={
                    **live_co2(event_id, ms, ppm, sequence=event_id),
                    "device_id": "esp32-test",
                    "boot_id": "boot-a",
                }
            )
            return pipeline.evaluate(body)["ai"]["co2"]

        self.assertEqual(feed(1, 0, 600.0)["error"], "FEATURE_UNAVAILABLE_WARMUP")
        self.assertEqual(feed(2, 60_000, 610.0)["error"], "FEATURE_UNAVAILABLE_WARMUP")
        self.assertEqual(feed(3, 120_000, 620.0)["error"], "FEATURE_UNAVAILABLE_WARMUP")
        ready = feed(4, 180_000, 1200.0)
        self.assertTrue(ready["available"])
        self.assertEqual(ready["model_id"], "C_B6_REDUCED_CO2_SLOPE_CANDIDATE_001")
        self.assertEqual(ready["metadata"]["feature_vector"][0], 1200.0)
        self.assertNotIn("humidity", json.dumps(ready["metadata"]["feature_contract"]))
        duplicate = feed(4, 180_000, 1200.0)
        self.assertEqual(duplicate["error"], "PHYSICAL_EVENT_DUPLICATE")
        self.assertEqual(duplicate["metadata"]["history_len"], 4)

    def test_thermal_b_does_not_use_minmax_or_historical_model(self) -> None:
        thermal = TB5Runtime()
        frame = np.full((62, 80), 22.769290618485442, dtype=np.float32)
        prepared = thermal.preprocess_celsius(frame)
        self.assertFalse(prepared["historical_minmax_used"])
        self.assertEqual(prepared["tensor_shape"], [1, 62, 80, 1])
        self.assertEqual(prepared["dtype"], "int8")
        self.assertAlmostEqual(prepared["post_normalization_mean"], 0.0, places=5)
        with self.assertRaisesRegex(FileNotFoundError, "THERMAL_B_ARTIFACT_UNAVAILABLE"):
            thermal.invoke(frame)
        called = []

        class Forbidden:
            def predict(self, *args):
                called.append(args)
                raise AssertionError("historical thermal adapter must not run in B mode")

        pipeline = OnDeviceAIPipeline(
            self.manager,
            {"thermal": Forbidden()},
            b_runtime=True,
        )
        idle = pipeline.evaluate(snapshot())["ai"]["thermal"]
        self.assertEqual(idle["error"], "THERMAL_B_ARTIFACT_UNAVAILABLE")
        live = snapshot(thermal=sensor(status="LIVE", values={"frame_available": True}))
        result = pipeline.evaluate(live)["ai"]["thermal"]
        self.assertEqual(result["error"], "THERMAL_B_ARTIFACT_UNAVAILABLE")
        self.assertFalse(result["metadata"]["historical_minmax_used"])
        self.assertEqual(called, [])

    def test_mmwave_live_gate_closed_even_with_window(self) -> None:
        called = []

        class Forbidden:
            def predict(self, *args):
                called.append(args)
                raise AssertionError("live mmWave B inference must stay gated")

        pipeline = OnDeviceAIPipeline(
            self.manager,
            {"mmwave": Forbidden()},
            b_runtime=True,
        )
        ready = snapshot(
            mmwave=sensor(status="LIVE", values={"respiration_phase_window": [0.1] * 300})
        )
        result = pipeline.evaluate(ready)["ai"]["mmwave"]
        self.assertEqual(result["error"], "MMWAVE_B_LIVE_GATE_CLOSED")
        self.assertEqual(result["metadata"]["live_gate"], "CLOSED")
        self.assertTrue(result["metadata"]["live_phase_window_ignored"])
        self.assertEqual(called, [])

    def test_mmwave_synthetic_invoke(self) -> None:
        if not MMWAVE_ARTIFACT.is_file():
            self.skipTest("mmWave B artifact not provisioned")
        runtime = MMWaveBRuntime()
        output = runtime.synthetic_infer(np.zeros(300, dtype=np.float64))
        self.assertEqual(len(output["probabilities"]), 3)
        self.assertIn(output["class_name"], {"NORMAL", "RAPID_OR_ABNORMAL", "APNEA"})
        self.assertEqual(runtime.live_gate, "CLOSED")

    def test_historical_lazy_mmwave_still_blocked(self) -> None:
        model = LazyModel("mmwave")
        with self.assertRaisesRegex(Exception, "MODEL_RELEASE_BLOCKED"):
            model.predict([0.0] * 300)


if __name__ == "__main__":
    unittest.main()
