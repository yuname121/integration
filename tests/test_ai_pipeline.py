from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from ai.pipeline import OnDeviceAIPipeline
from ai.result import AIResult
from ai.runtime import LazyModel, ModelRuntimeUnavailable
from gateway.protocol import PacketHeader, ThermalFrame
from state.manager import SensorStateManager


def sensor(status="LIVE", values=None, sequence=1, last_update=100.0):
    return {
        "status": status,
        "values": values or {},
        "sequence": sequence,
        "last_update": last_update,
    }


def snapshot(**overrides):
    sensors = {
        "thermal": sensor(values={"frame_available": True}),
        "mmwave": sensor(values={}),
        "co2": sensor(values={"ppm": 800.0}),
        "pir": sensor(values={"motion": True}),
    }
    sensors.update(overrides)
    return {"timestamp": 100.0, "revision": 7, "sensors": sensors}


class FakeModel:
    def __init__(self, prediction=None, error=None):
        self.prediction = prediction
        self.error = error
        self.calls = []

    def predict(self, *args):
        self.calls.append(args)
        if self.error:
            raise self.error
        return self.prediction


def prediction(class_name, probabilities, confidence=0.9, **extra):
    fields = dict(
        class_name=class_name,
        probabilities=probabilities,
        confidence=confidence,
        latency_ms=2.5,
        model_id="test_model",
        model_version="0.1.0",
    )
    fields.update(extra)
    return SimpleNamespace(**fields)


class AIPipelineTests(unittest.TestCase):
    def setUp(self):
        self.manager = SensorStateManager()

    def test_thermal_frame_runs_model_with_correct_shape(self):
        thermal = FakeModel(prediction("HUMAN_FALL", [0.01, 0.04, 0.95], 0.95))
        pipeline = OnDeviceAIPipeline(self.manager, {"thermal": thermal})
        pixels = [1000 + (index % 20) for index in range(80 * 62)]
        raw = b"".join(value.to_bytes(2, "big") for value in pixels)
        frame = ThermalFrame(PacketHeader(2, 1, 9936), 80, 62, 1, 10, 1000, 1019, raw)

        result = pipeline.evaluate(snapshot(), frame)["ai"]["thermal"]

        self.assertTrue(result["available"])
        self.assertEqual(result["state"], "HUMAN_FALL")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(thermal.calls[0][0].shape, (62, 80))
        self.assertFalse(result["metadata"]["temperature_calibrated"])
        preview = result["metadata"]["heatmap_preview"]
        self.assertEqual((preview["width"], preview["height"]), (20, 16))
        self.assertEqual(len(preview["values"]), 320)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in preview["values"]))

    def test_model_failure_isolated_and_pir_rule_continues(self):
        thermal = FakeModel(error=RuntimeError("invoke failed"))
        pipeline = OnDeviceAIPipeline(self.manager, {"thermal": thermal})
        raw = (1000).to_bytes(2, "big") * (80 * 62)
        frame = ThermalFrame(PacketHeader(2, 1, 9936), 80, 62, 1, 10, 1000, 1000, raw)

        output = pipeline.evaluate(snapshot(), frame)["ai"]

        self.assertFalse(output["thermal"]["available"])
        self.assertEqual(output["thermal"]["error"], "MODEL_RUNTIME_UNAVAILABLE")
        self.assertEqual(len(output["thermal"]["metadata"]["heatmap_preview"]["values"]), 320)
        self.assertTrue(output["pir"]["available"])
        self.assertEqual(output["pir"]["source"], "rule")

    def test_stale_sensor_does_not_call_model(self):
        thermal = FakeModel(prediction("HUMAN_NORMAL", [0.0, 1.0, 0.0]))
        pipeline = OnDeviceAIPipeline(self.manager, {"thermal": thermal})
        output = pipeline.evaluate(snapshot(thermal=sensor("STALE")))["ai"]["thermal"]
        self.assertEqual(output["error"], "SENSOR_STALE")
        self.assertEqual(thermal.calls, [])

    def test_mmwave_requires_canonical_freshness_evidence(self):
        mmwave = FakeModel(prediction("NORMAL", [0.9, 0.08, 0.02]))
        pipeline = OnDeviceAIPipeline(self.manager, {"mmwave": mmwave})
        missing = pipeline.evaluate(snapshot())["ai"]["mmwave"]
        self.assertEqual(missing["error"], "CANONICAL_FRESHNESS_METADATA_MISSING")
        self.assertEqual(mmwave.calls, [])

    def test_mmwave_heuristic_fallback_is_not_reported_as_ai(self):
        pred = prediction(
            "APNEA", [0.02, 0.03, 0.95], fallback_used=True, fallback_reason="NO_TFLITE"
        )
        pipeline = OnDeviceAIPipeline(self.manager, {"mmwave": FakeModel(pred)})
        ready = snapshot(mmwave=sensor(values={}))
        result = pipeline.evaluate(ready)["ai"]["mmwave"]
        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "CANONICAL_FRESHNESS_METADATA_MISSING")

    def test_co2_requires_humidity_and_history(self):
        co2 = FakeModel(prediction("OCCUPIED", [0.1, 0.9]))
        pipeline = OnDeviceAIPipeline(self.manager, {"co2": co2})
        result = pipeline.evaluate(snapshot())["ai"]["co2"]
        self.assertEqual(result["error"], "INPUT_UNAVAILABLE")
        self.assertEqual(co2.calls, [])

        first = snapshot(co2=sensor(values={"ppm": 800.0, "humidity_percent": 50.0}, sequence=1))
        second = snapshot(co2=sensor(values={"ppm": 830.0, "humidity_percent": 50.0}, sequence=2, last_update=160.0))
        self.assertEqual(pipeline.evaluate(first)["ai"]["co2"]["error"], "WINDOW_WARMING_UP")
        result = pipeline.evaluate(second)["ai"]["co2"]
        self.assertTrue(result["available"])
        self.assertAlmostEqual(co2.calls[0][0], 30.0)

    def test_result_rejects_nan_and_output_is_strict_json(self):
        with self.assertRaises(ValueError):
            AIResult("thermal", 1.0, True, "rule", "X", score=float("nan"))
        encoded = json.dumps(OnDeviceAIPipeline(self.manager).evaluate(snapshot()), allow_nan=False)
        self.assertIn('"degraded": true', encoded)

    def test_lazy_model_caches_load_failure(self):
        calls = []

        def broken_factory():
            calls.append(1)
            raise ModuleNotFoundError("tflite runtime missing")

        model = LazyModel("thermal", factory=broken_factory)
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "tflite runtime missing"):
                model.predict([])
        self.assertEqual(calls, [1])

    def test_historical_v0_1_0_mmwave_remains_release_blocked(self):
        root = Path(__file__).resolve().parent.parent / "sources" / "ondevice_ai"
        manifest = json.loads((root / "models" / "model_manifest.json").read_text(encoding="utf-8"))
        historical = manifest["models"]["mmwave_v0_1_0"]
        self.assertFalse(historical["deployment_allowed"])
        self.assertEqual(historical["block_reason"], "CLASS_COLLAPSE_ON_REPOSITORY_NPZ")
        self.assertEqual(historical["runtime_role"], "HISTORICAL_V0_1_0")
        active = manifest["models"]["mmwave"]
        self.assertEqual(active["model_id"], "MMWAVE_M_N9_FULL_INT8_V1")
        self.assertEqual(active["runtime_role"], "ACTIVE_M_N9")
        self.assertTrue(active["runtime_adapter_compatible"])
        self.assertEqual(active["deployment_scope"], "MAC_INTEGRATION_CANDIDATE")
        self.assertEqual(active["hardware_validation"], "NOT_PERFORMED")
        self.assertFalse(active["DEVICE_VALIDATED"])

    def test_frozen_model_hashes_match_manifest(self):
        root = Path(__file__).resolve().parent.parent / "sources" / "ondevice_ai"
        manifest = json.loads((root / "models" / "model_manifest.json").read_text(encoding="utf-8"))
        for name, metadata in manifest["models"].items():
            with self.subTest(name=name):
                model = root / metadata["path"]
                self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), metadata["sha256"])
                if "size_bytes" in metadata:
                    self.assertEqual(model.stat().st_size, metadata["size_bytes"])

    def test_latest_source_provenance_matches_snapshot(self):
        root = Path(__file__).resolve().parent.parent
        provenance = json.loads((root / "LATEST_SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))
        snapshot = root / "sources" / "ondevice_ai"
        overlay = snapshot / "models" / "rp_x0_b_complete"
        frozen = [
            path
            for path in snapshot.rglob("*")
            if path.is_file() and path.suffix != ".pyc" and overlay not in path.parents and path != overlay
        ]
        overlay_files = [path for path in overlay.rglob("*") if path.is_file()] if overlay.exists() else []
        self.assertEqual(provenance["latest_origin_main"], "fa8cf13")
        self.assertEqual(provenance["latest_component_source"], "77b1695ac66fd595bd037e4574d1626b8917654c")
        self.assertEqual(provenance["ondevice_ai_snapshot"]["tracked_file_count"], 1075)
        self.assertEqual(len(frozen), 1075)
        self.assertEqual(provenance["locked_b_stage_overlay"]["file_count"], len(overlay_files))
        self.assertEqual(len(overlay_files), 19)
        self.assertEqual(provenance["mmwave_m_n9_import"]["artifact_id"], "MMWAVE_M_N9_FULL_INT8_V1")
        self.assertTrue(provenance["mmwave_m_n9_import"]["active_runtime_selector"])
        self.assertEqual(provenance["integration_policy"]["mmwave_active_selector"], "MMWAVE_M_N9_FULL_INT8_V1")


if __name__ == "__main__":
    unittest.main()
