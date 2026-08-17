"""Mac-isolated M-N9 INT8 identity tests and M-N4 contract helpers.

No Raspberry Pi, no MR60, no live sensors. Distinguishes ACTIVE_M_N9 from
HISTORICAL_B_STAGE / historical v0.1.0. Does not rewrite gateway runtime.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
ONDEVICE = ROOT / "sources" / "ondevice_ai"
if str(ONDEVICE) not in sys.path:
    sys.path.insert(0, str(ONDEVICE))

from scripts.mmwave_m_n4_canonical import (  # noqa: E402
    CanonicalContractError,
    accept_phase_events,
    contract_self_check,
    form_canonical_window,
)

INT8_PATH = ONDEVICE / "models" / "mmwave" / "m_n9" / "MMWAVE_M_N9_FULL_INT8_V1.tflite"
INT8_SHA256 = "3b008af4be0facc4037c2afd3fe39292fb794208eb4370dbe6916b2d15aa38a4"
PRODUCTION_MANIFEST = ONDEVICE / "models" / "model_manifest.json"
M_N9_INVENTORY = ONDEVICE / "models" / "mmwave" / "m_n9" / "artifact_inventory.json"
M_N9_PROVISIONING = ROOT / "hil" / "rp_x0_m_n9_provisioning_manifest.json"
HISTORICAL_V010 = ONDEVICE / "models" / "mmwave" / "mmwave_resp_int8_v0.1.0.tflite"
HISTORICAL_MB3 = (
    ONDEVICE
    / "models"
    / "rp_x0_b_complete"
    / "mmwave"
    / "M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120_int8.tflite"
)


def make_interpreter(model_path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError:
            from tflite_runtime.interpreter import Interpreter
    interpreter = Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    return interpreter


def quantization(details: dict) -> tuple[float, int]:
    params = details.get("quantization_parameters") or {}
    scales = params.get("scales")
    zeros = params.get("zero_points")
    if scales is not None and len(scales) == 1:
        return float(scales[0]), int(zeros[0])
    scale, zero = details["quantization"]
    return float(scale), int(zero)


class MN9Int8ArtifactTests(unittest.TestCase):
    def test_sha256_and_size(self) -> None:
        self.assertTrue(INT8_PATH.is_file())
        self.assertEqual(INT8_PATH.stat().st_size, 11816)
        self.assertEqual(hashlib.sha256(INT8_PATH.read_bytes()).hexdigest(), INT8_SHA256)

    def test_mac_isolated_load_and_invoke(self) -> None:
        interpreter = make_interpreter(INT8_PATH)
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        self.assertEqual(list(inp["shape"]), [1, 240, 1])
        self.assertEqual(np.dtype(inp["dtype"]).name, "int8")
        self.assertEqual(list(out["shape"]), [1, 3])
        self.assertEqual(np.dtype(out["dtype"]).name, "int8")
        in_scale, in_zero = quantization(inp)
        out_scale, out_zero = quantization(out)
        self.assertAlmostEqual(in_scale, 0.5623255372047424)
        self.assertEqual(in_zero, 4)
        self.assertAlmostEqual(out_scale, 0.00390625)
        self.assertEqual(out_zero, -128)

        zero_q = np.full(inp["shape"], in_zero, dtype=inp["dtype"])
        interpreter.set_tensor(inp["index"], zero_q)
        interpreter.invoke()
        output = interpreter.get_tensor(out["index"])
        self.assertEqual(output.shape, (1, 3))
        self.assertEqual(output.dtype, np.int8)
        np.testing.assert_array_equal(output.reshape(-1), np.array([-128, -128, 127], dtype=np.int8))

    def test_active_pointer_is_m_n9_not_historical_b(self) -> None:
        manifest = json.loads(PRODUCTION_MANIFEST.read_text(encoding="utf-8"))
        active = manifest["models"]["mmwave"]
        self.assertEqual(active["runtime_role"], "ACTIVE_M_N9")
        self.assertEqual(active["model_id"], "MMWAVE_M_N9_FULL_INT8_V1")
        self.assertEqual(active["path"], "models/mmwave/m_n9/MMWAVE_M_N9_FULL_INT8_V1.tflite")
        self.assertEqual(active["sha256"], INT8_SHA256)
        self.assertEqual(active["hardware_validation"], "NOT_PERFORMED")
        self.assertFalse(active["DEVICE_VALIDATED"])
        self.assertTrue(active["PRESENCE_GATE_REQUIRED"])
        self.assertTrue(active["runtime_adapter_compatible"])
        self.assertEqual(manifest["models"]["co2"]["path"], "models/co2/co2_occupancy_int8_v0.1.0.tflite")
        self.assertEqual(
            manifest["models"]["thermal"]["path"],
            "models/thermal/thermal_fall_int8_v0.1.0.tflite",
        )
        self.assertTrue(HISTORICAL_V010.is_file())
        self.assertTrue(HISTORICAL_MB3.is_file())
        self.assertNotEqual(active["path"], "models/mmwave/mmwave_resp_int8_v0.1.0.tflite")
        self.assertNotIn("rp_x0_b_complete", active["path"])

    def test_inventory_records_required_flags(self) -> None:
        inventory = json.loads(M_N9_INVENTORY.read_text(encoding="utf-8"))
        provisioning = json.loads(M_N9_PROVISIONING.read_text(encoding="utf-8"))
        for doc in (inventory, provisioning):
            self.assertEqual(doc["source_ai"]["source_sha"] if "source_ai" in doc else doc["authoritative_ai_source"]["required_sha"], "390f3be3d75987a79a0e0438ba8a9d5e9e19dc97")
            self.assertEqual(doc["HISTORICAL_B_NOT_ACTIVE"], True)
            self.assertEqual(doc["DEVICE_VALIDATED"], "NO")
            self.assertEqual(doc["PI_SMOKE"], "NOT_PERFORMED")
            self.assertEqual(doc["PRESENCE_GATE_REQUIRED"], "YES")
            self.assertNotIn("/Users/", json.dumps(doc))


class MN4CanonicalHelperTests(unittest.TestCase):
    def test_self_check_clean(self) -> None:
        self.assertEqual(contract_self_check(), [])

    def test_production_accepts_freshness_fields_and_refuses_row_equals_sample(self) -> None:
        n = 250
        ts_monotonic_ms = np.arange(n, dtype=np.float64) * 125.0
        phase_age_ms = np.full(n, 3.0, dtype=np.float64)
        breath_phase = np.sin(2 * np.pi * 0.25 * ts_monotonic_ms / 1000.0)
        t_acc, x_acc, meta = accept_phase_events(
            ts_monotonic_ms,
            breath_phase,
            phase_age_ms,
            production=True,
            timestamps_are_seconds=False,
        )
        self.assertGreaterEqual(t_acc.size, 9)
        self.assertEqual(meta["notes"], ["PHASE_UPDATE_ESTIMATE_TS_MINUS_AGE"])
        window = form_canonical_window(t_acc, x_acc, 0.0)
        self.assertEqual(window.values.shape, (240,))
        self.assertEqual(window.values.dtype, np.float32)

        with self.assertRaises(CanonicalContractError) as ctx:
            accept_phase_events(
                ts_monotonic_ms,
                breath_phase,
                None,
                production=True,
                timestamps_are_seconds=False,
            )
        self.assertEqual(str(ctx.exception), "PRODUCTION_FRESHNESS_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
