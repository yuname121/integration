"""Identity and tensor-contract tests for locked B-stage runtime artifacts."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
ONDEVICE = ROOT / "sources" / "ondevice_ai"
B_COMPLETE = ONDEVICE / "models" / "rp_x0_b_complete"
PRODUCTION_MANIFEST = ONDEVICE / "models" / "model_manifest.json"
PROVISIONING_MANIFEST = ROOT / "hil" / "rp_x0_b_complete_provisioning_manifest.json"
INVENTORY = B_COMPLETE / "artifact_inventory.json"

PRODUCTION_MANIFEST_SHA256 = "ef9e9eea8dc4a9db139f7569b8d9579de3f5f5c06c69218a2e67a76ed0d8bf07"

CO2_ARTIFACT = B_COMPLETE / "co2" / "C_B6_REDUCED_CO2_SLOPE_CANDIDATE_001_full_integer_int8.tflite"
THERMAL_ARTIFACT = B_COMPLETE / "thermal" / "SMALL_CNN_BASELINE_V1_P1_full_int8.tflite"
MMWAVE_ARTIFACT = (
    B_COMPLETE
    / "mmwave"
    / "M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120_int8.tflite"
)

HISTORICAL = {
    "co2": ONDEVICE / "models" / "co2" / "co2_occupancy_int8_v0.1.0.tflite",
    "thermal": ONDEVICE / "models" / "thermal" / "thermal_fall_int8_v0.1.0.tflite",
    "mmwave": ONDEVICE / "models" / "mmwave" / "mmwave_resp_int8_v0.1.0.tflite",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def invoke_zero_point(interpreter) -> np.ndarray:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    _, zero = quantization(inp)
    tensor = np.full(inp["shape"], zero, dtype=inp["dtype"])
    interpreter.set_tensor(inp["index"], tensor)
    interpreter.invoke()
    return interpreter.get_tensor(out["index"])


class LockedBStageArtifactTests(unittest.TestCase):
    def test_production_manifest_selects_active_m_n9_not_historical_b(self) -> None:
        self.assertTrue(PRODUCTION_MANIFEST.is_file())
        self.assertEqual(sha256(PRODUCTION_MANIFEST), PRODUCTION_MANIFEST_SHA256)
        manifest = load_json(PRODUCTION_MANIFEST)
        # The CO2 selector is promoted to the C-B6 reduced-feature contract; the
        # 3-feature v0.1.0 entry is retained as history under its own key.
        self.assertEqual(
            manifest["models"]["co2"]["path"], "models/co2/co2_occupancy_int8_v0.1.0.tflite"
        )
        self.assertEqual(manifest["models"]["co2"]["runtime_role"], "HISTORICAL_CO2_V0_1_0")
        self.assertTrue(manifest["models"]["co2"]["HISTORICAL_NOT_ACTIVE"])
        active_co2 = manifest["models"]["co2_occupancy_c_b6"]
        self.assertEqual(active_co2["runtime_role"], "ACTIVE_C_B6")
        self.assertEqual(active_co2["model_id"], "C_B6_REDUCED_CO2_SLOPE_CANDIDATE_001")
        self.assertEqual(active_co2["path"], str(CO2_ARTIFACT.relative_to(ONDEVICE)))
        self.assertEqual(active_co2["input"]["feature_order"], ["CO2", "CO2_slope"])
        self.assertFalse(active_co2["input"]["humidity_included"])
        self.assertEqual(active_co2["risk_semantic"], "NONE")
        self.assertEqual(active_co2["safety_semantic"], "NONE")
        self.assertEqual(
            manifest["models"]["thermal"]["path"],
            "models/thermal/thermal_fall_int8_v0.1.0.tflite",
        )
        active = manifest["models"]["mmwave"]
        self.assertEqual(active["runtime_role"], "ACTIVE_M_N9")
        self.assertEqual(active["model_id"], "MMWAVE_M_N9_FULL_INT8_V1")
        self.assertEqual(active["path"], "models/mmwave/m_n9/MMWAVE_M_N9_FULL_INT8_V1.tflite")
        self.assertEqual(active["sha256"], "3b008af4be0facc4037c2afd3fe39292fb794208eb4370dbe6916b2d15aa38a4")
        self.assertEqual(active["input"]["shape"], [1, 240, 1])
        self.assertEqual(active["hardware_validation"], "NOT_PERFORMED")
        self.assertFalse(active["DEVICE_VALIDATED"])
        self.assertTrue(active["runtime_adapter_compatible"])
        historical = manifest["models"]["mmwave_v0_1_0"]
        self.assertEqual(historical["runtime_role"], "HISTORICAL_V0_1_0")
        self.assertEqual(historical["path"], "models/mmwave/mmwave_resp_int8_v0.1.0.tflite")
        self.assertFalse(historical["deployment_allowed"])
        self.assertNotIn("rp_x0_b_complete", active["path"])
        self.assertNotEqual(active["path"], str(MMWAVE_ARTIFACT.relative_to(ONDEVICE)))

    def test_historical_v0_1_0_preserved(self) -> None:
        for sensor, path in HISTORICAL.items():
            with self.subTest(sensor=sensor):
                self.assertTrue(path.is_file(), path)
                self.assertNotEqual(path.name, CO2_ARTIFACT.name)
                self.assertNotEqual(path.name, THERMAL_ARTIFACT.name)
                self.assertNotEqual(path.name, MMWAVE_ARTIFACT.name)

    def test_b_stage_filenames_are_not_historical_aliases(self) -> None:
        self.assertEqual(
            CO2_ARTIFACT.name,
            "C_B6_REDUCED_CO2_SLOPE_CANDIDATE_001_full_integer_int8.tflite",
        )
        self.assertEqual(THERMAL_ARTIFACT.name, "SMALL_CNN_BASELINE_V1_P1_full_int8.tflite")
        self.assertEqual(
            MMWAVE_ARTIFACT.name,
            "M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120_int8.tflite",
        )
        self.assertNotEqual(THERMAL_ARTIFACT.name, "thermal_fall_int8_v0.1.0.tflite")

    def test_artifact_sha256_and_size(self) -> None:
        expected = {
            CO2_ARTIFACT: ("c5969b367f5b5e28c4d27f1bdd6220f7d02da92e99604e3f85c9f1291e98dd3b", 1552),
            THERMAL_ARTIFACT: (
                "fa9730c29535477a3994c11e664474a0ca0116afaaa172889f47446ab2ac46be",
                318280,
            ),
            MMWAVE_ARTIFACT: (
                "6dff6aaa72c79d76715d40cf7e32bb1e6cd9b2c2e3ac78eaf2fda737561430c5",
                22080,
            ),
        }
        for path, (digest, size) in expected.items():
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertTrue(path.is_file(), path)
                self.assertEqual(path.stat().st_size, size)
                self.assertEqual(sha256(path), digest)

    def test_co2_contract_and_metadata(self) -> None:
        contract = load_json(B_COMPLETE / "co2" / "input_contract.json")
        scaler = load_json(B_COMPLETE / "co2" / "scaler_metadata.json")
        threshold = load_json(B_COMPLETE / "co2" / "threshold_contract.json")
        slope = load_json(B_COMPLETE / "co2" / "co2_slope_feature_profile.json")
        self.assertEqual(contract["feature_order"], ["CO2", "CO2_slope"])
        self.assertFalse(contract["humidity_included"])
        self.assertFalse(contract["temperature_included"])
        self.assertEqual(scaler["mean"], [606.5058118345612, 0.011527303414630624])
        self.assertEqual(scaler["scale"], [314.3524240597083, 5.661675596121919])
        self.assertEqual(threshold["threshold"], 0.43)
        self.assertFalse(threshold["b5_threshold_inherited"])
        self.assertEqual(slope["profile_id"], "CO2_SLOPE_FEATURE_PROFILE_001")
        self.assertEqual(slope["slope_method"], "ENDPOINT_DIFFERENCE")
        self.assertEqual(slope["feature_unit"], "ppm/min")
        self.assertEqual(slope["history_duration_seconds"], 150.0)
        self.assertEqual(slope["max_internal_gap_seconds"], 90.0)
        self.assertEqual(slope["gap_restart_status"], "FEATURE_UNAVAILABLE_GAP_RESTART")

        interpreter = make_interpreter(CO2_ARTIFACT)
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        self.assertEqual(list(inp["shape"]), [1, 2])
        self.assertEqual(np.dtype(inp["dtype"]).name, "int8")
        self.assertEqual(list(out["shape"]), [1, 1])
        self.assertEqual(np.dtype(out["dtype"]).name, "int8")
        scale, zero = quantization(inp)
        self.assertAlmostEqual(scale, 0.03921568766236305)
        self.assertEqual(zero, 0)
        out_scale, out_zero = quantization(out)
        self.assertAlmostEqual(out_scale, 0.00390625)
        self.assertEqual(out_zero, -128)
        output = invoke_zero_point(interpreter)
        self.assertEqual(output.shape, (1, 1))
        self.assertEqual(output.dtype, np.int8)
        np.testing.assert_array_equal(output.reshape(-1), np.array([-43], dtype=np.int8))

    def test_thermal_contract_and_p1_metadata(self) -> None:
        p1 = load_json(B_COMPLETE / "thermal" / "p1_lock.json")
        identity = load_json(B_COMPLETE / "thermal" / "identity.json")
        class_map = load_json(B_COMPLETE / "thermal" / "class_map.json")
        self.assertEqual(p1["profile_id"], "P1_TRAIN_FITTED_GLOBAL_ZSCORE")
        self.assertEqual(p1["mean"], 22.769290618485442)
        self.assertEqual(p1["std"], 2.8684523405441222)
        self.assertEqual(
            p1["statistics_checksum"],
            "10b5da044ef33f26715544c3d4bf56e9d999d6c65139e931f6997583b3f5b816",
        )
        self.assertEqual(identity["selected_candidate_id"], "FULL_INT8")
        self.assertFalse(identity["thermal44_deployment_validated"])
        self.assertEqual(class_map["0"], "NOT_HUMAN")
        self.assertEqual(class_map["1"], "HUMAN_NORMAL")
        self.assertEqual(class_map["2"], "HUMAN_FALL")

        interpreter = make_interpreter(THERMAL_ARTIFACT)
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        self.assertEqual(list(inp["shape"]), [1, 62, 80, 1])
        self.assertEqual(np.dtype(inp["dtype"]).name, "int8")
        self.assertEqual(list(out["shape"]), [1, 3])
        scale, zero = quantization(inp)
        self.assertAlmostEqual(scale, 0.31791284680366516)
        self.assertEqual(zero, -125)
        output = invoke_zero_point(interpreter)
        self.assertEqual(output.shape, (1, 3))
        np.testing.assert_array_equal(output.reshape(-1), np.array([-29, -70, -29], dtype=np.int8))

    def test_mmwave_contract_and_bpf_zscore_metadata(self) -> None:
        lock = load_json(B_COMPLETE / "mmwave" / "preprocessing_lock.json")
        summary = load_json(B_COMPLETE / "mmwave" / "locked_candidate_summary.json")
        self.assertEqual(lock["selected_profile_name"], "BPF_ZSCORE")
        self.assertEqual(lock["selected_profile_id"], "M-B1_D0_B1_Z1")
        self.assertEqual(
            lock["execution_preprocessing_contract_id"],
            "M-B10B_SELECTED_REAL_CANDIDATE_BPF_ZSCORE_V1",
        )
        self.assertEqual(lock["bpf"]["filter_family"], "Butterworth")
        self.assertEqual(lock["bpf"]["lowcut_hz"], 0.1)
        self.assertEqual(lock["bpf"]["highcut_hz"], 0.5)
        self.assertEqual(lock["bpf"]["order"], 4)
        self.assertEqual(lock["bpf"]["implementation"], "scipy.signal.filtfilt")
        self.assertEqual(lock["bpf"]["fs_hz"], 10.0)
        self.assertEqual(lock["zscore"]["mean"], 0.0031162832173884064)
        self.assertEqual(lock["zscore"]["std"], 2.955399434649939)
        self.assertEqual(
            summary["candidate_id"],
            "M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120",
        )
        production = load_json(PRODUCTION_MANIFEST)["models"]["mmwave"]
        self.assertEqual(production["runtime_role"], "ACTIVE_M_N9")
        self.assertNotEqual(production["path"], str(MMWAVE_ARTIFACT.relative_to(ONDEVICE)))
        self.assertNotEqual(list(production["input"]["shape"]), [1, 300, 1])

        interpreter = make_interpreter(MMWAVE_ARTIFACT)
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        self.assertEqual(list(inp["shape"]), [1, 300, 1])
        self.assertEqual(np.dtype(inp["dtype"]).name, "int8")
        self.assertEqual(list(out["shape"]), [1, 3])
        scale, zero = quantization(inp)
        self.assertAlmostEqual(scale, 0.041720833629369736)
        self.assertEqual(zero, -3)
        output = invoke_zero_point(interpreter)
        self.assertEqual(output.shape, (1, 3))
        np.testing.assert_array_equal(output.reshape(-1), np.array([-116, -104, 92], dtype=np.int8))

    def test_frozen_ondevice_snapshot_excludes_overlay(self) -> None:
        provenance = load_json(ROOT / "LATEST_SOURCE_PROVENANCE.json")
        snapshot = ONDEVICE
        overlay = B_COMPLETE
        frozen = [
            path
            for path in snapshot.rglob("*")
            if path.is_file() and path.suffix != ".pyc" and overlay not in path.parents and path != overlay
        ]
        overlay_files = [path for path in overlay.rglob("*") if path.is_file()]
        self.assertEqual(provenance["ondevice_ai_snapshot"]["tracked_file_count"], 1076)
        self.assertEqual(len(frozen), 1076)
        self.assertEqual(provenance["locked_b_stage_overlay"]["file_count"], 19)
        self.assertEqual(len(overlay_files), 19)

    def test_provisioning_manifest_is_non_production(self) -> None:
        manifest = load_json(PROVISIONING_MANIFEST)
        inventory = load_json(INVENTORY)
        self.assertTrue(manifest["not_a_production_model_manifest"])
        # CO2 is the only promoted selector; the mmWave gate stays closed.
        self.assertTrue(manifest["production_selection_changed"])
        self.assertEqual(
            manifest["production_selection_change_scope"], "CO2_ONLY_C_B6_REDUCED_FEATURE"
        )
        self.assertEqual(manifest["mmwave_live_b_gate"], "CLOSED")
        self.assertFalse(manifest["thermal44_deployment_validated"])
        self.assertTrue(inventory["production_selection_changed"])
        self.assertEqual(
            inventory["production_selection_change_scope"], "CO2_ONLY_C_B6_REDUCED_FEATURE"
        )
        classified = {
            item["path"]: item["classification"] for item in inventory["artifacts"]
        }
        self.assertIn("HISTORICAL", classified[str(HISTORICAL["co2"].relative_to(ROOT))])
        self.assertIn(
            "SUPERSEDED_BY_C_B6_REDUCED_FEATURE",
            classified[str(HISTORICAL["co2"].relative_to(ROOT))],
        )
        self.assertIn(
            "LOCKED_B_STAGE",
            classified[str(CO2_ARTIFACT.relative_to(ROOT))],
        )
        self.assertIn(
            "NOT_THERMAL44_DEPLOYMENT_VALIDATED",
            classified[str(THERMAL_ARTIFACT.relative_to(ROOT))],
        )
        self.assertIn(
            "LIVE_B_GATE_CLOSED",
            classified[str(MMWAVE_ARTIFACT.relative_to(ROOT))],
        )
        active = load_json(PRODUCTION_MANIFEST)["models"]["mmwave"]
        self.assertEqual(active["runtime_role"], "ACTIVE_M_N9")
        self.assertNotEqual(active["path"], str(MMWAVE_ARTIFACT.relative_to(ONDEVICE)))


if __name__ == "__main__":
    unittest.main()
