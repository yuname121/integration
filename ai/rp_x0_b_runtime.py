"""RP-X0 diagnostic B-complete adapters. Not the production default.

Classification: RP_X0_B_RUNTIME_PREPARATION
Enable only with SAFENEST_RP_X0_B_RUNTIME=1 or OnDeviceAIPipeline(b_runtime=True).
Historical v0.1.0 adapters remain the default and are never renamed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from ai.result import AIResult


REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_MODELS = REPO_ROOT / "sources" / "ondevice_ai" / "models" / "rp_x0_b_complete"
SELECTION_PATH = REPO_ROOT / "hil" / "rp_x0_b_runtime_selection.json"

CO2_ARTIFACT = (
    VENDOR_MODELS
    / "co2"
    / "C_B6_REDUCED_CO2_SLOPE_CANDIDATE_001_full_integer_int8.tflite"
)
CO2_SHA256 = "c5969b367f5b5e28c4d27f1bdd6220f7d02da92e99604e3f85c9f1291e98dd3b"
MMWAVE_ARTIFACT = (
    VENDOR_MODELS
    / "mmwave"
    / "M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120_int8.tflite"
)
MMWAVE_SHA256 = "6dff6aaa72c79d76715d40cf7e32bb1e6cd9b2c2e3ac78eaf2fda737561430c5"
THERMAL_EXPECTED_SHA256 = "fa9730c29535477a3994c11e664474a0ca0116afaaa172889f47446ab2ac46be"

HISTORY_SECONDS = 150.0
MAX_GAP_SECONDS = 90.0
MIN_SAMPLES = 2
SLOPE_UNIT = "ppm/min"
FEATURE_ORDER = ("CO2", "CO2_slope")
THRESHOLD = 0.43
SLOPE_PROFILE = "CO2_SLOPE_FEATURE_PROFILE_001"
P1_MEAN = 22.769290618485442
P1_STD = 2.8684523405441222
P1_EPSILON = 1e-6
THERMAL_IN_SCALE = 0.31791284680366516
THERMAL_IN_ZP = -125
MMWAVE_IN_SCALE = 0.041720833629369736
MMWAVE_IN_ZP = -3
MMWAVE_Z_MEAN = 0.0031162832173884064
MMWAVE_Z_STD = 2.955399434649939


def rp_x0_b_runtime_enabled(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    return os.getenv("SAFENEST_RP_X0_B_RUNTIME", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_selection() -> dict[str, Any]:
    return json.loads(SELECTION_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _interpreter(path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter  # type: ignore

    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


@dataclass(frozen=True)
class PhysicalCO2Event:
    device_id: str
    boot_id: str
    event_id: int
    monotonic_ms: int
    ppm: float

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.device_id, self.boot_id, self.event_id)


@dataclass
class SlopeDecision:
    status: str
    event: PhysicalCO2Event | None = None
    slope: float | None = None
    feature_vector: list[float] | None = None
    elapsed_seconds: float | None = None
    history_len: int = 0
    humidity_passed: bool = False


class CB6Runtime:
    """C-B6 2-feature interpreter plus frozen ENDPOINT_H150 slope history."""

    def __init__(self, artifact: Path = CO2_ARTIFACT) -> None:
        self.artifact = Path(artifact)
        scaler = json.loads(
            (VENDOR_MODELS / "co2" / "scaler_metadata.json").read_text(encoding="utf-8")
        )
        if tuple(scaler["feature_order"]) != FEATURE_ORDER:
            raise ValueError("C-B6 scaler feature_order drift")
        self.mean = np.asarray(scaler["mean"], dtype=np.float64)
        self.scale = np.asarray(scaler["scale"], dtype=np.float64)
        self.threshold = THRESHOLD
        self.history: list[PhysicalCO2Event] = []
        self._interpreter = None
        self.sha256 = _sha256(self.artifact) if self.artifact.is_file() else None
        if self.sha256 != CO2_SHA256:
            raise ValueError(
                f"C-B6 SHA mismatch: expected={CO2_SHA256} actual={self.sha256}"
            )

    def provenance(self) -> dict[str, Any]:
        return {
            "selection": "rp_x0_b",
            "candidate_id": "C_B6_REDUCED_CO2_SLOPE_CANDIDATE_001",
            "artifact_sha256": self.sha256,
            "preprocessing_profile": "CO2_C_B6_TRAIN_ONLY_STANDARD_SCALER_001",
            "feature_contract": list(FEATURE_ORDER),
            "slope_profile": SLOPE_PROFILE,
            "threshold": self.threshold,
            "humidity_removed": True,
        }

    def observe(self, event: PhysicalCO2Event) -> SlopeDecision:
        if event.boot_id != (self.history[-1].boot_id if self.history else event.boot_id):
            self.history.clear()
        if self.history and event.key == self.history[-1].key:
            return SlopeDecision(
                status="DUPLICATE_TRANSPORT",
                event=event,
                history_len=len(self.history),
            )
        if self.history and event.monotonic_ms <= self.history[-1].monotonic_ms:
            return SlopeDecision(
                status="NON_MONOTONIC",
                event=event,
                history_len=len(self.history),
            )
        if self.history:
            gap = (event.monotonic_ms - self.history[-1].monotonic_ms) / 1000.0
            if gap > MAX_GAP_SECONDS:
                self.history.clear()
                self.history.append(event)
                return SlopeDecision(
                    status="GAP_RESTART",
                    event=event,
                    history_len=len(self.history),
                )
        self.history.append(event)
        return self._slope_from_history()

    def _slope_from_history(self) -> SlopeDecision:
        current = self.history[-1]
        if len(self.history) < MIN_SAMPLES:
            return SlopeDecision(status="WARMUP", event=current, history_len=len(self.history))
        anchor = None
        for previous, later in zip(self.history, self.history[1:]):
            step = (later.monotonic_ms - previous.monotonic_ms) / 1000.0
            if step <= 0:
                return SlopeDecision(
                    status="NON_MONOTONIC", event=current, history_len=len(self.history)
                )
            if step > MAX_GAP_SECONDS:
                return SlopeDecision(
                    status="GAP_RESTART", event=current, history_len=len(self.history)
                )
        for candidate in reversed(self.history[:-1]):
            elapsed = (current.monotonic_ms - candidate.monotonic_ms) / 1000.0
            if elapsed >= HISTORY_SECONDS:
                anchor = candidate
                break
        if anchor is None:
            return SlopeDecision(status="WARMUP", event=current, history_len=len(self.history))
        elapsed = (current.monotonic_ms - anchor.monotonic_ms) / 1000.0
        slope = (current.ppm - anchor.ppm) / (elapsed / 60.0)
        vector = [float(current.ppm), float(slope)]
        return SlopeDecision(
            status="AVAILABLE",
            event=current,
            slope=float(slope),
            feature_vector=vector,
            elapsed_seconds=elapsed,
            history_len=len(self.history),
        )

    def infer(self, feature_vector: list[float]) -> dict[str, Any]:
        if list(FEATURE_ORDER) != ["CO2", "CO2_slope"]:
            raise RuntimeError("feature order contract broken")
        if len(feature_vector) != 2:
            raise ValueError("C-B6 requires [CO2, CO2_slope]")
        z = (np.asarray(feature_vector, dtype=np.float64) - self.mean) / self.scale
        interpreter = self._load()
        inn = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        scale, zp = inn["quantization"]
        quantized = np.clip(np.rint(z / scale + zp), -128, 127).astype(np.int8).reshape(1, 2)
        start = time.perf_counter()
        interpreter.set_tensor(inn["index"], quantized)
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])
        latency_ms = (time.perf_counter() - start) * 1000.0
        out_scale, out_zp = out["quantization"]
        occupied = float(((raw.astype(np.float32) - out_zp) * out_scale).reshape(-1)[0])
        occupied = min(1.0, max(0.0, occupied))
        occupied_flag = occupied >= self.threshold
        return {
            "class_name": "OCCUPIED" if occupied_flag else "VACANT",
            "probabilities": [1.0 - occupied, occupied],
            "occupied_probability": occupied,
            "confidence": occupied if occupied_flag else 1.0 - occupied,
            "latency_ms": latency_ms,
            "model_id": "C_B6_REDUCED_CO2_SLOPE_CANDIDATE_001",
            "model_version": "C-B6",
            "feature_vector": [float(feature_vector[0]), float(feature_vector[1])],
            "quantized_input": quantized.reshape(-1).tolist(),
        }

    def _load(self):
        if self._interpreter is None:
            self._interpreter = _interpreter(self.artifact)
        return self._interpreter


class TB5Runtime:
    """T-B5 P1 preprocessor. Invoke is fail-closed until the SSD binary is provisioned."""

    def __init__(self) -> None:
        self.artifact = VENDOR_MODELS / "thermal" / "SMALL_CNN_BASELINE_V1_P1_full_int8.tflite"
        self.missing = not self.artifact.is_file()

    def provenance(self) -> dict[str, Any]:
        return {
            "selection": "rp_x0_b",
            "candidate_id": "FULL_INT8",
            "artifact_sha256": THERMAL_EXPECTED_SHA256,
            "preprocessing_profile": "P1_TRAIN_FITTED_GLOBAL_ZSCORE",
            "historical_minmax_used": False,
            "physical_conversion": "DEFERRED_THERMAL44_UINT16",
            "orientation": "IDENTITY_SOURCE_AS_STORED",
            "geometry": "CANONICAL_62x80_NO_G1_CROP",
            "class_map": {"0": "NOT_HUMAN", "1": "HUMAN_NORMAL", "2": "HUMAN_FALL"},
            "human_fall_is_lying_proxy": True,
            "artifact_present": not self.missing,
        }

    def preprocess_celsius(self, frame: np.ndarray) -> dict[str, Any]:
        array = np.asarray(frame, dtype=np.float32)
        if array.shape == (62, 80):
            array = array[None, ..., None]
        elif array.shape == (62, 80, 1):
            array = array[None, ...]
        elif array.shape != (1, 62, 80, 1):
            raise ValueError(f"canonical thermal frame required, got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("thermal frame contains NaN or infinity")
        normalized = (array - np.float32(P1_MEAN)) / np.float32(max(P1_STD, P1_EPSILON))
        quantized = np.clip(
            np.rint(normalized / THERMAL_IN_SCALE + THERMAL_IN_ZP),
            -128,
            127,
        ).astype(np.int8)
        return {
            "input_physical_min": float(array.min()),
            "input_physical_max": float(array.max()),
            "post_normalization_mean": float(normalized.mean()),
            "post_normalization_std": float(normalized.std()),
            "quantized_min": int(quantized.min()),
            "quantized_max": int(quantized.max()),
            "tensor_shape": list(quantized.shape),
            "dtype": "int8",
            "quantized": quantized,
            "historical_minmax_used": False,
        }

    def invoke(self, frame: np.ndarray) -> dict[str, Any]:
        prepared = self.preprocess_celsius(frame)
        if self.missing:
            raise FileNotFoundError(
                "THERMAL_B_ARTIFACT_UNAVAILABLE: "
                "SMALL_CNN_BASELINE_V1_P1_full_int8.tflite is EXTERNAL_SSD_ONLY; "
                "historical thermal_fall_int8_v0.1.0.tflite must not be substituted"
            )
        actual = _sha256(self.artifact)
        if actual != THERMAL_EXPECTED_SHA256:
            raise ValueError(f"T-B5 SHA mismatch: expected={THERMAL_EXPECTED_SHA256} actual={actual}")
        interpreter = _interpreter(self.artifact)
        inn = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        interpreter.set_tensor(inn["index"], prepared["quantized"])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])
        scale, zp = out["quantization"]
        probabilities = ((raw.astype(np.float32) - zp) * scale).reshape(-1)
        class_index = int(np.argmax(probabilities))
        names = ("NOT_HUMAN", "HUMAN_NORMAL", "HUMAN_FALL")
        prepared.pop("quantized")
        prepared.update(
            {
                "class_index": class_index,
                "class_name": names[class_index],
                "probabilities": [float(value) for value in probabilities],
            }
        )
        return prepared


class MMWaveBRuntime:
    """B-candidate synthetic invoke. Live TCP phase must remain gated."""

    live_gate = "CLOSED"
    live_gate_reason = "PENDING_REAL_PHASE_EVIDENCE"

    def __init__(self) -> None:
        self.artifact = MMWAVE_ARTIFACT
        self.sha256 = _sha256(self.artifact) if self.artifact.is_file() else None
        if self.sha256 != MMWAVE_SHA256:
            raise ValueError(
                f"mmWave B SHA mismatch: expected={MMWAVE_SHA256} actual={self.sha256}"
            )
        self._interpreter = None

    def provenance(self) -> dict[str, Any]:
        return {
            "selection": "rp_x0_b",
            "candidate_id": "M-B3_CONV1D_GAP_BASELINE_seed42_M-B5_CAL_CLASS_BALANCED_120",
            "artifact_sha256": self.sha256,
            "preprocessing_profile": "M-B1_D0_B1_Z1",
            "execution_preprocessing_contract_id": "M-B10B_SELECTED_REAL_CANDIDATE_BPF_ZSCORE_V1",
            "live_gate": self.live_gate,
            "live_gate_reason": self.live_gate_reason,
        }

    def preprocess(self, window: np.ndarray) -> np.ndarray:
        from scipy.signal import butter, filtfilt

        values = np.asarray(window, dtype=np.float64).reshape(-1)
        if values.size != 300:
            raise ValueError("mmWave B window must contain 300 samples")
        if not np.all(np.isfinite(values)):
            raise ValueError("mmWave window contains NaN or infinity")
        sos_b, sos_a = butter(4, [0.1, 0.5], btype="bandpass", fs=10.0)
        filtered = filtfilt(sos_b, sos_a, values)
        zscored = (filtered - MMWAVE_Z_MEAN) / MMWAVE_Z_STD
        quantized = np.clip(
            np.rint(zscored / MMWAVE_IN_SCALE + MMWAVE_IN_ZP),
            -128,
            127,
        ).astype(np.int8)
        return quantized.reshape(1, 300, 1)

    def synthetic_infer(self, window: np.ndarray) -> dict[str, Any]:
        quantized = self.preprocess(window)
        interpreter = self._load()
        inn = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        start = time.perf_counter()
        interpreter.set_tensor(inn["index"], quantized)
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])
        latency_ms = (time.perf_counter() - start) * 1000.0
        scale, zp = out["quantization"]
        probabilities = ((raw.astype(np.float32) - zp) * scale).reshape(-1)
        names = ("NORMAL", "RAPID_OR_ABNORMAL", "APNEA")
        class_index = int(np.argmax(probabilities))
        return {
            "class_name": names[class_index],
            "probabilities": [float(value) for value in probabilities],
            "confidence": float(probabilities[class_index]),
            "latency_ms": latency_ms,
            "model_id": "M-B3_CONV1D_GAP_BASELINE_seed42_M-B6_STRICT_INT8",
            "model_version": "M-B6",
            "quantized_min": int(quantized.min()),
            "quantized_max": int(quantized.max()),
        }

    def _load(self):
        if self._interpreter is None:
            self._interpreter = _interpreter(self.artifact)
        return self._interpreter


def parse_physical_co2(sensor: Mapping[str, Any]) -> PhysicalCO2Event | None:
    values = sensor.get("values") if isinstance(sensor.get("values"), Mapping) else {}
    device_id = sensor.get("device_id")
    boot_id = sensor.get("boot_id")
    event_id = values.get("measurement_event_id")
    monotonic_ms = values.get("measurement_monotonic_ms")
    ppm = values.get("latest_measurement_ppm")
    if ppm is None:
        ppm = values.get("ppm")
    required = (device_id, boot_id, event_id, monotonic_ms, ppm)
    if any(item is None for item in required):
        return None
    if values.get("measurement_event_valid") is False:
        return None
    try:
        return PhysicalCO2Event(
            device_id=str(device_id),
            boot_id=str(boot_id),
            event_id=int(event_id),
            monotonic_ms=int(monotonic_ms),
            ppm=float(ppm),
        )
    except (TypeError, ValueError):
        return None


def b_unavailable(
    sensor_id: str,
    now: float,
    error: str,
    metadata: dict[str, Any] | None = None,
) -> AIResult:
    return AIResult(
        sensor_id=sensor_id,
        timestamp=now,
        available=False,
        source="unavailable",
        state="INPUT_UNAVAILABLE",
        error=error,
        metadata=metadata or {},
    )


def b_prediction(
    sensor_id: str,
    now: float,
    class_name: str,
    *,
    score: float,
    confidence: float,
    latency_ms: float,
    model_id: str,
    model_version: str,
    metadata: dict[str, Any],
) -> AIResult:
    return AIResult(
        sensor_id=sensor_id,
        timestamp=now,
        available=True,
        source="tflite",
        state=class_name,
        score=score,
        confidence=confidence,
        latency_ms=latency_ms,
        model_id=model_id,
        model_version=model_version,
        metadata=metadata,
    )
