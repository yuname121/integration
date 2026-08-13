#!/usr/bin/env python3
"""Reproduce the numeric examples used by the SafeNest advanced guide.

This verifier has no TensorFlow dependency.  It checks arithmetic derived from
the manifest/metadata plus current runtime boundaries: non-numeric RPM, an
over-budget mmWave gap, and missing/malformed time.  It is a learning oracle,
not field-performance evidence.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.mmwave_stream_adapter import MMWaveStreamAdapter
from risk.risk_rules import RiskRulesEvaluator


def load_co2_contract() -> tuple[dict, dict]:
    manifest = json.loads((ROOT / "models/model_manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (ROOT / "models/co2/co2_scaling_metadata_v0.1.0.json").read_text(encoding="utf-8")
    )
    return manifest["models"]["co2"], metadata


def verify_co2_ranges() -> tuple[float, float, list[tuple[float, float]]]:
    contract, metadata = load_co2_contract()
    s_quant = float(contract["input"]["scale"])
    zero = int(contract["input"]["zero_point"])
    z_min = s_quant * (-128 - zero)
    z_max = s_quant * (127 - zero)
    raw_ranges = [
        (mu + sigma * z_min, mu + sigma * z_max)
        for mu, sigma in zip(metadata["mean"], metadata["scale"])
    ]

    assert abs(z_min - (-1.078263244126)) < 1e-9
    assert abs(z_max - 0.407991497777) < 1e-9
    expected = [(-4.7045, 1.7955), (19.7650, 27.9878), (267.48, 734.73)]
    for actual, target in zip(raw_ranges, expected):
        assert abs(actual[0] - target[0]) < 0.03
        assert abs(actual[1] - target[1]) < 0.03
    return z_min, z_max, raw_ranges


def next_status(previous: str, risk: float) -> str:
    if previous == "DANGER":
        return "DANGER" if risk > 65.0 else ("CAUTION" if risk > 35.0 else "NORMAL")
    return "DANGER" if risk >= 75.0 else ("CAUTION" if risk >= 40.0 else "NORMAL")


def verify_filter_sequence() -> list[tuple[int, str, Optional[float], float, str]]:
    history: deque[float] = deque(maxlen=6)
    risk = 0.0
    status = "NORMAL"
    rows: list[tuple[int, str, Optional[float], float, str]] = []

    for tick, raw in enumerate([80.0] * 10):
        history.append(raw)
        moving_mean = sum(history) / len(history)
        risk += 0.25 * (moving_mean - risk)
        status = next_status(status, risk)
        rows.append((tick, "80", moving_mean, risk, status))

    # Emergency bypasses the filters and sets the IIR state, but does not clear
    # the six raw values already retained in risk_history.
    risk = 100.0
    status = "DANGER"
    rows.append((10, "EMERGENCY", None, risk, status))

    for tick in range(11, 17):
        history.append(0.0)
        moving_mean = sum(history) / len(history)
        risk += 0.25 * (moving_mean - risk)
        status = next_status(status, risk)
        rows.append((tick, "0", moving_mean, risk, status))

    lookup = {tick: (risk_value, state) for tick, _, _, risk_value, state in rows}
    assert abs(lookup[2][0] - 46.25) < 1e-9 and lookup[2][1] == "CAUTION"
    assert abs(lookup[9][0] - 75.49491882324219) < 1e-9 and lookup[9][1] == "DANGER"
    assert abs(lookup[14][0] - 60.338541666667) < 1e-9 and lookup[14][1] == "CAUTION"
    assert abs(lookup[16][0] - 36.4404296875) < 1e-9 and lookup[16][1] == "NORMAL"
    return rows


def verify_current_type_fault_boundary() -> str:
    """Prove the current non-numeric RPM crash boundary without hiding it."""
    evaluator = RiskRulesEvaluator()
    try:
        evaluator.evaluate_respiration(
            breath_rpm="16",  # type: ignore[arg-type]
            apnea=0,
            valid=True,
            sample_timestamp=0.0,
        )
    except TypeError as exc:
        return type(exc).__name__
    raise AssertionError("non-numeric RPM unexpectedly produced a structured result")


def verify_stream_gap_boundary() -> str:
    adapter = MMWaveStreamAdapter(window_samples=300, sample_rate_hz=10.0)
    assert adapter.push_sample(0.1, timestamp_s=0.0).accepted
    assert adapter.push_sample(0.2, timestamp_s=0.1).accepted
    result = adapter.push_sample(0.3, timestamp_s=0.7)
    assert not result.accepted
    assert result.reason == "MMWAVE_STREAM_GAP_TOO_LARGE"
    assert result.buffer_size == 0 and len(adapter.buffer) == 0
    return result.reason


def verify_current_time_boundaries() -> tuple[str, str]:
    """Expose current no-time promotion and adapter type-crash behavior."""
    evaluator = RiskRulesEvaluator()
    result = evaluator.evaluate_respiration(
        breath_rpm=0.0,
        apnea=0,
        valid=True,
        sample_timestamp=None,
        dt_s=None,
    )
    assert result.emergency_override and result.reasons == ["EMERGENCY_APNEA"]

    adapter = MMWaveStreamAdapter(window_samples=300, sample_rate_hz=10.0)
    try:
        adapter.push_sample(0.1, timestamp_s="bad")  # type: ignore[arg-type]
    except TypeError as exc:
        return result.reasons[0], type(exc).__name__
    raise AssertionError("non-numeric adapter timestamp unexpectedly produced a structured result")


def main() -> None:
    z_min, z_max, raw_ranges = verify_co2_ranges()
    rows = verify_filter_sequence()
    type_fault = verify_current_type_fault_boundary()
    gap_reason = verify_stream_gap_boundary()
    no_time_reason, time_type_fault = verify_current_time_boundaries()
    print(f"CO2_Z_RANGE={z_min:.4f}..{z_max:.4f}")
    for name, bounds in zip(("slope", "humidity", "co2"), raw_ranges):
        print(f"CO2_RAW_{name.upper()}={bounds[0]:.2f}..{bounds[1]:.2f}")
    selected = {tick: (risk, state) for tick, _, _, risk, state in rows}
    for tick in (2, 9, 14, 16):
        risk, state = selected[tick]
        print(f"FILTER_TICK_{tick}=R:{risk:.4f},STATUS:{state}")
    print(f"TYPE_FAULT_CURRENT={type_fault}")
    print(f"STREAM_GAP_CURRENT={gap_reason},BUFFER:0")
    print(f"TIME_NONE_CURRENT={no_time_reason}")
    print(f"TIME_TYPE_CURRENT={time_type_fault}")
    print("PASS: learning-guide arithmetic and runtime boundaries match checked-in contracts")


if __name__ == "__main__":
    main()
