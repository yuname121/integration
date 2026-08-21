#!/usr/bin/env python3
"""Pre-sensor-connection runtime audit: field capture replay through the real stack.

This tool answers three questions with evidence, using only artifacts that are
committed in this repository:

  Q1 INGEST  - does AI-readable data actually reach the model input boundary?
  Q2 COMPUTE - does the on-device AI actually run on it (real TFLite, no stub)?
  Q3 RISK    - does the risk formula produce a publishable score from that?

It replays the real Pi field captures under ``data/mmwave/*.jsonl`` and
``data/co2/*.jsonl`` as genuine ``safenest.telemetry.v1`` TCP frames into a live
``SafeNestRuntime`` over loopback sockets, and (because no real thermal capture
is committed) sends a clearly-labelled synthetic thermal UDP frame so the
thermal TFLite path is exercised rather than skipped.

No Raspberry Pi, no ESP32, no MR60, no MLX90640 is required.

Usage (from the repository root):
    python hil/preconnect_runtime_audit.py
    python hil/preconnect_runtime_audit.py --inject-presence
    python hil/preconnect_runtime_audit.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.pipeline import OnDeviceAIPipeline  # noqa: E402
from backend.runtime import SafeNestRuntime  # noqa: E402
from gateway.protocol import (  # noqa: E402
    HEADER,
    MAGIC,
    PACKET_TELEMETRY_JSON,
    PROTOCOL_VERSION,
    THERMAL_HEIGHT,
    THERMAL_META,
    THERMAL_WIDTH,
)
from gateway.thermal_udp import encode_thermal_udp_frame  # noqa: E402
from state.manager import SensorStateManager  # noqa: E402
from storage.sensor_logger import SensorStorageConfig  # noqa: E402

MMWAVE_DIR = REPO_ROOT / "data" / "mmwave"
CO2_DIR = REPO_ROOT / "data" / "co2"


# --------------------------------------------------------------------------- #
# Field capture loading
# --------------------------------------------------------------------------- #
def load_capture(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def newest_capture(directory: Path, suffix: str) -> Path | None:
    candidates = sorted(directory.glob(f"*{suffix}.jsonl"))
    return candidates[-1] if candidates else None


def co2_by_sequence(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(item["sequence"]): item for item in records if "sequence" in item}


# --------------------------------------------------------------------------- #
# Wire encoding (identical framing to the ESP32 canonical flash source)
# --------------------------------------------------------------------------- #
def telemetry_frame(
    record: dict[str, Any],
    co2_record: dict[str, Any] | None,
    *,
    inject_presence: bool | None = None,
    inject_humidity: float | None = None,
) -> bytes:
    sequence = int(record["sequence"])
    mmwave = dict(record.get("mmwave") or {})
    resp = record.get("respiration_rate_bpm")
    heart = record.get("heart_rate_bpm")
    resp_valid = bool(record.get("respiration_valid"))
    heart_valid = bool(record.get("heart_valid"))

    co2_ppm = None
    document_extra: dict[str, Any] = {}
    if co2_record is not None:
        co2_ppm = co2_record.get("co2_ppm")
        for key in (
            "co2_measurement_event_id",
            "co2_measurement_monotonic_ms",
            "co2_measurement_event_valid",
        ):
            if key in co2_record:
                document_extra[key] = co2_record[key]

    if inject_presence is not None:
        mmwave["human_detected_raw"] = bool(inject_presence)
    if inject_humidity is not None:
        document_extra["humidity_percent"] = float(inject_humidity)

    document = {
        "schema": "safenest.telemetry.v1",
        "device_id": str(record.get("device_id", "replay-node")),
        "boot_id": record.get("boot_id"),
        "seq": sequence,
        "uptime_ms": int(record.get("source_uptime_ms", 0)),
        "resp_rate_bpm": resp if resp_valid else None,
        "heart_rate_bpm": heart if heart_valid else None,
        "co2_ppm": co2_ppm,
        "pir_motion": bool(record.get("pir_motion", False)),
        "valid": {
            "respiration": resp_valid and resp is not None,
            "heart": heart_valid and heart is not None,
            "co2": co2_ppm is not None,
        },
        "mmwave": mmwave,
        **document_extra,
    }
    if document["boot_id"] is None:
        document.pop("boot_id")
    payload = json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return HEADER.pack(
        MAGIC, PROTOCOL_VERSION, PACKET_TELEMETRY_JSON, 0, sequence, len(payload)
    ) + payload


def synthetic_thermal_payload(sequence: int, *, shape: str) -> bytes:
    """SYNTHETIC frame - no real MLX90640 capture is committed to this repo.

    Purpose is only to prove the thermal tensor/TFLite path executes end to end
    and that the head is decisive. The label carries no physical meaning.
    """
    baseline, amplitude = 7400.0, 900.0  # plausible MLX90640-class raw counts
    geometry = {
        # (centre_y, centre_x, radius_y, radius_x)
        "upright": (28.0, 40.0, 22.0, 7.0),
        "lying": (45.0, 40.0, 7.0, 26.0),
    }
    values = [baseline] * (THERMAL_WIDTH * THERMAL_HEIGHT)
    if shape in geometry:
        cy, cx, ry, rx = geometry[shape]
        for y in range(THERMAL_HEIGHT):
            for x in range(THERMAL_WIDTH):
                falloff = max(0.0, 1.0 - (((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2))
                values[y * THERMAL_WIDTH + x] = baseline + amplitude * falloff
    pixels = [int(round(value)) for value in values]
    minimum_raw, maximum_raw = min(pixels), max(pixels)
    return THERMAL_META.pack(
        THERMAL_WIDTH, THERMAL_HEIGHT, sequence, 0, minimum_raw, maximum_raw
    ) + struct.pack(f"!{len(pixels)}H", *pixels)


# --------------------------------------------------------------------------- #
# Replay driver
# --------------------------------------------------------------------------- #
class ReplayAudit:
    def __init__(
        self,
        *,
        inject_presence: bool | None,
        inject_humidity: float | None,
        risk_engine: Any = None,
    ) -> None:
        self.inject_presence = inject_presence
        self.inject_humidity = inject_humidity
        # Fast replay needs the CO2 presentation throttle collapsed, otherwise a
        # 60 s real-time gate hides every replayed ppm value.
        self.manager = SensorStateManager(co2_update_interval_seconds=0.001)
        # Real LazyModel adapters - no scripted stand-ins anywhere.
        self.pipeline = OnDeviceAIPipeline(self.manager)
        if risk_engine is None:
            from risk.formula_v1 import SafeNestRiskFormulaV1

            risk_engine = SafeNestRiskFormulaV1()
        from database.store import PersistentRuntimeStore

        self.store = PersistentRuntimeStore(":memory:")
        self.runtime = SafeNestRuntime(
            sensor_host="127.0.0.1",
            sensor_port=0,
            thermal_udp_host="127.0.0.1",
            thermal_udp_port=0,
            packet_deadline_seconds=2.0,
            evaluation_interval_seconds=3600.0,
            manager=self.manager,
            ai_pipeline=self.pipeline,
            risk_engine=risk_engine,
            store=self.store,
            storage_config=SensorStorageConfig(root=".", enabled=False),
        )

    def __enter__(self) -> "ReplayAudit":
        self.runtime.start()
        _wait(lambda: self.runtime.server.port != 0 and self.runtime.thermal_udp_server.port != 0)
        return self

    def __exit__(self, *_exc: object) -> None:
        self.runtime.stop()
        self.store.close()

    def replay(self, frames: Iterator[bytes], then) -> Any:
        """Replay frames and run ``then`` while the TCP session is still open.

        The session must stay open: closing it makes the state manager report
        DISCONNECTED, which would gate every sensor off before evaluation.
        """
        sent = 0
        client = socket.create_connection(("127.0.0.1", self.runtime.server.port), timeout=2.0)
        try:
            for frame in frames:
                client.sendall(frame)
                sent += 1
                if sent % 200 == 0:
                    time.sleep(0.005)
            _wait(lambda: self.manager.snapshot()["sensors"]["mmwave"]["sequence"] is not None)
            time.sleep(0.3)
            self.sent = sent
            return then()
        finally:
            client.close()

    def send_thermal(self, sequence: int, *, shape: str) -> None:
        payload = synthetic_thermal_payload(sequence, shape=shape)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for datagram in encode_thermal_udp_frame(payload, sequence):
                sock.sendto(datagram, ("127.0.0.1", self.runtime.thermal_udp_server.port))
        finally:
            sock.close()
        _wait(lambda: self.manager.latest_thermal_frame() is not None)


def _wait(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.005)
    raise TimeoutError("audit condition was not reached")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def summarize_capture(records: list[dict[str, Any]]) -> dict[str, Any]:
    phases = [
        r["mmwave"]["breath_phase"]
        for r in records
        if isinstance(r.get("mmwave"), dict)
        and isinstance(r["mmwave"].get("breath_phase"), (int, float))
    ]
    resp = [
        r["respiration_rate_bpm"]
        for r in records
        if isinstance(r.get("respiration_rate_bpm"), (int, float)) and r.get("respiration_valid")
    ]
    ts = [
        r["mmwave"]["ts_monotonic_ms"]
        for r in records
        if isinstance(r.get("mmwave"), dict)
        and isinstance(r["mmwave"].get("ts_monotonic_ms"), (int, float))
    ]
    span_s = (max(ts) - min(ts)) / 1000.0 if len(ts) >= 2 else 0.0
    # Distinct phase-update instants, matching the runtime's dedup rule
    # (update_ms = ts_monotonic_ms - phase_age_ms, advance tolerance 8 ms).
    updates: list[float] = []
    for r in records:
        mm = r.get("mmwave")
        if not isinstance(mm, dict):
            continue
        t, age = mm.get("ts_monotonic_ms"), mm.get("phase_age_ms")
        if not isinstance(t, (int, float)) or not isinstance(age, (int, float)):
            continue
        update = float(t) - float(age)
        if not updates or update > updates[-1] + 8.0:
            updates.append(update)
    deltas = [b - a for a, b in zip(updates, updates[1:])]
    deltas_sorted = sorted(deltas)
    median_dt = deltas_sorted[len(deltas_sorted) // 2] if deltas_sorted else None
    gap_threshold = max(400.0, 4.0 * median_dt) if median_dt else None
    presence_field_count = sum(
        1
        for r in records
        if isinstance(r.get("mmwave"), dict) and "human_detected_raw" in r["mmwave"]
    )
    return {
        "record_count": len(records),
        "breath_phase_present": len(phases),
        "breath_phase_min": min(phases) if phases else None,
        "breath_phase_max": max(phases) if phases else None,
        "monotonic_span_seconds": round(span_s, 2),
        "effective_rate_hz": round(len(ts) / span_s, 2) if span_s > 0 else None,
        "distinct_phase_updates": len(updates),
        "update_dt_median_ms": median_dt,
        "update_dt_p95_ms": (
            deltas_sorted[int(len(deltas_sorted) * 0.95)] if deltas_sorted else None
        ),
        "update_dt_max_ms": deltas_sorted[-1] if deltas_sorted else None,
        "gap_threshold_ms": gap_threshold,
        "updates_exceeding_gap_threshold": (
            sum(1 for d in deltas if gap_threshold and d > gap_threshold)
        ),
        "respiration_rate_bpm_min": min(resp) if resp else None,
        "respiration_rate_bpm_max": max(resp) if resp else None,
        "respiration_rate_bpm_mean": round(sum(resp) / len(resp), 2) if resp else None,
        "human_detected_raw_records": presence_field_count,
        "humidity_percent_records": sum(1 for r in records if "humidity_percent" in r),
    }


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add("SafeNest pre-sensor-connection runtime audit")
    add("=" * 78)

    cap = report["capture"]
    add("")
    add("[SOURCE] committed field captures")
    add(f"  mmwave file          : {cap['mmwave_file']}")
    add(f"  co2 file             : {cap['co2_file']}")
    add(f"  thermal file         : {cap['thermal_file']}")
    add(f"  thermal frame        : {cap['thermal_frame']}")
    stats = cap["mmwave_stats"]
    add(f"  records replayed     : {stats['record_count']} (limit {cap['record_limit']})")
    add(f"  breath_phase samples : {stats['breath_phase_present']}"
        f"  range [{stats['breath_phase_min']}, {stats['breath_phase_max']}]")
    add(f"  device-time span     : {stats['monotonic_span_seconds']} s"
        f" @ ~{stats['effective_rate_hz']} Hz")
    add(f"  distinct phase upd.  : {stats['distinct_phase_updates']}"
        f"  dt median {stats['update_dt_median_ms']} ms"
        f" / p95 {stats['update_dt_p95_ms']} ms"
        f" / max {stats['update_dt_max_ms']} ms")
    add(f"  M-N4 gap threshold   : {stats['gap_threshold_ms']} ms"
        f"  exceeded {stats['updates_exceeding_gap_threshold']} time(s)")
    add(f"  resp_rate_bpm        : min {stats['respiration_rate_bpm_min']}"
        f" / mean {stats['respiration_rate_bpm_mean']}"
        f" / max {stats['respiration_rate_bpm_max']}")
    add(f"  human_detected_raw   : {stats['human_detected_raw_records']} / {stats['record_count']} records")
    add(f"  humidity_percent     : {stats['humidity_percent_records']} / {stats['record_count']} records")

    add("")
    add("[Q1 INGEST] sensor state after replay")
    add(f"  {'sensor':8} {'status':14} {'valid':6} seq        notes")
    for sensor_id, entry in report["state"]["sensors"].items():
        add(f"  {sensor_id:8} {entry['status']:14} {str(entry['valid']):6} "
            f"{str(entry['sequence']):10} {entry.get('error') or ''}")
    add(f"  system               : {report['state']['system']}")

    add("")
    add("[Q2 COMPUTE] AI pipeline results")
    add(f"  {'sensor':8} {'avail':6} {'source':12} {'state':36} {'score':>6} {'conf':>6} {'ms':>7}")
    for sensor_id, entry in report["ai"]["ai"].items():
        add(f"  {sensor_id:8} {str(entry['available']):6} {entry['source']:12} "
            f"{str(entry['state'])[:36]:36} {_num(entry['score']):>6} "
            f"{_num(entry['confidence']):>6} {_num(entry['latency_ms']):>7}")
    add(f"  all_models_available : {report['ai']['all_models_available']}")
    add(f"  degraded             : {report['ai']['degraded']}")
    for sensor_id, entry in report["ai"]["ai"].items():
        if not entry["available"]:
            add(f"  ! {sensor_id}: {entry.get('error')}")
            meta = entry.get("metadata") or {}
            for key in ("missing", "canonical_window_status", "suppression_reason",
                        "continuous_span_ms", "accepted_update_count", "detail"):
                if key in meta and meta[key] not in (None, [], ""):
                    add(f"      {key}: {meta[key]}")
        else:
            meta = entry.get("metadata") or {}
            for key in ("MAD", "median_update_dt_ms", "mad_collapsed", "input_shape",
                        "probabilities", "model_sha256"):
                if key in meta and meta[key] is not None:
                    add(f"      {sensor_id}.{key}: {meta[key]}")

    add("")
    add("[Q3 RISK] risk evaluation")
    risk = report["risk"]
    add(f"  formula              : {risk.get('formula_id', risk.get('config_status'))}")
    add(f"  risk_score           : {risk['risk_score']}")
    add(f"  risk_level           : {risk['risk_level']}")
    add(f"  system_health        : {risk['system_health']}")
    add(f"  degraded_mode        : {risk['degraded_mode']}")
    add(f"  is_emergency         : {risk['is_emergency']}")
    add(f"  presence             : {risk['presence_detected']} via {risk['presence_source']}")
    add(f"  component_scores     : {risk['component_scores']}")
    add(f"  component_status     : {risk['component_status']}")
    add(f"  weights              : {risk['weights']}")
    add(f"  reasons              : {list(risk['reasons'])}")
    if "effective_weight" in risk:
        add(f"  effective_weight     : {risk['effective_weight']}"
            f"  evidence_sufficient={risk['evidence_sufficient']}")
        add(f"  score_level          : {risk['score_level']} (level_source={risk['level_source']})")
        add(f"  escalation_floors    : {list(risk['escalation_floors'])}")

    add("")
    add("[Q3 PERSIST] publish -> SQLite")
    persistence = report["persistence"]
    add(f"  publication_revision : {persistence['publication_revision']}")
    add(f"  sqlite rows          : {persistence['sqlite_rows']}")
    latest = persistence.get("sqlite_latest") or {}
    add(f"  stored risk_score    : {latest.get('risk_score')}")
    add(f"  stored risk_level    : {latest.get('risk_level')}")
    add(f"  stored system_health : {latest.get('system_health')}")

    add("")
    add("[VERDICT]")
    for item in report["verdict"]:
        add(f"  {item['gate']:34} {item['result']:6}  {item['detail']}")
    add("=" * 78)
    return "\n".join(lines)


def _num(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}" if math.isfinite(value) else "nan"
    return str(value)


def build_verdict(report: dict[str, Any]) -> list[dict[str, str]]:
    verdict: list[dict[str, str]] = []
    state = report["state"]
    ai = report["ai"]["ai"]
    risk = report["risk"]

    def gate(name: str, ok: bool, detail: str) -> None:
        verdict.append({"gate": name, "result": "PASS" if ok else "FAIL", "detail": detail})

    live = [s for s, e in state["sensors"].items() if e["status"] == "LIVE"]
    gate("Q1 wire decode + state LIVE", len(live) == 4,
         f"LIVE={sorted(live)} of 4")

    mm_meta = (ai["mmwave"].get("metadata") or {})
    gate("Q1 mmwave canonical window", mm_meta.get("canonical_window_status") == "CANONICAL_WINDOW_READY",
         f"status={mm_meta.get('canonical_window_status')}, "
         f"span={mm_meta.get('continuous_span_ms')} ms")

    # thermal and co2 score from their heads; mmwave's head is observe-only while
    # MMWAVE_M_N9_FULL_INT8_V1 is DEVICE_VALIDATED: NO, so the gate checks that it
    # was invoked and adjudicated rather than that it was trusted.
    scoring = [s for s in ("thermal", "co2") if ai[s]["source"] == "tflite"]
    gate("Q2 thermal + co2 TFLite scoring", len(scoring) == 2,
         f"tflite={sorted(scoring)} of ['co2','thermal']")

    mm_invoked = (
        ai["mmwave"]["source"] == "tflite"
        or mm_meta.get("refused_class") is not None
        or mm_meta.get("probabilities") is not None
    )
    gate("Q2 mmwave M-N9 invoked + adjudicated", mm_invoked,
         f"state={ai['mmwave'].get('state')}, error={ai['mmwave'].get('error')}")

    spectral_ready = mm_meta.get("spectral_status") == "SPECTRAL_ESTIMATE_READY"
    gate("Q2 mmwave respiration signal usable", spectral_ready,
         f"spectral={mm_meta.get('spectral_status')}, "
         f"rate={mm_meta.get('spectral_rate_rpm')} rpm, "
         f"band={mm_meta.get('spectral_band_power_fraction')}")

    gate("Q3 risk score published", risk["risk_score"] is not None and risk["risk_level"] is not None,
         f"score={risk['risk_score']}, level={risk['risk_level']}")
    gate("Q3 all components contribute", "UNAVAILABLE" not in risk["component_status"].values(),
         f"status={risk['component_status']}")

    # DEGRADED is the correct published health while the mmWave head is
    # observe-only; FAILED or a missing level would not be.
    component = ((risk.get("components") or {}).get("mmwave") or {})
    observe_only = (component.get("metadata") or {}).get("neural_trust") == "OBSERVE_ONLY"
    expected_health = {"DEGRADED"} if observe_only else {"HEALTHY"}
    gate("Q3 health matches trust policy", risk["system_health"] in expected_health,
         f"health={risk['system_health']}, expected={sorted(expected_health)}"
         f" (neural_trust={'OBSERVE_ONLY' if observe_only else 'TRUSTED'})")

    persistence = report["persistence"]
    stored = persistence.get("sqlite_latest") or {}
    gate("Q3 risk persisted to SQLite",
         persistence["sqlite_rows"] > 0 and stored.get("risk_level") is not None,
         f"rows={persistence['sqlite_rows']}, level={stored.get('risk_level')}")
    return verdict


# --------------------------------------------------------------------------- #
def run_audit(args, limit: int, mmwave_path: Path, co2_index, mm_all) -> dict[str, Any]:
    mm_records = mm_all[:limit]
    risk_engine = None
    if args.risk == "legacy":
        from risk.engine import SafeNestRiskEngine

        risk_engine = SafeNestRiskEngine()

    with ReplayAudit(
        inject_presence=True if args.inject_presence else None,
        inject_humidity=args.inject_humidity,
        risk_engine=risk_engine,
    ) as audit:
        audit.send_thermal(
            mm_records[0]["sequence"] if mm_records else 1, shape=args.thermal_shape
        )

        # The firmware republishes the last valid CO2 reading on every packet;
        # the per-sensor log files only record the sequences where a new
        # measurement event landed, so carry the last one forward.
        def frames() -> Iterator[bytes]:
            held: dict[str, Any] | None = None
            for record in mm_records:
                sequence = int(record["sequence"])
                if sequence in co2_index:
                    held = co2_index[sequence]
                yield telemetry_frame(
                    record,
                    held,
                    inject_presence=True if args.inject_presence else None,
                    inject_humidity=args.inject_humidity,
                )

        def evaluate_live():
            # Refresh thermal so its 3 s TTL has not expired at evaluation time.
            audit.send_thermal(int(mm_records[-1]["sequence"]) + 1, shape=args.thermal_shape)
            # Drive the production path: state -> AI -> risk -> store -> SQLite.
            publication = audit.runtime.evaluate_once()
            persisted = audit.store.history(limit=1)
            return (
                publication["state"],
                publication["ai"],
                publication["risk"],
                {
                    "publication_revision": publication.get("publication_revision"),
                    "sqlite_rows": len(persisted),
                    "sqlite_latest": persisted[0] if persisted else None,
                },
            )

        state, ai, risk, persistence = audit.replay(frames(), evaluate_live)

    thermal_files = sorted((REPO_ROOT / "data" / "thermal").glob("*"))
    report: dict[str, Any] = {
        "generated_at": time.time(),
        "capture": {
            "mmwave_file": str(mmwave_path.relative_to(REPO_ROOT)),
            "co2_file": args.resolved_co2_file,
            "thermal_file": (
                str(thermal_files[0]) if thermal_files
                else "NONE (synthetic frame used; no real MLX90640 capture committed)"
            ),
            "record_limit": limit,
            "mmwave_stats": summarize_capture(mm_records),
            "injected_presence": bool(args.inject_presence),
            "injected_humidity": args.inject_humidity,
            "thermal_frame": f"SYNTHETIC_{args.thermal_shape.upper()}",
        },
        "state": state,
        "ai": ai,
        "risk": risk,
        "persistence": persistence,
    }
    report["verdict"] = build_verdict(report)
    return report


def render_sweep(rows: list[dict[str, Any]]) -> str:
    """Model behaviour across independent replay windows of the same capture."""

    lines = ["=" * 108,
             "mmWave behaviour sweep over independent windows of one committed capture",
             "=" * 108,
             f"{'records':>8} {'M-N9 class':<18} {'conf':>6} {'margin':>7}"
             f" {'spectral rpm':>12} {'band':>6} {'hold':>5}"
             f"  {'risk src':<10} {'risk':>7} {'level':<10} health"]
    for row in rows:
        lines.append(
            f"{row['records']:>8} {row['mmwave_state'][:18]:<18} {row['mmwave_confidence']:>6}"
            f" {row['mmwave_margin']:>7} {row['spectral_rate_rpm']:>12}"
            f" {row['spectral_band_power_fraction']:>6} {str(row['spectral_hold_evidence'])[:5]:>5}"
            f"  {row['respiration_rate_source'][:10]:<10} {row['risk_score']:>7}"
            f" {row['risk_level'][:10]:<10} {row['system_health']}"
        )
    published = [r for r in rows if "APNEA" in r["mmwave_state"]]
    refused = [r for r in rows if r["mmwave_error"] == "APNEA_CONTRADICTED_BY_SPECTRUM"]
    spectral = [r for r in rows if r["spectral_rate_rpm"] not in ("-", None)]
    lines.append("")
    lines.append(f"  windows                              : {len(rows)}")
    lines.append(f"  APNEA-proxy refused by spectrum      : {len(refused)}")
    lines.append(f"  APNEA-proxy still published          : {len(published)}"
                 "   (window contains a quiet stretch, so a real hold cannot be excluded)")
    lines.append(f"  spectral estimate available          : {len(spectral)} / {len(rows)}")
    lines.append("")
    lines.append("  The spectral column is a deterministic DSP readout of the same canonical")
    lines.append("  window, not a model. It is what the risk engine uses for the respiration")
    lines.append("  rule while M-N9 stays DEVICE_VALIDATED: NO, and it is what refuses an")
    lines.append("  APNEA-proxy class on a window that has no quiet stretch at all.")
    lines.append("=" * 108)
    return "\n".join(lines)


def sweep_row(report: dict[str, Any]) -> dict[str, Any]:
    mm = report["ai"]["ai"]["mmwave"]
    co2 = report["ai"]["ai"]["co2"]
    risk = report["risk"]
    meta = mm.get("metadata") or {}
    mm_component = ((risk.get("components") or {}).get("mmwave") or {})
    component_meta = mm_component.get("metadata") or {}
    probabilities = meta.get("probabilities") or meta.get("refused_probabilities") or []
    ordered = sorted((float(p) for p in probabilities), reverse=True)
    margin = round(ordered[0] - ordered[1], 4) if len(ordered) >= 2 else None
    return {
        "records": report["capture"]["record_limit"],
        "mmwave_state": str(mm.get("state")),
        "mmwave_confidence": _num(mm.get("confidence")),
        "mmwave_margin": "-" if margin is None else f"{margin:.3f}",
        "mmwave_decisive": "yes" if margin is not None and margin >= 0.15 else "no",
        "mmwave_probabilities": [round(float(p), 5) for p in probabilities],
        "mmwave_error": str(mm.get("error")) if mm.get("error") else None,
        "spectral_rate_rpm": _num(meta.get("spectral_rate_rpm")),
        "spectral_band_power_fraction": _num(meta.get("spectral_band_power_fraction")),
        "spectral_hold_evidence": meta.get("spectral_hold_evidence"),
        "spectral_contradicts_apnea": meta.get("spectral_contradicts_apnea"),
        "respiration_rate_source": str(
            component_meta.get("respiration_rate_source") or mm_component.get("source") or "-"
        ),
        "co2_state": str(co2.get("state")),
        "co2_source": str(co2.get("source")),
        "risk_score": _num(risk.get("risk_score")),
        "risk_level": str(risk.get("risk_level")),
        "system_health": str(risk.get("system_health")),
        "component_status": risk.get("component_status"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmwave", type=Path, default=None)
    parser.add_argument("--co2", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=2400,
                        help="max telemetry records to replay. The default clears both the "
                             "M-N4 30 s window and the C-B6 150 s CO2 history in the "
                             "committed 20260817 capture.")
    parser.add_argument("--sweep", default=None,
                        help="comma-separated record limits; reports model behaviour across "
                             "independent windows of the same capture")
    parser.add_argument("--inject-presence", action="store_true",
                        help="synthesize mmwave.human_detected_raw=true. The committed "
                             "captures predate firmware 1.3.0, which is what added the "
                             "field; this compensates for the capture, not for the "
                             "firmware. Retiring this flag needs a re-capture on >=1.3.0 "
                             "hardware, not a code change.")
    parser.add_argument("--inject-humidity", type=float, default=None,
                        help="synthesize humidity_percent; the C-B6 contract forbids it, so this "
                             "only demonstrates that it is ignored")
    parser.add_argument("--thermal-shape", choices=("upright", "lying", "flat"),
                        default="upright",
                        help="synthetic thermal geometry (no real capture is committed)")
    parser.add_argument("--risk", choices=("legacy", "v1"), default="v1",
                        help="risk formula under audit (runtime default is v1)")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    mmwave_path = args.mmwave or newest_capture(MMWAVE_DIR, "_mmwave")
    co2_path = args.co2 or newest_capture(CO2_DIR, "_co2")
    if mmwave_path is None:
        print("no committed mmwave field capture found under data/mmwave", file=sys.stderr)
        return 2
    args.resolved_co2_file = str(co2_path.relative_to(REPO_ROOT)) if co2_path else "NONE"

    mm_all = load_capture(mmwave_path)
    co2_index = co2_by_sequence(load_capture(co2_path)) if co2_path else {}

    if args.sweep:
        limits = [int(item) for item in args.sweep.split(",") if item.strip()]
        rows, reports = [], []
        for limit in limits:
            report = run_audit(args, limit, mmwave_path, co2_index, mm_all)
            rows.append(sweep_row(report))
            reports.append(report)
        print(render_sweep(rows))
        if args.json:
            args.json.write_text(
                json.dumps({"sweep": rows, "reports": reports}, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"\nwrote {args.json}")
        return 0

    report = run_audit(args, args.limit, mmwave_path, co2_index, mm_all)
    print(render(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0 if all(item["result"] == "PASS" for item in report["verdict"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
