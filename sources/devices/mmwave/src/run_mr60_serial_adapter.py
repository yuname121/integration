#!/usr/bin/env python3
"""Read ESP JSONL telemetry and emit Pi-ready SafeNest mmWave packets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))

from devices.mmwave.src.mr60_esp_adapter import MR60ESPAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--port", help="ESP serial port, e.g. /dev/ttyUSB0")
    source.add_argument("--replay", type=Path, help="Recorded ESP JSONL")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--allow-legacy-provenance", action="store_true",
        help="Allow archived records without the current ESP schema/firmware/config identifiers",
    )
    return parser.parse_args()


def serial_lines(port: str, baud: int):
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("pyserial is required for --port: pip install pyserial") from exc
    with serial.Serial(port, baudrate=baud, timeout=2.0) as stream:
        while True:
            raw = stream.readline()
            if raw:
                yield raw.decode("utf-8", errors="replace")


def replay_lines(path: Path):
    with path.open(encoding="utf-8") as stream:
        yield from stream


def main() -> int:
    args = parse_args()
    adapter = MR60ESPAdapter(
        args.config, strict_provenance=not args.allow_legacy_provenance,
    )
    lines = serial_lines(args.port, args.baud) if args.port else replay_lines(args.replay)
    for line in lines:
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict) or raw.get("kind") not in (None, "sensor"):
            continue
        packet = adapter.process(raw)
        print(adapter.to_json(packet), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
