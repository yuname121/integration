#!/usr/bin/env python3
"""Phase 3 Raspberry Pi receiver entry point."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import signal
import sys

if __package__ in {None, ""}:
    from pathlib import Path

    package_parent = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(package_parent))

from gateway.protocol import TelemetryPayload, ThermalFrame
from gateway.receiver import SafeNestTCPServer


def main() -> int:
    parser = argparse.ArgumentParser(description="SafeNest TCP v1 receiver")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--packet-deadline", type=float, default=5.0)
    args = parser.parse_args()

    def on_packet(packet, peer) -> None:
        if isinstance(packet, TelemetryPayload):
            data = asdict(packet)
            data["kind"] = "telemetry"
        elif isinstance(packet, ThermalFrame):
            data = {
                "kind": "thermal",
                "sequence": packet.frame_sequence,
                "uptime_ms": packet.uptime_ms,
                "width": packet.width,
                "height": packet.height,
                "minimum_raw": packet.minimum_raw,
                "maximum_raw": packet.maximum_raw,
            }
        else:  # pragma: no cover - protected by protocol decoder
            return
        data["peer"] = f"{peer[0]}:{peer[1]}"
        print(json.dumps(data, ensure_ascii=False), flush=True)

    def on_error(error: Exception, peer) -> None:
        label = "listener" if peer is None else f"{peer[0]}:{peer[1]}"
        print(f"[{label}] {type(error).__name__}: {error}", file=sys.stderr)

    server = SafeNestTCPServer(
        on_packet,
        host=args.host,
        port=args.port,
        on_error=on_error,
        packet_deadline_seconds=args.packet_deadline,
    )

    def stop(_signum, _frame) -> None:
        server.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f"SafeNest TCP receiver listening on {args.host}:{args.port}", file=sys.stderr)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
