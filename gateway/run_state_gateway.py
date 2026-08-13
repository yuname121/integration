#!/usr/bin/env python3
"""Run the PHASE 3 receiver with the PHASE 4 central sensor state manager."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time

if __package__ in {None, ""}:
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from gateway.protocol import ProtocolError
from gateway.receiver import SafeNestTCPServer
from state.manager import SensorStateManager


def main() -> int:
    parser = argparse.ArgumentParser(description="SafeNest receiver + sensor state")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--packet-deadline", type=float, default=5.0)
    parser.add_argument("--snapshot-interval", type=float, default=1.0)
    args = parser.parse_args()
    if args.snapshot_interval <= 0:
        parser.error("--snapshot-interval must be positive")

    manager = SensorStateManager()

    def on_packet(packet, peer) -> None:
        manager.ingest(packet, peer)

    def on_error(error: Exception, peer) -> None:
        if peer is not None and isinstance(error, ProtocolError):
            manager.mark_peer_disconnected(peer)
        label = "listener" if peer is None else f"{peer[0]}:{peer[1]}"
        print(f"[{label}] {type(error).__name__}: {error}", file=sys.stderr)

    server = SafeNestTCPServer(
        on_packet,
        host=args.host,
        port=args.port,
        on_error=on_error,
        packet_deadline_seconds=args.packet_deadline,
    )
    receiver_thread = threading.Thread(target=server.serve_forever, daemon=True)
    receiver_thread.start()

    stopping = False

    def stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True
        server.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while not stopping:
            print(json.dumps(manager.snapshot(), ensure_ascii=False), flush=True)
            time.sleep(args.snapshot_interval)
    finally:
        server.stop()
        receiver_thread.join(timeout=args.packet_deadline + 1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
