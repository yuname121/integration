"""Reconnect-safe Raspberry Pi TCP receiver for SafeNest protocol v1."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import threading
from typing import Callable

from .protocol import (
    ConnectionClosed,
    DecodedPacket,
    ProtocolError,
    SequenceTracker,
    TelemetryPayload,
    ThermalFrame,
    read_packet,
)


PacketCallback = Callable[[DecodedPacket, tuple[str, int]], None]
ErrorCallback = Callable[[Exception, tuple[str, int] | None], None]


@dataclass
class ReceiverStats:
    connections: int = 0
    disconnects: int = 0
    telemetry_packets: int = 0
    thermal_packets: int = 0
    sequence_gaps: int = 0
    protocol_errors: int = 0
    callback_errors: int = 0


class ConnectionProcessor:
    def __init__(
        self,
        on_packet: PacketCallback,
        *,
        on_error: ErrorCallback | None = None,
        packet_deadline_seconds: float = 5.0,
        stats: ReceiverStats | None = None,
    ) -> None:
        self.on_packet = on_packet
        self.on_error = on_error
        self.packet_deadline_seconds = packet_deadline_seconds
        self.stats = stats or ReceiverStats()

    def process(
        self,
        connection: socket.socket,
        peer: tuple[str, int],
        stop_event: threading.Event | None = None,
    ) -> None:
        tracker = SequenceTracker()
        self.stats.connections += 1
        try:
            while stop_event is None or not stop_event.is_set():
                packet = read_packet(
                    connection,
                    deadline_seconds=self.packet_deadline_seconds,
                )
                self.stats.sequence_gaps += tracker.accept(packet.header)
                if isinstance(packet, TelemetryPayload):
                    self.stats.telemetry_packets += 1
                elif isinstance(packet, ThermalFrame):
                    self.stats.thermal_packets += 1
                try:
                    self.on_packet(packet, peer)
                except Exception as exc:  # Consumer failure must not corrupt framing.
                    self.stats.callback_errors += 1
                    self._report(exc, peer)
        except ConnectionClosed as exc:
            self.stats.disconnects += 1
            self._report(exc, peer)
        except ProtocolError as exc:
            self.stats.protocol_errors += 1
            self._report(exc, peer)

    def _report(self, error: Exception, peer: tuple[str, int] | None) -> None:
        if self.on_error is not None:
            self.on_error(error, peer)


class SafeNestTCPServer:
    """Accept one ESP32 stream at a time and continue after disconnects."""

    def __init__(
        self,
        on_packet: PacketCallback,
        *,
        host: str = "0.0.0.0",
        port: int = 9000,
        on_error: ErrorCallback | None = None,
        packet_deadline_seconds: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.stop_event = threading.Event()
        self.stats = ReceiverStats()
        self.processor = ConnectionProcessor(
            on_packet,
            on_error=on_error,
            packet_deadline_seconds=packet_deadline_seconds,
            stats=self.stats,
        )
        self._listener: socket.socket | None = None

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            self._listener = listener
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.host, self.port))
            self.port = int(listener.getsockname()[1])
            listener.listen(2)
            listener.settimeout(0.5)
            while not self.stop_event.is_set():
                try:
                    connection, peer = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self.stop_event.is_set():
                        break
                    raise
                with connection:
                    connection.settimeout(0.25)
                    self.processor.process(connection, peer, self.stop_event)
            self._listener = None

    def stop(self) -> None:
        self.stop_event.set()
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass

