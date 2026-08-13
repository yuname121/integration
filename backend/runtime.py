"""Long-running TCP → state → AI → risk publication service."""

from __future__ import annotations

from dataclasses import asdict
import threading

from ai.pipeline import OnDeviceAIPipeline
from backend.store import RuntimeStore
from gateway.protocol import ConnectionClosed, ProtocolError
from gateway.receiver import SafeNestTCPServer
from risk.engine import SafeNestRiskEngine
from state.manager import SensorStateManager


class SafeNestRuntime:
    def __init__(
        self,
        *,
        sensor_host: str = "0.0.0.0",
        sensor_port: int = 9000,
        packet_deadline_seconds: float = 5.0,
        evaluation_interval_seconds: float = 1.0,
        manager: SensorStateManager | None = None,
        ai_pipeline: OnDeviceAIPipeline | None = None,
        risk_engine: SafeNestRiskEngine | None = None,
        store: RuntimeStore | None = None,
    ) -> None:
        if evaluation_interval_seconds <= 0:
            raise ValueError("evaluation interval must be positive")
        self.manager = manager or SensorStateManager()
        self.ai_pipeline = ai_pipeline or OnDeviceAIPipeline(self.manager)
        self.risk_engine = risk_engine or SafeNestRiskEngine()
        self.store = store or RuntimeStore()
        self.evaluation_interval_seconds = float(evaluation_interval_seconds)
        self.server = SafeNestTCPServer(
            self._on_packet,
            host=sensor_host,
            port=sensor_port,
            on_error=self._on_receiver_error,
            packet_deadline_seconds=packet_deadline_seconds,
        )
        self._stop_event = threading.Event()
        self._receiver_thread: threading.Thread | None = None
        self._evaluation_thread: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                return
            self._started = True
            self._stop_event.clear()
            self.evaluate_once()
            self._receiver_thread = threading.Thread(
                target=self.server.serve_forever,
                name="safenest-tcp-receiver",
                daemon=True,
            )
            self._evaluation_thread = threading.Thread(
                target=self._evaluation_loop,
                name="safenest-state-publisher",
                daemon=True,
            )
            self._receiver_thread.start()
            self._evaluation_thread.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            if not self._started:
                return
            self._started = False
            self._stop_event.set()
            self.server.stop()
            receiver = self._receiver_thread
            evaluator = self._evaluation_thread
        if evaluator is not None:
            evaluator.join(timeout=self.evaluation_interval_seconds + 1.0)
        if receiver is not None:
            receiver.join(timeout=self.server.processor.packet_deadline_seconds + 1.0)

    def receiver_stats(self) -> dict[str, object]:
        return {
            **asdict(self.server.stats),
            "host": self.server.host,
            "port": self.server.port,
            "runtime_started": self._started,
        }

    def _on_packet(self, packet, peer) -> None:
        self.manager.ingest(packet, peer)

    def _on_receiver_error(self, error: Exception, peer) -> None:
        if peer is not None and isinstance(error, ProtocolError):
            self.manager.mark_peer_disconnected(peer)
        if isinstance(error, ConnectionClosed):
            return
        source = "listener" if peer is None else f"receiver:{peer[0]}:{peer[1]}"
        self.store.record_runtime_error(source, error)

    def _evaluation_loop(self) -> None:
        while not self._stop_event.wait(self.evaluation_interval_seconds):
            try:
                self.evaluate_once()
            except Exception as error:
                self.store.record_runtime_error("evaluation", error)

    def evaluate_once(self) -> dict[str, object]:
        state = self.manager.snapshot()
        ai = self.ai_pipeline.evaluate(state, self.manager.latest_thermal_frame())
        risk = self.risk_engine.evaluate(state, ai)
        return self.store.publish(state, ai, risk.to_dict())
