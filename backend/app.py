"""FastAPI application factory; framework imports remain optional for core tests."""

import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any

from backend.runtime import SafeNestRuntime
from backend.store import RuntimeStore
from backend.views import (
    ROUTE_CONTRACTS,
    events_document,
    health_document,
    history_document,
    legacy_state_document,
    sensors_document,
    status_document,
)
from services.buzzer import BuzzerProtocol, create_buzzer_from_env
from services.emergency import EmergencyActionError, EmergencyActionService
from services.sms_service import SMSProvider


class BackendDependencyError(RuntimeError):
    pass


def create_app(
    runtime: SafeNestRuntime | None = None,
    *,
    store: RuntimeStore | None = None,
    start_runtime: bool = True,
    room: str = "밀폐공간 A-01",
    websocket_interval_seconds: float = 0.25,
    emergency_service: EmergencyActionService | None = None,
    sms_provider: SMSProvider | None = None,
    buzzer: BuzzerProtocol | None = None,
    sms_cooldown_seconds: float | None = None,
) -> Any:
    if websocket_interval_seconds <= 0:
        raise ValueError("websocket interval must be positive")
    try:
        from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as error:
        raise BackendDependencyError(
            "FastAPI is not installed; install requirements-backend.txt"
        ) from error

    if runtime is not None and store is not None and runtime.store is not store:
        raise ValueError("runtime and store must reference the same RuntimeStore")
    selected_store = store or (runtime.store if runtime is not None else RuntimeStore())
    selected_runtime = runtime or SafeNestRuntime(store=selected_store)
    selected_sms_cooldown = (
        _float_env("SAFENEST_SMS_COOLDOWN_SECONDS", 60.0)
        if sms_cooldown_seconds is None
        else float(sms_cooldown_seconds)
    )
    selected_buzzer = buzzer
    if emergency_service is None:
        selected_buzzer = selected_buzzer or create_buzzer_from_env()
        selected_emergency = EmergencyActionService(
            selected_store,
            sms_provider=sms_provider,
            room=room,
            sms_cooldown_seconds=selected_sms_cooldown,
            buzzer=selected_buzzer,
        )
    else:
        selected_emergency = emergency_service
    if selected_buzzer is not None:
        selected_store.attach_buzzer(selected_buzzer)

    @asynccontextmanager
    async def lifespan(_app):
        if start_runtime:
            selected_runtime.start()
        try:
            yield
        finally:
            if start_runtime:
                selected_runtime.stop()
            close = getattr(selected_store, "close", None)
            if callable(close):
                close()
            close_buzzer = getattr(selected_buzzer, "close", None)
            if callable(close_buzzer):
                close_buzzer()

    app = FastAPI(
        title="SafeNest Integrated Backend",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.safenest_runtime = selected_runtime
    app.state.safenest_store = selected_store
    app.state.safenest_emergency = selected_emergency

    dashboard_dir = Path(__file__).resolve().parent.parent / "web" / "dashboard"
    app.mount(
        "/dashboard/assets",
        StaticFiles(directory=str(dashboard_dir)),
        name="dashboard-assets",
    )

    @app.get("/dashboard", include_in_schema=False)
    @app.get("/dashboard/", include_in_schema=False)
    def dashboard() -> Any:
        return FileResponse(dashboard_dir / "index.html")

    @app.get("/")
    def root() -> dict[str, object]:
        return {"service": "SafeNest", "routes": ROUTE_CONTRACTS}

    @app.get("/api/status")
    def api_status() -> dict[str, Any]:
        return status_document(selected_store.latest())

    @app.get("/api/sensors")
    def api_sensors() -> dict[str, Any]:
        return sensors_document(selected_store.latest())

    @app.get("/api/events")
    def api_events(limit: int = 100) -> dict[str, Any]:
        try:
            events = selected_store.events(limit)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        diagnostics = selected_store.diagnostics()
        database = diagnostics.get("database")
        persistent = isinstance(database, dict) and database.get("available") is True
        return events_document(events, persistent=persistent)

    @app.get("/api/history")
    def api_history(limit: int = 100) -> dict[str, Any]:
        try:
            history = selected_store.history(limit)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        diagnostics = selected_store.diagnostics()
        database = diagnostics.get("database")
        persistent = isinstance(database, dict) and database.get("available") is True
        return history_document(history, persistent=persistent)

    @app.get("/api/state")
    def api_state_compatibility() -> dict[str, Any]:
        return legacy_state_document(selected_store.latest(), room=room)

    @app.get("/api/emergency/state")
    def api_emergency_state() -> dict[str, Any]:
        return {"ok": True, "emergency": selected_store.emergency_snapshot()}

    def action_error(error: EmergencyActionError) -> JSONResponse:
        body: dict[str, Any] = {
            "ok": False,
            "error_code": error.code,
            "message": str(error),
        }
        if error.retry_after_seconds is not None:
            body["retry_after_seconds"] = round(max(0.0, error.retry_after_seconds), 3)
        return JSONResponse(status_code=error.status_code, content=body)

    async def json_payload(request: Request) -> dict[str, Any]:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="JSON object body is required")
        return payload

    @app.post("/api/emergency/119/simulation/start", response_model=None)
    def start_119_simulation() -> dict[str, Any] | JSONResponse:
        try:
            return selected_emergency.start_119_simulation()
        except EmergencyActionError as error:
            return action_error(error)

    @app.post("/api/emergency/119/simulation/complete", response_model=None)
    async def complete_119_simulation(request: Request) -> dict[str, Any] | JSONResponse:
        payload = await json_payload(request)
        try:
            return selected_emergency.complete_119_simulation(str(payload.get("simulation_id", "")))
        except EmergencyActionError as error:
            return action_error(error)

    @app.post("/api/emergency/contact", response_model=None)
    async def contact_manager(request: Request) -> dict[str, Any] | JSONResponse:
        payload = await json_payload(request)
        try:
            return selected_emergency.send_manager_sms(
                idempotency_key=payload.get("idempotency_key"),
            )
        except EmergencyActionError as error:
            return action_error(error)

    @app.post("/api/emergency/acknowledge", response_model=None)
    def acknowledge_alarm() -> dict[str, Any] | JSONResponse:
        try:
            return selected_emergency.acknowledge_alarm()
        except (EmergencyActionError, RuntimeError) as error:
            if isinstance(error, EmergencyActionError):
                return action_error(error)
            return action_error(EmergencyActionError("DANGER_NOT_ACTIVE", str(error), status_code=409))

    @app.post("/api/emergency/voice", response_model=None)
    async def record_voice(request: Request) -> dict[str, Any] | JSONResponse:
        payload = await json_payload(request)
        try:
            return selected_emergency.record_voice_event(str(payload.get("action", "")))
        except EmergencyActionError as error:
            return action_error(error)

    @app.post("/api/client-connection", response_model=None)
    async def record_client_connection(request: Request) -> dict[str, Any] | JSONResponse:
        payload = await json_payload(request)
        try:
            return selected_emergency.record_client_connection(
                source=str(payload.get("source", "")),
                status=str(payload.get("status", "")),
            )
        except EmergencyActionError as error:
            return action_error(error)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return health_document(selected_store.diagnostics(), selected_runtime.receiver_stats())

    @app.websocket("/ws")
    async def websocket_status(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            selected_emergency.record_client_connection(source="websocket", status="online")
        except EmergencyActionError:
            pass
        last_revision = -1
        try:
            while True:
                publication = selected_store.latest()
                revision = publication.get("publication_revision", 0) if publication else 0
                if revision != last_revision:
                    await websocket.send_json(status_document(publication))
                    last_revision = int(revision)
                await asyncio.sleep(websocket_interval_seconds)
        except (WebSocketDisconnect, RuntimeError):
            return
        finally:
            try:
                selected_emergency.record_client_connection(source="websocket", status="offline")
            except EmergencyActionError:
                pass

    return app


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default
