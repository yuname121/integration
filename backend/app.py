"""FastAPI application factory; framework imports remain optional for core tests."""

import asyncio
from contextlib import asynccontextmanager
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


class BackendDependencyError(RuntimeError):
    pass


def create_app(
    runtime: SafeNestRuntime | None = None,
    *,
    store: RuntimeStore | None = None,
    start_runtime: bool = True,
    room: str = "밀폐공간 A-01",
    websocket_interval_seconds: float = 0.25,
) -> Any:
    if websocket_interval_seconds <= 0:
        raise ValueError("websocket interval must be positive")
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as error:
        raise BackendDependencyError(
            "FastAPI is not installed; install requirements-backend.txt"
        ) from error

    if runtime is not None and store is not None and runtime.store is not store:
        raise ValueError("runtime and store must reference the same RuntimeStore")
    selected_store = store or (runtime.store if runtime is not None else RuntimeStore())
    selected_runtime = runtime or SafeNestRuntime(store=selected_store)

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

    app = FastAPI(
        title="SafeNest Integrated Backend",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.safenest_runtime = selected_runtime
    app.state.safenest_store = selected_store

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

    @app.get("/health")
    def health() -> dict[str, Any]:
        return health_document(selected_store.diagnostics(), selected_runtime.receiver_stats())

    @app.websocket("/ws")
    async def websocket_status(websocket: WebSocket) -> None:
        await websocket.accept()
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

    return app
