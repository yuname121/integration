#!/usr/bin/env python3
"""SafeNest LCD display and laptop control server.

Runs entirely on the Raspberry Pi with Python's standard library.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
STATE_FILE = ROOT / "state.json"
ALLOWED_STATES = {
    "normal-empty",
    "normal-occupied",
    "warning",
    "danger",
    "emergency",
    "offline",
}
DEFAULT_STATE = {
    "state": "normal-empty",
    "room": "밀폐공간 A-01",
    "revision": 1,
    "updated_at": int(time.time()),
}
STATE_LOCK = threading.Lock()


class BuzzerController:
    """Drive a passive piezo buzzer with GPIO Zero's BCM pin numbering."""

    def __init__(
        self,
        pin: int = 18,
        frequency_hz: float = 880.0,
        enabled: bool = True,
    ) -> None:
        self.pin = pin
        self.frequency_hz = frequency_hz
        self.enabled = enabled
        self.available = False
        self.sounding = False
        self.error: str | None = None
        self._device: object | None = None
        self._lock = threading.Lock()

        if not enabled:
            return

        try:
            from gpiozero import TonalBuzzer

            self._device = TonalBuzzer(pin)
            self._device.stop()
            self.available = True
            print(f"피에조 부저 준비 완료: BCM GPIO{pin}, {frequency_hz:g} Hz")
        except Exception as error:  # GPIO support differs between Pi models/images.
            self.error = str(error)
            print(f"피에조 부저 초기화 실패(GPIO{pin}): {error}")

    def set_emergency(self, emergency: bool) -> None:
        with self._lock:
            if self._device is None:
                self.sounding = False
                return
            if emergency == self.sounding:
                return
            try:
                if emergency:
                    self._device.play(self.frequency_hz)
                else:
                    self._device.stop()
                self.sounding = emergency
                self.error = None
                print("피에조 부저: 긴급 경보 ON" if emergency else "피에조 부저: OFF")
            except Exception as error:
                self.sounding = False
                self.error = str(error)
                print(f"피에조 부저 제어 실패(GPIO{self.pin}): {error}")

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "available": self.available,
                "pin_bcm": self.pin,
                "frequency_hz": self.frequency_hz,
                "sounding": self.sounding,
                "error": self.error,
            }

    def close(self) -> None:
        with self._lock:
            if self._device is None:
                return
            try:
                self._device.stop()
                self._device.close()
            except Exception as error:
                self.error = str(error)
                print(f"피에조 부저 종료 실패(GPIO{self.pin}): {error}")
            finally:
                self.sounding = False
                self._device = None


def load_state() -> dict[str, object]:
    try:
        saved = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        state = saved.get("state")
        room = str(saved.get("room", "")).strip()
        if state in ALLOWED_STATES and room:
            return {
                "state": state,
                "room": room[:24],
                "revision": int(saved.get("revision", 1)),
                "updated_at": int(saved.get("updated_at", time.time())),
            }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return DEFAULT_STATE.copy()


APP_STATE = load_state()



def persist_state() -> None:
    temporary = STATE_FILE.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(APP_STATE, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, STATE_FILE)


def apply_state_change(
    buzzer: BuzzerController,
    new_state: str | None,
    new_room: str | None,
) -> dict[str, object]:
    """Persist one state update and keep the audible alarm in sync with it."""
    with STATE_LOCK:
        if new_state is not None:
            APP_STATE["state"] = new_state
        if new_room is not None:
            APP_STATE["room"] = new_room
        APP_STATE["revision"] = int(APP_STATE["revision"]) + 1
        APP_STATE["updated_at"] = int(time.time())
        try:
            persist_state()
        except OSError as error:
            print(f"상태 파일 저장 실패: {error}")
        buzzer.set_emergency(APP_STATE["state"] == "emergency")
        return APP_STATE.copy()


class SafeNestHandler(BaseHTTPRequestHandler):
    server_version = "SafeNestLCD/1.0"

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {format_string % args}")

    def send_common_headers(self, content_type: str, length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")

    def send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_common_headers(content_type, len(body))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_bytes(body, "application/json; charset=utf-8", status)

    def redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self.redirect("/control")
            return
        if path == "/control":
            path = "/control.html"
        elif path == "/display":
            path = "/display.html"
        elif path in {"/api/state", "/health"}:
            if path == "/health":
                self.send_json(
                    {
                        "ok": True,
                        "buzzer": self.server.buzzer.status(),
                    }
                )
            else:
                with STATE_LOCK:
                    self.send_json(APP_STATE.copy())
            return

        relative = path.lstrip("/")
        requested = (STATIC_DIR / relative).resolve()
        try:
            requested.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return

        if not requested.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_type = mimetypes.guess_type(requested.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {
            "application/javascript",
            "application/json",
        }:
            content_type += "; charset=utf-8"
        self.send_bytes(requested.read_bytes(), content_type)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/state":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json({"error": "잘못된 요청 길이입니다."}, HTTPStatus.BAD_REQUEST)
            return
        if content_length <= 0 or content_length > 4096:
            self.send_json({"error": "요청 크기가 올바르지 않습니다."}, HTTPStatus.BAD_REQUEST)
            return

        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json({"error": "JSON 형식이 올바르지 않습니다."}, HTTPStatus.BAD_REQUEST)
            return

        if not isinstance(payload, dict):
            self.send_json({"error": "객체 형식의 요청이 필요합니다."}, HTTPStatus.BAD_REQUEST)
            return

        new_state = payload.get("state")
        new_room = payload.get("room")
        if new_state is not None and new_state not in ALLOWED_STATES:
            self.send_json({"error": "지원하지 않는 화면 상태입니다."}, HTTPStatus.BAD_REQUEST)
            return
        if new_room is not None:
            if not isinstance(new_room, str) or not new_room.strip():
                self.send_json({"error": "공간 이름을 입력하세요."}, HTTPStatus.BAD_REQUEST)
                return
            new_room = new_room.strip()[:24]
        if new_state is None and new_room is None:
            self.send_json({"error": "변경할 항목이 없습니다."}, HTTPStatus.BAD_REQUEST)
            return

        response = apply_state_change(self.server.buzzer, new_state, new_room)
        self.send_json(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="SafeNest LCD remote-control server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--buzzer-pin", type=int, default=18)
    parser.add_argument("--buzzer-frequency", type=float, default=880.0)
    parser.add_argument("--disable-buzzer", action="store_true")
    args = parser.parse_args()

    buzzer = BuzzerController(
        pin=args.buzzer_pin,
        frequency_hz=args.buzzer_frequency,
        enabled=not args.disable_buzzer,
    )
    buzzer.set_emergency(APP_STATE["state"] == "emergency")
    server = ThreadingHTTPServer((args.host, args.port), SafeNestHandler)
    server.daemon_threads = True
    server.buzzer = buzzer

    def request_shutdown(signum: int, _frame: object) -> None:
        print(f"종료 신호 수신: {signal.Signals(signum).name}")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    print("SafeNest LCD 서버가 시작되었습니다.")
    print(f"LCD 화면: http://127.0.0.1:{args.port}/display")
    print(f"노트북 제어: http://<라즈베리파이-IP>:{args.port}/control")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        buzzer.close()
        print("SafeNest LCD 서버를 종료했습니다.")


if __name__ == "__main__":
    main()
