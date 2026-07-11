"""Helpers for spinning up the PWA app against an in-memory state for tests.

Used by the Playwright UI suite. Builds a minimal FastAPI app that serves
both the static PWA shell and the JSON API, with auth and the database
state-manager swapped out for in-memory stubs.
"""
from __future__ import annotations

import socket
import threading
import time
from datetime import date, datetime, time as _dtime, timedelta, timezone
from zoneinfo import ZoneInfo

import uvicorn
from fastapi import FastAPI

from vessel.auth import require_user_id
from vessel.config import get_settings
from vessel.models import CalendarEvent, StateData
from vessel.pwa.routes import router as pwa_router, mount_static


FAKE_USER = "test-user"


def test_now_local() -> datetime:
    """The pinned "now" the test fixture hands `_client_now`. Tests
    that build calendar events relative to "now" should anchor on
    THIS instead of `datetime.now()` so their events stay aligned
    with the server's notion of the current moment, regardless of
    actual wall-clock time."""
    tz = ZoneInfo(get_settings().timezone)
    return datetime.combine(datetime.now(tz).date(), _dtime(12, 0), tzinfo=tz)


def make_state_with_one_open_event(today: date) -> StateData:
    return StateData(
        calendar=[
            CalendarEvent(
                id="ev-open",
                title="open event",
                description="",
                start=datetime.combine(
                    today + timedelta(days=2), datetime.min.time()
                ).replace(hour=10, tzinfo=timezone.utc),
                end=datetime.combine(
                    today + timedelta(days=2), datetime.min.time()
                ).replace(hour=11, tzinfo=timezone.utc),
            ),
        ]
    )


def make_state_with_calendar_only(today: date) -> StateData:
    """Minimal state with a single calendar event for UI tests."""
    return StateData(
        calendar=[
            CalendarEvent(
                id="ev-gym",
                title="Go to the gym",
                description="cardio + lift",
                start=datetime.combine(
                    today + timedelta(days=1), datetime.min.time()
                ).replace(hour=8, tzinfo=timezone.utc),
                end=datetime.combine(
                    today + timedelta(days=1), datetime.min.time()
                ).replace(hour=9, tzinfo=timezone.utc),
            ),
        ]
    )


def make_state_full(today: date) -> StateData:
    """State with a mix of upcoming and completed events for UI tests."""
    base = datetime(2026, 4, 25, tzinfo=timezone.utc)
    return StateData(
        calendar=[
            CalendarEvent(
                id="ev-today",
                title="anytime today event",
                description="",
                start=datetime.combine(today, datetime.min.time()).replace(
                    hour=14, tzinfo=timezone.utc
                ),
                end=datetime.combine(today, datetime.min.time()).replace(
                    hour=14, minute=30, tzinfo=timezone.utc
                ),
            ),
            CalendarEvent(
                id="ev-tomorrow",
                title="tomorrow event",
                description="",
                start=datetime.combine(
                    today + timedelta(days=1), datetime.min.time()
                ).replace(hour=10, tzinfo=timezone.utc),
                end=datetime.combine(
                    today + timedelta(days=1), datetime.min.time()
                ).replace(hour=11, tzinfo=timezone.utc),
            ),
            CalendarEvent(
                id="ev-completed",
                title="completed event",
                description="",
                start=datetime.combine(today, datetime.min.time()).replace(
                    hour=9, tzinfo=timezone.utc
                ),
                end=datetime.combine(today, datetime.min.time()).replace(
                    hour=10, tzinfo=timezone.utc
                ),
                completed_at=base,
            ),
        ]
    )


# Backwards-compat alias used by some test helpers
make_state_with_one_open_task = make_state_with_one_open_event


def build_app(
    initial_state: StateData, *, chat_client=None
) -> tuple[FastAPI, dict]:
    """Build a FastAPI app whose state lives in-memory.

    The returned dict has key 'state' that tests can mutate to simulate
    server-side changes between page interactions.

    `chat_client` (optional) is the OpenAI-shaped client `/api/chat`
    will hand to `run_chat_assistant`. Tests pass a scripted fake so
    no real LLM is called.
    """
    from vessel.config import get_settings as _gs
    _settings = _gs()
    if _settings.bedtime_hour < 23:
        _settings.bedtime_hour = 23

    import vessel.pwa.routes as routes_mod

    def _fixed_client_now(client_now_header=None):
        return test_now_local()

    routes_mod._client_now = _fixed_client_now  # type: ignore[assignment]

    box: dict = {"state": initial_state}

    async def fake_read(_pool, _user_id):
        return StateData.model_validate(box["state"].model_dump(mode="python"))

    async def fake_write(_pool, _user_id, new_state):
        box["state"] = new_state

    async def fake_pool():
        return None

    routes_mod.state_manager.read = fake_read  # type: ignore[assignment]
    routes_mod.state_manager.write = fake_write  # type: ignore[assignment]
    routes_mod.get_pool = fake_pool  # type: ignore[assignment]

    # Inject (or clear) the chat endpoint's OpenAI client.
    routes_mod._set_chat_client_for_test(chat_client)

    app = FastAPI()
    app.include_router(pwa_router)
    app.dependency_overrides[require_user_id] = lambda: FAKE_USER
    mount_static(app)
    return app, box


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class ServerThread(threading.Thread):
    def __init__(self, app: FastAPI, port: int):
        super().__init__(daemon=True)
        self.config = uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="warning"
        )
        self.server = uvicorn.Server(self.config)

    def run(self) -> None:
        self.server.run()

    def stop(self) -> None:
        self.server.should_exit = True


def start_server(app: FastAPI) -> tuple[ServerThread, str]:
    port = _free_port()
    thread = ServerThread(app, port)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        raise RuntimeError("server did not start")
    return thread, base_url
