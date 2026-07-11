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
from pathlib import Path
from zoneinfo import ZoneInfo

import uvicorn
from fastapi import FastAPI

from vessel.auth import require_user_id
from vessel.config import get_settings
from vessel.models import StateData
from vessel.models.enums import Cadence, ProjectStatus, Tier, TimeWindow
from vessel.models.state import CalendarEvent, Project, Task
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


def make_state_with_one_open_task(today: date) -> StateData:
    project = Project(
        id="p1",
        name="Demo",
        status=ProjectStatus.active,
        tracked=True,
        cadence=Cadence.daily,
        last_touched=datetime(2026, 4, 25, tzinfo=timezone.utc),
    )
    return StateData(
        projects=[project],
        tasks=[
            Task(
                id="t-open",
                project_id="p1",
                title="open task",
                time_window=TimeWindow.evening,
                tier=Tier.must_today,
                due_date=today + timedelta(days=2),
                created_at=datetime(2026, 4, 25, tzinfo=timezone.utc),
            ),
        ],
    )


def make_state_with_calendar_only(today: date) -> StateData:
    """Mirrors the prod scenario the user hit: no tasks, only a calendar event."""
    project = Project(
        id="p_health",
        name="Health",
        status=ProjectStatus.active,
        tracked=True,
        cadence=Cadence.event_driven,
        last_touched=datetime(2026, 4, 25, tzinfo=timezone.utc),
    )
    return StateData(
        projects=[project],
        tasks=[],
        calendar=[
            CalendarEvent(
                id="ev-gym",
                project_id="p_health",
                title="Go to the gym",
                description="cardio + lift",
                start=datetime.combine(
                    today + timedelta(days=1), datetime.min.time()
                ).replace(hour=8, tzinfo=timezone.utc),
                end=datetime.combine(
                    today + timedelta(days=1), datetime.min.time()
                ).replace(hour=9, tzinfo=timezone.utc),
            ),
        ],
    )


def make_state_full(today: date) -> StateData:
    project = Project(
        id="p1",
        name="Demo",
        status=ProjectStatus.active,
        tracked=True,
        cadence=Cadence.daily,
        last_touched=datetime(2026, 4, 25, tzinfo=timezone.utc),
    )
    base = datetime(2026, 4, 25, tzinfo=timezone.utc)
    return StateData(
        projects=[project],
        tasks=[
            Task(
                id="t-today-anytime",
                project_id="p1",
                title="anytime today",
                time_window=TimeWindow.anytime,
                tier=Tier.must_today,
                # Small est so this task always fits the focus window,
                # even when the test runs minutes before bedtime.
                estimated_minutes=5,
                due_date=today,
                created_at=base,
            ),
            Task(
                id="t-tomorrow",
                project_id="p1",
                title="tomorrow task",
                time_window=TimeWindow.workday,
                tier=Tier.flex,
                due_date=today + timedelta(days=1),
                created_at=base,
            ),
            Task(
                id="t-completed",
                project_id="p1",
                title="completed task",
                time_window=TimeWindow.anytime,
                tier=Tier.flex,
                due_date=today,
                created_at=base,
                completed_at=base,
            ),
        ],
    )


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
    # Override bedtime_hour to 23 so focus-mode tests run regardless of
    # wall-clock time. Local .env has BEDTIME_HOUR=21 which makes the
    # focus card disappear after 9pm system time — every focus test
    # would flake when run late at night otherwise.
    from vessel.config import get_settings as _gs
    _settings = _gs()
    if _settings.bedtime_hour < 23:
        _settings.bedtime_hour = 23

    # Always pin "now" to noon today (in the test's configured TZ) so
    # focus-mode tests behave the same regardless of when they run —
    # past bedtime, before workday hours, doesn't matter. Tests that
    # build calendar events relative to "now" should use the
    # `test_now_local()` helper exported below so their events line
    # up with this pinned anchor.
    import vessel.pwa.routes as routes_mod

    def _fixed_client_now(client_now_header=None):
        return test_now_local()

    routes_mod._client_now = _fixed_client_now  # type: ignore[assignment]

    box: dict = {"state": initial_state, "history": []}

    async def fake_read(_pool, _user_id):
        return StateData.model_validate(box["state"].model_dump(mode="python"))

    async def fake_write(_pool, _user_id, new_state):
        box["state"] = new_state

    async def fake_pool():
        return None

    # Patch the symbols the routes module looks up at request time.
    import vessel.pwa.routes as routes_mod

    routes_mod.state_manager.read = fake_read  # type: ignore[assignment]
    routes_mod.state_manager.write = fake_write  # type: ignore[assignment]
    routes_mod.get_pool = fake_pool  # type: ignore[assignment]

    # Also patch insert_and_fanout — complete/push call it.
    async def fake_fanout(_pool, _user_id, _source, _payload):
        return None

    routes_mod.insert_and_fanout = fake_fanout  # type: ignore[assignment]

    # Stub task_history with an in-memory list so complete/skip can
    # archive and uncomplete/unskip can pop without a DB. Same model
    # the SQL table uses (newest first; pop_latest takes one off the
    # head when ids match).
    async def fake_archive(_pool, _user_id, task, closed_kind):
        box["history"].insert(
            0, {"task": task.model_copy(deep=True), "closed_kind": closed_kind}
        )

    async def fake_pop_latest(_pool, _user_id, task_id, closed_kind=None):
        for i, row in enumerate(box["history"]):
            if row["task"].id != task_id:
                continue
            if closed_kind is not None and row["closed_kind"] != closed_kind:
                continue
            return box["history"].pop(i)["task"]
        return None

    async def fake_list_recent(_pool, _user_id, *, limit=100):
        return [
            {
                "task_id": r["task"].id,
                "closed_kind": r["closed_kind"],
                "closed_at": "2026-04-26T00:00:00Z",
                "task": r["task"].model_dump(mode="json"),
            }
            for r in box["history"][:limit]
        ]

    routes_mod.task_history.archive = fake_archive  # type: ignore[assignment]
    routes_mod.task_history.pop_latest = fake_pop_latest  # type: ignore[assignment]
    routes_mod.task_history.list_recent = fake_list_recent  # type: ignore[assignment]

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
    # Wait for the server to be ready.
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
