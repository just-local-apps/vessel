"""Tests for the PWA HTTP routes — day navigation and all-tasks view.

Spins up a minimal FastAPI app wired only to vessel.pwa.router, overrides the
auth dependency, and monkeypatches the state_manager so we don't need a DB.
"""
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vessel.auth import require_user_id
from vessel.models import StateData
from vessel.models.enums import Cadence, ProjectStatus, Tier, TimeWindow
from vessel.models.state import CalendarEvent, Project, Task
from vessel.pwa.routes import _now_local, router as pwa_router


FAKE_USER = "test-user"


# ---------------------------------------------------------------------------
# Scripted-LLM shim for the chat endpoint.
#
# `/api/chat` calls `client.chat.completions.create(...)` via
# `run_chat_assistant` → `tool_loop`. We hand it a fake client whose
# `create` returns a queued OpenAI-compatible response per call. Each
# scripted message either contains text (final reply) or `tool_calls`
# (one per CRUD op the LLM is supposed to make).
# ---------------------------------------------------------------------------


@dataclass
class _FnCall:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FnCall
    type: str = "function"


@dataclass
class _FakeMessage:
    content: str = ""
    tool_calls: list[_FakeToolCall] = field(default_factory=list)


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResp:
    choices: list[_FakeChoice]


def _chat_msg(text: str = "", calls: list[tuple[str, dict[str, Any]]] | None = None) -> _FakeMessage:
    return _FakeMessage(
        content=text,
        tool_calls=[
            _FakeToolCall(
                id=f"call_{i}",
                function=_FnCall(name=name, arguments=json.dumps(args)),
            )
            for i, (name, args) in enumerate(calls or [])
        ],
    )


class _ChatLLM:
    """Returns a queued response per call; default = empty completion."""

    def __init__(self, scripted: list[_FakeMessage]):
        self._queue = list(scripted)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs) -> _FakeResp:
        self.calls.append(kwargs)
        if not self._queue:
            return _FakeResp(choices=[_FakeChoice(message=_chat_msg(text="(done)"))])
        return _FakeResp(choices=[_FakeChoice(message=self._queue.pop(0))])


class _ChatClient:
    """Mimics the AsyncOpenAI shape the route hands to run_chat_assistant.
    Mirrors the helper in test_assistant.py so the chat surface and the
    skip surface use the same fake."""

    def __init__(self, fn: _ChatLLM):
        self.chat = self
        self.completions = self
        self._fn = fn

    async def create(self, **kwargs):
        return await self._fn(**kwargs)


def _make_state(today: date) -> StateData:
    project = Project(
        id="p1",
        name="Demo",
        status=ProjectStatus.active,
        tracked=True,
        cadence=Cadence.daily,
        last_touched=datetime(2026, 4, 25, tzinfo=timezone.utc),
    )
    base_created = datetime(2026, 4, 25, tzinfo=timezone.utc)
    tasks = [
        Task(
            id="t-today-evening",
            project_id="p1",
            title="evening task today",
            time_window=TimeWindow.evening,
            tier=Tier.must_today,
            due_date=today,
            created_at=base_created,
        ),
        Task(
            id="t-today-workday",
            project_id="p1",
            title="workday task today",
            time_window=TimeWindow.workday,
            tier=Tier.flex,
            due_date=today,
            created_at=base_created,
        ),
        Task(
            id="t-tomorrow",
            project_id="p1",
            title="tomorrow task",
            time_window=TimeWindow.anytime,
            tier=Tier.flex,
            due_date=today + timedelta(days=1),
            created_at=base_created,
        ),
        Task(
            id="t-completed",
            project_id="p1",
            title="already done",
            time_window=TimeWindow.anytime,
            tier=Tier.flex,
            due_date=today,
            created_at=base_created,
            completed_at=datetime(2026, 4, 25, 10, tzinfo=timezone.utc),
        ),
    ]
    return StateData(projects=[project], tasks=tasks)


def _seed_state_with_project(project_id: str = "p_demo", name: str = "Demo") -> StateData:
    """Minimal state seeded with one project, for CRUD-endpoint tests."""
    return StateData(
        projects=[
            Project(
                id=project_id,
                name=name,
                status=ProjectStatus.active,
                tracked=True,
                cadence=Cadence.event_driven,
                last_touched=datetime(2026, 4, 28, tzinfo=timezone.utc),
            )
        ]
    )


@pytest.fixture
def client_state(monkeypatch):
    """Build a TestClient with auth overridden and state_manager stubbed.

    Defaults `settings.groq_api_key` to None so the skip/cancel
    endpoints' Stage-2 LLM branch is skipped — tests that want to
    exercise the assistant explicitly monkeypatch a fake key + fake
    OpenAI client. Without this default, .env's real Groq key would
    leak into route tests and the live LLM could mutate state in
    unpredictable ways (observed: a `cancel/change` test deleting
    its own event hard, breaking `unskip`)."""
    from vessel.config import get_settings as _gs

    monkeypatch.setattr(_gs(), "groq_api_key", None, raising=False)

    state_box = {"state": StateData()}

    async def fake_read(_pool, _user_id):
        return StateData.model_validate(
            state_box["state"].model_dump(mode="python")
        )

    async def fake_write(_pool, _user_id, new_state):
        state_box["state"] = new_state

    async def fake_pool():
        return None  # routes only pass it through to state_manager

    app = FastAPI()
    app.include_router(pwa_router)
    app.dependency_overrides[require_user_id] = lambda: FAKE_USER

    state_box["history"] = []  # archived (closed) tasks

    async def fake_archive(_pool, user_id, task, closed_kind):
        # Newest-first so pop_latest always returns the most recent.
        state_box["history"].insert(
            0, {"task": task.model_copy(deep=True), "closed_kind": closed_kind}
        )

    async def fake_pop_latest(_pool, user_id, task_id, closed_kind=None):
        for i, row in enumerate(state_box["history"]):
            if row["task"].id != task_id:
                continue
            if closed_kind is not None and row["closed_kind"] != closed_kind:
                continue
            state_box["history"].pop(i)
            return row["task"]
        return None

    async def fake_list_recent(_pool, user_id, *, limit=100):
        return [
            {
                "task_id": r["task"].id,
                "closed_kind": r["closed_kind"],
                "closed_at": "2026-04-26T00:00:00Z",
                "task": r["task"].model_dump(mode="json"),
            }
            for r in state_box["history"][:limit]
        ]

    with patch("vessel.pwa.routes.state_manager.read", side_effect=fake_read), \
         patch("vessel.pwa.routes.state_manager.write", side_effect=fake_write), \
         patch("vessel.pwa.routes.get_pool", side_effect=fake_pool), \
         patch("vessel.pwa.routes.task_history.archive", side_effect=fake_archive), \
         patch(
             "vessel.pwa.routes.task_history.pop_latest",
             side_effect=fake_pop_latest,
         ), \
         patch(
             "vessel.pwa.routes.task_history.list_recent",
             side_effect=fake_list_recent,
         ):
        client = TestClient(app)
        yield client, state_box


def test_tasks_all_includes_completed(client_state):
    client, box = client_state
    today = _now_local().date()
    box["state"] = _make_state(today)

    resp = client.get("/api/tasks/all")
    assert resp.status_code == 200
    data = resp.json()

    ids = [t["id"] for t in data["tasks"]]
    assert set(ids) == {
        "t-today-evening",
        "t-today-workday",
        "t-tomorrow",
        "t-completed",
    }
    assert data["open_count"] == 3
    assert data["total"] == 4
    # Completed must sort to the bottom.
    assert data["tasks"][-1]["id"] == "t-completed"


def test_tasks_all_orders_same_day_by_time_of_day_not_alphabet(client_state):
    """Within a single due_date, tasks must appear in chronological
    order of their `time_window` bucket (before_work → workday →
    after_work → evening → anytime), NOT alphabetical order.

    Regression: the `/api/tasks/all` sort used the raw enum string,
    which made `after_work` (a) come before `before_work` (b), so a
    morning task showed up below an evening task on the same day.
    """
    from datetime import time as _time

    client, box = client_state
    today = _now_local().date()

    project = Project(
        id="p1", name="P", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.daily,
        last_touched=datetime(2026, 4, 25, tzinfo=timezone.utc),
    )
    base = datetime(2026, 4, 25, tzinfo=timezone.utc)
    # Same due_date, four different time-of-day gates spanning the day.
    box["state"] = StateData(
        projects=[project],
        tasks=[
            Task(
                id="t-evening",
                project_id="p1",
                title="Evening task",
                tier=Tier.flex,
                due_date=today,
                created_at=base,
                start_after=_time(21, 0),
            ),
            Task(
                id="t-after-work",
                project_id="p1",
                title="After-work task",
                tier=Tier.flex,
                due_date=today,
                created_at=base,
                start_after=_time(18, 0),
            ),
            Task(
                id="t-before-work",
                project_id="p1",
                title="Before-work task",
                tier=Tier.flex,
                due_date=today,
                created_at=base,
                start_after=_time(6, 0),
            ),
            Task(
                id="t-anytime",
                project_id="p1",
                title="Anytime task",
                tier=Tier.flex,
                due_date=today,
                created_at=base,
                # No start_after → time_window = anytime
            ),
        ],
    )

    resp = client.get("/api/tasks/all")
    assert resp.status_code == 200
    ids = [t["id"] for t in resp.json()["tasks"]]
    assert ids == [
        "t-before-work",
        "t-after-work",
        "t-evening",
        "t-anytime",
    ], ids


def test_tasks_all_empty_state(client_state):
    client, _ = client_state
    resp = client.get("/api/tasks/all")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {
        "tasks": [],
        "events": [],
        "open_count": 0,
        "total": 0,
        "events_count": 0,
        "now": data["now"],
    }


def test_tasks_day_today_window_split(client_state):
    client, box = client_state
    today = _now_local().date()
    box["state"] = _make_state(today)

    resp = client.get("/api/tasks/day?offset=0")
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_today"] is True
    assert data["date"] == today.isoformat()
    assert data["offset"] == 0
    assert data["window"] is not None
    # The completed one must never appear.
    today_ids = {t["id"] for t in data["now"] + data["later"]}
    assert "t-completed" not in today_ids
    # And the tomorrow task isn't part of today.
    assert "t-tomorrow" not in today_ids
    # Both pending today tasks must be split between now/later.
    assert today_ids == {"t-today-evening", "t-today-workday"}


def test_tasks_day_offset_returns_that_day(client_state):
    client, box = client_state
    today = _now_local().date()
    box["state"] = _make_state(today)

    resp = client.get("/api/tasks/day?offset=1")
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_today"] is False
    assert data["offset"] == 1
    assert data["date"] == (today + timedelta(days=1)).isoformat()
    # Non-today views ignore window split — everything shows up under "now".
    assert data["later"] == []
    ids = [t["id"] for t in data["now"]]
    assert ids == ["t-tomorrow"]
    assert data["window"] is None


def _state_with_events(today: date) -> StateData:
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
                id="ev-today-gym",
                project_id="p_health",
                title="Gym",
                description="cardio",
                start=datetime.combine(today, datetime.min.time()).replace(
                    hour=8, tzinfo=timezone.utc
                ),
                end=datetime.combine(today, datetime.min.time()).replace(
                    hour=9, tzinfo=timezone.utc
                ),
            ),
            CalendarEvent(
                id="ev-tomorrow-meeting",
                project_id="p_health",
                title="Doctor",
                description="annual",
                start=datetime.combine(
                    today + timedelta(days=1), datetime.min.time()
                ).replace(hour=14, tzinfo=timezone.utc),
                end=datetime.combine(
                    today + timedelta(days=1), datetime.min.time()
                ).replace(hour=15, tzinfo=timezone.utc),
            ),
        ],
    )


def test_tasks_day_includes_events_for_target_date(client_state):
    client, box = client_state
    today = _now_local().date()
    box["state"] = _state_with_events(today)

    today_resp = client.get("/api/tasks/day?offset=0").json()
    assert [e["id"] for e in today_resp["events"]] == ["ev-today-gym"]

    tomorrow_resp = client.get("/api/tasks/day?offset=1").json()
    assert [e["id"] for e in tomorrow_resp["events"]] == ["ev-tomorrow-meeting"]


def test_tasks_all_includes_events(client_state):
    client, box = client_state
    today = _now_local().date()
    box["state"] = _state_with_events(today)

    data = client.get("/api/tasks/all").json()
    assert data["events_count"] == 2
    ids = {e["id"] for e in data["events"]}
    assert ids == {"ev-today-gym", "ev-tomorrow-meeting"}


def test_now_card_during_calendar_block_is_swipeable_event():
    """Active calendar block returns the event as a swipeable card —
    right marks done, left opens the move/skip dialog."""
    from vessel.pwa.routes import _pick_now_card

    now = _now_local()
    project = Project(
        id="p1", name="P", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.daily, last_touched=now,
    )
    active = CalendarEvent(
        id="ev-active", project_id="p1", title="Active block",
        description="",
        start=now - timedelta(minutes=10),
        end=now + timedelta(minutes=20),
    )
    state = StateData(projects=[project], calendar=[active])

    result = _pick_now_card(state, now)
    assert result["card"]["kind"] == "event"
    assert result["card"]["swipeable"] is True
    assert result["in_block"] is True


def test_now_card_outside_block_is_swipeable_true():
    """Plain task card → swipeable=True. UI shows the red/green borders."""
    from vessel.pwa.routes import _pick_now_card

    base = _now_local().replace(hour=9, minute=0, second=0, microsecond=0)
    project = Project(
        id="p1", name="P", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.daily, last_touched=base,
    )
    task = Task(
        id="t-fits", project_id="p1", title="Short",
        time_window=TimeWindow.workday, tier=Tier.must_today,
        due_date=base.date(), estimated_minutes=30, created_at=base,
    )
    state = StateData(projects=[project], tasks=[task])

    result = _pick_now_card(state, base)
    assert result["card"]["swipeable"] is True
    assert result["in_block"] is False


def test_break_card_appears_after_threshold_minutes_completed():
    """When cumulative completed-task minutes since last_break_acknowledged_at
    crosses 90, _pick_now_card returns a break card."""
    from vessel.pwa.routes import _pick_now_card

    now = _now_local().replace(hour=11, minute=0, second=0, microsecond=0)
    today = now.date()
    project = Project(
        id="p1", name="P", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.daily, last_touched=now,
    )
    completed_a = Task(
        id="t-done-a", project_id="p1", title="Done A",
        time_window=TimeWindow.workday, tier=Tier.must_today,
        due_date=today, estimated_minutes=60, created_at=now,
        completed_at=now,
    )
    completed_b = Task(
        id="t-done-b", project_id="p1", title="Done B",
        time_window=TimeWindow.workday, tier=Tier.must_today,
        due_date=today, estimated_minutes=40, created_at=now,
        completed_at=now,
    )
    state = StateData(projects=[project], tasks=[completed_a, completed_b])

    result = _pick_now_card(state, now)
    assert result["card"]["kind"] == "break"
    assert result["card"]["data"]["minutes_worked"] == 100


def test_break_card_does_not_appear_below_threshold():
    """30 minutes worked → no break card; the next task is shown."""
    from vessel.pwa.routes import _pick_now_card

    now = _now_local().replace(hour=11, minute=0, second=0, microsecond=0)
    today = now.date()
    project = Project(
        id="p1", name="P", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.daily, last_touched=now,
    )
    completed = Task(
        id="t-done", project_id="p1", title="Done",
        time_window=TimeWindow.workday, tier=Tier.must_today,
        due_date=today, estimated_minutes=30, created_at=now,
        completed_at=now,
    )
    open_task = Task(
        id="t-open", project_id="p1", title="Open",
        time_window=TimeWindow.workday, tier=Tier.must_today,
        due_date=today, estimated_minutes=15, created_at=now,
    )
    state = StateData(projects=[project], tasks=[completed, open_task])

    result = _pick_now_card(state, now)
    assert result["card"]["kind"] == "task"
    assert result["card"]["data"]["id"] == "t-open"


def test_break_acknowledgement_resets_the_counter():
    """After last_break_acknowledged_at is set, only work completed
    AFTER that timestamp counts toward the next break."""
    from vessel.pwa.routes import _pick_now_card

    now = _now_local().replace(hour=14, minute=0, second=0, microsecond=0)
    today = now.date()
    project = Project(
        id="p1", name="P", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.daily, last_touched=now,
    )
    # 100 min of work completed BEFORE the ack — must be ignored.
    pre_break = Task(
        id="t-pre", project_id="p1", title="Pre",
        time_window=TimeWindow.workday, tier=Tier.must_today,
        due_date=today, estimated_minutes=100,
        created_at=now - timedelta(hours=4),
        completed_at=now - timedelta(hours=3),
    )
    open_task = Task(
        id="t-open", project_id="p1", title="Open",
        time_window=TimeWindow.workday, tier=Tier.must_today,
        due_date=today, estimated_minutes=15, created_at=now,
    )
    state = StateData(
        projects=[project],
        tasks=[pre_break, open_task],
        last_break_acknowledged_at=now - timedelta(hours=2),
    )

    result = _pick_now_card(state, now)
    # No break card — pre_break completed before the ack and doesn't count.
    assert result["card"]["kind"] == "task"


def test_break_ack_route_sets_timestamp(client_state):
    client, box = client_state
    today = _now_local().date()
    box["state"] = _make_state(today)
    assert box["state"].last_break_acknowledged_at is None

    resp = client.post("/api/break/ack")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert box["state"].last_break_acknowledged_at is not None


def test_move_event_shifts_start_and_end(client_state):
    """POST /api/events/{id}/move shifts both start and end by N min."""
    client, box = client_state
    today = _now_local().date()
    box["state"] = _state_with_events(today)
    original = box["state"].calendar[0]
    orig_start = original.start
    orig_end = original.end

    resp = client.post(
        f"/api/events/{original.id}/move", json={"minutes": 30}
    )
    assert resp.status_code == 200
    moved = next(
        e for e in box["state"].calendar if e.id == original.id
    )
    assert (moved.start - orig_start) == timedelta(minutes=30)
    assert (moved.end - orig_end) == timedelta(minutes=30)


def test_move_event_404_on_unknown(client_state):
    client, _ = client_state
    resp = client.post(
        "/api/events/no-such/move", json={"minutes": 10}
    )
    assert resp.status_code == 404


def test_skip_mutates_task_with_reason_and_no_event(client_state):
    """POST /api/tasks/{id}/skip records `skipped_at` + `skip_reason`
    on the task and persists. The task is archived and removed from state."""
    client, box = client_state
    today = _now_local().date()
    box["state"] = _make_state(today)

    resp = client.post(
        "/api/tasks/t-today-evening/skip",
        json={"reason": "feeling sick, moving to tomorrow"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["task_id"] == "t-today-evening"
    assert body["skipped_at"]

    # The task is gone from state — skipped tasks move to the
    # archive table, not stay in the working state list.
    assert all(t.id != "t-today-evening" for t in box["state"].tasks)
    # And the archive row carries the reason for downstream stats.
    assert len(box["history"]) == 1
    archived = box["history"][0]
    assert archived["closed_kind"] == "skipped"
    assert archived["task"].id == "t-today-evening"
    assert archived["task"].skipped_at is not None
    assert archived["task"].skip_reason == "feeling sick, moving to tomorrow"


def test_skipped_task_disappears_from_today_view(client_state):
    """A task with `skipped_at` set must be filtered out of /api/tasks/day
    just like a completed one is."""
    client, box = client_state
    today = _now_local().date()
    box["state"] = _make_state(today)

    # First confirm the task is visible.
    pre = client.get("/api/tasks/day?offset=0").json()
    pre_ids = {t["id"] for t in pre["now"] + pre["later"]}
    assert "t-today-evening" in pre_ids

    # Skip it, then re-fetch — should be gone.
    client.post(
        "/api/tasks/t-today-evening/skip", json={"reason": "not today"}
    )
    post = client.get("/api/tasks/day?offset=0").json()
    post_ids = {t["id"] for t in post["now"] + post["later"]}
    assert "t-today-evening" not in post_ids


def test_skip_requires_reason(client_state):
    """A skip without a reason is the whole point — refuse it. The agent
    can't learn from a blank skip."""
    client, box = client_state
    today = _now_local().date()
    box["state"] = _make_state(today)

    resp = client.post(
        "/api/tasks/t-today-evening/skip", json={"reason": "   "}
    )
    assert resp.status_code == 400


def test_skip_404_on_unknown_task(client_state):
    client, box = client_state
    box["state"] = _make_state(_now_local().date())

    resp = client.post(
        "/api/tasks/no-such-task/skip", json={"reason": "oops"}
    )
    assert resp.status_code == 404


def test_uncomplete_restores_task_from_history(client_state):
    """Right-swipe → undo: complete archives the task to history;
    uncomplete pops it back into state with completed_at cleared."""
    client, box = client_state
    today = _now_local().date()
    box["state"] = _make_state(today)

    client.post("/api/tasks/t-today-evening/complete")
    # Task is OUT of state (archived) and present in the history fixture.
    assert all(t.id != "t-today-evening" for t in box["state"].tasks)
    assert any(
        r["task"].id == "t-today-evening" and r["closed_kind"] == "completed"
        for r in box["history"]
    )

    resp = client.post("/api/tasks/t-today-evening/uncomplete")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["source"] == "history"
    # Restored to state with completed_at cleared.
    restored = next(t for t in box["state"].tasks if t.id == "t-today-evening")
    assert restored.completed_at is None
    # And popped out of history.
    assert all(r["task"].id != "t-today-evening" for r in box["history"])


def test_unskip_restores_task_from_history(client_state):
    """Left-swipe → undo: skip archives + removes; unskip pops back."""
    client, box = client_state
    today = _now_local().date()
    box["state"] = _make_state(today)

    client.post(
        "/api/tasks/t-today-evening/skip", json={"reason": "not today"}
    )
    assert all(t.id != "t-today-evening" for t in box["state"].tasks)

    resp = client.post("/api/tasks/t-today-evening/unskip")
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "history"
    restored = next(t for t in box["state"].tasks if t.id == "t-today-evening")
    assert restored.skipped_at is None
    assert restored.skip_reason is None
    assert all(r["task"].id != "t-today-evening" for r in box["history"])


def test_uncomplete_task_404_on_unknown(client_state):
    client, box = client_state
    box["state"] = _make_state(_now_local().date())
    resp = client.post("/api/tasks/no-such/uncomplete")
    assert resp.status_code == 404


def test_unskip_event_clears_skip_state(client_state):
    client, box = client_state
    today = _now_local().date()
    box["state"] = _state_with_events(today)

    client.post(
        "/api/events/ev-today-gym/skip", json={"reason": "rolled my ankle"}
    )
    resp = client.post("/api/events/ev-today-gym/unskip")
    assert resp.status_code == 200
    ev = next(e for e in box["state"].calendar if e.id == "ev-today-gym")
    assert ev.skipped_at is None
    assert ev.skip_reason is None


def test_focus_card_hides_task_with_start_after_in_future(client_state):
    """A task gated by start_after must not surface as the focus card
    until the local wall clock crosses that time. Tasks without a gate
    (start_after=None) keep working unchanged."""
    from datetime import time as _time
    client, box = client_state
    today = _now_local().date()
    project = Project(
        id="p1", name="Demo", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.daily,
        last_touched=datetime(2026, 4, 25, tzinfo=timezone.utc),
    )
    base = datetime(2026, 4, 25, tzinfo=timezone.utc)
    # Anchor "future" relative to the test's actual wall clock so the
    # gate is reliably in the future regardless of when the suite runs.
    now_local = _now_local()
    far_future = _time(23, 59) if now_local.time() < _time(23, 0) else _time(23, 59, 59)
    box["state"] = StateData(
        projects=[project],
        tasks=[
            Task(
                id="t-gated", project_id="p1", title="Wash dishes",
                time_window=TimeWindow.evening, tier=Tier.must_today,
                due_date=today, estimated_minutes=15, created_at=base,
                start_after=far_future,
            ),
        ],
    )
    resp = client.get("/api/now")
    assert resp.status_code == 200
    # Gate has not opened → no candidate → empty focus card.
    assert resp.json().get("card") is None


def test_focus_card_surfaces_task_once_start_after_passes(client_state):
    """A task whose start_after time has already passed today must be
    eligible as the focus card."""
    from datetime import time as _time
    client, box = client_state
    today = _now_local().date()
    project = Project(
        id="p1", name="Demo", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.daily,
        last_touched=datetime(2026, 4, 25, tzinfo=timezone.utc),
    )
    base = datetime(2026, 4, 25, tzinfo=timezone.utc)
    box["state"] = StateData(
        projects=[project],
        tasks=[
            Task(
                id="t-open", project_id="p1", title="Wash dishes",
                time_window=TimeWindow.anytime, tier=Tier.must_today,
                due_date=today, estimated_minutes=15, created_at=base,
                start_after=_time(0, 0),  # midnight → always passed
            ),
        ],
    )
    resp = client.get("/api/now")
    assert resp.status_code == 200
    card = resp.json().get("card")
    assert card is not None
    assert card["data"]["id"] == "t-open"


def test_complete_daily_recurring_task_spawns_next_day_instance(client_state):
    """Completing a recurrence='daily' task must spawn an OPEN clone for
    due_date+1 with the same title / project / start_after / tier / etc.
    The id swaps the YYYYMMDD suffix; the original stays completed."""
    from datetime import time as _time
    client, box = client_state
    today = _now_local().date()
    today_suffix = today.strftime("%Y%m%d")
    project = Project(
        id="p1", name="Demo", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.daily,
        last_touched=datetime(2026, 4, 25, tzinfo=timezone.utc),
    )
    base = datetime(2026, 4, 25, tzinfo=timezone.utc)
    box["state"] = StateData(
        projects=[project],
        tasks=[
            Task(
                id=f"task_wash_dishes_{today_suffix}", project_id="p1",
                title="Wash dishes",
                time_window=TimeWindow.evening, tier=Tier.flex,
                due_date=today, estimated_minutes=15, created_at=base,
                start_after=_time(19, 0), recurrence="daily",
            ),
        ],
    )
    resp = client.post(f"/api/tasks/task_wash_dishes_{today_suffix}/complete")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    next_suffix = (today + timedelta(days=1)).strftime("%Y%m%d")
    expected_new_id = f"task_wash_dishes_{next_suffix}"
    assert body.get("spawned_id") == expected_new_id
    new_task = next(t for t in box["state"].tasks if t.id == expected_new_id)
    assert new_task.title == "Wash dishes"
    assert new_task.due_date == today + timedelta(days=1)
    assert new_task.completed_at is None
    assert new_task.skipped_at is None
    assert new_task.recurrence == "daily"
    assert new_task.start_after == _time(19, 0)
    # And the original is now archived to history (not in state).
    assert all(
        t.id != f"task_wash_dishes_{today_suffix}" for t in box["state"].tasks
    )
    archived = next(
        r for r in box["history"]
        if r["task"].id == f"task_wash_dishes_{today_suffix}"
    )
    assert archived["closed_kind"] == "completed"
    assert archived["task"].completed_at is not None


def test_chat_with_daily_recurrence_pre_populates_seven_days(client_state):
    """End-to-end shape: a chat instruction that produces a single
    `recurrence='daily'` task must have the 7-day window materialized
    by the route's expander before write — so the user sees today plus
    the next 6 days right away, not just one task. Now driven by the
    chat tool-loop assistant: the scripted LLM emits one `add_task`
    call with `recurrence="daily"`, and the route expander does the
    rest."""
    from datetime import time as _time
    import vessel.pwa.routes as routes_mod
    client, box = client_state
    today = _now_local().date()
    project = Project(
        id="p1", name="P", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.daily,
        last_touched=datetime(2026, 4, 25, tzinfo=timezone.utc),
    )
    box["state"] = StateData(projects=[project])

    fake = _ChatLLM([
        _chat_msg(calls=[(
            "add_task",
            {"fields": {
                "project_id": "p1",
                "title": "Wash dishes",
                "due_date": today.isoformat(),
                "tier": "flex",
                "estimated_minutes": 15,
                "start_after": "19:00",
                "recurrence": "daily",
            }},
        )]),
        _chat_msg(text="added wash dishes daily after 7pm"),
    ])

    routes_mod._set_chat_client_for_test(_ChatClient(fake))
    try:
        resp = client.post("/api/chat", json={"text": "wash dishes daily after 7pm"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["applied"] is True
        # 7 forward open instances must exist after the expander runs.
        open_dishes = [
            t for t in box["state"].tasks
            if t.title == "Wash dishes"
            and t.completed_at is None
            and t.skipped_at is None
        ]
        assert len(open_dishes) == 7, [t.due_date for t in open_dishes]
        dates = sorted(t.due_date for t in open_dishes)
        assert dates[0] == today
        assert dates[-1] == today + timedelta(days=6)
        # Every instance carries the recurrence + start_after metadata.
        for t in open_dishes:
            assert t.recurrence == "daily"
            assert t.start_after == _time(19, 0)
            assert t.estimated_minutes == 15
    finally:
        routes_mod._set_chat_client_for_test(None)


def test_complete_daily_recurring_does_not_double_spawn(client_state):
    """If the user completes a daily task and an open instance for
    tomorrow already exists with the same title, do NOT create a
    second copy. Same dedup contract as the intake agent."""
    from datetime import time as _time
    client, box = client_state
    today = _now_local().date()
    tomorrow = today + timedelta(days=1)
    today_suffix = today.strftime("%Y%m%d")
    tomorrow_suffix = tomorrow.strftime("%Y%m%d")
    project = Project(
        id="p1", name="Demo", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.daily,
        last_touched=datetime(2026, 4, 25, tzinfo=timezone.utc),
    )
    base = datetime(2026, 4, 25, tzinfo=timezone.utc)
    box["state"] = StateData(
        projects=[project],
        tasks=[
            Task(
                id=f"task_wash_dishes_{today_suffix}", project_id="p1",
                title="Wash dishes", time_window=TimeWindow.evening,
                tier=Tier.flex, due_date=today, estimated_minutes=15,
                created_at=base, recurrence="daily",
            ),
            Task(
                id=f"task_wash_dishes_{tomorrow_suffix}_existing",
                project_id="p1", title="Wash dishes",
                time_window=TimeWindow.evening, tier=Tier.flex,
                due_date=tomorrow, estimated_minutes=15, created_at=base,
                recurrence="daily",
            ),
        ],
    )
    resp = client.post(f"/api/tasks/task_wash_dishes_{today_suffix}/complete")
    assert resp.status_code == 200
    # Tomorrow should still have exactly one open Wash dishes — the
    # pre-existing one — not two.
    open_tomorrow = [
        t for t in box["state"].tasks
        if t.due_date == tomorrow
        and t.title == "Wash dishes"
        and t.completed_at is None
        and t.skipped_at is None
    ]
    assert len(open_tomorrow) == 1


def test_crud_add_project_minimal(client_state):
    client, box = client_state
    resp = client.post("/api/projects", json={"name": "Health"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["project"]["id"] == "p_health"
    assert any(p.id == "p_health" for p in box["state"].projects)


def test_crud_add_project_id_conflict(client_state):
    client, box = client_state
    box["state"] = _seed_state_with_project("p_taken", "Taken")
    resp = client.post("/api/projects", json={"id": "p_taken", "name": "x"})
    assert resp.status_code == 409


def test_crud_add_projects_bulk(client_state):
    client, box = client_state
    resp = client.post(
        "/api/projects/bulk",
        json=[{"name": "A"}, {"name": "B"}],
    )
    assert resp.status_code == 200
    assert len(resp.json()["projects"]) == 2
    assert {p.id for p in box["state"].projects} == {"p_a", "p_b"}


def test_crud_update_project_changes_field(client_state):
    client, box = client_state
    box["state"] = _seed_state_with_project("p_demo", "Demo")
    resp = client.patch("/api/projects/p_demo", json={"importance": "high"})
    assert resp.status_code == 200
    proj = next(p for p in box["state"].projects if p.id == "p_demo")
    assert proj.importance.value == "high"


def test_crud_delete_project_with_open_task_returns_409(client_state):
    client, box = client_state
    box["state"] = _seed_state_with_project("p_demo", "Demo")
    client.post("/api/tasks", json={"title": "x", "project_id": "p_demo"})
    resp = client.delete("/api/projects/p_demo")
    assert resp.status_code == 409


def test_crud_add_task_defaults_and_persists(client_state):
    client, box = client_state
    box["state"] = _seed_state_with_project("p_demo", "Demo")
    resp = client.post(
        "/api/tasks",
        json={"title": "Buy milk", "project_id": "p_demo"},
    )
    assert resp.status_code == 200
    task = resp.json()["task"]
    assert task["title"] == "Buy milk"
    assert task["tier"] == "flex"
    assert task["estimated_minutes"] == 30
    assert any(t.id == task["id"] for t in box["state"].tasks)


def test_crud_add_task_unknown_project_returns_400(client_state):
    client, box = client_state
    resp = client.post(
        "/api/tasks", json={"title": "x", "project_id": "p_nope"}
    )
    assert resp.status_code == 400


def test_crud_add_tasks_bulk(client_state):
    client, box = client_state
    box["state"] = _seed_state_with_project("p_demo", "Demo")
    resp = client.post(
        "/api/tasks/bulk",
        json=[
            {"title": "A", "project_id": "p_demo"},
            {"title": "B", "project_id": "p_demo"},
        ],
    )
    assert resp.status_code == 200
    assert len(resp.json()["tasks"]) == 2
    assert len(box["state"].tasks) == 2


def test_crud_update_task_sets_recurrence(client_state):
    client, box = client_state
    box["state"] = _seed_state_with_project("p_demo", "Demo")
    add = client.post(
        "/api/tasks", json={"title": "Wash dishes", "project_id": "p_demo"}
    )
    task_id = add.json()["task"]["id"]
    resp = client.patch(
        f"/api/tasks/{task_id}",
        json={"recurrence": "daily", "start_after": "19:00:00"},
    )
    assert resp.status_code == 200
    task = next(t for t in box["state"].tasks if t.id == task_id)
    assert task.recurrence == "daily"
    assert task.start_after.isoformat().startswith("19:00")


def test_crud_delete_task(client_state):
    client, box = client_state
    box["state"] = _seed_state_with_project("p_demo", "Demo")
    add = client.post(
        "/api/tasks", json={"title": "x", "project_id": "p_demo"}
    )
    task_id = add.json()["task"]["id"]
    resp = client.delete(f"/api/tasks/{task_id}")
    assert resp.status_code == 200
    assert all(t.id != task_id for t in box["state"].tasks)


def test_crud_calendar_add_update_delete_round_trip(client_state):
    client, box = client_state
    box["state"] = _seed_state_with_project("p_demo", "Demo")
    add = client.post(
        "/api/calendar",
        json={
            "project_id": "p_demo",
            "title": "Standup",
            "start": "2026-05-01T09:00:00+00:00",
            "end": "2026-05-01T09:30:00+00:00",
        },
    )
    assert add.status_code == 200
    ev = add.json()["calendar_event"]

    upd = client.patch(
        f"/api/calendar/{ev['id']}",
        json={"location": "Zoom"},
    )
    assert upd.status_code == 200
    assert any(
        e.id == ev["id"] and e.location == "Zoom"
        for e in box["state"].calendar
    )

    rm = client.delete(f"/api/calendar/{ev['id']}")
    assert rm.status_code == 200
    assert all(e.id != ev["id"] for e in box["state"].calendar)


def test_crud_calendar_bulk(client_state):
    client, box = client_state
    box["state"] = _seed_state_with_project("p_demo", "Demo")
    base = "2026-05-01T09:00:00+00:00"
    end = "2026-05-01T10:00:00+00:00"
    resp = client.post(
        "/api/calendar/bulk",
        json=[
            {"project_id": "p_demo", "title": "A", "start": base, "end": end},
            {
                "project_id": "p_demo", "title": "B",
                "start": "2026-05-01T11:00:00+00:00",
                "end": "2026-05-01T12:00:00+00:00",
            },
        ],
    )
    assert resp.status_code == 200
    assert len(resp.json()["calendar_events"]) == 2
    assert len(box["state"].calendar) == 2


def test_crud_routine_add_update_delete(client_state):
    client, box = client_state
    add = client.post(
        "/api/routines",
        json={
            "label": "Morning gym",
            "start_time": "07:00:00",
            "duration_minutes": 60,
        },
    )
    assert add.status_code == 200
    rid = add.json()["routine"]["id"]
    upd = client.patch(f"/api/routines/{rid}", json={"duration_minutes": 45})
    assert upd.status_code == 200
    assert next(
        r for r in box["state"].routines if r.id == rid
    ).duration_minutes == 45
    rm = client.delete(f"/api/routines/{rid}")
    assert rm.status_code == 200
    assert box["state"].routines == []


def test_state_uses_x_vessel_client_now_header(client_state):
    """Server must echo the client's wall clock when given. Without
    this an EDT user sees 'today' answered in LA time."""
    client, box = client_state
    box["state"] = _seed_state_with_project("p_demo", "Demo")
    iso = "2026-04-29T21:30:00-04:00"
    resp = client.get("/api/state", headers={"X-Vessel-Client-Now": iso})
    assert resp.status_code == 200
    # The /api/state route reflects whatever now we computed.
    assert resp.json()["now"] == iso


def test_state_falls_back_to_server_local_when_header_missing(client_state):
    """No header → server-local. Pre-existing curl / Claude Desktop
    callers keep working."""
    client, box = client_state
    box["state"] = _seed_state_with_project("p_demo", "Demo")
    resp = client.get("/api/state")
    assert resp.status_code == 200
    # We can't assert exact value, but it should parse and not be the
    # client-supplied one we never sent.
    assert "now" in resp.json()


def test_state_ignores_malformed_client_now(client_state):
    """Garbage header should not crash the request — silently fall
    back to server-local time."""
    client, box = client_state
    box["state"] = _seed_state_with_project("p_demo", "Demo")
    resp = client.get(
        "/api/state", headers={"X-Vessel-Client-Now": "not-a-timestamp"}
    )
    assert resp.status_code == 200


def test_skip_route_invokes_assistant_when_llm_configured(client_state, monkeypatch):
    """End-to-end: POST /api/tasks/{id}/skip with a reason that means
    "delete all instances" should (a) archive the swiped task and
    (b) invoke the LLM tool-use loop, which deletes the matching open
    siblings. Both happen in a single HTTP call. Asserts on the
    returned `assistant.tool_calls` payload so the bug we hit live
    (assistant never invoked) can't regress silently."""
    from datetime import date as _date
    client, box = client_state

    # State: 4 open Wash dishes — one is the one we'll skip, the
    # other three are siblings the LLM should delete.
    today = _date.today()
    state = _seed_state_with_project("p_demo", "Demo")
    base = datetime(2026, 4, 28, tzinfo=timezone.utc)
    for offset in range(4):
        d = today + timedelta(days=offset)
        state.tasks.append(
            Task(
                id=f"task_wash_dishes_{d.strftime('%Y%m%d')}",
                project_id="p_demo",
                title="Wash dishes",
                time_window=TimeWindow.evening,
                tier=Tier.flex,
                estimated_minutes=15,
                due_date=d,
                created_at=base,
            )
        )
    box["state"] = state
    swiped_id = state.tasks[0].id  # the user is left-swiping today's
    sibling_ids = [t.id for t in state.tasks[1:]]

    # Pretend a Groq key is set so the route enters the assistant branch.
    from vessel.config import get_settings as _gs
    settings = _gs()
    monkeypatch.setattr(settings, "groq_api_key", "fake-key", raising=False)
    monkeypatch.setattr(settings, "groq_model", "fake-model", raising=False)

    # Build a fake AsyncOpenAI client that returns scripted responses.
    import json as _json
    from dataclasses import dataclass as _dc, field as _field
    from typing import Any as _Any, Optional as _Optional

    @_dc
    class _FnCall:
        name: str; arguments: str

    @_dc
    class _FakeToolCall:
        id: str; function: _FnCall; type: str = "function"

    @_dc
    class _FakeMessage:
        content: str = ""
        tool_calls: list = _field(default_factory=list)

    @_dc
    class _FakeChoice:
        message: _FakeMessage

    @_dc
    class _FakeResp:
        choices: list

    queue = [
        _FakeMessage(
            content="",
            tool_calls=[
                _FakeToolCall(
                    id=f"c{i}",
                    function=_FnCall(
                        name="delete_task",
                        arguments=_json.dumps({"id": tid}),
                    ),
                )
                for i, tid in enumerate(sibling_ids)
            ],
        ),
        _FakeMessage(content=f"deleted {len(sibling_ids)} more wash-dishes tasks"),
    ]

    class _Client:
        def __init__(self, q):
            self._q = q
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            msg = self._q.pop(0) if self._q else _FakeMessage(content="")
            return _FakeResp(choices=[_FakeChoice(message=msg)])

    fake_client = _Client(queue)

    # Patch the AsyncOpenAI constructor to hand back our fake.
    import openai as _openai
    monkeypatch.setattr(_openai, "AsyncOpenAI", lambda **kwargs: fake_client)

    resp = client.post(
        f"/api/tasks/{swiped_id}/skip",
        json={"reason": "back pain. no more wash dishes."},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True

    # Assistant ran and made the deletes.
    a = body["assistant"]
    assert a["invoked"] is True, a
    assert a["mutated"] is True, a
    assert a["stopped_reason"] == "completed"
    assert len(a["tool_calls"]) == len(sibling_ids)
    assert all(c["ok"] for c in a["tool_calls"]), a

    # State: swiped task archived, siblings gone, no Wash dishes left open.
    open_wash = [
        t for t in box["state"].tasks
        if t.title == "Wash dishes"
        and t.completed_at is None
        and t.skipped_at is None
    ]
    assert open_wash == [], [t.id for t in open_wash]


def test_skip_route_works_when_llm_unconfigured(client_state, monkeypatch):
    """If GROQ_API_KEY is unset, skip must still archive the task and
    return 200 — the assistant is best-effort and never blocks the
    swipe."""
    client, box = client_state
    state = _seed_state_with_project("p_demo", "Demo")
    state.tasks.append(
        Task(
            id="task_lonely_20260430",
            project_id="p_demo",
            title="Lonely task",
            time_window=TimeWindow.anytime,
            tier=Tier.flex,
            estimated_minutes=5,
            due_date=date(2026, 4, 30),
            created_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
        )
    )
    box["state"] = state

    from vessel.config import get_settings as _gs
    settings = _gs()
    monkeypatch.setattr(settings, "groq_api_key", None, raising=False)

    resp = client.post(
        "/api/tasks/task_lonely_20260430/skip",
        json={"reason": "not today"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    # Assistant payload says it wasn't invoked.
    assert body["assistant"]["invoked"] is False


def test_uncomplete_event_clears_completed_at(client_state):
    client, box = client_state
    today = _now_local().date()
    box["state"] = _state_with_events(today)

    client.post("/api/events/ev-today-gym/complete")
    resp = client.post("/api/events/ev-today-gym/uncomplete")
    assert resp.status_code == 200
    ev = next(e for e in box["state"].calendar if e.id == "ev-today-gym")
    assert ev.completed_at is None


def test_now_returns_active_calendar_block_when_inside_one(client_state):
    """If `now` is inside a calendar event, that event is the card. Nothing
    else competes during a calendar block — calendar owns time."""
    from vessel.pwa.routes import _pick_now_card

    now = _now_local()
    today = now.date()
    project = Project(
        id="p1", name="P", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.daily, last_touched=now,
    )
    # Block that contains `now` (start = now-15min, end = now+30min).
    active = CalendarEvent(
        id="ev-active", project_id="p1", title="Active block", description="",
        start=now - timedelta(minutes=15), end=now + timedelta(minutes=30),
    )
    fitting_task = Task(
        id="t-fits", project_id="p1", title="Fits the gap",
        time_window=TimeWindow.workday, tier=Tier.must_today,
        due_date=today, estimated_minutes=20, created_at=now,
    )
    state = StateData(projects=[project], tasks=[fitting_task], calendar=[active])

    result = _pick_now_card(state, now)
    assert result["in_block"] is True
    assert result["card"]["kind"] == "event"
    assert result["card"]["data"]["id"] == "ev-active"


def test_now_picks_highest_priority_fitting_task(client_state):
    """Outside any calendar block, pick the highest-priority task whose
    estimated_minutes ≤ free_minutes."""
    from vessel.pwa.routes import _pick_now_card

    # Pin "now" at 9 AM in the configured TZ so we have a generous gap to bedtime.
    base = _now_local().replace(hour=9, minute=0, second=0, microsecond=0)
    today = base.date()
    p_high = Project(
        id="p_high", name="High", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.daily, last_touched=base,
    )
    p_low = Project(
        id="p_low", name="Low", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.daily, last_touched=base,
    )
    # Task on the higher-priority project, 30 min — should win.
    winner = Task(
        id="t-winner", project_id="p_high", title="Top",
        time_window=TimeWindow.workday, tier=Tier.must_today,
        due_date=today, estimated_minutes=30, created_at=base,
    )
    # Task on a lower-priority project, even though must_today → loses to ranking.
    loser = Task(
        id="t-loser", project_id="p_low", title="Less",
        time_window=TimeWindow.workday, tier=Tier.must_today,
        due_date=today, estimated_minutes=30, created_at=base,
    )
    # Task that doesn't fit (480 min) — must be filtered.
    doesnt_fit = Task(
        id="t-toobig", project_id="p_high", title="Marathon",
        time_window=TimeWindow.workday, tier=Tier.must_today,
        due_date=today, estimated_minutes=480, created_at=base,
    )

    state = StateData(
        projects=[p_high, p_low],
        tasks=[winner, loser, doesnt_fit],
        calendar=[],
        priority_ranking=["p_high", "p_low"],
    )

    result = _pick_now_card(state, base)
    assert result["in_block"] is False
    assert result["card"]["kind"] == "task"
    assert result["card"]["data"]["id"] == "t-winner"
    # free_minutes should be (bedtime - 9am) = 12 hours = 720 min if bedtime=21
    assert result["free_minutes"] >= 60


def test_now_returns_no_card_when_nothing_fits(client_state):
    """5 min before next event + 60 min task → no card. Empty state with
    a hint of what's next."""
    from vessel.pwa.routes import _pick_now_card

    now = _now_local().replace(hour=12, minute=55, second=0, microsecond=0)
    today = now.date()
    project = Project(
        id="p1", name="P", status=ProjectStatus.active, tracked=True,
        cadence=Cadence.daily, last_touched=now,
    )
    # Next event in 5 minutes.
    next_event = CalendarEvent(
        id="ev-1pm", project_id="p1", title="Lunch", description="",
        start=now + timedelta(minutes=5), end=now + timedelta(minutes=35),
    )
    # 60-minute task — doesn't fit.
    big_task = Task(
        id="t-too-big", project_id="p1", title="Big",
        time_window=TimeWindow.workday, tier=Tier.must_today,
        due_date=today, estimated_minutes=60, created_at=now,
    )
    state = StateData(projects=[project], tasks=[big_task], calendar=[next_event])

    result = _pick_now_card(state, now)
    assert result["card"] is None
    assert result["free_minutes"] == 5


def test_now_endpoint_returns_card_through_http(client_state):
    """Smoke-test the actual route, not just the helper."""
    client, box = client_state
    today = _now_local().date()
    box["state"] = _make_state(today)

    resp = client.get("/api/now")
    assert resp.status_code == 200
    body = resp.json()
    assert "card" in body
    assert "free_minutes" in body
    assert "in_block" in body


def test_chat_applies_state_when_assistant_calls_crud_tools(client_state):
    """POST /api/chat: the chat assistant emits CRUD tool calls; the
    route persists the mutated state and responds with applied=true +
    diff. Replaces the legacy "intake returns StateData" path."""
    from vessel.pwa import routes as r

    client, box = client_state
    box["state"] = StateData()  # empty start

    fake = _ChatLLM([
        _chat_msg(calls=[(
            "add_project",
            {"fields": {
                "id": "p_new",
                "name": "New",
                "status": "active",
                "tracked": True,
                "cadence": "event_driven",
                "importance": "medium",
            }},
        )]),
        _chat_msg(text="added project 'New'"),
    ])

    r._set_chat_client_for_test(_ChatClient(fake))
    try:
        resp = client.post("/api/chat", json={"text": "add project new"})
    finally:
        r._set_chat_client_for_test(None)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["applied"] is True
    assert "diff" in body
    assert body["diff"]["summary"]["projects"]["added"] == 1
    # The assistant payload echoes the tool call back to the chat UI.
    assert body["assistant"]["stopped_reason"] == "completed"
    assert body["assistant"]["tool_calls"][0]["name"] == "add_project"
    assert body["assistant"]["tool_calls"][0]["ok"] is True
    # State actually persisted.
    assert any(p.id == "p_new" for p in box["state"].projects)


def test_chat_text_only_reply_does_not_mutate(client_state):
    """If the chat assistant emits no tool calls (the user said
    something conversational), the state must be untouched and
    `applied` must be False. Replaces the old clarifications path —
    the new agent simply replies with text instead of asking."""
    from vessel.pwa import routes as r

    client, box = client_state
    today = _now_local().date()
    box["state"] = _make_state(today)
    snapshot = box["state"].model_dump_json()

    fake = _ChatLLM([_chat_msg(text="thanks for the note")])

    r._set_chat_client_for_test(_ChatClient(fake))
    try:
        resp = client.post("/api/chat", json={"text": "thanks"})
    finally:
        r._set_chat_client_for_test(None)

    assert resp.status_code == 200
    body = resp.json()
    assert body["applied"] is False
    assert body["diff"] is None
    assert body["assistant"]["summary"] == "thanks for the note"
    assert body["assistant"]["tool_calls"] == []
    # State must be untouched when the assistant only replies with text.
    assert box["state"].model_dump_json() == snapshot


def test_chat_rejects_empty_text(client_state):
    client, _ = client_state
    resp = client.post("/api/chat", json={"text": "   "})
    assert resp.status_code == 400


def test_chat_unauthorized_without_token():
    """Sanity: real require_user_id rejects /api/chat too."""
    app = FastAPI()
    app.include_router(pwa_router)
    client = TestClient(app)
    resp = client.post("/api/chat", json={"text": "hi"})
    assert resp.status_code == 401


def test_skip_event_mutates_calendar_with_reason_no_event(client_state):
    """POST /api/events/{id}/skip sets skipped_at + skip_reason on
    the calendar entry and persists."""
    client, box = client_state
    today = _now_local().date()
    box["state"] = _state_with_events(today)

    resp = client.post(
        "/api/events/ev-today-gym/skip", json={"reason": "rolled my ankle"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["event_id"] == "ev-today-gym"
    assert body["skipped_at"]

    state = box["state"]
    ev = next(e for e in state.calendar if e.id == "ev-today-gym")
    assert ev.skipped_at is not None
    assert ev.skip_reason == "rolled my ankle"
    assert ev.completed_at is None


def test_complete_event_marks_done_no_event(client_state):
    client, box = client_state
    today = _now_local().date()
    box["state"] = _state_with_events(today)

    resp = client.post("/api/events/ev-today-gym/complete")
    assert resp.status_code == 200, resp.text
    assert resp.json()["ok"] is True

    ev = next(e for e in box["state"].calendar if e.id == "ev-today-gym")
    assert ev.completed_at is not None
    assert ev.skipped_at is None


def test_skipped_event_disappears_from_today_view(client_state):
    """Same contract as tasks: a skipped event is filtered out of /api/tasks/day."""
    client, box = client_state
    today = _now_local().date()
    box["state"] = _state_with_events(today)

    pre = client.get("/api/tasks/day?offset=0").json()
    pre_ids = {e["id"] for e in pre["events"]}
    assert "ev-today-gym" in pre_ids

    client.post(
        "/api/events/ev-today-gym/skip", json={"reason": "skip it"}
    )
    post = client.get("/api/tasks/day?offset=0").json()
    post_ids = {e["id"] for e in post["events"]}
    assert "ev-today-gym" not in post_ids


def test_skip_event_requires_reason(client_state):
    client, box = client_state
    today = _now_local().date()
    box["state"] = _state_with_events(today)

    resp = client.post(
        "/api/events/ev-today-gym/skip", json={"reason": "   "}
    )
    assert resp.status_code == 400


def test_skip_event_404_on_unknown(client_state):
    client, box = client_state
    box["state"] = _state_with_events(_now_local().date())

    resp = client.post(
        "/api/events/no-such-event/skip", json={"reason": "x"}
    )
    assert resp.status_code == 404


def test_tasks_day_unauthorized_without_override():
    """Sanity check: real require_user_id rejects missing token."""
    app = FastAPI()
    app.include_router(pwa_router)
    client = TestClient(app)
    resp = client.get("/api/tasks/day")
    assert resp.status_code == 401
