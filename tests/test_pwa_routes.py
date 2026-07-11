"""Tests for the PWA HTTP routes — calendar-only.

Spins up a minimal FastAPI app wired only to vessel.pwa.router, overrides the
auth dependency, and monkeypatches the state_manager so we don't need a DB.
"""
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vessel.auth import require_user_id
from vessel.models import CalendarEvent, StateData
from vessel.pwa.routes import _now_local, router as pwa_router


FAKE_USER = "test-user"


# ---------------------------------------------------------------------------
# Scripted-LLM shim for the chat endpoint.
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
    def __init__(self, fn: _ChatLLM):
        self.chat = self
        self.completions = self
        self._fn = fn

    async def create(self, **kwargs):
        return await self._fn(**kwargs)


def _make_state(today=None) -> StateData:
    """Minimal state with two calendar events — one today, one tomorrow."""
    if today is None:
        today = _now_local().date()
    today_start = datetime.combine(today, datetime.min.time()).replace(
        hour=9, tzinfo=timezone.utc
    )
    tomorrow = today + timedelta(days=1)
    tomorrow_start = datetime.combine(tomorrow, datetime.min.time()).replace(
        hour=14, tzinfo=timezone.utc
    )
    return StateData(
        calendar=[
            CalendarEvent(
                id="ev-today",
                title="Gym",
                description="cardio",
                start=today_start,
                end=today_start + timedelta(hours=1),
            ),
            CalendarEvent(
                id="ev-tomorrow",
                title="Doctor",
                description="annual",
                start=tomorrow_start,
                end=tomorrow_start + timedelta(hours=1),
            ),
        ]
    )


@pytest.fixture
def client_state(monkeypatch):
    """Build a TestClient with auth overridden and state_manager stubbed."""
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
        return None

    app = FastAPI()
    app.include_router(pwa_router)
    app.dependency_overrides[require_user_id] = lambda: FAKE_USER

    with patch("vessel.pwa.routes.state_manager.read", side_effect=fake_read), \
         patch("vessel.pwa.routes.state_manager.write", side_effect=fake_write), \
         patch("vessel.pwa.routes.get_pool", side_effect=fake_pool):
        client = TestClient(app)
        yield client, state_box


# ---------------------------------------------------------------------------
# /api/state
# ---------------------------------------------------------------------------


def test_state_returns_calendar(client_state):
    client, box = client_state
    today = _now_local().date()
    box["state"] = _make_state(today)

    resp = client.get("/api/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "state" in data
    assert "calendar" in data["state"]
    assert len(data["state"]["calendar"]) == 2


def test_state_uses_x_vessel_client_now_header(client_state):
    """Server must echo the client's wall clock when given."""
    client, box = client_state
    iso = "2026-04-29T21:30:00-04:00"
    resp = client.get("/api/state", headers={"X-Vessel-Client-Now": iso})
    assert resp.status_code == 200
    assert resp.json()["now"] == iso


def test_state_falls_back_to_server_local_when_header_missing(client_state):
    client, box = client_state
    resp = client.get("/api/state")
    assert resp.status_code == 200
    assert "now" in resp.json()


def test_state_ignores_malformed_client_now(client_state):
    client, box = client_state
    resp = client.get(
        "/api/state", headers={"X-Vessel-Client-Now": "not-a-timestamp"}
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/now
# ---------------------------------------------------------------------------


def test_now_returns_active_event_when_inside_one(client_state):
    client, box = client_state
    now = _now_local()
    box["state"] = StateData(
        calendar=[
            CalendarEvent(
                id="ev-active",
                title="Active meeting",
                start=now - timedelta(minutes=10),
                end=now + timedelta(minutes=20),
            )
        ]
    )
    resp = client.get("/api/now")
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "event"
    assert body["event"]["id"] == "ev-active"


def test_now_returns_next_upcoming_when_not_in_block(client_state):
    client, box = client_state
    now = _now_local()
    box["state"] = StateData(
        calendar=[
            CalendarEvent(
                id="ev-upcoming",
                title="Upcoming",
                start=now + timedelta(hours=1),
                end=now + timedelta(hours=2),
            )
        ]
    )
    resp = client.get("/api/now")
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "event"
    assert body["event"]["id"] == "ev-upcoming"


def test_now_returns_empty_when_no_events(client_state):
    client, box = client_state
    box["state"] = StateData()
    resp = client.get("/api/now")
    assert resp.status_code == 200
    assert resp.json()["type"] == "empty"


def test_now_skips_completed_and_skipped_events(client_state):
    client, box = client_state
    now = _now_local()
    box["state"] = StateData(
        calendar=[
            CalendarEvent(
                id="ev-done",
                title="Done",
                start=now + timedelta(hours=1),
                end=now + timedelta(hours=2),
                completed_at=now,
            ),
            CalendarEvent(
                id="ev-skipped",
                title="Skipped",
                start=now + timedelta(hours=3),
                end=now + timedelta(hours=4),
                skipped_at=now,
            ),
        ]
    )
    resp = client.get("/api/now")
    assert resp.status_code == 200
    assert resp.json()["type"] == "empty"


# ---------------------------------------------------------------------------
# Event CRUD endpoints
# ---------------------------------------------------------------------------


def test_crud_calendar_add_update_delete_round_trip(client_state):
    client, box = client_state
    add = client.post(
        "/api/calendar",
        json={
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
    resp = client.post(
        "/api/calendar/bulk",
        json=[
            {
                "title": "A",
                "start": "2026-05-01T09:00:00+00:00",
                "end": "2026-05-01T10:00:00+00:00",
            },
            {
                "title": "B",
                "start": "2026-05-01T11:00:00+00:00",
                "end": "2026-05-01T12:00:00+00:00",
            },
        ],
    )
    assert resp.status_code == 200
    assert len(resp.json()["calendar_events"]) == 2
    assert len(box["state"].calendar) == 2


# ---------------------------------------------------------------------------
# Event complete / skip / unskip / uncomplete
# ---------------------------------------------------------------------------


def test_complete_event_marks_done(client_state):
    client, box = client_state
    today = _now_local().date()
    box["state"] = _make_state(today)

    resp = client.post("/api/events/ev-today/complete")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    ev = next(e for e in box["state"].calendar if e.id == "ev-today")
    assert ev.completed_at is not None
    assert ev.skipped_at is None


def test_uncomplete_event_clears_completed_at(client_state):
    client, box = client_state
    today = _now_local().date()
    box["state"] = _make_state(today)

    client.post("/api/events/ev-today/complete")
    resp = client.post("/api/events/ev-today/uncomplete")
    assert resp.status_code == 200
    ev = next(e for e in box["state"].calendar if e.id == "ev-today")
    assert ev.completed_at is None


def test_skip_event_sets_reason(client_state):
    client, box = client_state
    today = _now_local().date()
    box["state"] = _make_state(today)

    resp = client.post(
        "/api/events/ev-today/skip", json={"reason": "rolled my ankle"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["event_id"] == "ev-today"
    assert body["skipped_at"]

    ev = next(e for e in box["state"].calendar if e.id == "ev-today")
    assert ev.skipped_at is not None
    assert ev.skip_reason == "rolled my ankle"
    assert ev.completed_at is None


def test_skip_event_requires_reason(client_state):
    client, box = client_state
    today = _now_local().date()
    box["state"] = _make_state(today)

    resp = client.post("/api/events/ev-today/skip", json={"reason": "   "})
    assert resp.status_code == 400


def test_skip_event_404_on_unknown(client_state):
    client, box = client_state
    box["state"] = _make_state(_now_local().date())
    resp = client.post(
        "/api/events/no-such-event/skip", json={"reason": "x"}
    )
    assert resp.status_code == 404


def test_unskip_event_clears_skip_state(client_state):
    client, box = client_state
    today = _now_local().date()
    box["state"] = _make_state(today)

    client.post("/api/events/ev-today/skip", json={"reason": "rolled my ankle"})
    resp = client.post("/api/events/ev-today/unskip")
    assert resp.status_code == 200
    ev = next(e for e in box["state"].calendar if e.id == "ev-today")
    assert ev.skipped_at is None
    assert ev.skip_reason is None


def test_move_event_shifts_start_and_end(client_state):
    client, box = client_state
    today = _now_local().date()
    box["state"] = _make_state(today)
    original = box["state"].calendar[0]
    orig_start = original.start
    orig_end = original.end

    resp = client.post(
        f"/api/events/{original.id}/move", json={"minutes": 30}
    )
    assert resp.status_code == 200
    moved = next(e for e in box["state"].calendar if e.id == original.id)
    assert (moved.start - orig_start) == timedelta(minutes=30)
    assert (moved.end - orig_end) == timedelta(minutes=30)


def test_move_event_404_on_unknown(client_state):
    client, _ = client_state
    resp = client.post("/api/events/no-such/move", json={"minutes": 10})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/chat
# ---------------------------------------------------------------------------


def test_chat_applies_state_when_assistant_calls_crud_tools(client_state):
    """POST /api/chat: the chat assistant emits CRUD tool calls; the
    route persists the mutated state and responds with applied=true + diff."""
    from vessel.pwa import routes as r

    client, box = client_state
    box["state"] = StateData()

    fake = _ChatLLM([
        _chat_msg(calls=[(
            "add_calendar_event",
            {"fields": {
                "title": "Dentist",
                "start": "2026-05-08T10:00:00+00:00",
                "end": "2026-05-08T10:30:00+00:00",
            }},
        )]),
        _chat_msg(text="added dentist appointment"),
    ])

    r._set_chat_client_for_test(_ChatClient(fake))
    try:
        resp = client.post("/api/chat", json={"text": "dentist appointment May 8 at 10am"})
    finally:
        r._set_chat_client_for_test(None)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["applied"] is True
    assert "diff" in body
    assert body["diff"]["summary"]["calendar"]["added"] == 1
    assert body["assistant"]["stopped_reason"] == "completed"
    assert body["assistant"]["tool_calls"][0]["name"] == "add_calendar_event"
    assert body["assistant"]["tool_calls"][0]["ok"] is True
    assert any(e.title == "Dentist" for e in box["state"].calendar)


def test_chat_text_only_reply_does_not_mutate(client_state):
    """If the chat assistant emits no tool calls, the state must be untouched
    and `applied` must be False."""
    from vessel.pwa import routes as r

    client, box = client_state
    today = _now_local().date()
    box["state"] = _make_state(today)
    snapshot = box["state"].model_dump_json()

    fake = _ChatLLM([_chat_msg(text="I only manage your calendar")])

    r._set_chat_client_for_test(_ChatClient(fake))
    try:
        resp = client.post("/api/chat", json={"text": "thanks"})
    finally:
        r._set_chat_client_for_test(None)

    assert resp.status_code == 200
    body = resp.json()
    assert body["applied"] is False
    assert body["diff"] is None
    assert body["assistant"]["tool_calls"] == []
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


def test_skip_event_invokes_assistant_when_llm_configured(client_state, monkeypatch):
    """End-to-end: POST /api/events/{id}/skip with a reason that causes the
    assistant to create a new event. Both stages happen in one HTTP call."""
    client, box = client_state

    now = _now_local()
    ev = CalendarEvent(
        id="cal_dentist_20260501",
        title="Dentist",
        start=now + timedelta(hours=2),
        end=now + timedelta(hours=2, minutes=30),
    )
    box["state"] = StateData(calendar=[ev])

    from vessel.config import get_settings as _gs
    settings = _gs()
    monkeypatch.setattr(settings, "groq_api_key", "fake-key", raising=False)
    monkeypatch.setattr(settings, "groq_model", "fake-model", raising=False)

    from dataclasses import dataclass as _dc, field as _field

    @_dc
    class _FnCall2:
        name: str
        arguments: str

    @_dc
    class _FakeTc:
        id: str
        function: _FnCall2
        type: str = "function"

    @_dc
    class _FakeMsg2:
        content: str = ""
        tool_calls: list = _field(default_factory=list)

    @_dc
    class _FakeCh:
        message: _FakeMsg2

    @_dc
    class _FakeRsp:
        choices: list

    import json as _json
    new_start = (now + timedelta(days=7)).replace(microsecond=0)
    queue = [
        _FakeMsg2(
            content="",
            tool_calls=[
                _FakeTc(
                    id="c0",
                    function=_FnCall2(
                        name="add_calendar_event",
                        arguments=_json.dumps({"fields": {
                            "title": "Dentist",
                            "start": new_start.isoformat(),
                            "end": (new_start + timedelta(minutes=30)).isoformat(),
                        }}),
                    ),
                )
            ],
        ),
        _FakeMsg2(content="rescheduled dentist to next week"),
    ]

    class _Client2:
        def __init__(self, q):
            self._q = q
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            msg = self._q.pop(0) if self._q else _FakeMsg2(content="")
            return _FakeRsp(choices=[_FakeCh(message=msg)])

    import openai as _openai
    monkeypatch.setattr(_openai, "AsyncOpenAI", lambda **kwargs: _Client2(queue))

    resp = client.post(
        f"/api/events/{ev.id}/skip",
        json={"reason": "moved to next week"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True

    a = body["assistant"]
    assert a["invoked"] is True
    assert a["mutated"] is True
    assert a["stopped_reason"] == "completed"
    assert len(a["tool_calls"]) == 1
    assert a["tool_calls"][0]["ok"] is True

    # New event added to calendar.
    assert any(
        e.title == "Dentist" and e.id != ev.id
        for e in box["state"].calendar
    )


def test_skip_event_works_when_llm_unconfigured(client_state, monkeypatch):
    """If GROQ_API_KEY is unset, skip must still mark the event and return 200."""
    client, box = client_state
    now = _now_local()
    ev = CalendarEvent(
        id="cal_gym_20260501",
        title="Gym",
        start=now + timedelta(hours=1),
        end=now + timedelta(hours=2),
    )
    box["state"] = StateData(calendar=[ev])

    from vessel.config import get_settings as _gs
    settings = _gs()
    monkeypatch.setattr(settings, "groq_api_key", None, raising=False)

    resp = client.post(
        f"/api/events/{ev.id}/skip",
        json={"reason": "not today"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["assistant"]["invoked"] is False
