"""Tests for the LLM tool-use loop and the skip-with-reason assistant.

The loop is the heart of every LLM-using surface (skip + chat). The
fake client mimics the OpenAI ChatCompletion response shape so the
loop runs end-to-end without an actual LLM call.

Test coverage:
- One-shot completion (no tool calls) → final_message lands.
- Single tool call → executed against state, result fed back, then
  model finishes.
- Multiple tool calls in one assistant turn → all execute.
- Cap fires after MAX_TOOL_CALLS and stops the loop cleanly.
- CRUD error from a tool call comes back to the model and the loop continues.
- Unknown tool name → marked as error in the LoopResult.
- Invalid JSON args → marked as error, loop continues.
- skip_assistant happy path: "moved to next Friday" creates new event.
- skip_assistant does nothing when reason has no actionable intent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from vessel.assistant import run_skip_assistant
from vessel.assistant.tool_loop import MAX_TOOL_CALLS, ToolCall, tool_loop
from vessel.assistant.tool_schema import TOOLS
from vessel.models import CalendarEvent, StateData


# ---------------------------------------------------------------------------
# Fake OpenAI ChatCompletion response shape
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


def _msg(text: str = "", calls: list[tuple[str, dict[str, Any]]] = None) -> _FakeMessage:
    """Build an assistant message: either a plain text reply or one with tool_calls."""
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


def _resp(msg: _FakeMessage) -> _FakeResp:
    return _FakeResp(choices=[_FakeChoice(message=msg)])


class _ScriptedLLM:
    """Calls `chat.completions.create` returns a queued response per call."""

    def __init__(self, scripted: list[_FakeMessage]):
        self._queue = list(scripted)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs) -> _FakeResp:
        self.calls.append(kwargs)
        if not self._queue:
            return _resp(_msg(text="(done)"))
        return _resp(self._queue.pop(0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_event(
    state: StateData,
    eid: str = "ev1",
    title: str = "Test event",
    start: datetime | None = None,
    end: datetime | None = None,
) -> CalendarEvent:
    now = start or datetime(2026, 5, 1, 9, tzinfo=timezone.utc)
    ev = CalendarEvent(
        id=eid,
        title=title,
        start=now,
        end=end or now + timedelta(hours=1),
    )
    state.calendar.append(ev)
    return ev


# ---------------------------------------------------------------------------
# tool_loop: pure mechanics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_loop_no_tool_calls_returns_final_message():
    state = StateData()
    fake = _ScriptedLLM([_msg(text="hello there")])
    result = await tool_loop(
        chat_complete=fake,
        model="x",
        system_prompt="be helpful",
        user_message="hi",
        state=state,
        tools=TOOLS,
    )
    assert result.stopped_reason == "completed"
    assert result.final_message == "hello there"
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_tool_loop_executes_a_single_delete_then_finishes():
    state = StateData()
    ev = _seed_event(state, "ev-delete-me")
    fake = _ScriptedLLM(
        [
            _msg(calls=[("delete_calendar_event", {"id": ev.id})]),
            _msg(text="deleted 1 event"),
        ]
    )
    result = await tool_loop(
        chat_complete=fake, model="x", system_prompt="x",
        user_message="delete it", state=state, tools=TOOLS,
    )
    assert result.stopped_reason == "completed"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "delete_calendar_event"
    assert result.tool_calls[0].error is None
    assert state.calendar == []


@pytest.mark.asyncio
async def test_tool_loop_executes_multiple_tool_calls_in_one_assistant_turn():
    state = StateData()
    ids = []
    for i in range(3):
        ev = _seed_event(
            state,
            eid=f"ev-{i}",
            title=f"Event {i}",
            start=datetime(2026, 5, i + 1, 9, tzinfo=timezone.utc),
        )
        ids.append(ev.id)
    fake = _ScriptedLLM(
        [
            _msg(calls=[("delete_calendar_event", {"id": eid}) for eid in ids]),
            _msg(text="deleted 3"),
        ]
    )
    result = await tool_loop(
        chat_complete=fake, model="x", system_prompt="x",
        user_message="delete all", state=state, tools=TOOLS,
    )
    assert result.stopped_reason == "completed"
    assert len(result.tool_calls) == 3
    assert all(c.error is None for c in result.tool_calls)
    assert state.calendar == []


@pytest.mark.asyncio
async def test_tool_loop_caps_at_max_tool_calls():
    state = StateData()

    class _Forever:
        calls: list = []

        async def __call__(self, **kwargs):
            return _resp(_msg(calls=[("get_state", {})]))

    forever = _Forever()
    result = await tool_loop(
        chat_complete=forever, model="x", system_prompt="x",
        user_message="loop please", state=state, tools=TOOLS,
        max_tool_calls=3,
    )
    assert result.stopped_reason == "cap_hit"
    assert len(result.tool_calls) == 3


@pytest.mark.asyncio
async def test_tool_loop_crud_error_does_not_kill_loop():
    state = StateData()
    fake = _ScriptedLLM(
        [
            _msg(calls=[("delete_calendar_event", {"id": "does-not-exist"})]),
            _msg(text="oops, not found"),
        ]
    )
    result = await tool_loop(
        chat_complete=fake, model="x", system_prompt="x",
        user_message="delete x", state=state, tools=TOOLS,
    )
    assert result.stopped_reason == "completed"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].error is not None
    assert "not_found" in result.tool_calls[0].result.get("kind", "")


@pytest.mark.asyncio
async def test_tool_loop_unknown_tool_name_marked_error():
    state = StateData()
    fake = _ScriptedLLM(
        [
            _msg(calls=[("nonexistent_tool", {"x": 1})]),
            _msg(text="that tool doesn't exist"),
        ]
    )
    result = await tool_loop(
        chat_complete=fake, model="x", system_prompt="x",
        user_message="x", state=state, tools=TOOLS,
    )
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].error
    assert "unknown tool" in result.tool_calls[0].error


@pytest.mark.asyncio
async def test_tool_loop_invalid_json_args_marked_error():
    state = StateData()
    bad = _FakeMessage(
        content="",
        tool_calls=[
            _FakeToolCall(id="c0", function=_FnCall(name="get_state", arguments="not json"))
        ],
    )
    state2 = {"step": 0}

    async def two_step(**kwargs):
        state2["step"] += 1
        return _resp(bad if state2["step"] == 1 else _msg(text="done"))

    result = await tool_loop(
        chat_complete=two_step, model="x", system_prompt="x",
        user_message="x", state=state, tools=TOOLS,
    )
    assert len(result.tool_calls) == 1
    assert "invalid JSON" in (result.tool_calls[0].error or "")


# ---------------------------------------------------------------------------
# skip_assistant: end-to-end with a scripted LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skip_assistant_creates_new_event_when_reason_says_reschedule():
    """End-to-end: user skipped a dentist appointment with reason 'moved to
    next Friday'. The assistant creates a new calendar event."""
    state = StateData()
    skipped = CalendarEvent(
        id="cal_dentist_20260501",
        title="Dentist",
        start=datetime(2026, 5, 1, 10, tzinfo=timezone.utc),
        end=datetime(2026, 5, 1, 10, 30, tzinfo=timezone.utc),
        skipped_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        skip_reason="moved to next Friday",
    )
    new_start = datetime(2026, 5, 8, 10, tzinfo=timezone.utc)
    fake = _ScriptedLLM(
        [
            _msg(calls=[(
                "add_calendar_event",
                {"fields": {
                    "title": "Dentist",
                    "start": new_start.isoformat(),
                    "end": (new_start + timedelta(minutes=30)).isoformat(),
                }},
            )]),
            _msg(text="rescheduled dentist to next Friday"),
        ]
    )

    class _Client:
        def __init__(self, fn):
            self.chat = self
            self.completions = self
            self._fn = fn

        async def create(self, **kwargs):
            return await self._fn(**kwargs)

    result = await run_skip_assistant(
        reason="moved to next Friday",
        skipped_task=skipped,
        state=state,
        client=_Client(fake),
        model="x",
    )
    assert result.stopped_reason == "completed"
    assert len(result.mutating_calls()) == 1
    assert len(state.calendar) == 1
    assert state.calendar[0].title == "Dentist"
    assert "rescheduled" in result.final_message.lower()


@pytest.mark.asyncio
async def test_skip_assistant_does_nothing_when_reason_is_just_explanation():
    """Reason like "just not feeling it" — the assistant should reply
    with a text message and not call any tools."""
    state = StateData()
    ev = _seed_event(state, "ev-gym", "Gym")
    skipped = CalendarEvent(
        id="cal_gym_20260501",
        title="Gym",
        start=datetime(2026, 5, 1, 7, tzinfo=timezone.utc),
        end=datetime(2026, 5, 1, 8, tzinfo=timezone.utc),
    )
    fake = _ScriptedLLM([_msg(text="ok, just for today")])

    class _Client:
        def __init__(self, fn):
            self.chat = self
            self.completions = self
            self._fn = fn

        async def create(self, **kwargs):
            return await self._fn(**kwargs)

    result = await run_skip_assistant(
        reason="just not feeling it tonight",
        skipped_task=skipped,
        state=state,
        client=_Client(fake),
        model="x",
    )
    assert result.tool_calls == []
    assert len(state.calendar) == 1  # the original ev-gym survives


@pytest.mark.asyncio
async def test_skip_assistant_user_message_carries_today_date():
    """The user message must include the current local datetime, the weekday,
    and an explicit 'start >= today' rule."""
    state = StateData()
    skipped = CalendarEvent(
        id="cal_x_20260429",
        title="Meeting",
        start=datetime(2026, 4, 29, 10, tzinfo=timezone.utc),
        end=datetime(2026, 4, 29, 11, tzinfo=timezone.utc),
    )
    captured: list[dict[str, Any]] = []

    async def capture(**kwargs):
        captured.append(kwargs)
        return _resp(_msg(text="ok"))

    class _Client:
        def __init__(self, fn):
            self.chat = self
            self.completions = self
            self._fn = fn

        async def create(self, **kwargs):
            return await self._fn(**kwargs)

    fixed_now = datetime(2026, 4, 29, 21, 30, tzinfo=timezone.utc)
    await run_skip_assistant(
        reason="reschedule for tomorrow",
        skipped_task=skipped,
        state=state,
        client=_Client(capture),
        model="x",
        now=fixed_now,
    )
    user_msg = next(
        m["content"] for m in captured[0]["messages"] if m["role"] == "user"
    )
    assert "2026-04-29" in user_msg
    assert "Wednesday" in user_msg  # 2026-04-29 is a Wednesday
    assert "Now:" in user_msg
    assert "MUST be on or after 2026-04-29" in user_msg


@pytest.mark.asyncio
async def test_skip_assistant_user_message_includes_reason_and_event_summary():
    """The prompt must carry the reason verbatim and include the event summary."""
    state = StateData()
    _seed_event(state, "ev-upcoming", "Doctor")
    skipped = CalendarEvent(
        id="cal_dentist_20260501",
        title="Dentist",
        start=datetime(2026, 5, 1, 10, tzinfo=timezone.utc),
        end=datetime(2026, 5, 1, 10, 30, tzinfo=timezone.utc),
    )
    captured: list[dict[str, Any]] = []

    async def capture(**kwargs):
        captured.append(kwargs)
        return _resp(_msg(text="noted"))

    class _Client:
        def __init__(self, fn):
            self.chat = self
            self.completions = self
            self._fn = fn

        async def create(self, **kwargs):
            return await self._fn(**kwargs)

    await run_skip_assistant(
        reason="moved to next week",
        skipped_task=skipped,
        state=state,
        client=_Client(capture),
        model="x",
    )
    user_msg = next(
        m["content"] for m in captured[0]["messages"] if m["role"] == "user"
    )
    assert "moved to next week" in user_msg
    assert "Dentist" in user_msg
    # The skipped event's id must NOT appear in the open-events list.
    open_section = user_msg.split("Upcoming calendar events")[1]
    assert skipped.id not in open_section
