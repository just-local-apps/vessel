"""Tests for the LLM tool-use loop and the skip-with-reason assistant.

The loop is the heart of every LLM-using surface (skip + chat). The
fake client mimics the OpenAI ChatCompletion response shape so the
loop runs end-to-end without an actual LLM call.

Test coverage:
- One-shot completion (no tool calls) → final_message lands.
- Single tool call → executed against state, result fed back, then
  model finishes.
- Multiple tool calls in one assistant turn (the bulk-delete pattern)
  → all execute.
- Cap fires after MAX_TOOL_CALLS and stops the loop cleanly.
- CRUD error from a tool call comes back to the model as a
  `kind:"…"` envelope and the loop continues.
- Unknown tool name → marked as error in the LoopResult.
- Invalid JSON args → marked as error, loop continues.
- skip_assistant happy path: "delete all wash dishes" → all open
  Wash dishes tasks deleted; final_message present.
- skip_assistant ignores the just-skipped task in its prompt (it's
  already archived).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import pytest

from vessel.assistant import run_skip_assistant
from vessel.assistant.tool_loop import MAX_TOOL_CALLS, ToolCall, tool_loop
from vessel.assistant.tool_schema import TOOLS
from vessel.models import StateData
from vessel.models.enums import Cadence, ProjectStatus, Tier, TimeWindow
from vessel.models.state import Project, Task


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
            # Default: empty completion = "I'm done".
            return _resp(_msg(text="(done)"))
        return _resp(self._queue.pop(0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_project(state: StateData, pid: str = "p_demo", name: str = "Demo") -> None:
    state.projects.append(
        Project(
            id=pid,
            name=name,
            status=ProjectStatus.active,
            tracked=True,
            cadence=Cadence.event_driven,
            last_touched=datetime(2026, 4, 28, tzinfo=timezone.utc),
        )
    )


def _seed_wash_dishes(state: StateData, dates: list[date]) -> list[str]:
    """Insert one open 'Wash dishes' task per date. Returns the ids."""
    ids = []
    base_dt = datetime(2026, 4, 28, tzinfo=timezone.utc)
    for d in dates:
        suffix = d.strftime("%Y%m%d")
        tid = f"task_wash_dishes_{suffix}"
        state.tasks.append(
            Task(
                id=tid,
                project_id="p_demo",
                title="Wash dishes",
                time_window=TimeWindow.evening,
                tier=Tier.flex,
                estimated_minutes=15,
                due_date=d,
                created_at=base_dt,
            )
        )
        ids.append(tid)
    return ids


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
    _seed_project(state)
    ids = _seed_wash_dishes(state, [date(2026, 4, 30)])
    fake = _ScriptedLLM(
        [
            _msg(calls=[("delete_task", {"id": ids[0]})]),
            _msg(text="deleted 1 task"),
        ]
    )
    result = await tool_loop(
        chat_complete=fake, model="x", system_prompt="x",
        user_message="delete it", state=state, tools=TOOLS,
    )
    assert result.stopped_reason == "completed"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "delete_task"
    assert result.tool_calls[0].error is None
    assert state.tasks == []  # task actually removed from state


@pytest.mark.asyncio
async def test_tool_loop_executes_multiple_tool_calls_in_one_assistant_turn():
    state = StateData()
    _seed_project(state)
    ids = _seed_wash_dishes(
        state, [date(2026, 4, 30), date(2026, 5, 1), date(2026, 5, 2)]
    )
    fake = _ScriptedLLM(
        [
            _msg(calls=[("delete_task", {"id": tid}) for tid in ids]),
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
    assert state.tasks == []


@pytest.mark.asyncio
async def test_tool_loop_caps_at_max_tool_calls():
    state = StateData()
    _seed_project(state)
    # Always emit one tool call, never finish — the cap must stop us.
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
    _seed_project(state)
    fake = _ScriptedLLM(
        [
            _msg(calls=[("delete_task", {"id": "task_does_not_exist"})]),
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
    # Hand-craft a bad-args call (raw arguments string isn't valid JSON).
    bad = _FakeMessage(
        content="",
        tool_calls=[
            _FakeToolCall(id="c0", function=_FnCall(name="get_state", arguments="not json"))
        ],
    )
    class _Once:
        async def __call__(self, **kwargs):
            return _resp(bad if not getattr(self, "fired", False) else _msg(text="done"))
    fake = _Once()
    # Two-step trick so the first call returns bad, the second returns done.
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
async def test_skip_assistant_deletes_all_wash_dishes_when_reason_says_so():
    """End-to-end: user skipped one wash-dishes with reason "no more".
    The assistant gets the reason + state and is expected to call
    delete_task on every other open Wash dishes."""
    state = StateData()
    _seed_project(state)
    ids = _seed_wash_dishes(
        state,
        [date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 3)],
    )
    skipped = Task(
        id="task_wash_dishes_20260430",
        project_id="p_demo",
        title="Wash dishes",
        time_window=TimeWindow.evening,
        tier=Tier.flex,
        estimated_minutes=15,
        due_date=date(2026, 4, 30),
        created_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
        skipped_at=datetime(2026, 4, 30, tzinfo=timezone.utc),
        skip_reason="back pain. no more wash dishes",
    )
    fake = _ScriptedLLM(
        [
            _msg(calls=[("delete_task", {"id": tid}) for tid in ids]),
            _msg(text="deleted 3 wash-dishes tasks"),
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
        reason="back pain. no more wash dishes",
        skipped_task=skipped,
        state=state,
        client=_Client(fake),
        model="x",
    )
    assert result.stopped_reason == "completed"
    assert len(result.mutating_calls()) == 3
    assert all(t.title != "Wash dishes" for t in state.tasks), state.tasks
    assert "deleted" in result.final_message.lower()


@pytest.mark.asyncio
async def test_skip_assistant_does_nothing_when_reason_is_just_explanation():
    """Reason like "didn't feel like it" — the assistant should reply
    with a text message and not call any tools."""
    state = StateData()
    _seed_project(state)
    _seed_wash_dishes(state, [date(2026, 5, 1)])
    skipped = Task(
        id="task_wash_dishes_20260430",
        project_id="p_demo",
        title="Wash dishes",
        time_window=TimeWindow.evening,
        tier=Tier.flex,
        estimated_minutes=15,
        due_date=date(2026, 4, 30),
        created_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
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
        reason="didn't feel like it tonight",
        skipped_task=skipped,
        state=state,
        client=_Client(fake),
        model="x",
    )
    assert result.tool_calls == []
    assert len(state.tasks) == 1  # the future wash-dishes survives


@pytest.mark.asyncio
async def test_skip_assistant_user_message_carries_today_date():
    """Regression: the LLM was inventing yesterday's date for new
    tasks because it had no temporal grounding. The user message must
    now include the current local datetime, the weekday, and an
    explicit 'due_date >= today' rule."""
    state = StateData()
    _seed_project(state)
    skipped = Task(
        id="task_x_20260429", project_id="p_demo", title="x",
        time_window=TimeWindow.anytime, tier=Tier.flex,
        estimated_minutes=5, due_date=date(2026, 4, 29),
        created_at=datetime(2026, 4, 29, tzinfo=timezone.utc),
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
    # Header carries the absolute timestamp, the weekday name, and the
    # ISO date — three independent anchors so the model can't latch on
    # to a stale prior.
    assert "2026-04-29" in user_msg
    assert "Wednesday" in user_msg  # 2026-04-29 is a Wednesday
    assert "Now:" in user_msg
    # The "due_date MUST be on or after" guardrail is present.
    assert "MUST be on or after 2026-04-29" in user_msg


@pytest.mark.asyncio
async def test_skip_assistant_user_message_includes_reason_and_state_summary():
    """Inspect the prompt the assistant sends — it must carry the
    reason verbatim and a list of currently-open tasks so the model
    can act without an extra get_state round-trip."""
    state = StateData()
    _seed_project(state)
    _seed_wash_dishes(state, [date(2026, 5, 1), date(2026, 5, 2)])
    skipped = Task(
        id="task_wash_dishes_20260430",
        project_id="p_demo",
        title="Wash dishes",
        time_window=TimeWindow.evening,
        tier=Tier.flex,
        estimated_minutes=15,
        due_date=date(2026, 4, 30),
        created_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
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
        reason="back pain, no more wash dishes",
        skipped_task=skipped,
        state=state,
        client=_Client(capture),
        model="x",
    )
    user_msg = next(
        m["content"] for m in captured[0]["messages"] if m["role"] == "user"
    )
    assert "back pain, no more wash dishes" in user_msg
    assert "Wash dishes" in user_msg
    # The skipped task is named in the header (so the model knows what
    # was just archived) — but it must NOT appear in the open-tasks
    # list. Split on the open-tasks header and check only that section.
    open_section = user_msg.split("Open tasks currently in state")[1]
    assert skipped.id not in open_section
