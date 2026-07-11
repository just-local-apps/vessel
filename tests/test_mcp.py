"""In-process MCP integration test.

Spins up the Vessel MCP server in memory (no SSE, no FastAPI) and
exercises every tool through a real ClientSession.
"""
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from vessel.mcp_server import build_mcp_server
from vessel.models import StateData


class FakeStateStore:
    def __init__(self, initial: StateData | None = None):
        self.state = initial or StateData()

    async def read(self) -> StateData:
        return StateData.model_validate(self.state.model_dump(mode="python"))

    async def write(self, state: StateData) -> None:
        self.state = state


# ---- Scripted OpenAI-shaped client (mirrors the chat-tests pattern) ----


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


def _msg(text: str = "", calls: list[tuple[str, dict[str, Any]]] | None = None) -> _FakeMessage:
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


class _ScriptedChat:
    """Returns one queued response per `create()` call. `delay` makes
    the call hold for that many seconds — used to verify the Gate
    stays engaged for the lifetime of the tool invocation."""

    def __init__(self, scripted: list[_FakeMessage], *, delay: float = 0.0):
        self._queue = list(scripted)
        self._delay = delay
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs) -> _FakeResp:
        if self._delay:
            await asyncio.sleep(self._delay)
        self.calls.append(kwargs)
        if not self._queue:
            return _FakeResp(choices=[_FakeChoice(message=_msg(text="(done)"))])
        return _FakeResp(choices=[_FakeChoice(message=self._queue.pop(0))])


class _Client:
    """AsyncOpenAI-shaped wrapper around a scripted callable."""

    def __init__(self, fn: _ScriptedChat):
        self.chat = self
        self.completions = self
        self._fn = fn

    async def create(self, **kwargs):
        return await self._fn(**kwargs)


def _build(*, scripted: list[_FakeMessage] | None = None, delay: float = 0.0):
    store = FakeStateStore()
    fn = _ScriptedChat(scripted or [_msg(text="(noop)")], delay=delay)
    client = _Client(fn)

    server = build_mcp_server(
        read_state=store.read,
        write_state=store.write,
        chat_client=client,
        chat_model="fake-model",
    )
    return server, store, fn


@pytest.mark.asyncio
async def test_get_state_round_trip():
    server, store, _llm = _build()
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("get_state", {})
        text = result.content[0].text
        data = json.loads(text)
    assert data == {
        "projects": [],
        "tasks": [],
        "calendar": [],
        "routines": [],
        "priority_ranking": [],
        "last_break_acknowledged_at": None,
    }


@pytest.mark.asyncio
async def test_list_tools():
    server, *_ = _build()
    async with create_connected_server_and_client_session(server) as session:
        tools = await session.list_tools()
    names = {t.name for t in tools.tools}
    assert {"get_state", "apply_instruction"} <= names


@pytest.mark.asyncio
async def test_apply_instruction_writes_state():
    """The chat assistant emits an `add_project` tool call → MCP runs
    it through CRUD, persists state, and returns applied=true + diff."""
    server, store, fn = _build(scripted=[
        _msg(calls=[(
            "add_project",
            {"fields": {
                "id": "p_taxes",
                "name": "Taxes",
                "status": "active",
                "tracked": True,
                "cadence": "daily",
                "importance": "medium",
            }},
        )]),
        _msg(text="added taxes project"),
    ])

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(
            "apply_instruction", {"text": "set up a project for taxes"}
        )
        data = json.loads(result.content[0].text)

    assert data["applied"] is True
    diff = data["diff"]
    assert diff["summary"]["projects"] == {"added": 1, "removed": 0, "changed": 0}
    assert diff["projects"]["added"][0]["id"] == "p_taxes"
    assert diff["tasks"] == {"added": [], "removed": [], "changed": []}
    assert store.state.projects[0].id == "p_taxes"
    # The assistant's summary surfaces alongside the diff.
    assert data["assistant"]["stopped_reason"] == "completed"
    assert data["assistant"]["tool_calls"][0]["name"] == "add_project"
    assert data["assistant"]["tool_calls"][0]["ok"] is True
    # Two LLM calls: the tool-call turn + the final text turn.
    assert len(fn.calls) == 2


@pytest.mark.asyncio
async def test_apply_instruction_text_only_reply_does_not_write():
    """If the assistant emits no tool calls (just a text reply), state
    must be untouched and `applied` must be False. Replaces the old
    clarifications path — the new agent never asks questions."""
    server, store, _fn = _build(scripted=[
        _msg(text="nothing actionable in that note")
    ])
    initial_state_dump = store.state.model_dump(mode="json")

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(
            "apply_instruction", {"text": "remind me about the gym"}
        )
        data = json.loads(result.content[0].text)

    assert data["applied"] is False
    assert data["diff"] is None
    assert data["assistant"]["summary"] == "nothing actionable in that note"
    assert data["assistant"]["tool_calls"] == []
    # State must NOT have been written.
    assert store.state.model_dump(mode="json") == initial_state_dump


@pytest.mark.asyncio
async def test_apply_instruction_missing_text_is_rejected():
    server, _store, _llm = _build()
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("apply_instruction", {})
    # MCP enforces required arguments via inputSchema → isError=True.
    assert result.isError


# ---------------------------------------------------------------------------
# CRUD tools — exposed alongside apply_instruction so any MCP client (Claude
# Desktop or the vessel chat assistant) can mutate state without an LLM.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_tools_includes_crud_surface():
    server, *_ = _build()
    async with create_connected_server_and_client_session(server) as session:
        tools = await session.list_tools()
    names = {t.name for t in tools.tools}
    expected = {
        "add_project", "add_projects_bulk", "update_project", "delete_project",
        "add_task", "add_tasks_bulk", "update_task", "delete_task",
        "add_calendar_event", "add_calendar_events_bulk",
        "update_calendar_event", "delete_calendar_event",
        "add_routine", "add_routines_bulk", "update_routine", "delete_routine",
    }
    missing = expected - names
    assert not missing, f"missing CRUD tools: {missing}"


@pytest.mark.asyncio
async def test_add_project_then_add_task_persists():
    server, store, *_ = _build()
    async with create_connected_server_and_client_session(server) as session:
        proj_resp = await session.call_tool(
            "add_project", {"fields": {"name": "Health"}}
        )
        proj = json.loads(proj_resp.content[0].text)
        assert proj["ok"] is True
        assert proj["project"]["id"] == "p_health"

        task_resp = await session.call_tool(
            "add_task",
            {"fields": {"title": "Buy vitamins", "project_id": "p_health"}},
        )
        task = json.loads(task_resp.content[0].text)
        assert task["ok"] is True
        assert task["task"]["title"] == "Buy vitamins"

    # State actually persisted.
    state = store.state
    assert any(p.id == "p_health" for p in state.projects)
    assert any(t.title == "Buy vitamins" for t in state.tasks)


@pytest.mark.asyncio
async def test_add_tasks_bulk_creates_many_in_one_call():
    server, store, *_ = _build()
    async with create_connected_server_and_client_session(server) as session:
        await session.call_tool(
            "add_project", {"fields": {"name": "Demo"}}
        )
        bulk = await session.call_tool(
            "add_tasks_bulk",
            {
                "items": [
                    {"title": "A", "project_id": "p_demo"},
                    {"title": "B", "project_id": "p_demo"},
                    {"title": "C", "project_id": "p_demo"},
                ]
            },
        )
        body = json.loads(bulk.content[0].text)
    assert body["ok"] is True
    assert len(body["tasks"]) == 3
    assert len(store.state.tasks) == 3


@pytest.mark.asyncio
async def test_add_calendar_events_bulk_creates_many():
    server, store, *_ = _build()
    async with create_connected_server_and_client_session(server) as session:
        await session.call_tool(
            "add_project", {"fields": {"name": "Family"}}
        )
        items = [
            {
                "project_id": "p_family", "title": f"event-{i}",
                "start": f"2026-05-0{i+1}T09:00:00+00:00",
                "end": f"2026-05-0{i+1}T10:00:00+00:00",
            }
            for i in range(4)
        ]
        bulk = await session.call_tool(
            "add_calendar_events_bulk", {"items": items}
        )
        body = json.loads(bulk.content[0].text)
    assert body["ok"] is True
    assert len(body["calendar_events"]) == 4
    assert len(store.state.calendar) == 4


@pytest.mark.asyncio
async def test_update_task_changes_recurrence_and_start_after():
    server, store, *_ = _build()
    async with create_connected_server_and_client_session(server) as session:
        await session.call_tool("add_project", {"fields": {"name": "Demo"}})
        add = await session.call_tool(
            "add_task",
            {"fields": {"title": "Wash dishes", "project_id": "p_demo"}},
        )
        task_id = json.loads(add.content[0].text)["task"]["id"]
        upd = await session.call_tool(
            "update_task",
            {
                "id": task_id,
                "fields": {"recurrence": "daily", "start_after": "19:00:00"},
            },
        )
    body = json.loads(upd.content[0].text)
    assert body["ok"] is True
    assert body["task"]["recurrence"] == "daily"
    task = next(t for t in store.state.tasks if t.id == task_id)
    assert task.recurrence == "daily"
    assert task.start_after.isoformat().startswith("19:00")


@pytest.mark.asyncio
async def test_delete_task_removes_from_state():
    server, store, *_ = _build()
    async with create_connected_server_and_client_session(server) as session:
        await session.call_tool("add_project", {"fields": {"name": "Demo"}})
        add = await session.call_tool(
            "add_task",
            {"fields": {"title": "Throwaway", "project_id": "p_demo"}},
        )
        task_id = json.loads(add.content[0].text)["task"]["id"]
        rm = await session.call_tool("delete_task", {"id": task_id})
    body = json.loads(rm.content[0].text)
    assert body["ok"] is True
    assert all(t.id != task_id for t in store.state.tasks)


@pytest.mark.asyncio
async def test_delete_project_with_open_task_returns_still_referenced():
    server, store, *_ = _build()
    async with create_connected_server_and_client_session(server) as session:
        await session.call_tool("add_project", {"fields": {"name": "Demo"}})
        await session.call_tool(
            "add_task",
            {"fields": {"title": "Pinned", "project_id": "p_demo"}},
        )
        rm = await session.call_tool("delete_project", {"id": "p_demo"})
    body = json.loads(rm.content[0].text)
    assert body.get("kind") == "still_referenced"
    # Project survives the failed delete.
    assert any(p.id == "p_demo" for p in store.state.projects)


@pytest.mark.asyncio
async def test_add_task_without_project_id_returns_missing_reference():
    server, *_ = _build()
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(
            "add_task", {"fields": {"title": "Orphan"}}
        )
    body = json.loads(result.content[0].text)
    assert body.get("kind") == "missing_reference"


@pytest.mark.asyncio
async def test_routine_round_trip_via_mcp():
    server, store, *_ = _build()
    async with create_connected_server_and_client_session(server) as session:
        add = await session.call_tool(
            "add_routine",
            {
                "fields": {
                    "label": "Morning gym",
                    "start_time": "07:00:00",
                    "duration_minutes": 60,
                }
            },
        )
        rid = json.loads(add.content[0].text)["routine"]["id"]
        upd = await session.call_tool(
            "update_routine",
            {"id": rid, "fields": {"duration_minutes": 45}},
        )
        rm = await session.call_tool("delete_routine", {"id": rid})
    assert json.loads(upd.content[0].text)["routine"]["duration_minutes"] == 45
    assert json.loads(rm.content[0].text)["ok"] is True
    assert store.state.routines == []
