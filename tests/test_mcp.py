"""In-process MCP integration test — calendar-only.

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
    assert data == {"calendar": []}


@pytest.mark.asyncio
async def test_list_tools():
    server, *_ = _build()
    async with create_connected_server_and_client_session(server) as session:
        tools = await session.list_tools()
    names = {t.name for t in tools.tools}
    assert {"get_state", "apply_instruction"} <= names


@pytest.mark.asyncio
async def test_list_tools_includes_crud_surface():
    server, *_ = _build()
    async with create_connected_server_and_client_session(server) as session:
        tools = await session.list_tools()
    names = {t.name for t in tools.tools}
    expected = {
        "add_calendar_event", "add_calendar_events_bulk",
        "update_calendar_event", "delete_calendar_event",
    }
    missing = expected - names
    assert not missing, f"missing CRUD tools: {missing}"


@pytest.mark.asyncio
async def test_apply_instruction_writes_state():
    """The chat assistant emits an `add_calendar_event` tool call → MCP runs
    it through CRUD, persists state, and returns applied=true + diff."""
    server, store, fn = _build(scripted=[
        _msg(calls=[(
            "add_calendar_event",
            {"fields": {
                "title": "Dentist",
                "start": "2026-05-08T10:00:00+00:00",
                "end": "2026-05-08T10:30:00+00:00",
            }},
        )]),
        _msg(text="added dentist appointment"),
    ])

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(
            "apply_instruction", {"text": "dentist appointment May 8 at 10am"}
        )
        data = json.loads(result.content[0].text)

    assert data["applied"] is True
    diff = data["diff"]
    assert diff["summary"]["calendar"] == {"added": 1, "removed": 0, "changed": 0}
    assert diff["calendar"]["added"][0]["title"] == "Dentist"
    assert store.state.calendar[0].title == "Dentist"
    assert data["assistant"]["stopped_reason"] == "completed"
    assert data["assistant"]["tool_calls"][0]["name"] == "add_calendar_event"
    assert data["assistant"]["tool_calls"][0]["ok"] is True


@pytest.mark.asyncio
async def test_apply_instruction_text_only_reply_does_not_write():
    """If the assistant emits no tool calls, state must be untouched."""
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
    assert store.state.model_dump(mode="json") == initial_state_dump


@pytest.mark.asyncio
async def test_apply_instruction_missing_text_is_rejected():
    server, _store, _llm = _build()
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("apply_instruction", {})
    assert result.isError


# ---------------------------------------------------------------------------
# CRUD tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_calendar_event_and_delete():
    server, store, *_ = _build()
    async with create_connected_server_and_client_session(server) as session:
        add = await session.call_tool(
            "add_calendar_event",
            {"fields": {
                "title": "Standup",
                "start": "2026-05-01T09:00:00+00:00",
                "end": "2026-05-01T09:30:00+00:00",
            }},
        )
        body = json.loads(add.content[0].text)
        assert body["ok"] is True
        ev_id = body["calendar_event"]["id"]

        rm = await session.call_tool("delete_calendar_event", {"id": ev_id})
        rm_body = json.loads(rm.content[0].text)
        assert rm_body["ok"] is True

    assert store.state.calendar == []


@pytest.mark.asyncio
async def test_add_calendar_events_bulk_creates_many():
    server, store, *_ = _build()
    async with create_connected_server_and_client_session(server) as session:
        bulk = await session.call_tool(
            "add_calendar_events_bulk",
            {
                "items": [
                    {
                        "title": f"event-{i}",
                        "start": f"2026-05-0{i+1}T09:00:00+00:00",
                        "end": f"2026-05-0{i+1}T10:00:00+00:00",
                    }
                    for i in range(4)
                ]
            },
        )
        body = json.loads(bulk.content[0].text)
    assert body["ok"] is True
    assert len(body["calendar_events"]) == 4
    assert len(store.state.calendar) == 4


@pytest.mark.asyncio
async def test_update_calendar_event_changes_location():
    server, store, *_ = _build()
    async with create_connected_server_and_client_session(server) as session:
        add = await session.call_tool(
            "add_calendar_event",
            {"fields": {
                "title": "Doctor",
                "start": "2026-05-01T14:00:00+00:00",
                "end": "2026-05-01T15:00:00+00:00",
            }},
        )
        ev_id = json.loads(add.content[0].text)["calendar_event"]["id"]
        upd = await session.call_tool(
            "update_calendar_event",
            {"id": ev_id, "fields": {"location": "123 Main St"}},
        )
    body = json.loads(upd.content[0].text)
    assert body["ok"] is True
    assert body["calendar_event"]["location"] == "123 Main St"
    ev = next(e for e in store.state.calendar if e.id == ev_id)
    assert ev.location == "123 Main St"


@pytest.mark.asyncio
async def test_delete_nonexistent_event_returns_not_found():
    server, *_ = _build()
    async with create_connected_server_and_client_session(server) as session:
        rm = await session.call_tool("delete_calendar_event", {"id": "no-such"})
    body = json.loads(rm.content[0].text)
    assert body.get("kind") == "not_found"
