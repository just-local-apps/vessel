"""Vessel's MCP server — direct LLM ↔ calendar state interface.

Exposes tools:
- `get_state` — read the current calendar
- `apply_instruction(text)` — natural-language → chat tool-loop → mutated state
- `add_calendar_event` — create an event directly
- `update_calendar_event` — update an event
- `delete_calendar_event` — delete an event
- `add_calendar_events_bulk` — create multiple events
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from mcp.server import Server
from mcp.types import TextContent, Tool

from . import crud as _crud
from .models import StateData

logger = logging.getLogger(__name__)


ReadStateFn = Callable[[], Awaitable[StateData]]
WriteStateFn = Callable[[StateData], Awaitable[None]]


def _state_to_text(state: StateData) -> TextContent:
    return TextContent(type="text", text=state.model_dump_json(indent=2))


def _diff_collection(before: list, after: list) -> dict[str, list]:
    before_by_id = {item.id: item for item in before}
    after_by_id = {item.id: item for item in after}
    added = [
        after_by_id[i].model_dump(mode="json")
        for i in after_by_id.keys() - before_by_id.keys()
    ]
    removed = [
        before_by_id[i].model_dump(mode="json")
        for i in before_by_id.keys() - after_by_id.keys()
    ]
    changed = []
    for i in before_by_id.keys() & after_by_id.keys():
        b = before_by_id[i].model_dump(mode="json")
        a = after_by_id[i].model_dump(mode="json")
        if b != a:
            changed.append({"id": i, "before": b, "after": a})
    return {"added": added, "removed": removed, "changed": changed}


def _state_diff(before: StateData, after: StateData) -> dict[str, Any]:
    diff: dict[str, Any] = {
        "calendar": _diff_collection(before.calendar, after.calendar),
    }
    summary = {
        "calendar": {
            "added": len(diff["calendar"]["added"]),
            "removed": len(diff["calendar"]["removed"]),
            "changed": len(diff["calendar"]["changed"]),
        }
    }
    diff["summary"] = summary
    return diff


def build_mcp_server(
    *,
    read_state: ReadStateFn,
    write_state: WriteStateFn,
    chat_client: Any,
    chat_model: str,
) -> Server:
    """Construct an MCP Server bound to the supplied dependencies."""
    server: Server = Server("vessel")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name="get_state",
                description="Read Vessel's current calendar — returns all calendar events.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="apply_instruction",
                description=(
                    "Apply a natural-language instruction to Vessel's calendar. "
                    "Runs the chat tool-loop assistant — the LLM infers from "
                    "(now, calendar, instruction) and calls calendar CRUD tools. "
                    "Returns: {applied: bool, diff: {...} | null, "
                    "assistant: {summary, tool_calls, stopped_reason}}. "
                    "Use this to add, update, or delete calendar events from "
                    "natural-language descriptions."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Instruction in plain English.",
                        }
                    },
                    "required": ["text"],
                },
            ),
            # ----- CRUD: calendar ------------------------------------------
            Tool(
                name="add_calendar_event",
                description=(
                    "Add ONE calendar event. Required: `title`, `start` (ISO8601), "
                    "`end` (ISO8601). Optional: `description`, `url`, `location`, "
                    "`arrive_by` (ISO8601). id auto-generates."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"fields": {"type": "object"}},
                    "required": ["fields"],
                },
            ),
            Tool(
                name="add_calendar_events_bulk",
                description="Add many calendar events in one call. `items` is an array of event dicts.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": ["items"],
                },
            ),
            Tool(
                name="update_calendar_event",
                description=(
                    "Update a subset of a calendar event's fields by id. "
                    "Accepted: title, description, url, start, end, location, arrive_by."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "fields": {"type": "object"},
                    },
                    "required": ["id", "fields"],
                },
            ),
            Tool(
                name="delete_calendar_event",
                description="Delete a calendar event by id.",
                inputSchema={
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            ),
        ]

    @server.call_tool()
    async def _call_tool(
        name: str, arguments: dict[str, Any] | None
    ) -> list[TextContent]:
        arguments = arguments or {}

        if name == "get_state":
            state = await read_state()
            return [_state_to_text(state)]

        if name == "apply_instruction":
            text = str(arguments.get("text", "")).strip()
            if not text:
                return [
                    TextContent(
                        type="text",
                        text='{"error":"missing required argument: text"}',
                    )
                ]
            from .assistant.chat_assistant import run_chat_assistant

            state_before = await read_state()
            loop_state = await read_state()
            error: Optional[str] = None
            state_after = state_before
            loop_result = None
            try:
                loop_result = await run_chat_assistant(
                    text=text,
                    state=loop_state,
                    client=chat_client,
                    model=chat_model,
                    now=datetime.now(timezone.utc),
                )
                if loop_result.mutating_calls():
                    await write_state(loop_state)
                    state_after = loop_state
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                logger.exception("chat assistant failed")

            if error is not None:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": error}),
                    )
                ]
            mutated = bool(loop_result and loop_result.mutating_calls())
            payload = {
                "applied": mutated,
                "diff": _state_diff(state_before, state_after) if mutated else None,
                "assistant": {
                    "stopped_reason": loop_result.stopped_reason,
                    "summary": loop_result.final_message,
                    "tool_calls": [
                        {
                            "name": c.name,
                            "arguments": c.arguments,
                            "ok": c.error is None,
                            "error": c.error,
                        }
                        for c in loop_result.tool_calls
                    ],
                },
            }
            return [
                TextContent(
                    type="text",
                    text=json.dumps(payload, default=str, indent=2),
                )
            ]

        # ----- CRUD dispatch -----------------------------------------------
        crud_op = _CRUD_DISPATCH.get(name)
        if crud_op is not None:
            state = await read_state()
            try:
                result = crud_op(state, arguments)
            except _crud.CrudError as exc:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": str(exc),
                                "kind": _crud_error_kind(exc),
                            }
                        ),
                    )
                ]
            await write_state(state)
            return [
                TextContent(
                    type="text",
                    text=json.dumps(result, default=str, indent=2),
                )
            ]

        return [
            TextContent(
                type="text", text=json.dumps({"error": f"unknown tool: {name}"})
            )
        ]

    return server


def _crud_error_kind(exc: _crud.CrudError) -> str:
    if isinstance(exc, _crud.NotFound):
        return "not_found"
    if isinstance(exc, _crud.IdConflict):
        return "id_conflict"
    if isinstance(exc, _crud.StillReferenced):
        return "still_referenced"
    if isinstance(exc, _crud.MissingReference):
        return "missing_reference"
    return "bad_field"


def _result_one(model_obj, key: str) -> dict[str, Any]:
    return {"ok": True, key: model_obj.model_dump(mode="json")}


def _result_many(model_objs, key: str) -> dict[str, Any]:
    return {
        "ok": True,
        key: [m.model_dump(mode="json") for m in model_objs],
    }


def _op_add_calendar_event(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
    return _result_one(
        _crud.add_calendar_event(state, args.get("fields") or {}),
        "calendar_event",
    )


def _op_add_calendar_events_bulk(
    state: StateData, args: dict[str, Any]
) -> dict[str, Any]:
    return _result_many(
        _crud.add_calendar_events_bulk(state, args.get("items") or []),
        "calendar_events",
    )


def _op_update_calendar_event(
    state: StateData, args: dict[str, Any]
) -> dict[str, Any]:
    return _result_one(
        _crud.update_calendar_event(state, args["id"], args.get("fields") or {}),
        "calendar_event",
    )


def _op_delete_calendar_event(
    state: StateData, args: dict[str, Any]
) -> dict[str, Any]:
    _crud.delete_calendar_event(state, args["id"])
    return {"ok": True, "event_id": args["id"]}


_CRUD_DISPATCH: dict[str, Callable[[StateData, dict[str, Any]], dict[str, Any]]] = {
    "add_calendar_event": _op_add_calendar_event,
    "add_calendar_events_bulk": _op_add_calendar_events_bulk,
    "update_calendar_event": _op_update_calendar_event,
    "delete_calendar_event": _op_delete_calendar_event,
}
