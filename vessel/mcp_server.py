"""Vessel's MCP server — direct LLM ↔ state interface.

Exposes a small set of tools that a Claude (or any MCP client) can call:

- `get_state` — read the current StateData
- `apply_instruction(text)` — natural-language → chat tool-loop → new state

`apply_instruction` runs the SAME tool-loop the PWA chat box uses, so
there is one LLM path across chat / skip / MCP. The model receives
(now, state, instruction) and emits CRUD tool calls; this module
persists the mutated state and returns a diff.
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
from .scheduler.priority import compute_priority_ranking

logger = logging.getLogger(__name__)


ReadStateFn = Callable[[], Awaitable[StateData]]
WriteStateFn = Callable[[StateData], Awaitable[None]]


def _state_to_text(state: StateData) -> TextContent:
    return TextContent(type="text", text=state.model_dump_json(indent=2))


def _diff_collection(before: list, after: list) -> dict[str, list]:
    """Diff two id-keyed pydantic-model lists. Returns added/removed/changed.

    `changed` entries are {"id", "before", "after"} with full objects so the
    caller can see exactly which fields moved.
    """
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
        "projects": _diff_collection(before.projects, after.projects),
        "tasks": _diff_collection(before.tasks, after.tasks),
        "calendar": _diff_collection(before.calendar, after.calendar),
    }
    if before.priority_ranking != after.priority_ranking:
        diff["priority_ranking"] = {
            "before": before.priority_ranking,
            "after": after.priority_ranking,
        }
    summary = {
        kind: {
            "added": len(diff[kind]["added"]),
            "removed": len(diff[kind]["removed"]),
            "changed": len(diff[kind]["changed"]),
        }
        for kind in ("projects", "tasks", "calendar")
    }
    summary["priority_ranking_changed"] = (
        before.priority_ranking != after.priority_ranking
    )
    diff["summary"] = summary
    return diff


def build_mcp_server(
    *,
    read_state: ReadStateFn,
    write_state: WriteStateFn,
    chat_client: Any,
    chat_model: str,
) -> Server:
    """Construct an MCP Server bound to the supplied dependencies.

    `chat_client` is anything shaped like an AsyncOpenAI (i.e. with
    `.chat.completions.create`) — the chat tool-loop hands it tool
    calls and streams back the model's responses. `chat_model` is the
    model id passed through to that client (e.g. Groq's gpt-oss-120b).
    """
    server: Server = Server("vessel")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name="get_state",
                description=(
                    "Read Vessel's current state — projects, tasks, "
                    "calendar entries, and priority_ranking."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="apply_instruction",
                description=(
                    "Apply a natural-language instruction to Vessel's "
                    "state. Runs the chat tool-loop assistant — the LLM "
                    "infers from (now, state, instruction) and calls "
                    "CRUD tools to mutate state. Returns: "
                    "{applied: bool, diff: {...} | null, "
                    "assistant: {summary, tool_calls, stopped_reason}}. "
                    "`applied` is True when any mutating tool call "
                    "landed; `summary` is the assistant's one-line "
                    "confirmation. Use this to add projects, complete "
                    "tasks, push due dates, schedule calendar entries, "
                    "etc. The agent does NOT ask clarifying questions "
                    "— it infers from context or replies with text."
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
            # ----- CRUD: projects --------------------------------------
            Tool(
                name="add_project",
                description=(
                    "Add ONE project. Required: `name`. Optional: `id`, "
                    "`status`, `tracked`, `cadence`, `importance`, `goal`, "
                    "`target_date`, `why`. id auto-generates from name."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"fields": {"type": "object"}},
                    "required": ["fields"],
                },
            ),
            Tool(
                name="add_projects_bulk",
                description="Add many projects in one call. `items` is an array of project dicts.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": ["items"],
                },
            ),
            Tool(
                name="update_project",
                description="Update a subset of a project's fields by id.",
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
                name="delete_project",
                description=(
                    "Delete a project by id. Refuses if any open task or "
                    "calendar entry still references it."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            ),
            # ----- CRUD: tasks -----------------------------------------
            Tool(
                name="add_task",
                description=(
                    "Add ONE task. Required: `title`. Optional: "
                    "`project_id` (defaults to most-recent project), "
                    "`due_date`, `tier`, `estimated_minutes`, `notes`, "
                    "`start_after` (HH:MM clock gate — the displayed "
                    "window is derived from this; do NOT pass "
                    "`time_window` directly), `recurrence` "
                    "(\"none\"|\"daily\"). id auto-generates."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"fields": {"type": "object"}},
                    "required": ["fields"],
                },
            ),
            Tool(
                name="add_tasks_bulk",
                description="Add many tasks in one call. `items` is an array of task dicts.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": ["items"],
                },
            ),
            Tool(
                name="update_task",
                description=(
                    "Update a subset of a task's fields by id. Use this to "
                    "change recurrence, start_after, due_date, tier, etc."
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
                name="delete_task",
                description="Delete a task by id (hard delete from state, no archive).",
                inputSchema={
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            ),
            # ----- CRUD: calendar --------------------------------------
            Tool(
                name="add_calendar_event",
                description=(
                    "Add ONE calendar event. Required: `title`, `start`, "
                    "`end`. Optional: `project_id`, `description`, "
                    "`location`, `phone_number`."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"fields": {"type": "object"}},
                    "required": ["fields"],
                },
            ),
            Tool(
                name="add_calendar_events_bulk",
                description="Add many calendar events in one call.",
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
                description="Update a subset of a calendar event's fields by id.",
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
            # ----- CRUD: routines --------------------------------------
            Tool(
                name="add_routine",
                description=(
                    "Add ONE routine slot. Required: `label`, "
                    "`start_time` (HH:MM), `duration_minutes`. Optional: "
                    "`days`, `kind`, `source`, `confidence`."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"fields": {"type": "object"}},
                    "required": ["fields"],
                },
            ),
            Tool(
                name="add_routines_bulk",
                description="Add many routines in one call.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": ["items"],
                },
            ),
            Tool(
                name="update_routine",
                description="Update a subset of a routine's fields by id.",
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
                name="delete_routine",
                description="Delete a routine by id.",
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
            # Read again so the loop mutates a live copy without
            # racing read_state's deep-copy contract.
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
                    derived = compute_priority_ranking(loop_state)
                    if derived != loop_state.priority_ranking:
                        loop_state = loop_state.model_copy(
                            update={"priority_ranking": derived}
                        )
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

        # ----- CRUD dispatch ------------------------------------
        #
        # Single dispatcher for every CRUD tool. Reads state, runs
        # the named crud op, writes state on success, returns the
        # affected record(s) as JSON. Errors map onto a uniform
        # `{error, kind}` envelope so the LLM caller can branch on
        # the kind ("not_found" / "id_conflict" / "still_referenced"
        # / "missing_reference" / "bad_field").
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


def _op_add_project(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
    return _result_one(_crud.add_project(state, args.get("fields") or {}), "project")


def _op_add_projects_bulk(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
    return _result_many(
        _crud.add_projects_bulk(state, args.get("items") or []), "projects"
    )


def _op_update_project(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
    return _result_one(
        _crud.update_project(state, args["id"], args.get("fields") or {}),
        "project",
    )


def _op_delete_project(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
    _crud.delete_project(state, args["id"])
    return {"ok": True, "project_id": args["id"]}


def _op_add_task(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
    return _result_one(_crud.add_task(state, args.get("fields") or {}), "task")


def _op_add_tasks_bulk(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
    return _result_many(
        _crud.add_tasks_bulk(state, args.get("items") or []), "tasks"
    )


def _op_update_task(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
    return _result_one(
        _crud.update_task(state, args["id"], args.get("fields") or {}), "task"
    )


def _op_delete_task(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
    _crud.delete_task(state, args["id"])
    return {"ok": True, "task_id": args["id"]}


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


def _op_add_routine(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
    return _result_one(_crud.add_routine(state, args.get("fields") or {}), "routine")


def _op_add_routines_bulk(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
    return _result_many(
        _crud.add_routines_bulk(state, args.get("items") or []), "routines"
    )


def _op_update_routine(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
    return _result_one(
        _crud.update_routine(state, args["id"], args.get("fields") or {}),
        "routine",
    )


def _op_delete_routine(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
    _crud.delete_routine(state, args["id"])
    return {"ok": True, "routine_id": args["id"]}


_CRUD_DISPATCH: dict[str, Callable[[StateData, dict[str, Any]], dict[str, Any]]] = {
    "add_project": _op_add_project,
    "add_projects_bulk": _op_add_projects_bulk,
    "update_project": _op_update_project,
    "delete_project": _op_delete_project,
    "add_task": _op_add_task,
    "add_tasks_bulk": _op_add_tasks_bulk,
    "update_task": _op_update_task,
    "delete_task": _op_delete_task,
    "add_calendar_event": _op_add_calendar_event,
    "add_calendar_events_bulk": _op_add_calendar_events_bulk,
    "update_calendar_event": _op_update_calendar_event,
    "delete_calendar_event": _op_delete_calendar_event,
    "add_routine": _op_add_routine,
    "add_routines_bulk": _op_add_routines_bulk,
    "update_routine": _op_update_routine,
    "delete_routine": _op_delete_routine,
}
