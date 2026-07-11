"""Generic OpenAI-style tool-use loop driving vessel's CRUD layer.

Sits between an LLM client (the OpenAI Python SDK pointed at Groq) and
`vessel.crud` / `vessel.mcp_server`'s CRUD dispatchers. Used by:
  - skip-with-reason (`run_skip_assistant`)
  - the chat box (`run_chat_assistant`, coming next)

Both surfaces want exactly the same behavior: send the model a system
prompt, a user instruction, the current state, and the CRUD tool
schema; loop on every `tool_calls` reply by executing each call against
state and feeding the result back; cap at N iterations; surface every
tool call and its result so the caller can show them in the UI.

The loop is LLM-client-agnostic: it asks for `await client.chat.completions.create(...)`
shaped responses but accepts any object with the same shape, so unit
tests can pass a fake client without monkeypatching SDK internals.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .. import crud as _crud
from ..models import StateData

logger = logging.getLogger(__name__)


# Hard cap so a confused model can't burn the whole budget. 6 calls is
# enough for "delete all wash dishes" (1 get_state + 7 deletes is over,
# but most of the time the model only needs the deletes — it can
# decide from the state snapshot we already provide).
MAX_TOOL_CALLS = 8


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def is_mutation(self) -> bool:
        # get_state is read-only; everything else mutates.
        return self.name != "get_state"


@dataclass
class LoopResult:
    """What `tool_loop()` returns to its caller."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    final_message: str = ""
    stopped_reason: str = "completed"  # 'completed' | 'cap_hit' | 'error'
    error: Optional[str] = None

    def mutating_calls(self) -> list[ToolCall]:
        return [c for c in self.tool_calls if c.is_mutation() and c.error is None]


# ---------------------------------------------------------------------------
# Tool dispatch — the same dict mcp_server uses, replicated here so we
# don't import the MCP module just for this. Each entry is
# `(state, args) -> result_dict` (sync, mutates state in place).
# ---------------------------------------------------------------------------


def _result_one(model_obj, key: str) -> dict[str, Any]:
    return {"ok": True, key: model_obj.model_dump(mode="json")}


def _result_many(model_objs, key: str) -> dict[str, Any]:
    return {"ok": True, key: [m.model_dump(mode="json") for m in model_objs]}


def _error_kind(exc: _crud.CrudError) -> str:
    if isinstance(exc, _crud.NotFound):
        return "not_found"
    if isinstance(exc, _crud.IdConflict):
        return "id_conflict"
    if isinstance(exc, _crud.StillReferenced):
        return "still_referenced"
    if isinstance(exc, _crud.MissingReference):
        return "missing_reference"
    return "bad_field"


def _op_get_state(state: StateData, _args: dict[str, Any]) -> dict[str, Any]:
    # Returned to the model so it can re-read state mid-loop if it needs to.
    return {"ok": True, "state": state.model_dump(mode="json")}


def _op_add_project(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
    return _result_one(_crud.add_project(state, args.get("fields") or {}), "project")


def _op_add_projects_bulk(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
    return _result_many(
        _crud.add_projects_bulk(state, args.get("items") or []), "projects"
    )


def _op_update_project(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
    return _result_one(
        _crud.update_project(state, args["id"], args.get("fields") or {}), "project"
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
        _crud.add_calendar_event(state, args.get("fields") or {}), "calendar_event"
    )


def _op_add_calendar_events_bulk(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
    return _result_many(
        _crud.add_calendar_events_bulk(state, args.get("items") or []),
        "calendar_events",
    )


def _op_update_calendar_event(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
    return _result_one(
        _crud.update_calendar_event(state, args["id"], args.get("fields") or {}),
        "calendar_event",
    )


def _op_delete_calendar_event(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
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
        _crud.update_routine(state, args["id"], args.get("fields") or {}), "routine"
    )


def _op_delete_routine(state: StateData, args: dict[str, Any]) -> dict[str, Any]:
    _crud.delete_routine(state, args["id"])
    return {"ok": True, "routine_id": args["id"]}


DISPATCH: dict[str, Callable[[StateData, dict[str, Any]], dict[str, Any]]] = {
    "get_state": _op_get_state,
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


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


# Type for a chat-completions-creating coroutine. Anything that returns
# an object with `.choices[0].message.{content, tool_calls}` works —
# real OpenAI SDK and the test fakes both fit.
ChatCompleteFn = Callable[..., Awaitable[Any]]


def _execute_tool_call(state: StateData, name: str, raw_args: str) -> ToolCall:
    """Dispatch one model-emitted tool call. Captures the args, the
    result, and any CRUD error in a `ToolCall` record."""
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError as exc:
        return ToolCall(name=name, arguments={}, error=f"invalid JSON arguments: {exc}")
    op = DISPATCH.get(name)
    if op is None:
        return ToolCall(
            name=name, arguments=args, error=f"unknown tool: {name!r}"
        )
    try:
        result = op(state, args)
        return ToolCall(name=name, arguments=args, result=result)
    except _crud.CrudError as exc:
        return ToolCall(
            name=name,
            arguments=args,
            error=str(exc),
            result={"error": str(exc), "kind": _error_kind(exc)},
        )
    except Exception as exc:  # noqa: BLE001 — catch-all so the loop continues
        logger.exception("tool %r raised", name)
        return ToolCall(name=name, arguments=args, error=f"tool raised: {exc}")


async def tool_loop(
    *,
    chat_complete: ChatCompleteFn,
    model: str,
    system_prompt: str,
    user_message: str,
    state: StateData,
    tools: list[dict[str, Any]],
    max_tool_calls: int = MAX_TOOL_CALLS,
) -> LoopResult:
    """Drive an LLM through repeated tool calls until it stops emitting
    them (or the cap fires).

    State is mutated in place — caller persists once the loop returns.
    The LoopResult lists every tool call attempted (with results and
    errors) so the chat UI can show what actually happened.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    result = LoopResult()
    while True:
        if len(result.tool_calls) >= max_tool_calls:
            result.stopped_reason = "cap_hit"
            break

        try:
            resp = await chat_complete(
                model=model,
                messages=messages,
                tools=tools,
                # Let the model decide whether to call a tool or finish.
                tool_choice="auto",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM call failed")
            result.stopped_reason = "error"
            result.error = str(exc)
            break

        choice = resp.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            # Final message — model is done.
            result.final_message = msg.content or ""
            result.stopped_reason = "completed"
            break

        # Append the assistant turn (with tool_calls) so the model sees
        # its own previous output in the next round.
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        # Execute each tool call against state and append the tool
        # result message (one per call) for the next round.
        for tc in tool_calls:
            call = _execute_tool_call(
                state, tc.function.name, tc.function.arguments
            )
            result.tool_calls.append(call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps(
                        call.result if call.error is None
                        else {"error": call.error},
                        default=str,
                    ),
                }
            )

    return result
