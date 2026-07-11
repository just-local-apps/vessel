"""Generic OpenAI-style tool-use loop driving vessel's calendar CRUD layer.

Sits between an LLM client (the OpenAI Python SDK pointed at Groq) and
`vessel.crud`. Used by:
  - skip-with-reason (`run_skip_assistant`)
  - the chat box (`run_chat_assistant`)

The loop is LLM-client-agnostic: it asks for
`await client.chat.completions.create(...)` shaped responses but accepts
any object with the same shape, so unit tests can pass a fake client
without monkeypatching SDK internals.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .. import crud as _crud
from ..models import StateData

logger = logging.getLogger(__name__)


# Hard cap so a confused model can't burn the whole budget.
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
# Tool dispatch
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
    return {"ok": True, "state": state.model_dump(mode="json")}


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


DISPATCH: dict[str, Callable[[StateData, dict[str, Any]], dict[str, Any]]] = {
    "get_state": _op_get_state,
    "add_calendar_event": _op_add_calendar_event,
    "add_calendar_events_bulk": _op_add_calendar_events_bulk,
    "update_calendar_event": _op_update_calendar_event,
    "delete_calendar_event": _op_delete_calendar_event,
}


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


ChatCompleteFn = Callable[..., Awaitable[Any]]


def _execute_tool_call(state: StateData, name: str, raw_args: str) -> ToolCall:
    """Dispatch one model-emitted tool call."""
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
    except Exception as exc:  # noqa: BLE001
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
            result.final_message = msg.content or ""
            result.stopped_reason = "completed"
            break

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
