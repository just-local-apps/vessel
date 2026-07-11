import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..auth import require_user_id
from ..config import get_settings
from ..db import get_pool
from ..models import StateData
from ..scheduler import state_manager

logger = logging.getLogger(__name__)


# Test injection: swap the OpenAI client `/api/chat` uses with a fake.
_chat_client_for_test: Optional[Any] = None


def _set_chat_client_for_test(client: Optional[Any]) -> None:
    """Test hook — replaces the OpenAI client `/api/chat` would build
    from settings with a caller-supplied fake. Pass `None` to clear."""
    global _chat_client_for_test
    _chat_client_for_test = client


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


router = APIRouter()

STATIC_DIR = Path(__file__).parent / "static"


def _now_local() -> datetime:
    return datetime.now(ZoneInfo(get_settings().timezone))


def _client_now(client_now_header: Optional[str] = None) -> datetime:
    """Return the user-perceived "now" — preferring the client's
    timestamp if it was supplied, falling back to the server's local
    time otherwise.

    The PWA sends `X-Vessel-Client-Now: <ISO8601>` on every API call
    so the server (and any LLM the server invokes) reasons about the
    user's wall clock, not Fly's."""
    if not client_now_header:
        return _now_local()
    try:
        iso = client_now_header.strip()
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        logger.warning("ignoring malformed X-Vessel-Client-Now=%r", client_now_header)
        return _now_local()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(get_settings().timezone))
    return parsed


def _with_tz(dt: datetime, tz_now: datetime) -> datetime:
    """Promote a naive datetime to `tz_now`'s timezone."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=tz_now.tzinfo)


@router.get("/api/state")
async def read_state(
    user_id: str = Depends(require_user_id),
    x_vessel_client_now: Optional[str] = Header(default=None),
) -> dict:
    pool = await get_pool()
    state = await state_manager.read(pool, user_id)
    now = _client_now(x_vessel_client_now)
    return {
        "state": state.model_dump(mode="json"),
        "now": now.isoformat(),
    }


@router.get("/api/now")
async def now_view(
    user_id: str = Depends(require_user_id),
    x_vessel_client_now: Optional[str] = Header(default=None),
) -> dict:
    """Return the event currently in progress or the next upcoming one.

    Response shape:
      {type: "event", event: {...}}   — if an event is found
      {type: "empty"}                 — if no events are upcoming
    """
    pool = await get_pool()
    state = await state_manager.read(pool, user_id)
    now = _client_now(x_vessel_client_now)

    # Check for an in-progress event first.
    for ev in state.calendar:
        if ev.completed_at is not None or ev.skipped_at is not None:
            continue
        start = _with_tz(ev.start, now)
        end = _with_tz(ev.end, now)
        if start <= now < end:
            return {"type": "event", "event": ev.model_dump(mode="json")}

    # Otherwise find the next upcoming event.
    upcoming = [
        ev for ev in state.calendar
        if ev.completed_at is None
        and ev.skipped_at is None
        and _with_tz(ev.start, now) > now
    ]
    if upcoming:
        nxt = min(upcoming, key=lambda e: _with_tz(e.start, now))
        return {"type": "event", "event": nxt.model_dump(mode="json")}

    return {"type": "empty"}


class MoveEventBody(BaseModel):
    minutes: int


@router.post("/api/events/{event_id}/move")
async def move_event(
    event_id: str,
    body: MoveEventBody,
    user_id: str = Depends(require_user_id),
) -> dict:
    """Push a calendar event's start/end forward (or backward, with a
    negative `minutes`) by the given offset."""
    pool = await get_pool()
    state = await state_manager.read(pool, user_id)
    found = next((e for e in state.calendar if e.id == event_id), None)
    if found is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="event not found"
        )
    delta = timedelta(minutes=int(body.minutes))
    found.start = found.start + delta
    found.end = found.end + delta
    await state_manager.write(pool, user_id, state)
    return {
        "ok": True,
        "event_id": event_id,
        "start": found.start.isoformat(),
        "end": found.end.isoformat(),
    }


@router.post("/api/events/{event_id}/uncomplete")
async def uncomplete_event(
    event_id: str, user_id: str = Depends(require_user_id)
) -> dict:
    pool = await get_pool()
    state = await state_manager.read(pool, user_id)
    found = next((e for e in state.calendar if e.id == event_id), None)
    if found is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="event not found"
        )
    found.completed_at = None
    await state_manager.write(pool, user_id, state)
    return {"ok": True, "event_id": event_id}


@router.post("/api/events/{event_id}/unskip")
async def unskip_event(
    event_id: str, user_id: str = Depends(require_user_id)
) -> dict:
    pool = await get_pool()
    state = await state_manager.read(pool, user_id)
    found = next((e for e in state.calendar if e.id == event_id), None)
    if found is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="event not found"
        )
    found.skipped_at = None
    found.skip_reason = None
    await state_manager.write(pool, user_id, state)
    return {"ok": True, "event_id": event_id}


class SkipBody(BaseModel):
    reason: str


@router.post("/api/events/{event_id}/skip")
async def skip_event(
    event_id: str,
    body: SkipBody,
    user_id: str = Depends(require_user_id),
    x_vessel_client_now: Optional[str] = Header(default=None),
) -> dict:
    """User declined a calendar event. Two stages:

    1. Mark the event skipped (skipped_at + skip_reason set, stays in state
       so unskip can recover it).
    2. If an LLM is configured, run the skip-assistant tool-loop on the
       reason so the user's implied intent (reschedule, clear the day, etc.)
       actually happens via CRUD tool calls.
    """
    reason = (body.reason or "").strip()
    if not reason:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="reason is required when skipping an event",
        )
    pool = await get_pool()
    state = await state_manager.read(pool, user_id)
    now = _client_now(x_vessel_client_now)
    found = next((e for e in state.calendar if e.id == event_id), None)
    if found is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="event not found"
        )

    found.skipped_at = now
    found.skip_reason = reason
    await state_manager.write(pool, user_id, state)

    # ---- Stage 2: skip-reason assistant ----
    assistant_payload: dict[str, Any] = {"invoked": False}
    try:
        from ..assistant.skip_assistant import run_skip_assistant
        from openai import AsyncOpenAI as _AsyncOpenAI

        settings = get_settings()
        if settings.groq_api_key:
            client = _AsyncOpenAI(
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            loop_state = await state_manager.read(pool, user_id)
            result = await run_skip_assistant(
                reason=reason,
                skipped_task=found,
                state=loop_state,
                client=client,
                model=settings.groq_model,
                now=now,
            )
            mutated = bool(result.mutating_calls())
            if mutated:
                await state_manager.write(pool, user_id, loop_state)
            assistant_payload = {
                "invoked": True,
                "stopped_reason": result.stopped_reason,
                "summary": result.final_message,
                "tool_calls": [
                    {
                        "name": c.name,
                        "arguments": c.arguments,
                        "ok": c.error is None,
                        "error": c.error,
                    }
                    for c in result.tool_calls
                ],
                "mutated": mutated,
            }
    except Exception as exc:  # noqa: BLE001
        logger.exception("event skip assistant failed")
        assistant_payload = {"invoked": True, "error": str(exc)}

    return {
        "ok": True,
        "event_id": event_id,
        "skipped_at": now.isoformat(),
        "assistant": assistant_payload,
    }


@router.post("/api/events/{event_id}/complete")
async def complete_event(
    event_id: str, user_id: str = Depends(require_user_id)
) -> dict:
    """Mark a calendar event as done. Pure state mutation, no LLM."""
    pool = await get_pool()
    state = await state_manager.read(pool, user_id)
    now = _now_local()
    found = next((e for e in state.calendar if e.id == event_id), None)
    if found is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="event not found"
        )
    found.completed_at = now
    await state_manager.write(pool, user_id, state)
    return {"ok": True, "event_id": event_id, "completed_at": now.isoformat()}


class ChatBody(BaseModel):
    text: str


@router.post("/api/chat")
async def chat(
    body: ChatBody,
    user_id: str = Depends(require_user_id),
    x_vessel_client_now: Optional[str] = Header(default=None),
) -> dict:
    """Run the chat tool-use loop. The LLM sees (now, calendar, instruction)
    and emits calendar CRUD tool calls; the route persists state once the
    loop returns.

    Response shape:
      {
        applied: bool,
        diff: {...} | null,
        assistant: {stopped_reason, summary, tool_calls: [{name, arguments, ok, error}]}
      }
    """
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="text is required",
        )
    pool = await get_pool()
    state_before = await state_manager.read(pool, user_id)

    from ..assistant.chat_assistant import run_chat_assistant
    from openai import AsyncOpenAI as _AsyncOpenAI

    settings = get_settings()
    if _chat_client_for_test is not None:
        client = _chat_client_for_test
    else:
        if not settings.groq_api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM not configured: GROQ_API_KEY missing",
            )
        client = _AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    now = _client_now(x_vessel_client_now)

    loop_state = await state_manager.read(pool, user_id)
    try:
        result = await run_chat_assistant(
            text=text,
            state=loop_state,
            client=client,
            model=settings.groq_model,
            now=now,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat assistant failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"agent error: {exc}",
        )

    mutated = bool(result.mutating_calls())
    if mutated:
        await state_manager.write(pool, user_id, loop_state)

    return {
        "applied": mutated,
        "diff": _state_diff(state_before, loop_state) if mutated else None,
        "assistant": {
            "stopped_reason": result.stopped_reason,
            "summary": result.final_message,
            "tool_calls": [
                {
                    "name": c.name,
                    "arguments": c.arguments,
                    "ok": c.error is None,
                    "error": c.error,
                }
                for c in result.tool_calls
            ],
        },
    }


# ---------------------------------------------------------------------------
# CRUD endpoints — dumb create / update / delete over StateData.
# ---------------------------------------------------------------------------


from .. import crud as _crud  # noqa: E402


def _crud_to_http(exc: _crud.CrudError) -> HTTPException:
    """Map CrudError subclasses to HTTP status codes."""
    if isinstance(exc, _crud.NotFound):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, _crud.IdConflict):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, _crud.StillReferenced):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


async def _read_mut_write(user_id: str):
    """Helper: read state, return (state, write-callback)."""
    pool = await get_pool()
    state = await state_manager.read(pool, user_id)

    async def _write(new_state: StateData) -> None:
        await state_manager.write(pool, user_id, new_state)

    return state, _write


# ----- Calendar events -------------------------------------------------------


class CalendarIn(BaseModel):
    model_config = {"extra": "allow"}


@router.post("/api/calendar")
async def crud_add_calendar(
    body: CalendarIn, user_id: str = Depends(require_user_id)
) -> dict:
    state, write = await _read_mut_write(user_id)
    try:
        ev = _crud.add_calendar_event(state, body.model_dump())
    except _crud.CrudError as exc:
        raise _crud_to_http(exc)
    await write(state)
    return {"ok": True, "calendar_event": ev.model_dump(mode="json")}


@router.post("/api/calendar/bulk")
async def crud_add_calendar_bulk(
    body: list[dict], user_id: str = Depends(require_user_id)
) -> dict:
    state, write = await _read_mut_write(user_id)
    try:
        added = _crud.add_calendar_events_bulk(state, body)
    except _crud.CrudError as exc:
        raise _crud_to_http(exc)
    await write(state)
    return {
        "ok": True,
        "calendar_events": [e.model_dump(mode="json") for e in added],
    }


@router.patch("/api/calendar/{event_id}")
async def crud_update_calendar(
    event_id: str,
    body: CalendarIn,
    user_id: str = Depends(require_user_id),
) -> dict:
    state, write = await _read_mut_write(user_id)
    try:
        ev = _crud.update_calendar_event(state, event_id, body.model_dump())
    except _crud.CrudError as exc:
        raise _crud_to_http(exc)
    await write(state)
    return {"ok": True, "calendar_event": ev.model_dump(mode="json")}


@router.delete("/api/calendar/{event_id}")
async def crud_delete_calendar(
    event_id: str, user_id: str = Depends(require_user_id)
) -> dict:
    state, write = await _read_mut_write(user_id)
    try:
        _crud.delete_calendar_event(state, event_id)
    except _crud.CrudError as exc:
        raise _crud_to_http(exc)
    await write(state)
    return {"ok": True, "event_id": event_id}


def mount_static(app) -> None:
    app.mount(
        "/pwa",
        StaticFiles(directory=str(STATIC_DIR), html=True),
        name="pwa-static",
    )
