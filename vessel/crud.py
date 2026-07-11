"""Pure CRUD operations on a StateData blob — calendar events only.

Both the HTTP routes (`vessel/pwa/routes.py`) and the MCP tool layer
(`vessel/mcp_server.py`) call into here. No LLM, no DB, no I/O —
every function takes a `StateData`, mutates it in place (or raises),
and returns the affected record so callers can echo it back.

Validation contract:
- IDs must be unique within the calendar list. `add_calendar_event`
  raises `IdConflict` if the caller passes an id already in use.
- IDs auto-generate from a title slug + date suffix if omitted.
- Updates take a partial dict (`fields`); only present keys are set.
  Unknown keys raise `BadField`.
- Every mutation re-runs Pydantic validation by reconstructing the
  model from the updated dict — no skirting required-field rules.

This module is intentionally synchronous. Persistence happens in the
caller, not here.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Iterable

from .models import StateData
from .models.state import CalendarEvent


class CrudError(Exception):
    """Base class for any expected user-facing error from this module."""


class NotFound(CrudError):
    pass


class IdConflict(CrudError):
    pass


class MissingReference(CrudError):
    pass


class BadField(CrudError):
    pass


class StillReferenced(CrudError):
    """Kept for API compat; not raised by calendar-only ops."""


def _slugify(text: str) -> str:
    """Lowercase + underscore-separated slug, stripped of punctuation.
    Empty input returns 'item' so we always produce a usable id stub."""
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return s or "item"


def _today_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _ensure_no_id_conflict(existing_ids: set[str], new_id: str, kind: str) -> None:
    if new_id in existing_ids:
        raise IdConflict(f"{kind} id {new_id!r} already exists")


def _calendar_ids(state: StateData) -> set[str]:
    return {e.id for e in state.calendar}


# ---------------------------------------------------------------------------
# Calendar events
# ---------------------------------------------------------------------------


def add_calendar_event(
    state: StateData, fields: dict[str, Any]
) -> CalendarEvent:
    fields = dict(fields or {})
    for required in ("title", "start", "end"):
        if required not in fields:
            raise BadField(f"calendar event requires `{required}`")
    fields.setdefault("description", "")

    if "id" not in fields:
        start = fields["start"]
        if isinstance(start, str):
            suffix = start[:10].replace("-", "")
        elif isinstance(start, datetime):
            suffix = start.strftime("%Y%m%d")
        else:
            suffix = _today_suffix()
        base_id = f"cal_{_slugify(fields['title'])}_{suffix}"
        existing = _calendar_ids(state)
        new_id = base_id
        n = 2
        while new_id in existing:
            new_id = f"{base_id}_{n}"
            n += 1
        fields["id"] = new_id
    _ensure_no_id_conflict(_calendar_ids(state), fields["id"], "calendar event")

    try:
        event = CalendarEvent.model_validate(fields)
    except Exception as exc:  # noqa: BLE001
        raise BadField(str(exc)) from exc
    state.calendar.append(event)
    return event


def add_calendar_events_bulk(
    state: StateData, items: Iterable[dict[str, Any]]
) -> list[CalendarEvent]:
    return [add_calendar_event(state, item) for item in items]


def update_calendar_event(
    state: StateData, event_id: str, fields: dict[str, Any]
) -> CalendarEvent:
    ev = next((e for e in state.calendar if e.id == event_id), None)
    if ev is None:
        raise NotFound(f"calendar event {event_id!r} not found")
    if "id" in fields and fields["id"] != event_id:
        raise BadField("changing `id` is not supported via update")
    merged = ev.model_dump()
    merged.update(fields or {})
    try:
        new_ev = CalendarEvent.model_validate(merged)
    except Exception as exc:  # noqa: BLE001
        raise BadField(str(exc)) from exc
    idx = state.calendar.index(ev)
    state.calendar[idx] = new_ev
    return new_ev


def delete_calendar_event(state: StateData, event_id: str) -> None:
    if not any(e.id == event_id for e in state.calendar):
        raise NotFound(f"calendar event {event_id!r} not found")
    state.calendar = [e for e in state.calendar if e.id != event_id]
