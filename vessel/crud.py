"""Pure CRUD operations on a StateData blob.

Both the HTTP routes (`vessel/pwa/routes.py`) and the MCP tool layer
(`vessel/mcp_server.py`) call into here so adding/updating/deleting an
entity has exactly one implementation. No LLM, no DB, no I/O — every
function takes a `StateData`, mutates it in place (or raises), and
returns the affected record so callers can echo it back.

Validation contract:
- IDs must be unique within their entity list. `add_*` raises
  `IdConflict` if the caller passes an id already in use.
- IDs auto-generate from a title slug + date suffix if omitted.
- Foreign keys (task.project_id, calendar.project_id) must point at an
  existing project. Raises `MissingReference` otherwise.
- `delete_project` refuses if any open task or calendar entry still
  references the project — caller decides whether to delete the
  dependents first or refuse.
- Updates take a partial dict (`fields`); only present keys are set.
  Unknown keys raise `BadField`.
- Every mutation re-runs Pydantic validation by reconstructing the
  model from the updated dict — no skirting required-field rules.

This module is intentionally synchronous. Persistence happens in the
caller, not here, so a route can compose multiple ops in one
`state_manager.write` if needed.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, Iterable

from .models import StateData
from .models.state import CalendarEvent, Project, RoutineSlot, Task


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
    """Raised when delete would orphan tasks/calendar entries."""


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


def _project_ids(state: StateData) -> set[str]:
    return {p.id for p in state.projects}


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


_PROJECT_DEFAULTS = {
    "status": "active",
    "tracked": True,
    "cadence": "event_driven",
    "importance": "medium",
}


def add_project(state: StateData, fields: dict[str, Any]) -> Project:
    fields = dict(fields or {})
    if "name" not in fields or not str(fields["name"]).strip():
        raise BadField("project requires a non-empty `name`")

    if "id" not in fields:
        fields["id"] = f"p_{_slugify(fields['name'])}"
    fields["id"] = str(fields["id"])
    _ensure_no_id_conflict(_project_ids(state), fields["id"], "project")

    for k, v in _PROJECT_DEFAULTS.items():
        fields.setdefault(k, v)
    fields.setdefault("last_touched", datetime.now(timezone.utc))

    try:
        project = Project.model_validate(fields)
    except Exception as exc:  # noqa: BLE001
        raise BadField(str(exc)) from exc
    state.projects.append(project)
    return project


def add_projects_bulk(
    state: StateData, items: Iterable[dict[str, Any]]
) -> list[Project]:
    return [add_project(state, item) for item in items]


def update_project(
    state: StateData, project_id: str, fields: dict[str, Any]
) -> Project:
    proj = next((p for p in state.projects if p.id == project_id), None)
    if proj is None:
        raise NotFound(f"project {project_id!r} not found")
    if "id" in fields and fields["id"] != project_id:
        raise BadField("changing `id` is not supported via update")
    merged = proj.model_dump()
    merged.update(fields or {})
    try:
        new_proj = Project.model_validate(merged)
    except Exception as exc:  # noqa: BLE001
        raise BadField(str(exc)) from exc
    idx = state.projects.index(proj)
    state.projects[idx] = new_proj
    return new_proj


def delete_project(state: StateData, project_id: str) -> None:
    proj = next((p for p in state.projects if p.id == project_id), None)
    if proj is None:
        raise NotFound(f"project {project_id!r} not found")
    open_tasks = [
        t.id for t in state.tasks
        if t.project_id == project_id
        and t.completed_at is None
        and t.skipped_at is None
    ]
    open_events = [
        e.id for e in state.calendar
        if e.project_id == project_id
        and e.completed_at is None
        and e.skipped_at is None
    ]
    if open_tasks or open_events:
        raise StillReferenced(
            f"project {project_id!r} still has {len(open_tasks)} open "
            f"task(s) and {len(open_events)} open calendar entry/-ies; "
            "delete or reassign them first"
        )
    state.projects = [p for p in state.projects if p.id != project_id]
    state.priority_ranking = [
        pid for pid in state.priority_ranking if pid != project_id
    ]


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


_TASK_DEFAULTS = {
    # `time_window` is intentionally omitted: it is a computed field on
    # Task (derived from `start_after`). Setting a default would let it
    # be stored, which is exactly the redundancy CLAUDE.md forbids.
    "tier": "flex",
    "estimated_minutes": 30,
    "recurrence": "none",
}


def _default_project_id(state: StateData) -> str:
    """Pick the most-recently-touched project as a fallback. If no
    project exists, raise — the caller must add one first."""
    if not state.projects:
        raise MissingReference(
            "no projects exist; add a project before adding tasks"
        )
    return max(state.projects, key=lambda p: p.last_touched).id


def add_task(state: StateData, fields: dict[str, Any]) -> Task:
    fields = dict(fields or {})
    if "title" not in fields or not str(fields["title"]).strip():
        raise BadField("task requires a non-empty `title`")

    fields.setdefault("project_id", _default_project_id(state))
    if fields["project_id"] not in _project_ids(state):
        raise MissingReference(
            f"task references unknown project_id={fields['project_id']!r}"
        )

    fields.setdefault("due_date", date.today())
    fields.setdefault("created_at", datetime.now(timezone.utc))
    for k, v in _TASK_DEFAULTS.items():
        fields.setdefault(k, v)

    if "id" not in fields:
        suffix = (
            fields["due_date"].strftime("%Y%m%d")
            if isinstance(fields["due_date"], date)
            else str(fields["due_date"]).replace("-", "")
        )
        base_id = f"task_{_slugify(fields['title'])}_{suffix}"
        existing = {t.id for t in state.tasks}
        new_id = base_id
        n = 2
        while new_id in existing:
            new_id = f"{base_id}_{n}"
            n += 1
        fields["id"] = new_id
    _ensure_no_id_conflict({t.id for t in state.tasks}, fields["id"], "task")

    try:
        task = Task.model_validate(fields)
    except Exception as exc:  # noqa: BLE001
        raise BadField(str(exc)) from exc
    state.tasks.append(task)
    return task


def add_tasks_bulk(
    state: StateData, items: Iterable[dict[str, Any]]
) -> list[Task]:
    return [add_task(state, item) for item in items]


def update_task(
    state: StateData, task_id: str, fields: dict[str, Any]
) -> Task:
    task = next((t for t in state.tasks if t.id == task_id), None)
    if task is None:
        raise NotFound(f"task {task_id!r} not found")
    if "id" in fields and fields["id"] != task_id:
        raise BadField("changing `id` is not supported via update")
    if "project_id" in fields and fields["project_id"] not in _project_ids(state):
        raise MissingReference(
            f"task references unknown project_id={fields['project_id']!r}"
        )
    merged = task.model_dump()
    merged.update(fields or {})
    try:
        new_task = Task.model_validate(merged)
    except Exception as exc:  # noqa: BLE001
        raise BadField(str(exc)) from exc
    idx = state.tasks.index(task)
    state.tasks[idx] = new_task
    return new_task


def delete_task(state: StateData, task_id: str) -> None:
    if not any(t.id == task_id for t in state.tasks):
        raise NotFound(f"task {task_id!r} not found")
    state.tasks = [t for t in state.tasks if t.id != task_id]


# ---------------------------------------------------------------------------
# Calendar events
# ---------------------------------------------------------------------------


def _calendar_ids(state: StateData) -> set[str]:
    return {e.id for e in state.calendar}


def add_calendar_event(
    state: StateData, fields: dict[str, Any]
) -> CalendarEvent:
    fields = dict(fields or {})
    for required in ("title", "start", "end"):
        if required not in fields:
            raise BadField(f"calendar event requires `{required}`")
    fields.setdefault("description", "")
    fields.setdefault("project_id", _default_project_id(state))
    if fields["project_id"] not in _project_ids(state):
        raise MissingReference(
            f"calendar event references unknown project_id="
            f"{fields['project_id']!r}"
        )

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
    if "project_id" in fields and fields["project_id"] not in _project_ids(state):
        raise MissingReference(
            f"calendar event references unknown project_id={fields['project_id']!r}"
        )
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


# ---------------------------------------------------------------------------
# Routines
# ---------------------------------------------------------------------------


def _routine_ids(state: StateData) -> set[str]:
    return {r.id for r in state.routines}


def add_routine(state: StateData, fields: dict[str, Any]) -> RoutineSlot:
    fields = dict(fields or {})
    for required in ("label", "start_time", "duration_minutes"):
        if required not in fields:
            raise BadField(f"routine requires `{required}`")
    fields.setdefault("days", [])
    fields.setdefault("kind", "fixed")
    fields.setdefault("source", "user")
    fields.setdefault("confidence", 1.0)

    if "id" not in fields:
        base_id = f"routine_{_slugify(fields['label'])}"
        existing = _routine_ids(state)
        new_id = base_id
        n = 2
        while new_id in existing:
            new_id = f"{base_id}_{n}"
            n += 1
        fields["id"] = new_id
    _ensure_no_id_conflict(_routine_ids(state), fields["id"], "routine")

    try:
        routine = RoutineSlot.model_validate(fields)
    except Exception as exc:  # noqa: BLE001
        raise BadField(str(exc)) from exc
    state.routines.append(routine)
    return routine


def add_routines_bulk(
    state: StateData, items: Iterable[dict[str, Any]]
) -> list[RoutineSlot]:
    return [add_routine(state, item) for item in items]


def update_routine(
    state: StateData, routine_id: str, fields: dict[str, Any]
) -> RoutineSlot:
    routine = next((r for r in state.routines if r.id == routine_id), None)
    if routine is None:
        raise NotFound(f"routine {routine_id!r} not found")
    if "id" in fields and fields["id"] != routine_id:
        raise BadField("changing `id` is not supported via update")
    merged = routine.model_dump()
    merged.update(fields or {})
    try:
        new_r = RoutineSlot.model_validate(merged)
    except Exception as exc:  # noqa: BLE001
        raise BadField(str(exc)) from exc
    idx = state.routines.index(routine)
    state.routines[idx] = new_r
    return new_r


def delete_routine(state: StateData, routine_id: str) -> None:
    if not any(r.id == routine_id for r in state.routines):
        raise NotFound(f"routine {routine_id!r} not found")
    state.routines = [r for r in state.routines if r.id != routine_id]
