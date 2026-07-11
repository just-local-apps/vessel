import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..auth import require_user_id
from ..config import get_settings
from ..db import get_pool
from ..models import StateData, TimeWindow
from ..models.state import bucket_time_window
from ..scheduler import state_manager, task_history
from ..scheduler.priority import compute_priority_ranking

logger = logging.getLogger(__name__)


# Test injection: swap the OpenAI client `/api/chat` uses with a fake.
# The chat surface goes through `run_chat_assistant`, which in turn
# calls `client.chat.completions.create`. Hermetic tests pass a
# scripted client matching that shape; production builds the real
# Groq AsyncOpenAI from settings.
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

router = APIRouter()

STATIC_DIR = Path(__file__).parent / "static"


def current_window(now: datetime) -> TimeWindow:
    # Delegate to the shared bucket helper so a task whose
    # `time_window` (computed) is "evening" lands in the same window
    # this function reports for the user's wall clock at that moment.
    # One bucketing rule, two callers — never drift.
    return bucket_time_window(now.time())


def _now_local() -> datetime:
    return datetime.now(ZoneInfo(get_settings().timezone))


def _client_now(client_now_header: Optional[str] = None) -> datetime:
    """Return the user-perceived "now" — preferring the client's
    timestamp if it was supplied, falling back to the server's local
    time otherwise.

    The PWA sends `X-Vessel-Client-Now: <ISO8601>` on every API call
    so the server (and any LLM the server invokes) reasons about the
    user's wall clock, not Fly's. Without this, an EDT user gets
    "today" answered in LA time and sees yesterday's calendar entries,
    skip-assistants set due_date in the past, etc.

    If the header is missing or unparseable we silently fall back to
    `_now_local()` so curl / tests / Claude Desktop (which won't send
    the header) keep working."""
    if not client_now_header:
        return _now_local()
    try:
        # Accept both "...Z" suffix and "+00:00" form.
        iso = client_now_header.strip()
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        logger.warning("ignoring malformed X-Vessel-Client-Now=%r", client_now_header)
        return _now_local()
    if parsed.tzinfo is None:
        # Naive timestamp from the client → assume server-configured TZ.
        parsed = parsed.replace(tzinfo=ZoneInfo(get_settings().timezone))
    return parsed


# How many days of forward instances a daily-recurring task series keeps
# open at any time. Picked to match the show-all view's 7-day rolling
# window so the user always sees the full week of upcoming repeats.
RECURRENCE_LOOKAHEAD_DAYS = 7


def _expand_daily_recurrence(state: StateData, today_date) -> StateData:
    """For every `recurrence='daily'` task series, make sure an OPEN
    instance exists for each of the next RECURRENCE_LOOKAHEAD_DAYS days.

    A "series" is identified by (project_id, normalized title) — the
    Task model has no series id, and every instance carries
    `recurrence='daily'`, so this keying is what we have. Skipped or
    completed instances do NOT block creation of a new open one for the
    same date (they're closed; user expects the next day to still
    appear). But an existing OPEN instance for that date suppresses
    creation, which prevents drift after the user manually adds a copy.

    Mutates state in place and returns it. Idempotent: running it
    twice produces the same state."""
    from ..models.state import Task as _Task

    series: dict[tuple, "_Task"] = {}
    any_by_key_date: dict[tuple, set] = {}
    for t in state.tasks:
        if t.recurrence != "daily":
            continue
        key = (t.project_id, t.title.strip().lower())
        # Use the most recently due instance as the template (carries
        # the latest start_after / estimated_minutes; time_window is
        # computed from start_after, so it follows automatically).
        prev = series.get(key)
        if prev is None or t.due_date >= prev.due_date:
            series[key] = t
        # Track ANY instance per (series, date) — open OR closed —
        # because completing today's task shouldn't cause the expander
        # to immediately recreate a new "open today" copy on the next
        # request. The window slides forward by skipping dates already
        # represented in some form.
        any_by_key_date.setdefault(key, set()).add(t.due_date)

    for key, template in series.items():
        existing_any = any_by_key_date.get(key, set())
        for offset in range(RECURRENCE_LOOKAHEAD_DAYS):
            target = today_date + timedelta(days=offset)
            if target in existing_any:
                continue
            suffix = target.strftime("%Y%m%d")
            slug = template.title.strip().lower().replace(" ", "_")
            new_id = f"task_{slug}_{suffix}"
            # Collision-avoidance: if some other task already owns this
            # id (rare — different series with the same title slug),
            # tag it with `_r`.
            if any(t.id == new_id for t in state.tasks):
                new_id = f"{new_id}_r"
            state.tasks.append(
                _Task(
                    id=new_id,
                    project_id=template.project_id,
                    title=template.title,
                    notes=template.notes,
                    tier=template.tier,
                    estimated_minutes=template.estimated_minutes,
                    due_date=target,
                    created_at=datetime.now(ZoneInfo(get_settings().timezone)),
                    # `start_after` is the source of truth; `time_window`
                    # on the spawned task is computed from it.
                    start_after=template.start_after,
                    recurrence="daily",
                )
            )
    return state


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
        "current_window": current_window(now).value,
    }


@router.get("/api/tasks/history")
async def read_task_history(
    limit: int = 100,
    user_id: str = Depends(require_user_id),
) -> dict:
    """Read recent closed tasks from the archive table. Useful for a
    history / stats view; not used by the focus card directly."""
    pool = await get_pool()
    rows = await task_history.list_recent(pool, user_id, limit=limit)
    return {"history": rows, "limit": limit}


def _events_for_date(state: StateData, target_date) -> list[dict]:
    """Return calendar events whose start falls on target_date, sorted.
    Skipped or completed events are filtered out — same contract as
    tasks, so a left-swiped event disappears from the day view."""
    out = []
    for e in state.calendar:
        if e.start.date() != target_date:
            continue
        if e.completed_at is not None or e.skipped_at is not None:
            continue
        out.append(e.model_dump(mode="json"))
    out.sort(key=lambda e: e["start"])
    return out


def _with_tz(dt: datetime, tz_now: datetime) -> datetime:
    """Promote a naive datetime to `tz_now`'s timezone. Calendar entries
    occasionally come back without tzinfo from the agent's output —
    treating them as local-time keeps the math sane."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=tz_now.tzinfo)


def _pick_now_card(state: StateData, now: datetime) -> dict:
    """Single-card mode logic — answers 'what should I do next?'

    Order of precedence:
      1. If a calendar event covers `now`, that's the card. All cards
         (events too) are swipeable: right = done, left = move/skip.
      2. Else, if cumulative completed-task minutes since the user's
         last acknowledged break crosses the threshold, return a
         "take a break" card.
      3. Else, compute `free_minutes` until the next calendar event (or
         bedtime, whichever comes first). Pick the highest-priority
         open task whose `estimated_minutes` fits that window.
      4. If no task fits the window, return no card. The UI shows
         nothing — Vessel stays quiet when there's nothing to suggest.

    Ranking when multiple tasks fit:
      must_today first → priority_ranking position → most-pushed first
      → shortest first.
    """
    settings = get_settings()
    today = now.date()
    bedtime = now.replace(
        hour=settings.bedtime_hour, minute=0, second=0, microsecond=0
    )

    active_event = None
    for ev in state.calendar:
        if ev.completed_at is not None or ev.skipped_at is not None:
            continue
        start = _with_tz(ev.start, now)
        end = _with_tz(ev.end, now)
        if start <= now < end:
            active_event = ev
            break

    next_event_start = None
    for ev in state.calendar:
        if ev.completed_at is not None or ev.skipped_at is not None:
            continue
        start = _with_tz(ev.start, now)
        if start > now and start.date() == today:
            if next_event_start is None or start < next_event_start:
                next_event_start = start

    upper = bedtime
    if next_event_start is not None:
        upper = min(upper, next_event_start)
    free_minutes = max(0, int((upper - now).total_seconds() // 60))

    projects_by_id = {p.id: p for p in state.projects}

    def _annotate(card_kind: str, raw: dict, project_id: str) -> dict:
        proj = projects_by_id.get(project_id)
        raw["project_name"] = proj.name if proj else None
        return {"kind": card_kind, "data": raw, "swipeable": True}

    if active_event is not None:
        # During a calendar block: that event is the card. It's still
        # swipeable — right marks done, left opens a move/skip dialog
        # so the user can defer the event by 10/30/60 min or skip it
        # outright.
        return {
            "card": _annotate(
                "event",
                active_event.model_dump(mode="json"),
                active_event.project_id,
            ),
            "free_minutes": 0,
            "in_block": True,
            "now": now.isoformat(),
        }

    # Break logic — once enough cumulative work has happened since the
    # user's last acknowledged break, recommend stepping away. Counted
    # in minutes of `estimated_minutes` for tasks completed since
    # `last_break_acknowledged_at` (or start of today if no ack yet).
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    last_ack = state.last_break_acknowledged_at
    if last_ack is not None:
        last_ack_local = _with_tz(last_ack, now)
    else:
        last_ack_local = day_start
    work_since_break = 0
    for t in state.tasks:
        if t.completed_at is None:
            continue
        completed_local = _with_tz(t.completed_at, now)
        if completed_local < last_ack_local:
            continue
        if completed_local.date() != today:
            continue
        work_since_break += t.estimated_minutes or 30
    if work_since_break >= settings.work_before_break_min:
        return {
            "card": {
                "kind": "break",
                "data": {
                    "id": "break-card",
                    "title": "Take a break",
                    "notes": (
                        f"You've worked {work_since_break} minutes "
                        "since your last break. Step away for a few "
                        "minutes — water, stretch, look out a window."
                    ),
                    "minutes_worked": work_since_break,
                },
                "swipeable": True,
            },
            "free_minutes": free_minutes,
            "in_block": False,
            "now": now.isoformat(),
        }

    project_priority = {pid: i for i, pid in enumerate(state.priority_ranking)}
    candidates = [
        t
        for t in state.tasks
        if t.due_date == today
        and t.completed_at is None
        and t.skipped_at is None
        and (t.estimated_minutes or 30) <= free_minutes
        # Time-of-day gate: a task with `start_after=19:00` should not
        # surface as the focus card until 7pm local. Tasks with no
        # gate (start_after is None) always pass.
        and (t.start_after is None or now.time() >= t.start_after)
    ]
    candidates.sort(
        key=lambda t: (
            0 if t.tier.value == "must_today" else 1,
            project_priority.get(t.project_id, 999),
            -t.slide_count,
            t.estimated_minutes or 30,
        )
    )

    pick = candidates[0] if candidates else None
    return {
        "card": (
            _annotate("task", pick.model_dump(mode="json"), pick.project_id)
            if pick is not None
            else None
        ),
        "free_minutes": free_minutes,
        "in_block": False,
        "now": now.isoformat(),
    }


class MoveEventBody(BaseModel):
    minutes: int


@router.post("/api/events/{event_id}/move")
async def move_event(
    event_id: str,
    body: MoveEventBody,
    user_id: str = Depends(require_user_id),
) -> dict:
    """Push a calendar event's start/end forward (or backward, with a
    negative `minutes`) by the given offset. Pure state mutation."""
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


@router.post("/api/break/ack")
async def acknowledge_break(
    user_id: str = Depends(require_user_id),
) -> dict:
    """User saw the 'take a break' card and acknowledged it. Sets
    `last_break_acknowledged_at = now`, which resets the work-since
    counter so vessel won't show another break card until the user
    crosses the threshold again. Pure state mutation — no LLM."""
    pool = await get_pool()
    state = await state_manager.read(pool, user_id)
    now = _now_local()
    state = state.model_copy(update={"last_break_acknowledged_at": now})
    await state_manager.write(pool, user_id, state)
    return {"ok": True, "acknowledged_at": now.isoformat()}


@router.get("/api/now")
async def now_view(
    user_id: str = Depends(require_user_id),
    x_vessel_client_now: Optional[str] = Header(default=None),
) -> dict:
    """Return the single card to display in focus mode."""
    pool = await get_pool()
    state = await state_manager.read(pool, user_id)
    return _pick_now_card(state, _client_now(x_vessel_client_now))


@router.get("/api/tasks/day")
async def tasks_day(
    offset: int = 0,
    user_id: str = Depends(require_user_id),
    x_vessel_client_now: Optional[str] = Header(default=None),
) -> dict:
    pool = await get_pool()
    state = await state_manager.read(pool, user_id)
    now = _client_now(x_vessel_client_now)
    target = now.date() + timedelta(days=offset)
    is_today = offset == 0
    window = current_window(now)
    day_tasks = [
        t
        for t in state.tasks
        if t.completed_at is None
        and t.skipped_at is None
        and t.due_date == target
    ]
    if is_today:
        now_tasks = [
            t.model_dump(mode="json")
            for t in day_tasks
            if t.time_window in (window, TimeWindow.anytime)
        ]
        later_tasks = [
            t.model_dump(mode="json")
            for t in day_tasks
            if t.time_window not in (window, TimeWindow.anytime)
        ]
    else:
        now_tasks = [t.model_dump(mode="json") for t in day_tasks]
        later_tasks = []
    return {
        "now": now_tasks,
        "later": later_tasks,
        "events": _events_for_date(state, target),
        "date": target.isoformat(),
        "offset": offset,
        "is_today": is_today,
        "window": window.value if is_today else None,
        "now_iso": now.isoformat(),
    }


@router.get("/api/tasks/all")
async def tasks_all(
    user_id: str = Depends(require_user_id),
    x_vessel_client_now: Optional[str] = Header(default=None),
) -> dict:
    pool = await get_pool()
    state = await state_manager.read(pool, user_id)
    now = _client_now(x_vessel_client_now)
    items = [t.model_dump(mode="json") for t in state.tasks]
    # Open first, then completed/skipped at the bottom (both are "closed").
    # Within a day we sort by chronological position of the time_window
    # bucket — NOT the raw enum string. The string "after_work" < "before_work"
    # alphabetically, which made morning tasks appear below evening tasks
    # on the same day. Anytime sorts last so the user reads
    # before_work → workday → after_work → evening → "whenever".
    _TIME_WINDOW_ORDER = {
        TimeWindow.before_work.value: 0,
        TimeWindow.workday.value: 1,
        TimeWindow.after_work.value: 2,
        TimeWindow.evening.value: 3,
        TimeWindow.anytime.value: 4,
    }
    items.sort(
        key=lambda t: (
            t["completed_at"] is not None or t["skipped_at"] is not None,
            t["due_date"],
            _TIME_WINDOW_ORDER.get(t["time_window"], 99),
            t["title"],
        )
    )
    events = [e.model_dump(mode="json") for e in state.calendar]
    events.sort(
        key=lambda e: (
            e["completed_at"] is not None or e["skipped_at"] is not None,
            e["start"],
        )
    )
    open_count = sum(
        1
        for t in items
        if t["completed_at"] is None and t["skipped_at"] is None
    )
    return {
        "tasks": items,
        "events": events,
        "open_count": open_count,
        "total": len(items),
        "events_count": len(events),
        "now": now.isoformat(),
    }


class PushBody(BaseModel):
    days: int = 1


class SkipBody(BaseModel):
    reason: str


@router.post("/api/tasks/{task_id}/skip")
async def skip_task(
    task_id: str,
    body: SkipBody,
    user_id: str = Depends(require_user_id),
    x_vessel_client_now: Optional[str] = Header(default=None),
) -> dict:
    """User declined a task. Two-stage:

    1. Archive the task (set `skipped_at` + `skip_reason`, splice out
       of `state.tasks`, persist).
    2. If an LLM is configured, dispatch a tool-use loop that reads the
       reason and is allowed to call CRUD tools to act on the intent
       (e.g. "no more wash dishes" → delete every other open
       wash-dishes copy). Free-run scope; every tool call is returned
       in the response so the PWA can show what changed.

    If the LLM is not configured (no GROQ_API_KEY etc.) or the loop
    fails, the archive still happens — skip is never blocked on LLM
    availability."""
    reason = (body.reason or "").strip()
    if not reason:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="reason is required when skipping a task",
        )
    pool = await get_pool()
    state = await state_manager.read(pool, user_id)
    now = _client_now(x_vessel_client_now)
    found = next((t for t in state.tasks if t.id == task_id), None)
    if found is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="task not found"
        )
    found.skipped_at = now
    found.skip_reason = reason
    # Archive first, then prune from state. Skipped tasks leave the
    # working state for the same reason completed ones do — closed
    # items don't belong on the worklist and bloat the JSON blob.
    await task_history.archive(pool, user_id, found, "skipped")
    state.tasks = [t for t in state.tasks if t.id != task_id]
    await state_manager.write(pool, user_id, state)

    # ---- Stage 2: skip-reason assistant ----
    assistant_payload: dict[str, Any] = {"invoked": False}
    try:
        from ..assistant.skip_assistant import run_skip_assistant
        from ..config import get_settings
        from openai import AsyncOpenAI as _AsyncOpenAI

        settings = get_settings()
        if settings.groq_api_key:
            client = _AsyncOpenAI(
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            # The state we hand the assistant excludes the just-skipped
            # task (already removed above). Run loop, persist again if
            # any mutating tool calls landed.
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
    except Exception as exc:  # noqa: BLE001 — assistant must never block skip
        logger.exception("skip assistant failed")
        assistant_payload = {"invoked": True, "error": str(exc)}

    return {
        "ok": True,
        "task_id": task_id,
        "skipped_at": now.isoformat(),
        "assistant": assistant_payload,
    }


@router.post("/api/tasks/{task_id}/uncomplete")
async def uncomplete_task(
    task_id: str, user_id: str = Depends(require_user_id)
) -> dict:
    """Undo a recent completion. The completed task lives in the
    history table now (not `state.tasks`) so the working state stays
    small. Pop the most recent archived row, clear its `completed_at`,
    and re-insert into state. The 1-second undo toast in the PWA fires
    this. If the row is somehow still in state (legacy data from
    before history archival), clear `completed_at` in place."""
    pool = await get_pool()
    state = await state_manager.read(pool, user_id)
    found = next((t for t in state.tasks if t.id == task_id), None)
    if found is not None:
        found.completed_at = None
        await state_manager.write(pool, user_id, state)
        return {"ok": True, "task_id": task_id, "source": "state"}
    restored = await task_history.pop_latest(
        pool, user_id, task_id, closed_kind="completed"
    )
    if restored is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="task not found"
        )
    restored.completed_at = None
    state.tasks.append(restored)
    await state_manager.write(pool, user_id, state)
    return {"ok": True, "task_id": task_id, "source": "history"}


@router.post("/api/tasks/{task_id}/unskip")
async def unskip_task(
    task_id: str, user_id: str = Depends(require_user_id)
) -> dict:
    """Undo a recent skip. Mirror image of uncomplete: pop from
    history (skipped variant), clear skipped_at + skip_reason, restore
    to state."""
    pool = await get_pool()
    state = await state_manager.read(pool, user_id)
    found = next((t for t in state.tasks if t.id == task_id), None)
    if found is not None:
        found.skipped_at = None
        found.skip_reason = None
        await state_manager.write(pool, user_id, state)
        return {"ok": True, "task_id": task_id, "source": "state"}
    restored = await task_history.pop_latest(
        pool, user_id, task_id, closed_kind="skipped"
    )
    if restored is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="task not found"
        )
    restored.skipped_at = None
    restored.skip_reason = None
    state.tasks.append(restored)
    await state_manager.write(pool, user_id, state)
    return {"ok": True, "task_id": task_id, "source": "history"}


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


@router.post("/api/events/{event_id}/skip")
async def skip_event(
    event_id: str,
    body: SkipBody,
    user_id: str = Depends(require_user_id),
    x_vessel_client_now: Optional[str] = Header(default=None),
) -> dict:
    """User declined a calendar event. Two stages, mirroring
    `skip_task`:

    1. Audit-log the request + pre-mutation snapshot, then mark the
       event skipped (it stays in state with `skipped_at` set, so
       /api/events/{id}/unskip can fully recover the original).
    2. If an LLM is configured, run the skip-assistant tool-loop on
       the reason. The user typed something like "moved to next
       Friday" — the assistant can call CRUD tools to make that
       happen (insert a new calendar event, etc.). Same wiring as
       task skip so cancel/change actually drives a change rather
       than just hiding the row.

    The audit log lands in `vessel.events` BEFORE any state mutation
    so a buggy assistant or a bad cascade cannot leave us without a
    recoverable snapshot."""
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
                skipped_task=found,  # the assistant treats this generically
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
    except Exception as exc:  # noqa: BLE001 — assistant must never block the skip
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
    """Mark a calendar event as done (the user actually did the thing).
    Pure state mutation, no LLM."""
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


@router.post("/api/tasks/{task_id}/complete")
async def complete_task(
    task_id: str, user_id: str = Depends(require_user_id)
) -> dict:
    pool = await get_pool()
    state = await state_manager.read(pool, user_id)
    now = _now_local()
    found_task = None
    for task in state.tasks:
        if task.id == task_id:
            task.completed_at = now
            found_task = task
            break
    if found_task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="task not found"
        )
    # Recurrence: keep the 7-day rolling window topped up. The expander
    # is idempotent — completing a daily task ages the window by one
    # day, so it ensures a new tail-day open instance gets created.
    # IMPORTANT: run the expander BEFORE removing the completed task
    # from state. The expander uses each series' set of represented
    # dates to decide what to skip; the just-completed today instance
    # is what tells it "today is filled, don't re-create".
    spawned_ids: list[str] = []
    if found_task.recurrence == "daily":
        before_ids = {t.id for t in state.tasks}
        state = _expand_daily_recurrence(state, now.date())
        spawned_ids = [t.id for t in state.tasks if t.id not in before_ids]
    # Archive the completed task to the history table and splice it
    # out of `state.tasks`. Closed tasks don't belong on the worklist
    # and dragging them around in the JSON state blob inflates every
    # read/write.
    await task_history.archive(pool, user_id, found_task, "completed")
    state.tasks = [t for t in state.tasks if t.id != task_id]
    await state_manager.write(pool, user_id, state)
    if spawned_ids:
        return {"ok": True, "spawned_id": spawned_ids[0]}
    return {"ok": True}


@router.post("/api/tasks/{task_id}/push")
async def push_task(
    task_id: str,
    body: Optional[PushBody] = None,
    user_id: str = Depends(require_user_id),
) -> dict:
    pool = await get_pool()
    state = await state_manager.read(pool, user_id)
    now = _now_local()
    days = body.days if body else 1
    found = False
    for task in state.tasks:
        if task.id == task_id:
            task.due_date = task.due_date + timedelta(days=days)
            task.slide_count += 1
            found = True
            break
    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="task not found"
        )
    await state_manager.write(pool, user_id, state)
    return {"ok": True}


class ChatBody(BaseModel):
    text: str


@router.post("/api/chat")
async def chat(
    body: ChatBody,
    user_id: str = Depends(require_user_id),
    x_vessel_client_now: Optional[str] = Header(default=None),
) -> dict:
    """Run the chat tool-use loop. Same shape as the skip-assistant
    flow: the LLM sees (now, state, instruction) and emits CRUD tool
    calls; the route persists state once the loop returns.

    The legacy "intake agent emits a full StateData JSON" path is gone
    — chat and skip both go through `tool_loop` now, so the LLM has
    one job (CRUD via tools) on every surface.

    Response shape:
      {
        applied: bool,                 # any mutating tool call landed
        diff: {...},                   # before/after diff
        assistant: {                   # for the chat bubble UI
          stopped_reason, summary, tool_calls: [{name, arguments, ok, error}]
        }
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

    # Mutates loop_state in place via tool calls.
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
        # Re-derive priority + expand recurrence after the loop, same
        # invariants the old intake path enforced.
        derived = compute_priority_ranking(loop_state)
        if derived != loop_state.priority_ranking:
            loop_state = loop_state.model_copy(
                update={"priority_ranking": derived}
            )
        loop_state = _expand_daily_recurrence(loop_state, now.date())
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
#
# The LLM is not in this path. Both Claude Desktop (via MCP) and the
# vessel chat assistant (via OpenAI tool-use over MCP) call the same
# `vessel.crud` functions through these REST handlers. Adding /
# updating / deleting a project / task / calendar event / routine has
# exactly one implementation.
# ---------------------------------------------------------------------------


from .. import crud as _crud  # noqa: E402  (placed late on purpose)


def _crud_to_http(exc: _crud.CrudError) -> HTTPException:
    """Map CrudError subclasses to HTTP status codes."""
    if isinstance(exc, _crud.NotFound):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, _crud.IdConflict):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, _crud.StillReferenced):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    # MissingReference + BadField + anything else → 400.
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


async def _read_mut_write(user_id: str):
    """Helper: read state, return (state, write-callback). The caller
    mutates state via crud, then awaits write_state() once."""
    pool = await get_pool()
    state = await state_manager.read(pool, user_id)

    async def _write(new_state: StateData) -> None:
        await state_manager.write(pool, user_id, new_state)

    return state, _write


# ----- Projects ------------------------------------------------------------


class ProjectIn(BaseModel):
    # Permissive: any subset of Project fields. The CRUD layer
    # validates the shape via Pydantic.
    model_config = {"extra": "allow"}


@router.post("/api/projects")
async def crud_add_project(
    body: ProjectIn, user_id: str = Depends(require_user_id)
) -> dict:
    state, write = await _read_mut_write(user_id)
    try:
        proj = _crud.add_project(state, body.model_dump())
    except _crud.CrudError as exc:
        raise _crud_to_http(exc)
    await write(state)
    return {"ok": True, "project": proj.model_dump(mode="json")}


@router.post("/api/projects/bulk")
async def crud_add_projects_bulk(
    body: list[dict], user_id: str = Depends(require_user_id)
) -> dict:
    state, write = await _read_mut_write(user_id)
    try:
        added = _crud.add_projects_bulk(state, body)
    except _crud.CrudError as exc:
        raise _crud_to_http(exc)
    await write(state)
    return {"ok": True, "projects": [p.model_dump(mode="json") for p in added]}


@router.patch("/api/projects/{project_id}")
async def crud_update_project(
    project_id: str,
    body: ProjectIn,
    user_id: str = Depends(require_user_id),
) -> dict:
    state, write = await _read_mut_write(user_id)
    try:
        proj = _crud.update_project(state, project_id, body.model_dump())
    except _crud.CrudError as exc:
        raise _crud_to_http(exc)
    await write(state)
    return {"ok": True, "project": proj.model_dump(mode="json")}


@router.delete("/api/projects/{project_id}")
async def crud_delete_project(
    project_id: str, user_id: str = Depends(require_user_id)
) -> dict:
    state, write = await _read_mut_write(user_id)
    try:
        _crud.delete_project(state, project_id)
    except _crud.CrudError as exc:
        raise _crud_to_http(exc)
    await write(state)
    return {"ok": True, "project_id": project_id}


# ----- Tasks ---------------------------------------------------------------


class TaskIn(BaseModel):
    model_config = {"extra": "allow"}


@router.post("/api/tasks")
async def crud_add_task(
    body: TaskIn, user_id: str = Depends(require_user_id)
) -> dict:
    state, write = await _read_mut_write(user_id)
    try:
        task = _crud.add_task(state, body.model_dump())
    except _crud.CrudError as exc:
        raise _crud_to_http(exc)
    await write(state)
    return {"ok": True, "task": task.model_dump(mode="json")}


@router.post("/api/tasks/bulk")
async def crud_add_tasks_bulk(
    body: list[dict], user_id: str = Depends(require_user_id)
) -> dict:
    state, write = await _read_mut_write(user_id)
    try:
        added = _crud.add_tasks_bulk(state, body)
    except _crud.CrudError as exc:
        raise _crud_to_http(exc)
    await write(state)
    return {"ok": True, "tasks": [t.model_dump(mode="json") for t in added]}


@router.patch("/api/tasks/{task_id}")
async def crud_update_task(
    task_id: str,
    body: TaskIn,
    user_id: str = Depends(require_user_id),
) -> dict:
    state, write = await _read_mut_write(user_id)
    try:
        task = _crud.update_task(state, task_id, body.model_dump())
    except _crud.CrudError as exc:
        raise _crud_to_http(exc)
    await write(state)
    return {"ok": True, "task": task.model_dump(mode="json")}


@router.delete("/api/tasks/{task_id}")
async def crud_delete_task(
    task_id: str, user_id: str = Depends(require_user_id)
) -> dict:
    state, write = await _read_mut_write(user_id)
    try:
        _crud.delete_task(state, task_id)
    except _crud.CrudError as exc:
        raise _crud_to_http(exc)
    await write(state)
    return {"ok": True, "task_id": task_id}


# ----- Calendar events -----------------------------------------------------


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


# ----- Routines ------------------------------------------------------------


class RoutineIn(BaseModel):
    model_config = {"extra": "allow"}


@router.post("/api/routines")
async def crud_add_routine(
    body: RoutineIn, user_id: str = Depends(require_user_id)
) -> dict:
    state, write = await _read_mut_write(user_id)
    try:
        r = _crud.add_routine(state, body.model_dump())
    except _crud.CrudError as exc:
        raise _crud_to_http(exc)
    await write(state)
    return {"ok": True, "routine": r.model_dump(mode="json")}


@router.post("/api/routines/bulk")
async def crud_add_routines_bulk(
    body: list[dict], user_id: str = Depends(require_user_id)
) -> dict:
    state, write = await _read_mut_write(user_id)
    try:
        added = _crud.add_routines_bulk(state, body)
    except _crud.CrudError as exc:
        raise _crud_to_http(exc)
    await write(state)
    return {"ok": True, "routines": [r.model_dump(mode="json") for r in added]}


@router.patch("/api/routines/{routine_id}")
async def crud_update_routine(
    routine_id: str,
    body: RoutineIn,
    user_id: str = Depends(require_user_id),
) -> dict:
    state, write = await _read_mut_write(user_id)
    try:
        r = _crud.update_routine(state, routine_id, body.model_dump())
    except _crud.CrudError as exc:
        raise _crud_to_http(exc)
    await write(state)
    return {"ok": True, "routine": r.model_dump(mode="json")}


@router.delete("/api/routines/{routine_id}")
async def crud_delete_routine(
    routine_id: str, user_id: str = Depends(require_user_id)
) -> dict:
    state, write = await _read_mut_write(user_id)
    try:
        _crud.delete_routine(state, routine_id)
    except _crud.CrudError as exc:
        raise _crud_to_http(exc)
    await write(state)
    return {"ok": True, "routine_id": routine_id}


def mount_static(app) -> None:
    app.mount(
        "/pwa",
        StaticFiles(directory=str(STATIC_DIR), html=True),
        name="pwa-static",
    )
