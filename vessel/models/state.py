from datetime import date, datetime, time
from typing import Optional

from pydantic import BaseModel, computed_field, model_validator

from .enums import (
    Cadence,
    Importance,
    ProjectStatus,
    RoutineKind,
    Tier,
    TimeWindow,
    Weekday,
)


def bucket_time_window(t: Optional[time]) -> TimeWindow:
    """Project a clock time into the categorical TimeWindow bucket.

    Single source of truth for the bucketing — used by `Task.time_window`
    (computed field) AND by `vessel.pwa.routes.current_window(now)` so a
    task gated to `start_after=19:00` lands in the same bucket as the
    user's wall clock at 19:00. Two callers, one rule, never drift.

    Boundaries follow the user's workday/bedtime config so changing
    `WORKDAY_START_HOUR` etc. moves the labels in lock-step.
    """
    if t is None:
        return TimeWindow.anytime
    # Local import: keeps `vessel.models` import-time-pure (no .env read
    # at module load), while still letting the bucket follow user config.
    from ..config import get_settings

    settings = get_settings()
    h = t.hour
    if h < settings.workday_start_hour:
        return TimeWindow.before_work
    if h < settings.workday_end_hour:
        return TimeWindow.workday
    if h < settings.bedtime_hour - 2:
        return TimeWindow.after_work
    return TimeWindow.evening


# Migration map: when a caller still passes the legacy `time_window`
# field without a `start_after`, project the categorical bucket back
# into a clock gate so `start_after` becomes the source of truth from
# then on. Each value sits at the LOW end of its bucket so a round-trip
# through `bucket_time_window` returns the same window under default
# config (workday 9–17, bedtime 23 → evening starts at 21).
_LEGACY_TIME_WINDOW_TO_START_AFTER: dict[str, Optional[time]] = {
    "before_work": time(6, 0),
    "workday": time(9, 0),
    "after_work": time(17, 0),
    "evening": time(21, 0),
    "anytime": None,
}


class Project(BaseModel):
    id: str
    name: str
    status: ProjectStatus
    tracked: bool
    cadence: Cadence
    last_touched: datetime
    # Vessel's job is to drive these toward "done". Filled in by the intake
    # agent (asks clarifications when missing).
    importance: Importance = Importance.medium
    goal: Optional[str] = None              # Definition of done in plain English.
    target_date: Optional[date] = None      # When this needs to be achieved.
    why: Optional[str] = None               # One-line motivation, informs trade-offs.


class Task(BaseModel):
    id: str
    project_id: str
    title: str
    notes: Optional[str] = None
    # Optional link the user attached or the agent inferred from context
    # (e.g. a Confluence page, GitHub issue, calendar invite URL). Pure
    # display field — Vessel does not fetch or validate it.
    url: Optional[str] = None
    tier: Tier
    estimated_minutes: Optional[int] = None
    due_date: date
    created_at: datetime
    completed_at: Optional[datetime] = None
    # Set when the user explicitly declined the task (left-swipe / skip).
    # The reason is what the agent reads later to learn the pattern; the
    # skip itself is a pure state mutation and does NOT invoke an LLM.
    skipped_at: Optional[datetime] = None
    skip_reason: Optional[str] = None
    slide_count: int = 0
    # Optional time-of-day gate: the task is hidden from the focus card
    # until the local wall clock crosses this time on its due_date.
    # Lets the user say "wash dishes after 7" and not see the card all
    # afternoon. Stored as "HH:MM" in local TZ so it's trivially
    # comparable to `_now_local().time()`.
    #
    # `start_after` is the SINGLE source of truth for time-of-day gating.
    # The `time_window` categorical bucket below is DERIVED from it; do
    # not add a separate stored window field (see CLAUDE.md).
    start_after: Optional[time] = None
    # Recurrence cadence for auto-spawning the next instance on
    # completion. "none" (default) → one-off task. "daily" → completing
    # this task creates a fresh open task with due_date += 1 day,
    # preserving title/project/start_after/etc. Vessel keeps the
    # recurrence model tiny on purpose; weekly/monthly can be modeled
    # by completing on a different cadence or layered on later.
    recurrence: str = "none"

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_time_window(cls, data):
        """`time_window` used to be a stored column; it is now computed
        from `start_after`. If a caller (legacy DB blob, old test, LLM
        that hasn't seen the new schema) still passes `time_window`,
        strip it and synthesize a `start_after` so the categorical
        intent survives — but only when there's no explicit
        `start_after` to override. Explicit gate always wins."""
        if not isinstance(data, dict):
            return data
        legacy = data.pop("time_window", None)
        if legacy is None:
            return data
        if data.get("start_after") not in (None, ""):
            return data
        key = legacy.value if hasattr(legacy, "value") else str(legacy)
        synthesized = _LEGACY_TIME_WINDOW_TO_START_AFTER.get(key)
        if synthesized is not None:
            data["start_after"] = synthesized
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def time_window(self) -> TimeWindow:
        """Categorical bucket the task lives in for filter/display.

        Pure function of `start_after` + user config — never stored,
        never written. If you find yourself wanting to set this, set
        `start_after` instead.
        """
        return bucket_time_window(self.start_after)


class CalendarEvent(BaseModel):
    id: str
    project_id: str
    title: str
    description: str
    phone_number: Optional[str] = None
    # Optional link tied to the event (Zoom/Meet/Teams URL, calendar
    # invite, agenda doc, ticket). Display-only — no fetch / no validate.
    url: Optional[str] = None
    start: datetime
    end: datetime
    location: Optional[str] = None
    # Optional "arrive by" deadline — a meeting at 10:00 with travel
    # time means arrive_by=09:45. Display-only nudge; the scheduler
    # treats `start` as the canonical block boundary and is unaware
    # of arrive_by. Stored, not derived: depends only on `id` (the
    # user/agent sets it explicitly, it is not a function of `start`).
    arrive_by: Optional[datetime] = None
    # Set when the user marked this event as done (right-swipe) or
    # explicitly skipped it (left-swipe with a reason). Same shape as
    # Task, so the agent can read both signals from one place.
    completed_at: Optional[datetime] = None
    skipped_at: Optional[datetime] = None
    skip_reason: Optional[str] = None


class RoutineSlot(BaseModel):
    """A recurring time slot the user owns — wake, gym, dinner, etc.

    The planner uses these as anchors: don't schedule tasks during a `fixed`
    slot, and prefer to slot work *around* them rather than over them.
    """

    id: str
    label: str
    start_time: time            # Local time, e.g. 07:00
    duration_minutes: int       # How long the slot lasts
    days: list[Weekday] = []    # Empty list means every day.
    kind: RoutineKind = RoutineKind.fixed
    source: str = "user"        # "user" (stated explicitly) | "inferred"
    confidence: float = 1.0     # 0.0–1.0


class StateData(BaseModel):
    projects: list[Project] = []
    tasks: list[Task] = []
    calendar: list[CalendarEvent] = []
    routines: list[RoutineSlot] = []
    priority_ranking: list[str] = []
    # Last time the user acknowledged a "take a break" suggestion. The
    # ranker counts work-minutes-since this timestamp (or start-of-day
    # if None) and surfaces another break card once the threshold is
    # crossed. Set by `POST /api/break/ack`.
    last_break_acknowledged_at: Optional[datetime] = None

    @model_validator(mode="after")
    def validate_invariants(self):
        project_ids = {p.id for p in self.projects}
        for task in self.tasks:
            if task.project_id not in project_ids:
                raise ValueError(
                    f"Task {task.id} refs unknown project {task.project_id}"
                )
            if task.completed_at is not None and task.skipped_at is not None:
                raise ValueError(
                    f"Task {task.id} cannot be both completed and skipped"
                )
        for event in self.calendar:
            if event.project_id not in project_ids:
                raise ValueError(
                    f"CalendarEvent {event.id} refs unknown project {event.project_id}"
                )
            if event.completed_at is not None and event.skipped_at is not None:
                raise ValueError(
                    f"CalendarEvent {event.id} cannot be both completed and skipped"
                )
            if event.end < event.start:
                raise ValueError(
                    f"CalendarEvent {event.id} ends before it starts"
                )
        for pid in self.priority_ranking:
            if pid not in project_ids:
                raise ValueError(f"priority_ranking refs unknown project {pid}")
        return self
