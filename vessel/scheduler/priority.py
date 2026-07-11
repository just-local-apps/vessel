"""Project priority ranking — derived from importance, deadline, and momentum.

Returns a list of project ids ordered most-important first. Used by the
day-planner and replanner to decide whose tasks to surface today.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from ..models import Importance, ProjectStatus, StateData


_IMPORTANCE_SCORE = {
    Importance.critical: 4,
    Importance.high: 3,
    Importance.medium: 2,
    Importance.low: 1,
}


def _now_date() -> date:
    return datetime.now(timezone.utc).date()


def _project_score(project, *, today: date, momentum_ids: set[str]) -> tuple:
    """Return a sort key — larger tuples sort first when reversed.

    Components:
      1. Active projects beat dormant/closed.
      2. Importance (critical > high > medium > low).
      3. Deadline urgency: closer target_date scores higher.
      4. Momentum: project has open tasks → bonus.
      5. Recency of last_touched (newer wins on ties).
    """
    is_active = project.status in (ProjectStatus.active, ProjectStatus.autopilot)
    importance = _IMPORTANCE_SCORE.get(project.importance, 2)

    if project.target_date is not None:
        days_left = (project.target_date - today).days
        # Past-due → very high urgency. Within 7 days → high. Within 30 → med.
        if days_left < 0:
            urgency = 100
        elif days_left == 0:
            urgency = 90
        elif days_left <= 7:
            urgency = 70 - days_left
        elif days_left <= 30:
            urgency = 40 - days_left
        else:
            urgency = max(0, 30 - days_left // 10)
    else:
        urgency = 0

    momentum = 1 if project.id in momentum_ids else 0

    return (
        int(is_active),
        importance,
        urgency,
        momentum,
        project.last_touched.timestamp() if project.last_touched else 0,
    )


def compute_priority_ranking(state: StateData, today: date | None = None) -> list[str]:
    """Pure helper: re-derive priority_ranking from project metadata + tasks.

    Does NOT mutate state. Callers should set the returned list onto a copy.
    """
    if not state.projects:
        return []
    today = today or _now_date()
    momentum_ids = {
        t.project_id
        for t in state.tasks
        if t.completed_at is None
    }
    scored = sorted(
        state.projects,
        key=lambda p: _project_score(p, today=today, momentum_ids=momentum_ids),
        reverse=True,
    )
    return [p.id for p in scored]
