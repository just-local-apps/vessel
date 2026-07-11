"""Tests for compute_priority_ranking — the deterministic ordering helper
the planner consumes."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from vessel.models import (
    Cadence,
    Importance,
    Project,
    ProjectStatus,
    StateData,
    Task,
    Tier,
    TimeWindow,
)
from vessel.scheduler.priority import compute_priority_ranking


def _project(
    id: str,
    *,
    importance: Importance = Importance.medium,
    target_date: date | None = None,
    status: ProjectStatus = ProjectStatus.active,
    last_touched: datetime | None = None,
) -> Project:
    return Project(
        id=id,
        name=id,
        status=status,
        tracked=True,
        cadence=Cadence.event_driven,
        last_touched=last_touched or datetime(2026, 4, 25, tzinfo=timezone.utc),
        importance=importance,
        target_date=target_date,
    )


def test_critical_outranks_low_regardless_of_recency():
    a = _project("a", importance=Importance.critical)
    b = _project(
        "b",
        importance=Importance.low,
        last_touched=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    state = StateData(projects=[b, a])
    assert compute_priority_ranking(state) == ["a", "b"]


def test_near_target_date_outranks_far_target_within_same_importance():
    today = date(2026, 4, 25)
    near = _project(
        "near", importance=Importance.high, target_date=today + timedelta(days=2)
    )
    far = _project(
        "far", importance=Importance.high, target_date=today + timedelta(days=60)
    )
    state = StateData(projects=[far, near])
    assert compute_priority_ranking(state, today=today) == ["near", "far"]


def test_active_outranks_dormant_even_when_dormant_is_critical():
    # Dormant projects don't get the user's day even if importance is high.
    active_med = _project("active", importance=Importance.medium)
    dormant_crit = _project(
        "dormant", importance=Importance.critical, status=ProjectStatus.dormant
    )
    state = StateData(projects=[dormant_crit, active_med])
    assert compute_priority_ranking(state) == ["active", "dormant"]


def test_momentum_breaks_ties_within_same_importance():
    p1 = _project("p1", importance=Importance.medium)
    p2 = _project("p2", importance=Importance.medium)
    state = StateData(
        projects=[p1, p2],
        tasks=[
            Task(
                id="t1",
                project_id="p2",
                title="in flight",
                time_window=TimeWindow.anytime,
                tier=Tier.flex,
                due_date=date(2026, 4, 25),
                created_at=datetime(2026, 4, 25, tzinfo=timezone.utc),
            )
        ],
    )
    assert compute_priority_ranking(state) == ["p2", "p1"]


def test_empty_state_returns_empty_list():
    assert compute_priority_ranking(StateData()) == []
