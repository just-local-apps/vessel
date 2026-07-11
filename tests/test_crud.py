"""Hermetic tests for the CRUD layer.

The CRUD module is the single source of truth for "create / update /
delete a project / task / calendar event / routine". Both the HTTP
routes and the MCP tool layer delegate here, so every shape it
enforces locks behavior across both surfaces.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pytest

from vessel import crud
from vessel.models import StateData
from vessel.models.enums import (
    Cadence,
    Importance,
    ProjectStatus,
    Tier,
    TimeWindow,
)
from vessel.models.state import Project


def _seed_with_project() -> StateData:
    p = Project(
        id="p_demo",
        name="Demo",
        status=ProjectStatus.active,
        tracked=True,
        cadence=Cadence.event_driven,
        last_touched=datetime(2026, 4, 28, tzinfo=timezone.utc),
        importance=Importance.medium,
    )
    return StateData(projects=[p])


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


def test_add_project_minimal_input_fills_defaults():
    state = StateData()
    proj = crud.add_project(state, {"name": "Health"})
    assert proj.id == "p_health"
    assert proj.status == ProjectStatus.active
    assert proj.tracked is True
    assert proj.cadence == Cadence.event_driven
    assert proj.importance == Importance.medium
    assert state.projects[-1].id == "p_health"


def test_add_project_requires_name():
    with pytest.raises(crud.BadField):
        crud.add_project(StateData(), {})


def test_add_project_id_conflict_raises():
    state = _seed_with_project()
    with pytest.raises(crud.IdConflict):
        crud.add_project(state, {"id": "p_demo", "name": "Other"})


def test_add_projects_bulk_creates_many():
    state = StateData()
    out = crud.add_projects_bulk(
        state,
        [{"name": "A"}, {"name": "B"}, {"name": "C"}],
    )
    assert [p.name for p in out] == ["A", "B", "C"]
    assert len(state.projects) == 3


def test_update_project_changes_only_specified_fields():
    state = _seed_with_project()
    updated = crud.update_project(
        state, "p_demo", {"importance": "high", "goal": "ship draft"}
    )
    assert updated.importance == Importance.high
    assert updated.goal == "ship draft"
    # Untouched fields preserved.
    assert updated.cadence == Cadence.event_driven


def test_update_project_id_change_refused():
    state = _seed_with_project()
    with pytest.raises(crud.BadField):
        crud.update_project(state, "p_demo", {"id": "p_other"})


def test_update_project_404_for_unknown():
    with pytest.raises(crud.NotFound):
        crud.update_project(StateData(), "p_missing", {"name": "x"})


def test_delete_project_removes_it_and_clears_priority_ranking():
    state = _seed_with_project()
    state.priority_ranking = ["p_demo"]
    crud.delete_project(state, "p_demo")
    assert state.projects == []
    assert state.priority_ranking == []


def test_delete_project_refuses_when_open_task_references_it():
    state = _seed_with_project()
    crud.add_task(state, {"title": "Buy milk", "project_id": "p_demo"})
    with pytest.raises(crud.StillReferenced):
        crud.delete_project(state, "p_demo")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def test_add_task_defaults_project_to_most_recent():
    state = _seed_with_project()
    # Add a more-recently-touched project; the task should attach to it.
    newer = Project(
        id="p_newer",
        name="Newer",
        status=ProjectStatus.active,
        tracked=True,
        cadence=Cadence.event_driven,
        last_touched=datetime(2026, 4, 29, tzinfo=timezone.utc),
    )
    state.projects.append(newer)
    task = crud.add_task(state, {"title": "Read paper"})
    assert task.project_id == "p_newer"
    assert task.tier == Tier.flex
    assert task.time_window == TimeWindow.anytime
    assert task.estimated_minutes == 30
    assert task.due_date == date.today()
    assert task.recurrence == "none"


def test_add_task_uses_explicit_project_id_and_validates_it_exists():
    state = _seed_with_project()
    task = crud.add_task(
        state, {"title": "Buy milk", "project_id": "p_demo"}
    )
    assert task.project_id == "p_demo"


def test_add_task_with_unknown_project_raises():
    state = _seed_with_project()
    with pytest.raises(crud.MissingReference):
        crud.add_task(state, {"title": "x", "project_id": "p_nope"})


def test_add_task_no_projects_at_all_raises_helpful_error():
    with pytest.raises(crud.MissingReference):
        crud.add_task(StateData(), {"title": "x"})


def test_add_task_auto_id_avoids_collision():
    state = _seed_with_project()
    crud.add_task(
        state, {"title": "Wash dishes", "project_id": "p_demo",
                "due_date": date(2026, 4, 29)}
    )
    second = crud.add_task(
        state, {"title": "Wash dishes", "project_id": "p_demo",
                "due_date": date(2026, 4, 29)}
    )
    assert second.id == "task_wash_dishes_20260429_2"


def test_add_tasks_bulk_creates_many():
    state = _seed_with_project()
    out = crud.add_tasks_bulk(
        state,
        [
            {"title": "A", "project_id": "p_demo"},
            {"title": "B", "project_id": "p_demo"},
        ],
    )
    assert [t.title for t in out] == ["A", "B"]


def test_update_task_changes_recurrence_and_start_after():
    state = _seed_with_project()
    t = crud.add_task(
        state, {"title": "Wash dishes", "project_id": "p_demo"}
    )
    updated = crud.update_task(
        state, t.id, {"recurrence": "daily", "start_after": time(19, 0)}
    )
    assert updated.recurrence == "daily"
    assert updated.start_after == time(19, 0)


def test_update_task_validates_referenced_project():
    state = _seed_with_project()
    t = crud.add_task(state, {"title": "x", "project_id": "p_demo"})
    with pytest.raises(crud.MissingReference):
        crud.update_task(state, t.id, {"project_id": "p_nope"})


def test_delete_task_removes_it():
    state = _seed_with_project()
    t = crud.add_task(state, {"title": "x", "project_id": "p_demo"})
    crud.delete_task(state, t.id)
    assert state.tasks == []


# ---------------------------------------------------------------------------
# Calendar events
# ---------------------------------------------------------------------------


def test_add_calendar_event_minimal_fields():
    state = _seed_with_project()
    start = datetime(2026, 5, 1, 9, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    ev = crud.add_calendar_event(
        state,
        {
            "project_id": "p_demo",
            "title": "Standup",
            "start": start,
            "end": end,
        },
    )
    assert ev.id == "cal_standup_20260501"
    assert ev.description == ""
    assert state.calendar[-1].id == ev.id


def test_add_calendar_event_requires_start_end_title():
    state = _seed_with_project()
    with pytest.raises(crud.BadField):
        crud.add_calendar_event(state, {"project_id": "p_demo"})


def test_add_calendar_events_bulk():
    state = _seed_with_project()
    base = datetime(2026, 5, 1, 9, tzinfo=timezone.utc)
    out = crud.add_calendar_events_bulk(
        state,
        [
            {
                "project_id": "p_demo", "title": "A",
                "start": base, "end": base + timedelta(hours=1),
            },
            {
                "project_id": "p_demo", "title": "B",
                "start": base + timedelta(hours=2),
                "end": base + timedelta(hours=3),
            },
        ],
    )
    assert [e.title for e in out] == ["A", "B"]


def test_update_calendar_event_shifts_times():
    state = _seed_with_project()
    base = datetime(2026, 5, 1, 9, tzinfo=timezone.utc)
    ev = crud.add_calendar_event(
        state,
        {
            "project_id": "p_demo", "title": "Standup",
            "start": base, "end": base + timedelta(hours=1),
        },
    )
    new_start = base + timedelta(minutes=30)
    updated = crud.update_calendar_event(
        state, ev.id, {"start": new_start, "end": new_start + timedelta(hours=1)}
    )
    assert updated.start == new_start


def test_delete_calendar_event_removes_it():
    state = _seed_with_project()
    base = datetime(2026, 5, 1, 9, tzinfo=timezone.utc)
    ev = crud.add_calendar_event(
        state,
        {
            "project_id": "p_demo", "title": "Standup",
            "start": base, "end": base + timedelta(hours=1),
        },
    )
    crud.delete_calendar_event(state, ev.id)
    assert state.calendar == []


# ---------------------------------------------------------------------------
# Routines
# ---------------------------------------------------------------------------


def test_add_routine_minimal():
    state = StateData()
    r = crud.add_routine(
        state,
        {
            "label": "Morning gym",
            "start_time": time(7, 0),
            "duration_minutes": 60,
        },
    )
    assert r.id == "routine_morning_gym"
    assert r.kind.value == "fixed"
    assert r.source == "user"


def test_add_routine_requires_core_fields():
    with pytest.raises(crud.BadField):
        crud.add_routine(StateData(), {"label": "x"})


def test_update_routine_changes_duration():
    state = StateData()
    r = crud.add_routine(
        state,
        {"label": "Lunch", "start_time": time(12), "duration_minutes": 30},
    )
    updated = crud.update_routine(
        state, r.id, {"duration_minutes": 45}
    )
    assert updated.duration_minutes == 45


def test_delete_routine_removes_it():
    state = StateData()
    r = crud.add_routine(
        state,
        {"label": "Lunch", "start_time": time(12), "duration_minutes": 30},
    )
    crud.delete_routine(state, r.id)
    assert state.routines == []
