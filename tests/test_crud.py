"""Hermetic tests for the calendar CRUD layer.

The CRUD module is the single source of truth for calendar event operations.
Both the HTTP routes and the MCP tool layer delegate here.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from vessel import crud
from vessel.models import StateData


def _base_dt(offset_hours: int = 0) -> datetime:
    return datetime(2026, 5, 1, 9 + offset_hours, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Calendar events
# ---------------------------------------------------------------------------


def test_add_calendar_event_minimal_fields():
    state = StateData()
    start = _base_dt()
    end = start + timedelta(hours=1)
    ev = crud.add_calendar_event(
        state,
        {
            "title": "Standup",
            "start": start,
            "end": end,
        },
    )
    assert ev.id == "cal_standup_20260501"
    assert ev.description == ""
    assert state.calendar[-1].id == ev.id


def test_add_calendar_event_requires_title_start_end():
    state = StateData()
    with pytest.raises(crud.BadField):
        crud.add_calendar_event(state, {"title": "x"})  # missing start/end

    with pytest.raises(crud.BadField):
        crud.add_calendar_event(state, {})  # missing title


def test_add_calendar_event_id_conflict_raises():
    state = StateData()
    start = _base_dt()
    end = start + timedelta(hours=1)
    crud.add_calendar_event(state, {"id": "cal_x", "title": "X", "start": start, "end": end})
    with pytest.raises(crud.IdConflict):
        crud.add_calendar_event(state, {"id": "cal_x", "title": "Y", "start": start, "end": end})


def test_add_calendar_events_bulk():
    state = StateData()
    base = _base_dt()
    out = crud.add_calendar_events_bulk(
        state,
        [
            {
                "title": "A",
                "start": base,
                "end": base + timedelta(hours=1),
            },
            {
                "title": "B",
                "start": base + timedelta(hours=2),
                "end": base + timedelta(hours=3),
            },
        ],
    )
    assert [e.title for e in out] == ["A", "B"]
    assert len(state.calendar) == 2


def test_update_calendar_event_shifts_times():
    state = StateData()
    base = _base_dt()
    ev = crud.add_calendar_event(
        state,
        {
            "title": "Standup",
            "start": base,
            "end": base + timedelta(hours=1),
        },
    )
    new_start = base + timedelta(minutes=30)
    updated = crud.update_calendar_event(
        state, ev.id, {"start": new_start, "end": new_start + timedelta(hours=1)}
    )
    assert updated.start == new_start


def test_update_calendar_event_sets_location_and_url():
    state = StateData()
    base = _base_dt()
    ev = crud.add_calendar_event(
        state,
        {"title": "Doctor", "start": base, "end": base + timedelta(hours=1)},
    )
    updated = crud.update_calendar_event(
        state, ev.id, {"location": "123 Main St", "url": "https://example.com"}
    )
    assert updated.location == "123 Main St"
    assert updated.url == "https://example.com"


def test_update_calendar_event_id_change_refused():
    state = StateData()
    base = _base_dt()
    ev = crud.add_calendar_event(
        state, {"title": "X", "start": base, "end": base + timedelta(hours=1)}
    )
    with pytest.raises(crud.BadField):
        crud.update_calendar_event(state, ev.id, {"id": "other-id"})


def test_update_calendar_event_404_for_unknown():
    with pytest.raises(crud.NotFound):
        crud.update_calendar_event(StateData(), "missing-id", {"title": "x"})


def test_delete_calendar_event_removes_it():
    state = StateData()
    base = _base_dt()
    ev = crud.add_calendar_event(
        state,
        {
            "title": "Standup",
            "start": base,
            "end": base + timedelta(hours=1),
        },
    )
    crud.delete_calendar_event(state, ev.id)
    assert state.calendar == []


def test_delete_calendar_event_404_for_unknown():
    state = StateData()
    with pytest.raises(crud.NotFound):
        crud.delete_calendar_event(state, "no-such-id")


def test_add_calendar_event_auto_id_avoids_collision():
    state = StateData()
    base = _base_dt()
    ev1 = crud.add_calendar_event(
        state, {"title": "Meeting", "start": base, "end": base + timedelta(hours=1)}
    )
    ev2 = crud.add_calendar_event(
        state, {"title": "Meeting", "start": base, "end": base + timedelta(hours=1)}
    )
    assert ev1.id != ev2.id
    assert ev2.id == f"{ev1.id}_2"
