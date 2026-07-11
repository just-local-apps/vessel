from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from vessel.models import CalendarEvent, StateData


def _event(
    id: str = "c1",
    start: datetime | None = None,
    end: datetime | None = None,
    **kwargs,
) -> CalendarEvent:
    now = datetime.now(timezone.utc)
    return CalendarEvent(
        id=id,
        title="Test event",
        start=start or now,
        end=end or (start or now) + timedelta(hours=1),
        **kwargs,
    )


def test_empty_state_validates():
    s = StateData()
    assert s.calendar == []


def test_state_round_trip_json():
    now = datetime.now(timezone.utc)
    s = StateData(
        calendar=[
            CalendarEvent(
                id="c1",
                title="Dentist",
                start=now,
                end=now + timedelta(hours=1),
            )
        ]
    )
    blob = s.model_dump_json()
    back = StateData.model_validate_json(blob)
    assert back.calendar[0].id == "c1"


def test_calendar_event_cannot_end_before_it_starts():
    start = datetime.now(timezone.utc)
    bad = _event(start=start, end=start - timedelta(hours=1))
    with pytest.raises(ValidationError):
        StateData(calendar=[bad])


def test_calendar_event_cannot_be_both_completed_and_skipped():
    now = datetime.now(timezone.utc)
    bad = _event(
        start=now,
        end=now + timedelta(hours=1),
        completed_at=now,
        skipped_at=now,
    )
    with pytest.raises(ValidationError):
        StateData(calendar=[bad])


def test_calendar_event_valid_with_optional_fields():
    now = datetime.now(timezone.utc)
    ev = CalendarEvent(
        id="c2",
        title="Doctor",
        start=now,
        end=now + timedelta(hours=1),
        location="123 Main St",
        arrive_by=now - timedelta(minutes=15),
        url="https://example.com",
        description="Annual checkup",
    )
    s = StateData(calendar=[ev])
    assert s.calendar[0].location == "123 Main St"
    assert s.calendar[0].arrive_by is not None


# ---------------------------------------------------------------------------
# 3NF tripwire — CalendarEvent has no derived fields, but keep the
# test structure so the invariant is enforced if we add any.
# ---------------------------------------------------------------------------


def test_no_derived_columns_on_calendar_event():
    """CalendarEvent currently has no computed fields. All stored fields
    must depend only on `id` (3NF). This test documents the expectation."""
    # No computed fields expected on CalendarEvent.
    assert len(CalendarEvent.model_computed_fields) == 0
