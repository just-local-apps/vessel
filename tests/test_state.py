from datetime import date, datetime, time, timedelta, timezone

import pytest
from pydantic import ValidationError

from vessel.models import (
    CalendarEvent,
    Cadence,
    Project,
    ProjectStatus,
    StateData,
    Task,
    Tier,
    TimeWindow,
)


def _project() -> Project:
    return Project(
        id="p1",
        name="Home",
        status=ProjectStatus.active,
        tracked=True,
        cadence=Cadence.daily,
        last_touched=datetime.now(timezone.utc),
    )


def test_empty_state_validates():
    s = StateData()
    assert s.projects == []
    assert s.tasks == []


def test_task_refs_must_be_known():
    p = _project()
    task = Task(
        id="t1",
        project_id="nope",
        title="x",
        time_window=TimeWindow.workday,
        tier=Tier.flex,
        due_date=date.today(),
        created_at=datetime.now(timezone.utc),
    )
    with pytest.raises(ValidationError):
        StateData(projects=[p], tasks=[task])


def test_valid_references():
    p = _project()
    task = Task(
        id="t1",
        project_id="p1",
        title="x",
        time_window=TimeWindow.workday,
        tier=Tier.flex,
        due_date=date.today(),
        created_at=datetime.now(timezone.utc),
    )
    cal = CalendarEvent(
        id="c1",
        project_id="p1",
        title="meet",
        description="",
        start=datetime.now(timezone.utc),
        end=datetime.now(timezone.utc),
    )
    s = StateData(projects=[p], tasks=[task], calendar=[cal], priority_ranking=["p1"])
    assert s.tasks[0].id == "t1"
    assert s.priority_ranking == ["p1"]


def test_priority_ranking_must_reference_known():
    with pytest.raises(ValidationError):
        StateData(priority_ranking=["nope"])


def test_state_round_trip_json():
    p = _project()
    s = StateData(projects=[p])
    blob = s.model_dump_json()
    back = StateData.model_validate_json(blob)
    assert back.projects[0].id == "p1"


# ---------------------------------------------------------------------------
# 3NF tripwire — see CLAUDE.md "derived fields are derived, not stored".
# These tests fail if someone re-introduces a stored column for a value
# that is functionally dependent on another non-key field.
# ---------------------------------------------------------------------------


# field name -> the field it is functionally dependent on (must be derived).
_DERIVED_FIELDS_TASK: dict[str, str] = {
    "time_window": "start_after",
}


def test_no_derived_columns_on_task():
    """`time_window` is a categorical bucket of `start_after` — storing
    it caused the bucket to drift from the gate (UI said "anytime",
    scheduling said "after 18:00"). It must remain a `@computed_field`,
    never a regular field. If this test fails, you re-introduced the
    bug. Read CLAUDE.md before "fixing" it."""
    fields = set(Task.model_fields.keys())
    for derived, source in _DERIVED_FIELDS_TASK.items():
        assert derived not in fields, (
            f"Task.{derived} must be derived from Task.{source} "
            f"(use @computed_field). See CLAUDE.md."
        )
        assert derived in Task.model_computed_fields, (
            f"Task.{derived} must exist as a @computed_field so the "
            f"PWA still receives it in serialized output."
        )


def test_time_window_is_derived_from_start_after():
    """Round-trip: setting `start_after` is the ONLY way to influence
    `time_window`. Bucketing follows the user's workday/bedtime config
    (defaults: workday 9–17, bedtime 23 → evening at 21)."""
    base = dict(
        id="t",
        project_id="p1",
        title="x",
        tier=Tier.flex,
        due_date=date.today(),
        created_at=datetime.now(timezone.utc),
    )
    cases = [
        (None, TimeWindow.anytime),
        (time(6, 0), TimeWindow.before_work),
        (time(10, 0), TimeWindow.workday),
        (time(17, 30), TimeWindow.after_work),
        (time(21, 0), TimeWindow.evening),
    ]
    for start_after, expected in cases:
        t = Task(**base, start_after=start_after)
        assert t.time_window == expected, (
            f"start_after={start_after} should bucket to {expected}, "
            f"got {t.time_window}"
        )


def test_legacy_time_window_input_is_migrated_to_start_after():
    """Old callers (legacy DB blob, LLM that hasn't seen the new
    schema) may still emit `time_window`. The model's pre-validator
    strips it and projects the categorical bucket back onto
    `start_after`, so existing data continues to round-trip."""
    p = _project()
    raw = dict(
        id="t",
        project_id="p1",
        title="x",
        tier=Tier.flex,
        due_date=date.today(),
        created_at=datetime.now(timezone.utc),
        time_window="evening",
    )
    t = Task.model_validate(raw)
    assert t.start_after == time(21, 0)
    assert t.time_window == TimeWindow.evening
    # And explicit start_after wins over legacy time_window.
    raw["time_window"] = "anytime"
    raw["start_after"] = time(7, 0)
    t2 = Task.model_validate(raw)
    assert t2.start_after == time(7, 0)
    assert t2.time_window == TimeWindow.before_work
    StateData(projects=[p], tasks=[t])  # validates referentially


def test_task_cannot_be_both_completed_and_skipped():
    p = _project()
    now = datetime.now(timezone.utc)
    bad = Task(
        id="t",
        project_id="p1",
        title="x",
        tier=Tier.flex,
        due_date=date.today(),
        created_at=now,
        completed_at=now,
        skipped_at=now,
    )
    with pytest.raises(ValidationError):
        StateData(projects=[p], tasks=[bad])


def test_calendar_event_cannot_end_before_it_starts():
    p = _project()
    start = datetime.now(timezone.utc)
    bad = CalendarEvent(
        id="c",
        project_id="p1",
        title="m",
        description="",
        start=start,
        end=start - timedelta(hours=1),
    )
    with pytest.raises(ValidationError):
        StateData(projects=[p], calendar=[bad])


def test_calendar_event_cannot_be_both_completed_and_skipped():
    p = _project()
    now = datetime.now(timezone.utc)
    bad = CalendarEvent(
        id="c",
        project_id="p1",
        title="m",
        description="",
        start=now,
        end=now + timedelta(hours=1),
        completed_at=now,
        skipped_at=now,
    )
    with pytest.raises(ValidationError):
        StateData(projects=[p], calendar=[bad])
