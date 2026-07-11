from datetime import datetime
from typing import Optional

from pydantic import BaseModel, model_validator


class CalendarEvent(BaseModel):
    id: str
    title: str
    description: str = ""
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
    # explicitly skipped it (left-swipe with a reason).
    completed_at: Optional[datetime] = None
    skipped_at: Optional[datetime] = None
    skip_reason: Optional[str] = None


class StateData(BaseModel):
    calendar: list[CalendarEvent] = []

    @model_validator(mode="after")
    def validate_invariants(self):
        for event in self.calendar:
            if event.completed_at is not None and event.skipped_at is not None:
                raise ValueError(
                    f"CalendarEvent {event.id} cannot be both completed and skipped"
                )
            if event.end < event.start:
                raise ValueError(
                    f"CalendarEvent {event.id} ends before it starts"
                )
        return self
