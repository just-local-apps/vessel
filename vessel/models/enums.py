from enum import Enum


class ProjectStatus(str, Enum):
    active = "active"
    autopilot = "autopilot"
    dormant = "dormant"
    closed = "closed"


class Cadence(str, Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"
    event_driven = "event_driven"


class TimeWindow(str, Enum):
    before_work = "before_work"
    workday = "workday"
    after_work = "after_work"
    evening = "evening"
    anytime = "anytime"


class Tier(str, Enum):
    must_today = "must_today"
    flex = "flex"


class Importance(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Weekday(str, Enum):
    mon = "mon"
    tue = "tue"
    wed = "wed"
    thu = "thu"
    fri = "fri"
    sat = "sat"
    sun = "sun"


class RoutineKind(str, Enum):
    fixed = "fixed"  # anchor — never schedule tasks during this slot
    flex = "flex"    # preferred but movable
