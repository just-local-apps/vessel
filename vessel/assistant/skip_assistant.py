"""Skip-with-reason → calendar tool-use loop.

Invoked from `POST /api/events/{id}/skip` after the event has been
marked skipped. Gives the LLM the reason, the just-skipped event, the
*current local date/time*, and a state snapshot, then lets it call
calendar CRUD tools to act on the intent.

Example: user skips a dentist appointment with reason "moved to next
Friday" → the assistant can call update_calendar_event to reschedule it,
or add_calendar_event to create a new one.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from openai import AsyncOpenAI

from ..models import StateData
from ..models.state import CalendarEvent
from .tool_loop import LoopResult, tool_loop
from .tool_schema import TOOLS

logger = logging.getLogger(__name__)


# The system prompt is intentionally static so the prompt-prefix cache
# stays warm. Per-request facts (current time, state snapshot) live in
# the user message instead.
SKIP_SYSTEM_PROMPT = """You are Vessel's skip-reason agent.

A user just left-swiped a calendar event and gave a reason. Read the reason and
decide whether to make further changes to their calendar — and if so,
make them via the calendar CRUD tools provided.

Common patterns:
- "moved to next Friday" / "rescheduling" → add_calendar_event for the new
  date (or update_calendar_event if you can find the same event by a different id).
- "cancelled" / "no longer happening" → just do nothing (the event is already skipped).
- "feeling sick" / "fever" → consider skipping/deleting other events for
  the same day. Be aggressive about clearing the day if illness is implied.
- "duplicate" → just do nothing.

Date-handling rules (READ THESE EVERY TIME):
- The current local date and time are given in the user message under
  "Now:". You have NO independent knowledge of the date. Treat that
  value as ground truth.
- Any new event you add MUST have start/end >= today (the date in "Now:").
  Never default to a date in the past.
- "tomorrow" = current date + 1 day. "next week" = current date + 7 days.
  Calculate explicitly from "Now:", do not guess.

Other rules:
- The skipped event is ALREADY MARKED SKIPPED — do not try to skip/delete it
  again. Act on OTHER state or create new events.
- Make every change via tool calls. Do NOT describe what you would do in prose.
- If the reason has no actionable intent (just an explanation), reply
  with a short text message instead of any tool calls.
- After your tool calls, end with a brief one-sentence summary of
  what you did. Keep it terse — under 80 characters.

Tool schema is OpenAI-compatible. Arguments must be valid JSON."""


def _format_user_message(
    reason: str,
    event: CalendarEvent,
    state: StateData,
    now: datetime,
) -> str:
    """Pack (reason, event, state, now) into a single user prompt."""
    open_events = [
        {
            "id": e.id,
            "title": e.title,
            "start": e.start.isoformat(),
            "end": e.end.isoformat(),
        }
        for e in state.calendar
        if e.completed_at is None and e.skipped_at is None
    ]
    entity_summary = (
        f"{event.title!r} (id={event.id}, "
        f"start={event.start.isoformat()}, end={event.end.isoformat()})"
    )
    weekday = now.strftime("%A")
    today_iso = now.date().isoformat()
    return (
        f"Now: {now.isoformat()}  (today is {weekday}, {today_iso})\n"
        f"User cancelled calendar event: {entity_summary}\n"
        f"Reason: {reason!r}\n\n"
        f"Upcoming calendar events ({len(open_events)}):\n"
        f"{open_events!r}\n\n"
        f"Decide whether to act on this reason and make any changes via "
        f"the calendar CRUD tools. Any new event's start MUST be on or after "
        f"{today_iso}. End with a brief summary."
    )


async def run_skip_assistant(
    *,
    reason: str,
    skipped_task: CalendarEvent,  # named for historical compat; always CalendarEvent now
    state: StateData,
    client: AsyncOpenAI,
    model: str,
    now: Optional[datetime] = None,
) -> LoopResult:
    """Run the skip-reason tool-use loop. Mutates `state` in place.

    `skipped_task` is named for historical reasons but is always a
    CalendarEvent now — the parameter name is kept for call-site compat."""
    if now is None:
        now = datetime.now(timezone.utc)
    return await tool_loop(
        chat_complete=client.chat.completions.create,
        model=model,
        system_prompt=SKIP_SYSTEM_PROMPT,
        user_message=_format_user_message(reason, skipped_task, state, now),
        state=state,
        tools=TOOLS,
    )
