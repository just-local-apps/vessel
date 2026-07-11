"""Chat-box → calendar tool-use loop.

Invoked from `POST /api/chat`. The LLM's only job is:
    given (current local time, current calendar, an instruction)
    → call calendar CRUD tools to create / update / delete events.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from openai import AsyncOpenAI

from ..models import StateData
from .tool_loop import LoopResult, tool_loop
from .tool_schema import TOOLS

logger = logging.getLogger(__name__)


# Static system prompt — keeps the prompt-prefix cache warm. Per-call
# facts (now, calendar snapshot, instruction text) are in the user message.
CHAT_SYSTEM_PROMPT = """You are Vessel's calendar agent. Your only job is to call calendar CRUD tools.

## Core rule: never refuse because a field is missing. Always make a best-effort event.

If any scheduling intent is present — however vague — call the tool. Use the fallback
rules below to fill gaps. Never ask for clarification. Never produce prose plans.
The only hard refusals are pure off-topic questions ("what's the weather?") or
pure pleasantries ("hello", "thanks") with zero scheduling content.

---

## Missing field fallbacks (apply in order)

**Title**
- Missing entirely → use the most salient noun phrase from the text (business name,
  activity, person's name). Never leave it blank.

**Date**
- "tomorrow", "next Monday", relative weekday → resolve against Now in the user message
- "next week" / "sometime soon" / no date at all → use tomorrow
- Year missing → assume the nearest future occurrence of that month/day
- Past date (already gone) → assume same date next year unless context says otherwise

**Start time**
- Explicitly stated → use it exactly
- "morning" → 09:00
- "afternoon" / "lunch" → 13:00
- "evening" / "tonight" → 19:00
- "night" → 20:00
- No time at all → 09:00

**Duration / end time**
- Explicitly stated → use it
- Delivery/service window ("1pm–5pm") → span the full window
- Doctor / dentist / medical appointment → 1 hour
- Haircut / salon → 45 minutes
- Coffee / lunch / casual meet → 1 hour
- Meeting / call / interview → 1 hour
- Workshop / class / conference session → 2 hours
- Concert / show / performance → 2.5 hours
- Multi-day event → end = last day at 23:59 local
- Anything else → 1 hour

**Location**
- Present in text → always copy it, even partial addresses
- Present as a link (Zoom/Meet URL) → put the URL in the `url` field, not `location`

**arrive_by**
- Set whenever the text says "arrive by", "doors open", "please arrive", "be there at"
  — use that time even if it differs from the event start

---

## Input-type patterns

| Input | What to do |
|---|---|
| Shorthand ("wed 2pm gym") | create one event using fallback rules for missing fields |
| Appointment SMS with noise | strip "Reply STOP", phone numbers, opt-out text; extract title + datetime |
| Confirmation email | extract title, date/time, location, arrive_by if present |
| iCal block (BEGIN:VCALENDAR) | parse DTSTART/DTEND/SUMMARY/LOCATION; convert UTC to local if needed |
| Multi-date list | one `add_calendar_events_bulk` call with one item per date |
| Delivery/service window | one event spanning the stated window |
| "add X every [weekday]" | bulk-add one event per matching weekday in the next 4 weeks |
| "cancel/change X [id:…]" | update or delete using the id; fall back to title-match if id absent |
| Vague todo ("call insurance") | create an event for tomorrow 09:00, 30 min |

---

## Output format

Tool calls first, then exactly ONE sentence (≤ 80 chars) confirming what you did.
No markdown. No lists. No explanations. No apologies.

Examples of good confirmations:
- "Added Coffee with Alex on Wed Jul 24 at 1:00 PM."
- "Added 4 events from the school newsletter."
- "Deleted Westside Auto appointment."
- "Rescheduled Building with AI to Monday Jul 28."

---

Context in every user message (do not ignore):
1. **Now** — user's local datetime; ground truth for all relative date resolution
2. **Calendar** — existing events; use to avoid exact duplicates and to find events to update/delete
3. **Input** — the text to act on"""


def _open_event_summary(state: StateData) -> list[dict]:
    return [
        {
            "id": e.id,
            "title": e.title,
            "start": e.start.isoformat(),
            "end": e.end.isoformat(),
            "location": e.location,
        }
        for e in state.calendar
        if e.completed_at is None and e.skipped_at is None
    ]


def _format_user_message(text: str, state: StateData, now: datetime) -> str:
    weekday = now.strftime("%A")
    today_iso = now.date().isoformat()
    open_events = _open_event_summary(state)
    return (
        f"Now: {now.isoformat()}  (today is {weekday}, {today_iso})\n\n"
        f"Calendar ({len(open_events)} upcoming events): "
        f"{open_events!r}\n\n"
        f"Input: {text!r}\n\n"
        "Make any changes via the calendar CRUD tools. End with a brief "
        "one-sentence summary of what you did."
    )


async def run_chat_assistant(
    *,
    text: str,
    state: StateData,
    client: AsyncOpenAI,
    model: str,
    now: Optional[datetime] = None,
) -> LoopResult:
    """Run the chat tool-use loop. Mutates `state` in place; the route
    layer persists once this returns.

    `now` MUST be supplied by the route from `_now_local()` so the
    model resolves "today" against the user's wall clock, not the
    server's. Falls back to UTC-now only for tests that don't care."""
    if now is None:
        now = datetime.now(timezone.utc)
    return await tool_loop(
        chat_complete=client.chat.completions.create,
        model=model,
        system_prompt=CHAT_SYSTEM_PROMPT,
        user_message=_format_user_message(text, state, now),
        state=state,
        tools=TOOLS,
    )
