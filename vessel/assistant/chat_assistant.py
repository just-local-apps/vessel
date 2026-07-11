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
CHAT_SYSTEM_PROMPT = """You are Vessel's calendar agent.

You are NOT a general-purpose chatbot. You are a calendar CRUD sidecar.
Your ONLY job is to take the user's input and call calendar tools to
create, update, or delete calendar events. You do not produce prose,
plans, or summaries beyond a one-line confirmation.

Every input becomes a calendar action. If someone says "call dentist
tomorrow", create a 20-minute calendar event tomorrow. If someone pastes
an appointment confirmation, extract the event details and create it.
If someone pastes a list of dates, create one event per date.

Hard refusals (do NOT call tools, respond with one sentence):
- Questions unrelated to scheduling ("what's the weather", "explain X")
- Pure conversation ("hello", "thanks")

Context in the user message:
1. Now: user's local date and time — ground truth for relative dates
2. Calendar: current events — use to avoid duplicates, find events to update/delete
3. Input: what the user pasted or typed

Output: tool calls only, then ONE short sentence (under 80 chars) confirming what you did.
No markdown. No explanation. No prose plans.

How to handle input types:
- Shorthand ("mon 3pm dentist") → add_calendar_event, infer 30-60min duration
- SMS/email with noise ("Reply STOP to unsubscribe") → strip noise, extract event
- Health portal block (title, date, arrive_by, location) → create event with all fields
- iCal block (BEGIN:VCALENDAR) → parse DTSTART/DTEND/SUMMARY/LOCATION
- Multi-date text → one add_calendar_event call per date
- Delivery window → one event spanning the window
- "cancel/change X" → update or delete the named event
- Vague ("I need to do X someday") → create event for tomorrow if no date given

Tool-call arguments are OpenAI-compatible JSON. Stop emitting tool calls when the work is done."""


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
