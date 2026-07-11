"""Skip-with-reason → tool-use loop.

Invoked from `POST /api/tasks/{id}/skip` after the task has been
archived. Gives the LLM the reason, the just-skipped task, the
*current local date/time*, and a state snapshot, then lets it call
CRUD tools to act on the intent.

Free-run scope: the LLM may touch any task / event / project in state,
not just the skipped task's project. Every tool call is captured in
`LoopResult.tool_calls` so the route can echo them back to the PWA
chat bubble (so the user sees "deleted 6 tasks: …").
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from openai import AsyncOpenAI

from ..models import StateData
from ..models.state import CalendarEvent, Task
from .tool_loop import LoopResult, tool_loop
from .tool_schema import TOOLS

logger = logging.getLogger(__name__)


# The system prompt is intentionally static so the prompt-prefix cache
# stays warm. Per-request facts (current time, state snapshot) live in
# the user message instead.
SKIP_SYSTEM_PROMPT = """You are Vessel's skip-reason agent.

A user just left-swiped a task and gave a reason. Read the reason and
decide whether to make further changes to their state — and if so,
make them via the CRUD tools provided.

Common patterns:
- "back pain, no more X" / "stop X" / "cancel all X" / "remove all
  instances of X" → call `delete_task` on every open task with that
  title or topic. Use `get_state` first if you need to find them.
- "moved to next week" / "do later" → `update_task` to push due_date.
- "duplicate" → just do nothing (the task is already archived).
- "rescheduling" / "doing tomorrow instead" → `add_task` for the new
  date, or `update_task` on a sibling.
- "feeling sick", "fever" → consider clearing other tasks/events for
  the same day. Be aggressive about clearing the day if illness is
  implied; the user can always undo.

Date-handling rules (READ THESE EVERY TIME):
- The current local date and time are given in the user message under
  "Now:". You have NO independent knowledge of the date. Treat that
  value as ground truth.
- Any due_date you set on a NEW task MUST be >= today (the date in
  "Now:"). Never default to a date in the past — if the user implies
  "today" or doesn't specify, use the current date from "Now:".
- "tomorrow" = current date + 1 day. "next week" = current date + 7
  days. Calculate explicitly from "Now:", do not guess.

Other rules:
- The skipped task is ALREADY ARCHIVED — do not try to skip/delete it
  again. Act on OTHER state.
- Make every change via tool calls. Do NOT describe what you would do
  in prose; do it.
- If the reason has no actionable intent (just an explanation), reply
  with a short text message instead of any tool calls.
- After your tool calls, end with a brief one-sentence summary of
  what you did. Keep it terse — under 80 characters.

Tool schema is OpenAI-compatible. Arguments must be valid JSON."""


def _format_user_message(
    reason: str,
    entity: Task | CalendarEvent,
    state: StateData,
    now: datetime,
) -> str:
    """Pack (reason, entity, state, now) into a single user prompt so
    the model has everything — including the current date — in one
    shot. State is included verbatim; the model can also call
    `get_state` mid-loop if needed.

    `now` is REQUIRED so the model can resolve "today" / "tomorrow" /
    "next week" without guessing. Without it the LLM hallucinates
    yesterday's date roughly half the time (observed live).

    Accepts either a Task (left-swiped task) or a CalendarEvent
    (cancel/change on an event card). The noun in the prompt
    switches to match so the model picks the right CRUD tools."""
    open_tasks = [
        {
            "id": t.id,
            "title": t.title,
            "due_date": t.due_date.isoformat(),
            "project_id": t.project_id,
            "recurrence": t.recurrence,
        }
        for t in state.tasks
        if t.completed_at is None and t.skipped_at is None
    ]
    is_event = isinstance(entity, CalendarEvent)
    noun = "calendar event" if is_event else "task"
    if is_event:
        entity_summary = (
            f"{entity.title!r} (id={entity.id}, "
            f"start={entity.start.isoformat()}, end={entity.end.isoformat()})"
        )
    else:
        entity_summary = f"{entity.title!r} (id={entity.id})"
    weekday = now.strftime("%A")
    today_iso = now.date().isoformat()
    return (
        f"Now: {now.isoformat()}  (today is {weekday}, {today_iso})\n"
        f"User cancelled {noun}: {entity_summary}\n"
        f"Reason: {reason!r}\n\n"
        f"Open tasks currently in state ({len(open_tasks)}):\n"
        f"{open_tasks!r}\n\n"
        f"Decide whether to act on this reason and make any changes via "
        f"the CRUD tools. Any new {noun}'s date MUST be on or after "
        f"{today_iso}. End with a brief summary."
    )


async def run_skip_assistant(
    *,
    reason: str,
    skipped_task: Task | CalendarEvent,
    state: StateData,
    client: AsyncOpenAI,
    model: str,
    now: Optional[datetime] = None,
) -> LoopResult:
    """Run the skip-reason tool-use loop. Mutates `state` in place.

    The route layer passes `now` from `_now_local()` so the model
    knows what "today" means. Defaults to UTC-now if omitted (which
    only happens in tests that don't care about date-grounding).

    `skipped_task` is named for historical reasons but accepts a
    CalendarEvent too — the route layer passes either depending on
    which entity the user just cancelled."""
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
