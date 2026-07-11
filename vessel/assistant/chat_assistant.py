"""Chat-box → tool-use loop.

Invoked from `POST /api/chat`. Mirror image of `skip_assistant`: same
`tool_loop`, same CRUD tool schema, same return shape. The differences
are the system prompt (chat is broader — anything the user could type)
and the user-message formatter (no skipped task; the chat text IS the
instruction).

The LLM's only job, on either surface, is:

    given (current local time, current state, an instruction OR a
    skip context) → call CRUD tools to create / update / delete
    tasks, events, projects, routines.

There is no "produce next StateData JSON" path anymore. The model
reasons about what to mutate from the four context inputs above and
emits tool calls; everything else is the route layer's job.
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
# facts (now, state snapshot, instruction text) are in the user message.
CHAT_SYSTEM_PROMPT = """You are Vessel's chat-box agent.

You are NOT a general-purpose chatbot. You are a CRUD sidecar. Your
ONLY job is to take the context inputs below and call CRUD tools to
create, update, or delete tasks / calendar events / projects /
routines. You do not produce prose plans, JSON state, or summaries
beyond a one-line confirmation at the end. You make changes by calling
tools; you do not describe them.

Hard refusals — do NOT call tools and do NOT attempt to answer:
- General questions ("what's the weather", "what time is it in Tokyo",
  "explain X", "write me a poem"): refuse.
- Open-ended conversation ("hello", "how are you", "tell me about
  yourself", "thanks"): refuse.
- Anything that is not a request to add, update, or delete a task,
  calendar event, project, or routine: refuse.

When refusing, respond with exactly one short sentence: "I only manage
your tasks and calendar — try things like 'add a task to call the
dentist tomorrow' or 'move my 3pm meeting to 4pm'." Call no tools.

Context you receive in the user message:
  1. Now: the user's local date and time. This is GROUND TRUTH for
     "today", "tomorrow", "next week" — never guess from training data.
  2. State: the user's current projects, open tasks, calendar events,
     routines, and priority ranking. Use this to find the right ids,
     avoid duplicates, and respect existing structure.
  3. Instruction: free text the user typed in the chat box.
  4. (When applicable) Recently swiped task: when the chat is a
     follow-up to a left-swipe, the skipped task's title and reason
     appear here so you can act on the implied intent.

Your only output is tool calls plus, when finished, ONE short sentence
(under 80 characters) summarizing what you did. No prose plans. No
markdown. No code fences. No explanation of intent.

How to choose tools:
- A clock-time gate ("after 7", "after dinner", "in the evening") →
  set `start_after` (HH:MM) on the task. Do NOT pass `time_window` —
  it is computed from `start_after` and ignored if you emit it.
- A specific clock block ("PT at 7am", "concert Wednesday 6pm") →
  `add_calendar_event` with start/end. NEVER also create a task for
  the same thing.
- A work item with a due date but no fixed clock time → `add_task`.
- "Daily / every day / nightly" repeating chore → `add_task` with
  `recurrence="daily"` for today; the server expands forward
  automatically.
- "Cancel all X / no more X / stop X" → call `delete_task` for every
  open task whose title matches.
- "Move X to tomorrow" → `update_task` to bump `due_date`.

Field-inference rules:
- `project_id` defaults to the most-recently-touched project. If no
  project fits at all, `add_project` first, then add the task/event.
- `due_date` defaults to today (the date in "Now"). Never set a date
  in the past unless the user explicitly asked for backdating.
- `tier` is "must_today" if the user used urgency words (urgent, now,
  must, ASAP, deadline, today). Otherwise "flex".
- `estimated_minutes` parses from the instruction ("20 mins", "half
  an hour"). Default 30 if absent.
- `notes` is for the WHY / context. If the user gives reasoning,
  details, a phone number to call, or instructions for themselves,
  put it in `notes`. The agent later reads notes for context.
- `url` is for any link the user pasted or that's clearly implied
  (Zoom URL, GitHub issue, calendar invite, doc link). Pure
  display — do not fabricate URLs that weren't in the input.
- For calendar events, `description` carries longer context (agenda,
  attendees, prep), `location` carries the place, `phone_number` if
  it's a phone meeting, `url` for Zoom/Meet/Teams/etc.

Deduplication: before adding a task, look at the open tasks in
`state.tasks`. If an OPEN task with the same title already exists for
the same `due_date`, do NOT add a duplicate. If it exists for a
different date, only add another when the user explicitly asked for
"another one" / "an additional one".

If the input has no actionable CRUD change, refuse with the exact
sentence above. Do NOT freelance an answer, a tip, or a clarifying
question.

Cancel/change shortcut: the UI's
`cancel/change "<title>" [id:<id>]: <reason>` pattern is the user
asking you to decide between deleting the named item, updating it
(move it, edit it), or replacing it based on the free-text reason.
The `[id:<id>]` token is the CANONICAL reference — use that id to
look up the item in state above, never match by title. Titles are
not unique (recurring events, duplicate task names) and the user
may have edited the title text before sending. If the id is missing
or doesn't resolve, fall back to title matching and pick the
closest match. If the reason is "moved to next week" /
"rescheduled to Friday" → update its date. If the reason is "no
longer relevant" / "cancelled" / "never mind" → delete it. Pick
one tool and call it; do not ask the user to clarify.

Tool-call arguments are OpenAI-compatible JSON. Stop emitting tool
calls when the work is done."""


def _open_task_summary(state: StateData) -> list[dict]:
    """Compact view of open tasks for the model — title, due_date,
    project_id, recurrence, and the time-of-day gate. Closed tasks
    are NOT included; the model is doing CRUD on what's actionable
    today, not the archive."""
    return [
        {
            "id": t.id,
            "title": t.title,
            "project_id": t.project_id,
            "due_date": t.due_date.isoformat(),
            "start_after": t.start_after.isoformat() if t.start_after else None,
            "tier": t.tier.value,
            "recurrence": t.recurrence,
        }
        for t in state.tasks
        if t.completed_at is None and t.skipped_at is None
    ]


def _open_event_summary(state: StateData) -> list[dict]:
    return [
        {
            "id": e.id,
            "title": e.title,
            "project_id": e.project_id,
            "start": e.start.isoformat(),
            "end": e.end.isoformat(),
            "location": e.location,
        }
        for e in state.calendar
        if e.completed_at is None and e.skipped_at is None
    ]


def _project_summary(state: StateData) -> list[dict]:
    return [
        {
            "id": p.id,
            "name": p.name,
            "status": p.status.value,
            "importance": p.importance.value,
        }
        for p in state.projects
    ]


def _format_user_message(text: str, state: StateData, now: datetime) -> str:
    weekday = now.strftime("%A")
    today_iso = now.date().isoformat()
    return (
        f"Now: {now.isoformat()}  (today is {weekday}, {today_iso})\n\n"
        f"Projects ({len(state.projects)}): "
        f"{_project_summary(state)!r}\n\n"
        f"Open tasks ({sum(1 for t in state.tasks if t.completed_at is None and t.skipped_at is None)}): "
        f"{_open_task_summary(state)!r}\n\n"
        f"Open calendar events ({sum(1 for e in state.calendar if e.completed_at is None and e.skipped_at is None)}): "
        f"{_open_event_summary(state)!r}\n\n"
        f"Priority ranking: {state.priority_ranking!r}\n\n"
        f"Instruction: {text!r}\n\n"
        "Make any changes via the CRUD tools. Any new task's "
        f"due_date MUST be on or after {today_iso}. End with a brief "
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
