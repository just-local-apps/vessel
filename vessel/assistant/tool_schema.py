"""OpenAI-style function-calling schema for vessel's CRUD surface.

Same set of tools the MCP server exposes, expressed in the JSON-schema
shape the OpenAI Chat Completions API expects under `tools=[...]`. Both
Groq's gpt-oss model and OpenAI directly accept this format.

Keep this file in sync with `vessel/mcp_server.py`'s `_list_tools()`
output — adding a tool there means adding it here too. They're separate
because the MCP `Tool` type and the OpenAI tool dict are different
shapes; sharing a single source of truth would need a converter that's
more code than just maintaining both.
"""
from __future__ import annotations

from typing import Any


def _tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _fields_only_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "fields": {
                "type": "object",
                "description": "Field name → value. See entity schema for keys.",
            },
        },
        "required": ["fields"],
        "additionalProperties": False,
    }


def _items_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "object"},
            },
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def _id_only_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
        "additionalProperties": False,
    }


def _id_fields_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "fields": {"type": "object"},
        },
        "required": ["id", "fields"],
        "additionalProperties": False,
    }


TOOLS: list[dict[str, Any]] = [
    _tool("get_state", "Read the current StateData (projects, tasks, calendar, routines).",
          {"type": "object", "properties": {}, "additionalProperties": False}),

    # Projects
    _tool("add_project",
          "Add ONE project. fields: name (req), id, status, tracked, cadence, importance, goal, target_date, why.",
          _fields_only_schema()),
    _tool("add_projects_bulk", "Add many projects in one call.", _items_schema()),
    _tool("update_project", "Update a subset of a project's fields by id.", _id_fields_schema()),
    _tool("delete_project",
          "Delete a project by id. Refuses if open tasks/events still reference it.",
          _id_only_schema()),

    # Tasks
    _tool("add_task",
          "Add ONE task. fields: title (req), project_id, due_date, tier, "
          "estimated_minutes, notes, start_after (HH:MM — the displayed "
          "window is derived from this; do NOT pass time_window), "
          "recurrence ('none'|'daily').",
          _fields_only_schema()),
    _tool("add_tasks_bulk", "Add many tasks in one call.", _items_schema()),
    _tool("update_task",
          "Update a subset of a task's fields by id. Use to change recurrence, "
          "start_after, due_date, tier, etc.",
          _id_fields_schema()),
    _tool("delete_task", "Delete a task by id (hard delete from state).", _id_only_schema()),

    # Calendar events
    _tool("add_calendar_event",
          "Add ONE calendar event. fields: title (req), start (req), end (req), "
          "project_id, description, location, phone_number.",
          _fields_only_schema()),
    _tool("add_calendar_events_bulk", "Add many calendar events in one call.", _items_schema()),
    _tool("update_calendar_event",
          "Update a subset of a calendar event's fields by id.", _id_fields_schema()),
    _tool("delete_calendar_event", "Delete a calendar event by id.", _id_only_schema()),

    # Routines
    _tool("add_routine",
          "Add ONE routine slot. fields: label (req), start_time (HH:MM, req), "
          "duration_minutes (req), days, kind, source, confidence.",
          _fields_only_schema()),
    _tool("add_routines_bulk", "Add many routines in one call.", _items_schema()),
    _tool("update_routine", "Update a subset of a routine's fields by id.", _id_fields_schema()),
    _tool("delete_routine", "Delete a routine by id.", _id_only_schema()),
]
