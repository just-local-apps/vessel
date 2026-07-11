"""OpenAI-style function-calling schema for vessel's calendar CRUD surface.

The LLM has exactly 4 tools: add_calendar_event, update_calendar_event,
delete_calendar_event, get_state. Nothing else.

Keep this file in sync with `vessel/mcp_server.py`'s `_list_tools()`
output — adding a tool there means adding it here too.
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


def _event_fields_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "fields": {
                "type": "object",
                "description": (
                    "Calendar event fields. Accepted keys: title (string, required "
                    "for add), start (ISO8601 datetime), end (ISO8601 datetime), "
                    "description (string), url (string), location (string), "
                    "arrive_by (ISO8601 datetime)."
                ),
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "url": {"type": "string"},
                    "start": {"type": "string", "format": "date-time"},
                    "end": {"type": "string", "format": "date-time"},
                    "location": {"type": "string"},
                    "arrive_by": {"type": "string", "format": "date-time"},
                },
            },
        },
        "required": ["fields"],
        "additionalProperties": False,
    }


def _id_event_fields_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "fields": {
                "type": "object",
                "description": (
                    "Fields to update. Accepted keys: title, description, url, "
                    "start (ISO8601), end (ISO8601), location, arrive_by (ISO8601)."
                ),
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "url": {"type": "string"},
                    "start": {"type": "string", "format": "date-time"},
                    "end": {"type": "string", "format": "date-time"},
                    "location": {"type": "string"},
                    "arrive_by": {"type": "string", "format": "date-time"},
                },
            },
        },
        "required": ["id", "fields"],
        "additionalProperties": False,
    }


def _id_only_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
        "additionalProperties": False,
    }


def _items_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Array of calendar event field objects.",
            },
        },
        "required": ["items"],
        "additionalProperties": False,
    }


TOOLS: list[dict[str, Any]] = [
    _tool(
        "get_state",
        "Read the current calendar — returns all calendar events.",
        {"type": "object", "properties": {}, "additionalProperties": False},
    ),

    _tool(
        "add_calendar_event",
        (
            "Create ONE calendar event. Required fields: title, start (ISO8601), "
            "end (ISO8601). Optional: description, url, location, arrive_by (ISO8601). "
            "id is auto-generated from the title + date."
        ),
        _event_fields_schema(),
    ),

    _tool(
        "add_calendar_events_bulk",
        "Create multiple calendar events in one call. Each item in `items` is an event fields object.",
        _items_schema(),
    ),

    _tool(
        "update_calendar_event",
        (
            "Update fields on an existing calendar event by id. "
            "Only the fields you supply are changed. Accepted: title, description, "
            "url, start (ISO8601), end (ISO8601), location, arrive_by (ISO8601)."
        ),
        _id_event_fields_schema(),
    ),

    _tool(
        "delete_calendar_event",
        "Delete a calendar event by id. Hard delete — cannot be undone via tools.",
        _id_only_schema(),
    ),
]
