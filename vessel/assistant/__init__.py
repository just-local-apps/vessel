"""LLM-using surfaces of vessel — kept in their own package so vessel
core can boot without LLM credentials.

Two entry points today:
- `run_skip_assistant(reason, task, state)` — invoked when the user
  left-swipes a task with a reason. The LLM gets the reason + the
  archived task + a snapshot of state, and is free to call CRUD tools
  (delete, update, push, etc.) to act on the intent. Free-run scope:
  it can touch anything, not just the skipped task's project.
- (coming next) `run_chat_assistant(text, state)` — for the bottom
  chat input.

Both share `tool_loop()` and `TOOL_SCHEMA` so behavior is identical.
"""
from .skip_assistant import run_skip_assistant
from .tool_loop import LoopResult, ToolCall

__all__ = ["run_skip_assistant", "LoopResult", "ToolCall"]
