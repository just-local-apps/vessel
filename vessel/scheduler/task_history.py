"""Persistence layer for completed / skipped tasks.

A closed task is moved out of `StateData.tasks` and archived here so the
working state stays small. The route layer (`/api/tasks/{id}/complete`,
`/skip`) calls `archive`; the `/uncomplete`, `/unskip` routes call
`pop_latest` to restore the most recent matching row.

Tasks are stored encrypted at rest (same Fernet key used by the JSON
state blob) so a DB compromise can't read titles or notes.
"""
from __future__ import annotations

from typing import Optional

import asyncpg

from ..encryption import decrypt_json, encrypt_json
from ..models.state import Task


async def archive(
    pool: asyncpg.Pool,
    user_id: str,
    task: Task,
    closed_kind: str,
) -> None:
    """Insert one task into the history table. Idempotent only by
    (user_id, task_id, closed_at) — repeated calls with the same task
    create separate rows, which lets us restore the latest one cleanly."""
    blob = encrypt_json(task.model_dump(mode="json"))
    await pool.execute(
        """
        INSERT INTO vessel.task_history (user_id, task_id, closed_kind, encrypted_task)
        VALUES ($1, $2, $3, $4)
        """,
        user_id,
        task.id,
        closed_kind,
        blob,
    )


async def pop_latest(
    pool: asyncpg.Pool,
    user_id: str,
    task_id: str,
    closed_kind: Optional[str] = None,
) -> Optional[Task]:
    """Remove and return the most recent archived row for `task_id`.

    Used by the undo flow: the user swiped right to complete, then
    tapped undo within the toast window — we pop the row out of history
    and the caller re-inserts the Task into `StateData.tasks`.

    `closed_kind` (optional) restricts to either 'completed' or
    'skipped' so an uncomplete request doesn't accidentally restore a
    skipped row, and vice versa."""
    if closed_kind is None:
        row = await pool.fetchrow(
            """
            DELETE FROM vessel.task_history
            WHERE ctid = (
                SELECT ctid FROM vessel.task_history
                WHERE user_id = $1 AND task_id = $2
                ORDER BY closed_at DESC
                LIMIT 1
            )
            RETURNING encrypted_task
            """,
            user_id,
            task_id,
        )
    else:
        row = await pool.fetchrow(
            """
            DELETE FROM vessel.task_history
            WHERE ctid = (
                SELECT ctid FROM vessel.task_history
                WHERE user_id = $1 AND task_id = $2 AND closed_kind = $3
                ORDER BY closed_at DESC
                LIMIT 1
            )
            RETURNING encrypted_task
            """,
            user_id,
            task_id,
            closed_kind,
        )
    if row is None:
        return None
    raw = decrypt_json(bytes(row["encrypted_task"]))
    return Task.model_validate(raw)


async def list_recent(
    pool: asyncpg.Pool,
    user_id: str,
    *,
    limit: int = 100,
) -> list[dict]:
    """Read the most recent archived tasks for diagnostic / history
    views. Returns plain dicts so the caller can serialize directly."""
    rows = await pool.fetch(
        """
        SELECT task_id, closed_kind, closed_at, encrypted_task
        FROM vessel.task_history
        WHERE user_id = $1
        ORDER BY closed_at DESC
        LIMIT $2
        """,
        user_id,
        limit,
    )
    out = []
    for row in rows:
        task_json = decrypt_json(bytes(row["encrypted_task"]))
        out.append(
            {
                "task_id": row["task_id"],
                "closed_kind": row["closed_kind"],
                "closed_at": row["closed_at"].isoformat(),
                "task": task_json,
            }
        )
    return out
