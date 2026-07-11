-- Closed-task history table.
--
-- Tasks that the user has completed or skipped no longer live in the JSON
-- StateData blob — they're spliced out and archived here. This keeps the
-- working state small (the focus / show-all / calendar views never need
-- to filter long lists of done items) and gives us a permanent log to
-- compute completion stats from later.
--
-- The encrypted_task column carries the full Task JSON (id, project_id,
-- title, notes, tier, estimated_minutes, due_date, created_at,
-- completed_at, skipped_at, skip_reason, slide_count, start_after,
-- recurrence) so an `uncomplete` / `unskip` undo can restore the row
-- exactly as it was. `time_window` is derived from `start_after` and
-- not stored separately.
--
-- `closed_kind` is "completed" | "skipped". The route layer sets it
-- so we can distinguish stats and route undo correctly.

CREATE TABLE IF NOT EXISTS vessel.task_history (
    user_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    closed_kind TEXT NOT NULL,
    closed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    encrypted_task BYTEA NOT NULL,
    PRIMARY KEY (user_id, task_id, closed_at)
);

CREATE INDEX IF NOT EXISTS idx_task_history_user_closed
    ON vessel.task_history (user_id, closed_at DESC);
