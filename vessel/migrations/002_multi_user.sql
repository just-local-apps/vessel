-- Multi-user + at-rest encryption.
--
-- Drops the single-user state/events/invocations tables and recreates them
-- with `user_id TEXT NOT NULL` partition columns and `encrypted_*` BYTEA
-- columns in place of the previous plain-text JSON / TEXT columns. Any
-- existing data is wiped — single-user state was test data.
--
-- This migration is gated by `vessel._migrations`; once recorded, it never
-- runs again, so the destructive DROPs only fire on the first deploy.

DROP TABLE IF EXISTS vessel.invocations CASCADE;
DROP TABLE IF EXISTS vessel.events CASCADE;
DROP TABLE IF EXISTS vessel.state CASCADE;

CREATE TABLE vessel.state (
    user_id TEXT PRIMARY KEY,
    encrypted_data BYTEA NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE vessel.events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    sensor_name TEXT NOT NULL,
    encrypted_payload BYTEA NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

CREATE INDEX idx_events_unprocessed
    ON vessel.events (user_id, recorded_at ASC)
    WHERE processed_at IS NULL;

CREATE INDEX idx_events_user_recorded
    ON vessel.events (user_id, recorded_at DESC);

CREATE TABLE vessel.invocations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    event_id UUID NOT NULL REFERENCES vessel.events(id),
    agent_id TEXT NOT NULL,
    encrypted_state_before BYTEA NOT NULL,
    encrypted_state_after BYTEA NOT NULL,
    model TEXT NOT NULL,
    encrypted_rendered_prompt BYTEA NOT NULL,
    encrypted_raw_response BYTEA NOT NULL,
    latency_ms INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    parse_error TEXT,
    error TEXT
);

CREATE INDEX idx_invocations_event ON vessel.invocations (event_id);
CREATE INDEX idx_invocations_user_created ON vessel.invocations (user_id, created_at DESC);
