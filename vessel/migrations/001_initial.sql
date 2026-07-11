-- Vessel initial schema: events, state, invocations.

CREATE SCHEMA IF NOT EXISTS vessel;

CREATE TABLE IF NOT EXISTS vessel.events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sensor_name TEXT NOT NULL,
    payload JSONB NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_events_unprocessed
    ON vessel.events (recorded_at ASC)
    WHERE processed_at IS NULL;

CREATE TABLE IF NOT EXISTS vessel.state (
    id INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    data JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO vessel.state (id, data)
VALUES (1, '{"projects":[],"tasks":[],"calendar":[],"priority_ranking":[]}')
ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS vessel.invocations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID NOT NULL REFERENCES vessel.events(id),
    agent_id TEXT NOT NULL,
    state_before JSONB NOT NULL,
    state_after JSONB NOT NULL,
    model TEXT NOT NULL,
    rendered_prompt TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    latency_ms INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    parse_error TEXT,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_invocations_event ON vessel.invocations (event_id);
CREATE INDEX IF NOT EXISTS idx_invocations_created ON vessel.invocations (created_at DESC);
