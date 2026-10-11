-- =====================================================================
-- platform.sql  -  the platform's OWN tables (not a customer's data)
--
-- Three kinds of schema live in the one database:
--   platform   this file: who the users are, which workspace is theirs, run history
--   ws_<hex>   one per business: its six data tables, clean views, training_set
--   public     unused
--
-- Applied automatically when the API starts. Safe to run repeatedly.
-- =====================================================================
CREATE SCHEMA IF NOT EXISTS platform;

CREATE TABLE IF NOT EXISTS platform.users (
    user_id        serial       PRIMARY KEY,
    email          text         NOT NULL UNIQUE,      -- stored lower-case
    password_hash  text         NOT NULL,             -- scrypt, salted; the password itself is never stored
    created_at     timestamptz  NOT NULL DEFAULT now()
);

-- One workspace per user in this prototype (UNIQUE user_id). Allowing several would only
-- mean dropping that constraint and adding a "which workspace" parameter to the API.
CREATE TABLE IF NOT EXISTS platform.workspaces (
    workspace_id    serial       PRIMARY KEY,
    user_id         integer      NOT NULL UNIQUE REFERENCES platform.users(user_id) ON DELETE CASCADE,
    business_name   text         NOT NULL,
    schema_name     text         NOT NULL UNIQUE,     -- server-generated 'ws_<hex>', never from user input
    use_case_id     text,                             -- which use-case YAML the workspace is set to
    data_rows       jsonb,                            -- row counts after the last data load
    data_loaded_at  timestamptz,
    created_at      timestamptz  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS platform.runs (
    run_id        serial       PRIMARY KEY,
    workspace_id  integer      NOT NULL REFERENCES platform.workspaces(workspace_id) ON DELETE CASCADE,
    use_case_id   text         NOT NULL,
    status        text         NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    steps         jsonb        NOT NULL DEFAULT '[]'::jsonb,   -- progress shown in the UI
    error         text,
    started_at    timestamptz  NOT NULL DEFAULT now(),
    finished_at   timestamptz
);

-- The database itself guarantees at most ONE running pipeline per workspace.
-- Two quick clicks on "Run" cannot start two runs that fight over the same training_set table.
CREATE UNIQUE INDEX IF NOT EXISTS one_running_run_per_workspace
    ON platform.runs (workspace_id) WHERE status = 'running';

COMMENT ON TABLE platform.users IS 'Platform accounts. Passwords are stored only as salted scrypt hashes.';
COMMENT ON TABLE platform.workspaces IS 'One row per business workspace; schema_name is the PostgreSQL schema holding that business''s data.';
COMMENT ON TABLE platform.runs IS 'History of analysis pipeline runs per workspace.';
