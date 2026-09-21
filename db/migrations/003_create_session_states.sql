-- ============================================================================
-- Phase 14B — per-session behavioral verification state.
--
-- Persists ONLY the aggregate server-authoritative verification state for one
-- authentication session (one row per JWT login):
--
--     user_ref               TEXT (primary key part) — the authenticated username
--     session_id             TEXT (primary key part) — stable per-login id
--                            derived from the JWT claims (sub + iat)
--     state                  TEXT — one of: 'verified', 'reverification_required'
--     consecutive_suspicious INTEGER — how many consecutive SUSPICIOUS windows
--                            preceded the current state (>= 0)
--     last_verified_at       TIMESTAMPTZ — when the session last produced a
--                            VERIFIED continuous window (NULL until one exists)
--     created_at             TIMESTAMPTZ (first touch of the session)
--     updated_at             TIMESTAMPTZ (last state transition)
--
-- The composite PRIMARY KEY (user_ref, session_id) — with atomic
-- INSERT ... ON CONFLICT upserts in the repository — makes concurrent
-- requests race-safe: the first request creates the row, later ones update it
-- in place.
--
-- Storage & privacy
-- -----------------
-- Only aggregate, non-sensitive flags and timestamps live here. Raw
-- keyboard/mouse events, behavioural embeddings, centroids, key identity,
-- characters, passwords, JWT tokens and secrets are NEVER written to this
-- table. The CHECK constraint on ``state`` and the non-negative bound on
-- ``consecutive_suspicious`` are enforced by the database as well as by the
-- application.
--
-- A fresh login (a new JWT with a new ``iat`` -> a new ``session_id``) simply
-- inserts a clean new row; the previous session's row is left intact but is
-- never consulted again. Idempotent (CREATE TABLE IF NOT EXISTS), so applying
-- the migration twice is harmless.
-- ============================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS behavioral_session_states (
    user_ref               TEXT        NOT NULL,
    session_id             TEXT        NOT NULL,
    state                  TEXT        NOT NULL
                           CHECK (state IN ('verified', 'reverification_required')),
    consecutive_suspicious INTEGER     NOT NULL DEFAULT 0
                           CHECK (consecutive_suspicious >= 0),
    last_verified_at       TIMESTAMPTZ,
    created_at             TIMESTAMPTZ NOT NULL,
    updated_at             TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (user_ref, session_id)
);

COMMIT;