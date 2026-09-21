-- ============================================================================
-- Phase 10 — enrollment profile storage (PostgreSQL).
--
-- Persists ONLY enrollment aggregates, never raw behavioural events:
--
--     user_ref      unique identity metadata (text primary key)
--     centroid      DOUBLE PRECISION[] (Phase 8 centroid embedding)
--     embedding_dim fixed dimensionality of the centroid (must be > 0)
--     session_count number of enrolled sessions (must be >= 0)
--     created_at    TIMESTAMPTZ (enrollment time)
--     updated_at    TIMESTAMPTZ (metadata; Phase 10 never mutates profiles)
--
-- The primary key on user_ref IS the race-safe duplicate guard: a concurrent
-- second INSERT for the same user raises a unique-violation, which the
-- repository translates into the structured 409 ProfileExistsError.
--
-- Raw keyboard/mouse events, key identity, characters, passwords, tokens and
-- secrets are NEVER written here; no PHI/biometrics are stored beyond the
-- aggregate centroid. Idempotent (CREATE TABLE IF NOT EXISTS) so applying the
-- migration twice is harmless.
-- ============================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS behavioral_profiles (
    user_ref       TEXT             PRIMARY KEY,
    centroid       DOUBLE PRECISION[] NOT NULL,
    embedding_dim  INTEGER          NOT NULL CHECK (embedding_dim > 0),
    session_count  INTEGER          NOT NULL CHECK (session_count >= 0),
    created_at     TIMESTAMPTZ      NOT NULL,
    updated_at     TIMESTAMPTZ      NOT NULL
);

COMMIT;