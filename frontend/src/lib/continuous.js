/*
 * Behavioral Biometric Authentication — Phase 14A
 * Continuous (periodic) verification window logic.
 *
 * A continuous verification window is one bounded BehavioralSession of
 * keyboard + mouse activity, submitted as a BARE session (no user_ref) to
 * POST /api/v1/continuous-verification. The server derives identity from the
 * JWT alone and returns only {decision, distance, threshold}.
 *
 * Bounds: derived from the Phase 2 collector sampling (mousemove throttled to
 * <= 40 samples/s, key events <= ~20/s) so a window never approaches the
 * backend MAX_SESSION_EVENTS = 20000 ceiling.
 *
 * Privacy: this module is storage-free by design. A captured window lives only
 * in React state, is sent once, and is discarded on submit/reset. It never
 * reads localStorage/sessionStorage and never carries identity fields.
 *
 * Dependency-free ESM — consumed by the React hook and by the Node test suite.
 */
"use strict";

export const WINDOW_LIMITS = {
  MAX_KEYBOARD_EVENTS: 300,
  MAX_MOUSE_EVENTS: 1200,
  MAX_TOTAL_EVENTS: 1500,
  MAX_DURATION_MS: 30000,
};

export const MIN_WINDOW_KEYBOARD_EVENTS = 2;
export const MIN_WINDOW_MOUSE_EVENTS = 2;

export const CONTINUOUS_STATES = {
  AUTHENTICATED: "authenticated",
  COLLECTING: "collecting",
  VERIFYING: "verifying",
  VERIFIED: "verified",
  REVERIFICATION_REQUIRED: "reverification_required",
};

export const CONTINUOUS_DECISIONS = {
  VERIFIED: "VERIFIED",
  SUSPICIOUS: "SUSPICIOUS",
};

/*
 * Server-authoritative session verification states (Phase 14B), as returned
 * by GET /continuous-verification/state and the full continuous response.
 * Mirrors the backend SessionVerificationState constants.
 */
export const SESSION_STATES = {
  VERIFIED: "verified",
  REVERIFICATION_REQUIRED: "reverification_required",
};

/*
 * Map a continuous decision to the equivalent server session state, or null
 * for an unknown decision. Used as the fallback when the server omits the
 * session_state field (defensive; new servers always send it).
 */
export function sessionDecisionToState(decision) {
  if (decision === CONTINUOUS_DECISIONS.VERIFIED) {
    return SESSION_STATES.VERIFIED;
  }
  if (decision === CONTINUOUS_DECISIONS.SUSPICIOUS) {
    return SESSION_STATES.REVERIFICATION_REQUIRED;
  }
  return null;
}

/*
 * Map a server session_state string to the client machine state, or null when
 * the value is not a recognized server state (client never fabricates one).
 */
export function restoreServerState(serverState) {
  if (serverState === SESSION_STATES.VERIFIED) {
    return CONTINUOUS_STATES.VERIFIED;
  }
  if (serverState === SESSION_STATES.REVERIFICATION_REQUIRED) {
    return CONTINUOUS_STATES.REVERIFICATION_REQUIRED;
  }
  return null;
}

/*
 * One-shot restore coordinator for the Phase 14B server-state hydration effect.
 *
 * A coordinator remembers which token already had a COMPLETED restore. It is
 * created per hook instance and becomes empty again on remount (page refresh),
 * so a refreshed /continuous page always re-reads the server verdict.
 */
export function createRestoreCoordinator() {
  return { completedToken: null };
}

/*
 * Begin a one-shot server-state restore for `token`.
 *
 * Returns a dispose() to call from the Effect cleanup.
 *
 * The token is recorded as completed ONLY after a response actually arrives.
 * React 18 development StrictMode drives a mounted effect through
 * setup → cleanup → setup, and a cancelled in-flight attempt must never burn
 * the restore: when the retried effect starts a fresh request the coordinator
 * still lets it through, so GET /continuous-verification/state = 
 * reverification_required always lands on the machine and can never be
 * swallowed back into the default authenticated state. A completed restore
 * still runs exactly once per token.
 *
 * Non-401 failures leave the machine untouched; 401 routes to onUnauthorized
 * (the caller logs the session out, matching every other API call).
 */
export function restoreSessionState(coordinator, { token, fetchState, onRestored, onUnauthorized }) {
  if (!token || coordinator.completedToken === token) {
    return () => {};
  }
  let cancelled = false;
  (async () => {
    const outcome = await fetchState({ token });
    if (cancelled) {
      return;
    }
    if (outcome.ok) {
      coordinator.completedToken = token;
      onRestored(outcome.result);
      return;
    }
    if (outcome.status === 401 && onUnauthorized) {
      onUnauthorized();
    }
  })();
  return () => {
    cancelled = true;
  };
}

export function windowEventCounts(session) {
  if (!session) {
    return { keyboard: 0, mouse: 0 };
  }
  const keyboard = Array.isArray(session.keyboard_events) ? session.keyboard_events.length : 0;
  const mouse = Array.isArray(session.mouse_events) ? session.mouse_events.length : 0;
  return { keyboard, mouse };
}

export function windowTotalEvents(session) {
  const { keyboard, mouse } = windowEventCounts(session);
  return keyboard + mouse;
}

/* Window age in ms: ended_at - started_at, or atMs - started_at when open. */
export function windowDurationMs(session, atMs = Date.now()) {
  if (!session) {
    return 0;
  }
  if (typeof atMs !== "number" || !Number.isFinite(atMs)) {
    return 0;
  }
  const started = new Date(session.started_at).getTime();
  if (!Number.isFinite(started)) {
    return 0;
  }
  const ended = session.ended_at ? new Date(session.ended_at).getTime() : NaN;
  const end = Number.isFinite(ended) ? ended : atMs;
  return Math.max(0, end - started);
}

export function windowHasUsableContent(session) {
  const { keyboard, mouse } = windowEventCounts(session);
  return keyboard >= MIN_WINDOW_KEYBOARD_EVENTS && mouse >= MIN_WINDOW_MOUSE_EVENTS;
}

/*
 * Which bound the window has hit, or null when within every bound.
 * Reasons: "keyboard" | "mouse" | "total" | "duration".
 */
export function windowFullReason(session, options = {}) {
  const atMs = options.atMs;
  const { keyboard, mouse } = windowEventCounts(session);
  if (keyboard + mouse >= WINDOW_LIMITS.MAX_TOTAL_EVENTS) {
    return "total";
  }
  if (keyboard >= WINDOW_LIMITS.MAX_KEYBOARD_EVENTS) {
    return "keyboard";
  }
  if (mouse >= WINDOW_LIMITS.MAX_MOUSE_EVENTS) {
    return "mouse";
  }
  if (windowDurationMs(session, atMs) >= WINDOW_LIMITS.MAX_DURATION_MS) {
    return "duration";
  }
  return null;
}

export function shouldStopCollecting(session, options = {}) {
  return windowFullReason(session, options) !== null;
}

/*
 * A window is submittable only when it is fully bounded AND has enough of both
 * keyboard and mouse activity for the pipeline. Returns {can, reason} where
 * reason is null on success.
 */
export function canSubmitWindow(session, options = {}) {
  if (!windowHasUsableContent(session)) {
    return { can: false, reason: "no-usable-content" };
  }
  const full = windowFullReason(session, options);
  if (full) {
    return { can: false, reason: full };
  }
  return { can: true, reason: null };
}

/* Bare deep copy of the window — identity fields are never added. */
export function buildContinuousWindow(session) {
  return JSON.parse(JSON.stringify(session));
}

export function initialState() {
  return {
    state: CONTINUOUS_STATES.AUTHENTICATED,
    last: null,
    window: null,
  };
}

/*
 * Pure state machine for the periodic verification flow:
 *   authenticated -> collecting -> verifying -> verified / reverification_required
 *   verified / reverification_required -> collecting (next window)
 * Invalid transitions return the same state object (identity-safe for React).
 */
export function continuousReducer(state, action) {
  if (!action || typeof action.type !== "string") {
    return state;
  }
  switch (action.type) {
    case "START":
      if (
        state.state !== CONTINUOUS_STATES.AUTHENTICATED &&
        state.state !== CONTINUOUS_STATES.VERIFIED &&
        state.state !== CONTINUOUS_STATES.REVERIFICATION_REQUIRED
      ) {
        return state;
      }
      return { state: CONTINUOUS_STATES.COLLECTING, last: state.last, window: null };
    case "VERIFY":
      if (state.state !== CONTINUOUS_STATES.COLLECTING || !action.window) {
        return state;
      }
      return { state: CONTINUOUS_STATES.VERIFYING, last: state.last, window: action.window };
    case "DECISION":
      if (state.state !== CONTINUOUS_STATES.VERIFYING) {
        return state;
      }
      // Phase 14B: the server is the authority on session state. When it sends
      // a session_state we trust it over any client-side mapping; otherwise we
      // fall back to the decision mapping for older/defensive behavior.
      const effectiveState =
        restoreServerState(action.sessionState) ?? restoreServerState(sessionDecisionToState(action.decision));
      if (effectiveState !== CONTINUOUS_STATES.VERIFIED && effectiveState !== CONTINUOUS_STATES.REVERIFICATION_REQUIRED) {
        return state;
      }
      // Server session fields are included only when the server sends them,
      // keeping the decision payload identical to pre-14B otherwise.
      const last = {
        decision: action.decision,
        distance: action.distance,
        threshold: action.threshold,
        ...(action.sessionState !== undefined ? { sessionState: action.sessionState } : {}),
        ...(action.consecutiveSuspicious !== undefined
          ? { consecutiveSuspicious: action.consecutiveSuspicious }
          : {}),
        ...(action.lastVerifiedAt !== undefined ? { lastVerifiedAt: action.lastVerifiedAt } : {}),
      };
      return {
        state: effectiveState,
        last,
        window: state.window,
      };
    case "RESTORE":
      // Phase 14B: hydrate from the server on fresh page load. Only meaningful
      // from the untouched authenticated state; a restore is never allowed to
      // clobber an in-progress window or verdict.
      if (state.state !== CONTINUOUS_STATES.AUTHENTICATED) {
        return state;
      }
      {
        const restored = restoreServerState(action.sessionState);
        if (restored !== CONTINUOUS_STATES.VERIFIED && restored !== CONTINUOUS_STATES.REVERIFICATION_REQUIRED) {
          return state;
        }
        return {
          state: restored,
          last: {
            decision:
              restored === CONTINUOUS_STATES.VERIFIED
                ? CONTINUOUS_DECISIONS.VERIFIED
                : CONTINUOUS_DECISIONS.SUSPICIOUS,
            distance: null,
            threshold: null,
            sessionState: action.sessionState,
            consecutiveSuspicious: action.consecutiveSuspicious,
            lastVerifiedAt: action.lastVerifiedAt,
          },
          window: null,
        };
      }
    case "RESET":
      return { state: CONTINUOUS_STATES.AUTHENTICATED, last: state.last, window: null };
    default:
      return state;
  }
}