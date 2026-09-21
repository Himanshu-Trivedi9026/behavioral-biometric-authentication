/*
 * Behavioral Biometric Authentication — Phase 14A
 * useContinuousVerification — periodic continuous verification flow.
 *
 * Owns the bounded-window capture state machine (lib/continuous.js), the
 * BehavioralSession capture (reusing useCollector), automatic window capping,
 * and submission to POST /api/v1/continuous-verification.
 *
 * Privacy: captured windows live only in React state, are submitted once, and
 * are discarded after submit/reset — nothing is written to localStorage or
 * sessionStorage beyond the existing auth session. The window body never
 * carries identity fields (user_ref) or key identities.
 */
import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { useAuth, logOut } from "../auth.js";
import { continuousVerify, getContinuousSessionState } from "../api.js";
import { useCollector } from "./useCollector.js";
import {
  CONTINUOUS_STATES,
  buildContinuousWindow,
  canSubmitWindow,
  continuousReducer,
  createRestoreCoordinator,
  initialState,
  restoreSessionState,
  windowFullReason,
} from "../lib/continuous.js";

const POLL_MS = 500;

function errorText(outcome) {
  if (outcome.status === 401) {
    return "Your session has expired. Please log in again.";
  }
  if (outcome.code === "profile_not_found") {
    return "No behavioral profile exists for this account yet. Enroll first to build your profile.";
  }
  if (outcome.code === "invalid_session" || outcome.status === 422) {
    return "The captured window was rejected as invalid. Record a window with keyboard and mouse activity, then try again.";
  }
  if (outcome.status === 503) {
    return "Behavioral service unavailable. Please try again later.";
  }
  if (outcome.status === 500 || outcome.status === 502) {
    return "Unexpected server error. Please try again.";
  }
  if (outcome.status === 0) {
    return "Unable to reach the behavioral service. Please try again.";
  }
  return "Continuous verification failed. Please try again.";
}

export const WINDOW_CAP_LABELS = {
  keyboard: "keyboard event cap reached",
  mouse: "mouse event cap reached",
  total: "total event cap reached",
  duration: "window time cap reached",
};

export function useContinuousVerification() {
  const auth = useAuth();
  const collector = useCollector();
  const [machine, dispatch] = useReducer(continuousReducer, undefined, initialState);
  const [error, setError] = useState(null);
  const [errorCode, setErrorCode] = useState(null);
  const [capReason, setCapReason] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const collectorRef = useRef(collector);
  useEffect(() => {
    collectorRef.current = collector;
  });

  const evaluateWindow = useCallback(() => {
    const c = collectorRef.current;
    if (c.status !== "running") {
      return;
    }
    const startedAt = c.session ? c.session.started_at : null;
    const view = {
      started_at: startedAt,
      keyboard_events: new Array(c.keyboardCount),
      mouse_events: new Array(c.mouseCount),
    };
    const reason = windowFullReason(view);
    if (reason) {
      setCapReason(WINDOW_CAP_LABELS[reason]);
      c.stop();
    } else {
      setCapReason(null);
    }
  }, []);

  useEffect(() => {
    if (collector.status !== "running") {
      return undefined;
    }
    const timerId = setInterval(evaluateWindow, POLL_MS);
    return () => clearInterval(timerId);
  }, [collector.status, evaluateWindow]);

  useEffect(() => {
    if (collector.status === "running") {
      evaluateWindow();
    }
  }, [collector.status, collector.keyboardCount, collector.mouseCount, evaluateWindow]);

  /*
   * Phase 14B — hydrate from the server on (re)mount / refresh. The session is
   * logged in, so GET /continuous-verification/state returns the authoritative
   * verdict and the machine is restored EXACTLY once per token.
   *
   * The coordinator marks a token completed only after a response arrives.
   * React 18 development StrictMode runs a mounted effect as
   * setup → cleanup → setup: the first in-flight request is cancelled by its
   * own cleanup, but the retried effect issues a fresh request that still
   * passes the coordinator, so the server state ((re)verified /
   * reverification_required) always lands. The dependency is the stable token
   * string, never the useAuth() object (a fresh object literal each render,
   * which would re-run/cancel this effect after every unrelated re-render).
   */
  const restoreCoordinatorRef = useRef(null);
  if (restoreCoordinatorRef.current === null) {
    restoreCoordinatorRef.current = createRestoreCoordinator();
  }
  useEffect(() => {
    const token = auth ? auth.token : null;
    if (!token) {
      return undefined;
    }
    return restoreSessionState(restoreCoordinatorRef.current, {
      token,
      fetchState: getContinuousSessionState,
      onRestored(result) {
        dispatch({
          type: "RESTORE",
          sessionState: result.session_state,
          consecutiveSuspicious: result.consecutive_suspicious,
          lastVerifiedAt: result.last_verified_at,
        });
      },
      onUnauthorized() {
        logOut();
      },
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [auth ? auth.token : null]);

  const startWindow = useCallback(() => {
    setError(null);
    setErrorCode(null);
    setCapReason(null);
    dispatch({ type: "START" });
    collector.start();
  }, [collector]);

  const finishWindow = useCallback(() => {
    if (collector.status === "running") {
      collector.stop();
    }
  }, [collector]);

  const resetWindow = useCallback(() => {
    setError(null);
    setErrorCode(null);
    setCapReason(null);
    setSubmitting(false);
    dispatch({ type: "RESET" });
    collector.clear();
  }, [collector]);

  const submitWindow = useCallback(async () => {
    const c = collectorRef.current;
    const session = c.status === "stopped" ? c.session : machine.window;
    if (!session) {
      return;
    }
    const gate = canSubmitWindow(session);
    if (!gate.can) {
      setError("The captured window is not usable yet. Record a window with keyboard and mouse activity.");
      setErrorCode(gate.reason);
      return;
    }
    if (machine.state !== CONTINUOUS_STATES.COLLECTING) {
      return;
    }
    setSubmitting(true);
    setError(null);
    setErrorCode(null);
    dispatch({ type: "VERIFY", window: session });

    const token = auth ? auth.token : null;
    const outcome = await continuousVerify({
      session: buildContinuousWindow(session),
      token,
    });
    setSubmitting(false);

    if (outcome.ok) {
      const result = outcome.result;
      dispatch({
        type: "DECISION",
        decision: result.decision,
        distance: result.distance,
        threshold: result.threshold,
        sessionState: result.session_state,
        consecutiveSuspicious: result.consecutive_suspicious,
        lastVerifiedAt: result.last_verified_at,
      });
      return;
    }

    if (outcome.status === 401) {
      logOut();
    }
    setError(errorText(outcome));
    setErrorCode(outcome.code);
    dispatch({ type: "RESET" });
  }, [auth, machine.state, machine.window]);

  return {
    state: machine.state,
    lastDecision: machine.last,
    window: machine.window,
    captureSession: collector.session,
    error,
    errorCode,
    capReason,
    submitting,
    captureStatus: collector.status,
    keyboardCount: collector.keyboardCount,
    mouseCount: collector.mouseCount,
    lastEvent: collector.lastEvent,
    sessionId: collector.sessionId,
    startWindow,
    finishWindow,
    submitWindow,
    resetWindow,
  };
}