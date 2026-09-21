/*
 * Behavioral Biometric Authentication — Phase 14A
 * Continuous (periodic) verification frontend logic: window bounds, payload
 * builder, state machine, and the continuousVerify API client.
 *
 * Run with: npm test
 */
"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");

const { continuousVerify, getContinuousSessionState } = require("../frontend/src/api.js");
const {
  WINDOW_LIMITS,
  MIN_WINDOW_KEYBOARD_EVENTS,
  MIN_WINDOW_MOUSE_EVENTS,
  CONTINUOUS_STATES,
  CONTINUOUS_DECISIONS,
  SESSION_STATES,
  sessionDecisionToState,
  restoreServerState,
  createRestoreCoordinator,
  restoreSessionState,
  windowEventCounts,
  windowTotalEvents,
  windowDurationMs,
  windowHasUsableContent,
  windowFullReason,
  shouldStopCollecting,
  canSubmitWindow,
  buildContinuousWindow,
  initialState,
  continuousReducer,
} = require("../frontend/src/lib/continuous.js");

function jsonResponse(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

async function withFetch(fetchImpl, fn) {
  const original = globalThis.fetch;
  globalThis.fetch = fetchImpl;
  try {
    return await fn();
  } finally {
    if (original) {
      globalThis.fetch = original;
    } else {
      delete globalThis.fetch;
    }
  }
}

function window(kb, mouse, startedAt = "2025-01-01T00:00:00.000Z", endedAt = null, extra = {}) {
  return {
    session_id: "w1",
    started_at: startedAt,
    ended_at: endedAt,
    timestamp_source: "test",
    keyboard_events: Array.from({ length: kb }, () => ({})),
    mouse_events: Array.from({ length: mouse }, () => ({})),
    ...extra,
  };
}

/* ------------------------- window bounds --------------------------- */

test("42. continuous window limits match the bounded-window contract", () => {
  assert.deepEqual(WINDOW_LIMITS, {
    MAX_KEYBOARD_EVENTS: 300,
    MAX_MOUSE_EVENTS: 1200,
    MAX_TOTAL_EVENTS: 1500,
    MAX_DURATION_MS: 30000,
  });
  assert.equal(MIN_WINDOW_KEYBOARD_EVENTS, 2);
  assert.equal(MIN_WINDOW_MOUSE_EVENTS, 2);
});

test("43. windowEventCounts counts keyboard and mouse arrays safely", () => {
  assert.deepEqual(windowEventCounts(null), { keyboard: 0, mouse: 0 });
  assert.deepEqual(windowEventCounts({}), { keyboard: 0, mouse: 0 });
  const w = window(5, 7);
  assert.deepEqual(windowEventCounts(w), { keyboard: 5, mouse: 7 });
  assert.equal(windowTotalEvents(w), 12);
});

test("44. windowDurationMs uses ended_at when closed and atMs when open", () => {
  const closed = window(2, 2, "2025-01-01T00:00:00.000Z", "2025-01-01T00:00:10.000Z");
  assert.equal(windowDurationMs(closed), 10000);
  const open = window(2, 2, "2025-01-01T00:00:00.000Z");
  assert.equal(windowDurationMs(open, new Date("2025-01-01T00:00:40.000Z").getTime()), 40000);
  assert.equal(windowDurationMs(ww({}), 123), 0);
  assert.equal(windowDurationMs(null, 123), 0);
  assert.equal(windowDurationMs(open, NaN), 0);
});

function ww(s) {
  return s;
}

test("45. keyboard cap stops collection first", () => {
  const w = window(WINDOW_LIMITS.MAX_KEYBOARD_EVENTS, 1);
  assert.equal(windowFullReason(w), "keyboard");
  assert.equal(shouldStopCollecting(w), true);
});

test("46. mouse cap stops collection", () => {
  const w = window(1, WINDOW_LIMITS.MAX_MOUSE_EVENTS);
  assert.equal(windowFullReason(w), "mouse");
});

test("47. total cap stops collection", () => {
  const w = window(600, 900);
  assert.equal(windowFullReason(w), "total");
});

test("48. duration cap stops collection", () => {
  const w = window(5, 5, "2025-01-01T00:00:00.000Z");
  assert.equal(windowFullReason(w, { atMs: new Date("2025-01-01T00:00:30.000Z").getTime() }), "duration");
});

test("49. a window inside every bound reports no reason", () => {
  const w = window(4, 4, "2025-01-01T00:00:00.000Z");
  assert.equal(windowFullReason(w, { atMs: new Date("2025-01-01T00:00:05.000Z").getTime() }), null);
  assert.equal(shouldStopCollecting(w, { atMs: new Date("2025-01-01T00:00:05.000Z").getTime() }), false);
});

test("50. canSubmitWindow requires usable content and no cap breach", () => {
  const start = "2025-01-01T00:00:00.000Z";
  const at = new Date("2025-01-01T00:00:05.000Z").getTime();
  assert.deepEqual(canSubmitWindow(window(1, 1, start), { atMs: at }), { can: false, reason: "no-usable-content" });
  assert.deepEqual(canSubmitWindow(window(2, 2, start), { atMs: at }), { can: true, reason: null });
  assert.deepEqual(canSubmitWindow(window(2, 2, start), {
    atMs: new Date("2025-01-01T00:00:31.000Z").getTime(),
  }), { can: false, reason: "duration" });
  assert.equal(windowHasUsableContent(window(2, 2, start)), true);
  assert.equal(windowHasUsableContent(window(1, 2, start)), false);
});

test("51. buildContinuousWindow is a bare deep copy with no identity fields", () => {
  const source = window(2, 2);
  const built = buildContinuousWindow(source);
  assert.notEqual(built, source);
  assert.deepEqual(built, source);
  assert.equal("user_ref" in built, false);
  assert.deepEqual(Object.keys(built).sort(), [
    "ended_at",
    "keyboard_events",
    "mouse_events",
    "session_id",
    "started_at",
    "timestamp_source",
  ]);
});

/* --------------------------- state machine -------------------------- */

test("52. reducer starts authenticated with an empty machine", () => {
  const initial = initialState();
  assert.deepEqual(initial, { state: "authenticated", last: null, window: null });
  assert.equal(continuousReducer(initial, null), initial);
  assert.equal(continuousReducer(initial, { type: "nope" }), initial);
  assert.deepEqual(continuousReducer(initial, { type: "START" }), {
    state: "collecting",
    last: null,
    window: null,
  });
});

test("53. START is ignored while collecting or verifying", () => {
  const initial = initialState();
  const collecting = continuousReducer(initial, { type: "START" });
  assert.equal(continuousReducer(collecting, { type: "START" }), collecting);
  const verifying = continuousReducer(collecting, { type: "VERIFY", window: window(2, 2) });
  assert.equal(continuousReducer(verifying, { type: "START" }), verifying);
});

test("54. VERIFY is allowed only from collecting", () => {
  const initial = initialState();
  assert.equal(continuousReducer(initial, { type: "VERIFY", window: window(2, 2) }), initial);
  const collecting = continuousReducer(initial, { type: "START" });
  const w = window(2, 2);
  const verifying = continuousReducer(collecting, { type: "VERIFY", window: w });
  assert.deepEqual(verifying, { state: "verifying", last: null, window: w });
  assert.equal(continuousReducer(collecting, { type: "VERIFY" }), collecting);
});

test("55. DECISION VERIFIED lands on verified and keeps the decision", () => {
  const initial = initialState();
  const w = window(2, 2);
  const verifying = continuousReducer(continuousReducer(initial, { type: "START" }), {
    type: "VERIFY",
    window: w,
  });
  const outcome = continuousReducer(verifying, {
    type: "DECISION",
    decision: CONTINUOUS_DECISIONS.VERIFIED,
    distance: 0.12,
    threshold: 0.463513,
  });
  assert.equal(outcome.state, "verified");
  assert.deepEqual(outcome.last, {
    decision: "VERIFIED",
    distance: 0.12,
    threshold: 0.463513,
  });
  assert.equal(outcome.window, w);
});

test("56. DECISION SUSPICIOUS requests re-verification", () => {
  const initial = initialState();
  const verifying = continuousReducer(continuousReducer(initial, { type: "START" }), {
    type: "VERIFY",
    window: window(2, 2),
  });
  const outcome = continuousReducer(verifying, {
    type: "DECISION",
    decision: CONTINUOUS_DECISIONS.SUSPICIOUS,
    distance: 1.3037,
    threshold: 0.463513,
  });
  assert.equal(outcome.state, "reverification_required");
  assert.equal(outcome.last.decision, "SUSPICIOUS");
});

test("57. DECISION is ignored outside verifying and on unknown values", () => {
  const initial = initialState();
  assert.equal(continuousReducer(initial, { type: "DECISION", decision: "VERIFIED" }), initial);
  const collecting = continuousReducer(initial, { type: "START" });
  assert.equal(continuousReducer(collecting, { type: "DECISION", decision: "VERIFIED" }), collecting);
  const w = window(2, 2);
  const verifying = continuousReducer(collecting, { type: "VERIFY", window: w });
  assert.equal(continuousReducer(verifying, { type: "DECISION", decision: "MAYBE" }), verifying);
});

test("58. RESET returns to authenticated and keeps the last decision", () => {
  const initial = initialState();
  const w = window(2, 2);
  const verified = continuousReducer(
    continuousReducer(continuousReducer(initial, { type: "START" }), { type: "VERIFY", window: w }),
    { type: "DECISION", decision: "VERIFIED", distance: 0.1, threshold: 0.463513 }
  );
  const reset = continuousReducer(verified, { type: "RESET" });
  assert.deepEqual(reset, {
    state: "authenticated",
    last: { decision: "VERIFIED", distance: 0.1, threshold: 0.463513 },
    window: null,
  });
});

test("59. START is allowed again after a decision", () => {
  const initial = initialState();
  const verified = continuousReducer(
    continuousReducer(continuousReducer(initial, { type: "START" }), {
      type: "VERIFY",
      window: window(2, 2),
    }),
    { type: "DECISION", decision: "VERIFIED", distance: 0.1, threshold: 0.463513 }
  );
  assert.equal(continuousReducer(verified, { type: "START" }).state, "collecting");
});

/* --------------------------- API client ---------------------------- */

test("60. continuousVerify without a token is rejected client-side", async () => {
  await withFetch(async () => {
    throw new Error("fetch should not be called");
  }, async () => {
    const result = await continuousVerify({ session: {}, token: null });
    assert.equal(result.ok, false);
    assert.equal(result.status, 401);
    assert.equal(result.code, "missing_token");
  });
});

test("61. continuousVerify posts the bare window with bearer auth", async () => {
  const w = window(2, 2, "2025-01-01T00:00:00.000Z", "2025-01-01T00:00:05.000Z");
  await withFetch(async (url, init) => {
    assert.equal(url.endsWith("/continuous-verification"), true);
    assert.equal(init.method, "POST");
    assert.equal(init.headers.Authorization, "Bearer jwt-abc");
    const body = JSON.parse(init.body);
    assert.equal(body.session_id, "w1");
    assert.equal("user_ref" in body, false);
    assert.equal("sessions" in body, false);
    return jsonResponse(200, {
      decision: "VERIFIED",
      distance: 0.1024,
      threshold: 0.463513,
    });
  }, async () => {
    const result = await continuousVerify({ session: w, token: "jwt-abc" });
    assert.equal(result.ok, true);
    assert.deepEqual(result.result, {
      decision: "VERIFIED",
      distance: 0.1024,
      threshold: 0.463513,
    });
  });
});

/* ---------------------- Phase 14B: server session state -------------------- */

function verifiedWindow() {
  return continuousReducer(continuousReducer(initialState(), { type: "START" }), {
    type: "VERIFY",
    window: window(2, 2),
  });
}

test("62. server session state constants mirror the backend", () => {
  assert.deepEqual(SESSION_STATES, {
    VERIFIED: "verified",
    REVERIFICATION_REQUIRED: "reverification_required",
  });
  assert.equal(sessionDecisionToState("VERIFIED"), "verified");
  assert.equal(sessionDecisionToState("SUSPICIOUS"), "reverification_required");
  assert.equal(sessionDecisionToState("MAYBE"), null);
  assert.equal(restoreServerState("verified"), "verified");
  assert.equal(restoreServerState("reverification_required"), "reverification_required");
  assert.equal(restoreServerState("expired"), null);
  assert.equal(restoreServerState(undefined), null);
});

test("63. DECISION trusts the server session_state over the decision", () => {
  const outcome = continuousReducer(verifiedWindow(), {
    type: "DECISION",
    decision: "SUSPICIOUS",
    distance: 1.0,
    threshold: 0.463513,
    sessionState: "verified",
    consecutiveSuspicious: 0,
    lastVerifiedAt: "2026-01-01T00:00:00Z",
  });
  assert.equal(outcome.state, "verified");
  assert.deepEqual(outcome.last, {
    decision: "SUSPICIOUS",
    distance: 1.0,
    threshold: 0.463513,
    sessionState: "verified",
    consecutiveSuspicious: 0,
    lastVerifiedAt: "2026-01-01T00:00:00Z",
  });
});

test("64. DECISION falls back to the decision without a server session_state", () => {
  const outcome = continuousReducer(verifiedWindow(), {
    type: "DECISION",
    decision: "SUSPICIOUS",
    distance: 1.0,
    threshold: 0.463513,
  });
  assert.equal(outcome.state, "reverification_required");
  assert.deepEqual(outcome.last, {
    decision: "SUSPICIOUS",
    distance: 1.0,
    threshold: 0.463513,
  });
});

test("65. RESTORE hydrates from the server and never fabricates state", () => {
  const initial = initialState();
  const verified = continuousReducer(initial, {
    type: "RESTORE",
    sessionState: "verified",
    consecutiveSuspicious: 0,
    lastVerifiedAt: null,
  });
  assert.equal(verified.state, "verified");
  assert.deepEqual(verified.last, {
    decision: "VERIFIED",
    distance: null,
    threshold: null,
    sessionState: "verified",
    consecutiveSuspicious: 0,
    lastVerifiedAt: null,
  });
  assert.equal(verified.window, null);
  const recheck = continuousReducer(initial, {
    type: "RESTORE",
    sessionState: "reverification_required",
    consecutiveSuspicious: 2,
    lastVerifiedAt: "2026-01-01T00:00:00Z",
  });
  assert.equal(recheck.state, "reverification_required");
  assert.equal(recheck.last.decision, "SUSPICIOUS");
  assert.equal(recheck.last.consecutiveSuspicious, 2);
});

test("66. RESTORE is ignored for unknown states and outside authenticated", () => {
  const initial = initialState();
  assert.equal(continuousReducer(initial, { type: "RESTORE", sessionState: "expired" }), initial);
  assert.equal(continuousReducer(initial, { type: "RESTORE", sessionState: "MAYBE" }), initial);
  const collecting = continuousReducer(initial, { type: "START" });
  assert.equal(
    continuousReducer(collecting, { type: "RESTORE", sessionState: "verified" }),
    collecting
  );
  const verified = continuousReducer(initial, {
    type: "RESTORE",
    sessionState: "verified",
  });
  assert.equal(continuousReducer(verified, { type: "RESTORE", sessionState: "reverification_required" }), verified);
});

test("67. getContinuousSessionState requires a token", async () => {
  await withFetch(async () => {
    throw new Error("fetch should not be called");
  }, async () => {
    const result = await getContinuousSessionState({ token: null });
    assert.equal(result.ok, false);
    assert.equal(result.status, 401);
    assert.equal(result.code, "missing_token");
  });
});

test("68. getContinuousSessionState GETs the state endpoint with bearer auth", async () => {
  await withFetch(async (url, init) => {
    assert.equal(url.endsWith("/continuous-verification/state"), true);
    assert.equal(init.method, "GET");
    assert.equal(init.headers.Authorization, "Bearer jwt-abc");
    return jsonResponse(200, {
      session_state: "reverification_required",
      consecutive_suspicious: 3,
      last_verified_at: null,
    });
  }, async () => {
    const result = await getContinuousSessionState({ token: "jwt-abc" });
    assert.equal(result.ok, true);
    assert.deepEqual(result.result, {
      session_state: "reverification_required",
      consecutive_suspicious: 3,
      last_verified_at: null,
    });
  });
});

test("69. getContinuousSessionState surfaces failure status", async () => {
  await withFetch(async () => jsonResponse(503, { error: { code: "database_unavailable" } }), async () => {
    const result = await getContinuousSessionState({ token: "jwt-abc" });
    assert.equal(result.ok, false);
    assert.equal(result.status, 503);
    assert.equal(result.code, "database_unavailable");
  });
});

test("70. a time-capped 160/37 stopped window is intentionally not submittable", () => {
  // Exact manual-capture condition: auto-stopped at the 30 s time cap with
  // 160 keyboard and 37 mouse events. ended_at equals started_at + 30 s.
  const start = "2025-01-01T00:00:00.000Z";
  const capped = window(160, 37, start, "2025-01-01T00:00:30.000Z");
  assert.deepEqual(windowDurationMs(capped), 30000);
  assert.equal(windowFullReason(capped), "duration");
  assert.deepEqual(canSubmitWindow(capped), { can: false, reason: "duration" });

  // The same counts are submittable ONLY when the user stops the window while
  // still inside every bound (the designed manual action).
  const closedOnTime = window(160, 37, start, "2025-01-01T00:00:10.000Z");
  assert.deepEqual(canSubmitWindow(closedOnTime), { can: true, reason: null });
});

/* ------------- Phase 14B: frontend state restoration (hook/no-HTML) ---- */

function restoredState(serverState, consecutiveSuspicious, lastVerifiedAt) {
  return {
    session_state: serverState,
    consecutive_suspicious: consecutiveSuspicious,
    last_verified_at: lastVerifiedAt,
  };
}

test("71. server state verified → the UI machine restores to verified", async () => {
  const coordinator = createRestoreCoordinator();
  let restored = null;
  const dispose = restoreSessionState(coordinator, {
    token: "jwt-abc",
    fetchState: async () => ({ ok: true, result: restoredState("verified", 0, null) }),
    onRestored(result) {
      restored = result;
    },
  });
  assert.equal(typeof dispose, "function");
  await new Promise((resolve) => setImmediate(resolve));

  assert.deepEqual(restored, restoredState("verified", 0, null));
  const machine = continuousReducer(initialState(), {
    type: "RESTORE",
    sessionState: restored.session_state,
    consecutiveSuspicious: restored.consecutive_suspicious,
    lastVerifiedAt: restored.last_verified_at,
  });
  assert.equal(machine.state, CONTINUOUS_STATES.VERIFIED);
  assert.equal(machine.last.decision, CONTINUOUS_DECISIONS.VERIFIED);
});

test("72. server state reverification_required → the UI machine restores to re-verification-required", async () => {
  const coordinator = createRestoreCoordinator();
  let restored = null;
  restoreSessionState(coordinator, {
    token: "jwt-abc",
    fetchState: async () => ({
      ok: true,
      result: restoredState("reverification_required", 1, null),
    }),
    onRestored(result) {
      restored = result;
    },
  });
  await new Promise((resolve) => setImmediate(resolve));

  assert.deepEqual(restored, restoredState("reverification_required", 1, null));
  const machine = continuousReducer(initialState(), {
    type: "RESTORE",
    sessionState: restored.session_state,
    consecutiveSuspicious: restored.consecutive_suspicious,
    lastVerifiedAt: restored.last_verified_at,
  });
  assert.equal(machine.state, CONTINUOUS_STATES.REVERIFICATION_REQUIRED);
  assert.equal(machine.last.decision, CONTINUOUS_DECISIONS.SUSPICIOUS);
  assert.equal(machine.last.consecutiveSuspicious, 1);
});

test("73. StrictMode setup→cleanup→setup cannot swallow a reverification_required restore", async () => {
  const coordinator = createRestoreCoordinator();
  const resolvers = [];
  const fetchState = ({ token }) =>
    new Promise((resolve) => {
      resolvers.push({ token, resolve });
    });

  let restored = null;
  const disposeFirst = restoreSessionState(coordinator, {
    token: "jwt-abc",
    fetchState,
    onRestored(result) {
      restored = result;
    },
  });
  // The first attempt is burned by its own StrictMode cleanup BEFORE the
  // server answers...
  disposeFirst();
  // ...and the retried effect issues a fresh request that must still pass.
  const disposeSecond = restoreSessionState(coordinator, {
    token: "jwt-abc",
    fetchState,
    onRestored(result) {
      restored = result;
    },
  });
  assert.equal(resolvers.length, 2);

  resolvers[0].resolve({
    ok: true,
    result: restoredState("reverification_required", 1, null),
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(restored, null, "the cancelled attempt must not restore");

  resolvers[1].resolve({
    ok: true,
    result: restoredState("reverification_required", 1, null),
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(restored, restoredState("reverification_required", 1, null));

  // A completed restore never runs twice per token: no fetch, no dispatch.
  let secondRestore = false;
  restoreSessionState(coordinator, {
    token: "jwt-abc",
    fetchState,
    onRestored() {
      secondRestore = true;
    },
  });
  assert.equal(resolvers.length, 2);
  assert.equal(secondRestore, false);

  disposeSecond();
});

test("74. a fresh mount (refresh) re-reads the server state instead of the previous verdict", async () => {
  // First mount reads `verified`.
  const firstMount = createRestoreCoordinator();
  let firstRestored = null;
  restoreSessionState(firstMount, {
    token: "jwt-abc",
    fetchState: async () => ({ ok: true, result: restoredState("verified", 0, null) }),
    onRestored(result) {
      firstRestored = result;
    },
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(firstRestored, restoredState("verified", 0, null));

  // Refresh/remount creates a fresh coordinator; a brand-new coordinator must
  // re-query and land the server's NEWER verdict (reverification_required).
  const remount = createRestoreCoordinator();
  let secondRestored = null;
  restoreSessionState(remount, {
    token: "jwt-abc",
    fetchState: async () => ({
      ok: true,
      result: restoredState("reverification_required", 1, null),
    }),
    onRestored(result) {
      secondRestored = result;
    },
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(secondRestored, restoredState("reverification_required", 1, null));
});

test("75. restore failure paths leave the machine untouched; 401 routes to logout", async () => {
  const coordinator = createRestoreCoordinator();
  let restored = false;
  let loggedOut = false;

  // Transient failure: default authenticated machine is left untouched.
  restoreSessionState(coordinator, {
    token: "jwt-abc",
    fetchState: async () => ({ ok: false, status: 503, code: "database_unavailable" }),
    onRestored() {
      restored = true;
    },
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(restored, false);

  // 401: the session is logged out, nothing is restored.
  restoreSessionState(coordinator, {
    token: "jwt-rew",
    fetchState: async () => ({ ok: false, status: 401, code: null }),
    onRestored() {
      restored = true;
    },
    onUnauthorized() {
      loggedOut = true;
    },
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(restored, false);
  assert.equal(loggedOut, true);
  // No completed marker was written, so a later retry can still restore.
  assert.equal(coordinator.completedToken, null);
});