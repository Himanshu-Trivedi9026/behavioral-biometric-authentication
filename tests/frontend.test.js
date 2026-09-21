/*
 * Behavioral Biometric Authentication — Phase 12
 * Frontend integration tests (auth store, API client, payloads, privacy,
 * enrollment session limit).
 *
 * Run with: npm test (node --test ../tests/collector.test.js ../tests/frontend.test.js)
 */
"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");

const { createSessionStore, STORAGE_KEY } = require("../frontend/src/auth.js");
const {
  registerAccount,
  loginAccount,
  enrollSessions,
  verifySession,
} = require("../frontend/src/api.js");
const {
  MAX_ENROLLMENT_SESSIONS,
  hasUsableContent,
  hasReachedMaxSessions,
  canAddEnrollmentSession,
  collectEnrollmentSession,
  buildEnrollmentPayload,
  buildVerificationPayload,
} = require("../frontend/src/lib/enrollment.js");
const { BehavioralSession } = require("../frontend/src/lib/collector.js");

function memoryStorage() {
  const map = new Map();
  return {
    getItem(key) {
      return map.has(key) ? map.get(key) : null;
    },
    setItem(key, value) {
      map.set(key, value);
    },
    removeItem(key) {
      map.delete(key);
    },
  };
}

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

/* --------------------------- auth store ----------------------------- */

test("20. auth store login exposes token, username and flags", () => {
  const store = createSessionStore(null);
  store.logIn({ token: "tok123", username: "alice" });
  assert.equal(store.isAuthenticated(), true);
  assert.equal(store.getAccessToken(), "tok123");
  assert.equal(store.getUsername(), "alice");
  assert.deepEqual(store.getState(), { accessToken: "tok123", username: "alice" });
});

test("21. auth store ignores empty token on login", () => {
  const store = createSessionStore(null);
  store.logIn({ token: "", username: "alice" });
  assert.equal(store.isAuthenticated(), false);
  store.logIn({ token: null, username: "alice" });
  assert.equal(store.isAuthenticated(), false);
});

test("22. auth store logout clears the session", () => {
  const store = createSessionStore(null);
  store.logIn({ token: "tok123", username: "alice" });
  store.logOut();
  assert.equal(store.isAuthenticated(), false);
  assert.equal(store.getAccessToken(), null);
});

test("23. auth store restores a session from storage (page refresh)", () => {
  const storage = memoryStorage();
  const first = createSessionStore(storage);
  first.logIn({ token: "tok456", username: "bob" });
  const second = createSessionStore(storage);
  assert.equal(second.isAuthenticated(), true);
  assert.deepEqual(second.getState(), { accessToken: "tok456", username: "bob" });
});

test("24. auth store never persists a password", () => {
  const storage = memoryStorage();
  const store = createSessionStore(storage);
  store.logIn({ token: "tok789", username: "carol" });
  const raw = storage.getItem(STORAGE_KEY);
  assert.ok(raw);
  const parsed = JSON.parse(raw);
  assert.equal("password" in parsed, false);
});

test("25. auth store emits to subscribers on change", () => {
  const store = createSessionStore(null);
  let seen = null;
  store.subscribe(() => {
    seen = store.getSnapshot();
  });
  store.logIn({ token: "tok", username: "eve" });
  assert.deepEqual(seen, { accessToken: "tok", username: "eve" });
});

/* --------------------------- API client ----------------------------- */

test("26. registerAccount posts credentials and returns the user", async () => {
  await withFetch(async (url, init) => {
    assert.equal(url.endsWith("/auth/register"), true);
    assert.equal(init.method, "POST");
    assert.equal(init.headers["Content-Type"], "application/json");
    assert.deepEqual(JSON.parse(init.body), { username: "alice", password: "secret123" });
    return jsonResponse(201, { id: "u1", username: "alice", created_at: "2025-01-01" });
  }, async () => {
    const result = await registerAccount({ username: "alice", password: "secret123" });
    assert.equal(result.ok, true);
    assert.equal(result.user.username, "alice");
  });
});

test("27. API client maps structured error envelopes", async () => {
  await withFetch(async () => {
    return jsonResponse(409, {
      error: { code: "username_exists", message: "username already taken" },
    });
  }, async () => {
    const result = await registerAccount({ username: "alice", password: "secret123" });
    assert.equal(result.ok, false);
    assert.equal(result.status, 409);
    assert.equal(result.code, "username_exists");
    assert.equal(result.message, "username already taken");
  });
});

test("28. loginAccount returns the access token", async () => {
  await withFetch(async () => {
    return jsonResponse(200, { access_token: "jwt-abc", token_type: "bearer" });
  }, async () => {
    const result = await loginAccount({ username: "alice", password: "secret123" });
    assert.equal(result.ok, true);
    assert.equal(result.token, "jwt-abc");
    assert.equal(result.tokenType, "bearer");
  });
});

test("29. loginAccount fails when the token is absent", async () => {
  await withFetch(async () => {
    return jsonResponse(200, { token_type: "bearer" });
  }, async () => {
    const result = await loginAccount({ username: "alice", password: "secret123" });
    assert.equal(result.ok, false);
    assert.equal(result.status, 502);
  });
});

test("30. network failure surfaces status 0", async () => {
  await withFetch(async () => {
    throw new TypeError("fetch failed");
  }, async () => {
    const result = await registerAccount({ username: "alice", password: "secret123" });
    assert.equal(result.ok, false);
    assert.equal(result.status, 0);
  });
});

test("31. aborted request surfaces code timeout", async () => {
  await withFetch(async () => {
    const error = new Error("aborted");
    error.name = "AbortError";
    throw error;
  }, async () => {
    const result = await verifySession({ session: {}, token: "tok" });
    assert.equal(result.ok, false);
    assert.equal(result.status, 0);
    assert.equal(result.code, "timeout");
  });
});

test("32. enrollSessions without a token is rejected client-side", async () => {
  await withFetch(async () => {
    throw new Error("fetch should not be called");
  }, async () => {
    const result = await enrollSessions({ sessions: [], token: null });
    assert.equal(result.ok, false);
    assert.equal(result.status, 401);
    assert.equal(result.code, "missing_token");
  });
});

test("33. enrollSessions sends sessions with bearer auth", async () => {
  await withFetch(async (url, init) => {
    assert.equal(url.endsWith("/enrollment"), true);
    assert.equal(init.method, "POST");
    assert.equal(init.headers.Authorization, "Bearer jwt-abc");
    const body = JSON.parse(init.body);
    assert.equal(body.sessions.length, 1);
    assert.equal(body.sessions[0].session_id, "sess-1");
    return jsonResponse(201, {
      user_ref: "alice",
      status: "enrolled",
      embedding_dimension: 128,
      session_count: 1,
    });
  }, async () => {
    const result = await enrollSessions({
      sessions: [{ session_id: "sess-1" }],
      token: "jwt-abc",
    });
    assert.equal(result.ok, true);
    assert.equal(result.result.status, "enrolled");
    assert.equal(result.result.embedding_dimension, 128);
  });
});

test("34. verifySession sends the probe with bearer auth", async () => {
  await withFetch(async (url, init) => {
    assert.equal(url.endsWith("/verification"), true);
    assert.equal(init.headers.Authorization, "Bearer jwt-abc");
    const body = JSON.parse(init.body);
    assert.equal(body.session.session_id, "probe-1");
    return jsonResponse(200, {
      user_ref: "alice",
      decision: "VERIFIED",
      distance: 0.1024,
      threshold: 0.463513,
    });
  }, async () => {
    const result = await verifySession({
      session: { session_id: "probe-1" },
      token: "jwt-abc",
    });
    assert.equal(result.ok, true);
    assert.equal(result.result.decision, "VERIFIED");
    assert.equal(result.result.distance, 0.1024);
  });
});

/* ---------------------- payloads + privacy -------------------------- */

test("35. payload builders deep-copy without adding fields", () => {
  const session = { session_id: "s1", keyboard_events: [], mouse_events: [] };
  const enrolled = buildEnrollmentPayload([session, { session_id: "s2", keyboard_events: [], mouse_events: [] }]);
  assert.deepEqual(enrolled, {
    sessions: [
      { session_id: "s1", keyboard_events: [], mouse_events: [] },
      { session_id: "s2", keyboard_events: [], mouse_events: [] },
    ],
  });
  const probe = buildVerificationPayload(session);
  assert.deepEqual(probe, { session });
});

test("36. captured sessions carry no key identity", () => {
  const s = new BehavioralSession();
  assert.equal(s.start(), true);
  s.addKeyboardEvent("keydown", 100);
  s.addKeyboardEvent("keyup", 150);
  s.addMouseEvent("mousemove", 10, 20, 200);
  s.addMouseEvent("mousedown", 30, 40, 300);
  assert.equal(s.stop(), true);
  const session = s.getSession();
  assert.ok(session);
  for (const event of session.keyboard_events) {
    assert.deepEqual(Object.keys(event).sort(), ["event", "event_type", "timestamp"]);
    assert.equal("key" in event, false);
    assert.equal("code" in event, false);
    assert.equal("value" in event, false);
  }
  for (const event of session.mouse_events) {
    assert.deepEqual(Object.keys(event).sort(), ["event", "event_type", "timestamp", "x", "y"]);
    assert.equal("type" in event, false);
  }
});

test("37. hasUsableContent requires keyboard and mouse activity", () => {
  const counts = (kb, mouse) =>
    hasUsableContent({
      keyboard_events: Array.from({ length: kb }),
      mouse_events: Array.from({ length: mouse }),
    });
  assert.equal(counts(2, 2), true);
  assert.equal(counts(1, 2), false);
  assert.equal(counts(2, 1), false);
  assert.equal(counts(0, 0), false);
  assert.equal(hasUsableContent(null), false);
});

/* ----------------------- max session limit -------------------------- */

function usableSession(id) {
  return {
    session_id: id,
    keyboard_events: [{}, {}],
    mouse_events: [{}, {}],
  };
}

test("38. enrollment logic recognizes the 16-session maximum", () => {
  assert.equal(MAX_ENROLLMENT_SESSIONS, 16);
  assert.equal(hasReachedMaxSessions(15), false);
  assert.equal(hasReachedMaxSessions(16), true);
  assert.equal(hasReachedMaxSessions(17), true);
  assert.equal(canAddEnrollmentSession(15), true);
  assert.equal(canAddEnrollmentSession(16), false);
  assert.equal(canAddEnrollmentSession(17), false);
});

test("39. sessions accumulate up to the maximum of 16", () => {
  let sessions = [];
  for (let i = 1; i <= MAX_ENROLLMENT_SESSIONS; i += 1) {
    const outcome = collectEnrollmentSession(sessions, usableSession("s" + i));
    assert.equal(outcome.added, true, "session " + i + " should be added");
    assert.equal(outcome.reason, null);
    sessions = outcome.sessions;
    assert.equal(sessions.length, i);
  }
  assert.equal(sessions.length, 16);
});

test("40. a 17th session cannot be added", () => {
  let sessions = [];
  for (let i = 1; i <= MAX_ENROLLMENT_SESSIONS; i += 1) {
    sessions = collectEnrollmentSession(sessions, usableSession("s" + i)).sessions;
  }
  assert.equal(sessions.length, 16);
  const outcome = collectEnrollmentSession(sessions, usableSession("s17"));
  assert.equal(outcome.added, false);
  assert.equal(outcome.reason, "max-sessions");
  assert.equal(outcome.sessions, sessions, "the session list must not change");
});

test("41. collectEnrollmentSession preserves existing validation behavior", () => {
  const nullOutcome = collectEnrollmentSession([], null);
  assert.deepEqual(nullOutcome, { sessions: [], added: false, reason: "no-usable-content" });
  const thin = usableSession("thin");
  thin.keyboard_events = [{}];
  const thinOutcome = collectEnrollmentSession([], thin);
  assert.equal(thinOutcome.added, false);
  assert.equal(thinOutcome.reason, "no-usable-content");
  assert.deepEqual(thinOutcome.sessions, []);
  assert.deepEqual(collectEnrollmentSession(undefined, null).sessions, []);
  const one = collectEnrollmentSession([], usableSession("a")).sessions;
  const two = collectEnrollmentSession(one, usableSession("b"));
  assert.deepEqual(two.sessions.map((s) => s.session_id), ["a", "b"]);
});