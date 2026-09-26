/*
 * Behavioral Biometric Authentication — Phase 16A
 * Local-only enrollment session export: export document, filename, the
 * zero-session no-op, the browser download, the absence of any backend call,
 * privacy, and the unchanged 16-session cap.
 *
 * Run with: npm test
 */
"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const {
  EXPORT_VERSION,
  EXPORT_FILENAME_PREFIX,
  canExportSessions,
  exportTimestamp,
  buildExportFilename,
  buildSessionExport,
  downloadJsonFile,
  exportSessionsLocally,
} = require("../frontend/src/lib/sessionExport.js");
const {
  MAX_ENROLLMENT_SESSIONS,
  hasReachedMaxSessions,
  collectEnrollmentSession,
} = require("../frontend/src/lib/enrollment.js");
const { BehavioralSession } = require("../frontend/src/lib/collector.js");

const PAGE_SOURCE = fs.readFileSync(
  path.join(__dirname, "..", "frontend", "src", "enrollment", "EnrollmentPage.jsx"),
  "utf8"
);

const STAMP = "2026-09-26T11:22:33.456Z";
const STAMP_FILE = "2026-09-26T11-22-33-456Z";

/* Every key the collector may legitimately emit, at any nesting depth. */
const ALLOWED_KEYS = new Set([
  "export_version",
  "exported_at",
  "session_count",
  "sessions",
  "session_id",
  "started_at",
  "ended_at",
  "timestamp_source",
  "keyboard_events",
  "mouse_events",
  "event_type",
  "event",
  "timestamp",
  "x",
  "y",
]);

/* Anything that would be a privacy or identity leak if it ever appeared. */
const FORBIDDEN_KEYS = new Set([
  "username",
  "user",
  "user_ref",
  "userRef",
  "email",
  "password",
  "token",
  "access_token",
  "accessToken",
  "jwt",
  "key",
  "keys",
  "code",
  "value",
  "text",
  "typed_text",
  "content",
  "input",
  "clipboard",
  "cookie",
  "cookies",
  "dom",
  "html",
  "screenshot",
  "target",
  "isTrusted",
]);

function collectKeys(node, found) {
  if (Array.isArray(node)) {
    node.forEach((item) => collectKeys(item, found));
    return found;
  }
  if (node && typeof node === "object") {
    Object.keys(node).forEach((key) => {
      found.add(key);
      collectKeys(node[key], found);
    });
  }
  return found;
}

/* Minimal document stand-in: records the synthetic anchor and its click. */
function fakeDocument() {
  const created = [];
  const appended = [];
  const removed = [];
  return {
    created,
    appended,
    removed,
    body: {
      appendChild(node) {
        appended.push(node);
      },
      removeChild(node) {
        removed.push(node);
      },
    },
    createElement(tag) {
      const node = {
        tag,
        href: null,
        download: null,
        rel: null,
        clicks: 0,
        click() {
          node.clicks += 1;
        },
      };
      created.push(node);
      return node;
    },
  };
}

function withDocument(doc, fn) {
  const had = Object.prototype.hasOwnProperty.call(globalThis, "document");
  const previous = globalThis.document;
  globalThis.document = doc;
  try {
    return fn();
  } finally {
    if (had) {
      globalThis.document = previous;
    } else {
      delete globalThis.document;
    }
  }
}

/* Any network attempt at all fails the test loudly. */
async function withForbiddenNetwork(fn) {
  const originalFetch = globalThis.fetch;
  const originalXhr = globalThis.XMLHttpRequest;
  const originalSendBeacon =
    typeof globalThis.navigator !== "undefined" ? globalThis.navigator.sendBeacon : undefined;
  const calls = [];
  globalThis.fetch = (...args) => {
    calls.push(args);
    throw new Error("export must not perform a network request");
  };
  globalThis.XMLHttpRequest = function () {
    calls.push(["XMLHttpRequest"]);
    throw new Error("export must not perform a network request");
  };
  if (globalThis.navigator) {
    Object.defineProperty(globalThis.navigator, "sendBeacon", {
      configurable: true,
      value: () => {
        calls.push(["sendBeacon"]);
        return false;
      },
    });
  }
  try {
    return await fn(calls);
  } finally {
    if (originalFetch) {
      globalThis.fetch = originalFetch;
    } else {
      delete globalThis.fetch;
    }
    if (originalXhr) {
      globalThis.XMLHttpRequest = originalXhr;
    } else {
      delete globalThis.XMLHttpRequest;
    }
    if (globalThis.navigator && originalSendBeacon) {
      Object.defineProperty(globalThis.navigator, "sendBeacon", {
        configurable: true,
        value: originalSendBeacon,
      });
    }
  }
}

function withSilentConsole(fn) {
  const written = [];
  const originals = {};
  ["log", "info", "warn", "error", "debug", "trace"].forEach((level) => {
    originals[level] = console[level];
    console[level] = (...args) => written.push([level, args]);
  });
  try {
    fn();
  } finally {
    Object.keys(originals).forEach((level) => {
      console[level] = originals[level];
    });
  }
  return written;
}

/* A real collector session, so exports are verified against genuine data. */
function realSession(index) {
  const session = new BehavioralSession();
  assert.equal(session.start(), true);
  const base = 1000 * index;
  session.addKeyboardEvent("keydown", base + 10.5);
  session.addKeyboardEvent("keyup", base + 60.25);
  session.addKeyboardEvent("keydown", base + 120);
  session.addMouseEvent("mousemove", 10 + index, 20 + index, base + 30);
  session.addMouseEvent("mousemove", 40 + index, 80 + index, base + 90);
  session.addMouseEvent("mousedown", 55 + index, 61 + index, base + 140);
  session.addMouseEvent("mouseup", 56 + index, 62 + index, base + 200);
  assert.equal(session.stop(), true);
  return session.getSession();
}

function usableSession(id) {
  return {
    session_id: id,
    started_at: "2026-09-26T10:00:00.000Z",
    ended_at: "2026-09-26T10:00:30.000Z",
    timestamp_source: "monotonic high-resolution (performance.now, milliseconds)",
    keyboard_events: [
      { event_type: "keyboard", event: "keydown", timestamp: 1.5 },
      { event_type: "keyboard", event: "keyup", timestamp: 44.25 },
    ],
    mouse_events: [
      { event_type: "mouse", event: "mousemove", x: 12, y: 34, timestamp: 2.5 },
      { event_type: "mouse", event: "mouseup", x: 56, y: 78, timestamp: 90 },
    ],
  };
}

function collect(count) {
  let sessions = [];
  for (let i = 1; i <= count; i += 1) {
    sessions = collectEnrollmentSession(sessions, usableSession("s" + i)).sessions;
  }
  return sessions;
}

/* ---------------------- export button on the page -------------------- */

test("76. the enrollment page renders a visible Export Sessions button", () => {
  assert.ok(
    PAGE_SOURCE.includes("Export Sessions"),
    "EnrollmentPage must render an Export Sessions button"
  );
  assert.ok(
    PAGE_SOURCE.includes("from \"../lib/sessionExport.js\"") ||
      PAGE_SOURCE.includes("from '../lib/sessionExport.js'"),
    "EnrollmentPage must reuse the shared sessionExport helpers"
  );
  assert.ok(
    PAGE_SOURCE.includes("onClick={handleExport}"),
    "the Export Sessions button must be wired to the export handler"
  );
  assert.ok(
    PAGE_SOURCE.includes("disabled={!canExportSessions(sessions.length)}"),
    "the Export Sessions button must be disabled while no session is held"
  );
  assert.ok(
    PAGE_SOURCE.includes("no sessions to export"),
    "the disabled state must say there is nothing to export"
  );
});

test("77. the enrollment page states that export is local/download-only", () => {
  const lower = PAGE_SOURCE.toLowerCase();
  assert.ok(lower.includes("download only"), "privacy wording must say 'download only'");
  assert.ok(lower.includes("never uploaded to the server"));
  assert.ok(lower.includes("no name, email, password"));
  assert.equal(lower.includes("sendbeacon"), false);
  assert.equal(lower.includes("google-analytics"), false);
  assert.equal(lower.includes("gtag("), false);
});

/* ------------------------- zero-session no-op ------------------------ */

test("78. canExportSessions is false only while no session is held", () => {
  assert.equal(canExportSessions(0), false);
  assert.equal(canExportSessions(1), true);
  assert.equal(canExportSessions(10), true);
  assert.equal(canExportSessions(MAX_ENROLLMENT_SESSIONS), true);
  assert.equal(canExportSessions(-1), false);
  assert.equal(canExportSessions(NaN), false);
  assert.equal(canExportSessions(undefined), false);
  assert.equal(canExportSessions("3"), false);
});

test("79. exporting with zero sessions is a no-op: nothing is downloaded", () => {
  let downloads = 0;
  const record = () => {
    downloads += 1;
  };
  for (const empty of [[], undefined, null]) {
    const outcome = exportSessionsLocally(empty, { exportedAt: STAMP, download: record });
    assert.equal(outcome.ok, false);
    assert.equal(outcome.reason, "no-sessions");
    assert.equal(outcome.session_count, 0);
    assert.equal(outcome.filename, null);
    assert.equal(outcome.json, null);
    assert.equal(outcome.payload, null);
  }
  assert.equal(downloads, 0, "no download may be triggered for an empty session list");
});

test("80. the zero-session export also stays silent in the console", () => {
  const session = realSession(1);
  const written = withSilentConsole(() => {
    exportSessionsLocally([], { exportedAt: STAMP, download: () => {} });
    exportSessionsLocally([session], { exportedAt: STAMP, download: () => {} });
  });
  assert.deepEqual(written, [], "raw keyboard/mouse events must never reach the console");
});

/* ----------------------------- the export ---------------------------- */

test("81. the export document has the documented top-level shape", () => {
  const payload = buildSessionExport(collect(3), STAMP);
  assert.deepEqual(Object.keys(payload), [
    "export_version",
    "exported_at",
    "session_count",
    "sessions",
  ]);
  assert.equal(payload.export_version, "1.0");
  assert.equal(EXPORT_VERSION, "1.0");
  assert.equal(payload.exported_at, STAMP);
  assert.equal(payload.session_count, 3);
  assert.equal(payload.sessions.length, 3);
});

test("82. the export contains every session currently held on the page", () => {
  const sessions = collect(10);
  const outcome = exportSessionsLocally(sessions, { exportedAt: STAMP, download: () => {} });
  assert.equal(outcome.ok, true);
  assert.equal(outcome.reason, null);
  assert.equal(outcome.session_count, 10);
  const reparsed = JSON.parse(outcome.json);
  assert.deepEqual(Object.keys(reparsed), [
    "export_version",
    "exported_at",
    "session_count",
    "sessions",
  ]);
  assert.equal(reparsed.session_count, 10);
  assert.equal(reparsed.sessions.length, 10);
  assert.deepEqual(
    reparsed.sessions.map((s) => s.session_id),
    sessions.map((s) => s.session_id)
  );
});

test("83. exported session objects preserve their original fields and data", () => {
  const sessions = [realSession(1), realSession(2), realSession(3)];
  const outcome = exportSessionsLocally(sessions, { exportedAt: STAMP, download: () => {} });
  const reparsed = JSON.parse(outcome.json);
  reparsed.sessions.forEach((session, index) => {
    assert.deepEqual(session, sessions[index], "session " + (index + 1) + " must round-trip");
    assert.deepEqual(Object.keys(session).sort(), [
      "ended_at",
      "keyboard_events",
      "mouse_events",
      "session_id",
      "started_at",
      "timestamp_source",
    ]);
    assert.ok(session.ended_at, "ended_at must be stamped by the collector");
    session.keyboard_events.forEach((event) => {
      assert.deepEqual(Object.keys(event).sort(), ["event", "event_type", "timestamp"]);
    });
    session.mouse_events.forEach((event) => {
      assert.deepEqual(Object.keys(event).sort(), ["event", "event_type", "timestamp", "x", "y"]);
    });
  });
});

test("84. the export deep-copies, so the page state is never aliased", () => {
  const sessions = [realSession(1)];
  const payload = buildSessionExport(sessions, STAMP);
  assert.notEqual(payload.sessions[0], sessions[0]);
  assert.notEqual(payload.sessions[0].keyboard_events, sessions[0].keyboard_events);
  assert.deepEqual(payload.sessions[0], sessions[0]);
  sessions[0].keyboard_events.length = 0;
  assert.equal(payload.sessions[0].keyboard_events.length, 3);
});

/* --------------------------- local only ----------------------------- */

test("85. export performs no network request of any kind", async () => {
  const sessions = collect(4);
  await withForbiddenNetwork(async (calls) => {
    const doc = fakeDocument();
    const outcome = withDocument(doc, () =>
      exportSessionsLocally(sessions, { exportedAt: STAMP })
    );
    assert.equal(outcome.ok, true);
    assert.deepEqual(calls, [], "export must not touch fetch, XHR, or sendBeacon");
  });
});

test("86. the export module does not import the API client", () => {
  const source = fs.readFileSync(
    path.join(__dirname, "..", "frontend", "src", "lib", "sessionExport.js"),
    "utf8"
  );
  const importLines = source
    .split("\n")
    .map((line) => line.replace(/^\s*(\*|\/\/).*$/, ""))
    .filter((line) => /^\s*import\s|require\(/.test(line));
  assert.deepEqual(importLines, [], "sessionExport must stay dependency-free");
  assert.equal(source.includes("fetch("), false);
  assert.equal(source.includes("XMLHttpRequest"), false);
  assert.equal(source.includes("navigator."), false);
});

/* ----------------------------- filename ----------------------------- */

test("87. the filename is behavioral_sessions_<timestamp>.json", () => {
  const filename = buildExportFilename(STAMP);
  assert.equal(filename, EXPORT_FILENAME_PREFIX + STAMP_FILE + ".json");
  assert.equal(filename, "behavioral_sessions_2026-09-26T11-22-33-456Z.json");
  assert.ok(filename.endsWith(".json"));
  assert.equal(filename.includes(":"), false);
  assert.equal(filename.includes(" "), false);
  assert.equal(filename.includes("/"), false);
  assert.equal(filename.includes("\\"), false);
});

test("88. the filename never carries participant identity", () => {
  const secrets = [
    "user_001",
    "alice",
    "alice@example.com",
    "secret123",
    "jwt-abc",
    "eyJhbGciOiJIUzI1NiJ9",
  ];
  const filename = buildExportFilename(STAMP);
  secrets.forEach((secret) => {
    assert.equal(filename.includes(secret), false, "filename leaked " + secret);
  });
  assert.equal(filename.includes("session_id"), false);

  // The filename depends only on the timestamp, never on the payload.
  const withIdentity = collect(2);
  withIdentity[0].username = "alice";
  withIdentity[0].password = "secret123";
  const outcome = exportSessionsLocally(withIdentity, {
    exportedAt: STAMP,
    download: () => {},
  });
  assert.equal(outcome.filename, filename, "filename must not vary with session content");
  secrets.forEach((secret) => {
    assert.equal(outcome.filename.includes(secret), false);
  });
});

test("89. exportTimestamp accepts an epoch, a clock, or defaults to now", () => {
  assert.equal(exportTimestamp(1789000000000), new Date(1789000000000).toISOString());
  assert.equal(exportTimestamp(() => 1789000000000), new Date(1789000000000).toISOString());
  const now = exportTimestamp();
  assert.equal(new Date(now).toISOString(), now);
  assert.equal(exportTimestamp("nonsense").length > 0, true);
});

/* ------------------------------ privacy ------------------------------ */

test("90. the export introduces no privacy-sensitive field", () => {
  const sessions = [realSession(1), realSession(2)];
  const outcome = exportSessionsLocally(sessions, { exportedAt: STAMP, download: () => {} });
  const keys = collectKeys(JSON.parse(outcome.json), new Set());
  keys.forEach((key) => {
    assert.equal(FORBIDDEN_KEYS.has(key), false, "forbidden key in export: " + key);
    assert.equal(ALLOWED_KEYS.has(key), true, "unexpected key in export: " + key);
  });

  const secrets = ["alice", "alice@example.com", "secret123", "jwt-abc"];
  secrets.forEach((secret) => {
    assert.equal(outcome.json.includes(secret), false, "export leaked " + secret);
  });
  assert.equal(/eyJhbGciOi/.test(outcome.json), false, "no JWT may appear in the export");
  assert.equal(outcome.json.includes("Bearer"), false);
});

test("91. a full 10-session collection exports with identical event counts", () => {
  const sessions = [];
  for (let i = 1; i <= 10; i += 1) {
    sessions.push(realSession(i));
  }
  const expected = sessions.map((s) => ({
    session_id: s.session_id,
    keyboard: s.keyboard_events.length,
    mouse: s.mouse_events.length,
  }));
  const outcome = exportSessionsLocally(sessions, { exportedAt: STAMP, download: () => {} });
  assert.equal(outcome.ok, true);
  const reparsed = JSON.parse(outcome.json);
  assert.equal(reparsed.session_count, 10);
  assert.equal(reparsed.sessions.length, 10);
  reparsed.sessions.forEach((session, index) => {
    assert.equal(session.session_id, expected[index].session_id);
    assert.equal(session.keyboard_events.length, expected[index].keyboard);
    assert.equal(session.mouse_events.length, expected[index].mouse);
  });
});

test("92. the 16-session cap is unchanged and the cap set still exports", () => {
  assert.equal(MAX_ENROLLMENT_SESSIONS, 16);
  assert.equal(hasReachedMaxSessions(15), false);
  assert.equal(hasReachedMaxSessions(16), true);
  assert.equal(hasReachedMaxSessions(17), true);

  const sessions = collect(MAX_ENROLLMENT_SESSIONS);
  assert.equal(sessions.length, 16);
  const overflow = collectEnrollmentSession(sessions, usableSession("s17"));
  assert.equal(overflow.added, false);
  assert.equal(overflow.reason, "max-sessions");
  assert.equal(overflow.sessions.length, 16);

  const outcome = exportSessionsLocally(sessions, { exportedAt: STAMP, download: () => {} });
  assert.equal(outcome.ok, true);
  assert.equal(outcome.session_count, 16);
  assert.equal(JSON.parse(outcome.json).session_count, 16);
  assert.equal(collect(15).length, 15);
  assert.equal(collect(1).length, 1);
});

test("93. every session count from 1 to the cap is exportable", () => {
  for (let n = 1; n <= MAX_ENROLLMENT_SESSIONS; n += 1) {
    const outcome = exportSessionsLocally(collect(n), { exportedAt: STAMP, download: () => {} });
    assert.equal(outcome.ok, true, "count " + n + " must export");
    assert.equal(outcome.session_count, n);
    assert.equal(JSON.parse(outcome.json).sessions.length, n);
  }
});

/* ------------------------- browser download ------------------------- */

test("94. downloadJsonFile triggers a real browser file download", () => {
  const doc = fakeDocument();
  const json = JSON.stringify(buildSessionExport(collect(1), STAMP));
  const filename = buildExportFilename(STAMP);
  const saved = withDocument(doc, () => downloadJsonFile(json, filename));
  assert.equal(saved, true);
  assert.equal(doc.created.length, 1);
  const anchor = doc.created[0];
  assert.equal(anchor.tag, "a");
  assert.equal(anchor.download, filename);
  assert.equal(anchor.clicks, 1, "the synthetic anchor must be clicked exactly once");
  assert.equal(doc.appended.length, 1);
  assert.equal(doc.removed.length, 1, "the synthetic anchor must be removed again");
  assert.ok(anchor.href.startsWith("blob:"), "the download must use a local object URL");
});

test("95. the default download path is used when no saver is injected", () => {
  const doc = fakeDocument();
  const outcome = withDocument(doc, () =>
    exportSessionsLocally(collect(2), { exportedAt: STAMP })
  );
  assert.equal(outcome.ok, true);
  assert.equal(outcome.filename, "behavioral_sessions_" + STAMP_FILE + ".json");
  assert.equal(doc.created.length, 1);
  assert.equal(doc.created[0].download, outcome.filename);
  assert.equal(doc.created[0].clicks, 1);
});

test("96. a download that cannot start is reported, not thrown", () => {
  const original = globalThis.document;
  delete globalThis.document;
  try {
    const outcome = exportSessionsLocally(collect(1), { exportedAt: STAMP });
    assert.equal(outcome.ok, false);
    assert.equal(outcome.reason, "download-unavailable");
    assert.equal(outcome.filename, null);
    assert.ok(outcome.json.length > 0, "the JSON is still produced for inspection");
  } finally {
    if (original) {
      globalThis.document = original;
    }
  }
  assert.equal(downloadJsonFile("{}", "x.json", null), false);
});
