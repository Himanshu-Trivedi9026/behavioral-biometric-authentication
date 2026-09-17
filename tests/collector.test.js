/*
 * Behavioral Biometric Authentication — Phase 2
 * Unit tests for the core collector logic (frontend/js/collector.js).
 *
 * Run with:  node --test tests/
 *
 * Uses Node's built-in test runner (no external dependencies).
 * These tests exercise the pure data-model logic. The DOM wiring in
 * app.js is browser-glue and is covered by the manual browser
 * verification procedure documented in README.md.
 */
"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");

const { BehavioralSession, now } = require("../frontend/src/lib/collector.js");

function freshSession(opts) {
  return new BehavioralSession(opts);
}

test("1. a session can start", () => {
  const s = freshSession();
  assert.equal(s.start(), true);
  assert.equal(s.isRunning(), true);
  assert.equal(s.isIdle(), false);
  const sess = s.getSession();
  assert.ok(sess);
  assert.ok(Array.isArray(sess.keyboard_events));
  assert.ok(Array.isArray(sess.mouse_events));
});

test("2. a session receives keyboard events", () => {
  const s = freshSession();
  s.start();
  assert.equal(s.addKeyboardEvent("keydown", now()), true);
  assert.equal(s.addKeyboardEvent("keyup", now()), true);
  assert.equal(s.keyboardEventCount(), 2);
});

test("3. a session receives mouse events", () => {
  const s = freshSession();
  s.start();
  assert.equal(s.addMouseEvent("mousemove", 10, 20, now()), true);
  assert.equal(s.addMouseEvent("mousedown", 30, 40, now()), true);
  assert.equal(s.addMouseEvent("mouseup", 30, 40, now()), true);
  assert.equal(s.mouseEventCount(), 3);
});

test("4. keyboard events contain timestamps", () => {
  const s = freshSession();
  s.start();
  s.addKeyboardEvent("keydown", now());
  s.addKeyboardEvent("keyup", now());
  for (const ev of s.getSession().keyboard_events) {
    assert.equal(typeof ev.timestamp, "number");
    assert.ok(Number.isFinite(ev.timestamp));
  }
});

test("5. mouse events contain x/y/timestamp", () => {
  const s = freshSession();
  s.start();
  s.addMouseEvent("mousemove", 123, 456, now());
  const ev = s.getSession().mouse_events[0];
  assert.equal(ev.x, 123);
  assert.equal(ev.y, 456);
  assert.ok(Number.isFinite(ev.timestamp));
});

test("6. typed characters are NOT persisted", () => {
  const s = freshSession();
  s.start();
  // Simulate what the browser creates on keydown/keyup. Note that the
  // collector API does not even accept a key value, so even if a caller
  // passed extra data it would be dropped by the model.
  s.addKeyboardEvent("keydown", now());
  s.addKeyboardEvent("keyup", now());
  const sess = s.getSession();
  const json = JSON.stringify(sess);
  assert.equal(json.includes("password"), false);
  assert.equal(json.includes("secret"), false);
  assert.equal(json.includes("P@ssw0rd"), false);
  for (const ev of sess.keyboard_events) {
    assert.deepEqual(
      Object.keys(ev).sort(),
      ["event", "event_type", "timestamp"]
    );
    assert.equal(Object.prototype.hasOwnProperty.call(ev, "key"), false);
  }
});

test("7. session IDs are unique", () => {
  const seen = new Set();
  for (let i = 0; i < 200; i++) {
    const s = freshSession();
    assert.equal(s.start(), true);
    const id = s.getSession().session_id;
    assert.equal(typeof id, "string");
    assert.ok(id.length > 0);
    assert.equal(seen.has(id), false, "duplicate session id generated: " + id);
    seen.add(id);
  }
});

test("8. a session can be stopped", () => {
  const s = freshSession();
  s.start();
  s.addKeyboardEvent("keydown", now());
  assert.equal(s.isRunning(), true);
  assert.equal(s.stop(), true);
  assert.equal(s.isStopped(), true);
  assert.equal(s.isRunning(), false);
  const sess = s.getSession();
  assert.equal(typeof sess.ended_at, "string");
  // No events can be added after stopping.
  assert.equal(s.addKeyboardEvent("keydown", now()), false);
  assert.equal(s.keyboardEventCount(), 1);
});

test("9. session JSON can be generated", () => {
  const s = freshSession();
  s.start();
  s.addKeyboardEvent("keydown", now());
  s.addMouseEvent("mousemove", 1, 2, now());
  s.stop();
  const json = s.exportJSON();
  assert.equal(typeof json, "string");
  assert.doesNotThrow(() => JSON.parse(json));
});

test("10. generated JSON has the expected structure", () => {
  const s = freshSession();
  s.start();
  s.addKeyboardEvent("keydown", now());
  s.addKeyboardEvent("keyup", now());
  s.addMouseEvent("mousemove", 5, 6, now());
  s.addMouseEvent("mousedown", 7, 8, now());
  s.addMouseEvent("mouseup", 7, 8, now());
  s.stop();
  const parsed = JSON.parse(s.exportJSON());
  assert.deepEqual(
    Object.keys(parsed).sort(),
    [
      "ended_at",
      "keyboard_events",
      "mouse_events",
      "session_id",
      "started_at",
      "timestamp_source"
    ]
  );
  assert.ok(parsed.session_id);
  assert.equal(typeof parsed.started_at, "string");
  assert.equal(typeof parsed.ended_at, "string");
  assert.equal(Array.isArray(parsed.keyboard_events), true);
  assert.equal(Array.isArray(parsed.mouse_events), true);
  assert.equal(parsed.keyboard_events.length, 2);
  assert.equal(parsed.mouse_events.length, 3);
  // Keyboard event shape
  assert.deepEqual(
    Object.keys(parsed.keyboard_events[0]).sort(),
    ["event", "event_type", "timestamp"]
  );
  // Mouse event shape
  assert.deepEqual(
    Object.keys(parsed.mouse_events[0]).sort(),
    ["event", "event_type", "timestamp", "x", "y"]
  );
});

test("11. empty sessions are handled safely", () => {
  const s = freshSession();
  // Zero-event session
  assert.equal(s.start(), true);
  assert.equal(s.stop(), true);
  const parsed = JSON.parse(s.exportJSON());
  assert.equal(parsed.keyboard_events.length, 0);
  assert.equal(parsed.mouse_events.length, 0);
  assert.equal(typeof parsed.ended_at, "string");
  // exportJSON with no session at all
  const empty = freshSession();
  assert.equal(empty.exportJSON(), null);
  assert.equal(empty.getSession(), null);
});

test("12. clearing a session removes its in-memory data", () => {
  const s = freshSession();
  s.start();
  s.addKeyboardEvent("keydown", now());
  s.addMouseEvent("mousemove", 1, 2, now());
  assert.equal(s.keyboardEventCount(), 1);
  assert.equal(s.mouseEventCount(), 1);
  assert.equal(s.clear(), true);
  assert.equal(s.isIdle(), true);
  assert.equal(s.getSession(), null);
  assert.equal(s.exportJSON(), null);
  assert.equal(s.keyboardEventCount(), 0);
  assert.equal(s.mouseEventCount(), 0);
  // A fresh session can be started afterwards.
  assert.equal(s.start(), true);
});

test("events are ignored before a session starts", () => {
  const s = freshSession();
  assert.equal(s.addKeyboardEvent("keydown", now()), false);
  assert.equal(s.addMouseEvent("mousemove", 1, 2, now()), false);
  assert.equal(s.keyboardEventCount(), 0);
  assert.equal(s.mouseEventCount(), 0);
});

test("invalid/malformed events are rejected safely", () => {
  const s = freshSession();
  s.start();
  // Unknown event names
  assert.equal(s.addKeyboardEvent("click", now()), false);
  assert.equal(s.addMouseEvent("keydown", 1, 2, now()), false);
  // Non-finite / missing timestamps
  assert.equal(s.addKeyboardEvent("keydown", NaN), false);
  assert.equal(s.addKeyboardEvent("keydown", "now"), false);
  assert.equal(s.addKeyboardEvent("keydown", null), false);
  assert.equal(s.addKeyboardEvent("keydown", undefined), false);
  // Non-finite or missing coordinates
  assert.equal(s.addMouseEvent("mousemove", NaN, 2, now()), false);
  assert.equal(s.addMouseEvent("mousemove", 1, Infinity, now()), false);
  assert.equal(s.addMouseEvent("mousemove", 1, 2, NaN), false);
  // Nothing was recorded
  assert.equal(s.keyboardEventCount(), 0);
  assert.equal(s.mouseEventCount(), 0);
});

test("duplicate events are deduplicated safely", () => {
  const s = freshSession();
  s.start();
  const t = now();
  assert.equal(s.addKeyboardEvent("keydown", t), true);
  assert.equal(s.addKeyboardEvent("keydown", t), false);
  assert.equal(s.keyboardEventCount(), 1);
  assert.equal(s.addMouseEvent("mousemove", 100, 100, t), true);
  assert.equal(s.addMouseEvent("mousemove", 100, 100, t), false);
  assert.equal(s.mouseEventCount(), 1);
});

test("cannot start twice or restart while stopped", () => {
  const s = freshSession();
  assert.equal(s.start(), true);
  assert.equal(s.start(), false);
  s.stop();
  assert.equal(s.start(), false);
});

test("mousemove is throttled by interval and distance", () => {
  // Force aggressive throttling so every move within window is skipped.
  const s = freshSession({
    mouseMoveMinIntervalMs: 1000,
    mouseMoveMinDistancePx: 1000
  });
  s.start();
  const t0 = now();
  assert.equal(s.addMouseEvent("mousemove", 0, 0, t0), true);
  // Same position, shortly after: skipped by interval rule.
  assert.equal(s.addMouseEvent("mousemove", 1, 1, t0 + 10), false);
  // Far move at the same time: OR-rule captures it (distance exceeded).
  assert.equal(s.addMouseEvent("mousemove", 2000, 2000, t0 + 11), true);
  // Now inside interval again with small move: skipped.
  assert.equal(s.addMouseEvent("mousemove", 2005, 2005, t0 + 20), false);
  assert.equal(s.mouseEventCount(), 2);
});

test("configuration options are sanitized", () => {
  const s = freshSession({
    mouseMoveMinIntervalMs: "banana",
    mouseMoveMinDistancePx: -5,
    bogusOption: 3
  });
  s.start();
  const t = now();
  assert.equal(s.addMouseEvent("mousemove", 0, 0, t), true);
  // With the literal 0/0 default... defaults apply when values are invalid
  // strings/negatives. Distance default is 2, interval default is 25.
  assert.equal(s.addMouseEvent("mousemove", 1, 1, t + 5), false);
  assert.equal(s.mouseEventCount(), 1);
});

test("timestamps use the monotonic high-resolution source", () => {
  const sample = now();
  assert.equal(typeof sample, "number");
  assert.ok(Number.isFinite(sample));
  // A later call must not go backwards.
  const later = now();
  assert.ok(later >= sample);
});