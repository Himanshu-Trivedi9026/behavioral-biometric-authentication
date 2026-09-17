/*
 * Behavioral Biometric Authentication — Phase 2 Collector
 *
 * Browser-based raw behavioral event collector.
 *
 * PRIVACY: This collector records BEHAVIORAL METADATA ONLY.
 * It deliberately does NOT capture or persist:
 *   - typed characters / text content
 *   - passwords or input field values
 *   - clipboard content
 *   - cookies, history, or other browsing data
 *   - arbitrary DOM content
 *
 * A keyboard event stores only the event name and a high-resolution
 * timestamp. The value of the pressed key is never read or stored.
 *
 * Timestamp strategy:
 *   event timestamps use performance.now() — a monotonic, high-resolution
 *   timer (sub-millisecond) that is immune to wall-clock adjustments,
 *   which is what matters for behavioral timing. The session additionally
 *   records started_at / ended_at as ISO-8601 wall-clock strings for
 *   reference only.
 *
 * Mouse sampling strategy:
 *   mousemove is throttled so we record at most one sample per
 *   mouseMoveMinIntervalMs (default 25 ms) UNLESS the pointer has moved
 *   by at least mouseMoveMinDistancePx (default 2 px). This keeps a
 *   trajectory-shaping sample set without generating unbounded data at
 *   the native (often 1000+ Hz) pointer-event rate.
 *
 * This module is dependency-free ESM. It is consumed both from the
 * React + Vite frontend (browser ES module import) and from the Node
 * test suite (Node 22.12+/24 supports require() of ESM), so all
 * collection logic lives in one place and stays unit-testable.
 */

var KBD_EVENTS = { keydown: true, keyup: true };
var MOUSE_EVENTS = { mousemove: true, mousedown: true, mouseup: true };

var DEFAULT_OPTIONS = {
  mouseMoveMinIntervalMs: 25,
  mouseMoveMinDistancePx: 2
};

function isFiniteNumber(v) {
  return typeof v === "number" && Number.isFinite(v);
}

/* Monotonic high-resolution timestamp in milliseconds. */
function now() {
  if (typeof performance !== "undefined" && typeof performance.now === "function") {
    return performance.now();
  }
  return Date.now();
}

/* Reliable UUID. Uses Web Crypto when available (secure contexts);
   falls back to a v4-style random UUID otherwise. */
function generateSessionId() {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
    var r = (Math.random() * 16) | 0;
    var v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

function BehavioralSession(options) {
  this._options = {};
  Object.keys(DEFAULT_OPTIONS).forEach(function (k) {
    this._options[k] = DEFAULT_OPTIONS[k];
  }, this);
  if (options && typeof options === "object") {
    Object.keys(DEFAULT_OPTIONS).forEach(function (k) {
      if (typeof options[k] === "number" && isFiniteNumber(options[k]) && options[k] >= 0) {
        this._options[k] = options[k];
      }
    }, this);
  }
  this._state = "idle"; // idle | running | stopped
  this._session = null;
  this._lastMouseEvent = null;
  this._lastKeyboardEvent = null;
}

BehavioralSession.prototype.isIdle = function () {
  return this._state === "idle";
};

BehavioralSession.prototype.isRunning = function () {
  return this._state === "running";
};

BehavioralSession.prototype.isStopped = function () {
  return this._state === "stopped";
};

/* Start a new collection session. Returns false if one is already active. */
BehavioralSession.prototype.start = function () {
  if (this._state !== "idle") {
    return false;
  }
  var id = generateSessionId();
  if (typeof id !== "string" || id.length === 0) {
    return false;
  }
  this._session = {
    session_id: id,
    started_at: new Date().toISOString(),
    ended_at: null,
    timestamp_source:
      "monotonic high-resolution (performance.now, milliseconds)",
    keyboard_events: [],
    mouse_events: []
  };
  this._lastMouseEvent = null;
  this._lastKeyboardEvent = null;
  this._state = "running";
  return true;
};

/* Stop the active session, stamping ended_at. */
BehavioralSession.prototype.stop = function () {
  if (this._state !== "running") {
    return false;
  }
  this._session.ended_at = new Date().toISOString();
  this._state = "stopped";
  return true;
};

/* Clear the current in-memory session. Stops first if still running. */
BehavioralSession.prototype.clear = function () {
  if (this._state === "running") {
    this._session.ended_at = new Date().toISOString();
  }
  this._session = null;
  this._lastMouseEvent = null;
  this._lastKeyboardEvent = null;
  this._state = "idle";
  return true;
};

/*
 * Record a keyboard event (keydown | keyup).
 * Returns true if recorded, false if ignored (not running, unknown event
 * name, invalid timestamp, or an exact duplicate of the previous event).
 */
BehavioralSession.prototype.addKeyboardEvent = function (name, timestamp) {
  if (this._state !== "running") {
    return false;
  }
  if (!Object.prototype.hasOwnProperty.call(KBD_EVENTS, name)) {
    return false;
  }
  if (!isFiniteNumber(timestamp)) {
    return false;
  }
  var last = this._lastKeyboardEvent;
  if (last && last.event === name && last.timestamp === timestamp) {
    return false;
  }
  this._session.keyboard_events.push({
    event_type: "keyboard",
    event: name,
    timestamp: timestamp
  });
  this._lastKeyboardEvent = {
    event: name,
    timestamp: timestamp
  };
  return true;
};

/*
 * Record a mouse event (mousemove | mousedown | mouseup) with viewport
 * coordinates. mousemove is throttled/sampled (see header comment).
 */
BehavioralSession.prototype.addMouseEvent = function (name, x, y, timestamp) {
  if (this._state !== "running") {
    return false;
  }
  if (!Object.prototype.hasOwnProperty.call(MOUSE_EVENTS, name)) {
    return false;
  }
  if (!isFiniteNumber(x) || !isFiniteNumber(y) || !isFiniteNumber(timestamp)) {
    return false;
  }
  var last = this._lastMouseEvent;
  if (last && last.event === name && last.x === x && last.y === y && last.timestamp === timestamp) {
    return false;
  }
  if (name === "mousemove") {
    if (last && last.event === "mousemove") {
      var dt = timestamp - last.timestamp;
      var dx = x - last.x;
      var dy = y - last.y;
      var dist = Math.sqrt(dx * dx + dy * dy);
      if (
        dt < this._options.mouseMoveMinIntervalMs &&
        dist < this._options.mouseMoveMinDistancePx
      ) {
        return false;
      }
    }
  }
  this._session.mouse_events.push({
    event_type: "mouse",
    event: name,
    x: x,
    y: y,
    timestamp: timestamp
  });
  this._lastMouseEvent = {
    event: name,
    x: x,
    y: y,
    timestamp: timestamp
  };
  return true;
};

BehavioralSession.prototype.keyboardEventCount = function () {
  return this._session ? this._session.keyboard_events.length : 0;
};

BehavioralSession.prototype.mouseEventCount = function () {
  return this._session ? this._session.mouse_events.length : 0;
};

/* Shallow-clean copy of the session. Returns null when no session exists. */
BehavioralSession.prototype.getSession = function () {
  if (!this._session) {
    return null;
  }
  return JSON.parse(JSON.stringify(this._session));
};

/* Pretty-printed JSON export of the session. Returns null when no session. */
BehavioralSession.prototype.exportJSON = function () {
  if (!this._session) {
    return null;
  }
  return JSON.stringify(this._session, null, 2);
};

var keyboardEventsList = Object.keys(KBD_EVENTS);
var mouseEventsList = Object.keys(MOUSE_EVENTS);

export const version = "2.0.0";
export { BehavioralSession, now, generateSessionId };
export { keyboardEventsList as KEYBOARD_EVENTS, mouseEventsList as MOUSE_EVENTS };