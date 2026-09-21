/*
 * Behavioral Biometric Authentication — Phase 2
 * React hook wrapping the collector logic for the UI.
 *
 * All behavioral collection logic lives in lib/collector.js. This hook
 * only owns a BehavioralSession instance, attaches window-level listeners
 * while a session is running, and exposes React state + actions.
 *
 * Privacy: this file reads only clientX/clientY. It never reads or stores
 * typed text, input values, clipboard content, cookies, or DOM content.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { BehavioralSession, now } from "../lib/collector.js";

const STATUS_LABELS = {
  idle: "Idle — no active session",
  running: "Collecting",
  stopped: "Stopped — ready to export",
};

export function useCollector() {
  const sessionRef = useRef(new BehavioralSession());

  const [status, setStatus] = useState("idle"); // idle | running | stopped
  const [sessionId, setSessionId] = useState(null);
  const [keyboardCount, setKeyboardCount] = useState(0);
  const [mouseCount, setMouseCount] = useState(0);
  const [lastEvent, setLastEvent] = useState("ready — press Start Collection");
  const [session, setSession] = useState(null);

  const refreshFromSession = useCallback(() => {
    const c = sessionRef.current;
    const sess = c.getSession();
    setSession(sess);
    if (sess) {
      setSessionId(sess.session_id);
      setKeyboardCount(sess.keyboard_events.length);
      setMouseCount(sess.mouse_events.length);
    } else {
      setSessionId(null);
      setKeyboardCount(0);
      setMouseCount(0);
    }
  }, []);

  const record = useCallback(() => {
    const c = sessionRef.current;
    setKeyboardCount(c.keyboardEventCount());
    setMouseCount(c.mouseEventCount());
  }, []);

  const start = useCallback(() => {
    if (!sessionRef.current.start()) {
      return false;
    }
    setStatus("running");
    setLastEvent("session started — type and move your mouse");
    refreshFromSession();
    return true;
  }, [refreshFromSession]);

  const stop = useCallback(() => {
    if (!sessionRef.current.stop()) {
      return false;
    }
    setStatus("stopped");
    setLastEvent("session stopped");
    refreshFromSession();
    return true;
  }, [refreshFromSession]);

  const download = useCallback(() => {
    const c = sessionRef.current;
    const json = c.exportJSON();
    const sess = c.getSession();
    if (!json || !sess) {
      return false;
    }
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "behavioral-session-" + sess.session_id + ".json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setLastEvent("session exported to JSON");
    return true;
  }, []);

  const clear = useCallback(() => {
    sessionRef.current.clear();
    setStatus("idle");
    setSessionId(null);
    setKeyboardCount(0);
    setMouseCount(0);
    setSession(null);
    setLastEvent("ready — press Start Collection");
    return true;
  }, []);

  /* ---- window listeners (attached only while running) ---------------- */
  useEffect(() => {
    if (status !== "running") {
      return undefined;
    }
    const c = sessionRef.current;

    const onKeyDown = () => {
      const ts = now();
      if (c.addKeyboardEvent("keydown", ts)) {
        setLastEvent("keydown @" + ts.toFixed(3));
        record();
      }
    };
    const onKeyUp = () => {
      const ts = now();
      if (c.addKeyboardEvent("keyup", ts)) {
        setLastEvent("keyup @" + ts.toFixed(3));
        record();
      }
    };
    const onMouseMove = (e) => {
      const ts = now();
      if (c.addMouseEvent("mousemove", e.clientX, e.clientY, ts)) {
        setLastEvent("mousemove (" + e.clientX + ", " + e.clientY + ")");
        record();
      }
    };
    const onMouseDown = (e) => {
      const ts = now();
      if (c.addMouseEvent("mousedown", e.clientX, e.clientY, ts)) {
        setLastEvent("mousedown (" + e.clientX + ", " + e.clientY + ")");
        record();
      }
    };
    const onMouseUp = (e) => {
      const ts = now();
      if (c.addMouseEvent("mouseup", e.clientX, e.clientY, ts)) {
        setLastEvent("mouseup (" + e.clientX + ", " + e.clientY + ")");
        record();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, [status, record]);

  return {
    status,
    statusLabel: STATUS_LABELS[status],
    sessionId,
    keyboardCount,
    mouseCount,
    lastEvent,
    session,
    start,
    stop,
    download,
    clear,
  };
}