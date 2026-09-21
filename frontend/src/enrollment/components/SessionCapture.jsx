import { useEffect, useRef } from "react";

export default function SessionCapture({
  status,
  keyboardCount,
  mouseCount,
  lastEvent,
  sessionId,
  notice,
  sessionsCompleted,
  maxSessions,
  onStart,
  onFinish,
}) {
  const zoneRef = useRef(null);

  useEffect(() => {
    if (status === "running" && zoneRef.current) {
      zoneRef.current.focus({ preventScroll: true });
    }
  }, [status]);

  const running = status === "running";
  const maxReached = maxSessions > 0 && sessionsCompleted >= maxSessions;

  return (
    <div className="enr-capture">
      <div className="enr-capture-toolbar">
        <span className="enr-capture-label enr-mono">
          LIVE CAPTURE
          {running ? " · RUNNING" : status === "stopped" ? " · STOPPED" : " · READY"}
        </span>
        <span className="enr-capture-counts enr-mono">
          SESSIONS CAPTURED: {sessionsCompleted}
          {maxReached ? " / " + maxSessions + " MAX" : ""} · KBD {keyboardCount} · MOUSE{" "}
          {mouseCount}
        </span>
      </div>

      <textarea
        ref={zoneRef}
        className={running ? "enr-capture-zone enr-capture-zone--active" : "enr-capture-zone"}
        rows="4"
        placeholder={
          "Type harmless sample text and move your mouse here while the session runs\n" +
          "(e.g. 'the quick brown fox jumps over the lazy dog').\n" +
          "Never type passwords or personal information — key identities are never recorded."
        }
        aria-label="Behavioral capture area"
      />

      <div className="enr-capture-row">
        <p className="enr-capture-event">
          <span className="enr-mono">LAST EVENT:</span> {lastEvent}
        </p>
        {sessionId && (
          <p className="enr-capture-id" title={sessionId}>
            {sessionId.slice(0, 8)}…
          </p>
        )}
      </div>

      {running ? (
        <button type="button" className="enr-btn enr-btn-secondary enr-btn-block" onClick={onFinish}>
          Finish Session
        </button>
      ) : (
        <button
          type="button"
          className="enr-btn enr-btn-primary enr-btn-block"
          disabled={maxReached}
          onClick={onStart}
        >
          {maxReached
            ? "Maximum of " + maxSessions + " Sessions Reached"
            : "Start Session"}
        </button>
      )}

      {maxReached && (
        <p className="enr-capture-max" role="status">
          Session limit reached — {sessionsCompleted} of {maxSessions} sessions
          captured. Send your profile to the server.
        </p>
      )}

      {notice && (
        <p className="enr-capture-notice" role="alert">
          {notice}
        </p>
      )}
    </div>
  );
}