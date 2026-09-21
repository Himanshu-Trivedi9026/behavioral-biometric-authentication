import { useEffect, useRef } from "react";

export default function WindowCapture({
  status,
  keyboardCount,
  mouseCount,
  lastEvent,
  sessionId,
  capReason,
  disabled,
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

  return (
    <div className="ctn-capture">
      <div className="ctn-capture-toolbar">
        <span className="ctn-capture-label ctn-mono">
          LIVE WINDOW
          {running ? " · RUNNING" : status === "stopped" ? " · STOPPED" : " · READY"}
        </span>
        <span className="ctn-capture-counts ctn-mono">
          KBD {keyboardCount} · MOUSE {mouseCount}
        </span>
      </div>

      <textarea
        ref={zoneRef}
        className={running ? "ctn-capture-zone ctn-capture-zone--active" : "ctn-capture-zone"}
        rows="4"
        placeholder={
          "Type harmless sample text and move your mouse here while the window runs\n" +
          "(e.g. 'the quick brown fox jumps over the lazy dog').\n" +
          "Never type passwords or personal information — key identities are never recorded."
        }
        aria-label="Behavioral window capture area"
        readOnly={disabled}
      />

      <div className="ctn-capture-row">
        <p className="ctn-capture-event">
          <span className="ctn-mono">LAST EVENT:</span> {lastEvent}
        </p>
        {sessionId && (
          <p className="ctn-capture-id" title={sessionId}>
            {sessionId.slice(0, 8)}…
          </p>
        )}
      </div>

      {capReason && (
        <p className="ctn-capture-note ctn-mono" role="status">
          CAP: {capReason} — the window closed itself automatically.
        </p>
      )}

      {running ? (
        <button
          type="button"
          className="ctn-btn ctn-btn-secondary ctn-btn-block"
          onClick={onFinish}
          disabled={disabled}
        >
          Finish Window Early
        </button>
      ) : (
        <button
          type="button"
          className="ctn-btn ctn-btn-primary ctn-btn-block"
          onClick={onStart}
          disabled={disabled}
        >
          Start Window
        </button>
      )}
    </div>
  );
}