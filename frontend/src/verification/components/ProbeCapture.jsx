import { useEffect, useRef } from "react";

export default function ProbeCapture({
  status,
  keyboardCount,
  mouseCount,
  lastEvent,
  sessionId,
  probe,
  notice,
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
    <div className="vfy-capture">
      <div className="vfy-capture-toolbar">
        <span className="vfy-capture-label vfy-mono">
          LIVE PROBE
          {running ? " · RUNNING" : status === "stopped" ? " · STOPPED" : " · READY"}
        </span>
        <span className="vfy-capture-counts vfy-mono">
          KBD {keyboardCount} · MOUSE {mouseCount}
        </span>
      </div>

      <textarea
        ref={zoneRef}
        className={running ? "vfy-capture-zone vfy-capture-zone--active" : "vfy-capture-zone"}
        rows="4"
        placeholder={
          "Type harmless sample text and move your mouse here while the probe runs\n" +
          "(e.g. 'the quick brown fox jumps over the lazy dog').\n" +
          "Never type passwords or personal information — key identities are never recorded."
        }
        aria-label="Behavioral probe capture area"
      />

      <div className="vfy-capture-row">
        <p className="vfy-capture-event">
          <span className="vfy-mono">LAST EVENT:</span> {lastEvent}
        </p>
        {sessionId && (
          <p className="vfy-capture-id" title={sessionId}>
            {sessionId.slice(0, 8)}…
          </p>
        )}
      </div>

      {running ? (
        <button type="button" className="vfy-btn vfy-btn-secondary vfy-btn-block" onClick={onFinish}>
          Finish Probe
        </button>
      ) : (
        <button type="button" className="vfy-btn vfy-btn-primary vfy-btn-block" onClick={onStart}>
          Start Probe
        </button>
      )}

      {notice && (
        <p className="vfy-capture-notice" role="alert">
          {notice}
        </p>
      )}

      {probe && (
        <p className="vfy-capture-ready vfy-mono" role="status">
          PROBE READY — press Run Verification to check your identity.
        </p>
      )}
    </div>
  );
}