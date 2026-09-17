export default function StatusCard({
  status,
  statusLabel,
  sessionId,
  keyboardCount,
  mouseCount,
  lastEvent,
}) {
  return (
    <section className="card status-card">
      <h2>Collection Status</h2>
      <div className="status-row">
        <span className={"dot dot-" + status} aria-hidden="true" />
        <span className="status-label">{statusLabel}</span>
      </div>
      <ul className="stats">
        <li>
          Session ID: <strong>{sessionId || "—"}</strong>
        </li>
        <li>
          Keyboard events: <strong>{keyboardCount}</strong>
        </li>
        <li>
          Mouse events: <strong>{mouseCount}</strong>
        </li>
      </ul>
      <p className="last-event">
        Last event: <span>{lastEvent}</span>
      </p>
    </section>
  );
}