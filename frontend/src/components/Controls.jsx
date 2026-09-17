export default function Controls({ status, onStart, onStop, onDownload, onClear }) {
  const idle = status === "idle";
  const running = status === "running";
  const stopped = status === "stopped";

  return (
    <section className="card controls-card">
      <h2>Controls</h2>
      <div className="controls">
        <button
          type="button"
          className="btn btn-primary"
          onClick={onStart}
          disabled={!idle}
        >
          Start Collection
        </button>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={onStop}
          disabled={!running}
        >
          Stop Collection
        </button>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={onDownload}
          disabled={!stopped}
        >
          Download / Save Session (JSON)
        </button>
        <button
          type="button"
          className="btn btn-danger"
          onClick={onClear}
          disabled={!stopped}
        >
          Clear Session
        </button>
      </div>
    </section>
  );
}