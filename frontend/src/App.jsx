import { useCollector } from "./hooks/useCollector.js";
import StatusCard from "./components/StatusCard.jsx";
import Controls from "./components/Controls.jsx";
import TypeZone from "./components/TypeZone.jsx";
import PrivacyNotice from "./components/PrivacyNotice.jsx";

export default function App() {
  const {
    status,
    statusLabel,
    sessionId,
    keyboardCount,
    mouseCount,
    lastEvent,
    start,
    stop,
    download,
    clear,
  } = useCollector();

  return (
    <div className="container">
      <header className="app-header">
        <h1>Behavioral Biometric Authentication</h1>
        <p className="subtitle">
          Phase 2 — Browser-based Keyboard &amp; Mouse Data Collector{" "}
          <span className="badge">DEV / TEST ONLY</span>
        </p>
      </header>

      <StatusCard
        status={status}
        statusLabel={statusLabel}
        sessionId={sessionId}
        keyboardCount={keyboardCount}
        mouseCount={mouseCount}
        lastEvent={lastEvent}
      />

      <Controls
        status={status}
        onStart={start}
        onStop={stop}
        onDownload={download}
        onClear={clear}
      />

      <TypeZone running={status === "running"} />

      <PrivacyNotice />

      <footer className="footer">
        <p>College project — development interface, not for production use.</p>
      </footer>
    </div>
  );
}