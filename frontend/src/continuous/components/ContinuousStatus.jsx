import { CONTINUOUS_STATES } from "../../lib/continuous.js";

const CONTINUOUS_STATUS_STATES = [
  { id: CONTINUOUS_STATES.AUTHENTICATED, label: "Authenticated", kind: "normal" },
  { id: CONTINUOUS_STATES.COLLECTING, label: "Collecting", kind: "loading" },
  { id: CONTINUOUS_STATES.VERIFYING, label: "Verifying", kind: "loading" },
  { id: CONTINUOUS_STATES.VERIFIED, label: "Verified", kind: "success" },
  { id: CONTINUOUS_STATES.REVERIFICATION_REQUIRED, label: "Re-verify", kind: "reverify" },
];

export default function ContinuousStatus({ state, error, capReason }) {
  const activeEntry = CONTINUOUS_STATUS_STATES.find((entry) => entry.id === state);

  return (
    <div className="ctn-status-strip" role="status" aria-live="polite" aria-atomic="true">
      <div className="ctn-status-strip-row">
        <span className="ctn-status-strip-label ctn-mono">CONTINUOUS STATUS:</span>
        <div className="ctn-status-strip-chips">
          {CONTINUOUS_STATUS_STATES.map((entry) => (
            <span
              key={entry.id}
              className={
                entry.id === state
                  ? `ctn-status-chip ctn-status-chip--active ctn-status-chip--${entry.kind}`
                  : "ctn-status-chip"
              }
            >
              {entry.label}
            </span>
          ))}
        </div>
      </div>
      {activeEntry && capReason && state === CONTINUOUS_STATES.COLLECTING && (
        <p className="ctn-status-strip-note ctn-mono">CAP: {capReason} — window closed automatically.</p>
      )}
      {state === CONTINUOUS_STATES.REVERIFICATION_REQUIRED && (
        <p className="ctn-status-strip-note ctn-status-strip-note--reverify">
          The last window was suspicious. Record another behavioral window to re-verify your identity.
        </p>
      )}
      {error && (
        <div className="ctn-status-strip-error">
          <svg viewBox="0 0 20 20" width="18" height="18" className="ctn-error-icon" aria-hidden="true">
            <path
              d="M10 2.5a7.5 7.5 0 1 0 0 15 7.5 7.5 0 0 0 0-15Z"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            />
            <path d="M10 6.5v4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            <circle cx="10" cy="13.4" r="0.9" fill="currentColor" />
          </svg>
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}