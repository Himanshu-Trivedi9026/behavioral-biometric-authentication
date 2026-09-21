const VERIFICATION_STATES = [
  { id: "normal", label: "Normal" },
  { id: "loading", label: "Loading" },
  { id: "error", label: "Error" },
  { id: "success", label: "Success" },
];

export default function VerificationStatus({ phase, error }) {
  const active = phase === "probe" ? "normal" : phase;

  return (
    <div className="vfy-status-strip" role="status" aria-live="polite" aria-atomic="true">
      <div className="vfy-status-strip-row">
        <span className="vfy-status-strip-label vfy-mono">VERIFICATION STATUS:</span>
        <div className="vfy-status-strip-chips">
          {VERIFICATION_STATES.map((state) => (
            <span
              key={state.id}
              className={
                state.id === active
                  ? `vfy-status-chip vfy-status-chip--active vfy-status-chip--${state.id}`
                  : "vfy-status-chip"
              }
            >
              {state.label}
            </span>
          ))}
        </div>
      </div>
      {phase === "error" && error && (
        <div className="vfy-status-strip-error">
          <svg viewBox="0 0 20 20" width="18" height="18" className="vfy-error-icon" aria-hidden="true">
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