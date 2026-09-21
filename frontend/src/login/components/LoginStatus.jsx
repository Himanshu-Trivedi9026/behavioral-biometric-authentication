import LoginError from "./LoginError.jsx";

const LOGIN_STATES = [
  { id: "normal", label: "Normal" },
  { id: "loading", label: "Loading" },
  { id: "error", label: "Error" },
  { id: "success", label: "Success" },
];

export default function LoginStatus({ status, error }) {
  const active = status === "idle" ? "normal" : status;

  return (
    <div className="lgn-status-strip" role="status" aria-live="polite" aria-atomic="true">
      <div className="lgn-status-strip-row">
        <span className="lgn-status-strip-label lgn-mono">LOGIN STATUS:</span>
        <div className="lgn-status-strip-chips">
          {LOGIN_STATES.map((state) => (
            <span
              key={state.id}
              className={
                state.id === active
                  ? `lgn-status-chip lgn-status-chip--active lgn-status-chip--${state.id}`
                  : "lgn-status-chip"
              }
            >
              {state.label}
            </span>
          ))}
        </div>
      </div>
      {status === "error" && error && (
        <div className="lgn-status-strip-error">
          <LoginError message={error} />
        </div>
      )}
    </div>
  );
}