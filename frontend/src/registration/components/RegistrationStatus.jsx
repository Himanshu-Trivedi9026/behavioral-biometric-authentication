import RegistrationError from "./RegistrationError.jsx";

const REG_STATES = [
  { id: "normal", label: "Normal" },
  { id: "loading", label: "Loading" },
  { id: "error", label: "Error" },
  { id: "success", label: "Success" },
];

export default function RegistrationStatus({ status, error }) {
  const active = status === "idle" ? "normal" : status;

  return (
    <div className="reg-status-strip" role="status" aria-live="polite" aria-atomic="true">
      <div className="reg-status-strip-row">
        <span className="reg-status-strip-label reg-mono">REGISTRATION STATUS:</span>
        <div className="reg-status-strip-chips">
          {REG_STATES.map((state) => (
            <span
              key={state.id}
              className={
                state.id === active
                  ? `reg-status-chip reg-status-chip--active reg-status-chip--${state.id}`
                  : "reg-status-chip"
              }
            >
              {state.label}
            </span>
          ))}
        </div>
      </div>
      {status === "error" && error && (
        <div className="reg-status-strip-error">
          <RegistrationError message={error} />
        </div>
      )}
    </div>
  );
}