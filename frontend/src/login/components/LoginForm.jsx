import { useState } from "react";
import { Link } from "../../router.jsx";
import LoginStatus from "./LoginStatus.jsx";

function validate({ username, password }) {
  const errors = {};
  if (!username.trim()) {
    errors.username = "Enter your username.";
  }
  if (!password) {
    errors.password = "Enter your password.";
  }
  return errors;
}

function EyeIcon({ hidden }) {
  return (
    <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
      <g stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" fill="none">
        <path d="M3 10c1.6-3.2 4.3-4.8 7-4.8s5.4 1.6 7 4.8" />
        <path d="M3 10c1.6 3.2 4.3 4.8 7 4.8s5.4-1.6 7-4.8" />
        <circle cx="10" cy="10" r="2.4" />
      </g>
      {hidden && (
        <path d="M4 15L16 4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      )}
    </svg>
  );
}

function PasswordField({ value, onChange, error, disabled }) {
  const [visible, setVisible] = useState(false);
  const errorId = error ? "lgn-password-error" : undefined;
  const describedBy = ["lgn-password-help", errorId].filter(Boolean).join(" ");
  return (
    <div className="lgn-field">
      <label className="lgn-field-label" htmlFor="lgn-password">
        PASSWORD *
      </label>
      <div className="lgn-input-wrap">
        <input
          id="lgn-password"
          className="lgn-input"
          type={visible ? "text" : "password"}
          placeholder="Enter your password"
          autoComplete="current-password"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy || undefined}
          disabled={disabled}
          required
        />
        <button
          type="button"
          className="lgn-visibility-toggle"
          aria-label={visible ? "Hide password" : "Show password"}
          aria-pressed={visible}
          onClick={() => setVisible((value) => !value)}
          disabled={disabled}
        >
          <EyeIcon hidden={visible} />
        </button>
      </div>
      {errorId && (
        <p className="lgn-field-error" id={errorId}>
          {error}
        </p>
      )}
      <p className="lgn-field-help" id="lgn-password-help">
        Your password is verified securely using Argon2id.
      </p>
    </div>
  );
}

export default function LoginForm({ status, error, onSubmit }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [fieldErrors, setFieldErrors] = useState({});

  const loading = status === "loading";

  function handleSubmit(event) {
    event.preventDefault();
    if (loading) return;
    const errors = validate({ username, password });
    setFieldErrors(errors);
    if (Object.keys(errors).length > 0) {
      return;
    }
    setFieldErrors({});
    onSubmit(username.trim(), password);
  }

  return (
    <section className="lgn-card" aria-labelledby="lgn-card-title">
      <p className="lgn-gateway-kicker lgn-mono">AUTHENTICATION GATEWAY · SIGN IN</p>
      <h1 className="lgn-card-title" id="lgn-card-title">
        Welcome back
      </h1>
      <p className="lgn-welcome-sub">
        Sign in to continue to behavioral biometric authentication.
      </p>

      <div className="lgn-step">
        <p className="lgn-step-current lgn-mono">STEP 1 · ACCOUNT SIGN IN</p>
        <p className="lgn-step-next lgn-mono">NEXT: BEHAVIORAL VERIFICATION</p>
      </div>

      <LoginStatus status={status} error={error} />

      <form className="lgn-form" onSubmit={handleSubmit} noValidate aria-busy={loading}>
        <div className="lgn-field">
          <label className="lgn-field-label" htmlFor="lgn-username">
            USERNAME *
          </label>
          <input
            id="lgn-username"
            className="lgn-input"
            type="text"
            placeholder="Enter your username"
            autoComplete="username"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            aria-invalid={fieldErrors.username ? true : undefined}
            aria-describedby={[
              "lgn-username-help",
              fieldErrors.username && "lgn-username-error",
            ]
              .filter(Boolean)
              .join(" ")}
            disabled={loading}
            required
          />
          {fieldErrors.username && (
            <p className="lgn-field-error" id="lgn-username-error">
              {fieldErrors.username}
            </p>
          )}
          <p className="lgn-field-help" id="lgn-username-help">
            Use the username associated with your behavioral profile.
          </p>
        </div>

        <PasswordField value={password} onChange={setPassword} error={fieldErrors.password} disabled={loading} />

        <div className="lgn-account-row">
          <label className="lgn-remember">
            <input type="checkbox" className="lgn-remember-input" disabled />
            <span>Remember this device</span>
          </label>
          <span className="lgn-forgot-unavailable" aria-disabled="true">
            Forgot password? (Unavailable)
          </span>
        </div>

        <button type="submit" className="lgn-btn lgn-btn-primary lgn-btn-block" disabled={loading}>
          {loading ? (
            <>
              <span className="lgn-spinner" aria-hidden="true" />
              Signing in...
            </>
          ) : (
            "Log In →"
          )}
        </button>

        <p className="lgn-form-register">
          Don't have an account?{" "}
          <Link to="/register" className="lgn-form-register-link">
            Create Account
          </Link>
        </p>
      </form>
    </section>
  );
}