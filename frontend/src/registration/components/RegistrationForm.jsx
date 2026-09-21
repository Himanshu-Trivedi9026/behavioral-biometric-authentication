import { useState } from "react";
import { Link } from "../../router.jsx";
import RegistrationStatus from "./RegistrationStatus.jsx";

const USERNAME_PATTERN = /^[A-Za-z0-9._-]+$/;

function validate({ username, password, confirm }) {
  const errors = {};
  const name = username.trim();

  if (!name) {
    errors.username = "Username is required.";
  } else if (name.length < 3 || name.length > 64) {
    errors.username = "Username must be between 3 and 64 characters.";
  } else if (!USERNAME_PATTERN.test(name)) {
    errors.username = "Username can only contain letters, numbers, and . _ - characters.";
  }

  if (!password) {
    errors.password = "Password is required.";
  } else if (password.length < 8) {
    errors.password = "Password must be at least 8 characters.";
  } else if (password.length > 128) {
    errors.password = "Password must be at most 128 characters.";
  }

  if (!confirm) {
    errors.confirm = "Please confirm your password.";
  } else if (confirm !== password) {
    errors.confirm = "Passwords do not match.";
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

function PasswordField({
  id,
  label,
  value,
  onChange,
  error,
  help,
  visible,
  onToggleVisible,
  disabled,
}) {
  const errorId = error ? `${id}-error` : undefined;
  const helpId = help ? `${id}-help` : undefined;
  const describedBy = [errorId, helpId].filter(Boolean).join(" ");
  return (
    <div className="reg-field">
      <label className="reg-field-label" htmlFor={id}>
        {label}
      </label>
      <div className="reg-input-wrap">
        <input
          id={id}
          className="reg-input"
          type={visible ? "text" : "password"}
          autoComplete="new-password"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy || undefined}
          disabled={disabled}
          required
        />
        <button
          type="button"
          className="reg-visibility-toggle"
          aria-label={visible ? `Hide ${label.toLowerCase()}` : `Show ${label.toLowerCase()}`}
          aria-pressed={visible}
          onClick={onToggleVisible}
          disabled={disabled}
        >
          <EyeIcon hidden={visible} />
        </button>
      </div>
      {errorId && (
        <p className="reg-field-error" id={errorId}>
          {error}
        </p>
      )}
      {helpId && (
        <p className="reg-field-help" id={helpId}>
          {help}
        </p>
      )}
    </div>
  );
}

export default function RegistrationForm({ status, error, onSubmit }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [fieldErrors, setFieldErrors] = useState({});

  const loading = status === "loading";

  function handleSubmit(event) {
    event.preventDefault();
    if (loading) return;
    const errors = validate({ username, password, confirm });
    setFieldErrors(errors);
    if (Object.keys(errors).length > 0) {
      return;
    }
    setFieldErrors({});
    onSubmit(username.trim(), password);
  }

  return (
    <section className="reg-card" aria-labelledby="reg-card-title">
      <div className="reg-step">
        <div>
          <p className="reg-step-kicker reg-mono">STEP 1 OF 2</p>
          <h1 className="reg-step-name" id="reg-card-title">
            Create Account
          </h1>
        </div>
        <p className="reg-step-next reg-mono">NEXT: ENROLLMENT</p>
      </div>
      <p className="reg-step-hint">Behavioral enrollment follows after sign-in.</p>

      <RegistrationStatus status={status} error={error} />

      <form className="reg-form" onSubmit={handleSubmit} noValidate aria-busy={loading}>
        <div className="reg-field">
          <label className="reg-field-label" htmlFor="reg-username">
            Username
          </label>
          <input
            id="reg-username"
            className="reg-input"
            type="text"
            autoComplete="username"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            aria-invalid={fieldErrors.username ? true : undefined}
            aria-describedby={[
              "reg-username-help",
              fieldErrors.username && "reg-username-error",
            ]
              .filter(Boolean)
              .join(" ")}
            disabled={loading}
            required
          />
          {fieldErrors.username && (
            <p className="reg-field-error" id="reg-username-error">
              {fieldErrors.username}
            </p>
          )}
          <p className="reg-field-help" id="reg-username-help">
            Use a unique identifier for your authentication profile.
          </p>
        </div>

        <PasswordField
          id="reg-password"
          label="Password"
          value={password}
          onChange={setPassword}
          error={fieldErrors.password}
          help="Argon2id cryptographic hashing. Use a strong password you do not reuse elsewhere."
          visible={showPassword}
          onToggleVisible={() => setShowPassword((value) => !value)}
          disabled={loading}
        />

        <PasswordField
          id="reg-confirm"
          label="Confirm Password"
          value={confirm}
          onChange={setConfirm}
          error={fieldErrors.confirm}
          help="Re-enter the same password to confirm it matches."
          visible={showConfirm}
          onToggleVisible={() => setShowConfirm((value) => !value)}
          disabled={loading}
        />

        <button
          type="submit"
          className="reg-btn reg-btn-primary reg-btn-block"
          disabled={loading}
        >
          {loading ? (
            <>
              <span className="reg-spinner" aria-hidden="true" />
              Creating account...
            </>
          ) : (
            "Create Account"
          )}
        </button>

        <p className="reg-form-login">
          Already have an account?{" "}
          <Link to="/login" className="reg-form-login-link">
            Log In
          </Link>
        </p>
      </form>
    </section>
  );
}