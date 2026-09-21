import { Link } from "../../router.jsx";
import RegistrationStatus from "./RegistrationStatus.jsx";

export default function RegistrationSuccess({ user }) {
  return (
    <section
      className="reg-card reg-success"
      role="status"
      aria-live="polite"
      aria-labelledby="reg-success-title"
    >
      <div className="reg-success-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" width="34" height="34">
          <path
            d="M5 12.5l4.5 4.5L19 7.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>

      <h1 className="reg-success-title" id="reg-success-title">
        Account created
      </h1>
      <p className="reg-success-text">
        Your account is ready. Continue to sign in.
      </p>
      {user && <p className="reg-success-user reg-mono">USER: {user.username}</p>}

      <RegistrationStatus status="success" />

      <Link to="/login" className="reg-btn reg-btn-primary reg-btn-block">
        Continue to Log In
      </Link>
    </section>
  );
}