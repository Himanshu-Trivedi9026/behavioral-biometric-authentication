import { Link } from "../../router.jsx";
import LoginStatus from "./LoginStatus.jsx";

export default function LoginSuccess() {
  return (
    <section
      className="lgn-card lgn-success"
      role="status"
      aria-live="polite"
      aria-labelledby="lgn-success-title"
    >
      <div className="lgn-success-icon" aria-hidden="true">
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

      <h1 className="lgn-success-title" id="lgn-success-title">
        Signed in
      </h1>
      <p className="lgn-success-text">
        Authentication successful. Your behavioral enrollment awaits.
      </p>

      <LoginStatus status="success" />

      <Link to="/enrollment" className="lgn-btn lgn-btn-primary lgn-btn-block">
        Continue to Enrollment →
      </Link>
    </section>
  );
}