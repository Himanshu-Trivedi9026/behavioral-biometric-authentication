import { Link } from "../../router.jsx";

export default function EnrollmentSuccess({ result }) {
  return (
    <section
      className="enr-card enr-success"
      role="status"
      aria-live="polite"
      aria-labelledby="enr-success-title"
    >
      <div className="enr-success-icon" aria-hidden="true">
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

      <h1 className="enr-success-title" id="enr-success-title">
        Profile enrolled
      </h1>
      <p className="enr-success-text">
        Your behavioral profile was created from the captured sessions. The
        server stored only the aggregate profile, not your raw events.
      </p>

      {result && (
        <dl className="enr-success-meta">
          <div className="enr-success-meta-row">
            <dt>USER_REF</dt>
            <dd>{result.user_ref}</dd>
          </div>
          <div className="enr-success-meta-row">
            <dt>STATUS</dt>
            <dd>{result.status}</dd>
          </div>
          <div className="enr-success-meta-row">
            <dt>EMBEDDING_DIM</dt>
            <dd>{result.embedding_dimension}</dd>
          </div>
          <div className="enr-success-meta-row">
            <dt>SESSION_COUNT</dt>
            <dd>{result.session_count}</dd>
          </div>
        </dl>
      )}

      <Link to="/verification" className="enr-btn enr-btn-primary enr-btn-block">
        Continue to Verification →
      </Link>
    </section>
  );
}