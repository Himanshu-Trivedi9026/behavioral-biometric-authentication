export default function VerificationResult({ result, onRepeat }) {
  const verified = result && result.decision === "VERIFIED";

  return (
    <section
      className="vfy-card vfy-result"
      role="status"
      aria-live="polite"
      aria-labelledby="vfy-result-title"
    >
      <div
        className={
          verified ? "vfy-decision vfy-decision--verified" : "vfy-decision vfy-decision--suspicious"
        }
        aria-hidden="true"
      >
        <svg viewBox="0 0 24 24" width="34" height="34">
          {verified ? (
            <path
              d="M5 12.5l4.5 4.5L19 7.5"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ) : (
            <path
              d="M6 6l12 12M18 6L6 18"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
            />
          )}
        </svg>
      </div>

      <h1 className="vfy-result-title" id="vfy-result-title">
        {verified ? "Identity verified" : "Verification failed"}
      </h1>
      <p className="vfy-result-text">
        {verified
          ? "Your behavior matches the stored profile within the calibrated threshold."
          : "Your behavior did not match the stored profile within the calibrated threshold."}
      </p>

      {result && (
        <dl className="vfy-result-meta">
          <div className="vfy-result-meta-row">
            <dt>DECISION</dt>
            <dd className={verified ? "vfy-decision-value vfy-decision-value--verified" : "vfy-decision-value vfy-decision-value--suspicious"}>
              {result.decision}
            </dd>
          </div>
          <div className="vfy-result-meta-row">
            <dt>DISTANCE</dt>
            <dd>{Number(result.distance).toFixed(4)}</dd>
          </div>
          <div className="vfy-result-meta-row">
            <dt>CALIBRATED THRESHOLD</dt>
            <dd>{Number(result.threshold).toFixed(4)}</dd>
          </div>
          <div className="vfy-result-meta-row">
            <dt>USER_REF</dt>
            <dd>{result.user_ref}</dd>
          </div>
        </dl>
      )}

      <p className="vfy-result-note">
        The threshold is calibrated server-side and is never client-supplied.
        Distance ≤ threshold verifies; distance &gt; threshold is suspicious.
      </p>

      <button type="button" className="vfy-btn vfy-btn-ghost vfy-btn-block" onClick={onRepeat}>
        Verify Again
      </button>
    </section>
  );
}