import { CONTINUOUS_STATES } from "../../lib/continuous.js";

export default function ContinuousResult({ state, lastDecision, onRepeat }) {
  const verified = state === CONTINUOUS_STATES.VERIFIED;
  const suspicious = state === CONTINUOUS_STATES.REVERIFICATION_REQUIRED;

  return (
    <section
      className="ctn-card ctn-result"
      role="status"
      aria-live="polite"
      aria-labelledby="ctn-result-title"
    >
      <div
        className={
          verified
            ? "ctn-decision ctn-decision--verified"
            : "ctn-decision ctn-decision--reverify"
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

      <h1 className="ctn-result-title" id="ctn-result-title">
        {verified ? "Identity stays verified" : "Re-verification required"}
      </h1>
      <p className="ctn-result-text">
        {verified
          ? "This behavioral window matched your stored profile within the calibrated threshold."
          : "This behavioral window did not match your stored profile. Record another window to re-verify."}
      </p>

      {lastDecision && (
        <dl className="ctn-result-meta">
          <div className="ctn-result-meta-row">
            <dt>DECISION</dt>
            <dd
              className={
                verified
                  ? "ctn-decision-value ctn-decision-value--verified"
                  : "ctn-decision-value ctn-decision-value--reverify"
              }
            >
              {lastDecision.decision}
            </dd>
          </div>
          {Number.isFinite(Number(lastDecision.distance)) && (
            <div className="ctn-result-meta-row">
              <dt>DISTANCE</dt>
              <dd>{Number(lastDecision.distance).toFixed(4)}</dd>
            </div>
          )}
          {Number.isFinite(Number(lastDecision.threshold)) && (
            <div className="ctn-result-meta-row">
              <dt>CALIBRATED THRESHOLD</dt>
              <dd>{Number(lastDecision.threshold).toFixed(4)}</dd>
            </div>
          )}
        </dl>
      )}

      <p className="ctn-result-note">
        The threshold is calibrated server-side and is never client-supplied.
        Distance ≤ threshold verifies; distance &gt; threshold requests
        re-verification. The raw window was discarded after this decision.
      </p>

      <button type="button" className="ctn-btn ctn-btn-ghost ctn-btn-block" onClick={onRepeat}>
        {suspicious ? "Record Re-verification Window" : "Start Next Window"}
      </button>
    </section>
  );
}