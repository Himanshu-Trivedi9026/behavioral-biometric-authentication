const VECTOR_BUCKETS = 128;

function deterministicVector() {
  const heights = [];
  for (let i = 0; i < VECTOR_BUCKETS; i += 1) {
    heights.push((i * 37) % 72);
  }
  return heights;
}

export default function VerificationSpecification() {
  const heights = deterministicVector();

  return (
    <aside className="vfy-spec" aria-label="How behavioral verification works">
      <h2 className="vfy-spec-title">
        One probe, <br />
        compared server-side
      </h2>
      <p className="vfy-spec-intro">
        A single behavioral probe is embedded and compared with your stored
        profile using distance — the raw probe never leaves the request.
      </p>

      <div className="vfy-spec-cards">
        <div className="vfy-spec-card">
          <div className="vfy-spec-card-head">
            <span className="vfy-spec-card-num vfy-mono">01</span>
            <div>
              <h3 className="vfy-spec-card-title">One genuine probe</h3>
              <p className="vfy-spec-card-body">
                A probe is one short session of real typing and mouse movement.
                No key identities, text content, or screenshots are involved.
              </p>
            </div>
          </div>
        </div>

        <div className="vfy-spec-card">
          <div className="vfy-spec-card-head">
            <span className="vfy-spec-card-num vfy-mono">02</span>
            <div>
              <h3 className="vfy-spec-card-title">Distance vs threshold</h3>
              <p className="vfy-spec-card-body">
                The probe embedding is measured against the stored 128-D centroid.
                If the L2 distance lands within the calibrated threshold, the
                identity is verified.
              </p>
            </div>
          </div>

          <div className="vfy-vector-wrap">
            <div className="vfy-vector" aria-hidden="true">
              {heights.map((height, index) => (
                <span key={index} className="vfy-vector-bar" style={{ height: height + "%" }} />
              ))}
            </div>
            <span className="vfy-vector-cap vfy-mono">
              128-D EMBEDDING SPACE · L2 DISTANCE
            </span>
          </div>
        </div>

        <div className="vfy-spec-card">
          <div className="vfy-spec-card-head">
            <span className="vfy-spec-card-num vfy-mono">03</span>
            <div>
              <h3 className="vfy-spec-card-title">Protected verification</h3>
              <p className="vfy-spec-card-body">
                Verification is authenticated and returns only the decision
                (VERIFIED or SUSPICIOUS), distance, and the server-side calibrated
                threshold — never raw events or embeddings.
              </p>
            </div>
          </div>

          <div className="vfy-siamese">
            <div className="vfy-flow">
              <span className="vfy-flow-label vfy-mono">INPUT</span>
              <span className="vfy-flow-node">PROBE SESSION</span>
            </div>
            <div className="vfy-flow-merge" aria-hidden="true">
              ↓
            </div>
            <div className="vfy-flow">
              <span className="vfy-flow-label vfy-mono">PIPELINE</span>
              <span className="vfy-flow-node">PREPROCESS → CNN+GRU → L2</span>
            </div>
            <div className="vfy-flow-merge" aria-hidden="true">
              ↓
            </div>
            <div className="vfy-flow">
              <span className="vfy-flow-label vfy-mono">OUTPUT</span>
              <span className="vfy-flow-result">VERIFIED / SUSPICIOUS</span>
            </div>
          </div>
        </div>
      </div>

      <div className="vfy-privacy">
        <svg
          viewBox="0 0 24 24"
          width="22"
          height="22"
          className="vfy-privacy-icon"
          aria-hidden="true"
        >
          <path
            d="M12 3l7 3v5c0 4.4-3 7.7-7 9-4-1.3-7-4.6-7-9V6l7-3Z"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinejoin="round"
          />
          <path
            d="M9.5 12l1.8 1.8 3.4-3.6"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <div>
          <p className="vfy-privacy-title">Privacy: no key identity logging</p>
          <p className="vfy-privacy-text">
            This system records behavioral metadata only. Typed text, input
            values, and clipboard content are never read or stored.
          </p>
        </div>
      </div>
    </aside>
  );
}