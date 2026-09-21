const EMBEDDING_VECTOR = Array.from({ length: 128 }, (_, i) => 24 + ((i * 37) % 72));

function LockIcon() {
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
      <rect
        x="5"
        y="11"
        width="14"
        height="9"
        rx="2"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
      />
      <path d="M8 11V7a4 4 0 0 1 8 0v4" fill="none" stroke="currentColor" strokeWidth="1.8" />
    </svg>
  );
}

function VectorVisual() {
  return (
    <div className="lgn-vector-wrap" aria-hidden="true">
      <div className="lgn-vector">
        {EMBEDDING_VECTOR.map((height, i) => (
          <span key={i} className="lgn-vector-bar" style={{ height: `${height}%` }} />
        ))}
      </div>
      <span className="lgn-vector-cap lgn-mono">128-D EMBEDDING</span>
    </div>
  );
}

function SiameseVisual() {
  return (
    <div className="lgn-siamese" aria-hidden="true">
      <div className="lgn-flow">
        <span className="lgn-flow-label lgn-mono">REFERENCE</span>
        <span className="lgn-flow-node lgn-mono">CNN + GRU</span>
        <span className="lgn-flow-label lgn-mono">128-D EMBEDDING</span>
      </div>
      <div className="lgn-flow">
        <span className="lgn-flow-label lgn-mono">LIVE SESSION</span>
        <span className="lgn-flow-node lgn-mono">CNN + GRU</span>
        <span className="lgn-flow-label lgn-mono">128-D EMBEDDING</span>
      </div>
      <div className="lgn-flow-merge">↓</div>
      <div className="lgn-flow-result lgn-mono">L2 DISTANCE COMPARISON</div>
    </div>
  );
}

export default function BehavioralSpecification() {
  return (
    <aside className="lgn-spec" aria-labelledby="lgn-spec-title">
      <h2 className="lgn-spec-title" id="lgn-spec-title">
        Authentication goes beyond your password.
      </h2>
      <p className="lgn-spec-intro">
        Your password confirms your account. Behavioral verification adds another
        layer by comparing how you type and interact against your enrolled
        behavioral profile.
      </p>

      <div className="lgn-spec-cards">
        <article className="lgn-spec-card">
          <div className="lgn-spec-card-head">
            <span className="lgn-spec-card-num lgn-mono" aria-hidden="true">
              01
            </span>
            <div>
              <h3 className="lgn-spec-card-title">Account Authentication</h3>
              <p className="lgn-spec-card-body">
                Your credentials are verified through the protected authentication
                API before behavioral verification begins.
              </p>
            </div>
          </div>
        </article>

        <article className="lgn-spec-card">
          <div className="lgn-spec-card-head">
            <span className="lgn-spec-card-num lgn-mono" aria-hidden="true">
              02
            </span>
            <div>
              <h3 className="lgn-spec-card-title">Behavioral Profile</h3>
              <p className="lgn-spec-card-body">
                Your enrolled profile is represented as a learned behavioral
                embedding rather than stored keystrokes.
              </p>
            </div>
          </div>
          <VectorVisual />
        </article>

        <article className="lgn-spec-card">
          <div className="lgn-spec-card-head">
            <span className="lgn-spec-card-num lgn-mono" aria-hidden="true">
              03
            </span>
            <div>
              <h3 className="lgn-spec-card-title">Siamese Verification</h3>
              <p className="lgn-spec-card-body">
                A shared CNN + GRU encoder compares new behavioral activity against
                the enrolled profile using L2 distance.
              </p>
            </div>
          </div>
          <SiameseVisual />
        </article>
      </div>

      <div className="lgn-privacy">
        <span className="lgn-privacy-icon" aria-hidden="true">
          <LockIcon />
        </span>
        <div>
          <h3 className="lgn-privacy-title">Privacy-first authentication</h3>
          <p className="lgn-privacy-text">
            Actual key characters are never recorded, stored, or sent. Behavioral
            verification uses derived timing and movement features.
          </p>
        </div>
      </div>
    </aside>
  );
}