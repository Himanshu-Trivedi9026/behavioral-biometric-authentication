export default function ContinuousSpecification() {
  return (
    <aside className="ctn-spec" aria-label="How periodic continuous verification works">
      <h2 className="ctn-spec-title">
        Periodic re-checks, <br />
        while you stay signed in
      </h2>
      <p className="ctn-spec-intro">
        Each short behavioral window is matched against your stored profile
        using the same calibrated threshold as one-shot verification. Your
        identity stays VERIFIED when windows keep landing within it.
      </p>

      <div className="ctn-spec-cards">
        <div className="ctn-spec-card">
          <div className="ctn-spec-card-head">
            <span className="ctn-spec-card-num ctn-mono">01</span>
            <div>
              <h3 className="ctn-spec-card-title">Bounded windows</h3>
              <p className="ctn-spec-card-body">
                A window captures a few seconds of typing and mouse movement and
                auto-closes at its event/time cap. Windows are never stored —
                they live in memory, are sent once, and are discarded.
              </p>
            </div>
          </div>
        </div>

        <div className="ctn-spec-card">
          <div className="ctn-spec-card-head">
            <span className="ctn-spec-card-num ctn-mono">02</span>
            <div>
              <h3 className="ctn-spec-card-title">Same decision rule</h3>
              <p className="ctn-spec-card-body">
                The window is embedded and measured against the stored centroid.
                If the L2 distance lands within the calibrated threshold the
                decision is VERIFIED; otherwise RE-VERIFICATION REQUIRED.
              </p>
            </div>
          </div>

          <div className="ctn-flow" aria-hidden="true">
            <div className="ctn-flow-row">
              <span className="ctn-flow-label ctn-mono">WINDOW</span>
              <span className="ctn-flow-node">PREPROCESS → CNN+GRU → L2</span>
              <span className="ctn-flow-result">VERIFIED / RE-VERIFY</span>
            </div>
          </div>
        </div>

        <div className="ctn-spec-card">
          <div className="ctn-spec-card-head">
            <span className="ctn-spec-card-num ctn-mono">03</span>
            <div>
              <h3 className="ctn-spec-card-title">Identity from your JWT</h3>
              <p className="ctn-spec-card-body">
                The window body carries behavioral data only — no identity
                fields. The server selects your profile from your access token,
                so no other profile can be queried, and returns only the
                decision, distance, and threshold.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="ctn-privacy">
        <svg
          viewBox="0 0 24 24"
          width="22"
          height="22"
          className="ctn-privacy-icon"
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
          <p className="ctn-privacy-title">Privacy: windows never persist</p>
          <p className="ctn-privacy-text">
            Raw windows are not written to localStorage or sessionStorage, and
            typed text, input values, and clipboard content are never read or
            stored.
          </p>
        </div>
      </div>
    </aside>
  );
}