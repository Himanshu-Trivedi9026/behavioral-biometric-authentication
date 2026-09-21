const VECTOR_BUCKETS = 128;

function deterministicVector() {
  const heights = [];
  for (let i = 0; i < VECTOR_BUCKETS; i += 1) {
    heights.push((i * 37) % 72);
  }
  return heights;
}

export default function EnrollmentSpecification() {
  const heights = deterministicVector();

  return (
    <aside className="enr-spec" aria-label="How behavioral enrollment works">
      <h2 className="enr-spec-title">
        Behavioral enrollment, <br />
        server-side and private
      </h2>
      <p className="enr-spec-intro">
        Your typing rhythm and mouse movement are turned into a compact profile —
        without ever recording what you typed.
      </p>

      <div className="enr-spec-cards">
        <div className="enr-spec-card">
          <div className="enr-spec-card-head">
            <span className="enr-spec-card-num enr-mono">01</span>
            <div>
              <h3 className="enr-spec-card-title">Sessions are real behavior</h3>
              <p className="enr-spec-card-body">
                Finish each session while genuinely typing and moving the mouse.
                Only press/release timings and pointer coordinates are collected —
                no key identities, text, or screenshots.
              </p>
            </div>
          </div>
        </div>

        <div className="enr-spec-card">
          <div className="enr-spec-card-head">
            <span className="enr-spec-card-num enr-mono">02</span>
            <div>
              <h3 className="enr-spec-card-title">Aggregate profile only</h3>
              <p className="enr-spec-card-body">
                Sessions are preprocessed and encoded into a 128-D embedding. The
                server stores only the aggregate centroid; raw events are never
                persisted or echoed back.
              </p>
            </div>
          </div>

          <div className="enr-vector-wrap">
            <div className="enr-vector" aria-hidden="true">
              {heights.map((height, index) => (
                <span
                  key={index}
                  className="enr-vector-bar"
                  style={{ height: height + "%" }}
                />
              ))}
            </div>
            <span className="enr-vector-cap enr-mono">
              128-D EMBEDDING SPACE · {heights.length} DIMENSIONS
            </span>
          </div>
        </div>

        <div className="enr-spec-card">
          <div className="enr-spec-card-head">
            <span className="enr-spec-card-num enr-mono">03</span>
            <div>
              <h3 className="enr-spec-card-title">Protected API workflow</h3>
              <p className="enr-spec-card-body">
                Enrollment sends your sessions once over an authenticated request.
                The response returns only metadata — status, embedding dimension,
                and how many sessions were used.
              </p>
            </div>
          </div>

          <div className="enr-siamese">
            <div className="enr-flow">
              <span className="enr-flow-label enr-mono">INPUT</span>
              <span className="enr-flow-node">RAW SESSIONS</span>
              <span className="enr-flow-branch enr-mono">1–16</span>
            </div>
            <div className="enr-flow-merge" aria-hidden="true">
              ↓
            </div>
            <div className="enr-flow">
              <span className="enr-flow-label enr-mono">PIPELINE</span>
              <span className="enr-flow-node">PREPROCESS → CNN+GRU</span>
            </div>
            <div className="enr-flow-merge" aria-hidden="true">
              ↓
            </div>
            <div className="enr-flow">
              <span className="enr-flow-label enr-mono">OUTPUT</span>
              <span className="enr-flow-result">PROFILE · ENROLLED</span>
            </div>
          </div>
        </div>
      </div>

      <div className="enr-privacy">
        <svg
          viewBox="0 0 24 24"
          width="22"
          height="22"
          className="enr-privacy-icon"
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
          <p className="enr-privacy-title">Privacy: no key identity logging</p>
          <p className="enr-privacy-text">
            This system records behavioral metadata only. Typed text, input
            values, and clipboard content are never read or stored.
          </p>
        </div>
      </div>
    </aside>
  );
}