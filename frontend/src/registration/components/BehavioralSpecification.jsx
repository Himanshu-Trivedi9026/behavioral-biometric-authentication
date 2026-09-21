const HOLD_BARS = [58, 42, 66, 36, 52, 64, 40, 56, 32, 54, 62, 38, 50];
const FLIGHT_BARS = [28, 44, 20, 50, 16, 38, 32, 46, 24, 34, 40, 26, 42];

const TRAJECTORY =
  "M16 58 C 44 52 46 34 68 30 C 90 26 102 44 126 42 C 148 40 150 22 170 26 C 190 30 206 40 224 36";

function KeystrokeVisual() {
  return (
    <div className="reg-visual" aria-hidden="true">
      <div className="reg-visual-row">
        <span className="reg-visual-label reg-mono">HOLD</span>
        <div className="reg-visual-bars">
          {HOLD_BARS.map((height, i) => (
            <span
              key={i}
              className="reg-visual-bar reg-visual-bar--hold"
              style={{ height: `${height}%` }}
            />
          ))}
        </div>
      </div>
      <div className="reg-visual-row">
        <span className="reg-visual-label reg-mono">FLIGHT</span>
        <div className="reg-visual-bars">
          {FLIGHT_BARS.map((height, i) => (
            <span
              key={i}
              className="reg-visual-bar reg-visual-bar--flight"
              style={{ height: `${height}%` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function PointerVisual() {
  return (
    <div className="reg-visual reg-visual--pointer" aria-hidden="true">
      <svg viewBox="0 0 240 76" className="reg-visual-svg">
        <g opacity="0.1">
          {Array.from({ length: 10 }, (_, r) => (
            <line key={`h${r}`} x1="0" y1={r * 8} x2="240" y2={r * 8} stroke="#94A3B8" strokeWidth="0.5" />
          ))}
          {Array.from({ length: 20 }, (_, c) => (
            <line key={`v${c}`} x1={c * 12} y1="0" x2={c * 12} y2="76" stroke="#94A3B8" strokeWidth="0.5" />
          ))}
        </g>
        <path
          d={TRAJECTORY}
          fill="none"
          stroke="#38BDF8"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="16" cy="58" r="3" fill="#22D3EE" />
        <circle cx="224" cy="36" r="2.2" fill="#38BDF8" opacity="0.8" />
      </svg>
    </div>
  );
}

export default function BehavioralSpecification() {
  return (
    <aside className="reg-spec" aria-labelledby="reg-spec-title">
      <h2 className="reg-spec-title" id="reg-spec-title">
        Your behavior becomes part of your authentication profile.
      </h2>
      <p className="reg-spec-intro">
        Behavioral signals are transformed into derived representations that can
        be compared against an enrolled profile.
      </p>

      <div className="reg-spec-cards">
        <article className="reg-spec-card">
          <div className="reg-spec-card-head">
            <span className="reg-spec-card-num reg-mono" aria-hidden="true">
              1
            </span>
            <div>
              <h3 className="reg-spec-card-title">Keystroke Dynamics</h3>
              <p className="reg-spec-card-body">
                Keyboard timing captures hold and flight intervals without recording
                the actual characters you type.
              </p>
            </div>
          </div>
          <KeystrokeVisual />
        </article>

        <article className="reg-spec-card">
          <div className="reg-spec-card-head">
            <span className="reg-spec-card-num reg-mono" aria-hidden="true">
              2
            </span>
            <div>
              <h3 className="reg-spec-card-title">Pointer Trajectory</h3>
              <p className="reg-spec-card-body">
                Mouse movement is represented through positional deltas, timing,
                distance, and speed to capture interaction patterns.
              </p>
            </div>
          </div>
          <PointerVisual />
        </article>

        <article className="reg-spec-card">
          <div className="reg-spec-card-head">
            <span className="reg-spec-card-num reg-mono" aria-hidden="true">
              3
            </span>
            <div>
              <h3 className="reg-spec-card-title">Siamese Verification</h3>
              <p className="reg-spec-card-body">
                A shared CNN + GRU encoder produces behavioral embeddings that are
                compared using L2 distance against the enrolled profile.
              </p>
            </div>
          </div>
        </article>
      </div>

      <div className="reg-privacy">
        <svg className="reg-privacy-icon" viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
          <rect x="4.5" y="8" width="11" height="9" rx="1.6" fill="none" stroke="currentColor" strokeWidth="1.5" />
          <path d="M6.5 8V6.5a3.5 3.5 0 0 1 7 0V8" fill="none" stroke="currentColor" strokeWidth="1.5" />
        </svg>
        <p className="reg-privacy-text">
          Actual key characters are never recorded, stored, or sent. Keyboard
          timing features such as hold time and flight time are used for behavioral
          verification.
        </p>
      </div>
    </aside>
  );
}