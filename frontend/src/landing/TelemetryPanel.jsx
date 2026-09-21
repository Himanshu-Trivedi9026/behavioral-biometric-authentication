const HOLD_HEIGHTS = [
  62, 45, 70, 38, 55, 68, 42, 60, 34, 58, 66, 40, 53, 72, 36, 64, 48, 56, 60, 43,
];

const FLIGHT_HEIGHTS = [
  30, 48, 22, 55, 18, 42, 34, 50, 26, 38, 44, 28, 46, 32, 52, 24, 40, 36, 48, 30,
];

function hash(i) {
  return Math.abs(Math.sin(i * 12.9898 + 78.233) * 43758.5453 % 1);
}

const EMBEDDING_CELLS = Array.from({ length: 64 }, (_, i) => hash(i));

const TRAJECTORY = "M18 118 C58 92 48 50 105 58 C145 64 130 32 182 40 C226 47 220 22 268 42 C305 60 298 96 246 102 C200 108 212 128 172 122 C140 118 150 138 118 130";

const TRAJECTORY_NODES = [
  { cx: 18, cy: 118 },
  { cx: 105, cy: 58 },
  { cx: 182, cy: 40 },
  { cx: 268, cy: 42 },
  { cx: 246, cy: 102 },
  { cx: 118, cy: 130 },
];

export default function TelemetryPanel() {
  return (
    <aside className="lnd-tele" aria-label="Static behavioral analysis preview">
      <div className="lnd-tele-head">
        <div>
          <p className="lnd-tele-title lnd-mono">BEHAVIORAL ANALYSIS PREVIEW</p>
          <p className="lnd-tele-sub">
            Static demo visualisation — this page never collects your activity.
          </p>
        </div>
        <div className="lnd-tele-badges">
          <span className="lnd-badge lnd-badge-warn">DEMO SESSION</span>
          <span className="lnd-badge lnd-badge-info">SIMULATED</span>
        </div>
      </div>

      <div className="lnd-tele-grid">
        <div className="lnd-tele-tile">
          <p className="lnd-tile-label lnd-mono">
            MOUSE TRAJECTORY <span className="lnd-tile-tag">demo</span>
          </p>
          <div className="lnd-trajectory-wrap">
            <svg viewBox="0 0 340 150" className="lnd-trajectory-svg" aria-hidden="true">
              <defs>
                <linearGradient id="lnd-traj-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#22D3EE" />
                  <stop offset="100%" stopColor="#3B82F6" />
                </linearGradient>
              </defs>
              <g opacity="0.08">
                {Array.from({ length: 18 }, (_, r) => (
                  <line key={`h${r}`} x1="0" y1={r * 9} x2="340" y2={r * 9} stroke="#94A3B8" strokeWidth="0.5" />
                ))}
                {Array.from({ length: 22 }, (_, c) => (
                  <line key={`v${c}`} x1={c * 17} y1="0" x2={c * 17} y2="150" stroke="#94A3B8" strokeWidth="0.5" />
                ))}
              </g>
              <path
                d={TRAJECTORY}
                fill="none"
                stroke="url(#lnd-traj-grad)"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                filter="url(#lnd-traj-glow)"
              />
              <defs>
                <filter id="lnd-traj-glow">
                  <feGaussianBlur stdDeviation="3" result="glow" />
                  <feMerge>
                    <feMergeNode in="glow" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
              {TRAJECTORY_NODES.map((node, i) => (
                <circle
                  key={i}
                  cx={node.cx}
                  cy={node.cy}
                  r={i === 0 ? 4 : 2.5}
                  fill={i === 0 ? "#22D3EE" : "#38BDF8"}
                  opacity={i === 0 ? 1 : 0.7}
                />
              ))}
            </svg>
          </div>
        </div>

        <div className="lnd-tele-tile">
          <p className="lnd-tile-label lnd-mono">
            KEYSTROKE TIMING <span className="lnd-tile-tag">demo</span>
          </p>
          <div className="lnd-bars-group">
            <p className="lnd-bars-label lnd-mono">hold</p>
            <div className="lnd-bars">
              {HOLD_HEIGHTS.map((h, i) => (
                <span key={i} className="lnd-bar lnd-bar--cyan" style={{ height: `${h}%` }} />
              ))}
            </div>
          </div>
          <div className="lnd-bars-group">
            <p className="lnd-bars-label lnd-mono">flight</p>
            <div className="lnd-bars">
              {FLIGHT_HEIGHTS.map((h, i) => (
                <span key={i} className="lnd-bar lnd-bar--blue" style={{ height: `${h}%` }} />
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="lnd-tele-tile lnd-embed-tile">
        <p className="lnd-tile-label lnd-mono">
          BEHAVIORAL EMBEDDING — 128-D VECTOR <span className="lnd-tile-tag">preview</span>
        </p>
        <div className="lnd-embed" aria-hidden="true">
          {EMBEDDING_CELLS.map((v, i) => (
            <span
              key={i}
              className="lnd-embed-cell"
              style={{
                backgroundColor: `rgba(34, 211, 238, ${(0.18 + v * 0.72).toFixed(3)})`,
              }}
            />
          ))}
        </div>
      </div>

      <div className="lnd-tele-pipeline">
        <span className="lnd-pipe-chip">1 CAPTURE</span>
        <span className="lnd-pipe-sep" aria-hidden="true" />
        <span className="lnd-pipe-chip">2 EMBED</span>
        <span className="lnd-pipe-sep" aria-hidden="true" />
        <span className="lnd-pipe-chip">3 COMPARE</span>
        <span className="lnd-pipe-sep" aria-hidden="true" />
        <span className="lnd-pipe-chip lnd-pipe-chip--end">4 VERIFY</span>
        <span className="lnd-pipe-note lnd-mono">SIMULATED PIPELINE · NO SCORE OUTPUT</span>
      </div>
    </aside>
  );
}