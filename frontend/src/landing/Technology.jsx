const TECH_CARDS = [
  {
    title: "Keyboard Dynamics",
    body: "Key-hold and key-flight timing capture the pace of your typing. Only timing and tempo are recorded — never the keys themselves.",
    tag: "input signal",
  },
  {
    title: "Mouse Trajectories",
    body: "Cursor movement is sampled as delta positions (dx, dy) with timestamps, distance and speed, capturing the curves of hand motion.",
    tag: "input signal",
  },
  {
    title: "CNN + GRU Encoder",
    body: "A 1D convolutional front end extracts local movement patterns, and a GRU models their sequence into a 128-dimensional embedding.",
    tag: "model",
  },
  {
    title: "Siamese Verification",
    body: "A Siamese network shares the encoder across reference and live samples, measuring similarity with L2 distance against a calibrated threshold.",
    tag: "decision",
  },
];

const STACK = [
  "React + Vite",
  "FastAPI",
  "PyTorch CNN + GRU",
  "Siamese Verification",
  "PostgreSQL Profile Store",
];

export default function Technology() {
  return (
    <section className="lnd-section lnd-tech" id="technology" aria-labelledby="tech-heading">
      <div className="lnd-container">
        <p className="lnd-eyebrow lnd-mono">UNDER THE HOOD</p>
        <h2 className="lnd-heading" id="tech-heading">
          The technology behind behavioral authentication
        </h2>
        <p className="lnd-sub">
          A privacy-first pipeline that turns raw behavioral events into compact,
          comparable profiles.
        </p>

        <div className="lnd-card-grid lnd-card-grid--2">
          {TECH_CARDS.map((card) => (
            <article key={card.title} className="lnd-card">
              <span className="lnd-card-tag lnd-mono">{card.tag}</span>
              <h3 className="lnd-card-title">{card.title}</h3>
              <p className="lnd-card-body">{card.body}</p>
            </article>
          ))}
        </div>

        <div className="lnd-stack" aria-label="Technology stack">
          {STACK.map((item, i) => (
            <span key={item} className="lnd-stack-item">
              <span className="lnd-stack-name lnd-mono">{item}</span>
              {i < STACK.length - 1 && (
                <span className="lnd-stack-sep" aria-hidden="true">
                  /
                </span>
              )}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}