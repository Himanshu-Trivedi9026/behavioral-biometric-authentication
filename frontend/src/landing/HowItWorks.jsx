const STEPS = [
  {
    step: "01",
    title: "Capture",
    body: "Client-side instrumentation records keystroke and mouse events — key-hold and key-flight times, cursor position, velocity and distance.",
    note: "Typed characters are never sampled.",
  },
  {
    step: "02",
    title: "Learn",
    body: "During enrollment, the encoder models the distinctive rhythm of your behavior into a compact 128-dimensional embedding.",
    note: "A few short sessions are enough.",
  },
  {
    step: "03",
    title: "Compare",
    body: "At verification time, a fresh session is embedded and measured against your enrolled profile using a Siamese distance metric.",
    note: "Reference vs. live — same encoder.",
  },
  {
    step: "04",
    title: "Verify",
    body: "A calibrated distance threshold resolves the comparison into one of two outcomes: VERIFIED or SUSPICIOUS.",
    note: "A decision, expressed as a label.",
  },
];

export default function HowItWorks() {
  return (
    <section className="lnd-section lnd-how" id="how-it-works" aria-labelledby="how-heading">
      <div className="lnd-container">
        <p className="lnd-eyebrow lnd-mono">THE PIPELINE</p>
        <h2 className="lnd-heading" id="how-heading">
          How behavioral authentication works
        </h2>
        <div className="lnd-step-grid">
          {STEPS.map((item) => (
            <article key={item.step} className="lnd-step">
              <span className="lnd-step-num lnd-mono" aria-hidden="true">
                STEP {item.step}
              </span>
              <h3 className="lnd-step-title">{item.title}</h3>
              <p className="lnd-step-body">{item.body}</p>
              <span className="lnd-step-note lnd-mono">{item.note}</span>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}