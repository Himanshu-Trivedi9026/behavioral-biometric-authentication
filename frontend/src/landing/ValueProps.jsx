const VALUE_PROPS = [
  {
    title: "Behavior-Based",
    body: "Identity is drawn from how you type and move — the subtle rhythm, pace and tempo behind every session that is difficult to imitate.",
  },
  {
    title: "Few-Shot Enrollment",
    body: "A handful of short sessions is enough to build a personal behavioral profile — no long surveys, no extra hardware, no calibration scripts.",
  },
  {
    title: "Privacy-Aware",
    body: "Only keystroke timing and cursor motion are captured locally. Typed characters are never recorded, transmitted or stored.",
  },
  {
    title: "Behavioral Verification",
    body: "At sign-in, live behavior is embedded and compared against the enrolled profile, producing a VERIFIED or SUSPICIOUS outcome.",
  },
];

export default function ValueProps() {
  return (
    <section className="lnd-section lnd-valueprops" aria-labelledby="valueprops-heading">
      <div className="lnd-container">
        <p className="lnd-eyebrow lnd-mono">WHY BEHAVIORAL BIOMETRICS</p>
        <h2 className="lnd-heading" id="valueprops-heading">
          More than a password — a pattern only you can produce
        </h2>
        <div className="lnd-card-grid lnd-card-grid--4">
          {VALUE_PROPS.map((prop) => (
            <article key={prop.title} className="lnd-card">
              <h3 className="lnd-card-title">{prop.title}</h3>
              <p className="lnd-card-body">{prop.body}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}