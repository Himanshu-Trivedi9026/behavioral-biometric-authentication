const PRINCIPLES = [
  {
    title: "No Key Identity Logging",
    body: "Only timing is captured — key-hold and key-flight. The characters themselves are never recorded, transmitted or stored.",
  },
  {
    title: "Derived Behavioral Representations",
    body: "Comparisons run on learned embeddings rather than raw logs, so stored profiles encode behavioral patterns — not keystrokes or screen positions.",
  },
  {
    title: "Protected API Workflow",
    body: "Behavioral data flows to the backend through its protected, authenticated API endpoints rather than exposed, open routes.",
  },
];

export default function PrivacySecurity() {
  return (
    <section className="lnd-section lnd-security" id="security" aria-labelledby="security-heading">
      <div className="lnd-container">
        <p className="lnd-eyebrow lnd-mono">PRIVACY &amp; SECURITY</p>
        <h2 className="lnd-heading" id="security-heading">
          Designed with privacy in mind
        </h2>
        <p className="lnd-sub">
          The pipeline treats behavioral data as sensitive by default — limiting what is
          captured, how it is represented and where it travels.
        </p>
        <div className="lnd-card-grid lnd-card-grid--3">
          {PRINCIPLES.map((principle) => (
            <article key={principle.title} className="lnd-card">
              <h3 className="lnd-card-title">{principle.title}</h3>
              <p className="lnd-card-body">{principle.body}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}