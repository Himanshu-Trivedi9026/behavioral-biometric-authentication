export default function TypeZone({ running }) {
  return (
    <section className="card">
      <h2>Type Here</h2>
      <textarea
        id="type-zone"
        className={running ? "active" : ""}
        rows="6"
        placeholder={
          "Type harmless sample text here while collecting\n" +
          "(e.g. 'the quick brown fox jumps over the lazy dog').\n" +
          "Never type passwords or personal information."
        }
      />
      <p className="hint">
        Keyboard events are captured globally on the page while a session is
        running. The typed characters themselves are{" "}
        <strong>never stored</strong> — only press/release timestamps.
      </p>
    </section>
  );
}