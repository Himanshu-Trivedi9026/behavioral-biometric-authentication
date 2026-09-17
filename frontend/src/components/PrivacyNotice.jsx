export default function PrivacyNotice() {
  return (
    <section className="card privacy-card">
      <h2>Privacy Notice</h2>
      <p>
        This prototype collects <strong>behavioral metadata only</strong> — key
        press/release timing and mouse trajectory coordinates. It deliberately
        does <em>not</em> record or persist: typed text, passwords, input field
        values, clipboard content, cookies, browser history, or any other
        personal data. See the README for full details.
      </p>
    </section>
  );
}