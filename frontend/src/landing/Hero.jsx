import { Link } from "../router.jsx";
import TelemetryPanel from "./TelemetryPanel.jsx";

export default function Hero() {
  return (
    <section className="lnd-section lnd-hero">
      <div className="lnd-container">
        <div className="lnd-hero-grid">
          <div className="lnd-hero-copy">
            <p className="lnd-eyebrow lnd-mono">BEHAVIORAL BIOMETRIC AUTHENTICATION</p>
            <h1 className="lnd-title">
              Authentication That <span className="lnd-title-grad">Learns How You Behave.</span>
            </h1>
            <p className="lnd-lede">
              Verify identity through the unique patterns of your keystrokes and mouse
              movements — without relying solely on passwords.
            </p>
            <div className="lnd-cta-row">
              <Link to="/register" className="lnd-btn lnd-btn-primary lnd-btn-lg">
                Get Started
              </Link>
              <a href="#how-it-works" className="lnd-btn lnd-btn-ghost lnd-btn-lg">
                How It Works
              </a>
            </div>
            <ul className="lnd-trust" aria-label="Highlights">
              <li className="lnd-trust-item">
                <span className="lnd-trust-dot" aria-hidden="true" />
                Behavior-based identity
              </li>
              <li className="lnd-trust-item">
                <span className="lnd-trust-dot" aria-hidden="true" />
                Few-shot enrollment
              </li>
              <li className="lnd-trust-item">
                <span className="lnd-trust-dot" aria-hidden="true" />
                Privacy-aware by design
              </li>
            </ul>
          </div>

          <TelemetryPanel />
        </div>
      </div>
    </section>
  );
}