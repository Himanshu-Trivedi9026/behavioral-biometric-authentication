import { Link } from "../router.jsx";

export default function FinalCta() {
  return (
    <section className="lnd-section lnd-final-cta" aria-labelledby="cta-heading">
      <div className="lnd-container lnd-cta-panel">
        <h2 className="lnd-heading lnd-cta-title" id="cta-heading">
          Ready to verify your identity through behavior?
        </h2>
        <p className="lnd-cta-sub">
          Enroll your behavioral profile in a few short sessions, then verify future
          sign-ins with a rhythm only you can produce.
        </p>
        <div className="lnd-cta-row lnd-cta-row--center">
          <Link to="/register" className="lnd-btn lnd-btn-primary lnd-btn-lg">
            Get Started
          </Link>
          <Link to="/login" className="lnd-btn lnd-btn-ghost lnd-btn-lg">
            Log In
          </Link>
        </div>
      </div>
    </section>
  );
}