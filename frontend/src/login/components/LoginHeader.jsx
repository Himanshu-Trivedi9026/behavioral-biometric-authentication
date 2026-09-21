import { Link } from "../../router.jsx";
import Emblem from "../../landing/Emblem.jsx";

export default function LoginHeader() {
  return (
    <header className="lgn-header">
      <div className="lgn-container lgn-header-inner">
        <Link to="/" className="lgn-brand">
          <Emblem size={36} className="lgn-brand-emblem" />
          <span className="lgn-brand-text">Behavioral Biometrics</span>
          <span className="lgn-brand-badge">v1.0 CAPSTONE</span>
        </Link>

        <nav className="lgn-header-nav" aria-label="Account">
          <Link to="/" className="lgn-back-home">
            <span className="lgn-back-home-arrow" aria-hidden="true">
              ←
            </span>
            Back to Home
          </Link>
          <Link to="/register" className="lgn-btn lgn-btn-ghost lgn-btn-sm">
            Create Account
          </Link>
        </nav>
      </div>

      <div className="lgn-security-strip">
        <div className="lgn-container lgn-security-strip-inner">
          <span className="lgn-security-item lgn-mono">ARGON2ID PASSWORD PROTECTION</span>
          <span className="lgn-security-side">
            <span className="lgn-security-item lgn-mono">NO KEY IDENTITY LOGGING</span>
            <span className="lgn-security-sep" aria-hidden="true" />
            <span className="lgn-security-item lgn-mono">PROTECTED API WORKFLOW</span>
          </span>
        </div>
      </div>
    </header>
  );
}