import { Link } from "../../router.jsx";
import Emblem from "../../landing/Emblem.jsx";

const SECURITY_ITEMS = [
  "ARGON2ID PASSWORD PROTECTION",
  "NO KEY IDENTITY LOGGING",
  "PROTECTED API WORKFLOW",
];

export default function RegistrationHeader() {
  return (
    <header className="reg-header">
      <div className="reg-container reg-header-inner">
        <Link to="/" className="reg-brand">
          <Emblem size={36} className="reg-brand-emblem" />
          <span className="reg-brand-text">Behavioral Biometrics</span>
          <span className="reg-brand-badge">v1.0 CAPSTONE</span>
        </Link>

        <nav className="reg-header-nav" aria-label="Account">
          <Link to="/" className="reg-back-home">
            <span className="reg-back-home-arrow" aria-hidden="true">
              ←
            </span>
            Back to Home
          </Link>
          <Link to="/login" className="reg-btn reg-btn-ghost reg-btn-sm">
            Log In
          </Link>
        </nav>
      </div>

      <div className="reg-security-strip">
        <div className="reg-container reg-security-strip-inner">
          {SECURITY_ITEMS.map((item, index) => (
            <span key={item} className="reg-security-item reg-mono">
              {index > 0 && <span className="reg-security-sep" aria-hidden="true" />}
              {item}
            </span>
          ))}
        </div>
      </div>
    </header>
  );
}