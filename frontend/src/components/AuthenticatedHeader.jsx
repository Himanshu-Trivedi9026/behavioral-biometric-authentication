import { Link } from "../router.jsx";
import { useAuth, logOut } from "../auth.js";
import Emblem from "../landing/Emblem.jsx";
import "./authenticated-header.css";

export default function AuthenticatedHeader({ active }) {
  const session = useAuth();

  function handleLogout() {
    if (session) logOut();
  }

  return (
    <header className="auth-header">
      <div className="auth-container auth-header-inner">
        <Link to="/" className="auth-brand">
          <Emblem size={36} className="auth-brand-emblem" />
          <span className="auth-brand-text">Behavioral Biometrics</span>
          <span className="auth-brand-badge">v1.0 CAPSTONE</span>
        </Link>

        <nav className="auth-header-nav" aria-label="Account">
          <Link
            to="/enrollment"
            className={
              active === "enrollment" ? "auth-nav-link auth-nav-link--active" : "auth-nav-link"
            }
          >
            ENROLLMENT
          </Link>
          <Link
            to="/verification"
            className={
              active === "verification" ? "auth-nav-link auth-nav-link--active" : "auth-nav-link"
            }
          >
            VERIFICATION
          </Link>
          <Link
            to="/continuous"
            className={
              active === "continuous" ? "auth-nav-link auth-nav-link--active" : "auth-nav-link"
            }
          >
            CONTINUOUS
          </Link>
          <span className="auth-header-sep" aria-hidden="true" />
          <Link to="/" className="auth-back-home">
            <span className="auth-back-home-arrow" aria-hidden="true">
              ←
            </span>
            Home
          </Link>
          <span className="auth-username" title="Authenticated as">
            {session ? session.user : ""}
          </span>
          <button
            type="button"
            className="auth-logout"
            onClick={handleLogout}
            disabled={!session}
          >
            Log Out
          </button>
        </nav>
      </div>
    </header>
  );
}