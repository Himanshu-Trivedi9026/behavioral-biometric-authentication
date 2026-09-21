import { useState } from "react";
import { Link } from "../router.jsx";
import Emblem from "./Emblem.jsx";

const NAV_LINKS = [
  { label: "How It Works", href: "#how-it-works" },
  { label: "Security", href: "#security" },
  { label: "Technology", href: "#technology" },
];

export default function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <header className="lnd-nav">
      <div className="lnd-nav-inner">
        <a
          className="lnd-brand"
          href="#landing-top"
          onClick={(event) => {
            event.preventDefault();
            window.scrollTo({ top: 0, behavior: "smooth" });
          }}
        >
          <Emblem size={36} className="lnd-brand-emblem" />
          <span className="lnd-brand-text">Behavioral Biometrics</span>
          <span className="lnd-brand-badge">v1.0 CAPSTONE</span>
        </a>

        <nav className="lnd-nav-links" aria-label="Primary">
          {NAV_LINKS.map((link) => (
            <a key={link.href} className="lnd-nav-link" href={link.href}>
              {link.label}
            </a>
          ))}
        </nav>

        <div className="lnd-nav-actions">
          <Link to="/login" className="lnd-btn lnd-btn-ghost">
            Log In
          </Link>
          <Link to="/register" className="lnd-btn lnd-btn-primary">
            Get Started
          </Link>
        </div>

        <button
          type="button"
          className="lnd-nav-toggle"
          aria-expanded={open}
          aria-controls="lnd-nav-panel"
          aria-label="Toggle navigation menu"
          onClick={() => setOpen((value) => !value)}
        >
          <span className="lnd-nav-toggle-bar" />
          <span className="lnd-nav-toggle-bar" />
          <span className="lnd-nav-toggle-bar" />
        </button>
      </div>

      {open && (
        <div className="lnd-nav-panel" id="lnd-nav-panel">
          {NAV_LINKS.map((link) => (
            <a
              key={link.href}
              className="lnd-nav-link lnd-nav-link--mobile"
              href={link.href}
              onClick={() => setOpen(false)}
            >
              {link.label}
            </a>
          ))}
          <div className="lnd-nav-panel-actions">
            <Link to="/login" className="lnd-btn lnd-btn-ghost" onClick={() => setOpen(false)}>
              Log In
            </Link>
            <Link to="/register" className="lnd-btn lnd-btn-primary" onClick={() => setOpen(false)}>
              Get Started
            </Link>
          </div>
        </div>
      )}
    </header>
  );
}