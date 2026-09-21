import { Link } from "../router.jsx";
import Emblem from "./Emblem.jsx";

const PRODUCT_LINKS = [
  { label: "How It Works", href: "#how-it-works" },
  { label: "Security", href: "#security" },
  { label: "Technology", href: "#technology" },
];

const PLATFORM_LINKS = [
  { label: "Log In", to: "/login" },
  { label: "Register", to: "/register" },
];

const TECH_ITEMS = [
  "React + Vite",
  "FastAPI",
  "CNN + GRU",
  "Siamese Verification",
  "PostgreSQL",
];

export default function LandingFooter() {
  return (
    <footer className="lnd-footer">
      <div className="lnd-container">
        <div className="lnd-footer-grid">
          <div className="lnd-footer-brand">
            <div className="lnd-brand">
              <Emblem size={32} className="lnd-brand-emblem" />
              <span className="lnd-brand-text">Behavioral Biometrics</span>
            </div>
            <p className="lnd-footer-description">
              A behavioral-biometric authentication system that answers the question:
              are you really who you say you are?
            </p>
          </div>

          <nav className="lnd-footer-col" aria-label="Product">
            <h3 className="lnd-footer-heading">Product</h3>
            {PRODUCT_LINKS.map((link) => (
              <a key={link.href} className="lnd-footer-link" href={link.href}>
                {link.label}
              </a>
            ))}
          </nav>

          <nav className="lnd-footer-col" aria-label="Platform">
            <h3 className="lnd-footer-heading">Platform</h3>
            {PLATFORM_LINKS.map((link) => (
              <Link key={link.to} to={link.to} className="lnd-footer-link">
                {link.label}
              </Link>
            ))}
          </nav>

          <div className="lnd-footer-col" aria-label="Technology">
            <h3 className="lnd-footer-heading">Technology</h3>
            {TECH_ITEMS.map((item) => (
              <span key={item} className="lnd-footer-item lnd-mono">
                {item}
              </span>
            ))}
          </div>
        </div>

        <div className="lnd-footer-bottom">
          <span className="lnd-mono">Behavioral Biometrics · v1.0 Capstone</span>
          <span>College project — development system, not for production use.</span>
        </div>
      </div>
    </footer>
  );
}