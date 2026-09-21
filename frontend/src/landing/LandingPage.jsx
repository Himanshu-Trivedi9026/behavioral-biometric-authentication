import Navbar from "./Navbar.jsx";
import Hero from "./Hero.jsx";
import ValueProps from "./ValueProps.jsx";
import HowItWorks from "./HowItWorks.jsx";
import Technology from "./Technology.jsx";
import PrivacySecurity from "./PrivacySecurity.jsx";
import FinalCta from "./FinalCta.jsx";
import LandingFooter from "./LandingFooter.jsx";
import "./landing.css";

export default function LandingPage() {
  return (
    <div className="landing" id="landing-top">
      <a className="lnd-skip-link" href="#landing-main">
        Skip to content
      </a>
      <Navbar />
      <main id="landing-main">
        <Hero />
        <ValueProps />
        <HowItWorks />
        <Technology />
        <PrivacySecurity />
        <FinalCta />
      </main>
      <LandingFooter />
    </div>
  );
}