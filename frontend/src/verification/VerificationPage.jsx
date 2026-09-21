import { useEffect, useRef, useState } from "react";
import { Link } from "../router.jsx";
import AuthenticatedHeader from "../components/AuthenticatedHeader.jsx";
import { useCollector } from "../hooks/useCollector.js";
import { useAuth, logOut } from "../auth.js";
import { verifySession } from "../api.js";
import {
  hasUsableContent,
  sessionEventCounts,
  buildVerificationPayload,
} from "../lib/enrollment.js";
import VerificationStatus from "./components/VerificationStatus.jsx";
import VerificationResult from "./components/VerificationResult.jsx";
import ProbeCapture from "./components/ProbeCapture.jsx";
import VerificationSpecification from "./components/VerificationSpecification.jsx";
import "./styles/verification.css";

function probeErrorText(result) {
  if (result.status === 401) {
    return "Your session has expired. Please log in again.";
  }
  if (result.code === "profile_not_found") {
    return "No behavioral profile exists for this account yet. Enroll first to build your profile.";
  }
  if (result.status === 422) {
    return "The captured probe was rejected as invalid. Record a session with keyboard and mouse activity, then try again.";
  }
  if (result.status === 503) {
    return "Behavioral service unavailable. Please try again later.";
  }
  if (result.status === 500 || result.status === 502) {
    return "Unexpected server error. Please try again.";
  }
  if (result.status === 0) {
    return "Unable to reach the behavioral service. Please try again.";
  }
  return "Verification failed. Please try again.";
}

export default function VerificationPage() {
  const auth = useAuth();
  const collector = useCollector();
  const [probe, setProbe] = useState(null);
  const [notice, setNotice] = useState(null);
  const [phase, setPhase] = useState("probe");
  const [submitError, setSubmitError] = useState(null);
  const [submitCode, setSubmitCode] = useState(null);
  const [result, setResult] = useState(null);
  const acceptedIds = useRef(new Set());

  useEffect(() => {
    if (collector.status !== "stopped" || !collector.session) {
      return;
    }
    const sess = collector.session;
    if (acceptedIds.current.has(sess.session_id)) {
      return;
    }
    acceptedIds.current.add(sess.session_id);

    if (!hasUsableContent(sess)) {
      const counts = sessionEventCounts(sess);
      setNotice(
        "That probe recorded " +
          counts.keyboard +
          " keyboard and " +
          counts.mouse +
          " mouse events only. Type a few words and move the mouse over the capture area, then finish again."
      );
      collector.clear();
      return;
    }

    setNotice(null);
    setProbe(sess);
    collector.clear();
  }, [collector.status, collector.session, collector.clear]);

  function handleStart() {
    setNotice(null);
    setSubmitError(null);
    setSubmitCode(null);
    setProbe(null);
    collector.start();
  }

  function handleFinish() {
    if (collector.status === "running") {
      collector.stop();
    }
  }

  async function handleVerify() {
    if (phase === "submitting" || !probe) {
      return;
    }
    setPhase("submitting");
    setSubmitError(null);
    setSubmitCode(null);

    const token = auth ? auth.token : null;
    const outcome = await verifySession({
      session: buildVerificationPayload(probe).session,
      token,
    });

    if (outcome.ok) {
      setResult(outcome.result);
      setPhase("success");
      return;
    }

    if (outcome.status === 401) {
      logOut();
    }

    setPhase("error");
    setSubmitError(probeErrorText(outcome));
    setSubmitCode(outcome.code);
  }

  function handleReset() {
    setPhase("probe");
    setSubmitError(null);
    setSubmitCode(null);
    setProbe(null);
    setResult(null);
  }

  return (
    <div className="verification">
      <AuthenticatedHeader active="verification" />

      <main className="verification-main">
        <div className="vfy-container verification-grid">
          <div className="verification-form-col">
            {phase === "success" ? (
              <VerificationResult result={result} onRepeat={handleReset} />
            ) : (
              <section className="vfy-card" aria-labelledby="vfy-card-title">
                <div className="vfy-step">
                  <div>
                    <p className="vfy-step-kicker vfy-mono">IDENTITY CHECK</p>
                    <h1 className="vfy-step-name" id="vfy-card-title">
                      Verification
                    </h1>
                  </div>
                  <p className="vfy-step-next vfy-mono">BEHAVIORAL PROBE</p>
                </div>
                <p className="vfy-step-hint">
                  Capture one short session of typing and mouse movement. It is
                  matched against your stored behavioral profile.
                </p>

                <VerificationStatus phase={phase} error={submitError} />

                <ProbeCapture
                  status={collector.status}
                  keyboardCount={collector.keyboardCount}
                  mouseCount={collector.mouseCount}
                  lastEvent={collector.lastEvent}
                  sessionId={collector.sessionId}
                  probe={probe}
                  notice={notice}
                  onStart={handleStart}
                  onFinish={handleFinish}
                />

                <div className="vfy-probe-status">
                  <span className="vfy-probe-status-label vfy-mono">PROBE SESSION:</span>
                  {probe ? (
                    <span className="vfy-probe-status-ready vfy-mono">
                      READY · KBD {probe.keyboard_events.length} · MOUSE{" "}
                      {probe.mouse_events.length}
                    </span>
                  ) : (
                    <span className="vfy-probe-status-empty">Not captured yet</span>
                  )}
                </div>

                <button
                  type="button"
                  className="vfy-btn vfy-btn-primary vfy-btn-block"
                  disabled={phase === "submitting" || !probe}
                  onClick={handleVerify}
                >
                  {phase === "submitting" ? (
                    <>
                      <span className="vfy-spinner" aria-hidden="true" />
                      Verifying behavior...
                    </>
                  ) : !probe ? (
                    "Run Verification (disabled until a probe is captured)"
                  ) : (
                    "Run Verification"
                  )}
                </button>

                {phase === "error" && submitCode !== "profile_not_found" && (
                  <button
                    type="button"
                    className="vfy-btn vfy-btn-ghost vfy-btn-block"
                    onClick={handleReset}
                  >
                    Record a New Probe
                  </button>
                )}

                {phase === "error" && submitCode === "profile_not_found" && (
                  <Link to="/enrollment" className="vfy-btn vfy-btn-primary vfy-btn-block">
                    Continue to Enrollment →
                  </Link>
                )}
              </section>
            )}
          </div>

          <div className="verification-spec-col">
            <VerificationSpecification />
          </div>
        </div>
      </main>

      <footer className="vfy-footer">
        <div className="vfy-container vfy-footer-inner">
          <span className="vfy-mono vfy-footer-title">
            Behavioral Biometrics · Verification
          </span>
          <span className="vfy-footer-note">
            College project — development system, not for production use.
          </span>
        </div>
      </footer>
    </div>
  );
}