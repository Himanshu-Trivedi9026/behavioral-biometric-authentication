import { Link } from "../router.jsx";
import AuthenticatedHeader from "../components/AuthenticatedHeader.jsx";
import { useContinuousVerification } from "../hooks/useContinuousVerification.js";
import { CONTINUOUS_STATES, canSubmitWindow } from "../lib/continuous.js";
import ContinuousStatus from "./components/ContinuousStatus.jsx";
import WindowCapture from "./components/WindowCapture.jsx";
import ContinuousResult from "./components/ContinuousResult.jsx";
import ContinuousSpecification from "./components/ContinuousSpecification.jsx";
import "./styles/continuous.css";

export default function ContinuousVerificationPage() {
  const v = useContinuousVerification();

  const machineState = v.state;
  const collecting = machineState === CONTINUOUS_STATES.COLLECTING;
  const verifying = machineState === CONTINUOUS_STATES.VERIFYING;
  const finished = v.captureStatus === "stopped" && collecting;
  const session = finished ? v.captureSession || v.window : v.window;
  const windowGate = session ? canSubmitWindow(session) : { can: false, reason: null };
  const captureDisabled = verifying || v.submitting;

  const showResult =
    v.lastDecision &&
    (machineState === CONTINUOUS_STATES.VERIFIED ||
      machineState === CONTINUOUS_STATES.REVERIFICATION_REQUIRED);

  function handleVerify() {
    if (finished && !captureDisabled) {
      v.submitWindow();
    }
  }

  return (
    <div className="continuous">
      <AuthenticatedHeader active="continuous" />

      <main className="continuous-main">
        <div className="ctn-container continuous-grid">
          <div className="continuous-form-col">
            {showResult ? (
              <ContinuousResult state={machineState} lastDecision={v.lastDecision} onRepeat={v.resetWindow} />
            ) : (
              <section className="ctn-card" aria-labelledby="ctn-card-title">
                <div className="ctn-step">
                  <div>
                    <p className="ctn-step-kicker ctn-mono">PERIODIC IDENTITY CHECK</p>
                    <h1 className="ctn-step-name" id="ctn-card-title">
                      Continuous Verification
                    </h1>
                  </div>
                  <p className="ctn-step-next ctn-mono">BEHAVIORAL WINDOW</p>
                </div>
                <p className="ctn-step-hint">
                  Keep your identity verified by periodically closing short
                  windows of typing and mouse movement. Each window is matched
                  against your stored behavioral profile.
                </p>

                <ContinuousStatus state={machineState} error={v.error} capReason={v.capReason} />

                <WindowCapture
                  status={v.captureStatus}
                  keyboardCount={v.keyboardCount}
                  mouseCount={v.mouseCount}
                  lastEvent={v.lastEvent}
                  sessionId={v.sessionId}
                  capReason={v.capReason}
                  disabled={captureDisabled}
                  onStart={v.startWindow}
                  onFinish={v.finishWindow}
                />

                <div className="ctn-window-status">
                  <span className="ctn-window-status-label ctn-mono">WINDOW:</span>
                  {session ? (
                    <span className="ctn-window-status-ready ctn-mono">
                      READY · KBD {v.keyboardCount} · MOUSE {v.mouseCount}
                    </span>
                  ) : (
                    <span className="ctn-window-status-empty">Not captured yet</span>
                  )}
                </div>

                <button
                  type="button"
                  className="ctn-btn ctn-btn-primary ctn-btn-block"
                  disabled={!finished || !windowGate.can || captureDisabled}
                  onClick={handleVerify}
                >
                  {verifying || v.submitting ? (
                    <>
                      <span className="ctn-spinner" aria-hidden="true" />
                      Verifying behavior...
                    </>
                  ) : !finished ? (
                    "Verify Window (disabled until a window is captured)"
                  ) : (
                    "Verify Window"
                  )}
                </button>

                {finished && !windowGate.can && (
                  <button
                    type="button"
                    className="ctn-btn ctn-btn-ghost ctn-btn-block"
                    onClick={v.resetWindow}
                  >
                    Discard Window and Start Over
                  </button>
                )}

                {v.errorCode === "profile_not_found" && (
                  <Link to="/enrollment" className="ctn-btn ctn-btn-primary ctn-btn-block">
                    Continue to Enrollment →
                  </Link>
                )}
              </section>
            )}
          </div>

          <div className="continuous-spec-col">
            <ContinuousSpecification />
          </div>
        </div>
      </main>

      <footer className="ctn-footer">
        <div className="ctn-container ctn-footer-inner">
          <span className="ctn-mono ctn-footer-title">Behavioral Biometrics · Continuous Verification</span>
          <span className="ctn-footer-note">
            College project — development system, not for production use.
          </span>
        </div>
      </footer>
    </div>
  );
}