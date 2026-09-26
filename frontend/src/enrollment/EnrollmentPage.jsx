import { useEffect, useRef, useState } from "react";
import { Link } from "../router.jsx";
import AuthenticatedHeader from "../components/AuthenticatedHeader.jsx";
import { useCollector } from "../hooks/useCollector.js";
import { useAuth, logOut } from "../auth.js";
import { enrollSessions } from "../api.js";
import {
  MAX_ENROLLMENT_SESSIONS,
  hasReachedMaxSessions,
  hasUsableContent,
  sessionEventCounts,
  buildEnrollmentPayload,
} from "../lib/enrollment.js";
import { canExportSessions, exportSessionsLocally } from "../lib/sessionExport.js";
import EnrollmentStatus from "./components/EnrollmentStatus.jsx";
import EnrollmentSuccess from "./components/EnrollmentSuccess.jsx";
import SessionCapture from "./components/SessionCapture.jsx";
import EnrollmentSpecification from "./components/EnrollmentSpecification.jsx";
import "./styles/enrollment.css";

function submitErrorText(result) {
  if (result.status === 401) {
    return "Your session has expired. Please log in again.";
  }
  if (result.code === "profile_exists") {
    return "A behavioral profile already exists for this account. You can proceed to verification.";
  }
  if (result.status === 422) {
    return "The captured sessions were rejected as invalid. Record sessions with keyboard and mouse activity, then try again.";
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
  return "Enrollment failed. Please try again.";
}

export default function EnrollmentPage() {
  const auth = useAuth();
  const collector = useCollector();
  const [sessions, setSessions] = useState([]);
  const [notice, setNotice] = useState(null);
  const [phase, setPhase] = useState("collect");
  const [submitError, setSubmitError] = useState(null);
  const [submitCode, setSubmitCode] = useState(null);
  const [result, setResult] = useState(null);
  const [exportNotice, setExportNotice] = useState(null);
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
        "That session recorded " +
          counts.keyboard +
          " keyboard and " +
          counts.mouse +
          " mouse events only. Type a few words and move the mouse over the capture area, then finish again."
      );
      collector.clear();
      return;
    }

    if (hasReachedMaxSessions(sessions.length)) {
      setNotice(
        "Maximum of " +
          MAX_ENROLLMENT_SESSIONS +
          " enrollment sessions reached. Send your profile to the server."
      );
      collector.clear();
      return;
    }

    setNotice(null);
    setSessions((previous) => [...previous, sess]);
    collector.clear();
  }, [collector.status, collector.session, collector.clear, sessions]);

  function handleStart() {
    if (hasReachedMaxSessions(sessions.length)) {
      return;
    }
    setNotice(null);
    setSubmitError(null);
    setExportNotice(null);
    collector.start();
  }

  function handleFinish() {
    if (collector.status === "running") {
      collector.stop();
    }
  }

  /*
   * Local-only export. Everything happens in this browser tab: the sessions
   * already held in page state are serialized to JSON and handed to the
   * browser's own download mechanism. Nothing is sent to FastAPI, nothing is
   * written to PostgreSQL, and no external service is contacted.
   */
  function handleExport() {
    const outcome = exportSessionsLocally(sessions);
    if (!outcome.ok) {
      setExportNotice(
        outcome.reason === "no-sessions"
          ? "There are no sessions to export yet. Record a session first, then export."
          : "This browser could not start the download. Check that downloads are allowed for this page, then try again."
      );
      return;
    }
    setExportNotice(
      "Exported " +
        outcome.session_count +
        " session" +
        (outcome.session_count === 1 ? "" : "s") +
        " to " +
        outcome.filename +
        ". The file stays on this device — the raw behavioral data was not uploaded."
    );
  }

  async function handleSubmit() {
    if (phase === "submitting" || sessions.length < 1) {
      return;
    }
    setPhase("submitting");
    setSubmitError(null);
    setSubmitCode(null);

    const token = auth ? auth.token : null;
    const outcome = await enrollSessions({
      sessions: buildEnrollmentPayload(sessions).sessions,
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
    setSubmitError(submitErrorText(outcome));
    setSubmitCode(outcome.code);
  }

  function handleReenroll() {
    setPhase("collect");
    setSubmitError(null);
    setSubmitCode(null);
    setExportNotice(null);
    setSessions([]);
    setResult(null);
  }

  const showSuccess = phase === "success";

  return (
    <div className="enrollment">
      <AuthenticatedHeader active="enrollment" />

      <main className="enrollment-main">
        <div className="enr-container enrollment-grid">
          <div className="enrollment-form-col">
            {showSuccess ? (
              <EnrollmentSuccess result={result} />
            ) : (
              <section className="enr-card" aria-labelledby="enr-card-title">
                <div className="enr-step">
                  <div>
                    <p className="enr-step-kicker enr-mono">BEHAVIORAL PROFILE</p>
                    <h1 className="enr-step-name" id="enr-card-title">
                      Enrollment
                    </h1>
                  </div>
                  <p className="enr-step-next enr-mono">NEXT: VERIFICATION</p>
                </div>
                <p className="enr-step-hint">
                  Record real keyboard and mouse behavior. Sessions are processed
                  server-side and only an aggregate profile is stored.
                </p>

                <EnrollmentStatus phase={phase} error={submitError} />

                <SessionCapture
                  status={collector.status}
                  keyboardCount={collector.keyboardCount}
                  mouseCount={collector.mouseCount}
                  lastEvent={collector.lastEvent}
                  sessionId={collector.sessionId}
                  notice={notice}
                  sessionsCompleted={sessions.length}
                  maxSessions={MAX_ENROLLMENT_SESSIONS}
                  onStart={handleStart}
                  onFinish={handleFinish}
                />

                <div className="enr-session-list">
                  <p className="enr-session-list-title enr-mono">SESSIONS READY TO SUBMIT</p>
                  {sessions.length === 0 ? (
                    <p className="enr-session-list-empty">
                      No sessions captured yet. Start a session above and finish it
                      with keyboard and mouse activity.
                    </p>
                  ) : (
                    <ol className="enr-session-list-items">
                      {sessions.map((sess, index) => (
                        <li key={sess.session_id} className="enr-session-item">
                          <span className="enr-session-item-index enr-mono">
                            SESSION {index + 1}
                          </span>
                          <span className="enr-session-item-stats enr-mono">
                            KBD {sess.keyboard_events.length} · MOUSE{" "}
                            {sess.mouse_events.length}
                          </span>
                          <span className="enr-session-item-id">{sess.session_id}</span>
                        </li>
                      ))}
                    </ol>
                  )}
                </div>

                <div className="enr-export">
                  <button
                    type="button"
                    className="enr-btn enr-btn-secondary enr-btn-block"
                    disabled={!canExportSessions(sessions.length)}
                    onClick={handleExport}
                  >
                    {canExportSessions(sessions.length)
                      ? "Export Sessions (" +
                        sessions.length +
                        ") — download JSON to this device"
                      : "Export Sessions (disabled — no sessions to export)"}
                  </button>
                  <p className="enr-export-privacy">
                    Download only. The exported JSON is written straight to your
                    device by your browser. It is never uploaded to the server, never
                    stored in the database, and never sent to any third party. The
                    file name and its contents contain no name, email, password, or
                    other account details — only the raw keyboard/mouse timing data
                    you just recorded.
                  </p>
                  {exportNotice && (
                    <p className="enr-export-notice" role="status">
                      {exportNotice}
                    </p>
                  )}
                </div>

                <button
                  type="button"
                  className="enr-btn enr-btn-primary enr-btn-block"
                  disabled={phase === "submitting" || sessions.length < 1}
                  onClick={handleSubmit}
                >
                  {phase === "submitting" ? (
                    <>
                      <span className="enr-spinner" aria-hidden="true" />
                      Building your profile...
                    </>
                  ) : sessions.length < 1 ? (
                    "Send Profile to Server (disabled until a session is captured)"
                  ) : (
                    "Send " +
                      sessions.length +
                      " Session" +
                      (sessions.length === 1 ? "" : "s") +
                      " to Server"
                  )}
                </button>

                {phase === "error" && submitCode !== "profile_exists" && (
                  <button
                    type="button"
                    className="enr-btn enr-btn-ghost enr-btn-block"
                    onClick={handleReenroll}
                  >
                    Record Sessions Again
                  </button>
                )}

                {phase === "error" && submitCode === "profile_exists" && (
                  <Link to="/verification" className="enr-btn enr-btn-primary enr-btn-block">
                    Continue to Verification →
                  </Link>
                )}
              </section>
            )}
          </div>

          <div className="enrollment-spec-col">
            <EnrollmentSpecification />
          </div>
        </div>
      </main>

      <footer className="enr-footer">
        <div className="enr-container enr-footer-inner">
          <span className="enr-mono enr-footer-title">
            Behavioral Biometrics · Enrollment
          </span>
          <span className="enr-footer-note">
            College project — development system, not for production use.
          </span>
        </div>
      </footer>
    </div>
  );
}