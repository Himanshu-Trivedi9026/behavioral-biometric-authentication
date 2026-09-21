import { useState } from "react";
import LoginHeader from "./components/LoginHeader.jsx";
import LoginForm from "./components/LoginForm.jsx";
import LoginSuccess from "./components/LoginSuccess.jsx";
import BehavioralSpecification from "./components/BehavioralSpecification.jsx";
import { loginAccount } from "../api.js";
import { logIn } from "../auth.js";
import "./styles/login.css";

function errorMessage(result) {
  if (result.status === 401) {
    return "Invalid username or password.";
  }
  if (result.status === 422) {
    return "Invalid login details. Enter your username and password and try again.";
  }
  if (result.status === 503) {
    return "Authentication service unavailable. Please try again later.";
  }
  if (result.status === 500 || result.status === 502) {
    return "Unexpected server error. Please try again.";
  }
  if (result.status === 0) {
    return "Unable to reach the authentication service. Please try again.";
  }
  return "Log in failed. Please try again.";
}

export default function LoginPage() {
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);

  async function handleLogin(username, password) {
    if (status === "loading") return;
    setStatus("loading");
    setError(null);

    const result = await loginAccount({ username, password });

    if (result.ok) {
      logIn({ token: result.token, username });
      setStatus("success");
    } else {
      setError(errorMessage(result));
      setStatus("error");
    }
  }

  return (
    <div className="login">
      <LoginHeader />

      <main className="login-main">
        <div className="lgn-container login-grid">
          <div className="login-form-col">
            {status === "success" ? (
              <LoginSuccess />
            ) : (
              <LoginForm status={status} error={error} onSubmit={handleLogin} />
            )}
          </div>
          <div className="login-spec-col">
            <BehavioralSpecification />
          </div>
        </div>
      </main>

      <footer className="lgn-footer">
        <div className="lgn-container lgn-footer-inner">
          <span className="lgn-mono lgn-footer-title">
            Behavioral Biometrics Authentication Gateway
          </span>
          <span className="lgn-footer-mid">
            v1.0 Capstone · Privacy-aware behavioral verification
          </span>
          <span className="lgn-footer-note">
            College project — development system, not for production use.
          </span>
        </div>
      </footer>
    </div>
  );
}