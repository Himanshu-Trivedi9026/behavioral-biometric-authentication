import { useState } from "react";
import RegistrationHeader from "./components/RegistrationHeader.jsx";
import RegistrationForm from "./components/RegistrationForm.jsx";
import RegistrationSuccess from "./components/RegistrationSuccess.jsx";
import BehavioralSpecification from "./components/BehavioralSpecification.jsx";
import { registerAccount } from "../api.js";
import "./styles/registration.css";

function errorMessage(result) {
  if (result.status === 409) {
    return "Username already exists.";
  }
  if (result.status === 503) {
    return "Authentication service unavailable. Please try again later.";
  }
  if (result.status === 422) {
    return "Invalid registration details. Check your username and password and try again.";
  }
  if (result.status === 500) {
    return "Unexpected server error. Please try again.";
  }
  if (result.status === 0) {
    return "Unable to reach the authentication service. Please try again.";
  }
  return "Registration failed. Please try again.";
}

export default function RegistrationPage() {
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);
  const [user, setUser] = useState(null);

  async function handleRegister(username, password) {
    if (status === "loading") return;
    setStatus("loading");
    setError(null);

    const result = await registerAccount({ username, password });

    if (result.ok) {
      setUser(result.user);
      setStatus("success");
    } else {
      setError(errorMessage(result));
      setStatus("error");
    }
  }

  return (
    <div className="registration">
      <RegistrationHeader />

      <main className="registration-main">
        <div className="reg-container registration-grid">
          <div className="registration-form-col">
            {status === "success" ? (
              <RegistrationSuccess user={user} />
            ) : (
              <RegistrationForm status={status} error={error} onSubmit={handleRegister} />
            )}
          </div>
          <div className="registration-spec-col">
            <BehavioralSpecification />
          </div>
        </div>
      </main>

      <footer className="reg-footer">
        <div className="reg-container reg-footer-inner">
          <span className="reg-mono">Behavioral Biometrics · v1.0 Capstone</span>
          <span>College project — development system, not for production use.</span>
        </div>
      </footer>
    </div>
  );
}