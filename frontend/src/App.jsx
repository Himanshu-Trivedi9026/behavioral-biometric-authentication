import { useEffect } from "react";
import { useLocation, navigate } from "./router.jsx";
import { useAuth } from "./auth.js";
import LandingPage from "./landing/LandingPage.jsx";
import RegistrationPage from "./registration/RegistrationPage.jsx";
import LoginPage from "./login/LoginPage.jsx";
import CollectorPage from "./pages/CollectorPage.jsx";
import EnrollmentPage from "./enrollment/EnrollmentPage.jsx";
import VerificationPage from "./verification/VerificationPage.jsx";
import ContinuousVerificationPage from "./continuous/ContinuousVerificationPage.jsx";

const PROTECTED_ROUTES = {
  enrollment: true,
  verification: true,
  continuous: true,
};

export default function App() {
  const { pathname, route } = useLocation();
  const session = useAuth();

  useEffect(() => {
    if (route === null && pathname !== "/") {
      history.replaceState({}, "", "/");
    }
  }, [pathname, route]);

  const requiresAuth = PROTECTED_ROUTES[route] === true;

  useEffect(() => {
    if (requiresAuth && !session) {
      navigate("/login");
    }
  }, [requiresAuth, session]);

  useEffect(() => {
    if (route === "login" && session) {
      navigate("/enrollment");
    }
  }, [route, session]);

  if (requiresAuth && !session) {
    return null;
  }

  if (route === "registration") {
    return <RegistrationPage />;
  }

  if (route === "login") {
    return <LoginPage />;
  }

  if (route === "collector") {
    return <CollectorPage />;
  }

  if (route === "enrollment") {
    return <EnrollmentPage />;
  }

  if (route === "verification") {
    return <VerificationPage />;
  }

  if (route === "continuous") {
    return <ContinuousVerificationPage />;
  }

  return <LandingPage />;
}