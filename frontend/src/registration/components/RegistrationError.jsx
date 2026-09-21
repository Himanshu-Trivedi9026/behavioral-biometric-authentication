export default function RegistrationError({ message }) {
  return (
    <div className="reg-error" role="alert">
      <svg
        className="reg-error-icon"
        viewBox="0 0 20 20"
        width="18"
        height="18"
        aria-hidden="true"
      >
        <circle cx="10" cy="10" r="9" fill="none" stroke="currentColor" strokeWidth="1.6" />
        <path d="M10 5.5v5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        <circle cx="10" cy="13.6" r="0.9" fill="currentColor" />
      </svg>
      <span>{message}</span>
    </div>
  );
}