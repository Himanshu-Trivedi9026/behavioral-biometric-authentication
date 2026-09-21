export default function LoginError({ message }) {
  return (
    <p className="lgn-error" role="alert">
      <svg
        className="lgn-error-icon"
        viewBox="0 0 24 24"
        width="18"
        height="18"
        aria-hidden="true"
      >
        <circle
          cx="12"
          cy="12"
          r="9"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
        />
        <path
          d="M12 8v4.5M12 15.2v.1"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
        />
      </svg>
      {message}
    </p>
  );
}