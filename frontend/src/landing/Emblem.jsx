import emblemUrl from "./assets/emblem.svg";

export default function Emblem({ size = 40, className = "" }) {
  return (
    <img
      src={emblemUrl}
      width={size}
      height={size}
      className={className}
      alt=""
      aria-hidden="true"
    />
  );
}