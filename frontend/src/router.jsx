import { useEffect, useState } from "react";

const ROUTES = {
  "/": "landing",
  "/register": "registration",
  "/login": "login",
  "/collector": "collector",
  "/enrollment": "enrollment",
  "/verification": "verification",
  "/continuous": "continuous",
};

const NAV_ROUTE_CHANGE = "bb:routechange";

export function resolveRoute(pathname) {
  return ROUTES[pathname] ?? null;
}

export function useLocation() {
  const [pathname, setPathname] = useState(window.location.pathname);

  useEffect(() => {
    const onChange = () => setPathname(window.location.pathname);
    window.addEventListener("popstate", onChange);
    window.addEventListener(NAV_ROUTE_CHANGE, onChange);
    return () => {
      window.removeEventListener("popstate", onChange);
      window.removeEventListener(NAV_ROUTE_CHANGE, onChange);
    };
  }, []);

  return { pathname, route: resolveRoute(pathname) };
}

export function navigate(to) {
  if (to === window.location.pathname) return;
  history.pushState({}, "", to);
  window.dispatchEvent(new Event(NAV_ROUTE_CHANGE));
}

export function Link({ to, className, children, onClick, ...rest }) {
  return (
    <a
      href={to}
      className={className}
      onClick={(event) => {
        event.preventDefault();
        navigate(to);
        if (onClick) onClick(event);
      }}
      {...rest}
    >
      {children}
    </a>
  );
}