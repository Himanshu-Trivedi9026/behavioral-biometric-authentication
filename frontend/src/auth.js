import { useSyncExternalStore } from "react";

export const STORAGE_KEY = "bba.session.v1";

function createSessionStore(storage) {
  let state = null;
  const listeners = new Set();

  function readStored() {
    if (!storage) return null;
    try {
      const raw = storage.getItem(STORAGE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (
        parsed &&
        typeof parsed.accessToken === "string" &&
        parsed.accessToken.length > 0 &&
        typeof parsed.username === "string"
      ) {
        return { accessToken: parsed.accessToken, username: parsed.username };
      }
      storage.removeItem(STORAGE_KEY);
    } catch {
      /* corrupt storage is treated as no session */
    }
    return null;
  }

  state = readStored();

  function persist(next) {
    if (!storage) return;
    try {
      if (next) {
        storage.setItem(STORAGE_KEY, JSON.stringify(next));
      } else {
        storage.removeItem(STORAGE_KEY);
      }
    } catch {
      /* storage unavailable: keep in-memory state only */
    }
  }

  function emit() {
    listeners.forEach((listener) => listener());
  }

  return {
    getState: () => state,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    getSnapshot: () => state,
    getAccessToken: () => (state ? state.accessToken : null),
    getUsername: () => (state ? state.username : null),
    isAuthenticated: () => Boolean(state && state.accessToken),
    logIn({ token, username }) {
      if (typeof token !== "string" || token.length === 0) return;
      const next = {
        accessToken: token,
        username: typeof username === "string" ? username : "",
      };
      state = next;
      persist(next);
      emit();
    },
    logOut() {
      if (state === null) return;
      state = null;
      persist(null);
      emit();
    },
  };
}

const browserStorage =
  typeof globalThis !== "undefined" && typeof globalThis.sessionStorage !== "undefined"
    ? globalThis.sessionStorage
    : null;

export const sessionStore = createSessionStore(browserStorage);

export function logIn(session) {
  sessionStore.logIn(session);
}

export function logOut() {
  sessionStore.logOut();
}

export function getAccessToken() {
  return sessionStore.getAccessToken();
}

export function getUsername() {
  return sessionStore.getUsername();
}

export function isAuthenticated() {
  return sessionStore.isAuthenticated();
}

export function useAuth() {
  const session = useSyncExternalStore(sessionStore.subscribe, sessionStore.getSnapshot);
  if (!session) return null;
  return { user: session.username, token: session.accessToken };
}

export { createSessionStore };