const DEFAULT_API_BASE = "http://localhost:8000/api/v1";
const DEFAULT_TIMEOUT_MS = 30000;

const envApiBase = import.meta.env?.VITE_API_BASE_URL;

export const API_BASE_URL = (envApiBase || DEFAULT_API_BASE).replace(/\/+$/, "");

function withTimeout(milliseconds) {
  if (typeof AbortController === "undefined" || typeof setTimeout !== "function") {
    return { signal: null, timerId: null };
  }
  const controller = new AbortController();
  const timerId = setTimeout(() => controller.abort(), milliseconds);
  return { signal: controller.signal, timerId };
}

async function request(path, { method = "GET", token = null, body = undefined, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const { signal, timerId } = withTimeout(timeoutMs);

  let response;
  try {
    const headers = {};
    if (body !== undefined) headers["Content-Type"] = "application/json";
    if (token) headers.Authorization = "Bearer " + token;
    response = await fetch(API_BASE_URL + path, {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: signal || undefined,
    });
  } catch (error) {
    const aborted = error && typeof error.name === "string" && error.name === "AbortError";
    return { ok: false, status: 0, code: aborted ? "timeout" : null, message: null };
  } finally {
    if (timerId) clearTimeout(timerId);
  }

  let data = null;
  try {
    data = await response.json();
  } catch {
    /* backend returned an unparseable body */
  }

  if (response.ok) {
    return { ok: true, status: response.status, data };
  }

  if (data && data.error) {
    return {
      ok: false,
      status: response.status,
      code: typeof data.error.code === "string" ? data.error.code : null,
      message: typeof data.error.message === "string" ? data.error.message : null,
    };
  }

  return { ok: false, status: response.status, code: null, message: null };
}

export async function registerAccount({ username, password }) {
  const result = await request("/auth/register", {
    method: "POST",
    body: { username, password },
  });
  if (!result.ok) {
    return {
      ok: false,
      status: result.status,
      code: result.code,
      message: result.message,
    };
  }
  return { ok: true, user: result.data };
}

export async function loginAccount({ username, password }) {
  const result = await request("/auth/login", {
    method: "POST",
    body: { username, password },
  });
  if (!result.ok) {
    return {
      ok: false,
      status: result.status,
      code: result.code,
      message: result.message,
    };
  }
  const body = result.data || {};
  const token = typeof body.access_token === "string" ? body.access_token : null;
  const tokenType = typeof body.token_type === "string" ? body.token_type : "bearer";
  if (!token) {
    return { ok: false, status: 502, code: null, message: null };
  }
  return { ok: true, token, tokenType };
}

export async function enrollSessions({ sessions, token, timeoutMs }) {
  if (!token) {
    return { ok: false, status: 401, code: "missing_token", message: null };
  }
  const result = await request("/enrollment", {
    method: "POST",
    token,
    body: { sessions },
    timeoutMs,
  });
  if (!result.ok) {
    return {
      ok: false,
      status: result.status,
      code: result.code,
      message: result.message,
    };
  }
  return { ok: true, result: result.data };
}

export async function verifySession({ session, token, timeoutMs }) {
  if (!token) {
    return { ok: false, status: 401, code: "missing_token", message: null };
  }
  const result = await request("/verification", {
    method: "POST",
    token,
    body: { session },
    timeoutMs,
  });
  if (!result.ok) {
    return {
      ok: false,
      status: result.status,
      code: result.code,
      message: result.message,
    };
  }
  return { ok: true, result: result.data };
}

export async function continuousVerify({ session, token, timeoutMs }) {
  if (!token) {
    return { ok: false, status: 401, code: "missing_token", message: null };
  }
  const result = await request("/continuous-verification", {
    method: "POST",
    token,
    body: session,
    timeoutMs,
  });
  if (!result.ok) {
    return {
      ok: false,
      status: result.status,
      code: result.code,
      message: result.message,
    };
  }
  return { ok: true, result: result.data };
}

/*
 * GET /continuous-verification/state (Phase 14B) — read the
 * server-authoritative behavioral verification state for the current session.
 * Used on page mount/refresh so the UI decision truth comes from the server,
 * never from client storage.
 */
export async function getContinuousSessionState({ token, timeoutMs }) {
  if (!token) {
    return { ok: false, status: 401, code: "missing_token", message: null };
  }
  const result = await request("/continuous-verification/state", {
    method: "GET",
    token,
    timeoutMs,
  });
  if (!result.ok) {
    return {
      ok: false,
      status: result.status,
      code: result.code,
      message: result.message,
    };
  }
  return { ok: true, result: result.data };
}