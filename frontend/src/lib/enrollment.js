export const MIN_KEYBOARD_EVENTS = 2;
export const MIN_MOUSE_EVENTS = 2;
export const MAX_ENROLLMENT_SESSIONS = 16;

export function sessionEventCounts(session) {
  return {
    keyboard: session && Array.isArray(session.keyboard_events) ? session.keyboard_events.length : 0,
    mouse: session && Array.isArray(session.mouse_events) ? session.mouse_events.length : 0,
  };
}

export function hasUsableContent(session) {
  const { keyboard, mouse } = sessionEventCounts(session);
  return keyboard >= MIN_KEYBOARD_EVENTS && mouse >= MIN_MOUSE_EVENTS;
}

export function hasReachedMaxSessions(sessionCount) {
  return sessionCount >= MAX_ENROLLMENT_SESSIONS;
}

export function canAddEnrollmentSession(sessionCount) {
  return !hasReachedMaxSessions(sessionCount);
}

export function collectEnrollmentSession(sessions, session) {
  const list = Array.isArray(sessions) ? sessions : [];
  if (!hasUsableContent(session)) {
    return { sessions: list, added: false, reason: "no-usable-content" };
  }
  if (hasReachedMaxSessions(list.length)) {
    return { sessions: list, added: false, reason: "max-sessions" };
  }
  return { sessions: list.concat(session), added: true, reason: null };
}

export function buildEnrollmentPayload(sessions) {
  return { sessions: sessions.map((session) => JSON.parse(JSON.stringify(session))) };
}

export function buildVerificationPayload(session) {
  return { session: JSON.parse(JSON.stringify(session)) };
}