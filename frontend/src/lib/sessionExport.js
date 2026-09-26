/*
 * Behavioral Biometric Authentication — Phase 16A
 * Local-only export of the behavioral sessions accumulated on the Enrollment
 * page, so a participant can download them as a JSON file from the browser.
 *
 * PRIVACY / SCOPE CONTRACT:
 *   - This module is LOCAL ONLY. It performs no network request, imports
 *     nothing from api.js, and never touches the backend, the database, or
 *     any external service. No analytics, no telemetry.
 *   - buildSessionExport() deep-copies the raw session objects produced by
 *     lib/collector.js verbatim. No field is added, renamed, re-ordered in
 *     meaning, or removed: session_id, started_at, ended_at,
 *     timestamp_source, keyboard_events, mouse_events.
 *   - It never introduces identity or sensitive material: no username, no
 *     user_ref, no email, no password, no JWT, no key identity, no typed
 *     text, no clipboard, no cookies, no DOM content, no screenshots.
 *   - The download filename is derived from the export timestamp alone, so it
 *     cannot carry participant identity.
 *
 * This module is dependency-free ESM and is consumed both by the React +
 * Vite frontend and by the Node test suite (Node 22.12+/24 supports
 * require() of ESM), so all export logic lives in one testable place.
 */

export const EXPORT_VERSION = "1.0";
export const EXPORT_FILENAME_PREFIX = "behavioral_sessions_";
export const EXPORT_MIME_TYPE = "application/json";

/* An export is possible as soon as at least one session is held in memory. */
export function canExportSessions(sessionCount) {
  return Number.isFinite(sessionCount) && sessionCount > 0;
}

/*
 * ISO-8601 export stamp. `now` is injectable for deterministic tests; it may
 * be a function or a millisecond epoch value.
 */
export function exportTimestamp(now) {
  let epoch;
  if (typeof now === "function") {
    epoch = now();
  } else if (typeof now === "number" && Number.isFinite(now)) {
    epoch = now;
  } else {
    epoch = Date.now();
  }
  return new Date(epoch).toISOString();
}

/*
 * Filesystem-safe filename: behavioral_sessions_<timestamp>.json
 * The timestamp is the only variable part, so no participant identity,
 * session id, or credential can appear in the name.
 */
export function buildExportFilename(exportedAt) {
  const iso =
    typeof exportedAt === "string" && exportedAt.length > 0
      ? exportedAt
      : exportTimestamp();
  return EXPORT_FILENAME_PREFIX + iso.replace(/[:.]/g, "-") + ".json";
}

/*
 * The export document. Sessions are deep-copied exactly as captured, so the
 * caller can keep mutating its own state without touching the export.
 */
export function buildSessionExport(sessions, exportedAt) {
  const list = Array.isArray(sessions) ? sessions : [];
  const stamp =
    typeof exportedAt === "string" && exportedAt.length > 0
      ? exportedAt
      : exportTimestamp();
  return {
    export_version: EXPORT_VERSION,
    exported_at: stamp,
    session_count: list.length,
    sessions: list.map((session) => JSON.parse(JSON.stringify(session))),
  };
}

/*
 * Hand the JSON to the browser as a file download via a temporary object URL
 * and a synthetic anchor. Entirely client-side: nothing is transmitted.
 * `doc` is injectable so the Node test suite can drive it without a DOM.
 * Returns true when a download was triggered.
 */
export function downloadJsonFile(json, filename, doc) {
  const target = doc || (typeof document !== "undefined" ? document : null);
  if (!target || typeof Blob !== "function") {
    return false;
  }
  if (typeof URL === "undefined" || typeof URL.createObjectURL !== "function") {
    return false;
  }
  const blob = new Blob([json], { type: EXPORT_MIME_TYPE });
  const url = URL.createObjectURL(blob);
  const anchor = target.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = "noopener";
  target.body.appendChild(anchor);
  anchor.click();
  target.body.removeChild(anchor);
  URL.revokeObjectURL(url);
  return true;
}

/*
 * Export the currently accumulated sessions locally.
 *
 * With zero sessions this is a no-op: no filename is generated, no download is
 * triggered, and no file is produced.
 *
 * options.download may be injected (tests); it defaults to the browser
 * download above. options.exportedAt / options.now pin the timestamp.
 */
export function exportSessionsLocally(sessions, options) {
  const opts = options || {};
  const list = Array.isArray(sessions) ? sessions : [];

  if (!canExportSessions(list.length)) {
    return {
      ok: false,
      reason: "no-sessions",
      session_count: 0,
      filename: null,
      json: null,
      payload: null,
    };
  }

  const exportedAt =
    typeof opts.exportedAt === "string" && opts.exportedAt.length > 0
      ? opts.exportedAt
      : exportTimestamp(opts.now);
  const payload = buildSessionExport(list, exportedAt);
  const filename = buildExportFilename(exportedAt);
  const json = JSON.stringify(payload, null, 2);

  const download = typeof opts.download === "function" ? opts.download : downloadJsonFile;
  const saved = download(json, filename) !== false;

  return {
    ok: saved,
    reason: saved ? null : "download-unavailable",
    session_count: list.length,
    filename: saved ? filename : null,
    json,
    payload,
  };
}
