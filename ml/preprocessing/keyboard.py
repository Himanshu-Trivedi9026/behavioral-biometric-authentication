"""
Keyboard preprocessing and feature extraction.

The Phase 2 collector deliberately stores NO key identity — only
``keydown``/``keyup`` names and high-resolution timestamps. Everything below
therefore works without key names and never reconstructs characters.

Input  : list of validated raw keyboard events
         ``{"event_type": "keyboard", "event": "keydown"|"keyup",
         "timestamp": float}``
Output : ``list`` of feature dicts (see :func:`preprocess_keyboard`).

Features
--------
Each valid key press produces one feature sample:

``hold_time``
    keyup timestamp - keydown timestamp (ms). Non-negative.
``flight_time``
    current keydown timestamp - previous keydown timestamp (ms).
    Defined between consecutive valid presses. The FIRST press in a session
    has no predecessor; its ``flight_time`` is set to ``0.0`` as a
    documented, deterministic placeholder (keeps the sequence numeric).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

KEYBOARD_FEATURE_COLUMNS = ("hold_time", "flight_time")


def _sort_events(events: Sequence[dict]) -> List[Tuple[float, int, dict]]:
    """Return events sorted by (timestamp, original index) — stable & deterministic."""
    indexed = [(float(e["timestamp"]), i, e) for i, e in enumerate(events)]
    indexed.sort(key=lambda t: (t[0], t[1]))
    return indexed


def preprocess_keyboard(
    events: Sequence[dict],
) -> Tuple[List[Dict[str, float]], Dict[str, int]]:
    """Reduce raw keyboard events to behavioral timing features.

    Pairing rule (event-order based, no key identity used):
      Events are walked in chronological order. ``keydown`` events are pushed
      onto a stack (LIFO). Each ``keyup`` pops the MOST RECENT unmatched
      ``keydown`` — this correctly resolves interleaved/nested presses when
      keys are unknown.

    Output order:
      After pairing, valid presses are **sorted by keydown timestamp** and
      features are computed in that order. This ensures ``flight_time`` is
      always non-negative for valid (chronologically sorted) input.

    Malformed handling (documented; never aborts the pipeline):
      * unmatched ``keyup`` (empty stack)      -> dropped, counted
      * unmatched ``keydown`` left at the end  -> dropped, counted
      * negative hold time (keyup before keydown) -> press dropped, counted
      * exact consecutive duplicate events     -> dropped, counted

    Args:
        events: raw keyboard event dicts.

    Returns:
        ``(sequence, metadata)`` where ``sequence`` is a list of
        ``{"hold_time": float, "flight_time": float}`` dicts and ``metadata``
        summarises counts for traceability.
    """
    metadata = {
        "input_events": len(events),
        "valid_presses": 0,
        "dropped_unmatched_keyup": 0,
        "dropped_unmatched_keydown": 0,
        "dropped_negative_hold": 0,
        "dropped_duplicates": 0,
    }

    # --- Phase 1: LIFO pairing (unchanged from before) ---
    stack: List[float] = []
    valid_presses: List[Tuple[float, float]] = []  # (keydown_ts, keyup_ts)
    last_seen: None | Tuple[str, float] = None

    for _, _, event in _sort_events(events):
        name = event["event"]
        ts = float(event["timestamp"])

        if last_seen is not None and last_seen == (name, ts):
            metadata["dropped_duplicates"] += 1
            continue
        last_seen = (name, ts)

        if name == "keydown":
            stack.append(ts)
            continue

        # name == "keyup"
        if not stack:
            metadata["dropped_unmatched_keyup"] += 1
            continue

        keydown_ts = stack.pop()
        hold_time = ts - keydown_ts
        if hold_time < 0:
            metadata["dropped_negative_hold"] += 1
            continue

        valid_presses.append((keydown_ts, ts))

    metadata["dropped_unmatched_keydown"] = len(stack)

    # --- Phase 2: sort by keydown timestamp and compute features ---
    valid_presses.sort(key=lambda p: p[0])

    sequence: List[Dict[str, float]] = []
    previous_keydown: None | float = None

    for kd, ku in valid_presses:
        hold_time = ku - kd
        flight_time = (kd - previous_keydown) if previous_keydown is not None else 0.0
        sequence.append({"hold_time": hold_time, "flight_time": flight_time})
        previous_keydown = kd

    metadata["valid_presses"] = len(sequence)
    return sequence, metadata