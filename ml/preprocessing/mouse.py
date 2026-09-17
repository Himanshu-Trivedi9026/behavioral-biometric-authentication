"""
Mouse preprocessing and feature extraction.

Input  : list of validated raw mouse events
         ``{"event_type": "mouse", "event": "mousemove"|"mousedown"|"mouseup",
         "x": float, "y": float, "timestamp": float}``
Output : separate trajectory sequence + preserved action events.

Trajectory (mousemove only)
---------------------------
For each pair of chronologically consecutive mouse positions:

``dx``        horizontal displacement (px)
``dy``        vertical displacement (px)
``dt``        time between samples (ms)
``distance``  sqrt(dx^2 + dy^2)  (px)
``speed``     distance / dt  (px/ms)

Safe rule for ``dt <= 0``: speed is undefined without a positive time
span, so ``speed`` is set to ``0.0``. The sample is still kept so dx/dy/dt/
distance remain inspectable; downstream phases may filter.

Shape note: the first mousemove has no predecessor, so the trajectory
sequence has ``(number of move points) - 1`` samples.

Action events
-------------
``mousedown`` / ``mouseup`` are preserved verbatim (chronological order) as
``mouse_action_events`` for later modeling. No categorical encoding is
introduced yet.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

MOUSE_FEATURE_COLUMNS = ("dx", "dy", "dt", "distance", "speed")
MOUSE_ACTION_EVENTS = ("mousedown", "mouseup")


def _sort_events(events: Sequence[dict]) -> List[Tuple[float, int, dict]]:
    """Return events sorted by (timestamp, original index) — stable & deterministic."""
    indexed = [(float(e["timestamp"]), i, e) for i, e in enumerate(events)]
    indexed.sort(key=lambda t: (t[0], t[1]))
    return indexed


def preprocess_mouse(
    events: Sequence[dict],
) -> Tuple[List[Dict[str, float]], List[Dict[str, Any]], Dict[str, int]]:
    """Reduce raw mouse events to trajectory/speed features + action events.

    Malformed handling (documented; never aborts the pipeline):
      * exact consecutive duplicate events  -> dropped, counted
      * ``dt <= 0`` samples                 -> kept, speed set to 0.0

    Privacy: only viewport-relative trajectory shape is used. No screen
    identifier, browser identity, or page content is requested or emitted.

    Args:
        events: raw mouse event dicts.

    Returns:
        A ``(move_sequence, action_events, metadata)`` tuple.

        ``move_sequence`` is the trajectory feature list
        ``[{"dx": .., "dy": .., "dt": .., "distance": .., "speed": ..}, ...]``.
        ``action_events`` is the preserved mousedown/mouseup list
        ``[{"event": .., "x": .., "y": .., "timestamp": ..}, ...]``.
        ``metadata`` summarises counts for traceability.
    """
    metadata = {
        "input_events": len(events),
        "move_points": 0,
        "move_samples": 0,
        "action_events": 0,
        "dropped_duplicates": 0,
    }

    move_points: List[Tuple[float, float, float]] = []  # (x, y, timestamp)
    action_events: List[Dict[str, Any]] = []
    last_seen: None | Tuple[str, float, float, float] = None

    for _, _, event in _sort_events(events):
        name = event["event"]
        x = float(event["x"])
        y = float(event["y"])
        ts = float(event["timestamp"])

        if last_seen is not None and last_seen == (name, x, y, ts):
            metadata["dropped_duplicates"] += 1
            continue
        last_seen = (name, x, y, ts)

        if name == "mousemove":
            move_points.append((x, y, ts))
        else:  # mousedown | mouseup
            action_events.append(
                {"event": name, "x": x, "y": y, "timestamp": ts}
            )

    move_sequence: List[Dict[str, float]] = []
    for i in range(1, len(move_points)):
        x0, y0, t0 = move_points[i - 1]
        x1, y1, t1 = move_points[i]
        dx = x1 - x0
        dy = y1 - y0
        dt = t1 - t0
        distance = math.sqrt(dx * dx + dy * dy)
        speed = distance / dt if dt > 0 else 0.0
        move_sequence.append(
            {
                "dx": dx,
                "dy": dy,
                "dt": dt,
                "distance": distance,
                "speed": speed,
            }
        )

    metadata["move_points"] = len(move_points)
    metadata["move_samples"] = len(move_sequence)
    metadata["action_events"] = len(action_events)
    return move_sequence, action_events, metadata