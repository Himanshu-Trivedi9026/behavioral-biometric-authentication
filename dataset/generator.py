"""
Deterministic synthetic / demo dataset generation.

Why a synthetic generator?
    Phase 2 collection requires a live browser, so only a few real manual
    sessions exist. To develop and test the Phase 4 dataset + baseline we
    generate a deterministic, schema-compatible synthetic dataset.

Schema compatibility
    Every generated entry uses the exact same entry schema as the Phase 4
    ``dataset.schema`` (which mirrors the Phase 3 processed-session layout):
    ``user_id``, ``session_id``, ``keyboard_sequence``, ``mouse_sequence``,
    ``mouse_action_events`` and optional ``metadata``.

Determinism
    ``random.Random`` is seeded *per user* and *per session* using fixed
    derived seed strings (``f"syn:{seed}:{user_id}"`` / ``...:session_idx``).
    Python's ``random.Random(str)`` seeds deterministically across runs and
    processes, so the exact same dataset is regenerated for the same seed —
    independent of ordering or global RNG state.

Behavioural plausibility
    Each user gets their own statistical "typing/mouse profile":

    * keyboard: per-user ``gauss``-shaped hold_time and flight_time
      distributions (users type at different speeds/rhythms),
    * mouse: per-user step-size and speed distributions plus realistic sample
      intervals (``dt`` ~ 16-40 ms, like the Phase 2 collector),
    * occasional longer pauses / jumps to mimic natural behaviour.

    Sessions are **not** duplicate copies — each one is freshly sampled from
    the user's distribution with its own seed, producing realistic within-user
    and across-user variation.

Privacy
    Only numerical timing/geometry is generated. No key names, characters,
    passwords, or raw text appear anywhere.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class SyntheticDatasetGenerator:
    """Deterministic multi-user synthetic behavioural dataset generator."""

    def __init__(
        self,
        seed: int = 0,
        n_users: int = 5,
        sessions_per_user: int = 10,
    ) -> None:
        """Configure generation scale.

        Defaults: 5 users x 10 sessions = 50 sessions. With the default 70/30
        per-user split that yields 7 enrolment + 3 evaluation sessions per
        user (15 genuine + 60 impostor attempts) — enough for a meaningful
        baseline while staying fully deterministic and stdlib-only.
        """
        if n_users < 1:
            raise ValueError("n_users must be >= 1")
        if sessions_per_user < 2:
            raise ValueError("sessions_per_user must be >= 2")

        self.seed = seed
        self.n_users = n_users
        self.sessions_per_user = sessions_per_user

    # ------------------------------------------------------------------
    # User + session generation
    # ------------------------------------------------------------------

    def _user_rng(self, index: int) -> random.Random:
        return random.Random("syn:{}:user:{}".format(self.seed, index))

    def _session_rng(self, user_index: int, session_index: int) -> random.Random:
        seed_key = "syn:{seed}:user-{user}:session-{session}".format(
            seed=self.seed, user=user_index, session=session_index
        )
        return random.Random(seed_key)

    def _user_parameters(self, user_index: int) -> Dict[str, float]:
        """Each user gets a stable, distinct behavioural profile."""
        rng = self._user_rng(user_index)
        return {
            "hold_mean": _clamp(rng.gauss(85.0, 25.0), 30.0, 180.0),
            "hold_std": _clamp(rng.gauss(18.0, 8.0), 6.0, 40.0),
            "flight_mean": _clamp(rng.gauss(220.0, 70.0), 60.0, 450.0),
            "flight_std": _clamp(rng.gauss(70.0, 30.0), 15.0, 160.0),
            "flight_pause_prob": rng.uniform(0.04, 0.14),
            "flight_pause_scale": rng.uniform(2.5, 5.5),
            "dx_std": _clamp(rng.gauss(16.0, 9.0), 2.0, 45.0),
            "dy_std": _clamp(rng.gauss(16.0, 9.0), 2.0, 45.0),
            "speed_mean": _clamp(rng.gauss(2.0, 1.0), 0.4, 5.0),
            "speed_std": _clamp(rng.gauss(1.0, 0.5), 0.2, 2.5),
            "dt_min": rng.uniform(16.0, 20.0),
            "dt_jitter": rng.uniform(8.0, 22.0),
            "jump_prob": rng.uniform(0.02, 0.08),
            "jump_scale": rng.uniform(4.0, 10.0),
            "click_prob": rng.uniform(0.2, 0.6),
        }

    def _make_keyboard_sequence(
        self, params: Dict[str, float], rng: random.Random
    ) -> List[Dict[str, float]]:
        n_presses = rng.randint(15, 35)
        sequence: List[Dict[str, float]] = []
        for press_i in range(n_presses):
            hold = abs(rng.gauss(params["hold_mean"], params["hold_std"]))
            hold = _clamp(hold, 10.0, 800.0)

            if rng.random() < params["flight_pause_prob"]:
                flight = abs(
                    rng.gauss(
                        params["flight_mean"] * params["flight_pause_scale"],
                        params["flight_std"],
                    )
                )
            else:
                flight = abs(rng.gauss(params["flight_mean"], params["flight_std"]))
            flight = _clamp(flight, 0.0, 12000.0)

            # First press flight_time = 0.0 documented placeholder (Phase 3 semantics)
            flight_time = 0.0 if press_i == 0 else flight

            sequence.append({"hold_time": round(hold, 3), "flight_time": round(flight_time, 3)})
        return sequence

    def _make_mouse_sequence(
        self, params: Dict[str, float], rng: random.Random
    ) -> List[Dict[str, float]]:
        # Build a random-walk of move points, then derive trajectory features
        # exactly like Phase 3 mouse preprocessing does (consecutive pairs).
        n_points = rng.randint(30, 80)
        x = rng.uniform(50.0, 700.0)
        y = rng.uniform(50.0, 450.0)
        t = rng.uniform(100.0, 500.0)

        move_points: List[Any] = [(x, y, t)]
        for _ in range(n_points - 1):
            dt = params["dt_min"] + abs(rng.gauss(0.0, params["dt_jitter"]))
            if rng.random() < params["jump_prob"]:
                dx = rng.gauss(0.0, params["dx_std"] * params["jump_scale"])
                dy = rng.gauss(0.0, params["dy_std"] * params["jump_scale"])
            else:
                dx = rng.gauss(0.0, params["dx_std"])
                dy = rng.gauss(0.0, params["dy_std"])
            x = _clamp(x + dx, 0.0, 1920.0)
            y = _clamp(y + dy, 0.0, 1080.0)
            t += dt
            move_points.append((x, y, t))

        sequence: List[Dict[str, float]] = []
        for i in range(1, len(move_points)):
            x0, y0, t0 = move_points[i - 1]
            x1, y1, t1 = move_points[i]
            dx = x1 - x0
            dy = y1 - y0
            dt = t1 - t0
            distance = math.sqrt(dx * dx + dy * dy)
            speed = distance / dt if dt > 0 else 0.0
            sequence.append(
                {
                    "dx": round(dx, 3),
                    "dy": round(dy, 3),
                    "dt": round(dt, 3),
                    "distance": round(distance, 3),
                    "speed": round(speed, 3),
                }
            )
        return sequence

    def _make_action_events(
        self, params: Dict[str, float], rng: random.Random
    ) -> List[Dict[str, float]]:
        events: List[Dict[str, float]] = []
        n_clicks = rng.randint(0, 4)
        t = rng.uniform(50.0, 300.0)
        for _ in range(n_clicks):
            if rng.random() > params["click_prob"]:
                continue
            x = rng.uniform(50.0, 1500.0)
            y = rng.uniform(50.0, 800.0)
            t += rng.uniform(20.0, 120.0)
            events.append({"event": "mousedown", "x": round(x, 3), "y": round(y, 3), "timestamp": round(t, 3)})
            t += rng.uniform(20.0, 90.0)
            events.append({"event": "mouseup", "x": round(x, 3), "y": round(y, 3), "timestamp": round(t, 3)})
        return events

    def generate_session(self, user_index: int, session_index: int) -> Dict[str, Any]:
        """Generate one dataset entry for ``user_index`` / ``session_index``."""
        params = self._user_parameters(user_index)
        rng = self._session_rng(user_index, session_index)

        user_id = "user_{:03d}".format(user_index + 1)
        session_id = "syn-user_{:03d}-session-{:04d}".format(
            user_index + 1, session_index
        )

        keyboard_sequence = self._make_keyboard_sequence(params, rng)
        mouse_sequence = self._make_mouse_sequence(params, rng)
        action_events = self._make_action_events(params, rng)

        return {
            "user_id": user_id,
            "session_id": session_id,
            "keyboard_sequence": keyboard_sequence,
            "mouse_sequence": mouse_sequence,
            "mouse_action_events": action_events,
            "metadata": {
                "generated": True,
                "seed": self.seed,
                "user_index": user_index,
                "session_index": session_index,
                "source": None,
                "preprocessing": {"version": "synthetic"},
                "keyboard_samples": len(keyboard_sequence),
                "mouse_samples": len(mouse_sequence),
            },
        }

    def generate(self) -> List[Dict[str, Any]]:
        """Generate the full dataset (deterministic for the configured seed)."""
        entries: List[Dict[str, Any]] = []
        for u in range(self.n_users):
            for s in range(self.sessions_per_user):
                entries.append(self.generate_session(u, s))
        return entries


__all__ = ["SyntheticDatasetGenerator"]