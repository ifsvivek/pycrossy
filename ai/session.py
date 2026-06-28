"""In-process AI playback: Auto-Play and Replay Viewer.

A :class:`AISession` owns a headless :class:`~ai.env.CrossyEnv` but advances its engine one
fixed tick at a time so the hop animations render smoothly inside the game's own main loop
(unlike ``env.step``, which fast-forwards a whole decision). At each settle point it asks a
*policy* for the next action — either a trained model loaded from a checkpoint, a recorded
replay's action list, or a simple built-in heuristic when nothing is trained yet.

This keeps "watch the AI play" and "re-watch a saved run" entirely in the existing window,
with no extra process or window, while reusing the exact observation/action space the
training stack uses (so what you watch is what was learned).
"""
from __future__ import annotations

import os
from typing import List, Optional

from pycrossy import config
from pycrossy.engine import Direction
from pycrossy.tween import tween

from . import observation
from .env import Action, CrossyEnv

_DIR = {Action.UP: Direction.UP, Action.DOWN: Direction.DOWN,
        Action.LEFT: Direction.LEFT, Action.RIGHT: Direction.RIGHT}


class AISession:
    def __init__(self, mode: str = "auto", algo: str = "neat", seed: int = 0,
                 wait_ticks: int = 8):
        self.mode = mode                      # 'auto' | 'replay'
        self.algo_name = algo
        self.seed = int(seed)
        self.wait_ticks = wait_ticks
        self.env = CrossyEnv(seed=self.seed)
        self.policy = None                    # object with best_act(obs) -> int
        self.replay_actions: List[int] = []
        self.label = ""
        self.source = ""
        self._replay_idx = 0
        self._wait = 0
        self.max_z = 0
        self.score = 0
        self.elapsed = 0.0
        self.dead_timer = 0.0
        self.finished = False
        self._load_source()
        self.reset()

    # -- setup -------------------------------------------------------------
    def _load_source(self) -> None:
        if self.mode == "replay":
            self._load_replay()
        else:
            self._load_policy()

    def _load_policy(self) -> None:
        path = os.path.join("runs", self.algo_name, "best_model.pkl")
        if os.path.exists(path):
            try:
                from .trainer import Trainer
                self.policy = Trainer.load_policy(path)
                self.label = f"{self.algo_name.upper()} TRAINED"
                self.source = path
                return
            except Exception as exc:                       # pragma: no cover - defensive
                print(f"[ai] could not load {path}: {exc}")
        self.label = f"{self.algo_name.upper()} HEURISTIC"

    def _load_replay(self) -> None:
        from . import replay as replay_mod
        path = os.path.join("runs", self.algo_name, "best_replay.json")
        if os.path.exists(path):
            try:
                rp = replay_mod.load(path)
                self.replay_actions = list(rp.actions)
                self.seed = int(rp.seed)
                self.label = f"REPLAY {self.algo_name.upper()} SCORE {rp.score}"
                self.source = path
                return
            except Exception as exc:                       # pragma: no cover - defensive
                print(f"[ai] could not load replay {path}: {exc}")
        self.finished = True
        self.label = "No replay found"

    @property
    def available(self) -> bool:
        return not (self.mode == "replay" and not self.replay_actions)

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self.env.reset(seed=self.seed)
        self.env.engine.hero.stop_idle()
        self._replay_idx = 0
        self._wait = 0
        self.max_z = 0
        self.score = 0
        self.elapsed = 0.0
        self.dead_timer = 0.0
        self.finished = self.mode == "replay" and not self.replay_actions

    def restart(self) -> None:
        # Replays are deterministic and finite — don't loop; Auto-play restarts forever.
        if self.mode == "replay":
            self.finished = True
            return
        self.reset()

    @property
    def scene(self):
        return self.env.engine.scene

    def hud(self) -> dict:
        return {"score": self.score, "label": self.label}

    # -- per-tick ----------------------------------------------------------
    def frame_step(self, dt: float) -> None:
        if self.finished:                 # replay exhausted / nothing more to do
            return
        eng = self.env.engine
        tween.update(dt)
        eng.tick(dt)
        self.elapsed += dt
        self._track_score()

        if not eng.hero.is_alive:
            self.dead_timer += dt
            if self.dead_timer > 1.4:
                self.restart()
            return

        if eng.hero.moving:
            return
        if self._wait > 0:
            self._wait -= 1
            return

        action = self._next_action()
        if action is None:
            self.finished = True
            return
        if action == int(Action.WAIT):
            self._wait = self.wait_ticks
            return
        eng.begin_move_with_direction()
        eng.move_with_direction(_DIR[Action(action)])

    def _track_score(self) -> None:
        import math
        z = max(0, math.floor(self.env.engine.hero.position.z) - config.STARTING_ROW)
        self.max_z = max(self.max_z, z)
        self.score = self.max_z

    def _next_action(self) -> Optional[int]:
        if self.mode == "replay":
            if self._replay_idx >= len(self.replay_actions):
                return None
            a = self.replay_actions[self._replay_idx]
            self._replay_idx += 1
            return int(a)
        obs = observation.build(self.env.engine, self.max_z, self.elapsed)
        if self.policy is not None:
            try:
                return int(self.policy.best_act(obs))
            except Exception:                              # pragma: no cover - defensive
                pass
        return self._heuristic()

    def _heuristic(self) -> int:
        """A modest hand-written policy for when no model is trained yet.

        Prefer hopping forward; if the row immediately ahead is a road or water, occasionally
        wait a beat or sidestep so the demo doesn't just walk into the first car.
        """
        eng = self.env.engine
        hero = eng.hero
        ahead = eng.game_map.get_row(hero.position.z + 1) or {}
        kind = ahead.get("type")
        r = self.env._rng.random()
        if kind in ("road", "water"):
            if r < 0.45:
                return int(Action.WAIT)
            if r < 0.6:
                return int(Action.LEFT)
            if r < 0.75:
                return int(Action.RIGHT)
        elif r < 0.08:
            return int(Action.LEFT if r < 0.04 else Action.RIGHT)
        return int(Action.UP)
