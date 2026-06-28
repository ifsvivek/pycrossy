"""AI Swarm — watch a whole population play at once, camera on the leader.

Many chickens run simultaneously in the *same* world, each driven by its own policy. The
trick that makes this cheap and faithful: every chicken gets its own headless
:class:`~pycrossy.engine.Engine` seeded identically and ticked in lockstep, so all worlds are
bit-for-bit identical (same rows, same traffic) while the chickens diverge by their actions.
We then render only the **leader's** engine scene and overlay the other chickens as "ghost"
chicken meshes — because every engine's ``world`` group shares the same absolute row-index
coordinate frame, a ghost placed at another chicken's local position lands exactly where it
belongs in the leader's framing. The camera follows the leader for free (its own engine eases
to follow it); when the leader dies or is overtaken we simply render the next-best engine,
which has been smoothly tracking its own chicken all along — so there's no camera jump.

Policies come from the saved population when one is trained (genuinely different behaviours),
from an ε-varied copy of a single trained policy, or from a forward-biased heuristic when
nothing is trained yet — so the swarm is always lively.
"""
from __future__ import annotations

import math
import os
import pickle
import random
from typing import Callable, List

from pycrossy import config, primitives
from pycrossy.audio import NullAudio
from pycrossy.engine import Direction, Engine
from pycrossy.scene import Group, Mesh
from pycrossy.tween import tween

from . import algorithms  # noqa: F401 — registers all algorithms so make_algo can find them
from . import observation, vec_env
from .base import make as make_algo
from .env import NUM_ACTIONS, OBS_SIZE, Action

_DIR = {Action.UP: Direction.UP, Action.DOWN: Direction.DOWN,
        Action.LEFT: Direction.LEFT, Action.RIGHT: Direction.RIGHT}

_CROWN_GEO = primitives.box(0.45, 0.18, 0.45)
_CROWN_TINT = (1.0, 0.82, 0.18, 1.0)


class SwarmSession:
    mode = "swarm"

    def __init__(self, algo: str = "neat", seed: int = 0, size: int = 12,
                 wait_ticks: int = 8, character: str = "chicken"):
        self.algo_name = algo
        self.seed = int(seed)
        self.character = character
        self.wait_ticks = wait_ticks

        self.policies, self.label = self._load_policies(algo, max(2, int(size)), self.seed)
        self.n = len(self.policies)

        self.engines: List[Engine] = [Engine(audio=NullAudio()) for _ in range(self.n)]
        for e in self.engines:
            e.is_game_state_ended = lambda: False
        # Persistent ghost chickens (one per agent) + a crown marking the leader.
        from pycrossy.models import registry
        self._reg = registry()
        self.ghosts: List[Group] = []
        for _ in range(self.n):
            g = Group()
            g.add(self._reg.make_hero_node(character))
            self.ghosts.append(g)
        self.ghost_group = Group()
        for g in self.ghosts:
            self.ghost_group.add(g)
        self.crown = Mesh(_CROWN_GEO, None, tint=_CROWN_TINT,
                          cast_shadow=False, receive_shadow=False)

        self.alive = [True] * self.n
        self.max_z = [0] * self.n
        self.wait = [0] * self.n
        self.best = 0
        self.score = 0
        self.round = 1
        self.elapsed = 0.0
        self.dead_timer = 0.0
        self.finished = False
        self._world_seed = self.seed
        self.reset()

    # -- policy loading ----------------------------------------------------
    def _load_policies(self, algo: str, size: int, seed: int):
        path = os.path.join("runs", algo, "checkpoint.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as fh:
                    state = pickle.load(fh)
                cfg = state.get("config", {}) or {}
                a = make_algo(algo, OBS_SIZE, NUM_ACTIONS, cfg.get("algo_cfg", {}), seed=seed)
                a.load_state_dict(state["algo"])
                if getattr(a, "supports_parallel", False):
                    payloads = a.population_payloads()[:size]
                    pols = [vec_env._make_policy(k, pl) for (k, pl) in payloads]
                    if pols:
                        return pols, f"{algo.upper()} SWARM {len(pols)}"
                # single trained policy -> ε-varied copies for visible variety
                base = a.best_act
                pols = [self._epsilon(base, 0.15, seed * 131 + i) for i in range(size)]
                return pols, f"{algo.upper()} VARIED"
            except Exception as exc:                      # pragma: no cover - defensive
                print(f"[swarm] could not load {path}: {exc}")
        return [self._heuristic(seed * 131 + i) for i in range(size)], f"{algo.upper()} UNTRAINED"

    @staticmethod
    def _epsilon(base: Callable, eps: float, seed: int) -> Callable:
        rng = random.Random(seed)

        def policy(obs):
            if rng.random() < eps:
                return rng.randrange(NUM_ACTIONS)
            try:
                return int(base(obs))
            except Exception:
                return int(Action.UP)
        return policy

    @staticmethod
    def _heuristic(seed: int) -> Callable:
        rng = random.Random(seed)

        def policy(_obs):
            r = rng.random()
            if r < 0.70:
                return int(Action.UP)
            if r < 0.82:
                return int(Action.WAIT)
            if r < 0.91:
                return int(Action.LEFT)
            return int(Action.RIGHT)
        return policy

    @property
    def available(self) -> bool:
        return self.n > 0

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        random_state = self._world_seed
        for i, e in enumerate(self.engines):
            random.seed(random_state)          # identical world for every chicken
            e.setup_game(self.character)
            e.init()
            e.hero.stop_idle()
            self.alive[i] = True
            self.max_z[i] = 0
            self.wait[i] = 0
        self.best = 0
        self.score = 0
        self.elapsed = 0.0
        self.dead_timer = 0.0
        self.finished = False
        self._update_ghosts()                  # parent ghosts to the leader, ready to render

    def restart(self) -> None:
        self.round += 1
        self._world_seed = self.seed + self.round * 7919
        self.reset()

    @property
    def scene(self):
        return self.engines[self.best].scene

    def hud(self) -> dict:
        return {"score": self.score, "label": self.label}

    # -- per-tick ----------------------------------------------------------
    def frame_step(self, dt: float) -> None:
        if self.finished:
            return
        tween.update(dt)
        any_alive = False
        for i in range(self.n):
            if not self.alive[i]:
                continue
            e = self.engines[i]
            hero = e.hero
            if hero.is_alive and not hero.moving:
                if self.wait[i] > 0:
                    self.wait[i] -= 1
                else:
                    self._decide(i)
            e.tick(dt)
            if not hero.is_alive:
                self.alive[i] = False
            else:
                any_alive = True
                z = max(0, math.floor(hero.position.z) - config.STARTING_ROW)
                self.max_z[i] = max(self.max_z[i], z)
        self.elapsed += dt

        self._update_best()
        self._update_ghosts()

        if not any_alive:
            self.dead_timer += dt
            if self.dead_timer > 2.0:
                self.restart()

    def _decide(self, i: int) -> None:
        e = self.engines[i]
        obs = observation.build(e, self.max_z[i], self.elapsed)
        try:
            action = int(self.policies[i](obs))
        except Exception:
            action = int(Action.UP)
        if action == int(Action.WAIT):
            self.wait[i] = self.wait_ticks
            return
        e.begin_move_with_direction()
        e.move_with_direction(_DIR[Action(action)])

    def _update_best(self) -> None:
        best, best_z = self.best, -1
        for i in range(self.n):
            if self.alive[i] and self.max_z[i] > best_z:
                best, best_z = i, self.max_z[i]
        # keep the current leader if still alive and nobody strictly beat it (stable camera)
        if self.alive[self.best] and self.max_z[self.best] >= best_z:
            best = self.best
        self.best = best
        self.score = self.max_z[best]

    def _update_ghosts(self) -> None:
        world = self.engines[self.best].world
        if self.ghost_group.parent is not world:
            world.add(self.ghost_group)        # auto-reparents from the previous leader
            world.add(self.crown)
        for i in range(self.n):
            g = self.ghosts[i]
            if i == self.best or not self.alive[i]:
                g.visible = False               # leader drawn by its own engine; dead hidden
                continue
            hero = self.engines[i].hero
            g.visible = True
            g.position.copy(hero.position)
            g.rotation.set(hero.rotation.x, hero.rotation.y, hero.rotation.z)
            g.scale.copy(hero.scale)
        bh = self.engines[self.best].hero
        self.crown.visible = self.alive[self.best]
        self.crown.position.set(bh.position.x, bh.position.y + 1.15, bh.position.z)

    @property
    def alive_count(self) -> int:
        return sum(self.alive)
