"""Gym-like environment wrapping the headless game engine.

One :class:`CrossyEnv` owns one :class:`~pycrossy.engine.Engine` (with silent audio) and
exposes ``reset`` / ``step`` over the 5-action discrete space. A *step* applies an action
and advances the fixed-60 Hz simulation until the hop settles (or a few ticks for WAIT),
so each step is one discrete decision. The reward shaping rewards forward progress and
survival while penalising death, idling and backward moves.

Note: the game uses a process-global tween manager, so only ONE env may be *active* per
process at a time. Parallel rollouts therefore use subprocess workers (see ``vec_env``),
while population/episode evaluation inside a process runs sequentially — which is correct
because each episode resets the engine (and clears tweens) before it begins.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

from pycrossy import config
from pycrossy.audio import NullAudio
from pycrossy.engine import Engine, Direction
from pycrossy.tween import tween

from . import observation


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    WAIT = 4


_DIR = {
    Action.UP: Direction.UP,
    Action.DOWN: Direction.DOWN,
    Action.LEFT: Direction.LEFT,
    Action.RIGHT: Direction.RIGHT,
}

NUM_ACTIONS = len(Action)
OBS_SIZE = observation.OBS_SIZE


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: dict


class CrossyEnv:
    def __init__(self, character: str = "chicken", max_steps: int = 1500,
                 wait_ticks: int = 8, settle_cap: int = 24, max_idle: int = 150,
                 seed: Optional[int] = None):
        self.character = character
        self.max_steps = max_steps
        self.wait_ticks = wait_ticks
        self.settle_cap = settle_cap
        self.max_idle = max_idle
        self.engine = Engine(audio=NullAudio())
        self.engine.is_game_state_ended = lambda: False  # "playing" until death
        self._rng = random.Random(seed)
        self.max_z = 0
        self.steps = 0
        self.idle = 0
        self.elapsed = 0.0
        self._np_random = np.random.default_rng(seed)
        # Optional per-tick callback used by the live trainer to render the AI play window
        # (and pump its events) as the episode advances. None in headless training.
        self.on_tick = None

    # -- gym API -----------------------------------------------------------
    def seed(self, seed: int) -> None:
        self._rng.seed(seed)
        self._np_random = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.seed(seed)
        # Seed the game's module-level RNG so world generation is reproducible per episode.
        random.seed(self._rng.randint(0, 2 ** 31 - 1))
        self.engine.setup_game(self.character)
        self.engine.init()
        self.engine.hero.stop_idle()
        self.max_z = 0
        self.steps = 0
        self.idle = 0
        self.elapsed = 0.0
        return self._obs()

    def step(self, action: int) -> StepResult:
        action = Action(int(action))
        prev_max = self.max_z
        if action == Action.WAIT:
            self._advance(self.wait_ticks)
        else:
            self.engine.begin_move_with_direction()
            self.engine.move_with_direction(_DIR[action])
            self._advance_until_settled()

        self.steps += 1
        new_z = max(0, math.floor(self.engine.hero.position.z) - config.STARTING_ROW)
        self.max_z = max(self.max_z, new_z)
        forward = self.max_z - prev_max
        alive = self.engine.hero.is_alive

        # Reward design (see docs/AI_AUDIT.md R2): a uniform NET-NEGATIVE time cost makes
        # standing still strictly lose, so any forward progress dominates loitering — the
        # old +0.05/step survival bonus rewarded waiting (it scaled with episode length).
        # Forward progress is potential-based shaping on max_z (Δ of the furthest row), so
        # the optimal policy is unchanged; the death penalty is ~one row, not a 5-row cliff.
        reward = -0.01                                  # time cost: progress or perish
        reward += forward * 1.0                         # forward progress (the only + term)
        if action == Action.DOWN:
            reward -= 0.3                               # discourage backtracking
        elif action == Action.UP and forward == 0:
            reward -= 0.05                              # wasted hop into an obstacle/wall

        if forward > 0:
            self.idle = 0                               # idle counts steps since a NEW max row
        else:
            self.idle += 1

        done = False
        if not alive:
            reward -= 1.0                               # death ≈ losing one row of progress
            done = True
        elif self.steps >= self.max_steps:
            done = True
        elif self.idle >= self.max_idle:
            done = True                                 # stuck too long; the time cost is penalty enough

        info = {"score": self.max_z, "steps": self.steps, "alive": alive,
                "elapsed": self.elapsed}
        return StepResult(self._obs(), float(reward), done, info)

    # -- helpers -----------------------------------------------------------
    def _advance(self, ticks: int) -> None:
        dt = config.FIXED_DT
        for _ in range(ticks):
            tween.update(dt)
            self.engine.tick(dt)
            self.elapsed += dt
            if self.on_tick is not None:
                self.on_tick()
            if not self.engine.hero.is_alive:
                break

    def _advance_until_settled(self) -> None:
        dt = config.FIXED_DT
        for _ in range(self.settle_cap):
            tween.update(dt)
            self.engine.tick(dt)
            self.elapsed += dt
            if self.on_tick is not None:
                self.on_tick()
            if not self.engine.hero.is_alive:
                break
            if not self.engine.hero.moving:
                break

    def _obs(self) -> np.ndarray:
        return observation.build(self.engine, self.max_z, self.elapsed)

    @property
    def scene(self):
        return self.engine.scene
