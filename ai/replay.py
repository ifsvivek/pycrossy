"""Replay recording & playback.

An episode is fully determined by its reset ``seed`` (which fixes world generation) plus
the agent's ``actions``. So a replay stores just those — replaying re-runs the env and
reproduces the episode exactly, which doubles as a determinism check.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from .env import CrossyEnv


@dataclass
class Replay:
    seed: int
    actions: List[int] = field(default_factory=list)
    algo: str = ""
    score: int = 0
    reward: float = 0.0
    length: int = 0
    timestamp: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_dict(d: dict) -> "Replay":
        return Replay(seed=int(d["seed"]), actions=[int(a) for a in d["actions"]],
                      algo=d.get("algo", ""), score=int(d.get("score", 0)),
                      reward=float(d.get("reward", 0.0)), length=int(d.get("length", 0)),
                      timestamp=float(d.get("timestamp", 0.0)))


class ReplayRecorder:
    """Captures the seed + actions of the current episode."""

    def __init__(self, algo: str = ""):
        self.algo = algo
        self.seed = 0
        self.actions: List[int] = []

    def start(self, seed: int) -> None:
        self.seed = seed
        self.actions = []

    def record(self, action: int) -> None:
        self.actions.append(int(action))

    def finish(self, score: int, reward: float) -> Replay:
        return Replay(seed=self.seed, actions=list(self.actions), algo=self.algo,
                      score=score, reward=reward, length=len(self.actions),
                      timestamp=time.time())


def save(replay: Replay, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(replay.to_json())


def load(path: str) -> Replay:
    with open(path) as fh:
        return Replay.from_dict(json.load(fh))


def play(replay: Replay, env: Optional[CrossyEnv] = None,
         on_step: Optional[Callable] = None) -> dict:
    """Re-run a replay. ``on_step(env, action, result)`` is called each step (for rendering)."""
    env = env or CrossyEnv()
    env.reset(seed=replay.seed)
    last_info = {"score": 0}
    total = 0.0
    for action in replay.actions:
        res = env.step(action)
        total += res.reward
        last_info = res.info
        if on_step is not None:
            on_step(env, action, res)
        if res.done:
            break
    return {"score": last_info.get("score", 0), "reward": total, "info": last_info}


def verify(replay: Replay, env: Optional[CrossyEnv] = None, tol: float = 1e-6) -> bool:
    """True if replaying reproduces the recorded final score (determinism check)."""
    result = play(replay, env)
    return result["score"] == replay.score
