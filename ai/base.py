"""The pluggable algorithm interface + registry.

Every algorithm — gradient (PPO/A2C/DQN) or evolutionary (NEAT/ES/GA/CMA-ES) — implements
the same episodic learner protocol, so the trainer and the live dashboard drive them
identically and they are selectable by name without touching game or trainer code:

    begin_episode()                   # choose the policy to evaluate this episode
    act(obs, deterministic) -> int    # action from that policy
    observe(transition)               # record a step (RL methods use it; evolution ignores)
    end_episode(total_reward, info)   # learn / advance generation; returns a metrics dict
    best_act(obs) -> int              # best-known policy (eval / live "champion" view)
    state_dict()/load_state_dict()    # checkpointing
    progress                          # extra live stats (generation, species, epsilon, ...)
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, List, Type

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class Algorithm(abc.ABC):
    """Common interface for all interchangeable learning algorithms."""

    name: str = "base"

    # Planning agents (e.g. Minimax) need the live simulator, not just the obs vector, so the
    # drivers call ``bind_env`` before running an episode. Learners leave this False / no-op.
    uses_planning: bool = False

    def bind_env(self, env) -> None:
        """Give a planning agent access to the env it is about to act in (no-op for learners)."""

    def __init__(self, obs_size: int, num_actions: int, cfg: Dict | None = None,
                 seed: int = 0):
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.cfg = cfg or {}
        self.rng = np.random.default_rng(seed)
        self.total_episodes = 0
        self.total_steps = 0

    # -- per-episode lifecycle --------------------------------------------
    def begin_episode(self) -> None:
        """Select / prepare the policy evaluated during the next episode."""

    @abc.abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Return an action for the current episode's policy."""

    def observe(self, tr: Transition) -> None:
        """Record one transition (used by value/policy-gradient methods)."""
        self.total_steps += 1

    @abc.abstractmethod
    def end_episode(self, total_reward: float, info: Dict) -> Dict:
        """Finish the episode (maybe learn / advance a generation). Return metrics."""

    # -- evaluation / inference -------------------------------------------
    @abc.abstractmethod
    def best_act(self, obs: np.ndarray) -> int:
        """Action from the best-known policy (deterministic)."""

    # -- persistence -------------------------------------------------------
    @abc.abstractmethod
    def state_dict(self) -> Dict:
        ...

    @abc.abstractmethod
    def load_state_dict(self, state: Dict) -> None:
        ...

    # -- live stats --------------------------------------------------------
    @property
    def progress(self) -> Dict:
        """Algorithm-specific live numbers for the dashboard."""
        return {}

    # -- optional parallel-evaluation protocol ----------------------------
    # Population-based algorithms (NEAT, ES, GA, CMA-ES) implement these so the trainer
    # can evaluate the whole population across subprocess workers (see ai.vec_env).
    supports_parallel: bool = False

    def population_payloads(self) -> List:
        """Return picklable ``(kind, payload)`` policy specs for the current population."""
        raise NotImplementedError

    def set_population_fitness(self, fits: List[float]) -> Dict:
        """Apply parallel-evaluated fitnesses, advance a generation, return metrics."""
        raise NotImplementedError

    def set_validated_champion(self, payload) -> None:
        """Adopt ``payload`` (a ``(kind, spec)`` from :meth:`population_payloads`) as the
        best-known policy used by :meth:`best_act` and saved by the trainer. Called by the
        trainer after a candidate wins on the held-out validation seed bank, so the saved
        champion is a re-validated policy rather than a single-episode lucky outlier."""
        raise NotImplementedError

    def center_payload(self):
        """An extra ``(kind, spec)`` policy to also put through held-out validation (e.g. the
        search distribution's centre for ES/CMA-ES). ``None`` if there is no such candidate."""
        return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Type[Algorithm]] = {}


def register(name: str):
    def deco(cls: Type[Algorithm]) -> Type[Algorithm]:
        cls.name = name
        _REGISTRY[name.lower()] = cls
        return cls
    return deco


def make(name: str, obs_size: int, num_actions: int, cfg: Dict | None = None,
         seed: int = 0) -> Algorithm:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown algorithm '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](obs_size, num_actions, cfg, seed)


def available() -> List[str]:
    return sorted(_REGISTRY)
