"""Rigorous, deterministic evaluation of a trained policy.

A model is judged by its statistics over MANY independent held-out episodes — never a
single run. :func:`evaluate_policy` rolls a deterministic policy across a fixed bank of
seeds and returns the full distribution (mean / median / best / worst / std / percentiles /
success-rate at several score thresholds / survival time). The held-out seed bank is
disjoint from the seeds used during training so reported numbers reflect generalisation,
not memorised maps.

Usable three ways:
    from ai.evaluate import evaluate_policy, evaluate_model
    python -m ai.evaluate runs/neat/best_model.pkl --episodes 200
    python -m ai.evaluate runs/neat/best_model.pkl runs/ppo/best_model.pkl --episodes 200
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .env import CrossyEnv

# Held-out evaluation seeds live in a high, fixed band so they never overlap the
# training seeds (seed*100003 + episode) or the small banks used for in-training eval.
EVAL_SEED_BASE = 9_000_000
SUCCESS_THRESHOLDS = (5, 10, 20, 30, 50)


def eval_seeds(n: int, base: int = EVAL_SEED_BASE) -> List[int]:
    return [base + i for i in range(n)]


@dataclass
class EvalReport:
    n: int
    mean: float
    median: float
    best: int
    worst: int
    std: float
    p25: float
    p75: float
    mean_length: float
    mean_survival_s: float
    success_rate: Dict[int, float] = field(default_factory=dict)
    scores: List[int] = field(default_factory=list)
    lengths: List[int] = field(default_factory=list)
    elapsed_s: float = 0.0

    def summary(self) -> Dict:
        d = asdict(self)
        d.pop("scores"); d.pop("lengths")
        return d

    def format(self, name: str = "") -> str:
        sr = "  ".join(f">={t}:{self.success_rate.get(t, 0.0)*100:4.0f}%" for t in SUCCESS_THRESHOLDS)
        return (f"{name:<22} n={self.n:<4d} mean={self.mean:6.2f}  median={self.median:5.1f}  "
                f"best={self.best:3d}  worst={self.worst:3d}  std={self.std:5.2f}  "
                f"p25={self.p25:4.1f} p75={self.p75:4.1f}  len={self.mean_length:5.1f}  | {sr}")


def _report(scores: List[int], lengths: List[int], survivals: List[float],
            elapsed: float) -> EvalReport:
    sc = sorted(scores)
    n = len(scores)
    def pct(p):
        return float(sc[min(n - 1, max(0, int(p * n)))]) if n else 0.0
    return EvalReport(
        n=n,
        mean=float(statistics.mean(scores)) if scores else 0.0,
        median=float(statistics.median(scores)) if scores else 0.0,
        best=max(scores) if scores else 0,
        worst=min(scores) if scores else 0,
        std=float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0,
        p25=pct(0.25), p75=pct(0.75),
        mean_length=float(statistics.mean(lengths)) if lengths else 0.0,
        mean_survival_s=float(statistics.mean(survivals)) if survivals else 0.0,
        success_rate={t: (sum(1 for x in scores if x >= t) / n if n else 0.0)
                      for t in SUCCESS_THRESHOLDS},
        scores=list(scores), lengths=list(lengths), elapsed_s=elapsed,
    )


def evaluate_policy(act_fn: Callable[[np.ndarray], int], seeds: Sequence[int],
                    max_steps: int = 1500, env: Optional[CrossyEnv] = None) -> EvalReport:
    """Roll a deterministic ``act_fn`` over ``seeds`` and return the score distribution."""
    own = env is None
    env = env or CrossyEnv(max_steps=max_steps)
    scores: List[int] = []
    lengths: List[int] = []
    survivals: List[float] = []
    t0 = time.time()
    for s in seeds:
        obs = env.reset(seed=int(s))
        done = False
        res = None
        while not done:
            res = env.step(int(act_fn(obs)))
            obs = res.obs
            done = res.done
        scores.append(int(res.info["score"]))
        lengths.append(int(res.info["steps"]))
        survivals.append(float(res.info.get("elapsed", 0.0)))
    rep = _report(scores, lengths, survivals, time.time() - t0)
    if own:
        del env
    return rep


def evaluate_model(path: str, episodes: int = 100, max_steps: int = 1500,
                   seeds: Optional[Sequence[int]] = None) -> EvalReport:
    """Load a saved ``best_model.pkl`` and evaluate its deterministic ``best_act`` policy."""
    from .trainer import Trainer
    policy = Trainer.load_policy(path)
    env = CrossyEnv(max_steps=max_steps)
    if getattr(policy, "uses_planning", False):
        policy.bind_env(env)                 # planning agents act on the live engine
    return evaluate_policy(policy.best_act, seeds or eval_seeds(episodes),
                           max_steps=max_steps, env=env)


def evaluate_algo(name: str, cfg: Optional[dict] = None, episodes: int = 100,
                  max_steps: int = 1500, seeds: Optional[Sequence[int]] = None) -> EvalReport:
    """Construct an algorithm by name (no checkpoint needed) and evaluate it — used for
    planning agents like ``minimax`` that have nothing to load."""
    from . import algorithms  # noqa: F401 — ensure the registry is populated
    from .base import make
    from .env import NUM_ACTIONS, OBS_SIZE
    algo = make(name, OBS_SIZE, NUM_ACTIONS, cfg or {})
    env = CrossyEnv(max_steps=max_steps)
    if getattr(algo, "uses_planning", False):
        algo.bind_env(env)
    return evaluate_policy(algo.best_act, seeds or eval_seeds(episodes),
                           max_steps=max_steps, env=env)


def _cli(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluate trained PyCrossy policies over many held-out seeds")
    p.add_argument("models", nargs="+", help="path(s) to best_model.pkl (or a run dir)")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--json", default=None, help="write the full report table to this JSON file")
    args = p.parse_args(argv)

    seeds = eval_seeds(args.episodes)
    out = {}
    print(f"Evaluating over {args.episodes} held-out seeds (base {EVAL_SEED_BASE})\n")
    for m in args.models:
        path = m if os.path.isfile(m) else os.path.join(m, "best_model.pkl")
        name = os.path.basename(os.path.dirname(path)) or path
        if not os.path.exists(path):
            print(f"{name:<22} (missing: {path})")
            continue
        rep = evaluate_model(path, episodes=args.episodes, max_steps=args.max_steps, seeds=seeds)
        print(rep.format(name))
        out[name] = rep.summary()
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    _cli()
