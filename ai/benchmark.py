"""Algorithm benchmarking — train and/or evaluate every algorithm under identical conditions.

Two uses:

* **Compare existing models** (fast) — evaluate each ``best_model.pkl`` on ONE shared bank of
  held-out seeds (identical maps for every algorithm), so the comparison is apples-to-apples:

      python -m ai.benchmark --compare runs/neat2 runs/cmaes2 runs/dqn2 runs/ppo2 --episodes 200

* **Full benchmark** (train then evaluate) — train each algorithm across several random seeds,
  then evaluate each on the shared bank and aggregate to mean ± 95% CI across the training
  seeds (so a lucky single training run can't win):

      python -m ai.benchmark --algos neat,ppo,dqn,cmaes --seeds 0,1,2 --episodes 6000

Results (the full statistic table the audit asks for) are written to ``runs/bench/results.json``
and printed. Reuses :mod:`ai.evaluate` for the deterministic multi-seed evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .evaluate import EvalReport, eval_seeds, evaluate_algo, evaluate_model

BENCH_DIR = "runs/bench"


def _is_planning_name(name: str) -> bool:
    """True if ``name`` is a registered planning agent (e.g. ``minimax``) — evaluated by
    construction rather than from a saved checkpoint."""
    from . import algorithms  # noqa: F401 — populate registry
    from .base import _REGISTRY
    cls = _REGISTRY.get(name.lower())
    return bool(cls) and getattr(cls, "uses_planning", False)


def _ci95(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))


def compare(model_paths: Sequence[str], episodes: int = 200, max_steps: int = 1500
            ) -> Dict[str, EvalReport]:
    """Evaluate each model on the SAME held-out seed bank and print a comparison table."""
    seeds = eval_seeds(episodes)
    reports: Dict[str, EvalReport] = {}
    print(f"Comparing {len(model_paths)} models over {episodes} shared held-out seeds\n")
    for m in model_paths:
        try:
            if not (os.path.isfile(m) or os.path.isdir(m)) and _is_planning_name(m):
                rep = evaluate_algo(m, {}, episodes=episodes, max_steps=max_steps, seeds=seeds)
                reports[m] = rep
                print(rep.format(m))
                continue
            path = m if os.path.isfile(m) else os.path.join(m, "best_model.pkl")
            name = os.path.basename(os.path.dirname(path)) or path
            if not os.path.exists(path):
                print(f"{name:<20} (missing {path})")
                continue
            rep = evaluate_model(path, episodes=episodes, max_steps=max_steps, seeds=seeds)
        except Exception as e:                       # e.g. a model from an incompatible obs/net
            print(f"{m:<20} (load/eval failed: {type(e).__name__}: {e})")
            continue
        reports[name] = rep
        print(rep.format(name))
    return reports


def _train_one(algo: str, seed: int, episodes: int, eval_episodes: int,
               extra_set: Sequence[str], logdir: str) -> bool:
    """Shell out to train.py (headless) for one (algo, seed). Returns success."""
    cmd = [sys.executable, "train.py", "--algo", algo, "--headless",
           "--episodes", str(episodes), "--eval-episodes", str(eval_episodes),
           "--seed", str(seed), "--no-tensorboard", "--logdir", logdir]
    for kv in extra_set or []:
        cmd += ["--set", kv]
    print(f"  [train] {algo} seed={seed} → {logdir}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [train] FAILED ({algo} seed={seed}):\n{r.stderr[-800:]}")
        return False
    return True


def benchmark(algos: Sequence[str], train_seeds: Sequence[int] = (0,), episodes: int = 6000,
              eval_episodes: int = 4, n_eval: int = 100, max_steps: int = 1500,
              train: bool = True, extra_set: Optional[Sequence[str]] = None) -> Dict:
    """Train (optionally) and evaluate every algorithm across ``train_seeds``; aggregate to
    mean ± 95% CI of the held-out mean score across seeds. Returns a results dict and writes
    ``runs/bench/results.json``."""
    os.makedirs(BENCH_DIR, exist_ok=True)
    seeds = eval_seeds(n_eval)
    results: Dict[str, Dict] = {}
    for algo in algos:
        per_seed_mean: List[float] = []
        pooled_scores: List[int] = []
        runs: List[Dict] = []
        t0 = time.time()
        for s in train_seeds:
            logdir = os.path.join(BENCH_DIR, algo, f"seed{s}")
            model = os.path.join(logdir, "best_model.pkl")
            if train:
                if not _train_one(algo, s, episodes, eval_episodes, extra_set or [], logdir):
                    continue
            if not os.path.exists(model):
                print(f"  [eval] no model for {algo} seed={s} ({model})")
                continue
            rep = evaluate_model(model, episodes=n_eval, max_steps=max_steps, seeds=seeds)
            per_seed_mean.append(rep.mean)
            pooled_scores.extend(rep.scores)
            runs.append({"seed": s, **rep.summary()})
            print(f"  {rep.format(f'{algo}:seed{s}')}")
        if not per_seed_mean:
            continue
        ps = sorted(pooled_scores)
        results[algo] = {
            "n_train_seeds": len(per_seed_mean),
            "mean_score": float(np.mean(per_seed_mean)),
            "ci95": _ci95(per_seed_mean),
            "best_seed_mean": float(max(per_seed_mean)),
            "worst_seed_mean": float(min(per_seed_mean)),
            "pooled_best": int(max(ps)) if ps else 0,
            "pooled_median": float(ps[len(ps) // 2]) if ps else 0.0,
            "pooled_success_rate_5": sum(1 for x in ps if x >= 5) / len(ps) if ps else 0.0,
            "wall_clock_s": round(time.time() - t0, 1),
            "runs": runs,
        }
    _report(results)
    out = os.path.join(BENCH_DIR, "results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out}")
    return results


def _report(results: Dict) -> None:
    print("\n" + "=" * 78)
    print(f"{'algo':<10} {'mean±CI95':>16} {'best_seed':>10} {'pooled_best':>12} "
          f"{'succ@5':>8} {'time_s':>8}")
    print("-" * 78)
    for algo, r in sorted(results.items(), key=lambda kv: kv[1]["mean_score"], reverse=True):
        print(f"{algo:<10} {r['mean_score']:8.2f} ±{r['ci95']:5.2f} {r['best_seed_mean']:10.2f} "
              f"{r['pooled_best']:12d} {r['pooled_success_rate_5']*100:7.0f}% {r['wall_clock_s']:8.0f}")
    print("=" * 78)


def _cli(argv=None) -> None:
    p = argparse.ArgumentParser(description="Benchmark PyCrossy RL algorithms under identical conditions")
    p.add_argument("--compare", nargs="+", help="evaluate existing run dirs/models on shared seeds and exit")
    p.add_argument("--algos", default="neat,cmaes,dqn,ppo", help="comma-separated algorithms to benchmark")
    p.add_argument("--seeds", default="0", help="comma-separated training seeds")
    p.add_argument("--episodes", type=int, default=6000)
    p.add_argument("--eval-episodes", type=int, default=4, help="CRN K / per-eval episodes for training")
    p.add_argument("--n-eval", type=int, default=100, help="held-out seeds for final evaluation")
    p.add_argument("--no-train", action="store_true", help="evaluate existing runs/bench models only")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="passthrough --set to train.py")
    args = p.parse_args(argv)

    if args.compare:
        compare(args.compare, episodes=args.n_eval)
        return
    benchmark([a.strip() for a in args.algos.split(",")],
              train_seeds=[int(s) for s in args.seeds.split(",")],
              episodes=args.episodes, eval_episodes=args.eval_episodes, n_eval=args.n_eval,
              train=not args.no_train, extra_set=args.overrides)


if __name__ == "__main__":
    _cli()
