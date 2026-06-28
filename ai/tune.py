"""Automated hyperparameter optimization — no external dependencies.

Random or grid search over a per-algorithm search space. Each trial is scored by the **same
held-out evaluation** used everywhere else (``ai.evaluate``), so trials are comparable and the
ranking reflects generalisation, not training-fitness noise. Learners are trained (via
``train.py`` in a subprocess) then evaluated; planning agents (``minimax``) are evaluated
directly across configs. Results persist to ``runs/tune/<algo>/trials.csv`` + ``best.json``.

    python -m ai.tune --algo minimax --trials 8 --n-eval 40
    python -m ai.tune --algo ddqn --trials 6 --episodes 1500 --n-eval 40
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
from typing import Dict, List

from .evaluate import eval_seeds, evaluate_algo, evaluate_model

TUNE_DIR = "runs/tune"

# Tuned search spaces (NOT library defaults). Values are concrete candidates per knob.
SEARCH_SPACES: Dict[str, Dict[str, list]] = {
    "ddqn": {
        "lr": [1e-3, 5e-4, 3e-4, 1e-4],
        "gamma": [0.97, 0.99, 0.995],
        "batch": [32, 64, 128],
        "buffer_size": [50_000, 100_000],
        "target_sync": [500, 1_000, 2_000],
        "n_step": [1, 3, 5],
        "per_alpha": [0.4, 0.6, 0.8],
        "hidden": ["128,128", "256,256", "128,128,64"],
        "eps_decay_steps": [8_000, 15_000, 30_000],
    },
    "dqn": {
        "lr": [1e-3, 5e-4, 3e-4],
        "gamma": [0.97, 0.99],
        "batch": [32, 64, 128],
        "target_sync": [250, 500, 1_000],
        "hidden": ["128,128", "256,256"],
        "eps_decay_steps": [8_000, 15_000, 30_000],
    },
    "ppo": {
        "lr": [3e-4, 1e-3, 1e-4],
        "gamma": [0.97, 0.99, 0.995],
        "gae_lambda": [0.9, 0.95, 0.98],
        "clip": [0.1, 0.2, 0.3],
        "ent_coef": [0.0, 0.01, 0.03],
        "rollout_size": [256, 512, 1024],
        "hidden": ["64,64", "128,128"],
    },
    "minimax": {
        "max_depth": [3, 4, 5, 6, 7, 8],
        "w_progress": [6.0, 10.0, 14.0],
        "w_safety": [3.0, 6.0, 9.0],
        "w_edge": [1.0, 2.0, 4.0],
        "time_budget_ms": [40.0],
    },
}

_PLANNING = {"minimax"}


def _candidates(space: Dict[str, list], n_trials: int, grid: bool, rng: random.Random) -> List[dict]:
    if grid:
        keys = list(space)
        combos = list(itertools.product(*(space[k] for k in keys)))
        rng.shuffle(combos)
        return [dict(zip(keys, c)) for c in combos[:n_trials]]
    out, seen = [], set()
    for _ in range(n_trials * 5):
        cfg = {k: rng.choice(v) for k, v in space.items()}
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            out.append(cfg)
        if len(out) >= n_trials:
            break
    return out


def _coerce(cfg: dict) -> dict:
    """Turn CLI-style values (e.g. hidden='128,128') into proper python for in-process use."""
    out = {}
    for k, v in cfg.items():
        if isinstance(v, str) and "," in v:
            out[k] = tuple(int(x) for x in v.split(","))
        else:
            out[k] = v
    return out


def _train_subprocess(algo: str, cfg: dict, episodes: int, logdir: str, seed: int) -> bool:
    cmd = [sys.executable, "train.py", "--algo", algo, "--headless", "--episodes", str(episodes),
           "--seed", str(seed), "--no-tensorboard", "--logdir", logdir, "--eval-every", "100"]
    for k, v in cfg.items():
        cmd += ["--set", f"{k}={v}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"   trial train failed: {r.stderr[-400:]}")
    return r.returncode == 0


def tune(algo: str, n_trials: int = 8, episodes: int = 1500, n_eval: int = 40,
         grid: bool = False, seed: int = 0) -> dict:
    if algo not in SEARCH_SPACES:
        raise SystemExit(f"no search space for {algo!r}; have {sorted(SEARCH_SPACES)}")
    rng = random.Random(seed)
    cfgs = _candidates(SEARCH_SPACES[algo], n_trials, grid, rng)
    out_dir = os.path.join(TUNE_DIR, algo)
    os.makedirs(out_dir, exist_ok=True)
    seeds = eval_seeds(n_eval)
    rows, best = [], None
    print(f"Tuning {algo}: {len(cfgs)} trials, scored on {n_eval} held-out seeds\n")
    for i, cfg in enumerate(cfgs):
        if algo in _PLANNING:
            rep = evaluate_algo(algo, _coerce(cfg), episodes=n_eval, seeds=seeds)
        else:
            logdir = os.path.join(out_dir, f"trial{i}")
            if not _train_subprocess(algo, cfg, episodes, logdir, seed):
                continue
            model = os.path.join(logdir, "best_model.pkl")
            if not os.path.exists(model):
                continue
            rep = evaluate_model(model, episodes=n_eval, seeds=seeds)
        score = rep.mean
        rows.append({"trial": i, "score": round(score, 3), "median": rep.median,
                     "best": rep.best, **cfg})
        print(f"  trial {i:2d}  mean={score:6.2f}  median={rep.median:5.1f}  best={rep.best:4d}  {cfg}")
        if best is None or score > best["score"]:
            best = {"score": round(score, 3), "config": cfg}
    # persist
    if rows:
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(os.path.join(out_dir, "trials.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(out_dir, "best.json"), "w") as fh:
        json.dump(best, fh, indent=2)
    print(f"\nBEST {algo}: {best}")
    print(f"wrote {out_dir}/trials.csv + best.json")
    return best


def _cli(argv=None) -> None:
    p = argparse.ArgumentParser(description="Hyperparameter search for PyCrossy RL algorithms")
    p.add_argument("--algo", default="minimax")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--episodes", type=int, default=1500, help="training episodes per learner trial")
    p.add_argument("--n-eval", type=int, default=40, help="held-out seeds to score each trial")
    p.add_argument("--grid", action="store_true", help="grid search instead of random")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    tune(args.algo, n_trials=args.trials, episodes=args.episodes, n_eval=args.n_eval,
         grid=args.grid, seed=args.seed)


if __name__ == "__main__":
    _cli()
