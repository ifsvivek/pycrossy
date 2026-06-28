"""Parallel environment evaluation via a subprocess pool.

Population-based algorithms expose ``(kind, payload)`` policy specs (a NEAT genome or an
MLP parameter vector); this module evaluates a whole generation across worker processes —
one episode per candidate — and returns ``(reward, score, length)`` for each.

This is the right parallelism for this codebase: the game uses a process-global tween
manager, so each worker runs its own env in its own process (sequential episodes within a
worker), avoiding cross-env interference while using all cores.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Callable, List, Tuple

import numpy as np

from .env import CrossyEnv


def _pool_context():
    """Multiprocessing context for the worker pool, portable across OSes.

    Prefer ``fork`` where available (Linux): it copies the already-loaded game modules, so
    workers start instantly, and it avoids Python 3.14's default ``forkserver`` (slower start,
    brittle once native GUI libs are imported). On platforms without ``fork`` (Windows, and the
    safer default on macOS) fall back to ``spawn`` — workers only run numpy/env logic and all
    payloads are picklable, so it works the same, just with slower process startup.
    """
    return mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")


def _make_policy(kind: str, payload) -> Callable[[np.ndarray], int]:
    if kind == "neat":
        genome = payload
        genome._rebuild_index()
        return lambda obs: int(np.argmax(genome.forward(obs)))
    if kind == "mlp":
        from .networks import MLP
        params, sizes, activation = payload
        net = MLP(list(sizes), activation=activation)
        net.set_params(np.asarray(params))
        return lambda obs: net.act_argmax(obs)
    raise ValueError(f"unknown policy kind {kind!r}")


# Per-worker cached env (reused across the worker's tasks).
_WORKER_ENV: CrossyEnv | None = None
_WORKER_MAX_STEPS = 1500


def _worker_init(max_steps: int) -> None:
    global _WORKER_ENV, _WORKER_MAX_STEPS
    _WORKER_MAX_STEPS = max_steps
    _WORKER_ENV = CrossyEnv(max_steps=max_steps)


def _eval_one(args) -> Tuple[float, int, int]:
    kind, payload, seed = args
    global _WORKER_ENV
    if _WORKER_ENV is None:
        _WORKER_ENV = CrossyEnv(max_steps=_WORKER_MAX_STEPS)
    env = _WORKER_ENV
    policy = _make_policy(kind, payload)
    obs = env.reset(seed=seed)
    total = 0.0
    length = 0
    res = None
    done = False
    while not done:
        res = env.step(policy(obs))
        total += res.reward
        obs = res.obs
        length += 1
        done = res.done
    return total, res.info["score"], length


def make_pool(n_workers: int | None = None, max_steps: int = 1500):
    """Create a persistent fork pool of evaluation workers, or ``None`` to run in-process.

    For live training, create this ONCE up front (from the main thread, before opening any
    GL window) and reuse it every generation via :func:`parallel_evaluate` — so workers are
    forked exactly once, never from inside the render loop or a background thread (which would
    be unsafe to fork while a GL context / extra threads exist).
    """
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    if n_workers <= 1:
        _worker_init(max_steps)
        return None
    ctx = _pool_context()
    return ctx.Pool(processes=n_workers, initializer=_worker_init, initargs=(max_steps,))


def parallel_evaluate(payloads: List[Tuple[str, object]], seeds: List[int],
                      max_steps: int = 1500, n_workers: int | None = None, pool=None
                      ) -> List[Tuple[float, int, int]]:
    """Evaluate each ``(kind, payload)`` for one episode in parallel. Returns per-candidate
    ``(reward, score, length)`` aligned with ``payloads``.

    If ``pool`` (from :func:`make_pool`) is given it is reused; otherwise a throwaway pool is
    created for this call (or evaluated in-process when ``n_workers == 1``).
    """
    tasks = [(k, pl, s) for (k, pl), s in zip(payloads, seeds)]
    if pool is not None:
        return pool.map(_eval_one, tasks)
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    if n_workers == 1:
        _worker_init(max_steps)
        return [_eval_one(t) for t in tasks]
    ctx = _pool_context()      # 'fork' on Linux (fast), 'spawn' on Windows/macOS (see _pool_context)
    with ctx.Pool(processes=n_workers, initializer=_worker_init, initargs=(max_steps,)) as p:
        return p.map(_eval_one, tasks)
