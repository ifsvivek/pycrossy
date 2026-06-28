"""Live-parallel training plumbing: persistent worker pool, per-generation eval, showcase.

These back the menu's "AI Training" launch with Parallel Environments > 0, which previously
had no effect in the watched (live) mode.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from ai import vec_env
from ai.trainer import Trainer, TrainConfig


def _trainer(tmp_path):
    cfg = TrainConfig(algo="neat", logdir=str(tmp_path / "run"), seed=0,
                      target_episodes=24, eval_every=0, checkpoint_every=0,
                      use_tensorboard=False, algo_cfg={"pop_size": 6, "hidden": (16,)})
    return Trainer(cfg)


def test_make_pool_single_worker_is_inline():
    # n_workers <= 1 means "run in-process" (no real pool object)
    assert vec_env.make_pool(1, max_steps=200) is None


def test_train_generation_uses_persistent_pool(tmp_path):
    tr = _trainer(tmp_path)
    pop = len(tr.algo.population_payloads())
    pool = vec_env.make_pool(2, max_steps=300)
    assert pool is not None
    try:
        gens = []
        m1 = tr.train_generation(n_workers=2, pool=pool,
                                 on_generation=lambda m, s: gens.append(s))
        assert tr.episode == pop                      # one full generation advanced
        assert isinstance(m1, dict)
        assert len(gens) == 1
        tr.train_generation(n_workers=2, pool=pool)
        assert tr.episode == 2 * pop                  # pool reused for a second generation
    finally:
        pool.terminate()
        pool.join()
        tr.logger.close()


def test_train_generation_logger_stays_open(tmp_path):
    # Unlike train_parallel, train_generation must NOT close the logger (live loop calls it
    # repeatedly and closes the logger itself at the end).
    tr = _trainer(tmp_path)
    tr.train_generation(n_workers=1)
    tr.train_generation(n_workers=1)                  # would raise if the logger were closed
    tr.logger.close()


def test_showcase_episode_runs_without_learning(tmp_path):
    tr = _trainer(tmp_path)
    before = tr.episode
    score = tr.showcase_episode(seed=123)
    assert tr.episode == before                       # showcase does not advance training
    assert isinstance(score, int) and score >= 0
    tr.logger.close()
