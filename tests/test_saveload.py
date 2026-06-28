"""Save/load round-trips for every algorithm + trainer checkpoint/resume."""
from __future__ import annotations

import pytest

import ai.algorithms as algorithms
from ai.base import Transition, make
from ai.env import CrossyEnv, NUM_ACTIONS, OBS_SIZE
from ai.trainer import Trainer, TrainConfig


def _cfg(name):
    if name in ("neat", "es", "ga", "cmaes"):
        return {"pop_size": 8, "hidden": (16,)}
    if name == "dqn":
        return {"warmup": 20, "hidden": (32,)}
    return {"rollout_size": 64, "hidden": (32,)}


def _run_episodes(env, algo, n):
    for _ in range(n):
        algo.begin_episode()
        obs = env.reset()
        done = False
        res = None
        while not done:
            a = algo.act(obs)
            res = env.step(a)
            algo.observe(Transition(obs, a, res.reward, res.obs, res.done))
            obs = res.obs
            done = res.done
        algo.end_episode(0.0, res.info)


@pytest.mark.parametrize("name", algorithms.available())
def test_algorithm_state_roundtrip(name):
    env = CrossyEnv(seed=0)
    cfg = _cfg(name)
    algo = make(name, OBS_SIZE, NUM_ACTIONS, cfg, seed=0)
    n = cfg.get("pop_size", 3)
    _run_episodes(env, algo, n)

    state = algo.state_dict()
    clone = make(name, OBS_SIZE, NUM_ACTIONS, cfg, seed=99)
    clone.load_state_dict(state)

    obs = env.reset(seed=5)
    # The best-known policy must be identical after a save/load round-trip.
    assert algo.best_act(obs) == clone.best_act(obs)


def test_trainer_checkpoint_resume(tmp_path):
    cfg = TrainConfig(algo="dqn", logdir=str(tmp_path / "run"), use_tensorboard=False,
                      eval_every=0, checkpoint_every=0, algo_cfg={"warmup": 20})
    tr = Trainer(cfg)
    for _ in range(5):
        tr.train_episode()
    path = tr.save_checkpoint()
    tr.logger.close()

    cfg2 = TrainConfig(algo="dqn", logdir=str(tmp_path / "run2"), use_tensorboard=False,
                       algo_cfg={"warmup": 20})
    tr2 = Trainer(cfg2)
    tr2.load_checkpoint(path)
    assert tr2.episode == tr.episode
    assert tr2.best_score == tr.best_score
    tr2.logger.close()
