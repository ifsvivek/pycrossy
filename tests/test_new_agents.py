"""Tests for the two new agents: DDQN (n-step / PER / anneal) and the Minimax planner."""
from __future__ import annotations

import numpy as np

import ai.algorithms  # noqa: F401 — registers ddqn / minimax / … in the algorithm registry
from ai.base import make
from ai.base import Transition
from ai.env import Action, CrossyEnv, NUM_ACTIONS, OBS_SIZE


# --------------------------------------------------------------------------- DDQN
def test_ddqn_nstep_return_is_discounted_sum():
    """A 3-step window with rewards [1,1,1] and γ=0.9 must store R = 1 + 0.9 + 0.81."""
    algo = make("ddqn", OBS_SIZE, NUM_ACTIONS,
                {"n_step": 3, "gamma": 0.9, "obs_norm": False, "per": False,
                 "warmup": 10_000, "reward_norm": False}, seed=0)
    z = np.zeros(OBS_SIZE, np.float32)
    for _ in range(3):
        algo.observe(Transition(z, 0, 1.0, z, False))
    assert len(algo.buffer) == 1
    assert abs(float(algo.buffer.rewards[0]) - (1 + 0.9 + 0.81)) < 1e-5


def test_ddqn_per_buffer_samples_valid():
    from ai.algorithms.ddqn import PrioritizedReplay
    buf = PrioritizedReplay(64, OBS_SIZE, alpha=0.6)
    z = np.zeros(OBS_SIZE, np.float32)
    for i in range(20):
        buf.add(z, i % 5, float(i), z, 0.0, 1)
    data_idx, tree_idx, weights = buf.sample(8, beta=0.4)
    assert len(data_idx) == 8 and len(weights) == 8
    assert (weights > 0).all() and (weights <= 1.0 + 1e-6).all()
    buf.update_priorities(tree_idx, np.ones(8))      # must not raise


def test_ddqn_trains_and_anneals():
    algo = make("ddqn", OBS_SIZE, NUM_ACTIONS,
                {"warmup": 200, "buffer_size": 5000, "hidden": (64, 64),
                 "eps_decay_steps": 1000}, seed=0)
    env = CrossyEnv(max_steps=120, seed=0)
    obs = env.reset(seed=0)
    for _ in range(8):
        done = False
        while not done:
            a = algo.act(obs)
            res = env.step(a)
            algo.observe(Transition(obs, a, res.reward, res.obs, res.done))
            obs = res.obs
            done = res.done
        obs = env.reset(seed=int(algo.total_steps))
    assert algo.total_steps > 200
    assert algo.epsilon < algo.eps_start          # ε annealed
    assert np.isfinite(algo.last_loss)            # learning produced a finite loss


# --------------------------------------------------------------------------- Minimax
def test_minimax_forward_model_predicts_death_and_avoids_it():
    from ai.algorithms.search_agent import _Model, _Row, _Mover, _State, _Search
    rows = {
        8: _Row("grass", [], frozenset(), 22.0),                       # current row (safe)
        9: _Row("road", [_Mover(0.0, 0.0, 0.5, False)], frozenset(), 22.0),  # car parked on col 0
        10: _Row("grass", [], frozenset(), 22.0),
    }
    m = _Model(rows, 7, 12)
    s = _State(0.0, 8, False, 0, True, 0)
    srch = _Search(m, 10.0, 6.0, 2.0)
    assert srch.step(s, Action.UP).alive is False          # stepping into the car is fatal
    best, _ = srch.best_action(s, 4)
    assert best != Action.UP                               # the planner must not walk into it


def test_minimax_far_outperforms_random():
    mm = make("minimax", OBS_SIZE, NUM_ACTIONS, {"max_depth": 4, "time_budget_ms": 20})
    env = CrossyEnv(max_steps=600)
    mm.bind_env(env)
    rng = np.random.default_rng(0)
    mm_scores, rnd_scores = [], []
    for s in range(9_100_000, 9_100_000 + 5):
        obs = env.reset(seed=s)
        done = False
        while not done:
            res = env.step(mm.act(obs))
            done = res.done
        mm_scores.append(res.info["score"])
        obs = env.reset(seed=s)
        done = False
        while not done:
            res = env.step(int(rng.integers(0, NUM_ACTIONS)))
            done = res.done
        rnd_scores.append(res.info["score"])
    assert np.mean(mm_scores) > 3 * max(1.0, np.mean(rnd_scores))


def test_minimax_no_engine_is_safe():
    mm = make("minimax", OBS_SIZE, NUM_ACTIONS, {})
    assert 0 <= mm.best_act(np.zeros(OBS_SIZE, np.float32)) < NUM_ACTIONS   # no crash unbound
