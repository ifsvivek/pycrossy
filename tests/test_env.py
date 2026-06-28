"""AI environment: observation shape, step contract, reward signs, determinism."""
from __future__ import annotations

import numpy as np

from ai.env import Action, CrossyEnv, NUM_ACTIONS, OBS_SIZE


def test_observation_shape_and_step_contract():
    env = CrossyEnv(seed=0)
    obs = env.reset(seed=0)
    assert obs.shape == (OBS_SIZE,) and obs.dtype == np.float32
    res = env.step(Action.UP)
    assert res.obs.shape == (OBS_SIZE,)
    assert isinstance(res.reward, float)
    assert isinstance(res.done, bool)
    assert {"score", "steps", "alive"} <= set(res.info)


def test_forward_progress_is_rewarded():
    env = CrossyEnv(seed=0)
    env.reset(seed=0)
    res = env.step(Action.UP)               # row 8 -> 9 (grass) is safe forward progress
    assert res.reward > 0


def test_death_is_penalised_and_terminal():
    env = CrossyEnv(seed=0)
    env.reset(seed=0)
    last = None
    for _ in range(60):
        last = env.step(Action.UP)          # blind forward eventually drowns/crashes
        if last.done:
            break
    assert last.done
    if not last.info["alive"]:
        assert last.reward < 0


def test_reset_is_deterministic():
    o1 = CrossyEnv().reset(seed=123)
    o2 = CrossyEnv().reset(seed=123)
    assert np.allclose(o1, o2)


def test_action_space_size():
    assert NUM_ACTIONS == len(Action) == 5
