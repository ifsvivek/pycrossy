"""Regression tests for the RL-pipeline fixes (see docs/AI_AUDIT.md).

Each test pins down one of the audited root-cause fixes so they can't silently regress:
reward shaping (R2), entropy-gradient sign (§3.7), Common Random Numbers (R1), and the
held-out validated champion (R3).
"""
from __future__ import annotations

import numpy as np
import pytest

from ai.base import make
from ai.env import Action, CrossyEnv, NUM_ACTIONS, OBS_SIZE
from ai.networks import ActorCritic
from ai.trainer import Trainer, TrainConfig


def test_reward_idling_loses_progress_wins():
    """R2: standing still must have a NET-NEGATIVE return (no more cowardice optimum),
    while a forward hop is positive."""
    env = CrossyEnv(seed=0)
    env.reset(seed=0)
    total = 0.0
    for _ in range(12):
        res = env.step(Action.WAIT)          # spawn grass is safe, so this only burns time
        total += res.reward
        if res.done:
            break
    assert total < 0.0

    env.reset(seed=0)
    res = env.step(Action.UP)                # grass row 8 → 9 is safe forward progress
    assert res.reward > 0.0


def test_entropy_bonus_increases_entropy():
    """§3.7: with zero advantage and returns == value (no policy/value gradient), the only
    driver is the entropy bonus, which must INCREASE entropy. The old sign decreased it."""
    ac = ActorCritic(4, 3, hidden=(8,), lr=1e-2, seed=0)
    ac.bp[:] = np.array([3.0, 0.0, 0.0], dtype=np.float32)   # start peaked → low entropy
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((32, 4)).astype(np.float32)
    acts = rng.integers(0, 3, 32)

    def entropy():
        p = ac.policy(obs)[0]
        return float(np.mean(-np.sum(p * np.log(p + 1e-8), axis=1)))

    e0 = entropy()
    for _ in range(80):
        _, v = ac.policy(obs)
        ac.update(obs, acts, np.zeros(32, np.float32), v.astype(np.float32),
                  np.zeros(32, np.float32), ent_coef=0.2, ppo=False)
    assert entropy() > e0 + 0.05


def test_crn_identical_policies_get_identical_fitness(tmp_path):
    """R1: within a generation every candidate is scored on the SAME shared seed bank, so two
    identical policies must receive identical fitness (under the old per-candidate seeds they
    would differ)."""
    cfg = TrainConfig(algo="neat", logdir=str(tmp_path / "run"), use_tensorboard=False,
                      max_steps=60, eval_every=0, checkpoint_every=0, eval_episodes=2,
                      algo_cfg={"pop_size": 4})
    tr = Trainer(cfg)
    tr.algo.population[1] = tr.algo.population[0].clone()     # force two identical genomes
    captured = {}
    orig = tr.algo.set_population_fitness

    def spy(fits):
        captured["fits"] = list(fits)
        return orig(fits)

    tr.algo.set_population_fitness = spy
    tr.train_generation(n_workers=1)
    f = captured["fits"]
    assert abs(f[0] - f[1]) < 1e-9
    tr.logger.close()


class _FakeMesh:
    def __init__(self, x):
        self.position = type("P", (), {"x": x})()


class _FakeMover:
    def __init__(self, x, speed, cb=0.5):
        self.mesh = _FakeMesh(x)
        self.speed = speed
        self.collision_box = cb


class _FakeRoad:
    def __init__(self, cars):
        self.cars = cars


def test_observation_arrival_time_safety():
    """The obs encodes safety AT ARRIVAL, not just now: a car that is clear at t=0 but will
    arrive on the column during the hop window must read as unsafe; a parked-far car as safe."""
    from ai import observation as O
    # Car currently 1.5 units left of the column (clear now) moving +0.12/tick toward it →
    # reaches the column well within the ~20-tick hop window → must be flagged unsafe (0.0).
    incoming = _FakeRoad([_FakeMover(x=-1.5, speed=0.12)])
    assert O._arrival_safety("road", incoming, 0.0) == 0.0
    # Car far away and moving away → the column stays clear → safe (1.0).
    clear = _FakeRoad([_FakeMover(x=-9.0, speed=-0.12)])
    assert O._arrival_safety("road", clear, 0.0) == 1.0


@pytest.mark.parametrize("name", ["neat", "cmaes", "es", "ga"])
def test_validated_champion_drives_best_act(name):
    """R3: adopting a validated candidate as the champion makes best_act use that policy
    deterministically (the saved model is a re-validated policy, not a noisy outlier)."""
    algo = make(name, OBS_SIZE, NUM_ACTIONS, {"pop_size": 6, "hidden": (8,)}, seed=0)
    payloads = algo.population_payloads()
    algo.set_validated_champion(payloads[2])
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    a1, a2 = algo.best_act(obs), algo.best_act(obs)
    assert 0 <= a1 < NUM_ACTIONS and a1 == a2
