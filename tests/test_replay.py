"""Replay recording, save/load, and determinism verification."""
from __future__ import annotations

import numpy as np

from ai import replay as R
from ai.env import CrossyEnv


def test_replay_reproduces_episode():
    env = CrossyEnv()
    rec = R.ReplayRecorder("test")
    seed = 42
    rec.start(seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(0)
    total = 0.0
    res = None
    done = False
    while not done:
        a = int(rng.integers(0, 5))
        rec.record(a)
        res = env.step(a)
        total += res.reward
        done = res.done
    rp = rec.finish(res.info["score"], total)
    assert R.verify(rp)                       # replays to the same score


def test_replay_save_load(tmp_path):
    rp = R.Replay(seed=7, actions=[0, 2, 0, 1, 3], algo="x", score=3, reward=1.5, length=5)
    path = str(tmp_path / "r.json")
    R.save(rp, path)
    rp2 = R.load(path)
    assert rp2.seed == 7
    assert rp2.actions == [0, 2, 0, 1, 3]
    assert rp2.score == 3 and rp2.length == 5
