"""Numpy network toolkit: gradient correctness + learning sanity."""
from __future__ import annotations

import numpy as np

from ai.networks import ActorCritic, MLP, QNetwork


def test_mlp_backward_matches_finite_differences():
    rng = np.random.default_rng(0)
    net = MLP([3, 5, 2], activation="tanh", seed=1)
    net.W = [w.astype(np.float64) for w in net.W]
    net.b = [b.astype(np.float64) for b in net.b]
    x = rng.standard_normal((4, 3))
    tgt = rng.standard_normal((4, 2))

    out, cache = net.forward(x)
    dout = 2 * (out - tgt) / out.size          # d/dout of mean-squared-error
    dW, db, _ = net.backward(dout, cache)

    def loss():
        o, _ = net.forward(x)
        return np.mean((o - tgt) ** 2)

    eps = 1e-6
    for li in range(len(net.W)):
        W = net.W[li]
        for idx in np.ndindex(W.shape):
            o = W[idx]
            W[idx] = o + eps; lp = loss()
            W[idx] = o - eps; lm = loss()
            W[idx] = o
            assert abs((lp - lm) / (2 * eps) - dW[li][idx]) < 1e-6


def test_qnetwork_fits_targets():
    rng = np.random.default_rng(2)
    q = QNetwork(4, 3, hidden=(16,), lr=1e-2, seed=2)
    obs = rng.standard_normal((64, 4)).astype("f4")
    acts = rng.integers(0, 3, 64)
    tg = rng.standard_normal(64).astype("f4")
    l0 = q.update(obs, acts, tg)
    for _ in range(300):
        lf = q.update(obs, acts, tg)
    assert lf < l0 * 0.2


def test_actorcritic_policy_improves():
    rng = np.random.default_rng(3)
    ac = ActorCritic(4, 3, hidden=(16,), lr=5e-3, seed=3)
    obs = rng.standard_normal((128, 4)).astype("f4")
    acts = np.zeros(128, dtype=int)
    adv = np.ones(128, dtype="f4")
    ret = np.ones(128, dtype="f4")
    oldlp = np.zeros(128, dtype="f4")
    p_before = ac.policy(obs)[0].mean(0)[0]
    for _ in range(200):
        ac.update(obs, acts, adv, ret, oldlp, ppo=False)
    p_after = ac.policy(obs)[0].mean(0)[0]
    assert p_after > p_before + 0.3
