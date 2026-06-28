"""Minimal numpy neural-network toolkit.

Pure-numpy layers with explicit forward/backward + an Adam optimizer, so every algorithm
runs with no deep-learning dependency. Provides:

* :class:`MLP` — a stack of linear layers with an activation, flat param get/set
  (used directly as a deterministic policy by ES/GA/CMA-ES).
* :class:`ActorCritic` — shared trunk + policy and value heads (PPO/A2C).
* :class:`QNetwork` — state→action-value MLP with a TD-MSE update (DQN).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _he(rng, fan_in, fan_out):
    return rng.standard_normal((fan_in, fan_out)).astype(np.float32) * np.sqrt(2.0 / fan_in)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _act(z, kind):
    if kind == "relu":
        return np.maximum(z, 0.0)
    if kind == "tanh":
        return np.tanh(z)
    return z


def _act_grad(a, z, kind):
    if kind == "relu":
        return (z > 0).astype(np.float32)
    if kind == "tanh":
        return 1.0 - a * a
    return np.ones_like(z)


class MLP:
    """Feed-forward network. ``sizes = [in, h1, ..., out]``."""

    def __init__(self, sizes: List[int], activation: str = "tanh", seed: int = 0):
        self.sizes = sizes
        self.activation = activation
        rng = np.random.default_rng(seed)
        self.W = [_he(rng, sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1], dtype=np.float32) for i in range(len(sizes) - 1)]

    # -- inference / training ---------------------------------------------
    def forward(self, x: np.ndarray):
        x = np.atleast_2d(x).astype(np.float32)
        cache = [x]
        a = x
        n = len(self.W)
        for i in range(n):
            z = a @ self.W[i] + self.b[i]
            if i < n - 1:
                a = _act(z, self.activation)
            else:
                a = z                       # linear output
            cache.append((z, a))
        return a, cache

    def backward(self, dout: np.ndarray, cache) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        n = len(self.W)
        dW = [None] * n
        db = [None] * n
        x0 = cache[0]
        delta = dout
        for i in reversed(range(n)):
            z, a = cache[i + 1]
            if i < n - 1:
                delta = delta * _act_grad(a, z, self.activation)
            a_prev = x0 if i == 0 else cache[i][1]
            dW[i] = a_prev.T @ delta
            db[i] = delta.sum(axis=0)
            delta = delta @ self.W[i].T
        return dW, db, delta

    # -- flat params (for evolution / checkpoint) -------------------------
    @property
    def num_params(self) -> int:
        return sum(w.size for w in self.W) + sum(bb.size for bb in self.b)

    def get_params(self) -> np.ndarray:
        return np.concatenate([w.ravel() for w in self.W] + [bb.ravel() for bb in self.b])

    def set_params(self, flat: np.ndarray) -> None:
        i = 0
        for k in range(len(self.W)):
            n = self.W[k].size
            self.W[k] = flat[i:i + n].reshape(self.W[k].shape).astype(np.float32)
            i += n
        for k in range(len(self.b)):
            n = self.b[k].size
            self.b[k] = flat[i:i + n].reshape(self.b[k].shape).astype(np.float32)
            i += n

    def act_argmax(self, obs: np.ndarray) -> int:
        out, _ = self.forward(obs)
        return int(np.argmax(out[0]))


class Adam:
    """Adam optimizer over a list of parameter arrays (updated in place)."""

    def __init__(self, params: List[np.ndarray], lr: float = 3e-4,
                 betas=(0.9, 0.999), eps: float = 1e-8):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> None:
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / bc1
            vhat = self.v[i] / bc2
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


class ActorCritic:
    """Shared trunk + policy (logits) and value (scalar) heads. Used by PPO/A2C."""

    def __init__(self, obs_size: int, num_actions: int, hidden=(64, 64),
                 lr: float = 3e-4, seed: int = 0):
        self.trunk = MLP([obs_size, *hidden], activation="tanh", seed=seed)
        rng = np.random.default_rng(seed + 1)
        h = hidden[-1]
        self.Wp = _he(rng, h, num_actions) * 0.1
        self.bp = np.zeros(num_actions, dtype=np.float32)
        self.Wv = _he(rng, h, 1) * 0.1
        self.bv = np.zeros(1, dtype=np.float32)
        self.num_actions = num_actions
        self._params = self.trunk.W + self.trunk.b + [self.Wp, self.bp, self.Wv, self.bv]
        self.opt = Adam(self._params, lr=lr)

    def _features(self, obs):
        feat, cache = self.trunk.forward(obs)
        feat = _act(feat, "tanh")           # trunk output passed through tanh
        return feat, cache

    def forward(self, obs):
        feat, cache = self._features(obs)
        logits = feat @ self.Wp + self.bp
        value = (feat @ self.Wv + self.bv)[:, 0]
        return logits, value, feat, cache

    def policy(self, obs):
        logits, value, _, _ = self.forward(obs)
        return softmax(logits), value

    def act(self, obs, rng, deterministic=False):
        probs, value = self.policy(obs)
        p = probs[0]
        if deterministic:
            a = int(np.argmax(p))
        else:
            a = int(rng.choice(self.num_actions, p=p))
        logp = float(np.log(p[a] + 1e-8))
        return a, logp, float(value[0])

    def update(self, obs, actions, advantages, returns, old_logp,
               clip: float = 0.2, vf_coef: float = 0.5, ent_coef: float = 0.01,
               ppo: bool = True):
        """One gradient update over a batch. Returns (policy_loss, value_loss, entropy)."""
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        advantages = np.asarray(advantages, dtype=np.float32)
        returns = np.asarray(returns, dtype=np.float32)
        old_logp = np.asarray(old_logp, dtype=np.float32)
        B = obs.shape[0]

        feat, cache = self._features(obs)
        z_trunk_last = cache[-1][0]
        logits = feat @ self.Wp + self.bp
        value = (feat @ self.Wv + self.bv)[:, 0]
        probs = softmax(logits)
        logp = np.log(probs[np.arange(B), actions] + 1e-8)

        if ppo:
            ratio = np.exp(logp - old_logp)
            clipped = np.clip(ratio, 1 - clip, 1 + clip)
            obj = np.minimum(ratio * advantages, clipped * advantages)
            policy_loss = -np.mean(obj)
            # d(policy_loss)/d(logits): use unclipped branch where it's active.
            use_unclipped = (ratio * advantages <= clipped * advantages).astype(np.float32)
            coeff = -(advantages * ratio * use_unclipped) / B
        else:  # A2C
            policy_loss = -np.mean(logp * advantages)
            coeff = -advantages / B

        # dL/dlogits for the chosen-action log-prob term: coeff * (onehot - probs)
        onehot = np.zeros_like(probs)
        onehot[np.arange(B), actions] = 1.0
        dlogits = coeff[:, None] * (onehot - probs)

        # entropy bonus gradient (maximize entropy -> subtract from loss)
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
        ent = np.mean(entropy)
        # d(entropy)/dlogits = -probs * (logp_all + entropy) ; we add -ent_coef*entropy to loss
        logits_logp = np.log(probs + 1e-8)
        dent = -probs * (logits_logp + entropy[:, None])
        dlogits += (-ent_coef) * (-dent) / B   # loss has -ent_coef*entropy

        # value loss (MSE)
        value_err = value - returns
        value_loss = np.mean(value_err ** 2)
        dvalue = (vf_coef * 2.0 * value_err / B)[:, None]

        # Head grads.
        dWp = feat.T @ dlogits
        dbp = dlogits.sum(axis=0)
        dWv = feat.T @ dvalue
        dbv = dvalue.sum(axis=0)
        dfeat = dlogits @ self.Wp.T + dvalue @ self.Wv.T

        # Backprop through trunk's final tanh, then the MLP.
        dfeat = dfeat * _act_grad(feat, z_trunk_last, "tanh")
        dW, db, _ = self.trunk.backward(dfeat, cache)

        grads = dW + db + [dWp, dbp, dWv, dbv]
        # clip global norm for stability
        total = np.sqrt(sum(float(np.sum(g * g)) for g in grads)) + 1e-8
        scale = min(1.0, 0.5 / total) if total > 0.5 else 1.0
        if scale < 1.0:
            grads = [g * scale for g in grads]
        self.opt.step(self._params, grads)
        return float(policy_loss), float(value_loss), float(ent)

    def get_params(self):
        return [p.copy() for p in self._params]

    def set_params(self, params):
        for dst, src in zip(self._params, params):
            dst[...] = src


class QNetwork:
    """State→action-value MLP with a TD-MSE update (DQN)."""

    def __init__(self, obs_size: int, num_actions: int, hidden=(64, 64),
                 lr: float = 5e-4, seed: int = 0):
        self.net = MLP([obs_size, *hidden, num_actions], activation="relu", seed=seed)
        self.opt = Adam(self.net.W + self.net.b, lr=lr)
        self.num_actions = num_actions

    def q(self, obs):
        out, _ = self.net.forward(obs)
        return out

    def act_argmax(self, obs):
        return int(np.argmax(self.q(obs)[0]))

    def update(self, obs, actions, targets):
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.float32)
        B = obs.shape[0]
        out, cache = self.net.forward(obs)
        q_sa = out[np.arange(B), actions]
        err = q_sa - targets
        dout = np.zeros_like(out)
        dout[np.arange(B), actions] = 2.0 * err / B
        dW, db, _ = self.net.backward(dout, cache)
        self.opt.step(self.net.W + self.net.b, dW + db)
        return float(np.mean(err ** 2))

    def copy_from(self, other: "QNetwork"):
        self.net.set_params(other.net.get_params())
