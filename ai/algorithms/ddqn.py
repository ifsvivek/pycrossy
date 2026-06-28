"""Double Deep Q-Network (DDQN) — a modular, production-quality value learner.

A from-scratch (pure-numpy) DDQN whose Rainbow-style components are independently toggled
through ``cfg`` so the same class spans "vanilla Double DQN" → "Double + Dueling + PER +
n-step". Every feature defaults ON to a tuned configuration for PyCrossy (not library
defaults). Selectable as ``"ddqn"``.

Components (all configurable):
  * **Double DQN** — action chosen by the online net, valued by the target net (curbs the
    max-operator overestimation of vanilla DQN).                                  ``double``
  * **Dueling network** — separate state-value / advantage streams.               ``dueling``
  * **Prioritized Experience Replay** — sample transitions by TD error, with        ``per``
    importance-sampling correction (sum-tree, O(log n)).             ``per_alpha`` ``per_beta``
  * **n-step returns** — bootstrap on the n-step discounted return.                ``n_step``
  * **Observation normalization** — running mean/std whitening of inputs.        ``obs_norm``
  * **Reward normalization** — divide rewards by a running std.               ``reward_norm``
  * **Linear LR + ε schedules**, **Huber loss**, **global-norm gradient clipping**, hard or
    soft (``tau``) target sync, and deterministic **eval mode** (``act(deterministic=True)``).

Not implemented (documented as future work in docs/AI_NEW_AGENTS.md): C51/distributional and
NoisyNets — the env is near-fully-observed (positions+velocities), so frame-stacking/NoisyNets
add little here, and distributional value learning is a large addition with marginal expected
gain for a 5-action discrete task.
"""
from __future__ import annotations

import pickle
from collections import deque
from typing import Dict

import numpy as np

from ..base import Algorithm, Transition, register
from ..networks import DuelingQNetwork


# ---------------------------------------------------------------------------
# Prioritized replay (sum-tree). With per_alpha=0 it degrades to a uniform buffer.
# ---------------------------------------------------------------------------
class _SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.size = 0
        self.write = 0

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def add(self, p: float) -> int:
        idx = self.write + self.capacity
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

    def update(self, idx: int, p: float) -> None:
        change = p - self.tree[idx]
        self.tree[idx] = p
        idx //= 2
        while idx >= 1:
            self.tree[idx] += change
            idx //= 2

    def get(self, s: float) -> int:
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        return idx                                   # leaf index in the tree array


class PrioritizedReplay:
    """Proportional prioritized replay. ``alpha=0`` → uniform sampling with unit IS weights."""

    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.6, eps: float = 1e-3):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.tree = _SumTree(capacity)
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.dones = np.zeros(capacity, np.float32)
        self.nsteps = np.zeros(capacity, np.float32)
        self.max_prio = 1.0

    def __len__(self) -> int:
        return self.tree.size

    def add(self, obs, action, reward, next_obs, done, n) -> None:
        i = self.tree.write
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.nsteps[i] = n
        self.tree.add(self.max_prio ** self.alpha)

    def sample(self, batch: int, beta: float):
        idxs = np.empty(batch, np.int64)
        data_idx = np.empty(batch, np.int64)
        seg = self.tree.total / batch
        for k in range(batch):
            s = (k + np.random.random()) * seg       # stratified sampling across the mass
            leaf = self.tree.get(s)
            idxs[k] = leaf
            data_idx[k] = leaf - self.capacity
        probs = self.tree.tree[idxs] / max(self.tree.total, 1e-8)
        weights = (len(self) * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)
        return data_idx, idxs, weights

    def update_priorities(self, tree_idx, td_errors) -> None:
        prios = (np.abs(td_errors) + self.eps)
        self.max_prio = max(self.max_prio, float(prios.max()))
        for i, p in zip(tree_idx, prios):
            self.tree.update(int(i), float(p) ** self.alpha)


class _RunningNorm:
    """Welford running mean/variance for observation whitening."""

    def __init__(self, dim: int):
        self.mean = np.zeros(dim, np.float64)
        self.var = np.ones(dim, np.float64)
        self.count = 1e-4

    def update(self, x) -> None:
        x = np.asarray(x, np.float64)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += (delta * (x - self.mean) - self.var) / self.count

    def norm(self, x):
        return ((np.asarray(x, np.float32) - self.mean) / (np.sqrt(self.var) + 1e-8)).astype(np.float32)


@register("ddqn")
class DDQN(Algorithm):
    def __init__(self, obs_size, num_actions, cfg=None, seed=0):
        super().__init__(obs_size, num_actions, cfg, seed)
        c = self.cfg
        self.hidden = tuple(c.get("hidden", (128, 128)))
        self.gamma = c.get("gamma", 0.99)
        self.lr = c.get("lr", 5e-4)
        self.lr_end = c.get("lr_end", 5e-5)
        self.lr_decay_steps = int(c.get("lr_decay_steps", 100_000))
        self.batch = int(c.get("batch", 64))
        self.buffer_size = int(c.get("buffer_size", 100_000))
        self.warmup = int(c.get("warmup", 1_000))
        self.train_every = int(c.get("train_every", 1))
        self.grad_clip = c.get("grad_clip", 10.0)
        # toggles
        self.double = bool(c.get("double", True))
        self.dueling = bool(c.get("dueling", True))
        self.per = bool(c.get("per", True))
        self.n_step = int(c.get("n_step", 3))
        self.obs_norm = bool(c.get("obs_norm", True))
        self.reward_norm = bool(c.get("reward_norm", False))
        # target sync: hard every N learn-steps, or soft polyak if tau > 0
        self.target_sync = int(c.get("target_sync", 1_000))
        self.tau = float(c.get("tau", 0.0))
        # exploration
        self.eps_start = c.get("eps_start", 1.0)
        self.eps_end = c.get("eps_end", 0.05)
        self.eps_decay_steps = int(c.get("eps_decay_steps", 15_000))
        self.eps_decay = c.get("eps_decay", "linear")        # 'linear' | 'exp'
        # PER schedule
        self.per_alpha = c.get("per_alpha", 0.6) if self.per else 0.0
        self.per_beta0 = c.get("per_beta", 0.4)
        self.per_beta_steps = int(c.get("per_beta_steps", 100_000))

        self.q = DuelingQNetwork(obs_size, num_actions, hidden=self.hidden, lr=self.lr,
                                 grad_clip=self.grad_clip, dueling=self.dueling, seed=seed)
        self.target = DuelingQNetwork(obs_size, num_actions, hidden=self.hidden, lr=self.lr,
                                      grad_clip=self.grad_clip, dueling=self.dueling, seed=seed)
        self.target.copy_from(self.q)

        self.buffer = PrioritizedReplay(self.buffer_size, obs_size, alpha=self.per_alpha)
        self.normer = _RunningNorm(obs_size) if self.obs_norm else None
        self._rstd = 1.0
        self._rcount = 1e-4
        self._nstep_buf: deque = deque()           # managed manually (not maxlen — see observe)
        self.epsilon = self.eps_start
        self.last_loss = 0.0
        self._learn_steps = 0

    # -- schedules ---------------------------------------------------------
    def _epsilon(self) -> float:
        t = self.total_steps
        if self.eps_decay == "exp":
            return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-3.0 * t / max(1, self.eps_decay_steps))
        frac = min(1.0, t / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _lr(self) -> float:
        frac = min(1.0, self._learn_steps / max(1, self.lr_decay_steps))
        return self.lr + frac * (self.lr_end - self.lr)

    def _beta(self) -> float:
        frac = min(1.0, self.total_steps / max(1, self.per_beta_steps))
        return self.per_beta0 + frac * (1.0 - self.per_beta0)

    def _n(self, obs):
        return self.normer.norm(obs) if self.normer is not None else np.asarray(obs, np.float32)

    # -- policy ------------------------------------------------------------
    def act(self, obs, deterministic=False) -> int:
        self.epsilon = self._epsilon()
        if (not deterministic) and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        return self.q.act_argmax(self._n(obs))

    def best_act(self, obs) -> int:
        return self.q.act_argmax(self._n(obs))

    # -- experience --------------------------------------------------------
    def observe(self, tr: Transition) -> None:
        super().observe(tr)
        if self.normer is not None:
            self.normer.update(tr.obs)
        r = tr.reward
        if self.reward_norm:
            self._rcount += 1
            self._rstd += (r * r - self._rstd) / self._rcount
            r = r / (np.sqrt(self._rstd) + 1e-8)
        self._nstep_buf.append((tr.obs, tr.action, r, tr.next_obs, tr.done))
        if tr.done:                                  # episode end → flush every remaining start
            while self._nstep_buf:
                self._emit_nstep()
                self._nstep_buf.popleft()
        elif len(self._nstep_buf) >= self.n_step:    # window full → emit + advance the oldest
            self._emit_nstep()
            self._nstep_buf.popleft()

        if len(self.buffer) >= self.warmup and self.total_steps % self.train_every == 0:
            self._learn()

    def _emit_nstep(self) -> None:
        """Fold up to ``n_step`` entries from the front of the window into one
        (s, a, R_n, s', done) transition, stopping early at a terminal step."""
        buf = self._nstep_buf
        R, gamma_k, done, next_obs, n_used = 0.0, 1.0, 0.0, buf[0][3], 0
        for k in range(min(self.n_step, len(buf))):
            _, _, r, nobs, d = buf[k]
            R += gamma_k * r
            gamma_k *= self.gamma
            next_obs = nobs
            n_used = k + 1
            if d:
                done = 1.0
                break
        self.buffer.add(buf[0][0], buf[0][1], R, next_obs, done, n_used)

    def _learn(self) -> None:
        beta = self._beta()
        data_idx, tree_idx, weights = self.buffer.sample(self.batch, beta)
        obs = self.buffer.obs[data_idx]
        next_obs = self.buffer.next_obs[data_idx]
        actions = self.buffer.actions[data_idx]
        rewards = self.buffer.rewards[data_idx]
        dones = self.buffer.dones[data_idx]
        ns = self.buffer.nsteps[data_idx]
        if self.normer is not None:
            obs = self.normer.norm(obs)
            next_obs = self.normer.norm(next_obs)

        if self.double:                              # online picks the action, target values it
            next_actions = self.q.q(next_obs).argmax(axis=1)
            next_q = self.target.q(next_obs)[np.arange(self.batch), next_actions]
        else:
            next_q = self.target.q(next_obs).max(axis=1)
        targets = rewards + (self.gamma ** ns) * next_q * (1.0 - dones)

        self.q.set_lr(self._lr())
        self.last_loss, td = self.q.update(obs, actions, targets, weights=weights)
        if self.per:
            self.buffer.update_priorities(tree_idx, td)
        self._learn_steps += 1

        if self.tau > 0.0:                           # soft (polyak) target update every step
            tp, qp = self.target.get_params(), self.q.get_params()
            self.target.set_params([(1 - self.tau) * t + self.tau * q for t, q in zip(tp, qp)])
        elif self._learn_steps % self.target_sync == 0:
            self.target.copy_from(self.q)

    def end_episode(self, total_reward, info) -> Dict:
        self.total_episodes += 1
        return {"loss": self.last_loss, "epsilon": self.epsilon, "buffer": len(self.buffer),
                "lr": self._lr(), "beta": self._beta()}

    @property
    def progress(self) -> Dict:
        return {"loss": self.last_loss, "epsilon": self.epsilon, "buffer": len(self.buffer),
                "lr": self._lr()}

    # -- persistence -------------------------------------------------------
    def state_dict(self) -> Dict:
        norm = None
        if self.normer is not None:
            norm = {"mean": self.normer.mean, "var": self.normer.var, "count": self.normer.count}
        return {"q": pickle.dumps(self.q.get_params()), "hidden": self.hidden,
                "dueling": self.dueling, "norm": pickle.dumps(norm),
                "steps": self.total_steps, "episodes": self.total_episodes,
                "learn_steps": self._learn_steps}

    def load_state_dict(self, state: Dict) -> None:
        hidden = tuple(state.get("hidden", self.hidden))
        dueling = bool(state.get("dueling", self.dueling))
        if hidden != tuple(self.hidden) or dueling != self.dueling:
            self.hidden, self.dueling = hidden, dueling
            self.q = DuelingQNetwork(self.obs_size, self.num_actions, hidden=hidden,
                                     grad_clip=self.grad_clip, dueling=dueling)
            self.target = DuelingQNetwork(self.obs_size, self.num_actions, hidden=hidden,
                                          grad_clip=self.grad_clip, dueling=dueling)
        self.q.set_params(pickle.loads(state["q"]))
        self.target.copy_from(self.q)
        norm = pickle.loads(state["norm"]) if state.get("norm") else None
        if norm is not None and self.normer is not None:
            self.normer.mean, self.normer.var, self.normer.count = norm["mean"], norm["var"], norm["count"]
        self.total_steps = state.get("steps", 0)
        self.total_episodes = state.get("episodes", 0)
        self._learn_steps = state.get("learn_steps", 0)
