"""Deep Q-Network (DQN) with experience replay + a target network.

Off-policy value learning over the :class:`~ai.networks.QNetwork` (numpy). Epsilon-greedy
exploration with linear decay, a circular replay buffer, periodic target-network sync, and
TD-MSE minibatch updates. Selectable as ``"dqn"``.
"""
from __future__ import annotations

import pickle
from collections import deque
from typing import Dict

import numpy as np

from ..base import Algorithm, Transition, register
from ..networks import QNetwork


@register("dqn")
class DQN(Algorithm):
    def __init__(self, obs_size, num_actions, cfg=None, seed=0):
        super().__init__(obs_size, num_actions, cfg, seed)
        c = self.cfg
        hidden = tuple(c.get("hidden", (128, 128)))
        self.gamma = c.get("gamma", 0.99)
        self.lr = c.get("lr", 5e-4)
        self.batch = int(c.get("batch", 64))
        self.buffer_size = int(c.get("buffer_size", 50000))
        self.warmup = int(c.get("warmup", 500))
        self.target_sync = int(c.get("target_sync", 500))
        self.train_every = int(c.get("train_every", 1))
        self.eps_start = c.get("eps_start", 1.0)
        self.eps_end = c.get("eps_end", 0.05)
        self.eps_decay_steps = int(c.get("eps_decay_steps", 15000))
        self.double = bool(c.get("double", True))      # Double DQN on by default
        self.hidden = hidden

        self.q = QNetwork(obs_size, num_actions, hidden=hidden, lr=self.lr, seed=seed)
        self.target = QNetwork(obs_size, num_actions, hidden=hidden, lr=self.lr, seed=seed)
        self.target.copy_from(self.q)

        self.buffer = deque(maxlen=self.buffer_size)
        self.epsilon = self.eps_start
        self.last_loss = 0.0
        self._learn_steps = 0

    def _epsilon(self) -> float:
        frac = min(1.0, self.total_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, obs, deterministic=False) -> int:
        self.epsilon = self._epsilon()
        if (not deterministic) and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        return self.q.act_argmax(obs)

    def best_act(self, obs) -> int:
        return self.q.act_argmax(obs)

    def observe(self, tr: Transition) -> None:
        super().observe(tr)
        self.buffer.append((tr.obs, tr.action, tr.reward, tr.next_obs, tr.done))
        if len(self.buffer) >= self.warmup and self.total_steps % self.train_every == 0:
            self._learn()
        if self._learn_steps > 0 and self._learn_steps % self.target_sync == 0:
            self.target.copy_from(self.q)

    def _learn(self) -> None:
        idx = self.rng.integers(0, len(self.buffer), size=self.batch)
        batch = [self.buffer[i] for i in idx]
        obs = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_obs = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        if self.double:
            # Double DQN: pick next action with the ONLINE net, value it with the TARGET net.
            # Decouples action selection from evaluation, curbing the max-operator overestimation
            # that vanilla DQN suffers (see AI_AUDIT §3.6).
            next_actions = self.q.q(next_obs).argmax(axis=1)
            next_q = self.target.q(next_obs)[np.arange(len(next_actions)), next_actions]
        else:
            next_q = self.target.q(next_obs).max(axis=1)
        targets = rewards + self.gamma * next_q * (1.0 - dones)
        self.last_loss = self.q.update(obs, actions, targets)
        self._learn_steps += 1

    def end_episode(self, total_reward, info) -> Dict:
        self.total_episodes += 1
        return {"loss": self.last_loss, "epsilon": self.epsilon,
                "buffer": len(self.buffer), "lr": self.lr}

    @property
    def progress(self) -> Dict:
        return {"loss": self.last_loss, "epsilon": self.epsilon, "buffer": len(self.buffer)}

    def state_dict(self) -> Dict:
        return {"q": pickle.dumps(self.q.net.get_params()), "hidden": tuple(self.hidden),
                "steps": self.total_steps, "episodes": self.total_episodes}

    def load_state_dict(self, state: Dict) -> None:
        hidden = tuple(state.get("hidden", self.hidden))
        if hidden != tuple(self.hidden):                 # rebuild to the saved architecture
            self.hidden = hidden
            self.q = QNetwork(self.obs_size, self.num_actions, hidden=hidden, lr=self.lr)
            self.target = QNetwork(self.obs_size, self.num_actions, hidden=hidden, lr=self.lr)
        self.q.net.set_params(pickle.loads(state["q"]))
        self.target.copy_from(self.q)
        self.total_steps = state.get("steps", 0)
        self.total_episodes = state.get("episodes", 0)
