"""On-policy policy-gradient methods — PPO and A2C.

Both share the :class:`~ai.networks.ActorCritic` (numpy) and a rollout buffer with GAE.
PPO does several clipped-surrogate epochs over each rollout; A2C does a single
advantage-weighted update. Selectable as ``"ppo"`` or ``"a2c"``.
"""
from __future__ import annotations

import pickle
from typing import Dict, List

import numpy as np

from ..base import Algorithm, Transition, register
from ..networks import ActorCritic


class _PolicyGradient(Algorithm):
    ppo = True

    def __init__(self, obs_size, num_actions, cfg=None, seed=0):
        super().__init__(obs_size, num_actions, cfg, seed)
        c = self.cfg
        hidden = tuple(c.get("hidden", (64, 64)))
        self.gamma = c.get("gamma", 0.99)
        self.lam = c.get("gae_lambda", 0.95)
        self.clip = c.get("clip", 0.2)
        self.vf_coef = c.get("vf_coef", 0.5)
        self.ent_coef = c.get("ent_coef", 0.01)
        self.lr = c.get("lr", 3e-4)
        # Smaller rollout than the original 1024 → several× more policy updates for the same
        # number of (short) episodes, which on-policy methods need to learn at all here.
        self.rollout_size = int(c.get("rollout_size", 512))
        self.epochs = int(c.get("epochs", 4 if self.ppo else 1))
        self.minibatch = int(c.get("minibatch", 128))
        self.hidden = hidden
        self.net = ActorCritic(obs_size, num_actions, hidden=hidden, lr=self.lr, seed=seed)

        self._buf: List[dict] = []
        self._last_logp = 0.0
        self._last_value = 0.0
        self.last_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    def act(self, obs, deterministic=False) -> int:
        a, logp, value = self.net.act(obs, self.rng, deterministic=deterministic)
        self._last_logp = logp
        self._last_value = value
        return a

    def best_act(self, obs) -> int:
        a, _, _ = self.net.act(obs, self.rng, deterministic=True)
        return a

    def observe(self, tr: Transition) -> None:
        super().observe(tr)
        self._buf.append({"obs": tr.obs, "action": tr.action, "reward": tr.reward,
                          "done": tr.done, "logp": self._last_logp, "value": self._last_value})

    def end_episode(self, total_reward, info) -> Dict:
        self.total_episodes += 1
        if len(self._buf) >= self.rollout_size:
            self._learn()
        m = dict(self.last_metrics)
        m.update({"lr": self.lr, "buffer": len(self._buf)})
        return m

    # -- learning ----------------------------------------------------------
    def _compute_gae(self):
        n = len(self._buf)
        adv = np.zeros(n, dtype=np.float32)
        last = 0.0
        for t in reversed(range(n)):
            done = self._buf[t]["done"]
            next_v = 0.0 if (t == n - 1 or self._buf[t]["done"]) else self._buf[t + 1]["value"]
            if done:
                next_v = 0.0
            delta = self._buf[t]["reward"] + self.gamma * next_v - self._buf[t]["value"]
            last = delta + self.gamma * self.lam * (0.0 if done else last)
            adv[t] = last
        returns = adv + np.array([b["value"] for b in self._buf], dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    def _learn(self) -> None:
        adv, returns = self._compute_gae()
        obs = np.array([b["obs"] for b in self._buf], dtype=np.float32)
        actions = np.array([b["action"] for b in self._buf], dtype=np.int64)
        old_logp = np.array([b["logp"] for b in self._buf], dtype=np.float32)
        n = len(self._buf)
        pls, vls, ents = [], [], []
        for _ in range(self.epochs):
            idx = self.rng.permutation(n)
            for s in range(0, n, self.minibatch):
                mb = idx[s:s + self.minibatch]
                pl, vl, ent = self.net.update(
                    obs[mb], actions[mb], adv[mb], returns[mb], old_logp[mb],
                    clip=self.clip, vf_coef=self.vf_coef, ent_coef=self.ent_coef, ppo=self.ppo)
                pls.append(pl); vls.append(vl); ents.append(ent)
        self.last_metrics = {"policy_loss": float(np.mean(pls)),
                             "value_loss": float(np.mean(vls)),
                             "entropy": float(np.mean(ents))}
        self._buf.clear()

    @property
    def progress(self) -> Dict:
        return {**self.last_metrics, "buffer": len(self._buf), "lr": self.lr}

    def state_dict(self) -> Dict:
        return {"params": pickle.dumps(self.net.get_params()), "hidden": tuple(self.hidden),
                "episodes": self.total_episodes, "steps": self.total_steps}

    def load_state_dict(self, state: Dict) -> None:
        hidden = tuple(state.get("hidden", self.hidden))
        if hidden != tuple(self.hidden):                 # rebuild to the saved architecture
            self.hidden = hidden
            self.net = ActorCritic(self.obs_size, self.num_actions, hidden=hidden, lr=self.lr)
        self.net.set_params(pickle.loads(state["params"]))
        self.total_episodes = state.get("episodes", 0)
        self.total_steps = state.get("steps", 0)


@register("ppo")
class PPO(_PolicyGradient):
    ppo = True


@register("a2c")
class A2C(_PolicyGradient):
    ppo = False

    def __init__(self, obs_size, num_actions, cfg=None, seed=0):
        cfg = dict(cfg or {})
        cfg.setdefault("rollout_size", 256)
        cfg.setdefault("epochs", 1)
        super().__init__(obs_size, num_actions, cfg, seed)
