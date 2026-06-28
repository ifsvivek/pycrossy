"""Training orchestration.

Ties the env, a registry algorithm, the metrics logger, the replay recorder and
checkpointing together. Supports headless fast training and a live mode (per-tick render
hook + per-episode snapshot callback). Handles periodic deterministic evaluation,
automatic best-model saving, checkpoint save/resume and best-episode replay capture.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from . import algorithms  # noqa: F401 — registers all algorithms in the registry
from . import base, replay
from .base import Transition
from .env import CrossyEnv
from .metrics import MetricsLogger


@dataclass
class TrainConfig:
    algo: str = "neat"
    logdir: str = "runs/latest"
    seed: int = 0
    target_episodes: int = 2000
    max_steps: int = 1500
    eval_every: int = 50          # episodes between evaluations (0 = never)
    eval_episodes: int = 5
    checkpoint_every: int = 100   # episodes between checkpoints (0 = never)
    record_best_replay: bool = True
    use_tensorboard: bool = True
    algo_cfg: Dict = field(default_factory=dict)


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        os.makedirs(cfg.logdir, exist_ok=True)
        self.env = CrossyEnv(max_steps=cfg.max_steps, seed=cfg.seed)
        self.eval_env = CrossyEnv(max_steps=cfg.max_steps, seed=cfg.seed + 9999)
        self.algo = base.make(cfg.algo, self.env_obs_size, self.env_actions,
                              cfg.algo_cfg, seed=cfg.seed)
        self.logger = MetricsLogger(cfg.logdir, cfg.algo, cfg.target_episodes,
                                    cfg.use_tensorboard)
        self.recorder = replay.ReplayRecorder(cfg.algo)
        self.episode = 0
        self.best_eval = -1e18
        self.best_score = 0
        self._rng = np.random.default_rng(cfg.seed)

    @property
    def env_obs_size(self) -> int:
        from .env import OBS_SIZE
        return OBS_SIZE

    @property
    def env_actions(self) -> int:
        from .env import NUM_ACTIONS
        return NUM_ACTIONS

    # -- one episode -------------------------------------------------------
    def train_episode(self, render_tick: Optional[Callable] = None) -> Dict:
        ep_seed = self.cfg.seed * 100003 + self.episode
        self.env.on_tick = render_tick
        self.recorder.start(ep_seed)
        self.algo.begin_episode()
        obs = self.env.reset(seed=ep_seed)
        total = 0.0
        length = 0
        info = {"score": 0, "alive": True}
        done = False
        while not done:
            action = self.algo.act(obs)
            self.recorder.record(action)
            res = self.env.step(action)
            self.algo.observe(Transition(obs, action, res.reward, res.obs, res.done))
            total += res.reward
            obs = res.obs
            info = res.info
            length += 1
            done = res.done
        self.env.on_tick = None
        metrics = self.algo.end_episode(total, info)
        metrics["alive"] = info.get("alive", False)
        self.episode += 1

        eval_score = None
        if self.cfg.eval_every and self.episode % self.cfg.eval_every == 0:
            eval_score = self._evaluate()
            if eval_score > self.best_eval:
                self.best_eval = eval_score
                self.save_best()

        if info["score"] > self.best_score:
            self.best_score = info["score"]
            if self.cfg.record_best_replay:
                rp = self.recorder.finish(info["score"], total)
                replay.save(rp, os.path.join(self.cfg.logdir, "best_replay.json"))

        self.logger.log_episode(total, info["score"], length, metrics, eval_score)

        if self.cfg.checkpoint_every and self.episode % self.cfg.checkpoint_every == 0:
            self.save_checkpoint()

        return {"episode": self.episode, "reward": total, "score": info["score"],
                "length": length, "metrics": metrics, "eval_score": eval_score}

    def showcase_episode(self, render_tick: Optional[Callable] = None,
                         seed: Optional[int] = None) -> int:
        """Render the current *best* policy playing one episode — no learning.

        Used by live parallel training to show a representative run in the window while the
        population is evaluated in the background.
        """
        self.env.on_tick = render_tick
        if seed is None:
            seed = 50_000 + self.episode
        obs = self.env.reset(seed=seed)
        info = {"score": 0}
        done = False
        while not done:
            res = self.env.step(self.algo.best_act(obs))
            obs = res.obs
            info = res.info
            done = res.done
        self.env.on_tick = None
        return info.get("score", 0)

    def _evaluate(self) -> float:
        scores = []
        for i in range(self.cfg.eval_episodes):
            obs = self.eval_env.reset(seed=10_000 + i)
            done = False
            while not done:
                res = self.eval_env.step(self.algo.best_act(obs))
                obs = res.obs
                done = res.done
            scores.append(res.info["score"])
        return float(np.mean(scores))

    # -- headless loop -----------------------------------------------------
    def train(self, num_episodes: Optional[int] = None,
              on_episode: Optional[Callable] = None) -> None:
        n = num_episodes or self.cfg.target_episodes
        for _ in range(n):
            summary = self.train_episode()
            if on_episode is not None:
                on_episode(summary, self.logger.snapshot())
        self.logger.close()

    # -- parallel generation loop (population algorithms) -----------------
    def train_generation(self, n_workers: Optional[int] = None,
                         on_generation: Optional[Callable] = None, pool=None) -> Dict:
        """Evaluate ONE generation's population across worker processes.

        Only valid for algorithms with ``supports_parallel`` (NEAT/ES/GA/CMA-ES). Unlike
        :meth:`train_parallel`, this does *not* close the logger, so it can be called once
        per generation from a live loop that interleaves rendering. Pass a persistent
        ``pool`` (from ``vec_env.make_pool``) to reuse workers across generations.
        """
        from . import vec_env
        if not getattr(self.algo, "supports_parallel", False):
            raise ValueError(f"{self.cfg.algo} does not support parallel evaluation; use train()")
        payloads = self.algo.population_payloads()
        seeds = [self.cfg.seed * 100003 + self.episode + i for i in range(len(payloads))]
        results = vec_env.parallel_evaluate(payloads, seeds, self.cfg.max_steps, n_workers, pool=pool)
        rewards = [r for r, _, _ in results]
        metrics = self.algo.set_population_fitness(rewards)
        pop = len(payloads)
        for (reward, score, length) in results:
            self.episode += 1
            self.best_score = max(self.best_score, score)
            self.logger.log_episode(reward, score, length, metrics)
        if self.cfg.checkpoint_every and self.episode % max(1, self.cfg.checkpoint_every) < pop:
            self.save_checkpoint()
            self.save_best()
        if on_generation is not None:
            on_generation(metrics, self.logger.snapshot())
        return metrics

    def train_parallel(self, generations: Optional[int] = None, n_workers: Optional[int] = None,
                       on_generation: Optional[Callable] = None) -> None:
        """Evaluate each generation's population across worker processes.

        Only valid for algorithms with ``supports_parallel`` (NEAT/ES/GA/CMA-ES).
        """
        if not getattr(self.algo, "supports_parallel", False):
            raise ValueError(f"{self.cfg.algo} does not support parallel evaluation; use train()")
        pop = len(self.algo.population_payloads())
        target_gens = generations or max(1, self.cfg.target_episodes // pop)
        for _ in range(target_gens):
            self.train_generation(n_workers=n_workers, on_generation=on_generation)
        self.logger.close()

    # -- persistence -------------------------------------------------------
    def save_checkpoint(self, path: Optional[str] = None) -> str:
        path = path or os.path.join(self.cfg.logdir, "checkpoint.pkl")
        with open(path, "wb") as fh:
            pickle.dump({"algo": self.algo.state_dict(), "episode": self.episode,
                         "best_eval": self.best_eval, "best_score": self.best_score,
                         "config": self.cfg.__dict__}, fh)
        self.logger.checkpoint_info.update(
            {"last_checkpoint_ep": self.episode, "best_score": self.best_score})
        return path

    def save_best(self, path: Optional[str] = None) -> str:
        path = path or os.path.join(self.cfg.logdir, "best_model.pkl")
        with open(path, "wb") as fh:
            pickle.dump({"algo": self.algo.state_dict(), "algo_name": self.cfg.algo,
                         "best_eval": self.best_eval}, fh)
        best_eval = round(self.best_eval, 2) if self.best_eval > -1e17 else None
        self.logger.checkpoint_info.update(
            {"best_model_ep": self.episode, "best_eval": best_eval})
        return path

    def load_checkpoint(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(self.cfg.logdir, "checkpoint.pkl")
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        self.algo.load_state_dict(state["algo"])
        self.episode = state["episode"]
        self.best_eval = state["best_eval"]
        self.best_score = state["best_score"]

    @staticmethod
    def load_policy(path: str):
        """Load a saved best model into a fresh algorithm for inference."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        from .env import OBS_SIZE, NUM_ACTIONS
        algo = base.make(state["algo_name"], OBS_SIZE, NUM_ACTIONS)
        algo.load_state_dict(state["algo"])
        return algo
