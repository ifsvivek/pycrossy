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

# Fixed validation seed bank, disjoint from training seeds (seed*100003 + …) and from the
# held-out evaluation band used by ai.evaluate (9_000_000+). A champion is only saved when it
# improves on THIS bank, so the saved model is a re-validated policy, not a lucky single run.
VAL_SEED_BASE = 7_000_000
VAL_SEEDS = [VAL_SEED_BASE + i for i in range(16)]   # wider bank → less champion overfit


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
        self._last_eval = None          # most recent validated mean score (for logging)
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
        if self.algo.uses_planning:
            self.algo.bind_env(self.env)
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
        if self.algo.uses_planning:
            self.algo.bind_env(self.env)
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
        if self.algo.uses_planning:
            self.algo.bind_env(self.eval_env)
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

        Fixes the two measurement bugs that made evolution a lottery (see docs/AI_AUDIT.md):
        * **Common Random Numbers** — every candidate is scored on the SAME bank of ``K``
          seeds (shared maps), so selection compares *policies* not map difficulty. The bank
          rotates each generation so the population can't overfit one map.
        * **Held-out validation** — the top candidates (plus the search-distribution centre)
          are re-scored on a fixed disjoint bank; the best is adopted as the champion and the
          model is saved only when it beats the best validated score so far.
        """
        from . import vec_env
        if not getattr(self.algo, "supports_parallel", False):
            raise ValueError(f"{self.cfg.algo} does not support parallel evaluation; use train()")
        payloads = self.algo.population_payloads()
        pop = len(payloads)
        K = max(1, int(self.cfg.eval_episodes))          # CRN episodes per candidate
        gen = int(getattr(self.algo, "generation", self.episode // max(1, pop)))
        base = self.cfg.seed * 100003 + gen * 1009       # shared across candidates, rotates per gen
        seed_bank = [base + j for j in range(K)]

        exp_payloads = [p for p in payloads for _ in seed_bank]
        exp_seeds = [s for _ in payloads for s in seed_bank]
        flat = vec_env.parallel_evaluate(exp_payloads, exp_seeds, self.cfg.max_steps,
                                         n_workers, pool=pool)
        rewards, scores, lengths = [], [], []
        for ci in range(pop):
            chunk = flat[ci * K:(ci + 1) * K]
            rewards.append(float(np.mean([r for r, _, _ in chunk])))
            scores.append(int(max(s for _, s, _ in chunk)))
            lengths.append(int(np.mean([l for _, _, l in chunk])))
        metrics = self.algo.set_population_fitness(rewards)

        # Held-out validation → gated champion/save (only every ``eval_every`` episodes).
        do_eval = self.cfg.eval_every and (self.episode // max(1, pop)) % \
            max(1, self.cfg.eval_every // max(1, pop)) == 0
        if do_eval:
            self._validate_population(payloads, rewards, n_workers, pool)

        for ci in range(pop):
            self.episode += 1
            self.best_score = max(self.best_score, scores[ci])
            self.logger.log_episode(rewards[ci], scores[ci], lengths[ci], metrics,
                                    eval_score=self._last_eval if (do_eval and ci == 0) else None)
        if self.cfg.checkpoint_every and self.episode % max(1, self.cfg.checkpoint_every) < pop:
            self.save_checkpoint()
        if on_generation is not None:
            on_generation(metrics, self.logger.snapshot())
        return metrics

    def _validate_population(self, payloads, rewards, n_workers, pool) -> None:
        """Re-score the most promising candidates (by CRN training fitness) plus the search
        centre on the fixed held-out :data:`VAL_SEEDS`; adopt the best as champion and save
        when it improves on ``best_eval`` (the validated mean SCORE)."""
        from . import vec_env
        pop = len(payloads)
        order = sorted(range(pop), key=lambda i: rewards[i], reverse=True)[:min(5, pop)]
        cands = [payloads[i] for i in order]
        center = self.algo.center_payload()
        if center is not None:
            cands.append(center)
        vtasks = [c for c in cands for _ in VAL_SEEDS]
        vseeds = [s for _ in cands for s in VAL_SEEDS]
        vres = vec_env.parallel_evaluate(vtasks, vseeds, self.cfg.max_steps, n_workers, pool=pool)
        nb = len(VAL_SEEDS)
        val_scores = [float(np.mean([vres[k * nb + t][1] for t in range(nb)])) for k in range(len(cands))]
        bestk = int(np.argmax(val_scores))
        champ_val = val_scores[bestk]
        self._last_eval = champ_val
        if champ_val > self.best_eval:
            self.best_eval = champ_val
            self.algo.set_validated_champion(cands[bestk])
            self.save_best()

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
        payload = {"algo": self.algo.state_dict(), "algo_name": self.cfg.algo,
                   "best_eval": self.best_eval, "episode": self.episode, "seed": self.cfg.seed}
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        best_eval = round(self.best_eval, 2) if self.best_eval > -1e17 else None
        self.logger.checkpoint_info.update(
            {"best_model_ep": self.episode, "best_eval": best_eval})
        self._register_best(best_eval)
        return path

    def _register_best(self, best_eval) -> None:
        """Append to the run's best-history log and update the cross-run registry index, so
        every validated improvement is recorded and runs can be compared/replayed later."""
        import json
        try:
            with open(os.path.join(self.cfg.logdir, "best_history.jsonl"), "a") as fh:
                fh.write(json.dumps({"episode": self.episode, "best_eval": best_eval,
                                     "best_score": self.best_score}) + "\n")
            runs_root = os.path.dirname(os.path.normpath(self.cfg.logdir)) or "runs"
            index = os.path.join(runs_root, "index.json")
            reg = {}
            if os.path.exists(index):
                with open(index) as fh:
                    reg = json.load(fh)
            reg[self.cfg.logdir] = {"algo": self.cfg.algo, "seed": self.cfg.seed,
                                    "episode": self.episode, "best_eval": best_eval,
                                    "best_score": self.best_score,
                                    "model": os.path.join(self.cfg.logdir, "best_model.pkl")}
            with open(index, "w") as fh:
                json.dump(reg, fh, indent=2)
        except Exception:
            pass        # registry is best-effort; never break training on a logging hiccup

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
