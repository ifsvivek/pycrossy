"""Metrics logging — CSV, JSON, TensorBoard + in-memory history for the dashboard.

One :class:`MetricsLogger` per run. ``log_episode`` appends a row to CSV, writes scalars
to TensorBoard (if ``tensorboardX`` is installed), and keeps rolling history (rewards,
scores, losses, …) used by the live dashboard and by :meth:`snapshot` (which also derives
moving averages, training speed, FPS, elapsed and ETA).
"""
from __future__ import annotations

import csv
import json
import os
import time
from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

try:  # optional — TB is nice-to-have; CSV/JSON always work
    from tensorboardX import SummaryWriter
    _HAVE_TB = True
except Exception:  # pragma: no cover
    _HAVE_TB = False


class MetricsLogger:
    def __init__(self, logdir: str, algo_name: str, target_episodes: Optional[int] = None,
                 use_tensorboard: bool = True, history: int = 5000,
                 success_threshold: int = 5):
        self.logdir = logdir
        self.algo_name = algo_name
        self.target_episodes = target_episodes
        self.success_threshold = success_threshold
        self.checkpoint_info: Dict[str, object] = {}
        os.makedirs(logdir, exist_ok=True)

        self.csv_path = os.path.join(logdir, "metrics.csv")
        self.json_path = os.path.join(logdir, "summary.json")
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_fields: list = []      # ordered union of all columns seen so far
        self._csv_rows: list = []        # buffered rows, so the header can be rebuilt on new keys

        self.tb = SummaryWriter(logdir) if (use_tensorboard and _HAVE_TB) else None

        self.start_time = time.time()
        self.episode = 0
        self.total_steps = 0
        self.death_count = 0
        self.best_score = 0
        self.best_reward = -1e18

        h = history
        self.rewards: Deque[float] = deque(maxlen=h)
        self.scores: Deque[int] = deque(maxlen=h)
        self.lengths: Deque[int] = deque(maxlen=h)
        self.moving_reward: Deque[float] = deque(maxlen=h)
        self.eval_scores: Deque[float] = deque(maxlen=h)
        # algorithm-specific scalar series, created lazily
        self.series: Dict[str, Deque[float]] = {}
        self._latest: Dict[str, float] = {}

    # -- logging -----------------------------------------------------------
    def log_episode(self, reward: float, score: int, length: int, metrics: Dict,
                    eval_score: Optional[float] = None) -> Dict:
        self.episode += 1
        self.total_steps += length
        if score == 0 and reward < 0:
            self.death_count += 1
        elif not metrics.get("alive", True):
            self.death_count += 1
        self.best_score = max(self.best_score, score)
        self.best_reward = max(self.best_reward, reward)

        self.rewards.append(reward)
        self.scores.append(score)
        self.lengths.append(length)
        window = list(self.rewards)[-100:]
        self.moving_reward.append(float(np.mean(window)))
        if eval_score is not None:
            self.eval_scores.append(eval_score)

        row = {"episode": self.episode, "reward": round(reward, 4), "score": score,
               "length": length, "death_count": self.death_count,
               "elapsed": round(time.time() - self.start_time, 2)}
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                row[k] = round(float(v), 6)
                self.series.setdefault(k, deque(maxlen=self.rewards.maxlen)).append(float(v))
                self._latest[k] = float(v)
        if eval_score is not None:
            row["eval_score"] = round(eval_score, 4)

        self._write_csv(row)
        if self.tb:
            self.tb.add_scalar("reward/episode", reward, self.episode)
            self.tb.add_scalar("reward/moving_avg_100", self.moving_reward[-1], self.episode)
            self.tb.add_scalar("score/episode", score, self.episode)
            self.tb.add_scalar("score/best", self.best_score, self.episode)
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    self.tb.add_scalar(f"algo/{k}", float(v), self.episode)
            if eval_score is not None:
                self.tb.add_scalar("eval/score", eval_score, self.episode)
        return row

    def _write_csv(self, row: Dict) -> None:
        self._csv_rows.append(row)
        new_keys = [k for k in row if k not in self._csv_fields]
        if new_keys:
            # A column appeared that the frozen header lacks (e.g. eval_score, which is only
            # present on evaluation episodes). Extend the schema and rewrite the whole file so
            # late-appearing columns are never silently dropped.
            self._csv_fields.extend(new_keys)
            self._csv_file.seek(0)
            self._csv_file.truncate()
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)
            self._csv_writer.writeheader()
            for r in self._csv_rows:
                self._csv_writer.writerow({k: r.get(k, "") for k in self._csv_fields})
        else:
            self._csv_writer.writerow({k: row.get(k, "") for k in self._csv_fields})
        self._csv_file.flush()

    # -- derived stats -----------------------------------------------------
    def snapshot(self) -> Dict:
        elapsed = time.time() - self.start_time
        eps_per_s = self.episode / elapsed if elapsed > 0 else 0.0
        steps_per_s = self.total_steps / elapsed if elapsed > 0 else 0.0
        eta = None
        if self.target_episodes and eps_per_s > 0:
            remaining = max(0, self.target_episodes - self.episode)
            eta = remaining / eps_per_s
        recent = list(self.scores)[-100:]
        success_rate = (sum(1 for s in recent if s >= self.success_threshold) / len(recent)
                        if recent else 0.0)
        return {
            "algo": self.algo_name,
            "episode": self.episode,
            "target_episodes": self.target_episodes,
            "total_steps": self.total_steps,
            "death_count": self.death_count,
            "best_score": self.best_score,
            "best_reward": self.best_reward,
            "elapsed": elapsed,
            "eta": eta,
            "episodes_per_sec": eps_per_s,
            "steps_per_sec": steps_per_s,
            "success_rate": success_rate,
            "success_threshold": self.success_threshold,
            "checkpoint_info": dict(self.checkpoint_info),
            "rewards": list(self.rewards),
            "scores": list(self.scores),
            "lengths": list(self.lengths),
            "moving_reward": list(self.moving_reward),
            "eval_scores": list(self.eval_scores),
            "series": {k: list(v) for k, v in self.series.items()},
            "latest": dict(self._latest),
        }

    def save_json(self) -> None:
        snap = self.snapshot()
        summary = {k: snap[k] for k in (
            "algo", "episode", "total_steps", "death_count", "best_score",
            "best_reward", "elapsed", "episodes_per_sec", "steps_per_sec", "latest")}
        with open(self.json_path, "w") as fh:
            json.dump(summary, fh, indent=2)

    def close(self) -> None:
        self.save_json()
        try:
            self._csv_file.close()
        except Exception:
            pass
        if self.tb:
            self.tb.close()
