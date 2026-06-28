"""Evolutionary policy search — ES, GA and (separable) CMA-ES.

All three optimise the flat parameter vector of a small :class:`~ai.networks.MLP` policy
(``obs -> logits``, argmax action). They share the episodic interface via a common base:
one candidate is evaluated per episode; once the whole population is scored the optimiser
updates and proposes the next generation.

* ``es``    — OpenAI-style Evolution Strategies (rank-normalised gradient estimate).
* ``ga``    — elitist Genetic Algorithm (tournament select + uniform crossover + mutation).
* ``cmaes`` — separable CMA-ES (diagonal covariance, O(dim) — scales to the net's params).
"""
from __future__ import annotations

import pickle
from typing import Dict, List

import numpy as np

from ..base import Algorithm, register
from ..networks import MLP


class _Evolution(Algorithm):
    def __init__(self, obs_size, num_actions, cfg=None, seed=0):
        super().__init__(obs_size, num_actions, cfg, seed)
        c = self.cfg
        hidden = tuple(c.get("hidden", (32,)))
        self.pop_size = int(c.get("pop_size", 40))
        self.policy = MLP([obs_size, *hidden, num_actions], activation="tanh", seed=seed)
        self.dim = self.policy.num_params
        self.generation = 0
        self.best_fitness = -1e9
        self.mean_fitness = 0.0
        self.best_params = self.policy.get_params()
        self.population: List[np.ndarray] = self._ask()
        self._fits = np.full(self.pop_size, -1e9)
        self._idx = 0
        self._cur_fitness = 0.0
        self._diversity = 0.0           # std of the generation's fitnesses

    # -- subclass hooks ----------------------------------------------------
    def _ask(self) -> List[np.ndarray]:
        raise NotImplementedError

    def _tell(self, fits: np.ndarray) -> None:
        raise NotImplementedError

    # -- episodic interface ------------------------------------------------
    def begin_episode(self) -> None:
        self.policy.set_params(self.population[self._idx])

    def act(self, obs, deterministic=False) -> int:
        return self.policy.act_argmax(obs)

    def best_act(self, obs) -> int:
        self.policy.set_params(self.best_params)
        a = self.policy.act_argmax(obs)
        self.policy.set_params(self.population[self._idx])  # restore current candidate
        return a

    def end_episode(self, total_reward, info) -> Dict:
        self.total_episodes += 1
        self._cur_fitness = total_reward
        self._fits[self._idx] = total_reward
        if total_reward > self.best_fitness:
            self.best_fitness = total_reward
            self.best_params = self.population[self._idx].copy()
        self._idx += 1
        advanced = False
        if self._idx >= self.pop_size:
            self.mean_fitness = float(np.mean(self._fits))
            self._diversity = float(np.std(self._fits))
            self._tell(self._fits.copy())
            self.population = self._ask()
            self._fits = np.full(self.pop_size, -1e9)
            self._idx = 0
            self.generation += 1
            advanced = True
        return self._metrics(advanced)

    def _metrics(self, advanced: bool) -> Dict:
        return {"generation": self.generation, "best_fitness": self.best_fitness,
                "mean_fitness": self.mean_fitness, "current_fitness": self._cur_fitness,
                "population": self.pop_size, "advanced_generation": advanced,
                "diversity": self._diversity, "sigma": float(getattr(self, "sigma", 0.0))}

    # -- parallel evaluation protocol -------------------------------------
    supports_parallel = True

    def population_payloads(self):
        return [("mlp", (p.copy(), tuple(self.policy.sizes), self.policy.activation))
                for p in self.population]

    def set_population_fitness(self, fits) -> Dict:
        fits = np.asarray(fits, dtype=np.float64)
        bi = int(np.argmax(fits))
        self._cur_fitness = float(fits[bi])
        if fits[bi] > self.best_fitness:
            self.best_fitness = float(fits[bi])
            self.best_params = self.population[bi].copy()
        self.total_episodes += len(fits)
        self.mean_fitness = float(np.mean(fits))
        self._diversity = float(np.std(fits))
        self._tell(fits)
        self.population = self._ask()
        self.generation += 1
        return self._metrics(advanced=True)

    @property
    def progress(self) -> Dict:
        return {"generation": self.generation, "best_fitness": self.best_fitness,
                "mean_fitness": self.mean_fitness, "eval_index": self._idx}

    def state_dict(self) -> Dict:
        return {"best_params": pickle.dumps(self.best_params),
                "generation": self.generation, "best_fitness": self.best_fitness,
                "extra": pickle.dumps(self._extra_state())}

    def load_state_dict(self, state: Dict) -> None:
        self.best_params = pickle.loads(state["best_params"])
        self.generation = state["generation"]
        self.best_fitness = state["best_fitness"]
        self._load_extra(pickle.loads(state["extra"]))
        self.population = self._ask()
        self._fits = np.full(self.pop_size, -1e9)
        self._idx = 0

    def _extra_state(self) -> Dict:
        return {}

    def _load_extra(self, s: Dict) -> None:
        ...

    @staticmethod
    def _centered_ranks(fits: np.ndarray) -> np.ndarray:
        order = np.argsort(fits)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(fits))
        ranks = ranks / (len(fits) - 1) - 0.5
        return ranks


@register("es")
class ES(_Evolution):
    def __init__(self, obs_size, num_actions, cfg=None, seed=0):
        c = cfg or {}
        self.sigma = c.get("sigma", 0.1)
        self.lr = c.get("lr", 0.05)
        self.weight_decay = c.get("weight_decay", 0.005)
        self._theta = None
        super().__init__(obs_size, num_actions, c, seed)

    def _ask(self) -> List[np.ndarray]:
        if self._theta is None:
            self._theta = self.policy.get_params().astype(np.float64)
        self._eps = self.rng.standard_normal((self.pop_size, self.dim))
        return [self._theta + self.sigma * e for e in self._eps]

    def _tell(self, fits: np.ndarray) -> None:
        ranks = self._centered_ranks(fits)
        grad = (self._eps.T @ ranks) / (self.pop_size * self.sigma)
        self._theta = (1 - self.weight_decay) * self._theta + self.lr * grad
        # Track the centre as a strong candidate for the champion view.
        if fits.max() > self.best_fitness:
            pass  # best_params already handled in end_episode

    def _extra_state(self) -> Dict:
        return {"theta": self._theta, "sigma": self.sigma}

    def _load_extra(self, s: Dict) -> None:
        self._theta = s["theta"]
        self.sigma = s["sigma"]


@register("ga")
class GA(_Evolution):
    def __init__(self, obs_size, num_actions, cfg=None, seed=0):
        c = cfg or {}
        self.elite = int(c.get("elite", 4))
        self.mut_rate = c.get("mut_rate", 0.1)
        self.mut_std = c.get("mut_std", 0.1)
        self.tournament = int(c.get("tournament", 3))
        self.init_std = c.get("init_std", 0.5)
        super().__init__(obs_size, num_actions, c, seed)

    def _ask(self) -> List[np.ndarray]:
        # Before the first _tell (fresh start OR just-resumed checkpoint) there is no
        # bred population yet — seed one around the best-known params.
        if not hasattr(self, "_next_population"):
            base = getattr(self, "best_params", None)
            if base is None:
                base = self.policy.get_params()
            return [base + self.init_std * self.rng.standard_normal(self.dim).astype(np.float32)
                    for _ in range(self.pop_size)]
        return self._next_population

    def _select(self, fits: np.ndarray) -> int:
        cand = self.rng.integers(0, self.pop_size, size=self.tournament)
        return int(cand[np.argmax(fits[cand])])

    def _tell(self, fits: np.ndarray) -> None:
        order = np.argsort(fits)[::-1]
        new_pop = [self.population[i].copy() for i in order[:self.elite]]
        while len(new_pop) < self.pop_size:
            p1 = self.population[self._select(fits)]
            p2 = self.population[self._select(fits)]
            mask = self.rng.random(self.dim) < 0.5
            child = np.where(mask, p1, p2).astype(np.float32)
            mut = self.rng.random(self.dim) < self.mut_rate
            child[mut] += self.mut_std * self.rng.standard_normal(int(mut.sum())).astype(np.float32)
            new_pop.append(child)
        self._next_population = new_pop


@register("cmaes")
class CMAES(_Evolution):
    """Separable (diagonal) CMA-ES — O(dim) covariance, scalable to the net's params."""

    def __init__(self, obs_size, num_actions, cfg=None, seed=0):
        c = cfg or {}
        self.sigma = c.get("sigma", 0.3)
        self._cma_ready = False
        super().__init__(obs_size, num_actions, c, seed)

    def _init_cma(self) -> None:
        n = self.dim
        self.mean = self.policy.get_params().astype(np.float64)
        self.C = np.ones(n)                      # diagonal covariance
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        mu = self.pop_size // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.w = w / w.sum()
        self.mu = mu
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = 4.0 / (n + 4)
        self.cs = (self.mueff + 2) / (n + self.mueff + 3)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        # Separable speed-up factor (Ros & Hansen 2008).
        self.c1 = min(self.c1 * (n + 2) / 3.0, 1.0)
        self.cmu = min(self.cmu * (n + 2) / 3.0, 1 - self.c1)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        self._gen_count = 0
        self._cma_ready = True

    def _ask(self) -> List[np.ndarray]:
        if not self._cma_ready:
            self._init_cma()
        self._z = self.rng.standard_normal((self.pop_size, self.dim))
        sqrtC = np.sqrt(self.C)
        self._y = self._z * sqrtC
        return [(self.mean + self.sigma * y).astype(np.float32) for y in self._y]

    def _tell(self, fits: np.ndarray) -> None:
        order = np.argsort(fits)[::-1]      # maximise
        y_sel = self._y[order[:self.mu]]
        yw = (self.w[:, None] * y_sel).sum(axis=0)
        self.mean = self.mean + self.sigma * yw

        invsqrtC = 1.0 / np.sqrt(self.C)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (invsqrtC * yw)
        self._gen_count += 1
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self._gen_count)) / self.chiN
                < 1.4 + 2 / (self.dim + 1))
        self.pc = (1 - self.cc) * self.pc + (1.0 if hsig else 0.0) * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw

        delta = (1 - hsig) * self.cc * (2 - self.cc)
        rank_mu = (self.w[:, None] * (y_sel ** 2)).sum(axis=0)
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (self.pc ** 2 + delta * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = float(np.clip(self.sigma, 1e-8, 1e3))

    def _extra_state(self) -> Dict:
        return {"mean": self.mean, "C": self.C, "sigma": self.sigma,
                "pc": self.pc, "ps": self.ps, "gen": self._gen_count}

    def _load_extra(self, s: Dict) -> None:
        self._init_cma()                     # set hyperparams + mark ready (avoid re-init)
        self.mean = s["mean"]; self.C = s["C"]; self.sigma = s["sigma"]
        self.pc = s["pc"]; self.ps = s["ps"]; self._gen_count = s["gen"]
