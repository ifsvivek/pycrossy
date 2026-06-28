"""NEAT — NeuroEvolution of Augmenting Topologies (from scratch).

A faithful implementation of Stanley & Miikkulainen's NEAT: genomes encode nodes +
innovation-numbered connections, evolved by weight perturbation, add-connection and
add-node mutations, innovation-aligned crossover, and speciation with fitness sharing.

Fits the episodic :class:`~ai.base.Algorithm` interface: each genome is evaluated for one
episode; once the whole population has been scored, the population speciates and
reproduces into the next generation.
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..base import Algorithm, register


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-4.9 * x)) if -20 < x < 20 else (0.0 if x <= 0 else 1.0)


@dataclass
class Connection:
    in_node: int
    out_node: int
    weight: float
    enabled: bool
    innov: int


@dataclass
class Genome:
    inputs: int
    outputs: int
    nodes: List[int] = field(default_factory=list)        # all node ids
    node_type: Dict[int, str] = field(default_factory=dict)  # input/output/hidden/bias
    conns: Dict[int, Connection] = field(default_factory=dict)  # by innovation
    fitness: float = 0.0
    adj_fitness: float = 0.0

    def clone(self) -> "Genome":
        g = Genome(self.inputs, self.outputs, list(self.nodes), dict(self.node_type))
        g.conns = {i: Connection(c.in_node, c.out_node, c.weight, c.enabled, c.innov)
                   for i, c in self.conns.items()}
        return g

    # -- feed-forward activation ------------------------------------------
    def forward(self, obs: np.ndarray) -> np.ndarray:
        values: Dict[int, float] = {}
        for i in range(self.inputs):
            values[i] = float(obs[i])
        values[self.inputs] = 1.0          # bias node
        # Evaluate hidden/output nodes in dependency order (graph is acyclic).
        order = self._topo_order()
        for nid in order:
            if self.node_type[nid] in ("input", "bias"):
                continue
            s = 0.0
            for c in self._incoming.get(nid, ()):
                if c.enabled:
                    s += values.get(c.in_node, 0.0) * c.weight
            values[nid] = _sigmoid(s)
        out_start = self.inputs + 1
        return np.array([values.get(out_start + j, 0.0) for j in range(self.outputs)])

    _incoming: Dict[int, List[Connection]] = field(default_factory=dict, repr=False)
    _order_cache: Optional[List[int]] = field(default=None, repr=False)

    def _rebuild_index(self) -> None:
        self._incoming = {}
        for c in self.conns.values():
            self._incoming.setdefault(c.out_node, []).append(c)
        self._order_cache = None

    def _topo_order(self) -> List[int]:
        if self._order_cache is not None:
            return self._order_cache
        # Kahn topological sort over enabled edges.
        adj: Dict[int, List[int]] = {n: [] for n in self.nodes}
        indeg: Dict[int, int] = {n: 0 for n in self.nodes}
        for c in self.conns.values():
            if c.enabled and c.in_node in indeg and c.out_node in indeg:
                adj[c.in_node].append(c.out_node)
                indeg[c.out_node] += 1
        queue = [n for n in self.nodes if indeg[n] == 0]
        order: List[int] = []
        while queue:
            n = queue.pop()
            order.append(n)
            for m in adj[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
        self._order_cache = order
        return order


class _Innovation:
    """Tracks global innovation numbers for connections and node-splits."""

    def __init__(self, start_node: int):
        self.conn_innov: Dict[Tuple[int, int], int] = {}
        self._next_innov = 0
        self._next_node = start_node

    def conn(self, a: int, b: int) -> int:
        key = (a, b)
        if key not in self.conn_innov:
            self.conn_innov[key] = self._next_innov
            self._next_innov += 1
        return self.conn_innov[key]

    def new_node(self) -> int:
        n = self._next_node
        self._next_node += 1
        return n


@register("neat")
class NEAT(Algorithm):
    def __init__(self, obs_size, num_actions, cfg=None, seed=0):
        super().__init__(obs_size, num_actions, cfg, seed)
        c = self.cfg
        self.pop_size = int(c.get("pop_size", 60))
        self.c1 = c.get("c1", 1.0)
        self.c2 = c.get("c2", 1.0)
        self.c3 = c.get("c3", 0.4)
        self.compat_threshold = c.get("compat_threshold", 3.0)
        self.weight_mut_rate = c.get("weight_mut_rate", 0.8)
        self.weight_perturb = c.get("weight_perturb", 0.5)
        self.add_conn_rate = c.get("add_conn_rate", 0.1)
        self.add_node_rate = c.get("add_node_rate", 0.05)
        self.survival = c.get("survival_threshold", 0.3)
        self.elitism = c.get("elitism", 1)
        self.eval_episodes = int(c.get("eval_episodes", 1))

        self.inputs = obs_size
        self.outputs = num_actions
        self.innov = _Innovation(start_node=obs_size + 1 + num_actions)
        self.generation = 0
        self.species: List[List[Genome]] = []
        self.population = self._initial_population()
        self._eval_idx = 0
        self._ep_rewards: List[float] = []
        self.best_genome: Genome = self.population[0]
        self.best_fitness = -1e9
        self.mean_fitness = 0.0
        self.num_species = 0
        self._cur = self.population[0]

    # -- population init ---------------------------------------------------
    def _base_genome(self) -> Genome:
        g = Genome(self.inputs, self.outputs)
        for i in range(self.inputs):
            g.nodes.append(i); g.node_type[i] = "input"
        bias = self.inputs
        g.nodes.append(bias); g.node_type[bias] = "bias"
        for j in range(self.outputs):
            nid = self.inputs + 1 + j
            g.nodes.append(nid); g.node_type[nid] = "output"
        return g

    def _initial_population(self) -> List[Genome]:
        pop = []
        out_start = self.inputs + 1
        for _ in range(self.pop_size):
            g = self._base_genome()
            # Start minimally connected: bias + a small random subset of inputs -> outputs.
            sources = [self.inputs] + list(self.rng.choice(self.inputs, size=min(8, self.inputs), replace=False))
            for j in range(self.outputs):
                for s in sources:
                    if self.rng.random() < 0.6:
                        innov = self.innov.conn(int(s), out_start + j)
                        g.conns[innov] = Connection(int(s), out_start + j,
                                                    float(self.rng.normal(0, 1)), True, innov)
            g._rebuild_index()
            pop.append(g)
        return pop

    # -- episodic interface ------------------------------------------------
    def begin_episode(self) -> None:
        self._cur = self.population[self._eval_idx]
        self._cur._rebuild_index()
        self._acc_reward = 0.0
        self._episodes_done = 0

    def act(self, obs, deterministic=False) -> int:
        return int(np.argmax(self._cur.forward(obs)))

    def best_act(self, obs) -> int:
        self.best_genome._rebuild_index()
        return int(np.argmax(self.best_genome.forward(obs)))

    def end_episode(self, total_reward, info) -> Dict:
        self.total_episodes += 1
        self._acc_reward = getattr(self, "_acc_reward", 0.0) + total_reward
        self._episodes_done = getattr(self, "_episodes_done", 0) + 1
        if self._episodes_done < self.eval_episodes:
            return self._metrics(advancing=False)

        fitness = self._acc_reward / self.eval_episodes
        self._cur.fitness = fitness
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_genome = self._cur.clone()

        self._eval_idx += 1
        if self._eval_idx >= len(self.population):
            self._reproduce()
            self._eval_idx = 0
        return self._metrics(advancing=True)

    # -- speciation + reproduction ----------------------------------------
    def _distance(self, a: Genome, b: Genome) -> float:
        ca, cb = a.conns, b.conns
        all_innov = set(ca) | set(cb)
        if not all_innov:
            return 0.0
        max_a = max(ca) if ca else 0
        max_b = max(cb) if cb else 0
        excess = disjoint = matching = 0
        wdiff = 0.0
        for i in all_innov:
            in_a, in_b = i in ca, i in cb
            if in_a and in_b:
                matching += 1
                wdiff += abs(ca[i].weight - cb[i].weight)
            else:
                if (in_a and i > max_b) or (in_b and i > max_a):
                    excess += 1
                else:
                    disjoint += 1
        n = max(len(ca), len(cb))
        n = 1 if n < 20 else n
        wmean = (wdiff / matching) if matching else 0.0
        return self.c1 * excess / n + self.c2 * disjoint / n + self.c3 * wmean

    def _speciate(self) -> None:
        reps = [s[0] for s in self.species] if self.species else []
        new_species: List[List[Genome]] = [[] for _ in reps]
        for g in self.population:
            placed = False
            for k, rep in enumerate(reps):
                if self._distance(g, rep) < self.compat_threshold:
                    new_species[k].append(g)
                    placed = True
                    break
            if not placed:
                reps.append(g)
                new_species.append([g])
        self.species = [s for s in new_species if s]
        self.num_species = len(self.species)

    def _reproduce(self) -> None:
        self.mean_fitness = float(np.mean([g.fitness for g in self.population]))
        self._speciate()

        # Fitness sharing: adjusted fitness = fitness / species size.
        offset = min(g.fitness for g in self.population)
        offset = -offset + 1.0 if offset <= 0 else 0.0
        species_adj = []
        for sp in self.species:
            for g in sp:
                g.adj_fitness = (g.fitness + offset) / len(sp)
            species_adj.append(sum(g.adj_fitness for g in sp))
        total_adj = sum(species_adj) or 1.0

        new_pop: List[Genome] = []
        for k, sp in enumerate(self.species):
            sp.sort(key=lambda g: g.fitness, reverse=True)
            n_offspring = int(round(species_adj[k] / total_adj * self.pop_size))
            if n_offspring <= 0:
                continue
            # Elitism: keep the best of each species.
            if len(sp) > 0 and self.elitism > 0:
                new_pop.append(sp[0].clone())
                n_offspring -= 1
            survivors = sp[:max(1, int(math.ceil(len(sp) * self.survival)))]
            for _ in range(max(0, n_offspring)):
                if len(survivors) == 1 or self.rng.random() < 0.25:
                    child = self.rng.choice(survivors).clone()
                else:
                    p1, p2 = self.rng.choice(len(survivors), size=2, replace=False)
                    child = self._crossover(survivors[p1], survivors[p2])
                self._mutate(child)
                child._rebuild_index()
                new_pop.append(child)

        # Pad/trim to exact population size.
        while len(new_pop) < self.pop_size:
            child = self.rng.choice(self.population).clone()
            self._mutate(child); child._rebuild_index(); new_pop.append(child)
        self.population = new_pop[:self.pop_size]
        self.generation += 1

    def _crossover(self, a: Genome, b: Genome) -> Genome:
        if b.fitness > a.fitness:
            a, b = b, a   # a is the fitter parent
        child = Genome(self.inputs, self.outputs, list(a.nodes), dict(a.node_type))
        for innov, ca in a.conns.items():
            cb = b.conns.get(innov)
            src = ca if (cb is None or self.rng.random() < 0.5) else cb
            enabled = True
            if (cb is not None) and (not ca.enabled or not cb.enabled):
                enabled = self.rng.random() > 0.75
            child.conns[innov] = Connection(src.in_node, src.out_node, src.weight, enabled, innov)
            for nd in (src.in_node, src.out_node):
                if nd not in child.node_type:
                    child.nodes.append(nd)
                    child.node_type[nd] = "hidden"
        return child

    # -- mutation ----------------------------------------------------------
    def _mutate(self, g: Genome) -> None:
        if self.rng.random() < self.weight_mut_rate:
            for c in g.conns.values():
                if self.rng.random() < 0.9:
                    c.weight += float(self.rng.normal(0, self.weight_perturb))
                else:
                    c.weight = float(self.rng.normal(0, 1))
        if self.rng.random() < self.add_conn_rate:
            self._mutate_add_connection(g)
        if self.rng.random() < self.add_node_rate:
            self._mutate_add_node(g)

    def _creates_cycle(self, g: Genome, a: int, b: int) -> bool:
        # Would an edge a->b create a cycle? True if b can already reach a.
        stack = [b]
        seen = set()
        edges: Dict[int, List[int]] = {}
        for c in g.conns.values():
            if c.enabled:
                edges.setdefault(c.in_node, []).append(c.out_node)
        while stack:
            n = stack.pop()
            if n == a:
                return True
            for m in edges.get(n, ()):
                if m not in seen:
                    seen.add(m); stack.append(m)
        return False

    def _mutate_add_connection(self, g: Genome) -> None:
        non_input = [n for n in g.nodes if g.node_type[n] not in ("input", "bias")]
        non_output = [n for n in g.nodes if g.node_type[n] != "output"]
        if not non_input or not non_output:
            return
        for _ in range(20):
            a = int(self.rng.choice(non_output))
            b = int(self.rng.choice(non_input))
            if a == b:
                continue
            innov = self.innov.conn(a, b)
            if innov in g.conns:
                continue
            if self._creates_cycle(g, a, b):
                continue
            g.conns[innov] = Connection(a, b, float(self.rng.normal(0, 1)), True, innov)
            return

    def _mutate_add_node(self, g: Genome) -> None:
        enabled = [c for c in g.conns.values() if c.enabled]
        if not enabled:
            return
        c = self.rng.choice(enabled)
        c.enabled = False
        new_node = self.innov.new_node()
        g.nodes.append(new_node); g.node_type[new_node] = "hidden"
        i1 = self.innov.conn(c.in_node, new_node)
        i2 = self.innov.conn(new_node, c.out_node)
        g.conns[i1] = Connection(c.in_node, new_node, 1.0, True, i1)
        g.conns[i2] = Connection(new_node, c.out_node, c.weight, True, i2)

    # -- metrics / persistence --------------------------------------------
    def _metrics(self, advancing: bool) -> Dict:
        nodes = len(self.best_genome.nodes)
        conns = sum(1 for c in self.best_genome.conns.values() if c.enabled)
        return {
            "generation": self.generation,
            "species": self.num_species,
            "diversity": float(self.num_species),     # species count = population diversity
            "genome_count": len(self.population),
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "current_fitness": self._cur.fitness,
            "num_nodes": nodes,
            "num_conns": conns,
            "mutation_rate": self.add_node_rate,
            "advanced_generation": advancing and self._eval_idx == 0,
        }

    # -- parallel evaluation protocol -------------------------------------
    supports_parallel = True

    def population_payloads(self):
        for g in self.population:
            g._rebuild_index()
        return [("neat", g) for g in self.population]

    def set_population_fitness(self, fits) -> Dict:
        fits = [float(f) for f in fits]
        for g, f in zip(self.population, fits):
            g.fitness = f
        bi = int(np.argmax(fits))
        if fits[bi] > self.best_fitness:
            self.best_fitness = fits[bi]
            self.best_genome = self.population[bi].clone()
        self.total_episodes += len(fits)
        self._cur = self.population[bi]
        self._reproduce()
        return self._metrics(advancing=True)

    @property
    def progress(self) -> Dict:
        return {"generation": self.generation, "species": self.num_species,
                "genome_count": len(self.population), "best_fitness": self.best_fitness,
                "eval_index": self._eval_idx}

    def state_dict(self) -> Dict:
        return {"population": pickle.dumps(self.population),
                "best": pickle.dumps(self.best_genome),
                "generation": self.generation, "best_fitness": self.best_fitness,
                "innov_conn": self.innov.conn_innov,
                "innov_next": self.innov._next_innov, "node_next": self.innov._next_node}

    def load_state_dict(self, state: Dict) -> None:
        self.population = pickle.loads(state["population"])
        self.best_genome = pickle.loads(state["best"])
        self.generation = state["generation"]
        self.best_fitness = state["best_fitness"]
        self.innov.conn_innov = state["innov_conn"]
        self.innov._next_innov = state["innov_next"]
        self.innov._next_node = state["node_next"]
        self._eval_idx = 0
        self._cur = self.population[0]
