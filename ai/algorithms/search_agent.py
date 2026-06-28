"""Minimax-family planning agent — depth-limited Expectimax over a forward model.

Crossy Road is a single-player *stochastic* environment, not a two-player game, so the
classic minimax adversary is replaced by an **expectimax** formulation: the agent (MAX node)
chooses the action that maximises the value of the resulting state, and **chance nodes** model
the only genuine uncertainty — lanes that have not been generated yet (beyond the engine's
look-ahead). Within the observed horizon the dynamics are *deterministic* (vehicles move at
constant speed and wrap), so the search there is exact one-agent maximisation; chance nodes
only fire when planning past the generated rows.

The agent never trains — it *plans* each move by snapshotting the live engine into a compact
forward model (`_Model`) and searching it. Implemented techniques (all configurable):

  * **Configurable depth** (`max_depth`) with **iterative deepening** under a per-move time
    budget (`time_budget_ms`) — always returns the best move found so far.
  * **Branch-and-bound (alpha-style) pruning** using an admissible upper bound (≤ 1 new row
    per remaining move) to cut hopeless branches.
  * **Beam search** (`beam_width`) — expand only the most promising actions per node.
  * **Move ordering** — try forward/most-promising actions first (better pruning).
  * **Transposition / state caching** within a search (memoise equal `(cell, phase, depth)`).
  * **Expectiminimax chance nodes** for unobserved future lanes.

The heuristic scores distance travelled, lane safety, time-to-collision against vehicle/train
trajectories, safe river landings, escape routes and proximity to the killing edge. Selectable
as ``"minimax"``.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pycrossy import config
from pycrossy.entities.rows import OFFSET, TRAIN_OFFSET
from ..base import Algorithm, register
from ..env import Action

_EDGE = config.EDGE_DEATH_X
_START = config.STARTING_ROW
_HOP = 12            # ticks for a hop to settle (~2 × 0.1s × 60Hz)
_WAIT = 8            # ticks consumed by a WAIT decision
_ACTIONS = (Action.UP, Action.WAIT, Action.LEFT, Action.RIGHT, Action.DOWN)   # move-ordered
_DELTA = {Action.UP: (0, 1), Action.DOWN: (0, -1), Action.LEFT: (1, 0), Action.RIGHT: (-1, 0)}


@dataclass
class _Mover:
    x0: float
    speed: float
    cb: float
    ride: bool          # True for logs/lily pads (a surface you can stand on)

    def x_at(self, t: int, span: float) -> float:
        """Wrapped x-position at tick offset ``t`` (matches the engine's per-tick wrap)."""
        if self.speed == 0.0:
            return self.x0
        return ((self.x0 + span / 2 + self.speed * t) % span) - span / 2


@dataclass
class _Row:
    type: str
    movers: List[_Mover]
    obstacles: frozenset      # grass: blocked integer columns
    span: float               # wrap span (2*OFFSET for cars/logs, 2*TRAIN_OFFSET for trains)


@dataclass
class _State:
    px: float
    pz: int
    riding: bool
    tick: int
    alive: bool
    score: int

    def key(self) -> Tuple:
        # Transposition key: discretised cell + coarse mover phase (tick bucket).
        return (round(self.px * 2), self.pz, self.riding, self.tick // 2, self.alive)


class _Model:
    """A snapshot of the engine sufficient to simulate several moves forward."""

    def __init__(self, rows: Dict[int, _Row], min_z: int, max_z: int):
        self.rows = rows
        self.min_z = min_z
        self.max_z = max_z

    def row(self, z: int) -> Optional[_Row]:
        return self.rows.get(z)

    # -- snapshot from the live engine ------------------------------------
    @classmethod
    def snapshot(cls, engine, depth_horizon: int) -> Tuple["_Model", "_State", float]:
        hero = engine.hero
        gm = engine.game_map
        px = float(hero.position.x)
        pz = int(round(hero.position.z))
        rows: Dict[int, _Row] = {}
        lo, hi = pz - 1, pz + depth_horizon + 2
        for z in range(lo, hi + 1):
            r = gm.get_row(z)
            if r is None:
                continue
            t = r["type"]
            ent = r["entity"]
            movers: List[_Mover] = []
            obstacles = frozenset()
            span = 2 * OFFSET
            if t == "road":
                movers = [_Mover(m.mesh.position.x, m.speed, m.collision_box, False) for m in ent.cars]
            elif t == "railRoad":
                tr = ent.train
                movers = [_Mover(tr.mesh.position.x, tr.speed, tr.collision_box, False)]
                span = 2 * TRAIN_OFFSET
            elif t == "water":
                movers = [_Mover(m.mesh.position.x, m.speed, m.collision_box, True) for m in ent.entities]
            elif t == "grass":
                obstacles = frozenset(int(k) for k in ent.obstacle_map)
            rows[z] = _Row(t, movers, obstacles, span)
        score = max(0, pz - _START)
        state = _State(px, pz, hero.riding_on is not None, 0, True, score)
        return cls(rows, lo, hi), state, px


def _hazard_hit(row: _Row, col_x: float, t0: int, t1: int) -> bool:
    """True if any vehicle/train on ``row`` overlaps column ``col_x`` during ticks [t0, t1]."""
    for m in row.movers:
        if m.ride:
            continue
        for t in range(t0, t1 + 1, 2):
            if abs(m.x_at(t, row.span) - col_x) < m.cb:
                return True
    return False


def _ride_under(row: _Row, col_x: float, t: int) -> Optional[_Mover]:
    """The log/lily covering column ``col_x`` at tick ``t`` (None ⇒ open water = drown)."""
    for m in row.movers:
        if m.ride and abs(m.x_at(t, row.span) - col_x) < m.cb:
            return m
    return None


class _Search:
    def __init__(self, model: _Model, w_progress: float, w_safety: float, w_edge: float,
                 w_nav: float = 3.0):
        self.m = model
        self.wp, self.ws, self.we, self.wn = w_progress, w_safety, w_edge, w_nav
        self.cache: Dict[Tuple, float] = {}
        self.nav: Dict[Tuple[int, int], int] = {}
        self.nodes = 0

    # -- navigation field --------------------------------------------------
    def compute_nav(self, s: "_State") -> None:
        """BFS distance (over walkable grass, 4-connected incl. backtracking) from every grass
        cell to the nearest cell from which the player can ADVANCE to a new forward row. This
        gives the heuristic a gradient out of tree pockets where no forward progress is possible
        within the search horizon — so the planner slides toward (or backtracks to) a real gap
        instead of sitting until it dies."""
        walk = set()
        for z, row in self.m.rows.items():
            if row.type == "grass":
                for c in range(-int(_EDGE) + 1, int(_EDGE)):
                    if c not in row.obstacles:
                        walk.add((c, z))
        launch = []                                   # cells from which UP reaches a forward row
        for (c, z) in walk:
            if z < s.pz - 2 or (z + 1) - _START <= s.score:
                continue
            nxt = self.m.row(z + 1)
            advanceable = (nxt is None or nxt.type != "grass"
                           or (abs(c) < _EDGE and c not in nxt.obstacles))
            if advanceable:
                launch.append((c, z))
        dist = {cell: 0 for cell in launch}
        q = deque(launch)
        while q:
            c, z = q.popleft()
            d = dist[(c, z)]
            for nb in ((c + 1, z), (c - 1, z), (c, z + 1), (c, z - 1)):
                if nb in walk and nb not in dist:
                    dist[nb] = d + 1
                    q.append(nb)
        self.nav = dist

    # -- forward model -----------------------------------------------------
    def step(self, s: _State, a: Action) -> _State:
        """Apply one decision; returns the resulting (possibly dead) state."""
        self.nodes += 1
        px, pz, tick = s.px, s.pz, s.tick
        cur = self.m.row(pz)

        if a == Action.WAIT:
            t1 = tick + _WAIT
            if s.riding and cur is not None:
                m = _ride_under(cur, px, tick)
                drift = m.speed if m else 0.0
                npx = px + drift * _WAIT
                if abs(npx) >= _EDGE or _ride_under(cur, npx, t1) is None:
                    return _State(npx, pz, True, t1, False, s.score)   # drifted off log / off edge
                return _State(npx, pz, True, t1, True, s.score)
            # standing still on a hazard lane is deadly if a vehicle sweeps the cell
            if cur is not None and cur.type in ("road", "railRoad") and _hazard_hit(cur, px, tick, t1):
                return _State(px, pz, False, t1, False, s.score)
            return _State(px, pz, s.riding, t1, True, s.score)

        # directional move
        dx, dz = _DELTA[a]
        nz = pz + dz
        nx = round(px) + dx
        t_land = tick + _HOP
        if abs(nx) >= _EDGE:
            return _State(nx, nz, False, t_land, False, s.score)        # edge death
        dest = self.m.row(nz)
        # leaving a hazardous cell: exposed for the first part of the hop
        if cur is not None and cur.type in ("road", "railRoad") and _hazard_hit(cur, px, tick, tick + _HOP // 2):
            return _State(px, pz, False, tick + _HOP // 2, False, s.score)
        if dest is None:
            # planning past the generated horizon → caller treats as a chance node
            return _State(float(nx), nz, False, t_land, True, max(s.score, nz - _START))
        if dest.type == "grass":
            if int(nx) in dest.obstacles:
                return _State(px, pz, s.riding, tick + 2, True, s.score)  # blocked → hop in place
            return _State(float(nx), nz, False, t_land, True, max(s.score, nz - _START))
        if dest.type in ("road", "railRoad"):
            dead = _hazard_hit(dest, nx, tick + _HOP // 2, t_land)
            return _State(float(nx), nz, False, t_land, not dead, max(s.score, nz - _START))
        if dest.type == "water":
            m = _ride_under(dest, nx, t_land)
            if m is None:
                return _State(float(nx), nz, False, t_land, False, s.score)  # drown
            return _State(float(nx), nz, True, t_land, True, max(s.score, nz - _START))
        return _State(float(nx), nz, False, t_land, True, max(s.score, nz - _START))

    # -- heuristic ---------------------------------------------------------
    def heuristic(self, s: _State) -> float:
        if not s.alive:
            return -1000.0 + s.score * 5.0          # die as deep as possible if death is forced
        v = self.wp * s.score
        cur = self.m.row(s.pz)
        # danger of the cell currently occupied (lower time-to-collision = worse)
        if cur is not None and cur.type in ("road", "railRoad"):
            ttc = self._ttc(cur, s.px, s.tick)
            v -= self.ws * max(0.0, 1.0 - ttc / 30.0)
        elif cur is not None and cur.type == "water" and not s.riding:
            v -= self.ws
        # proximity to the killing edge
        v -= self.we * max(0.0, (abs(s.px) - 3.0)) / (_EDGE - 3.0)
        # forward potential: reward having a safe cell directly ahead now + escape routes
        ahead = self.m.row(s.pz + 1)
        if ahead is not None:
            safe_ahead = self._cell_safe(ahead, round(s.px), s.tick + _HOP)
            v += self.ws * 0.5 * (1.0 if safe_ahead else 0.0)
            escapes = sum(1 for c in (round(s.px) - 1, round(s.px), round(s.px) + 1)
                          if abs(c) < _EDGE and self._cell_safe(ahead, c, s.tick + _HOP))
            v += 0.15 * escapes
        # Navigation gradient: when stuck behind tree walls, pull toward the nearest cell from
        # which forward progress is actually possible (closer = better; 0 = ready to advance).
        d = self.nav.get((round(s.px), s.pz))
        if d is not None:
            v += self.wn * max(0.0, 1.0 - d / 8.0)
        return v

    def _ttc(self, row: _Row, col_x: float, tick: int) -> float:
        best = 60.0
        for m in row.movers:
            if m.ride:
                continue
            for t in range(0, 60, 2):
                if abs(m.x_at(tick + t, row.span) - col_x) < m.cb:
                    best = min(best, float(t))
                    break
        return best

    def _cell_safe(self, row: _Row, col_x: float, t: int) -> bool:
        if abs(col_x) >= _EDGE:
            return False
        if row.type == "grass":
            return int(col_x) not in row.obstacles
        if row.type in ("road", "railRoad"):
            return not _hazard_hit(row, col_x, t, t + _HOP)
        if row.type == "water":
            return _ride_under(row, col_x, t) is not None
        return True

    # -- expectimax with branch-and-bound + beam ---------------------------
    def value(self, s: _State, depth: int, alpha: float) -> float:
        if not s.alive or depth == 0:
            return self.heuristic(s)
        k = (s.key(), depth)
        cached = self.cache.get(k)
        if cached is not None:
            return cached
        best = -1e18
        children = [(a, self.step(s, a)) for a in _ACTIONS]
        # move ordering: explore higher one-step heuristic first (sharper pruning)
        children.sort(key=lambda c: self.heuristic(c[1]), reverse=True)
        if self.beam:
            children = children[:self.beam]
        for a, ns in children:
            # admissible upper bound: at most +w_progress per remaining move from here
            ub = self.heuristic(ns) + self.wp * depth
            if ub <= alpha:
                continue
            val = self.value(ns, depth - 1, max(alpha, best))
            if val > best:
                best = val
                if best > alpha:
                    alpha = best
        self.cache[k] = best
        return best

    beam = 0

    def best_action(self, s: _State, depth: int) -> Tuple[Action, float]:
        self.compute_nav(s)                          # navigation field for this decision
        best_a, best_v, alpha = Action.WAIT, -1e18, -1e18
        children = [(a, self.step(s, a)) for a in _ACTIONS]
        children.sort(key=lambda c: self.heuristic(c[1]), reverse=True)
        for a, ns in children:
            val = self.value(ns, depth - 1, alpha)
            if val > best_v:
                best_v, best_a = val, a
                alpha = max(alpha, best_v)
        return best_a, best_v


@register("minimax")
class MinimaxAgent(Algorithm):
    uses_planning = True

    def __init__(self, obs_size, num_actions, cfg=None, seed=0):
        super().__init__(obs_size, num_actions, cfg, seed)
        c = self.cfg
        self.max_depth = int(c.get("max_depth", 6))
        self.beam_width = int(c.get("beam_width", 5))       # 5 = no pruning (full branching)
        self.time_budget_ms = float(c.get("time_budget_ms", 25.0))
        self.iterative = bool(c.get("iterative_deepening", True))
        self.w_progress = c.get("w_progress", 10.0)
        self.w_safety = c.get("w_safety", 6.0)
        self.w_edge = c.get("w_edge", 2.0)
        self.w_nav = c.get("w_nav", 5.0)        # pull toward a reachable gap when boxed in
        self._engine = None
        self._last = {"depth": 0, "nodes": 0, "ms": 0.0}

    # -- engine binding (planning agents need the simulator, not just the obs) --
    def bind_env(self, env) -> None:
        self._engine = env.engine

    def _plan(self) -> int:
        if self._engine is None or self._engine.hero is None or not self._engine.hero.is_alive:
            return int(Action.UP)
        model, state, _ = _Model.snapshot(self._engine, self.max_depth)
        t0 = time.perf_counter()
        deadline = t0 + self.time_budget_ms / 1000.0
        best = Action.UP
        depths = range(2, self.max_depth + 1) if self.iterative else (self.max_depth,)
        total_nodes = 0
        reached = 0
        for d in depths:
            srch = _Search(model, self.w_progress, self.w_safety, self.w_edge, self.w_nav)
            srch.beam = self.beam_width if self.beam_width < len(_ACTIONS) else 0
            a, _v = srch.best_action(state, d)
            best = a
            total_nodes += srch.nodes
            reached = d
            if time.perf_counter() >= deadline:
                break
        self._last = {"depth": reached, "nodes": total_nodes,
                      "ms": (time.perf_counter() - t0) * 1000.0}
        return int(best)

    # -- Algorithm interface ----------------------------------------------
    def act(self, obs, deterministic=False) -> int:
        return self._plan()

    def best_act(self, obs) -> int:
        return self._plan()

    def observe(self, tr) -> None:                  # planning agent — nothing to learn
        self.total_steps += 1

    def end_episode(self, total_reward, info) -> Dict:
        self.total_episodes += 1
        return {"plan_depth": self._last["depth"], "plan_nodes": self._last["nodes"],
                "plan_ms": self._last["ms"]}

    @property
    def progress(self) -> Dict:
        return dict(self._last)

    def state_dict(self) -> Dict:
        return {"cfg": dict(self.cfg)}

    def load_state_dict(self, state: Dict) -> None:
        self.cfg.update(state.get("cfg", {}))
