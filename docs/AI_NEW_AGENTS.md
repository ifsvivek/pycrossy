# New AI Agents — Minimax Planner & Double DQN

Two agents added to the framework, both selectable by name everywhere (`train.py --algo …`,
Settings ▸ AI ▸ Algorithm, Auto-Play, Evaluation, Benchmark). They drop in via the same
`@register` registry — no game/trainer changes needed.

---

## 1. Minimax / Expectimax Search Agent (`minimax`)

A **planning** agent — it does not train, it *searches*. Each move it snapshots the live engine
into a compact forward model and runs a depth-limited search, choosing the action that maximises
expected long-term survival. File: [`ai/algorithms/search_agent.py`](../ai/algorithms/search_agent.py).

### Adapting minimax to a single-player stochastic game
Classic minimax assumes an adversary. Crossy Road has none, so the adversary layer is replaced
by **expectimax**: the agent is the only decision-maker (MAX nodes), and **chance nodes** model
the sole genuine uncertainty — *lanes that have not been generated yet*. Crucially, within the
engine's look-ahead (~22 rows are always generated ahead of the player) the dynamics are
**deterministic** — vehicles/logs/trains move at constant speed and wrap — so the search there is
exact one-agent maximisation. Chance nodes only fire when a plan would step past the generated
horizon (rare at the depths used), where the outcome is averaged over the possible lane types.

### Forward model (`_Model` / `_Search.step`)
Snapshotted per decision from the engine: player cell + riding state, and for each nearby row
its type, obstacle columns (grass), and every mover as `(x₀, speed, collision_box)`. A mover's
position at any future tick is computed in O(1) by the same wrap arithmetic the engine uses
(`((x₀ + span/2 + speed·t) mod span) − span/2`). One *decision* advances the model by the ticks it
really costs (a hop ≈ 12 ticks, a WAIT = 8) and resolves collisions exactly as the engine does:
a vehicle kills you if it overlaps your column while you occupy a road/rail row (including the
dwell when waiting or starting to leave); water drowns you unless a log/lily covers your landing
column, after which you **ride** and drift with it; the edge kills at |x| ≥ 5.

### Heuristic (the factors the task asked for)
`heuristic(state)` combines: **distance travelled** (dominant, `w_progress·score`), **time-to-
collision** against vehicle/train trajectories on the current row, **safe-landing** availability
on water, **escape routes** (how many of the three reachable forward columns are safe), **lane
safety ahead**, **proximity to the killing edge**, and a **navigation gradient** (`w_nav`). Forced
death returns a large negative that still prefers dying *deeper* (so it never gives up early).

**Navigation gradient (anti-stuck).** In tree "pockets" — where the column directly ahead is
walled off and the only way forward is to slide sideways or *back up and approach from a different
column* — there is no score gain within the search horizon, so a pure progress heuristic has no
signal and the agent can sit until it dies. To fix this, each decision runs a small **BFS over the
walkable grass** (4-connected, including backtracking) measuring the distance from every cell to the
nearest one from which it can actually advance, and the heuristic rewards being closer to that
frontier. Measured effect (40 held-out seeds, depth 6): **stuck-rate 12% → 5%** and mean score
**49 → 55**. Weights are configurable.

### Search techniques (all implemented)
- **Configurable depth** (`max_depth`) + **iterative deepening** under a per-move time budget
  (`time_budget_ms`) — always returns the best move found so far (real-time safe).
- **Branch-and-bound (alpha) pruning** with an admissible upper bound (≤ 1 new row per remaining
  move) to cut hopeless branches.
- **Beam search** (`beam_width`) — expand only the top actions per node.
- **Move ordering** — order children by a one-step heuristic (sharper pruning).
- **Transposition / state caching** within a search (memoise equal `(cell, phase, depth)`).
- **Expectiminimax chance nodes** for unobserved future lanes.

### Configuration
`max_depth` (6), `beam_width` (5), `time_budget_ms` (25), `iterative_deepening` (true),
`w_progress` (10), `w_safety` (6), `w_edge` (2), `w_nav` (5). Override via `--set`, e.g.
`python train.py --algo minimax --set max_depth=8 --set w_safety=9`. In the app, **AI Settings ▸
Minimax Search Depth** sets the depth for both Auto-Play and Training.

> **Running it.** Minimax is a *planner*, not a learner — `train.py --algo minimax` only *showcases*
> it (no metrics improve; use **Auto-Play** in the app to just watch). `--parallel` does **not**
> apply (it isn't population-based) and `--speed` paces the simulation but each move still costs its
> search time, so **`max_depth` is the real speed/strength dial** (depth 3–4 is much faster and still
> strong; depth 6 is the default). The CLI prints these notes when you launch it.

### Results — 100 held-out seeds (band 9,000,000+)
At ~15 ms / decision (depth 6, ~550 nodes), the planner is **by far** the strongest agent:

| config | mean | median | best | worst | std | ≥10 | ≥20 | ≥30 | ≥50 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| minimax depth 4 | 28.8 | 22.5 | 150 | 1 | 25.9 | 74% | 52% | 40% | 20% |
| minimax depth 6 | **47.7** | **33** | **331** | 1 | 51.3 | 83% | 65% | 57% | 34% |

*(depth 6 with the navigation-gradient anti-stuck heuristic, `w_nav=5`; 100 held-out seeds.)*

That is ~**8×** the best *learned* agent (NEAT 5.4) and far above the timing-aware engine oracle
(7.9), because it *plans several moves ahead* rather than reacting. Deeper search → higher score
at more inference time; depth 3–4 stays fast for high `--speed` play. Variance is high (some seeds
present an early unavoidable hazard → score ~1; others run past 200) which is intrinsic to the game.
HPO (`ai/tune.py`) over depth and heuristic weights confirms depth 6 + `(w_progress 10, w_safety 6,
w_edge 4)` as a strong setting.

---

## 2. Double DQN (`ddqn`)

A complete, **modular** DDQN whose Rainbow-style components are toggled through `cfg`; every
feature defaults to a tuned configuration (not library defaults).
File: [`ai/algorithms/ddqn.py`](../ai/algorithms/ddqn.py), network in `ai/networks.py`.

### Components (config flag → effect)
| flag | default | effect |
|---|---|---|
| `double` | on | online net selects the next action, target net values it (curbs overestimation) |
| `dueling` | on | separate state-value V(s) and advantage A(s,a) streams, `Q = V + (A − mean A)` |
| `per` (`per_alpha`, `per_beta`) | on | prioritized replay by TD-error, sum-tree O(log n), IS-weighted updates |
| `n_step` | 3 | bootstrap on the n-step discounted return |
| `obs_norm` | on | running mean/std whitening of observations |
| `reward_norm` | off | divide rewards by a running std |
| `target_sync` / `tau` | 1000 / 0 | hard target copy every N steps, or soft Polyak if `tau>0` |
| `eps_decay` (`eps_*`) | linear | ε-greedy schedule (`linear`|`exp`) to a floor |
| `lr` / `lr_end` / `lr_decay_steps` | 5e-4→5e-5 | linear LR schedule |
| `grad_clip` | 10 | global-norm gradient clipping; Huber (clipped) TD loss |

**Training/eval modes**: `act(obs, deterministic=True)` is greedy/no-noise (evaluation);
`deterministic=False` explores. Checkpoints are **architecture-aware** (store hidden sizes +
dueling flag + normalizer stats) so any config reloads correctly. Full pipeline support: headless
long runs, automatic checkpointing/resume, CSV + JSON + TensorBoard logging, periodic evaluation
and validated best-model saving (inherited from the trainer).

### Not implemented (documented choices)
**NoisyNets**, **distributional/C51**, and **frame-stacking** are intentionally omitted: the
PyCrossy observation is near-fully-observed (positions *and* velocities are in the vector), so
frame-stacking/NoisyNets add little, and C51 is a large addition with marginal expected gain for a
5-action task. The config is structured so they could be added as further `DuelingQNetwork`
variants without touching the training loop.

### Hyperparameter optimization
`ai/tune.py` provides random/grid search scored by held-out evaluation. The DDQN search space
covers lr, gamma, batch, buffer, target sync, n-step, PER α, hidden sizes and ε-decay:
```bash
python -m ai.tune --algo ddqn --trials 6 --episodes 1500 --n-eval 40
```
It writes `runs/tune/ddqn/trials.csv` + `best.json`.

---

## Integration & benchmarking

Both agents are wired into: **Main Menu / AI Settings** (`pycrossy/settings.py` `ai_algorithm`
choices), **Auto-Play & Replay** (`ai/session.py` builds a planner live and binds the engine;
learners load their checkpoint), **Training** (`train.py --algo …`), **Evaluation**
(`ai/evaluate.py`, incl. `evaluate_algo` for checkpoint-less planners), **Benchmark**
(`ai/benchmark.py`) and the **analytics dashboard** (generic metric series). Switching agents needs
no code changes — pick it in Settings or pass `--algo`.

Benchmark everything under identical conditions (shared held-out seeds):
```bash
python -m ai.benchmark --compare minimax runs/ddqn runs/neat2 runs/ppo2 runs/dqn2 runs/cmaes2 --n-eval 100
```

### Final comparison — all agents, 100 shared held-out seeds (`runs/bench/results.json`)

| agent | type | mean | median | best | ≥10 | ≥30 | infer/decision |
|---|---|--:|--:|--:|--:|--:|--:|
| **minimax (depth 6)** | planning | **47.7** | **33** | **331** | 83% | 57% | ~16 ms |
| minimax (depth 4) | planning | 28.8 | 22.5 | 150 | 74% | 40% | ~7 ms |
| neat | neuroevolution | 5.42 | 4.5 | 24 | 11% | 0% | <1 ms |
| ppo | policy gradient | 4.76 | 4.0 | 17 | 8% | 0% | <1 ms |
| cmaes | CMA-ES | 3.98 | 3.0 | 22 | 4% | 0% | <1 ms |
| dqn | value | 3.59 | 3.0 | 13 | 5% | 0% | <1 ms |
| ddqn | value | 3.54 | 3.0 | 13 | 4% | 0% | <1 ms |

**Takeaways.** (1) The **Minimax planner dominates by ~8×** — planning a few moves ahead over an
accurate forward model solves the water/timing crossings that reactive learned policies can't.
(2) Among *learners*, NEAT leads; the trained learners are bunched at 3.5–5.4 — the env's
difficulty (multi-step water crossings) and pure-numpy sample efficiency, not the algorithms, are
the ceiling. (3) DDQN's dueling/PER/n-step stack does **not** beat plain DQN here, which is itself
informative: those Rainbow components pay off most with high-dimensional/pixel inputs, whereas this
observation is compact and near-Markov. The planner is the right tool when a simulator is available;
the learners are the right tool when one is not.

