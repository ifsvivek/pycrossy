# RL Framework Audit & Redesign — PyCrossy

*Lead RL researcher review. All findings below are code-verified (file:line cited). One hypothesis (death-penalty cliff) was refuted and is noted in §6. Throughout, "the parallel path" = `ai/trainer.py:train_generation`, which is the path NEAT/CMA-ES/ES/GA actually ran on.*

---

## 1. Executive Summary — Root Causes, Ranked

The reported pathology ("best score 28–30, mean_fitness 1.79, species=1") is **not** an algorithm-quality problem. It is the predictable output of three independent measurement/selection bugs that turn evolution into a lottery, plus one reward bug that makes the lottery's *expected* winner a coward. Fix these four and the headline numbers will converge; the per-component RL improvements (§3) raise the ceiling afterward.

### R1 (CRITICAL) — Within-generation selection is decided by map luck, not policy skill
**Evidence:** `ai/trainer.py:169` — `seeds = [self.cfg.seed*100003 + self.episode + i for i in range(len(payloads))]`. The `+ i` gives every candidate in a generation a *different* map; `ai/vec_env.py:54` `env.reset(seed=seed)` makes the seed fully determine the map and all hazard motion (`ai/env.py:85-90` reseeds both `self._rng` and the module-level `random`). Crossy map difficulty variance (an easy seed → score ~30, a hard one → ~2) dwarfs the skill gap between neighbouring genomes, so `argmax`/rank selection in NEAT (`neat_algo.py:405-416`), ES (`evolutionary.py:135-140`), GA (`:197-203`) and CMA-ES (`:256-258`) mostly picks whoever drew the easiest map. Across generations the carried elite is re-scored on a *new* map and regresses to the mean.
**Expected impact:** This is the single biggest lever. With Common Random Numbers (CRN), mean_fitness should begin climbing within a few generations instead of pinning at the passive baseline. Converts "no learning" → "learning."

### R2 (CRITICAL) — The reward function selects for cowardice
**Evidence:** `ai/env.py:115` `reward = 0.05` (unconditional per-step survival bonus, always positive) and `:130-131` `reward -= 5.0` on death, with fitness = raw sum of per-step rewards (`trainer.py:171-172`). A do-nothing genome that waits to the idle cap scores `40×(0.05−0.02) − 1.0 = +0.2` and survives; a brave genome that advances 3 rows then dies scores `3×1.05 − 4.95 = −1.80`. **Cowardice beats bravery by ~+4 fitness** until a genome banks >5 rows — but only row 9 is guaranteed grass. Observed mean_fitness 1.79 sits exactly on the "creep 2 free grass rows then loiter" baseline (~+2.2).
**Expected impact:** Largest *single-file* lever. Removes the dominant local optimum; mean fitness decouples from "stall on grass" and starts tracking actual progress. Trivial effort.

### R3 (CRITICAL) — The champion / "best score" is an unvalidated max-of-noise order statistic
**Evidence:** `neat_algo.py:409-412` keeps `best_fitness` as a running max over single-episode rewards and never re-tests it; `evolutionary.py:96-100` does the same for `best_params`; the ES/CMA-ES distribution centre is never evaluated at all (`evolutionary.py:163-165` is a dead `pass`). `train_generation` (`trainer.py:156-183`) **never calls `_evaluate()`**, so `best_eval` stays at its `-1e18` init (`trainer.py:51`) and `save_best` (`:215`) persists `best_eval=None`. "Best score 28–30" is the luckiest of ~3000 noisy draws, not a reproducible policy.
**Expected impact:** Makes the headline number honest and the saved model deployable. Collapses the misleading 30× best-vs-mean gap from a *measurement artifact* into a real (and now improvable) signal.

### R4 (CRITICAL, DQN-specific) — DQN was run rendered/real-time-paced, not headless
**Evidence:** `runs/dqn/summary.json`: 9650 steps / 1394 s = **6.92 steps/s**. `train.py:32` `--headless` is opt-in; the default `run_live` path paces the sim via `time.sleep` in `render_tick` (`train.py:141-143`) at speed=3 → 180 ticks/s, and `env._advance_until_settled` fires `on_tick` every tick (`env.py:155-166`). 24 ticks × 5.56 ms ≈ 133 ms/step ≈ 7.5 steps/s ceiling. The numpy learner is *not* the bottleneck (`_learn` ≈ 0.85 ms, ~27% over the 3.2 ms env step). Epsilon stuck at 0.54 is a *symptom* of only reaching 9650/20000 decay steps.
**Expected impact:** ~30–40× throughput from one CLI flag; epsilon anneals; DQN collects enough data to learn. (H6's "train_every=1 is 50× too slow" premise is **refuted** for headless.)

**Net:** R1+R3 are the same disease (noisy single-sample selection with no held-out validation) seen from the selection side and the reporting side; fix them together. R2 is orthogonal and trivial. R4 is DQN-only. Everything in §3 is secondary until these land.

---

## 2. Per-Dimension Findings (severity-ordered)

### Evaluation / Noise (the dominant dimension)
- **CRITICAL** — Per-candidate different seeds destroy within-generation comparability (`trainer.py:169`). [R1]
- **CRITICAL** — Parallel path never re-validates; champion + saved model are single-episode outliers (`trainer.py:156-183`, `neat_algo.py:409-412`). [R3]
- **HIGH** — `eval_episodes` silently ignored in the parallel path: exactly one noisy episode per candidate (`vec_env.py:47-65`; `TrainConfig.eval_episodes=5` only feeds the sequential `_evaluate`). Must be fixed *jointly* with R1 — averaging only helps over a *shared* seed bank.
- **HIGH** — Reported `best_score = max(...)` over 3000 random maps (`trainer.py:176`, `metrics.py:70-71`) is an upper-tail statistic, not capability.

### Reward
- **CRITICAL** — Cowardice trap: absolute +0.05/step survival + −5.0 death cliff invert selection (`env.py:115,130-131`). [R2]
- **HIGH** — Idle counter resets only on a new max row (`env.py:124-127`), so lateral repositioning and log-riding count as "idle" and trigger premature −1.0 termination (`:135-137`) — exactly the maneuvers water/obstacle crossings *require*.
- **HIGH** — Survival reward has the wrong sign; enables wait-padding (one forward hop resets idle, then up to 39 WAITs at +0.03 each → a score-28 run banks ~+60 instead of +28). *Partially confirmed:* the "+75 standing still" figure is bounded to ~+1 by the idle cap; the real exploit is per-row wait-padding.

### Environment
- **HIGH** — `max_idle=40` terminates mid-crossing; a worst-case railroad wait needs ~34 consecutive WAITs (train cycle 275 ticks / wait_ticks 8), already 85% of budget before lateral aiming (`env.py:61,124-137`). *Correction:* keep the reset condition as "forward>0 only" — do **not** reset on lateral moves or you destroy the loop detector; just raise the cap.
- **HIGH** — `wait_ticks=8` (0.133 s) is shorter than a hop (~12 ticks) and 1–2 orders below hazard cycles (1–18 s), so timing a gap requires long WAIT bursts that blow the idle budget (`env.py:61,102-103`).
- *(env summary)* Roads/railroads generate hazards once at construction and recycle (~20 distinct layouts/kind); no difficulty curve. Not a bug, but limits generalization.

### Observation
- **CRITICAL** — No time-to-collision: all col-safety/hazard features are decision-instant snapshots, but the hop settles ~12 ticks later during which cars move 0.24–0.96 units (`observation.py:55-76`; `env.py:155-166`). The agent learned a partial speed heuristic (hence score ~28), so this *caps the ceiling and injects variance* rather than wholly blocking learning.
- **CRITICAL→HIGH** — Railroad: train distance saturates at `_DIST_NORM=12` (visible only ~15 ticks out) and the warning-light state (`light_ringing`, a real ~2.3 s look-ahead) is never encoded (`observation.py:20,46`; `rows.py:364,428`). *Partially confirmed:* `_col_safety` still gives a ~6–15-tick railroad signal, so it is short-horizon, not literally un-observable.
- **HIGH→MEDIUM** — Water logs (safe) share feature slots with cars (deadly), distinguished only by the lane type one-hot; safety is snapshot-only (`observation.py:35-52,62-68`). *Partially confirmed:* the one-hot does provide gating; the real defect is the shared snapshot semantics (same root cause as the TTC finding).

### NEAT
- **CRITICAL** — Single-episode per-genome seeds destroy selection + elitism (same as R1, surfaced in NEAT). [R1]
- **HIGH** — `best_genome` is an un-revalidated single-episode outlier (`neat_algo.py:409-412`). [R3]
- **HIGH→MEDIUM** — Fixed `compat_threshold=3.0`, no adaptive control → speciation collapses to 1 (`neat_algo.py:133`; distance divides excess/disjoint by n≈24 so weight term dominates). *Partially confirmed:* `diversity` is literally `float(num_species)` (`:386`), so "diversity=1.0" just restates species=1; collapse is real but multi-factor, and the fix won't move mean_fitness unless R1 lands first.
- **HIGH (latent)** — Minority species rounded to 0 offspring with no stagnation tracking or min-allocation (`neat_algo.py:276-277`). *Confirmed but inert today* — only 1 species ever forms, so it never fires; becomes load-bearing only after the threshold fix. Note: `self.species` is rebuilt anonymously each gen, so per-species stagnation needs a persistent `Species` object first.

### DQN
- **CRITICAL** — Run was rendered/throttled, not headless (`train.py:32,141-143`). [R4]
- **HIGH** — Epsilon decay horizon (20000) far exceeds data collected (9650) → 54% random actions throughout (`dqn.py:34,45-47`). *Note:* already configurable via `algo_cfg`; the gap is it isn't auto-scaled to `target_episodes`. Fixed automatically once R4 lands.
- **HIGH→MEDIUM** — Unclipped MSE, no Huber, no grad clipping (contrast `ActorCritic` clips to 0.5) (`networks.py:266-278` vs `:234-238`). *Partially confirmed:* Adam *normalizes* per-parameter, so the "unbounded amplification" mechanism is wrong; recorded loss 0.41 is small. Real but secondary — the symptom is undertraining, not instability.
- *(dqn summary)* Vanilla (non-Double) target max → overestimation; time-limit truncation treated as terminal (no bootstrap). Standard improvements, not root causes.

### Policy Gradient (PPO/A2C — never run, all latent)
- **HIGH** — Entropy-bonus gradient has an **inverted sign** (`networks.py:215`): `dlogits += (-ent_coef)*(-dent)/B` = `+ent_coef*dent/B`, which *decreases* entropy → drives the policy toward determinism. Real correctness bug, masked by existing tests because constant advantage dominates the 0.01 entropy term.
- **HIGH→MEDIUM** — `rollout_size=1024` vs few-step episodes → ~10–45 rollouts over the whole budget (`policy_gradient.py:31`). *Partially confirmed:* the buffer accumulates *across* episodes (it's not reset per episode), so "mid-episode flushing" buys nothing and would *break* the GAE bootstrap (`:71` correctly zeroes `next_v` only because the buffer always ends on a terminal step). Fix = lower `rollout_size` + budget in *steps*, not episodes.

### Evolutionary (ES/GA/CMA-ES — only CMA-ES ran)
- **CRITICAL** — Per-candidate different maps destroy ES/GA/CMA selection (same as R1). [R1]
- **CRITICAL** — Champion = luckiest unvalidated episode; ES θ / CMA-ES mean never evaluated (`evolutionary.py:65-67,96-100,163-165`). [R3] *Correction:* GA elites *are* copied forward and re-evaluated, giving weak implicit validation; capture the elite *during* `set_population_fitness` (before `_ask` overwrites the population).
- **HIGH** — CMA-ES under-resourced: ~3909-dim search, pop=40, c1≈1.7e-4/cmu≈1.6e-3 → covariance turns over only ~12.5% across 75 gens; degenerates into slow fixed-isotropic weighted-ES (`evolutionary.py:27-30,220,236-243`). *Correction:* rates scale ~1/n (not n²); the mean *does* still move. Also: `algo_cfg` is never populated by the CLI, so pop/sigma/hidden are stuck at defaults.
- **HIGH** — ES uses no mirrored/antithetic sampling (`evolutionary.py:153-161`); SNR≈√(40/3909)≈0.10. Mirroring is a *no-op* unless each ±eps pair shares a map seed — strictly downstream of R1.

### Infrastructure
- **CRITICAL** — No multi-algo benchmarking harness; one `--algo`, one seed, no multi-seed aggregation (`train.py:240-251`).
- **CRITICAL** — Parallel path saves `best_model` with `best_eval=-1e18`, unvalidated (same as R3).
- **HIGH** — Hyperparameters not configurable from CLI: `algo_cfg` plumbed but never set (`train.py:47-53`); no sweep/HPO code. *Severity → medium for immediate impact* (HPO can't help while R1/R3 corrupt the signal).
- **HIGH** — Degenerate model registry: fixed `checkpoint.pkl`/`best_model.pkl` paths overwritten every run (`trainer.py:200-218`); `runs/neat` has 13 tfevents files but one pkl — proof of clobbering. `AISession._load_policy` (`session.py:58`) hardcodes the path, blocking compare/replay.
- **HIGH** — Metrics CSV header freezes on episode 1 (`metrics.py:105-111`), so `eval_score` (first non-None at ep 50) is **never written to disk**. Verified: `runs/dqn/metrics.csv` has no eval column despite evals at 50…300.

---

## 3. Redesign Plan (code-level, per component)

### 3.1 Evaluation / Noise — **do this first, it gates everything**

**(a) Common Random Numbers + K-seed averaging in `train_generation`.** Replace the per-candidate seed line and the flat evaluate call:

```python
# ai/trainer.py, train_generation()
payloads = self.algo.population_payloads()
pop = len(payloads)
K   = max(3, self.cfg.eval_episodes)                  # was a no-op in parallel; now real
gen = getattr(self.algo, "generation", self.episode // max(1, pop))
base = self.cfg.seed * 100003 + gen * 1009            # SHARED across candidates, rotates per gen
seed_bank = [base + j for j in range(K)]

exp_payloads = [p for p in payloads for _ in seed_bank]
exp_seeds    = [s for _ in payloads for s in seed_bank]
flat = vec_env.parallel_evaluate(exp_payloads, exp_seeds, self.cfg.max_steps, n_workers, pool=pool)

rewards, scores, lengths = [], [], []
for i in range(pop):
    chunk = flat[i*K:(i+1)*K]
    rewards.append(float(np.mean([r for r,_,_ in chunk])))
    scores.append (max  (s for _,s,_ in chunk))
    lengths.append(int  (np.mean([l for _,_,l in chunk])))
metrics = self.algo.set_population_fitness(rewards)
# advance self.episode by pop (one logical episode per candidate), not by pop*K
```

*Notes:* use `gen*1009` (not `gen`) so consecutive generations' banks don't overlap. The same line feeds NEAT and all evolutionary algos, so this one change fixes R1 everywhere. The minimal one-liner stopgap (`seeds = [base]*pop`) restores comparability but overfits one map/gen — prefer K≥3.

**(b) Held-out validation + gated best-model saving** (fixes R3). Add a deterministic eval over a fixed bank *disjoint* from training seeds, run on the existing pool, and gate `save_best`:

```python
VAL_SEEDS = list(range(10_000, 10_008))    # disjoint from seed*100003+episode
# after set_population_fitness, every eval_every:
order = sorted(range(pop), key=lambda i: rewards[i], reverse=True)[:5]   # shortlist by (now-meaningful) train fitness
vp, vs = [], []
for i in order:
    for s in VAL_SEEDS: vp.append(payloads[i]); vs.append(s)
vres = vec_env.parallel_evaluate(vp, vs, self.cfg.max_steps, n_workers, pool=pool)
val_means = [np.mean([vres[j*len(VAL_SEEDS)+t][0] for t in range(len(VAL_SEEDS))]) for j in range(len(order))]
champ = order[int(np.argmax(val_means))]; champ_val = float(max(val_means))
if champ_val > self.best_eval:
    self.best_eval = champ_val
    self.algo.set_validated_champion(payloads[champ])   # new hook; see below
    self.save_best()
self.logger.log_episode(np.mean(rewards), max(scores), int(np.mean(lengths)), metrics, eval_score=champ_val)
```

Add `champion_payload()` / `set_validated_champion()` hooks to each algo, and **remove the running-max update** at `neat_algo.py:410-412` and `evolutionary.py:65-67,96-100`. For ES/CMA-ES the validated champion candidate should include the *distribution centre* (θ / mean), which is usually the best low-variance policy — delete the dead `pass` at `evolutionary.py:163-165`.

### 3.2 Reward — `ai/env.py:step()`

Flip the alive/death signs, shrink the death cliff to ~one row, drop the explicit WAIT penalty (the time cost already taxes idling), and switch idle to a *cell*-based stall detector:

```python
reward  = -0.01                      # uniform time cost: progress or perish (was +0.05)
reward += forward * 1.0              # only positive term
if action == Action.DOWN: reward -= 0.3
# delete the WAIT -=0.02 and the +0.05 survival bonus
...
if not alive: reward -= 1.0          # was -5.0; comparable to one row

# cell-based idle (resets on lateral moves AND log-riding, which drift x):
cell = (round(hero.position.x), math.floor(hero.position.z))
self.idle = 0 if cell != self._prev_cell else self.idle + 1
self._prev_cell = cell
```

Payoffs become: do-nothing → loses; advance-3-then-die → +1.96; advance-8 → +6.9. Bravery strictly dominates at depth ≥2. **Caveat:** with cell-based reset, infinite L/R pacing never trips the cap — this is *only safe because the survival bonus is now net-negative*. Keep `max_idle` (raised to ~120, see 3.3) as a backstop, not the primary loiter deterrent. This is potential-based shaping with Φ(s)=max_z, so the optimal policy is unchanged.

### 3.3 Environment

- `ai/env.py:61` `max_idle: int = 120` (covers the ~34-step worst-case train wait + aiming margin); keep the reset condition as forward>0 only (cell-based reset from 3.2 handles laterals).
- Make WAIT event-driven instead of a fixed 8-tick poll: advance until the player's front lane locally clears or a `cap=36` (one hop ≤ cap < idle starvation):
```python
def _advance_wait(self, cap=36):
    for _ in range(cap):
        self.engine.tick(config.FIXED_DT); self.elapsed += config.FIXED_DT
        if self.on_tick: self.on_tick()
        if not self.engine.hero.is_alive or self._front_lane_clear(): break
```
This turns WAIT from a polling primitive into one "wait for the next opening" decision, collapsing the long WAIT bursts that blow the idle budget.

### 3.4 Observation — `ai/observation.py`

Replace decision-instant safety with **arrival-time prediction** (interval test over the hop dwell window, not endpoint), keep snapshot as an extra channel, and add the missing railroad/water features. Use the true hop horizon T≈12 ticks:

```python
def _proj_x(m, T):
    x = m.mesh.position.x + m.speed * T
    OFF = 11
    if x >  OFF and m.speed > 0: x -= 2*OFF
    if x < -OFF and m.speed < 0: x += 2*OFF
    return x
# col safe iff no mover's projected box covers col_x over t in [6, 18]:
safe = not any(abs(_proj_x(m, t) - col_x) < m.collision_box
               for m in movers for t in (6, 12, 18))
```
Additionally:
- **Railroad warning:** bump `_PER_LANE` 11→13; for railRoad rows write `obs[i+11]=1.0 if entity.light_ringing else 0.0`, `obs[i+12]=entity.ring_count/15.0`. This surfaces the ~2.3 s look-ahead the game already provides. Train distance: use a TRAIN_OFFSET-scaled signed TTC instead of `_DIST_NORM=12`.
- **Water:** apply the same projection in the water branch; add signed riding-drift velocity (when `hero.riding_on`) and projected distance-to-edge `5 - |px + drift*T|` so edge-death is an explicit feature.
- Emit a real TTC per side only when `sign(v)` opposes `sign(dx)`, normalized so near-term threats don't saturate.

*This changes `OBS_SIZE`, invalidating saved nets/genomes — acceptable since everything retrains, but bump net input dims in lockstep.*

### 3.5 NEAT — `ai/algorithms/neat_algo.py`

- **Adaptive compat threshold** (at end of `_reproduce`, after `_speciate` sets `num_species`): `target=8`; `if num_species>target: compat_threshold += 0.3 elif <target: compat_threshold = max(0.5, compat_threshold-0.3)`. Seed it lower (~1.0–1.5).
- **Persistent `Species` object** with stable id, `best_fitness`, `last_improved_gen`; cull species stagnant >15 gens but always keep ≥2.
- **Largest-remainder offspring allocation** with `max(1, ...)` per surviving species and a ~40% cap; drop the `if n_offspring<=0: continue` so elitism always runs.
- Do these **only together with R1** — otherwise you inflate the diversity metric while mean_fitness stays flat. Verify by checking mean_fitness rises, not just species count.

### 3.6 DQN — `ai/algorithms/dqn.py`, `networks.py`

- **Always run `--headless`** (R4). Make headless the default for DQN/PPO/A2C or print a loud warning when launched live.
- **Auto-scale epsilon** to the step budget: `eps_decay_steps = max(2000, min(20000, int(0.4 * target_episodes * 32)))`; switch to exponential decay with a 0.05 floor.
- **Huber + grad clip** in `QNetwork.update`: `d = np.clip(err, -1, 1); dout[arange, actions] = d/B`; add global-norm clip to 10 (mirror `ActorCritic`). Reduce death penalty to ~−1 (covered by 3.2).
- Secondary: Double-DQN target (`a*=argmax online q(next); next_q = target.q(next)[a*]`); bootstrap on time-limit truncation rather than treating it as terminal.

### 3.7 PPO / A2C — `ai/networks.py`, `ai/algorithms/policy_gradient.py`

- **Fix entropy sign** (`networks.py:215`): `dlogits += (-ent_coef) * dent / B`. Add a regression test: with `adv=0, returns=value`, entropy must *increase* over updates.
- **Lower PPO `rollout_size`** 1024→256–512 (match A2C) for 2–4× more updates; keep `minibatch ≤ rollout_size`.
- **Budget in steps, not episodes:** drive training to ~5e5–1e6 env steps. Do **not** flush mid-episode (breaks the GAE bootstrap at `:71`).
- Add reward/advantage normalization and LR annealing.

### 3.8 Evolutionary — `ai/algorithms/evolutionary.py`

- CRN (3.1) is co-primary and fixes the dominant bug for all three.
- **Mirrored ES sampling** (only meaningful with CRN, each ±pair sharing a seed): draw `pop//2` base perturbations, concatenate `[base, -base]`; `_tell` is unchanged (centered ranks + signed rows already give the antithetic estimator).
- **CMA-ES:** shrink the search dim — start `hidden=(16,)` (dim 1957, ~2× rates) or linear `[116,5]` (585, ~6.7×); raise `pop_size` to 64, `sigma` to 0.5. Expose these via `algo_cfg` (see 4.3). Validate the **mean** as champion.
- **GA:** capture the elite during `set_population_fitness`; reduce `init_std` and use less-destructive crossover for the 3909-dim weight vector.

---

## 4. Infrastructure to Build

### 4.1 Evaluation harness — `ai/benchmark.py`
`BenchmarkSpec(algos, seeds, episodes, eval_episodes, fixed_eval_seeds)`. For each (algo, seed): build `TrainConfig(logdir=runs/bench/<algo>/seed<seed>)`, run `train_parallel()` or `train()`, then a **final deterministic eval on a SHARED fixed seed set** (25–50 seeds, identical maps for every algo/seed — generalize `Trainer._evaluate` to take an explicit seed list). Collect `final_eval_mean/std`, `best_eval`, `best_score`, reward-curve AUC, wall-clock, steps/s. Aggregate across seeds → mean ± 95% CI (`1.96*std/√n`). Emit `runs/bench/results.csv` + a printed table. CLI: `python benchmark.py --algos neat,cmaes,ppo,dqn --seeds 0,1,2,3,4 --episodes 3000 --eval-seeds 40`. Reuse `vec_env.make_pool`. **Run after R1/R3**, else it faithfully reports broken training.

### 4.2 Eval/metrics persistence fixes — `ai/metrics.py`, `ai/trainer.py`
- Always write `row["eval_score"]` (empty string when absent) so the frozen CSV header includes it from episode 1; additionally harden `_write_csv` to rewrite the header when a new key appears (buffer rows in memory).
- Wire the parallel path to actually pass `eval_score` into `log_episode` (3.1b).
- Add `best_eval`/`lucky_max_score` to the metrics snapshot and make the dashboard headline read validated `best_eval`, demoting `best_score` to a "peak single-episode (luck)" diagnostic.

### 4.3 HPO — CLI config + `ai/tune.py`
- Add `--set key=val` (repeatable) and `--config file.json` to `train.py`; merge into `TrainConfig.algo_cfg` with type coercion (int→float→bool→str). This single change unblocks all the tuning in §3.
- `ai/tune.py`: no-dep random/grid search over per-algo `SEARCH_SPACE`, scoring each trial by `Trainer._evaluate()` (the only apples-to-apples signal), writing `runs/tune/<algo>/trials.csv`. Optional Optuna backend behind a try-import. **Precondition: R1/R3 fixed** or the tuner ranks noise.

### 4.4 Model registry — `ai/trainer.py`, `ai/session.py`
- Unique run dirs: `runs/<algo>/<timestamp>` + a `latest` pointer (stops the 13× clobber).
- Immutable periodic snapshots `checkpoints/ep<NNNNN>.pkl` (keep last K) + `best/best_ep<NNNNN>.pkl` + `best_history.jsonl` on each validated improvement.
- `runs/index.json` registry (run_id, algo, seed, final_eval, best_score, path).
- Give `AISession._load_policy` an optional `model_path`; add `--compare runA runB ...` evaluating each `best_model.pkl` on shared eval seeds.

### 4.5 Dashboard additions
Per-episode (not UI-rate) diversity/success persisted to CSV; action-distribution and survival/length histograms; an honest `best_eval` progression curve.

---

## 5. One-Session Roadmap — ordered by impact-per-effort

**Do these 6 FIRST (they move the score the most; ~½ day):**

| # | Change | File(s) | Effort | Fixes |
|---|--------|---------|--------|-------|
| 1 | **CRN + K-seed averaging** in `train_generation` | `trainer.py:169` | small | R1 |
| 2 | **Reward sign flip** (alive −0.01, death −1.0, drop survival bonus & WAIT penalty) | `env.py:115-131` | trivial | R2 |
| 3 | **Held-out validation + gated `save_best`** (remove running-max champion) | `trainer.py:156-183`, `neat_algo.py:409-412`, `evolutionary.py:65-67,96-100` | medium | R3 |
| 4 | **Cell-based idle + `max_idle=120`** | `env.py:61,124-137` | small | reward/env |
| 5 | **DQN `--headless` default + epsilon auto-scale** | `train.py:32,248`, `dqn.py:34` | trivial | R4 |
| 6 | **Eval persistence:** write `eval_score` to CSV + pass it through parallel path | `metrics.py:89-111`, `trainer.py:177` | small | observability |

**Then, same session if time (raise the ceiling):**
7. Arrival-time observation + railroad warning-light features (`observation.py`) — medium, biggest *ceiling* unlock after the selection fixes.
8. CLI `--set`/`--config` for `algo_cfg` + CMA-ES `hidden=(16,), pop=64, sigma=0.5` (`train.py`, `evolutionary.py`) — small, unblocks tuning.
9. Entropy sign fix + PPO `rollout_size=256` (`networks.py:215`, `policy_gradient.py:31`) — trivial, makes PPO/A2C viable.
10. `ai/benchmark.py` harness (§4.1) — medium, the measurement layer.

**Defer:** NEAT speciation overhaul (§3.5), model registry (§4.4), HPO sweeps (§4.3) — high value but only after the above prove the signal is clean.

### What to measure (A/B on fixed eval seeds)
Lock a held-out validation set (`seeds 10_000…10_039`). For each change, record **mean ± 95% CI of `final_eval` over 5 training seeds**, plus the **best_eval progression curve** and the **mean-vs-best gap**.

- **Before (current):** mean_fitness ~1.79, best_score ~28 (outlier), gap ~30×, species=1.
- **Predicted gates after #1–#4:** mean_fitness rises off ~1.8 across generations (the primary success signal); `best_eval` (validated) and mean converge to within a small factor; cowardice baseline disappears (passive genomes score negative). If mean_fitness *still* sits at the passive baseline after #1–#3, the CRN wiring or the episode-bookkeeping (advance `self.episode` by `pop`, not `pop*K`) is wrong — check there first.
- **After #5 (DQN):** steps/s 6.9 → ~250; epsilon reaches ~0.05; best_score climbs past 8.
- **After #7:** road/rail crossing success rate (fraction of episodes passing a hazard row) rises; fitness variance drops.

### Honest uncertainty
- The **magnitude** of score gains is uncertain — R1/R2/R3 are high-confidence *mechanism* fixes, but how high mean score climbs depends on representation (§3.4), which is only partially addressed in the first pass.
- The headless DQN throughput (~250 steps/s) is an auditor estimate, not re-measured here; the *direction* (~30–40×) is robust.
- Several findings (PPO/A2C entropy, NEAT speciation, evolutionary mirroring) are **latent** — they affect algorithms that were never run, so they will not change the *currently observed* symptoms; their value is making those algorithms viable for the benchmark.

---

## 6. Refuted (do not act on)
- **"Death penalty −5.0 vs +1/row is a non-monotonic cliff suppressing early risk-taking."** *Refuted.* The −5.0 is a flat constant paid once at death, independent of where you die, so episode fitness is strictly increasing in score — no non-monotonicity. All symptomatic algorithms are **rank-based** (NEAT rebases to positive before fitness sharing at `neat_algo.py:264-265`; CMA-ES/ES/GA sort by rank), so a constant offset is selection-invariant and changes no ranking, weight, or offspring allocation. The death magnitude *can* matter for DQN/PPO value scale (handled in §3.2/3.6), but it provably cannot cause the flat-mean-fitness symptom for the evolutionary algos that actually ran. The true causes are R1/R3, not the death-penalty magnitude.",
    "num_findings": 74,
    "confirmed": 30,
    "refuted": 1,
    "confirmed_findings": [
      {
        "dim": "environment",
        "title": "max_idle=40 terminates episodes mid-crossing; railroads alone can need ~34 consecutive WAITs",
        "severity": "high",
        "verdict": "confirmed",
        "rank": 2,
        "fix": "Reject the 'reset idle on any lateral move (moved_x)' part of the proposed fix: resetting on LEFT/RIGHT lets an agent oscillate sideways forever and never trip the cap, destroying the loop detector — and max_steps=1500 (ai/env.py:60) already bounds total episode length, so the counter's only job is 'no NET forward progress for too long'. The current counter is already exactly 'steps since max_z last increased', which is the correct quantity; it should NOT be reset by lateral oscillation. Concretely: (a) raise the default cap to ~120 in ai/env.py:61 (`max_idle: int = 120`) — this covers the ~34-step worst-case train wait plus aiming margin for a couple of sequential hazards (going to ~150 also covers a worst-case slow-car lane, at the cost of letting genuinely-stuck episodes run longer). (b) Leave the reset condition as-is (reset only on forward>0). (c) Optionally soften the WAIT bias so legitimate waiting isn't double-punished: drop the idle-termination reward from -1.0 toward ~-0.5 (still << the -5.0 death penalty) so timing out a hazard-wait isn't treated almost as harshly as dying. (d) If true infinite-loop detection is still wanted alongside a high cap, add a separate short-horizon position-history/visited-state hash check rather than overloading the forward-progress counter. Net minimal change that captures ~all the benefit: `max_idle=120` and reduce the timeout penalty to -0.5."
      },
      {
        "dim": "environment",
        "title": "WAIT advances only 8 ticks (0.133 s), far shorter than hop or hazard cycles",
        "severity": "high",
        "verdict": "confirmed",
        "rank": 5,
        "fix": "Two coupled changes, not one:

1) Make WAIT event-driven AND uniform-ish in cost. Advance until the player's current lane materially clears (or a cap), instead of a fixed 8-tick poll:

```python
if action == Action.WAIT:
    self._advance_wait()
...
def _advance_wait(self, cap: int = 36):  # 0.6s cap; >= one hop, < idle starvation
    dt = config.FIXED_DT
    px = self.engine.hero.position.x
    pz = round(self.engine.hero.position.z)
    for _ in range(cap):
        tween.update(dt); self.engine.tick(dt); self.elapsed += dt
        if self.on_tick: self.on_tick()
        if not self.engine.hero.is_alive: break
        # stop early once the lane in front (pz+1) is locally clear at px,
        # i.e. nearest mover in that row is far from px in its travel direction
        if self._front_lane_clear(px, pz + 1): break
```
where _front_lane_clear inspects the row at pz+1 and returns True when no car/train collision box overlaps a small window around px (reuse the same lane data observation.build already reads). Keep a hard cap (~36 ticks) so WAIT can never stall indefinitely on water rows (logs always present) or empty grass.

2) Critically, stop WAIT from starving the idle budget. A *productive* wait must not count as idle, or the agent can never sit out an 18 s lane. Either:
   - don't increment self.idle on WAIT when alive and a hazard is still pending (track a separate `self.waits` capped much higher, e.g. equivalent to ~20 s), or
   - raise max_idle and instead terminate on a wall-clock budget (self.elapsed) rather than a non-forward-step count.

If you want the minimal one-liner as a stopgap, set wait_ticks=12 to match a hop (uniform time-per-step) AND raise max_idle so 40 waits exceed the slowest cycle — but the event-driven version is strictly better because it turns WAIT from a polling primitive into a single "wait for the next opening" decision, collapsing the long WAIT bursts that currently both blow the idle cap and explode episode length."
      },
      {
        "dim": "observation",
        "title": "No time-to-collision / arrival-time safety: all hazard & col-safety features are decision-instant snapshots",
        "severity": "critical",
        "verdict": "confirmed",
        "rank": 3,
        "fix": "Replace the binary safe-now landing values with an arrival-time prediction, keeping the snapshot as an extra channel. Use the true hop horizon T_move=12 ticks (WAIT uses 8). For each landing column col_x in a forward lane, and each mover m with center mx and per-tick velocity v=m.speed (signed): compute the closest approach over the dwell window. predicted_safe = 0.0 if there exists t in [t_arrive, t_arrive+dwell] with |mx + v*t - col_x| < m.collision_box, else 1.0, where t_arrive ~= 6 (player becomes resident on the destination row roughly mid-hop, per rows.py rounding) and dwell covers until the next likely decision (use T_move, i.e. window ~[6, 18]); this is an INTERVAL test, not an endpoint test, so a car sweeping through is caught. Also emit a real TTC per side: only when sign(v) opposes sign(dx) (approaching), ttc_ticks = (col_x - mx)/v, else +inf; report clip(ttc_ticks/24, 0, 1) so near-term threats produce a non-saturated gradient (the current _DIST_NORM=12 distance saturates and discards approach direction entirely). Handle wrap at OFFSET=11 by projecting the wrapped copy too (matters mainly for the fast train: 9.6 units/hop). Concretely in _col_safety, for road/railRoad return min over movers of the interval-overlap test instead of the instantaneous overlap; m.speed is already per-tick velocity. Add the fixed hop duration / chosen-action horizon as a scalar so the policy need not infer it. Effort: ~1-2 hrs to rewrite both functions plus a retrain; medium. This raises the achievable ceiling and cuts fitness variance, but fix H1/H2 (per-candidate differing seeds and no re-validation of best) first, since those dominate the flat mean-fitness symptom."
      },
      {
        "dim": "observation",
        "title": "Railroad is effectively un-observable: train distance saturates and the warning-light state is never encoded",
        "severity": "critical",
        "verdict": "partially_confirmed",
        "rank": 5,
        "fix": "Two-part, both in ai/observation.py. (1) Make _lane_hazards wrap-aware per mover instead of a single _DIST_NORM=12: for the train use a TRAIN_OFFSET-scaled signed time-to-collision so approach is visible >15 ticks out. Concretely, since the train always has dir=1/speed>0, compute ttc = (player_x - train_x)/train.speed when train_x < player_x and normalize, e.g. norm = max(abs(speed),1e-6)*K with K~150 ticks; report d = min(max((player_x-mx),0)/(0.8*150), 1.0). This keeps cars on their wrap=11 scale and the train on its 110 scale. (2) Encode the warning state. Per-lane block is currently full (11 = 4 type + 3 col + 4 hazard, observation.py:23-26), so DO add capacity rather than overwrite: bump _PER_LANE to 13, OBS_SIZE recomputes, and in build()'s loop add, for railRoad rows only: obs[i+11] = 1.0 if entity.light_ringing else 0.0; obs[i+12] = entity.ring_count/15.0 (ring_count rises 0->15 as the train approaches, a usable TTC proxy). Leave them 0 for non-railroad lanes. This is an OBS_SIZE change so it invalidates saved genomes/nets — acceptable since everything is being retrained, but note it must be bumped in lockstep with any net-input dims. This lets the agent learn 'wait while ringing, cross right after the train passes (ring just ended / signed TTC just flipped)'. Note this is a secondary fix; land the H1/H2 selection-seed fixes first or this signal still won't be learnable."
      },
      {
        "dim": "observation",
        "title": "Water logs (safe ride surfaces) share the same feature slot as cars (deadly) with no semantic separation; safety is also snapshot-only",
        "severity": "high",
        "verdict": "partially_confirmed",
        "rank": 6,
        "fix": "Replace the decision-instant water check with predicted-on-arrival coverage, reusing the same projection machinery proposed for cars in finding 1. Concretely, in _col_safety (and ideally _lane_hazards) project each mover by the hop horizon before testing overlap:

  T = 12  # ticks the forward hop+settle takes (measure: ~0.2s * 60Hz; or expose from env)
  def _proj_x(m):
      x = m.mesh.position.x + m.speed * T
      # wrap like rows._move (OFFSET=11 for logs/cars)
      if x >  OFFSET and m.speed > 0: x -= 2*OFFSET
      if x < -OFFSET and m.speed < 0: x += 2*OFFSET
      return x
  # water safe if a log's projected box covers col_x:
  return 1.0 if any(_proj_x(m)-m.collision_box < col_x < _proj_x(m)+m.collision_box for m in entity.entities) else 0.0

Apply the identical projection to the road/rail branch so the same code path serves all moving-lane types. Then add two cheap scalar/lane features that ARE currently missing in a usable form:
 (a) signed riding-drift velocity for the current lane (only meaningful when hero.riding_on) so the agent need not infer it from the conflated dz=0 hazard slot; and
 (b) projected signed distance-to-edge = 5 - |px + ride_drift*T| while riding, so drifting off the board (|x|>5 edge death) is an explicit feature rather than something inferred from raw px.
The "give water its own dedicated slot" idea is optional/low-value: the existing per-lane type one-hot already lets the network gate behaviour, so spend the effort on the projection + drift/edge features, not on duplicating slots. Keep T configurable (or measure the actual settle tick count per env config) rather than hard-coding, since DIFFICULTY_SPEED scales mover speed."
      },
      {
        "dim": "reward",
        "title": "Cowardice trap: do-nothing genome out-scores a brave genome that dies — absolute survival bonus + large death cliff invert selection pressure",
        "severity": "critical",
        "verdict": "confirmed",
        "rank": 1,
        "fix": "In ai/env.py step(), flip the alive/death signs and shrink the death cliff to ~one row, and avoid double-penalizing the WAITs that road/water crossings legitimately require. Line 115: reward = -0.01 (time cost — progress or perish; net-negative for loitering). Keep line 116 reward += forward * 1.0. Keep DOWN -=0.3 and the UP&forward==0 -=0.05 bump penalty, but DROP the explicit WAIT -=0.02 (lines 119-120): the -0.01 time cost already taxes idling, and an extra WAIT penalty wrongly punishes the strategic waiting needed to time gaps. Line 131: reward -= 1.0 (comparable to one row, not 5x it). With these constants: do-nothing to the idle cap = 40×(-0.01) − 1.0 ≈ −1.4 (loitering loses); advance-3-then-die = 3×0.99 − 1.01 ≈ +1.96; advance-8-then-die ≈ +6.9 — bravery strictly dominates cowardice at every depth ≥2. Principled equivalent (preferred, leaves optimal policy invariant while adding dense pressure): potential-based shaping with Φ(s)=max_z, giving r' = (max_z_new − max_z_old)·1.0 − time_cost + terminal; the forward term already equals ΔΦ, so only the alive/death signs need correcting. Also note the max_idle=40 terminal (lines 135-136) becomes largely redundant under a per-step time cost and could be removed or its threshold raised so a legit agent waiting out a wide road is not prematurely killed."
      },
      {
        "dim": "reward",
        "title": "Idle counter resets ONLY on a new max row — counts lateral repositioning and log-riding as 'idle' and terminates legitimate water/obstacle navigation",
        "severity": "high",
        "verdict": "confirmed",
        "rank": 3,
        "fix": "Adopt the cell-based idle definition but keep a (raised) cap and fix the survival-bonus interaction. In reset() initialize `self._prev_cell = (round(self.engine.hero.position.x), math.floor(self.engine.hero.position.z))`. In step(), after advancing, replace env.py:124-127 with:\
\
  cell = (round(self.engine.hero.position.x), math.floor(self.engine.hero.position.z))\
  if cell != self._prev_cell:\
      self.idle = 0\
  else:\
      self.idle += 1\
  self._prev_cell = cell\
\
This resets idle on lateral repositioning, on DOWN, and crucially on log-riding (the log drifts the hero's x so round(x) changes each tick). Raise max_idle to ~60-80 (env.py:61) so a full slow-lane car cycle fits.\
\
But do NOT also drop the cap or you must first neutralize the loiter exploit: with reward=0.05 survival (env.py:115) and cell-resetting idle, infinite L/R pacing earns positive reward forever. So keep the termination, OR change the survival bonus to a small NET-NEGATIVE per-step time cost (e.g. reward = -0.01 baseline, forward*1.0 progress) so loitering bleeds score and the cap becomes a backstop rather than the primary loiter deterrent. Keep the -1.0 terminal penalty modest or remove only after the per-step term is net-negative. Also consider not counting WAIT toward idle at all when the front cell is currently occupied by a car/train (i.e. waiting is provably correct), gated on the observation already computed."
      },
      {
        "dim": "reward",
        "title": "Survival reward has the wrong sign and enables loiter-farming (reward hacking)",
        "severity": "high",
        "verdict": "partially_confirmed",
        "rank": 3,
        "fix": "Two-line change in ai/env.py step():

1. Delete the unconditional survival bonus: remove `reward = 0.05` (line 115) and initialise `reward = 0.0`.
2. Add a small uniform time cost so every non-progressing step is strictly negative and progress is the only positive signal:

```python
reward = 0.0
reward += forward * 1.0          # the only positive term
reward -= 0.01                   # uniform time cost (dithering is always net-negative)
if action == Action.DOWN:
    reward -= 0.3
# WAIT and UP-into-obstacle now inherit only the time cost (-0.01), no bonus
```

Effect on the exploit: a WAIT now nets -0.01 (was +0.03), so wait-padding a row costs ~39*0.01=-0.39 instead of paying +1.17 — the agent keeps WAITs only when they enable a future forward hop (their true purpose). A forward hop still nets +0.99. Keep the existing `forward>0 -> idle=0` reset and the death penalty; they now correctly act as the implicit cost of dying.

Caveat the original fix missed: do NOT simply delete 0.05 while leaving `-0.02` for WAIT, because then UP-into-obstacle (currently -0.05) and DOWN (-0.3) stay, but a clean uniform -0.01 floor is better than per-action ad-hoc costs and removes the loophole uniformly. Also retune `max_idle` upward (e.g. 60-80) ONLY after this change, since crossing wide water/rail sections legitimately needs long WAIT chains and the timeout currently double-guards against farming the now-removed bonus."
      },
      {
        "dim": "neat",
        "title": "Single-episode, per-genome random seeds destroy NEAT's selection signal and elitism (the proximate cause of flat mean fitness)",
        "severity": "critical",
        "verdict": "confirmed",
        "rank": 1,
        "fix": "Apply Common Random Numbers: evaluate the WHOLE population on the SAME fixed seed set each generation, average K episodes per genome, and keep the set fixed (or slow-cycle a curriculum) so good genomes/elites stay good. Corrected to match the real APIs (parallel_evaluate takes two aligned lists; Trainer has no .generation; eval_episodes defaults to 1 so K must be raised):

In Trainer.train_generation (replace lines 168-177):
```python
payloads = self.algo.population_payloads()
pop = len(payloads)
K = max(3, getattr(self.algo, "eval_episodes", 1))   # >=3 to average out map luck
base = self.cfg.seed * 100003                          # FIXED across generations
# optional slow curriculum: base += (self.algo.generation // 20) * 1000
seed_set = [base + j for j in range(K)]
exp_payloads = [p for p in payloads for _ in range(K)]   # K copies per genome
exp_seeds    = [s for _ in payloads for s in seed_set]    # same seed_set for every genome
results = vec_env.parallel_evaluate(exp_payloads, exp_seeds, self.cfg.max_steps, n_workers, pool=pool)
rewards, scores, lengths = [], [], []
for i in range(pop):
    chunk = results[i*K:(i+1)*K]
    rewards.append(float(np.mean([r for r, _, _ in chunk])))
    scores.append(max(s for _, s, _ in chunk))
    lengths.append(int(np.mean([l for _, _, l in chunk])))
metrics = self.algo.set_population_fitness(rewards)
for reward, score, length in zip(rewards, scores, lengths):
    self.episode += 1
    self.best_score = max(self.best_score, score)
    self.logger.log_episode(reward, score, length, metrics)
```
Also set eval_episodes>=4 in the NEAT algo_cfg (currently defaults to 1 at neat_algo.py:140). Because the elite is re-inserted into self.population (neat_algo.py:280-281) it is automatically re-evaluated under the fixed seed set, so no separate elite-revalidation code is needed once CRN is in place; optionally guard best_genome promotion (neat_algo.py:410) behind a >=2-seed margin to avoid promoting a one-map fluke. This is purely a within-generation seed-sharing change; it does not touch the env or NEAT operators."
      },
      {
        "dim": "neat",
        "title": "best_genome/best_fitness is an un-revalidated single-episode outlier",
        "severity": "high",
        "verdict": "confirmed",
        "rank": 2,
        "fix": "Revalidate candidates on a FIXED shared seed set before crowning, and make validation genome-parameterized (the existing `_evaluate` only scores `best_genome`, so it cannot be reused directly).

Cleanest place is `train_generation` (trainer.py), since that owns the worker pool:

1. After `rewards = [r for r,_,_ in results]`, pick the top-K indices by training reward (K≈5).
2. Build a validation batch: for each of the K genomes, replicate it across M fixed validation seeds (e.g. `VAL_SEEDS = range(10_000, 10_000+M)`, M≈8) — same set `_evaluate` uses — and run them through the SAME `vec_env.parallel_evaluate` (genomes are already the payload type, so no new worker code needed):
```python
TOPK, VAL = 5, list(range(10_000, 10_000+8))
order = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)[:TOPK]
val_payloads, val_seeds = [], []
for i in order:
    for s in VAL:
        val_payloads.append(("neat", payloads[i][1])); val_seeds.append(s)
vres = vec_env.parallel_evaluate(val_payloads, val_seeds, self.cfg.max_steps, n_workers, pool=pool)
val_means = [np.mean([vres[j*len(VAL)+t][0] for t in range(len(VAL))]) for j in range(len(order))]
champ_local = order[int(np.argmax(val_means))]; champ_val = float(max(val_means))
```
3. Pass the validated champion + score to the algo and update best only from the validated score. Add a method:
```python
def update_validated_best(self, genome, val_fitness):
    self.population_fitness_set(...)  # keep training-fitness reproduction as-is
    if val_fitness > self.best_fitness:
        self.best_fitness = val_fitness
        self.best_genome = genome.clone()
```
and REMOVE the running-max update at neat_algo.py:410-412 (keep reproduction/selection on training fitness — selection within a generation is a separate issue, H1). Report `best_fitness` as the validated value.

Crucially, fix the same single-seed pathology that feeds it: give every candidate in a generation the SAME seed (or same small seed set) at trainer.py:169 so within-generation ranking — and hence the top-K shortlist — is not itself pure luck. Without that, you are still validating a shortlist chosen by noise. Also bump the algo's own `eval_episodes` default (neat_algo.py:140) above 1, or set it in the run config, so the non-parallel path is not single-sample either.

Cost: the revalidation adds K*M≈40 episodes per generation (~one extra small batch), negligible vs pop*generations, and runs on the existing pool."
      },
      {
        "dim": "neat",
        "title": "Fixed compatibility threshold with no adaptive control → speciation collapses to 1 species",
        "severity": "high",
        "verdict": "partially_confirmed",
        "rank": 4,
        "fix": "Add a target-species controller, but place it correctly and pair it with the prerequisite fixes. Insert at the END of `_reproduce` (after `_speciate()` has set `self.num_species`), not inside `_speciate` (which is called before reproduction and would adjust mid-pass):

```python
# end of _reproduce, after self.generation += 1
target = self.cfg.get("target_species", 8)
step = self.cfg.get("compat_step", 0.3)
if self.num_species > target:
    self.compat_threshold += step
elif self.num_species < target:
    self.compat_threshold = max(0.5, self.compat_threshold - step)
```

Refinements over the proposed version: (1) Seed `compat_threshold` lower (e.g. 1.0-1.5) so multiple species form on the first `_reproduce`. (2) Make it config-driven (c1/c2/c3, target_species, compat_step) since none are exposed today. (3) Add per-species stagnation tracking: record each species' best fitness across generations and cull species that haven't improved in N generations (e.g. 15) so a single lineage cannot monopolize all offspring. (4) Reconsider the `n = 1 if n < 20 else n` clamp at line 239 — with genomes near the 20-conn boundary it causes a discontinuity in distance; use the real n or a smaller floor.

CRITICAL precondition: do this only together with deterministic shared evaluation seeds per generation and eval_episodes>=3 (the H1/H2 fixes). Without those, this change will inflate the diversity metric (which is just species count) while mean_fitness stays flat, giving a false sense of progress. Verify by checking that mean_fitness rises across generations, not merely that species count hits the target."
      },
      {
        "dim": "neat",
        "title": "Minority species rounded to zero offspring with no stagnation handling or minimum-allocation → diversity is irreversibly lost",
        "severity": "high",
        "verdict": "confirmed",
        "rank": 7,
        "fix": "Do this together with the adaptive-threshold fix (H3) — alone it is inert because only 1 species ever forms.

1) Introduce a persistent Species class so history survives across generations:
   class Species: id, members, representative, best_fitness=-inf, last_improved_gen, age. Maintain `self.species_registry` keyed by id; in `_speciate`, match each genome to an existing Species' representative (carry id forward) instead of rebuilding anonymous lists, and create new ids only for unmatched genomes.

2) Replace deterministic rounding with largest-remainder (Hare) allocation that guarantees survivors >=1 and caps any lineage:
   raw = [species_adj[k]/total_adj * self.pop_size for k]
   base = [int(math.floor(x)) for x in raw]
   # guarantee 1 to every non-stagnant surviving species
   base = [max(1, b) for b in base]
   # distribute leftover by largest fractional remainder
   leftover = self.pop_size - sum(base)
   order = sorted(range(len(raw)), key=lambda k: raw[k]-math.floor(raw[k]), reverse=True)
   i=0
   while leftover>0: base[order[i%len(order)]]+=1; leftover-=1; i+=1
   while leftover<0: # trim from largest, never below 1
       j=max(range(len(base)), key=lambda k: base[k])
       if base[j]>1: base[j]-=1; leftover+=1
       else: break
   Then cap each at e.g. int(0.4*self.pop_size) and redistribute the overflow the same way. Use base[k] as n_offspring and DROP the `if n_offspring<=0: continue` line (now unreachable for survivors), so elitism always runs.

3) Stagnation: after fitness eval each gen, update Species.best_fitness/last_improved_gen. Before allocation, mark a species stagnant if (gen - last_improved_gen) > 15 and give it 0 offspring — BUT only when len(non_stagnant_species) >= 2 (always keep at least the top 2 species alive regardless of stagnation, NEAT-style). Protect the global best genome from removal.

4) Also raise/adapt compat_threshold (H3) so >1 species actually forms, otherwise none of the above engages."
      },
      {
        "dim": "evaluation",
        "title": "Each candidate in a generation is evaluated on a DIFFERENT seed → fitness ranking is luck, mean fitness cannot rise",
        "severity": "critical",
        "verdict": "confirmed",
        "rank": 1,
        "fix": "Use Common Random Numbers within a generation, with a per-generation K-seed bank averaged per candidate, rotated across generations. Concretely in ai/trainer.py train_generation():

```python
payloads = self.algo.population_payloads()
gen = getattr(self.algo, "generation", self.episode // max(1, len(payloads)))
K = max(1, self.cfg.eval_episodes)
bank = [self.cfg.seed * 100003 + gen * 1000 + j for j in range(K)]  # shared across candidates, rotates per gen
tasks_payloads = [pl for pl in payloads for _ in range(K)]
seeds          = [s   for _  in payloads for s in bank]
flat = vec_env.parallel_evaluate(tasks_payloads, seeds, self.cfg.max_steps, n_workers, pool=pool)
# average the K runs back down to one entry per candidate
rewards, scores, lengths = [], [], []
for i in range(len(payloads)):
    chunk = flat[i*K:(i+1)*K]
    rewards.append(float(np.mean([r for r, _, _ in chunk])))
    scores.append(float(np.mean([s for _, s, _ in chunk])))
    lengths.append(int(np.mean([l for _, _, l in chunk])))
metrics = self.algo.set_population_fitness(rewards)
```

Notes that sharpen the original sketch:
- `gen * 1000` (not `gen` alone) prevents the K-bank of one generation overlapping the next generation's bank when K>1 — the original `gen_seed + j` banks of consecutive generations would collide.
- Keep K>1 (e.g. 3-5): a single shared seed (the one-line `[gen_seed]*pop` variant) fixes comparability but overfits to one map per generation; averaging K shared seeds gives both fairness and variance reduction.
- Also fix the validation half (this is H2): the global best_genome captured at neat_algo.py:409-412 from a single noisy episode must be re-validated. Have train_generation call self._evaluate() on the incumbent best on the fixed eval seeds (trainer.py:135, seeds 10_000+i) and only save_best() when that averaged eval score improves — otherwise the reported "best score" remains an outlier even after CRN."
      },
      {
        "dim": "evaluation",
        "title": "Parallel path never runs deterministic re-evaluation; champion & best model are single-episode outliers",
        "severity": "critical",
        "verdict": "confirmed",
        "rank": 2,
        "fix": "Reuse the existing pool + parallel_evaluate (no new infra needed). The audit's sketch is right in spirit but references methods that don't exist and uses self.algo.generation inconsistently. Concrete version:

1) Add champion_payload() to each parallel algo:
  NEAT (neat_algo.py): `def champion_payload(self): self.best_genome._rebuild_index(); return ("neat", self.best_genome)`
  _Evolution (evolutionary.py): `def champion_payload(self): return ("mlp", (self.best_params.copy(), tuple(self.policy.sizes), self.policy.activation))`

2) Add a fixed held-out seed bank evaluated via the SAME pool:
```python
EVAL_SEED_BASE = 10_000  # disjoint from training seeds (seed*100003+episode)
def _evaluate_parallel(self, pool, n_seeds=None):
    n = n_seeds or self.cfg.eval_episodes
    payload = self.algo.champion_payload()
    from . import vec_env
    seeds = [EVAL_SEED_BASE + i for i in range(n)]
    res = vec_env.parallel_evaluate([payload]*n, seeds, self.cfg.max_steps, pool=pool)
    return float(np.mean([s for _, s, _ in res]))  # mean SCORE (or reward — be consistent with _evaluate)
```

3) In train_generation, REMOVE the unconditional save_best at line 180 and gate on the deterministic eval. Trigger every eval_every episodes (matching the sequential cadence) rather than only on checkpoint cadence:
```python
# after set_population_fitness / logging loop, before on_generation:
gens_done = self.episode // max(1, pop)
if self.cfg.eval_every and (self.episode % self.cfg.eval_every) < pop:
    ev = self._evaluate_parallel(pool)
    self.logger.log_eval(ev)  # surface honest curve
    if ev > self.best_eval:
        self.best_eval = ev
        self.save_best()
if self.cfg.checkpoint_every and self.episode % max(1, self.cfg.checkpoint_every) < pop:
    self.save_checkpoint()  # checkpoint only; no save_best here
```

Notes the audit missed:
- Champion must ALSO be selected by deterministic eval, not just gated for saving: keeping best_genome=argmax(single-episode fits) means the in-memory champion (used by best_act/showcase and pickled into state_dict) is still an outlier even if best_model.pkl's save is gated. Either re-rank the top-k of each generation on the fixed bank and set best_genome to the deterministic winner, or at minimum store the champion that produced the best _evaluate_parallel score.
- Use the persistent `pool` already threaded into train_generation (param at trainer.py:157) so eval forks no new workers.
- Keep the eval seed bank DISJOINT from training seeds. Current training seeds are seed*100003+episode+i; the 10_000-range bank only stays held-out if seed*100003 doesn't collide — with seed=0, episode reaches ~3000 and i<60, so training seeds span 0..~3060, safely disjoint from 10_000+. For nonzero seed, offset the bank to e.g. 2**31-1-i to guarantee no overlap."
      },
      {
        "dim": "evaluation",
        "title": "Reported best_score / best_fitness are max-over-noisy-episodes order statistics, not policy capability",
        "severity": "high",
        "verdict": "confirmed",
        "rank": 2,
        "fix": "The proposed fix is correct in spirit but underspecified: in the parallel path there is currently NO best_eval to surface, so just "adding best_eval_mean to the snapshot" would display None. The real fix has three parts. (1) Actually run honest eval in the population loop. In train_generation(), every `eval_every` generations, evaluate the generation champion with a deterministic multi-seed rollout and record it. Reuse the existing _evaluate machinery but evaluate the genome that wins the eval, not the lucky-max best_genome: e.g. after set_population_fitness, take the top-K candidates by training fitness, re-score each over the fixed eval seeds (10_000+i), pick the argmax mean as the true champion, set self.best_eval = max(self.best_eval, champion_mean), and call save_best() on that champion. Code sketch inside train_generation after `metrics = self.algo.set_population_fitness(rewards)`:
```
if self.cfg.eval_every and (self.episode // pop) % max(1, self.cfg.eval_every // pop) == 0:
    eval_mean = self._evaluate()  # uses algo.best_act -> best_genome over eval_episodes fixed seeds
    self.logger.eval_scores.append(eval_mean)  # or pass eval_score into log_episode
    if eval_mean > self.best_eval:
        self.best_eval = eval_mean
        self.save_best()
```
(2) In metrics.py track and expose it: add `self.best_eval = -1e18`, update it in/around log_episode when an eval_sc
