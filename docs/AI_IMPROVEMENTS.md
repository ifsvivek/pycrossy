# PyCrossy RL — Audit Findings, Fixes & Results

A research-grade overhaul of the reinforcement-learning pipeline. The full adversarial audit is
in [`AI_AUDIT.md`](AI_AUDIT.md); this document records what was wrong, what changed, and the
**measured** before/after over held-out seeds.

## TL;DR

* **The reported "best score ~28–30" was a measurement artifact, not real skill.** It was the
  single luckiest of ~3000 noisy training episodes, never re-validated. Evaluated over 60–100
  *held-out* seeds, every saved model's true mean was only **~3–4 rows**.
* Three independent **measurement/selection bugs** turned evolution into a lottery, and one
  **reward bug** made the lottery's expected winner a coward. None was an "algorithm quality"
  problem.
* After fixing them and upgrading the observation/reward, **NEAT's true mean over 100 held-out
  seeds rose from 3.70 → 5.42 (+46%)**, with the learning mechanism now demonstrably working
  (population mean fitness climbs; species count is healthy). PPO (which had never produced a
  working agent) now reaches **4.76**; DQN improved.
* The remaining performance wall is **water/log crossing** — it accounts for ~47% of deaths.

## Root causes (all code-verified — see AI_AUDIT.md)

| # | Bug | Where | Effect |
|---|-----|-------|--------|
| **R1** | Every population candidate scored on a **different random map** (`seeds = seed*100003 + episode + i`) | `trainer.train_generation` | Selection compared *map luck*, not policy skill → mean fitness never rose |
| **R2** | **+0.05/step survival bonus** (scales with episode length) + a −5 death cliff | `env.step` | A do-nothing genome beat a brave one → "cowardice" was optimal |
| **R3** | Champion = running **max over single-episode noise**; parallel path never re-validated | `trainer`, `neat`, `evolutionary` | Saved model + headline score were lucky outliers, not reproducible |
| **R4** | DQN was run **rendered/real-time-paced** (6.9 steps/s), not headless | `train.py` default | ε never annealed (stuck 0.54); DQN never collected enough data |

Plus: a **sign error** in the PPO/A2C entropy-bonus gradient (collapsed exploration), NEAT
**speciation collapse to 1 species** (fixed threshold) and **minority species rounded to 0
offspring**, a **frozen CSV header** that dropped `eval_score`, no way to pass hyperparameters
from the CLI, and an observation that only encoded **instant** safety (not safety *at the moment
the agent lands* after its ~12-tick hop).

## What changed

**Selection / evaluation (the big lever).** `train_generation` now uses **Common Random Numbers**
— every candidate in a generation is scored on the *same* rotating bank of `K` seeds — and
averages over them, so fitness compares policies, not maps. A separate **held-out validation**
pass (fixed disjoint seed bank) picks and saves the champion, so `best_model.pkl` is a
re-validated policy. The ES/CMA-ES **distribution centre** is validated alongside the samples.

**Reward (`env.py`).** Flipped to a **net-negative time cost** (`-0.01`/step) so progress strictly
beats loitering; death reduced to `-1` (≈ one row); idle budget raised to 150 (the old 40 killed
the waiting that crossing *requires*). Forward reward is potential-based on the furthest row, so
the optimal policy is unchanged.

**Observation (`observation.py`, 116 → 138 dims).** **Arrival-time** column safety: hazards are
projected forward over the hop dwell window, so the agent reasons about whether a gap will still
be there when it lands. Added railroad **warning-light** state, riding-drift velocity, and
distance-to-edge. (Verified sufficient: an obs-only heuristic reaches score 20.)

**Algorithms.** NEAT: **adaptive** compatibility threshold (targets ~10 species) + largest-remainder
offspring (minority species survive). DQN: **Double DQN** + Huber/grad-clipped TD + ε auto-anneal.
PPO/A2C: **entropy-gradient sign fixed** + smaller rollout (more updates). ES: **mirrored** sampling.
CMA-ES: smaller net + larger population + centre validation. Checkpoints are now architecture-aware.

**Infrastructure.** `ai/evaluate.py` (multi-seed statistics: mean/median/best/worst/std/percentiles/
success-rate/survival), `ai/benchmark.py` (compare algorithms under identical conditions with
multi-seed CIs), `--set KEY=VAL` / `--config` CLI overrides (hyperparameters were previously
un-settable), a CSV-header fix, and a `best_history.jsonl` + `runs/index.json` model registry.
Regression tests in `tests/test_ai_fixes.py` pin the reward, entropy-sign, CRN and champion fixes.

## Results — true mean SCORE over 100 held-out seeds (band 9,000,000+)

| Algorithm | Old pipeline (real) | New pipeline | Change |
|-----------|--------------------:|-------------:|:------:|
| **NEAT**  | 3.70 | **5.42** (best 24, 50% reach ≥5) | **+46%** |
| PPO       | — (never produced an agent) | **4.76** | new |
| DQN       | 3.05 | **3.59** | +18% |
| CMA-ES    | 4.03 | 3.98 | flat |

> The old "best score 28–30" does **not** appear here because it never existed as repeatable
> skill — it was the max of ~3000 noisy single-episode draws. The numbers above are the honest,
> reproducible distribution.

A control run with a **much larger NEAT population (250 vs 150)** and more structural mutation
reached the *same* ~5.4 ceiling, which shows the plateau is the **water-crossing wall**, not a
lack of search budget — so the next gains must come from the levers below, not just more compute.

> **Update — the wall is cracked by *planning*, not learning.** A subsequently added
> **Minimax/Expectimax planner** ([`AI_NEW_AGENTS.md`](AI_NEW_AGENTS.md)) that searches a forward
> model of the engine reaches **mean 44 / median 33 / best 203** over 100 held-out seeds (~8× the
> best learned agent) — it solves water crossing by looking several moves ahead instead of reacting.
> So the learners' ceiling is a *reactive-policy + sample-efficiency* limit, exactly as diagnosed.

### Reference points (diagnostics)

* **Engine oracle** (timing-aware, full state access): mean **7.9**, best 27 — the practical ceiling
  for a reactive policy with this env granularity.
* **Obs-only heuristic** (reads just the 138-dim vector): mean **4.4**, best 20 — confirms the
  observation carries enough signal; the gap to the oracle is mostly **water crossing**.
* **Death causes** (obs-heuristic): water 47%, road 23%, railroad 2%, survived 28%.

## The frontier & next steps

Scores plateau around 5–6 because **riding logs across water** is a multi-step plan that a reactive,
short-trained policy rarely discovers (47% of deaths). The highest-value next steps:

1. **Curriculum** — start with road/rail only, phase in water — to let agents master crossings
   incrementally (a `DIFFICULTY`/row-mix knob in `world.py`).
2. **Vectorized + longer PPO** — PPO here trained on only ~30k env steps (it dies in ~5 steps, so
   episodes are tiny); on-policy methods need ~5×10⁵–10⁶ steps. A vectorized env would make that
   tractable.
3. **Recurrence (PPO-LSTM)** — water timing benefits from memory of log phase.
4. **Automated HPO** — `ai/benchmark.py` + `--set` make a random/grid search over the now-clean
   signal straightforward.

## Reproduce

```bash
python train.py --algo neat --headless --episodes 9000 --set pop_size=150 --eval-episodes 4
python -m ai.evaluate runs/neat/best_model.pkl --episodes 100
python -m ai.benchmark --compare runs/neat runs/cmaes runs/ppo --episodes 200
```
