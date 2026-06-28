# PyCrossy — Architecture & Systems

This document maps the codebase and explains how the major systems fit together. For the
exact gameplay spec, see [`../DESIGN.md`](../DESIGN.md).

```
pycrossy/                 the game
  config.py               all gameplay constants (GameSettings) + window/GPU options
  gpu.py                  dedicated-GPU detection/selection + startup logging
  assets.py               asset path registries (models / audio / images / fonts)
  obj_loader.py           Wavefront OBJ -> interleaved (pos,normal,uv) vertex buffers
  textures.py             PNG -> RGBA bytes (palette-safe, headless)
  primitives.py           procedural box/plane geometry (particles)
  mathutils.py            matrices (closed-form TRS), eases, helpers
  scene.py                Object3D scene graph (Vec3/Euler/Mesh/Group)
  tween.py                tween engine (GSAP-style easing semantics)
  models.py               model registry (ModelLoader + Node/* setup)
  particles.py            Feathers / Water / Foam
  renderer.py             moderngl renderer (ortho camera, shadow map, sRGB Lambert,
                          GPU instancing, letterboxed presentation, overlay textures)
  layout.py               resolution-independent presentation layout + display modes
  ui.py                   resolution-independent HUD + menus (score, home, game-over, pause)
  audio.py                AudioManager (pygame.mixer) + NullAudio
  persistence.py          JSON save (high score, character)
  entities/player.py      CrossyPlayer (hop/squash/rotation/idle/hit animations)
  entities/rows.py        Grass / Road / Water / RailRoad rows + spawned movers
  world.py                CrossyGameMap (recycled row pools + generation state machine)
  engine.py               GameEngine (tick loop, movement, collision, camera, scoring, death)
  game.py                 playable app shell (window, states, input, present, hotkeys)

ai/                       the reinforcement-learning framework
  env.py                  gym-like CrossyEnv (reset/step, 5 actions, reward shaping)
  observation.py          fixed 116-float state builder
  base.py                 Algorithm interface + registry (make / register / available)
  networks.py             numpy NN toolkit (MLP, Adam, ActorCritic, QNetwork)
  algorithms/
    neat_algo.py          NEAT (from scratch)
    policy_gradient.py    PPO + A2C
    dqn.py                DQN
    evolutionary.py       ES + GA + CMA-ES
  vec_env.py              subprocess-pool parallel population evaluation
  trainer.py              orchestration (episodes, eval, checkpoints, best-model, replays)
  metrics.py              CSV / JSON / TensorBoard logging + dashboard snapshots
  replay.py               record / play / verify episodes (determinism)
  dashboard.py            live multi-panel dashboard (separate process)
  game_window.py          live AI play window (shared renderer)

main.py                   entry: play          train.py  entry: train an AI
tests/                    pytest suite          docs/     screenshots + this file
```

## Rendering pipeline

The game logic manipulates a lightweight scene graph (`scene.Object3D`). Each frame the
renderer:

1. Recomputes world matrices (`compose` is a closed-form TRS with **static-local caching**
   so the thousands of fixed objects — trees, floors, walls — are composed once).
2. Collects visible drawables and **groups them by geometry/material**, frustum-culling
   parked/off-screen rows.
3. Renders a directional **shadow-map** depth pass, then the main pass — each geometry group
   drawn once with **GPU instancing** (per-instance model matrices), textured sRGB Lambert
   shading with PCF shadows, into an offscreen FBO sized to the gameplay rect.
4. **Presents** the FBO letterboxed into the window (`layout.compute` decides the rect, bg
   and bezel per display mode), then the UI overlay (re-uploaded only when it changes),
   then an optional device bezel — and calls `ctx.finish()` before the buffer swap so the
   cross-GPU presentation copy on hybrid (NVIDIA PRIME) laptops never reads an incomplete
   buffer (eliminates flicker/black frames). Frustum culling (step 2) is view-projection
   based with generous margins and only applies to **compact** objects (trees/cars/…); the
   full-width floor rows are never centre-culled (their centre can leave the screen while
   their body is still visible), so the sky never shows between rows — the GPU clips their
   off-screen parts for free via instancing.

The camera is orthographic, oriented by a fixed `lookAt` and translated to the runtime
position. The **vertical** framing is the invariant (the same rows are always visible at any
resolution); the **horizontal** extent is derived as `half_height × aspect` ("Hor+"), so wide
windows reveal more scenery left/right rather than letterboxing — never distortion.

## Game logic

`engine.GameEngine` runs the core game loop: a fixed 60 Hz tick updates rows,
applies passive movement (riding logs / being carried by a car), eases the camera-follow
world transform (generating a new row per unit advanced), checks fall/edge death, and runs
the swipe movement state machine. `world.CrossyGameMap` owns the recycled row pools and the
generation rules (first 10 grass; then grass / road¾ | railroad¼ / water; obstacle fill;
neighbour coordination for winnable paths). `tween` drives all animations on real `dt`, so
timing is frame-rate independent.

## Display & responsiveness

`layout.py` computes the **gameplay rect** within any window for each `DisplayMode` (Native
fills the whole window — the 3D camera widens its horizontal frustum to match the window aspect
so there are no bars; Mobile centres a phone with a bezel; Stretch letterboxes a fixed phone
aspect with black bars; Dynamic auto-picks by aspect). `WindowMode` toggles windowed /
borderless / fullscreen at runtime (recreating the GL context). `ui.py` sizes every element from
a single `scale` factor (rect height / reference), so the HUD and menus reposition and scale
together — no hardcoded coordinates.

## GPU selection

`gpu.prefer_high_performance_gpu()` runs before any GL context is created: it detects an
NVIDIA GPU via `nvidia-smi` and sets PRIME render-offload env vars (and prefers X11/GLX for
reliable offload). `gpu.create_standalone_context()` enumerates EGL devices for the headless
path. `gpu.log_startup()` prints the resolved device (name / API / driver / VRAM) and warns
with instructions if it ends up on the integrated GPU while a dedicated one exists.

## AI framework

`env.CrossyEnv` wraps a headless engine: a `step(action)` applies a move and advances the
sim until the hop settles (or a few ticks for `WAIT`), returning a 116-float observation,
shaped reward, done flag and info. Every algorithm implements the same episodic interface
(`begin_episode / act / observe / end_episode / best_act / state_dict / progress`), so the
trainer and dashboard drive gradient and evolutionary methods identically, and algorithms
are chosen by name from a registry. Networks are pure numpy with finite-difference-verified
backprop.

## Training pipeline

`trainer.Trainer` runs episodes (live render hook or headless), periodically **evaluates**
the best policy deterministically, **autosaves** the best model, writes **checkpoints**
(resumable), records the **best replay**, and logs every metric. `vec_env.parallel_evaluate`
fans a whole population out across subprocess workers (the right parallelism here, since the
game uses a process-global tween manager). Live mode launches the **dashboard in a separate
process** and the **AI play window** in the main process, exchanging metric snapshots over a
queue.

## Replay system

An episode is fully determined by its reset `seed` (which fixes world generation) plus the
agent's action list, so `replay.Replay` stores only those. `replay.play` re-runs the env to
reproduce the episode (also a determinism check via `replay.verify`). Replays are JSON.

## Save system

`persistence.py` stores the high score and selected character as JSON in `~/.pycrossy/`.
Training artefacts (checkpoints, best models, replays, metrics) live under `runs/<algo>/`.
