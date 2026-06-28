# PyCrossy — Gameplay & Design Spec

PyCrossy is an endless arcade hopper built in Python. It renders low-poly `.obj`/`.png`
voxel models and animates them with a lightweight tween engine. This doc captures the
exact mechanics implemented in `pycrossy/`.

The bundled art, audio, and font assets were originally created for an MIT-licensed
project (© Evan Bacon) and are reused here under that licence; see `assets/ATTRIBUTION.md`
for details.

## Coordinate system
- 3D, Y up. **+Z = forward** (away from spawn, score increases). **X = lateral.**
- One tile = 1 world unit. Rows sit at integer Z. Row models are 25 units wide in X.
- Player spawns at `(0, groundLevel=0.4, startingRow=8)`. Score 0 == row 8.

## Camera
- `OrthographicCamera`, position `(-1, 2.8, -2.9)`, `lookAt(0,0,0)`, near `-30`, far `30`.
- The frustum frames a fixed world width (~9 columns) so framing is
  resolution-independent while keeping the camera angle constant.
- **World-follow** (`forwardScene`, easing `0.03`): `world.z` eases toward `-(hero.z - 8)`
  so the hero stays screen-centered; `world.x` eases toward `clamp(-3, 2, -hero.x)`.
  The camera *settles* when you stop — **there is no eagle / forced scroll**. Deaths are
  only: vehicle/train collision, drowning, or leaving the field (`|x| > 5`, or falling
  more than 1 unit behind the camera — effectively unreachable here).

## Lighting / render
- `DirectionalLight(0xffffff, 1.0)` at `(20, 30, 0.05)`, casts shadows
  (`shadow.camera` left −15 / right 9 / top 6 / bottom −6, far 100, bias ~1e-4,
  map 2048²). Plus `AmbientLight(0xffffff, 1.8)` inside the world group.
- `MeshLambertMaterial`, textures `NearestFilter` + sRGB, `flipY` default.

## Player — animation timing is the *feel*
- `BASE_ANIMATION_TIME = 0.1 s`.
- Hero mesh: `scaleLongestSideToSize(node, 1)` then center X/Z, base on ground.
- **Jump (position):** 2 phases × 0.1 s. Phase 1 → in-air point: 75% of the lateral
  delta, `y = targetY + 0.5`. Phase 2 → exact target. Total 0.2 s arc.
- **Squash/stretch (scale):** 3 phases × 0.1 s: `(1,1.2,1)` → `(1,0.8,1)` →
  `(1,1,1)` ease `Bounce.out`.
- **Rotation:** 0.1 s to target, `Power1.inOut`. Targets: up `0`, down `π`,
  left `π/2`, right `3π/2`. Reset rotation = `(0, π, 0)`.
- **Anticipation** (key-down `runPosieAnimation`): scale → `(1.2, 0.75, 1)` over 0.2 s.
- **Idle** (menu only, `IDLE_DURING_GAME_PLAY=false`): scale.y bobs 1 ↔ 0.8.
- Run-over by car: `y = road.top - 0.05`, scale `(1.7, 0.05, 1.7)`. Side hit: pushed to
  `road.z ± 0.52`, `hitBy = car` (player slides with the car via `moveOnCar`).

## Movement & collision (`moveWithDirection`)
- Left → `x+1`, Right → `x-1`, Up → `z+1`, Down → `z-1` (target rounded to int X;
  when riding, X is floored/ceiled toward the log's travel direction).
- `treeCollision`: if target row is grass and `obstacleMap[x]` exists → blocked → hero
  jumps in place (`moving=false`, target=initial).
- Target Y = next row `entity.top` (default `groundLevel`). Into water: if a log/lily
  pad is ridable at the target X → `top+mid`; else the sunken (drown) Y.
- `playMoveSound` (cycles buck1..12). Score updates on jump complete:
  `score = max(floor(hero.z) - 8, 0)`.

## Rows / world generation
- Each row type (grass/road/water/railroad) is a recycled pool of `maxRows=20`.
- `init`: row 0 generated, then `maxRows+3` `newRow()` calls. First **10 rows forced
  grass**. Obstacle fill by row index: `<5` solid, `<10` empty, else random.
- `newRow` (after row 10): pick from `["grass","roadtype","water"]` uniformly;
  `roadtype` → **¼ railroad**, else road. New rows spawn at the front as the world scrolls
  (one row per world-unit advanced). Generation coordinates with neighbours so there's
  always a winnable path (lily-pad ↔ clear-grass alignment).

### Grass
- Walls: obstacle at every border column (`_x ≤ -5` or `_x ≥ 5`) → tree border.
- Interior obstacles: `count = round(rand·2)+1` (1–3), placed where `_x≠0` and `rand>0.6`.
- Obstacle = 40% boulder / 60% tree (`getRandom`). `obstacleMap[x]` blocks movement.
- `top = 0.4`.

### Road
- `carGen`: `speed = rand·0.06 + 0.02` (0.02–0.08), `numCars = floor(rand·2)+1` (1–2),
  `xDir = ±1`, start `x = -6·xDir`, spacing `-= (rand·3+5)·xDir`. Car y `0.25`,
  `rotation.y = (π/2)·xDir`. Wrap at `|x| > 11`.
- Collision when `round(player.z) == row.z`: `|player.x - car.x| < collisionBox`
  (`= heroWidth/2 + carWidth/2 - 0.1`). Run-over vs side-hit per `moveWithDirection`.
- First lane of a road block uses the striped texture; subsequent lanes blank. `top=0.3`.

### Water
- `isStaticRow = (z % 2 == 0)` → **lily pads** (static, gently rotate); else **driftwood
  logs** (moving). Lily: `numItems = floor(rand·2)+2` (2–3), `top .2 / min .01 / mid .125`.
  Logs: `speed = rand·0.05 + 0.02`, `xDir = ±1`, `top .3 / min -.3 / mid -.1`.
- Standing on a ridable → `ridingOn`, player slides at the log speed (`moveOnEntity`),
  bounce dip animation. On water with nothing ridable & not moving → drown
  (`onCollide(water)` → water particles + death). Foam particles at `x = ±4.5`.

### Railroad
- Train: `speed = 0.8`, `size = rand·2+1` middles, wrap at `|x| > 110`. On each wrap:
  start the warning lights blinking (15 × 200 ms) + train pass sound.
- Collision identical pattern to cars (run-over / side-hit). `top = 0.5`.

## Particles
- **Feathers** (death): 20 white boxes, cubic-bezier burst biased by hit direction.
- **Water** (drown): 15 blue boxes, bounce-out burst.
- **Foam** (river edges): 6 planes, scale-in/out loop drifting outward.

## Audio
- Move: `buck1..12` cycled. Death: `chickendeath{,2}`. Car hit: `carhit`/`carsquish3`.
  Passive (entering road): 50% `car-horn`. Water: `watersplashlow`. Train die: `trainsplat`.
  Game-over banner: `bannerhit3-g` ×3 staggered.

## Game states
`none` (home) → `playing` → `gameOver` → `none`/`playing`; `paused`. Input: swipe or
keys `W/A/S/D` + arrows + space (space/up = up). Tap = up. Key-down triggers the
anticipation squash; key-up commits the move.

## HUD
- Score: `retro` font, 48 px, white, 4 px black outline, top-left.
- `TOP <highscore>`: 14 px, yellow, shown once a highscore exists.

---
Module layout: `config.py` (game settings), `renderer.py` (render stack),
`tween.py` (animation tweens), `entities/` (nodes, rows, player), `world.py`
(world/map generation), `engine.py` (game engine), `audio.py`, `particles.py`,
`ui/` (screens/HUD), `game.py` (app shell).
