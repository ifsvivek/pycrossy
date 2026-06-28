"""Observation builder.

Turns the live engine state into a fixed-length float vector covering everything an
agent needs to plan: player position/state, the upcoming lanes' types, where the safe
landing columns are *at the moment the agent would arrive there* (not just right now),
the nearest hazards with velocity, the railroad warning-light state, and scalar context
(score, time, camera, riding drift, distance-to-edge).

The key design choice (see docs/AI_AUDIT.md §3.4) is **arrival-time prediction**: a hop
settles ~12 ticks after the decision, during which cars/logs/trains move. Column safety is
therefore computed by projecting every mover forward over the hop dwell window and testing
occupancy, so the agent can reason about whether a gap will *still be there* when it lands.

The layout is deterministic and documented so any algorithm (NEAT → deep RL) consumes the
same vector. ``OBS_SIZE`` is the total length.
"""
from __future__ import annotations

from typing import List

import numpy as np

from pycrossy import config
from pycrossy.entities.rows import OFFSET, TRAIN_OFFSET

# Lane offsets relative to the player's current row (−1 behind .. +8 ahead).
LANE_OFFSETS: List[int] = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
_LANDING_COLS = (-1, 0, 1)          # reachable landing columns (left / stay / right)
_DIST_NORM = 12.0                   # normalise hazard distances (car/log wrap at 11)
_TYPE_INDEX = {"grass": 0, "road": 1, "water": 2, "railRoad": 3}

# Hop dwell window (ticks after a decision) over which we test arrival-time safety. A hop
# settles in ~12 ticks; the agent then re-decides, so a cell must stay clear through ~landing.
_CAR_WINDOW = (2, 8, 14, 20)
_TRAIN_WINDOW = (2, 6, 10, 14, 18)
_EDGE = config.EDGE_DEATH_X

# Per-lane: 4 (type one-hot) + 3 (arrival col safety) + 4 (nearest hazard L/R dist+speed)
#           + 2 (railroad warning light + ring progress) = 13
_PER_LANE = 13
_SCALARS = 8
OBS_SIZE = _SCALARS + _PER_LANE * len(LANE_OFFSETS)


def _project_x(x0: float, speed: float, t: int, wrap: float) -> float:
    """Where a mover at ``x0`` moving at ``speed``/tick will be after ``t`` ticks (wrapped)."""
    x = x0 + speed * t
    if speed > 0 and x > wrap:
        x -= 2 * wrap
    elif speed < 0 and x < -wrap:
        x += 2 * wrap
    return x


def _movers_clear(movers, col_x, window, wrap) -> bool:
    """True iff no mover's collision box covers ``col_x`` now or at any projected tick."""
    for m in movers:
        x0 = m.mesh.position.x
        cb = m.collision_box
        for t in (0, *window):
            x = _project_x(x0, m.speed, t, wrap)
            if x - cb < col_x < x + cb:
                return False
    return True


def _lane_hazards(row_type, entity, player_x):
    """Nearest moving hazard to the left and right of ``player_x`` in a lane.

    Returns (left_dist, left_speed, right_dist, right_speed) normalised. Works for cars,
    trains and logs (logs are tracked as moving objects even though they are ridable).
    """
    movers = []
    if row_type == "road":
        movers = entity.cars
    elif row_type == "railRoad":
        movers = [entity.train]
    elif row_type == "water":
        movers = entity.entities
    left_d, left_s, right_d, right_s = 1.0, 0.0, 1.0, 0.0
    for m in movers:
        mx = m.mesh.position.x
        dx = mx - player_x
        d = min(abs(dx) / _DIST_NORM, 1.0)
        spd = float(np.clip(m.speed, -1.0, 1.0))
        if dx <= 0 and d < left_d:
            left_d, left_s = d, spd
        elif dx > 0 and d < right_d:
            right_d, right_s = d, spd
    return left_d, left_s, right_d, right_s


def _arrival_safety(rtype, entity, col_x) -> float:
    """Safety of occupying ``col_x`` on this row *when the agent arrives*.

    1.0 = will be safe, 0.0 = deadly at/around arrival, 0.3 = blocked grass (hop-in-place),
    0.5 = unknown. Roads/rails project hazards forward; water requires a ridable to be under
    the column at arrival; grass checks the static obstacle map.
    """
    if rtype == "grass":
        if abs(col_x) >= _EDGE:
            return 0.0
        return 0.3 if int(col_x) in entity.obstacle_map else 1.0
    if rtype == "road":
        return 1.0 if _movers_clear(entity.cars, col_x, _CAR_WINDOW, OFFSET) else 0.0
    if rtype == "railRoad":
        return 1.0 if _movers_clear([entity.train], col_x, _TRAIN_WINDOW, TRAIN_OFFSET) else 0.0
    if rtype == "water":
        # Safe only if a log/lily pad will cover this column at arrival (~12 ticks). Static
        # lily pads (speed 0) just need current coverage; drifting logs are projected.
        for m in entity.entities:
            x0 = m.mesh.position.x
            cb = m.collision_box
            for t in (0, 8, 12, 16):
                x = _project_x(x0, m.speed, t, OFFSET)
                if x - cb < col_x < x + cb:
                    return 1.0
        return 0.0
    return 0.5


def build(engine, max_z: int, elapsed: float) -> np.ndarray:
    """Construct the observation vector for the current engine state."""
    hero = engine.hero
    gm = engine.game_map
    px = hero.position.x
    pz = int(round(hero.position.z))

    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[0] = np.clip(px / 5.0, -1.0, 1.0)
    obs[1] = 1.0 if hero.moving else 0.0
    obs[2] = 1.0 if hero.riding_on else 0.0
    obs[3] = min(max_z / 100.0, 1.0)
    obs[4] = min(elapsed / 60.0, 1.0)
    obs[5] = np.clip(engine.world.position.x / 3.0, -1.0, 1.0)        # camera lateral offset
    # Riding drift velocity (a log carries the player in x; relevant for edge-death planning).
    drift = hero.riding_on.speed if hero.riding_on else 0.0
    obs[6] = float(np.clip(drift / 0.15, -1.0, 1.0))
    # Headroom before lateral edge-death (1 = centred, 0 = at the killing edge).
    obs[7] = float(np.clip((_EDGE - abs(px)) / _EDGE, 0.0, 1.0))

    i = _SCALARS
    for dz in LANE_OFFSETS:
        row = gm.get_row(pz + dz)
        if row is not None:
            entity = row["entity"]
            rtype = row["type"]
            ti = _TYPE_INDEX.get(rtype)
            if ti is not None:
                obs[i + ti] = 1.0
            for j, cdx in enumerate(_LANDING_COLS):
                obs[i + 4 + j] = _arrival_safety(rtype, entity, round(px) + cdx)
            ld, ls, rd, rs = _lane_hazards(rtype, entity, px)
            obs[i + 7] = ld
            obs[i + 8] = ls
            obs[i + 9] = rd
            obs[i + 10] = rs
            if rtype == "railRoad":
                obs[i + 11] = 1.0 if getattr(entity, "light_ringing", False) else 0.0
                obs[i + 12] = float(getattr(entity, "ring_count", 0)) / 15.0
        else:
            # Unknown/ungenerated lane: neutral safety.
            obs[i + 4:i + 7] = 0.5
        i += _PER_LANE
    return obs
