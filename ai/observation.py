"""Observation builder.

Turns the live engine state into a fixed-length float vector covering everything an
agent needs: player position/state, the upcoming lanes' types, where the safe landing
columns are, and the nearest hazards (cars/trains/logs) with their velocities — plus
scalar context (score, elapsed time, camera offset).

The layout is deterministic and documented so any algorithm (NEAT → deep RL) consumes
the same vector. ``OBS_SIZE`` is the total length.
"""
from __future__ import annotations

from typing import List

import numpy as np

# Lane offsets relative to the player's current row (−1 behind .. +8 ahead).
LANE_OFFSETS: List[int] = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
_LANDING_COLS = (-1, 0, 1)          # reachable landing columns (left / stay / right)
_DIST_NORM = 12.0                   # normalise hazard distances (wrap happens at 11)
_TYPE_INDEX = {"grass": 0, "road": 1, "water": 2, "railRoad": 3}

# Per-lane: 4 (type one-hot) + 3 (col safety) + 4 (nearest hazard L/R dist+speed) = 11
_PER_LANE = 11
_SCALARS = 6
OBS_SIZE = _SCALARS + _PER_LANE * len(LANE_OFFSETS)


def _lane_hazards(row_type, entity, player_x):
    """Nearest moving hazard to the left and right of ``player_x`` in a lane.

    Returns (left_dist, left_speed, right_dist, right_speed) normalised. Works for cars,
    trains and logs (logs are 'hazards' only in the sense of being moving objects to track).
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


def _col_safety(row, entity, col_x) -> float:
    """1 = safe to occupy ``col_x`` now, 0 = deadly, 0.5 = blocked (can't enter)."""
    if row is None:
        return 0.5
    rtype = row["type"]
    if rtype == "grass":
        return 0.3 if int(col_x) in entity.obstacle_map else 1.0
    if rtype == "water":
        # Safe only if a log/lily pad currently covers this column.
        for m in entity.entities:
            cb = m.collision_box
            if m.mesh.position.x - cb < col_x < m.mesh.position.x + cb:
                return 1.0
        return 0.0
    if rtype in ("road", "railRoad"):
        movers = entity.cars if rtype == "road" else [entity.train]
        for m in movers:
            cb = m.collision_box
            if m.mesh.position.x - cb < col_x < m.mesh.position.x + cb:
                return 0.0
        return 1.0
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
    obs[5] = np.clip(engine.world.position.x / 3.0, -1.0, 1.0)  # camera lateral offset

    i = _SCALARS
    for dz in LANE_OFFSETS:
        row = gm.get_row(pz + dz)
        if row is not None:
            entity = row["entity"]
            ti = _TYPE_INDEX.get(row["type"])
            if ti is not None:
                obs[i + ti] = 1.0
            for j, cdx in enumerate(_LANDING_COLS):
                obs[i + 4 + j] = _col_safety(row, entity, round(px) + cdx)
            ld, ls, rd, rs = _lane_hazards(row["type"], entity, px)
            obs[i + 7] = ld
            obs[i + 8] = ls
            obs[i + 9] = rd
            obs[i + 10] = rs
        else:
            # Unknown/ungenerated lane: neutral safety.
            obs[i + 4:i + 7] = 0.5
        i += _PER_LANE
    return obs
