"""Movement mechanics: direction deltas, rotation targets, blocked hops."""
from __future__ import annotations

import math

from pycrossy import config
from pycrossy.engine import Direction
from tests._helpers import make_engine, settle


def test_move_up_advances_one_row():
    eng = make_engine(0)
    z0 = eng.hero.position.z
    eng.move_with_direction(Direction.UP)
    settle(eng)
    assert round(eng.hero.position.z) == round(z0) + 1
    assert eng.hero.is_alive


def test_move_left_right_change_x():
    eng = make_engine(0)
    x0 = eng.hero.position.x
    eng.move_with_direction(Direction.LEFT)   # left increases x (per source)
    settle(eng)
    assert round(eng.hero.position.x) == round(x0) + 1

    eng2 = make_engine(0)
    x0 = eng2.hero.position.x
    eng2.move_with_direction(Direction.RIGHT)  # right decreases x
    settle(eng2)
    assert round(eng2.hero.position.x) == round(x0) - 1


def test_rotation_targets():
    eng = make_engine(0)
    eng.move_with_direction(Direction.UP)
    assert abs(eng.hero.target_rotation - 0.0) < 1e-6
    eng = make_engine(0)
    eng.move_with_direction(Direction.LEFT)
    assert abs(eng.hero.target_rotation - config.PI_2) < 1e-6
    eng = make_engine(0)
    eng.move_with_direction(Direction.DOWN)
    assert abs(eng.hero.target_rotation - math.pi) < 1e-6


def test_tree_blocks_movement():
    """A grass obstacle ahead makes the hop happen in place (no row advance)."""
    eng = make_engine(0)
    # Force an obstacle directly ahead of the hero.
    pz = int(round(eng.hero.position.z))
    row = eng.game_map.get_row(pz + 1)
    # ensure the target row is grass; if not, find a grass row ahead and teleport hero under it
    if row is None or row["type"] != "grass":
        for dz in range(1, 12):
            r = eng.game_map.get_row(pz + dz)
            if r and r["type"] == "grass":
                eng.hero.position.z = pz + dz - 1
                eng.hero.initial_position = None
                row = r
                break
    grass = row["entity"]
    tx = int(round(eng.hero.position.x))
    grass.obstacle_map[tx] = {"index": 0}     # block the landing column
    z_before = round(eng.hero.position.z)
    eng.move_with_direction(Direction.UP)
    settle(eng)
    assert round(eng.hero.position.z) == z_before   # blocked -> stayed
