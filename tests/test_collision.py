"""Collision behaviour: car kill, water drown, riding logs."""
from __future__ import annotations

from pycrossy import config
from pycrossy.entities.rows import Mover
from tests._helpers import make_engine

DT = config.FIXED_DT


def _first_row(eng, rtype, pred=lambda e: True):
    for _, info in eng.game_map.floor_map.items():
        if info["type"] == rtype and pred(info["entity"]):
            return info["entity"]
    return None


def test_car_collision_kills():
    eng = make_engine(2)
    road = _first_row(eng, "road", lambda e: bool(e.cars))
    assert road is not None
    car = road.cars[0]
    car.speed = 0.0
    car.mesh.position.x = 0.0
    eng.hero.position.set(0.0, road.top, road.position.z)
    eng.hero.moving = False
    eng.hero.initial_position = None
    road.active = True
    road.update(DT, eng.hero)
    assert eng.hero.is_alive is False


def test_water_drown_without_support():
    eng = make_engine(3)
    water = _first_row(eng, "water")
    assert water is not None
    water.entities = []                       # remove all logs/lily pads -> must drown
    eng.hero.position.set(0.0, water.top, water.position.z)
    eng.hero.moving = False
    eng.hero.riding_on = None
    eng.hero.initial_position = None
    water.active = True
    water.update(DT, eng.hero)
    assert eng.hero.is_alive is False


def test_riding_log_carries_player():
    eng = make_engine(4)
    water = _first_row(eng, "water")
    assert water is not None
    mesh = water.reg.log.make_random()
    mesh.position.set(0.0, -0.1, 0.0)
    water.floor.add(mesh)
    mover = Mover(mesh, dir=1, width=1, collision_box=1.0)
    mover.speed, mover.top, mover.min, mover.mid = 0.05, 0.3, -0.3, -0.1
    water.entities = [mover]
    eng.hero.position.set(0.0, water.top, water.position.z)
    eng.hero.moving = False
    eng.hero.riding_on = None
    eng.hero.initial_position = None
    water.active = True
    water.update(DT, eng.hero)
    assert eng.hero.riding_on is mover and eng.hero.is_alive

    x_before = eng.hero.position.x
    eng.hero.move_on_entity()                 # slides with the log
    assert abs(eng.hero.position.x - x_before - mover.speed) < 1e-6
