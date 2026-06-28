"""Headless helpers shared across tests."""
from __future__ import annotations

import random

from pycrossy import config
from pycrossy.engine import Engine
from pycrossy.tween import tween


def make_engine(seed: int = 0) -> Engine:
    random.seed(seed)
    eng = Engine(audio=None)
    eng.is_game_state_ended = lambda: False
    eng.setup_game("chicken")
    eng.init()
    eng.hero.stop_idle()
    return eng


def settle(engine: Engine, max_ticks: int = 30) -> None:
    dt = config.FIXED_DT
    for _ in range(max_ticks):
        tween.update(dt)
        engine.tick(dt)
        if not engine.hero.is_alive or not engine.hero.moving:
            break
