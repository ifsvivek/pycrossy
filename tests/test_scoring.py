"""Scoring: score = max(floor(z) - startingRow, 0), monotonic on forward progress."""
from __future__ import annotations

import math

from pycrossy import config
from pycrossy.engine import Direction
from tests._helpers import make_engine, settle


def test_start_score_is_zero():
    eng = make_engine(0)
    captured = []
    eng.on_update_score = captured.append
    eng.update_score()
    assert captured[-1] == max(math.floor(config.STARTING_ROW) - config.STARTING_ROW, 0) == 0


def test_forward_increases_score():
    eng = make_engine(0)
    scores = []
    eng.on_update_score = scores.append
    eng.move_with_direction(Direction.UP)
    settle(eng)
    assert eng.hero.is_alive
    assert max(scores) == 1            # floor(9) - 8


def test_backward_does_not_score():
    eng = make_engine(0)
    scores = []
    eng.on_update_score = scores.append
    eng.move_with_direction(Direction.DOWN)   # z -> 7 -> max(7-8,0) = 0
    settle(eng)
    assert all(s == 0 for s in scores)
