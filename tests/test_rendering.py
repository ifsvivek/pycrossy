"""Rendering regression: the floor rows must tile with no background showing through.

Renders a populated scene with a distinctive clear colour and asserts none of it bleeds
into the play area — guarding against over-aggressive frustum culling that would remove
full-width floor rows and reveal the sky between them. Skipped where no GL context exists.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from pycrossy import config
from pycrossy.engine import Direction, Engine
from pycrossy.tween import tween


@pytest.fixture
def gl_ctx():
    try:
        ctx = moderngl.create_standalone_context()
    except Exception as exc:  # pragma: no cover - depends on host GPU/EGL
        pytest.skip(f"no standalone GL context: {exc}")
    yield ctx
    ctx.release()


def test_no_background_bleed_between_rows(gl_ctx, monkeypatch):
    # Distinctive clear colour so any background showing through is unmistakable.
    monkeypatch.setattr(config, "SCENE_COLOR", (1.0, 0.0, 1.0))
    from pycrossy.renderer import Renderer

    random.seed(11)
    r = Renderer(gl_ctx, 480, 820)
    eng = Engine(audio=None)
    eng.setup_game("chicken")
    eng.init()
    dt = config.FIXED_DT
    for _ in range(120):
        tween.update(dt)
        eng.tick(dt)
    for _ in range(6):
        eng.begin_move_with_direction()
        eng.move_with_direction(Direction.UP)
        for _ in range(14):
            tween.update(dt)
            eng.tick(dt)

    worst = 0
    # Scroll DEEP (past row recycling and into the camera-follow lag regime) — the regime
    # that exposed sky between rows. The whole screen must stay covered every frame.
    for _ in range(120):
        eng.hero.is_alive = True
        if not eng.hero.moving:
            eng.begin_move_with_direction()
            eng.move_with_direction(Direction.UP)
        for _ in range(13):
            tween.update(dt)
            eng.tick(dt)
        eng.hero.is_alive = True
        r.render(eng.scene)
        img = r.read_pixels()
        magenta = (img[:, :, 0] > 200) & (img[:, :, 1] < 80) & (img[:, :, 2] > 200)
        worst = max(worst, int(magenta.sum()))
    assert worst == 0, f"{worst} background pixels bled through the terrain (gap regression)"
