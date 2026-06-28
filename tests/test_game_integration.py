"""End-to-end game-shell integration over a real (offscreen) GL context.

The shell needs an OpenGL context, and the SDL video driver is process-global, so this runs
in an isolated subprocess using SDL's ``offscreen`` driver (a real GL context, no window).
It is skipped automatically where no GL/EGL context can be created (e.g. CI without a GPU).
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SCRIPT = r'''
import os
os.environ["SDL_VIDEODRIVER"] = "offscreen"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYCROSSY_GPU"] = os.environ.get("PYCROSSY_GPU", "integrated")
import pygame

from pycrossy.game import Game, State
from pycrossy import settings as st
from pycrossy.menu import Screen, FOOTER

try:
    g = Game(width=1280, height=800, audio_enabled=False)
except Exception as exc:                      # no usable GL/EGL context here
    if any(k in str(exc).lower() for k in ("context", "egl", "glx", "opengl")):
        print("SKIP_NO_GL", exc); raise SystemExit(0)
    raise
assert g.state == State.MENU and g._menu_mode == "desktop"

def frames(n):
    for _ in range(n):
        g._update_logic(); g._present()

frames(20)                                   # attract backdrop
g.renderer.read_pixels()

g._on_launch("play", {}); assert g.state == State.PLAYING
frames(20)
g.engine.game_over(); frames(2)
assert g.state == State.GAME_OVER
assert g.save.games_played >= 1
g.restart(); assert g.state == State.PLAYING
g.toggle_pause(); assert g.state == State.PAUSED
g.toggle_pause(); assert g.state == State.PLAYING
g._enter_menu(); assert g.state == State.MENU

# settings apply: shadows off + brightness, committed to the renderer
g.menu._go(Screen.SETTINGS)
g.cfg.set("shadow_quality", "off"); g.cfg.set("brightness", 130)
g._on_preview("brightness")
assert abs(g.renderer.brightness - 1.3) < 1e-6
g.menu.region = "footer"
g.menu.footer_focus = [i for i, (f, _) in enumerate(FOOTER) if f == "apply"][0]
g.menu.activate()
assert g.renderer.shadows_enabled is False
assert not g.cfg.dirty

# render scale resizes the scene FBO
g.cfg.set("render_scale", 50); g._on_preview("render_scale")
assert g.renderer.width < g.layout.w

# benchmark returns to the menu with a result
g._enter_menu(); g._enter_bench(); g._bench["dur"] = 0.2
import time
t0 = time.time()
while g.state == State.BENCH and time.time() - t0 < 5:
    a = time.perf_counter(); g._update_logic(); g._present()
    g._bench_record((time.perf_counter() - a) * 1000.0)
assert g.state == State.MENU and "Benchmark" in g.menu.toast

pygame.quit()
print("GAME_OK")
'''


def test_game_shell_end_to_end():
    proc = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=_ROOT,
                          capture_output=True, text=True, timeout=180)
    out = proc.stdout + proc.stderr
    if "SKIP_NO_GL" in out:
        pytest.skip("no GL/EGL context available for the game shell")
    assert "GAME_OK" in out, f"integration failed (rc={proc.returncode}):\n{out[-3000:]}"
