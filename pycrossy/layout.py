"""Resolution-independent presentation layout.

The 3D game is always rendered to an offscreen framebuffer at a **fixed gameplay aspect
ratio**, so the camera framing, movement and collisions never depend on the window size.
This module computes where that fixed-aspect "gameplay rect" sits inside an arbitrary
window for each display mode, plus the letterbox background and whether to draw a device
bezel. All UI is anchored to this rect, so every element scales and repositions together.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from . import config


class DisplayMode(str, Enum):
    NATIVE = "native"       # fill the whole window; the camera widens horizontally to match
    MOBILE = "mobile"       # phone aspect, centred, dark background + device bezel
    STRETCH = "stretch"     # aspect-preserving fill, black letterbox bars
    DYNAMIC = "dynamic"     # auto-pick based on the window's aspect ratio


class WindowMode(str, Enum):
    WINDOWED = "windowed"
    BORDERLESS = "borderless"
    FULLSCREEN = "fullscreen"


# The game's intended aspect ratio (portrait, phone-like). Camera framing is tied to this.
GAME_ASPECT = config.WINDOW_WIDTH / config.WINDOW_HEIGHT
REFERENCE_HEIGHT = config.WINDOW_HEIGHT     # UI scale = rect_height / REFERENCE_HEIGHT

_SCENE_BG = tuple(int(c * 255) for c in config.SCENE_COLOR)
_DARK_BG = (16, 20, 28)
_BLACK_BG = (0, 0, 0)


@dataclass
class Layout:
    x: int
    y: int
    w: int
    h: int
    bg: tuple              # 0..255 letterbox fill
    bezel: bool
    scale: float           # UI scale factor relative to the reference height

    @property
    def rect(self):
        return (self.x, self.y, self.w, self.h)

    @property
    def size(self):
        return (self.w, self.h)


def _contain(win_w: int, win_h: int, aspect: float, fill: float = 1.0):
    """Largest ``aspect`` rect fitting in ``fill`` of the window, centred."""
    avail_w, avail_h = win_w * fill, win_h * fill
    if avail_w / avail_h > aspect:
        h = avail_h
        w = h * aspect
    else:
        w = avail_w
        h = w / aspect
    w, h = int(round(w)), int(round(h))
    x = (win_w - w) // 2
    y = (win_h - h) // 2
    return x, y, w, h


def compute(win_w: int, win_h: int, mode: DisplayMode,
            aspect: float = GAME_ASPECT) -> Layout:
    win_w = max(win_w, 16)
    win_h = max(win_h, 16)

    if mode == DisplayMode.DYNAMIC:
        # Portrait-ish window -> behave like NATIVE (fill); wide desktop -> centred phone.
        mode = DisplayMode.NATIVE if (win_w / win_h) <= aspect * 1.2 else DisplayMode.MOBILE

    if mode == DisplayMode.MOBILE:
        x, y, w, h = _contain(win_w, win_h, aspect, fill=0.94)
        bg, bezel = _DARK_BG, True
    elif mode == DisplayMode.STRETCH:
        x, y, w, h = _contain(win_w, win_h, aspect, fill=1.0)
        bg, bezel = _BLACK_BG, False
    else:  # NATIVE — fill the whole window. The 3D camera widens its horizontal frustum to
        # the window aspect (Hor+), so there are no letterbox bars; wide/desktop windows simply
        # reveal more scenery (the ground slab backs the edges), portrait windows are unchanged.
        x, y, w, h = 0, 0, win_w, win_h
        bg, bezel = _SCENE_BG, False

    return Layout(x, y, w, h, bg, bezel, scale=h / REFERENCE_HEIGHT)
