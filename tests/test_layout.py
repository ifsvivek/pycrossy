"""Responsive layout: aspect-preserving fit across resolutions and display modes."""
from __future__ import annotations

import pytest

from pycrossy.layout import DisplayMode, GAME_ASPECT, compute

# 16:9, 16:10, 21:9, 32:9, 4:3, portrait phone, square
RESOLUTIONS = [(1920, 1080), (1920, 1200), (2560, 1080), (3840, 1080),
               (1280, 1024), (480, 820), (1080, 1920), (1000, 1000)]


@pytest.mark.parametrize("w,h", RESOLUTIONS)
@pytest.mark.parametrize("mode", list(DisplayMode))
def test_rect_fits_within_window(w, h, mode):
    L = compute(w, h, mode)
    assert 0 < L.w <= w
    assert 0 < L.h <= h
    assert L.x >= 0 and L.y >= 0
    assert L.x + L.w <= w and L.y + L.h <= h
    assert L.scale > 0


@pytest.mark.parametrize("w,h", RESOLUTIONS)
@pytest.mark.parametrize("mode", [DisplayMode.MOBILE, DisplayMode.STRETCH])
def test_fixed_aspect_modes_preserve_aspect(w, h, mode):
    # Mobile/Stretch keep the game's portrait aspect (letterboxed); the camera is unchanged.
    L = compute(w, h, mode)
    assert abs((L.w / L.h) - GAME_ASPECT) < 0.02


@pytest.mark.parametrize("w,h", RESOLUTIONS)
def test_native_fills_the_window(w, h):
    # Native fills the whole window (no letterbox); the 3D camera widens horizontally to match.
    L = compute(w, h, DisplayMode.NATIVE)
    assert (L.x, L.y, L.w, L.h) == (0, 0, w, h)


def test_native_uses_scene_background_and_no_bezel():
    L = compute(1920, 1080, DisplayMode.NATIVE)
    assert L.bezel is False


def test_mobile_centers_with_bezel():
    L = compute(1920, 1080, DisplayMode.MOBILE)
    assert L.bezel is True
    assert L.h < 1080            # centred at < full height


def test_dynamic_adapts_to_aspect():
    portrait = compute(1080, 1920, DisplayMode.DYNAMIC)   # tall -> fill like native
    wide = compute(3440, 1440, DisplayMode.DYNAMIC)       # ultrawide -> centred phone
    assert portrait.bezel is False
    assert wide.bezel is True
