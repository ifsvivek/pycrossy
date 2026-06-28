"""Menu system: rendering across views, navigation, settings apply/preview, keybinds.

Pure pygame-surface rendering (the dummy SDL video driver), no GL context required.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from pycrossy import settings as st
from pycrossy.persistence import SaveData


@pytest.fixture(scope="module", autouse=True)
def _pg():
    pygame.init()
    pygame.display.set_mode((1, 1))
    yield
    pygame.quit()


def _menu(tmp_path):
    from pycrossy.menu import MenuSystem
    cfg = st.Config(path=str(tmp_path / "settings.json"))
    launched, applied, previewed = [], [], []
    m = MenuSystem(cfg, SaveData(games_played=3, total_score=30),
                   lambda n, p: launched.append((n, p)),
                   lambda c: applied.append(list(c)),
                   lambda k: previewed.append(k),
                   info_provider=lambda: {"gpu_info": "GPU", "renderer_info": "GL"})
    return m, cfg, launched, applied, previewed


def test_renders_every_screen_in_both_modes(tmp_path):
    from pycrossy.menu import Screen
    m, *_ = _menu(tmp_path)
    for mode in ("desktop", "phone"):
        for w, h, sc in ((1280, 800, 1.0), (480, 820, 0.6)):
            m.resize(w, h, sc, mode)
            for scr in (Screen.MAIN, Screen.SETTINGS, Screen.STATS, Screen.CREDITS):
                m.screen = scr
                m._items_sig = None
                surf = m.render()
                assert surf.get_size() == (w, h)
                assert isinstance(m.to_bytes(), bytes)


def test_navigate_all_settings_tabs_and_rows(tmp_path):
    from pycrossy.menu import Screen
    m, cfg, *_ = _menu(tmp_path)
    m.resize(1280, 800, 1.0, "desktop")
    m._go(Screen.SETTINGS)
    for ti in range(len(st.CATEGORIES)):
        m.tab = ti
        m.region = "list"
        m.list_focus = 0
        m._items_sig = None
        for _ in range(40):
            m.nav_vertical(1)
            m.nav_horizontal(1)
            m.nav_horizontal(-1)
            m.render()
    # never escaped the item bounds
    assert 0 <= m.list_focus < max(1, len(m.items))


def test_apply_commits_and_persists(tmp_path):
    from pycrossy.menu import Screen, FOOTER
    m, cfg, _, applied, _ = _menu(tmp_path)
    m.resize(1280, 800, 1.0, "desktop")
    m._go(Screen.SETTINGS)
    cfg.set("brightness", 130)
    assert cfg.dirty
    m.region = "footer"
    m.footer_focus = [i for i, (f, _) in enumerate(FOOTER) if f == "apply"][0]
    m.activate()
    assert not cfg.dirty
    assert cfg.committed("brightness") == 130
    assert "brightness" in applied[-1]
    # persisted to disk
    assert st.load(cfg.path).get("brightness") == 130


def test_preview_fires_for_live_not_restart(tmp_path):
    from pycrossy.menu import Screen
    m, cfg, _, _, previewed = _menu(tmp_path)
    m.resize(1280, 800, 1.0, "desktop")
    m._go(Screen.SETTINGS)
    # brightness is live -> previews
    m.tab = [i for i, c in enumerate(st.CATEGORIES) if c == "Graphics"][0]
    m._items_sig = None
    m._sync_items()
    bi = next(i for i, it in enumerate(m.items) if it.spec and it.spec.key == "brightness")
    m.list_focus = bi
    m.nav_horizontal(1)
    assert "brightness" in previewed
    # vsync needs restart -> must NOT live-preview
    previewed.clear()
    vi = next(i for i, it in enumerate(m.items) if it.spec and it.spec.key == "vsync")
    m.list_focus = vi
    m.nav_horizontal(1)
    assert "vsync" not in previewed


def test_keybind_capture(tmp_path):
    from pycrossy.menu import Screen
    m, cfg, *_ = _menu(tmp_path)
    m.resize(1280, 800, 1.0, "desktop")
    m._go(Screen.SETTINGS)
    m.tab = [i for i, c in enumerate(st.CATEGORIES) if c == "Controls"][0]
    m._items_sig = None
    m._sync_items()
    ki = next(i for i, it in enumerate(m.items) if it.spec and it.spec.key == "key_up")
    m.list_focus = ki
    m.activate()
    assert m.capturing == "key_up"
    assert m.capture_key("w")
    assert cfg.get("key_up") == "w"
    # escape cancels capture without changing the binding
    m.list_focus = ki
    m.activate()
    assert m.capture_key("escape")
    assert cfg.get("key_up") == "w"


def test_back_cancels_dirty_and_returns_home(tmp_path):
    from pycrossy.menu import Screen
    m, cfg, _, applied, _ = _menu(tmp_path)
    m.resize(1280, 800, 1.0, "desktop")
    m._go(Screen.SETTINGS)
    cfg.set("brightness", 70)
    assert cfg.dirty
    assert m.back()
    assert not cfg.dirty
    assert m.screen == Screen.MAIN
    assert applied[-1] == []          # asked the shell to restore committed values


def test_launch_actions(tmp_path):
    m, cfg, launched, *_ = _menu(tmp_path)
    m.resize(1280, 800, 1.0, "desktop")
    m.open_main()
    m.list_focus = 0
    m.activate()
    assert launched[-1][0] == "play"
    assert "algo" in launched[-1][1]


def test_defaults_footer_resets(tmp_path):
    from pycrossy.menu import Screen
    m, cfg, *_ = _menu(tmp_path)
    m.resize(1280, 800, 1.0, "desktop")
    cfg.set("difficulty", "hard")
    cfg.apply()
    m._go(Screen.SETTINGS)
    m._do_footer("defaults")
    assert cfg.dirty                         # staged, not yet applied
    cfg.apply()
    assert cfg.to_dict() == st.defaults()


def test_mouse_tab_click_changes_category(tmp_path):
    from pycrossy.menu import Screen
    m, cfg, *_ = _menu(tmp_path)
    m.resize(1280, 800, 1.0, "desktop")
    m._go(Screen.SETTINGS)
    m.render()                               # lays out tab rects
    assert m.tab_rects
    m.mouse_down(m.tab_rects[3].center)
    assert m.tab == 3
