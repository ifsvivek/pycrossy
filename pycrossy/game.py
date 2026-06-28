"""Playable app shell + front-end.

Opens a pygame OpenGL window on the dedicated GPU when available, drives the :class:`Engine`
at a fixed 60 Hz logic step while rendering at up to the configured FPS, and presents the
3D game inside any window size (Native fills the window; Mobile/Stretch letterbox a phone aspect).

On top of the game it hosts the full front-end: a main menu, a tabbed settings system (with
live-applied, persisted configuration), statistics, credits, and several launch modes —
normal play, AI auto-play / replay (in-process), AI training (spawned dual-window), a
benchmark, and a self-running demo/attract backdrop. Settings drive the renderer, engine,
audio, window and input live; see :mod:`pycrossy.settings` for the schema.

Run with ``python main.py`` (or ``python -m pycrossy.game``).

Hotkeys: F11 fullscreen · F10 borderless · F9 cycle view · Esc back/pause.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from enum import Enum
from typing import Optional, Tuple

import moderngl
import pygame

from . import config, gpu, persistence, settings as st
from .audio import AudioManager
from .engine import Engine, Direction
from .layout import DisplayMode, WindowMode, compute as compute_layout
from .menu import MenuSystem
from .renderer import Renderer
from .tween import tween
from .ui import UI

_DIFFICULTY_SPEED = {"easy": 0.8, "normal": 1.0, "hard": 1.3}
_VIEW_CYCLE = ["auto", "desktop", "phone"]


class State(str, Enum):
    MENU = "menu"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "gameOver"
    WATCH = "watch"          # AI auto-play / replay spectator
    BENCH = "bench"          # benchmark


class Game:
    def __init__(self, width: Optional[int] = None, height: Optional[int] = None,
                 audio_enabled: bool = True, vsync: Optional[bool] = None,
                 display_mode: Optional[DisplayMode] = None):
        self.cfg = st.load()
        # GPU preference + native video backend must be chosen before pygame.init().
        config.PREFER_DEDICATED_GPU = bool(self.cfg.get("prefer_dedicated_gpu"))
        self._gpu_request = gpu.prefer_high_performance_gpu(config.PREFER_DEDICATED_GPU)
        pygame.init()
        try:
            pygame.joystick.init()
            self._joys = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
            for j in self._joys:
                j.init()
        except Exception:
            self._joys = []

        self._forced_display = display_mode  # CLI override of the view, else None
        self.display_mode = DisplayMode.NATIVE
        self._sync_derived()                 # needs pygame for key-name resolution
        if vsync is not None:
            self.vsync = vsync

        w0, h0 = self._initial_size(width, height)
        self._windowed_size: Tuple[int, int] = (w0, h0)
        self.win_w, self.win_h = w0, h0
        self.layout = None
        self._bezel_bytes: Optional[bytes] = None
        self._bezel_dirty = False
        self._menu_area = pygame.Rect(0, 0, w0, h0)
        self._menu_mode = "desktop"

        self._create_window(w0, h0)
        gpu.log_startup(self.ctx, self._gpu_request)
        self._gpu_info = gpu.gl_info(self.ctx)

        self.save = persistence.load()
        self.audio = AudioManager(enabled=audio_enabled)
        self.character = self.save.character

        self.engine = Engine(audio=self.audio)
        self._wire_engine()
        self.engine.setup_game(self.character)
        self.engine.init()

        self.menu = MenuSystem(self.cfg, self.save, self._on_launch, self._on_apply,
                               self._on_preview, info_provider=self._menu_info)
        if getattr(self, "_pending_menu_geom", None) is not None:
            self.menu.resize(*self._pending_menu_geom)
        self.apply_settings(None)            # full apply to the freshly built systems

        self.state = State.MENU
        self.menu.open_main()
        self.score = 0
        self.new_best = False
        self._game_seconds = 0.0
        self._mouse_down: Optional[tuple] = None
        self._banner_timer = 0.0
        self.session = None                  # AISession when in WATCH
        self._bench = None
        self._attract_wait = 0
        self._attract_dead = 0.0
        self._setup_attract()

        self.clock = pygame.time.Clock()
        self._accum = 0.0
        self.running = True
        self._visible = True
        self._focused = True
        self._ui_sig = None
        self._frame = 0
        self._fps = 0.0

    # -- derive runtime fields from config --------------------------------
    def _sync_derived(self) -> None:
        c = self.cfg
        self.vsync = bool(c.get("vsync"))
        self.window_mode = WindowMode(c.get("window_mode"))
        self.render_scale = c.get("render_scale") / 100.0
        self.ui_scale = c.get("ui_scale") / 100.0
        self.fps_limit = int(c.get("fps_limit"))
        self.show_hints = bool(c.get("show_hints"))
        self.show_perf = bool(c.get("show_fps")) or bool(c.get("perf_overlay"))
        self.perf_detail = bool(c.get("perf_overlay"))
        self.debug = bool(c.get("debug_mode"))
        self._rebuild_keymap()

    @staticmethod
    def _easing(smoothness: int) -> float:
        # smoothness 1 (snappy) .. 10 (very smooth); 7 ≈ a 0.03 lerp.
        return max(0.012, min(0.06, 0.066 - 0.0052 * float(smoothness)))

    def _keycode(self, name: str) -> Optional[int]:
        try:
            return pygame.key.key_code(name)
        except (ValueError, TypeError):
            return None

    def _rebuild_keymap(self) -> None:
        base = {pygame.K_w: Direction.UP, pygame.K_SPACE: Direction.UP,
                pygame.K_s: Direction.DOWN, pygame.K_a: Direction.LEFT,
                pygame.K_d: Direction.RIGHT, pygame.K_UP: Direction.UP,
                pygame.K_DOWN: Direction.DOWN, pygame.K_LEFT: Direction.LEFT,
                pygame.K_RIGHT: Direction.RIGHT}
        for key, direction in (("key_up", Direction.UP), ("key_down", Direction.DOWN),
                               ("key_left", Direction.LEFT), ("key_right", Direction.RIGHT)):
            kc = self._keycode(self.cfg.get(key))
            if kc is not None:
                base[kc] = direction
        self.key_dir = base
        self.pause_key = self._keycode(self.cfg.get("key_pause")) or pygame.K_p

    def _initial_size(self, width, height) -> Tuple[int, int]:
        if width and height:
            return width, height
        res = self.cfg.get("resolution")
        if isinstance(res, str) and "x" in res:
            try:
                w, h = (int(v) for v in res.split("x"))
                return w, h
            except ValueError:
                pass
        return config.WINDOW_WIDTH, config.WINDOW_HEIGHT

    # -- window / GL lifecycle --------------------------------------------
    def _create_window(self, width: int, height: int) -> None:
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if self.window_mode == WindowMode.WINDOWED:
            flags |= pygame.RESIZABLE
            size = (width, height)
        elif self.window_mode == WindowMode.BORDERLESS:
            flags |= pygame.NOFRAME
            size = pygame.display.get_desktop_sizes()[0]
        else:  # FULLSCREEN
            flags |= pygame.FULLSCREEN
            size = pygame.display.get_desktop_sizes()[0]
        gpu.safe_set_mode(size, flags, vsync=self.vsync)
        pygame.display.set_caption(config.WINDOW_TITLE)
        self.win_w, self.win_h = pygame.display.get_window_size()
        self.ctx = moderngl.create_context()
        self._build_gl()
        # Re-apply brightness/shadow/zoom/etc. to the freshly built renderer (skipped during
        # the very first construction, before audio/engine exist — apply_settings does it then).
        if getattr(self, "engine", None) is not None and getattr(self, "audio", None) is not None:
            self._apply_av(lambda *names: True)

    def _fbo_size(self, rect) -> Tuple[int, int]:
        return (max(16, int(rect.w * self.render_scale)),
                max(16, int(rect.h * self.render_scale)))

    def _resolve_view(self) -> Tuple[DisplayMode, str]:
        """Map the View setting (+ window orientation for Auto) to a concrete game display
        mode and a menu layout. Desktop fills the window; Phone uses the portrait device
        frame; Auto picks by window orientation so wide windows get the desktop front-end."""
        if self._forced_display is not None:
            dm = self._forced_display
            return dm, ("phone" if dm == DisplayMode.MOBILE else "desktop")
        view = self.cfg.get("view_mode")
        if view == "desktop":
            return DisplayMode.NATIVE, "desktop"
        if view == "phone":
            return DisplayMode.MOBILE, "phone"
        # auto: landscape window -> desktop front-end, portrait -> phone
        if self.win_w >= self.win_h:
            return DisplayMode.NATIVE, "desktop"
        return DisplayMode.MOBILE, "phone"

    def _build_gl(self) -> None:
        self.display_mode, self._menu_mode = self._resolve_view()
        self.layout = compute_layout(self.win_w, self.win_h, self.display_mode)
        fw, fh = self._fbo_size(self.layout)
        self.renderer = Renderer(self.ctx, fw, fh)
        self._build_ui()
        self._build_bezel()
        self._ui_sig = None

    def _build_ui(self) -> None:
        r = self.layout
        self.ui = UI(max(r.w, 16), max(r.h, 16), self.ui_scale)
        # Menu geometry: full-window on desktop, the device rect on phone.
        if self._menu_mode == "phone":
            area = pygame.Rect(r.x, r.y, r.w, r.h)
            mscale = r.scale * self.ui_scale
        else:                                        # full-window desktop menu
            area = pygame.Rect(0, 0, self.win_w, self.win_h)
            mscale = (self.win_h / 820.0) * self.ui_scale
        self._menu_area = area
        mscale = max(0.55, min(2.2, mscale))
        if getattr(self, "menu", None) is not None:
            self.menu.resize(area.w, area.h, mscale, self._menu_mode)
        else:
            self._pending_menu_geom = (area.w, area.h, mscale, self._menu_mode)

    def _relayout(self) -> None:
        self.display_mode, self._menu_mode = self._resolve_view()
        new = compute_layout(self.win_w, self.win_h, self.display_mode)
        fw, fh = self._fbo_size(new)
        self.layout = new
        self.renderer.resize(fw, fh)
        self._build_ui()
        self._build_bezel()
        self._ui_sig = None

    def _build_bezel(self) -> None:
        if not self.layout.bezel:
            self._bezel_bytes = None
            return
        surf = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)
        r = self.layout
        pad = max(4, int(r.h * 0.012))
        frame = pygame.Rect(r.x - pad, r.y - pad, r.w + 2 * pad, r.h + 2 * pad)
        radius = max(8, int(r.h * 0.04))
        pygame.draw.rect(surf, (40, 44, 56, 255), frame, border_radius=radius)
        pygame.draw.rect(surf, (70, 76, 92, 255), frame, width=max(2, pad // 2),
                         border_radius=radius)
        pygame.draw.rect(surf, (0, 0, 0, 0), (r.x, r.y, r.w, r.h))
        self._bezel_bytes = pygame.image.tostring(surf, "RGBA", True)
        self._bezel_dirty = True

    def set_window_mode(self, mode: WindowMode) -> None:
        if mode == self.window_mode:
            return
        if self.window_mode == WindowMode.WINDOWED:
            self._windowed_size = (self.win_w, self.win_h)
        self.window_mode = mode
        w, h = self._windowed_size
        self._create_window(w, h)            # rebuilds GL + re-applies A/V to the fresh renderer

    def _recreate_window(self) -> None:
        w, h = (self._windowed_size if self.window_mode == WindowMode.WINDOWED
                else pygame.display.get_desktop_sizes()[0])
        self._create_window(w, h)

    def _apply_resolution(self) -> None:
        if self.window_mode != WindowMode.WINDOWED:
            return
        res = self.cfg.get("resolution")
        if not isinstance(res, str) or "x" not in res:
            return
        try:
            w, h = (int(v) for v in res.split("x"))
        except ValueError:
            return
        self._windowed_size = (w, h)
        self._create_window(w, h)

    # -- settings application ---------------------------------------------
    def _apply_av(self, need) -> None:
        """Push audio / renderer / engine knobs into the live systems (no layout/window)."""
        g = self.cfg.get
        if need("master_volume", "sfx_volume", "ui_volume", "mute"):
            self.audio.set_levels(master=g("master_volume") / 100.0,
                                  sfx=g("sfx_volume") / 100.0,
                                  ui=g("ui_volume") / 100.0, mute=g("mute"))
        if need("brightness"):
            self.renderer.set_brightness(g("brightness") / 100.0)
        if need("shadow_quality"):
            self.renderer.set_shadow_quality(g("shadow_quality"))
        if need("camera_zoom"):
            self.renderer.set_camera_zoom(g("camera_zoom") / 100.0)
        if need("camera_smoothness"):
            self.engine.camera_easing = self._easing(g("camera_smoothness"))
        if need("screen_shake"):
            self.engine.screen_shake = bool(g("screen_shake"))
        if need("difficulty"):
            config.DIFFICULTY_SPEED = _DIFFICULTY_SPEED.get(g("difficulty"), 1.0)
        if need("logging_level"):
            self._apply_logging(g("logging_level"))

    def apply_settings(self, keys=None) -> None:
        """Push config values into the live game. ``keys=None``/empty re-applies everything."""
        self._sync_derived()
        ks = set(keys) if keys else None

        def need(*names) -> bool:
            return ks is None or any(n in ks for n in names)

        self._apply_av(need)
        # Layout-affecting settings need a relayout / renderer resize.
        if need("render_scale", "ui_scale", "view_mode"):
            self._relayout()
        # Window-recreating settings (only reached via an explicit Apply commit). _create_window
        # rebuilds the GL + UI/menu and re-applies A/V, so no extra relayout is needed here.
        g = self.cfg.get
        if ks is not None:
            if "window_mode" in ks:
                self.set_window_mode(WindowMode(g("window_mode")))
            if "resolution" in ks:
                self._apply_resolution()
            if "vsync" in ks:
                self._recreate_window()

    def _apply_logging(self, level: str) -> None:
        import logging
        logging.getLogger().setLevel(
            {"error": logging.ERROR, "warn": logging.WARNING,
             "info": logging.INFO, "debug": logging.DEBUG}.get(level, logging.INFO))

    def _menu_info(self) -> dict:
        r = self._gpu_info
        return {"gpu_info": r.get("renderer", "—"),
                "renderer_info": f"{r.get('vendor', '')} · {r.get('version', '')}".strip(" ·")}

    # -- menu callbacks ----------------------------------------------------
    def _on_apply(self, changed) -> None:
        self.apply_settings(changed)

    def _on_preview(self, key) -> None:
        self.apply_settings([key])

    def _on_launch(self, action: str, payload: dict) -> None:
        if action == "play":
            self._enter_play()
        elif action == "quit":
            self.running = False
        elif action == "benchmark":
            self._enter_bench()
        elif action in ("ai_auto", "replay", "ai_swarm"):
            self._enter_watch(action, payload)
        elif action == "ai_train":
            self._launch_training(payload)

    # -- engine callbacks --------------------------------------------------
    def _wire_engine(self) -> None:
        def on_update_score(pos: int) -> None:
            if self.score < pos:
                self.score = pos

        def on_game_ended() -> None:
            if self.state != State.PLAYING:        # ignore attract/menu deaths
                return
            self.state = State.GAME_OVER
            self.ui.on_game_over()
            self._banner_timer = 0.0
            self.new_best = self.save.record_game(self.score, self.character, self._game_seconds)
            persistence.save(self.save)

        self.engine.on_update_score = on_update_score
        self.engine.on_game_init = lambda: setattr(self, "score", 0)
        self.engine.on_game_ready = lambda: None
        self.engine.on_game_ended = on_game_ended
        self.engine.is_game_state_ended = lambda: self.state != State.PLAYING

    # -- state transitions -------------------------------------------------
    def _enter_play(self) -> None:
        config.DIFFICULTY_SPEED = _DIFFICULTY_SPEED.get(self.cfg.get("difficulty"), 1.0)
        self.engine.setup_game(self.character)
        self.engine.init()
        self.engine.camera_easing = self._easing(self.cfg.get("camera_smoothness"))
        self.engine.screen_shake = bool(self.cfg.get("screen_shake"))
        self.score = 0
        self._game_seconds = 0.0
        self.state = State.PLAYING
        self.engine.move_with_direction(Direction.UP)

    def restart(self) -> None:
        self._enter_play()

    def _enter_menu(self) -> None:
        self.session = None
        self._bench = None
        self.state = State.MENU
        self.menu.open_main()
        self._setup_attract()

    def toggle_pause(self) -> None:
        if self.state == State.PLAYING:
            self.state = State.PAUSED
        elif self.state == State.PAUSED:
            self.state = State.PLAYING

    def _enter_watch(self, action: str, payload: dict) -> None:
        try:
            if action == "ai_swarm":
                from ai.swarm import SwarmSession
                self.session = SwarmSession(algo=payload.get("algo", "neat"),
                                            seed=payload.get("seed", 0),
                                            size=int(payload.get("swarm", 12)),
                                            character=self.character)
            else:
                from ai.session import AISession
                self.session = AISession(mode="replay" if action == "replay" else "auto",
                                         algo=payload.get("algo", "neat"),
                                         seed=payload.get("seed", 0),
                                         algo_cfg={"max_depth": int(payload.get("depth", 6))})
        except Exception as exc:
            self.menu.notify(f"AI unavailable: {exc}")
            self.session = None
            return
        if not self.session.available:
            self.menu.notify("No replay saved for this algorithm yet")
            self.session = None
            return
        config.DIFFICULTY_SPEED = 1.0
        self.state = State.WATCH

    def _enter_bench(self) -> None:
        config.DIFFICULTY_SPEED = 1.0
        self.engine.setup_game(self.character)
        self.engine.init()
        self.engine.screen_shake = False
        self.state = State.BENCH
        self._bench = {"frames": 0, "sum_ms": 0.0, "worst_ms": 0.0, "dur": 6.0,
                       "start": None}

    def _launch_training(self, payload: dict) -> None:
        # Resolve train.py relative to this package (project root), not the CWD, so launching
        # works regardless of where the game was started from. Run it there too, so its
        # relative paths (runs/, assets) resolve correctly.
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_py = os.path.join(project_root, "train.py")
        if not os.path.isfile(train_py):
            self.menu.notify("train.py not found next to the game")
            return
        algo = str(payload.get("algo", "neat"))
        args = [sys.executable, train_py,
                "--algo", algo,
                "--speed", str(payload.get("speed", 3)),
                "--seed", str(payload.get("seed", 0))]
        if int(payload.get("parallel", 0)) > 0:
            args += ["--parallel", str(int(payload["parallel"]))]
        if algo == "minimax":                       # planner: thread the in-game search depth
            args += ["--set", f"max_depth={int(payload.get('depth', 6))}"]
        try:
            subprocess.Popen(args, cwd=project_root, env=dict(os.environ))
            self.menu.notify(f"Launched {payload.get('algo', 'neat').upper()} training "
                             "(new windows)")
        except Exception as exc:
            self.menu.notify(f"Could not launch training: {exc}")

    # -- attract / demo backdrop ------------------------------------------
    def _setup_attract(self) -> None:
        config.DIFFICULTY_SPEED = 1.0
        self.engine.setup_game(self.character)
        self.engine.init()
        self.engine.screen_shake = False
        self.engine.hero.stop_idle()
        self._attract_wait = 0
        self._attract_dead = 0.0

    def _attract_step(self, dt: float) -> None:
        import random
        eng = self.engine
        hero = eng.hero
        if not hero.is_alive:
            self._attract_dead += dt
            if self._attract_dead > 1.2:
                self._setup_attract()
            return
        if hero.moving:
            return
        if self._attract_wait > 0:
            self._attract_wait -= 1
            return
        ahead = eng.game_map.get_row(hero.position.z + 1) or {}
        r = random.random()
        if ahead.get("type") in ("road", "water"):
            if r < 0.45:
                self._attract_wait = 7
                return
            d = Direction.LEFT if r < 0.6 else (Direction.RIGHT if r < 0.72 else Direction.UP)
        else:
            d = Direction.UP if r > 0.12 else (Direction.LEFT if r < 0.06 else Direction.RIGHT)
        eng.begin_move_with_direction()
        eng.move_with_direction(d)

    # -- input -------------------------------------------------------------
    def _menu_local(self, pos) -> Tuple[int, int]:
        a = self._menu_area
        return (pos[0] - a.x, pos[1] - a.y)

    def _process_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)
            elif event.type == pygame.KEYUP:
                self._handle_keyup(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._handle_mouse_up(event.pos)
            elif event.type == pygame.MOUSEMOTION:
                if self.state == State.MENU:
                    self.menu.mouse_move(self._menu_local(event.pos))
            elif event.type == pygame.MOUSEWHEEL:
                if self.state == State.MENU:
                    self.menu.wheel(event.y)
            elif event.type == pygame.VIDEORESIZE and self.window_mode == WindowMode.WINDOWED:
                self.win_w, self.win_h = max(event.w, 64), max(event.h, 64)
                self._relayout()
            elif event.type in (pygame.JOYBUTTONDOWN, pygame.JOYHATMOTION, pygame.JOYAXISMOTION):
                self._handle_joy(event)
            elif event.type in (pygame.WINDOWMINIMIZED, pygame.WINDOWHIDDEN):
                self._visible = False
            elif event.type == pygame.WINDOWFOCUSLOST:
                self._focused = False
                if self.cfg.get("auto_pause") and self.state == State.PLAYING:
                    self.state = State.PAUSED
            elif event.type == pygame.WINDOWFOCUSGAINED:
                self._focused = True
            elif event.type in (pygame.WINDOWRESTORED, pygame.WINDOWSHOWN, pygame.WINDOWEXPOSED):
                self._visible = True

    def _handle_keydown(self, event) -> None:
        key = event.key
        # global display hotkeys
        if key == pygame.K_F11:
            self._toggle_window_mode(WindowMode.FULLSCREEN)
            return
        if key == pygame.K_F10:
            self._toggle_window_mode(WindowMode.BORDERLESS)
            return
        if key == pygame.K_F9:
            self._cycle_view()
            return

        if self.state == State.MENU:
            self._menu_key(event)
        elif self.state == State.PLAYING:
            self._play_key(key)
        elif self.state == State.PAUSED:
            self._pause_key(key)
        elif self.state == State.GAME_OVER:
            if key == pygame.K_ESCAPE:
                self._enter_menu()
            elif key in self.key_dir or key in (pygame.K_RETURN, pygame.K_SPACE):
                if self.ui.gameover_t > 0.4:
                    self.restart()
        elif self.state in (State.WATCH, State.BENCH):
            if key == pygame.K_ESCAPE:
                self._enter_menu()
            elif key in (pygame.K_r,) and self.session is not None:
                self.session.reset()

    def _menu_key(self, event) -> None:
        key = event.key
        if self.menu.capturing:
            self.menu.capture_key(pygame.key.name(key))
            return
        if key in (pygame.K_UP, pygame.K_w):
            self.menu.nav_vertical(-1)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.menu.nav_vertical(1)
        elif key in (pygame.K_LEFT, pygame.K_a):
            self.menu.nav_horizontal(-1)
        elif key in (pygame.K_RIGHT, pygame.K_d):
            self.menu.nav_horizontal(1)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self.menu.activate()
        elif key == pygame.K_TAB:
            self.menu.next_tab(-1 if event.mod & pygame.KMOD_SHIFT else 1)
        elif key in (pygame.K_q, pygame.K_LEFTBRACKET):
            self.menu.next_tab(-1)
        elif key in (pygame.K_e, pygame.K_RIGHTBRACKET):
            self.menu.next_tab(1)
        elif key == pygame.K_ESCAPE:
            if not self.menu.back():
                pass        # already at the root menu; stay put

    def _play_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE or key == self.pause_key:
            self.toggle_pause()
            return
        if key in self.key_dir:
            self.engine.begin_move_with_direction()

    def _pause_key(self, key: int) -> None:
        if key == self.pause_key or key == pygame.K_p:
            self.toggle_pause()
        elif key == pygame.K_r:
            self.restart()
        elif key == pygame.K_ESCAPE:
            self._enter_menu()

    def _handle_keyup(self, key: int) -> None:
        if self.state != State.PLAYING:
            return
        direction = self.key_dir.get(key)
        if direction is not None:
            self.engine.move_with_direction(direction)

    def _handle_mouse_down(self, event) -> None:
        if self.state == State.MENU:
            self.menu.mouse_down(self._menu_local(event.pos), event.button)
            return
        if event.button != 1:
            return
        self._mouse_down = event.pos
        if self.state == State.PLAYING:
            self.engine.begin_move_with_direction()

    def _handle_mouse_up(self, pos) -> None:
        if self.state == State.MENU:
            self.menu.mouse_up(self._menu_local(pos))
            return
        if self.state in (State.WATCH, State.BENCH):
            return
        if self._mouse_down is None:
            return
        dx = pos[0] - self._mouse_down[0]
        dy = pos[1] - self._mouse_down[1]
        self._mouse_down = None
        if self.state == State.GAME_OVER:
            if self.ui.gameover_t > 0.4:
                self.restart()
            return
        if self.state != State.PLAYING:
            return
        thresh = max(12, self.win_h // 40)
        if abs(dx) < thresh and abs(dy) < thresh:
            direction = Direction.UP
        elif abs(dx) > abs(dy):
            direction = Direction.RIGHT if dx > 0 else Direction.LEFT
        else:
            direction = Direction.DOWN if dy > 0 else Direction.UP
        self.engine.move_with_direction(direction)

    def _handle_joy(self, event) -> None:
        if event.type == pygame.JOYBUTTONDOWN:
            if self.state == State.MENU:
                if event.button in (0, 7):           # A / Start
                    self.menu.activate()
                elif event.button == 1:              # B
                    self.menu.back()
                elif event.button in (4,):           # LB
                    self.menu.next_tab(-1)
                elif event.button in (5,):           # RB
                    self.menu.next_tab(1)
            elif self.state == State.PLAYING and event.button == 0:
                self.engine.begin_move_with_direction()
                self.engine.move_with_direction(Direction.UP)
            elif self.state in (State.WATCH, State.BENCH, State.GAME_OVER) and event.button == 1:
                self._enter_menu()
        elif event.type == pygame.JOYHATMOTION:
            hx, hy = event.value
            if self.state == State.MENU:
                if hy:
                    self.menu.nav_vertical(-1 if hy > 0 else 1)
                if hx:
                    self.menu.nav_horizontal(1 if hx > 0 else -1)
            elif self.state == State.PLAYING:
                d = ({(0, 1): Direction.UP, (0, -1): Direction.DOWN,
                      (-1, 0): Direction.LEFT, (1, 0): Direction.RIGHT}).get((hx, hy))
                if d:
                    self.engine.begin_move_with_direction()
                    self.engine.move_with_direction(d)

    def _toggle_window_mode(self, mode: WindowMode) -> None:
        target = WindowMode.WINDOWED if self.window_mode == mode else mode
        self.set_window_mode(target)
        self.cfg.values["window_mode"] = target.value
        self.cfg.save()

    def _cycle_view(self) -> None:
        cur = self.cfg.get("view_mode")
        nxt = _VIEW_CYCLE[(_VIEW_CYCLE.index(cur) + 1) % len(_VIEW_CYCLE)] \
            if cur in _VIEW_CYCLE else "auto"
        self.cfg.values["view_mode"] = nxt
        self.cfg.save()
        self.apply_settings(["view_mode"])
        self.menu.notify(f"View: {nxt}")

    # -- main loop ---------------------------------------------------------
    def _update_logic(self) -> None:
        dt = config.FIXED_DT
        if self.state == State.MENU:
            tween.update(dt)
            self.engine.tick(dt)
            self._attract_step(dt)
            self.menu.update(dt)
        elif self.state == State.PLAYING:
            tween.update(dt)
            self.engine.tick(dt)
            self._game_seconds += dt
            self.ui.update(dt, "playing")
            self._update_banner_audio(dt)
        elif self.state == State.GAME_OVER:
            tween.update(dt)
            self.engine.tick(dt)
            self.ui.update(dt, "gameOver")
            self._update_banner_audio(dt)
        elif self.state == State.WATCH and self.session is not None:
            speed = max(1, int(self.cfg.get("ai_speed")))
            try:
                for _ in range(speed):
                    self.session.frame_step(dt)
            except Exception as exc:        # never let an AI hiccup crash the app
                print(f"[ai] playback error: {exc}")
                self._enter_menu()
                self.menu.notify("AI playback stopped (error)")
        elif self.state == State.BENCH:
            self._bench_step(dt)
        elif self.state == State.PAUSED:
            pass        # frozen; the pause veil is drawn each present

    def _update_banner_audio(self, dt: float) -> None:
        if self.state != State.GAME_OVER:
            return
        prev = self._banner_timer
        self._banner_timer += dt
        if prev < 0.6 <= self._banner_timer:
            self.audio.play_banner()

    def _bench_step(self, dt: float) -> None:
        eng = self.engine
        tween.update(dt)
        eng.tick(dt)
        if not eng.hero.is_alive:
            self.engine.setup_game(self.character)
            self.engine.init()
            self.engine.screen_shake = False
        elif not eng.hero.moving:
            eng.begin_move_with_direction()
            eng.move_with_direction(Direction.UP)

    def _bench_record(self, ms: float) -> None:
        b = self._bench
        if b is None:
            return
        if b["start"] is None:
            b["start"] = time.perf_counter()
        b["frames"] += 1
        b["sum_ms"] += ms
        b["worst_ms"] = max(b["worst_ms"], ms)
        if time.perf_counter() - b["start"] >= b["dur"]:
            self._finish_bench()

    def _perf_lines(self):
        if not self.show_perf:
            return None
        lines = [f"{self._fps:4.0f} FPS"]
        if self.perf_detail:
            ms = 1000.0 / self._fps if self._fps > 0 else 0.0
            lines.append(f"{ms:4.1f} ms")
            lines.append(self._gpu_info.get("renderer", "")[:22])
            lines.append(f"{self.layout.w}x{self.layout.h} @{int(self.render_scale*100)}%")
        if self.debug:
            lines.append(f"state {self.state.value}")
        return lines

    def _present(self) -> None:
        rect = self.layout
        scene = (self.session.scene if (self.state == State.WATCH and self.session)
                 else self.engine.scene)
        self.renderer.render(scene)
        self.renderer.clear_window(rect.bg, self.win_w, self.win_h)
        self.renderer.present_scene(rect.rect)

        if self.state == State.MENU:
            self._present_menu()
        else:
            self._present_hud(rect)

        if rect.bezel and self._bezel_bytes is not None:
            self.renderer.draw_overlay("bezel",
                                       self._bezel_bytes if self._bezel_dirty else None,
                                       self.win_w, self.win_h, (0, 0, self.win_w, self.win_h))
            self._bezel_dirty = False
        if config.GPU_SYNC_BEFORE_SWAP:
            self.renderer.finish()
        pygame.display.flip()

    def _present_menu(self) -> None:
        a = self._menu_area
        self.menu.render()
        self.renderer.draw_overlay("menu", self.menu.to_bytes(), a.w, a.h,
                                   (a.x, a.y, a.w, a.h))

    def _present_hud(self, rect) -> None:
        perf = self._perf_lines()
        if self.state in (State.WATCH, State.BENCH):
            score = self.session.score if (self.session and self.state == State.WATCH) else 0
            label = (self.session.label if self.session else "Benchmark")
            if self.state == State.BENCH:
                label = "Benchmark"
            elif self.session is not None and hasattr(self.session, "alive_count"):
                label = f"{label}   {self.session.alive_count} OF {self.session.n} ALIVE"
            hint = "Esc  Main Menu"
            self.ui.render_watch(score, label, hint, perf)
            self.renderer.draw_overlay("ui", self.ui.to_bytes(), rect.w, rect.h, rect.rect)
            return
        self.ui.render(self.state.value if self.state != State.PLAYING else "playing",
                       self.score, self.save.highscore,
                       paused=self.state == State.PAUSED, new_best=self.new_best,
                       show_hints=self.show_hints, perf_lines=perf)
        self.renderer.draw_overlay("ui", self.ui.to_bytes(), rect.w, rect.h, rect.rect)

    def run(self) -> None:
        while self.running:
            frame_dt = self.clock.tick(0 if self.state == State.BENCH else self.fps_limit) / 1000.0
            self._fps = self.clock.get_fps()
            self._frame += 1
            self._process_events()
            self._accum += frame_dt
            steps = 0
            while self._accum >= config.FIXED_DT and steps < 5:
                self._update_logic()
                self._accum -= config.FIXED_DT
                steps += 1
            if steps == 5:
                self._accum = 0.0
            if not self._visible:
                pygame.time.wait(30)
            elif self.state == State.BENCH:
                t0 = time.perf_counter()
                self._present()
                self._bench_record((time.perf_counter() - t0) * 1000.0)
            else:
                self._present()

        persistence.save(self.save)
        pygame.quit()

    def _finish_bench(self) -> None:
        b = self._bench
        elapsed = max(1e-6, time.perf_counter() - b.get("start", time.perf_counter()))
        fps = b["frames"] / elapsed
        avg_ms = b["sum_ms"] / max(1, b["frames"])
        self._enter_menu()
        self.menu.notify(f"Benchmark: {fps:.0f} FPS avg · {avg_ms:.1f} ms · "
                         f"worst {b['worst_ms']:.1f} ms")


def main(display_mode: str = "native") -> None:
    try:
        mode = DisplayMode(display_mode)
    except ValueError:
        mode = None
    Game(display_mode=mode).run()


if __name__ == "__main__":
    main()
