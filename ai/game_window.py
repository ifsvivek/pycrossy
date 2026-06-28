"""Live play window — renders the env scene while the AI plays, with a small HUD.

Runs in the trainer's process (the dashboard is a separate process). Uses the dedicated
GPU when available and presents the game filling a resizable / fullscreen window (the camera
widens horizontally to the window aspect), so it scales to any resolution without distortion.
"""
from __future__ import annotations

from typing import Dict

import moderngl
import pygame

from pycrossy import assets, config, gpu
from pycrossy.layout import DisplayMode, compute as compute_layout
from pycrossy.renderer import Renderer


class LivePlayWindow:
    def __init__(self, width: int = 480, height: int = 760, title: str = "PyCrossy — AI Playing"):
        self._gpu_request = gpu.prefer_high_performance_gpu(config.PREFER_DEDICATED_GPU)
        pygame.init()
        pygame.font.init()
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        gpu.safe_set_mode((width, height), flags, vsync=False)
        pygame.display.set_caption(title)
        self.win_w, self.win_h = pygame.display.get_window_size()
        self.ctx = moderngl.create_context()
        gpu.log_startup(self.ctx, self._gpu_request)
        self._fullscreen = False
        self._rebuild()
        self.clock = pygame.time.Clock()
        self.running = True

    def _rebuild(self) -> None:
        self.layout = compute_layout(self.win_w, self.win_h, DisplayMode.NATIVE)
        r = self.layout
        self.renderer = Renderer(self.ctx, max(r.w, 16), max(r.h, 16))
        s = self.layout.scale
        self._hud = pygame.Surface((max(r.w, 16), max(r.h, 16)), pygame.SRCALPHA)
        self.font = pygame.font.Font(assets.RETRO_FONT, max(14, int(40 * s)))
        self.small = pygame.font.Font(assets.RETRO_FONT, max(9, int(16 * s)))

    def _relayout(self) -> None:
        new = compute_layout(self.win_w, self.win_h, DisplayMode.NATIVE)
        if (new.w, new.h) != (self.layout.w, self.layout.h):
            self.renderer.resize(max(new.w, 16), max(new.h, 16))
            self.layout = new
            self._rebuild_hud()
        else:
            self.layout = new

    def _rebuild_hud(self) -> None:
        r = self.layout
        s = r.scale
        self._hud = pygame.Surface((max(r.w, 16), max(r.h, 16)), pygame.SRCALPHA)
        self.font = pygame.font.Font(assets.RETRO_FONT, max(14, int(40 * s)))
        self.small = pygame.font.Font(assets.RETRO_FONT, max(9, int(16 * s)))

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if self._fullscreen:
            size = pygame.display.get_desktop_sizes()[0]
            gpu.safe_set_mode(size, flags | pygame.FULLSCREEN)
        else:
            gpu.safe_set_mode((480, 760), flags | pygame.RESIZABLE)
        self.win_w, self.win_h = pygame.display.get_window_size()
        self.ctx = moderngl.create_context()
        self._rebuild()

    def pump(self) -> bool:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.running = False
                elif e.key == pygame.K_F11:
                    self._toggle_fullscreen()
            elif e.type == pygame.VIDEORESIZE and not self._fullscreen:
                self.win_w, self.win_h = max(e.w, 64), max(e.h, 64)
                self._relayout()
        return self.running

    def _outline(self, text, x, y, font, color=(255, 255, 255), ow=3):
        base = font.render(text, True, color)
        outl = font.render(text, True, (0, 0, 0))
        for dx, dy in ((-ow, 0), (ow, 0), (0, -ow), (0, ow), (-ow, -ow), (ow, ow), (-ow, ow), (ow, -ow)):
            self._hud.blit(outl, (x + dx, y + dy))
        self._hud.blit(base, (x, y))

    def render(self, scene, hud: Dict, fps_cap: int = 0) -> None:
        rect = self.layout
        self.renderer.render(scene)
        self.renderer.clear_window(rect.bg, self.win_w, self.win_h)
        self.renderer.present_scene(rect.rect)
        self._hud.fill((0, 0, 0, 0))
        m = max(8, int(16 * rect.scale))
        self._outline(str(hud.get("score", 0)), m, m, self.font, ow=max(2, int(3 * rect.scale)))
        line = f"{hud.get('algo', '')}  ep {hud.get('episode', 0)}  best {hud.get('best_score', 0)}"
        gen = hud.get("generation")
        if gen is not None:
            line += f"  gen {gen}"
        workers = hud.get("workers", 0)
        if workers:
            line += f"  x{workers} workers"
        self._outline(line, m, m + int(48 * rect.scale), self.small, (255, 240, 120),
                      ow=max(1, int(2 * rect.scale)))
        self.renderer.draw_overlay("ui", pygame.image.tostring(self._hud, "RGBA", True),
                                   rect.w, rect.h, rect.rect)
        if config.GPU_SYNC_BEFORE_SWAP:
            self.renderer.finish()      # complete GPU work before the (PRIME) buffer swap
        pygame.display.flip()
        if fps_cap:
            self.clock.tick(fps_cap)

    def close(self) -> None:
        pygame.quit()
