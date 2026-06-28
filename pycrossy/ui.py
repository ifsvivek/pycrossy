"""2D UI overlay — HUD + menus, fully resolution-independent.

Rendered to a pygame ``Surface`` sized to the gameplay rect and uploaded to the GL overlay
texture. Every size/position derives from a ``scale`` factor (rect height / reference
height), so the HUD, home title, neutral game-over result card, pause veil and hints all
scale and reposition together at any resolution or aspect ratio.
"""
from __future__ import annotations

import math
import pygame

from . import assets
from .mathutils import clamp, elastic_out, power1_inout

WHITE = (255, 255, 255)
YELLOW = (255, 225, 70)
BLACK = (0, 0, 0)
PANEL = (255, 255, 255, 235)
PANEL_DARK = (28, 32, 44, 230)
ACCENT = (54, 145, 235)


class UI:
    REFERENCE_HEIGHT = 820

    def __init__(self, width: int, height: int, ui_scale: float = 1.0):
        pygame.font.init()
        self.width = max(width, 16)
        self.height = max(height, 16)
        self.scale = (self.height / self.REFERENCE_HEIGHT) * max(0.5, ui_scale)
        self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        s = self.scale
        self.font_score = pygame.font.Font(assets.RETRO_FONT, max(12, int(48 * s)))
        self.font_top = pygame.font.Font(assets.RETRO_FONT, max(8, int(15 * s)))
        self.font_card = pygame.font.Font(assets.RETRO_FONT, max(10, int(22 * s)))
        self.font_card_big = pygame.font.Font(assets.RETRO_FONT, max(16, int(56 * s)))
        self.font_hint = pygame.font.Font(assets.RETRO_FONT, max(9, int(17 * s)))

        title = pygame.image.load(assets.TITLE_IMAGE).convert_alpha()
        tw = int(self.width * 0.82)
        th = max(1, int(title.get_height() * (tw / title.get_width())))
        self.title_img = pygame.transform.smoothscale(title, (tw, th))

        self.title_t = 0.0
        self.gameover_t = 0.0
        self.hint_t = 0.0

    # -- animation state ---------------------------------------------------
    def on_enter_home(self) -> None:
        self.title_t = 0.0

    def on_game_over(self) -> None:
        self.gameover_t = 0.0

    def update(self, dt: float, state: str) -> None:
        self.hint_t += dt
        if state == "none":
            self.title_t = min(1.0, self.title_t + dt / 0.8)
        elif state == "gameOver":
            self.gameover_t += dt

    # -- helpers -----------------------------------------------------------
    def _outline_text(self, font, text, color, x, y, ow=None, anchor="topleft", outline=BLACK):
        ow = ow if ow is not None else max(1, int(4 * self.scale))
        base = font.render(text, True, color)
        rect = base.get_rect(**{anchor: (x, y)})
        outl = font.render(text, True, outline)
        for dx, dy in ((-ow, 0), (ow, 0), (0, -ow), (0, ow),
                       (-ow, -ow), (ow, -ow), (-ow, ow), (ow, ow)):
            self.surface.blit(outl, (rect.x + dx, rect.y + dy))
        self.surface.blit(base, rect)
        return rect

    def _center(self, surf, cy):
        self.surface.blit(surf, (self.width // 2 - surf.get_width() // 2, int(cy)))

    # -- screens -----------------------------------------------------------
    def render(self, state: str, score: int, highscore: int, paused: bool = False,
               new_best: bool = False, show_hints: bool = True,
               perf_lines=None) -> pygame.Surface:
        self.surface.fill((0, 0, 0, 0))
        self.show_hints = show_hints
        m = max(8, int(16 * self.scale))

        if state != "none":
            self._outline_text(self.font_score, str(score), WHITE, m, m)
            if highscore > 0:
                self._outline_text(self.font_top, f"TOP {highscore}", YELLOW,
                                   m + 2, m + int(60 * self.scale),
                                   ow=max(1, int(2 * self.scale)))

        if state == "none":
            self._render_home(highscore)
        elif state == "gameOver":
            self._render_game_over(score, highscore, new_best)

        if paused:
            self._render_pause()
        if perf_lines:
            self._render_perf(perf_lines)
        return self.surface

    def _render_pause(self) -> None:
        veil = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        veil.fill((105, 201, 230, 200))
        self.surface.blit(veil, (0, 0))
        self._outline_text(self.font_card_big, "PAUSED", WHITE,
                           self.width // 2, int(self.height * 0.42),
                           ow=max(1, int(3 * self.scale)), anchor="center")
        for i, line in enumerate(("P  Resume", "R  Restart", "Esc  Main Menu")):
            img = self.font_hint.render(line, True, WHITE)
            self._center(img, self.height * 0.54 + i * int(26 * self.scale))

    def _render_perf(self, lines) -> None:
        x = self.width - max(8, int(10 * self.scale))
        y = max(8, int(10 * self.scale))
        for line in lines:
            img = self.font_hint.render(line, True, (180, 255, 180))
            outl = self.font_hint.render(line, True, BLACK)
            rect = img.get_rect(topright=(x, y))
            self.surface.blit(outl, (rect.x + 1, rect.y + 1))
            self.surface.blit(img, rect)
            y += img.get_height() + int(2 * self.scale)

    def render_watch(self, score: int, label: str, hint: str,
                     perf_lines=None) -> pygame.Surface:
        """HUD for AI Auto-Play / Replay / Benchmark spectator modes."""
        self.surface.fill((0, 0, 0, 0))
        m = max(8, int(16 * self.scale))
        self._outline_text(self.font_score, str(score), WHITE, m, m)
        if label:
            self._outline_text(self.font_hint, label, YELLOW, m + 2,
                               m + int(58 * self.scale), ow=max(1, int(2 * self.scale)))
        if hint:
            img = self.font_hint.render(hint, True, WHITE)
            img.set_alpha(200)
            self._center(img, self.height - int(48 * self.scale))
        if perf_lines:
            self._render_perf(perf_lines)
        return self.surface

    def _render_home(self, highscore: int) -> None:
        t = power1_inout(clamp(self.title_t, 0, 1))
        x = (self.width - self.title_img.get_width()) // 2 + int((1 - t) * -self.width)
        y = int(self.height * 0.2) + int((1 - t) * -100 * self.scale)
        self.surface.blit(self.title_img, (x, y))

        if highscore > 0:
            self._outline_text(self.font_top, f"TOP {highscore}", YELLOW,
                               self.width - int(14 * self.scale), int(16 * self.scale),
                               ow=max(1, int(2 * self.scale)), anchor="topright")

        if getattr(self, "show_hints", True):
            alpha = 0.5 + 0.5 * math.sin(self.hint_t * 3.0)
            hint = self.font_hint.render("PRESS SPACE OR TAP TO PLAY", True, WHITE)
            hint.set_alpha(int(120 + 135 * alpha))
            self._center(hint, self.height * 0.74)

    def _render_game_over(self, score: int, highscore: int, new_best: bool) -> None:
        # Slide a neutral result card in from the top with an elastic ease.
        progress = elastic_out(clamp(self.gameover_t / 0.9, 0, 1))
        cw = int(self.width * 0.74)
        ch = int(self.height * 0.30)
        cx = (self.width - cw) // 2
        target_y = int(self.height * 0.26)
        cy = int(-ch + (target_y + ch) * progress)

        card = pygame.Surface((cw, ch), pygame.SRCALPHA)
        pygame.draw.rect(card, PANEL_DARK, (0, 0, cw, ch), border_radius=int(18 * self.scale))
        pygame.draw.rect(card, (*ACCENT, 255), (0, 0, cw, ch),
                         width=max(2, int(3 * self.scale)), border_radius=int(18 * self.scale))
        self.surface.blit(card, (cx, cy))

        label = self.font_card.render("SCORE", True, (200, 210, 225))
        self.surface.blit(label, (cx + (cw - label.get_width()) // 2, cy + int(ch * 0.12)))
        big = self.font_card_big.render(str(score), True, WHITE)
        self.surface.blit(big, (cx + (cw - big.get_width()) // 2, cy + int(ch * 0.28)))
        best_txt = "NEW BEST!" if new_best else f"BEST  {highscore}"
        best = self.font_card.render(best_txt, True, YELLOW if new_best else (200, 210, 225))
        self.surface.blit(best, (cx + (cw - best.get_width()) // 2, cy + int(ch * 0.66)))

        if self.gameover_t > 1.0 and getattr(self, "show_hints", True):
            alpha = 0.5 + 0.5 * math.sin(self.hint_t * 3.0)
            hint = self.font_hint.render("TAP OR SPACE  ·  ESC FOR MENU", True, WHITE)
            hint.set_alpha(int(120 + 135 * alpha))
            self._center(hint, self.height - int(70 * self.scale))

    def to_bytes(self) -> bytes:
        return pygame.image.tostring(self.surface, "RGBA", True)
