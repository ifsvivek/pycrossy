"""Reusable, resolution-independent UI primitives for the menu / settings system.

A :class:`Theme` carries the colour palette and a set of fonts scaled to the current UI
size, and the module provides stateless draw helpers plus *geometry helpers* (e.g.
:func:`slider_track_rect`). The menu draws a control and hit-tests it through the **same**
geometry helper, so the visuals and the mouse targets can never drift apart.

Everything renders onto a ``pygame.Surface`` (later uploaded to the GL overlay), and every
size derives from a ``scale`` factor, so the UI is crisp and proportional at any resolution.
"""
from __future__ import annotations

from typing import List, Tuple

import pygame

from . import assets


class Theme:
    # Surfaces
    backdrop = (10, 13, 20)
    scrim = (8, 11, 18, 165)         # darken the live 3D backdrop behind a menu
    panel = (24, 30, 45)
    panel_alt = (30, 37, 55)
    rail = (17, 22, 34)
    row_focus = (38, 78, 130)
    row_hover = (33, 42, 62)
    divider = (45, 54, 75)
    # Text
    text = (230, 236, 246)
    text_dim = (156, 167, 186)
    text_mute = (104, 114, 134)
    # Accent / status
    accent = (74, 158, 240)
    accent_soft = (120, 190, 250)
    good = (92, 210, 140)
    warn = (245, 202, 90)
    danger = (236, 102, 102)
    # Controls
    track = (49, 58, 78)
    track_fill = (74, 158, 240)
    knob = (236, 241, 250)

    def __init__(self, scale: float):
        pygame.font.init()
        self.scale = scale
        s = scale
        self.f_title = pygame.font.Font(assets.RETRO_FONT, max(16, int(34 * s)))
        self.f_head = pygame.font.Font(assets.RETRO_FONT, max(11, int(20 * s)))
        self.f_item = pygame.font.Font(None, max(15, int(27 * s)))
        self.f_item_b = pygame.font.Font(None, max(15, int(27 * s)))
        self.f_item_b.set_bold(True)
        self.f_small = pygame.font.Font(None, max(12, int(21 * s)))
        self.f_tiny = pygame.font.Font(None, max(11, int(18 * s)))

    def px(self, v: float) -> int:
        return max(1, int(round(v * self.scale)))


# ---------------------------------------------------------------------------
# Low-level drawing
# ---------------------------------------------------------------------------
def text(surf, font, s, pos, color, anchor="topleft") -> pygame.Rect:
    img = font.render(s, True, color)
    rect = img.get_rect(**{anchor: pos})
    surf.blit(img, rect)
    return rect


def text_clip(surf, font, s, pos, color, max_w, anchor="topleft") -> pygame.Rect:
    """Render text, truncating with an ellipsis if it would exceed ``max_w``."""
    if font.size(s)[0] > max_w:
        while s and font.size(s + "…")[0] > max_w:
            s = s[:-1]
        s = s + "…"
    return text(surf, font, s, pos, color, anchor)


def rounded(surf, rect, color, radius, width=0) -> None:
    pygame.draw.rect(surf, color, rect, width=width, border_radius=radius)


def panel(surf, rect, color, radius, alpha=255, border=None, border_w=2) -> None:
    r = pygame.Rect(rect)
    if alpha < 255:
        tmp = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
        pygame.draw.rect(tmp, (*color, alpha), (0, 0, r.w, r.h), border_radius=radius)
        surf.blit(tmp, r.topleft)
    else:
        pygame.draw.rect(surf, color, r, border_radius=radius)
    if border is not None:
        pygame.draw.rect(surf, border, r, width=border_w, border_radius=radius)


def scrim(surf, rect, rgba) -> None:
    r = pygame.Rect(rect)
    tmp = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
    tmp.fill(rgba)
    surf.blit(tmp, r.topleft)


# ---------------------------------------------------------------------------
# Geometry helpers (shared by draw + hit-test)
# ---------------------------------------------------------------------------
def value_zone(row: pygame.Rect) -> pygame.Rect:
    """The right-hand area of a settings row where the value/control sits."""
    w = int(row.w * 0.46)
    return pygame.Rect(row.right - w - row.h // 3, row.y, w, row.h)


def slider_track_rect(row: pygame.Rect, theme: Theme) -> pygame.Rect:
    z = value_zone(row)
    h = theme.px(6)
    lbl = theme.px(46)                       # room for the % value on the right
    return pygame.Rect(z.x, z.centery - h // 2, max(20, z.w - lbl), h)


def choice_arrow_rects(row: pygame.Rect, theme: Theme) -> Tuple[pygame.Rect, pygame.Rect]:
    z = value_zone(row)
    a = z.h
    left = pygame.Rect(z.x, z.y, a, z.h)
    right = pygame.Rect(z.right - a, z.y, a, z.h)
    return left, right


# ---------------------------------------------------------------------------
# Control renderers
# ---------------------------------------------------------------------------
def row_background(surf, row, theme: Theme, focused: bool, hover: bool) -> None:
    if focused:
        panel(surf, row, theme.row_focus, theme.px(8))
        pygame.draw.rect(surf, theme.accent_soft,
                         (row.x, row.y, theme.px(4), row.h),
                         border_radius=theme.px(2))
    elif hover:
        panel(surf, row, theme.row_hover, theme.px(8))


def draw_label(surf, row, theme: Theme, label: str, focused: bool, enabled=True) -> None:
    color = theme.text if enabled else theme.text_mute
    if focused and enabled:
        color = (255, 255, 255)
    pad = theme.px(14)
    text_clip(surf, theme.f_item, label, (row.x + pad, row.centery), color,
              int(row.w * 0.5), anchor="midleft")


def draw_toggle(surf, row, theme: Theme, on: bool, focused: bool) -> None:
    z = value_zone(row)
    w, h = theme.px(46), theme.px(24)
    track = pygame.Rect(z.right - w, z.centery - h // 2, w, h)
    panel(surf, track, theme.track_fill if on else theme.track, h // 2)
    kr = theme.px(9)
    kx = track.right - kr - theme.px(4) if on else track.x + kr + theme.px(4)
    pygame.draw.circle(surf, theme.knob, (kx, track.centery), kr)
    text(surf, theme.f_small, "ON" if on else "OFF",
         (track.x - theme.px(10), track.centery),
         theme.good if on else theme.text_mute, anchor="midright")


def draw_choice(surf, row, theme: Theme, label_value: str, focused: bool) -> None:
    z = value_zone(row)
    left, right = choice_arrow_rects(row, theme)
    col = theme.accent_soft if focused else theme.text_dim
    text(surf, theme.f_item_b, "‹", left.center, col, anchor="center")
    text(surf, theme.f_item_b, "›", right.center, col, anchor="center")
    text_clip(surf, theme.f_item, label_value,
              (z.centerx, z.centery), theme.text,
              right.x - left.right - theme.px(6), anchor="center")


def draw_slider(surf, row, theme: Theme, frac: float, value_text: str, focused: bool) -> None:
    track = slider_track_rect(row, theme)
    frac = max(0.0, min(1.0, frac))
    panel(surf, track, theme.track, track.h // 2)
    fill = pygame.Rect(track.x, track.y, int(track.w * frac), track.h)
    if fill.w > 0:
        panel(surf, fill, theme.track_fill, track.h // 2)
    kx = track.x + int(track.w * frac)
    pygame.draw.circle(surf, theme.knob, (kx, track.centery), theme.px(8))
    if focused:
        pygame.draw.circle(surf, theme.accent_soft, (kx, track.centery), theme.px(8),
                           width=theme.px(2))
    z = value_zone(row)
    text(surf, theme.f_small, value_text, (z.right, row.centery), theme.text_dim,
         anchor="midright")


def draw_keybind(surf, row, theme: Theme, key_label: str, focused: bool, capturing: bool) -> None:
    z = value_zone(row)
    w = theme.px(120)
    box = pygame.Rect(z.right - w, z.centery - theme.px(15), w, theme.px(30))
    border = theme.warn if capturing else (theme.accent if focused else theme.divider)
    panel(surf, box, theme.panel_alt, theme.px(6), border=border, border_w=theme.px(2))
    label = "Press a key…" if capturing else key_label.upper()
    text_clip(surf, theme.f_small, label, box.center,
              theme.warn if capturing else theme.text, box.w - theme.px(12), anchor="center")


def draw_action(surf, row, theme: Theme, focused: bool) -> None:
    # The label itself acts as the button; add a chevron to signal it does something.
    z = value_zone(row)
    text(surf, theme.f_item_b, "›", (z.right, row.centery),
         theme.accent_soft if focused else theme.text_dim, anchor="midright")


def draw_info(surf, row, theme: Theme, value: str) -> None:
    z = value_zone(row)
    text_clip(surf, theme.f_small, value, (z.right, row.centery), theme.text_dim,
              z.w + theme.px(40), anchor="midright")


def draw_badge(surf, pos, theme: Theme, label: str, color, anchor="topleft") -> pygame.Rect:
    pad = theme.px(6)
    img = theme.f_tiny.render(label, True, (12, 16, 24))
    w = img.get_width() + pad * 2
    h = img.get_height() + pad
    rect = pygame.Rect(0, 0, w, h)
    setattr(rect, anchor, pos)
    panel(surf, rect, color, h // 2)
    surf.blit(img, img.get_rect(center=rect.center))
    return rect


def draw_tooltip(surf, area: pygame.Rect, theme: Theme, title: str, body: str) -> None:
    """Draw a wrapped tooltip panel pinned to the bottom of ``area``."""
    if not body:
        return
    pad = theme.px(12)
    inner_w = area.w - pad * 4
    lines = _wrap(theme.f_small, body, inner_w)
    line_h = theme.f_small.get_linesize()
    h = pad * 2 + line_h * (len(lines) + (1 if title else 0))
    box = pygame.Rect(area.x + pad, area.bottom - h - pad, area.w - pad * 2, h)
    panel(surf, box, theme.panel_alt, theme.px(8), alpha=242,
          border=theme.divider, border_w=theme.px(1))
    y = box.y + pad
    if title:
        text(surf, theme.f_small, title, (box.x + pad, y), theme.accent_soft)
        y += line_h
    for ln in lines:
        text(surf, theme.f_small, ln, (box.x + pad, y), theme.text_dim)
        y += line_h


def _wrap(font, s: str, max_w: int) -> List[str]:
    words = s.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if font.size(trial)[0] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines
