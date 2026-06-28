"""The front-end menu system: main hub, tabbed settings, statistics and credits.

``MenuSystem`` is a self-contained screen manager that renders onto a ``pygame.Surface``
(uploaded to the GL overlay by the game shell) and consumes keyboard, mouse and controller
input through a small *semantic* API (:meth:`nav_vertical`, :meth:`activate`, …) shared by
all input devices. It adapts between a **desktop** layout (full-window, left tab rail) and a
**phone** layout (portrait, top chip rail), driven by the schema in :mod:`pycrossy.settings`.

The shell wires three callbacks: ``on_launch(name, payload)`` to start a game/AI/replay mode,
``on_apply(changed_keys)`` to (re)apply committed settings to the live game, and
``on_preview(key)`` to live-preview a single pending value as the user drags a slider. The
menu never touches the renderer/engine itself — it only edits the :class:`~pycrossy.settings.Config`
and asks the shell to apply.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

import pygame

from . import assets, settings as st, widgets as W
from .mathutils import clamp
from .widgets import Theme


class Screen(str, Enum):
    MAIN = "main"
    SETTINGS = "settings"
    STATS = "stats"
    CREDITS = "credits"


# Main-menu entries: (action, label, subtitle, enabled-predicate key)
@dataclass
class MainEntry:
    action: str
    label: str
    subtitle: str
    tooltip: str = ""


MAIN_ENTRIES: List[MainEntry] = [
    MainEntry("play", "Play", "Hop across the road", "Start a normal game."),
    MainEntry("ai_auto", "AI Auto Play", "Watch a trained agent",
              "Load the best saved model for the selected algorithm and watch it play. "
              "Falls back to a simple heuristic if no model is trained yet."),
    MainEntry("ai_swarm", "AI Swarm", "Watch the whole population",
              "Spawn many chickens at once — each running its own policy from the population — "
              "with the camera following whoever is in the lead. Set the count in Settings ▸ AI."),
    MainEntry("ai_train", "AI Training", "Teach an agent to play",
              "Launch dual-window training (game + live analytics dashboard) for the "
              "selected algorithm. Configure it in Settings ▸ AI."),
    MainEntry("replay", "Replay Viewer", "Re-watch a saved run",
              "Replay the best recorded episode for the selected algorithm."),
    MainEntry("benchmark", "Benchmark", "Stress-test performance",
              "Render a dense scene at full tilt and report average FPS / frame time."),
    MainEntry("settings", "Settings", "Graphics, audio, controls…",
              "Configure graphics, audio, gameplay, controls, performance and AI."),
    MainEntry("stats", "Statistics", "Your lifetime numbers",
              "Games played, best scores and totals."),
    MainEntry("credits", "Credits", "Who made this",
              "Attribution and the tech behind the game."),
    MainEntry("quit", "Quit", "Leave the game", "Exit PyCrossy."),
]

# Footer buttons on the settings screen: (id, label).
FOOTER = [
    ("defaults", "Defaults"),
    ("import", "Import"),
    ("export", "Export"),
    ("backup", "Backup"),
    ("cancel", "Cancel"),
    ("apply", "Apply"),
]


@dataclass
class Item:
    """One interactive row in the current screen, with its laid-out rect."""
    kind: str                       # 'main' | 'spec' | 'back'
    rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 0, 0))
    spec: Optional[st.Spec] = None
    entry: Optional[MainEntry] = None
    focusable: bool = True


class MenuSystem:
    def __init__(self, config: st.Config, save, on_launch: Callable[[str, dict], None],
                 on_apply: Callable[[List[str]], None],
                 on_preview: Callable[[str], None],
                 info_provider: Optional[Callable[[], Dict[str, str]]] = None):
        self.cfg = config
        self.save = save
        self.on_launch = on_launch
        self.on_apply = on_apply
        self.on_preview = on_preview
        self.info_provider = info_provider or (lambda: {})

        self.screen = Screen.MAIN
        self.mode = "desktop"
        self.w = self.h = 16
        self.theme = Theme(1.0)
        self.surface = pygame.Surface((16, 16), pygame.SRCALPHA)
        self.title_img = None

        # navigation state
        self.region = "list"          # 'tabs' | 'list' | 'footer' (settings only)
        self.list_focus = 0
        self.tab = 0                   # current settings category index
        self.footer_focus = len(FOOTER) - 1
        self.capturing: Optional[str] = None   # spec key currently rebinding
        self.scroll = 0.0
        self._drag_key: Optional[str] = None

        # laid-out hit targets (rebuilt each render)
        self.items: List[Item] = []
        self._items_sig = None
        self.tab_rects: List[pygame.Rect] = []
        self.footer_rects: List[pygame.Rect] = []
        self._content_clip: Optional[pygame.Rect] = None
        self._max_scroll = 0.0

        # animation
        self.anim = 1.0               # screen-entry slide (0->1)
        self.time = 0.0
        self.toast = ""
        self.toast_t = 0.0

    # ------------------------------------------------------------------ size
    def resize(self, w: int, h: int, scale: float, mode: str) -> None:
        self.w, self.h, self.mode = max(w, 16), max(h, 16), mode
        self.theme = Theme(scale)
        self.surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        try:
            img = pygame.image.load(assets.TITLE_IMAGE).convert_alpha()
            tw = int(self.w * (0.25 if mode == "desktop" else 0.74))
            th = max(1, int(img.get_height() * (tw / img.get_width())))
            self.title_img = pygame.transform.smoothscale(img, (tw, th))
        except Exception:
            self.title_img = None

    # ------------------------------------------------------------- lifecycle
    def open_main(self) -> None:
        self.screen = Screen.MAIN
        self.region = "list"
        self.list_focus = 0
        self.scroll = 0.0
        self.anim = 0.0

    def _go(self, screen: Screen) -> None:
        self.screen = screen
        self.list_focus = 0
        self.scroll = 0.0
        self.region = "list"
        self.anim = 0.0
        self.capturing = None

    def update(self, dt: float) -> None:
        self.time += dt
        if self.anim < 1.0:
            self.anim = min(1.0, self.anim + dt / 0.22)
        if self.toast_t > 0:
            self.toast_t = max(0.0, self.toast_t - dt)

    def notify(self, msg: str) -> None:
        self.toast, self.toast_t = msg, 2.4

    # ---------------------------------------------------------- item building
    def _build_items(self) -> List[Item]:
        items: List[Item] = []
        if self.screen == Screen.MAIN:
            for e in MAIN_ENTRIES:
                items.append(Item("main", entry=e))
        elif self.screen == Screen.SETTINGS:
            for sp in st.specs_for(st.CATEGORIES[self.tab]):
                items.append(Item("spec", spec=sp, focusable=sp.kind != st.INFO))
        else:  # STATS / CREDITS
            items.append(Item("back"))
        self.items = items
        self._items_sig = (self.screen, self.tab)
        return items

    def _sync_items(self) -> None:
        """Rebuild the logical item list when the screen/tab changes.

        Items keep their laid-out rects across frames (mouse hit-testing uses last frame's
        layout); we only recreate the list — and reset rects — when the screen actually
        changes, never on every keystroke.
        """
        if not self.items or getattr(self, "_items_sig", None) != (self.screen, self.tab):
            self._build_items()

    def _focusables(self) -> List[int]:
        return [i for i, it in enumerate(self.items) if it.focusable]

    # ------------------------------------------------------------ navigation
    def nav_vertical(self, d: int) -> None:
        self._sync_items()
        self.capturing = None
        if self.screen != Screen.SETTINGS:
            foc = self._focusables()
            if foc:
                cur = foc.index(self.list_focus) if self.list_focus in foc else 0
                self.list_focus = foc[clampi(cur + d, 0, len(foc) - 1)]
            return
        # settings: tabs <-> list <-> footer
        foc = self._focusables()
        if self.region == "tabs":
            if d > 0 and foc:
                self.region, self.list_focus = "list", foc[0]
        elif self.region == "list":
            if not foc:
                self.region = "footer"
                return
            cur = foc.index(self.list_focus) if self.list_focus in foc else 0
            nxt = cur + d
            if nxt < 0:
                self.region = "tabs"
            elif nxt >= len(foc):
                self.region = "footer"
            else:
                self.list_focus = foc[nxt]
            self._ensure_visible()
        elif self.region == "footer":
            if d < 0 and foc:
                self.region, self.list_focus = "list", foc[-1]
                self._ensure_visible()

    def nav_horizontal(self, d: int) -> None:
        self._sync_items()
        if self.screen != Screen.SETTINGS:
            return
        if self.region == "tabs":
            self.tab = (self.tab + d) % len(st.CATEGORIES)
            self.list_focus = 0
            self.scroll = 0.0
        elif self.region == "footer":
            self.footer_focus = clampi(self.footer_focus + d, 0, len(FOOTER) - 1)
        elif self.region == "list":
            self._adjust_focused(d)

    def next_tab(self, d: int) -> None:
        if self.screen != Screen.SETTINGS:
            return
        self.tab = (self.tab + d) % len(st.CATEGORIES)
        self.region = "list"
        self.list_focus = 0
        self.scroll = 0.0
        self._sync_items()

    def _adjust_focused(self, d: int) -> None:
        it = self.items[self.list_focus] if 0 <= self.list_focus < len(self.items) else None
        if it is None or it.spec is None:
            return
        sp = it.spec
        if sp.kind == st.TOGGLE:
            self.cfg.set(sp.key, not self.cfg.get(sp.key))
            self._preview(sp)
        elif sp.kind == st.CHOICE:
            self.cfg.cycle_choice(sp.key, d)
            self._preview(sp)
        elif sp.kind == st.SLIDER:
            self.cfg.nudge_slider(sp.key, d)
            self._preview(sp)

    def activate(self) -> None:
        self._sync_items()
        if self.screen == Screen.MAIN:
            it = self.items[self.list_focus] if 0 <= self.list_focus < len(self.items) else None
            if it and it.entry:
                self._do_main(it.entry.action)
            return
        if self.screen in (Screen.STATS, Screen.CREDITS):
            self.open_main()
            return
        # settings
        if self.region == "tabs":
            foc = self._focusables()
            self.region, self.list_focus = "list", (foc[0] if foc else 0)
        elif self.region == "footer":
            self._do_footer(FOOTER[self.footer_focus][0])
        elif self.region == "list":
            self._activate_spec()

    def _activate_spec(self) -> None:
        it = self.items[self.list_focus] if 0 <= self.list_focus < len(self.items) else None
        if it is None or it.spec is None:
            return
        sp = it.spec
        if sp.kind == st.TOGGLE:
            self.cfg.set(sp.key, not self.cfg.get(sp.key))
            self._preview(sp)
        elif sp.kind == st.CHOICE:
            self.cfg.cycle_choice(sp.key, 1)
            self._preview(sp)
        elif sp.kind == st.SLIDER:
            self.cfg.nudge_slider(sp.key, 1)
            self._preview(sp)
        elif sp.kind == st.KEYBIND:
            self.capturing = sp.key
        elif sp.kind == st.ACTION:
            self._do_action(sp.action)

    def back(self) -> bool:
        """Go back one screen. Returns False if already at the main menu (caller decides)."""
        if self.capturing:
            self.capturing = None
            return True
        if self.screen != Screen.MAIN:
            if self.screen == Screen.SETTINGS and self.cfg.dirty:
                self.cfg.cancel()
                self.on_apply([])
            self.open_main()
            return True
        return False

    # ------------------------------------------------------------- actions
    def _do_main(self, action: str) -> None:
        if action == "settings":
            self._go(Screen.SETTINGS)
        elif action == "stats":
            self._go(Screen.STATS)
        elif action == "credits":
            self._go(Screen.CREDITS)
        else:
            payload = {
                "algo": self.cfg.get("ai_algorithm"),
                "speed": self.cfg.get("ai_speed"),
                "parallel": int(self.cfg.get("ai_parallel")),
                "seed": int(self.cfg.get("ai_seed")),
                "swarm": int(self.cfg.get("ai_swarm_size")),
                "depth": int(self.cfg.get("ai_minimax_depth")),
            }
            self.on_launch(action, payload)

    def _do_action(self, action: str) -> None:
        if action == "reset_controls":
            for k in ("key_up", "key_down", "key_left", "key_right", "key_pause"):
                self.cfg.set(k, st.SPEC_BY_KEY[k].default)
            self.notify("Controls reset to defaults")

    def _do_footer(self, fid: str) -> None:
        if fid == "apply":
            if self.cfg.dirty:
                changed = self.cfg.apply()
                self.on_apply(changed)
                self.notify("Settings applied" + (" — restart for some changes"
                            if any(st.SPEC_BY_KEY[k].needs_restart for k in changed) else ""))
        elif fid == "cancel":
            if self.cfg.dirty:
                self.cfg.cancel()
                self.on_apply([])           # restore live values to committed
                self.notify("Changes cancelled")
        elif fid == "defaults":
            self.cfg.reset_to_defaults()
            for sp in st.SPECS:
                if sp.kind not in (st.INFO, st.ACTION):
                    self._preview(sp)
            self.notify("Restored factory defaults (not yet saved)")
        elif fid == "backup":
            dst = self.cfg.backup()
            self.notify("Backed up settings" if dst else "Backup failed")
        elif fid == "export":
            path = st.CONFIG_PATH.replace(".json", "-export.json")
            ok = self.cfg.export_to(path)
            self.notify(f"Exported to {path}" if ok else "Export failed")
        elif fid == "import":
            path = st.CONFIG_PATH.replace(".json", "-export.json")
            if self.cfg.import_from(path):
                for sp in st.SPECS:
                    if sp.kind not in (st.INFO, st.ACTION):
                        self._preview(sp)
                self.notify("Imported settings (review, then Apply)")
            else:
                self.notify(f"No import file at {path}")

    def _preview(self, sp: st.Spec) -> None:
        """Live-preview a non-restart setting as it changes (and refocus capture off)."""
        self.capturing = None
        if not sp.needs_restart and sp.wired:
            self.on_preview(sp.key)

    # ------------------------------------------------------- key capture
    def capture_key(self, key_name: str) -> bool:
        """Feed a captured key name to the pending rebind. Returns True if consumed."""
        if not self.capturing:
            return False
        if key_name != "escape":
            self.cfg.set(self.capturing, key_name)
        self.capturing = None
        return True

    # ----------------------------------------------------------- mouse
    def mouse_move(self, pos) -> None:
        self._sync_items()
        if self._drag_key is not None:
            self._drag_slider(pos)
            return
        for i, it in enumerate(self.items):
            if it.focusable and it.rect.collidepoint(pos):
                if self.screen == Screen.SETTINGS:
                    self.region = "list"
                self.list_focus = i
                return

    def mouse_down(self, pos, button: int = 1) -> None:
        self._sync_items()
        if button in (4, 5):
            self.wheel(1 if button == 4 else -1)
            return
        # tabs
        for i, r in enumerate(self.tab_rects):
            if r.collidepoint(pos):
                self.tab, self.region, self.list_focus, self.scroll = i, "list", 0, 0.0
                return
        # footer
        for i, r in enumerate(self.footer_rects):
            if r.collidepoint(pos):
                self.region, self.footer_focus = "footer", i
                self._do_footer(FOOTER[i][0])
                return
        # rows
        for i, it in enumerate(self.items):
            if not it.rect.collidepoint(pos):
                continue
            self.list_focus = i
            if self.screen == Screen.SETTINGS:
                self.region = "list"
            if it.kind == "main" and it.entry:
                self._do_main(it.entry.action)
            elif it.kind == "back":
                self.open_main()
            elif it.spec is not None:
                self._click_spec(it, pos)
            return

    def _click_spec(self, it: Item, pos) -> None:
        sp = it.spec
        if sp.kind == st.SLIDER:
            track = W.slider_track_rect(it.rect, self.theme)
            hot = track.inflate(self.theme.px(18), self.theme.px(22))
            if hot.collidepoint(pos) or W.value_zone(it.rect).collidepoint(pos):
                self._drag_key = sp.key
                self._drag_slider(pos)
        elif sp.kind == st.CHOICE:
            left, right = W.choice_arrow_rects(it.rect, self.theme)
            if left.collidepoint(pos):
                self.cfg.cycle_choice(sp.key, -1); self._preview(sp)
            else:
                self.cfg.cycle_choice(sp.key, 1); self._preview(sp)
        elif sp.kind == st.TOGGLE:
            self.cfg.set(sp.key, not self.cfg.get(sp.key)); self._preview(sp)
        elif sp.kind == st.KEYBIND:
            self.capturing = sp.key
        elif sp.kind == st.ACTION:
            self._do_action(sp.action)

    def _drag_slider(self, pos) -> None:
        sp = st.SPEC_BY_KEY.get(self._drag_key)
        it = next((x for x in self.items if x.spec is sp), None)
        if sp is None or it is None:
            return
        track = W.slider_track_rect(it.rect, self.theme)
        frac = clamp((pos[0] - track.x) / max(1, track.w), 0.0, 1.0)
        value = sp.lo + frac * (sp.hi - sp.lo)
        self.cfg.set(sp.key, value)
        self._preview(sp)

    def mouse_up(self, pos) -> None:
        self._drag_key = None

    def wheel(self, dy: int) -> None:
        if self._max_scroll > 0:
            self.scroll = clamp(self.scroll - dy * self.theme.px(48), 0.0, self._max_scroll)

    def _ensure_visible(self) -> None:
        if self._content_clip is None or not (0 <= self.list_focus < len(self.items)):
            return
        r = self.items[self.list_focus].rect
        clip = self._content_clip
        if r.top < clip.top:
            self.scroll -= (clip.top - r.top)
        elif r.bottom > clip.bottom:
            self.scroll += (r.bottom - clip.bottom)
        self.scroll = clamp(self.scroll, 0.0, max(0.0, self._max_scroll))

    # ============================================================ RENDERING
    def render(self) -> pygame.Surface:
        self.surface.fill((0, 0, 0, 0))
        # Darken the live 3D backdrop so text is legible.
        W.scrim(self.surface, (0, 0, self.w, self.h), self.theme.scrim)
        self._sync_items()
        if self.screen == Screen.MAIN:
            self._render_main()
        elif self.screen == Screen.SETTINGS:
            self._render_settings()
        elif self.screen == Screen.STATS:
            self._render_stats()
        elif self.screen == Screen.CREDITS:
            self._render_credits()
        self._render_toast()
        return self.surface

    def to_bytes(self) -> bytes:
        return pygame.image.tostring(self.surface, "RGBA", True)

    # ---- shared chrome ---------------------------------------------------
    def _slide(self) -> int:
        return int((1.0 - _ease(self.anim)) * self.theme.px(36))

    def _header(self, title: str, area: pygame.Rect) -> None:
        t = self.theme
        text = W.text(self.surface, t.f_title, title, (area.x, area.y), t.text)
        pygame.draw.line(self.surface, t.divider, (area.x, text.bottom + t.px(6)),
                         (area.right, text.bottom + t.px(6)), t.px(2))

    def _render_toast(self) -> None:
        if self.toast_t <= 0:
            return
        t = self.theme
        a = clamp(self.toast_t / 0.5, 0.0, 1.0)
        img = t.f_small.render(self.toast, True, t.text)
        pad = t.px(12)
        box = pygame.Rect(0, 0, img.get_width() + pad * 2, img.get_height() + pad)
        box.midbottom = (self.w // 2, self.h - t.px(16))
        W.panel(self.surface, box, t.accent, t.px(8), alpha=int(235 * a))
        self.surface.blit(img, img.get_rect(center=box.center))

    # ---- MAIN ------------------------------------------------------------
    def _render_main(self) -> None:
        t = self.theme
        desktop = self.mode == "desktop"
        if desktop:
            panel_w = int(self.w * 0.42)
            W.scrim(self.surface, (0, 0, panel_w, self.h), (8, 11, 18, 130))
            x = t.px(40)
            aw = panel_w - t.px(80)
            ty = int(self.h * 0.06)
        else:
            x = t.px(24)
            aw = self.w - t.px(48)
            ty = int(self.h * 0.05)

        title_bottom = ty
        if self.title_img:
            tx = x if desktop else (self.w - self.title_img.get_width()) // 2
            self.surface.blit(self.title_img, (tx, ty))
            title_bottom = ty + self.title_img.get_height()

        hint_y = self.h - t.px(22)
        top = title_bottom + t.px(22)
        bottom = hint_y - t.px(10)
        n = max(1, len(self.items))
        gap = t.px(8)
        row_h = max(t.px(38), min(t.px(62), int((bottom - top) / n) - gap))

        slide = self._slide()
        for idx, it in enumerate(self.items):
            if it.entry is None:                     # defensive: MAIN rows always carry an entry
                continue
            r = pygame.Rect(x - slide, top + idx * (row_h + gap), aw, row_h)
            it.rect = r
            focused = idx == self.list_focus
            W.row_background(self.surface, r, t, focused, False)
            color = (255, 255, 255) if focused else t.text
            show_sub = desktop and row_h >= t.px(44) and it.entry.subtitle
            if show_sub:
                W.text(self.surface, t.f_head, it.entry.label,
                       (r.x + t.px(14), r.y + t.px(7)), color)
                W.text(self.surface, t.f_tiny, it.entry.subtitle,
                       (r.x + t.px(14), r.bottom - t.px(8)), t.text_dim, anchor="bottomleft")
            else:
                W.text(self.surface, t.f_head, it.entry.label,
                       (r.x + t.px(14), r.centery), color, anchor="midleft")
            if focused:
                W.text(self.surface, t.f_item_b, "›", (r.right - t.px(16), r.centery),
                       t.accent_soft, anchor="midright")

        hint = "Arrows  move    Enter  select    Esc  back"
        W.text(self.surface, t.f_tiny, hint, (x, hint_y), t.text_mute, anchor="midleft")
        if desktop and 0 <= self.list_focus < len(self.items):
            e = self.items[self.list_focus].entry
            if e and e.tooltip:
                tip = pygame.Rect(int(self.w * 0.46), int(self.h * 0.76),
                                  int(self.w * 0.50), int(self.h * 0.20))
                W.draw_tooltip(self.surface, tip, t, e.label, e.tooltip)

    # ---- SETTINGS --------------------------------------------------------
    def _render_settings(self) -> None:
        t = self.theme
        desktop = self.mode == "desktop"
        margin = t.px(28) if desktop else t.px(16)
        footer_h = t.px(54)
        header_h = t.px(58)

        # header
        W.text(self.surface, t.f_title, "Settings", (margin, t.px(18)), t.text)
        if self.cfg.dirty:
            W.draw_badge(self.surface, (self.w - margin, t.px(26)), t,
                         "UNSAVED", t.warn, anchor="topright")

        if desktop:
            rail_w = int(self.w * 0.24)
            rail = pygame.Rect(margin, header_h, rail_w, self.h - header_h - footer_h - t.px(12))
            self._render_rail_vertical(rail)
            content = pygame.Rect(rail.right + t.px(20), header_h,
                                  self.w - rail.right - t.px(20) - margin,
                                  rail.h)
        else:
            chip = pygame.Rect(margin, header_h - t.px(8), self.w - margin * 2, t.px(34))
            self._render_rail_chips(chip)
            content = pygame.Rect(margin, chip.bottom + t.px(8),
                                  self.w - margin * 2,
                                  self.h - chip.bottom - footer_h - t.px(16))

        self._render_rows(content)
        self._render_footer(footer_h)
        self._render_settings_tooltip(content, desktop)

    def _render_rail_vertical(self, rail: pygame.Rect) -> None:
        t = self.theme
        W.panel(self.surface, rail, t.rail, t.px(10), alpha=235)
        self.tab_rects = []
        ih = t.px(40)
        y = rail.y + t.px(8)
        for i, cat in enumerate(st.CATEGORIES):
            r = pygame.Rect(rail.x + t.px(6), y, rail.w - t.px(12), ih)
            self.tab_rects.append(r)
            active = i == self.tab
            if active:
                W.panel(self.surface, r, t.row_focus, t.px(8))
            elif self.region == "tabs" and i == self.tab:
                W.panel(self.surface, r, t.row_hover, t.px(8))
            col = (255, 255, 255) if active else t.text_dim
            W.text(self.surface, t.f_item, cat, (r.x + t.px(12), r.centery), col,
                   anchor="midleft")
            y += ih + t.px(4)

    def _render_rail_chips(self, bar: pygame.Rect) -> None:
        t = self.theme
        self.tab_rects = []
        x = bar.x
        gap = t.px(6)
        for i, cat in enumerate(st.CATEGORIES):
            label = cat
            w = t.f_small.size(label)[0] + t.px(20)
            r = pygame.Rect(x, bar.y, w, bar.h)
            self.tab_rects.append(r)
            active = i == self.tab
            W.panel(self.surface, r, t.row_focus if active else t.panel, t.px(8))
            W.text(self.surface, t.f_small, label, r.center,
                   (255, 255, 255) if active else t.text_dim, anchor="center")
            x += w + gap

    def _render_rows(self, content: pygame.Rect) -> None:
        t = self.theme
        self._content_clip = content
        row_h = t.px(44)
        gap = t.px(6)
        total = len(self.items) * (row_h + gap)
        self._max_scroll = max(0.0, total - content.h)
        self.scroll = clamp(self.scroll, 0.0, self._max_scroll)

        prev_clip = self.surface.get_clip()
        self.surface.set_clip(content)
        slide = self._slide()
        y0 = content.y - int(self.scroll)
        for idx, it in enumerate(self.items):
            r = pygame.Rect(content.x - slide, y0 + idx * (row_h + gap), content.w, row_h)
            it.rect = r
            if r.bottom < content.y or r.top > content.bottom:
                continue
            self._render_spec_row(it, idx)
        self.surface.set_clip(prev_clip)

        # scrollbar
        if self._max_scroll > 0:
            track_h = content.h
            knob_h = max(t.px(24), int(track_h * content.h / total))
            knob_y = content.y + int((track_h - knob_h) * self.scroll / self._max_scroll)
            bar = pygame.Rect(content.right + t.px(4), knob_y, t.px(4), knob_h)
            W.panel(self.surface, bar, t.divider, t.px(2))

    def _render_spec_row(self, it: Item, idx: int) -> None:
        t = self.theme
        sp, r = it.spec, it.rect
        focused = (self.region == "list" and idx == self.list_focus)
        if sp.kind != st.INFO:
            W.row_background(self.surface, r, t, focused, False)
        W.draw_label(self.surface, r, t, sp.label, focused, enabled=sp.wired or sp.kind == st.INFO)

        val = self.cfg.get(sp.key) if sp.kind not in (st.INFO, st.ACTION) else None
        if sp.kind == st.TOGGLE:
            W.draw_toggle(self.surface, r, t, bool(val), focused)
        elif sp.kind == st.CHOICE:
            W.draw_choice(self.surface, r, t, sp.choice_label(val), focused)
        elif sp.kind == st.SLIDER:
            frac = (val - sp.lo) / (sp.hi - sp.lo) if sp.hi > sp.lo else 0
            W.draw_slider(self.surface, r, t, frac, self._slider_text(sp, val), focused)
        elif sp.kind == st.KEYBIND:
            W.draw_keybind(self.surface, r, t, _key_display(val), focused,
                           self.capturing == sp.key)
        elif sp.kind == st.ACTION:
            W.draw_action(self.surface, r, t, focused)
        elif sp.kind == st.INFO:
            W.draw_info(self.surface, r, t, self.info_provider().get(sp.key, "—"))

        if not sp.wired and sp.kind not in (st.INFO,):
            W.draw_badge(self.surface, (r.x + t.f_item.size(sp.label)[0] + t.px(26), r.centery),
                         t, "N/A", t.text_mute, anchor="midleft")
        elif sp.needs_restart:
            W.draw_badge(self.surface, (r.x + t.f_item.size(sp.label)[0] + t.px(26), r.centery),
                         t, "RESTART", t.warn, anchor="midleft")

    def _slider_text(self, sp: st.Spec, val) -> str:
        if sp.key == "ai_parallel" and int(val) == 0:
            return "Off"
        return f"{val}{sp.unit}"

    def _render_footer(self, footer_h: int) -> None:
        t = self.theme
        bar = pygame.Rect(0, self.h - footer_h, self.w, footer_h)
        W.panel(self.surface, bar, t.rail, 0, alpha=240)
        pygame.draw.line(self.surface, t.divider, bar.topleft, bar.topright, t.px(1))
        self.footer_rects = []
        margin = t.px(20)
        bw = min(t.px(120), (self.w - margin * 2) // len(FOOTER) - t.px(8))
        bh = footer_h - t.px(16)
        x = self.w - margin - bw
        for i in reversed(range(len(FOOTER))):
            fid, label = FOOTER[i]
            r = pygame.Rect(x, bar.y + (footer_h - bh) // 2, bw, bh)
            self.footer_rects.insert(0, r)
            enabled = self._footer_enabled(fid)
            focused = self.region == "footer" and self.footer_focus == i
            if fid == "apply":
                base = t.accent if enabled else t.panel
            elif fid == "cancel":
                base = t.danger if (enabled and focused) else (t.panel_alt if enabled else t.panel)
            else:
                base = t.panel_alt
            W.panel(self.surface, r, base, t.px(8),
                    border=t.accent_soft if focused else None, border_w=t.px(2))
            col = t.text if enabled else t.text_mute
            W.text(self.surface, t.f_small, label, r.center, col, anchor="center")
            x -= bw + t.px(8)

    def _footer_enabled(self, fid: str) -> bool:
        if fid in ("apply", "cancel"):
            return self.cfg.dirty
        return True

    def _render_settings_tooltip(self, content: pygame.Rect, desktop: bool) -> None:
        if self.region != "list" or not (0 <= self.list_focus < len(self.items)):
            return
        sp = self.items[self.list_focus].spec
        if sp is None:
            return
        body = sp.tooltip
        if not sp.wired and sp.note:
            body = f"{sp.tooltip}  ({sp.note})" if sp.tooltip else sp.note
        if not body:
            return
        t = self.theme
        area = pygame.Rect(content.x, content.bottom - t.px(4),
                           content.w, t.px(64))
        # draw just above the footer
        area.bottom = self.h - t.px(58)
        W.draw_tooltip(self.surface, pygame.Rect(area.x, area.y, area.w, t.px(70)), t,
                       sp.label, body)

    # ---- STATS -----------------------------------------------------------
    def _render_stats(self) -> None:
        t = self.theme
        margin = t.px(28) if self.mode == "desktop" else t.px(18)
        area = pygame.Rect(margin, t.px(18), self.w - margin * 2, self.h)
        self._header("Statistics", pygame.Rect(area.x, area.y, area.w, t.px(40)))
        sv = self.save
        avg = sv.average_score
        mins = sv.play_seconds / 60.0
        rows = [
            ("High Score", str(sv.highscore)),
            ("Games Played", str(sv.games_played)),
            ("Average Score", f"{avg:.1f}"),
            ("Total Score", str(sv.total_score)),
            ("Time Played", f"{mins:.1f} min"),
        ]
        for char, best in sorted(sv.best_by_character.items(),
                                 key=lambda kv: -kv[1])[:4]:
            rows.append((f"Best · {char}", str(best)))

        y = area.y + t.px(58)
        slide = self._slide()
        rh = t.px(40)
        cardw = min(area.w, t.px(520))
        for label, value in rows:
            r = pygame.Rect(area.x - slide, y, cardw, rh - t.px(6))
            W.panel(self.surface, r, t.panel, t.px(8), alpha=225)
            W.text(self.surface, t.f_item, label, (r.x + t.px(14), r.centery), t.text_dim,
                   anchor="midleft")
            W.text(self.surface, t.f_item_b, value, (r.right - t.px(14), r.centery), t.text,
                   anchor="midright")
            y += rh
        self._render_back_hint()

    # ---- CREDITS ---------------------------------------------------------
    def _render_credits(self) -> None:
        t = self.theme
        margin = t.px(28) if self.mode == "desktop" else t.px(18)
        area = pygame.Rect(margin, t.px(18), self.w - margin * 2, self.h)
        self._header("Credits", pygame.Rect(area.x, area.y, area.w, t.px(40)))
        lines = [
            ("PyCrossy", "A GPU-rendered arcade hopper in Python.", t.accent_soft),
            ("Assets", "Art, audio & fonts © Evan Bacon, MIT.", t.text),
            ("", "See assets/ATTRIBUTION.md for full credits,", t.text_dim),
            ("", "with thanks to their authors.", t.text_dim),
            ("Engine", "pygame-ce · moderngl · OpenGL · NumPy", t.text),
            ("AI framework", "NEAT · PPO · A2C · DQN · ES · GA · CMA-ES (pure NumPy)", t.text),
        ]
        y = area.y + t.px(64)
        slide = self._slide()
        for head, body, color in lines:
            if head:
                W.text(self.surface, t.f_head, head, (area.x - slide, y), t.accent_soft)
                y += t.px(26)
            if body:
                W.text(self.surface, t.f_small, body, (area.x - slide, y), color)
                y += t.px(26)
            if not head and not body:
                y += t.px(12)
        self._render_back_hint()

    def _render_back_hint(self) -> None:
        t = self.theme
        self.items[0].rect = pygame.Rect(t.px(20), self.h - t.px(40),
                                         t.px(160), t.px(30))
        W.text(self.surface, t.f_small, "‹ Back (Esc)", (t.px(24), self.h - t.px(25)),
               t.accent_soft, anchor="midleft")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def clampi(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) * (1 - t)        # ease-out quad


def _key_display(name: str) -> str:
    pretty = {"up": "UP", "down": "DOWN", "left": "LEFT", "right": "RIGHT",
              "space": "SPACE", "return": "ENTER", "escape": "ESC",
              "left shift": "L-SHIFT", "right shift": "R-SHIFT"}
    return pretty.get(name, (name or "—").upper())
