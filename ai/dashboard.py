"""Live training dashboard — a professional multi-panel ML experiment view.

Runs in its own process (its own pygame window) consuming metric snapshots from a queue.
Dark theme, continuously updating. Panels: a header with an episode progress bar and live
speed/FPS, a grouped live-stats column (training / performance / algorithm / checkpoint),
and a 3×2 grid of graphs (reward, score timeline, loss-or-fitness, evaluation, population
diversity, success rate). Mouse-wheel zooms the time window, ``E`` exports high-res
matplotlib plots + a screenshot, ``P`` pauses, ``Esc`` closes.

Rendering is flicker-free and decoupled: every frame is composed onto an off-screen
back-buffer and blitted once to a double-buffered window, the event loop stays responsive
at ~60 Hz, and the expensive graph redraw is throttled to a configurable rate (default
12 Hz, 5–20 recommended) — so the game window keeps running at full speed independently.
"""
from __future__ import annotations

import os
import queue as queuelib
import time
from collections import deque
from typing import Dict, List, Optional, Sequence

import pygame

from pycrossy import gpu

BG = (15, 17, 23)
HEADER = (22, 25, 34)
PANEL = (26, 30, 40)
PANEL_HEAD = (34, 39, 52)
GRID = (42, 47, 62)
TEXT = (224, 228, 238)
MUTED = (135, 142, 162)
ACCENT = (0, 200, 255)
GOOD = (90, 222, 140)
WARN = (255, 198, 70)
BAD = (255, 95, 110)
PURPLE = (181, 132, 255)
TEAL = (60, 220, 200)


def _fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v != v:  # NaN
            return "—"
        if abs(v) >= 10000:
            return f"{v:,.0f}"
        return f"{v:.{nd}f}"
    return str(v)


def _fmt_time(s: Optional[float]) -> str:
    if s is None:
        return "—"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


class Dashboard:
    def __init__(self, width=1280, height=780, logdir="runs/latest", refresh_hz: float = 12.0):
        self.width, self.height = width, height
        self.logdir = logdir
        self.refresh_interval = 1.0 / max(1.0, min(refresh_hz, 60.0))
        # Create the window exactly ONCE. We then NEVER call set_mode again — on Wayland the
        # display surface auto-tracks the compositor-given size, and calling set_mode in
        # response to a resize creates a feedback loop (set_mode -> reconfigure -> resize ->
        # set_mode ...) that makes tiling compositors (Hyprland/Sway) flash the window
        # open/closed. Instead we render the layout at the *actual* window size each frame,
        # so the dashboard is fully responsive and fills its tile with no letterbox.
        self.screen = gpu.safe_set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("PyCrossy — Live Training Dashboard")
        self.width, self.height = pygame.display.get_window_size()
        # Off-screen back-buffer: the whole frame is composed here then blitted once, so the
        # window is never seen mid-clear (no flicker).
        self.surf = pygame.Surface((self.width, self.height)).convert()
        self.f_title = pygame.font.SysFont("monospace", 22, bold=True)
        self.f_h = pygame.font.SysFont("monospace", 14, bold=True)
        self.f = pygame.font.SysFont("monospace", 14)
        self.f_s = pygame.font.SysFont("monospace", 12)
        self.snap: Dict = {}
        self.window = 0
        self.paused = False
        self.running = True
        self.clock = pygame.time.Clock()
        self._success_hist: deque = deque(maxlen=4000)
        self._diversity_hist: deque = deque(maxlen=4000)
        self._dirty = True
        self._last_draw = 0.0
        self._draw_fps = 0.0

    # -- input -------------------------------------------------------------
    def handle_events(self) -> None:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.VIDEORESIZE:
                self._dirty = True   # actual resize handled by _sync_size (no set_mode)
            elif e.type == pygame.MOUSEWHEEL:
                base = self.window or len(self.snap.get("rewards", [])) or 200
                self.window = max(20, int(base * (0.8 if e.y > 0 else 1.25)))
                self._dirty = True
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.running = False
                elif e.key == pygame.K_p:
                    self.paused = not self.paused
                    self._dirty = True
                elif e.key == pygame.K_e:
                    self.export()

    def update_data(self, snap: Dict) -> None:
        if self.paused:
            return
        self.snap = snap
        self._success_hist.append(snap.get("success_rate", 0.0))
        div = snap.get("series", {}).get("diversity")
        self._diversity_hist.append(div[-1] if div else 0.0)
        self._dirty = True

    # -- primitives (draw onto the off-screen back-buffer) -----------------
    def _text(self, s, x, y, font=None, color=TEXT):
        self.surf.blit((font or self.f).render(s, True, color), (x, y))

    def _slice(self, data: Sequence[float]) -> List[float]:
        return list(data[-self.window:]) if (self.window and len(data) > self.window) else list(data)

    def _panel(self, rect, title=None):
        pygame.draw.rect(self.surf, PANEL, rect, border_radius=8)
        if title:
            head = pygame.Rect(rect[0], rect[1], rect[2], 24)
            pygame.draw.rect(self.surf, PANEL_HEAD, head, border_top_left_radius=8,
                             border_top_right_radius=8)
            self._text(title, rect[0] + 10, rect[1] + 5, self.f_s, MUTED)

    def _graph(self, rect, title, series, fmt_nd=1):
        x, y, w, h = rect
        self._panel(rect, title)
        px, py, pw, ph = x + 12, y + 32, w - 24, h - 50
        for gy in range(5):
            ly = py + int(ph * gy / 4)
            pygame.draw.line(self.surf, GRID, (px, ly), (px + pw, ly), 1)
        vals = [v for _, s, _ in series for v in self._slice(s)]
        if not vals:
            self._text("waiting for data…", px + 8, py + ph // 2, self.f_s, MUTED)
            return
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            hi, lo = hi + 1, lo - 1
        self._text(_fmt(hi, fmt_nd), px + pw - 56, py - 2, self.f_s, MUTED)
        self._text(_fmt(lo, fmt_nd), px + pw - 56, py + ph - 12, self.f_s, MUTED)
        lx = px + 4
        for name, data, color in series:
            d = self._slice(data)
            if len(d) >= 2:
                n = len(d)
                pts = [(px + int(pw * i / (n - 1)), py + ph - int(ph * (v - lo) / (hi - lo)))
                       for i, v in enumerate(d)]
                pygame.draw.lines(self.surf, color, False, pts, 2)
            self._text(name, lx, py + ph + 5, self.f_s, color)
            lx += len(name) * 7 + 14

    def _section(self, x, y, w, title, rows) -> int:
        h = 30 + len(rows) * 21
        self._panel((x, y, w, h), title)
        ry = y + 30
        vx = x + int(w * 0.52)
        for label, val, color in rows:
            self._text(label, x + 10, ry, self.f_s, MUTED)
            self._text(_fmt(val), vx, ry, self.f, color)
            ry += 21
        return y + h + 10

    def _progress_bar(self, x, y, w, frac, color):
        pygame.draw.rect(self.surf, GRID, (x, y, w, 8), border_radius=4)
        pygame.draw.rect(self.surf, color, (x, y, int(w * max(0, min(1, frac))), 8),
                         border_radius=4)

    # -- compose the frame onto the back-buffer ----------------------------
    def draw(self) -> None:
        self.surf.fill(BG)
        s = self.snap
        latest = s.get("latest", {})

        # ----- header -----
        pygame.draw.rect(self.surf, HEADER, (0, 0, self.width, 54))
        self._text("PyCrossy Trainer", 14, 8, self.f_title, ACCENT)
        self._text(f"algo: {s.get('algo', '—')}", 240, 6, self.f_h, GOOD)
        ep = s.get("episode", 0)
        tgt = s.get("target_episodes") or 0
        self._text(f"episode {ep}/{tgt or '∞'}", 240, 26, self.f_s, TEXT)
        if tgt:
            self._progress_bar(360, 30, 180, ep / tgt, ACCENT)
        self._text(f"UI {self._draw_fps:4.0f}Hz", self.width - 360, 6, self.f_s, MUTED)
        self._text(f"{s.get('episodes_per_sec', 0):.1f} ep/s", self.width - 360, 26, self.f_s, MUTED)
        self._text(f"{s.get('steps_per_sec', 0):,.0f} steps/s", self.width - 250, 6, self.f_s, MUTED)
        self._text(f"elapsed {_fmt_time(s.get('elapsed'))}", self.width - 250, 26, self.f_s, MUTED)
        self._text(f"ETA {_fmt_time(s.get('eta'))}", self.width - 110, 26, self.f_s, WARN)

        # ----- stats column (width adapts to the window) -----
        col_x = 12
        col_w = max(230, min(340, int(self.width * 0.22)))
        y = 64
        y = self._section(col_x, y, col_w, "TRAINING", [
            ("Total steps", s.get("total_steps", 0), TEXT),
            ("Episodes/s", _fmt(s.get("episodes_per_sec"), 2), TEXT),
            ("Steps/s", _fmt(s.get("steps_per_sec"), 0), TEXT),
            ("Elapsed", _fmt_time(s.get("elapsed")), TEXT),
            ("ETA", _fmt_time(s.get("eta")), WARN),
        ])
        y = self._section(col_x, y, col_w, "PERFORMANCE", [
            ("Best score", s.get("best_score", 0), GOOD),
            ("Best reward", _fmt(s.get("best_reward"), 1), GOOD),
            ("Success rate", f"{s.get('success_rate', 0) * 100:.0f}%  (≥{s.get('success_threshold', 5)})", TEAL),
            ("Deaths", s.get("death_count", 0), BAD),
        ])
        algo_rows = []
        for key, label, color in [
            ("generation", "Generation", ACCENT), ("species", "Species", ACCENT),
            ("genome_count", "Genomes", ACCENT), ("num_nodes", "Net nodes", MUTED),
            ("num_conns", "Net conns", MUTED), ("mutation_rate", "Mutation", MUTED),
            ("diversity", "Diversity", PURPLE), ("sigma", "Sigma", MUTED),
            ("best_fitness", "Best fit", GOOD), ("mean_fitness", "Mean fit", TEXT),
            ("policy_loss", "Policy loss", WARN), ("value_loss", "Value loss", WARN),
            ("loss", "TD loss", WARN), ("entropy", "Entropy", PURPLE),
            ("lr", "LR", MUTED), ("epsilon", "Epsilon", MUTED),
        ]:
            if key in latest:
                algo_rows.append((label, latest[key], color))
        if algo_rows:
            y = self._section(col_x, y, col_w, "ALGORITHM", algo_rows[:8])
        ck = s.get("checkpoint_info", {})
        self._section(col_x, y, col_w, "CHECKPOINT", [
            ("Last ckpt ep", ck.get("last_checkpoint_ep", "—"), TEXT),
            ("Best model ep", ck.get("best_model_ep", "—"), GOOD),
            ("Best eval", ck.get("best_eval", "—"), GOOD),
        ])

        # ----- 3x2 graph grid -----
        gx0 = col_x + col_w + 14
        gw_total = self.width - gx0 - 12
        gh = (self.height - 64 - 30) // 2 - 8
        cw = gw_total // 3 - 8
        cols = [gx0, gx0 + cw + 12, gx0 + 2 * (cw + 12)]
        r0, r1 = 64, 64 + gh + 14

        self._graph((cols[0], r0, cw, gh), "Reward (raw + avg100)", [
            ("reward", s.get("rewards", []), (70, 90, 120)),
            ("avg100", s.get("moving_reward", []), GOOD)])
        self._graph((cols[1], r0, cw, gh), "Score / episode", [
            ("score", s.get("scores", []), ACCENT)], fmt_nd=0)

        ser = s.get("series", {})
        if "best_fitness" in ser:
            self._graph((cols[2], r0, cw, gh), "Fitness (best / mean)", [
                ("best", ser.get("best_fitness", []), GOOD),
                ("mean", ser.get("mean_fitness", []), ACCENT)])
        elif "policy_loss" in ser:
            self._graph((cols[2], r0, cw, gh), "Loss (policy / value)", [
                ("policy", ser.get("policy_loss", []), WARN),
                ("value", ser.get("value_loss", []), BAD)])
        elif "loss" in ser:
            self._graph((cols[2], r0, cw, gh), "TD loss", [("loss", ser.get("loss", []), WARN)])
        else:
            self._graph((cols[2], r0, cw, gh), "—", [])

        self._graph((cols[0], r1, cw, gh), "Evaluation score",
                    [("eval", s.get("eval_scores", []), GOOD)], fmt_nd=1)
        self._graph((cols[1], r1, cw, gh), "Population diversity",
                    [("diversity", list(self._diversity_hist), PURPLE)], fmt_nd=2)
        self._graph((cols[2], r1, cw, gh), "Success rate",
                    [("success", [v * 100 for v in self._success_hist], TEAL)], fmt_nd=0)

        hint = "wheel: zoom   P: pause   E: export   Esc: close"
        if self.paused:
            hint = "[PAUSED]  " + hint
        self._text(hint, gx0, self.height - 20, self.f_s, MUTED)

    def _sync_size(self) -> None:
        """Track the compositor-given window size (no set_mode) and resize the back-buffer."""
        w, h = pygame.display.get_window_size()
        w, h = max(w, 320), max(h, 240)
        if (w, h) != (self.width, self.height):
            self.width, self.height = w, h
            self.surf = pygame.Surface((w, h)).convert()
            self._dirty = True

    def present(self) -> None:
        """Blit the composed back-buffer onto the (auto-tracking) window surface, once."""
        disp = pygame.display.get_surface() or self.screen
        if disp.get_size() == self.surf.get_size():
            disp.blit(self.surf, (0, 0))
        else:
            pygame.transform.scale(self.surf, disp.get_size(), disp)
        pygame.display.flip()

    def maybe_redraw(self, now: float) -> None:
        """Recompose the back-buffer only when data/UI changed or the throttle elapses."""
        if self._dirty or (now - self._last_draw) >= self.refresh_interval:
            self.draw()
            dt = now - self._last_draw
            self._draw_fps = (1.0 / dt) if dt > 0 else 0.0
            self._last_draw = now
            self._dirty = False

    # -- export ------------------------------------------------------------
    def export(self) -> None:
        plots = os.path.join(self.logdir, "plots")
        os.makedirs(plots, exist_ok=True)
        pygame.image.save(self.surf, os.path.join(plots, "dashboard.png"))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            s = self.snap
            fig, ax = plt.subplots(2, 3, figsize=(16, 9))
            ax[0, 0].plot(s.get("rewards", []), alpha=0.35, label="reward")
            ax[0, 0].plot(s.get("moving_reward", []), label="avg100")
            ax[0, 0].set_title("Reward"); ax[0, 0].legend()
            ax[0, 1].plot(s.get("scores", []), color="tab:cyan"); ax[0, 1].set_title("Score")
            ser = s.get("series", {})
            for key in ("best_fitness", "mean_fitness", "loss", "policy_loss", "value_loss"):
                if key in ser:
                    ax[0, 2].plot(ser[key], label=key)
            ax[0, 2].set_title("Algorithm"); ax[0, 2].legend()
            ax[1, 0].plot(s.get("eval_scores", []), color="tab:green"); ax[1, 0].set_title("Eval")
            ax[1, 1].plot(list(self._diversity_hist), color="tab:purple"); ax[1, 1].set_title("Diversity")
            ax[1, 2].plot([v * 100 for v in self._success_hist], color="tab:olive")
            ax[1, 2].set_title("Success rate %")
            fig.tight_layout()
            fig.savefig(os.path.join(plots, "report.png"), dpi=120)
            plt.close(fig)
        except Exception as exc:  # pragma: no cover
            print("export plot failed:", exc)


def run_dashboard(q, meta: Dict) -> None:
    gpu.prefer_high_performance_gpu(False)   # native video backend (Wayland); no dGPU needed
    pygame.init()
    pygame.font.init()
    dash = Dashboard(width=meta.get("width", 1280), height=meta.get("height", 780),
                     logdir=meta.get("logdir", "runs/latest"),
                     refresh_hz=meta.get("refresh_hz", 12.0))
    dash.draw()
    while dash.running:
        # Drain the queue to the most recent snapshot (thread/process-safe).
        latest = None
        try:
            while True:
                item = q.get_nowait()
                if item == "STOP":
                    dash.running = False
                    break
                latest = item
        except queuelib.Empty:
            pass
        if latest is not None:
            dash.update_data(latest)
        dash.handle_events()
        dash._sync_size()                      # adopt the compositor window size (responsive)
        dash.maybe_redraw(time.monotonic())    # heavy redraw is throttled
        dash.present()                         # cheap blit+swap every loop -> responsive
        dash.clock.tick(60)
    pygame.quit()
