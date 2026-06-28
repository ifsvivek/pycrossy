"""Game settings — a schema-driven, corruption-safe configuration model.

This is the single source of truth for every user-configurable option. A flat list of
:class:`Spec` objects describes each setting (type, default, range/choices, tooltip,
whether it needs a restart, and whether it is actually wired to behaviour). The schema
*drives the UI* — the menu renders whatever is in ``SPECS``, so adding a setting here makes
it appear, persist and validate automatically.

The model is deliberately free of any pygame / rendering imports so it can be unit-tested
and loaded headlessly. Keybindings are stored as human-readable key *names* (e.g. ``"up"``,
``"space"``) and resolved to keycodes by the input layer.

Persistence is hardened against corruption: a missing, unreadable or malformed config file
falls back to safe defaults, and every individual value is validated/clamped on load, so a
single bad field can never crash the game or wipe the rest of the config.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Where config lives
# ---------------------------------------------------------------------------
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".pycrossy")
CONFIG_PATH = os.path.join(CONFIG_DIR, "settings.json")
BACKUP_DIR = os.path.join(CONFIG_DIR, "backups")

# Categories, in display order.
GRAPHICS = "Graphics"
AUDIO = "Audio"
GAMEPLAY = "Gameplay"
CONTROLS = "Controls"
PERFORMANCE = "Performance"
AI = "AI"
DEVELOPER = "Developer"

CATEGORIES: Tuple[str, ...] = (GRAPHICS, AUDIO, GAMEPLAY, CONTROLS, PERFORMANCE, AI, DEVELOPER)

# Widget kinds.
TOGGLE = "toggle"      # bool
CHOICE = "choice"      # one of choices (value, label)
SLIDER = "slider"      # numeric in [lo, hi] stepped
KEYBIND = "keybind"    # a key name string
INFO = "info"          # read-only display (no value)
ACTION = "action"      # a button that emits an action


@dataclass
class Spec:
    key: str
    label: str
    kind: str
    category: str
    default: Any = None
    choices: Tuple[Tuple[Any, str], ...] = ()       # (value, human label)
    lo: float = 0.0
    hi: float = 100.0
    step: float = 1.0
    unit: str = ""
    tooltip: str = ""
    needs_restart: bool = False
    wired: bool = True
    note: str = ""            # shown when not wired (honest fallback note)
    action: str = ""          # for ACTION kind: the action name emitted

    def clamp(self, value: Any) -> Any:
        """Coerce ``value`` to a valid value for this spec (used on load + on set)."""
        if self.kind == TOGGLE:
            return bool(value)
        if self.kind == SLIDER:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return self.default
            v = max(self.lo, min(self.hi, v))
            # snap to step, keep ints integral
            if self.step:
                v = round((v - self.lo) / self.step) * self.step + self.lo
                v = round(v, 6)              # kill IEEE-754 dust from fractional steps
            v = max(self.lo, min(self.hi, v))
            return int(round(v)) if float(self.step).is_integer() and float(self.lo).is_integer() else v
        if self.kind == CHOICE:
            valid = [c[0] for c in self.choices]
            return value if value in valid else self.default
        if self.kind == KEYBIND:
            return str(value) if value else self.default
        return value

    def choice_label(self, value: Any) -> str:
        for v, label in self.choices:
            if v == value:
                return label
        return str(value)


# ---------------------------------------------------------------------------
# The curated schema — only settings that mean something for THIS game.
# ---------------------------------------------------------------------------
def _build_specs() -> List[Spec]:
    s: List[Spec] = []

    # ---- Graphics --------------------------------------------------------
    s += [
        Spec("window_mode", "Window Mode", CHOICE, GRAPHICS, default="windowed",
             choices=(("windowed", "Windowed"), ("borderless", "Borderless"),
                      ("fullscreen", "Fullscreen")),
             tooltip="How the game window is presented on your desktop."),
        Spec("view_mode", "View", CHOICE, GRAPHICS, default="auto",
             choices=(("auto", "Auto"), ("desktop", "Desktop"), ("phone", "Phone")),
             tooltip="Desktop fills the window; Phone shows the portrait device frame. "
                     "Auto picks based on the window shape."),
        Spec("resolution", "Resolution", CHOICE, GRAPHICS, default="auto",
             choices=(("auto", "Auto (window)"), ("960x600", "960 x 600"),
                      ("1280x720", "1280 x 720"), ("1600x900", "1600 x 900"),
                      ("1920x1080", "1920 x 1080"), ("480x820", "480 x 820 (phone)")),
             tooltip="Windowed-mode size. Auto keeps whatever you resize the window to."),
        Spec("vsync", "VSync", TOGGLE, GRAPHICS, default=True, needs_restart=True,
             tooltip="Synchronise to the monitor refresh to remove tearing. "
                     "Applied when the window is rebuilt."),
        Spec("fps_limit", "FPS Limit", CHOICE, GRAPHICS, default=120,
             choices=((30, "30"), (60, "60"), (120, "120"), (144, "144"),
                      (240, "240"), (0, "Unlimited")),
             tooltip="Upper bound on rendered frames per second."),
        Spec("shadow_quality", "Shadow Quality", CHOICE, GRAPHICS, default="high",
             choices=(("off", "Off"), ("low", "Low"), ("high", "High")),
             tooltip="Resolution of the directional shadow map (Off disables shadows)."),
        Spec("render_scale", "Render Scale", SLIDER, GRAPHICS, default=100,
             lo=50, hi=100, step=5, unit="%",
             tooltip="Render the 3D scene at a fraction of the window size, then upscale. "
                     "Lower is faster on weak GPUs."),
        Spec("brightness", "Brightness", SLIDER, GRAPHICS, default=100,
             lo=60, hi=140, step=5, unit="%",
             tooltip="Overall scene brightness."),
        Spec("ui_scale", "UI Scale", SLIDER, GRAPHICS, default=100,
             lo=75, hi=150, step=5, unit="%",
             tooltip="Scale of all menu and HUD text/controls."),
        Spec("camera_zoom", "Camera Zoom", SLIDER, GRAPHICS, default=100,
             lo=80, hi=130, step=5, unit="%",
             tooltip="How close the camera frames the action."),
        Spec("anti_aliasing", "Anti-Aliasing", CHOICE, GRAPHICS, default="off",
             choices=(("off", "Off"),), wired=False,
             note="Not yet supported by the renderer.",
             tooltip="Edge smoothing. Reserved for a future MSAA path."),
    ]

    # ---- Audio -----------------------------------------------------------
    s += [
        Spec("master_volume", "Master Volume", SLIDER, AUDIO, default=80,
             lo=0, hi=100, step=5, unit="%", tooltip="Overall output level."),
        Spec("sfx_volume", "Sound Effects", SLIDER, AUDIO, default=80,
             lo=0, hi=100, step=5, unit="%", tooltip="Hops, splashes, crashes."),
        Spec("ui_volume", "UI Sounds", SLIDER, AUDIO, default=70,
             lo=0, hi=100, step=5, unit="%", tooltip="Menu clicks and confirmations."),
        Spec("mute", "Mute", TOGGLE, AUDIO, default=False,
             tooltip="Silence all audio without losing your volume levels."),
    ]

    # ---- Gameplay --------------------------------------------------------
    s += [
        Spec("difficulty", "Difficulty", CHOICE, GAMEPLAY, default="normal",
             choices=(("easy", "Easy"), ("normal", "Normal"), ("hard", "Hard")),
             tooltip="Scales traffic/river speed. Easy is more forgiving."),
        Spec("camera_smoothness", "Camera Smoothness", SLIDER, GAMEPLAY, default=5,
             lo=1, hi=10, step=1,
             tooltip="How gently the camera eases to follow the player."),
        Spec("screen_shake", "Screen Shake", TOGGLE, GAMEPLAY, default=True,
             tooltip="The little jolt on collisions and deaths."),
        Spec("auto_pause", "Auto-Pause on Focus Loss", TOGGLE, GAMEPLAY, default=True,
             tooltip="Pause automatically when the window is minimised or loses focus."),
        Spec("show_hints", "Show Hints", TOGGLE, GAMEPLAY, default=True,
             tooltip="On-screen prompts like \"press space to play\"."),
    ]

    # ---- Controls (rebindable) ------------------------------------------
    s += [
        Spec("key_up", "Move Up / Forward", KEYBIND, CONTROLS, default="up",
             tooltip="Primary key to hop forward (W and Space always also work)."),
        Spec("key_down", "Move Down", KEYBIND, CONTROLS, default="down",
             tooltip="Primary key to hop backward."),
        Spec("key_left", "Move Left", KEYBIND, CONTROLS, default="left",
             tooltip="Primary key to hop left."),
        Spec("key_right", "Move Right", KEYBIND, CONTROLS, default="right",
             tooltip="Primary key to hop right."),
        Spec("key_pause", "Pause", KEYBIND, CONTROLS, default="p",
             tooltip="Pause / resume the game."),
        Spec("controls_reset", "Reset Controls to Defaults", ACTION, CONTROLS,
             action="reset_controls", tooltip="Restore the default key bindings."),
    ]

    # ---- Performance -----------------------------------------------------
    s += [
        Spec("prefer_dedicated_gpu", "Prefer High-Performance GPU", TOGGLE, PERFORMANCE,
             default=True, needs_restart=True,
             tooltip="On hybrid laptops, request the dedicated GPU. Applied on next launch."),
        Spec("show_fps", "Show FPS", TOGGLE, PERFORMANCE, default=False,
             tooltip="A small frames-per-second counter."),
        Spec("perf_overlay", "Performance Overlay", TOGGLE, PERFORMANCE, default=False,
             tooltip="FPS, frame time and renderer info overlaid on the game."),
        Spec("gpu_info", "GPU", INFO, PERFORMANCE,
             tooltip="The graphics device currently in use."),
        Spec("renderer_info", "Renderer", INFO, PERFORMANCE,
             tooltip="The OpenGL renderer / API in use."),
    ]

    # ---- AI --------------------------------------------------------------
    s += [
        Spec("ai_algorithm", "Algorithm", CHOICE, AI, default="neat",
             choices=(("neat", "NEAT"), ("ppo", "PPO"), ("a2c", "A2C"), ("dqn", "DQN"),
                      ("ddqn", "Double DQN"), ("es", "Evolution Strategies"),
                      ("ga", "Genetic Algorithm"), ("cmaes", "CMA-ES"),
                      ("minimax", "Minimax (search)")),
             tooltip="Which AI to train / auto-play with. Minimax plans live (no training); "
                     "the others learn. Double DQN is a full DDQN with prioritized replay."),
        Spec("ai_speed", "Training / Play Speed", SLIDER, AI, default=3,
             lo=1, hi=10, step=1, unit="x",
             tooltip="Real-time multiplier for the watched AI window."),
        Spec("ai_parallel", "Parallel Environments", SLIDER, AI, default=0,
             lo=0, hi=16, step=1,
             tooltip="Worker processes for population training (0 = single process). "
                     "Used by NEAT / ES / GA / CMA-ES."),
        Spec("ai_seed", "Random Seed", SLIDER, AI, default=0,
             lo=0, hi=9999, step=1,
             tooltip="Seed for reproducible training runs."),
        Spec("ai_swarm_size", "Swarm Size", SLIDER, AI, default=12,
             lo=2, hi=32, step=1,
             tooltip="How many chickens run at once in AI Swarm — each on its own policy, "
                     "with the camera following the leader."),
        Spec("ai_minimax_depth", "Minimax Search Depth", SLIDER, AI, default=6,
             lo=2, hi=8, step=1,
             tooltip="How many moves ahead the Minimax planner searches. Higher = smarter "
                     "(depth 4 ≈ score 28, depth 6 ≈ 44) but slower per decision."),
    ]

    # ---- Developer -------------------------------------------------------
    s += [
        Spec("debug_mode", "Debug Mode", TOGGLE, DEVELOPER, default=False,
             tooltip="Enable developer diagnostics."),
        Spec("show_collision", "Collision Visualisation", TOGGLE, DEVELOPER, default=False,
             wired=False, note="Collision-box overlay is planned, not yet drawn.",
             tooltip="Draw collision/bounding boxes for debugging."),
        Spec("logging_level", "Logging Level", CHOICE, DEVELOPER, default="info",
             choices=(("error", "Error"), ("warn", "Warn"), ("info", "Info"),
                      ("debug", "Debug")),
             tooltip="Verbosity of console logging."),
    ]
    return s


SPECS: List[Spec] = _build_specs()
SPEC_BY_KEY: Dict[str, Spec] = {sp.key: sp for sp in SPECS}


def specs_for(category: str) -> List[Spec]:
    return [sp for sp in SPECS if sp.category == category]


def defaults() -> Dict[str, Any]:
    return {sp.key: sp.default for sp in SPECS
            if sp.kind not in (INFO, ACTION)}


# ---------------------------------------------------------------------------
# The live config object
# ---------------------------------------------------------------------------
class Config:
    """Holds the committed values plus a pending overlay for Apply / Cancel.

    Reads go through :meth:`get` (pending overrides committed). Edits go to ``pending``
    via :meth:`set`; :meth:`apply` commits + persists; :meth:`cancel` discards them.
    """

    def __init__(self, values: Optional[Dict[str, Any]] = None, path: str = CONFIG_PATH):
        self.path = path
        self.values: Dict[str, Any] = defaults()
        if values:
            for k, v in values.items():
                if k in SPEC_BY_KEY:
                    self.values[k] = SPEC_BY_KEY[k].clamp(v)
        self.pending: Dict[str, Any] = {}

    # -- access ------------------------------------------------------------
    def get(self, key: str) -> Any:
        if key in self.pending:
            return self.pending[key]
        return self.values.get(key, SPEC_BY_KEY[key].default if key in SPEC_BY_KEY else None)

    def committed(self, key: str) -> Any:
        return self.values.get(key)

    def set(self, key: str, value: Any) -> Any:
        """Stage an edit (validated/clamped). Returns the stored value."""
        spec = SPEC_BY_KEY.get(key)
        if spec is None or spec.kind in (INFO, ACTION):
            return None
        value = spec.clamp(value)
        if value == self.values.get(key):
            self.pending.pop(key, None)          # back to committed -> no longer pending
        else:
            self.pending[key] = value
        return value

    def cycle_choice(self, key: str, direction: int = 1) -> Any:
        spec = SPEC_BY_KEY.get(key)
        if spec is None or spec.kind != CHOICE or not spec.choices:
            return None
        vals = [c[0] for c in spec.choices]
        try:
            i = vals.index(self.get(key))
        except ValueError:
            i = 0
        return self.set(key, vals[(i + direction) % len(vals)])

    def nudge_slider(self, key: str, direction: int = 1) -> Any:
        spec = SPEC_BY_KEY.get(key)
        if spec is None or spec.kind != SLIDER:
            return None
        return self.set(key, self.get(key) + direction * spec.step)

    # -- pending / restart -------------------------------------------------
    @property
    def dirty(self) -> bool:
        return bool(self.pending)

    def pending_keys(self) -> List[str]:
        return list(self.pending.keys())

    def needs_restart(self) -> bool:
        return any(SPEC_BY_KEY[k].needs_restart for k in self.pending if k in SPEC_BY_KEY)

    # -- commit / discard --------------------------------------------------
    def apply(self) -> List[str]:
        """Commit pending edits into the live values and persist. Returns changed keys."""
        changed = list(self.pending.keys())
        self.values.update(self.pending)
        self.pending.clear()
        self.save()
        return changed

    def cancel(self) -> None:
        self.pending.clear()

    def reset_to_defaults(self) -> None:
        """Stage a full reset to factory defaults (commit with :meth:`apply`)."""
        self.pending = {}
        d = defaults()
        for k, v in d.items():
            if v != self.values.get(k):
                self.pending[k] = v

    def reset_category(self, category: str) -> None:
        for sp in specs_for(category):
            if sp.kind in (INFO, ACTION):
                continue
            self.set(sp.key, sp.default)

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return dict(self.values)

    def save(self) -> bool:
        """Atomically write the committed values. Never raises."""
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"version": 1, "values": self.values}, fh, indent=2)
            os.replace(tmp, self.path)          # atomic on POSIX
            return True
        except OSError:
            return False

    def export_to(self, path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"version": 1, "values": self.values}, fh, indent=2)
            return True
        except OSError:
            return False

    def import_from(self, path: str) -> bool:
        """Replace pending with values from ``path`` (validated). Apply to commit."""
        data = _read_json(path)
        if data is None:
            return False
        incoming = data.get("values", data) if isinstance(data, dict) else {}
        if not isinstance(incoming, dict):
            return False
        self.pending = {}
        for k, v in incoming.items():
            spec = SPEC_BY_KEY.get(k)
            if spec is None or spec.kind in (INFO, ACTION):
                continue
            self.set(k, v)
        return True

    def backup(self) -> Optional[str]:
        """Copy the current config file into the backups dir. Returns the backup path."""
        try:
            os.makedirs(BACKUP_DIR, exist_ok=True)
            if not os.path.exists(self.path):
                self.save()
            stamp = time.strftime("%Y%m%d-%H%M%S")
            dst = os.path.join(BACKUP_DIR, f"settings-{stamp}.json")
            shutil.copyfile(self.path, dst)
            return dst
        except OSError:
            return None


def _read_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def load(path: str = CONFIG_PATH) -> Config:
    """Load config from disk, falling back to safe defaults on any problem.

    A missing, unreadable, or malformed file (or one whose values are the wrong types)
    yields a default :class:`Config` — corruption can never crash the game. Individual bad
    fields are clamped/dropped while the rest are kept.
    """
    data = _read_json(path)
    if not isinstance(data, dict):
        return Config(path=path)
    values = data.get("values")
    if not isinstance(values, dict):
        # Tolerate a flat {key: value} file too.
        values = data if all(k in SPEC_BY_KEY for k in data) else {}
    return Config(values=values, path=path)


# Convenience for callers that just want a validated dict (e.g. on_apply consumers).
ApplyFn = Callable[["Config", List[str]], None]
