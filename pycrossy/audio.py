"""Audio manager built on ``pygame.mixer``.

Caches ``Sound`` objects by path and plays them on free channels. Handles the move
buck-cycling, passive-car coin-flip, and random death/hit selection. Safe to disable
(``enabled=False``) for headless AI training.
"""
from __future__ import annotations

import random
from typing import Dict

from . import assets


class AudioManager:
    def __init__(self, enabled: bool = True, volume: float = 0.7):
        self.enabled = enabled
        self.sounds = assets.AUDIO
        self.volume = volume
        # Runtime mixer levels (0..1), set from settings via set_levels().
        self.master = volume
        self.sfx = 1.0
        self.ui = 1.0
        self.muted = False
        self._cache: Dict[str, object] = {}
        self._move_index = 0
        self._mixer = None
        if enabled:
            try:
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.set_num_channels(32)
                self._mixer = pygame.mixer
            except Exception as exc:  # pragma: no cover - audio is non-critical
                print(f"[audio] disabled (mixer init failed: {exc})")
                self.enabled = False

    def set_levels(self, master=None, sfx=None, ui=None, mute=None) -> None:
        """Update mixer levels (each 0..1) from settings. Any ``None`` is left unchanged."""
        if master is not None:
            self.master = max(0.0, min(1.0, float(master)))
        if sfx is not None:
            self.sfx = max(0.0, min(1.0, float(sfx)))
        if ui is not None:
            self.ui = max(0.0, min(1.0, float(ui)))
        if mute is not None:
            self.muted = bool(mute)

    def _level(self, kind: str) -> float:
        if self.muted:
            return 0.0
        return self.master * (self.ui if kind == "ui" else self.sfx)

    def _sound(self, path: str):
        s = self._cache.get(path)
        if s is None:
            try:
                s = self._mixer.Sound(path)
            except Exception:
                s = False  # mark as failed so we don't retry every call
            self._cache[path] = s
        return s or None

    def play(self, path: str, kind: str = "sfx") -> None:
        if not self.enabled or self._mixer is None:
            return
        vol = self._level(kind)
        if vol <= 0.0:
            return
        s = self._sound(path)
        if s is not None:
            ch = self._mixer.find_channel(True)
            if ch is not None:
                ch.set_volume(vol)
                ch.play(s)

    # -- high-level cues ---------------------------------------------------
    def play_move_sound(self) -> None:
        moves = self.sounds["chicken"]["move"]
        self.play(moves[str(self._move_index)])
        self._move_index = (self._move_index + 1) % len(moves)

    def play_passive_car_sound(self) -> None:
        if random.randint(0, 1) == 0:
            self.play(self.sounds["car"]["passive"]["1"])

    def play_death_sound(self) -> None:
        self.play(self.sounds["chicken"]["die"][str(random.randint(0, 1))])

    def play_car_hit_sound(self) -> None:
        self.play(self.sounds["car"]["die"][str(random.randint(0, 1))])

    def play_banner(self) -> None:
        self.play(self.sounds["banner"])

    def play_button(self, down: bool = True) -> None:
        self.play(self.sounds["button_in" if down else "button_out"], kind="ui")


class NullAudio:
    """No-op audio for headless/AI use (same interface)."""

    def __init__(self):
        self.sounds = assets.AUDIO

    def set_levels(self, *_a, **_k): ...
    def play(self, *_a, **_k): ...
    def play_move_sound(self): ...
    def play_passive_car_sound(self): ...
    def play_death_sound(self): ...
    def play_car_hit_sound(self): ...
    def play_banner(self): ...
    def play_button(self, *_a, **_k): ...
