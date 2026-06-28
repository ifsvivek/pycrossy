"""JSON persistence for the high score, selected character, and lifetime statistics.

Hardened against corruption: a missing or malformed save file yields fresh defaults rather
than raising, and unknown/garbage fields are ignored. The on-disk schema is forward- and
backward-compatible — older saves (with only ``highscore`` / ``character``) load fine, and
new fields default sensibly.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict

_SAVE_PATH = os.path.join(os.path.expanduser("~"), ".pycrossy", "save.json")


@dataclass
class SaveData:
    highscore: int = 0
    character: str = "chicken"
    # Lifetime statistics (shown on the Statistics screen).
    games_played: int = 0
    total_score: int = 0
    total_deaths: int = 0
    best_by_character: Dict[str, int] = field(default_factory=dict)
    play_seconds: float = 0.0

    # -- stats updates -----------------------------------------------------
    def record_game(self, score: int, character: str, seconds: float = 0.0) -> bool:
        """Fold one finished game into the lifetime stats. Returns True on a new best."""
        self.games_played += 1
        self.total_score += max(0, int(score))
        self.total_deaths += 1
        self.play_seconds += max(0.0, float(seconds))
        prev = self.best_by_character.get(character, 0)
        if score > prev:
            self.best_by_character[character] = int(score)
        new_best = score > self.highscore
        if new_best:
            self.highscore = int(score)
        return new_best

    @property
    def average_score(self) -> float:
        return self.total_score / self.games_played if self.games_played else 0.0


def load() -> SaveData:
    try:
        with open(_SAVE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return SaveData()
        best = data.get("best_by_character", {})
        if not isinstance(best, dict):
            best = {}
        return SaveData(
            highscore=int(data.get("highscore", 0)),
            character=str(data.get("character", "chicken")),
            games_played=int(data.get("games_played", 0)),
            total_score=int(data.get("total_score", 0)),
            total_deaths=int(data.get("total_deaths", 0)),
            best_by_character={str(k): int(v) for k, v in best.items()},
            play_seconds=float(data.get("play_seconds", 0.0)),
        )
    except (OSError, ValueError, TypeError, KeyError):
        return SaveData()


def save(data: SaveData) -> None:
    try:
        os.makedirs(os.path.dirname(_SAVE_PATH), exist_ok=True)
        tmp = _SAVE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(asdict(data), fh, indent=2)
        os.replace(tmp, _SAVE_PATH)
    except OSError:
        pass
