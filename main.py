#!/usr/bin/env python3
"""Play PyCrossy — an endless arcade hopper game in Python.

Launches the main menu. From there: Play, watch / train an AI, view replays, run a
benchmark, or open Settings. All preferences are saved and restored automatically, so the
flags below are optional one-off overrides of the saved configuration.

Controls in game: W/A/S/D or arrow keys to hop, Space/Up to move forward, P to pause,
Esc to pause / go back. In menus: arrows + Enter, mouse, or a controller.

Display hotkeys: F11 fullscreen · F10 borderless · F9 cycle view.
"""
from __future__ import annotations

import argparse

from pycrossy.game import Game
from pycrossy.layout import DisplayMode


def main() -> None:
    p = argparse.ArgumentParser(description="Play PyCrossy")
    p.add_argument("--mode", default=None, choices=[m.value for m in DisplayMode],
                   help="override the saved view: native | mobile | stretch | dynamic")
    p.add_argument("--width", type=int, default=None, help="override window width")
    p.add_argument("--height", type=int, default=None, help="override window height")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--no-vsync", action="store_true")
    args = p.parse_args()

    display_mode = DisplayMode(args.mode) if args.mode else None
    Game(width=args.width, height=args.height, audio_enabled=not args.no_audio,
         vsync=False if args.no_vsync else None, display_mode=display_mode).run()


if __name__ == "__main__":
    main()
