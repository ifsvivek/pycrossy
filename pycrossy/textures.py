"""Texture decoding.

Decodes PNG textures (many are palette-mode 'P') to tightly-packed RGBA bytes via
Pillow so it works headless (no display / SDL surface required). Results are cached.
Textures are sampled with nearest-neighbour filtering in sRGB by the renderer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from PIL import Image


@dataclass
class TextureData:
    width: int
    height: int
    rgba: bytes


_CACHE: Dict[str, TextureData] = {}


def load_rgba(path: str, flip_y: bool = True) -> TextureData:
    """Load a PNG as RGBA bytes. ``flip_y`` matches three.js' default ``texture.flipY``."""
    cached = _CACHE.get((path, flip_y))  # type: ignore[arg-type]
    if cached is not None:
        return cached
    img = Image.open(path).convert("RGBA")
    if flip_y:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    data = TextureData(img.width, img.height, img.tobytes())
    _CACHE[(path, flip_y)] = data  # type: ignore[index]
    return data
