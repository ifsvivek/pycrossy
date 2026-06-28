"""Particle systems for death, drown, and river-edge foam effects.

Each system is a small pool of tinted boxes/planes animated with the tween engine,
using bezier-path bursts (feathers/water) and looping foam drift.
"""
from __future__ import annotations

import math
import random
from typing import List

from . import primitives
from .scene import Group, Mesh, Object3D
from .tween import tween

_WHITE = (1.0, 1.0, 1.0, 1.0)
_WATER_BLUE = (0x71 / 255, 0xD7 / 255, 0xFF / 255, 1.0)


def _rand(a: float = 0.0, b: float = 1.0) -> float:
    return random.random() * (b - a) + a


class Feathers:
    """Death burst: 20 white boxes flung outward along the hit direction."""

    SIZE = 0.1

    def __init__(self):
        self.mesh = Group()
        geo = primitives.box(self.SIZE, self.SIZE, 0.01)
        self.parts: List[Mesh] = []
        for _ in range(20):
            p = Mesh(geo, None, tint=_WHITE, cast_shadow=False, receive_shadow=False)
            p.visible = False
            self.parts.append(p)
            self.mesh.add(p)

    def run(self, _type: str = "feathers", direction: float = 0.0) -> None:
        explosion_speed = 0.3
        for p in self.parts:
            m = -1 if direction < 0 else 1
            tx = _rand(2, 7) * m
            ty = _rand(1, 3)
            tz = _rand(1, 3)
            values = [(0, 0, 0), (tx * 0.25, ty * 0.25, tz * 0.25),
                      (tx * 0.5, ty * 0.5, tz * 0.5), (tx, ty, tz)]
            p.position.set(0, 0, 0)
            p.scale.set(1, 1, 1)
            p.visible = True
            delay = explosion_speed + random.random() * 0.5
            tween.to(p.position, delay * 5, bezier=values)
            tween.to(p.rotation, delay * 5, delay=delay,
                     x=_rand(0.2, math.pi * 2 + 0.2), y=_rand(0.2, math.pi * 2 + 0.2),
                     z=_rand(0.2, math.pi * 2 + 0.2))
            tween.to(p.scale, delay, delay=delay * 3, x=0.01, y=0.01, z=0.01,
                     on_complete=lambda pp=p: setattr(pp, "visible", False))


class Water:
    """Drown splash: 15 blue boxes bounced outward."""

    def __init__(self):
        self.mesh = Group()
        geo = primitives.box(0.2, 0.3, 0.2)
        self.parts: List[Mesh] = []
        for _ in range(15):
            p = Mesh(geo, None, tint=_WATER_BLUE, cast_shadow=False, receive_shadow=False)
            p.visible = False
            self.parts.append(p)
            self.mesh.add(p)

    def run(self, _type: str = "water", _direction: float = 0.0) -> None:
        explosion_speed = 0.3
        for p in self.parts:
            tx = -1.0 + random.random()
            ty = random.random() * 2.0 + 1
            tz = -1.0 + random.random()
            values = [(0, 0, 0), (tx, ty, tz), (tx * 0.8, ty * 0.8, tz * 0.8),
                      (tx * (_rand(1.1, 1.6)), 0, tz * (_rand(1.1, 1.6)))]
            p.position.set(0, 0, 0)
            p.scale.set(1, 1, 1)
            p.visible = True
            s = explosion_speed + random.random() * 0.5
            tween.to(p.position, s * 4, bezier=values, ease="bounce.out")
            tween.to(p.scale, s, delay=s * 3, x=0.01, y=0.01, z=0.01,
                     on_complete=lambda pp=p: setattr(pp, "visible", False))


class Foam(Object3D):
    """River-edge foam: 6 planes scaling in/out and drifting outward, looping."""

    SIZE = 0.6

    def __init__(self, direction: int):
        super().__init__()
        self.direction = direction
        geo = primitives.plane(self.SIZE, self.SIZE)
        self.parts: List[Mesh] = []
        for _ in range(6):
            p = Mesh(geo, None, tint=_WHITE, cast_shadow=False, receive_shadow=False)
            p.rotation.x = math.pi / 2
            p.visible = False
            self.parts.append(p)
            self.add(p)

    def run(self) -> None:
        for i, p in enumerate(self.parts):
            self._run_one(p, i)

    def _setup(self, n: Mesh, i: int) -> None:
        n.position.set(_rand(-0.1, 0.1), 0, (self.SIZE / len(self.parts)) * i + 0.2)
        n.visible = True
        n.scale.set(0.01, 0.01, 0.01)
        n.rotation.y = random.random() * 0.6 - 0.3

    def _run_one(self, n: Mesh, i: int) -> None:
        self._setup(n, i)
        m_dur, l_dur = 0.4, 1.2
        start_delay = (m_dur + l_dur) * 0.2 * i

        def grow_done(nn=n, ii=i):
            tween.to(nn.scale, l_dur, x=0.01, y=0.01, z=0.01, ease="power2.in")
            tween.to(nn.position, l_dur,
                     x=nn.position.x + _rand(0.2, 1.0) * self.direction,
                     on_complete=lambda: self._run_one(nn, ii))

        tween.to(n.scale, m_dur, delay=start_delay, x=1, y=1, z=1,
                 ease="bounce.out", on_complete=grow_done)
