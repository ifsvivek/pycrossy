"""A small tween engine with GSAP-style ``TweenMax`` / ``TimelineMax`` semantics.

It provides timed property tweens with eases, delays, ``onComplete``, ``repeat``
(incl. infinite), ``yoyo``, sequential timelines, and bezier paths (for the
particle bursts). All tweens are driven by real wall-clock ``dt`` so timing is
consistent regardless of framerate.

Usage:

    tween.to(player.scale, 0.2, x=1.2, y=0.75, z=1)
    tl = tween.timeline(repeat=-1)
    tl.to(player.scale, 0.3, y=0.8, ease="power1.in").to(player.scale, 0.3, y=1.0)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from . import mathutils as mu

_PROPS = ("x", "y", "z")


def _catmull_rom(points: Sequence[Sequence[float]], t: float) -> List[float]:
    """Sample a Catmull-Rom spline through ``points`` (each len-3) at t in [0,1]."""
    n = len(points)
    if n == 1:
        return list(points[0])
    if n == 2:
        return [mu.lerp(points[0][i], points[1][i], t) for i in range(3)]
    seg = n - 1
    ft = t * seg
    i = min(int(ft), seg - 1)
    lt = ft - i
    p0 = points[max(i - 1, 0)]
    p1 = points[i]
    p2 = points[i + 1]
    p3 = points[min(i + 2, n - 1)]
    out = []
    for k in range(3):
        a = 2 * p1[k]
        b = p2[k] - p0[k]
        c = 2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]
        d = -p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]
        out.append(0.5 * (a + b * lt + c * lt * lt + d * lt ** 3))
    return out


class Tween:
    """Animate properties of a target object over a duration."""

    def __init__(self, target, duration: float, props: Dict[str, float],
                 ease: str = "power1.inout", delay: float = 0.0,
                 on_complete: Optional[Callable] = None,
                 on_update: Optional[Callable] = None,
                 repeat: int = 0, yoyo: bool = False,
                 bezier: Optional[Sequence[Sequence[float]]] = None):
        self.target = target
        self.duration = max(duration, 1e-6)
        self.props = props
        self.ease = mu.EASES.get(ease, mu.linear)
        self.delay = delay
        self.on_complete = on_complete
        self.on_update = on_update
        self.repeat = repeat
        self.yoyo = yoyo
        self.bezier = [list(p) for p in bezier] if bezier else None

        self._elapsed = -delay
        self._start: Dict[str, float] = {}
        self._started = False
        self._iteration = 0
        self.active = True
        self.done = False

    def _capture_start(self) -> None:
        if self.bezier is not None:
            # First bezier value is the start; capture nothing else.
            self._start = {}
        else:
            self._start = {k: float(getattr(self.target, k)) for k in self.props}
        self._started = True

    def _apply(self, frac: float) -> None:
        e = self.ease(frac)
        if self.bezier is not None:
            pt = _catmull_rom(self.bezier, e)
            self.target.x, self.target.y, self.target.z = pt[0], pt[1], pt[2]
        else:
            for k, end in self.props.items():
                setattr(self.target, k, mu.lerp(self._start[k], end, e))
        if self.on_update:
            self.on_update()

    def update(self, dt: float) -> None:
        if not self.active or self.done:
            return
        self._elapsed += dt
        if self._elapsed < 0:
            return
        if not self._started:
            self._capture_start()
        frac = self._elapsed / self.duration
        if frac >= 1.0:
            # Handle repeats.
            if self.repeat != 0:
                self._apply(1.0 if not self.yoyo or self._iteration % 2 == 0 else 0.0)
                self._iteration += 1
                if self.repeat > 0:
                    self.repeat -= 1
                self._elapsed -= self.duration
                # On yoyo, swap start/end by recapturing inverted.
                if self.yoyo:
                    if self.bezier is not None:
                        self.bezier.reverse()
                    else:
                        new_start = dict(self.props)
                        self.props = {k: self._start[k] for k in self.props}
                        self._start = new_start
                return
            self._apply(1.0)
            self.done = True
            self.active = False
            if self.on_complete:
                self.on_complete()
            return
        self._apply(frac)

    def pause(self) -> None:
        self.active = False

    def play(self) -> None:
        if not self.done:
            self.active = True

    def kill(self) -> None:
        self.active = False
        self.done = True


class Timeline:
    """A sequence of tweens played one after another, with optional repeat."""

    def __init__(self, repeat: int = 0, on_complete: Optional[Callable] = None):
        self.steps: List[Tween] = []
        self.repeat = repeat
        self.on_complete = on_complete
        self._index = 0
        self.active = True
        self.done = False

    def to(self, target, duration: float, ease: str = "power1.inout", delay: float = 0.0,
           on_complete: Optional[Callable] = None, **props) -> "Timeline":
        bezier = props.pop("bezier", None)
        self.steps.append(Tween(target, duration, props, ease=ease, delay=delay,
                                on_complete=on_complete, bezier=bezier))
        return self

    def update(self, dt: float) -> None:
        if not self.active or self.done or not self.steps:
            return
        step = self.steps[self._index]
        step.update(dt)
        if step.done:
            self._index += 1
            if self._index >= len(self.steps):
                if self.repeat != 0:
                    if self.repeat > 0:
                        self.repeat -= 1
                    self._index = 0
                    for s in self.steps:           # reset for replay
                        s._elapsed = -s.delay
                        s._started = False
                        s.done = False
                        s.active = True
                else:
                    self.done = True
                    self.active = False
                    if self.on_complete:
                        self.on_complete()

    def pause(self) -> None:
        self.active = False

    def play(self) -> None:
        if not self.done:
            self.active = True

    def kill(self) -> None:
        self.active = False
        self.done = True


class TweenManager:
    """Global registry; ``update(dt)`` advances every live tween/timeline."""

    def __init__(self):
        self._items: List = []

    def to(self, target, duration: float, ease: str = "power1.inout", delay: float = 0.0,
           on_complete: Optional[Callable] = None, on_update: Optional[Callable] = None,
           **props) -> Tween:
        bezier = props.pop("bezier", None)
        repeat = props.pop("repeat", 0)
        yoyo = props.pop("yoyo", False)
        t = Tween(target, duration, props, ease=ease, delay=delay, on_complete=on_complete,
                  on_update=on_update, repeat=repeat, yoyo=yoyo, bezier=bezier)
        self._items.append(t)
        return t

    def add(self, item) -> None:
        self._items.append(item)

    def timeline(self, repeat: int = 0, on_complete: Optional[Callable] = None) -> Timeline:
        tl = Timeline(repeat=repeat, on_complete=on_complete)
        self._items.append(tl)
        return tl

    def update(self, dt: float) -> None:
        for item in self._items:
            item.update(dt)
        if self._items:
            self._items = [it for it in self._items if not it.done]

    def clear(self) -> None:
        self._items.clear()


# Global singleton, GSAP-style global timeline.
tween = TweenManager()
