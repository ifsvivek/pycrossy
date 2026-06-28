"""Model registry — loads geometries and builds the scene-node setup classes.

Loads every geometry once and hands out fresh :class:`scene.Mesh` nodes (cloned per
use). Shadow flags are set per node type.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional


from . import assets, obj_loader
from .scene import Group, Mesh
from .obj_loader import Mesh as Geometry


@dataclass
class _Spec:
    geometry: Geometry
    texture: str
    cast_shadow: bool
    receive_shadow: bool


class _Category:
    """A keyed set of model specs (e.g. the 4 tree variants)."""

    def __init__(self):
        self.specs: Dict[str, _Spec] = {}

    def register(self, key: str, mt: dict, cast: bool, recv: bool) -> None:
        self.specs[key] = _Spec(obj_loader.load(mt["model"]), mt["texture"], cast, recv)

    def make(self, key: str = "0") -> Mesh:
        s = self.specs[key]
        return Mesh(s.geometry, s.texture, cast_shadow=s.cast_shadow, receive_shadow=s.receive_shadow)

    def make_random(self) -> Mesh:
        return self.make(random.choice(list(self.specs.keys())))

    def keys(self) -> List[str]:
        return list(self.specs.keys())


class ModelRegistry:
    def __init__(self):
        env = assets.ENVIRONMENT
        veh = assets.VEHICLES

        self.grass = _Category()
        for k in ("0", "1"):
            self.grass.register(k, env["grass"][k], cast=False, recv=True)

        self.road = _Category()
        for k in ("0", "1"):
            self.road.register(k, env["road"][k], cast=False, recv=True)

        self.tree = _Category()
        for k in ("0", "1", "2", "3"):
            self.tree.register(k, env["tree"][k], cast=True, recv=False)

        self.boulder = _Category()
        for k in ("0", "1"):
            self.boulder.register(k, env["boulder"][k], cast=True, recv=False)

        self.log = _Category()
        for k in ("0", "1", "2", "3"):
            self.log.register(k, env["log"][k], cast=True, recv=True)

        self.lily_pad = _Category()
        self.lily_pad.register("0", env["lily_pad"], cast=True, recv=True)

        self.river = _Category()
        self.river.register("0", env["river"], cast=False, recv=True)

        self.railroad = _Category()
        self.railroad.register("0", env["railroad"], cast=False, recv=True)

        self.train_light = _Category()
        self.train_light.register("0", env["train_light"]["inactive"], cast=False, recv=False)
        self.train_light.register("active_0", env["train_light"]["active"]["0"], cast=False, recv=False)
        self.train_light.register("active_1", env["train_light"]["active"]["1"], cast=False, recv=False)

        self.car = _Category()
        for i, name in enumerate(assets.CAR_NAMES):
            self.car.register(str(i), veh[name], cast=True, recv=True)

        self.train = _Category()
        for part in ("front", "middle", "back"):
            self.train.register(part, veh["train"][part], cast=True, recv=True)

        # Hero geometries (per character).
        self.hero_specs: Dict[str, _Spec] = {}
        for cid, mt in assets.CHARACTERS.items():
            self.hero_specs[cid] = _Spec(obj_loader.load(mt["model"]), mt["texture"], True, True)

    # -- helpers -----------------------------------------------------------
    def make_train(self, size: int) -> Group:
        """Build a train of front + ``size`` middles + back, offset along X.

        Depth is measured from each part's X extent.
        """
        group = Group()
        front = self.train.make("front")
        front.mark_static()
        group.add(front)
        offset = round(float(front.geometry.size[0]))
        for _ in range(size):
            mid = self.train.make("middle")
            mid.position.x = offset
            mid.mark_static()
            group.add(mid)
            offset += round(float(mid.geometry.size[0]))
        back = self.train.make("back")
        back.position.x = offset
        back.mark_static()
        group.add(back)
        return group

    def make_hero_node(self, character: str) -> Mesh:
        """Scale the hero's longest side to 1 and center X/Z with the base on the ground.

        Scales the longest side to unit size, then aligns the mesh on the ground.
        """
        spec = self.hero_specs.get(character) or next(iter(self.hero_specs.values()))
        geo = spec.geometry
        s = 1.0 / float(geo.size.max())
        node = Mesh(geo, spec.texture, cast_shadow=True, receive_shadow=True)
        node.scale.set(s, s, s)
        center = geo.center
        node.position.set(-center[0] * s, -geo.bbox_min[1] * s, -center[2] * s)
        return node

    def character_ids(self) -> List[str]:
        return list(self.hero_specs.keys())


# Lazily-created singleton (needs a GL-free environment; only loads geometry/paths).
_REGISTRY: Optional[ModelRegistry] = None


def registry() -> ModelRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ModelRegistry()
    return _REGISTRY
