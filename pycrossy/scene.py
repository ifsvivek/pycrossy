"""A minimal retained-mode scene graph.

Entity code manipulates ``Object3D`` instances directly
(``mesh.position.set(...)``, ``node.rotation.y = ...``, ``group.add(child)``) through a
small surface: :class:`Vec3`, :class:`Euler`, :class:`Object3D`, :class:`Group`, and a
drawable :class:`Mesh`.

World matrices are recomputed each frame by :meth:`Object3D.update_world_matrix`; the
renderer then walks the graph collecting drawables.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from . import mathutils as mu
from .obj_loader import Mesh as Geometry


class Vec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def set(self, x: float, y: float, z: float) -> "Vec3":
        self.x, self.y, self.z = float(x), float(y), float(z)
        return self

    def copy(self, other) -> "Vec3":
        self.x, self.y, self.z = float(other.x), float(other.y), float(other.z)
        return self

    def clone(self) -> "Vec3":
        return Vec3(self.x, self.y, self.z)

    def as_tuple(self):
        return (self.x, self.y, self.z)

    def __repr__(self) -> str:
        return f"Vec3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


class Euler:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def set(self, x: float, y: float, z: float) -> "Euler":
        self.x, self.y, self.z = float(x), float(y), float(z)
        return self


class Object3D:
    """Transform node with optional children, organized as an ``Object3D`` graph."""

    __slots__ = ("position", "rotation", "scale", "children", "parent", "visible",
                 "matrix_world", "_uid", "name", "_static", "_local_cache")
    _counter = 0

    def __init__(self):
        self.position = Vec3()
        self.rotation = Euler()
        self.scale = Vec3(1.0, 1.0, 1.0)
        self.children: List["Object3D"] = []
        self.parent: Optional["Object3D"] = None
        self.visible = True
        self.matrix_world = np.identity(4, dtype=np.float64)
        Object3D._counter += 1
        self._uid = Object3D._counter
        self.name = ""
        self._static = False          # static-local nodes cache their local matrix
        self._local_cache = None

    def add(self, child: "Object3D") -> "Object3D":
        if child.parent is not None and child in child.parent.children:
            child.parent.children.remove(child)
        child.parent = self
        self.children.append(child)
        return self

    def remove(self, child: "Object3D") -> "Object3D":
        if child in self.children:
            self.children.remove(child)
            child.parent = None
        return self

    def local_matrix(self) -> np.ndarray:
        if self._static and self._local_cache is not None:
            return self._local_cache
        m = mu.compose(
            (self.position.x, self.position.y, self.position.z),
            (self.rotation.x, self.rotation.y, self.rotation.z),
            (self.scale.x, self.scale.y, self.scale.z),
        )
        if self._static:
            self._local_cache = m
        return m

    def mark_static(self) -> None:
        """Flag this node's local transform as fixed so it is composed only once.

        Call after positioning objects that never move again (trees, boulders, floors,
        lights, train carriages). The world matrix is still re-derived from the parent.
        """
        self._static = True
        self._local_cache = None

    def update_world_matrix(self, parent_world: Optional[np.ndarray] = None) -> None:
        local = self.local_matrix()
        self.matrix_world = local if parent_world is None else parent_world @ local
        for c in self.children:
            c.update_world_matrix(self.matrix_world)


class Mesh(Object3D):
    """A drawable leaf: geometry + a texture key (resolved by the renderer)."""

    __slots__ = ("geometry", "texture_key", "tint", "cast_shadow", "receive_shadow")

    def __init__(self, geometry: Geometry, texture_key: Optional[str], tint=None,
                 cast_shadow: bool = True, receive_shadow: bool = True):
        super().__init__()
        self.geometry = geometry
        self.texture_key = texture_key
        self.tint = tint                # (r,g,b,a) floats, used when texture_key is None
        self.cast_shadow = cast_shadow
        self.receive_shadow = receive_shadow

    def clone(self) -> "Mesh":
        m = Mesh(self.geometry, self.texture_key, self.tint, self.cast_shadow, self.receive_shadow)
        m.position.copy(self.position)
        m.rotation.set(self.rotation.x, self.rotation.y, self.rotation.z)
        m.scale.copy(self.scale)
        m.visible = self.visible
        return m


class Group(Object3D):
    """A pure transform container (no geometry)."""


def collect_drawables(root: Object3D, out: List[Mesh]) -> None:
    """Depth-first collect visible :class:`Mesh` leaves (world matrices must be current)."""
    stack = [root]
    while stack:
        node = stack.pop()
        if not node.visible:
            continue
        if isinstance(node, Mesh) and node.geometry is not None and node.geometry.vertex_count:
            out.append(node)
        # children added in order; order doesn't matter (depth-sorted by GL z-buffer)
        for c in node.children:
            stack.append(c)
