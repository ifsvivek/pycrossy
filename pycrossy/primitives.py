"""Procedural primitive geometries (boxes, planes) for particles.

Returns the same :class:`obj_loader.Mesh` interleaved format the model loader produces,
so the renderer treats them identically to loaded models.
"""
from __future__ import annotations

import numpy as np

from .obj_loader import Mesh as Geometry


def box(w: float, h: float, d: float) -> Geometry:
    hx, hy, hz = w / 2, h / 2, d / 2
    # (position, normal) per face; uv is a dummy (0,0). 6 faces x 2 tris x 3 verts.
    faces = [
        # +X
        ((hx, -hy, -hz), (hx, -hy, hz), (hx, hy, hz), (hx, hy, -hz), (1, 0, 0)),
        # -X
        ((-hx, -hy, hz), (-hx, -hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (-1, 0, 0)),
        # +Y
        ((-hx, hy, -hz), (hx, hy, -hz), (hx, hy, hz), (-hx, hy, hz), (0, 1, 0)),
        # -Y
        ((-hx, -hy, hz), (hx, -hy, hz), (hx, -hy, -hz), (-hx, -hy, -hz), (0, -1, 0)),
        # +Z
        ((-hx, -hy, hz), (-hx, hy, hz), (hx, hy, hz), (hx, -hy, hz), (0, 0, 1)),
        # -Z
        ((hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz), (-hx, -hy, -hz), (0, 0, -1)),
    ]
    verts = []
    for a, b, c, d2, n in faces:
        for tri in ((a, b, c), (a, c, d2)):
            for p in tri:
                verts.append([p[0], p[1], p[2], n[0], n[1], n[2], 0.0, 0.0])
    data = np.asarray(verts, dtype=np.float32)
    return Geometry(data=data, bbox_min=np.array([-hx, -hy, -hz]), bbox_max=np.array([hx, hy, hz]))


def plane(w: float, h: float) -> Geometry:
    """A plane in the XY plane facing +Z (three.js PlaneGeometry orientation convention)."""
    hx, hy = w / 2, h / 2
    quad = [(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0)]
    n = (0, 0, 1)
    verts = []
    for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
        for p in tri:
            verts.append([p[0], p[1], p[2], n[0], n[1], n[2], 0.0, 0.0])
    data = np.asarray(verts, dtype=np.float32)
    return Geometry(data=data, bbox_min=np.array([-hx, -hy, 0]), bbox_max=np.array([hx, hy, 0]))
