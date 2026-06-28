"""Wavefront OBJ loader.

Parses the low-poly models into interleaved ``(position, normal, uv)`` vertex
buffers ready for a GL VBO. The models were exported by Misfit Model 3D as triangle
soup with per-face normals, which is exactly the faceted look we want. Missing normals
are reconstructed per face; missing UVs default to (0, 0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Mesh:
    """A loaded mesh: interleaved float32 vertices + bounding box metadata.

    Layout per vertex (8 floats): px py pz  nx ny nz  u v
    """
    data: np.ndarray              # (N, 8) float32
    bbox_min: np.ndarray          # (3,)
    bbox_max: np.ndarray          # (3,)

    @property
    def size(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min

    @property
    def center(self) -> np.ndarray:
        return (self.bbox_max + self.bbox_min) * 0.5

    @property
    def vertex_count(self) -> int:
        return self.data.shape[0]


_CACHE: Dict[str, Mesh] = {}


def _parse_index(token: str, n_v: int, n_vt: int, n_vn: int) -> Tuple[int, int, int]:
    """Parse a ``v/vt/vn`` face token (1-based, possibly negative) -> 0-based ints (-1 = none)."""
    parts = token.split("/")
    vi = int(parts[0])
    vi = vi - 1 if vi > 0 else n_v + vi
    ti = -1
    ni = -1
    if len(parts) >= 2 and parts[1] != "":
        ti = int(parts[1])
        ti = ti - 1 if ti > 0 else n_vt + ti
    if len(parts) >= 3 and parts[2] != "":
        ni = int(parts[2])
        ni = ni - 1 if ni > 0 else n_vn + ni
    return vi, ti, ni


def load(path: str) -> Mesh:
    """Load (and cache) an OBJ file into a :class:`Mesh`."""
    cached = _CACHE.get(path)
    if cached is not None:
        return cached

    positions: List[Tuple[float, float, float]] = []
    texcoords: List[Tuple[float, float]] = []
    normals: List[Tuple[float, float, float]] = []
    faces: List[List[Tuple[int, int, int]]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            tag, _, rest = line.partition(" ")
            if tag == "v":
                vals = rest.split()
                positions.append((float(vals[0]), float(vals[1]), float(vals[2])))
            elif tag == "vt":
                vals = rest.split()
                u = float(vals[0])
                v = float(vals[1]) if len(vals) > 1 else 0.0
                texcoords.append((u, v))
            elif tag == "vn":
                vals = rest.split()
                normals.append((float(vals[0]), float(vals[1]), float(vals[2])))
            elif tag == "f":
                toks = rest.split()
                faces.append([_parse_index(t, len(positions), len(texcoords), len(normals)) for t in toks])

    pos_arr = np.asarray(positions, dtype=np.float64)
    tex_arr = np.asarray(texcoords, dtype=np.float64) if texcoords else np.zeros((0, 2))
    nrm_arr = np.asarray(normals, dtype=np.float64) if normals else np.zeros((0, 3))

    verts: List[List[float]] = []
    for face in faces:
        # Fan-triangulate polygons with >3 vertices.
        for k in range(1, len(face) - 1):
            tri = (face[0], face[k], face[k + 1])
            # Flat normal fallback if the face has no vn.
            if tri[0][2] < 0:
                p0 = pos_arr[tri[0][0]]
                p1 = pos_arr[tri[1][0]]
                p2 = pos_arr[tri[2][0]]
                fn = np.cross(p1 - p0, p2 - p0)
                norm = np.linalg.norm(fn)
                fn = fn / norm if norm > 1e-12 else np.array([0.0, 1.0, 0.0])
            else:
                fn = None
            for (vi, ti, ni) in tri:
                p = pos_arr[vi]
                n = nrm_arr[ni] if (ni >= 0 and ni < len(nrm_arr)) else fn
                if n is None:
                    n = np.array([0.0, 1.0, 0.0])
                t = tex_arr[ti] if (ti >= 0 and ti < len(tex_arr)) else (0.0, 0.0)
                verts.append([p[0], p[1], p[2], n[0], n[1], n[2], t[0], t[1]])

    data = np.asarray(verts, dtype=np.float32)
    if data.size == 0:
        data = np.zeros((0, 8), dtype=np.float32)
    mesh = Mesh(
        data=data,
        bbox_min=pos_arr.min(axis=0) if len(pos_arr) else np.zeros(3),
        bbox_max=pos_arr.max(axis=0) if len(pos_arr) else np.zeros(3),
    )
    _CACHE[path] = mesh
    return mesh
