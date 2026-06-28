"""Matrix math + easing functions.

Matrices are row-major numpy ``(4, 4)`` arrays using the column-vector convention
(``v' = M @ v``). ``to_gl`` transposes to OpenGL's column-major layout for moderngl.

The easing functions follow the well-known gsap ease semantics
(``Power1``, ``Power2``, ``Bounce``, ``Elastic``) to drive animation movement.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

Vec3 = Sequence[float]


def identity() -> np.ndarray:
    return np.identity(4, dtype=np.float64)


def to_gl(m: np.ndarray) -> bytes:
    """Row-major math matrix -> column-major float32 bytes for a GL ``mat4`` uniform."""
    return np.ascontiguousarray(m.T, dtype="f4").tobytes()


def translate(x: float, y: float, z: float) -> np.ndarray:
    m = identity()
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scale(x: float, y: float, z: float) -> np.ndarray:
    m = identity()
    m[0, 0] = x
    m[1, 1] = y
    m[2, 2] = z
    return m


def rotate_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = identity()
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def rotate_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = identity()
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def rotate_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = identity()
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


def compose(position: Vec3, rotation: Vec3, scl: Vec3) -> np.ndarray:
    """Build a TRS model matrix using the XYZ Euler order.

    Closed-form (no 4x4 matmuls) for speed — this runs for every scene node every frame.
    Equivalent to ``translate @ rotate_x @ rotate_y @ rotate_z @ scale``.
    """
    px, py, pz = position
    rx, ry, rz = rotation
    sx, sy, sz = scl
    m = np.empty((4, 4), dtype=np.float64)

    if rx == 0.0 and ry == 0.0 and rz == 0.0:
        m[0, 0] = sx; m[0, 1] = 0.0; m[0, 2] = 0.0
        m[1, 0] = 0.0; m[1, 1] = sy; m[1, 2] = 0.0
        m[2, 0] = 0.0; m[2, 1] = 0.0; m[2, 2] = sz
    else:
        cx, sx_ = math.cos(rx), math.sin(rx)
        cy, sy_ = math.cos(ry), math.sin(ry)
        cz, sz_ = math.cos(rz), math.sin(rz)
        # R = Rx @ Ry @ Rz (columns then scaled by sx/sy/sz).
        r00 = cy * cz
        r01 = -cy * sz_
        r02 = sy_
        r10 = cx * sz_ + sx_ * sy_ * cz
        r11 = cx * cz - sx_ * sy_ * sz_
        r12 = -sx_ * cy
        r20 = sx_ * sz_ - cx * sy_ * cz
        r21 = sx_ * cz + cx * sy_ * sz_
        r22 = cx * cy
        m[0, 0] = r00 * sx; m[0, 1] = r01 * sy; m[0, 2] = r02 * sz
        m[1, 0] = r10 * sx; m[1, 1] = r11 * sy; m[1, 2] = r12 * sz
        m[2, 0] = r20 * sx; m[2, 1] = r21 * sy; m[2, 2] = r22 * sz

    m[0, 3] = px; m[1, 3] = py; m[2, 3] = pz
    m[3, 0] = 0.0; m[3, 1] = 0.0; m[3, 2] = 0.0; m[3, 3] = 1.0
    return m


def look_at(eye: Vec3, target: Vec3, up: Vec3 = (0.0, 1.0, 0.0)) -> np.ndarray:
    """Right-handed view matrix using the standard lookAt convention for a camera."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    f = target - eye
    if np.linalg.norm(f) < 1e-9:
        f = np.array([0.0, 0.0, -1.0])
    z = -f / np.linalg.norm(f)        # camera looks down -z
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-9:
        x = np.array([1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)

    m = identity()
    m[0, :3] = x
    m[1, :3] = y
    m[2, :3] = z
    m[0, 3] = -np.dot(x, eye)
    m[1, 3] = -np.dot(y, eye)
    m[2, 3] = -np.dot(z, eye)
    return m


def ortho(left: float, right: float, bottom: float, top: float, near: float, far: float) -> np.ndarray:
    """OpenGL orthographic projection (clip z in [-1, 1])."""
    m = identity()
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def transform_point(m: np.ndarray, p: Vec3) -> np.ndarray:
    v = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    out = m @ v
    return out[:3] / out[3]


def normalize_angle(a: float) -> float:
    """Wrap to (-pi, pi] via ``atan2(sin(a), cos(a))``."""
    return math.atan2(math.sin(a), math.cos(a))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ---------------------------------------------------------------------------
# Easing functions (gsap-compatible)
# ---------------------------------------------------------------------------
def linear(t: float) -> float:
    return t


def power1_in(t: float) -> float:
    return t * t


def power1_out(t: float) -> float:
    return 1.0 - (1.0 - t) ** 2


def power1_inout(t: float) -> float:
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - ((-2.0 * t + 2.0) ** 2) / 2.0


def power2_in(t: float) -> float:
    return t ** 3


def power2_out(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def power2_inout(t: float) -> float:
    if t < 0.5:
        return 4.0 * t ** 3
    return 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def bounce_out(t: float) -> float:
    n1, d1 = 7.5625, 2.75
    if t < 1 / d1:
        return n1 * t * t
    if t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    if t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    t -= 2.625 / d1
    return n1 * t * t + 0.984375


def elastic_out(t: float) -> float:
    if t in (0.0, 1.0):
        return t
    c4 = (2.0 * math.pi) / 3.0
    return math.pow(2.0, -10.0 * t) * math.sin((t * 10.0 - 0.75) * c4) + 1.0


EASES = {
    "linear": linear,
    "power1.in": power1_in,
    "power1.out": power1_out,
    "power1.inout": power1_inout,
    "power2.in": power2_in,
    "power2.out": power2_out,
    "power2.inout": power2_inout,
    "bounce.out": bounce_out,
    "elastic.out": elastic_out,
}
