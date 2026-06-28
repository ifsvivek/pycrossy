"""moderngl renderer.

Draws the scene with an orthographic camera at a fixed angle, a directional light plus
ambient (Lambert shading), nearest-filtered sRGB textures, and directional shadow mapping
with PCF. Works against either a pygame OpenGL window or a moderngl standalone (headless)
context, drawing into an offscreen framebuffer.

Performance: drawables are grouped by geometry and rendered with **GPU instancing** (one
draw call per geometry/material group instead of per object), and parked/off-screen rows
are frustum-culled — so a fully-populated scene renders well above 120 FPS.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import moderngl
import numpy as np

from . import config, mathutils as mu, textures
from .obj_loader import Mesh as Geometry
from .scene import Mesh, Object3D, collect_drawables

# ---------------------------------------------------------------------------
# Shaders (per-instance model matrix via the in_model mat4 attribute)
# ---------------------------------------------------------------------------
_MAIN_VS = """
#version 330
in vec3 in_pos;
in vec3 in_norm;
in vec2 in_uv;
in mat4 in_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat4 u_light_vp;
out vec3 v_norm;
out vec2 v_uv;
out vec4 v_lightspace;
void main() {
    vec4 world = in_model * vec4(in_pos, 1.0);
    gl_Position = u_proj * u_view * world;
    v_norm = mat3(in_model) * in_norm;
    v_uv = in_uv;
    v_lightspace = u_light_vp * world;
}
"""

_MAIN_FS = """
#version 330
in vec3 v_norm;
in vec2 v_uv;
in vec4 v_lightspace;
out vec4 frag;
uniform sampler2D u_tex;
uniform sampler2D u_shadow;
uniform vec3 u_light_dir;
uniform float u_ambient;
uniform float u_dir_intensity;
uniform vec4 u_tint;
uniform int u_use_tex;
uniform int u_receive_shadow;
uniform float u_shadow_texel;
vec3 to_linear(vec3 c) { return pow(c, vec3(2.2)); }
vec3 to_srgb(vec3 c)   { return pow(c, vec3(1.0 / 2.2)); }
float shadow_factor() {
    if (u_receive_shadow == 0) return 0.0;
    vec3 proj = v_lightspace.xyz / v_lightspace.w;
    proj = proj * 0.5 + 0.5;
    if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0)
        return 0.0;
    float current = proj.z;
    float bias = 0.0016;
    float sh = 0.0;
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y) {
            float closest = texture(u_shadow, proj.xy + vec2(x, y) * u_shadow_texel).r;
            sh += (current - bias > closest) ? 1.0 : 0.0;
        }
    return sh / 9.0;
}
void main() {
    vec4 base = (u_use_tex == 1) ? texture(u_tex, v_uv) : u_tint;
    if (base.a < 0.5) discard;
    vec3 base_lin = to_linear(base.rgb);
    vec3 n = normalize(v_norm);
    float ndl = max(dot(n, normalize(u_light_dir)), 0.0);
    float shadow = shadow_factor();
    float irradiance = u_ambient + u_dir_intensity * ndl * (1.0 - shadow * 0.7);
    frag = vec4(to_srgb(clamp(base_lin * irradiance, 0.0, 1.0)), base.a);
}
"""

_DEPTH_VS = """
#version 330
in vec3 in_pos;
in mat4 in_model;
uniform mat4 u_light_vp;
void main() { gl_Position = u_light_vp * in_model * vec4(in_pos, 1.0); }
"""
_DEPTH_FS = "#version 330\nvoid main() {}"

class _MeshGL:
    """GPU buffers for one geometry: shared vertex VBO + a growable instance VBO."""

    __slots__ = ("vbo", "inst", "vao_main", "vao_depth", "count", "capacity")

    def __init__(self, ctx, geom: Geometry, main_prog, depth_prog):
        self.vbo = ctx.buffer(geom.data.tobytes())
        self.count = geom.vertex_count
        self.capacity = 64
        self.inst = ctx.buffer(reserve=self.capacity * 64)  # 16 floats * 4 bytes
        vfmt = (self.vbo, "3f 3f 2f", "in_pos", "in_norm", "in_uv")
        ifmt = (self.inst, "16f/i", "in_model")
        self.vao_main = ctx.vertex_array(main_prog, [vfmt, ifmt])
        self.vao_depth = ctx.vertex_array(depth_prog, [(self.vbo, "3f 5x4", "in_pos"), ifmt])

    def ensure(self, ctx, n: int) -> None:
        if n > self.capacity:
            self.capacity = max(n, self.capacity * 2)
            self.inst.orphan(self.capacity * 64)


class _Group:
    __slots__ = ("geo", "texture_key", "tint", "use_tex", "receive", "cast", "mats")

    def __init__(self, geo, texture_key, tint, use_tex, receive, cast):
        self.geo = geo
        self.texture_key = texture_key
        self.tint = tint
        self.use_tex = use_tex
        self.receive = receive
        self.cast = cast
        self.mats: List[np.ndarray] = []


class Renderer:
    def __init__(self, ctx: moderngl.Context, width: int, height: int, samples: int = 0):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.samples = samples

        self.main_prog = ctx.program(vertex_shader=_MAIN_VS, fragment_shader=_MAIN_FS)
        self.depth_prog = ctx.program(vertex_shader=_DEPTH_VS, fragment_shader=_DEPTH_FS)

        self._base_ambient = 0.62
        self._base_dir = 0.65
        self.brightness = 1.0
        self.ambient = self._base_ambient
        self.dir_intensity = self._base_dir
        self.view_half_width = config.CAMERA_VIEW_HALF_WIDTH       # zoom base (reference)
        self.view_half_height = config.CAMERA_VIEW_HALF_HEIGHT     # invariant vertical extent
        self.shadows_enabled = True
        self.cull_enabled = True

        self._mesh_cache: Dict[int, _MeshGL] = {}
        self._tex_cache: Dict[str, moderngl.Texture] = {}
        self._radius_cache: Dict[int, float] = {}

        self._make_scene_fbo(width, height)
        self._shadow_size = 0
        self._shadow_depth = None
        self._shadow_fbo = None
        self._make_shadow_fbo(config.SHADOW_MAP_SIZE)
        self._light_vp = self._compute_light_vp()

        self._blit_prog = ctx.program(
            vertex_shader="#version 330\nin vec2 in_pos; out vec2 uv;"
                          "void main(){uv=in_pos*0.5+0.5; gl_Position=vec4(in_pos,0,1);}",
            fragment_shader="#version 330\nuniform sampler2D tex; in vec2 uv; out vec4 c;"
                            "void main(){c=texture(tex,uv);}")
        quad = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
        self._blit_vao = ctx.vertex_array(self._blit_prog, [(ctx.buffer(quad.tobytes()), "2f", "in_pos")])
        self._overlays: Dict[str, moderngl.Texture] = {}
        ctx.enable(moderngl.DEPTH_TEST)

    # -- resources ---------------------------------------------------------
    def _make_scene_fbo(self, width: int, height: int) -> None:
        self.color_tex = self.ctx.texture((width, height), 4)
        self.color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth_rb = self.ctx.depth_renderbuffer((width, height))
        self.scene_fbo = self.ctx.framebuffer(color_attachments=[self.color_tex], depth_attachment=self.depth_rb)

    def resize(self, width: int, height: int) -> None:
        if width == self.width and height == self.height:
            return
        self.width, self.height = width, height
        self.scene_fbo.release(); self.color_tex.release(); self.depth_rb.release()
        self._make_scene_fbo(width, height)

    def _make_shadow_fbo(self, size: int) -> None:
        if size == self._shadow_size:
            return
        if self._shadow_fbo is not None:
            self._shadow_fbo.release()
        if self._shadow_depth is not None:
            self._shadow_depth.release()
        self._shadow_depth = self.ctx.depth_texture((size, size))
        self._shadow_depth.repeat_x = self._shadow_depth.repeat_y = False
        self._shadow_depth.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._shadow_fbo = self.ctx.framebuffer(depth_attachment=self._shadow_depth)
        self._shadow_texel = 1.0 / size
        self._shadow_size = size

    # -- live quality knobs (driven by settings) --------------------------
    def set_shadow_quality(self, quality: str) -> None:
        """``'off'`` disables shadows; ``'low'``/``'high'`` pick the shadow-map resolution."""
        if quality == "off":
            self.shadows_enabled = False
            return
        self.shadows_enabled = True
        self._make_shadow_fbo(512 if quality == "low" else 1024)

    def set_brightness(self, factor: float) -> None:
        """Scale scene lighting. ``1.0`` is the reference look."""
        self.brightness = max(0.3, min(2.0, float(factor)))
        self.ambient = self._base_ambient * self.brightness
        self.dir_intensity = self._base_dir * self.brightness

    def set_camera_zoom(self, factor: float) -> None:
        """``factor`` > 1 frames tighter (zooms in); 1.0 is the reference framing."""
        factor = max(0.5, min(2.0, float(factor)))
        self.view_half_width = config.CAMERA_VIEW_HALF_WIDTH / factor
        self.view_half_height = config.CAMERA_VIEW_HALF_HEIGHT / factor

    def _geom_gl(self, geom: Geometry) -> _MeshGL:
        gl = self._mesh_cache.get(id(geom))
        if gl is None:
            gl = _MeshGL(self.ctx, geom, self.main_prog, self.depth_prog)
            self._mesh_cache[id(geom)] = gl
        return gl

    def _texture(self, key: str) -> moderngl.Texture:
        tex = self._tex_cache.get(key)
        if tex is None:
            td = textures.load_rgba(key, flip_y=True)
            tex = self.ctx.texture((td.width, td.height), 4, td.rgba)
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._tex_cache[key] = tex
        return tex

    # -- camera / light ----------------------------------------------------
    def _compute_light_vp(self) -> np.ndarray:
        view = mu.look_at(config.DIR_LIGHT_POSITION, (0.0, 0.0, 0.0))
        proj = mu.ortho(config.SHADOW_CAM_LEFT, config.SHADOW_CAM_RIGHT,
                        config.SHADOW_CAM_BOTTOM, config.SHADOW_CAM_TOP, 0.5, config.SHADOW_CAM_FAR)
        return proj @ view

    def camera_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        orient = mu.look_at(config.CAMERA_ORIENT_EYE, config.CAMERA_TARGET)
        rot3 = orient[:3, :3]
        view = np.identity(4, dtype=np.float64)
        view[:3, :3] = rot3
        view[:3, 3] = -rot3 @ np.asarray(config.CAMERA_POSITION, dtype=np.float64)
        # Vertical extent is the invariant; widen horizontally with the window aspect ("Hor+")
        # so wide/desktop windows reveal more scenery instead of letterboxing. At the portrait
        # design aspect this reproduces the original half_w = 5.2 framing exactly.
        aspect = self.width / self.height
        half_h = self.view_half_height
        half_w = half_h * aspect
        proj = mu.ortho(-half_w, half_w, -half_h, half_h, config.CAMERA_NEAR, config.CAMERA_FAR)
        return view, proj

    # -- grouping ----------------------------------------------------------
    def _build_groups(self, drawables: List[Mesh], vp: Optional[np.ndarray] = None) -> List[_Group]:
        """Group drawables by geometry/material, frustum-culling off-screen objects.

        Culling projects each node's origin through the view-projection matrix and drops it
        only if it lies well outside the clip volume (generous margins), so it is correct at
        any aspect ratio / zoom and never removes geometry that is even partly visible —
        eliminating tile pop-in/out. (Row meshes are centred on x≈0, so they are never
        culled horizontally while their z-band is on screen.)"""
        groups: Dict[tuple, _Group] = {}
        cull = self.cull_enabled and vp is not None
        mxy, mz = 1.6, 1.5
        for node in drawables:
            mw = node.matrix_world
            # Only centre-cull COMPACT objects (trees/cars/boulders/…). The full-width floor
            # rows (radius ~12.5) are never centre-culled — their centre can leave the screen
            # while their body is still visible, so culling them would show the sky between
            # rows. They are left to the GPU's clipping (negligible cost via instancing).
            if cull and self._cull_radius(node.geometry) <= 3.0:
                ox, oy, oz = mw[0, 3], mw[1, 3], mw[2, 3]
                cw = vp[3, 0] * ox + vp[3, 1] * oy + vp[3, 2] * oz + vp[3, 3]
                if cw != 0.0:
                    cx = vp[0, 0] * ox + vp[0, 1] * oy + vp[0, 2] * oz + vp[0, 3]
                    cy = vp[1, 0] * ox + vp[1, 1] * oy + vp[1, 2] * oz + vp[1, 3]
                    cz = vp[2, 0] * ox + vp[2, 1] * oy + vp[2, 2] * oz + vp[2, 3]
                    lim_xy, lim_z = mxy * abs(cw), mz * abs(cw)
                    if abs(cx) > lim_xy or abs(cy) > lim_xy or cz < -lim_z or cz > lim_z:
                        continue
            use_tex = node.texture_key is not None
            tint = tuple(node.tint) if node.tint else (1.0, 1.0, 1.0, 1.0)
            key = (id(node.geometry), node.texture_key, tint, node.receive_shadow, node.cast_shadow)
            g = groups.get(key)
            if g is None:
                g = _Group(node.geometry, node.texture_key, tint, use_tex, node.receive_shadow, node.cast_shadow)
                groups[key] = g
            g.mats.append(mw)
        return list(groups.values())

    def _cull_radius(self, geo) -> float:
        """Cached horizontal bounding radius (max of the X/Z half-extents) of a geometry."""
        r = self._radius_cache.get(id(geo))
        if r is None:
            s = geo.size
            r = 0.5 * max(float(s[0]), float(s[2]))
            self._radius_cache[id(geo)] = r
        return r

    @staticmethod
    def _instance_bytes(mats: List[np.ndarray]) -> bytes:
        # Column-major float32 mat4 per instance (transpose of our row-major matrices).
        stacked = np.stack(mats)                          # (N,4,4) row-major
        return np.ascontiguousarray(np.transpose(stacked, (0, 2, 1)), dtype="f4").tobytes()

    # -- render ------------------------------------------------------------
    def render(self, scene: Object3D) -> None:
        scene.update_world_matrix()
        drawables: List[Mesh] = []
        collect_drawables(scene, drawables)

        view, proj = self.camera_matrices()
        vp = proj @ view
        groups = self._build_groups(drawables, vp)

        # Upload instance buffers once; reused by both passes.
        for g in groups:
            gl = self._geom_gl(g.geo)
            gl.ensure(self.ctx, len(g.mats))
            gl.inst.write(self._instance_bytes(g.mats))

        light_dir = np.asarray(config.DIR_LIGHT_POSITION, dtype=np.float64)
        light_dir = light_dir / np.linalg.norm(light_dir)

        if self.shadows_enabled:
            self._shadow_pass(groups)

        self.scene_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.clear(*config.SCENE_COLOR, 1.0, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        p = self.main_prog
        p["u_view"].write(mu.to_gl(view))
        p["u_proj"].write(mu.to_gl(proj))
        p["u_light_vp"].write(mu.to_gl(self._light_vp))
        p["u_light_dir"].value = tuple(light_dir)
        p["u_ambient"].value = self.ambient
        p["u_dir_intensity"].value = self.dir_intensity
        p["u_shadow_texel"].value = self._shadow_texel
        p["u_tex"].value = 0
        p["u_shadow"].value = 1
        self._shadow_depth.use(location=1)

        for g in groups:
            gl = self._geom_gl(g.geo)
            if g.use_tex:
                p["u_use_tex"].value = 1
                self._texture(g.texture_key).use(location=0)
            else:
                p["u_use_tex"].value = 0
                p["u_tint"].value = g.tint
            p["u_receive_shadow"].value = 1 if (g.receive and self.shadows_enabled) else 0
            gl.vao_main.render(instances=len(g.mats))

    def _shadow_pass(self, groups: List[_Group]) -> None:
        self._shadow_fbo.use()
        self.ctx.viewport = (0, 0, self._shadow_size, self._shadow_size)
        self._shadow_fbo.clear(depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.depth_prog["u_light_vp"].write(mu.to_gl(self._light_vp))
        for g in groups:
            if not g.cast:
                continue
            self._geom_gl(g.geo).vao_depth.render(instances=len(g.mats))

    def blit_to_screen(self, screen_fbo=None) -> None:
        """Draw the scene FBO to fill the whole default framebuffer (no letterbox)."""
        target = screen_fbo or self.ctx.screen
        target.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.color_tex.use(location=0)
        self._blit_prog["tex"].value = 0
        self._blit_vao.render()

    # -- letterboxed presentation -----------------------------------------
    def clear_window(self, bg, win_w: int, win_h: int) -> None:
        """Fill the whole default framebuffer with the letterbox background colour."""
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, win_w, win_h)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.clear(bg[0] / 255.0, bg[1] / 255.0, bg[2] / 255.0, 1.0)

    def present_scene(self, rect) -> None:
        """Draw the scene FBO into a sub-rectangle of the window (aspect-preserved)."""
        self.ctx.screen.use()
        self.ctx.viewport = tuple(int(v) for v in rect)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.color_tex.use(location=0)
        self._blit_prog["tex"].value = 0
        self._blit_vao.render()

    def draw_overlay(self, key: str, rgba, ui_w: int, ui_h: int, rect) -> None:
        """Draw a cached RGBA overlay (keyed) into a window sub-rectangle, blended.

        Pass ``rgba`` bytes to (re)upload; pass ``None`` to redraw the cached texture
        (so static overlays like the device bezel upload only when they change).
        """
        tex = self._overlays.get(key)
        if tex is None or tex.size != (ui_w, ui_h):
            if tex is not None:
                tex.release()
            tex = self.ctx.texture((ui_w, ui_h), 4)
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._overlays[key] = tex
            if rgba is None:
                return
        if rgba is not None:
            tex.write(rgba)
        self.ctx.screen.use()
        self.ctx.viewport = tuple(int(v) for v in rect)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        tex.use(location=0)
        self._blit_prog["tex"].value = 0
        self._blit_vao.render()
        self.ctx.disable(moderngl.BLEND)

    def finish(self) -> None:
        """Force the GPU to complete all submitted work.

        Called right before the buffer swap on hybrid-GPU (NVIDIA PRIME render-offload)
        setups: it guarantees the back buffer is fully rendered before the cross-GPU
        presentation copy reads it, eliminating intermittent black/torn frames. The CPU
        stall is tiny (the frame's GPU work is a few ms) and only happens once per frame.
        """
        self.ctx.finish()

    def read_pixels(self) -> np.ndarray:
        data = self.scene_fbo.read(components=3, alignment=1)
        return np.flipud(np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3))
