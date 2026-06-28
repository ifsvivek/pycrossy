"""GPU detection, high-performance selection, and startup logging.

On hybrid laptops (e.g. NVIDIA dGPU + AMD/Intel iGPU) the default OpenGL renderer is
usually the integrated GPU. This module requests the **dedicated** GPU before any GL
context is created — via NVIDIA PRIME render-offload env vars for the windowed (GLX/X11)
path, and EGL device enumeration for the headless path — then logs which GPU is actually
in use and, if stuck on the iGPU while a dedicated GPU exists, prints instructions.

Selection is best-effort and always falls back to the best available device, so the game
runs on any machine (including pure-integrated or software rasterizers).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, Optional

# Renderer-string keywords that indicate a high-performance / discrete GPU.
_HIGH_PERF_KEYWORDS = ("nvidia", "geforce", "rtx", "gtx", "quadro", "radeon rx",
                       "radeon pro", "arc ")


def detect_nvidia() -> Optional[Dict[str, str]]:
    """Return ``{name, driver, vram_mb}`` for the first NVIDIA GPU, or None."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        name, driver, vram = (p.strip() for p in out.stdout.strip().splitlines()[0].split(","))
        return {"name": name, "driver": driver, "vram_mb": vram}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _prefer_native_video_driver() -> str:
    """Pick SDL's native video backend for the session (call before pygame.init).

    Crucially, on a Wayland session (e.g. Hyprland/Sway) use the **wayland** backend rather
    than routing through XWayland — XWayland + a tiling Wayland compositor + NVIDIA is a
    notorious source of texture flicker, black frames and windows that appear to close/reopen.
    NVIDIA PRIME render-offload still selects the dedicated GPU under native Wayland.
    A user-set ``SDL_VIDEODRIVER`` is always respected.
    """
    if "SDL_VIDEODRIVER" in os.environ:
        return os.environ["SDL_VIDEODRIVER"]
    if os.environ.get("WAYLAND_DISPLAY"):
        os.environ["SDL_VIDEODRIVER"] = "wayland"
    elif os.environ.get("DISPLAY"):
        os.environ["SDL_VIDEODRIVER"] = "x11"
    return os.environ.get("SDL_VIDEODRIVER", "")


def prefer_high_performance_gpu(enable: bool = True) -> Dict[str, object]:
    """Set env vars to request the dedicated GPU + the native window backend.

    Call BEFORE creating any window/GL context (and before spawning subprocesses, so they
    inherit it). Returns a dict describing the request. ``PYCROSSY_GPU=integrated`` opts out
    of the dedicated GPU (the native video backend is still preferred, for stability).
    """
    if os.environ.get("PYCROSSY_GPU", "").lower() == "integrated":
        enable = False
    nvidia = detect_nvidia()
    driver = _prefer_native_video_driver()
    if not enable or nvidia is None:
        return {"requested": "default", "nvidia": nvidia, "video_driver": driver}

    # NVIDIA PRIME render offload — selects the dGPU under both X11 (GLX) and Wayland (EGL).
    os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    os.environ.setdefault("__VK_LAYER_NV_optimus", "NVIDIA_only")
    return {"requested": "nvidia", "nvidia": nvidia, "video_driver": driver}


def safe_set_mode(size, flags, vsync: bool = False):
    """``pygame.display.set_mode`` with graceful fallbacks.

    Tries vsync → no-vsync, and if the chosen video backend can't create the window at all,
    drops the forced ``SDL_VIDEODRIVER`` and lets SDL auto-select — so the app starts on any
    session (Wayland, X11 or headless) without crashing.
    """
    import pygame
    for use_vsync in ((True, False) if vsync else (False,)):
        try:
            return (pygame.display.set_mode(size, flags, vsync=1) if use_vsync
                    else pygame.display.set_mode(size, flags))
        except pygame.error:
            continue
    if os.environ.pop("SDL_VIDEODRIVER", None) is not None:
        try:
            pygame.display.quit()
            pygame.display.init()
        except pygame.error:
            pass
    return pygame.display.set_mode(size, flags)


def create_standalone_context(prefer_high_perf: bool = True):
    """Create a headless moderngl context on the dedicated GPU when possible."""
    import moderngl
    if prefer_high_perf and detect_nvidia() is not None:
        for idx in range(8):
            try:
                ctx = moderngl.create_standalone_context(backend="egl", device_index=idx)
            except Exception:
                break
            if any(k in ctx.info["GL_RENDERER"].lower() for k in _HIGH_PERF_KEYWORDS):
                return ctx
            ctx.release()
    return moderngl.create_standalone_context()


def gl_info(ctx) -> Dict[str, str]:
    info = ctx.info
    code = getattr(ctx, "version_code", None)         # e.g. 460 for GLSL 4.60
    glsl = f"{code // 100}.{(code % 100) // 10}0" if code else "—"
    return {
        "renderer": info.get("GL_RENDERER", "—"),
        "vendor": info.get("GL_VENDOR", "—"),
        "version": info.get("GL_VERSION", "—"),
        "glsl": glsl,
    }


def is_high_performance(renderer: str) -> bool:
    r = renderer.lower()
    return any(k in r for k in _HIGH_PERF_KEYWORDS)


def log_startup(ctx, requested: Optional[Dict] = None, headless: bool = False) -> Dict[str, str]:
    """Print a GPU banner and return the resolved info. Warn if stuck on the iGPU."""
    info = gl_info(ctx)
    nvidia = (requested or {}).get("nvidia") or detect_nvidia()
    on_high_perf = is_high_performance(info["renderer"])
    vram = nvidia["vram_mb"] + " MiB" if (nvidia and on_high_perf and "nvidia" in info["renderer"].lower()) else "—"
    driver = nvidia["driver"] if (nvidia and "nvidia" in info["renderer"].lower()) else "—"
    api = f"OpenGL {info['version'].split(' ')[0]}" + (" (headless/EGL)" if headless else "")
    backend = (requested or {}).get("video_driver") or os.environ.get("SDL_VIDEODRIVER", "auto")

    bar = "─" * 58
    print(f"┌{bar}┐")
    print(f"│ GPU       : {info['renderer'][:44]:<44} │")
    print(f"│ Vendor    : {info['vendor'][:44]:<44} │")
    print(f"│ API       : {api[:44]:<44} │")
    print(f"│ GLSL      : {info['glsl'][:44]:<44} │")
    print(f"│ Driver    : {driver[:44]:<44} │")
    print(f"│ VRAM      : {vram[:44]:<44} │")
    print(f"│ Backend   : {('window: ' + str(backend))[:44]:<44} │")
    print(f"│ Mode      : {'HIGH-PERFORMANCE (dedicated)' if on_high_perf else 'integrated / software':<44} │")
    print(f"└{bar}┘")

    if not on_high_perf and nvidia is not None:
        print(f"[GPU] ⚠ Running on the integrated GPU while a dedicated "
              f"{nvidia['name']} is available.")
        print("      To use the dedicated GPU, launch with:")
        print("        __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \\")
        print("        SDL_VIDEODRIVER=x11 python main.py")
        print("      (or configure PRIME render-offload in your system / NVIDIA settings).")
    return {**info, "high_performance": str(on_high_perf), "vram": vram, "driver": driver}
