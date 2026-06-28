"""Global game constants.

Core gameplay settings plus a handful of derived/tuning constants used by the
Python renderer and app shell. Keeping every gameplay-affecting number here in one
place makes the tuning easy to audit and adjust.
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Core gameplay settings
# ---------------------------------------------------------------------------
GROUND_LEVEL: float = 0.4            # groundLevel
SCENE_COLOR_HEX: int = 0x87C6FF      # sceneColor (sky blue)
STARTING_ROW: int = 8                # startingRow — player spawns here, scores 0
MAX_ROWS: int = 20                   # maxRows — size of each recycled row pool
DISABLE_DRIFTWOOD: bool = False      # disableDriftwood
CAMERA_EASING: float = 0.03          # CAMERA_EASING — world follow lerp factor
MAP_OFFSET: float = -30.0            # MAP_OFFSET — parking spot for inactive rows
BASE_ANIMATION_TIME: float = 0.1     # BASE_ANIMATION_TIME — one jump-arc phase (s)
IDLE_DURING_GAME_PLAY: bool = False  # IDLE_DURING_GAME_PLAY
PI_2: float = math.pi * 0.5          # PI_2
PLAYER_IDLE_SCALE: float = 0.8       # PLAYER_IDLE_SCALE
DEBUG_CAMERA_CONTROLS: bool = False  # DEBUG_CAMERA_CONTROLS

# Convenience colour tuple (0..1 floats) for clearing the GL framebuffer.
SCENE_COLOR = (
    ((SCENE_COLOR_HEX >> 16) & 0xFF) / 255.0,
    ((SCENE_COLOR_HEX >> 8) & 0xFF) / 255.0,
    (SCENE_COLOR_HEX & 0xFF) / 255.0,
)

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
# Orientation is fixed by looking at the target from (-1, 2.8, -2.9); the runtime
# camera keeps that viewing angle but sits at z=1.
CAMERA_ORIENT_EYE = (-1.0, 2.8, -2.9)  # eye that defines the viewing angle
CAMERA_POSITION = (-1.0, 2.8, 1.0)     # runtime camera position (z = 1)
CAMERA_TARGET = (0.0, 0.0, 0.0)        # look-at target
CAMERA_NEAR = -30.0
CAMERA_FAR = 30.0
# Frame a fixed number of world units so the view matches regardless of resolution.
# ~9 playable columns + a little margin → half-width of ~5.2 world units.
CAMERA_VIEW_HALF_WIDTH = 5.2

# ---------------------------------------------------------------------------
# Lighting (directional + ambient lights)
# ---------------------------------------------------------------------------
DIR_LIGHT_POSITION = (20.0, 30.0, 0.05)   # directional light position
DIR_LIGHT_INTENSITY = 1.0
AMBIENT_LIGHT_INTENSITY = 1.8             # white ambient light intensity
# Shadow camera bounds (left, right, top, bottom, far)
SHADOW_CAM_LEFT = -15.0
SHADOW_CAM_RIGHT = 9.0
SHADOW_CAM_TOP = 6.0
SHADOW_CAM_BOTTOM = -6.0
SHADOW_CAM_FAR = 100.0
SHADOW_BIAS = 0.0008
# 1024² is ample for the blocky shadows over the ~24-unit play area (~43 texels/unit) and
# keeps the per-frame shadow depth pass cheap so a GPU sync before swap stays well under 120 FPS.
SHADOW_MAP_SIZE = 1024

# ---------------------------------------------------------------------------
# Player / world tuning
# ---------------------------------------------------------------------------
HERO_WIDTH = 0.7                     # hero collision width
EDGE_DEATH_X = 5.0                   # |x| > 5 → off-screen death

# Row "top" surface heights (the Y the player rests at on each row type)
GRASS_TOP = 0.4
ROAD_TOP = 0.3
WATER_TOP = 0.25
RAILROAD_TOP = 0.5

# ---------------------------------------------------------------------------
# App-shell / window settings
# ---------------------------------------------------------------------------
WINDOW_WIDTH = 480
WINDOW_HEIGHT = 820                  # portrait, phone-like aspect
TARGET_FPS = 120
WINDOW_TITLE = "Crossy Road — PyCrossy"

# Prefer the dedicated/high-performance GPU when one is available (set PYCROSSY_GPU=
# integrated, or this flag to False, to keep the default device).
PREFER_DEDICATED_GPU = True

# Force GPU completion before each buffer swap. Required for flicker-free presentation on
# hybrid-GPU laptops (NVIDIA PRIME render-offload), where the cross-GPU display copy can
# otherwise read an incomplete back buffer and show black/torn frames. Negligible cost.
GPU_SYNC_BEFORE_SWAP = True

# Logical update rate. Vehicles/logs/trains move by PER-FRAME increments (so their
# intended speed is their 60 Hz behaviour) while easing tweens use wall-clock seconds.
# We therefore step game logic at a FIXED 60 Hz timestep — one logic frame per tick —
# and render at up to TARGET_FPS. This keeps motion speed independent of render rate.
FIXED_TICK_HZ = 60.0
FIXED_DT = 1.0 / FIXED_TICK_HZ

# Tile geometry helpers (rows are 25 units wide, 1 unit deep).
ROW_WIDTH = 25.0

# How many rows to keep generated AHEAD of the player. The camera-follow eases (lags) during
# continuous movement, which shifts the framing so the top of the screen looks further ahead
# than the equilibrium position — so the buffer must comfortably exceed the number of rows
# visible at once (~15 in portrait) to prevent the sky showing past the furthest row.
ROW_LOOKAHEAD = 22

# Difficulty multiplier applied to vehicle / river-mover speeds (1.0 = default tuning).
# Set at runtime by the Gameplay ▸ Difficulty setting (easy < 1.0 < hard).
DIFFICULTY_SPEED = 1.0
