"""Game engine — the core fixed-timestep game loop.

Owns the scene graph (scene > worldWithCamera > world), the player, the row map, and the
particle systems. Drives one fixed-timestep tick: row updates, passive movement (riding /
being hit), camera follow with new-row generation, fall/edge death checks, and the swipe
movement state machine. UI/audio are injected via callbacks so the same engine runs both
the playable app and the headless AI environment.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Callable, Optional

from . import config, primitives
from .mathutils import clamp, normalize_angle
from .models import registry
from .particles import Feathers, Water
from .scene import Group, Mesh, Vec3
from .tween import tween
from .world import CrossyGameMap
from .entities.player import Player

# A large flat ground slab sits just beneath every row, fixed relative to the camera, so if
# the camera-follow lag or row recycling ever leaves the visible field momentarily uncovered,
# the ground shows through instead of the sky — never a flickering blue gap. Tinted to a
# grass green so it reads as the field extending to the horizon.
_GROUND_GEO = primitives.box(120.0, 0.5, 160.0)
_GROUND_TINT = (0.36, 0.56, 0.20, 1.0)

PI_2 = config.PI_2


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class Engine:
    def __init__(self, audio=None):
        self.audio = audio
        self.scene: Optional[Group] = None
        self.world_with_camera: Optional[Group] = None
        self.world: Optional[Group] = None
        self.game_map: Optional[CrossyGameMap] = None
        self.hero: Optional[Player] = None
        self.feathers: Optional[Feathers] = None
        self.water_particles: Optional[Water] = None
        self.cam_count = 0.0
        # Live gameplay knobs (overridden by settings; default tuning shown here).
        self.camera_easing = config.CAMERA_EASING
        self.screen_shake = True

        # Callbacks (set by the host app / AI env); default to no-ops.
        self.on_update_score: Callable[[int], None] = lambda pos: None
        self.on_game_init: Callable[[], None] = lambda: None
        self.on_game_ready: Callable[[], None] = lambda: None
        self.on_game_ended: Callable[[], None] = lambda: None
        self.is_game_state_ended: Callable[[], bool] = lambda: False

    # -- setup -------------------------------------------------------------
    def setup_game(self, character: str = "chicken") -> None:
        reg = registry()
        # Fresh game -> drop any tweens from a previous game (foam loops forever, so the
        # global manager would otherwise grow without bound across restarts). Safe because
        # all gameplay tweens are created *after* this point.
        tween.clear()
        self.scene = Group()
        self.world_with_camera = Group()
        self.world = Group()
        self.world_with_camera.add(self.world)
        self.scene.add(self.world_with_camera)
        self.world_with_camera.position.z = -config.STARTING_ROW

        # Ground backdrop, fixed relative to the camera and just below every row, so no sky
        # can ever show through (camera lag / row recycling / aspect changes all become safe).
        self.ground = Mesh(_GROUND_GEO, None, tint=_GROUND_TINT,
                           cast_shadow=False, receive_shadow=False)
        self.ground.position.set(0.0, -0.35, config.STARTING_ROW)
        self.ground.mark_static()
        self.world_with_camera.add(self.ground)

        self.game_map = CrossyGameMap(reg, config.HERO_WIDTH, self.world, self.on_collide, self.audio)
        self.cam_count = 0.0

        self.hero = Player(character)
        self.world.add(self.hero)

        self.feathers = Feathers()
        self.water_particles = Water()
        self.world.add(self.feathers.mesh)
        self.world.add(self.water_particles.mesh)

    def is_game_ended(self) -> bool:
        return (not self.hero.is_alive) or self.is_game_state_ended()

    # -- collision / death -------------------------------------------------
    def on_collide(self, obstacle=None, type: str = "feathers", collision: Optional[str] = None) -> None:
        if self.is_game_ended():
            return
        self.hero.is_alive = False
        self.hero.stop_idle()
        if self.audio:
            if collision == "car":
                self.audio.play_car_hit_sound()
                self.audio.play_death_sound()
            elif collision == "train":
                self.audio.play(self.audio.sounds["train"]["die"]["0"])
                self.audio.play_death_sound()
        speed = getattr(obstacle, "speed", 0) or 0
        self.use_particle(self.hero, type, speed)
        self.rumble()
        self.game_over()

    def use_particle(self, model, type: str, direction: float = 0.0) -> None:
        if type == "water":
            self.water_particles.mesh.position.copy(model.position)
            self.water_particles.run(type)
            if self.audio:
                self.audio.play(self.audio.sounds["water"])
        elif type == "feathers":
            self.feathers.mesh.position.copy(model.position)
            self.feathers.run(type, direction)

    def reset_particles(self, position) -> None:
        self.feathers.mesh.position.copy(position)
        self.water_particles.mesh.position.copy(position)
        self.feathers.mesh.position.y = 0
        self.water_particles.mesh.position.y = 0

    def rumble(self) -> None:
        if not self.screen_shake:
            return
        tween.to(self.scene.position, 0.2, x=0, y=0, z=1)
        tween.to(self.scene.position, 0.2, delay=0.2, x=0, y=0, z=0)

    def game_over(self) -> None:
        self.hero.moving = False
        self.hero.stop_animations()
        self.on_game_ended()

    # -- lifecycle ---------------------------------------------------------
    def init(self) -> None:
        self.on_game_init()
        # Reset the camera-follow transforms so a re-init (restart) always starts aligned,
        # even if init() is called without a fresh setup_game().
        self.world.position.set(0.0, 0.0, 0.0)
        self.scene.position.set(0.0, 0.0, 0.0)
        self.hero.reset()
        self.reset_particles(self.hero.position)
        self.cam_count = 0.0
        self.game_map.reset()
        self.hero.idle()
        self.game_map.init()
        self.on_game_ready()

    # -- per-frame ---------------------------------------------------------
    def forward_scene(self) -> None:
        w = self.world
        ease = self.camera_easing
        w.position.z -= (self.hero.position.z - config.STARTING_ROW + w.position.z) * ease
        target_x = clamp(-self.hero.position.x, -3, 2)
        w.position.x += (target_x - w.position.x) * ease
        # Generate rows so a fixed buffer always exists ahead of the PLAYER (not the lagging
        # eased camera), guaranteeing the top of the screen is always covered by terrain.
        target = int(self.hero.position.z) + config.ROW_LOOKAHEAD
        while self.game_map.row_count < target:
            self.cam_count = -w.position.z
            self.game_map.new_row()

    def tick(self, dt: float) -> None:
        self.game_map.tick(dt, self.hero)
        if not self.hero.moving:
            self.hero.move_on_entity()
            self.hero.move_on_car()
            self.check_fallen_out_of_frame()
        self.forward_scene()

    def check_fallen_out_of_frame(self) -> None:
        if self.is_game_ended():
            return
        # camera.position.z == 1 (set in init); hero local z stays >= 0 so this is a guard.
        if self.hero.position.z < config.CAMERA_POSITION[2] - 1:
            self.rumble()
            self.game_over()
            if self.audio:
                self.audio.play_death_sound()
        if self.hero.position.x < -config.EDGE_DEATH_X or self.hero.position.x > config.EDGE_DEATH_X:
            self.rumble()
            self.game_over()
            if self.audio:
                self.audio.play_death_sound()

    def update_score(self) -> None:
        position = max(math.floor(self.hero.position.z) - config.STARTING_ROW, 0)
        self.on_update_score(position)

    # -- input -------------------------------------------------------------
    def begin_move_with_direction(self) -> None:
        if self.is_game_ended():
            return
        self.hero.run_posie_animation()

    def move_with_direction(self, direction: Direction) -> None:
        if self.is_game_ended():
            return
        hero = self.hero
        hero.riding_on = None

        if hero.initial_position is None:
            hero.initial_position = hero.position          # alias (same object, not a copy)
            hero.target_position = hero.initial_position

        hero.skip_pending_movement()

        velocity = Vec3(0, 0, 0)
        hero.target_rotation = normalize_angle(hero.rotation.y)
        init = hero.initial_position

        if direction == Direction.LEFT:
            hero.target_rotation = PI_2
            velocity = Vec3(1, 0, 0)
            hero.target_position = Vec3(init.x + 1, init.y, init.z)
            hero.moving = True
        elif direction == Direction.RIGHT:
            if (int(hero.target_rotation) != -int(PI_2)
                    and int(hero.target_rotation) != int(math.pi + PI_2)):
                hero.target_rotation = math.pi + PI_2
            velocity = Vec3(-1, 0, 0)
            hero.target_position = Vec3(init.x - 1, init.y, init.z)
            hero.moving = True
        elif direction == Direction.UP:
            hero.target_rotation = 0.0
            row_object = self.game_map.get_row(init.z) or {}
            if row_object.get("type") == "road" and self.audio:
                self.audio.play_passive_car_sound()
            velocity = Vec3(0, 0, 1)
            hero.target_position = Vec3(init.x, init.y, init.z + 1)
            self._snap_target_x(hero)
            hero.moving = True
        elif direction == Direction.DOWN:
            hero.target_rotation = math.pi
            velocity = Vec3(0, 0, -1)
            hero.target_position = Vec3(init.x, init.y, init.z - 1)
            self._snap_target_x(hero)
            hero.moving = True

        # Tree/obstacle collision -> hop in place.
        if self.game_map.tree_collision(hero.target_position):
            hero.target_position = Vec3(init.x, init.y, init.z)
            hero.moving = False

        target_row = self.game_map.get_row(init.z + velocity.z) or {}
        entity = target_row.get("entity")
        final_y = (entity.top if entity is not None else config.GROUND_LEVEL)
        if target_row.get("type") == "water":
            ridable = entity.get_ridable_for_position(hero.target_position)
            if not ridable:
                final_y = entity.get_player_sunken_position()
            else:
                final_y = entity.get_player_lower_bounce_position_for_entity(ridable)

        if self.audio:
            self.audio.play_move_sound()

        hero.target_position.y = final_y
        hero.commit_movement_animations(on_complete=self.update_score)

    @staticmethod
    def _snap_target_x(hero) -> None:
        hero.target_position.x = round(hero.target_position.x)
        riding = hero.riding_on
        if riding and riding.dir:
            if riding.dir < 0:
                hero.target_position.x = math.floor(hero.target_position.x)
            elif riding.dir > 0:
                hero.target_position.x = math.ceil(hero.target_position.x)
            else:
                hero.target_position.x = round(hero.target_position.x)
