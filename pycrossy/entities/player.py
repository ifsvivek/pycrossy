"""The player character and its animation behaviours.

Drives the hop arc (two 0.1 s phases with a 0.5-unit apex at 75% of the lateral
delta), the squash/stretch scale timeline, the rotation tween, idle bob, the anticipation
"posie" squash on key-down, and the car/train hit deformations.
"""
from __future__ import annotations

import math
import random
from typing import Callable, List, Optional

from .. import config
from ..mathutils import normalize_angle
from ..models import registry
from ..scene import Group, Vec3
from ..tween import tween

BASE = config.BASE_ANIMATION_TIME


class Player(Group):
    def __init__(self, character: str = "chicken"):
        super().__init__()
        self.animations: List = []
        self._character: Optional[str] = None
        self.node = None

        self.initial_position: Optional[Vec3] = None
        self.target_position: Optional[Vec3] = None
        self.target_rotation: float = 0.0
        self.moving = False
        self.hit_by = None
        self.hit_by_train = None
        self.riding_on = None
        self.riding_on_offset: Optional[float] = None
        self.is_alive = True
        self.idle_animation = None
        self.last_position: Optional[Vec3] = None

        self.set_character(character)
        self.reset()

    # -- character ---------------------------------------------------------
    def set_character(self, character: str) -> None:
        if self._character == character:
            return
        self._character = character
        node = registry().make_hero_node(character)
        if self.node is not None:
            self.remove(self.node)
        self.node = node
        self.add(node)

    # -- passive movement (riding / being hit) -----------------------------
    def move_on_entity(self) -> None:
        if not self.riding_on:
            return
        self.position.x += self.riding_on.speed
        if self.initial_position:
            self.initial_position.x = self.position.x

    def move_on_car(self) -> None:
        if not self.hit_by:
            return
        target = self.hit_by.mesh.position.x
        self.position.x += self.hit_by.speed
        if self.initial_position:
            self.initial_position.x = target

    # -- animation control -------------------------------------------------
    def stop_animations(self) -> None:
        for a in self.animations:
            if hasattr(a, "pause"):
                a.pause()
        self.animations = []

    def reset(self) -> None:
        self.position.set(0, config.GROUND_LEVEL, config.STARTING_ROW)
        self.scale.set(1, 1, 1)
        self.rotation.set(0, math.pi, 0)
        self.initial_position = None
        self.target_position = None
        self.moving = False
        self.hit_by = None
        self.hit_by_train = None
        self.riding_on = None
        self.riding_on_offset = None
        self.is_alive = True

    def skip_pending_movement(self) -> None:
        if not self.moving:
            return
        self.position.set(self.target_position.x, self.target_position.y, self.target_position.z)
        if self.target_rotation:
            self.rotation.y = normalize_angle(self.target_rotation)

    def finished_moving_animation(self) -> None:
        self.moving = False
        if config.IDLE_DURING_GAME_PLAY:
            if self.idle_animation:
                self.idle_animation.play()
            else:
                self.idle()
        self.last_position = self.position.clone()

    def stop_idle(self) -> None:
        if self.idle_animation and hasattr(self.idle_animation, "pause"):
            self.idle_animation.pause()
        self.idle_animation = None
        self.scale.set(1, 1, 1)

    def idle(self) -> None:
        if self.idle_animation:
            return
        self.stop_idle()
        tl = tween.timeline(repeat=-1)
        tl.to(self.scale, 0.3, y=config.PLAYER_IDLE_SCALE, ease="power1.in")
        tl.to(self.scale, 0.3, y=1.0, ease="power1.out")
        self.idle_animation = tl

    def run_posie_animation(self) -> None:
        self.stop_idle()
        tween.to(self.scale, 0.2, x=1.2, y=0.75, z=1)

    # -- the hop -----------------------------------------------------------
    def commit_movement_animations(self, on_complete: Callable) -> None:
        init = self.initial_position
        tgt = self.target_position
        dx = tgt.x - init.x
        dz = tgt.z - init.z
        in_air = (init.x + dx * 0.75, tgt.y + 0.5, init.z + dz * 0.75)

        def _finish():
            self.finished_moving_animation()
            on_complete()

        pos_tl = tween.timeline(on_complete=_finish)
        pos_tl.to(self.position, BASE, x=in_air[0], y=in_air[1], z=in_air[2])
        pos_tl.to(self.position, BASE, x=tgt.x, y=tgt.y, z=tgt.z)

        scale_tl = tween.timeline()
        scale_tl.to(self.scale, BASE, x=1, y=1.2, z=1)
        scale_tl.to(self.scale, BASE, x=1.0, y=0.8, z=1)
        scale_tl.to(self.scale, BASE, x=1, y=1, z=1, ease="bounce.out")

        def _norm_rot():
            self.rotation.y = normalize_angle(self.rotation.y)

        rot = tween.to(self.rotation, BASE, y=self.target_rotation,
                       ease="power1.inout", on_complete=_norm_rot)

        self.animations = [pos_tl, scale_tl, rot]
        self.initial_position = self.target_position

    # -- collisions (deformations) ----------------------------------------
    def collide_with_car(self, road, car) -> None:
        if self.moving and abs(self.position.z - round(self.position.z)) > 0.1:
            self.get_hit_by_car(road, car)
        else:
            self.get_run_over_by_car(road, car)

    def get_run_over_by_car(self, road, car) -> None:
        self.position.y = road.top - 0.05
        tween.to(self.scale, 0.2, y=0.05, x=1.7, z=1.7)
        tween.to(self.rotation, 0.2, y=random.random() * math.pi - math.pi / 2)

    def get_hit_by_car(self, road, car) -> None:
        self.hit_by = car
        forward = (self.position.z - round(self.position.z)) > 0
        self.position.z = road.position.z + (0.52 if forward else -0.52)
        tween.to(self.scale, 0.15, y=1.5, z=0.2)
        tween.to(self.rotation, 0.15, z=random.random() * math.pi - math.pi / 2)
