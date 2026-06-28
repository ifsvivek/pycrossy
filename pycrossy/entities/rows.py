"""Row entities — grass, road, water, and railroad lanes.

Each row is a transform node holding a floor mesh plus its spawned objects (trees/boulders,
cars, logs/lily pads, trains). Spawn parameters, movement (per-frame increments at the
fixed 60 Hz tick), collision boxes and hazard logic are defined here.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional

from .. import config
from ..models import ModelRegistry
from ..particles import Foam
from ..scene import Group, Mesh, Object3D
from ..tween import tween

OFFSET = 11  # car/log wrap distance
TRAIN_OFFSET = 22 * 5


class Mover:
    """A spawned, possibly-moving object (car/log/lily/train) with collision metadata."""

    __slots__ = ("mesh", "dir", "width", "collision_box", "speed", "top", "min", "mid", "_rot_tween")

    def __init__(self, mesh: Mesh, dir: int = 0, width: float = 0.0, collision_box: float = 0.0):
        self.mesh = mesh
        self.dir = dir
        self.width = width
        self.collision_box = collision_box
        self.speed = 0.0
        self.top = 0.0
        self.min = 0.0
        self.mid = 0.0
        self._rot_tween = None


def _round_extent(geo, axis: int) -> int:
    return int(round(float(geo.size[axis])))


# ---------------------------------------------------------------------------
# Grass
# ---------------------------------------------------------------------------
class GrassRow(Object3D):
    SOLID = "solid"
    EMPTY = "empty"
    RANDOM = "random"

    def __init__(self, reg: ModelRegistry, hero_width: float, on_collide):
        super().__init__()
        self.reg = reg
        self.hero_width = hero_width
        self.on_collide = on_collide
        self.top = config.GRASS_TOP
        self.entities: List[Mesh] = []
        self.obstacle_map = {}
        self.required_clear = set()
        self.floor = reg.grass.make("0")
        self.floor.mark_static()
        self.add(self.floor)

    def generate(self, fill: str = "random", required_clear_positions: Optional[List[int]] = None) -> None:
        for e in self.entities:
            self.floor.remove(e)
        self.entities = []
        self.obstacle_map = {}
        self.required_clear = set(required_clear_positions or [])
        # Alternate light/dark grass by row parity (both textures are available).
        self.floor.texture_key = self.reg.grass.specs["1" if int(round(self.position.z)) % 2 else "0"].texture
        self._tree_gen(fill)

    def _add_obstacle(self, x: int) -> None:
        if int(x) in self.required_clear:
            return
        if random.random() < 0.4:
            mesh = self.reg.boulder.make_random()
        else:
            mesh = self.reg.tree.make_random()
        self.obstacle_map[int(x)] = {"index": len(self.entities)}
        self.entities.append(mesh)
        self.floor.add(mesh)
        mesh.position.set(x, config.GROUND_LEVEL, 0)
        mesh.mark_static()

    def get_blocked_positions(self) -> List[int]:
        return list(self.obstacle_map.keys())

    def _tree_gen(self, fill: str) -> None:
        row_count = 0
        count = round(random.random() * 2) + 1
        for x in range(-3, 12):
            _x = x - 4
            if fill == self.SOLID:
                self._add_obstacle(_x)
                continue
            if x >= 9 or x <= -1:                 # walls
                self._add_obstacle(_x)
                continue
            if row_count < count and _x != 0 and random.random() > 0.6:
                self._add_obstacle(_x)
                row_count += 1


# ---------------------------------------------------------------------------
# Road
# ---------------------------------------------------------------------------
class RoadRow(Object3D):
    def __init__(self, reg: ModelRegistry, hero_width: float, on_collide):
        super().__init__()
        self.reg = reg
        self.hero_width = hero_width
        self.on_collide = on_collide
        self.active = False
        self.top = config.ROAD_TOP
        self.cars: List[Mover] = []
        self.floor = reg.road.make("1")          # default blank
        self.floor.mark_static()
        self.add(self.floor)
        self.car_gen()

    def is_first_lane(self, is_first: bool) -> None:
        self.floor.texture_key = self.reg.road.specs["1" if is_first else "0"].texture

    def car_gen(self) -> None:
        for c in self.cars:
            self.floor.remove(c.mesh)
        self.cars = []
        speed = (random.random() * 0.06 + 0.02) * config.DIFFICULTY_SPEED
        num_cars = math.floor(random.random() * 2) + 1
        x_dir = 1 if random.random() <= 0.5 else -1
        x_pos = -6 * x_dir
        for i in range(num_cars):
            mesh = self.reg.car.make_random()
            width = _round_extent(mesh.geometry, 2)
            mover = Mover(mesh, dir=x_dir, width=width,
                          collision_box=self.hero_width / 2 + width / 2 - 0.1)
            self.cars.append(mover)
            self.floor.add(mesh)
            mover.mesh.position.set(x_pos, 0.25, 0)
            mover.speed = speed * x_dir
            mover.mesh.rotation.y = (math.pi / 2) * x_dir
            x_pos -= (random.random() * 3 + 5) * x_dir

    def update(self, dt: float, player) -> None:
        if not self.active:
            return
        for car in self.cars:
            self._drive(player, car)

    def _drive(self, player, car: Mover) -> None:
        car.mesh.position.x += car.speed
        if car.mesh.position.x > OFFSET and car.speed > 0:
            car.mesh.position.x = -OFFSET
            if car is player.hit_by:
                player.hit_by = None
        elif car.mesh.position.x < -OFFSET and car.speed < 0:
            car.mesh.position.x = OFFSET
            if car is player.hit_by:
                player.hit_by = None
        else:
            self._check_collision(player, car)

    def _check_collision(self, player, car: Mover) -> None:
        if round(player.position.z) == round(self.position.z) and player.is_alive:
            cx = car.mesh.position.x
            if cx - car.collision_box < player.position.x < cx + car.collision_box:
                player.collide_with_car(self, car)
                self.on_collide(car, "feathers", "car")


# ---------------------------------------------------------------------------
# Water
# ---------------------------------------------------------------------------
class WaterRow(Object3D):
    def __init__(self, reg: ModelRegistry, hero_width: float, on_collide):
        super().__init__()
        self.reg = reg
        self.hero_width = hero_width
        self.on_collide = on_collide
        self.active = False
        self.top = config.WATER_TOP
        self.entities: List[Mover] = []
        self.lily_pad_positions: List[int] = []
        self.sine_count = 0.0
        self.sine_inc = math.pi / 50
        self.floor = reg.river.make("0")
        self.floor.mark_static()
        self.add(self.floor)
        foam_l = Foam(1)
        foam_l.position.set(4.5, 0.2, -0.5)
        foam_l.run()
        self.add(foam_l)
        foam_r = Foam(-1)
        foam_r.position.set(-4.5, 0.2, -0.5)
        foam_r.run()
        self.add(foam_r)

    def is_static_row(self, index: int) -> bool:
        return index % 2 == 0

    def generate(self, clear_positions: Optional[List[int]] = None) -> None:
        for e in self.entities:
            self.floor.remove(e.mesh)
            rot = getattr(e, "_rot_tween", None)   # stop the infinite lily-pad spin
            if rot is not None:
                rot.kill()
        self.entities = []
        self.lily_pad_positions = []
        if self.is_static_row(int(round(self.position.z))):
            self._generate_static(clear_positions or [])
        elif not config.DISABLE_DRIFTWOOD:
            self._generate_dynamic()

    def get_lily_pad_positions(self) -> List[int]:
        return self.lily_pad_positions

    def _generate_static(self, clear_positions: List[int]) -> None:
        num_items = math.floor(random.random() * 2) + 2
        positions: List[int] = []
        x_pos = math.floor(random.random() * 2 - 4)
        for _ in range(num_items):
            positions.append(x_pos)
            x_pos += math.floor(random.random() * 2 + 2)
        if clear_positions:
            if not any((p | 0) in clear_positions for p in positions):
                clear_pos = random.choice(clear_positions)
                positions[0] = clear_pos
                positions.sort()
        self.lily_pad_positions = [int(p) for p in positions]
        for i, pos in enumerate(positions):
            mesh = self.reg.lily_pad.make_random()
            width = _round_extent(mesh.geometry, 0)
            mover = Mover(mesh, dir=0, width=width,
                          collision_box=self.hero_width / 2 + width / 2 - 0.1)
            mover.top, mover.min, mover.mid, mover.speed = 0.2, 0.01, 0.125, 0.0
            self.entities.append(mover)
            self.floor.add(mesh)
            mover.mesh.position.set(pos, 0.125, 0)
            mover._rot_tween = tween.to(mover.mesh.rotation, random.random() * 2 + 2,
                                        y=random.random() * 1.5 + 0.5, yoyo=True, repeat=-1,
                                        ease="power2.inout")

    def _generate_dynamic(self) -> None:
        speed = (random.random() * 0.05 + 0.02) * config.DIFFICULTY_SPEED
        num_items = math.floor(random.random() * 2) + 2
        x_dir = 1 if random.random() <= 0.5 else -1
        x_pos = -6 * x_dir
        for _ in range(num_items):
            mesh = self.reg.log.make_random()
            width = _round_extent(mesh.geometry, 0)
            mover = Mover(mesh, dir=x_dir, width=width,
                          collision_box=self.hero_width / 2 + width / 2 - 0.1)
            mover.top, mover.min, mover.mid = 0.3, -0.3, -0.1
            self.entities.append(mover)
            self.floor.add(mesh)
            mover.mesh.position.set(x_pos, -0.1, 0)
            mover.speed = speed * x_dir
            x_pos -= (random.random() * 3 + 5) * x_dir

    def _bounce(self, entity: Mover, player) -> None:
        timing = 0.2
        tween.to(entity.mesh.position, timing * 0.9, y=entity.min)
        tween.to(entity.mesh.position, timing, delay=timing, y=entity.mid)
        tween.to(player.position, timing * 0.9, y=entity.top + entity.min)
        tween.to(player.position, timing, delay=timing, y=entity.top + entity.mid)

    def update(self, dt: float, player) -> None:
        if not self.active:
            return
        for entity in self.entities:
            self._move(entity)
        if not player.moving and not player.riding_on:
            for entity in self.entities:
                self._check_collision(player, entity)
            self._check_hazard(player)

    def _move(self, entity: Mover) -> None:
        entity.mesh.position.x += entity.speed
        if entity.mesh.position.x > OFFSET and entity.speed > 0:
            entity.mesh.position.x = -OFFSET
        elif entity.mesh.position.x < -OFFSET and entity.speed < 0:
            entity.mesh.position.x = OFFSET

    def get_ridable_for_position(self, position) -> Optional[Mover]:
        if round(position.z) != round(self.position.z):
            return None
        return self._get_collision_log(position)

    def get_player_lower_bounce_position_for_entity(self, entity: Mover) -> float:
        return entity.top + entity.mid

    def get_player_sunken_position(self) -> float:
        return math.sin(self.sine_count) * 0.08 - 0.2

    def _check_hazard(self, player) -> None:
        if round(player.position.z) == round(self.position.z) and not player.moving:
            if not player.riding_on:
                if player.is_alive:
                    self.on_collide(self.floor, "water", None)
                else:
                    player.position.y = self.get_player_sunken_position()
                    self.sine_count += self.sine_inc
                    player.rotation.y += 0.01
                    if self.entities:
                        player.position.x += self.entities[0].speed

    def _get_collision_log(self, position) -> Optional[Mover]:
        for entity in self.entities:
            if self._will_collide(position, entity):
                return entity
        return None

    def _will_collide(self, position, entity: Mover) -> bool:
        cx = entity.mesh.position.x
        return cx - entity.collision_box < position.x < cx + entity.collision_box

    def _check_collision(self, player, entity: Mover) -> None:
        if round(player.position.z) == round(self.position.z) and player.is_alive:
            if self._will_collide(player.position, entity):
                player.riding_on = entity
                player.riding_on_offset = player.position.x - entity.mesh.position.x
                self._bounce(entity, player)


# ---------------------------------------------------------------------------
# Railroad
# ---------------------------------------------------------------------------
class RailRoadRow(Object3D):
    def __init__(self, reg: ModelRegistry, hero_width: float, on_collide, audio=None):
        super().__init__()
        self.reg = reg
        self.hero_width = hero_width
        self.on_collide = on_collide
        self.audio = audio
        self.active = False
        self.top = config.RAILROAD_TOP

        self.railroad = reg.railroad.make("0")
        self.railroad.mark_static()
        self.light = reg.train_light.make("0")
        self.active_light_a = reg.train_light.make("active_0")
        self.active_light_b = reg.train_light.make("active_1")

        train_mesh = reg.make_train(math.floor(random.random() * 2 + 1))
        width = self._group_width(train_mesh)
        self.train = Mover(train_mesh, dir=1, width=width,
                           collision_box=self.hero_width / 2 + width / 2 - 0.1)
        self.train.speed = 0.8

        for lt in (self.light, self.active_light_a, self.active_light_b):
            self._setup_light(lt)
            lt.mark_static()
        self.active_light_a.visible = False
        self.active_light_b.visible = False

        self.railroad.add(train_mesh)
        train_mesh.position.y = config.GROUND_LEVEL
        train_mesh.position.z = 0.1
        self.add(self.railroad)

        self.light_ringing = False
        self.ring_count = 0
        self._ring_timer = 0.0

    @staticmethod
    def _group_width(group: Group) -> int:
        # x-extent across all parts (parts offset along +x; each spans ±size.x/2 locally).
        mn, mx = 1e9, -1e9
        for part in group.children:
            half = float(part.geometry.size[0]) / 2.0
            cx = part.position.x
            mn = min(mn, cx - half)
            mx = max(mx, cx + half)
        return int(round(mx - mn))

    def _setup_light(self, light: Mesh) -> None:
        light.position.z = -0.5
        light.rotation.y = math.pi
        self.railroad.add(light)

    def update(self, dt: float, player) -> None:
        if not self.active:
            return
        self._drive(player)
        self._tick_lights(dt)

    def _drive(self, player) -> None:
        train = self.train
        train.mesh.position.x += train.speed
        if train.mesh.position.x > TRAIN_OFFSET and train.speed > 0:
            train.mesh.position.x = -TRAIN_OFFSET
            self._start_ringing()
            if self.audio:
                self.audio.play(self.audio.sounds["train"]["move"]["0"])
            if train is player.hit_by_train:
                player.hit_by_train = None
        elif train.mesh.position.x < -TRAIN_OFFSET and train.speed < 0:
            train.mesh.position.x = TRAIN_OFFSET
            self._start_ringing()
            if self.audio:
                self.audio.play(self.audio.sounds["train"]["move"]["0"])
            if train is player.hit_by_train:
                player.hit_by_train = None
        elif not player.moving:
            self._train_collision(player)

    def _train_collision(self, player) -> None:
        train = self.train
        if round(player.position.z) == round(self.position.z) and player.is_alive:
            cx = train.mesh.position.x
            if cx - train.collision_box < player.position.x < cx + train.collision_box:
                if player.moving and abs(player.position.z - round(player.position.z)) > 0.1:
                    forward = (player.position.z - round(player.position.z)) > 0
                    player.position.z = self.position.z + (0.52 if forward else -0.52)
                    tween.to(player.scale, 0.3, y=1.5, z=0.2)
                    tween.to(player.rotation, 0.3, z=random.random() * math.pi - math.pi / 2)
                    self.on_collide(train, "feathers", "train")
                    return
                else:
                    player.position.y = config.GROUND_LEVEL
                    tween.to(player.scale, 0.3, y=0.2, x=1.5)
                    tween.to(player.rotation, 0.3, y=random.random() * math.pi - math.pi / 2)
                self.on_collide(train, "feathers", "train")

    def _start_ringing(self) -> None:
        self.light_ringing = True
        self.ring_count = 0
        self._ring_timer = 0.0
        self._ring_light()

    def _tick_lights(self, dt: float) -> None:
        if not self.light_ringing:
            return
        self._ring_timer += dt
        if self._ring_timer >= 0.2:
            self._ring_timer -= 0.2
            self._ring_light()

    def _ring_light(self) -> None:
        if self.light_ringing and self.ring_count < 15:
            self.light.visible = False
            self.ring_count += 1
            self.active_light_b.visible = self.active_light_a.visible
            self.active_light_a.visible = not self.active_light_a.visible
        else:
            self.light_ringing = False
            self.ring_count = 0
            self.light.visible = True
            self.active_light_a.visible = self.active_light_b.visible = False
