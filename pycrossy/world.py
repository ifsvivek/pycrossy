"""World / row map — the ``CrossyGameMap`` that drives endless row generation.

Owns the recycled pools of grass/road/water/railroad rows, the floor map keyed by Z, the
row-generation state machine (first 10 rows grass; then grass / road(¾) | railroad(¼) /
water; obstacle fill solid<5, empty<10, random after), and neighbour coordination so a
winnable path always exists across grass↔water boundaries.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from . import config
from .entities.rows import GrassRow, RoadRow, WaterRow, RailRoadRow
from .models import ModelRegistry
from .scene import Object3D


class _Container:
    __slots__ = ("items", "count")

    def __init__(self):
        self.items: List = []
        self.count = 0


class CrossyGameMap:
    def __init__(self, reg: ModelRegistry, hero_width: float, world: Object3D, on_collide, audio=None):
        self.reg = reg
        self.hero_width = hero_width
        self.world = world
        self.on_collide = on_collide
        self.audio = audio

        self.floor_map: Dict[int, dict] = {}
        self.grasses = _Container()
        self.water = _Container()
        self.roads = _Container()
        self.rail_roads = _Container()
        self.row_count = 0

        for i in range(config.MAX_ROWS):
            g = GrassRow(reg, hero_width, on_collide)
            w = WaterRow(reg, hero_width, on_collide)
            r = RoadRow(reg, hero_width, on_collide)
            rr = RailRoadRow(reg, hero_width, on_collide, audio=audio)
            self.grasses.items.append(g)
            self.water.items.append(w)
            self.roads.items.append(r)
            self.rail_roads.items.append(rr)
            world.add(g)
            world.add(w)
            world.add(r)
            world.add(rr)

    # -- base GameMap ------------------------------------------------------
    def get_row(self, index) -> Optional[dict]:
        return self.floor_map.get(int(index))

    def set_row(self, index, value) -> None:
        self.floor_map[int(index)] = value

    def tree_collision(self, position) -> bool:
        target_z = int(position.z)            # truncate toward 0 to key the floor map
        row = self.floor_map.get(target_z)
        if row is not None and row["type"] == "grass":
            key = int(position.x)
            if key in row["entity"].obstacle_map:
                return True
        return False

    def tick(self, dt: float, hero) -> None:
        for rr in self.rail_roads.items:
            rr.update(dt, hero)
        for r in self.roads.items:
            r.update(dt, hero)
        for w in self.water.items:
            w.update(dt, hero)

    # -- generation --------------------------------------------------------
    def get_clear_positions_from_grass(self, grass_entity) -> List[int]:
        blocked = set(grass_entity.get_blocked_positions())
        return [x for x in range(-4, 5) if x not in blocked]

    def new_row(self, row_kind: Optional[str] = None) -> None:
        if self.grasses.count == config.MAX_ROWS:
            self.grasses.count = 0
        if self.roads.count == config.MAX_ROWS:
            self.roads.count = 0
        if self.water.count == config.MAX_ROWS:
            self.water.count = 0
        if self.rail_roads.count == config.MAX_ROWS:
            self.rail_roads.count = 0
        if self.row_count < 10:
            row_kind = "grass"

        if row_kind is None:
            row_kind = random.choice(["grass", "roadtype", "water"])

        previous_row = self.get_row(self.row_count - 1)

        if row_kind == "grass":
            g = self.grasses.items[self.grasses.count]
            g.position.z = self.row_count
            required_clear: List[int] = []
            if previous_row and previous_row["type"] == "water":
                required_clear = previous_row["entity"].get_lily_pad_positions()
            g.generate(self._map_row_to_obstacle(self.row_count), required_clear)
            self.set_row(self.row_count, {"type": "grass", "entity": g})
            self.grasses.count += 1

        elif row_kind == "roadtype":
            if int(random.random() * 4) == 0:
                rr = self.rail_roads.items[self.rail_roads.count]
                rr.position.z = self.row_count
                rr.active = True
                self.set_row(self.row_count, {"type": "railRoad", "entity": rr})
                self.rail_roads.count += 1
            else:
                r = self.roads.items[self.roads.count]
                r.position.z = self.row_count
                prev_type = (self.get_row(self.row_count - 1) or {}).get("type")
                r.is_first_lane(prev_type != "road")
                # NOTE: cars are generated only when a Road is first constructed, so a
                # recycled lane keeps its existing car set (no car_gen on reuse).
                r.active = True
                self.set_row(self.row_count, {"type": "road", "entity": r})
                self.roads.count += 1

        elif row_kind == "water":
            w = self.water.items[self.water.count]
            w.position.z = self.row_count
            w.active = True
            clear_positions: List[int] = []
            if previous_row and previous_row["type"] == "grass":
                clear_positions = self.get_clear_positions_from_grass(previous_row["entity"])
            w.generate(clear_positions)
            self.set_row(self.row_count, {"type": "water", "entity": w})
            self.water.count += 1

        self.row_count += 1

    def reset(self) -> None:
        self.grasses.count = 0
        self.water.count = 0
        self.roads.count = 0
        self.rail_roads.count = 0
        self.row_count = 0
        self.floor_map = {}

    def init(self) -> None:
        for i in range(config.MAX_ROWS):
            self.grasses.items[i].position.z = config.MAP_OFFSET
            self.water.items[i].position.z = config.MAP_OFFSET
            self.water.items[i].active = False
            self.roads.items[i].position.z = config.MAP_OFFSET
            self.roads.items[i].active = False
            self.rail_roads.items[i].position.z = config.MAP_OFFSET
            self.rail_roads.items[i].active = False

        g = self.grasses.items[self.grasses.count]
        g.position.z = self.row_count
        g.generate(self._map_row_to_obstacle(self.row_count))
        self.set_row(self.row_count, {"type": "grass", "entity": g})
        self.grasses.count += 1
        self.row_count += 1

        for _ in range(config.MAX_ROWS + 3):
            self.new_row()

    def _map_row_to_obstacle(self, row: int) -> str:
        if self.row_count < 5:
            return GrassRow.SOLID
        elif self.row_count < 10:
            return GrassRow.EMPTY
        return GrassRow.RANDOM
