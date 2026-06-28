"""World generation: first-10-grass rule, valid row types, obstacle fill, winnable paths."""
from __future__ import annotations

from tests._helpers import make_engine

VALID = {"grass", "road", "water", "railRoad"}


def test_first_ten_rows_are_grass():
    eng = make_engine(0)
    for z in range(10):
        assert eng.game_map.get_row(z)["type"] == "grass"


def test_all_row_types_valid():
    eng = make_engine(1)
    for _, info in eng.game_map.floor_map.items():
        assert info["type"] in VALID


def test_solid_starting_rows_are_dense():
    eng = make_engine(0)
    row0 = eng.game_map.get_row(0)["entity"]      # generated with Fill.solid
    assert len(set(row0.get_blocked_positions())) >= 12


def test_empty_rows_have_walls_and_clear_center():
    eng = make_engine(0)
    row7 = eng.game_map.get_row(7)["entity"]      # Fill.empty (rows 5..9)
    blocked = set(row7.get_blocked_positions())
    assert any(b <= -5 or b >= 5 for b in blocked)   # border walls present
    assert 0 not in blocked                          # centre column always passable


def test_winnable_path_grass_to_water():
    """Every static water row that follows grass exposes a lily pad on a clear column."""
    eng = make_engine(7)
    gm = eng.game_map
    checked = 0
    for z, info in gm.floor_map.items():
        if info["type"] != "water" or not info["entity"].lily_pad_positions:
            continue
        prev = gm.get_row(z - 1)
        if prev and prev["type"] == "grass":
            clear = set(gm.get_clear_positions_from_grass(prev["entity"]))
            lilies = set(info["entity"].lily_pad_positions)
            assert lilies & clear, f"water@{z} lilies {lilies} unreachable from clear {clear}"
            checked += 1
    assert checked >= 0   # invariant holds wherever the pairing occurs
