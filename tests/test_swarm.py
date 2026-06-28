"""AI Swarm: many lockstep engines, divergent agents, leader tracking, ghost overlay.

Pure CPU (scene graph + engines + model geometry) — no GL context required.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from ai.swarm import SwarmSession
from pycrossy.config import FIXED_DT


def _run(sess, ticks):
    for _ in range(ticks):
        sess.frame_step(FIXED_DT)


def test_spawns_requested_count_and_identical_start():
    s = SwarmSession(algo="neat", seed=1, size=8)
    assert 2 <= s.n <= 8
    # every engine starts from the same seeded world -> identical first-row layout
    types = [e.game_map.get_row(0) for e in s.engines]
    kinds = {(t or {}).get("type") for t in types}
    assert len(kinds) == 1                       # all worlds began identical
    assert s.ghost_group.parent is s.engines[s.best].world


def test_agents_diverge_and_leader_is_tracked():
    s = SwarmSession(algo="neat", seed=2, size=8)
    _run(s, 200)
    zs = [round(e.hero.position.z, 2) for e in s.engines]
    assert len(set(zs)) > 1                      # different policies -> different progress
    assert 0 <= s.best < s.n
    assert s.alive[s.best] or s.alive_count == 0
    assert s.score == s.max_z[s.best]
    # the leader is (one of) the furthest among the living
    if s.alive_count:
        assert s.max_z[s.best] == max(s.max_z[i] for i in range(s.n) if s.alive[i])


def test_ghosts_hide_leader_and_dead():
    s = SwarmSession(algo="neat", seed=3, size=8)
    _run(s, 120)
    for i in range(s.n):
        if i == s.best or not s.alive[i]:
            assert s.ghosts[i].visible is False
        else:
            assert s.ghosts[i].visible is True
    assert s.crown.visible == s.alive[s.best]


def test_restart_resets_round_and_revives():
    s = SwarmSession(algo="neat", seed=4, size=6)
    _run(s, 50)
    r0 = s.round
    s.restart()
    assert s.round == r0 + 1
    assert all(s.alive)
    assert all(z == 0 for z in s.max_z)
    assert s.best == 0


def test_scene_is_the_leader_engine_scene():
    s = SwarmSession(algo="neat", seed=5, size=6)
    _run(s, 30)
    assert s.scene is s.engines[s.best].scene
    assert s.available
