"""Shared test fixtures (all headless — no GL needed for game logic)."""
from __future__ import annotations

import pytest

from tests._helpers import make_engine


@pytest.fixture
def engine():
    return make_engine(0)
