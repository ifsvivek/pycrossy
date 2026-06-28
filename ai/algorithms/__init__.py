"""Algorithm implementations — importing this package registers them all.

Each module decorates its class with ``@register("name")`` so ``ai.base.make(name, ...)``
can construct it. New algorithms drop in here without touching the trainer or game.
"""
from . import neat_algo          # noqa: F401  -> "neat"
from . import policy_gradient    # noqa: F401  -> "ppo", "a2c"
from . import dqn                # noqa: F401  -> "dqn"
from . import ddqn               # noqa: F401  -> "ddqn"
from . import evolutionary       # noqa: F401  -> "es", "ga", "cmaes"
from . import search_agent       # noqa: F401  -> "minimax"

from ..base import available, make  # noqa: F401  re-export
