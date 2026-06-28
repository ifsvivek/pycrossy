"""PyCrossy reinforcement-learning framework.

A modular RL stack around the headless game engine: a gym-like :class:`~ai.env.CrossyEnv`,
a pluggable :class:`~ai.base.Algorithm` interface with a registry of interchangeable
algorithms (NEAT, PPO, A2C, DQN, ES, GA, CMA-ES), parallel envs, checkpointing, metrics
(TensorBoard/CSV/JSON), replay recording, and a live dual-window training mode.
"""
