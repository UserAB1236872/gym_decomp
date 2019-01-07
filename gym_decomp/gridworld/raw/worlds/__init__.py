"""
Unlike Gridworld, which is a *general* implementation of the gridworld concept,
these define *specific* instances of the gridworld.
"""

from gym_decomp.gridworld.raw.worlds.cliff import Cliffworld
from gym_decomp.gridworld.raw.worlds.miniworld import MiniGridworld

ALL_WORLDS = [Cliffworld, MiniGridworld]
