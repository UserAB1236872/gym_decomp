from abc import ABCMeta

import gym
from gym import spaces
from gym.utils import seeding


# pylint: disable=C0103


class __Gridworld(gym.Env, metaclass=ABCMeta):
    """
    A basic gridworld containing multiple reward types (dependent on the exact domain).

    Observation Space: A vector that one-hot encodes the map position
    Action Space: `Discrete(4)` corresponding to up, down, left, or right
    """

    metadata = {'render.modes': ['println']}

    def __init__(self, world):
        from gym_decomp.gridworld.raw.q_world import QWorld as __QWorld

        self.np_random, _ = seeding.np_random(None)
        self.__world = __QWorld(world, self.np_random)
        self.__curr_state = None
        self.action_space = spaces.Discrete(4)
        self.__action_map = ['u', 'd', 'l', 'r']
        self.seed()

    @property
    def reward_types(self):
        return self.__world.reward_types

    @property
    def action_meanings(self):
        return ['up', 'down', 'left', 'right']

    def reset(self):
        self.__curr_state = self.__world.reset()

        return self.__world.statify(self.__curr_state).flatten()

    def step(self, action):
        action = self.__action_map[action]
        nxt, decomp_reward, reward, terminal = self.__world.act(
            self.__curr_state, action)
        self.__curr_state = nxt
        info = {'reward_decomposition': decomp_reward}

        state = self.__world.statify(self.__curr_state).flatten()

        return state, reward, terminal, info

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31

        self.__world.np_random = self.np_random
        return [seed1, seed2]

    def render(self, mode="println"):
        state = self.__world.statify(self.__curr_state)
        return state

class CliffworldV0(__Gridworld):
    """
    A small toy problem with several reward types and a "cliff" along the bottom you can fall off
    """

    def __init__(self):
        from gym_decomp.gridworld.raw.worlds import Cliffworld as __Cliffworld

        world = __Cliffworld()
        super().__init__(world)


class MiniGridworldV0(__Gridworld):
    """
    A tiny toy problem, mainly for testing purposes
    """

    def __init__(self):
        from gym_decomp.gridworld.raw.worlds import MiniGridworld as __MiniGridworld

        world = __MiniGridworld()
        super().__init__(world)


class CliffworldDeterministicV0(__Gridworld):
    """
    A small toy problem with several reward types and a "cliff" along the bottom you can fall off.

    Unlike the normal one, this one has no ability to "fall"
    """

    def __init__(self):
        from gym_decomp.gridworld.raw.worlds import Cliffworld as __Cliffworld

        world = __Cliffworld(misfire_prob=0.0)
        super().__init__(world)
