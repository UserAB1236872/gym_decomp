"""
A gym wrapper over the HIV Simulator at https://github.com/Zaerei/hip-mdp-public
(Fork of https://github.com/dtak/hip-mdp-public).

The fork is slightly modified to be pip-installable, as well as have a function to
grab each reward component individually.

The originally computed reward is just a linear combination of a few features, so each component
is simply those features before their addition together.
"""
import logging

from hiv_simulator.hiv import HIVTreatment
import gym
from gym import spaces

import numpy as np


class HivSimV0(gym.Env):
    """
    A thin wrapper over the HIVTreatment environment to conform to gym style.

    For more info on the environment see the original repository in the module documentation.
    """

    def __init__(self):
        self.__world = HIVTreatment()
        self.action_space = spaces.Discrete(4)

    @property
    def state_meanings(self):
        """
        What each cell in the returned state means
        """
        return ['T1: non-infected CD4+ T-lymphocytes [cells / ml]',
                'T1*: infected CD4+ T-lymphocytes [cells / ml]',
                'T2: non-infected macrophages [cells / ml]',
                'T2*: infected macrophages [cells / ml]',
                'V: number of free HI viruses [copies / ml]',
                'E: number of cytotoxic T-lymphocytes [cells / ml]']

    @property
    def action_meanings(self):
        """
        The meanings of the actions we can take, in order of index
        """
        return ["None active", "RTI Active", "PI Active", "RTI & PI Active"]

    @property
    def reward_types(self):
        """
        The valid reward types
        """
        return ["V: Free HI viruses",
                "Episode value 1",
                "Episode value 2",
                "E: Cytotoxic T-lymphocytes"]

    def reset(self):
        self.__world.reset()

        return self.__world.observe()

    def step(self, action):
        reward, nxt = self.__world.perform_action(action)
        # In the code for perform_action, the total reward is calculated after the updates
        # So this is proper
        typed_reward = self.__world.typed_reward(action)
        terminal = self.__world.is_done()

        info = {'reward_decomposition': typed_reward}

        reward_sum = 0
        for val in typed_reward.values():
            reward_sum += val

        if reward_sum - reward > 1e-8 or np.isnan(reward_sum):
            logging.warning("Warning, HIV Decomposition =/= returned reward:\nReward: %f\n\
            Decomposition: %s (sum: %f)\n", reward, typed_reward, reward_sum)
            info['warning_decomposition_mismatch'] = True

        return nxt, reward, terminal, info

    def render(self, mode=None):
        pass
