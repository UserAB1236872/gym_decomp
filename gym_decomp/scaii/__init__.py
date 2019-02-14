"""
Wrappers for SCAII scenarios, primarily four towers derivatives.
"""
import os
from pathlib import Path
import logging

import gym
from gym import spaces

from scaii.env.sky_rts.env.scenarios.city_attack import CityAttack
from scaii.env.explanation import Explanation, BarChart, BarGroup, Bar


REPLAY_PATH = (Path.home() / ".scaii/replays/")


class FourTowersV1(gym.Env):
    """
    The SCAII City Attack scenario (an expanded Four Towers with cities and enemy tanks)
    """

    def __init__(self):
        self.__world = CityAttack()
        self.__record = False
        self.recording_ep = 0
        self.__flatten_state = True

        self.__curr_state = None

        self.action_space = spaces.Discrete(4)

        if REPLAY_PATH.exists() and os.listdir(REPLAY_PATH):
            logging.warning(
                "Warning, replays in .scaii/replays may get clobbered,\
                appending .bak and making a fresh directory")

            num_baks = 1
            basepath = REPLAY_PATH
            suffix = '.bak'*num_baks
            target = basepath.with_suffix(suffix)

            while target.exists():
                num_baks += 1
                suffix = '.bak'*num_baks
                target = basepath.with_suffix(suffix)

            basepath.rename(target)

    def change_map(self, map_name):
        """
        Change the map to another one in the Sky-RTS backend maps directory.

        Note, this does not and cannot verify properly that you load another Four Towers scenario!
        Make sure you're loading it properly!

        This is primarily useful for loading hand-coded or
        "static" versions for testing hand-tailored states.
        """
        self.__curr_state = None
        self.__world = CityAttack(map_name=map_name)

        return self.reset()

    @property
    def flatten_state(self):
        """
        Whether to `.flatten()` the state in an observation.
        By default this is true, but you can deactivate it to get
        the true 40x40x8 map (e.g. if you want to use convolution)
        """
        return self.__flatten_state

    @flatten_state.setter
    def flatten_state(self, val):
        self.__flatten_state = val

    @property
    def record(self):
        """
        Whether to dump this to a SCAII replay file
        """
        return self.__record

    @record.setter
    def record(self, val):
        self.__record = val

    @property
    def action_meanings(self):
        """
        The meanings of the actions we can take, in order of index
        """
        return ["Q4", "Q1", "Q3", "Q2"]

    @property
    def reward_types(self):
        """
        The set of possible reward types, presented as a list
        """
        return [*self.__world.reward_types()]

    @property
    def curr_state(self):
        """
        The current state. Internally this is not flattened,
        so if you temporarily turn off `flatten_state`
        you can get a peek at the current state unflattened if you need it.

        However, if you want this info just once, it's recommended to just use `unflattened_state`
        """
        if self.__flatten_state:
            return self.__curr_state.flatten()
        else:
            return self.__curr_state

    @property
    def unflattened_state(self):
        """
        The raw, unflattened state, if you need it
        """
        return self.__curr_state

    def render(self, mode='print'):
        return self.unflattened_state

    def reset(self):
        if self.record and not REPLAY_PATH.exists():
            REPLAY_PATH.mkdir()

        if self.record:
            replay_file = REPLAY_PATH / "replay.scr"
            if replay_file.exists():
                target = REPLAY_PATH / ("replay%d.scr" % self.recording_ep)
                (REPLAY_PATH / "replay.scr").replace(target)

            self.__curr_state = self.__world.reset(
                record=self.record).state
        else:
            self.__curr_state = self.__world.reset().state

        if self.record:
            self.recording_ep += 1

        return self.curr_state

    # pylint: disable=W0221
    def step(self, action, q_vals=None):
        """
        The normal step function overridden from gym.Env, however, it has a
        `q_vals` optional argument for sending chart data back to the replay (if applicable).

        If specified this should be in the form of a map with one key for each reward type.
        Each of these entries should contain a list, in order, for each of the four actions.
        """
        assert action in range(0, 4)
        a = self.__world.new_action()
        # pylint: disable=E1101
        a.attack_quadrant(action+1)

        obs = None
        if self.record:
            assert q_vals is not None
            explanation = self.__build_explanation(q_vals)
            obs = self.__world.act(a, explanation=explanation)
        else:
            obs = self.__world.act(a)

        self.__curr_state = obs.state
        terminal = obs.is_terminal()

        reward = 0.0
        for val in obs.typed_reward.values():
            reward += val

        for r_type in self.reward_types:
            if r_type not in obs.typed_reward:
                obs.typed_reward[r_type] = 0.0

        info = {"reward_decomposition": obs.typed_reward}

        return self.curr_state, reward, terminal, info

    def __build_explanation(self, q_vals):
        explanation = Explanation("Predicted Reward Per Quadrant")
        chart = BarChart("Move Explanation", "Actions", "QVal By Reward Type")

        for quad in range(0, 4):
            group = BarGroup("Attack %s" % self.action_meanings[quad])

            for r_type in self.reward_types:
                r_bar = Bar(r_type, q_vals[r_type][quad])
                group.add_bar(r_bar)

            chart.add_bar_group(group)

        explanation.with_bar_chart(chart)

        return explanation
