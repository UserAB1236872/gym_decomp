import numpy as np

from gym_decomp.gridworld.raw import Gridworld


class MiniGridworld(Gridworld):
    def __init__(self, misfire_prob=0.1):
        rewards = {
            "success": np.array([
                [0, 0],
                [0, 0],
                [0, 1],
                [0, 0]], dtype=float),
            "fail": np.array([
                [0, 0],
                [0, 0],
                [0, 0],
                [0, -1]
            ], dtype=float)
        }

        terminals = np.array([
            [False, False],
            [False, False],
            [False, True],
            [False, True]
        ])

        misfires = np.array([
            ['', ''],
            ['', ''],
            ['', ''],
            ['r', ''],
        ])

        impassable = np.array([
            [True, True],
            [False, False],
            [False, False],
            [False, False],
        ])

        super().__init__(rewards, terminals, misfires,
                         impassable, terminals.shape, "MiniWorld", misfire_prob=misfire_prob)
