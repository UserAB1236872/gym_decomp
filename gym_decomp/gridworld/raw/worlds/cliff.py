import numpy as np

from gym_decomp.gridworld.raw import Gridworld


class CliffWorld(Gridworld):
    def __init__(self, misfire_prob=0.1):
        cliff = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, -10, -10, -10, 0],
            [0, 0, 0, 0, 0]
        ], dtype=float)

        success = np.array([
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 15],
            [0, 0, 0, 0, 0]
        ], dtype=float)

        fail = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, -20],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=float)

        gold = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 10, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=float)

        rewards = {
            "cliff": cliff,
            "success": success,
            "fail": fail,
            "gold": gold,
        }

        terminals = np.array([
            [False, False, False, False, True],
            [False, False, False, False, True],
            [False, True, True, True, True],
            [False, False, False, False, False]
        ])

        misfires = np.array([
            ['', '', '', '', ''],
            ['', '', '', '', ''],
            ['', '', '', '', ''],
            ['', 'u', 'u', 'u', '']
        ])

        impassable = np.full(terminals.shape, False)

        super().__init__(rewards, terminals, misfires,
                         impassable, terminals.shape, "Cliffworld", misfire_prob=misfire_prob)
