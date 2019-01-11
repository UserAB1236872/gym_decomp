"""
Test suite for the "qworld" exploration wrapper
"""
from gym.utils import seeding

from gym_decomp.gridworld.raw.worlds import MiniGridworld
from gym_decomp.gridworld.raw.q_world import QWorld

TEST_POINTS = 1000

# We don't want to document every test
# pylint: disable=C0111


def test_reset():
    underlying = MiniGridworld()
    np_rand, _ = seeding.np_random(0)
    world = QWorld(underlying, np_rand)
    terminals = underlying.terminals
    impassable = underlying.impassable

    for _ in range(TEST_POINTS):
        state = world.reset()
        assert not terminals[state] and not impassable[state]


def test_act():
    np_rand, _ = seeding.np_random(0)
    world = QWorld(MiniGridworld(), np_rand)

    state = (2, 0)
    nxt = (2, 1)

    (s_sp, rewards, total_reward, terminal) = world.act(state, 'r')

    assert s_sp == nxt
    assert rewards['success'] == 1
    assert rewards['fail'] == 0
    assert total_reward == 1
    assert terminal

    nxt = (3, 0)
    (s_sp, rewards, total_reward, terminal) = world.act(state, 'd')
    assert s_sp == nxt
    assert rewards['success'] == 0
    assert rewards['fail'] == 0
    assert total_reward == 0
    assert not terminal
