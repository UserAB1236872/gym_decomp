"""
Test suite for the main gridworld
"""

import itertools

import numpy as np
import pytest

from gym_decomp.gridworld.raw.worlds import MiniGridworld

# We don't want to document every test
# pylint: disable=C0111


def test_gridworld():
    gridworld = MiniGridworld()
    # just make sure things work
    assert gridworld.rewards["success"][2, 1] == 1


def test_states():
    gridworld = MiniGridworld()

    states = set(gridworld.states)
    # Make sure you can call it twice
    assert states == set(gridworld.states)

    shape = gridworld.shape
    states2 = set(filter(lambda x: not gridworld.impassable[x], itertools.product(
        range(shape[0]), range(shape[1]))))

    assert states == states2

    nonterminals = set(filter(lambda x: not gridworld.terminals[x], states2))

    assert nonterminals == set(gridworld.nonterminal_states())


def test_succ_map():
    gridworld = MiniGridworld()

    source = (1, 0)
    succs = gridworld.succ_map(source)

    # Testing impassable terrain
    assert succs['u'][0] == source
    assert succs['d'][0] == (2, 0)
    assert succs['l'][0] == source
    assert succs['r'][0] == (1, 1)

    # This is a terminal state, but good for testing
    source = (2, 1)
    succs = gridworld.succ_map(source)
    assert succs['u'][0] == (1, 1)
    assert succs['r'][0] == source
    assert succs['l'][0] == (2, 0)
    assert succs['d'][0] == (3, 1)


def test_successors():
    gridworld = MiniGridworld()

    source = (1, 0)
    succ_u = gridworld.successors(source, 'u')
    succ_d = gridworld.successors(source, 'd')
    succ_l = gridworld.successors(source, 'l')
    succ_r = gridworld.successors(source, 'r')

    # Testing impassable terrain
    assert succ_u[0] == source
    assert succ_d[0] == (2, 0)
    assert succ_l[0] == source
    assert succ_r[0] == (1, 1)


def test_succ_misfire():
    gridworld = MiniGridworld()

    source = (3, 0)
    succs = gridworld.succ_map(source)

    assert succs['r'][0] == (3, 1)
    assert len(succs['r']) == 1

    misfire = succs['r'][0]
    for a, succ in succs.items():
        if a == 'r':
            continue
        assert misfire in succ
        assert len(succ) == 2


def test_all_successors():
    gridworld = MiniGridworld()

    source = (1, 0)
    succs = set([(1, 1), source, (2, 0)])
    assert gridworld.all_successors(source) == succs


def test_transition_prob():
    gridworld = MiniGridworld()

    source = (1, 0)
    nxt = (1, 1)
    assert abs(gridworld.transition_prob(source, 'r', nxt) - 1.0) < 1e-4
    for state in gridworld.states:
        if state == nxt:
            continue
        assert gridworld.transition_prob(source, 'r', state) == 0

    assert abs(gridworld.transition_prob(source, 'u', source) - 1.0) < 1e-4
    assert abs(gridworld.transition_prob(source, 'l', source) - 1.0) < 1e-4
    for state in gridworld.states:
        if state == source:
            continue
        assert gridworld.transition_prob(source, 'u', state) == 0
        assert gridworld.transition_prob(source, 'l', state) == 0

    mat = gridworld.transition_matrix(source, 'r')
    verify = np.zeros(gridworld.shape)
    verify[1, 1] = 1.0
    assert (mat == verify).all()
    mat[1, 1] = 0.0
    assert mat.sum() == 0.0


def test_transition_misfire_prob():
    gridworld = MiniGridworld()

    source = (3, 0)
    misfire = (3, 1)
    nxt = (2, 0)
    assert abs(gridworld.transition_prob(source, 'u', nxt) - 0.9) < 1e-4
    assert abs(gridworld.transition_prob(source, 'u', misfire) - 0.1) < 1e-4

    for state in gridworld.states:
        if state == nxt or state == misfire:
            continue
        assert gridworld.transition_prob(source, 'u', state) == 0

    nxt = misfire
    assert abs(gridworld.transition_prob(source, 'r', nxt) - 1.0) < 1e-4
    for state in gridworld.states:
        if state == nxt:
            continue
        assert gridworld.transition_prob(source, 'r', state) == 0

    # This is technically behavior we don't want to rely on right now, misfires
    # shouldn't fire for moving into walls, but testing this reminds us to
    # change this to test for the right thing if we change our minds
    # and disallow this in the future
    assert abs(gridworld.transition_prob(source, 'l', source) - 0.9) < 1e-4
    assert abs(gridworld.transition_prob(source, 'l', misfire) - 0.1) < 1e-4
    for state in gridworld.states:
        if state == nxt or state == source:
            continue
        assert gridworld.transition_prob(source, 'r', state) == 0


def test_printable_state():
    gridworld = MiniGridworld()

    verify = np.array([
        ['x', 'x'],
        [' ', ' '],
        [' ', '1.0'],
        ['r', '-1.0']
    ])

    printables = gridworld.printable()
    assert (verify == printables['total']).all()

    verify[2, 1] = ' '
    assert (verify == printables['fail']).all()

    verify[2, 1] = '1.0'
    verify[3, 1] = ' '
    assert (verify == printables['success']).all()


def test_statify_impassable():
    gridworld = MiniGridworld()

    with pytest.raises(Exception):
        gridworld.statify((0, 0))
        gridworld.statify((0, 1))


def test_statify():
    gridworld = MiniGridworld()

    expected = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
    ])

    assert (expected == gridworld.statify((3, 0))).all()
