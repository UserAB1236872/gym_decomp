"""
Tests for the HIV Environment
"""
import gym

import gym_decomp as _

# pylint: disable=C0111


def test_init_hiv():
    _hiv = gym.make('HivSimulator-v0')


def test_reset_hiv():
    hiv = gym.make('HivSimulator-v0')

    state = hiv.reset()
    assert len(state) == 6


def test_act_hiv():
    hiv = gym.make('HivSimulator-v0')

    init_state = hiv.reset()

    nxt, reward, terminal, info = hiv.step(2)

    assert len(nxt) == len(init_state) and (nxt != init_state).any()
    assert isinstance(reward, float)
    assert not terminal

    assert len(info) == 1

    typed_rewards = info['reward_decomposition']
    assert len(typed_rewards) == 4

    total = 0.0
    for val in typed_rewards.values():
        total += val

    assert total - reward < 1e-8
