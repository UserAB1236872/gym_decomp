"""
Tests for the SCAII Four Towers
"""
import unittest
import os

import gym

import gym_decomp as _
from gym_decomp.scaii import REPLAY_PATH

# pylint: disable=C0111


def test_init_scaii():
    _scaii = gym.make('ScaiiFourTowers-v1')


def test_reset_scaii():
    scaii = gym.make('ScaiiFourTowers-v1')

    state = scaii.reset()
    # hp, friend, enemy, tank, large tower, small tower, large city, small city
    # 40x40 each
    assert len(state) == 40*40*8


def test_act_scaii():
    scaii = gym.make('ScaiiFourTowers-v1')

    init_state = scaii.reset()

    print("hi")
    
    nxt, reward, terminal, info = scaii.step(2)

    print(nxt, reward, terminal, info)
    assert len(nxt) == len(init_state) and (nxt != init_state).any()
    assert isinstance(reward, float)
    assert terminal in [True, False]

    assert len(info) == 1

    typed_rewards = info['reward_decomposition']
    expected_types = {"Enemy Destroyed", "Friend Destroyed", "City Destroyed",
                      "City Damaged", "Friend Damaged", "Enemy Damaged", "Living"}
    assert expected_types == set(typed_rewards.keys())

    total = 0.0
    for val in typed_rewards.values():
        total += val

    assert total - reward < 1e-8
