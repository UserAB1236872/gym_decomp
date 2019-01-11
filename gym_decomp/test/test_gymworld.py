import gym
import gym_decomp as _


def test_state_rep():
    env = gym.make('MiniGridworld-v0')

    state = env.reset()
    assert state.shape == (8,)

    nxt, reward, terminal, info = env.step(2)
    assert nxt.shape == (8,)
    assert isinstance(reward, float)
    #pylint: disable=c0121
    assert terminal == True or terminal == False

    assert 'reward_decomposition' in info

    total = 0.0
    for name, val in info['reward_decomposition'].items():
        assert name in ['success', 'fail']
        total += val

    assert total - reward < 1e-4

    _ = gym.make('Cliffworld-v0')
