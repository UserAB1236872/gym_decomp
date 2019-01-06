from gym.envs.registration import register

register(
    id='MiniGridworld-v0',
    entry_point='gym_decomp.gridworld:MiniGridworldV0'
)

register(
    id='Cliffworld-v0',
    entry_point='gym_decomp.gridworld:CliffworldV0'
)
