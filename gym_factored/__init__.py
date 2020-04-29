from gym.envs.registration import register

register(
    id='taxi-fuel-v0',
    entry_point='gym_factored.envs:TaxiFuelEnv',
    max_episode_steps=200,
    kwargs={
        'fuel_capacity': 14,
        'map_name': "5x5",
    }
)
register(
    id='chain-v0',
    entry_point='gym_factored.envs.unittest:ChainEnv',
    max_episode_steps=200
)
register(
    id='slippery_chain-v0',
    entry_point='gym_factored.envs.unittest:SlipperyChainEnv',
    max_episode_steps=200
)

register(
    id='non_absorbing_chain-v0',
    entry_point='gym_factored.envs.unittest:NonAbsorbingChainEnv',
    max_episode_steps=200
)
