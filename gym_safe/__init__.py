from gym.envs.registration import register

register(
    id='taxi-fuel-v0',
    entry_point='gym_safe.envs:TaxiFuelEnv',
    max_episode_steps=200,
    kwargs={
        'fuel_capacity': 14,
    }
)
register(
    id='chain-v0',
    entry_point='gym_safe.envs.unittest:ChainEnv',
    max_episode_steps=200
)
register(
    id='slippery_chain-v0',
    entry_point='gym_safe.envs.unittest:SlipperyChainEnv',
    max_episode_steps=200
)
