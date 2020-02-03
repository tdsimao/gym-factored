from gym.envs.registration import register

register(
    id='taxi-fuel-v0',
    entry_point='gym_safe.envs:TaxiFuelEnv',
    max_episode_steps=200,
    kwargs={
        'penalty_fuel_below_min_level': 0,
        'min_fuel_level': 0,
        'fuel_capacity': 14,
    }
)
