from gym.envs.registration import register

register(
    id='taxi-fuel-v0',
    entry_point='gym_factored.envs.taxi_fuel:TaxiFuelEnv',
    max_episode_steps=200,
    kwargs={
        'fuel_capacity': 14,
        'map_name': "5x5",
    }
)
register(
    id='taxi-fuel-small-v0',
    entry_point='gym_factored.envs.taxi_fuel:TaxiFuelEnv',
    max_episode_steps=40,
    kwargs={
        'fuel_capacity': 10,
        'map_name': "4x4",
    }
)
register(
    id='taxi-fuel-tiny-v0',
    entry_point='gym_factored.envs.taxi_fuel:TaxiFuelEnv',
    max_episode_steps=8,
    kwargs={
        'fuel_capacity': 5,
        'map_name': "2x2",
    }
)
register(
    id='chain-v0',
    entry_point='gym_factored.envs.simple_chain:ChainEnv',
    max_episode_steps=200
)
register(
    id='slippery_chain-v0',
    entry_point='gym_factored.envs.simple_chain:SlipperyChainEnv',
    max_episode_steps=200
)
register(
    id='non_absorbing_chain-v0',
    entry_point='gym_factored.envs.simple_chain:NonAbsorbingChainEnv',
    max_episode_steps=200
)
register(
    id='chain2d-v0',
    entry_point='gym_factored.envs.simple_chain2d:Chain2DEnv',
    max_episode_steps=200
)
register(
    id='slippery_chain2d-v0',
    entry_point='gym_factored.envs.simple_chain2d:SlipperyChain2DEnv',
    max_episode_steps=200
)
register(
    id='non_absorbing_chain2d-v0',
    entry_point='gym_factored.envs.simple_chain2d:NonAbsorbingChain2DEnv',
    max_episode_steps=200
)
register(
    id='sysadmin-v0',
    entry_point='gym_factored.envs.sysadmin:SysAdminEnv',
    max_episode_steps=40,
    kwargs={'size': 8},
)
for i in range(3, 50):
    register(
        id='sysadmin{}-v0'.format(i),
        entry_point='gym_factored.envs.sysadmin:SysAdminEnv',
        max_episode_steps=40,
        kwargs={'size': i},
    )
register(
    id='stock-trading-v0',
    entry_point='gym_factored.envs.stock_trading:StockTradingEnv',
    max_episode_steps=40,
    kwargs={
        'number_of_sectors': 3,
        'number_of_stocks_per_sector': 2,
    },
)
for i in range(1, 5):
    for j in range(1, 5):
        register(
            id='stock-trading_{}_{}-v0'.format(i, j),
            entry_point='gym_factored.envs.stock_trading:StockTradingEnv',
            max_episode_steps=40,
            kwargs={
                'number_of_sectors': i,
                'number_of_stocks_per_sector': j
            },
        )
register(
    id='bridge-v0',
    entry_point='gym_factored.envs.bridge:BridgeEnv',
    max_episode_steps=250,
    kwargs={
        'bridge_len': 10,
        'max_swimming_len': 5
    },
)
register(
    id='cliff_walking_cost-v0',
    max_episode_steps=100,
    entry_point='gym_factored.envs.cliff_walking:CliffWalkingCostEnv',
    kwargs={
        'num_rows': 4,
        'num_cols': 12
    },
)
register(
    id='cmdp-v0',
    max_episode_steps=6,
    entry_point='gym_factored.envs.simple_cmdp:CMDPEnv',
    kwargs={
        'ns': 3
    },
)
register(
    id='difficult_cmdp-v0',
    max_episode_steps=2,
    entry_point='gym_factored.envs.difficult_cmdp:DifficultCMDPEnv',
    kwargs={
        'prob_y_zero': 0.1
    },
)

register(
    id='small_cost_chain-v0',
    max_episode_steps=6,
    entry_point='gym_factored.envs.cost_chain:CostChainEnv',
    kwargs={
        'prob_y_zero': 0.1,
        'n': 3
    },
)
