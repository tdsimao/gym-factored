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
    max_episode_steps=200,
    kwargs={
        'bridge_len': 10,
        'max_swimming_len': 5
    },
)
