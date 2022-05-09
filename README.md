# gym-factored
A collection of factored environments.

## Quick-start 

### Installation
```bash
git clone https://github.com/tdsimao/gym-factored.git
pip install -e gym-factored
```

Alternative;
```bash
pip install -e git+git@github.com:tdsimao/gym-factored.git#egg=gym_factored
```

### Tests
```bash
python -m unittest discover gym-factored/tests/
```


### Example
```python
import gym

env = gym.make('gym_factored:taxi-fuel-v0')
ob = env.reset()
decoded_state = list(env.decode(ob))
assert ob == env.encode(*decoded_state)
while True:
    action = env.action_space.sample()
    ob, reward, done, _ = env.step(action)
    decoded_state = list(env.decode(ob))
    print(ob, decoded_state)
    assert ob == env.encode(*decoded_state)
    if done:
        break
```

Executing the code above should give an output like:
```
1241 [0, 4, 2, 0, 9]
1240 [0, 4, 2, 0, 8]
959 [0, 3, 2, 0, 7]
958 [0, 3, 2, 0, 6]
957 [0, 3, 2, 0, 5]
956 [0, 3, 2, 0, 4]
955 [0, 3, 2, 0, 3]
954 [0, 3, 2, 0, 2]
1233 [0, 4, 2, 0, 1]
2632 [1, 4, 2, 0, 0]
```

## List of environments

1. [TaxiFuel](gym_factored/envs/taxi_fuel.py): `gym.make('gym_factored:taxi-fuel-v0')`
2. [SysAdmin](./gym_factored/envs/sysadmin.py): `gym.make('gym_factored:sysadmin-v0')`
3. [StockTrading](./gym_factored/envs/stock_trading.py): `gym.make('gym_factored:stock-trading-v0')`
4. [Bridge](./gym_factored/envs/bridge.py): `gym.make('gym_factored:bridge-v0')`
