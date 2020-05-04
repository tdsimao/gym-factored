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
pip install -e  git+git://github.com/tdsimao/gym-factored.git#egg=gym_factored
```

### Tests
```bash
python -m unittest discover gym-factored/tests/
```


### Example
```python
import gym

if __name__ == '__main__':
    env = gym.make('gym_factored:taxi-fuel-v0')
    env.seed(42)
    ob = env.reset()
    print(ob, list(env.decode(ob)), env.encode(*list(env.decode(ob))))
    while True:
        action = env.action_space.sample()
        ob, reward, done, _ = env.step(action)
        print(ob, list(env.decode(ob)), env.encode(*list(env.decode(ob))))
        if done:
            break
```

Executing the code above should give an output like:
```
2611 [1, 4, 1, 2, 7] 2611
2610 [1, 4, 1, 2, 6] 2610
2609 [1, 4, 1, 2, 5] 2609
4008 [2, 4, 1, 2, 4] 4008
4007 [2, 4, 1, 2, 3] 4007
5406 [3, 4, 1, 2, 2] 5406
5405 [3, 4, 1, 2, 1] 5405
5404 [3, 4, 1, 2, 0] 5404
```

## List of environments

1. [TaxiFuel](gym_factored/envs/taxi_fuel.py): `gym.make('gym_factored:taxi-fuel-v0')`
2. [SysAdmin](./gym_factored/envs/sysadmin.py): `gym.make('gym_factored:sysadmin-v0')`
3. [StockTrading](./gym_factored/envs/stock_trading.py): `gym.make('gym_factored:stock-trading-v0')`
4. [Bridge](./gym_factored/envs/bridge.py): `gym.make('gym_factored:bridge-v0')`
