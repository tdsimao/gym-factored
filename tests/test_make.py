import unittest
import gym


class TestMakeEnvs(unittest.TestCase):
    def test_make_chain_env(self):
        env = gym.make("gym_factored:chain-v0")
        ob = env.reset()
        self.assertIn(ob, range(1, 3))

    def test_make_chain_env_imported_module(self):
        env = gym.make("chain-v0")
        state = env.reset()
        self.assertIn(state, range(1, 3))

    def test_make_taxi(self):
        env = gym.make("gym_factored:taxi-fuel-v0")
        state = env.reset()
        print(env.nS)
        self.assertIn(state, range(8400))

    def test_make_taxi_map(self):
        env = gym.make("gym_factored:taxi-fuel-v0", map_name="5x5")
        state = env.reset()
        self.assertIn(state, range(8400))

    def test_make_taxi_different_map(self):
        env = gym.make("gym_factored:taxi-fuel-v0", map_name="7x7")
        state = env.reset()
        self.assertIn(state, range(16464))
        self.assertIn(16463, env.P)
        self.assertEqual(len(env.P), 16464)

    def test_make_taxi_invalid_map(self):
        with self.assertRaises(AssertionError):
            gym.make("gym_factored:taxi-fuel-v0", map_name="0x0")

