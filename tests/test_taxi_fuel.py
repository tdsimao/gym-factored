import unittest
import gym


class TestTaxiEnvs(unittest.TestCase):
    def test_taxi_suc_trial(self):
        env = gym.make("gym_factored:taxi-fuel-v0")
        env.reset()
        # move agent next to goal state
        env.unwrapped.s = env.encode(0, 0, 4, 0, 10)
        # move to goal state
        state, reward, done, info = env.step(5)
        self.assertEqual(state, env.encode(0, 0, 0, 0, 9))
        self.assertEqual(reward, 20)
        self.assertTrue(done)
        self.assertTrue(info['suc'])

    def test_taxi_failed_trial(self):
        env = gym.make("gym_factored:taxi-fuel-v0")
        env.reset()
        # move agent next to dead end
        env.unwrapped.s = env.encode(0, 0, 4, 0, 1)
        # move to dead-end state
        state, reward, done, info = env.step(4)
        self.assertEqual(state, env.encode(0, 0, 4, 0, 0))
        self.assertEqual(reward, -20)
        self.assertTrue(done)
        self.assertFalse(info['suc'])
