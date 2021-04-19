import unittest
import gym


class TestTaxiEnvs(unittest.TestCase):
    def test_taxi_suc_trial(self):
        env = gym.make("gym_factored:taxi-fuel-v0")
        env.reset()
        # move agent next to goal state
        env.unwrapped.s = env.encode(0, 0, 4, 0, 10)
        # drop passenger at destination
        state, reward, done, info = env.step(5)
        self.assertEqual(state, env.encode(0, 0, 5, 0, 9))
        self.assertEqual(reward, 20)
        self.assertFalse(done)  # episode only ends if out of fuel or end of horizon
        self.assertEqual(info['cost'], 0)

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
        self.assertTrue(info['fail'])
        self.assertEqual(info['cost'], 1)

    def test_taxi_timout_trial(self):
        env = gym.make("gym_factored:taxi-fuel-v0")
        env.reset()
        # move agent next to refuel station
        env.unwrapped.s = env.encode(3, 2, 4, 0, 13)
        # take as many steps as possible
        while True:
            state, reward, done, info = env.step(6)
            self.assertEqual(state, env.encode(3, 2, 4, 0, 13))
            self.assertEqual(reward, -1)
            self.assertEqual(info['cost'], 0)
            if done:
                self.assertFalse(info['fail'])
                self.assertFalse(info['suc'])
                self.assertTrue(info['TimeLimit.truncated'])
                break

    def test_taxi_small(self):
        env = gym.make("gym_factored:taxi-fuel-small-v0")
        env.reset()
        # move agent next to refuel station
        env.unwrapped.s = env.encode(2, 1, 0, 1, 9)
        # take as many steps as possible
        while True:
            state, reward, done, info = env.step(6)
            self.assertEqual(state, env.encode(2, 1, 0, 1, 9))
            self.assertEqual(reward, -1)
            self.assertEqual(info['cost'], 0)
            if done:
                self.assertFalse(info['fail'])
                self.assertFalse(info['suc'])
                self.assertTrue(info['TimeLimit.truncated'])
                break

    def test_taxi_tiny(self):
        env = gym.make("gym_factored:taxi-fuel-tiny-v0")
        env.reset()
        # move agent next to refuel station
        env.unwrapped.s = env.encode(1, 0, 0, 1, 1)
        # take as many steps as possible
        while True:
            state, reward, done, info = env.step(6)
            self.assertEqual(state, env.encode(1, 0, 0, 1, 4))
            self.assertEqual(reward, -1)
            self.assertEqual(info['cost'], 0)
            if done:
                self.assertFalse(info['fail'])
                self.assertFalse(info['suc'])
                self.assertTrue(info['TimeLimit.truncated'])
                break
