import unittest
import gym
import gym_safe


class TestChainEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("gym_safe:chain-v0")

    def test_reset_env(self):
        ob = self.env.reset()
        self.assertIn(ob, range(1, 3))

    def test_move_left(self):
        state = self.env.reset()
        done = False
        reward = 0
        for i in range(4):
            state, reward, done, info = self.env.step(0)
            if done:
                break
        self.assertEqual(state, 0)
        self.assertEqual(reward, 1)
        self.assertTrue(done)

    def test_move_right(self):
        state = self.env.reset()
        done = False
        reward = 0
        for i in range(4):
            state, reward, done, info = self.env.step(1)
            if done:
                break
        self.assertEqual(reward, 10)
        self.assertTrue(done)
        self.assertEqual(state, 3)

