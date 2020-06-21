import unittest
import gym

INITIAL_STATES = [5, 6, 9, 10]


class TestChain2DEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("gym_factored:chain2d-v0")

    def test_reset_env(self):
        ob = self.env.reset()
        self.assertIn(ob, INITIAL_STATES)

    def test_move_left(self):
        for s in INITIAL_STATES:
            with self.subTest(s=s):
                self.env.reset()
                self.env.unwrapped.s = s
                _, y0 = list(self.env.decode(s))
                done = False
                reward = 0
                info = {}
                for i in range(4):
                    state, reward, done, info = self.env.step(0)
                    if done:
                        break
                x, y = list(self.env.decode(state))
                self.assertEqual(y, y0)
                self.assertEqual(x, 0)
                self.assertEqual(reward, 1)
                self.assertTrue(done)
                self.assertEqual(info['cost'], 0)

    def test_move_right(self):
        for s in INITIAL_STATES:
            with self.subTest(s=s):
                self.env.reset()
                self.env.unwrapped.s = s
                _, y0 = list(self.env.decode(s))
                done = False
                reward = 0
                info = {}
                for i in range(4):
                    state, reward, done, info = self.env.step(1)
                    if done:
                        break
                x, y = list(self.env.decode(state))
                self.assertEqual(y, y0)
                self.assertEqual(x, 3)
                self.assertEqual(reward, 10)
                self.assertTrue(done)
                self.assertEqual(info['cost'], 1)

    def test_move_up(self):
        for s in INITIAL_STATES:
            with self.subTest(s=s):
                self.env.reset()
                self.env.unwrapped.s = s
                x0, _ = list(self.env.decode(s))
                done = False
                reward = 0
                info = {}
                for i in range(4):
                    state, reward, done, info = self.env.step(2)
                    if done:
                        break
                x, y = list(self.env.decode(state))
                self.assertEqual(x, x0)
                self.assertEqual(y, 0)
                self.assertEqual(reward, 1)
                self.assertTrue(done)
                self.assertEqual(info['cost'], 0)

    def test_move_down(self):
        for s in INITIAL_STATES:
            with self.subTest(s=s):
                self.env.reset()
                self.env.unwrapped.s = s
                x0, _ = list(self.env.decode(s))
                done = False
                reward = 0
                info = {}
                for i in range(4):
                    state, reward, done, info = self.env.step(3)
                    if done:
                        break
                x, y = list(self.env.decode(state))
                self.assertEqual(x, x0)
                self.assertEqual(y, 3)
                self.assertEqual(reward, 10)
                self.assertTrue(done)
                self.assertEqual(info['cost'], 2)

class TestSlipperyChain2DEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("gym_factored:slippery_chain2d-v0")