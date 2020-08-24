import unittest
import gym


class TestCMDPEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("gym_factored:cmdp-v0")

    def test_reset_env(self):
        ob = self.env.reset()
        self.assertIn(ob, range(3))

    def test_stay(self):
        initial_state = self.env.reset()
        done = False
        for i in range(6):
            state, reward, done, info = self.env.step(0)
            self.assertEqual(state, initial_state)
            self.assertEqual(reward, 0)
            self.assertEqual(info['cost'], 0)
        self.assertTrue(done)

    def test_move(self):
        state = self.env.reset()
        done = False
        for i in range(6):
            new_state, reward, done, info = self.env.step(1)
            self.assertEqual(new_state, (state + 1) % 3)
            self.assertEqual(reward, state + 1)
            self.assertEqual(info['cost'], 1)
            state = new_state
        self.assertTrue(done)
