import unittest
import gym
import numpy as np


class TestDifficultCMDPEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = np.random.random()
        cls.env = gym.make("gym_factored:difficult_cmdp-v0", prob_y_zero=cls.p)

    def test_reset_env(self):
        ob = self.env.reset()
        self.assertIn(ob, range(6))

    def test_encode(self):
        self.env.reset()
        self.assertEqual(2, self.env.encode(1, 0))
        self.assertEqual(4, self.env.encode(2, 0))

    def test_decode(self):
        self.env.reset()
        self.assertListEqual([1, 1], list(self.env.decode(3)))
        self.assertListEqual([2, 1], list(self.env.decode(5)))

    def test_encode_decode(self):
        self.env.reset()
        for i in range(6):
            self.assertEqual(i, self.env.encode(*list(self.env.decode(i))))

    def test_a(self):
        initial_state = self.env.reset()
        self.assertIn(initial_state, [0, 1])
        middle_state, _, _, _ = self.env.step(0)
        final_state, reward, done, info = self.env.step(0)
        if middle_state == 2:
            self.assertEqual(reward, 0)
        else:
            self.assertEqual(middle_state, 3)
            self.assertEqual(reward, 1)
        self.assertEqual(info['cost'], 1)
        self.assertTrue(done)
        self.assertIn(final_state, [4, 5])

    def test_b(self):
        self.env.reset()
        state, _, _, _ = self.env.step(0)
        final_state, reward, done, info = self.env.step(1)
        if state == 2:
            self.assertEqual(reward, 1)
        else:
            self.assertEqual(reward, 0)
        self.assertEqual(info['cost'], 0)
        self.assertTrue(done)
        self.assertIn(final_state, [4, 5])

    def test_stochastic_transition(self):
        repetitions = 20000
        total = 0
        self.env.seed(42)
        for i in range(repetitions):
            self.env.reset()
            state, _, _, _ = self.env.step(1)
            total += int(state == 2)
        np.testing.assert_almost_equal(self.p, total/repetitions, decimal=2)
