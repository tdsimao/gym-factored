import unittest
import gym
import numpy as np


class TestCostChainEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make('gym_factored:small_cost_chain-v0')

    def test_reset_env(self):
        ob = self.env.reset()
        self.assertIn(ob, range(2))

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
        self.assertFalse(done)
        self.assertIn(final_state, [4, 5])
        final_state, reward, done, info = self.env.step(0)
        self.assertTrue(done)
        self.assertEqual(reward, 0)
        self.assertEqual(info['cost'], 0)
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
        self.assertFalse(done)
        self.assertIn(final_state, [4, 5])
        final_state, reward, done, info = self.env.step(1)
        self.assertTrue(done)
        self.assertEqual(reward, 0)
        self.assertEqual(info['cost'], 0)
        self.assertIn(final_state, [4, 5])

    def test_reset(self):
        initial_state = self.env.reset()
        for _ in range(5):
            state, reward, done, info = self.env.step(2)
            self.assertEqual(list(self.env.decode(initial_state)), list(self.env.decode(state)))
            self.assertEqual(reward, 0)
            self.assertEqual(info['cost'], 0)
            self.assertFalse(done)
        final_state, reward, done, info = self.env.step(2)
        self.assertEqual(list(self.env.decode(initial_state)), list(self.env.decode(final_state)))
        self.assertEqual(reward, 0)
        self.assertEqual(info['cost'], 0)
        self.assertTrue(done)

    def test_stochastic_transition(self):
        repetitions = 20000
        total = 0
        self.env.seed(42)
        for i in range(repetitions):
            self.env.reset()
            state, _, _, _ = self.env.step(1)
            total += int(state == 2)
        np.testing.assert_almost_equal(0.1, total/repetitions, decimal=2)
