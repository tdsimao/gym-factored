import unittest
import gym
import numpy as np


class TestSmallCostChainEnv(unittest.TestCase):
    p = 0.1
    horizon = 4
    n = 3

    @classmethod
    def setUpClass(cls):
        cls.env = gym.make('gym_factored:small_cost_chain-v0', prob_y_zero=cls.p)

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
        for i in range(2):
            self.assertEqual(i, self.env.encode(*list(self.env.decode(i))))

    def test_a(self):
        initial_state = self.env.reset()
        self.assertIn(initial_state, range(2))
        for step in range(1, self.horizon - 1, 2):
            state_odd_step, _, _, _ = self.env.step(0)
            state_even_step, reward, done, info = self.env.step(0)
            if state_odd_step % 2:
                self.assertEqual(state_odd_step, 2 * step + 1)
                self.assertEqual(reward, 0.01)
            else:
                self.assertEqual(reward, 0)
            self.assertEqual(info['cost'], 1)
            self.assertFalse(done)
            self.assertIn(state_odd_step, [2 * step, 2 * step + 1])
            self.assertIn(state_even_step, [2 * (step + 1), 2 * (step + 1) + 1])
        final_state, reward, done, info = self.env.step(0)
        self.assertTrue(done)
        self.assertEqual(reward, 1)
        self.assertEqual(info['cost'], 0)
        self.assertIn(final_state, range(self.n * 2 - 2, self.n * 2))

    def test_b(self):
        initial_state = self.env.reset()
        self.assertIn(initial_state, range(2))

        for step in range(1, self.horizon - 1, 2):
            state_odd_step, _, _, _ = self.env.step(0)
            state_even_step, reward, done, info = self.env.step(1)
            if state_odd_step % 2:
                self.assertEqual(reward, 0)
            else:
                self.assertEqual(reward, 0.01)
            self.assertEqual(info['cost'], 0)
            self.assertFalse(done)
            self.assertIn(state_odd_step, [2 * step, 2 * step + 1])
            self.assertIn(state_even_step, [2 * (step + 1), 2 * (step + 1) + 1])
        final_state, reward, done, info = self.env.step(1)
        self.assertTrue(done)
        self.assertEqual(reward, 1)
        self.assertEqual(info['cost'], 0)
        self.assertIn(final_state, range(self.n * 2 - 2, self.n * 2))

    def test_reset(self):
        initial_state = self.env.reset()
        for _ in range(self.horizon - 1):
            state, reward, done, info = self.env.step(2)
            self.assertEqual(list(self.env.decode(initial_state)), list(self.env.decode(state)))
            self.assertEqual(reward, 0.01)
            self.assertEqual(info['cost'], 0)
            self.assertFalse(done)
        final_state, reward, done, info = self.env.step(2)
        self.assertEqual(list(self.env.decode(initial_state)), list(self.env.decode(final_state)))
        self.assertEqual(reward, 0.01)
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
        np.testing.assert_almost_equal(self.p, total/repetitions, decimal=2)


class TestCostChain1Env(TestSmallCostChainEnv):
    p = 1


class TestCostChain0Env(TestSmallCostChainEnv):
    p = 0


class TestCostChainEnv(TestSmallCostChainEnv):
    horizon = 12
    n = 11

    @classmethod
    def setUpClass(cls):
        cls.p = np.random.random()
        cls.env = gym.make("gym_factored:cost_chain-v0", prob_y_zero=cls.p, n=cls.n)
