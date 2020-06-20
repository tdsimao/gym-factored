import unittest
import gym


class TestChainEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("gym_factored:non_absorbing_chain-v0")

    def test_non_absorbing_left(self):
        successors = self.env.P[0][1]
        self.assertEqual(len(successors), 1)
        transition_prob, new_state, reward, done, info = successors[0]
        self.assertAlmostEqual(transition_prob, 1)
        self.assertEqual(new_state, 1)
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertEqual(info['cost'], 1)

    def test_non_absorbing_right(self):
        successors = self.env.P[3][0]
        self.assertEqual(len(successors), 1)
        transition_prob, new_state, reward, done, info = successors[0]
        self.assertAlmostEqual(transition_prob, 1)
        self.assertEqual(new_state, 2)
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertEqual(info['cost'], 0)
