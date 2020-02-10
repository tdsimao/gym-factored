import unittest
import gym


class TestMakeEnvs(unittest.TestCase):
    @classmethod
    def setUp(self) -> None:
        self.env = gym.make("gym_safe:chain-v0")
        self.env.reset()

    def test_make_chain_env(self):
        self.env = gym.make("gym_safe:chain-v0")
        ob = self.env.reset()
        self.assertIn(ob, range(1, 3))

    def test_make_chain_env_imported_module(self):
        self.env = gym.make("chain-v0")
        state = self.env.reset()
        self.assertIn(state, range(1, 3))
