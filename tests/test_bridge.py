import unittest
import gym


class TestBridgeEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("gym_factored:bridge-v0")

    def test_reset_env(self):
        state = self.env.reset()
        self.assertEqual(state, 0)

    def test_episode_length(self):
        self.env.reset()
        episode_length = 0
        done = False
        while not done:
            state, reward, done, info = self.env.step(1)
            episode_length += 1
        self.assertEqual(episode_length, 200)

    def test_decode(self):
        initial_state = self.env.reset()
        decoded_initial_state = list(self.env.decode(initial_state))
        self.assertListEqual(decoded_initial_state, [0, 0])
