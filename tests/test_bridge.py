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
        self.assertEqual(state, 0)
        self.assertEqual(episode_length, 250)
        self.assertFalse(info["suc"])
        self.assertFalse(info["fail"])

    def test_fail_end(self):
        self.env.reset()
        episode_length = 0
        state, reward, info = 0, 0, {}
        done = False
        while not done:
            state, reward, done, info = self.env.step(2)
            episode_length += 1
        self.assertIn(episode_length, range(3, 6))
        self.assertEqual(reward, -1)
        self.assertEqual(state, episode_length - 1)
        self.assertListEqual(list(self.env.decode(state)), [0, episode_length - 1])
        self.assertTrue(info['fail'])

    def test_success(self):
        self.env.reset()
        episode_length = 0
        state, reward, info = 0, 0, {}
        done = False
        while not done:
            state, reward, done, info = self.env.step(0)
            episode_length += 1

        if list(self.env.decode(state))[1] == 0:
            self.assertEqual(reward, 100)
            self.assertListEqual(list(self.env.decode(state)), [9, 0])
            self.assertTrue(info['suc'])
        else:
            self.assertEqual(reward, -1)
            self.assertTrue(info['fail'])

    def test_decode(self):
        initial_state = self.env.reset()
        decoded_initial_state = list(self.env.decode(initial_state))
        self.assertListEqual(decoded_initial_state, [0, 0])
