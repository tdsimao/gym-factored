import unittest
import gym


class TestSysAdminEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("gym_factored:sysadmin-v0")

    def test_reset_env(self):
        state = self.env.reset()
        self.assertEqual(state, 255)

    def test_episode_length(self):
        self.env.reset()
        episode_length = 0
        done = False
        while not done:
            state, reward, done, info = self.env.step(self.env.action_space.sample())
            episode_length += 1
        self.assertEqual(episode_length, 40)

    def test_decode(self):
        initial_state = self.env.reset()
        decoded_initial_state = list(self.env.decode(initial_state))
        self.assertListEqual(decoded_initial_state, [1, 1, 1, 1, 1, 1, 1, 1])

    def test_rendering(self):
        self.env.reset()
        done = False
        while not done:
            action = self.env.action_space.sample()
            _, _, done, _ = self.env.step(action)
            out = self.env.render(mode='ansi')
            lines = out.getvalue().split('\n')
            if action < 8:
                self.assertEqual(lines[0][action], 'O')
            self.assertEqual(lines[1][action], '^')


class TestSysAdminXEnv(unittest.TestCase):
    def test_eight_machines(self):
        env = gym.make("gym_factored:sysadmin8-v0")
        initial_state = env.reset()
        decoded_initial_state = list(env.decode(initial_state))
        self.assertListEqual(decoded_initial_state, [1, 1, 1, 1, 1, 1, 1, 1])

    def test_five_machines(self):
        env = gym.make("gym_factored:sysadmin5-v0")
        initial_state = env.reset()
        decoded_initial_state = list(env.decode(initial_state))
        self.assertListEqual(decoded_initial_state, [1, 1, 1, 1, 1])

